"""Model Initialization File

Configures the LLM model used throughout the course.

Default (LOCAL MODIFICATION): Snowflake Cortex REST API.

═══════════════════════════════════════════════════════════════════════════
  Why Snowflake instead of a direct Anthropic/OpenAI key
═══════════════════════════════════════════════════════════════════════════

Snowflake Cortex exposes an OpenAI-compatible Chat Completions endpoint at
`/api/v2/cortex/v1/chat/completions`, so `ChatOpenAI` talks to it with nothing
but a `base_url` swap. Inference runs inside the Snowflake perimeter, billed as
Snowflake credits, authenticated with a Snowflake PAT — no third-party model
key needed, and no course data leaves the governance boundary.

  Docs: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api

Tool calling and streaming are both supported on this endpoint, which is what
the Deep Agents framework in this course relies on.

═══════════════════════════════════════════════════════════════════════════
  ⚠  Gotchas specific to the Snowflake endpoint (learned by testing, not docs)
═══════════════════════════════════════════════════════════════════════════

1. `max_tokens` is REJECTED with HTTP 400:
       "max_tokens is deprecated in favor of max_completion_tokens"
   The Snowflake doc's own quickstart still shows `max_tokens`, so this bites
   immediately. That is why the models below do NOT pass `max_tokens=`. If you
   need an output cap, pass it through `model_kwargs`:
       model_kwargs={"max_completion_tokens": 4096}

2. The REST API uses the *user's default role*, so that role must hold either
   SNOWFLAKE.CORTEX_USER or SNOWFLAKE.CORTEX_REST_API_USER. Ours holds the
   latter (inference only, no data access).

3. Usage is NOT written to AI_OBSERVABILITY_EVENTS. To see spend, query
   SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY.

4. PARALLEL TOOL CALLS are broken on the Claude models. Claude-on-Cortex will
   happily EMIT an assistant turn with 2+ tool_calls, but the follow-up request
   carrying those tool results is rejected:
       HTTP 400 "Each 'toolUse' block must be accompanied with a matching
                 'toolResult' block."
   This happens even when every tool_call IS correctly paired with a result, and
   in every message shape tested (content=None / "" / omitted, results in order
   and reversed). The `toolUse`/`toolResult` wording is AWS Bedrock Converse
   vocabulary, so the Claude path appears to translate through Bedrock and drops
   all but one tool block. Sending `parallel_tool_calls: false` is ACCEPTED
   (HTTP 200) but IGNORED — the model still returns 2 calls.

   Verified 2026-09-03 on this account:
       claude-*         emits parallel, REJECTS the paired follow-up
       openai-gpt-5.4   parallel tool calls fully work
       openai-gpt-5-mini  parallel tool calls fully work
       llama4-maverick  "tool calling is not supported for this model"
       mistral-large2   "tool calling is not supported for this model"

   This breaks deepagents outright — the framework delegates to subagents via a
   `task` tool and routinely batches calls.

   FIX (implemented in `_split_parallel_tool_turns` + `SnowflakeCortexChat`):
   the constraint is one tool block per assistant TURN, not one per request. So
   rewrite the OUTBOUND history, splitting a batched turn into consecutive
   single-call pairs:

       assistant[A,B] + tool(A) + tool(B)               -> HTTP 400
       assistant[A] + tool(A) + assistant[B] + tool(B)  -> HTTP 200   <- verified

   Because only the history is reshaped, the model still emits parallel tool
   calls and LangGraph still executes them concurrently — no lost parallelism,
   no discarded work, and token streaming is unaffected. Measured on the
   todo+filesystem stress test: batches of 3 preserved, 10 messages vs 14 for
   the naive "keep only the first call" approach.

   `coerce_single_tool_call=True` is retained as a blunt fallback (drop all but
   the first call, agent works sequentially) in case the split is ever rejected.

═══════════════════════════════════════════════════════════════════════════
  Model availability
═══════════════════════════════════════════════════════════════════════════

Verified callable on this account (2026-09-03), newest of each family:

    claude-opus-5        <- latest Opus   (used for strong_model)
    claude-sonnet-5      <- latest Sonnet (used for model)
    claude-haiku-4-5     <- latest Haiku; there is no haiku-5
    openai-gpt-5.4       <- latest GPT available here

Confirmed NOT to exist on this account: claude-haiku-5, claude-haiku-4-6,
claude-opus-5-1, claude-sonnet-5-1, claude-opus-6 (all HTTP 400).

Requires CORTEX_ENABLED_CROSS_REGION for models not hosted in your own region;
this account is set to ANY_REGION.

To swap to a different Snowflake model, just change the string — every model
above is reachable with the same credential.

To revert to a stock provider (Anthropic, OpenAI, Ollama, ...), see the
commented sections at the bottom of this file.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _keep_first_tool_call(message: AIMessage) -> AIMessage:
    """Drop all but the first tool call on an assistant message.

    Works around the Claude-on-Cortex parallel tool-call limitation (gotcha 4 in
    the module docstring). The agent loses nothing semantically: it executes the
    first tool, sees the result, and requests the next one on the following turn.
    """
    tool_calls = message.tool_calls or []
    if len(tool_calls) <= 1:
        return message

    kept = tool_calls[0]
    extra_kwargs = dict(message.additional_kwargs)
    raw = extra_kwargs.get("tool_calls")
    if isinstance(raw, list) and raw:
        matching = [t for t in raw if t.get("id") == kept.get("id")]
        extra_kwargs["tool_calls"] = matching or raw[:1]

    return message.model_copy(
        update={
            "tool_calls": [kept],
            "additional_kwargs": extra_kwargs,
            "invalid_tool_calls": [],
        }
    )


def _trim_result(result: ChatResult) -> ChatResult:
    generations = []
    for gen in result.generations:
        msg = gen.message
        if isinstance(msg, AIMessage):
            msg = _keep_first_tool_call(msg)
        generations.append(ChatGeneration(message=msg, generation_info=gen.generation_info))
    return ChatResult(generations=generations, llm_output=result.llm_output)


def _as_chunk(message: AIMessage) -> ChatGenerationChunk:
    """Repackage a complete AIMessage as a single streaming chunk."""
    chunks = [
        tool_call_chunk(
            name=tc.get("name"),
            args=json.dumps(tc.get("args") or {}),
            id=tc.get("id"),
            index=i,
        )
        for i, tc in enumerate(message.tool_calls or [])
    ]
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=message.content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            tool_call_chunks=chunks,
            usage_metadata=message.usage_metadata,
            id=message.id,
        )
    )


def _split_parallel_tool_turns(messages: list[dict]) -> list[dict]:
    """Rewrite one assistant turn holding N tool calls into N assistant/tool pairs.

    This is the fix for gotcha 4. Cortex's Claude path rejects an assistant turn
    carrying 2+ `toolUse` blocks, but it happily accepts the same work spread
    across consecutive single-call turns:

        assistant[A,B] + tool(A) + tool(B)        -> HTTP 400
        assistant[A] + tool(A) + assistant[B] + tool(B) -> HTTP 200

    Crucially this only rewrites the OUTBOUND history, so the model still emits
    parallel tool calls and LangGraph still executes them concurrently. We only
    reshape how the completed work is described on the next request. Verified on
    2, 3 and 4-way splits.

    Any assistant text is kept on the first synthetic turn only, so it is not
    duplicated into the transcript N times.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        tool_calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None

        if not tool_calls or len(tool_calls) <= 1:
            out.append(msg)
            i += 1
            continue

        # Gather the tool results that answer this turn.
        j = i + 1
        results: dict[str, dict] = {}
        while j < len(messages) and messages[j].get("role") == "tool":
            results[messages[j].get("tool_call_id")] = messages[j]
            j += 1

        # No results yet (agent still mid-flight) — nothing safe to reshape.
        if not results:
            out.append(msg)
            i += 1
            continue

        for position, call in enumerate(tool_calls):
            turn = dict(msg)
            turn["tool_calls"] = [call]
            if position > 0:
                turn["content"] = None
            out.append(turn)
            answer = results.pop(call.get("id"), None)
            if answer is not None:
                out.append(answer)

        # Preserve any tool result that didn't match a call on this turn.
        out.extend(results.values())
        i = j

    return out


_TOOL_RESULT_TRUNCATION_NOTE = (
    "\n\n[...truncated by SnowflakeCortexChat: Cortex's Claude path returns "
    "HTTP 500 'internal error' on large tool results. See SNOWFLAKE.md.]"
)


def _cap_tool_results(messages: list[dict], limit: int) -> tuple[list[dict], int]:
    """Truncate oversized tool-role message content. Returns (messages, n_truncated).

    Cortex's Claude path 500s on large tool results (gotcha 7). Truncating is
    strictly better than the alternative, which is the whole agent run dying.
    """
    out, truncated = [], 0
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "tool" and isinstance(content, str) and len(content) > limit:
            copy_ = dict(msg)
            copy_["content"] = content[:limit] + _TOOL_RESULT_TRUNCATION_NOTE
            out.append(copy_)
            truncated += 1
        else:
            out.append(msg)
    return out, truncated


class SnowflakeCortexChat(ChatOpenAI):
    """ChatOpenAI pointed at Snowflake Cortex, with two Claude-path workarounds.

    Knobs, all defaulting to the right thing and all auto-disabled for openai-*
    models, which need none of them:

    * `split_parallel_tool_turns` (default True) — gotcha 4. Keeps genuine
      parallel tool execution and reshapes only the outbound history.
    * `max_tool_result_chars` (default 6000, None = off) — gotcha 7. Caps
      tool-result content, which otherwise 500s the request.
    * `coerce_single_tool_call` (default False) — blunt fallback for gotcha 4.
      Discards all but the first tool call so the agent works sequentially.
    """

    split_parallel_tool_turns: bool = True
    coerce_single_tool_call: bool = False
    max_tool_result_chars: int | None = 6000

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = payload.get("messages")
        if not messages:
            return payload

        if self.split_parallel_tool_turns:
            messages = _split_parallel_tool_turns(messages)

        if self.max_tool_result_chars:
            messages, n = _cap_tool_results(messages, self.max_tool_result_chars)
            if n:
                logger.warning(
                    "Truncated %d tool result(s) to %d chars to avoid the Cortex "
                    "Claude-path HTTP 500 on large tool results.",
                    n, self.max_tool_result_chars,
                )

        payload["messages"] = messages
        return payload

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _trim_result(result) if self.coerce_single_tool_call else result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _trim_result(result) if self.coerce_single_tool_call else result

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Only needed by the blunt fallback: a partial stream can't be trimmed
        # safely. With the default split fix, normal token streaming is kept.
        if self.coerce_single_tool_call and kwargs.get("tools"):
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            yield _as_chunk(result.generations[0].message)
            return
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        if self.coerce_single_tool_call and kwargs.get("tools"):
            result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            yield _as_chunk(result.generations[0].message)
            return
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


# Context and output limits as documented by SNOWFLAKE for the Cortex REST API —
# not Anthropic's or OpenAI's own numbers, which differ. Snowflake serves
# claude-sonnet-5 at a 1M context, whereas LangChain's registry lists 200K for
# the 4-5 generation. Source:
#   https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql
# (model availability / context window table, read 2026-09-03)
_CORTEX_MODEL_LIMITS: dict[str, tuple[int, int]] = {
    "claude-opus-5":     (1_000_000, 128_000),
    "claude-opus-4-8":   (1_000_000, 128_000),
    "claude-opus-4-7":   (1_000_000, 128_000),
    "claude-opus-4-6":   (1_000_000, 128_000),
    "claude-sonnet-5":   (1_000_000,  64_000),
    "claude-sonnet-4-6": (1_000_000,  64_000),
    "claude-sonnet-4-5": (  200_000,  64_000),
    "claude-opus-4-5":   (  200_000,  64_000),
    "claude-haiku-4-5":  (  200_000,  64_000),
    "openai-gpt-4.1":    (  128_000,  32_000),
    "llama4-maverick":   (  128_000,   8_192),
    "mistral-large2":    (  128_000,   8_192),
}

_DEFAULT_LIMITS = (200_000, 64_000)


def _build_profile(model_name: str) -> dict:
    """Return a LangChain-shaped model profile for a Cortex model.

    Needed because `.profile` is resolved from LangChain's provider+model
    registry, and `SnowflakeCortexChat` is a ChatOpenAI subclass pointed at a
    custom base_url — so no registry entry exists and `.profile` would be None.

    That is not cosmetic. Lessons that introspect the context window do this:

        model.profile = {**model.profile, "max_input_tokens": 700}

    which raises `TypeError: 'NoneType' object is not a mapping` on a None
    profile. Module 3's summarization lessons (m3.1) fail at import without it.
    """
    max_in, max_out = _CORTEX_MODEL_LIMITS.get(model_name, _DEFAULT_LIMITS)
    is_claude = model_name.startswith("claude-")
    # llama/mistral on Cortex return "tool calling is not supported for this model"
    supports_tools = is_claude or model_name.startswith("openai-")

    return {
        "name": f"{model_name} (Snowflake Cortex)",
        "max_input_tokens": max_in,
        "max_output_tokens": max_out,
        "open_weights": not (is_claude or model_name.startswith("openai-")),
        "text_inputs": True,
        "image_inputs": is_claude or model_name.startswith("openai-"),
        "audio_inputs": False,
        "pdf_inputs": is_claude,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": is_claude,
        "tool_calling": supports_tools,
        "structured_output": supports_tools,
        "attachment": is_claude,
        "temperature": True,
        "image_url_inputs": is_claude,
        "tool_call_streaming": supports_tools,
    }


def _snowflake_base_url() -> str:
    """Build the Cortex OpenAI-compatible base URL from the account identifier."""
    explicit = os.environ.get("SNOWFLAKE_CORTEX_BASE_URL")
    if explicit:
        return explicit.rstrip("/")

    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    if not account:
        raise RuntimeError(
            "Set SNOWFLAKE_ACCOUNT (e.g. MYORG-MYACCOUNT) in python/.env, or set "
            "SNOWFLAKE_CORTEX_BASE_URL to the full .../api/v2/cortex/v1 URL."
        )
    return f"https://{account}.snowflakecomputing.com/api/v2/cortex/v1"


def snowflake_chat_model(model_name: str, **kwargs) -> SnowflakeCortexChat:
    """Return a chat model pointed at Snowflake Cortex.

    Deliberately does not set `max_tokens` — the Snowflake endpoint rejects it
    (gotcha 1). Use `model_kwargs={"max_completion_tokens": N}` instead.

    The parallel tool-call workaround (gotcha 4) is enabled by default and
    auto-disabled for openai-* models, which don't need it.
    """
    pat = os.environ.get("SNOWFLAKE_PAT")
    if not pat:
        raise RuntimeError(
            "SNOWFLAKE_PAT is not set. Add your Snowflake programmatic access "
            "token to python/.env. Regenerate with:\n"
            "  ALTER USER LCA_DEEPAGENTS_SVC ADD PROGRAMMATIC ACCESS TOKEN ..."
        )

    kwargs.setdefault("split_parallel_tool_turns", not model_name.startswith("openai-"))
    kwargs.setdefault("profile", _build_profile(model_name))
    if model_name.startswith("openai-"):
        # openai-* takes 48KB of tool results without complaint; no cap needed.
        kwargs.setdefault("max_tool_result_chars", None)

    return SnowflakeCortexChat(
        model=model_name,
        base_url=_snowflake_base_url(),
        api_key=pat,
        **kwargs,
    )


# ═══ Default Models (Snowflake Cortex) ═══════════════════════════════════════
# Timeouts are deliberately higher than the upstream course values (60s / 120s).
# Upstream paired 60s with claude-haiku-4-5; sonnet-5 and opus-5 are slower per
# call, and the Module 4/5 research labs stuff large Tavily result sets into
# context. At 60s, m4.2_run_newsletter.py died with openai.APITimeoutError
# mid-delegation. Raise these rather than downgrading the model.
# Workhorse model, used by nearly every lesson.
# model = snowflake_chat_model("claude-sonnet-5", timeout=180, max_retries=2)

# A more capable model for steps that need stronger reasoning.
strong_model = snowflake_chat_model("claude-opus-5", timeout=300, max_retries=2)

# ═══ Alternative Snowflake models ════════════════════════════════════════════
# Cheaper/faster default — swap in if the research labs in Modules 4-5 burn
# more credits than you want. This is the newest Haiku (no haiku-5 exists).
model = snowflake_chat_model("claude-haiku-4-5", timeout=60, max_retries=2)
#
# model = snowflake_chat_model("claude-sonnet-4-6", timeout=60, max_retries=2)
# strong_model = snowflake_chat_model("claude-opus-4-8", timeout=120, max_retries=2)
#
# Non-Claude, same credential — handy for the Models lesson:
# model = snowflake_chat_model("openai-gpt-5.4", timeout=60, max_retries=2)


# ═════════════════════════════════════════════════════════════════════════════
#  ORIGINAL COURSE PROVIDERS (unchanged, for reference / reverting)
# ═════════════════════════════════════════════════════════════════════════════
#
#   Provider              Install command              Already installed?
#   --------------------  ---------------------------  ---------------------
#   Anthropic             -                            yes (default dep)
#   OpenAI                -                            yes (default dep)
#   Azure OpenAI          uv sync --extra azure        no - install first
#   AWS Bedrock           uv sync --extra bedrock      no - install first
#   Google Vertex/Gemini  uv sync --extra google       no - install first
#
# To revert to the course default, comment out the two Snowflake lines above
# and uncomment these (requires ANTHROPIC_API_KEY in .env):
#
# from langchain.chat_models import init_chat_model
# model = init_chat_model("anthropic:claude-haiku-4-5", timeout=60, max_retries=2)
# strong_model = init_chat_model("anthropic:claude-sonnet-4-6", timeout=120, max_retries=2)

# ─── Other options ───────────────────────────────────────────────────────────
# model = init_chat_model("openai:gpt-4.1-mini")
#
# Groq:   uv add langchain-groq ; GROQ_API_KEY in .env
# model = init_chat_model("groq:llama-3.3-70b-versatile")
#
# Ollama (local, no API key): install https://ollama.com ; ollama pull qwen2.5:7b
# model = init_chat_model("ollama:qwen2.5:7b")
#
# OpenRouter (OpenAI-compatible, like Snowflake above):
# model = ChatOpenAI(model="nvidia/nemotron-3-ultra-550b-a55b:free",
#                    base_url="https://openrouter.ai/api/v1",
#                    api_key=os.environ["OPENROUTER_API_KEY"])
#
# ─── Azure OpenAI ─── uv sync --extra azure
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1", api_version="2024-12-01-preview")
#
# ─── AWS Bedrock ─── uv sync --extra bedrock
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(model_id="anthropic.claude-sonnet-4-6", region_name="us-east-1")
#
# ─── Google Gemini ─── uv sync --extra google
# model = init_chat_model("google_genai:gemini-2.5-flash")
