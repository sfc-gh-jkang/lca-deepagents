# Running this course on Snowflake Cortex

This fork points the course's LLM inference at **Snowflake Cortex** instead of a
direct Anthropic or OpenAI key. Inference runs inside the Snowflake perimeter and
bills as Snowflake credits, so no third-party model key is needed.

Upstream: <https://github.com/langchain-ai/lca-deepagents>

## What changed vs upstream

Three files. Everything else is untouched, and every lesson works unmodified
because the `model` / `strong_model` names are preserved.

| File | Change |
|---|---|
| `python/models.py` | `ChatOpenAI` pointed at Cortex's OpenAI-compatible endpoint, plus `SnowflakeCortexChat` (see *Parallel tool calls* below) |
| `python/.env.example` | `SNOWFLAKE_ACCOUNT` / `SNOWFLAKE_PAT` block; model-provider keys commented out |
| `python/env_utils.py` | secret masking widened beyond `*API_KEY` so `SNOWFLAKE_PAT` isn't printed in cleartext |

## Setup

```bash
cd python
cp .env.example .env       # then fill in SNOWFLAKE_ACCOUNT + SNOWFLAKE_PAT
uv sync
uv run python env_utils.py # verify
```

Cortex exposes an OpenAI-compatible endpoint, so the only real change is a
`base_url` swap:

```
https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1
```

Auth is a Snowflake [programmatic access token](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens)
(PAT). Create a dedicated service user rather than using your own login — the
token then can only run inference, not touch data:

```sql
CREATE ROLE LCA_DEEPAGENTS_RL;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_REST_API_USER TO ROLE LCA_DEEPAGENTS_RL;

CREATE USER LCA_DEEPAGENTS_SVC TYPE = SERVICE DEFAULT_ROLE = LCA_DEEPAGENTS_RL;
GRANT ROLE LCA_DEEPAGENTS_RL TO USER LCA_DEEPAGENTS_SVC;

-- service users must be subject to a network policy to use a PAT
CREATE NETWORK POLICY LCA_DEEPAGENTS_NP ALLOWED_IP_LIST = ('0.0.0.0/0');
ALTER USER LCA_DEEPAGENTS_SVC SET NETWORK_POLICY = LCA_DEEPAGENTS_NP;

ALTER USER LCA_DEEPAGENTS_SVC ADD PROGRAMMATIC ACCESS TOKEN LCA_COURSE_PAT
  ROLE_RESTRICTION = 'LCA_DEEPAGENTS_RL' DAYS_TO_EXPIRY = 90;
```

`SNOWFLAKE.CORTEX_REST_API_USER` grants the REST API only — not Cortex Analyst,
Search, or the AI functions. Note the REST API uses the user's **default role**,
so it must be set. Widen `ALLOWED_IP_LIST` to taste.

## Two API surfaces — pick the right one

Cortex exposes the same models through two schemas, and the choice matters more
than the model choice:

| | Chat Completions `/v1/chat/completions` | Messages API `/v1/messages` |
|---|---|---|
| Shape | OpenAI | Anthropic |
| Models | all | Claude only |
| LangChain class | `ChatOpenAI` | `ChatAnthropic` |
| Parallel tool calls on Claude | **broken** (gotcha 4) | **work natively** |
| Prompt caching on Claude | **impossible** (gotcha 7) | **works** |
| `m4.2` newsletter lab | **HTTP 500** | **passes** |

`models.py` therefore routes **Claude through the Messages API**
(`snowflake_anthropic_model`) and keeps Chat Completions
(`snowflake_chat_model`) for the `openai-*` models. Three separate defects
disappear as a result. If you only take one thing from this file: do not drive
Claude on Cortex through Chat Completions.

## Gotchas worth knowing

Measured on 2026-09-03, not taken from docs. See the `models.py` docstring for the
full detail and evidence.

**1. `max_tokens` is rejected.** HTTP 400, *"max_tokens is deprecated in favor of
max_completion_tokens"* — even though Snowflake's own quickstart still shows
`max_tokens`. `models.py` therefore never sets it; pass
`model_kwargs={"max_completion_tokens": N}` if you need a cap.

**2. Parallel tool calls break on Claude — ON CHAT COMPLETIONS ONLY.** The
Messages API handles them natively, which is why `models.py` uses it. The rest of
this entry applies only if you deliberately drive Claude via `snowflake_chat_model`.
 Claude-on-Cortex will emit
an assistant turn with 2+ `tool_calls`, then reject the follow-up request carrying
those results:

```
HTTP 400  Each 'toolUse' block must be accompanied with a matching 'toolResult' block.
```

This fires even when every call *is* correctly paired, in every message shape
tested. `parallel_tool_calls: false` is accepted (HTTP 200) and then ignored. It
breaks `deepagents` outright, since subagent delegation goes through a `task` tool
and the framework batches routinely.

The constraint is one tool block per assistant **turn**, not per request — so
`SnowflakeCortexChat` rewrites only the *outbound history*, splitting a batched
turn into consecutive single-call pairs:

```
assistant[A,B] + tool(A) + tool(B)               -> 400
assistant[A] + tool(A) + assistant[B] + tool(B)  -> 200
```

Parallel execution is preserved (the model still emits batches, LangGraph still
runs them concurrently); only the transcript shape changes. `coerce_single_tool_call=True`
remains as a blunt fallback that drops extra calls and works sequentially.

The `toolUse`/`toolResult` wording is AWS Bedrock Converse vocabulary, and the
Claude responses return empty `id` and empty `finish_reason` while `openai-*`
responses carry a real `chatcmpl-…` id and an Azure-style `moderation` key —
consistent with the two model families sitting behind different providers. That
is inference from response shape, not confirmed internals.

**3. Claude returns an empty `finish_reason`.** Anything branching on
`finish_reason == "tool_calls"` will mis-route. `langchain-openai` is fine because
it inspects the `tool_calls` array instead.

**4. Non-Claude tool-calling support varies.** `openai-gpt-5.4` and
`openai-gpt-5-mini` handle parallel tool calls natively (both workarounds
auto-disable for `openai-*`). `llama4-maverick` and `mistral-large2` return
*"tool calling is not supported for this model"*, so they cannot drive an agent.

**5. Underscored account identifiers fail Python's TLS hostname check.** An
account like `MYORG-MY_ACCOUNT_1` raises
`CERTIFICATE_VERIFY_FAILED: Hostname mismatch` under `urllib`/`ssl`, while `curl`
and `httpx` accept it. The course works because `langchain-openai` uses `httpx`;
a plain `urllib` script against the same URL will not.

**6. `model.profile` is None unless you seed it.** LangChain resolves `.profile`
from a provider+model registry. A `ChatOpenAI` subclass on a custom `base_url`
has no entry, so `.profile` is `None` — and the Module 3 summarization lessons do:

```python
model.profile = {**model.profile, "max_input_tokens": 700}
```

which raises `TypeError: 'NoneType' object is not a mapping` at import. `models.py`
therefore seeds a profile from `_CORTEX_MODEL_LIMITS`, using **Snowflake's**
documented context windows rather than Anthropic's — they disagree, e.g. Snowflake
serves `claude-sonnet-5` at 1M context where LangChain's registry lists 200K for
the 4-5 generation. Anything that introspects the context window depends on this.

**7. Prompt caching on Claude requires the Messages API.** Identical
~12,300-token stable prefix, `cache_read` on the second call:

| path | result |
|---|---|
| `claude-sonnet-5` via **Messages API** + `cache_control` | **14,371 of 14,455 cached** |
| `openai-gpt-5-mini` via Chat Completions (implicit) | **8,064 of 8,140 cached** |
| `claude-sonnet-5` via Chat Completions | `0`, always |

Snowflake is not truncating the field — it is present and reaches LangSmith, it is
genuinely zero on that third path. The reason: OpenAI caching is *implicit* (no
request-side expression needed) while Anthropic caching is *explicit*, requiring
`cache_control` markers that the OpenAI Chat Completions schema has nowhere to
put. Both working paths engage on the **second** call, exactly as Module 1.4 says.

Sub-gotcha: Cortex rejects a **top-level** `cache_control` on the request body
with `400 "cache_control: Extra inputs are not permitted"`.
`langchain_anthropic`'s prompt-caching middleware sets it there *in addition to*
the system blocks and tools. Bisected on a real captured payload, removing the
top-level key alone fixes it; the system- and tool-level markers are accepted and
are what earn the hits. `models.py` strips only that key (`CortexChatAnthropic`).

Also note a minimum cacheable prefix (~1024-2048 tokens by model). A small agent
prompt reporting 0 cached tokens is correct, not broken.
`m1/m1.4_prompt_caching.py` demonstrates both paths without touching `models.py`.

Cost caveat: the token accounting is measured; whether Cortex *bills* cached input
at a discount is unverified — check `CORTEX_REST_API_USAGE_HISTORY`.

**8. `m4.2` (newsletter) 500s on Chat Completions — FIXED by the Messages API.**
The subagent-team lab hit a deterministic `500 {'message': 'internal error'}` on
Chat Completions that resisted every client-side fix: not a timeout, not
whole-request size (an 800KB user message is fine), not the parallel tool-call
bug, not `cache_control`, not tool-schema shape, not encoding, not tool-result
size. Different models had different thresholds (`claude-sonnet-5` failed in ~2
minutes, `claude-opus-5` lasted ~57 calls over ~25 minutes). Switching Claude to
the Messages API cleared it outright — the lab now completes with `EXIT=0`,
writing the newsletter plus all four researcher archives.

## Verification status

Re-verified 2026-09-04 **on the Messages API transport** (an earlier pass on Chat
Completions is superseded). Across 32 scripts and 14 run logs: **zero**
`cache_control` rejections, **zero** `max_tokens` errors, **zero**
`NoneType ... not a mapping`, **zero** Cortex 400s, **zero** Cortex 500s.

| Module | Result |
|---|---|
| 1 | 12/12 pass — incl. remote MCP (`docs.langchain.com`, `deepwiki`), both HITL interrupts, and the new caching lesson |
| 2 | 5/5 runnable pass; m2.3 blocked (see below) |
| 3 | 6/6 pass |
| 4 | 5/5 pass — including `m4.2_run_newsletter.py`, ~20 min |
| 5 | 4/4 invokable pass; 2 import-only, blocked on a local MCP server / LangGraph runtime injection |

The caching workarounds are genuinely exercised rather than merely absent: every
`create_deep_agent` script loads `langchain_anthropic`'s prompt-caching
middleware, which now really fires against a real `ChatAnthropic`.

Two caveats worth knowing, neither transport-related:

- **`m4.2` writes only 3 of its 4 `research/<genre>/` archives**, non-deterministically.
  Confirmed by mtime, not by existence check: on one run Rock/Metal/Alternative were
  fresh while `research/Latin/sources.md` was stale from an earlier run. The Latin
  researcher does run — its segment appears in the newsletter — its raw `sources.md`
  just does not always land in the agent `files` channel the host script mirrors from.
  `newsletter.html` itself is complete, with all four genre sections.
- **`m2.3` is blocked by an ORG ENTITLEMENT, not a missing key.** With
  `LANGSMITH_API_KEY` set it returns `403 Forbidden` →
  `SandboxAuthenticationError: Sandbox feature is not enabled for this organization`.
  The key is accepted; rotating it will not help. Same for `m5/sales_assistant_sandbox`.

Other lessons needing things this setup does not have — **none are Cortex issues**:

- **A local MCP server**: `m5/sales_assistant` expects the mock-mail MCP its
  `start.sh` launches on port 5002.
- **LangGraph server runtime**: `m5/sales_assistant_sandbox`'s `make_graph` needs
  `config`/`runtime` injected by `langgraph dev`; it cannot be invoked in-process.

`m5/async_lab/specialized_agent` is an isolated deployment (`"dependencies": ["."]`,
own `pyproject.toml`) so `python/` is not on `sys.path` under `langgraph dev`. It
carries a deliberate inline copy of the Messages-API client rather than importing
`models.py`.

Bare `*_homework.py` files are intentionally incomplete student templates and are
not verification targets; `*_homework_filled.py` are the solutions.

## Models

Verified callable, newest of each family:

```
claude-opus-5      <- strong_model
claude-sonnet-5    <- model
claude-haiku-4-5   <- latest Haiku; there is no haiku-5
openai-gpt-5.4     <- latest GPT here; no workaround needed
```

Models outside your own region need
[`CORTEX_ENABLED_CROSS_REGION`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference).

## Cost

Cortex REST API usage is **not** written to `AI_OBSERVABILITY_EVENTS`. Query:

```sql
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY
ORDER BY START_TIME DESC;
```

## Licensing

Upstream `langchain-ai/lca-deepagents` carries **no LICENSE file**, so its course
content is all-rights-reserved by default. This is a GitHub fork, which is how
that content is legally carried here — do not extract it into a standalone repo.
Only this file and the changes listed in the table above are original work.
