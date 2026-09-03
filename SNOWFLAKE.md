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

## Gotchas worth knowing

Measured on 2026-09-03, not taken from docs. See the `models.py` docstring for the
full detail and evidence.

**1. `max_tokens` is rejected.** HTTP 400, *"max_tokens is deprecated in favor of
max_completion_tokens"* — even though Snowflake's own quickstart still shows
`max_tokens`. `models.py` therefore never sets it; pass
`model_kwargs={"max_completion_tokens": N}` if you need a cap.

**2. Parallel tool calls break on the Claude models.** Claude-on-Cortex will emit
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

**7. Prompt caching only works on the `openai-*` models.** Resending an identical
~12,300-token system prefix three times:

| | `claude-sonnet-5` | `openai-gpt-5-mini` |
|---|---|---|
| `usage.prompt_tokens_details.cached_tokens` | `0` every call | `8,064` of 8,140 |
| LangChain `input_token_details.cache_read` | `0` | `8,064` |

Cortex does implement prompt caching and reports it correctly — Snowflake is not
truncating the field, it is present and reaches LangSmith, it is simply zero on
Claude. Two independent reasons Claude shows nothing: Cortex reports
`cached_tokens = 0` for it, and Anthropic-style *explicit* caching via
`cache_control` never fires because `langchain_anthropic`'s prompt-caching
middleware (which deepagents loads) no-ops for a `ChatOpenAI` subclass — verified
as zero `cache_control` occurrences across 17 captured payloads. What `openai-*`
provides is OpenAI-style *implicit* caching, needing no markers.

Also: the cache engaged on the **third** identical call, not the second as
Module 1.4 states — calls 1 and 2 both reported `cache_read=0`. Comparing only
two runs makes it look broken. `m1/m1.4_prompt_caching.py` pins its own models
and shows both paths, so you never have to edit `models.py` for that exercise.

Cost caveat: the token accounting is measured, but whether Cortex *bills* cached
input at a discount is unverified — check
`SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY`.

**8. `m4.2` (newsletter) is KNOWN BROKEN — a Cortex-side HTTP 500.** The
subagent-team lab hits a deterministic `500 {'message': 'internal error'}` with a
request id. Ruled out: timeout, whole-request size (an 800KB user message is
fine), the parallel tool-call bug, `cache_control`, tool-schema shape, encoding,
and tool-result size. It correlates with accumulated conversation growth and the
threshold varies by model — `claude-sonnet-5` fails within ~2 minutes,
`claude-opus-5` survives ~57 model calls over ~25 minutes and then fails the same
way. `openai-*` clears the 500 but Azure's content filter then rejects the
music-news results, which is not tunable. No client-side fix found; this needs
reporting to Snowflake with the request ids. The other three Module 4 lessons
pass, and `m4.3_run_manuscript.py` demonstrates the same subagent-team concept
with 60 subagents, so nothing is lost pedagogically.

## Verification status
Verified 2026-09-03 against `claude-sonnet-5` / `claude-opus-5`. **Zero
`toolUse`/`toolResult` rejections anywhere**, including the heaviest delegation
path in the course (`m4.3_run_manuscript.py`, 60 subagents in 2 rounds of 30).

| Module | Result |
|---|---|
| 1 | 13/13 pass — incl. remote MCP (`docs.langchain.com`, `deepwiki`) and all three HITL interrupts |
| 2 | 5/5 runnable pass; m2.3 needs LangSmith (see below) |
| 3 | 6/6 pass after the profile fix above |
| 4 | 3 pass; m4.2 newsletter KNOWN BROKEN (gotcha 8, Cortex 500) |
| 5 | 8/8 graphs import; 5/8 invoke — remainder need a local MCP server or LangSmith |

Lessons needing keys this setup deliberately omits — **none are Cortex issues**:

- **LangSmith** (`LANGSMITH_API_KEY`): m2.3's three sandbox scripts construct
  `SandboxClient()` at module top level, so they 401 against
  `api.smith.langchain.com` before any request reaches Cortex. Also
  `m5/sales_assistant_sandbox`.
- **Tavily** (`TAVILY_API_KEY`): `m4_2_newsletter_agent.py` raises at import and
  does **not** degrade gracefully; `m4.2_run_newsletter.py` inherits that. By
  contrast `m5/sales_assistant` gates on the key and degrades cleanly.
- **A local MCP server**: `m5/sales_assistant` expects the mock-mail MCP that its
  `start.sh` launches on port 5002.

`m5/async_lab/specialized_agent` is repointed separately: it is an isolated
deployment (`"dependencies": ["."]`, own `pyproject.toml`), so `python/` is not on
`sys.path` under `langgraph dev` and it cannot import `models.py`. The Cortex
client and the tool-call fix are inlined there on purpose.

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
