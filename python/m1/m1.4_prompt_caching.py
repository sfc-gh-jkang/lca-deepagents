# python/m1/m1.4_prompt_caching.py
"""M1.4 (optional) — Prompt caching on Snowflake Cortex.

The lesson asks you to run an agent twice and compare the `usage` field: the
second run should show a chunk of the input tokens read from cache instead of
processed fresh. That works on Cortex — on BOTH Claude and OpenAI models — but
only if you reach Claude through the right API. This script pins its own models
so you never have to edit models.py to do the exercise.

MEASURED (2026-09-03, this account), identical ~12,300-token system prefix:

    claude-sonnet-5   via Messages API (/v1/messages)         call 2: 14,371 of 14,455 cached
    openai-gpt-5-mini via Chat Completions (/chat/completions) call 2:  8,064 of  8,140 cached
    claude-sonnet-5   via Chat Completions                    never caches, always 0

WHY THE THIRD ROW IS ZERO — AND WHY IT IS NOT SNOWFLAKE TRUNCATING ANYTHING
The two providers cache by different mechanisms:

  * OpenAI caching is IMPLICIT. Send a large enough stable prefix twice and it
    caches. Nothing in the request asks for it.
  * Anthropic caching is EXPLICIT. It only happens if the request carries
    `cache_control` markers on the blocks you want cached. No markers, no cache.

The OpenAI Chat Completions schema has nowhere to put those markers. So driving
Claude through /chat/completions means you cannot ask for caching at all, and
Cortex correctly reports `cached_tokens: 0`. The field is present and reaches
LangSmith — it is genuinely zero, not missing.

Switch Claude to Cortex's Anthropic-compatible Messages API and the markers
become expressible, and Cortex honours them. models.py now routes all Claude
models that way by default (`snowflake_anthropic_model`).

TWO GOTCHAS ON THE MESSAGES API PATH
  1. Cortex rejects a TOP-LEVEL `cache_control` on the request body with
     400 "cache_control: Extra inputs are not permitted". langchain_anthropic's
     prompt-caching middleware sets it there in addition to the system blocks and
     tools, so models.py strips just that one key. System- and tool-level markers
     are accepted and are what earn the cache hits.
  2. There is a minimum cacheable prefix (~1024-2048 tokens depending on model).
     A small agent prompt will report 0 cached tokens and that is correct
     behaviour, not a failure. Hence the deliberately large prefix below.

CAVEAT ON COST
The lesson says cached tokens cost "a fraction of the price". The token
accounting below is measured and real. Whether Cortex *bills* cached input at a
discount is NOT verified here — Cortex meters credits via
SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY, which is where you would
confirm the pricing claim.

RUN
  uv run python m1/m1.4_prompt_caching.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from models import snowflake_anthropic_model, snowflake_chat_model

# A large, STABLE prefix is the precondition for caching. In a real agent this
# bulk comes from your system prompt plus tool definitions plus skills.
SYSTEM_PROMPT = (
    "You are a meticulous assistant for a music distribution company. "
    "Follow these standing instructions exactly on every turn. "
) * 400


@tool
def get_genre_sales(genre: str) -> str:
    """Look up last quarter's sales for a music genre."""
    return f"{genre}: 12,400 units"


def report(model_name: str, via: str) -> None:
    """Invoke the same prompt three times and show what got cached.

    via="messages" -> Anthropic Messages API (Claude, explicit cache_control)
    via="chat"     -> OpenAI Chat Completions (openai-*, implicit caching)
    """
    if via == "messages":
        model = snowflake_anthropic_model(model_name, timeout=300, max_retries=2)
        # Explicit: mark the stable prefix, or nothing is cached. Inside
        # create_deep_agent the middleware adds these for you; calling the model
        # directly, we add one ourselves.
        messages = [
            SystemMessage(content=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }]),
            HumanMessage(content="Reply with exactly: OK"),
        ]
    else:
        model = snowflake_chat_model(model_name, timeout=180, max_retries=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Reply with exactly: OK"},
        ]

    bound = model.bind_tools([get_genre_sales])
    print(f"\n{model_name}  (via {via})")
    print(f"  {'call':<6}{'input_tokens':>14}{'cache_read':>13}{'cache_creation':>16}{'fresh':>9}")
    cached = total = 0
    for i in (1, 2, 3):
        result = bound.invoke(messages)
        usage = result.usage_metadata or {}
        details = usage.get("input_token_details", {}) or {}
        total = usage.get("input_tokens") or 0
        cached = details.get("cache_read") or 0
        created = details.get("cache_creation") or 0
        print(f"  {i:<6}{total:>14,}{cached:>13,}{created:>16,}{total - cached:>9,}")

    if cached:
        print(f"  -> {100 * cached / total:.0f}% of input tokens read from cache on the last call")
    else:
        print("  -> nothing cached (see the gotchas in this file's docstring)")


if __name__ == "__main__":
    print(f"Stable system prefix: {len(SYSTEM_PROMPT):,} chars (~{len(SYSTEM_PROMPT)//4:,} tokens)")

    # Claude — the course default, through the Messages API.
    report("claude-sonnet-5", via="messages")

    # OpenAI — implicit caching, no markers needed, for contrast.
    report("openai-gpt-5-mini", via="chat")

    print(
        "\nBoth paths show the effect the lesson describes, and both engage on the\n"
        "SECOND call exactly as it says. The difference is how you ask: Claude needs\n"
        "explicit cache_control markers on the Messages API, OpenAI needs nothing at\n"
        "all. The same numbers appear in the LangSmith `usage` field."
    )
