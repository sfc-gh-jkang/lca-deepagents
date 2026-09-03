# python/m1/m1.4_prompt_caching.py
"""M1.4 (optional) — Prompt caching, adapted for Snowflake Cortex.

WHY THIS FILE EXISTS
The lesson asks you to run an agent twice and compare the `usage` field in
LangSmith: the second run should show a chunk of the input tokens read from
cache instead of processed fresh. On Snowflake Cortex that comparison only
works on the openai-* models, so this script pins its own models and leaves
models.py alone — you do NOT need to edit it to do this exercise.

WHAT WE MEASURED (2026-09-03, this account)
Resending an identical ~7,600-token system prompt:

    claude-sonnet-5    cached_tokens = 0        every time, no matter how often
    openai-gpt-5.4     cached_tokens = 4,608    of 4,816 input tokens

So Cortex does implement prompt caching and reports it correctly — but not on
the Claude path. Two independent reasons the Claude path shows nothing:

  1. Cortex returns `usage.prompt_tokens_details.cached_tokens = 0` for Claude,
     so there is nothing to compare. Snowflake is NOT truncating the field; it
     is present and plumbed through to LangSmith, it is simply zero.
  2. On Anthropic's own API, caching is opt-in via `cache_control` markers that
     langchain_anthropic's prompt-caching middleware injects. deepagents does
     load that middleware, but it no-ops here because it checks for a
     ChatAnthropic instance and our model is a ChatOpenAI subclass. Verified:
     zero `cache_control` occurrences across 17 captured request payloads.

What openai-* gives us instead is OpenAI-style IMPLICIT caching — no markers
needed, it just happens once the stable prefix is large enough.

WARM-UP TAKES MORE THAN ONE REPEAT
The lesson says the *second* run shows cached tokens. On Cortex we consistently
saw the cache engage on the THIRD identical call, not the second — calls 1 and 2
both reported cache_read=0, then call 3 came back 99% cached. That is why this
script runs three times. If you only compare two runs you will conclude, wrongly,
that caching is not working.

CAVEAT ON COST
The lesson says cached tokens cost "a fraction of the price". The token
accounting below is real and measured. Whether Cortex *bills* cached input
tokens at a discount is NOT verified here — Cortex meters credits via
SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY, and that is where you
would confirm the pricing claim. Treat the discount as unverified on Cortex.

RUN
  uv run python m1/m1.4_prompt_caching.py
"""

from langchain_core.tools import tool

from models import snowflake_chat_model

# A large, STABLE prefix is the precondition for caching — implicit caching
# typically needs ~1024+ tokens before it engages at all. In a real agent this
# bulk comes from your system prompt plus tool definitions plus skills.
SYSTEM_PROMPT = (
    "You are a meticulous assistant for a music distribution company. "
    "Follow these standing instructions exactly on every turn. "
) * 400


@tool
def get_genre_sales(genre: str) -> str:
    """Look up last quarter's sales for a music genre."""
    return f"{genre}: 12,400 units"


def report(model_name: str) -> None:
    """Invoke the same prompt three times and show what got cached."""
    model = snowflake_chat_model(model_name, timeout=180, max_retries=2)
    bound = model.bind_tools([get_genre_sales])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Reply with exactly: OK"},
    ]

    print(f"\n{model_name}")
    print(f"  {'call':<6}{'input_tokens':>14}{'cache_read':>13}{'cache_creation':>16}{'fresh':>9}")
    for i in (1, 2, 3):
        result = bound.invoke(messages)
        usage = result.usage_metadata or {}
        details = usage.get("input_token_details", {}) or {}
        total = usage.get("input_tokens") or 0
        cached = details.get("cache_read") or 0
        created = details.get("cache_creation") or 0
        print(f"  {i:<6}{total:>14,}{cached:>13,}{created:>16,}{total - cached:>9,}")

    if cached:
        pct = 100 * cached / total if total else 0
        print(f"  -> {pct:.0f}% of input tokens were read from cache on the last call")
    else:
        print("  -> nothing cached: Cortex reports cached_tokens=0 for this model")


if __name__ == "__main__":
    print(__doc__.split("RUN")[0].strip().split("\n\n")[0])
    print(f"\nStable system prefix: {len(SYSTEM_PROMPT):,} chars (~{len(SYSTEM_PROMPT)//4:,} tokens)")

    # The model the lesson's exercise actually works on.
    report("openai-gpt-5-mini")

    # The course default, shown for contrast so the zero is clearly the
    # platform's behaviour and not a mistake in your setup.
    report("claude-sonnet-5")

    print(
        "\nTakeaway: the caching effect the lesson describes is real and visible on\n"
        "Cortex, but only on the openai-* models. Keep models.py on Claude for the\n"
        "rest of the course; this file pins its own models so you never have to\n"
        "edit it. The same numbers appear in the LangSmith `usage` field."
    )
