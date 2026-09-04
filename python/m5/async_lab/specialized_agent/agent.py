# python/m5/async_lab/specialized_agent/agent.py
"""M5.4 Lab: the "specialized" deployment.

THE IDEA
This is a second, independent deployment, not a folder inside the shared
course environment. It has its own pyproject.toml and its own model setup,
so pandas (needed for the analysis tool below) is installed only here,
never in the shared environment every other lab in the course depends on.
Its langgraph.json also declares "dependencies": ["."] instead of the usual
["../.."], which is what makes that isolation real.

RUN
  cd python/m5/async_lab/specialized_agent
  uv run langgraph dev --port 2025
Leave this running, then start ../main_agent in a second terminal. The main
agent reaches this one over HTTP at http://127.0.0.1:2025, exactly like any
other remote deployment.
"""

import os
import time

import pandas as pd
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

SALES = pd.DataFrame(
    {
        "region": ["West", "West", "East", "East", "Central", "Central"],
        "product": ["Widget", "Gadget", "Widget", "Gadget", "Widget", "Gadget"],
        "units_sold": [1200, 850, 980, 1400, 630, 720],
        "revenue": [36000, 42500, 29400, 70000, 18900, 36000],
    }
)


@tool
def analyze_sales(group_by: str = "region") -> str:
    """Run a full sales breakdown, grouped by "region" or by "product", ranked highest revenue first.

    This is a slow, heavyweight analysis job, not something you'd want
    blocking the main agent's own model calls.
    """
    time.sleep(20)  # stands in for a genuinely slow job (a big pandas pipeline, a model call, etc.)
    key = group_by if group_by in ("region", "product") else "region"
    grouped = SALES.groupby(key)[["units_sold", "revenue"]].sum().sort_values("revenue", ascending=False)
    lines = [
        f"{name}: ${int(row['revenue']):,} revenue, {int(row['units_sold']):,} units"
        for name, row in grouped.iterrows()
    ]
    return "\n".join(lines)


class _CortexChat(ChatAnthropic):
    """Minimal inline copy of python/models.py::CortexChatAnthropic.

    Duplicated rather than imported ON PURPOSE. This deployment declares
    "dependencies": ["."] so python/ is NOT on sys.path at runtime — importing
    ../../../models.py would work in a plain shell and then fail under
    `langgraph dev`, which is the whole point of the isolation this lab teaches.

    Claude on Cortex goes through the Anthropic Messages API, not the
    OpenAI-compatible Chat Completions endpoint: on Chat Completions, parallel
    tool calls are rejected and prompt caching is impossible. The only
    incompatibility on this path is a TOP-LEVEL cache_control, which Cortex
    rejects with 400 "Extra inputs are not permitted" — so strip that one key.
    Full evidence in SNOWFLAKE.md at the repo root.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("cache_control", None)
        return payload


_account = os.environ["SNOWFLAKE_ACCOUNT"]
model = _CortexChat(
    model="claude-haiku-4-5",
    base_url=f"https://{_account}.snowflakecomputing.com/api/v2/cortex",
    api_key="unused-snowflake-uses-bearer-header",
    default_headers={"Authorization": f"Bearer {os.environ['SNOWFLAKE_PAT']}"},
    max_tokens=8192,
    timeout=120,
)

# langgraph.json points at this module-level variable: "./agent.py:graph"
graph = create_deep_agent(model=model, tools=[analyze_sales])
