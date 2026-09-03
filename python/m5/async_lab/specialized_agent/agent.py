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
from langchain_openai import ChatOpenAI

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


class _CortexChat(ChatOpenAI):
    """Minimal inline copy of python/models.py::SnowflakeCortexChat.

    Duplicated rather than imported ON PURPOSE. This deployment declares
    "dependencies": ["."] so python/ is NOT on sys.path at runtime — importing
    ../../../models.py would work in a plain shell and then fail under
    `langgraph dev`, which is the whole point of the isolation this lab teaches.

    Cortex's Claude path rejects a follow-up request whose assistant turn holds
    2+ tool calls, so split such a turn into consecutive single-call pairs. Full
    explanation and evidence in SNOWFLAKE.md at the repo root.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        msgs, out, i = payload.get("messages") or [], [], 0
        while i < len(msgs):
            m = msgs[i]
            calls = m.get("tool_calls") if m.get("role") == "assistant" else None
            if not calls or len(calls) <= 1:
                out.append(m)
                i += 1
                continue
            j, results = i + 1, {}
            while j < len(msgs) and msgs[j].get("role") == "tool":
                results[msgs[j].get("tool_call_id")] = msgs[j]
                j += 1
            if not results:
                out.append(m)
                i += 1
                continue
            for pos, call in enumerate(calls):
                turn = dict(m)
                turn["tool_calls"] = [call]
                if pos > 0:
                    turn["content"] = None
                out.append(turn)
                if (ans := results.pop(call.get("id"), None)) is not None:
                    out.append(ans)
            out.extend(results.values())
            i = j
        if out:
            payload["messages"] = out
        return payload


_account = os.environ["SNOWFLAKE_ACCOUNT"]
model = _CortexChat(
    model="claude-haiku-4-5",
    base_url=f"https://{_account}.snowflakecomputing.com/api/v2/cortex/v1",
    api_key=os.environ["SNOWFLAKE_PAT"],
    # Cortex rejects max_tokens in favour of max_completion_tokens, so leave it unset.
    profile={"max_input_tokens": 200_000, "max_output_tokens": 64_000, "tool_calling": True},
)
# langgraph.json points at this module-level variable: "./agent.py:graph"
graph = create_deep_agent(model=model, tools=[analyze_sales])
