#!/usr/bin/env python
"""Probe the three Cortex Chat Completions defects, and the paths that avoid them.

Run this to answer one question: *are the defects this course works around still
there?* Each check records the behaviour observed on 2026-09-14 as its baseline and
reports DRIFT when today disagrees. Exit code is non-zero on any drift, so this is
safe to schedule and alert on.

Drift in either direction is worth knowing. If a defect is fixed we can simplify
`models.py`; if a working path breaks, the course breaks with it.

    uv run python probe_cortex_defects.py            # all checks
    uv run python probe_cortex_defects.py --quick    # skip the slow 500-rate check

Needs SNOWFLAKE_ACCOUNT and SNOWFLAKE_PAT (loaded from .env like the lessons).

Host note: account identifiers containing underscores must be written with hyphens
in a URL (`MYORG-MY_ACCOUNT_1` -> `myorg-my-account-1.snowflakecomputing.com`).
The underscored spelling resolves and serves, but its certificate fails Python's
hostname check with `CERTIFICATE_VERIFY_FAILED: Hostname mismatch`. This probe
normalizes; see SNOWFLAKE.md gotcha 5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

load_dotenv()

ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "")
PAT = os.environ.get("SNOWFLAKE_PAT", "")
# Underscores are legal in an account identifier but not in a hostname.
HOST = ACCOUNT.replace("_", "-")
BASE = f"https://{HOST}.snowflakecomputing.com/api/v2/cortex/v1"
CHAT, MESSAGES = f"{BASE}/chat/completions", f"{BASE}/messages"
MODEL = "claude-sonnet-5"

# Behaviour measured on 2026-09-14. See SNOWFLAKE.md gotchas 2, 7, 8.
# Each value is the set of statuses that are NOT drift. The deterministic checks
# admit exactly one; `chat_500` admits two because that defect is probabilistic —
# it failed 10/10 on one captured payload and 1/6 on a rebuilt one, so a clean run
# is entirely normal and is NOT evidence the defect is gone. Pinning it to a single
# expected status would raise a false alarm on roughly half of all runs.
BASELINE: dict[str, tuple[str, ...]] = {
    "chat_parallel_tool_calls": ("defect_present",),
    "messages_parallel_tool_calls": ("works",),
    "chat_cache_control_usage": ("defect_present",),
    "messages_caching": ("works",),
    "chat_500": ("defect_present", "no_failures"),
    "messages_500": ("works",),
}

TIMEOUT = httpx.Timeout(240.0)

# A request that never reached Snowflake tells us nothing about a defect. Track
# transport failures separately so a flat network can never read as a verdict.
TRANSPORT_ERRORS: list[str] = []


def _post(url: str, body: dict, anthropic: bool = False) -> tuple[int, dict]:
    """POST and return (status_code, json). status_code 0 means it never got there."""
    headers = {"Authorization": f"Bearer {PAT}", "Content-Type": "application/json"}
    if anthropic:
        headers["anthropic-version"] = "2023-06-01"
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    except httpx.HTTPError as exc:
        TRANSPORT_ERRORS.append(f"{type(exc).__name__}: {str(exc)[:110]}")
        return 0, {"_transport_error": str(exc)}


def _big_prefix(nonce: str, secret: str) -> str:
    """~18K tokens with a planted fact in the middle, and a unique cold-cache head."""
    filler = "Snowflake Cortex architecture detail. " * 600
    return (
        f"Session {nonce}. Reference material follows.\n{filler}\n"
        f"IMPORTANT FACT: the internal project codename is {secret}.\n{filler}"
    )


# --------------------------------------------------------------------------- 1

_ADD_TOOL_CHAT = [{
    "type": "function",
    "function": {
        "name": "add", "description": "Add two integers.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]},
    },
}]
_ADD_TOOL_MSG = [{
    "name": "add", "description": "Add two integers.",
    "input_schema": {"type": "object",
                     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                     "required": ["a", "b"]},
}]


def check_chat_parallel_tool_calls() -> tuple[str, str]:
    """An assistant turn with 2 tool_calls, both results supplied and paired."""
    code, body = _post(CHAT, {
        "model": MODEL, "tools": _ADD_TOOL_CHAT, "max_completion_tokens": 60,
        "messages": [
            {"role": "user", "content": "Add 2+2 and 10+5."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "add", "arguments": '{"a": 10, "b": 5}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "4"},
            {"role": "tool", "tool_call_id": "c2", "content": "15"},
        ],
    })
    if code == 400 and "toolUse" in json.dumps(body):
        return "defect_present", "400, every call correctly paired"
    if code == 200:
        return "works", "200 — parallel fan-out now accepted"
    return "unexpected", f"HTTP {code} {str(body)[:70]}"


def check_messages_parallel_tool_calls() -> tuple[str, str]:
    """The same fan-out on the Messages API, results coalesced into one user turn."""
    code, _ = _post(MESSAGES, {
        "model": MODEL, "max_tokens": 80, "tools": _ADD_TOOL_MSG,
        "messages": [
            {"role": "user", "content": "Add 2+2 and 10+5."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 2}},
                {"type": "tool_use", "id": "c2", "name": "add", "input": {"a": 10, "b": 5}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "4"},
                {"type": "tool_result", "tool_use_id": "c2", "content": "15"},
            ]},
        ],
    }, anthropic=True)
    return ("works", "200") if code == 200 else ("broken", f"HTTP {code}")


# --------------------------------------------------------------------------- 2

def check_chat_cache_control_usage() -> tuple[str, str]:
    """cache_control on a content part: accepted, but does it misreport usage?

    A cache *read* on a uniquely-nonced prefix is impossible. If one is reported
    while the model still recalls the planted fact, Cortex is reporting the cache
    write as a read and `cached_tokens` cannot be trusted.
    """
    nonce = uuid.uuid4().hex[:8]
    secret = f"ZEPHYR-{nonce.upper()}"
    code, body = _post(CHAT, {
        "model": MODEL, "max_completion_tokens": 40,
        "messages": [
            {"role": "system", "content": [{
                "type": "text", "text": _big_prefix(nonce, secret),
                "cache_control": {"type": "ephemeral"},
            }]},
            {"role": "user",
             "content": "What is the internal project codename? Reply with just the codename."},
        ],
    })
    if code != 200:
        return "unexpected", f"HTTP {code} {str(body)[:70]}"
    usage = body.get("usage") or {}
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    answer = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    recalled = secret in answer.upper()
    if cached > 1000 and prompt < 1000:
        detail = f"cold prefix reported cached={cached:,} prompt={prompt:,}"
        detail += "; fact recalled, so the tokens were read" if recalled else ""
        return "defect_present", detail
    if cached == 0:
        return "no_caching", f"cached=0 prompt={prompt:,} (honest; still no caching here)"
    return "unexpected", f"cached={cached:,} prompt={prompt:,} recalled={recalled}"


def check_messages_caching() -> tuple[str, str]:
    """Real explicit caching on the Messages API: cold write, then a genuine read."""
    nonce = uuid.uuid4().hex[:8]
    body_req = {
        "model": MODEL, "max_tokens": 20,
        "system": [{"type": "text", "text": _big_prefix(nonce, "IRRELEVANT"),
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    }
    reads = []
    for _ in range(2):
        code, body = _post(MESSAGES, body_req, anthropic=True)
        if code != 200:
            return "broken", f"HTTP {code} {str(body)[:70]}"
        reads.append((body.get("usage") or {}).get("cache_read_input_tokens") or 0)
    if reads[1] > 1000:
        return "works", f"call1 read={reads[0]:,} call2 read={reads[1]:,}"
    return "broken", f"no cache read on call 2 (reads={reads})"


# --------------------------------------------------------------------------- 3

def _research_conversation() -> tuple[list[dict], list[dict]]:
    """A 12-message research transcript of the shape that triggers the 500."""
    tools = [
        {"type": "function", "function": {
            "name": "ls", "description": "List files in the workspace.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {
            "name": "internet_search", "description": "Search the web for a topic.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"]}}},
    ]

    def blob(topic: str, n: int = 3136) -> str:
        s = ""
        while len(s) < n:
            s += (f"Title: {topic} outlook {uuid.uuid4().hex[:6]}\n"
                  f"URL: https://example.com/{topic}/{uuid.uuid4().hex[:8]}\n"
                  "Content: Analysts report continued expansion as adoption widens "
                  "across enterprise segments, with platform consolidation and "
                  "AI-driven workloads cited as primary drivers.\n\n")
        return s[:n]

    msgs: list[dict] = [
        {"role": "system", "content": (
            "You are a research supervisor coordinating subagents. Delegate research, "
            "gather findings, synthesize a newsletter, and write results to a file. ") * 4},
        {"role": "user", "content": (
            "Research the music industry across four genres and produce a newsletter "
            "summarizing the key trends for each. ") * 5},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_ls", "type": "function",
             "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_ls", "content": "['notes.md']"},
    ]
    for i, genre in enumerate(("jazz", "rock", "hiphop", "classical")):
        cid = f"call_s{i}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {
                "name": "internet_search",
                "arguments": json.dumps({"query": f"{genre} industry trends", "max_results": 3})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": blob(genre)})
    return msgs, tools


def _to_messages_shape(msgs: list[dict], tools: list[dict]) -> dict:
    system = "".join(m["content"] for m in msgs if m["role"] == "system")
    out: list[dict] = []
    for m in msgs:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for t in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": t["id"], "name": t["function"]["name"],
                               "input": json.loads(t["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": blocks})
        else:
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return {
        "model": MODEL, "max_tokens": 512, "system": system, "messages": out,
        "tools": [{"name": t["function"]["name"],
                   "description": t["function"].get("description", ""),
                   "input_schema": t["function"].get("parameters", {})} for t in tools],
    }


def check_500_rates(samples: int) -> tuple[tuple[str, str], tuple[str, str]]:
    """Sample both surfaces with the same conversation. The 500 is intermittent."""
    msgs, tools = _research_conversation()
    chat_body = {"model": MODEL, "messages": msgs, "tools": tools,
                 "max_completion_tokens": 300}
    msg_body = _to_messages_shape(msgs, tools)

    def rate(url: str, body: dict, anthropic: bool) -> tuple[int, int, list[int]]:
        with ThreadPoolExecutor(max_workers=samples) as ex:
            codes = list(ex.map(lambda _: _post(url, body, anthropic)[0], range(samples)))
        # 0 = never reached Snowflake. Counting those as failures would invent a defect.
        unreached = sum(1 for c in codes if c == 0)
        failed = sum(1 for c in codes if c not in (0, 200))
        return failed, unreached, codes

    chat_fail, chat_unreached, chat_codes = rate(CHAT, chat_body, False)
    msg_fail, msg_unreached, msg_codes = rate(MESSAGES, msg_body, True)

    if chat_unreached or msg_unreached:
        note = f"{chat_unreached + msg_unreached} request(s) never reached Snowflake"
        return ("unreachable", note), ("unreachable", note)

    chat = (("defect_present", f"{chat_fail}/{samples} failed {chat_codes}")
            if chat_fail else ("no_failures", f"0/{samples} — intermittent, may still exist"))
    msg = (("works", f"0/{samples} failed") if not msg_fail
           else ("broken", f"{msg_fail}/{samples} failed {msg_codes}"))
    return chat, msg


# --------------------------------------------------------------------------- run

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the slow 500-rate check")
    ap.add_argument("--samples", type=int, default=6, help="calls per surface for the 500 check")
    args = ap.parse_args()

    if not ACCOUNT or not PAT:
        print("SNOWFLAKE_ACCOUNT and SNOWFLAKE_PAT must be set (see .env.example).")
        return 2

    print(f"Probing Cortex defects on {ACCOUNT}, model={MODEL}\n")
    results: dict[str, tuple[str, str]] = {
        "chat_parallel_tool_calls": check_chat_parallel_tool_calls(),
        "messages_parallel_tool_calls": check_messages_parallel_tool_calls(),
        "chat_cache_control_usage": check_chat_cache_control_usage(),
        "messages_caching": check_messages_caching(),
    }
    if args.quick:
        print("(--quick: skipping the 500-rate check)\n")
    else:
        chat_500, msg_500 = check_500_rates(args.samples)
        results["chat_500"] = chat_500
        results["messages_500"] = msg_500

    print(f"{'check':32}{'accepted':30}{'observed':15}detail")
    drift = []
    for name, (status, detail) in results.items():
        accepted = BASELINE.get(name, ())
        ok = status in accepted
        flag = "" if ok else "   <-- DRIFT"
        if not ok:
            drift.append((name, "|".join(accepted), status))
        print(f"{name:32}{'|'.join(accepted):30}{status:15}{detail}{flag}")

    print()
    # Distinguish "the network was down" from "the defects changed". Reporting drift
    # off unreachable requests would be a false alarm every time the VPN drops.
    if TRANSPORT_ERRORS:
        print(f"INCONCLUSIVE — {len(TRANSPORT_ERRORS)} request(s) never reached Snowflake.")
        for err in dict.fromkeys(TRANSPORT_ERRORS[:3]):
            print(f"  {err}")
        print("\nNo verdict on any defect. Check the network, the account identifier, and\n"
              "that SNOWFLAKE_PAT is unexpired, then re-run.")
        return 2

    if not drift:
        skipped = " (500-rate check skipped)" if args.quick else ""
        print(f"No drift — every defect and every workaround behaves as recorded{skipped}.")
        return 0
    print("DRIFT DETECTED:")
    for name, expected, got in drift:
        print(f"  {name}: expected {expected}, observed {got}")
    print("\nA fixed defect means models.py can be simplified; a broken workaround means\n"
          "the course is broken. Either way, re-read SNOWFLAKE.md and update the baseline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
