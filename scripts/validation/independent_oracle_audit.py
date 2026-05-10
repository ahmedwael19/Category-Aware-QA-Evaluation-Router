"""Construct-validity audit of the symbolic ground truth.

The router's symbolic execution path and the dataset ground truth both rely
on `evaluation/deterministic.py`: a bug in those functions would be invisible
to end-to-end evaluation because the "system" and the "ground truth" would
see the same bug.

This script recomputes symbolic ground truth on the 305-pair benchmark via
four independent implementations, none of which import
`evaluation/deterministic.py`, `evaluation/system_runner.py`, or any router
module:

    Method A   dateutil.parser.isoparse + list comprehensions
    Method B   datetime.strptime + index-based loops
    Method C   pandas DataFrame operations
    Method D   GPT-4o tool-calling, with Method-A implementations as tools

Agreement between each method and the dataset-shipped ground truth is
reported per category, per subcategory, and overall; disagreements are saved
for inspection. If all four methods agree, the "same code defines truth and
execution" objection is closed on the agreement subset.

Reads:  data/evaluation_dataset.csv
Writes: results/independent_oracle/

Usage:
    python -m scripts.validation.independent_oracle_audit
    python -m scripts.validation.independent_oracle_audit --skip-llm
    python -m scripts.validation.independent_oracle_audit --sample 30
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time as timer
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from config import ROOT
from dateutil.parser import isoparse


EVAL_DATA = ROOT / "data" / "evaluation_dataset.csv"
OUT_DIR = ROOT / "results" / "independent_oracle"

log = logging.getLogger("ind-oracle")


# ---------------------------------------------------------------------------
# 1. Independent conversation parser
#    No dataclasses or helpers imported from evaluation/. Flat dicts only.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Msg:
    role: str
    text: str
    ts_iso: str
    is_public: bool
    channel: str


@dataclass(frozen=True, slots=True)
class _Conv:
    messages: tuple[_Msg, ...]
    tags: tuple[str, ...]
    channel: str
    resolution_time_minutes: float | None


def parse_conv(raw: str) -> _Conv:
    """Parse conversation_json into an independent local dataclass."""
    d = json.loads(raw)
    msgs = tuple(
        _Msg(
            role=m["role"],
            text=m["text"],
            ts_iso=m["timestamp"],
            is_public=bool(m.get("is_public", True)),
            channel=m.get("channel", d.get("channel", "")),
        )
        for m in d["messages"]
    )
    rt = d.get("resolution_time_minutes")
    return _Conv(
        messages=msgs,
        tags=tuple(d.get("tags", [])),
        channel=d.get("channel", ""),
        resolution_time_minutes=float(rt) if rt is not None else None,
    )


# ---------------------------------------------------------------------------
# 2. Parameter normalisation (shared by Methods A, B, C, D)
# ---------------------------------------------------------------------------

def _normalise_threshold_minutes(threshold: float, unit: str) -> float:
    u = (unit or "minutes").lower().strip()
    if u in ("minute", "minutes", "min", "mins"):
        return float(threshold)
    if u in ("second", "seconds", "sec", "secs"):
        return float(threshold) / 60.0
    if u in ("hour", "hours", "hr", "hrs"):
        return float(threshold) * 60.0
    if u in ("day", "days"):
        return float(threshold) * 60.0 * 24.0
    raise ValueError(f"unknown unit: {unit}")


def _apply_op(actual: float, threshold: float, operator: str) -> str:
    op = (operator or "<=").strip()
    if op in ("<", "lt"):
        return "yes" if actual < threshold else "no"
    if op in ("<=", "lte", "le"):
        return "yes" if actual <= threshold else "no"
    if op in (">", "gt"):
        return "yes" if actual > threshold else "no"
    if op in (">=", "gte", "ge"):
        return "yes" if actual >= threshold else "no"
    if op in ("=", "==", "eq"):
        return "yes" if math.isclose(actual, threshold) else "no"
    raise ValueError(f"unknown operator: {operator}")


def _parse_params(params: Any) -> dict:
    if params is None or (isinstance(params, float) and math.isnan(params)):
        return {}
    if isinstance(params, str):
        return json.loads(params)
    if isinstance(params, dict):
        return params
    return {}


# ---------------------------------------------------------------------------
# 3. Method A — dateutil + list comprehensions
# ---------------------------------------------------------------------------

class MethodA:
    name = "A_dateutil"

    @staticmethod
    def _first_response_minutes(conv: _Conv) -> float | None:
        cust = next((m for m in conv.messages if m.role == "customer"), None)
        agent = next((m for m in conv.messages if m.role == "agent"), None)
        if cust is None or agent is None:
            return None
        return (isoparse(agent.ts_iso) - isoparse(cust.ts_iso)).total_seconds() / 60.0

    @staticmethod
    def _count_public_messages(conv: _Conv) -> int:
        """Count all public (customer-visible) messages, both roles."""
        return sum(1 for m in conv.messages if m.is_public)

    @classmethod
    def symbolic_time(cls, conv: _Conv, p: dict) -> str:
        d = cls._first_response_minutes(conv)
        if d is None:
            return "no"
        thr = _normalise_threshold_minutes(float(p["threshold"]), p.get("unit", "minutes"))
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_time requires explicit operator; got {p}")
        return _apply_op(d, thr, op)

    @classmethod
    def symbolic_count(cls, conv: _Conv, p: dict) -> str:
        n = cls._count_public_messages(conv)
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_count requires explicit operator; got {p}")
        return _apply_op(float(n), float(p["threshold"]), op)

    @staticmethod
    def symbolic_metadata(conv: _Conv, p: dict) -> str:
        sub = p.get("subtype", "tag")
        if sub == "tag":
            return "yes" if p.get("value", "") in conv.tags else "no"
        if sub == "channel":
            return "yes" if conv.channel == p.get("value", "") else "no"
        if sub == "notes":
            return "yes" if any(not m.is_public for m in conv.messages) else "no"
        raise ValueError(f"unknown metadata subtype: {sub}")


# ---------------------------------------------------------------------------
# 4. Method B — strptime + index-based loops
# ---------------------------------------------------------------------------

class MethodB:
    name = "B_strptime"

    _FMTS = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    )

    @classmethod
    def _parse_ts(cls, s: str) -> datetime:
        for fmt in cls._FMTS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"unparseable timestamp: {s}")

    @classmethod
    def _first_response_minutes(cls, conv: _Conv) -> float | None:
        first_cust_idx, first_agent_idx = None, None
        for i in range(len(conv.messages)):
            m = conv.messages[i]
            if first_cust_idx is None and m.role == "customer":
                first_cust_idx = i
            if first_agent_idx is None and m.role == "agent":
                first_agent_idx = i
            if first_cust_idx is not None and first_agent_idx is not None:
                break
        if first_cust_idx is None or first_agent_idx is None:
            return None
        t_cust = cls._parse_ts(conv.messages[first_cust_idx].ts_iso)
        t_agent = cls._parse_ts(conv.messages[first_agent_idx].ts_iso)
        return (t_agent - t_cust).total_seconds() / 60.0

    @staticmethod
    def _count_public_messages(conv: _Conv) -> int:
        n = 0
        for i in range(len(conv.messages)):
            if conv.messages[i].is_public:
                n += 1
        return n

    @classmethod
    def symbolic_time(cls, conv: _Conv, p: dict) -> str:
        d = cls._first_response_minutes(conv)
        if d is None:
            return "no"
        thr = _normalise_threshold_minutes(float(p["threshold"]), p.get("unit", "minutes"))
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_time requires explicit operator; got {p}")
        return _apply_op(d, thr, op)

    @classmethod
    def symbolic_count(cls, conv: _Conv, p: dict) -> str:
        n = cls._count_public_messages(conv)
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_count requires explicit operator; got {p}")
        return _apply_op(float(n), float(p["threshold"]), op)

    @staticmethod
    def symbolic_metadata(conv: _Conv, p: dict) -> str:
        sub = p.get("subtype", "tag")
        val = p.get("value", "")
        if sub == "tag":
            for t in conv.tags:
                if t == val:
                    return "yes"
            return "no"
        if sub == "channel":
            return "yes" if conv.channel == val else "no"
        if sub == "notes":
            for m in conv.messages:
                if not m.is_public:
                    return "yes"
            return "no"
        raise ValueError(f"unknown metadata subtype: {sub}")


# ---------------------------------------------------------------------------
# 5. Method C — pandas DataFrame operations
# ---------------------------------------------------------------------------

class MethodC:
    name = "C_pandas"

    @staticmethod
    def _df(conv: _Conv) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "role": [m.role for m in conv.messages],
                "text": [m.text for m in conv.messages],
                "ts": pd.to_datetime([m.ts_iso for m in conv.messages], format="ISO8601"),
                "is_public": [m.is_public for m in conv.messages],
                "channel": [m.channel for m in conv.messages],
            }
        )

    @classmethod
    def _first_response_minutes(cls, conv: _Conv) -> float | None:
        df = cls._df(conv)
        cust = df[df.role == "customer"].head(1)
        agent = df[df.role == "agent"].head(1)
        if cust.empty or agent.empty:
            return None
        delta = (agent.iloc[0].ts - cust.iloc[0].ts).total_seconds() / 60.0
        return float(delta)

    @classmethod
    def _count_public_messages(cls, conv: _Conv) -> int:
        df = cls._df(conv)
        return int(df.is_public.sum())

    @classmethod
    def symbolic_time(cls, conv: _Conv, p: dict) -> str:
        d = cls._first_response_minutes(conv)
        if d is None:
            return "no"
        thr = _normalise_threshold_minutes(float(p["threshold"]), p.get("unit", "minutes"))
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_time requires explicit operator; got {p}")
        return _apply_op(d, thr, op)

    @classmethod
    def symbolic_count(cls, conv: _Conv, p: dict) -> str:
        n = cls._count_public_messages(conv)
        op = p.get("operator")
        if op is None:
            raise ValueError(f"symbolic_count requires explicit operator; got {p}")
        return _apply_op(float(n), float(p["threshold"]), op)

    @classmethod
    def symbolic_metadata(cls, conv: _Conv, p: dict) -> str:
        sub = p.get("subtype", "tag")
        val = p.get("value", "")
        if sub == "tag":
            return "yes" if val in conv.tags else "no"
        if sub == "channel":
            return "yes" if conv.channel == val else "no"
        if sub == "notes":
            df = cls._df(conv)
            return "yes" if (~df.is_public).any() else "no"
        raise ValueError(f"unknown metadata subtype: {sub}")


# ---------------------------------------------------------------------------
# 6. Dispatcher — maps (category, params) to a method answer
# ---------------------------------------------------------------------------

METHODS = {
    "A_dateutil": MethodA,
    "B_strptime": MethodB,
    "C_pandas": MethodC,
}


def run_method(method_name: str, category: str, conv: _Conv, params: dict) -> str:
    m = METHODS[method_name]
    if category == "SYMBOLIC_TIME":
        return m.symbolic_time(conv, params)
    if category == "SYMBOLIC_COUNT":
        return m.symbolic_count(conv, params)
    if category == "SYMBOLIC_METADATA":
        return m.symbolic_metadata(conv, params)
    if category == "UNSUPPORTED":
        return "reject"
    raise ValueError(f"category {category} is not symbolic")


# ---------------------------------------------------------------------------
# 7. Method D — GPT-4o tool calling oracle
# ---------------------------------------------------------------------------

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "first_response_minutes",
            "description": (
                "Returns the number of minutes between the first customer "
                "message and the first subsequent agent message. Unit is "
                "minutes (not seconds, not hours). Returns null if no such "
                "pair exists."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_public_messages",
            "description": (
                "Returns the integer count of public (customer-visible) "
                "messages in the conversation, across both roles. Internal "
                "notes (is_public false) are excluded."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tag_present",
            "description": "Returns true iff the given tag is in the conversation tag list.",
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string"}},
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "channel_equals",
            "description": "Returns true iff the conversation channel equals the given value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "has_internal_notes",
            "description": "Returns true iff the conversation contains at least one message with is_public = false.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _tool_impl(conv: _Conv, name: str, args: dict) -> str:
    """Execute a tool call using Method A (treated as the canonical independent oracle)."""
    if name == "first_response_minutes":
        v = MethodA._first_response_minutes(conv)
        return "null" if v is None else f"{v:.6f}"
    if name == "count_public_messages":
        return str(MethodA._count_public_messages(conv))
    if name == "tag_present":
        return "true" if args.get("tag", "") in conv.tags else "false"
    if name == "channel_equals":
        return "true" if conv.channel == args.get("value", "") else "false"
    if name == "has_internal_notes":
        return "true" if any(not m.is_public for m in conv.messages) else "false"
    return f'error: unknown tool {name}'


_SYSTEM_PROMPT = """You are a deterministic QA evaluator. You have tools that \
compute exact properties of a customer-support conversation. Use the tools \
to answer. Do not reason in natural language about timestamps, counts, or \
tag contents; call a tool and use its return value.

Answer format: call at most two tools, then produce a single final message \
consisting of exactly one token: "yes", "no", or "reject". Do not explain."""


def _prompt_user(conv_raw: str, prompt: str) -> str:
    return (
        f"CONVERSATION_JSON:\n{conv_raw}\n\n"
        f"EVALUATION_PROMPT:\n{prompt}\n\n"
        "Use the available tools to compute the answer. Return yes, no, or reject."
    )


def run_llm_oracle(client, model: str, conv_raw: str, conv: _Conv, prompt: str, max_hops: int = 4) -> dict:
    """Run one tool-calling dialogue for one prompt. Returns a dict with answer, tool_trace, tokens, latency."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _prompt_user(conv_raw, prompt)},
    ]
    trace = []
    tokens_total = 0
    t0 = timer.time()
    for _ in range(max_hops):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SPEC,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=200,
        )
        tokens_total += getattr(resp.usage, "total_tokens", 0) or 0
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = _tool_impl(conv, tc.function.name, args)
                trace.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue
        answer_raw = (msg.content or "").strip().lower().rstrip(".")
        if answer_raw in ("yes", "no", "reject"):
            final = answer_raw
        elif "reject" in answer_raw and "yes" not in answer_raw and "no" not in answer_raw:
            final = "reject"
        elif "yes" in answer_raw and "no" not in answer_raw:
            final = "yes"
        elif "no" in answer_raw and "yes" not in answer_raw:
            final = "no"
        else:
            final = "invalid"
        return {
            "answer": final,
            "tool_trace": trace,
            "total_tokens": tokens_total,
            "latency_ms": round((timer.time() - t0) * 1000, 1),
            "raw": msg.content,
        }
    return {
        "answer": "invalid",
        "tool_trace": trace,
        "total_tokens": tokens_total,
        "latency_ms": round((timer.time() - t0) * 1000, 1),
        "raw": "max_hops_exceeded",
    }


# ---------------------------------------------------------------------------
# 8. Agreement reporting
# ---------------------------------------------------------------------------

def agreement_table(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Overall and per-category agreement of each method with dataset ground truth."""
    rows = []
    for m in methods:
        col = f"{m}_answer"
        if col not in df.columns:
            continue
        overall = (df[col] == df["ground_truth"]).mean()
        rows.append({"method": m, "category": "ALL", "n": len(df), "agreement": round(float(overall), 4)})
        for cat, sub in df.groupby("category", observed=True):
            agree = (sub[col] == sub["ground_truth"]).mean()
            rows.append({"method": m, "category": cat, "n": len(sub), "agreement": round(float(agree), 4)})
    return pd.DataFrame(rows)


def cross_agreement(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Pairwise agreement between methods (how often method X equals method Y)."""
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i:]:
            ca, cb = f"{a}_answer", f"{b}_answer"
            if ca not in df.columns or cb not in df.columns:
                continue
            rate = (df[ca] == df[cb]).mean()
            rows.append({"method_a": a, "method_b": b, "agreement": round(float(rate), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Runner
# ---------------------------------------------------------------------------

def run(sample: int | None, run_llm: bool, llm_model: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Reading %s", EVAL_DATA)
    ed = pd.read_csv(EVAL_DATA).reset_index(drop=True)
    sym = ed[ed.category.str.startswith("SYMBOLIC")].reset_index(drop=True)
    log.info("Symbolic pairs: %d (of %d total)", len(sym), len(ed))
    if sample is not None and sample < len(sym):
        sym = sym.sample(n=sample, random_state=42).reset_index(drop=True)
        log.info("Subsampled to N=%d", len(sym))

    # Methods A, B, C (fast, no API)
    log.info("Running Methods A, B, C on %d prompts", len(sym))
    for method in ("A_dateutil", "B_strptime", "C_pandas"):
        answers = []
        failures = []
        for i, row in sym.iterrows():
            try:
                conv = parse_conv(row["conversation_json"])
                params = _parse_params(row["params"])
                a = run_method(method, row["category"], conv, params)
            except Exception as exc:  # noqa: BLE001
                a = "error"
                failures.append({"idx": i, "prompt": row["prompt"], "err": str(exc)})
            answers.append(a)
        sym[f"{method}_answer"] = answers
        if failures:
            log.warning("%s: %d failures", method, len(failures))

    methods_ran = ["A_dateutil", "B_strptime", "C_pandas"]

    # Method D (slow, requires API)
    if run_llm:
        log.info("Running Method D (LLM tool-calling, model=%s) on %d prompts", llm_model, len(sym))
        from thesis_router import get_openai_client
        client = get_openai_client()
        answers = []
        traces = []
        tokens_list = []
        lat_list = []
        for i, row in sym.iterrows():
            try:
                conv = parse_conv(row["conversation_json"])
                out = run_llm_oracle(client, llm_model, row["conversation_json"], conv, row["prompt"])
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM oracle failed at idx=%d: %s", i, exc)
                out = {"answer": "error", "tool_trace": [], "total_tokens": 0, "latency_ms": 0, "raw": str(exc)}
            answers.append(out["answer"])
            traces.append(out["tool_trace"])
            tokens_list.append(out["total_tokens"])
            lat_list.append(out["latency_ms"])
            if (i + 1) % 25 == 0:
                agree_so_far = sum(1 for a, g in zip(answers, sym["ground_truth"].iloc[: i + 1]) if a == g)
                log.info(
                    "  %d/%d  running agreement=%.3f  tokens=%d  p50-lat=%.0fms",
                    i + 1, len(sym), agree_so_far / (i + 1), sum(tokens_list), np.median(lat_list),
                )
        sym["D_llm_tool_answer"] = answers
        sym["D_llm_tool_trace"] = [json.dumps(t) for t in traces]
        sym["D_llm_tool_tokens"] = tokens_list
        sym["D_llm_tool_latency_ms"] = lat_list
        methods_ran.append("D_llm_tool")

    # Reports
    overall = agreement_table(sym, methods_ran)
    cross = cross_agreement(sym, methods_ran)
    disagreements = sym[
        (sym["A_dateutil_answer"] != sym["ground_truth"]) |
        (sym["B_strptime_answer"] != sym["ground_truth"]) |
        (sym["C_pandas_answer"] != sym["ground_truth"]) |
        (sym.get("D_llm_tool_answer", sym["ground_truth"]) != sym["ground_truth"])
    ].copy()

    overall.to_csv(OUT_DIR / "agreement_vs_ground_truth.csv", index=False)
    cross.to_csv(OUT_DIR / "pairwise_agreement.csv", index=False)
    disagreements.to_csv(OUT_DIR / "disagreements.csv", index=False)
    sym.drop(columns=["conversation_json"], errors="ignore").to_csv(OUT_DIR / "full_results.csv", index=False)

    summary = {
        "n_symbolic": int(len(sym)),
        "methods": methods_ran,
        "overall_agreement_vs_ground_truth": {
            m: round(float((sym[f"{m}_answer"] == sym["ground_truth"]).mean()), 4)
            for m in methods_ran
        },
        "per_category_agreement": {
            m: {
                str(cat): round(float((sub[f"{m}_answer"] == sub["ground_truth"]).mean()), 4)
                for cat, sub in sym.groupby("category", observed=True)
            }
            for m in methods_ran
        },
        "triple_consensus_agreement_vs_ground_truth": round(
            float((
                (sym["A_dateutil_answer"] == sym["B_strptime_answer"]) &
                (sym["B_strptime_answer"] == sym["C_pandas_answer"]) &
                (sym["A_dateutil_answer"] == sym["ground_truth"])
            ).mean()),
            4,
        ),
        "n_disagreements": int(len(disagreements)),
        "artifacts": {
            "agreement_vs_ground_truth": "results/independent_oracle/agreement_vs_ground_truth.csv",
            "pairwise_agreement": "results/independent_oracle/pairwise_agreement.csv",
            "disagreements": "results/independent_oracle/disagreements.csv",
            "full_results": "results/independent_oracle/full_results.csv",
        },
    }
    if run_llm:
        summary["llm_oracle"] = {
            "model": llm_model,
            "total_tokens": int(sum(tokens_list)),
            "p50_latency_ms": float(np.median(lat_list)),
            "p95_latency_ms": float(np.percentile(lat_list, 95)),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("----")
    log.info("Summary:")
    print(json.dumps(summary, indent=2))
    log.info("Artifacts under %s", OUT_DIR)


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-llm", action="store_true", help="skip Method D (saves ~$0.50 and ~5 min)")
    ap.add_argument("--sample", type=int, default=None, help="run on a random subsample (default: all 175)")
    ap.add_argument("--llm-model", default="gpt-4o", help="model for Method D (default: gpt-4o)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(sample=args.sample, run_llm=not args.skip_llm, llm_model=args.llm_model)
