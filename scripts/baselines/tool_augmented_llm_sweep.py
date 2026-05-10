"""Tool-augmented LLM baseline across OpenAI and Claude models.

Every evaluated model is given the same five deterministic oracle tools, whose
implementations return ground-truth-grade values. This isolates whether a
model's error on symbolic prompts is a computation failure (the model cannot
subtract timestamps or count messages) or an application failure (the model
sees the correct value and still answers wrong): the residual gap between
tool-augmented accuracy and direct deterministic dispatch bounds the
application-failure component.

Models evaluated on the 175 symbolic pairs:
  GPT-4o-mini, GPT-4o, o3-mini, o1 (OpenAI)
  Claude Sonnet 4.5, Claude Opus 4.7 (Anthropic)

Per-model artefacts are written incrementally to
results/independent_oracle/tool_aug_<slug>.json so interruption does not lose
progress; pass --resume to skip models whose output already exists.

Usage:
    python -m scripts.baselines.tool_augmented_llm_sweep
    python -m scripts.baselines.tool_augmented_llm_sweep --models gpt-4o-mini gpt-4o
    python -m scripts.baselines.tool_augmented_llm_sweep --sample 10
    python -m scripts.baselines.tool_augmented_llm_sweep --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import time as timer
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR
from thesis_router import get_anthropic_client, get_openai_client
from scripts.validation.independent_oracle_audit import (
    _Conv,
    parse_conv,
    TOOL_SPEC as OPENAI_TOOL_SPEC,
    _tool_impl as oracle_tool_impl,
)

EVAL_DATA = DATA_DIR / "evaluation_dataset.csv"
OUT_DIR = RESULTS_DIR / "independent_oracle"

log = logging.getLogger("tool-aug-sweep")


# ---------------------------------------------------------------------------
# 1. Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelSpec:
    slug: str
    provider: str            # "openai" or "anthropic"
    model_id: str
    family: str              # "gpt-4o-family", "openai-reasoning", "claude"
    reasoning: bool
    max_tokens: int

MODELS: dict[str, ModelSpec] = {
    "gpt-4o-mini": ModelSpec(
        slug="gpt-4o-mini", provider="openai",
        model_id="gpt-4o-mini", family="gpt-4o-family",
        reasoning=False, max_tokens=512,
    ),
    "gpt-4o": ModelSpec(
        slug="gpt-4o", provider="openai",
        model_id="gpt-4o", family="gpt-4o-family",
        reasoning=False, max_tokens=512,
    ),
    "o3-mini": ModelSpec(
        slug="o3-mini", provider="openai",
        model_id="o3-mini", family="openai-reasoning",
        reasoning=True, max_tokens=8192,
    ),
    "o1": ModelSpec(
        slug="o1", provider="openai",
        model_id="o1", family="openai-reasoning",
        reasoning=True, max_tokens=8192,
    ),
    "claude-sonnet-4-5": ModelSpec(
        slug="claude-sonnet-4-5", provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        family="claude", reasoning=False, max_tokens=1024,
    ),
    "claude-opus-4-7": ModelSpec(
        slug="claude-opus-4-7", provider="anthropic",
        model_id="claude-opus-4-7",
        family="claude", reasoning=False, max_tokens=1024,
    ),
}

# Hard cap on tool-call rounds per prompt. The system prompt instructs the
# model to call at most two tools; MAX_HOPS provides headroom against models
# that over-call before emitting a final answer (logged as "max_hops_exceeded").
MAX_HOPS = 6


# ---------------------------------------------------------------------------
# 2. Shared system prompt (identical wording across providers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a deterministic QA evaluator. You have tools that \
compute exact properties of a customer-support conversation. Use the tools \
to answer. Do not reason in natural language about timestamps, counts, or \
tag contents; call a tool and use its return value.

Answer format: call at most two tools, then produce a single final message \
consisting of exactly one token: "yes", "no", or "reject". Do not explain."""


def user_message(conv_raw: str, prompt: str) -> str:
    return (
        f"CONVERSATION_JSON:\n{conv_raw}\n\n"
        f"EVALUATION_PROMPT:\n{prompt}\n\n"
        "Use the available tools to compute the answer. Return yes, no, or reject."
    )


def parse_final_answer(text: str) -> str:
    t = (text or "").strip().lower().rstrip(".")
    if t in ("yes", "no", "reject"):
        return t
    if "reject" in t and "yes" not in t and "no" not in t:
        return "reject"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return "invalid"


# ---------------------------------------------------------------------------
# 3. Anthropic (Bedrock-gateway) tool schemas
# ---------------------------------------------------------------------------

ANTHROPIC_TOOL_SPEC = [
    {
        "name": "first_response_minutes",
        "description": (
            "Returns the number of minutes between the first customer message "
            "and the first subsequent agent message. Unit is minutes. Returns "
            "null if no such pair exists."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "count_public_messages",
        "description": (
            "Returns the integer count of public messages in the conversation, "
            "across both roles. Internal notes are excluded."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "tag_present",
        "description": "Returns true iff the given tag is in the conversation tag list.",
        "input_schema": {
            "type": "object",
            "properties": {"tag": {"type": "string"}},
            "required": ["tag"],
        },
    },
    {
        "name": "channel_equals",
        "description": "Returns true iff the conversation channel equals the given value.",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
    {
        "name": "has_internal_notes",
        "description": "Returns true iff the conversation contains at least one non-public message.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# 4. Provider-specific runners
# ---------------------------------------------------------------------------

def run_openai(spec: ModelSpec, client, conv_raw: str, conv: _Conv, prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(conv_raw, prompt)},
    ]
    trace: list[dict] = []
    tokens_total = 0
    reasoning_tokens_total = 0
    t0 = timer.time()

    for _ in range(MAX_HOPS):
        kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "messages": messages,
            "tools": OPENAI_TOOL_SPEC,
            "tool_choice": "auto",
            "max_completion_tokens": spec.max_tokens,
        }
        if not spec.reasoning:
            kwargs["temperature"] = 0.0
        resp = client.chat.completions.create(**kwargs)

        if resp.usage:
            tokens_total += resp.usage.total_tokens or 0
            details = getattr(resp.usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens_total += getattr(details, "reasoning_tokens", 0) or 0

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = oracle_tool_impl(conv, tc.function.name, args)
                trace.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        final = parse_final_answer(msg.content or "")
        return {
            "answer": final,
            "tool_trace": trace,
            "total_tokens": tokens_total,
            "reasoning_tokens": reasoning_tokens_total,
            "latency_ms": round((timer.time() - t0) * 1000, 1),
            "raw": msg.content,
        }

    return {
        "answer": "invalid",
        "tool_trace": trace,
        "total_tokens": tokens_total,
        "reasoning_tokens": reasoning_tokens_total,
        "latency_ms": round((timer.time() - t0) * 1000, 1),
        "raw": "max_hops_exceeded",
    }


def run_anthropic(spec: ModelSpec, client, conv_raw: str, conv: _Conv, prompt: str) -> dict:
    messages: list[dict] = [
        {"role": "user", "content": user_message(conv_raw, prompt)},
    ]
    trace: list[dict] = []
    tokens_total = 0
    t0 = timer.time()

    for _ in range(MAX_HOPS):
        resp = client.messages.create(
            model=spec.model_id,
            max_tokens=spec.max_tokens,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOL_SPEC,
            messages=messages,
        )

        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens_total += (usage.input_tokens or 0) + (usage.output_tokens or 0)

        content_blocks = [b.model_dump() for b in resp.content]
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]

        if tool_uses:
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tu in tool_uses:
                name = tu.get("name", "")
                args = tu.get("input", {}) or {}
                result = oracle_tool_impl(conv, name, args)
                trace.append({"name": name, "args": args, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id", ""),
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        raw = " ".join(b.get("text", "") for b in text_blocks)
        final = parse_final_answer(raw)
        return {
            "answer": final,
            "tool_trace": trace,
            "total_tokens": tokens_total,
            "reasoning_tokens": 0,
            "latency_ms": round((timer.time() - t0) * 1000, 1),
            "raw": raw,
        }

    return {
        "answer": "invalid",
        "tool_trace": trace,
        "total_tokens": tokens_total,
        "reasoning_tokens": 0,
        "latency_ms": round((timer.time() - t0) * 1000, 1),
        "raw": "max_hops_exceeded",
    }


# ---------------------------------------------------------------------------
# 5. Model dispatcher with retries
# ---------------------------------------------------------------------------

def run_one_prompt(spec: ModelSpec, ctx: dict, conv_raw: str, conv: _Conv, prompt: str, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            if spec.provider == "openai":
                return run_openai(spec, ctx["openai_client"], conv_raw, conv, prompt)
            if spec.provider == "anthropic":
                return run_anthropic(spec, ctx["anthropic_client"], conv_raw, conv, prompt)
            raise ValueError(f"unknown provider: {spec.provider}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries - 1:
                backoff = 2.0 ** attempt
                log.warning("  retry %d/%d after %.0fs: %s", attempt + 1, retries, backoff, exc)
                timer.sleep(backoff)
    return {
        "answer": "error",
        "tool_trace": [],
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "latency_ms": 0,
        "raw": f"{type(last_err).__name__}: {last_err}",
    }


# ---------------------------------------------------------------------------
# 6. Per-model run loop
# ---------------------------------------------------------------------------

def run_model(spec: ModelSpec, sym: pd.DataFrame, ctx: dict, resume: bool) -> dict:
    out_path = OUT_DIR / f"tool_aug_{spec.slug}.json"
    if resume and out_path.exists():
        log.info("SKIP %s (already exists: %s)", spec.slug, out_path)
        return json.load(open(out_path))

    log.info("=" * 70)
    log.info("MODEL %s (%s) — %d prompts", spec.slug, spec.model_id, len(sym))
    log.info("=" * 70)

    per_prompt = []
    start = timer.time()
    for i, row in sym.reset_index(drop=True).iterrows():
        conv = parse_conv(row["conversation_json"])
        out = run_one_prompt(spec, ctx, row["conversation_json"], conv, row["prompt"])
        per_prompt.append({
            "idx": int(row.get("conversation_idx", i)),
            "prompt": row["prompt"],
            "category": row["category"],
            "ground_truth": row["ground_truth"],
            "model_answer": out["answer"],
            "correct": int(out["answer"] == row["ground_truth"]),
            "total_tokens": out["total_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
            "latency_ms": out["latency_ms"],
            "n_tool_calls": len(out["tool_trace"]),
            "tool_trace": out["tool_trace"],
            "raw": (out["raw"] or "")[:500],
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(sym):
            corr = sum(p["correct"] for p in per_prompt)
            tokens = sum(p["total_tokens"] for p in per_prompt)
            med = np.median([p["latency_ms"] for p in per_prompt])
            log.info(
                "  %3d/%3d  acc=%.3f  tokens=%d  p50-lat=%.0fms",
                i + 1, len(sym), corr / (i + 1), tokens, med,
            )

    elapsed = timer.time() - start

    lat = np.array([p["latency_ms"] for p in per_prompt])
    tok = np.array([p["total_tokens"] for p in per_prompt])
    reason_tok = np.array([p["reasoning_tokens"] for p in per_prompt])

    per_cat = {}
    for cat in sorted(sym["category"].unique()):
        sub = [p for p in per_prompt if p["category"] == cat]
        if not sub:
            continue
        per_cat[cat] = {
            "n": len(sub),
            "accuracy": round(sum(p["correct"] for p in sub) / len(sub), 4),
        }

    summary = {
        "model": spec.slug,
        "model_id": spec.model_id,
        "provider": spec.provider,
        "family": spec.family,
        "reasoning": spec.reasoning,
        "n": len(per_prompt),
        "accuracy": round(float(np.mean([p["correct"] for p in per_prompt])), 4),
        "per_category_accuracy": per_cat,
        "total_tokens": int(tok.sum()),
        "reasoning_tokens": int(reason_tok.sum()),
        "mean_tokens_per_prompt": round(float(tok.mean()), 1),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_tool_calls_per_prompt": round(float(np.mean([p["n_tool_calls"] for p in per_prompt])), 2),
        "errors": int(sum(1 for p in per_prompt if p["model_answer"] == "error")),
        "invalids": int(sum(1 for p in per_prompt if p["model_answer"] == "invalid")),
        "elapsed_s": round(elapsed, 1),
        "per_prompt": per_prompt,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "WROTE %s   acc=%.4f  tokens=%d  p50-lat=%.0fms  elapsed=%.0fs",
        out_path.name, summary["accuracy"], summary["total_tokens"],
        summary["p50_latency_ms"], elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# 7. Final comparison table
# ---------------------------------------------------------------------------

def build_comparison(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        row = {
            "model": s["model"],
            "family": s["family"],
            "reasoning": s["reasoning"],
            "n": s["n"],
            "accuracy": s["accuracy"],
            "sym_time": s["per_category_accuracy"].get("SYMBOLIC_TIME", {}).get("accuracy"),
            "sym_count": s["per_category_accuracy"].get("SYMBOLIC_COUNT", {}).get("accuracy"),
            "sym_metadata": s["per_category_accuracy"].get("SYMBOLIC_METADATA", {}).get("accuracy"),
            "total_tokens": s["total_tokens"],
            "reasoning_tokens": s["reasoning_tokens"],
            "tokens_per_prompt": s["mean_tokens_per_prompt"],
            "p50_latency_ms": s["p50_latency_ms"],
            "p95_latency_ms": s["p95_latency_ms"],
            "tool_calls_per_prompt": s["mean_tool_calls_per_prompt"],
            "errors": s["errors"],
            "invalids": s["invalids"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Runner
# ---------------------------------------------------------------------------

def run(models: list[str], sample: int | None, resume: bool) -> None:
    from thesis_router import rebuild_baselines

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    comp_path = OUT_DIR / "tool_augmented_comparison.csv"
    all_path = OUT_DIR / "tool_augmented_all_models.json"
    if not rebuild_baselines() and comp_path.exists() and all_path.exists():
        log.info("[phase4.tool_aug] reproduce mode — committed %s and %s present; "
                 "skipping re-run (set REBUILD_BASELINES=1 to regenerate)",
                 comp_path.name, all_path.name)
        return

    ctx: dict = {}
    if any(MODELS[m].provider == "openai" for m in models):
        ctx["openai_client"] = get_openai_client()
    if any(MODELS[m].provider == "anthropic" for m in models):
        ctx["anthropic_client"] = get_anthropic_client()

    ed = pd.read_csv(EVAL_DATA).reset_index(drop=True)
    sym = ed[ed.category.str.startswith("SYMBOLIC")].reset_index(drop=True)
    if sample is not None and sample < len(sym):
        sym = sym.sample(n=sample, random_state=42).reset_index(drop=True)
    log.info("Symbolic pairs: N=%d", len(sym))
    log.info("Models: %s", ", ".join(models))

    summaries: list[dict] = []
    for m in models:
        if m not in MODELS:
            log.warning("unknown model: %s  (valid: %s)", m, ", ".join(MODELS))
            continue
        summary = run_model(MODELS[m], sym, ctx, resume)
        summaries.append(summary)

    if not summaries:
        return

    comp = build_comparison(summaries)
    comp_path = OUT_DIR / "tool_augmented_comparison.csv"
    comp.to_csv(comp_path, index=False)

    all_path = OUT_DIR / "tool_augmented_all_models.json"
    with open(all_path, "w") as f:
        json.dump(
            {"models": [{k: v for k, v in s.items() if k != "per_prompt"} for s in summaries]},
            f, indent=2,
        )

    log.info("=" * 70)
    log.info("FINAL COMPARISON (%s)", comp_path.name)
    log.info("=" * 70)
    print(comp.to_string(index=False))
    log.info("=" * 70)
    log.info("Wrote %s and %s", comp_path.name, all_path.name)


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help="subset of models to run (default: all)",
    )
    ap.add_argument("--sample", type=int, default=None, help="run on a random subsample")
    ap.add_argument("--resume", action="store_true", help="skip models whose output file already exists")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, sample=args.sample, resume=args.resume)
