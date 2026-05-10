"""Build the end-to-end evaluation benchmark.

Constructs (prompt, conversation, ground_truth) triples from the test split
and the full synthetic pool under the polarity-admission rule described in
the methodology chapter. Every SYMBOLIC_TIME and SYMBOLIC_COUNT prompt in the
result satisfies:

  1. its stored params carry an explicit comparison operator,
  2. the strict polarity parser recovers that operator from the prompt text,
  3. the threshold and polarity keyword appear literally in the prompt.

Prompts in the test split that fail (2) or (3) are replaced by parseable
prompts drawn from the full synthetic pool, same category. Ground truth is
then recomputed from the explicit operator via
`evaluation/deterministic.py`.

Other categories (SYMBOLIC_METADATA, SEMANTIC, HYBRID, UNSUPPORTED) are taken
directly from the test split.

Target sizes: 65 TIME, 60 COUNT, 50 METADATA, 60 SEMANTIC, 40 HYBRID,
30 UNSUPPORTED (305 total).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.conversation_generator import (
    generate_conversation,
    generate_conversation_for_symbolic,
)
from evaluation.deterministic import (
    check_channel, check_internal_notes_exist, check_tag_present,
)
from evaluation.polarity_parser import parse_polarity, polarity_to_operator

TARGET_COUNTS = {
    "SYMBOLIC_TIME": 65,
    "SYMBOLIC_COUNT": 60,
    "SYMBOLIC_METADATA": 50,
    "SEMANTIC": 60,
    "HYBRID": 40,
    "UNSUPPORTED": 30,
}

CAT_ORDER = list(TARGET_COUNTS.keys())


def _row_polarity(prompt: str, params) -> str | None:
    res = parse_polarity(prompt, params)
    return res.polarity


def _params_obj(params):
    return json.loads(params) if isinstance(params, str) else params


def _symbolic_pool(df_full: pd.DataFrame) -> pd.DataFrame:
    """Subset of the full synthetic pool whose prompts pass polarity admission."""
    pool = df_full[df_full.category.isin(["SYMBOLIC_TIME", "SYMBOLIC_COUNT"])].copy()
    pool["polarity"] = pool.apply(
        lambda r: _row_polarity(r["prompt"], r["params"]), axis=1,
    )
    return pool[pool["polarity"].isin(("under", "over"))].reset_index(drop=True)


def _pick_replacement(pool_df: pd.DataFrame, used_prompts: set, rng: random.Random) -> pd.Series:
    """Draw a fresh parseable prompt from the pool, avoiding reuse."""
    for _ in range(500):
        candidate = pool_df.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        if candidate["prompt"] not in used_prompts:
            return candidate
    raise RuntimeError(f"exhausted replacement pool after 500 tries "
                       f"({len(pool_df)} candidates available)")


def _build_symbolic_time_row(prompt: str, params: dict, scenario: str, idx: int) -> dict:
    conv_payload = generate_conversation_for_symbolic(
        {"prompt": prompt, "category": "SYMBOLIC_TIME", "params": params, "scenario": scenario},
    )
    return {
        "prompt": prompt,
        "category": "SYMBOLIC_TIME",
        "conversation_json": conv_payload["json"],
        "conversation_idx": idx,
        "ground_truth": conv_payload["ground_truth"],
        "ground_truth_source": "deterministic",
        "scenario": scenario,
        "params": json.dumps(params),
        "actual_response_time_min": conv_payload.get("actual_response_time_min"),
    }


def _build_symbolic_count_row(prompt: str, params: dict, scenario: str, idx: int) -> dict:
    conv_payload = generate_conversation_for_symbolic(
        {"prompt": prompt, "category": "SYMBOLIC_COUNT", "params": params, "scenario": scenario},
    )
    return {
        "prompt": prompt,
        "category": "SYMBOLIC_COUNT",
        "conversation_json": conv_payload["json"],
        "conversation_idx": idx,
        "ground_truth": conv_payload["ground_truth"],
        "ground_truth_source": "deterministic",
        "scenario": scenario,
        "params": json.dumps(params),
        "actual_count": conv_payload.get("actual_count"),
    }


def _build_symbolic_metadata_row(row, idx: int, rng: random.Random) -> dict | None:
    params = _params_obj(row["params"])
    if not isinstance(params, dict):
        return None
    subtype = params.get("subtype", "tag")
    make_yes = (idx % 2 == 0)

    if subtype == "tag":
        tag = params.get("value", "urgent")
        if make_yes:
            conv = generate_conversation(tags=[tag, "general"])
            scenario = "tag_present"
        else:
            conv = generate_conversation(tags=["general", "other"])
            scenario = "tag_absent"
        gt = check_tag_present(conv, tag)
    elif subtype == "channel":
        ch = params.get("value", "email")
        if make_yes:
            conv = generate_conversation(channel=ch)
            scenario = "channel_match"
        else:
            conv = generate_conversation(channel="phone" if ch != "phone" else "email")
            scenario = "channel_mismatch"
        gt = check_channel(conv, ch)
    elif subtype == "notes":
        conv = generate_conversation(include_internal_notes=make_yes)
        scenario = "notes_present" if make_yes else "notes_absent"
        gt = check_internal_notes_exist(conv)
    else:
        return None

    return {
        "prompt": row["prompt"],
        "category": "SYMBOLIC_METADATA",
        "conversation_json": json.dumps(asdict(conv), default=str),
        "conversation_idx": idx,
        "ground_truth": gt,
        "ground_truth_source": "deterministic",
        "scenario": scenario,
        "params": json.dumps(params),
    }


def _build_semantic_or_hybrid_row(row, category: str, idx: int, rng: random.Random) -> dict:
    conv = generate_conversation(n_turns=rng.randint(5, 10))
    return {
        "prompt": row["prompt"],
        "category": category,
        "conversation_json": json.dumps(asdict(conv), default=str),
        "conversation_idx": idx,
        "ground_truth": None,
        "ground_truth_source": "gpt5.2_judge",
        "scenario": "standard",
        "params": None,
    }


def _build_unsupported_row(row, idx: int, rng: random.Random) -> dict:
    conv = generate_conversation(n_turns=rng.randint(4, 8))
    return {
        "prompt": row["prompt"],
        "category": "UNSUPPORTED",
        "conversation_json": json.dumps(asdict(conv), default=str),
        "conversation_idx": idx,
        "ground_truth": "reject",
        "ground_truth_source": "rejection",
        "scenario": "unanswerable",
        "params": None,
    }


def _scenario_cycle(i: int) -> str:
    return ("clear_yes", "clear_no", "boundary")[i % 3]


def build_evaluation_dataset(
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,
    out_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the 305-pair end-to-end benchmark and write it to `out_path`.

    Parameters
    ----------
    df_test
        The test split (prompt, top_category, params, ...).
    df_full
        The full synthetic pool, used to replace prompts that fail the
        polarity admission rule.
    out_path
        Destination CSV (usually `config.EVAL_DATASET`).
    seed
        RNG seed for scenario assignment, conversation generation, and
        pool-based replacement.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    pool = _symbolic_pool(df_full)
    pool_time = pool[pool.category == "SYMBOLIC_TIME"].reset_index(drop=True)
    pool_count = pool[pool.category == "SYMBOLIC_COUNT"].reset_index(drop=True)

    used_prompts: set[str] = set()
    rows: list[dict] = []
    idx = 0

    # SYMBOLIC_TIME
    time_prompts = df_test[
        (df_test["top_category"] == "SYMBOLIC_TIME") & (df_test["params"].notna())
    ].reset_index(drop=True)
    kept = 0
    replaced = 0
    for i, r in time_prompts.iterrows():
        if kept + replaced >= TARGET_COUNTS["SYMBOLIC_TIME"]:
            break
        polarity = _row_polarity(r["prompt"], r["params"])
        scenario = _scenario_cycle(i)
        if polarity in ("under", "over"):
            params = _params_obj(r["params"])
            params["operator"] = polarity_to_operator(polarity, "SYMBOLIC_TIME")
            rows.append(_build_symbolic_time_row(r["prompt"], params, scenario, idx))
            used_prompts.add(r["prompt"])
            kept += 1
        else:
            cand = _pick_replacement(pool_time, used_prompts, rng)
            params = _params_obj(cand["params"])
            params["operator"] = polarity_to_operator(cand["polarity"], "SYMBOLIC_TIME")
            rows.append(_build_symbolic_time_row(cand["prompt"], params, scenario, idx))
            used_prompts.add(cand["prompt"])
            replaced += 1
        idx += 1
    print(f"  SYMBOLIC_TIME: {kept} kept, {replaced} replaced from pool")

    # SYMBOLIC_COUNT
    count_prompts = df_test[
        (df_test["top_category"] == "SYMBOLIC_COUNT") & (df_test["params"].notna())
    ].reset_index(drop=True)
    kept = 0
    replaced = 0
    for i, r in count_prompts.iterrows():
        if kept + replaced >= TARGET_COUNTS["SYMBOLIC_COUNT"]:
            break
        polarity = _row_polarity(r["prompt"], r["params"])
        scenario = _scenario_cycle(i)
        if polarity in ("under", "over"):
            params = _params_obj(r["params"])
            params["operator"] = polarity_to_operator(polarity, "SYMBOLIC_COUNT")
            rows.append(_build_symbolic_count_row(r["prompt"], params, scenario, idx))
            used_prompts.add(r["prompt"])
            kept += 1
        else:
            cand = _pick_replacement(pool_count, used_prompts, rng)
            params = _params_obj(cand["params"])
            params["operator"] = polarity_to_operator(cand["polarity"], "SYMBOLIC_COUNT")
            rows.append(_build_symbolic_count_row(cand["prompt"], params, scenario, idx))
            used_prompts.add(cand["prompt"])
            replaced += 1
        idx += 1
    print(f"  SYMBOLIC_COUNT: {kept} kept, {replaced} replaced from pool")

    # SYMBOLIC_METADATA
    meta_prompts = df_test[
        (df_test["top_category"] == "SYMBOLIC_METADATA") & (df_test["params"].notna())
    ].reset_index(drop=True)
    meta_n = 0
    for i, r in meta_prompts.iterrows():
        if meta_n >= TARGET_COUNTS["SYMBOLIC_METADATA"]:
            break
        built = _build_symbolic_metadata_row(r, idx, rng)
        if built is None:
            continue
        rows.append(built)
        meta_n += 1
        idx += 1
    print(f"  SYMBOLIC_METADATA: {meta_n} pairs")

    # SEMANTIC
    sem_sample = df_test[df_test["top_category"] == "SEMANTIC"].sample(
        n=min(TARGET_COUNTS["SEMANTIC"], (df_test["top_category"] == "SEMANTIC").sum()),
        random_state=seed,
    )
    for _, r in sem_sample.iterrows():
        rows.append(_build_semantic_or_hybrid_row(r, "SEMANTIC", idx, rng))
        idx += 1
    print(f"  SEMANTIC: {len(sem_sample)} pairs (ground truth pending)")

    # HYBRID
    hyb_sample = df_test[df_test["top_category"] == "HYBRID"].sample(
        n=min(TARGET_COUNTS["HYBRID"], (df_test["top_category"] == "HYBRID").sum()),
        random_state=seed,
    )
    for _, r in hyb_sample.iterrows():
        rows.append(_build_semantic_or_hybrid_row(r, "HYBRID", idx, rng))
        idx += 1
    print(f"  HYBRID: {len(hyb_sample)} pairs (ground truth pending)")

    # UNSUPPORTED
    unsup_sample = df_test[df_test["top_category"] == "UNSUPPORTED"].sample(
        n=min(TARGET_COUNTS["UNSUPPORTED"], (df_test["top_category"] == "UNSUPPORTED").sum()),
        random_state=seed,
    )
    for _, r in unsup_sample.iterrows():
        rows.append(_build_unsupported_row(r, idx, rng))
        idx += 1
    print(f"  UNSUPPORTED: {len(unsup_sample)} pairs")

    out = pd.DataFrame(rows)
    out["_sort"] = out["category"].map({c: i for i, c in enumerate(CAT_ORDER)}).fillna(99)
    out = out.sort_values(["_sort", "conversation_idx"]).drop(columns=["_sort"]).reset_index(drop=True)

    counts = out["category"].value_counts().to_dict()
    for cat, n in TARGET_COUNTS.items():
        got = counts.get(cat, 0)
        if got != n:
            print(f"  WARNING: {cat} count {got} != target {n}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nTotal evaluation pairs: {len(out)}")
    print(out["category"].value_counts().sort_index().to_string())
    print(f"Wrote {out_path}")
    return out
