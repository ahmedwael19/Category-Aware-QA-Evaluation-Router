"""Fail-loud gate that enforces the post-regeneration benchmark invariants.

Checks, in order:

  1. exactly 305 rows
  2. exact category counts: 65/60/50/60/40/30
  3. unique conversation_idx, no missing ground_truth
  4. every SYMBOLIC_TIME / SYMBOLIC_COUNT row carries an explicit operator
  5. every TIME / COUNT prompt is mechanically parseable by the strict parser,
     and the parsed polarity agrees with the stored operator
  6. independent ground-truth recomputation (Method A from
     `scripts.validation.independent_oracle_audit`) matches the shipped
     `ground_truth` on every SYMBOLIC_* row
  7. the production deterministic evaluator matches the shipped ground_truth
  8. a handful of polarity-inversion regressions produce the expected label

Exits nonzero on any failure.

Run: python -m scripts.validation.validate_final_benchmark
"""
from __future__ import annotations

import datetime as _dt
import json
import sys

import pandas as pd

from config import DATA_DIR
from evaluation.conversation_generator import generate_conversation
from evaluation.deterministic import (
    Conversation, Message,
    check_channel, check_internal_notes_exist,
    check_message_count, check_response_time, check_tag_present,
)
from evaluation.polarity_parser import operator_to_polarity, parse_polarity
from scripts.validation.independent_oracle_audit import (
    MethodA, parse_conv as parse_conv_ind,
)

EVAL = DATA_DIR / "evaluation_dataset.csv"

EXPECTED_COUNTS = {
    "SYMBOLIC_TIME": 65,
    "SYMBOLIC_COUNT": 60,
    "SYMBOLIC_METADATA": 50,
    "SEMANTIC": 60,
    "HYBRID": 40,
    "UNSUPPORTED": 30,
}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ed = pd.read_csv(EVAL)
    print(f"loaded {len(ed)} rows from {EVAL}")

    # 1. Shape
    if len(ed) != 305:
        fail(f"row count: got {len(ed)}, expected 305")
    counts = ed["category"].value_counts().to_dict()
    for cat, n in EXPECTED_COUNTS.items():
        if counts.get(cat, 0) != n:
            fail(f"category count {cat}: got {counts.get(cat,0)}, expected {n}")

    # 2. Unique conversation_idx
    if ed["conversation_idx"].duplicated().any():
        dupes = ed.loc[ed["conversation_idx"].duplicated(keep=False), "conversation_idx"].tolist()
        fail(f"duplicate conversation_idx: {sorted(set(dupes))}")

    # 3. Missing ground_truth
    if ed["ground_truth"].isna().any():
        fail(f"{ed['ground_truth'].isna().sum()} rows missing ground_truth")

    # 4-6. Symbolic row invariants
    sym_tc = ed[ed["category"].isin(["SYMBOLIC_TIME", "SYMBOLIC_COUNT"])]
    for i, r in sym_tc.iterrows():
        params = json.loads(r["params"]) if isinstance(r["params"], str) else r["params"]
        # explicit operator
        if not isinstance(params, dict) or "operator" not in params:
            fail(f"row {i}: {r['category']} params missing operator: {params}")
        stored_pol = operator_to_polarity(params["operator"])
        if stored_pol is None:
            fail(f"row {i}: unrecognised operator {params['operator']!r}")
        # parseable
        res = parse_polarity(r["prompt"], params)
        if res.polarity is None:
            fail(f"row {i}: prompt not parseable ({res.reason}): {r['prompt']!r}")
        # parsed polarity matches stored operator
        if res.polarity != stored_pol:
            fail(
                f"row {i}: prompt polarity {res.polarity} != stored operator polarity "
                f"{stored_pol}; prompt={r['prompt']!r}, params={params}"
            )

    # 7. Independent GT recomputation via Method A
    mismatches = []
    for i, r in ed.iterrows():
        if not str(r["category"]).startswith("SYMBOLIC"):
            continue
        conv = parse_conv_ind(r["conversation_json"])
        params = json.loads(r["params"]) if isinstance(r["params"], str) else r["params"]
        if r["category"] == "SYMBOLIC_TIME":
            # Method A inherits stored operator explicitly
            gt_ind = MethodA.symbolic_time(conv, params)
        elif r["category"] == "SYMBOLIC_COUNT":
            gt_ind = MethodA.symbolic_count(conv, params)
        elif r["category"] == "SYMBOLIC_METADATA":
            gt_ind = MethodA.symbolic_metadata(conv, params)
        else:
            continue
        if gt_ind != r["ground_truth"]:
            mismatches.append((i, r["category"], r["prompt"][:60], gt_ind, r["ground_truth"]))
    if mismatches:
        for m in mismatches[:10]:
            print(f"  {m}")
        fail(f"{len(mismatches)} symbolic rows: independent GT != shipped GT")

    # 8. Production deterministic evaluator matches shipped GT
    prod_mismatches = []
    for i, r in ed.iterrows():
        if not str(r["category"]).startswith("SYMBOLIC"):
            continue
        conv_d = json.loads(r["conversation_json"])
        msgs = [
            Message(
                role=m["role"], text=m["text"],
                timestamp=_dt.datetime.fromisoformat(m["timestamp"]),
                is_public=m.get("is_public", True),
                channel=m.get("channel", conv_d.get("channel", "")),
            )
            for m in conv_d["messages"]
        ]
        conv = Conversation(
            messages=msgs,
            tags=conv_d.get("tags", []),
            channel=conv_d.get("channel", ""),
            resolution_time_minutes=conv_d.get("resolution_time_minutes"),
        )
        params = json.loads(r["params"]) if isinstance(r["params"], str) else r["params"]
        if r["category"] == "SYMBOLIC_TIME":
            thr = float(params["threshold"])
            unit = params.get("unit", "minutes")
            if unit in ("seconds", "sec", "secs"):
                thr /= 60.0
            elif unit in ("hours", "hrs", "hr"):
                thr *= 60.0
            elif unit in ("days", "day"):
                thr *= 1440.0
            gt_prod = check_response_time(conv, thr, params["operator"])
        elif r["category"] == "SYMBOLIC_COUNT":
            gt_prod = check_message_count(conv, int(params["threshold"]), params["operator"])
        elif r["category"] == "SYMBOLIC_METADATA":
            sub = params.get("subtype", "tag")
            if sub == "tag":
                gt_prod = check_tag_present(conv, params.get("value", ""))
            elif sub == "channel":
                gt_prod = check_channel(conv, params.get("value", ""))
            elif sub == "notes":
                gt_prod = check_internal_notes_exist(conv)
            else:
                continue
        else:
            continue
        if gt_prod != r["ground_truth"]:
            prod_mismatches.append((i, r["category"], r["prompt"][:60], gt_prod, r["ground_truth"]))
    if prod_mismatches:
        for m in prod_mismatches[:10]:
            print(f"  {m}")
        fail(f"{len(prod_mismatches)} symbolic rows: prod evaluator != shipped GT")

    # 9. Regression examples
    regressions = [
        ("wider than", 16, "sec", 32, "yes"),       # "wider than 16 sec" + 32 sec → yes
        ("exceeded", 14, "minutes", 28, "yes"),     # "exceeded 14 min" + 28 min → yes
        ("stayed under", 18, None, 19, "no"),       # count: stayed under 18 + 19 turns → no
        ("fewer than", 17, None, 13, "yes"),        # count: fewer than 17 + 13 msgs → yes
    ]
    for kw, thr, unit, actual, expected in regressions:
        p = {"threshold": thr, "operator": ">" if kw in ("wider than", "exceeded") else ("<" if unit is None else "<")}
        if unit:
            p["unit"] = unit
        # Build a minimal synthetic conversation and test
        if unit:
            thr_min = thr if unit == "minutes" else (thr / 60.0 if unit in ("sec", "seconds") else thr * 60.0)
            conv = generate_conversation(response_time_minutes=actual)
            gt = check_response_time(conv, thr_min, p["operator"])
            if gt != expected:
                fail(f"regression: {kw} {thr} {unit} + actual {actual} expected {expected}, got {gt}")
        else:
            conv = generate_conversation(n_turns=actual)
            gt = check_message_count(conv, thr, p["operator"])
            if gt != expected:
                fail(f"regression: {kw} {thr} (count) + {actual} turns expected {expected}, got {gt}")
    print("PASS: all validations passed")


if __name__ == "__main__":
    main()
