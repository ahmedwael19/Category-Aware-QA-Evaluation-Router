"""Parameter Extraction Accuracy Test.

Measures how well regex-based extractors can recover symbolic parameters
(thresholds, units, operators, tag names, channels) from prompt text alone.
This quantifies the gap between the thesis's oracle assumption (params from
metadata) and a realistic deployment (params from free text).

Runs on all symbolic prompts in the full dataset (not just eval subset).
No API calls.

Run: python -m scripts.analyses.parameter_extraction
"""

import json
import re

from config import DATA_DIR, RESULTS_ANALYSES

import pandas as pd

if __name__ != "__main__":
    raise RuntimeError("This script must be invoked as a main program, e.g. python -m scripts.analyses.parameter_extraction")


# ── Load full dataset ─────────────────────────────────────────────────────────
full_csv = sorted(DATA_DIR.glob("synthetic_final_*_full.csv"))[-1]
df = pd.read_csv(full_csv)
sym = df[df["top_category"].str.startswith("SYMBOLIC")].copy()
print(f"Loaded {len(sym)} symbolic prompts from {full_csv.name}")
print(f"  Categories: {sym['top_category'].value_counts().to_dict()}")
print(f"  Subcategories: {sym['subcategory'].value_counts().to_dict()}")

# TIME: threshold number + unit. Matches "within 5 minutes", "under 3 hrs",
# "exceeded 30 min", "300 sec response window", "the 8 sec threshold", etc.
_TIME_NUMBER_UNIT = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(seconds?|sec|minutes?|min|mins?|hours?|hrs?)',
    re.IGNORECASE,
)

# Operator extraction from context
_TIME_UNDER = re.compile(
    r'within|under|less\s+than|before|fewer\s+than|met|honored|achieved|'
    r'in\s+under|faster\s+than|quicker\s+than',
    re.IGNORECASE,
)
_TIME_OVER = re.compile(
    r'more\s+than|over|exceed|longer\s+than|greater\s+than|'
    r'beyond|surpass|breach|broke|late|slow|delay',
    re.IGNORECASE,
)


def extract_time_params(prompt):
    """Extract threshold, unit, and operator from a time prompt."""
    m = _TIME_NUMBER_UNIT.search(prompt)
    if not m:
        return None

    threshold = float(m.group(1))
    raw_unit = m.group(2).lower()

    # Normalize unit
    if raw_unit in ("second", "seconds", "sec"):
        unit = "seconds"
    elif raw_unit in ("minute", "minutes", "min", "mins"):
        unit = "minutes"
    elif raw_unit in ("hour", "hours", "hr", "hrs"):
        unit = "hours"
    else:
        unit = raw_unit

    # Determine operator from context
    has_under = bool(_TIME_UNDER.search(prompt))
    has_over = bool(_TIME_OVER.search(prompt))

    if has_under and not has_over:
        operator = "<="
    elif has_over and not has_under:
        operator = ">"
    else:
        operator = "<="  # Default: "within" semantics

    return {"threshold": threshold, "unit": unit, "operator": operator}


# ── COUNT: extract threshold number ───────────────────────────────────────────
_COUNT_NUMBER = re.compile(
    r'(\d+)\s*(?:\+\s*)?(?:turns?|messages?|exchanges?|replies?|interactions?|'
    r'back.and.forth|responses?|round.trips?)',
    re.IGNORECASE,
)
_COUNT_NUMBER_ALT = re.compile(
    r'(?:than|exceeding?|surpass|above|beyond|over|fewer\s+than|'
    r'more\s+than|at\s+least|at\s+most|crossed|compare\s+to)\s+(\d+)',
    re.IGNORECASE,
)
# "Were 15 or more turns" — number ... unit with gap
_COUNT_NUMBER_GAP = re.compile(
    r'(\d+)\s+(?:or\s+(?:more|fewer))\s+(?:turns?|messages?|exchanges?|'
    r'replies?|interactions?|responses?)',
    re.IGNORECASE,
)
# "turn count exceeded 18", "total messages crossed 13"
_COUNT_UNIT_THEN_NUMBER = re.compile(
    r'(?:turns?|messages?|exchanges?|replies?|responses?|message\s+count|'
    r'turn\s+count|reply\s+count|exchange\s+count|total\s+messages?)'
    r'.*?(\d+)',
    re.IGNORECASE,
)


def extract_count_params(prompt):
    """Extract threshold from a count prompt."""
    m = _COUNT_NUMBER.search(prompt)
    if not m:
        m = _COUNT_NUMBER_GAP.search(prompt)
    if not m:
        m = _COUNT_NUMBER_ALT.search(prompt)
    if not m:
        m = _COUNT_UNIT_THEN_NUMBER.search(prompt)
    if not m:
        return None

    threshold = int(m.group(1))
    return {"threshold": threshold}


# ── METADATA: extract subtype + value ─────────────────────────────────────────
_META_TAG = re.compile(
    r"['\"]([a-z_-]+)['\"]"
    r"(?:\s+(?:tag|label|ticket|status|case))?",
    re.IGNORECASE,
)
_META_CHANNEL = re.compile(
    r'\b(email|chat|phone|twitter|whatsapp|sms)\b',
    re.IGNORECASE,
)
_META_NOTES = re.compile(
    r'internal\s+(note|comment)|behind.the.scenes\s+note|'
    r'note\s+for\s+the\s+team|private\s+(note|comment)',
    re.IGNORECASE,
)


def extract_metadata_params(prompt):
    """Extract subtype and value from a metadata prompt."""
    if _META_NOTES.search(prompt):
        return {"subtype": "notes", "value": None}

    tag_match = _META_TAG.search(prompt)
    channel_match = _META_CHANNEL.search(prompt)

    # Channel and tag regexes can both match on a prompt like "was the chat
    # channel tagged 'urgent'?" — disambiguate by checking which role the
    # channel token plays in the sentence.
    if channel_match:
        ch = channel_match.group(1).lower()
        if re.search(r'source|channel|via|came\s+(?:from|through)|identify\s+if\s+source',
                      prompt, re.IGNORECASE):
            return {"subtype": "channel", "value": ch}
        if re.search(rf'(?:was|is)\s+(?:the\s+)?channel\s+{re.escape(ch)}',
                      prompt, re.IGNORECASE):
            return {"subtype": "channel", "value": ch}
        if re.search(r'via\s+' + re.escape(ch), prompt, re.IGNORECASE):
            return {"subtype": "channel", "value": ch}

    if tag_match:
        return {"subtype": "tag", "value": tag_match.group(1).lower()}

    if channel_match:
        return {"subtype": "channel", "value": channel_match.group(1).lower()}

    return None


# ── RUN EXTRACTION ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Running parameter extraction on all symbolic prompts...")
print(f"{'='*70}")

results = []

for _, row in sym.iterrows():
    prompt = row["prompt"]
    category = row["top_category"]
    subcategory = row["subcategory"]
    true_params = json.loads(row["params"]) if pd.notna(row["params"]) else {}

    if category == "SYMBOLIC_TIME":
        extracted = extract_time_params(prompt)
    elif category == "SYMBOLIC_COUNT":
        extracted = extract_count_params(prompt)
    elif category == "SYMBOLIC_METADATA":
        extracted = extract_metadata_params(prompt)
    else:
        extracted = None

    # Compare extracted vs true
    if extracted is None:
        match_threshold = False
        match_unit = False
        match_operator = False
        match_subtype = False
        match_value = False
        extracted_any = False
    else:
        extracted_any = True

        if category == "SYMBOLIC_TIME":
            match_threshold = (extracted.get("threshold") == true_params.get("threshold"))
            # Normalize units for comparison
            true_unit = true_params.get("unit", "")
            ext_unit = extracted.get("unit", "")
            unit_map = {"sec": "seconds", "min": "minutes", "mins": "minutes",
                        "hrs": "hours", "hr": "hours"}
            true_unit_norm = unit_map.get(true_unit, true_unit)
            ext_unit_norm = unit_map.get(ext_unit, ext_unit)
            match_unit = (true_unit_norm == ext_unit_norm)
            match_operator = True  # operator not stored in all params
            match_subtype = True
            match_value = True

        elif category == "SYMBOLIC_COUNT":
            match_threshold = (extracted.get("threshold") == true_params.get("threshold"))
            match_unit = True
            match_operator = True
            match_subtype = True
            match_value = True

        elif category == "SYMBOLIC_METADATA":
            match_subtype = (extracted.get("subtype") == true_params.get("subtype"))
            true_val = (true_params.get("value") or "").lower()
            ext_val = (extracted.get("value") or "").lower()
            match_value = (true_val == ext_val) if true_params.get("subtype") != "notes" else True
            match_threshold = True
            match_unit = True
            match_operator = True

    # Overall correctness: all relevant fields match
    if category == "SYMBOLIC_TIME":
        correct = extracted_any and match_threshold and match_unit
    elif category == "SYMBOLIC_COUNT":
        correct = extracted_any and match_threshold
    elif category == "SYMBOLIC_METADATA":
        correct = extracted_any and match_subtype and match_value
    else:
        correct = False

    results.append({
        "prompt": prompt[:100],
        "category": category,
        "subcategory": subcategory,
        "true_params": json.dumps(true_params),
        "extracted_params": json.dumps(extracted) if extracted else "",
        "extracted_any": extracted_any,
        "match_threshold": match_threshold,
        "match_unit": match_unit,
        "match_subtype": match_subtype,
        "match_value": match_value,
        "correct": correct,
    })

results_df = pd.DataFrame(results)
RESULTS_ANALYSES.mkdir(parents=True, exist_ok=True)
_out_csv = RESULTS_ANALYSES / "parameter_extraction.csv"
results_df.to_csv(_out_csv, index=False)
print(f"Saved {_out_csv} ({len(results_df)} rows)")

# ── REPORT ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PARAMETER EXTRACTION RESULTS")
print(f"{'='*70}")

# Overall
total = len(results_df)
extracted = results_df["extracted_any"].sum()
correct = results_df["correct"].sum()
print(f"\n  Overall: {correct}/{total} correct ({correct/total:.1%})")
print(f"  Extraction attempted: {extracted}/{total} ({extracted/total:.1%})")

# Per category
print(f"\n  Per category:")
for cat in ["SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"]:
    cat_df = results_df[results_df["category"] == cat]
    n = len(cat_df)
    ext = cat_df["extracted_any"].sum()
    cor = cat_df["correct"].sum()
    print(f"    {cat:<22}: {cor}/{n} correct ({cor/n:.1%}), "
          f"{ext}/{n} extracted ({ext/n:.1%})")

# Per subcategory (the real insight — explicit vs implicit)
print(f"\n  Per subcategory:")
for subcat in sorted(results_df["subcategory"].unique()):
    sub_df = results_df[results_df["subcategory"] == subcat]
    n = len(sub_df)
    ext = sub_df["extracted_any"].sum()
    cor = sub_df["correct"].sum()
    print(f"    {subcat:<22}: {cor}/{n} correct ({cor/n:.1%}), "
          f"{ext}/{n} extracted ({ext/n:.1%})")

# Error analysis
print(f"\n  Extraction failures (extracted_any=False):")
failures = results_df[~results_df["extracted_any"]]
print(f"    Total: {len(failures)}")
for cat in failures["category"].unique():
    cat_fails = failures[failures["category"] == cat]
    print(f"    {cat}: {len(cat_fails)} failures")
    for _, row in cat_fails.head(3).iterrows():
        print(f"      - \"{row['prompt']}\"")
        print(f"        true_params: {row['true_params']}")

print(f"\n  Extraction wrong (extracted but incorrect):")
wrong = results_df[results_df["extracted_any"] & ~results_df["correct"]]
print(f"    Total: {len(wrong)}")
for _, row in wrong.head(5).iterrows():
    print(f"    - [{row['category']}] \"{row['prompt']}\"")
    print(f"      true: {row['true_params']}")
    print(f"      extracted: {row['extracted_params']}")

print(f"\n{'=' * 70}")
print("Summary")
print(f"{'=' * 70}")
print(f"  {correct/total:.1%} of symbolic prompts parameterised by regex")
print(f"  {(total-correct)/total:.1%} fall back to the LLM (implicit thresholds or metadata disambiguation)")
