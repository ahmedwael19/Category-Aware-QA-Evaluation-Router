"""Template overlap + cosine dedup audit.

E1: quantify template-level overlap between train and test splits.
E2: rerun semantic dedup on LLM-generated prompts and report the actual
    near-duplicate count.

Run: python -m scripts.validation.template_overlap_and_dedup_audit
"""

import sys
import re
import json


import pandas as pd
import numpy as np

from config import (
    RESULTS_VALIDATION,
    DATASET_TRAIN, DATASET_VAL, DATASET_TEST, DATASET_FULL,
)

if __name__ != "__main__":
    raise RuntimeError("Run as: python -m scripts.validation.template_overlap_and_dedup_audit")

# ── E1: Template overlap analysis ─────────────────────────────────────────
print("=" * 80)
print("E1: TEMPLATE-LEVEL OVERLAP BETWEEN TRAIN AND TEST SPLITS")
print("=" * 80)

df_train = pd.read_csv(DATASET_TRAIN)
df_val = pd.read_csv(DATASET_VAL)
df_test = pd.read_csv(DATASET_TEST)
df_full = pd.read_csv(DATASET_FULL)

print(f"\nSplit sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
print(f"Total: {len(df_full)}")

# Approach 1: Normalize prompts by replacing digits + common variable parts,
# then check if a normalized 'template' form of a test prompt also appears in train.
# This is conservative because it treats SYMBOLIC prompts (which are templated
# with parameter swapping) as potentially overlapping.

NUM = re.compile(r"\d+")
WORD = re.compile(r"\w+")


def normalize_template(prompt: str) -> str:
    """Collapse digits, lowercase, strip punctuation-only variation.

    Turns templated prompts like 'Did the agent respond within 5 minutes?'
    and 'Did the agent respond within 10 minutes?' into the SAME template.
    """
    p = prompt.lower().strip()
    p = NUM.sub("<NUM>", p)
    # collapse whitespace
    p = re.sub(r"\s+", " ", p)
    return p


train_templates = set(normalize_template(p) for p in df_train["prompt"])
test_templates = set(normalize_template(p) for p in df_test["prompt"])

# Overall overlap
overlap_templates = train_templates & test_templates
overlap_pct = 100.0 * len(overlap_templates) / len(test_templates)

print(f"\n  Train unique templates (after <NUM> masking): {len(train_templates):,}")
print(f"  Test unique templates  (after <NUM> masking): {len(test_templates):,}")
print(f"  Test templates also appearing in Train:        {len(overlap_templates):,}")
print(f"  Overlap rate:                                  {overlap_pct:.1f}%")

# Per-category
print("\nPer-category template overlap (% of test prompts whose normalized template appears in train):")
for cat in sorted(df_train["top_category"].unique()):
    train_cat = df_train[df_train["top_category"] == cat]
    test_cat = df_test[df_test["top_category"] == cat]
    train_t = set(normalize_template(p) for p in train_cat["prompt"])
    test_t = [normalize_template(p) for p in test_cat["prompt"]]
    if not test_t:
        continue
    overlap_count = sum(1 for t in test_t if t in train_t)
    print(f"  {cat:<22}  test_n={len(test_t):>4}  overlap={overlap_count:>4}  ({100.0 * overlap_count / len(test_t):.1f}%)")

# More stringent check: character-level prompt hash (exact string match)
train_exact = set(df_train["prompt"].tolist())
test_exact = set(df_test["prompt"].tolist())
exact_overlap = train_exact & test_exact
print(f"\n  EXACT string overlap (should be zero after dedup): {len(exact_overlap)} prompts")

# Write E1 report
e1_report = {
    "train_unique_templates": len(train_templates),
    "test_unique_templates": len(test_templates),
    "test_templates_overlapping_train": len(overlap_templates),
    "overlap_percent": round(overlap_pct, 2),
    "exact_string_overlap": len(exact_overlap),
    "per_category_overlap_percent": {
        cat: round(
            100.0 * sum(
                1 for p in df_test[df_test["top_category"] == cat]["prompt"]
                if normalize_template(p) in set(
                    normalize_template(q) for q in df_train[df_train["top_category"] == cat]["prompt"]
                )
            ) / max(1, (df_test["top_category"] == cat).sum()),
            2
        )
        for cat in sorted(df_train["top_category"].unique())
    },
}

RESULTS_VALIDATION.mkdir(parents=True, exist_ok=True)
_e1_path = RESULTS_VALIDATION / "template_overlap.json"
with open(_e1_path, "w") as f:
    json.dump(e1_report, f, indent=2)
print(f"\nE1 report saved to: {_e1_path}")


# ── E2: Cosine deduplication removal count ────────────────────────────────
print()
print("=" * 80)
print("E2: SEMANTIC DEDUP REMOVAL COUNT AT COSINE THRESHOLD = 0.99")
print("=" * 80)

# The committed dataset is already post-dedup, so this re-runs the dedup
# logic on it — the reported count is near-zero (pairs that survived at
# 0.99) rather than the original removal count.

llm_gen = df_full[df_full["subcategory"].isin(["llm_generated", "llm_generated_hybrid", "llm_hybrid", "llm_unsupported"]) |
                  df_full["category"].str.startswith("SEMANTIC_", na=False)]

# Be more liberal: all prompts from LLM-generation strategies
# subcategory marks generation source in most rows
print(f"\n  Total rows flagged as LLM-generated in final (post-dedup) dataset: {len(llm_gen):,}")

# Breakdown by category
print("\n  LLM-generated prompt counts per top_category (in final dataset):")
for cat, count in df_full[
    df_full["subcategory"].str.contains("llm", na=False, case=False)
]["top_category"].value_counts().items():
    print(f"    {cat:<22}  {count}")

# Reports pair counts at several cosine thresholds so the 0.99 choice can be
# compared against paraphrase density in the post-dedup data.

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"\n  [SKIP] sentence_transformers not installed: {e}")
    sys.exit(0)

print("\n  Loading multilingual-e5-large-instruct (this takes a minute)...")
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# Dedup is per category. Run per category like the original script.
results_e2 = {"threshold_used": 0.99, "per_category": {}}

for cat in sorted(llm_gen["category"].unique()):
    cat_df = llm_gen[llm_gen["category"] == cat]
    if len(cat_df) < 2:
        continue
    prompts = cat_df["prompt"].tolist()
    embs = model.encode(prompts, show_progress_bar=False, batch_size=64)
    sim = cosine_similarity(embs)
    # Count pairs above threshold (upper triangle only)
    n = len(prompts)
    iu = np.triu_indices(n, k=1)
    pair_sims = sim[iu]
    pairs_at_099 = int((pair_sims > 0.99).sum())
    pairs_at_098 = int((pair_sims > 0.98).sum())
    pairs_at_095 = int((pair_sims > 0.95).sum())
    pairs_at_090 = int((pair_sims > 0.90).sum())
    max_sim = float(pair_sims.max()) if len(pair_sims) else 0.0
    pct_at_099 = 100.0 * pairs_at_099 / max(1, len(pair_sims))
    results_e2["per_category"][cat] = {
        "n_prompts": n,
        "n_pairs": len(pair_sims),
        "max_cosine": round(max_sim, 4),
        "pairs_above_0.99": pairs_at_099,
        "pairs_above_0.98": pairs_at_098,
        "pairs_above_0.95": pairs_at_095,
        "pairs_above_0.90": pairs_at_090,
    }
    print(f"  {cat:<22}  n={n:>4}  max_sim={max_sim:.4f}  "
          f"pairs>0.99={pairs_at_099}  >0.98={pairs_at_098}  >0.95={pairs_at_095}  >0.90={pairs_at_090}")

# Summary across all categories
total_pairs_099 = sum(v["pairs_above_0.99"] for v in results_e2["per_category"].values())
total_pairs_095 = sum(v["pairs_above_0.95"] for v in results_e2["per_category"].values())
total_pairs_090 = sum(v["pairs_above_0.90"] for v in results_e2["per_category"].values())

print(f"\n  TOTAL pairs in post-dedup data above thresholds (should be 0 at 0.99 if threshold worked):")
print(f"    > 0.99:  {total_pairs_099}")
print(f"    > 0.95:  {total_pairs_095}")
print(f"    > 0.90:  {total_pairs_090}")

results_e2["summary"] = {
    "total_pairs_above_0.99": total_pairs_099,
    "total_pairs_above_0.95": total_pairs_095,
    "total_pairs_above_0.90": total_pairs_090,
}

_e2_path = RESULTS_VALIDATION / "cosine_dedup_audit.json"
with open(_e2_path, "w") as f:
    json.dump(results_e2, f, indent=2)

print(f"\nE2 report saved to: {_e2_path}")
print("\nDone.")