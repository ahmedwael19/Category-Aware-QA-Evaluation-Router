"""Compute inter-annotator agreement and annotator-vs-gold accuracy.

Produces:
  1. Cohen's kappa (unweighted) — appropriate for nominal categories
  2. Bootstrap 95% CI on kappa
  3. Full confusion matrix and classification report
  4. Per-category agreement (Jaccard-like + F1)
  5. Annotator-vs-gold kappa with CI
  6. Landis & Koch interpretation (correct thresholds)

Run: python -m scripts.analyses.compute_kappa [annotator_1.csv] [annotator_2.csv]
"""

import sys

from config import DATA_DIR, RESULTS_ANALYSES

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

SEED = 42
N_BOOTSTRAP = 5000 

# ── Landis & Koch interpretation scale ────────────────────────────────
# Exact thresholds from the original paper.
LANDIS_KOCH = [
    (0.81, 1.00, "Almost perfect"),
    (0.61, 0.80, "Substantial"),
    (0.41, 0.60, "Moderate"),
    (0.21, 0.40, "Fair"),
    (0.00, 0.20, "Slight"),
    (-1.0, 0.00, "Poor"),
]


def interpret_kappa(k):
    for lo, hi, label in LANDIS_KOCH:
        if lo <= k <= hi:
            return label
    return "Unknown"


def bootstrap_kappa(labels_a, labels_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for Cohen's kappa using percentile method."""
    rng = np.random.RandomState(seed)
    n = len(labels_a)
    kappas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        kappas[i] = cohen_kappa_score(labels_a[idx], labels_b[idx])
    return np.percentile(kappas, 2.5), np.percentile(kappas, 97.5)


def main():
    # ── Load annotator files ──────────────────────────────────────────────
    f1 = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "annotations" / "annotation_study_annotator_1.csv")
    f2 = sys.argv[2] if len(sys.argv) > 2 else str(DATA_DIR / "annotations" / "annotation_study_annotator_2.csv")

    a1 = pd.read_csv(f1)
    a2 = pd.read_csv(f2)

    # Validate expected columns
    for name, df in [("Annotator 1", a1), ("Annotator 2", a2)]:
        required = {"id", "prompt", "annotator_category"}
        missing = required - set(df.columns)
        assert not missing, f"{name} file missing columns: {missing}"

    # ── Merge on ID (NOT positional) ──────────────────────────────────────
    merged = a1[["id", "prompt", "annotator_category"]].rename(
        columns={"annotator_category": "ann1"}
    ).merge(
        a2[["id", "annotator_category"]].rename(columns={"annotator_category": "ann2"}),
        on="id",
        how="inner",
    )

    # Drop unfilled rows (empty strings become NaN after read_csv)
    merged = merged.dropna(subset=["ann1", "ann2"])
    merged["ann1"] = merged["ann1"].astype(str).str.strip().str.upper()
    merged["ann2"] = merged["ann2"].astype(str).str.strip().str.upper()
    merged = merged[(merged["ann1"] != "") & (merged["ann2"] != "")]
    merged = merged[(merged["ann1"] != "NAN") & (merged["ann2"] != "NAN")]

    if len(merged) == 0:
        print("ERROR: No annotated rows found. Have annotators returned their files?")
        sys.exit(1)

    print(f"Prompts annotated by both: {len(merged)}")
    print(f"Categories found: {sorted(set(merged['ann1']) | set(merged['ann2']))}")
    print()

    # ── 1. INTER-ANNOTATOR AGREEMENT ──────────────────────────────────────────
    ann1_arr = merged["ann1"].values
    ann2_arr = merged["ann2"].values

    kappa = cohen_kappa_score(ann1_arr, ann2_arr)
    ci_lo, ci_hi = bootstrap_kappa(ann1_arr, ann2_arr)
    interp = interpret_kappa(kappa)

    print(f"Cohen's kappa (inter-annotator): {kappa:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Interpretation (Landis & Koch 1977): {interp}")
    print(f"  Note: Unweighted kappa used because categories are nominal (no ordinal relationship)")

    # Raw agreement
    agree = (ann1_arr == ann2_arr).mean()
    print(f"Raw agreement: {agree:.1%}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    CATS = sorted(set(ann1_arr) | set(ann2_arr))
    print(f"\nConfusion Matrix (Annotator 1 = rows, Annotator 2 = columns):")
    cm = confusion_matrix(ann1_arr, ann2_arr, labels=CATS)
    # Print with labels
    header = f"{'':>22}" + "".join(f"{c:>12}" for c in CATS)
    print(header)
    for i, cat in enumerate(CATS):
        row_str = f"{cat:>22}" + "".join(f"{cm[i, j]:>12}" for j in range(len(CATS)))
        print(row_str)

    # ── Classification report (treating ann1 as "reference") ──────────────
    print(f"\nClassification Report (Ann1 as reference, Ann2 as predicted):")
    print(classification_report(ann1_arr, ann2_arr, labels=CATS, zero_division=0))

    # ── Per-category agreement ────────────────────────────────────────────
    print("Per-category breakdown:")
    print(f"  {'Category':<22} {'Both':>5} {'Either':>6} {'Jaccard':>8} {'Ann1':>5} {'Ann2':>5}")
    for cat in CATS:
        mask1 = ann1_arr == cat
        mask2 = ann2_arr == cat
        both = int((mask1 & mask2).sum())
        either = int((mask1 | mask2).sum())
        jaccard = both / either if either > 0 else 0.0
        print(f"  {cat:<22} {both:>5} {either:>6} {jaccard:>8.3f} {mask1.sum():>5} {mask2.sum():>5}")

    # ── Disagreement analysis ─────────────────────────────────────────────
    disagree = merged[merged["ann1"] != merged["ann2"]]
    print(f"\nDisagreements: {len(disagree)} / {len(merged)} ({len(disagree)/len(merged)*100:.1f}%)")
    if len(disagree) > 0:
        print("\n| ID | Prompt (first 60 chars) | Ann1 | Ann2 |")
        print("|---|---|---|---|")
        for _, row in disagree.iterrows():
            p = row["prompt"][:60].replace("|", "\\|")
            print(f"| {row['id']} | {p} | {row['ann1']} | {row['ann2']} |")

    # ── 2. ANNOTATOR vs GROUND TRUTH ──────────────────────────────────────────
    key_file = DATA_DIR / "annotations" / "annotation_answer_key.csv"
    if key_file.exists():
        key = pd.read_csv(key_file)
        merged_gt = merged.merge(key[["id", "true_category"]], on="id", how="inner")
        merged_gt["true_category"] = merged_gt["true_category"].str.strip().str.upper()

        if len(merged_gt) > 0:
            print(f"\n{'='*60}")
            print("ANNOTATOR vs GROUND TRUTH (generation labels)")
            print(f"{'='*60}")
            print("Note: 'Ground truth' = labels from the synthetic generation process.")
            print("      This measures whether annotators agree with the taxonomy")
            print("      as operationalized in data generation, not absolute truth.\n")

            gt_arr = merged_gt["true_category"].values
            a1_gt_arr = merged_gt["ann1"].values
            a2_gt_arr = merged_gt["ann2"].values

            k1_gt = cohen_kappa_score(a1_gt_arr, gt_arr)
            k1_ci = bootstrap_kappa(a1_gt_arr, gt_arr)
            k2_gt = cohen_kappa_score(a2_gt_arr, gt_arr)
            k2_ci = bootstrap_kappa(a2_gt_arr, gt_arr)

            print(f"Annotator 1 vs GT: kappa={k1_gt:.3f} [{k1_ci[0]:.3f}, {k1_ci[1]:.3f}] ({interpret_kappa(k1_gt)})")
            print(f"Annotator 2 vs GT: kappa={k2_gt:.3f} [{k2_ci[0]:.3f}, {k2_ci[1]:.3f}] ({interpret_kappa(k2_gt)})")

            # Per-category GT accuracy
            print(f"\nPer-category accuracy vs GT:")
            print(f"  {'Category':<22} {'Ann1':>6} {'Ann2':>6} {'N':>4}")
            for cat in sorted(set(gt_arr)):
                mask = gt_arr == cat
                n_cat = mask.sum()
                acc1 = (a1_gt_arr[mask] == gt_arr[mask]).mean() if n_cat > 0 else 0
                acc2 = (a2_gt_arr[mask] == gt_arr[mask]).mean() if n_cat > 0 else 0
                print(f"  {cat:<22} {acc1:>5.1%} {acc2:>5.1%} {n_cat:>4}")
    else:
        print(f"\nNo answer key found at {key_file} — skipping GT comparison.")

    # ── 3. SAVE RESULTS ───────────────────────────────────────────────────────
    results = {
        "inter_annotator_kappa": round(kappa, 4),
        "kappa_ci_lo": round(ci_lo, 4),
        "kappa_ci_hi": round(ci_hi, 4),
        "kappa_interpretation": interp,
        "raw_agreement": round(agree, 4),
        "n_annotated": len(merged),
        "n_disagreements": len(disagree),
        "n_categories": len(CATS),
        "bootstrap_iterations": N_BOOTSTRAP,
    }

    if key_file.exists() and len(merged_gt) > 0:
        results["ann1_vs_gt_kappa"] = round(k1_gt, 4)
        results["ann2_vs_gt_kappa"] = round(k2_gt, 4)

    import json
    RESULTS_ANALYSES.mkdir(parents=True, exist_ok=True)
    out = RESULTS_ANALYSES / "annotation_kappa.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # Save disagreements for resolution
    if len(disagree) > 0:
        disagree.to_csv(DATA_DIR / "annotations" / "annotation_disagreements.csv", index=False)
        print(f"Saved: {DATA_DIR / 'annotations' / 'annotation_disagreements.csv'} ({len(disagree)} rows for resolution)")


if __name__ == "__main__":
    main()
