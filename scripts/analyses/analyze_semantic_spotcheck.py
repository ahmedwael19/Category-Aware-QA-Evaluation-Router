"""Compute human-annotator agreement with GPT-5.2 on the semantic spot-check.

Reports Cohen's kappa with bootstrap 95% CI, PABAK (prevalence-adjusted),
per-category agreement, and a pre-registered acceptance check (kappa >= 0.70
validates, 0.50-0.70 usable with caveats, < 0.50 unreliable).

Run: python -m scripts.analyses.analyze_semantic_spotcheck [annotator_1.csv] [annotator_2.csv]
"""

import sys

from config import DATA_DIR, RESULTS_ANALYSES

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SEED = 42
N_BOOTSTRAP = 5000

# Pre-registered acceptance thresholds
ACCEPT_THRESHOLD = 0.70
CAUTION_THRESHOLD = 0.50


def bootstrap_kappa(a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(a)
    kappas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            kappas[i] = cohen_kappa_score(a[idx], b[idx])
        except ValueError:
            kappas[i] = np.nan
    kappas = kappas[~np.isnan(kappas)]
    return np.percentile(kappas, 2.5), np.percentile(kappas, 97.5)


def compute_pabak(a, b):
    """Prevalence-adjusted bias-adjusted kappa (Byrt et al. 1993).

    Addresses the known issue that Cohen's kappa is deflated when
    prevalence is high (e.g., 73% "no" answers). PABAK = 2*p_o - 1
    where p_o is the observed agreement proportion.
    """
    p_o = (a == b).mean()
    return 2 * p_o - 1


def main():
    # ── Load annotator files ──────────────────────────────────────────────
    f1 = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "annotations" / "semantic_spotcheck_annotator_1.csv")
    f2 = sys.argv[2] if len(sys.argv) > 2 else str(DATA_DIR / "annotations" / "semantic_spotcheck_annotator_2.csv")
    key_file = DATA_DIR / "annotations" / "semantic_spotcheck_answer_key.csv"

    a1 = pd.read_csv(f1, sep=None, engine="python")
    a2 = pd.read_csv(f2, sep=None, engine="python")

    # Validate
    for name, df in [("Annotator 1", a1), ("Annotator 2", a2)]:
        assert "annotator_answer" in df.columns, f"{name}: missing 'annotator_answer' column"
        assert "id" in df.columns, f"{name}: missing 'id' column"

    # Merge on ID (not positional)
    merged = a1[["id", "category", "prompt", "annotator_answer"]].rename(
        columns={"annotator_answer": "ann1"}
    ).merge(
        a2[["id", "annotator_answer"]].rename(columns={"annotator_answer": "ann2"}),
        on="id", how="inner",
    )

    # Drop unfilled
    merged = merged.dropna(subset=["ann1", "ann2"])
    if len(merged) == 0:
        print("No completed annotations in the spot-check CSVs. "
              "Fill `semantic_spotcheck_annotator_1.csv` and "
              "`semantic_spotcheck_annotator_2.csv` in data/annotations/ before running.")
        sys.exit(0)
    merged["ann1"] = merged["ann1"].astype(str).str.strip().str.lower()
    merged["ann2"] = merged["ann2"].astype(str).str.strip().str.lower()
    merged = merged[merged["ann1"].isin({"yes", "no"})]
    merged = merged[merged["ann2"].isin({"yes", "no"})]

    if len(merged) == 0:
        print("ERROR: No yes/no annotations found after filtering.")
        sys.exit(1)

    print(f"Annotated pairs: {len(merged)}")
    print(f"  SEMANTIC: {(merged['category'] == 'SEMANTIC').sum()}")
    print(f"  HYBRID: {(merged['category'] == 'HYBRID').sum()}")
    print()

    # ── Inter-annotator agreement (human vs human) ─────────────────────────
    ann1_arr = merged["ann1"].values
    ann2_arr = merged["ann2"].values

    kappa_inter = cohen_kappa_score(ann1_arr, ann2_arr)
    ci_inter = bootstrap_kappa(ann1_arr, ann2_arr)
    pabak_inter = compute_pabak(ann1_arr, ann2_arr)
    agree_inter = (ann1_arr == ann2_arr).mean()

    print("INTER-ANNOTATOR AGREEMENT (human vs human)")
    print(f"  Cohen's kappa: {kappa_inter:.3f} [{ci_inter[0]:.3f}, {ci_inter[1]:.3f}]")
    print(f"  PABAK:         {pabak_inter:.3f}")
    print(f"  Raw agreement: {agree_inter:.1%}")

    # Per-category
    for cat in ["SEMANTIC", "HYBRID"]:
        mask = merged["category"] == cat
        if mask.sum() < 5:
            continue
        a1c = merged.loc[mask, "ann1"].values
        a2c = merged.loc[mask, "ann2"].values
        try:
            k = cohen_kappa_score(a1c, a2c)
        except ValueError:
            k = float("nan")
        print(f"  {cat}: kappa={k:.3f} agree={( a1c == a2c).mean():.1%} (n={mask.sum()})")

    # ── Human vs GPT-5.2 agreement ─────────────────────────────────────────
    if not key_file.exists():
        print(f"\nAnswer key not found at {key_file} — cannot compute human-GPT agreement.")
        return

    key = pd.read_csv(key_file)
    # Merge key using source_idx
    if "source_idx" in merged.columns and "source_idx" in key.columns:
        merged_gt = merged.merge(key[["source_idx", "gpt5_2_answer"]], on="source_idx", how="inner")
    else:
        # Fallback: merge on prompt text
        merged_gt = merged.merge(key[["prompt", "gpt5_2_answer"]], on="prompt", how="inner")

    merged_gt["gpt"] = merged_gt["gpt5_2_answer"].str.strip().str.lower()

    print(f"\n{'='*60}")
    print("HUMAN vs GPT-5.2 AGREEMENT")
    print(f"{'='*60}")

    # Prevalence
    gpt_yes_pct = (merged_gt["gpt"] == "yes").mean()
    print(f"\nGPT answer prevalence: {gpt_yes_pct:.1%} yes, {1-gpt_yes_pct:.1%} no")

    # Each annotator vs GPT
    for ann_name, ann_col in [("Annotator 1", "ann1"), ("Annotator 2", "ann2")]:
        a_arr = merged_gt[ann_col].values
        g_arr = merged_gt["gpt"].values

        k = cohen_kappa_score(a_arr, g_arr)
        ci = bootstrap_kappa(a_arr, g_arr)
        pabak = compute_pabak(a_arr, g_arr)
        agree = (a_arr == g_arr).mean()

        print(f"\n{ann_name} vs GPT-5.2:")
        print(f"  Cohen's kappa: {k:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  PABAK:         {pabak:.3f}")
        print(f"  Raw agreement: {agree:.1%}")

        # Confusion matrix
        cm = confusion_matrix(g_arr, a_arr, labels=["yes", "no"])
        print(f"  Confusion (GPT rows, Human cols):  yes={cm[0,0]:>2}/{cm[0,0]+cm[0,1]:>2}  no={cm[1,1]:>2}/{cm[1,0]+cm[1,1]:>2}")

        # Per-category
        for cat in ["SEMANTIC", "HYBRID"]:
            mask = merged_gt["category"] == cat
            if mask.sum() < 5:
                continue
            ac = merged_gt.loc[mask, ann_col].values
            gc = merged_gt.loc[mask, "gpt"].values
            try:
                kc = cohen_kappa_score(ac, gc)
            except ValueError:
                kc = float("nan")
            print(f"  {cat}: kappa={kc:.3f} agree={(ac == gc).mean():.1%} (n={mask.sum()})")

        # Acceptance threshold check
        print(f"\n  ACCEPTANCE CHECK: kappa={k:.3f}", end="")
        if k >= ACCEPT_THRESHOLD:
            print(f" >= {ACCEPT_THRESHOLD} → PASS: GPT-5.2 ground truth validated")
        elif k >= CAUTION_THRESHOLD:
            print(f" >= {CAUTION_THRESHOLD} → CAUTION: GPT-5.2 usable with caveats")
        else:
            print(f" < {CAUTION_THRESHOLD} → FAIL: GPT-5.2 ground truth unreliable")

    # ── Disagreement analysis by GPT answer ───────────────────────────────
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'='*60}")

    for ann_col in ["ann1", "ann2"]:
        disagree = merged_gt[merged_gt[ann_col] != merged_gt["gpt"]]
        print(f"\n{ann_col} disagrees with GPT on {len(disagree)}/{len(merged_gt)} cases:")
        # By GPT answer
        for gpt_ans in ["yes", "no"]:
            n_total = (merged_gt["gpt"] == gpt_ans).sum()
            n_disagree = ((merged_gt["gpt"] == gpt_ans) & (merged_gt[ann_col] != merged_gt["gpt"])).sum()
            print(f"  GPT={gpt_ans}: {n_disagree}/{n_total} disagreements ({n_disagree/n_total*100:.0f}%)")

    # ── Save results ──────────────────────────────────────────────────────
    import json
    results = {
        "n_pairs": len(merged),
        "inter_annotator_kappa": round(kappa_inter, 4),
        "inter_annotator_pabak": round(pabak_inter, 4),
        "gpt_prevalence_yes": round(gpt_yes_pct, 3),
        "acceptance_threshold": ACCEPT_THRESHOLD,
        "bootstrap_n": N_BOOTSTRAP,
    }

    for ann_name, ann_col in [("ann1", "ann1"), ("ann2", "ann2")]:
        a_arr = merged_gt[ann_col].values
        g_arr = merged_gt["gpt"].values
        results[f"{ann_name}_vs_gpt_kappa"] = round(cohen_kappa_score(a_arr, g_arr), 4)
        results[f"{ann_name}_vs_gpt_agreement"] = round((a_arr == g_arr).mean(), 4)

    RESULTS_ANALYSES.mkdir(parents=True, exist_ok=True)
    out = RESULTS_ANALYSES / "semantic_spotcheck.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
