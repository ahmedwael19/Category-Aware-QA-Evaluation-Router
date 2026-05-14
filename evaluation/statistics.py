"""Statistical analysis: McNemar's test, bootstrap CIs, per-category accuracy."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


_BASELINES_RESCORED = "baselines_rescored.csv"
_ROUTER_VARIANTS_RESCORED = "router_variants_per_prompt.csv"

_LLM_SYSTEMS = [
    ("LLM GPT-4o-mini (bin)", "mini_bin_correct", "mini_bin_tokens", "mini_bin_latency"),
    ("LLM GPT-4o-mini (rej)", "mini_rej_correct", "mini_rej_tokens", "mini_rej_latency"),
    ("LLM GPT-4o (bin)",      "gpt4o_bin_correct", "gpt4o_bin_tokens", None),
    ("LLM GPT-4o (rej)",      "gpt4o_rej_correct", "gpt4o_rej_tokens", None),
]

_ROUTER_SYSTEMS = [
    ("Router (TF-IDF+SVM)", "svm_correct"),
    ("Router (TF-IDF+LR)",  "tfidf_lr_correct"),
    ("Router (Emb+LR)",     "emb_lr_correct"),
    ("Router (Ensemble)",   "ensemble_correct"),
    ("Router (Keyword)",    "keyword_correct"),
]


def _bootstrap_ci(values, n_boot=5000):
    boots = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _mcnemar(baseline_correct, router_correct):
    b = int(((1 - baseline_correct) * router_correct).sum())
    c = int((baseline_correct * (1 - router_correct)).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, chi2_stat, float(1 - chi2.cdf(chi2_stat, df=1))


def _cohens_h(p1, p2):
    p1 = max(0.001, min(0.999, p1))
    p2 = max(0.001, min(0.999, p2))
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def _holm_bonferroni(p_values):
    k = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.ones(k)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (k - rank))
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def compute_all_statistics(
    eval_with_llm,
    system_results,
    results_e2e_dir,
    n_boot=5000,
    seed=42,
):
    """Compute McNemar, bootstrap CIs, per-category accuracy from the rescored sources."""
    np.random.seed(seed)

    e2e_dir = Path(results_e2e_dir)
    baselines = pd.read_csv(e2e_dir / _BASELINES_RESCORED)
    routers = pd.read_csv(e2e_dir / _ROUTER_VARIANTS_RESCORED)

    if len(baselines) != len(routers):
        raise ValueError(f"Row mismatch: {_BASELINES_RESCORED} ({len(baselines)}) vs {_ROUTER_VARIANTS_RESCORED} ({len(routers)})")
    n = len(baselines)
    categories = baselines["category"].values

    summary_rows = []
    for name, col, tok_col, lat_col in _LLM_SYSTEMS:
        lat_mean = lat_p50 = lat_p95 = 0.0
        if lat_col and lat_col in baselines.columns:
            lat = baselines[lat_col].values
            lat_mean = round(float(np.mean(lat)), 1)
            lat_p50 = round(float(np.median(lat)), 1)
            lat_p95 = round(float(np.percentile(lat, 95)), 1)
        summary_rows.append({
            "system": name,
            "accuracy": round(float(baselines[col].mean()), 4),
            "n": n,
            "total_tokens": int(baselines[tok_col].sum()),
            "mean_latency_ms": lat_mean,
            "p50_latency_ms": lat_p50,
            "p95_latency_ms": lat_p95,
        })

    for name, col in _ROUTER_SYSTEMS:
        summary_rows.append({
            "system": name,
            "accuracy": round(float(routers[col].mean()), 4),
            "n": n,
            "total_tokens": "",
            "mean_latency_ms": "",
            "p50_latency_ms": "",
            "p95_latency_ms": "",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("System Comparison Summary:")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("McNEMAR'S TEST: Router vs LLM-only")
    print(f"{'='*80}")

    mcnemar_rows = []
    baseline_pairs = [
        ("LLM GPT-4o-mini (bin)", baselines["mini_bin_correct"].values),
        ("LLM GPT-4o-mini (rej)", baselines["mini_rej_correct"].values),
        ("LLM GPT-4o (bin)",      baselines["gpt4o_bin_correct"].values),
        ("LLM GPT-4o (rej)",      baselines["gpt4o_rej_correct"].values),
    ]
    for bl_name, bl_vec in baseline_pairs:
        p_bl = float(bl_vec.mean())
        print(f"\n  vs {bl_name} (acc={p_bl:.3f}):")
        for r_name, r_col in _ROUTER_SYSTEMS:
            r_vec = routers[r_col].values
            b, c, chi2_stat, p_val = _mcnemar(bl_vec, r_vec)
            h = _cohens_h(p_bl, float(r_vec.mean()))
            mcnemar_rows.append({
                "baseline": bl_name, "router": r_name,
                "b_router_wins": b, "c_baseline_wins": c,
                "chi2": round(chi2_stat, 4), "p_value": round(p_val, 6),
                "cohens_h": round(h, 4), "significant": p_val < 0.05,
            })
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {r_name:<22} b={b:>3} c={c:>3} chi2={chi2_stat:.2f} p={p_val:.4f} h={h:+.3f} {sig}")

    holm = _holm_bonferroni([row["p_value"] for row in mcnemar_rows])
    for i, row in enumerate(mcnemar_rows):
        row["p_holm"] = round(float(holm[i]), 6)
        row["significant_holm"] = bool(holm[i] < 0.05)

    print(f"\n  Holm-Bonferroni correction applied ({len(mcnemar_rows)} tests):")
    for row in mcnemar_rows:
        sig = "***" if row["p_holm"] < 0.001 else "**" if row["p_holm"] < 0.01 else "*" if row["p_holm"] < 0.05 else "ns"
        print(f"    {row['baseline'][:20]:<22} vs {row['router']:<22} p_raw={row['p_value']:.4f} p_holm={row['p_holm']:.4f} {sig}")

    print(f"\n{'='*80}")
    print("BOOTSTRAP 95% CIs")
    print(f"{'='*80}")

    ci_rows = []
    for bl_name, bl_vec in baseline_pairs:
        lo, hi = _bootstrap_ci(bl_vec, n_boot)
        ci_rows.append({"system": bl_name, "accuracy": round(float(bl_vec.mean()), 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    for r_name, r_col in _ROUTER_SYSTEMS:
        r_vec = routers[r_col].values
        lo, hi = _bootstrap_ci(r_vec, n_boot)
        ci_rows.append({"system": r_name, "accuracy": round(float(r_vec.mean()), 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    for r in ci_rows:
        print(f"  {r['system']:<28} acc={r['accuracy']:.3f} CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    print(f"\n{'='*80}")
    print("PER-CATEGORY BOOTSTRAP 95% CIs")
    print(f"{'='*80}")

    cat_ci_rows = []
    for cat in sorted(set(categories)):
        mask = categories == cat
        n_cat = int(mask.sum())
        for bl_name, bl_vec in baseline_pairs:
            vals = bl_vec[mask]
            if len(vals) > 1:
                lo, hi = _bootstrap_ci(vals, n_boot)
                cat_ci_rows.append({"system": bl_name, "category": cat, "n": n_cat,
                                    "accuracy": round(float(vals.mean()), 4),
                                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        for r_name, r_col in _ROUTER_SYSTEMS:
            vals = routers[r_col].values[mask]
            if len(vals) > 1:
                lo, hi = _bootstrap_ci(vals, n_boot)
                cat_ci_rows.append({"system": r_name, "category": cat, "n": n_cat,
                                    "accuracy": round(float(vals.mean()), 4),
                                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    cat_ci_df = pd.DataFrame(cat_ci_rows)
    cat_ci_df.to_csv(e2e_dir / "per_category_ci.csv", index=False)
    print("Saved per_category_ci.csv")

    print(f"\n{'='*80}")
    print("PER-CATEGORY ACCURACY")
    print(f"{'='*80}")

    cat_rows = []
    for cat in sorted(set(categories)):
        mask = categories == cat
        row = {"category": cat, "n": int(mask.sum())}
        for bl_name, bl_vec in baseline_pairs:
            row[bl_name] = round(float(bl_vec[mask].mean()), 4)
        for r_name, r_col in _ROUTER_SYSTEMS:
            row[r_name] = round(float(routers[r_col].values[mask].mean()), 4)
        cat_rows.append(row)

    cat_df = pd.DataFrame(cat_rows)
    print(cat_df.to_string(index=False))

    summary_df.to_csv(e2e_dir / "summary.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(e2e_dir / "mcnemar.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(e2e_dir / "bootstrap_ci.csv", index=False)
    cat_df.to_csv(e2e_dir / "per_category.csv", index=False)

    print(f"\nSaved: summary.csv, mcnemar.csv, bootstrap_ci.csv, per_category.csv, per_category_ci.csv")

    return {
        "summary_df": summary_df,
        "mcnemar_df": pd.DataFrame(mcnemar_rows),
        "ci_df": pd.DataFrame(ci_rows),
        "cat_ci_df": cat_ci_df,
        "cat_df": cat_df,
    }
