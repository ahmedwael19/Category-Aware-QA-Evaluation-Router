"""Statistical analysis: McNemar's test, bootstrap CIs, per-category accuracy.

"""
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2


def compute_all_statistics(
    eval_with_llm,
    system_results,
    results_e2e_dir,
    n_boot=5000,
    seed=42,
):
    """Compute McNemar's test, Cohen's h effect size, bootstrap CIs (overall + per-category).

    Parameters
    ----------
    eval_with_llm : pd.DataFrame
        Evaluation dataset with LLM baseline results.
    system_results : pd.DataFrame
        Full comparison results from run_full_comparison.
    results_e2e_dir : str
        Directory to save result CSVs.
    n_boot : int
        Number of bootstrap resamples (>= 5000 for publication-grade).
    seed : int
        Random seed for bootstrap.

    Returns
    -------
    dict
        Keys: 'summary_df', 'mcnemar_df', 'ci_df', 'cat_ci_df', 'cat_df'.
    """
    np.random.seed(seed)

    # ── Per-system accuracy ───────────────────────────────────────────
    _summary_rows = [
        {
            "system": "LLM-only (binary)",
            "accuracy": round(eval_with_llm["system_a_binary_correct"].mean(), 4),
            "n": len(eval_with_llm),
            "total_tokens": int(eval_with_llm["system_a_binary_tokens"].sum()),
            "mean_latency_ms": round(eval_with_llm["system_a_binary_latency_ms"].mean(), 1),
            "p50_latency_ms": round(eval_with_llm["system_a_binary_latency_ms"].median(), 1),
            "p95_latency_ms": round(np.percentile(eval_with_llm["system_a_binary_latency_ms"], 95), 1),
        },
        {
            "system": "LLM-only (with reject)",
            "accuracy": round(eval_with_llm["system_a_reject_correct"].mean(), 4),
            "n": len(eval_with_llm),
            "total_tokens": int(eval_with_llm["system_a_reject_tokens"].sum()),
            "mean_latency_ms": round(eval_with_llm["system_a_reject_latency_ms"].mean(), 1),
            "p50_latency_ms": round(eval_with_llm["system_a_reject_latency_ms"].median(), 1),
            "p95_latency_ms": round(np.percentile(eval_with_llm["system_a_reject_latency_ms"], 95), 1),
        },
    ]

    for _rname in system_results["router"].unique():
        _sub = system_results[system_results["router"] == _rname]
        _summary_rows.append({
            "system": f"Router ({_rname})",
            "accuracy": round(_sub["correct"].mean(), 4),
            "n": len(_sub),
            "total_tokens": int(_sub["tokens"].sum()),
            "mean_latency_ms": round(_sub["latency_ms"].mean(), 1),
            "p50_latency_ms": round(_sub["latency_ms"].median(), 1),
            "p95_latency_ms": round(np.percentile(_sub["latency_ms"], 95), 1),
        })

    _summary_df = pd.DataFrame(_summary_rows)
    print("System Comparison Summary:")
    print(_summary_df.to_string(index=False))

    # ── McNemar's test: each router vs LLM-only ──────────────────────
    print(f"\n{'='*80}")
    print("McNEMAR'S TEST: Router vs LLM-only")
    print(f"{'='*80}")

    _mcnemar_rows = []

    for _bl_name, _bl_col in [("LLM-only (binary)", "system_a_binary_correct"), ("LLM-only (with reject)", "system_a_reject_correct")]:
        _sys_a = eval_with_llm[_bl_col].values
        _p_a = eval_with_llm[_bl_col].mean()
        print(f"\n  vs {_bl_name} (acc={_p_a:.3f}):")

        for _rname in system_results["router"].unique():
            _sub = system_results[system_results["router"] == _rname].sort_values("idx")
            _sys_b = _sub["correct"].values

            _b = int(((1 - _sys_a) * _sys_b).sum())
            _c = int((_sys_a * (1 - _sys_b)).sum())

            # Edwards continuity-corrected McNemar's test (more conservative
            # for small discordant pairs; standard for N < 1000)
            _chi2_stat = (abs(_b - _c) - 1) ** 2 / (_b + _c) if (_b + _c) > 0 else 0.0
            _p_value = 1 - chi2.cdf(_chi2_stat, df=1) if (_b + _c) > 0 else 1.0

            _sub_acc = _sub["correct"].mean()
            _h = 2 * math.asin(math.sqrt(max(0.001, min(0.999, _sub_acc)))) - 2 * math.asin(math.sqrt(max(0.001, min(0.999, _p_a))))

            _mcnemar_rows.append({
                "baseline": _bl_name, "router": _rname,
                "b_router_wins": _b, "c_baseline_wins": _c,
                "chi2": round(_chi2_stat, 4), "p_value": round(_p_value, 6),
                "cohens_h": round(_h, 4), "significant": _p_value < 0.05,
            })
            _sig = "***" if _p_value < 0.001 else "**" if _p_value < 0.01 else "*" if _p_value < 0.05 else "ns"
            print(f"    {_rname:<20} b={_b:>3} c={_c:>3} chi2={_chi2_stat:.2f} p={_p_value:.4f} h={_h:+.3f} {_sig}")

    # ── Holm-Bonferroni correction for multiple comparisons ───────────
    _raw_ps = [r["p_value"] for r in _mcnemar_rows]
    _sorted_indices = np.argsort(_raw_ps)
    _k = len(_raw_ps)
    _corrected = np.ones(_k)
    for _rank, _orig_idx in enumerate(_sorted_indices):
        _corrected[_orig_idx] = min(1.0, _raw_ps[_orig_idx] * (_k - _rank))

    # Enforce monotonicity
    _running_max = 0.0
    for _rank, _orig_idx in enumerate(_sorted_indices):
        _corrected[_orig_idx] = max(_corrected[_orig_idx], _running_max)
        _running_max = _corrected[_orig_idx]

    for i, r in enumerate(_mcnemar_rows):
        r["p_holm"] = round(float(_corrected[i]), 6)
        r["significant_holm"] = _corrected[i] < 0.05

    print(f"\n  Holm-Bonferroni correction applied ({_k} tests):")
    for r in _mcnemar_rows:
        _sig_h = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else "*" if r["p_holm"] < 0.05 else "ns"
        print(f"    {r['baseline'][:15]:<16} vs {r['router']:<18} p_raw={r['p_value']:.4f} p_holm={r['p_holm']:.4f} {_sig_h}")

    # ── Bootstrap 95% CIs ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("BOOTSTRAP 95% CIs")
    print(f"{'='*80}")

    _ci_rows = []

    # Both baselines
    for _bl_name, _bl_col in [("LLM-only (binary)", "system_a_binary_correct"), ("LLM-only (with reject)", "system_a_reject_correct")]:
        _vals = eval_with_llm[_bl_col].values
        _boots = [np.mean(np.random.choice(_vals, size=len(_vals), replace=True)) for _ in range(n_boot)]
        _ci_rows.append({"system": _bl_name, "accuracy": round(np.mean(_vals), 4),
                          "ci_lo": round(np.percentile(_boots, 2.5), 4),
                          "ci_hi": round(np.percentile(_boots, 97.5), 4)})

    for _rname in system_results["router"].unique():
        _sub = system_results[system_results["router"] == _rname]
        _vals = _sub["correct"].values
        _boots = [np.mean(np.random.choice(_vals, size=len(_vals), replace=True)) for _ in range(n_boot)]
        _ci_rows.append({"system": f"Router ({_rname})", "accuracy": round(np.mean(_vals), 4),
                          "ci_lo": round(np.percentile(_boots, 2.5), 4),
                          "ci_hi": round(np.percentile(_boots, 97.5), 4)})

    for _r in _ci_rows:
        print(f"  {_r['system']:<30} acc={_r['accuracy']:.3f} CI=[{_r['ci_lo']:.3f}, {_r['ci_hi']:.3f}]")

    # ── Per-category bootstrap CIs ────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-CATEGORY BOOTSTRAP 95% CIs")
    print(f"{'='*80}")

    _cat_ci_rows = []
    for _cat in sorted(eval_with_llm["category"].unique()):
        _cat_mask = eval_with_llm["category"] == _cat

        # Both baselines
        for _bl_name, _bl_col in [("LLM-only (binary)", "system_a_binary_correct"), ("LLM-only (with reject)", "system_a_reject_correct")]:
            _vals = eval_with_llm.loc[_cat_mask, _bl_col].values
            if len(_vals) > 1:
                _boots = [np.mean(np.random.choice(_vals, size=len(_vals), replace=True)) for _ in range(n_boot)]
                _cat_ci_rows.append({"system": _bl_name, "category": _cat, "n": len(_vals),
                                      "accuracy": round(np.mean(_vals), 4),
                                      "ci_lo": round(np.percentile(_boots, 2.5), 4),
                                      "ci_hi": round(np.percentile(_boots, 97.5), 4)})

        # Each router
        for _rname in system_results["router"].unique():
            _sub = system_results[(system_results["router"] == _rname) & (system_results["true_category"] == _cat)]
            _vals = _sub["correct"].values
            if len(_vals) > 1:
                _boots = [np.mean(np.random.choice(_vals, size=len(_vals), replace=True)) for _ in range(n_boot)]
                _cat_ci_rows.append({"system": f"Router ({_rname})", "category": _cat, "n": len(_vals),
                                      "accuracy": round(np.mean(_vals), 4),
                                      "ci_lo": round(np.percentile(_boots, 2.5), 4),
                                      "ci_hi": round(np.percentile(_boots, 97.5), 4)})

    pd.DataFrame(_cat_ci_rows).to_csv(f"{results_e2e_dir}/per_category_ci.csv", index=False)
    print(f"Saved per_category_ci.csv")

    # ── Per-category breakdown ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-CATEGORY ACCURACY")
    print(f"{'='*80}")

    _cat_rows = []
    for _cat in sorted(eval_with_llm["category"].unique()):
        _cat_mask = eval_with_llm["category"] == _cat
        _n_cat = _cat_mask.sum()
        _row = {"category": _cat, "n": _n_cat}

        _row["LLM-only (binary)"] = round(eval_with_llm.loc[_cat_mask, "system_a_binary_correct"].mean(), 4)
        _row["LLM-only (reject)"] = round(eval_with_llm.loc[_cat_mask, "system_a_reject_correct"].mean(), 4)

        for _rname in system_results["router"].unique():
            _sub = system_results[(system_results["router"] == _rname)]
            _cat_sub = _sub[_sub["true_category"] == _cat]
            _row[f"Router ({_rname})"] = round(_cat_sub["correct"].mean(), 4) if len(_cat_sub) > 0 else None

        _cat_rows.append(_row)

    _cat_df = pd.DataFrame(_cat_rows)
    print(_cat_df.to_string(index=False))

    # ── Save all results ──────────────────────────────────────────────
    _summary_df.to_csv(f"{results_e2e_dir}/summary.csv", index=False)
    pd.DataFrame(_mcnemar_rows).to_csv(f"{results_e2e_dir}/mcnemar.csv", index=False)
    pd.DataFrame(_ci_rows).to_csv(f"{results_e2e_dir}/bootstrap_ci.csv", index=False)
    _cat_df.to_csv(f"{results_e2e_dir}/per_category.csv", index=False)

    # GPT-5.2 judge cost (for reproducibility reporting)
    _gt_calls = eval_with_llm[eval_with_llm["ground_truth_source"] == "gpt5.2_judge"].shape[0]
    _gt_total_tokens = int(eval_with_llm["gt_tokens"].sum())
    print(f"\nGPT-5.2 judge: {_gt_calls} calls, {_gt_total_tokens:,} tokens total")

    print(f"Saved: summary.csv, mcnemar.csv, bootstrap_ci.csv, per_category.csv, per_category_ci.csv")

    return {
        "summary_df": _summary_df,
        "mcnemar_df": pd.DataFrame(_mcnemar_rows),
        "ci_df": pd.DataFrame(_ci_rows),
        "cat_ci_df": pd.DataFrame(_cat_ci_rows),
        "cat_df": _cat_df,
    }
