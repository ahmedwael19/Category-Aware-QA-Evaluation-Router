"""Post-hoc analysis of the reasoning-model comparison.

Reads `results/reasoning/raw.csv` and `results/e2e/reliability.csv` and
produces the per-category, overthinking, and error-overlap analyses cited in
the reasoning-model section of the results chapter. No API calls.

Run: python -m scripts.baselines.reasoning_model_posthoc
"""


from config import DATA_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
from scipy import stats

if __name__ != "__main__":
    raise RuntimeError("This script must be invoked as a main program, e.g. python -m scripts.baselines.reasoning_model_posthoc")


df = pd.read_csv(RESULTS_DIR / "reasoning" / "raw.csv")
rel = pd.read_csv(RESULTS_DIR / "e2e" / "reliability.csv")
print(f"Loaded {len(df)} reasoning model rows, {len(rel)} reliability rows")
print(f"Models: {df['model'].unique().tolist()}")

# ── 1. PER-CATEGORY ACCURACY ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. PER-CATEGORY ACCURACY (averaged across 5 runs)")
print("=" * 70)
cat_acc = df.groupby(["model", "category"])["correct"].mean().unstack()
print(cat_acc.round(3).to_string())

# Also compute per-run per-category for CIs
print("\n  Per-category 95% CIs:")
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    for cat in sorted(mdf["category"].unique()):
        run_accs = mdf[mdf["category"] == cat].groupby("run")["correct"].mean()
        mean = run_accs.mean()
        if len(run_accs) > 1 and run_accs.std() > 0:
            se = stats.sem(run_accs)
            ci = stats.t.interval(0.95, df=len(run_accs) - 1, loc=mean, scale=se)
            print(f"    {model:<12} {cat:<20}: {mean:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")
        elif len(run_accs) > 1:
            # Zero variance across runs — deterministically consistent
            print(f"    {model:<12} {cat:<20}: {mean:.1%} [exact: zero variance across 5 runs]")
        else:
            print(f"    {model:<12} {cat:<20}: {mean:.1%} [single run]")

# ── 2. OVERALL 95% CIs (t-distribution, 5 runs) ───────────────────────────
print("\n" + "=" * 70)
print("2. OVERALL 95% CONFIDENCE INTERVALS (from 5 run-level accuracies)")
print("=" * 70)
for model in df["model"].unique():
    run_accs = df[df["model"] == model].groupby("run")["correct"].mean()
    mean = run_accs.mean()
    se = stats.sem(run_accs)
    ci = stats.t.interval(0.95, df=len(run_accs) - 1, loc=mean, scale=se)
    print(f"  {model:<12}: {mean:.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]")
print(f"  {'deterministic':<12}: 100.0%  [100.0%, 100.0%]")

# Add gpt-4o-mini from reliability data
rel_t0 = rel[rel["temperature"] == 0.0]
gpt4mini_run_accs = rel_t0.groupby("run").apply(
    lambda g: (g["llm_answer"] == g["ground_truth"]).mean()
)
mean_4m = gpt4mini_run_accs.mean()
se_4m = stats.sem(gpt4mini_run_accs)
ci_4m = stats.t.interval(0.95, df=len(gpt4mini_run_accs) - 1, loc=mean_4m, scale=se_4m)
print(f"  {'gpt-4o-mini':<12}: {mean_4m:.1%}  [{ci_4m[0]:.1%}, {ci_4m[1]:.1%}]")

# ── 3. INCONSISTENCY DETAILS — which prompts and categories? ──────────────
print("\n" + "=" * 70)
print("3. INCONSISTENT PROMPTS BY MODEL")
print("=" * 70)
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    inconsistent_idxs = []
    for idx in mdf["idx"].unique():
        valid = mdf[(mdf["idx"] == idx) & (mdf["answer"].isin(["yes", "no"]))]
        if valid["answer"].nunique() > 1:
            inconsistent_idxs.append(idx)

    if inconsistent_idxs:
        inc_df = mdf[mdf["idx"].isin(inconsistent_idxs)]
        cats = inc_df.groupby("idx")["category"].first().value_counts()
        print(f"\n  {model}: {len(inconsistent_idxs)} inconsistent prompts")
        print(f"    By category: {cats.to_dict()}")
        for idx in inconsistent_idxs[:3]:
            row = mdf[mdf["idx"] == idx].iloc[0]
            answers = mdf[mdf["idx"] == idx]["answer"].tolist()
            print(f"    idx={idx} [{row['category']}]: "
                  f"answers={answers}, gt={row['ground_truth']}")
            print(f"      prompt: {row['prompt']}")
    else:
        print(f"\n  {model}: 0 inconsistent prompts")

# ── 4. PROMPT-LEVEL: WHERE o1 BEATS vs LOSES TO gpt-4o ────────────────────
print("\n" + "=" * 70)
print("4. PROMPT-LEVEL: WHERE o1 BEATS vs LOSES TO gpt-4o")
print("=" * 70)
prompt_acc = df.pivot_table(
    index=["idx", "category"],
    columns="model",
    values="correct",
    aggfunc="mean",
).reset_index()

if "o1" in prompt_acc.columns and "gpt-4o" in prompt_acc.columns:
    o1_wins = prompt_acc[
        (prompt_acc["o1"] >= 0.8) & (prompt_acc["gpt-4o"] <= 0.2)
    ]
    print(f"\n  o1 wins (o1>=80%, gpt-4o<=20%): {len(o1_wins)} prompts")
    if len(o1_wins) > 0:
        print(f"    Categories: {o1_wins['category'].value_counts().to_dict()}")

    gpt4o_wins = prompt_acc[
        (prompt_acc["gpt-4o"] >= 0.8) & (prompt_acc["o1"] <= 0.2)
    ]
    print(f"  gpt-4o wins (gpt-4o>=80%, o1<=20%): {len(gpt4o_wins)} prompts")
    if len(gpt4o_wins) > 0:
        print(f"    Categories: {gpt4o_wins['category'].value_counts().to_dict()}")

    all_wrong = prompt_acc
    for m in ["o1", "gpt-4o", "o3-mini"]:
        if m in all_wrong.columns:
            all_wrong = all_wrong[all_wrong[m] <= 0.2]
    print(f"  ALL models wrong (all<=20%): {len(all_wrong)} prompts")
    if len(all_wrong) > 0:
        print(f"    Categories: {all_wrong['category'].value_counts().to_dict()}")

    all_right = prompt_acc
    for m in ["o1", "gpt-4o", "o3-mini"]:
        if m in all_right.columns:
            all_right = all_right[all_right[m] >= 0.8]
    print(f"  ALL models right (all>=80%): {len(all_right)} prompts")
    if len(all_right) > 0:
        print(f"    Categories: {all_right['category'].value_counts().to_dict()}")

# ── 5. COST ANALYSIS ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. COST ANALYSIS (per symbolic evaluation)")
print("=" * 70)

# USD per 1M tokens. Update when provider pricing changes.
PRICING = {
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o1": {"input": 15.00, "output": 60.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

for model in df["model"].unique():
    mdf = df[df["model"] == model]
    if model not in PRICING:
        continue
    p = PRICING[model]
    mean_input = mdf["prompt_tokens"].mean()
    mean_output = mdf["completion_tokens"].mean()
    cost_per_eval = (mean_input / 1e6 * p["input"] +
                     mean_output / 1e6 * p["output"])
    daily_cost_30k = cost_per_eval * 30_000
    print(f"  {model:<12}: ${cost_per_eval:.6f}/eval "
          f"(${daily_cost_30k:.2f}/day at 30K symbolic evals)")

# gpt-4o-mini from evaluation pipeline data
eval_df = pd.read_csv(DATA_DIR / "evaluation_with_llm.csv")
sym_eval = eval_df[eval_df["ground_truth_source"] == "deterministic"]
gpt4mini_mean_tokens = sym_eval["system_a_binary_tokens"].mean()
p_mini = PRICING["gpt-4o-mini"]
cost_mini = gpt4mini_mean_tokens / 1e6 * (p_mini["input"] + p_mini["output"]) / 2
daily_mini = cost_mini * 30_000
print(f"  {'gpt-4o-mini':<12}: ~${cost_mini:.6f}/eval "
      f"(~${daily_mini:.2f}/day at 30K symbolic evals)")
print(f"  {'deterministic':<12}: $0.000000/eval ($0.00/day)")

# Enterprise scenario
print(f"\n  Enterprise scenario: 100K evals/day, 30% symbolic (30K symbolic evals):")
for model in ["o1", "o3-mini", "gpt-4o"]:
    if model not in PRICING:
        continue
    mdf = df[df["model"] == model]
    p = PRICING[model]
    mean_input = mdf["prompt_tokens"].mean()
    mean_output = mdf["completion_tokens"].mean()
    cost = (mean_input / 1e6 * p["input"] + mean_output / 1e6 * p["output"]) * 30_000
    acc = mdf.groupby("run")["correct"].mean().mean()
    print(f"    {model:<12}: ${cost:>8.2f}/day at {acc:.1%} accuracy")
print(f"    {'deterministic':<12}: $    0.00/day at 100.0% accuracy")

# ── 6. REASONING TOKENS vs CORRECTNESS ────────────────────────────────────
print("\n" + "=" * 70)
print("6. REASONING TOKENS vs CORRECTNESS (reasoning models only)")
print("=" * 70)
for model in ["o3-mini", "o1"]:
    mdf = df[df["model"] == model]
    correct = mdf[mdf["correct"] == 1]["reasoning_tokens"]
    wrong = mdf[mdf["correct"] == 0]["reasoning_tokens"]
    print(f"\n  {model}:")
    print(f"    Correct answers:  mean={correct.mean():.0f} reasoning tokens "
          f"(median={correct.median():.0f}, n={len(correct)})")
    print(f"    Wrong answers:    mean={wrong.mean():.0f} reasoning tokens "
          f"(median={wrong.median():.0f}, n={len(wrong)})")
    if len(correct) > 1 and len(wrong) > 1:
        t, p_val = stats.ttest_ind(correct, wrong)
        print(f"    t-test: t={t:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            if correct.mean() > wrong.mean():
                print(f"    --> Models use MORE reasoning tokens when correct")
            else:
                print(f"    --> Models use MORE reasoning tokens when wrong (overthinking)")
        else:
            print(f"    --> No significant difference in reasoning effort "
                  f"between correct/wrong answers")
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(correct) - 1) * correct.std()**2 + (len(wrong) - 1) * wrong.std()**2) /
            (len(correct) + len(wrong) - 2)
        )
        if pooled_std > 0:
            d = (correct.mean() - wrong.mean()) / pooled_std
            print(f"    Cohen's d = {d:.3f}")

# ── 7. ERROR OVERLAP: Do all models fail on the same prompts? ─────────────
print("\n" + "=" * 70)
print("7. ERROR OVERLAP ACROSS MODELS")
print("=" * 70)

# Get majority-vote errors for each model
error_sets = {}
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    majority = mdf.groupby("idx")["correct"].mean()
    errors = set(majority[majority < 0.5].index)
    error_sets[model] = errors
    print(f"  {model:<12}: {len(errors)} prompts with majority-wrong")

# Add gpt-4o-mini from reliability data
rel_t0 = rel[rel["temperature"] == 0.0]
gpt4mini_majority = rel_t0.groupby("idx").apply(
    lambda g: (g["llm_answer"] == g["ground_truth"]).mean()
)
error_sets["gpt-4o-mini"] = set(gpt4mini_majority[gpt4mini_majority < 0.5].index)
print(f"  {'gpt-4o-mini':<12}: {len(error_sets['gpt-4o-mini'])} prompts with majority-wrong")

# Pairwise Jaccard similarity of error sets
print(f"\n  Pairwise Jaccard similarity of error sets:")
models_list = list(error_sets.keys())
for i in range(len(models_list)):
    for j in range(i + 1, len(models_list)):
        a, b = error_sets[models_list[i]], error_sets[models_list[j]]
        if len(a | b) > 0:
            jaccard = len(a & b) / len(a | b)
            overlap = len(a & b)
            print(f"    {models_list[i]} vs {models_list[j]}: "
                  f"Jaccard={jaccard:.2f} ({overlap} shared errors / "
                  f"{len(a | b)} total)")

# Universal errors (all models get wrong)
all_errors = set.intersection(*error_sets.values()) if error_sets else set()
print(f"\n  Universal errors (ALL models majority-wrong): {len(all_errors)} prompts")
if all_errors:
    # Show what these prompts are
    for idx in sorted(all_errors)[:5]:
        row = df[df["idx"] == idx].iloc[0]
        print(f"    idx={idx} [{row['category']}]: {row['prompt']}")
        print(f"      gt={row['ground_truth']}, "
              f"answers across all models: ", end="")
        for model in df["model"].unique():
            model_answers = df[(df["idx"] == idx) & (df["model"] == model)]["answer"].tolist()
            print(f"{model}={model_answers}", end=" ")
        print()

# Only-one-model-gets-right
for model in df["model"].unique():
    unique_correct = set()
    mdf = df[df["model"] == model]
    majority = mdf.groupby("idx")["correct"].mean()
    correct_set = set(majority[majority >= 0.5].index)
    other_errors = set.intersection(
        *[e for m, e in error_sets.items() if m != model]
    ) if len(error_sets) > 1 else set()
    unique_correct = correct_set & other_errors
    if unique_correct:
        print(f"\n  {model} uniquely solves {len(unique_correct)} prompts "
              f"that ALL other models get wrong")

# ── SAVE ANALYSIS SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DONE. Key numbers for thesis:")
print("=" * 70)
for model in df["model"].unique():
    run_accs = df[df["model"] == model].groupby("run")["correct"].mean()
    mean = run_accs.mean()
    se = stats.sem(run_accs)
    ci = stats.t.interval(0.95, df=len(run_accs) - 1, loc=mean, scale=se)
    inc = df[df["model"] == model].groupby("idx")["answer"].apply(
        lambda x: x.nunique() > 1).sum()
    n_prompts = df[df["model"] == model]["idx"].nunique()
    print(f"  {model:<12}: {mean:.1%} [{ci[0]:.1%}, {ci[1]:.1%}], "
          f"inconsistent={inc}/{n_prompts} ({inc/n_prompts:.0%})")
