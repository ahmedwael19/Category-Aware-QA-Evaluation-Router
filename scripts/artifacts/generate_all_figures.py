"""Generate ALL thesis figures from CSV results.

Reads from results_*.csv files — no notebook dependency.

Run: python -m scripts.artifacts.generate_all_figures
"""


from config import RESULTS_DIR, RESULTS_ANALYSES, FIGURES_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

if __name__ != "__main__":
    raise RuntimeError("This script must be invoked as a main program, e.g. python -m scripts.artifacts.generate_all_figures")


# ── Thesis style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 10.5,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
})

# Palette
C_RED = "#c0392b"
C_ORANGE = "#e67e22"
C_GREEN = "#27ae60"
C_BLUE = "#2980b9"
C_PURPLE = "#8e44ad"
C_TEAL = "#16a085"
C_GREY = "#7f8c8d"
C_DARK = "#2c3e50"


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=600)
    fig.savefig(FIGURES_DIR / name.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  {name}")


print("Generating all thesis figures...\n")

# ── Part A: router training figures ──────────────────────────────────────────
print("Router training figures:")

# ── A1: Difficulty Ladder ─────────────────────────────────────────────────────
bench = pd.read_csv(RESULTS_DIR / "router" / "benchmark.csv")

fig, ax = plt.subplots(figsize=(10, 4.5))

# Select the 6 ladder models in order
_ladder_names = ["Keyword/Regex", "TF-IDF+LR (bi, <NUM>)", "TF-IDF+SVM",
                 "Emb+LR", "Emb+MLP", "Ensemble (TF-IDF+Emb)"]
_short = ["Keyword", "TF-IDF+LR", "TF-IDF+SVM", "Emb+LR", "Emb+MLP", "Ensemble"]
_ladder = bench[bench["model"].isin(_ladder_names)].copy()
_ladder["model"] = pd.Categorical(_ladder["model"], categories=_ladder_names, ordered=True)
_ladder = _ladder.sort_values("model")

_colors = [C_RED] + [C_BLUE, C_GREEN, C_PURPLE, C_TEAL, C_DARK]
bars = ax.bar(_short, _ladder["test_f1"], color=_colors, alpha=0.88,
              edgecolor="white", linewidth=0.8, width=0.65)

ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.axhline(0.8, color=C_RED, linestyle="--", linewidth=0.8, alpha=0.5, label="80% threshold")

for bar, val in zip(bars, _ladder["test_f1"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.1%}",
            ha="center", fontsize=9, fontweight="bold")

ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc")
fig.suptitle("Classifier Difficulty Ladder (In-Distribution, N=634)", fontsize=12, y=1.01)
save(fig, "difficulty_ladder.png")

# ── A2: Generalization Gap ────────────────────────────────────────────────────
gap = pd.read_csv(RESULTS_DIR / "router" / "generalization_gap.csv")
gap = gap.dropna(subset=["holdout_f1"])

fig, ax = plt.subplots(figsize=(10, 4.5))

_order = gap.sort_values("gap", ascending=True)["model"].tolist()
_colors_gap = []
for m in _order:
    if "Keyword" in m:
        _colors_gap.append(C_RED)
    elif "Emb" in m or "Ensemble" in m:
        _colors_gap.append(C_PURPLE)
    elif "NB" in m or "NaiveBayes" in m:
        _colors_gap.append(C_GREY)
    else:
        _colors_gap.append(C_BLUE)

_gap_sorted = gap.set_index("model").loc[_order]

_y = np.arange(len(_order))
_short_names = [m.replace("TF-IDF+LR (bi, <NUM>)", "TF-IDF+LR (bi)")
                 .replace("TF-IDF+LR (uni, <NUM>)", "TF-IDF+LR (uni)")
                 .replace("TF-IDF+LR (bi, raw)", "TF-IDF+LR (raw)")
                 .replace("Ensemble (TF-IDF+Emb)", "Ensemble")
                for m in _order]

ax.barh(_y, _gap_sorted["test_f1"], height=0.4, color=_colors_gap, alpha=0.7, label="In-Distribution F1")
ax.barh(_y + 0.4, _gap_sorted["holdout_f1"], height=0.4, color=_colors_gap, alpha=0.35, label="Holdout F1")

ax.set_yticks(_y + 0.2)
ax.set_yticklabels(_short_names, fontsize=9)
ax.set_xlabel("Macro-F1")
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc")
fig.suptitle("Generalization Gap: In-Distribution vs Cross-Generator Holdout", fontsize=12, y=1.01)
save(fig, "generalization_gap.png")

# ── A3: Template-Family-Out Heatmap ──────────────────────────────────────────
tfo = pd.read_csv(RESULTS_DIR / "router" / "template_family_out.csv")

_model_cols = [c for c in tfo.columns if c not in ("category", "held_out", "n_test")]
_labels = [f"{r['category'].split('_')[-1]}\n({r['held_out']})" for _, r in tfo.iterrows()]

fig, ax = plt.subplots(figsize=(10, 4))
_data = tfo[_model_cols].values
_col_names = [c.replace("tfidf_", "TF-").replace("emb_", "Emb-").upper() for c in _model_cols]

im = ax.imshow(_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(_col_names)))
ax.set_xticklabels(_col_names, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(len(_labels)))
ax.set_yticklabels(_labels, fontsize=8.5)

for _i in range(len(_labels)):
    for _j in range(len(_col_names)):
        _v = _data[_i, _j]
        _c = "white" if _v < 0.5 else "black"
        ax.text(_j, _i, f"{_v:.0%}", ha="center", va="center", fontsize=8, color=_c)

fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
fig.suptitle("Leave-Template-Family-Out Accuracy", fontsize=12, y=1.01)
save(fig, "template_family_heatmap.png")

# ── Part B: end-to-end figures ────────────────────────────────────────────────
print("\nEnd-to-end figures:")

summary = pd.read_csv(RESULTS_DIR / "e2e" / "summary.csv")
per_cat = pd.read_csv(RESULTS_DIR / "e2e" / "per_category.csv")
rel = pd.read_csv(RESULTS_DIR / "e2e" / "reliability_summary.csv")

CAT_ORDER = ["SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA", "SEMANTIC", "HYBRID", "UNSUPPORTED"]
CAT_SHORT = {"SYMBOLIC_TIME": "Time", "SYMBOLIC_COUNT": "Count",
             "SYMBOLIC_METADATA": "Metadata", "SEMANTIC": "Semantic",
             "HYBRID": "Hybrid", "UNSUPPORTED": "Unsupported"}

per_cat["category"] = pd.Categorical(per_cat["category"], categories=CAT_ORDER, ordered=True)
per_cat = per_cat.sort_values("category").reset_index(drop=True)

# ── B1: Per-Category Accuracy (3 key systems) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.2))

x = np.arange(len(per_cat))
w = 0.24

bars1 = ax.bar(x - w, per_cat["LLM-only (binary)"], w, label="LLM-only (yes/no)",
               color=C_RED, alpha=0.88, edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x, per_cat["LLM-only (reject)"], w, label="LLM-only (yes/no/reject)",
               color=C_ORANGE, alpha=0.88, edgecolor="white", linewidth=0.8)
bars3 = ax.bar(x + w, per_cat["Router (TF-IDF+SVM)"], w, label="Router (TF-IDF+SVM)",
               color=C_GREEN, alpha=0.88, edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels([CAT_SHORT[c] for c in per_cat["category"]])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(frameon=True, fancybox=False, edgecolor="#ccc", fontsize=9, loc="upper right")

for bar in bars3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.0%}",
            ha="center", fontsize=8, fontweight="bold", color=C_GREEN)

ax.axvline(x=2.5, color="#bdc3c7", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(1.0, 1.08, "Symbolic", ha="center", fontsize=9, fontstyle="italic", color="#666")
ax.text(4.0, 1.08, "Semantic / Other", ha="center", fontsize=9, fontstyle="italic", color="#666")

fig.suptitle("Per-Category End-to-End Evaluation Accuracy (N=305)", fontsize=12, y=1.01)
save(fig, "e2e_per_category.png")

# ── B2: All-Models Summary (3 panels) ────────────────────────────────────────
_order = [
    "LLM-only (binary)", "LLM-only (with reject)", "Router (Keyword)",
    "Router (Emb+LR)", "Router (Ensemble)", "Router (TF-IDF+LR)", "Router (TF-IDF+SVM)",
]
_palette = {
    "LLM-only (binary)": C_RED, "LLM-only (with reject)": C_ORANGE,
    "Router (Keyword)": C_GREY, "Router (Emb+LR)": C_PURPLE,
    "Router (Ensemble)": C_TEAL, "Router (TF-IDF+LR)": C_BLUE,
    "Router (TF-IDF+SVM)": C_GREEN,
}

_s = summary[summary["system"].isin(_order)].copy()
_s["system"] = pd.Categorical(_s["system"], categories=_order, ordered=True)
_s = _s.sort_values("system")
_labels = _s["system"].tolist()
_cols = [_palette[s] for s in _labels]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

ax = axes[0]
bars = ax.barh(_labels, _s["accuracy"], color=_cols, alpha=0.9, height=0.62, edgecolor="white", linewidth=0.6)
ax.set_xlim(0, 1.08)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Accuracy")
ax.set_title("(a) Overall Accuracy")
for bar, val in zip(bars, _s["accuracy"]):
    ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=8.5, fontweight="bold")

ax = axes[1]
bars = ax.barh(_labels, _s["total_tokens"] / 1000, color=_cols, alpha=0.9, height=0.62, edgecolor="white", linewidth=0.6)
ax.set_xlabel("Tokens (thousands)")
ax.set_title("(b) Token Consumption")
for bar, val in zip(bars, _s["total_tokens"]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val / 1000:.0f}K", va="center", fontsize=8.5)

ax = axes[2]
bars = ax.barh(_labels, _s["mean_latency_ms"], color=_cols, alpha=0.9, height=0.62, edgecolor="white", linewidth=0.6)
ax.set_xlabel("Mean Latency (ms)")
ax.set_title("(c) Mean Latency")
for bar, val in zip(bars, _s["mean_latency_ms"]):
    ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=8.5)

fig.suptitle("End-to-End System Comparison (N=305)", fontsize=12, y=1.02)
plt.tight_layout(w_pad=1.5)
save(fig, "e2e_summary_all_models.png")

# ── B3: Reliability ───────────────────────────────────────────────────────────
rel_rows = []
for temp in sorted(rel["temperature"].unique()):
    sub = rel[rel["temperature"] == temp]
    rel_rows.append({
        "system": f"GPT-4o-mini\n(temp={temp})",
        "inconsistent": (sub["unique_answers"] > 1).mean(),
        "accuracy": (sub["correct_runs"] / sub["total_runs"]).mean(),
    })
rel_rows.append({"system": "Deterministic\nFunctions", "inconsistent": 0.0, "accuracy": 1.0})
rel_plot = pd.DataFrame(rel_rows)

fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(rel_plot))
w = 0.32
bars1 = ax.bar(x - w / 2, rel_plot["accuracy"], w, label="Accuracy", color=C_BLUE, alpha=0.88, edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + w / 2, rel_plot["inconsistent"], w, label="Inconsistent Across 5 Runs", color=C_RED, alpha=0.88, edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(rel_plot["system"], fontsize=10)
ax.set_ylim(0, 1.14)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("Rate")
ax.legend(frameon=True, fancybox=False, edgecolor="#ccc", fontsize=9.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.0%}", ha="center", fontsize=9, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, max(h, 0.01) + 0.02,
            f"{h:.0%}", ha="center", fontsize=9, color=C_RED)

fig.suptitle("Evaluation Reliability on Symbolic Prompts (5 Repeated Runs, N=50)", fontsize=11, y=1.01)
save(fig, "e2e_reliability.png")

# ── B4: Calibration Fallback Effect ───────────────────────────────────────────
# Same model WITH fallback (τ=0.85) vs WITHOUT fallback (τ=0). Both pulled from
# the committed τ sweep so the figure tracks whichever run produced it.

_tau_sweep = pd.read_csv(RESULTS_DIR / "e2e" / "tau_sweep_rescored.csv")
_tau0 = _tau_sweep[_tau_sweep["tau"] == 0.0].iloc[0]
_no_fallback = {
    "TF-IDF+LR": float(_tau0["tfidf_lr_acc"]),
    "Emb+LR": float(_tau0["emb_lr_acc"]),
    "Ensemble": float(_tau0["ens_acc"]),
}
_with_fallback = {}
for _name, _sys in [("TF-IDF+LR", "Router (TF-IDF+LR)"), ("Emb+LR", "Router (Emb+LR)"), ("Ensemble", "Router (Ensemble)")]:
    _row = summary[summary["system"] == _sys]
    if len(_row) > 0:
        _with_fallback[_name] = _row.iloc[0]["accuracy"]

_models = list(_no_fallback.keys())

fig, ax = plt.subplots(figsize=(8, 4.2))

x = np.arange(len(_models))
w = 0.32

bars1 = ax.bar(x - w / 2, [_no_fallback[m] for m in _models], w,
               label="Without fallback (τ=0)", color=C_BLUE, alpha=0.88, edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + w / 2, [_with_fallback.get(m, 0) for m in _models], w,
               label="With fallback (τ=0.85)", color=C_ORANGE, alpha=0.88, edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(_models, fontsize=10)
ax.set_ylabel("End-to-End Accuracy")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(frameon=True, fancybox=False, edgecolor="#ccc", fontsize=9.5)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.1%}",
            ha="center", fontsize=9, fontweight="bold", color=C_BLUE)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.1%}",
            ha="center", fontsize=9, fontweight="bold", color=C_ORANGE)


fig.suptitle("Effect of Confidence-Based Fallback on Calibrated Router Accuracy", fontsize=11, y=1.01)
save(fig, "e2e_calibration_fallback.png")
plt.tight_layout()

# ── Part C: additional experiment figures ────────────────────────────────────

_fig_count = 7

# ── C1: Reasoning Model Comparison ──────────────────────────────────────────
_reasoning_path = RESULTS_DIR / "reasoning" / "summary.csv"
if _reasoning_path.exists():
    print("\nReasoning model figures:")
    rmod = pd.read_csv(_reasoning_path)

    # Order: deterministic, o1, o3-mini, gpt-4o, gpt-4o-mini
    _rorder = ["deterministic", "o1", "o3-mini", "gpt-4o", "gpt-4o-mini (temp=0)"]
    _rlabels = ["Deterministic\nFunctions", "o1\n(reasoning)", "o3-mini\n(reasoning)",
                "GPT-4o", "GPT-4o-mini"]
    _rcolors = [C_GREEN, C_PURPLE, C_TEAL, C_BLUE, C_RED]

    rmod_ordered = rmod.set_index("model").reindex(_rorder).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel (a): Accuracy
    ax = axes[0]
    bars = ax.bar(_rlabels, rmod_ordered["mean_accuracy"], color=_rcolors,
                  alpha=0.88, edgecolor="white", linewidth=0.8, width=0.6)
    # Error bars for std
    ax.errorbar(range(len(_rlabels)), rmod_ordered["mean_accuracy"],
                yerr=rmod_ordered["std_accuracy"], fmt="none", ecolor="black",
                capsize=4, linewidth=1.2)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Accuracy (5 runs)")
    ax.set_title("(a) Accuracy on Symbolic Prompts")
    for bar, val in zip(bars, rmod_ordered["mean_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.1%}",
                ha="center", fontsize=9, fontweight="bold")

    # Panel (b): Inconsistency
    ax = axes[1]
    bars = ax.bar(_rlabels, rmod_ordered["inconsistent_pct"], color=_rcolors,
                  alpha=0.88, edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_ylim(0, 0.28)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Inconsistent Across 5 Runs")
    ax.set_title("(b) Output Inconsistency")
    for bar, val in zip(bars, rmod_ordered["inconsistent_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0.003) + 0.01,
                f"{val:.0%}", ha="center", fontsize=9, fontweight="bold")

    # Panel (c): Token cost (skip deterministic=0 and gpt-4o-mini=0)
    ax = axes[2]
    bars = ax.bar(_rlabels, rmod_ordered["mean_total_tokens"], color=_rcolors,
                  alpha=0.88, edgecolor="white", linewidth=0.8, width=0.6)
    # Overlay reasoning tokens in darker shade
    _reason_vals = rmod_ordered["mean_reasoning_tokens"].tolist()
    ax.bar(_rlabels, _reason_vals, color=_rcolors, alpha=0.4,
           edgecolor="none", width=0.6, label="Reasoning tokens")
    ax.set_ylabel("Mean Tokens / Prompt")
    ax.set_title("(c) Token Consumption")
    for bar, val in zip(bars, rmod_ordered["mean_total_tokens"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 10,
                    f"{val:.0f}", ha="center", fontsize=9)
    ax.text(0, -0.03 * ax.get_ylim()[1], "0", ha="center", fontsize=8, color=C_GREEN)

    fig.suptitle("Reasoning Model Comparison on Symbolic Prompts (N=50, 5 Repeated Runs)",
                 fontsize=12, y=1.02)
    plt.tight_layout(w_pad=1.5)
    save(fig, "reasoning_model_comparison.png")
    _fig_count += 1
else:
    print("\nSkipping reasoning model figure (reasoning/summary.csv not found)")

# ── C1b: Reasoning Tokens vs Correctness ("Overthinking") ────────────────
_raw_reasoning_path = RESULTS_DIR / "reasoning" / "raw.csv"
if _raw_reasoning_path.exists():
    rraw = pd.read_csv(_raw_reasoning_path)
    reasoning_models = rraw[rraw["model"].isin(["o3-mini", "o1"])]

    if len(reasoning_models) > 0 and reasoning_models["reasoning_tokens"].sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

        for ax_idx, model_name in enumerate(["o3-mini", "o1"]):
            ax = axes[ax_idx]
            mdf = reasoning_models[reasoning_models["model"] == model_name]

            correct = mdf[mdf["correct"] == 1]["reasoning_tokens"]
            wrong = mdf[mdf["correct"] == 0]["reasoning_tokens"]

            bp = ax.boxplot(
                [correct, wrong],
                tick_labels=["Correct", "Wrong"],
                widths=0.5,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
            )
            bp["boxes"][0].set_facecolor(C_GREEN)
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor(C_RED)
            bp["boxes"][1].set_alpha(0.6)

            ax.set_ylabel("Reasoning Tokens" if ax_idx == 0 else "")
            ax.set_title(f"{model_name}")

            # Annotate means
            for i, (data, label) in enumerate([(correct, "Correct"), (wrong, "Wrong")]):
                ax.text(i + 1, data.mean(), f"$\\mu$={data.mean():.0f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=C_GREEN if i == 0 else C_RED)

            # Significance annotation
            from scipy import stats as _stats
            if len(correct) > 1 and len(wrong) > 1:
                _, p_val = _stats.ttest_ind(correct, wrong)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                y_max = max(correct.max(), wrong.max())
                ax.plot([1, 1, 2, 2], [y_max * 1.02, y_max * 1.06, y_max * 1.06, y_max * 1.02],
                        color="black", linewidth=0.8)
                ax.text(1.5, y_max * 1.07, f"p={p_val:.4f} {sig}",
                        ha="center", fontsize=8.5)

        fig.suptitle("Reasoning Tokens: Correct vs Wrong Answers on Symbolic Prompts",
                     fontsize=12, y=1.02)
        plt.tight_layout(w_pad=2)
        save(fig, "reasoning_overthinking.png")
        _fig_count += 1
else:
    print("\nSkipping overthinking figure (reasoning/raw.csv not found)")

# ── C2: Tau Sensitivity ────────────────────────────────────────────────────
_tau_path = RESULTS_ANALYSES / "tau_sensitivity.csv"
if _tau_path.exists():
    print("\nTau sensitivity figure:")
    tau_df = pd.read_csv(_tau_path)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    _tau_models = {
        "TF-IDF+LR": (C_BLUE, "-", "o"),
        "Emb+LR": (C_PURPLE, "--", "s"),
        "Ensemble": (C_TEAL, "-.", "D"),
    }

    for model_name, (color, ls, marker) in _tau_models.items():
        model_data = tau_df[tau_df["model"] == model_name]
        if len(model_data) > 0:
            ax.plot(model_data["tau"], model_data["accuracy"], color=color,
                    linestyle=ls, marker=marker, markersize=5, linewidth=1.8,
                    label=model_name, alpha=0.9)

    # Mark the chosen tau
    ax.axvline(x=0.85, color=C_GREY, linestyle=":", linewidth=1.2, alpha=0.7,
               label="Selected $\\tau$ = 0.85")
    # Mark TF-IDF+SVM (no fallback) as reference
    _svm_row = tau_df[(tau_df["model"] == "TF-IDF+SVM") & (tau_df["tau"] == 0.0)]
    if len(_svm_row) > 0:
        _svm_acc = _svm_row.iloc[0]["accuracy"]
        ax.axhline(y=_svm_acc, color=C_GREEN, linestyle="--", linewidth=1.0,
                   alpha=0.6, label=f"TF-IDF+SVM (no fallback): {_svm_acc:.1%}")

    ax.set_xlabel("Confidence Threshold ($\\tau$)")
    ax.set_ylabel("End-to-End Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.45, 1.0)
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc", loc="best")
    fig.suptitle("Effect of Confidence Threshold on End-to-End Accuracy (N=305)",
                 fontsize=12, y=1.01)
    save(fig, "tau_sensitivity.png")
    _fig_count += 1
else:
    print("\nSkipping tau sensitivity figure (tau_sensitivity.csv not found)")

print(f"\nAll figures generated. Total: {_fig_count} figures in {FIGURES_DIR}")
