"""Prompt-length-adjusted overthinking test.

The raw observation is that reasoning models (o1, o3-mini) spend more reasoning
tokens on prompts they answer wrong. This script controls for prompt length:
regresses reasoning tokens on prompt length, then tests whether the residual
correct/wrong difference is still significant. Produces the values reported in
the appendix's `tab:app-overthinking`.

Run: python -m scripts.analyses.overthinking_confound
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "overthinking"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_DIR / "reasoning" / "raw.csv")
    rows = []

    for model in ["o1", "o3-mini"]:
        m = df[df["model"] == model].copy()
        m["prompt_len"] = m["prompt"].str.len()

        correct = m[m["correct"] == 1]["reasoning_tokens"]
        wrong = m[m["correct"] == 0]["reasoning_tokens"]
        t_raw, p_raw = stats.ttest_ind(correct, wrong)

        slope, intercept = np.polyfit(m["prompt_len"], m["reasoning_tokens"], 1)
        m["resid"] = m["reasoning_tokens"] - (slope * m["prompt_len"] + intercept)
        correct_resid = m[m["correct"] == 1]["resid"]
        wrong_resid = m[m["correct"] == 0]["resid"]
        t_adj, p_adj = stats.ttest_ind(correct_resid, wrong_resid)

        pooled_std = np.sqrt(
            ((len(correct_resid) - 1) * correct_resid.std() ** 2 +
             (len(wrong_resid) - 1) * wrong_resid.std() ** 2)
            / (len(correct_resid) + len(wrong_resid) - 2)
        )
        d_adj = (correct_resid.mean() - wrong_resid.mean()) / pooled_std if pooled_std > 0 else 0

        rows.append({
            "model": model,
            "raw_t": round(t_raw, 2), "raw_p": round(p_raw, 4),
            "adjusted_t": round(t_adj, 2), "adjusted_p": round(p_adj, 4),
            "adjusted_d": round(d_adj, 3),
            "survives": p_adj < 0.05,
        })

    result = pd.DataFrame(rows)
    out_path = OUT_DIR / "overthinking_confound.csv"
    result.to_csv(out_path, index=False)

    print("Prompt-length-adjusted overthinking test:")
    for _, r in result.iterrows():
        status = "SURVIVES" if r["survives"] else "does not survive"
        print(f"  {r['model']}: raw t={r['raw_t']}, adjusted t={r['adjusted_t']}, "
              f"d={r['adjusted_d']}, p={r['adjusted_p']} → {status}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
