"""Compute LLM-baseline accuracy on the 275 answerable prompts (excludes UNSUPPORTED).

Reproduces the thesis claim at Section 4.2.7:
    "On answerable prompts alone (275 of 305), accuracy drops from 72.0% to 43.3%"

Reads per-row LLM outcomes from results/e2e/baselines_rescored.csv 
and aggregates binary- and reject-capable accuracy on the
275 prompts whose category is not UNSUPPORTED.

Run: python -m scripts.analyses.answerable_only
"""

import csv
from collections import defaultdict

from config import RESULTS_ANALYSES, RESULTS_E2E

INPUT = RESULTS_E2E / "baselines_rescored.csv"
OUTPUT = RESULTS_ANALYSES / "answerable_only.csv"


def main() -> None:
    rows = list(csv.DictReader(open(INPUT)))
    by_cat_total: dict[str, int] = defaultdict(int)
    by_cat_bin: dict[str, int] = defaultdict(int)
    by_cat_rej: dict[str, int] = defaultdict(int)

    for r in rows:
        c = r["category"]
        by_cat_total[c] += 1
        if r["mini_bin_correct"] == "1":
            by_cat_bin[c] += 1
        if r["mini_rej_correct"] == "1":
            by_cat_rej[c] += 1

    answerable_cats = [c for c in by_cat_total if c != "UNSUPPORTED"]
    n = sum(by_cat_total[c] for c in answerable_cats)
    bin_correct = sum(by_cat_bin[c] for c in answerable_cats)
    rej_correct = sum(by_cat_rej[c] for c in answerable_cats)

    bin_acc = bin_correct / n
    rej_acc = rej_correct / n

    RESULTS_ANALYSES.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "answerable_accuracy", "n_correct", "n"])
        w.writerow(["LLM-only (binary)", f"{bin_acc:.4f}", bin_correct, n])
        w.writerow(["LLM-only (reject)", f"{rej_acc:.4f}", rej_correct, n])

    print("Answerable-only accuracy (N=275, UNSUPPORTED excluded):")
    print(f"  LLM-only (binary): {bin_correct}/{n} = {bin_acc:.1%}")
    print(f"  LLM-only (reject): {rej_correct}/{n} = {rej_acc:.1%}")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
