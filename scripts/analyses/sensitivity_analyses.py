"""Sensitivity analyses:

1. Cosine dedup threshold (0.95, 0.97, 0.99) — retrain TF-IDF+SVM, report test Macro-F1
2. Noise-mix sensitivity — retrain on clean-only prompts (drop noisy), report test Macro-F1

Does NOT regenerate the dataset. Uses the existing train split and:
- For cosine: simulates a stricter/looser threshold by removing additional pairs
  above the target threshold from the train set only (per-category, LLM-generated only),
  then retrains. Test set is unchanged.
- For noise: filters train to noise=='none' only, retrains. Test set unchanged.
"""

import json
import re


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ROOT,
    DATASET_TRAIN, DATASET_TEST,
    FEATURE_COL, TARGET_COL, SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
)

if __name__ != "__main__":
    raise RuntimeError("Run as: python -m scripts.analyses.sensitivity_analyses")

NUM = re.compile(r"\d+")
def mask(t): return NUM.sub("<NUM>", t)


def tfidf_fit(prompts_train, prompts_test):
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True)
    Xtr = tfidf.fit_transform([mask(p) for p in prompts_train])
    Xte = tfidf.transform([mask(p) for p in prompts_test])
    return Xtr, Xte

def svm_f1(Xtr, ytr, Xte, yte):
    m = LinearSVC(class_weight="balanced", max_iter=2000, random_state=SEED, C=1.0)
    m.fit(Xtr, ytr)
    return f1_score(yte, m.predict(Xte), average="macro")

def lr_f1(Xtr, ytr, Xte, yte):
    m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED, C=1.0)
    m.fit(Xtr, ytr)
    return f1_score(yte, m.predict(Xte), average="macro")


print("=" * 80)
print("SENSITIVITY ANALYSES")
print("=" * 80)

dtr = pd.read_csv(DATASET_TRAIN)
dte = pd.read_csv(DATASET_TEST)
print(f"train={len(dtr)}  test={len(dte)}")

yte = dte[TARGET_COL].values

# ── Baseline (current dataset) ──────────────────────────────────────────────
Xtr, Xte = tfidf_fit(dtr[FEATURE_COL], dte[FEATURE_COL])
baseline_svm = svm_f1(Xtr, dtr[TARGET_COL].values, Xte, yte)
baseline_lr = lr_f1(Xtr, dtr[TARGET_COL].values, Xte, yte)
print(f"\nBaseline (current dataset, n={len(dtr)}): SVM F1 = {baseline_svm:.4f}  LR F1 = {baseline_lr:.4f}")

results = {"baseline": {"n_train": len(dtr), "svm_f1": baseline_svm, "lr_f1": baseline_lr}}

# ── Cosine sensitivity ──────────────────────────────────────────────────────
print("\n─" * 40)
print("COSINE DEDUP SENSITIVITY")
print("─" * 40)

# Simulate stricter dedup (0.95, 0.97) on the LLM-generated subset of the
# committed train split, which was built at 0.99.
from sentence_transformers import SentenceTransformer
print("Loading multilingual-e5-large-instruct...")
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

llm_mask = dtr["subcategory"].astype(str).str.contains("llm", case=False, na=False)
llm_df = dtr[llm_mask].copy()
non_llm_df = dtr[~llm_mask].copy()

print(f"LLM-generated in train: {len(llm_df)}")


def simulate_stricter_threshold(llm_df, threshold):
    """Remove additional pairs above `threshold` per-category, keep first of each pair."""
    keep = np.ones(len(llm_df), dtype=bool)
    for cat in llm_df["category"].unique():
        idx = np.where(llm_df["category"].values == cat)[0]
        if len(idx) < 2:
            continue
        prompts = llm_df.iloc[idx][FEATURE_COL].tolist()
        embs = model.encode(prompts, show_progress_bar=False, batch_size=64)
        sim = cosine_similarity(embs)
        for i in range(len(idx)):
            if not keep[idx[i]]:
                continue
            for j in range(i + 1, len(idx)):
                if not keep[idx[j]]:
                    continue
                if sim[i, j] > threshold:
                    keep[idx[j]] = False
    return llm_df[keep].copy()

for thr in [0.95, 0.97]:
    print(f"\n  Simulating dedup at cosine > {thr} on LLM-generated train prompts...")
    llm_kept = simulate_stricter_threshold(llm_df, thr)
    removed = len(llm_df) - len(llm_kept)
    new_train = pd.concat([non_llm_df, llm_kept], ignore_index=True)
    Xtr2, Xte2 = tfidf_fit(new_train[FEATURE_COL], dte[FEATURE_COL])
    svm2 = svm_f1(Xtr2, new_train[TARGET_COL].values, Xte2, yte)
    lr2 = lr_f1(Xtr2, new_train[TARGET_COL].values, Xte2, yte)
    print(f"    removed {removed} from train (train now {len(new_train)}); SVM F1 = {svm2:.4f}  LR F1 = {lr2:.4f}")
    results[f"cosine_{thr}"] = {
        "threshold": thr,
        "n_train": int(len(new_train)),
        "removed_from_llm": int(removed),
        "svm_f1": float(svm2),
        "lr_f1": float(lr2),
        "svm_delta": float(svm2 - baseline_svm),
        "lr_delta": float(lr2 - baseline_lr),
    }

# ── Noise-mix sensitivity ──────────────────────────────────────────────────
print("\n─" * 40)
print("NOISE-MIX SENSITIVITY")
print("─" * 40)

if "noise" in dtr.columns:
    # Clean-only: drop any prompt with noise != 'none'
    clean_train = dtr[dtr["noise"].astype(str).str.lower() == "none"].copy()
    print(f"\nClean-only (noise=='none') train: n={len(clean_train)} (was {len(dtr)})")
    Xtr_c, Xte_c = tfidf_fit(clean_train[FEATURE_COL], dte[FEATURE_COL])
    svm_c = svm_f1(Xtr_c, clean_train[TARGET_COL].values, Xte_c, yte)
    lr_c = lr_f1(Xtr_c, clean_train[TARGET_COL].values, Xte_c, yte)
    print(f"  clean-only: SVM F1 = {svm_c:.4f}  LR F1 = {lr_c:.4f}")
    results["clean_only"] = {
        "n_train": int(len(clean_train)),
        "svm_f1": float(svm_c),
        "lr_f1": float(lr_c),
        "svm_delta": float(svm_c - baseline_svm),
        "lr_delta": float(lr_c - baseline_lr),
    }

    # Noisy-only (no clean): train on everything except clean
    noisy_train = dtr[dtr["noise"].astype(str).str.lower() != "none"].copy()
    print(f"\nNoisy-only (noise != 'none') train: n={len(noisy_train)}")
    Xtr_n, Xte_n = tfidf_fit(noisy_train[FEATURE_COL], dte[FEATURE_COL])
    svm_n = svm_f1(Xtr_n, noisy_train[TARGET_COL].values, Xte_n, yte)
    lr_n = lr_f1(Xtr_n, noisy_train[TARGET_COL].values, Xte_n, yte)
    print(f"  noisy-only: SVM F1 = {svm_n:.4f}  LR F1 = {lr_n:.4f}")
    results["noisy_only"] = {
        "n_train": int(len(noisy_train)),
        "svm_f1": float(svm_n),
        "lr_f1": float(lr_n),
        "svm_delta": float(svm_n - baseline_svm),
        "lr_delta": float(lr_n - baseline_lr),
    }
else:
    print("No 'noise' column in train CSV, skipping noise-mix sensitivity")

# ── Save ────────────────────────────────────────────────────────────────────
out = ROOT / "results" / "router" / "sensitivity.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("SENSITIVITY SUMMARY")
print("=" * 80)
for k, v in results.items():
    if k == "baseline":
        continue
    print(f"  {k:<16}  n_train={v['n_train']:>5}  SVM Δ={v.get('svm_delta', 0):+.4f}  LR Δ={v.get('lr_delta', 0):+.4f}")

print(f"\nSaved: {out}")
