"""Post-hoc validation of the synthetic benchmark dataset.

This script runs on the EXISTING  CSV files and produces validation
evidence for the thesis without modifying the data. It addresses:

  1. Cross-category embedding separability (are categories distinct?)
  2. Near-duplicate audit (how many near-dups survived the 0.99 threshold?)
  3. Keyword leakage re-check (aligned with generation forbidden words)
  4. Class imbalance documentation
  5. Style label audit (how many were affected by post-noise detection?)

Run: python -m scripts.validation.validate_synthetic_data
"""

import json
import re
from collections import Counter

from config import DATA_DIR, RESULTS_VALIDATION, MODELS_DIR

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

if __name__ != "__main__":
    raise RuntimeError("This script must be invoked as a main program, e.g. python -m scripts.validation.validate_synthetic_data")


# ── Find the canonical dataset ────────────────────────────────────────────────
full_files = sorted(DATA_DIR.glob("synthetic_final_*_full.csv"))
if not full_files:
    raise FileNotFoundError("No synthetic_final_*_full.csv found")

FULL_CSV = full_files[-1]  # Most recent
print(f"Validating: {FULL_CSV.name}")

df = pd.read_csv(FULL_CSV)
print(f"Total prompts: {len(df)}")
print(f"Categories: {sorted(df['top_category'].unique())}")
print()


# ── 1. CLASS IMBALANCE DOCUMENTATION ──────────────────────────────────────
print("=" * 70)
print("1. CLASS DISTRIBUTION")
print("=" * 70)

counts = df["top_category"].value_counts().sort_index()
total = len(df)
print(f"\n{'Category':<24} {'N':>5} {'%':>7} {'Ratio to min':>12}")
min_count = counts.min()
for cat, n in counts.items():
    print(f"  {cat:<22} {n:>5} {n/total*100:>6.1f}% {n/min_count:>11.1f}x")

imbalance_ratio = counts.max() / counts.min()
print(f"\nMax/min imbalance ratio: {imbalance_ratio:.1f}x")
print(f"Largest: {counts.idxmax()} ({counts.max()})")
print(f"Smallest: {counts.idxmin()} ({counts.min()})")

# Subcategory distribution
if "sub_category" in df.columns:
    print(f"\nSub-category distribution:")
    sub_counts = df["sub_category"].value_counts().sort_index()
    for cat, n in sub_counts.items():
        print(f"  {cat:<30} {n:>5}")


# ── 2. NEAR-DUPLICATE AUDIT ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("2. NEAR-DUPLICATE AUDIT")
print("=" * 70)

# Load embeddings if they exist, otherwise use TF-IDF as proxy
emb_file = MODELS_DIR / "embeddings" / "full.npy"
if emb_file.exists():
    print("Using pre-computed embeddings")
    embeddings = np.load(emb_file)
else:
    print("No embeddings found — using TF-IDF cosine similarity as proxy")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    embeddings = tfidf.fit_transform(df["prompt"]).toarray()

# Per-category near-duplicate check
THRESHOLDS = [0.95, 0.97, 0.99]
print(f"\n{'Category':<24}", end="")
for t in THRESHOLDS:
    print(f"  sim>{t:.2f}", end="")
print(f"  {'Total':>6}")

total_near_dups = {t: 0 for t in THRESHOLDS}
for cat in sorted(df["top_category"].unique()):
    mask = df["top_category"] == cat
    idx = np.where(mask)[0]
    if len(idx) < 2:
        continue
    cat_emb = embeddings[idx]
    sim = cosine_similarity(cat_emb)
    np.fill_diagonal(sim, 0)

    print(f"  {cat:<22}", end="")
    for t in THRESHOLDS:
        # Count pairs (not double-count)
        n_pairs = int((sim > t).sum() / 2)
        total_near_dups[t] += n_pairs
        print(f"  {n_pairs:>7}", end="")
    print(f"  {len(idx):>6}")

print(f"\n  {'TOTAL':<22}", end="")
for t in THRESHOLDS:
    print(f"  {total_near_dups[t]:>7}", end="")
print(f"  {len(df):>6}")

# Show worst near-duplicate pairs at 0.95
print(f"\nTop 10 most similar intra-category pairs (sim > 0.95):")
all_high_pairs = []
for cat in sorted(df["top_category"].unique()):
    mask = df["top_category"] == cat
    idx = np.where(mask)[0]
    if len(idx) < 2:
        continue
    cat_emb = embeddings[idx]
    sim = cosine_similarity(cat_emb)
    np.fill_diagonal(sim, 0)
    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            if sim[i, j] > 0.95:
                all_high_pairs.append({
                    "cat": cat,
                    "sim": sim[i, j],
                    "p1": df.iloc[idx[i]]["prompt"][:80],
                    "p2": df.iloc[idx[j]]["prompt"][:80],
                })

all_high_pairs.sort(key=lambda x: x["sim"], reverse=True)
for p in all_high_pairs[:10]:
    print(f"  [{p['cat']}] sim={p['sim']:.3f}")
    print(f"    A: {p['p1']}")
    print(f"    B: {p['p2']}")


# ── 3. KEYWORD LEAKAGE CHECK (aligned with generation forbidden words) ────
print(f"\n{'=' * 70}")
print("3. KEYWORD LEAKAGE CHECK")
print("=" * 70)

# Category-indicative keywords — any word that appears in >25% of one
# category's prompts and <5% of all other categories combined is a leak.
LEAK_THRESHOLD = 0.25
OUTSIDE_THRESHOLD = 0.05

# Build vocabulary per category
cat_vocabularies = {}
for cat in sorted(df["top_category"].unique()):
    cat_prompts = df[df["top_category"] == cat]["prompt"]
    words = Counter()
    for p in cat_prompts:
        for w in set(re.findall(r'\b\w+\b', p.lower())):
            words[w] += 1
    cat_vocabularies[cat] = {w: c / len(cat_prompts) for w, c in words.items()}

# Check each word for leakage
leaks = []
all_words = set()
for vocab in cat_vocabularies.values():
    all_words.update(vocab.keys())

for word in sorted(all_words):
    for cat, vocab in cat_vocabularies.items():
        freq_in = vocab.get(word, 0)
        if freq_in < LEAK_THRESHOLD:
            continue
        # Check frequency in all OTHER categories
        other_prompts = df[df["top_category"] != cat]["prompt"]
        freq_out = other_prompts.str.lower().str.contains(rf'\b{re.escape(word)}\b').mean()
        if freq_out < OUTSIDE_THRESHOLD:
            leaks.append({"word": word, "category": cat, "freq_in": freq_in, "freq_out": freq_out})

if leaks:
    print(f"\nLeaked keywords found: {len(leaks)}")
    print(f"  {'Word':<20} {'Category':<24} {'Freq in':>8} {'Freq out':>9}")
    for l in sorted(leaks, key=lambda x: x["freq_in"], reverse=True)[:20]:
        print(f"  {l['word']:<20} {l['category']:<24} {l['freq_in']:>7.1%} {l['freq_out']:>8.1%}")
else:
    print("\nNo keyword leakage detected (threshold: >25% in-category, <5% outside)")


# ── 4. CROSS-CATEGORY SEPARABILITY ────────────────────────────────────────
print(f"\n{'=' * 70}")
print("4. CROSS-CATEGORY SEPARABILITY")
print("=" * 70)

# Compute inter-category centroid distances
print("\nEmbedding centroid distances (cosine similarity):")
cats = sorted(df["top_category"].unique())
centroids = {}
for cat in cats:
    mask = df["top_category"] == cat
    centroids[cat] = embeddings[np.where(mask)[0]].mean(axis=0)

print(f"{'':>22}", end="")
for c in cats:
    print(f"  {c[:8]:>8}", end="")
print()
for c1 in cats:
    print(f"  {c1:<20}", end="")
    for c2 in cats:
        sim = np.dot(centroids[c1], centroids[c2]) / (
            np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2])
        )
        print(f"  {sim:>8.3f}", end="")
    print()

# Quick 5-fold cross-validation with TF-IDF + LR to measure baseline separability
print(f"\n5-fold CV separability (TF-IDF+LR, top_category):")
tfidf_cv = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_cv = tfidf_cv.fit_transform(df["prompt"])
y_cv = df["top_category"].to_numpy()

lr_cv = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
scores = cross_val_score(lr_cv, X_cv, y_cv, cv=5, scoring="f1_macro")
print(f"  Macro-F1: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"  Per-fold: {', '.join(f'{s:.3f}' for s in scores)}")

# Also test with just unigrams (simpler model)
tfidf_uni = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_uni = tfidf_uni.fit_transform(df["prompt"])
scores_uni = cross_val_score(lr_cv, X_uni, y_cv, cv=5, scoring="f1_macro")
print(f"  Unigram-only Macro-F1: {scores_uni.mean():.3f} +/- {scores_uni.std():.3f}")


# ── 5. STYLE LABEL AUDIT ──────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("5. STYLE LABEL AUDIT")
print("=" * 70)

if "style" in df.columns:
    # Check if style label matches actual content
    has_question_mark = df["prompt"].str.contains(r"\?")
    labeled_question = df["style"] == "question"

    mismatched = has_question_mark != labeled_question
    n_mismatch = mismatched.sum()
    print(f"Style label vs '?' presence: {n_mismatch} mismatches ({n_mismatch/len(df)*100:.1f}%)")

    if n_mismatch > 0:
        print(f"\n  Breakdown:")
        # Has ? but labeled instruction
        q_as_i = (has_question_mark & ~labeled_question).sum()
        # No ? but labeled question
        i_as_q = (~has_question_mark & labeled_question).sum()
        print(f"    Has '?' but labeled 'instruction': {q_as_i}")
        print(f"    No '?' but labeled 'question': {i_as_q}")
else:
    print("No 'style' column found")


# ── 6. SUMMARY ────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("VALIDATION SUMMARY")
print("=" * 70)

results = {
    "dataset": FULL_CSV.name,
    "total_prompts": len(df),
    "n_categories": len(cats),
    "imbalance_ratio": round(imbalance_ratio, 1),
    "near_duplicates_095": total_near_dups.get(0.95, 0),
    "near_duplicates_097": total_near_dups.get(0.97, 0),
    "near_duplicates_099": total_near_dups.get(0.99, 0),
    "keyword_leaks": len(leaks),
    "cv_macro_f1_bigram": round(scores.mean(), 3),
    "cv_macro_f1_unigram": round(scores_uni.mean(), 3),
}

RESULTS_VALIDATION.mkdir(parents=True, exist_ok=True)
_out = RESULTS_VALIDATION / "synthetic_validation.json"
with open(_out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {_out}")
print(json.dumps(results, indent=2))
