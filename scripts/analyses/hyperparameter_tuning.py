"""HP tuning: GridSearchCV on TF-IDF+SVM, TF-IDF+LR, Emb+LR, LightGBM.

Uses the existing 70/15/15 split. Tunes on train+val (5-fold CV on train).
Reports test Macro-F1 for default vs tuned.
"""

import json
import re


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from config import (
    ROOT,
    DATASET_TRAIN, DATASET_VAL, DATASET_TEST,
    FEATURE_COL, TARGET_COL, SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    EMBEDDINGS_TRAIN, EMBEDDINGS_VAL, EMBEDDINGS_TEST,
)

if __name__ != "__main__":
    raise RuntimeError("Run as: python -m scripts.analyses.hyperparameter_tuning")

NUM = re.compile(r"\d+")
def mask(t): return NUM.sub("<NUM>", t)

print("=" * 80)
print("HP TUNING: TF-IDF+LR, TF-IDF+SVM, Emb+LR, LightGBM")
print("=" * 80)

dtr = pd.read_csv(DATASET_TRAIN)
dv = pd.read_csv(DATASET_VAL)
dt = pd.read_csv(DATASET_TEST)
print(f"train={len(dtr)}  val={len(dv)}  test={len(dt)}")

y_train = dtr[TARGET_COL].astype(str).to_numpy()
y_val = dv[TARGET_COL].astype(str).to_numpy()
y_test = dt[TARGET_COL].astype(str).to_numpy()

# ── TF-IDF ───────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=True,
)
X_train_tf = tfidf.fit_transform([mask(p) for p in dtr[FEATURE_COL]])
X_val_tf = tfidf.transform([mask(p) for p in dv[FEATURE_COL]])
X_test_tf = tfidf.transform([mask(p) for p in dt[FEATURE_COL]])

# ── Embeddings (cached) ──────────────────────────────────────────────────────
E_train = np.load(str(EMBEDDINGS_TRAIN))
E_val = np.load(str(EMBEDDINGS_VAL))
E_test = np.load(str(EMBEDDINGS_TEST))

results = {}

# ── TF-IDF + LR ──────────────────────────────────────────────────────────────
print("\n[1/4] TF-IDF + LR")
base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
base_lr.fit(X_train_tf, y_train)
f1_default = f1_score(y_test, base_lr.predict(X_test_tf), average="macro")
grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    {"C": [0.1, 0.3, 1.0, 3.0, 10.0]},
    scoring="f1_macro", cv=5, n_jobs=1,
)
grid.fit(X_train_tf, y_train)
f1_tuned = f1_score(y_test, grid.best_estimator_.predict(X_test_tf), average="macro")
print(f"  default C=1.0 -> test F1 = {f1_default:.4f}")
print(f"  best    C={grid.best_params_['C']} -> test F1 = {f1_tuned:.4f}  (cv F1 = {grid.best_score_:.4f})")
results["tfidf_lr"] = {"default_C": 1.0, "default_test_f1": f1_default,
                       "best_params": grid.best_params_, "tuned_test_f1": f1_tuned,
                       "cv_score": grid.best_score_}

# ── TF-IDF + SVM ─────────────────────────────────────────────────────────────
print("\n[2/4] TF-IDF + SVM")
base_svm = LinearSVC(class_weight="balanced", max_iter=2000, random_state=SEED, C=1.0)
base_svm.fit(X_train_tf, y_train)
f1_default_svm = f1_score(y_test, base_svm.predict(X_test_tf), average="macro")
grid_svm = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=4000, random_state=SEED),
    {"C": [0.1, 0.3, 1.0, 3.0, 10.0]},
    scoring="f1_macro", cv=5, n_jobs=1,
)
grid_svm.fit(X_train_tf, y_train)
f1_tuned_svm = f1_score(y_test, grid_svm.best_estimator_.predict(X_test_tf), average="macro")
print(f"  default C=1.0 -> test F1 = {f1_default_svm:.4f}")
print(f"  best    C={grid_svm.best_params_['C']} -> test F1 = {f1_tuned_svm:.4f}  (cv F1 = {grid_svm.best_score_:.4f})")
results["tfidf_svm"] = {"default_C": 1.0, "default_test_f1": f1_default_svm,
                        "best_params": grid_svm.best_params_, "tuned_test_f1": f1_tuned_svm,
                        "cv_score": grid_svm.best_score_}

# ── Emb + LR ─────────────────────────────────────────────────────────────────
print("\n[3/4] Emb + LR")
base_emb_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
base_emb_lr.fit(E_train, y_train)
f1_default_emb = f1_score(y_test, base_emb_lr.predict(E_test), average="macro")
grid_emb = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    {"C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    scoring="f1_macro", cv=5, n_jobs=1,
)
grid_emb.fit(E_train, y_train)
f1_tuned_emb = f1_score(y_test, grid_emb.best_estimator_.predict(E_test), average="macro")
print(f"  default C=1.0 -> test F1 = {f1_default_emb:.4f}")
print(f"  best    C={grid_emb.best_params_['C']} -> test F1 = {f1_tuned_emb:.4f}  (cv F1 = {grid_emb.best_score_:.4f})")
results["emb_lr"] = {"default_C": 1.0, "default_test_f1": f1_default_emb,
                     "best_params": grid_emb.best_params_, "tuned_test_f1": f1_tuned_emb,
                     "cv_score": grid_emb.best_score_}

print("\n[4/4] Emb + LightGBM")
try:
    import lightgbm as lgb

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    class_counts = np.bincount(y_train_enc)
    n_samples = len(y_train_enc)
    n_classes = len(class_counts)
    class_weights = n_samples / (n_classes * class_counts)
    sw = np.array([class_weights[y] for y in y_train_enc])

    base_lgb = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.1,
        min_child_samples=10, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    base_lgb.fit(E_train, y_train_enc, sample_weight=sw)
    f1_default_lgb = f1_score(y_test_enc, base_lgb.predict(E_test), average="macro")

    grid_lgb = GridSearchCV(
        lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        {
            "n_estimators": [100, 200, 400],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1],
            "min_child_samples": [5, 10, 20],
        },
        scoring="f1_macro", cv=5, n_jobs=1,
    )
    grid_lgb.fit(E_train, y_train_enc, sample_weight=sw)
    f1_tuned_lgb = f1_score(y_test_enc, grid_lgb.best_estimator_.predict(E_test), average="macro")
    print(f"  defaults -> test F1 = {f1_default_lgb:.4f}")
    print(f"  best {grid_lgb.best_params_} -> test F1 = {f1_tuned_lgb:.4f}  (cv F1 = {grid_lgb.best_score_:.4f})")
    results["emb_lgbm"] = {
        "default_params": {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 10},
        "default_test_f1": f1_default_lgb,
        "best_params": grid_lgb.best_params_,
        "tuned_test_f1": f1_tuned_lgb,
        "cv_score": grid_lgb.best_score_,
    }
except (OSError, ImportError) as e:
    print(f"  SKIPPED: LightGBM unavailable ({e})")
    print("  On macOS: brew install libomp; on Linux: libgomp is usually bundled.")
    results["emb_lgbm"] = {"skipped": True, "reason": str(e)}

# ── Save ─────────────────────────────────────────────────────────────────────
out = ROOT / "results" / "router" / "hp_tuning.json"
out.parent.mkdir(parents=True, exist_ok=True)
# Numpy types → python
def clean(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
with open(out, "w") as f:
    json.dump(clean(results), f, indent=2)

print("\n" + "=" * 80)
print("SUMMARY: tuned test Macro-F1 vs default test Macro-F1")
print("=" * 80)
for name, r in results.items():
    if r.get("skipped"):
        print(f"  {name:<12}  skipped ({r['reason']})")
        continue
    delta = r["tuned_test_f1"] - r["default_test_f1"]
    print(f"  {name:<12}  default={r['default_test_f1']:.4f}  tuned={r['tuned_test_f1']:.4f}  Δ={delta:+.4f}")

print(f"\nSaved: {out}")
