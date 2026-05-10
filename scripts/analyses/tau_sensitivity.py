"""Confidence-threshold (tau) sensitivity sweep.

Sweeps tau from 0.0 to 0.95 for each calibrated router and measures end-to-end
accuracy and coverage. Re-uses the cached LLM answers in
`evaluation_with_llm.csv`; no API calls.

Run: python -m scripts.analyses.tau_sensitivity
"""

import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    DATA_DIR, DATASET_TRAIN, EMBEDDINGS_TRAIN, RESULTS_ANALYSES, SEED, TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

DETERMINISTIC_CATS = {"SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"}
LLM_CATS = {"SEMANTIC", "HYBRID"}
REJECT_CATS = {"UNSUPPORTED"}

NUM_RE = re.compile(r"\d+")


def compute_e2e_accuracy(preds, confs, tau, eval_df):
    """End-to-end accuracy at the given tau.

    Mirrors the routing policy in `notebooks/03_evaluation_pipeline.py`: prompts
    with confidence below tau fall back to the LLM answer; SYMBOLIC_* prompts
    are answered by the deterministic function (100% correct by construction);
    SEMANTIC/HYBRID use the cached LLM answer; UNSUPPORTED resolves to reject.
    """
    correct = 0
    fallback_count = 0

    for i, (_, row) in enumerate(eval_df.iterrows()):
        pred = preds[i]
        conf = confs[i]
        gt = row["ground_truth"]

        if conf < tau:
            fallback_count += 1
            answer = row["system_a_binary_answer"]
        elif pred in DETERMINISTIC_CATS:
            answer = gt
        elif pred in LLM_CATS:
            answer = row["system_a_binary_answer"]
        elif pred in REJECT_CATS:
            answer = "reject"
        else:
            answer = row["system_a_binary_answer"]

        if answer == gt:
            correct += 1

    return correct / len(eval_df), 1 - (fallback_count / len(eval_df)), fallback_count


def main() -> None:
    print("Loading data...")
    eval_df = pd.read_csv(DATA_DIR / "evaluation_with_llm.csv")
    train_df = pd.read_csv(DATASET_TRAIN)

    train_prompts_masked = train_df["prompt"].apply(lambda t: NUM_RE.sub("<NUM>", t))
    train_labels = train_df["top_category"]
    print(f"  Eval: {len(eval_df)} pairs, Train: {len(train_df)} prompts")

    print("Training routers...")
    tfidf_vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True,
    )
    X_train_tfidf = tfidf_vec.fit_transform(train_prompts_masked)

    cal_tfidf = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=SEED),
        method="sigmoid", cv=3,
    )
    cal_tfidf.fit(X_train_tfidf, train_labels)

    tfidf_svm = LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, random_state=SEED)
    tfidf_svm.fit(X_train_tfidf, train_labels)

    emb_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    if EMBEDDINGS_TRAIN.exists():
        emb_train = np.load(EMBEDDINGS_TRAIN)
    else:
        emb_train = emb_model.encode(
            [f"query: {t}" for t in train_df["prompt"]],
            batch_size=64, normalize_embeddings=True,
        )

    cal_emb = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=SEED),
        method="sigmoid", cv=3,
    )
    cal_emb.fit(emb_train, train_labels)

    print("Classifying eval prompts...")
    eval_prompts = eval_df["prompt"].tolist()
    eval_prompts_masked = [NUM_RE.sub("<NUM>", p) for p in eval_prompts]

    X_eval_tfidf = tfidf_vec.transform(eval_prompts_masked)
    tfidf_lr_probs = cal_tfidf.predict_proba(X_eval_tfidf)
    tfidf_lr_preds = cal_tfidf.classes_[np.argmax(tfidf_lr_probs, axis=1)]
    tfidf_lr_confs = np.max(tfidf_lr_probs, axis=1)
    svm_preds = tfidf_svm.predict(X_eval_tfidf)

    eval_embs = emb_model.encode(
        [f"query: {p}" for p in eval_prompts], batch_size=64, normalize_embeddings=True,
    )
    emb_lr_probs = cal_emb.predict_proba(eval_embs)
    emb_lr_preds = cal_emb.classes_[np.argmax(emb_lr_probs, axis=1)]
    emb_lr_confs = np.max(emb_lr_probs, axis=1)

    ens_probs = (tfidf_lr_probs + emb_lr_probs) / 2
    ens_preds = cal_tfidf.classes_[np.argmax(ens_probs, axis=1)]
    ens_confs = np.max(ens_probs, axis=1)

    tau_values = sorted({round(t, 2) for t in [0.0, *np.arange(0.50, 0.96, 0.05)]})
    print(f"\nSweeping tau over {len(tau_values)} values")

    all_rows = []
    models = {
        "TF-IDF+LR": (tfidf_lr_preds, tfidf_lr_confs),
        "Emb+LR": (emb_lr_preds, emb_lr_confs),
        "Ensemble": (ens_preds, ens_confs),
    }

    for model_name, (preds, confs) in models.items():
        print(f"\n  {model_name}:")
        for tau in tau_values:
            acc, cov, fb = compute_e2e_accuracy(preds, confs, tau, eval_df)
            all_rows.append({
                "model": model_name,
                "tau": tau,
                "accuracy": round(acc, 4),
                "coverage": round(cov, 4),
                "fallback_count": fb,
            })
            marker = " <-- selected" if abs(tau - 0.85) < 0.001 else ""
            print(f"    tau={tau:.2f}: acc={acc:.1%}, coverage={cov:.0%}, fallback={fb}{marker}")
        sys.stdout.flush()

    svm_acc, _, _ = compute_e2e_accuracy(svm_preds, np.ones(len(svm_preds)), 0.0, eval_df)
    all_rows.append({
        "model": "TF-IDF+SVM",
        "tau": 0.0,
        "accuracy": round(svm_acc, 4),
        "coverage": 1.0,
        "fallback_count": 0,
    })

    results_df = pd.DataFrame(all_rows)
    RESULTS_ANALYSES.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ANALYSES / "tau_sensitivity.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(results_df)} rows)")

    print("\nOptimal tau per model:")
    for model_name in models:
        model_rows = results_df[results_df["model"] == model_name]
        best = model_rows.loc[model_rows["accuracy"].idxmax()]
        print(f"  {model_name}: tau={best['tau']:.2f}  acc={best['accuracy']:.1%}  "
              f"coverage={best['coverage']:.0%}")
    print(f"  TF-IDF+SVM (no tau): acc={svm_acc:.1%}")


if __name__ == "__main__":
    main()
