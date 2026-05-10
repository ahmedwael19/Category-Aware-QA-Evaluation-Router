"""Comprehensive benchmark orchestration.

Trains all 10 classifier configurations, evaluates on test + holdout,
computes ECE/Brier, bootstrap CIs, per-class F1, and saves all CSVs.
"""

import re
import time as timer

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from router.classifiers import (
    train_tfidf_lr,
    train_tfidf_svm,
    train_tfidf_nb,
    train_emb_lr,
    train_emb_lgbm,
    train_emb_mlp,
    train_ensemble,
    ensemble_predict,
)
from router.calibration import compute_ece_from_strings, compute_brier_from_strings

_NUM_PAT = re.compile(r"\d+")
_N_INFER_RUNS = 100


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_inference(predict_fn, X, n_runs=_N_INFER_RUNS):
    """Median inference time per sample in ms, averaged over n_runs."""
    times = []
    for _ in range(n_runs):
        t = timer.time()
        predict_fn(X)
        times.append(timer.time() - t)
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    return np.median(times) / n * 1000


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark(
    df_train, df_test,
    df_train_full,
    feature_col, target_col, classes,
    emb_train, emb_test,
    data_dir, results_dir, figures_dir,
    keyword_classify_fn,
    holdout_path=None,
):
    """Run all 10 classifiers, return benchmark_results list.

    Parameters
    ----------
    df_train, df_test : DataFrame
        Already <NUM>-masked splits with *feature_col* and *target_col*.
    df_train_full : DataFrame
        Full training set (with raw prompts for the raw-digit ablation).
    feature_col, target_col : str
    classes : list of str
    emb_train, emb_test : ndarray
    data_dir, results_dir, figures_dir : Path
    keyword_classify_fn : callable
        Keyword router function.
    holdout_path : Path or None

    Returns
    -------
    all_results : list[dict]
        One dict per model with test_acc, test_f1, ho_f1, etc.
    cal_results : list[dict]
        ECE / Brier for calibratable models.
    all_preds_test : dict[str, ndarray]
        Predictions on test set keyed by model name.
    all_preds_ho : dict[str, ndarray] or None
    """
    from pathlib import Path as _Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    train_prompts = df_train[feature_col]
    train_labels = df_train[target_col]
    test_labels = df_test[target_col].values

    # ── Holdout ─────────────────────────────────────────────────────
    has_holdout = holdout_path is not None and _Path(holdout_path).exists()
    ho_df = ho_prompts_norm = ho_labels = ho_emb = None
    if has_holdout:
        ho_df = pd.read_csv(holdout_path)
        ho_df = ho_df[ho_df["top_category"].isin(classes)].reset_index(drop=True)
        ho_prompts_norm = ho_df["prompt"].apply(lambda t: _NUM_PAT.sub("<NUM>", t))
        ho_labels = ho_df["top_category"].values
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        emb_m = SentenceTransformer(EMBEDDING_MODEL)
        ho_emb = emb_m.encode(
            [f"query: {t}" for t in ho_df["prompt"]],
            batch_size=64, normalize_embeddings=True,
        )
        print(f"Holdout loaded: {len(ho_df)} prompts")

    # ── TF-IDF features for all sets ──────────────────────────────
    tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf_vec.fit_transform(train_prompts)
    X_test = tfidf_vec.transform(df_test[feature_col])
    X_ho = tfidf_vec.transform(ho_prompts_norm) if has_holdout else None

    # TF-IDF unigram ablation
    tfidf_uni = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=True)
    X_train_uni = tfidf_uni.fit_transform(train_prompts)
    X_test_uni = tfidf_uni.transform(df_test[feature_col])
    X_ho_uni = tfidf_uni.transform(ho_prompts_norm) if has_holdout else None

    # TF-IDF raw digits ablation
    train_csv = sorted(data_dir.glob("synthetic_final_*_train.csv"))[-1]
    test_csv = sorted(data_dir.glob("synthetic_final_*_test.csv"))[-1]
    train_raw_df = pd.read_csv(train_csv)
    test_raw_df = pd.read_csv(test_csv)
    train_raw = train_raw_df["prompt"]
    train_raw_labels = train_raw_df["top_category"]
    test_raw = test_raw_df["prompt"]
    tfidf_raw = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_raw = tfidf_raw.fit_transform(train_raw)
    X_test_raw = tfidf_raw.transform(test_raw)
    ho_raw = ho_df["prompt"] if has_holdout else None
    X_ho_raw = tfidf_raw.transform(ho_raw) if has_holdout else None

    # LabelEncoder for LightGBM/MLP
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_labels)

    print("=" * 80)
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("=" * 80)

    all_results = []

    def _eval_model(name, model_or_fn, X_te, X_holdout, train_time, preds_test, preds_ho):
        if callable(model_or_fn) and X_te is not None:
            infer_ms = _time_inference(model_or_fn, X_te)
        else:
            infer_ms = 0.0

        r = {
            "model": name,
            "test_acc": round(accuracy_score(test_labels, preds_test), 4),
            "test_f1": round(f1_score(test_labels, preds_test, average="macro", zero_division=0), 4),
            "test_wf1": round(f1_score(test_labels, preds_test, average="weighted", zero_division=0), 4),
            "train_s": round(train_time, 3),
            "infer_ms": round(infer_ms, 4),
        }
        if preds_ho is not None:
            r["ho_acc"] = round(accuracy_score(ho_labels, preds_ho), 4)
            r["ho_f1"] = round(f1_score(ho_labels, preds_ho, average="macro", zero_division=0), 4)
        all_results.append(r)
        ho_str = f"ho={r['ho_f1']:.3f}" if "ho_f1" in r else "ho=--"
        print(f"  {name:<30} test={r['test_f1']:.3f}  {ho_str}  infer={r['infer_ms']:.3f}ms")

    # ── 1. Keyword/Regex ──────────────────────────────────────────
    t0 = timer.time()
    kw_test = np.array([keyword_classify_fn(p) for p in df_test["prompt"]])
    kw_tt = timer.time() - t0
    kw_ho = np.array([keyword_classify_fn(p) for p in ho_df["prompt"]]) if has_holdout else None
    kw_times = []
    for _ in range(50):
        t = timer.time()
        [keyword_classify_fn(p) for p in df_test["prompt"]]
        kw_times.append(timer.time() - t)
    kw_infer = np.median(kw_times) / len(df_test) * 1000
    _eval_model("Keyword/Regex", None, None, None, 0, kw_test, kw_ho)
    all_results[-1]["infer_ms"] = round(kw_infer, 4)

    # ── 2. TF-IDF + LR (bigram, <NUM>) ───────────────────────────
    t0 = timer.time()
    lr = train_tfidf_lr(X_train, train_labels)
    tt = timer.time() - t0
    _eval_model("TF-IDF+LR (bi, <NUM>)", lr.predict, X_test, X_ho, tt,
                lr.predict(X_test), lr.predict(X_ho) if has_holdout else None)

    # ── 3. TF-IDF + LR (unigram, <NUM>) ──────────────────────────
    t0 = timer.time()
    lr_uni = train_tfidf_lr(X_train_uni, train_labels)
    tt = timer.time() - t0
    _eval_model("TF-IDF+LR (uni, <NUM>)", lr_uni.predict, X_test_uni, X_ho_uni, tt,
                lr_uni.predict(X_test_uni), lr_uni.predict(X_ho_uni) if has_holdout else None)

    # ── 4. TF-IDF + LR (bigram, raw digits) ──────────────────────
    t0 = timer.time()
    lr_raw = train_tfidf_lr(X_train_raw, train_raw_labels)
    tt = timer.time() - t0
    _eval_model("TF-IDF+LR (bi, raw)", lr_raw.predict, X_test_raw, X_ho_raw, tt,
                lr_raw.predict(X_test_raw), lr_raw.predict(X_ho_raw) if has_holdout else None)

    # ── 5. TF-IDF + SVM ──────────────────────────────────────────
    t0 = timer.time()
    svm = train_tfidf_svm(X_train, train_labels)
    tt = timer.time() - t0
    _eval_model("TF-IDF+SVM", svm.predict, X_test, X_ho, tt,
                svm.predict(X_test), svm.predict(X_ho) if has_holdout else None)

    # ── 6. TF-IDF + NB ───────────────────────────────────────────
    t0 = timer.time()
    nb = train_tfidf_nb(X_train, train_labels)
    tt = timer.time() - t0
    _eval_model("TF-IDF+NB", nb.predict, X_test, X_ho, tt,
                nb.predict(X_test), nb.predict(X_ho) if has_holdout else None)

    # ── 7. Emb + LR ──────────────────────────────────────────────
    t0 = timer.time()
    emb_lr = train_emb_lr(emb_train, train_labels)
    tt = timer.time() - t0
    _eval_model("Emb+LR", emb_lr.predict, emb_test, ho_emb if has_holdout else None, tt,
                emb_lr.predict(emb_test), emb_lr.predict(ho_emb) if has_holdout else None)

    # ── 8. Emb + LightGBM ────────────────────────────────────────
    # LightGBM requires libomp (OpenMP). On macOS: `brew install libomp`.
    # If unavailable, skip this variant rather than crashing the whole benchmark.
    gbm = gbm_le = gbm_test = gbm_ho = None
    try:
        t0 = timer.time()
        gbm, gbm_le = train_emb_lgbm(emb_train, train_labels)
        tt = timer.time() - t0
        gbm_test = gbm_le.inverse_transform(gbm.predict(emb_test))
        gbm_ho = gbm_le.inverse_transform(gbm.predict(ho_emb)) if has_holdout else None
        gbm_infer = _time_inference(lambda X: gbm.predict(X), emb_test)
        _eval_model("Emb+LightGBM", None, None, None, tt, gbm_test, gbm_ho)
        all_results[-1]["infer_ms"] = round(gbm_infer, 4)
    except (ImportError, OSError) as e:
        print(f"  Emb+LightGBM SKIPPED: {e}")
        print("  On macOS: brew install libomp; on Linux: libgomp usually ships with the wheel.")

    # ── 9. Emb + MLP ─────────────────────────────────────────────
    t0 = timer.time()
    mlp, mlp_le = train_emb_mlp(emb_train, train_labels)
    tt = timer.time() - t0
    mlp_test = mlp_le.inverse_transform(mlp.predict(emb_test))
    mlp_ho = mlp_le.inverse_transform(mlp.predict(ho_emb)) if has_holdout else None
    mlp_infer = _time_inference(lambda X: mlp.predict(X), emb_test)
    _eval_model("Emb+MLP", None, None, None, tt, mlp_test, mlp_ho)
    all_results[-1]["infer_ms"] = round(mlp_infer, 4)

    # ── 10. Ensemble (TF-IDF+LR + Emb+LR calibrated avg) ────────
    t0 = timer.time()
    cal_tfidf, cal_emb = train_ensemble(X_train, emb_train, train_labels)
    ens_tt = timer.time() - t0

    ens_test, ens_test_probs = ensemble_predict(cal_tfidf, cal_emb, X_test, emb_test)
    ens_ho = ens_ho_probs = None
    if has_holdout:
        ens_ho, ens_ho_probs = ensemble_predict(cal_tfidf, cal_emb, X_ho, ho_emb)

    ens_times = []
    for _ in range(50):
        t = timer.time()
        ensemble_predict(cal_tfidf, cal_emb, X_test, emb_test)
        ens_times.append(timer.time() - t)
    ens_infer = np.median(ens_times) / len(df_test) * 1000
    _eval_model("Ensemble (TF-IDF+Emb)", None, None, None, ens_tt, ens_test, ens_ho)
    all_results[-1]["infer_ms"] = round(ens_infer, 4)

    # ── ECE / Brier for calibratable models ───────────────────────
    print(f"\n{'=' * 80}")
    print("CALIBRATION METRICS (ECE, Brier) -- models with predict_proba")
    print("=" * 80)

    calibratable = [
        ("TF-IDF+LR", lr, X_test, X_ho),
        ("TF-IDF+NB", nb, X_test, X_ho),
        ("Emb+LR", emb_lr, emb_test, ho_emb if has_holdout else None),
    ]
    if gbm is not None:
        calibratable.append(("Emb+LightGBM", gbm, emb_test, ho_emb if has_holdout else None))
    calibratable += [
        ("Emb+MLP", mlp, emb_test, ho_emb if has_holdout else None),
        ("Ensemble", None, None, None),
    ]
    cal_results = []
    for name, model, Xte, Xho in calibratable:
        if name == "Ensemble":
            probs_te = ens_test_probs
            probs_ho = ens_ho_probs if has_holdout else None
        else:
            probs_te = model.predict_proba(Xte)
            probs_ho = model.predict_proba(Xho) if Xho is not None else None
        ece_te = compute_ece_from_strings(probs_te, test_labels, classes)
        brier_te = compute_brier_from_strings(probs_te, test_labels, classes)
        cr = {"model": name, "test_ece": ece_te, "test_brier": brier_te}
        if probs_ho is not None:
            cr["ho_ece"] = compute_ece_from_strings(probs_ho, ho_labels, classes)
            cr["ho_brier"] = compute_brier_from_strings(probs_ho, ho_labels, classes)
        cal_results.append(cr)
        print(f"  {name:<25} ECE={ece_te:.4f}  Brier={brier_te:.4f}" +
              (f"  | holdout ECE={cr['ho_ece']:.4f}" if "ho_ece" in cr else ""))

    # ── Per-category holdout confusion matrices for top 2 models ──
    if has_holdout:
        ranked = sorted(
            [r for r in all_results if "ho_f1" in r],
            key=lambda r: r["ho_f1"], reverse=True,
        )
        model_map = {
            "TF-IDF+LR (bi, <NUM>)": lambda: lr.predict(X_ho),
            "TF-IDF+LR (uni, <NUM>)": lambda: lr_uni.predict(X_ho_uni),
            "TF-IDF+LR (bi, raw)": lambda: lr_raw.predict(X_ho_raw),
            "TF-IDF+SVM": lambda: svm.predict(X_ho),
            "TF-IDF+NB": lambda: nb.predict(X_ho),
            "Emb+LR": lambda: emb_lr.predict(ho_emb),
            "Emb+MLP": lambda: mlp.predict(ho_emb),
            "Ensemble (TF-IDF+Emb)": lambda: ens_ho,
            "Keyword/Regex": lambda: kw_ho,
        }
        if gbm is not None:
            model_map["Emb+LightGBM"] = lambda: gbm_le.inverse_transform(gbm.predict(ho_emb))

        fig_top2, axes_top2 = plt.subplots(1, min(2, len(ranked)), figsize=(20, 8))
        if not hasattr(axes_top2, "__len__"):
            axes_top2 = [axes_top2]
        for ti, r in enumerate(ranked[:2]):
            pred_fn = model_map.get(r["model"])
            if pred_fn is None:
                continue
            preds = pred_fn()
            cm = confusion_matrix(ho_labels, preds, labels=classes)
            cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            sns.heatmap(
                cm_n, annot=True, fmt=".2f",
                cmap="Oranges" if ti == 0 else "Purples",
                xticklabels=classes, yticklabels=classes, ax=axes_top2[ti],
            )
            axes_top2[ti].set_xlabel("Predicted")
            axes_top2[ti].set_ylabel("True")
            axes_top2[ti].set_title(f"Holdout: {r['model']} (F1={r['ho_f1']:.3f})")
        plt.tight_layout()
        out_path = figures_dir / "holdout_top2_confusion.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_top2.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved {out_path.name}")

    # ── Generalization gap ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("GENERALIZATION GAP (Synthetic F1 - Holdout F1)")
    print("=" * 80)
    for r in all_results:
        if "ho_f1" in r:
            gap = r["test_f1"] - r["ho_f1"]
            print(f"  {r['model']:<30} test={r['test_f1']:.3f}  ho={r['ho_f1']:.3f}  gap={gap:+.3f}")

    # ── Collect all predictions ───────────────────────────────────
    all_preds_test = {
        "Keyword/Regex": kw_test,
        "TF-IDF+LR (bi, <NUM>)": lr.predict(X_test),
        "TF-IDF+LR (uni, <NUM>)": lr_uni.predict(X_test_uni),
        "TF-IDF+LR (bi, raw)": lr_raw.predict(X_test_raw),
        "TF-IDF+SVM": svm.predict(X_test),
        "TF-IDF+NB": nb.predict(X_test),
        "Emb+LR": emb_lr.predict(emb_test),
        "Emb+MLP": mlp_test,
        "Ensemble (TF-IDF+Emb)": ens_test,
    }
    if gbm_test is not None:
        all_preds_test["Emb+LightGBM"] = gbm_test

    all_preds_ho = None
    if has_holdout:
        all_preds_ho = {
            "Keyword/Regex": kw_ho,
            "TF-IDF+LR (bi, <NUM>)": lr.predict(X_ho),
            "TF-IDF+LR (uni, <NUM>)": lr_uni.predict(X_ho_uni),
            "TF-IDF+LR (bi, raw)": lr_raw.predict(X_ho_raw),
            "TF-IDF+SVM": svm.predict(X_ho),
            "TF-IDF+NB": nb.predict(X_ho),
            "Emb+LR": emb_lr.predict(ho_emb),
            "Emb+MLP": mlp_ho,
            "Ensemble (TF-IDF+Emb)": ens_ho,
        }
        if gbm_ho is not None:
            all_preds_ho["Emb+LightGBM"] = gbm_ho

    # ── Inference note ────────────────────────────────────────────
    for r in all_results:
        r["infer_note"] = "+8ms embed" if r["model"].startswith("Emb") or r["model"].startswith("Ensemble") else "total"

    # ── Per-class F1 ──────────────────────────────────────────────
    per_class_rows = []
    for model_name, preds in all_preds_test.items():
        for c in classes:
            pt = round(f1_score(test_labels == c, preds == c, average="binary", zero_division=0), 4)
            per_class_rows.append({"model": model_name, "category": c, "f1": pt})
    router_dir = results_dir / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class_rows).to_csv(str(router_dir / "per_class_f1.csv"), index=False)
    print("Saved per_class_f1.csv (all models x all categories)")

    # ── Bootstrap 95% CIs ─────────────────────────────────────────
    n_boot = 500
    np.random.seed(42)
    n_test = len(test_labels)
    boot_rows = []
    for model_name, preds in all_preds_test.items():
        boot_f1s = []
        for _ in range(n_boot):
            idx = np.random.choice(n_test, size=n_test, replace=True)
            boot_f1s.append(f1_score(test_labels[idx], preds[idx], average="macro", zero_division=0))
        boot_rows.append({
            "model": model_name,
            "macro_f1": round(f1_score(test_labels, preds, average="macro", zero_division=0), 4),
            "ci_lo": round(np.percentile(boot_f1s, 2.5), 4),
            "ci_hi": round(np.percentile(boot_f1s, 97.5), 4),
        })
    pd.DataFrame(boot_rows).to_csv(str(router_dir / "bootstrap_ci.csv"), index=False)
    print("Saved bootstrap_ci.csv (all models)")

    # ── Holdout predictions CSV ───────────────────────────────────
    if has_holdout and all_preds_ho is not None:
        ho_preds_df = ho_df[["prompt", "top_category"]].copy()
        for mn, pp in all_preds_ho.items():
            ho_preds_df[mn] = pp
        ho_preds_df.to_csv(str(router_dir / "holdout_predictions.csv"), index=False)
        print("Saved holdout_predictions.csv (all models)")

    # ── Save benchmark CSV ────────────────────────────────────────
    bench_df = pd.DataFrame(all_results)
    bench_df.to_csv(str(router_dir / "benchmark.csv"), index=False)
    print(f"Saved benchmark.csv ({len(bench_df)} models)")

    # ── Generalization gap CSV ────────────────────────────────────
    gap_df = pd.DataFrame([
        {
            "model": r["model"],
            "test_f1": r["test_f1"],
            "holdout_f1": r.get("ho_f1"),
            "gap": round(r["test_f1"] - r["ho_f1"], 4) if "ho_f1" in r else None,
        }
        for r in all_results
    ])
    gap_df.to_csv(str(router_dir / "generalization_gap.csv"), index=False)
    print("Saved generalization_gap.csv")

    # ── Calibration CSV ───────────────────────────────────────────
    for cr in cal_results:
        for k in cr:
            if isinstance(cr[k], float):
                cr[k] = round(cr[k], 4)
    cal_df = pd.DataFrame(cal_results)
    cal_df.to_csv(str(router_dir / "calibration.csv"), index=False)
    print("Saved calibration.csv")

    return all_results, cal_results, all_preds_test, all_preds_ho
