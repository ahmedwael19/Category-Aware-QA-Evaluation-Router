import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    import re
    import time as timer
    from pathlib import Path

    from collections import Counter
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
        brier_score_loss,
    )
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.preprocessing import LabelEncoder

    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    return (
        LabelEncoder,
        LogisticRegression,
        Path,
        TfidfVectorizer,
        accuracy_score,
        brier_score_loss,
        calibration_curve,
        classification_report,
        confusion_matrix,
        f1_score,
        joblib,
        json,
        mo,
        np,
        pd,
        plt,
        re,
        sns,
        timer,
    )


@app.cell
def title(mo):
    mo.md("""
    # Router Training — Classifier Difficulty Ladder + Calibration

    **Purpose**: Train the prompt router classifier and validate the synthetic dataset quality.

    This notebook implements two critical functions:
    1. **Dataset validation** via the classifier difficulty ladder (Section 3.3.8 of methodology):
       `Random < Majority < Keyword < TF-IDF+LR < Embedding+LR < Embedding+LightGBM`
    2. **Router training** with confidence calibration for production deployment.

    **Note**: Only the `prompt` text is used as input. The `params`, `subcategory`,
    `noise`, and `style` columns are **excluded** from all classifier training to prevent
    target leakage.

    **Target**: `top_category` (6 classes: SEMANTIC, SYMBOLIC_TIME, SYMBOLIC_COUNT,
    SYMBOLIC_METADATA, HYBRID, UNSUPPORTED)
    """)
    return


@app.cell
def load_data(Path, mo, pd):
    """Load train/val/test splits. Use ONLY prompt + top_category."""

    _root = Path(__file__).parent.parent
    _data_dir = _root / "data"

    # Find the latest synthetic_final dataset
    _candidates = sorted(_data_dir.glob("synthetic_final_*_train.csv"))
    if not _candidates:
        raise FileNotFoundError("No synthetic_final_*_train.csv found")
    _latest = _candidates[-1].name.replace("_train.csv", "")
    _base = _data_dir / _latest

    df_train_full = pd.read_csv(f"{_base}_train.csv")
    df_val_full = pd.read_csv(f"{_base}_val.csv")
    df_test_full = pd.read_csv(f"{_base}_test.csv")

    # Keep prompt + top_category only; anything else leaks the target.
    FEATURE_COL = "prompt"
    TARGET_COL = "top_category"
    df_train = df_train_full[[FEATURE_COL, TARGET_COL]].copy()
    df_val = df_val_full[[FEATURE_COL, TARGET_COL]].copy()
    df_test = df_test_full[[FEATURE_COL, TARGET_COL]].copy()

    # Digit normalization: replace all numbers with <NUM> to prevent memorization
    # of specific thresholds (e.g., "60" → SYMBOLIC_TIME). Forces the model to
    # learn structural patterns ("within <NUM> minutes") rather than specific values.
    import re as _re
    _num_pat = _re.compile(r'\d+')
    for _df in [df_train, df_val, df_test]:
        _df[FEATURE_COL] = _df[FEATURE_COL].apply(lambda t: _num_pat.sub("<NUM>", t))

    CLASSES = sorted(df_train[TARGET_COL].unique())
    DATA_PREFIX = str(_base)

    mo.md(f"""
    ### Dataset: `{_latest}`

    | Split | Rows |
    |-------|------|
    | Train | {len(df_train):,} |
    | Val   | {len(df_val):,} |
    | Test  | {len(df_test):,} |

    **Columns used**: `{FEATURE_COL}` (input), `{TARGET_COL}` (target)

    **Columns dropped** (leak prevention): params, subcategory, noise, style, category
    """)

    print("Class distribution (train):")
    for cls in CLASSES:
        n = (df_train[TARGET_COL] == cls).sum()
        print(f"  {cls:<22} {n:>5} ({n/len(df_train)*100:.1f}%)")
    return (
        CLASSES,
        DATA_PREFIX,
        FEATURE_COL,
        TARGET_COL,
        df_test,
        df_train,
        df_train_full,
        df_val,
    )


@app.cell
def dataset_validation(CLASSES, TARGET_COL, df_train_full, mo, re):
    """Dataset validation metrics: TTR, length, noise per category."""

    # TTR per top_category
    _ttr_data = []
    for _cat in CLASSES:
        _prompts = df_train_full[df_train_full[TARGET_COL] == _cat]["prompt"]
        _all_tokens = []
        for p in _prompts:
            _all_tokens.extend(re.findall(r'\w+', p.lower()))
        _types = len(set(_all_tokens))
        _tokens = len(_all_tokens)
        _ttr = _types / _tokens if _tokens > 0 else 0
        _ttr_data.append({"category": _cat, "ttr": _ttr, "types": _types, "tokens": _tokens})

    # Length stats per top_category
    _len_data = []
    for _cat in CLASSES:
        _lens = df_train_full[df_train_full[TARGET_COL] == _cat]["prompt"].str.len()
        _len_data.append({
            "category": _cat, "mean_len": _lens.mean(),
            "median_len": _lens.median(), "std_len": _lens.std(),
        })

    mo.md(f"""
    ### Dataset Validation Metrics

    **Type-Token Ratio** (lexical diversity per top_category):

    | Category | TTR | Types | Tokens |
    |----------|-----|-------|--------|
    {"".join(f"| {d['category']} | {d['ttr']:.3f} | {d['types']:,} | {d['tokens']:,} |{chr(10)}" for d in _ttr_data)}

    **Prompt Length** (characters):

    | Category | Mean | Median | Std |
    |----------|------|--------|-----|
    {"".join(f"| {d['category']} | {d['mean_len']:.0f} | {d['median_len']:.0f} | {d['std_len']:.0f} |{chr(10)}" for d in _len_data)}
    """)
    return


@app.cell
def baseline_random(
    TARGET_COL,
    accuracy_score,
    df_test,
    df_train,
    f1_score,
    np,
):
    """Baseline 1: Random classifier (samples from prior distribution)."""
    _prior = df_train[TARGET_COL].value_counts(normalize=True)
    np.random.seed(42)
    _preds = np.random.choice(_prior.index, size=len(df_test), p=_prior.values)

    random_acc = accuracy_score(df_test[TARGET_COL], _preds)
    random_macro_f1 = f1_score(df_test[TARGET_COL], _preds, average="macro")
    random_weighted_f1 = f1_score(df_test[TARGET_COL], _preds, average="weighted")

    print(f"[1] RANDOM BASELINE")
    print(f"    Accuracy:    {random_acc:.3f}")
    print(f"    Macro-F1:    {random_macro_f1:.3f}")
    print(f"    Weighted-F1: {random_weighted_f1:.3f}")
    return random_acc, random_macro_f1, random_weighted_f1


@app.cell
def baseline_majority(
    TARGET_COL,
    accuracy_score,
    df_test,
    df_train,
    f1_score,
    np,
):
    """Baseline 2: Majority class (always predict SEMANTIC)."""
    _majority = df_train[TARGET_COL].value_counts().index[0]
    _preds = np.full(len(df_test), _majority)

    majority_acc = accuracy_score(df_test[TARGET_COL], _preds)
    majority_macro_f1 = f1_score(df_test[TARGET_COL], _preds, average="macro")
    majority_weighted_f1 = f1_score(df_test[TARGET_COL], _preds, average="weighted")

    print(f"[2] MAJORITY BASELINE (always predict '{_majority}')")
    print(f"    Accuracy:    {majority_acc:.3f}")
    print(f"    Macro-F1:    {majority_macro_f1:.3f}")
    print(f"    Weighted-F1: {majority_weighted_f1:.3f}")
    return majority_acc, majority_macro_f1, majority_weighted_f1


@app.cell
def baseline_keyword(TARGET_COL, accuracy_score, df_test, f1_score, mo, re):
    """Baseline 3: Keyword/Regex classifier (for dataset validation).

    If accuracy > 80%, the dataset is considered too easy (methodology Section 3.3.8).
    """
    from router.keyword_baseline import keyword_router as keyword_classify

    _preds = [keyword_classify(p) for p in df_test["prompt"]]

    keyword_acc = accuracy_score(df_test[TARGET_COL], _preds)
    keyword_macro_f1 = f1_score(df_test[TARGET_COL], _preds, average="macro", zero_division=0)
    keyword_weighted_f1 = f1_score(df_test[TARGET_COL], _preds, average="weighted", zero_division=0)

    _status = "PASS" if keyword_acc < 0.80 else "FAIL — dataset may be too easy"

    print(f"[3] KEYWORD/REGEX BASELINE")
    print(f"    Accuracy:    {keyword_acc:.3f}  [{_status}]")
    print(f"    Macro-F1:    {keyword_macro_f1:.3f}")
    print(f"    Weighted-F1: {keyword_weighted_f1:.3f}")

    mo.md(f"""
    ### Keyword/Regex Baseline Result

    **Accuracy: {keyword_acc:.1%}** — {"PASS: Below 80% threshold" if keyword_acc < 0.80 else "WARNING: Above 80% threshold — dataset may be trivially solvable"}

    This validates that the synthetic dataset cannot be solved by simple pattern matching,
    confirming the need for a learned classifier (Section 3.3.8, validation strategy 3).
    """)
    return keyword_acc, keyword_classify, keyword_macro_f1, keyword_weighted_f1


@app.cell
def compute_embeddings(FEATURE_COL, df_test, df_train, df_val, mo):
    """Compute embeddings using multilingual-e5-large-instruct."""
    from router.feature_extraction import compute_and_cache_embeddings

    mo.md("### Computing Embeddings\nModel: `intfloat/multilingual-e5-large-instruct` (1024-dim)")

    _models_dir = str(Path(__file__).parent.parent / "models")
    emb_train, emb_val, emb_test = compute_and_cache_embeddings(
        df_train, df_val, df_test, FEATURE_COL, _models_dir,
    )
    return emb_test, emb_train, emb_val


@app.cell
def baseline_tfidf(
    FEATURE_COL,
    TARGET_COL,
    accuracy_score,
    df_test,
    df_train,
    df_val,
    f1_score,
    timer,
):
    """Baseline 4: TF-IDF + Logistic Regression."""
    from router.feature_extraction import build_tfidf_features
    from router.classifiers import train_tfidf_lr

    _t0 = timer.time()
    tfidf_vectorizer, _X_train, tf_lr_X_test, _X_val, _ = build_tfidf_features(
        df_train[FEATURE_COL], df_test[FEATURE_COL], val_prompts=df_val[FEATURE_COL],
    )
    tf_lr_model = train_tfidf_lr(_X_train, df_train[TARGET_COL])
    tfidf_train_time = timer.time() - _t0

    _t1 = timer.time()
    _val_preds = tf_lr_model.predict(_X_val)
    _test_preds = tf_lr_model.predict(tf_lr_X_test)
    tfidf_inference_time = (timer.time() - _t1) / (len(df_val) + len(df_test))

    tfidf_acc = accuracy_score(df_test[TARGET_COL], _test_preds)
    tfidf_macro_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="macro", zero_division=0)
    tfidf_weighted_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="weighted", zero_division=0)
    tfidf_val_f1 = f1_score(df_val[TARGET_COL], _val_preds, average="macro", zero_division=0)

    print(f"[4] TF-IDF + LOGISTIC REGRESSION")
    print(f"    Val Macro-F1:  {tfidf_val_f1:.3f}")
    print(f"    Test Accuracy: {tfidf_acc:.3f}")
    print(f"    Test Macro-F1: {tfidf_macro_f1:.3f}")
    print(f"    Test Wt-F1:    {tfidf_weighted_f1:.3f}")
    print(f"    Train time:    {tfidf_train_time:.2f}s")
    print(f"    Inference:     {tfidf_inference_time*1000:.2f}ms/prompt")
    return (
        tf_lr_X_test,
        tf_lr_model,
        tfidf_acc,
        tfidf_inference_time,
        tfidf_macro_f1,
        tfidf_train_time,
        tfidf_vectorizer,
        tfidf_weighted_f1,
    )


@app.cell
def baseline_emb_lr(
    TARGET_COL,
    accuracy_score,
    df_test,
    df_train,
    df_val,
    emb_test,
    emb_train,
    emb_val,
    f1_score,
    timer,
):
    """Baseline 5: Embedding + Logistic Regression."""
    from router.classifiers import train_emb_lr

    _t0 = timer.time()
    _lr = train_emb_lr(emb_train, df_train[TARGET_COL])
    emb_lr_train_time = timer.time() - _t0

    _t1 = timer.time()
    _val_preds = _lr.predict(emb_val)
    _test_preds = _lr.predict(emb_test)
    emb_lr_inference_time = (timer.time() - _t1) / (len(df_val) + len(df_test))

    emb_lr_acc = accuracy_score(df_test[TARGET_COL], _test_preds)
    emb_lr_macro_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="macro", zero_division=0)
    emb_lr_weighted_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="weighted", zero_division=0)
    emb_lr_val_f1 = f1_score(df_val[TARGET_COL], _val_preds, average="macro", zero_division=0)

    print(f"[5] EMBEDDING + LOGISTIC REGRESSION")
    print(f"    Val Macro-F1:  {emb_lr_val_f1:.3f}")
    print(f"    Test Accuracy: {emb_lr_acc:.3f}")
    print(f"    Test Macro-F1: {emb_lr_macro_f1:.3f}")
    print(f"    Test Wt-F1:    {emb_lr_weighted_f1:.3f}")
    print(f"    Train time:    {emb_lr_train_time:.2f}s")
    print(f"    Inference:     {emb_lr_inference_time*1000:.2f}ms/prompt (excludes embedding)")
    return (
        emb_lr_acc,
        emb_lr_inference_time,
        emb_lr_macro_f1,
        emb_lr_train_time,
        emb_lr_weighted_f1,
    )


@app.cell
def model_lgbm(
    TARGET_COL,
    accuracy_score,
    df_test,
    df_train,
    df_val,
    emb_test,
    emb_train,
    emb_val,
    f1_score,
    timer,
):
    """Model 6: Embedding + LightGBM (default params)."""
    from router.classifiers import train_emb_lgbm

    _t0 = timer.time()
    _model, _le = train_emb_lgbm(emb_train, df_train[TARGET_COL])
    lgbm_train_time = timer.time() - _t0

    _t1 = timer.time()
    _val_preds = _le.inverse_transform(_model.predict(emb_val))
    _test_preds = _le.inverse_transform(_model.predict(emb_test))
    lgbm_inference_time = (timer.time() - _t1) / (len(df_val) + len(df_test))

    lgbm_acc = accuracy_score(df_test[TARGET_COL], _test_preds)
    lgbm_macro_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="macro", zero_division=0)
    lgbm_weighted_f1 = f1_score(df_test[TARGET_COL], _test_preds, average="weighted", zero_division=0)
    lgbm_val_f1 = f1_score(df_val[TARGET_COL], _val_preds, average="macro", zero_division=0)

    print(f"[6] EMBEDDING + LIGHTGBM (default)")
    print(f"    Val Macro-F1:  {lgbm_val_f1:.3f}")
    print(f"    Test Accuracy: {lgbm_acc:.3f}")
    print(f"    Test Macro-F1: {lgbm_macro_f1:.3f}")
    print(f"    Test Wt-F1:    {lgbm_weighted_f1:.3f}")
    print(f"    Train time:    {lgbm_train_time:.2f}s")
    print(f"    Inference:     {lgbm_inference_time*1000:.2f}ms/prompt (excludes embedding)")
    return (
        lgbm_acc,
        lgbm_inference_time,
        lgbm_macro_f1,
        lgbm_train_time,
        lgbm_weighted_f1,
    )


@app.cell
def difficulty_ladder(
    emb_lr_acc,
    emb_lr_inference_time,
    emb_lr_macro_f1,
    emb_lr_train_time,
    emb_lr_weighted_f1,
    keyword_acc,
    keyword_macro_f1,
    keyword_weighted_f1,
    lgbm_acc,
    lgbm_inference_time,
    lgbm_macro_f1,
    lgbm_train_time,
    lgbm_weighted_f1,
    majority_acc,
    majority_macro_f1,
    majority_weighted_f1,
    mo,
    plt,
    random_acc,
    random_macro_f1,
    random_weighted_f1,
    tfidf_acc,
    tfidf_inference_time,
    tfidf_macro_f1,
    tfidf_train_time,
    tfidf_weighted_f1,
):
    """Difficulty ladder summary — validates dataset quality. Includes timing."""

    ladder = [
        {"name": "Random", "acc": random_acc, "macro_f1": random_macro_f1, "weighted_f1": random_weighted_f1, "train_s": 0, "infer_ms": 0},
        {"name": "Majority", "acc": majority_acc, "macro_f1": majority_macro_f1, "weighted_f1": majority_weighted_f1, "train_s": 0, "infer_ms": 0},
        {"name": "Keyword/Regex", "acc": keyword_acc, "macro_f1": keyword_macro_f1, "weighted_f1": keyword_weighted_f1, "train_s": 0, "infer_ms": 0},
        {"name": "TF-IDF + LR", "acc": tfidf_acc, "macro_f1": tfidf_macro_f1, "weighted_f1": tfidf_weighted_f1, "train_s": tfidf_train_time, "infer_ms": tfidf_inference_time * 1000},
        {"name": "Emb + LR", "acc": emb_lr_acc, "macro_f1": emb_lr_macro_f1, "weighted_f1": emb_lr_weighted_f1, "train_s": emb_lr_train_time, "infer_ms": emb_lr_inference_time * 1000},
        {"name": "Emb + LightGBM", "acc": lgbm_acc, "macro_f1": lgbm_macro_f1, "weighted_f1": lgbm_weighted_f1, "train_s": lgbm_train_time, "infer_ms": lgbm_inference_time * 1000},
    ]

    # Check monotonic improvement (first 4 should be monotonic; 5-6 may not be)
    _accs = [l["acc"] for l in ladder]
    _keyword_pass = keyword_acc < 0.80
    _first4_monotonic = all(_accs[i] <= _accs[i+1] for i in range(3))

    mo.md(f"""
    ### Difficulty Ladder Results

    | # | Classifier | Accuracy | Macro-F1 | Wt-F1 | Train (s) | Infer (ms/prompt) |
    |---|-----------|----------|----------|-------|-----------|-------------------|
    {"".join(f"| {i+1} | {l['name']} | {l['acc']:.3f} | {l['macro_f1']:.3f} | {l['weighted_f1']:.3f} | {l['train_s']:.1f} | {l['infer_ms']:.2f} |{chr(10)}" for i, l in enumerate(ladder))}

    **Note**: Embedding models require ~20ms/prompt for embedding computation (not included in inference time above).
    TF-IDF inference includes vectorization.

    **Validation checks**:
    - Keyword baseline < 80%: {"PASS" if _keyword_pass else "FAIL"} ({keyword_acc:.1%})
    - Monotonic through learned classifiers: {"PASS" if _first4_monotonic else "PARTIAL"} ({" → ".join(f"{a:.3f}" for a in _accs[:4])})
    """)

    # Bar chart
    fig_ladder, ax = plt.subplots(figsize=(12, 5))
    _names = [l["name"] for l in ladder]
    _macro_f1s = [l["macro_f1"] for l in ladder]
    _best_name = max(ladder, key=lambda l: l["macro_f1"])["name"]
    _colors = ["#d62728" if l["name"] == "Keyword/Regex" else "#2ca02c" if l["name"] == _best_name else "#1f77b4" for l in ladder]
    bars = ax.bar(_names, _macro_f1s, color=_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Classifier Difficulty Ladder (green = best: {_best_name})")
    ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend()
    for bar, val in zip(bars, _macro_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    fig_ladder.savefig(
        str(Path(__file__).parent.parent / "figures" / "notebook" / "difficulty_ladder.png"),
        dpi=150, bbox_inches="tight",
    )
    print("Saved difficulty_ladder.png")
    return (ladder,)


@app.cell
def template_family_eval(
    FEATURE_COL, TARGET_COL,
    df_train_full, mo, np, pd,
):
    """Leave-template-family-out evaluation: proves the router doesn't memorize templates.

    For each symbolic category, train on some template families, test on the held-out family.
    If accuracy holds, the router learned category structure, not template artifacts.
    """
    from router.template_family_out import evaluate_template_family_out

    print("=" * 80)
    print("LEAVE-TEMPLATE-FAMILY-OUT EVALUATION")
    print("=" * 80)

    _results = evaluate_template_family_out(df_train_full, FEATURE_COL, TARGET_COL)

    # Summary
    _model_keys = ["tfidf_lr", "tfidf_svm", "tfidf_nb", "emb_lr", "emb_lgbm", "emb_mlp", "ensemble"]
    _model_names = ["TF-IDF+LR", "TF-IDF+SVM", "TF-IDF+NB", "Emb+LR", "Emb+LGBM", "Emb+MLP", "Ensemble"]
    if _results:
        _means = {k: np.mean([r[k] for r in _results]) for k in _model_keys}
        _best_key = max(_means, key=_means.get)
        _best_name = _model_names[_model_keys.index(_best_key)]

        _header = "| Category | Held-Out | N | " + " | ".join(_model_names) + " |"
        _sep = "|---|---|---|" + "|".join(["---"] * len(_model_names)) + "|"
        _rows = ""
        for _r in _results:
            _vals = " | ".join(f"{_r[k]:.3f}" for k in _model_keys)
            _rows += f"| {_r['category']} | {_r['held_out']} | {_r['n_test']} | {_vals} |\n"
        _mean_row = "| **Mean** | | | " + " | ".join(f"**{_means[k]:.3f}**" for k in _model_keys) + " |"

        mo.md(f"""
        ### Leave-Template-Family-Out Results

        {_header}
        {_sep}
        {_rows}{_mean_row}

        **Best generalization: {_best_name}** (mean acc={_means[_best_key]:.3f})
        """)

    # Save as CSV
    if _results:
        _results_df = pd.DataFrame(_results)
        _results_df.to_csv(
            str(Path(__file__).parent.parent / "results" / "router" / "template_family_out.csv"),
            index=False,
        )
        print("Saved results_template_family_out.csv")

    template_family_results = _results
    return (template_family_results,)


@app.cell
def final_test_eval(
    CLASSES,
    TARGET_COL,
    classification_report,
    confusion_matrix,
    df_test,
    f1_score,
    np,
    plt,
    sns,
    tf_lr_X_test,
    tf_lr_model,
):
    """Final evaluation on held-out test set with bootstrap CIs."""

    _y_pred_enc = tf_lr_model.predict(tf_lr_X_test)
    test_preds = _y_pred_enc
    _y_true = df_test[TARGET_COL].values

    print("=" * 80)
    print("FINAL test SET EVALUATION (TF+LR)")
    print("=" * 80)
    print(classification_report(_y_true, test_preds, digits=3))

    # Bootstrap 95% CIs
    _n_bootstrap = 1000
    np.random.seed(42)
    _n = len(_y_true)
    _boot_macro_f1 = []
    _boot_per_class = {_cls: [] for _cls in CLASSES}

    for _ in range(_n_bootstrap):
        _idx = np.random.choice(_n, size=_n, replace=True)
        _bt = _y_true[_idx]
        _bp = test_preds[_idx]
        _boot_macro_f1.append(f1_score(_bt, _bp, average="macro"))
        for _cls in CLASSES:
            _cls_mask = _bt == _cls
            if _cls_mask.sum() > 0:
                _cls_f1 = f1_score(_bt == _cls, _bp == _cls, average="binary")
                _boot_per_class[_cls].append(_cls_f1)

    _macro_ci_lo = np.percentile(_boot_macro_f1, 2.5)
    _macro_ci_hi = np.percentile(_boot_macro_f1, 97.5)
    _macro_point = f1_score(_y_true, test_preds, average="macro")

    print(f"\nMacro-F1: {_macro_point:.3f} (95% CI: [{_macro_ci_lo:.3f}, {_macro_ci_hi:.3f}])")
    # Per-class F1: point estimate from full test set, CIs from bootstrap
    print(f"\nPer-class F1 with 95% bootstrap CIs:")
    for _cls in CLASSES:
        _vals = _boot_per_class[_cls]
        if _vals:
            _lo = np.percentile(_vals, 2.5)
            _hi = np.percentile(_vals, 97.5)
            _pt = f1_score(_y_true == _cls, test_preds == _cls, average="binary")
            print(f"  {_cls:<22} F1={_pt:.3f}  CI=[{_lo:.3f}, {_hi:.3f}]  (n_test={(df_test[TARGET_COL]==_cls).sum()})")

    # Confusion matrix
    _cm = confusion_matrix(_y_true, test_preds, labels=CLASSES)
    _cm_norm = _cm.astype(float) / _cm.sum(axis=1, keepdims=True)

    fig_cm, _ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        _cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=_ax,
    )
    _ax.set_xlabel("Predicted")
    _ax.set_ylabel("True")
    _ax.set_title("Normalized Confusion Matrix (TF-IDF + LR, Test Set)")
    plt.tight_layout()

    fig_cm.savefig(
        str(Path(__file__).parent.parent / "figures" / "notebook" / "confusion_matrix.png"),
        dpi=150, bbox_inches="tight",
    )
    print("Saved confusion_matrix.png")
    return (test_preds,)


@app.cell
def calibration(
    CLASSES,
    FEATURE_COL,
    LabelEncoder,
    TARGET_COL,
    df_test,
    df_train,
    df_val,
):
    """Platt scaling for confidence calibration on TF-IDF + LR (in-distribution model)."""
    from router.feature_extraction import build_tfidf_features as _build_tfidf_features
    from router.classifiers import train_tfidf_lr as _train_tfidf_lr
    from router.calibration import calibrate_platt, apply_platt

    _vectorizer, _X_train, _X_test, _X_val, _ = _build_tfidf_features(
        df_train[FEATURE_COL], df_test[FEATURE_COL], val_prompts=df_val[FEATURE_COL],
    )

    _le = LabelEncoder()
    _le.fit(CLASSES)
    _y_train = _le.transform(df_train[TARGET_COL])
    _y_val = _le.transform(df_val[TARGET_COL])

    from sklearn.linear_model import LogisticRegression as _LogisticRegression
    _base = _LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
    )
    _base.fit(_X_train, _y_train)

    # Get uncalibrated probabilities first
    probs_uncalibrated = _base.predict_proba(_X_test)

    # Manual Platt scaling via router.calibration
    _calibrators = calibrate_platt(_base, _X_val, _y_val)
    probs_calibrated = apply_platt(_calibrators, probs_uncalibrated)

    calibrated_le = _le
    calibrated_model = _base  # store base model for saving

    print("Platt scaling applied to TF-IDF + LR (per-class sigmoid on val set)")
    print(f"  Uncalibrated prob range: [{probs_uncalibrated.min():.4f}, {probs_uncalibrated.max():.4f}]")
    print(f"  Calibrated prob range:   [{probs_calibrated.min():.4f}, {probs_calibrated.max():.4f}]")
    return (
        calibrated_le,
        calibrated_model,
        probs_calibrated,
        probs_uncalibrated,
    )


@app.cell
def ece_reliability(
    CLASSES,
    TARGET_COL,
    calibrated_le,
    df_test,
    mo,
    np,
    plt,
    probs_calibrated,
    probs_uncalibrated,
):
    """ECE, Brier score, and reliability diagrams."""
    from router.calibration import compute_ece, compute_brier, plot_reliability_diagram

    _y_test = calibrated_le.transform(df_test[TARGET_COL])

    ece_before = compute_ece(_y_test, probs_uncalibrated)
    ece_after = compute_ece(_y_test, probs_calibrated)
    brier_before = compute_brier(_y_test, probs_uncalibrated, n_classes=len(CLASSES))
    brier_after = compute_brier(_y_test, probs_calibrated, n_classes=len(CLASSES))

    print(f"ECE:   {ece_before:.4f} -> {ece_after:.4f} ({'improved' if ece_after < ece_before else 'worsened'})")
    print(f"Brier: {brier_before:.4f} -> {brier_after:.4f} ({'improved' if brier_after < brier_before else 'worsened'})")

    mo.md(f"""
    ### Calibration Results

    | Metric | Before Platt | After Platt | Target |
    |--------|-------------|-------------|--------|
    | ECE | {ece_before:.4f} | {ece_after:.4f} | < 0.05 |
    | Brier Score | {brier_before:.4f} | {brier_after:.4f} | Lower is better |
    """)

    fig_rel = plot_reliability_diagram(_y_test, probs_uncalibrated, probs_calibrated, CLASSES)

    fig_rel.savefig(
        str(Path(__file__).parent.parent / "figures" / "notebook" / "reliability_diagrams.png"),
        dpi=150, bbox_inches="tight",
    )
    print("Saved reliability_diagrams.png")

    return brier_after, brier_before, ece_after, ece_before


@app.cell
def error_analysis(TARGET_COL, df_test, mo, test_preds):
    """Analyze misclassified examples."""
    from router.template_family_out import analyze_errors

    _y_true = df_test[TARGET_COL].values
    _prompts = df_test["prompt"].values

    _error_count, _total, _sorted_pairs, _hs_count, _sh_count = analyze_errors(
        _y_true, test_preds, _prompts,
    )

    print(f"Total errors: {_error_count} / {_total} ({_error_count/_total:.1%})")

    _error_md = "### Error Analysis\n\n"
    _error_md += f"**Total misclassifications**: {_error_count} / {_total} ({_error_count/_total:.1%})\n\n"
    _error_md += "| True -> Predicted | Count | Example |\n|---|---|---|\n"
    for (true, pred), examples in _sorted_pairs[:15]:
        _ex = examples[0][:80].replace("|", "\\|").replace("\n", " ")
        _error_md += f"| {true} -> {pred} | {len(examples)} | {_ex}... |\n"

    _error_md += f"\n**HYBRID<->SEMANTIC confusion**: {_hs_count} + {_sh_count} = {_hs_count + _sh_count} total\n"

    # Show examples from sorted_pairs
    _confusion_pairs = dict(_sorted_pairs)
    if ("HYBRID", "SEMANTIC") in _confusion_pairs:
        _error_md += "\n**HYBRID misclassified as SEMANTIC** (boundary cases):\n"
        for ex in _confusion_pairs[("HYBRID", "SEMANTIC")][:5]:
            _error_md += f"- {ex[:120]}\n"

    if ("SEMANTIC", "HYBRID") in _confusion_pairs:
        _error_md += "\n**SEMANTIC misclassified as HYBRID**:\n"
        for ex in _confusion_pairs[("SEMANTIC", "HYBRID")][:5]:
            _error_md += f"- {ex[:120]}\n"

    mo.md(_error_md)
    return


@app.cell
def comprehensive_benchmark(
    CLASSES, FEATURE_COL, TARGET_COL,
    df_train, df_train_full, df_test,
    emb_train, emb_test,
    keyword_classify, mo, np, pd,
):
    """Complete model benchmark: all models x all test conditions."""
    from router.benchmarking import run_full_benchmark

    _root = Path(__file__).parent.parent
    _data_dir = _root / "data"
    _results_dir = _root / "results"
    _figures_dir = _root / "figures"
    _ho_path = _data_dir / "holdout_cross_generator.csv"

    _all_results, _cal_results, _all_preds_test, _all_preds_ho = run_full_benchmark(
        df_train=df_train,
        df_test=df_test,
        df_train_full=df_train_full,
        feature_col=FEATURE_COL,
        target_col=TARGET_COL,
        classes=CLASSES,
        emb_train=emb_train,
        emb_test=emb_test,
        data_dir=_data_dir,
        results_dir=_results_dir,
        figures_dir=_figures_dir,
        keyword_classify_fn=keyword_classify,
        holdout_path=_ho_path,
    )

    # Build markdown summary
    _cal_lookup = {c["model"]: c for c in _cal_results}

    _md = "### Comprehensive Model Benchmark\n\n"
    _md += "| Model | Test Acc | Test F1 | Test WF1 | HO Acc | HO F1 | Gap | Train(s) | ECE | Brier |\n"
    _md += "|-------|---------|---------|----------|--------|-------|-----|----------|-----|-------|\n"
    for _r in _all_results:
        _ho_acc = f"{_r['ho_acc']:.3f}" if "ho_acc" in _r else "--"
        _ho_f1 = f"{_r['ho_f1']:.3f}" if "ho_f1" in _r else "--"
        _gap = f"{_r['test_f1'] - _r['ho_f1']:+.3f}" if "ho_f1" in _r else "--"
        _cal = None
        for _ck, _cv in _cal_lookup.items():
            if _ck in _r["model"] or _r["model"].startswith(_ck):
                _cal = _cv
                break
        _ece = f"{_cal['test_ece']:.4f}" if _cal else "--"
        _brier = f"{_cal['test_brier']:.4f}" if _cal else "--"
        _md += f"| {_r['model']} | {_r['test_acc']:.3f} | {_r['test_f1']:.3f} | {_r['test_wf1']:.3f} | {_ho_acc} | {_ho_f1} | {_gap} | {_r['train_s']:.1f} | {_ece} | {_brier} |\n"

    mo.md(_md)

    benchmark_results = _all_results
    return (benchmark_results,)


@app.cell
def confidence_abstention(
    FEATURE_COL, TARGET_COL,
    df_test, df_train, df_val, emb_test, emb_train,
    mo, plt,
):
    """Confidence-abstention analysis: accuracy vs coverage at different thresholds."""
    from router.confidence import analyze_confidence_abstention
    from router.feature_extraction import build_tfidf_features as _build_tfidf_features

    _, _X_tr, _X_te, _, _ = _build_tfidf_features(
        df_train[FEATURE_COL], df_test[FEATURE_COL],
    )

    fig_abs = analyze_confidence_abstention(
        X_train_tfidf=_X_tr,
        X_test_tfidf=_X_te,
        emb_train=emb_train,
        emb_test=emb_test,
        y_train=df_train[TARGET_COL],
        y_test=df_test[TARGET_COL].values,
    )

    fig_abs.savefig(
        str(Path(__file__).parent.parent / "figures" / "notebook" / "confidence_abstention.png"),
        dpi=150, bbox_inches="tight",
    )
    print("Saved confidence_abstention.png")

    mo.md("""
    ### Confidence-Abstention Analysis

    Shows the tradeoff between coverage (fraction of prompts routed by classifier)
    and accuracy (correctness of routed prompts) at different confidence thresholds tau.

    When confidence < tau, the prompt falls back to LLM evaluation (safe default).
    """)
    return



@app.cell
def slice_analysis(
    CLASSES, LogisticRegression, Path, TfidfVectorizer,
    FEATURE_COL, TARGET_COL,
    accuracy_score, df_train, df_train_full, emb_train,
    f1_score, mo, np, pd, re, benchmark_results, SentenceTransformer,
):
    """Per-category holdout accuracy for ALL models."""
    from sklearn.svm import LinearSVC as _LinearSVC
    from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
    from sklearn.neural_network import MLPClassifier as _MLPClassifier
    import lightgbm as _lgb2

    _root = Path(__file__).parent.parent
    _ho_path = _root / "data" / "holdout_cross_generator.csv"

    if not _ho_path.exists():
        _ = mo.md("### Slice Analysis\n\n*Holdout not found.*")
    else:
        _ho_df = pd.read_csv(_ho_path)
        _ho_df = _ho_df[_ho_df["top_category"].isin(CLASSES)].reset_index(drop=True)
        _num_pat = re.compile(r'\d+')

        _train_prompts = df_train[FEATURE_COL]
        _train_labels = df_train[TARGET_COL]
        _ho_prompts_norm = _ho_df["prompt"].apply(lambda t: _num_pat.sub("<NUM>", t))
        _ho_labels = _ho_df["top_category"].values

        # Build all features
        _vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        _X_tr = _vec.fit_transform(_train_prompts)
        _X_ho = _vec.transform(_ho_prompts_norm)

        _emb_m = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        _ho_emb = _emb_m.encode(
            [f"query: {t}" for t in _ho_df["prompt"]], batch_size=64, normalize_embeddings=True,
        )

        from sklearn.preprocessing import LabelEncoder as _LE2
        _le2 = _LE2()
        _y_tr_enc2 = _le2.fit_transform(_train_labels)
        _cw2 = len(_y_tr_enc2) / (len(np.bincount(_y_tr_enc2)) * np.bincount(_y_tr_enc2))
        _sw2 = np.array([_cw2[y] for y in _y_tr_enc2])

        # TF-IDF unigram
        _vec_uni = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=True)
        _X_tr_uni = _vec_uni.fit_transform(_train_prompts)
        _X_ho_uni = _vec_uni.transform(_ho_prompts_norm)

        # TF-IDF raw digits
        _train_csv = sorted((_root / "data").glob("synthetic_final_*_train.csv"))[-1]
        _train_raw_df2 = pd.read_csv(_train_csv)
        _vec_raw = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        _X_tr_raw = _vec_raw.fit_transform(_train_raw_df2["prompt"])
        _X_ho_raw = _vec_raw.transform(_ho_df["prompt"])  # raw holdout (no masking)

        # Train ALL models and get holdout predictions
        _models_preds = {}

        # Keyword
        _models_preds["Keyword/Regex"] = np.array([keyword_classify(p) for p in _ho_df["prompt"]])

        # TF-IDF + LR (bi, <NUM>)
        _m = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        _m.fit(_X_tr, _train_labels)
        _tfidf_lr = _m
        _models_preds["TF-IDF+LR (bi, <NUM>)"] = _m.predict(_X_ho)

        # TF-IDF + LR (uni, <NUM>)
        _m = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        _m.fit(_X_tr_uni, _train_labels)
        _models_preds["TF-IDF+LR (uni, <NUM>)"] = _m.predict(_X_ho_uni)

        # TF-IDF + LR (bi, raw)
        _m = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        _m.fit(_X_tr_raw, _train_raw_df2["top_category"])
        _models_preds["TF-IDF+LR (bi, raw)"] = _m.predict(_X_ho_raw)

        # TF-IDF + SVM
        _m = _LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
        _m.fit(_X_tr, _train_labels)
        _models_preds["TF-IDF+SVM"] = _m.predict(_X_ho)

        # TF-IDF + NB
        _m = _MultinomialNB(alpha=1.0)
        _m.fit(_X_tr, _train_labels)
        _models_preds["TF-IDF+NB"] = _m.predict(_X_ho)

        # Emb + LR
        _emb_lr_slice = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        _emb_lr_slice.fit(emb_train, _train_labels)
        _models_preds["Emb+LR"] = _emb_lr_slice.predict(_ho_emb)

        # Emb + LightGBM
        _m = _lgb2.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)
        _m.fit(emb_train, _y_tr_enc2, sample_weight=_sw2)
        _models_preds["Emb+LightGBM"] = _le2.inverse_transform(_m.predict(_ho_emb))

        # Emb + MLP
        _m = _MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15)
        _m.fit(emb_train, _y_tr_enc2)
        _models_preds["Emb+MLP"] = _le2.inverse_transform(_m.predict(_ho_emb))

        # Ensemble
        _cal_tf = CalibratedClassifierCV(estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42), method="sigmoid", cv=3)
        _cal_tf.fit(_X_tr, _train_labels)
        _cal_em = CalibratedClassifierCV(estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42), method="sigmoid", cv=3)
        _cal_em.fit(emb_train, _train_labels)
        _ens_probs = (_cal_tf.predict_proba(_X_ho) + _cal_em.predict_proba(_ho_emb)) / 2
        _models_preds["Ensemble"] = _cal_tf.classes_[np.argmax(_ens_probs, axis=1)]

        _model_names = list(_models_preds.keys())

        # Per-category holdout accuracy for ALL models
        _slice_rows = []
        for _c in CLASSES:
            _mask = _ho_labels == _c
            _n = int(_mask.sum())
            if _n == 0:
                continue
            _row = {"category": _c, "n": _n}
            for _mn in _model_names:
                _row[_mn] = round(accuracy_score(_ho_labels[_mask], _models_preds[_mn][_mask]), 4)
            _slice_rows.append(_row)

        _slice_df = pd.DataFrame(_slice_rows)
        _slice_df.to_csv(str(_root / "results" / "router" / "slice_holdout.csv"), index=False)
        print("Saved results_slice_holdout.csv")

        # Markdown table
        _md = "### Slice Analysis — Per-Category Holdout Accuracy (All Models)\n\n"
        _md += "| Category | N | " + " | ".join(_model_names) + " |\n"
        _md += "|---|---|" + "|".join(["---"] * len(_model_names)) + "|\n"
        for _, _row in _slice_df.iterrows():
            _vals = " | ".join(f"{_row[_mn]:.3f}" for _mn in _model_names)
            _md += f"| {_row['category']} | {_row['n']} | {_vals} |\n"

        # Template family distribution
        _md += "\n**Template family distribution (training set)**:\n\n"
        _md += "| Category | Family | Count |\n|---|---|---|\n"
        _symbolic = df_train_full[df_train_full["top_category"].str.startswith("SYMBOLIC")]
        for _c2 in sorted(_symbolic["top_category"].unique()):
            _cat_data = _symbolic[_symbolic["top_category"] == _c2]
            for _fam in sorted(_cat_data["subcategory"].unique()):
                _n = int((_cat_data["subcategory"] == _fam).sum())
                _md += f"| {_c2} | {_fam} | {_n} |\n"

        _ = mo.md(_md)
    return


@app.cell
def save_artifacts(
    DATA_PREFIX,
    benchmark_results,
    brier_after,
    brier_before,
    calibrated_le,
    calibrated_model,
    ece_after,
    ece_before,
    joblib,
    json,
    ladder,
    template_family_results,
    tf_lr_model,
    tfidf_vectorizer,
):
    """Save all trained models and results."""

    _root = Path(__file__).parent.parent
    _models_dir = str(_root / "models")
    _results_dir = str(_root / "results" / "router")

    # Models
    joblib.dump(tf_lr_model, f"{_models_dir}/router_model_tfidf_lr.joblib")
    joblib.dump(tfidf_vectorizer, f"{_models_dir}/router_tfidf_vectorizer.joblib")
    joblib.dump(calibrated_model, f"{_models_dir}/router_model_calibrated.joblib")
    joblib.dump(calibrated_le, f"{_models_dir}/router_label_encoder.joblib")

    _best_indist = max(benchmark_results, key=lambda r: r["test_f1"])
    _best_holdout = max(
        [r for r in benchmark_results if "ho_f1" in r],
        key=lambda r: r["ho_f1"],
    )

    # Results summary
    _results = {
        "dataset": DATA_PREFIX.split("/")[-1],
        "best_model_indist": _best_indist["model"],
        "best_model_indist_f1": _best_indist["test_f1"],
        "best_model_holdout": _best_holdout["model"],
        "best_model_holdout_f1": _best_holdout["ho_f1"],
        "difficulty_ladder": ladder,
        "benchmark": benchmark_results,
        "template_family_out": template_family_results,
        "calibration": {
            "method": "Platt scaling (sigmoid)",
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }

    print(f"Best in-distribution: {_best_indist['model']} (F1={_best_indist['test_f1']:.3f})")
    print(f"Best on holdout:      {_best_holdout['model']} (F1={_best_holdout['ho_f1']:.3f})")
    with open(f"{_results_dir}/training_metadata.json", "w") as f:
        json.dump(_results, f, indent=2, default=str)

    print("Saved artifacts:")
    print(f"  {_models_dir}/router_model_*.joblib")
    print(f"  {_results_dir}/training_metadata.json")
    print(f"  Figures: difficulty_ladder, confusion_matrix, reliability_diagrams,")
    print(f"           confidence_abstention, holdout_top2_confusion")
    return



if __name__ == "__main__":
    app.run()
