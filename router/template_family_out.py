"""Evaluation helpers: leave-template-family-out + error analysis."""

import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from config import EMBEDDING_MODEL, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from router.classifiers import (
    ensemble_predict,
    train_emb_lr,
    train_emb_mlp,
    train_ensemble,
    train_tfidf_lr,
    train_tfidf_nb,
    train_tfidf_svm,
)

_NUM_PAT = re.compile(r"\d+")
_SYMBOLIC_CATS = ("SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA")


def _get_family(subcat):
    for fam in (
        "explicit", "moderate", "implicit", "gt", "lt", "eq",
        "meta_tag", "meta_channel", "meta_notes",
    ):
        if fam in str(subcat):
            return fam
    return "other"


def _tfidf_features(train_prompts, test_prompts):
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(train_prompts)
    X_te = vec.transform(test_prompts)
    return X_tr, X_te


def _embed(emb_model, prompts):
    return emb_model.encode(
        [f"query: {t}" for t in prompts],
        batch_size=64, normalize_embeddings=True,
    )


def evaluate_template_family_out(df_train_full, feature_col, target_col):
    """Leave-template-family-out evaluation for symbolic categories.

    For each symbolic category, holds out one template family at a time, trains
    every classifier variant on the remaining families plus all non-symbolic
    data, and reports held-out accuracy.

    Returns a list of per-fold dicts with one entry per classifier.
    """
    symbolic = df_train_full[df_train_full[target_col].isin(_SYMBOLIC_CATS)].copy()
    symbolic["family"] = symbolic["subcategory"].apply(_get_family)
    non_symbolic = df_train_full[~df_train_full[target_col].isin(_SYMBOLIC_CATS)].copy()

    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    results = []

    try:
        import lightgbm  # noqa: F401
        from router.classifiers import train_emb_lgbm
        _HAS_LGBM = True
    except (ImportError, OSError) as e:
        print(f"  (LightGBM unavailable: {e}; emb_lgbm will be null)")
        _HAS_LGBM = False

    for cat in _SYMBOLIC_CATS:
        cat_data = symbolic[symbolic[target_col] == cat]
        families = sorted(cat_data["family"].unique())
        if len(families) < 2:
            print(f"\n  {cat}: only {len(families)} family, skipping")
            continue
        print(f"\n  {cat} — families: {families}")

        for held_out_fam in families:
            test_set = cat_data[cat_data["family"] == held_out_fam]
            train_symbolic = cat_data[cat_data["family"] != held_out_fam]
            if len(test_set) < 5:
                continue

            other_symbolic = symbolic[symbolic[target_col] != cat]
            train_all = pd.concat([non_symbolic, other_symbolic, train_symbolic])

            train_prompts_raw = train_all["prompt"]
            test_prompts_raw = test_set["prompt"]
            train_prompts = train_prompts_raw.apply(lambda t: _NUM_PAT.sub("<NUM>", t))
            test_prompts = test_prompts_raw.apply(lambda t: _NUM_PAT.sub("<NUM>", t))
            train_labels = train_all[target_col]
            test_labels = test_set[target_col]

            X_tr, X_te = _tfidf_features(train_prompts, test_prompts)
            emb_tr = _embed(emb_model, train_prompts_raw)
            emb_te = _embed(emb_model, test_prompts_raw)

            fold = {
                "category": cat,
                "held_out": held_out_fam,
                "n_test": len(test_set),
                "tfidf_lr":  round(accuracy_score(test_labels, train_tfidf_lr(X_tr, train_labels).predict(X_te)), 4),
                "tfidf_svm": round(accuracy_score(test_labels, train_tfidf_svm(X_tr, train_labels).predict(X_te)), 4),
                "tfidf_nb":  round(accuracy_score(test_labels, train_tfidf_nb(X_tr, train_labels).predict(X_te)), 4),
                "emb_lr":    round(accuracy_score(test_labels, train_emb_lr(emb_tr, train_labels).predict(emb_te)), 4),
            }

            if _HAS_LGBM:
                lgbm_model, lgbm_le = train_emb_lgbm(emb_tr, train_labels)
                lgbm_preds = lgbm_le.inverse_transform(lgbm_model.predict(emb_te))
                fold["emb_lgbm"] = round(accuracy_score(test_labels, lgbm_preds), 4)
            else:
                fold["emb_lgbm"] = None

            mlp_model, mlp_le = train_emb_mlp(emb_tr, train_labels)
            mlp_preds = mlp_le.inverse_transform(mlp_model.predict(emb_te))
            fold["emb_mlp"] = round(accuracy_score(test_labels, mlp_preds), 4)

            cal_tf, cal_em = train_ensemble(X_tr, emb_tr, train_labels)
            ens_preds, _ = ensemble_predict(cal_tf, cal_em, X_te, emb_te)
            fold["ensemble"] = round(accuracy_score(test_labels, ens_preds), 4)

            results.append(fold)
            lgbm_str = f"LGBM={fold['emb_lgbm']:.3f}" if fold["emb_lgbm"] is not None else "LGBM=n/a"
            print(
                f"    Hold out '{held_out_fam}' ({len(test_set):>3}): "
                f"TF-LR={fold['tfidf_lr']:.3f} SVM={fold['tfidf_svm']:.3f} "
                f"NB={fold['tfidf_nb']:.3f} Emb-LR={fold['emb_lr']:.3f} "
                f"{lgbm_str} MLP={fold['emb_mlp']:.3f} Ens={fold['ensemble']:.3f}"
            )

    return results


def analyze_errors(y_true, y_pred, prompts):
    """Group misclassified examples by (true, predicted) confusion pair."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prompts = np.asarray(prompts)
    errors = y_true != y_pred

    confusion_pairs = {}
    for i in range(len(y_true)):
        if errors[i]:
            key = (str(y_true[i]), str(y_pred[i]))
            confusion_pairs.setdefault(key, []).append(prompts[i])

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -len(x[1]))
    hs_count = len(confusion_pairs.get(("HYBRID", "SEMANTIC"), []))
    sh_count = len(confusion_pairs.get(("SEMANTIC", "HYBRID"), []))

    return int(errors.sum()), len(y_true), sorted_pairs, hs_count, sh_count
