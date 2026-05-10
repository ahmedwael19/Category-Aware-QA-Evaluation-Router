"""Confidence-abstention analysis: accuracy vs coverage at different tau.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def analyze_confidence_abstention(
    X_train_tfidf, X_test_tfidf,
    emb_train, emb_test,
    y_train, y_test,
    thresholds=None,
):
    """Confidence-abstention analysis with tau sweep for TF-IDF+LR and Emb+LR.

    Parameters
    ----------
    X_train_tfidf, X_test_tfidf : sparse matrix
        TF-IDF feature matrices.
    emb_train, emb_test : ndarray
        Embedding feature matrices.
    y_train : array-like of str
        Training labels.
    y_test : array-like of str
        Test labels.
    thresholds : ndarray, optional
        Confidence thresholds to sweep. Defaults to np.arange(0.3, 1.0, 0.05).

    Returns
    -------
    fig : matplotlib.Figure
        Two-panel figure (TF-IDF+LR, Emb+LR).
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 1.0, 0.05)

    y_true = np.asarray(y_test)

    # Calibrate both models
    cal_tfidf = CalibratedClassifierCV(
        estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
        ),
        method="sigmoid",
        cv=3,
    )
    cal_tfidf.fit(X_train_tfidf, y_train)

    cal_emb = CalibratedClassifierCV(
        estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
        ),
        method="sigmoid",
        cv=3,
    )
    cal_emb.fit(emb_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, cal, X) in enumerate([
        ("TF-IDF + LR", cal_tfidf, X_test_tfidf),
        ("Embedding + LR", cal_emb, emb_test),
    ]):
        probs = cal.predict_proba(X)
        max_conf = np.max(probs, axis=1)
        preds = cal.classes_[np.argmax(probs, axis=1)]

        coverages = []
        accuracies = []
        f1s = []
        for tau in thresholds:
            mask = max_conf >= tau
            cov = mask.mean()
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], preds[mask])
                f = f1_score(y_true[mask], preds[mask], average="macro", zero_division=0)
            else:
                acc = 0
                f = 0
            coverages.append(cov)
            accuracies.append(acc)
            f1s.append(f)

        ax = axes[idx]
        ax.plot(thresholds, coverages, "b-o", markersize=4, label="Coverage")
        ax.plot(thresholds, accuracies, "g-s", markersize=4, label="Accuracy")
        ax.plot(thresholds, f1s, "r-^", markersize=4, label="Macro-F1")
        ax.set_xlabel("Confidence threshold tau")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{name}: Confidence-Abstention")
        ax.legend()
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
