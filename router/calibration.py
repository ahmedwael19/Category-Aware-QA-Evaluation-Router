"""Confidence calibration: Platt scaling, ECE, Brier score, reliability diagrams.

"""

import numpy as np
from sklearn.calibration import _SigmoidCalibration, calibration_curve
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def calibrate_platt(base_model, X_val, y_val):
    """Apply per-class Platt (sigmoid) scaling on validation set probabilities.

    Parameters
    ----------
    base_model : classifier with ``predict_proba``
        Already-trained model (e.g. LogisticRegression).
    X_val : sparse matrix or ndarray
        Validation features.
    y_val : ndarray of int
        Encoded validation labels (integer class indices).

    Returns
    -------
    calibrators : list of _SigmoidCalibration
        One per class, fitted on validation probabilities.
    """
    val_probs = base_model.predict_proba(X_val)
    calibrators = []
    for ci in range(val_probs.shape[1]):
        y_bin = (y_val == ci).astype(float)
        sig = _SigmoidCalibration()
        sig.fit(val_probs[:, ci], y_bin)
        calibrators.append(sig)
    return calibrators


def apply_platt(calibrators, raw_probs):
    """Apply fitted Platt calibrators to raw probability matrix.

    Returns normalised (row-sum = 1) calibrated probabilities.
    """
    cal = raw_probs.copy()
    for ci, sig in enumerate(calibrators):
        cal[:, ci] = sig.predict(raw_probs[:, ci])
    # Normalise
    cal = cal / cal.sum(axis=1, keepdims=True)
    return cal


# ---------------------------------------------------------------------------
# ECE and Brier
# ---------------------------------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error (multi-class, confidence-based).

    Parameters
    ----------
    y_true : ndarray of int
        True class indices.
    y_prob : ndarray, shape (n, n_classes)
        Predicted probability matrix.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
    """
    confs = np.max(y_prob, axis=1)
    preds = np.argmax(y_prob, axis=1)
    correct = (preds == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def compute_ece_from_strings(probs, y_true_str, classes, n_bins=15):
    """ECE variant that accepts string labels and a class list."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true_str])
    return compute_ece(y_idx, probs, n_bins=n_bins)


def compute_brier(y_true, y_prob, n_classes=None):
    """Multi-class Brier score (one-vs-rest average).

    Parameters
    ----------
    y_true : ndarray of int
        True class indices.
    y_prob : ndarray, shape (n, n_classes)
        Predicted probability matrix.
    n_classes : int, optional
        Inferred from y_prob if not given.

    Returns
    -------
    float
    """
    if n_classes is None:
        n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean([
        brier_score_loss(y_onehot[:, i], y_prob[:, i])
        for i in range(n_classes)
    ])


def compute_brier_from_strings(probs, y_true_str, classes):
    """Brier variant that accepts string labels and a class list."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true_str])
    return compute_brier(y_idx, probs, n_classes=len(classes))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(y_test_encoded, probs_uncalibrated, probs_calibrated, classes, n_bins=10):
    """Per-class reliability diagrams (before vs after Platt scaling).

    Returns matplotlib figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        y_bin = (y_test_encoded == i).astype(int)

        # Before calibration
        frac_pos_b, mean_pred_b = calibration_curve(y_bin, probs_uncalibrated[:, i], n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred_b, frac_pos_b, "s-", color="red", alpha=0.7, label="Before")

        # After calibration
        frac_pos_a, mean_pred_a = calibration_curve(y_bin, probs_calibrated[:, i], n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred_a, frac_pos_a, "o-", color="blue", alpha=0.7, label="After")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Reliability Diagrams (Before vs After Platt Scaling)", fontsize=12)
    plt.tight_layout()
    return fig
