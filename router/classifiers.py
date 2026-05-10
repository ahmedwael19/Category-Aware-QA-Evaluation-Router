"""Classifier training functions for the prompt router.

Each function takes feature matrices and labels, returns a trained model.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# TF-IDF classifiers
# ---------------------------------------------------------------------------

def train_tfidf_lr(X_train, y_train):
    """TF-IDF + Logistic Regression (balanced, bigram, C=1.0)."""
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_tfidf_svm(X_train, y_train):
    """TF-IDF + LinearSVC (balanced, C=1.0)."""
    model = LinearSVC(
        class_weight="balanced", max_iter=2000, C=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_tfidf_nb(X_train, y_train):
    """TF-IDF + Multinomial Naive Bayes (alpha=1.0)."""
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Embedding classifiers
# ---------------------------------------------------------------------------

def train_emb_lr(X_train, y_train):
    """Embedding + Logistic Regression (balanced, C=1.0)."""
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_emb_lgbm(X_train, y_train):
    """Embedding + LightGBM (200 trees, balanced via sample weights).

    Parameters
    ----------
    X_train : ndarray
        Embedding features.
    y_train : array-like of str
        String category labels.

    Returns
    -------
    model : LGBMClassifier
        Trained model (predicts encoded ints).
    le : LabelEncoder
        Fitted label encoder for inverse_transform.
    """
    import lightgbm as lgb

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    class_counts = np.bincount(y_enc)
    n_samples = len(y_enc)
    n_classes = len(class_counts)
    class_weights = n_samples / (n_classes * class_counts)
    sample_weights = np.array([class_weights[y] for y in y_enc])

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_enc, sample_weight=sample_weights)
    return model, le


def train_emb_mlp(X_train, y_train):
    """Embedding + MLP (256-64 hidden, early stopping).

    Parameters
    ----------
    X_train : ndarray
        Embedding features.
    y_train : array-like of str
        String category labels.

    Returns
    -------
    model : MLPClassifier
        Trained model (predicts encoded ints).
    le : LabelEncoder
        Fitted label encoder for inverse_transform.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    model.fit(X_train, y_enc)
    return model, le


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def train_ensemble(X_train_tfidf, X_train_emb, y_train):
    """Calibrated ensemble: averaged probabilities of TF-IDF+LR and Emb+LR.

    Returns
    -------
    cal_tfidf : CalibratedClassifierCV
    cal_emb : CalibratedClassifierCV
    """
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
    cal_emb.fit(X_train_emb, y_train)

    return cal_tfidf, cal_emb


def ensemble_predict(cal_tfidf, cal_emb, X_tfidf, X_emb):
    """Predict using averaged calibrated probabilities.

    Returns
    -------
    preds : ndarray of str
        Predicted class labels.
    probs : ndarray
        Averaged probability matrix.
    """
    probs = (cal_tfidf.predict_proba(X_tfidf) + cal_emb.predict_proba(X_emb)) / 2
    preds = cal_tfidf.classes_[np.argmax(probs, axis=1)]
    return preds, probs
