"""Feature extraction: embeddings and TF-IDF vectorization.

"""

import re
import time as timer
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import EMBEDDING_MODEL, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

_NUM_PAT = re.compile(r"\d+")


def compute_and_cache_embeddings(df_train, df_val, df_test, feature_col, models_dir):
    """Compute or load cached sentence-transformer embeddings.

    Parameters
    ----------
    df_train, df_val, df_test : DataFrame
        Each must contain *feature_col*.
    feature_col : str
        Column name holding prompt text.
    models_dir : str | Path
        Directory under which ``embeddings/`` will be created.

    Returns
    -------
    emb_train, emb_val, emb_test : ndarray
        Normalised embedding matrices.
    """
    emb_dir = Path(models_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    train_path = emb_dir / "train.npy"
    val_path = emb_dir / "val.npy"
    test_path = emb_dir / "test.npy"

    # Try loading from cache
    if train_path.exists() and val_path.exists() and test_path.exists():
        emb_train = np.load(str(train_path))
        emb_val = np.load(str(val_path))
        emb_test = np.load(str(test_path))
        if (
            emb_train.shape[0] == len(df_train)
            and emb_val.shape[0] == len(df_val)
            and emb_test.shape[0] == len(df_test)
        ):
            print(f"Loaded cached embeddings from {emb_dir}/")
            print(f"  shapes: train={emb_train.shape}, val={emb_val.shape}, test={emb_test.shape}")
            return emb_train, emb_val, emb_test

    # Compute from scratch
    model = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(texts):
        # E5-instruct models require "query: " prefix
        prefixed = [f"query: {t}" for t in texts]
        return model.encode(prefixed, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

    t0 = timer.time()
    print("Embedding train set...")
    emb_train = _embed(df_train[feature_col].tolist())
    print("Embedding val set...")
    emb_val = _embed(df_val[feature_col].tolist())
    print("Embedding test set...")
    emb_test = _embed(df_test[feature_col].tolist())
    elapsed = timer.time() - t0

    print(f"\nEmbedding shapes: train={emb_train.shape}, val={emb_val.shape}, test={emb_test.shape}")
    total_prompts = len(df_train) + len(df_val) + len(df_test)
    print(f"Total time: {elapsed:.1f}s ({elapsed / total_prompts * 1000:.1f}ms per prompt)")

    # Save
    np.save(str(train_path), emb_train)
    np.save(str(val_path), emb_val)
    np.save(str(test_path), emb_test)
    print(f"Saved embeddings to {emb_dir}/")

    return emb_train, emb_val, emb_test


def build_tfidf_features(train_prompts, test_prompts, val_prompts=None, holdout_prompts=None):
    """Build TF-IDF matrices with <NUM> masking already applied to inputs.

    Parameters
    ----------
    train_prompts, test_prompts : array-like of str
        Already <NUM>-masked prompt texts.
    val_prompts, holdout_prompts : array-like of str, optional
        Additional splits to transform (not fit).

    Returns
    -------
    tfidf_vec : TfidfVectorizer
        Fitted vectorizer.
    X_train, X_test : sparse matrix
    X_val : sparse matrix or None
    X_holdout : sparse matrix or None
    """
    tfidf_vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train = tfidf_vec.fit_transform(train_prompts)
    X_test = tfidf_vec.transform(test_prompts)
    X_val = tfidf_vec.transform(val_prompts) if val_prompts is not None else None
    X_holdout = tfidf_vec.transform(holdout_prompts) if holdout_prompts is not None else None

    return tfidf_vec, X_train, X_test, X_val, X_holdout


def mask_digits(text):
    """Replace all digit sequences with ``<NUM>``."""
    return _NUM_PAT.sub("<NUM>", text)
