"""Deduplication stages for synthetic benchmark data.

Stage 1: Exact string deduplication.
Stage 2: Semantic deduplication using embedding cosine similarity (LLM-generated only).
"""

import numpy as np


def dedup_exact(df):
    """Remove exact duplicate prompts.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``prompt`` column.

    Returns
    -------
    pandas.DataFrame
        Copy with exact duplicates removed.
    """
    n0 = len(df)
    df1 = df.drop_duplicates(subset=["prompt"]).copy()
    print(f"  Exact duplicates removed: {n0 - len(df1)}")
    print(f"  After exact dedup: {len(df1):,}")
    return df1


def dedup_semantic(df, threshold=0.99):
    """Remove near-paraphrase LLM-generated prompts via embedding cosine similarity.

    Only applies to rows where ``subcategory == 'llm_generated'``.
    Template-generated prompts (SYMBOLIC, template HYBRID/UNSUPPORTED) have
    legitimate structural similarity (same template, different parameters) that
    must be preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``prompt``, ``category``, and ``subcategory`` columns.
    threshold : float
        Cosine similarity threshold above which to deduplicate (default 0.99).

    Returns
    -------
    pandas.DataFrame
        Copy with semantic duplicates removed and index reset.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    prompts = df["prompt"].tolist()

    print(f"  Computing embeddings for {len(prompts)} prompts...")
    embeddings = model.encode(prompts, show_progress_bar=True, batch_size=64)

    # Per-category semantic dedup
    keep_mask = np.ones(len(prompts), dtype=bool)
    removed = 0

    for cat in df["category"].unique():
        # Only apply semantic dedup to LLM-generated prompts, regardless of category.
        # Template-generated prompts (SYMBOLIC, template HYBRID/UNSUPPORTED) have
        # legitimate structural similarity (same template, different parameters) that
        # must be preserved. LLM-generated prompts in ANY category (SEMANTIC_*,
        # HYBRID llm_generated, UNSUPPORTED llm_generated) are deduped to remove
        # near-paraphrases that inflate dataset size without adding diversity.
        cat_idx = df.index[
            (df["category"] == cat) &
            (df["subcategory"] == "llm_generated")
        ].tolist()
        if len(cat_idx) < 2:
            continue

        cat_positions = [df.index.get_loc(i) for i in cat_idx]
        cat_emb = embeddings[cat_positions]
        sim = cosine_similarity(cat_emb)

        # Remove only near-paraphrases (cosine > threshold)
        for i in range(len(cat_positions)):
            if not keep_mask[cat_positions[i]]:
                continue
            for j in range(i + 1, len(cat_positions)):
                if not keep_mask[cat_positions[j]]:
                    continue
                if sim[i, j] > threshold:
                    keep_mask[cat_positions[j]] = False
                    removed += 1

    df_deduped = df[keep_mask].reset_index(drop=True)
    print(f"  Semantic duplicates removed: {removed}")
    print(f"  Final dataset: {len(df_deduped):,}")
    print(f"\nCategory distribution after all dedup:")
    print(df_deduped["category"].value_counts().to_string())
    return df_deduped
