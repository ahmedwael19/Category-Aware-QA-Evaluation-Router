"""Generate a cross-instruction holdout set.

The router is trained on prompts from phase-1 synthetic data generation, which
uses a specific template + prompt combination. This script regenerates prompts
from category descriptions only (no templates, no few-shot, no forbidden-word
filter) to test generalisation to instruction variation rather than to a
different generator family.

Run: python -m scripts.artifacts.generate_holdout
"""

import json
import time
import random
import re
import glob
from datetime import datetime

from config import DATA_DIR, HOLDOUT_CSV
from thesis_router import get_openai_client

import numpy as np
import pandas as pd

SEED = 99
random.seed(SEED)
np.random.seed(SEED)

_cached_client = None


def client():
    global _cached_client
    if _cached_client is None:
        _cached_client = get_openai_client()
    return _cached_client

# ── Category descriptions (shared taxonomy, different instructions) ──────────
CATEGORIES = {
    "SYMBOLIC_TIME": {
        "description": "Prompts that can be answered by comparing timestamps or measuring durations. The answer requires arithmetic over time values.",
        "examples_of_real_use": "SLA checks, response time verification, wait time thresholds, duration limits",
        "n": 30,
    },
    "SYMBOLIC_METADATA": {
        "description": "Prompts that can be answered by checking whether a specific tag, channel, or metadata field has a particular value. Exact-match lookup.",
        "examples_of_real_use": "Tag verification, channel detection, source type checks, field presence",
        "n": 30,
    },
    "SYMBOLIC_COUNT": {
        "description": "Prompts that can be answered by counting messages, turns, exchanges, or other enumerable conversation elements.",
        "examples_of_real_use": "Turn limits, message count thresholds, exchange counting",
        "n": 30,
    },
    "SEMANTIC": {
        "description": "Prompts that require subjective human judgment about communication quality. No deterministic answer exists. Covers: tone, empathy, professionalism, helpfulness, comprehension, greeting quality, closing quality, solution quality.",
        "examples_of_real_use": "Tone assessment, empathy checks, solution quality, greeting appropriateness, professionalism evaluation",
        "n": 30,
    },
    "HYBRID": {
        "description": "Prompts that combine a verifiable numeric or metadata condition WITH a subjective quality judgment. Both parts are needed to answer. If you remove the condition, the evaluation changes.",
        "examples_of_real_use": "Conditional quality checks: 'if wait was long, did agent apologize?', 'for email, was tone formal?'",
        "n": 30,
    },
    "UNSUPPORTED": {
        "description": "Prompts that reference data NOT available in a single conversation transcript. This includes: customer history across other tickets, CRM/billing data, post-conversation satisfaction scores, future predictions, agent performance comparisons.",
        "examples_of_real_use": "Cross-ticket history, lifetime value, churn prediction, CSAT scores, CRM lookups",
        "n": 30,
    },
}

# ── Generation prompt — deliberately DIFFERENT from main pipeline ─────────────
SYSTEM_PROMPT = """You are helping create a test dataset for a QA evaluation system.
Write evaluation prompts that a real customer service QA manager might write.
Be natural and varied. Use different sentence structures, lengths, and styles.
Some should be short and direct. Some should be longer and more detailed.
Mix question style ("Did the agent...?") with instruction style ("Check if...").
Do NOT use overly formal or academic language — write like a real person."""

USER_TEMPLATE = """Write exactly {n} QA evaluation prompts for the category: {category}

Category description: {description}
Real-world use cases: {examples}

Requirements:
- Each prompt must clearly belong to this category
- Use diverse phrasing — no two prompts should sound the same
- Include both easy/obvious and subtle/tricky examples
- Use a mix of operators where applicable (within/under/more than/fewer than/exactly/at least/at most)
- Vary sentence length and complexity

Return JSON: {{"prompts": ["...", ...]}}"""


def generate_batch(category, config, n=30):
    """Generate prompts for one category."""
    prompt = USER_TEMPLATE.format(
        n=n,
        category=category,
        description=config["description"],
        examples=config["examples_of_real_use"],
    )

    for attempt in range(3):
        try:
            resp = client().chat.completions.create(
                model="gpt-5.2",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_completion_tokens=4000,
            )
            result = json.loads(resp.choices[0].message.content)
            prompts = result.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts[:n]
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []


def inject_noise(text):
    """Apply light noise to ~30% of prompts to match the training distribution."""
    if random.random() > 0.30:
        return text

    ops = random.choice(["lower_start", "strip_punct", "add_ellipsis", "casual"])
    if ops == "lower_start" and len(text) > 0:
        return text[0].lower() + text[1:]
    if ops == "strip_punct":
        return text.rstrip("?.!")
    if ops == "add_ellipsis":
        return text.rstrip("?.!") + "..."
    replacements = {"please ": "", "Please ": "", "whether ": "if ", "determine ": "check "}
    for old, new in replacements.items():
        if old in text:
            return text.replace(old, new, 1)
    return text


def main() -> None:
    print("Generating cross-instruction holdout set...")
    all_prompts = []

    for cat, config in CATEGORIES.items():
        n = config["n"]
        batch1_n = n // 2
        batch2_n = n - batch1_n
        print(f"  {cat} ({n} prompts in 2 batches)...")

        batch1 = generate_batch(cat, config, n=batch1_n)
        time.sleep(1)
        batch2 = generate_batch(cat, config, n=batch2_n)
        prompts = batch1 + batch2

        if len(prompts) < n:
            print(f"    WARNING: Got {len(prompts)}, expected {n}. Generating supplementary batch...")
            prompts.extend(generate_batch(cat, config, n=n - len(prompts)))

        print(f"    Got {len(prompts)}")

        for p in prompts:
            all_prompts.append({
                "prompt": inject_noise(p),
                "prompt_original": p,
                "top_category": cat,
                "generation_source": "cross_instruction_holdout",
            })
        time.sleep(1)

    df = pd.DataFrame(all_prompts)

    n_before = len(df)
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    print(f"\nAfter exact dedup: {n_before} → {len(df)}")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        print("Computing embeddings for near-duplicate removal...")
        model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        embeddings = model.encode(
            ["query: " + p for p in df["prompt"]],
            show_progress_bar=True,
            batch_size=32,
        )

        NEAR_DUP_THRESHOLD = 0.95
        to_remove = set()
        for cat in df["top_category"].unique():
            idx = df[df["top_category"] == cat].index.tolist()
            if len(idx) < 2:
                continue
            cat_emb = embeddings[idx]
            sim = cos_sim(cat_emb)
            np.fill_diagonal(sim, 0)
            for i in range(len(sim)):
                for j in range(i + 1, len(sim)):
                    if sim[i, j] > NEAR_DUP_THRESHOLD and idx[j] not in to_remove:
                        to_remove.add(idx[j])

        if to_remove:
            print(f"Removing {len(to_remove)} near-duplicates (sim > {NEAR_DUP_THRESHOLD})")
            df = df.drop(index=to_remove).reset_index(drop=True)
        else:
            print("No near-duplicates found")
    except ImportError:
        print("WARNING: sentence-transformers not installed — skipping near-duplicate removal")

    print(f"After near-dedup: {len(df)}")

    train_files = sorted(glob.glob(str(DATA_DIR / "synthetic_final_*_train.csv")))
    val_files = sorted(glob.glob(str(DATA_DIR / "synthetic_final_*_val.csv")))
    test_files = sorted(glob.glob(str(DATA_DIR / "synthetic_final_*_test.csv")))

    all_training_prompts = set()
    for flist, label in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
        if flist:
            split_df = pd.read_csv(flist[-1])
            split_prompts = set(split_df["prompt"].str.lower().str.strip())
            overlap = df["prompt"].apply(lambda p: p.lower().strip() in split_prompts).sum()
            print(f"Exact overlap with {label}: {overlap}")
            all_training_prompts.update(split_prompts)

    total_overlap = df["prompt"].apply(lambda p: p.lower().strip() in all_training_prompts).sum()
    print(f"Total overlap with any split: {total_overlap}")

    if total_overlap > 0:
        df = df[~df["prompt"].apply(lambda p: p.lower().strip() in all_training_prompts)].reset_index(drop=True)
        print(f"After overlap removal: {len(df)}")

    print(f"\n{'Category':<24} {'N':>3} {'Mean len':>8} {'Q%':>4} {'TTR':>6}")
    for c in sorted(df["top_category"].unique()):
        sub = df[df["top_category"] == c]
        lens = sub["prompt"].str.len()
        q_pct = (sub["prompt"].str.contains(r"\?").sum() / len(sub)) * 100
        tokens: list[str] = []
        for p in sub["prompt"]:
            tokens.extend(re.findall(r"\w+", p.lower()))
        ttr = len(set(tokens)) / len(tokens) if tokens else 0
        print(f"  {c:<22} {len(sub):>3} {lens.mean():>8.0f} {q_pct:>3.0f}% {ttr:>6.3f}")

    df.to_csv(HOLDOUT_CSV, index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model": "gpt-5.2",
        "temperature": 1.0,
        "python_seed": SEED,
        "note_on_reproducibility": (
            "Python random seed controls noise injection only. "
            "LLM API calls are stochastic (no API-level seed). "
            "The saved CSV is the canonical artifact."
        ),
        "total_before_dedup": n_before,
        "total_after_dedup": len(df),
        "per_category": df["top_category"].value_counts().to_dict(),
        "batches_per_category": 2,
        "noise_injection": "30% of prompts receive light noise to match training distribution",
        "near_duplicate_threshold": 0.95,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "category_definitions": {k: v["description"] for k, v in CATEGORIES.items()},
        "overlap_with_any_split": int(total_overlap),
        "framing": (
            "This is a cross-instruction holdout, not a cross-model holdout. "
            "Both training and holdout use GPT-5.2. The test measures "
            "robustness to instruction variation, not cross-generator transfer."
        ),
    }
    with open(DATA_DIR / "holdout_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {HOLDOUT_CSV.name} and holdout_metadata.json ({len(df)} prompts)")


if __name__ == "__main__":
    main()
