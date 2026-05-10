import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    import random
    import hashlib
    import re
    import time
    from datetime import datetime
    from pathlib import Path

    return (
        Path,
        datetime,
        hashlib,
        json,
        mo,
        np,
        pd,
        random,
        re,
        time,
    )


@app.cell
def set_seed(np, random):
    """Fixed seed for full reproducibility"""
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")
    return (SEED,)


@app.cell
def setup_openai():
    """OpenAI client from OPENAI_API_KEY environment variable."""
    from thesis_router import get_openai_client
    llm_client = get_openai_client()
    print("OpenAI client ready")
    return (llm_client,)


@app.cell
def title(mo):
    mo.md("""
    # Synthetic Benchmark Generation

    - **Strategy 1**: template-based deterministic generation (SYMBOLIC)
    - **Strategy 2**: scenario-conditioned LLM generation (SEMANTIC)
    - **Strategy 3**: combined generation (HYBRID, UNSUPPORTED)
    - **Noise injection** for realism
    - **Deduplication**: exact + semantic (cosine > 0.99 on LLM-generated prompts)
    - **Stratified split**: train 70% / val 15% / test 15%, seed 42
    - **Language**: English only
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Deterministic noise injection using dictionary lookups and adjacent-key swapping.
    """)
    return


@app.cell
def noise_module(random):
    from benchmark.noise import apply_noise as _apply_noise_raw, pick_noise_level as _pick_noise_level_raw

    # Wrap to bind the random module (matching original Marimo closure interface)
    def apply_noise(text, intensity="medium"):
        return _apply_noise_raw(text, intensity, random_module=random)

    def pick_noise_level():
        return _pick_noise_level_raw(random_module=random)

    # Demo
    _demo = "Did the agent acknowledge the customer's frustration appropriately?"
    print("Noise injection demo:")
    for _lv in ["none", "light", "medium", "heavy"]:
        print(f"  {_lv:>6}: {apply_noise(_demo, _lv)}")
    return apply_noise, pick_noise_level


@app.cell
def _(mo):
    mo.md(r"""
    ## Balanced 50/50 question vs instruction style.

    Justification: Both question-style ("Did the agent...?") and instruction-style
    ("Detect whether...") are prevalent in QA evaluation systems. A balanced split
    prevents the classifier from using prompt style as a shortcut signal for
    category prediction, ensuring it must learn semantic features instead.
    """)
    return


@app.cell
def style_fn(random):
    from benchmark.noise import pick_style as _pick_style_raw

    def pick_style():
        return _pick_style_raw(random_module=random)
    return (pick_style,)


@app.cell
def _(mo):
    mo.md(r"""
    ## SYMBOLIC_TIME: response time, SLA, wait time
    """)
    return


@app.cell
def gen_time(apply_noise, json, pick_noise_level, pick_style, random):
    from benchmark.templates_time import generate_time_prompts

    def generate_time_prompt():
        return generate_time_prompts(apply_noise, pick_noise_level, pick_style, random, count=1)[0]

    print("SYMBOLIC_TIME samples:")
    for _ in range(6):
        _s = generate_time_prompt()
        print(f"  [{_s['style'][:3]}|{_s['subcategory']}] {_s['prompt']}")
    return (generate_time_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## SYMBOLIC_METADATA: tags, channel, internal notes.
    """)
    return


@app.cell
def gen_metadata(apply_noise, json, pick_noise_level, pick_style, random):
    from benchmark.templates_metadata import generate_metadata_prompts

    def generate_metadata_prompt():
        return generate_metadata_prompts(apply_noise, pick_noise_level, pick_style, random, count=1)[0]

    print("SYMBOLIC_METADATA samples:")
    for _ in range(6):
        _s = generate_metadata_prompt()
        print(f"  [{_s['style'][:3]}|{_s['subcategory']}] {_s['prompt']}")
    return (generate_metadata_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## SYMBOLIC_COUNT: turn counting, message counting
    """)
    return


@app.cell
def gen_count(apply_noise, json, pick_noise_level, pick_style, random):
    from benchmark.templates_count import generate_count_prompts

    def generate_count_prompt():
        return generate_count_prompts(apply_noise, pick_noise_level, pick_style, random, count=1)[0]

    print("SYMBOLIC_COUNT samples:")
    for _ in range(6):
        _s = generate_count_prompt()
        print(f"  [{_s['style'][:3]}|{_s['subcategory']}] {_s['prompt']}")
    return (generate_count_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Scenario dimensions and category definitions for SEMANTIC generation
    """)
    return


@app.cell
def semantic_config():
    from benchmark.semantic_generation import (
        SCENARIOS, STAGES, BEHAVIORS, STRUCTURES, SEMANTIC_CATS,
    )

    print(f"Scenarios: {len(SCENARIOS)} | Stages: {len(STAGES)} | Behaviors: {len(BEHAVIORS)}")
    print(f"Semantic categories: {len(SEMANTIC_CATS)}")
    return BEHAVIORS, SCENARIOS, SEMANTIC_CATS, STAGES, STRUCTURES


@app.cell
def _(mo):
    mo.md(r"""
    ## Scenario-conditioned LLM generation for SEMANTIC categories
    """)
    return


@app.cell
def gen_semantic(
    BEHAVIORS,
    SCENARIOS,
    SEMANTIC_CATS,
    STAGES,
    STRUCTURES,
    json,
    llm_client,
    random,
    time,
):
    from benchmark.semantic_generation import generate_semantic_batch as _generate_semantic_batch

    def generate_semantic_batch(cat_name, n=25, scenario=None, stage=None, behavior=None, structure=None):
        return _generate_semantic_batch(
            llm_client, cat_name, n=n, scenario=scenario, stage=stage,
            behavior=behavior, structure=structure, random_module=random,
        )

    # Test
    print("Testing scenario-conditioned generation (SEMANTIC_EMPATHY)...")
    _test = generate_semantic_batch(
        "SEMANTIC_EMPATHY", n=5,
        scenario="billing error", stage="escalation request",
        behavior="acknowledging frustration", structure="conditional",
    )
    for _p in _test[:5]:
        print(f"  - {_p}")
    return (generate_semantic_batch,)


@app.cell
def _(mo):
    mo.md(r"""
    ## HYBRID: symbolic condition + semantic judgment. Routed to LLM in evaluation.
    """)
    return


@app.cell
def gen_hybrid(apply_noise, pick_noise_level, pick_style, random):
    from benchmark.templates_hybrid import generate_hybrid_prompts

    def generate_hybrid_prompt():
        return generate_hybrid_prompts(apply_noise, pick_noise_level, pick_style, random, count=1)[0]

    print("HYBRID samples:")
    for _ in range(5):
        _s = generate_hybrid_prompt()
        print(f"  [{_s['style'][:3]}] {_s['prompt']}")
    return (generate_hybrid_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## UNSUPPORTED: references unavailable data. Parameterized templates for diversity
    """)
    return


@app.cell
def gen_unsupported(apply_noise, pick_noise_level, pick_style, random):
    from benchmark.templates_unsupported import generate_unsupported_prompts

    def generate_unsupported_prompt():
        return generate_unsupported_prompts(apply_noise, pick_noise_level, pick_style, random, count=1)[0]

    print("UNSUPPORTED template samples:")
    for _ in range(5):
        _s = generate_unsupported_prompt()
        print(f"  [{_s['style'][:3]}] {_s['prompt']}")
    return (generate_unsupported_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## LLM-generated UNSUPPORTED prompts for diversity beyond templates
    """)
    return


@app.cell
def gen_unsupported_llm(json, llm_client, random, time):
    from benchmark.semantic_generation import (
        generate_unsupported_llm_batch as _generate_unsupported_llm_batch,
        UNAVAIL_TYPES as _UNAVAIL_TYPES,
        UNSUPPORTED_DOMAINS as _DOMAINS,
    )

    def generate_unsupported_llm_batch(n=25, unavail_type=None, domain=None):
        return _generate_unsupported_llm_batch(
            llm_client, n=n, unavail_type=unavail_type, domain=domain,
            random_module=random,
        )

    print("Testing LLM UNSUPPORTED generation...")
    _test = generate_unsupported_llm_batch(n=3, unavail_type=_UNAVAIL_TYPES[0], domain=_DOMAINS[0])
    for _p in _test[:3]:
        print(f"  - {_p}")
    return (generate_unsupported_llm_batch,)


@app.cell
def _(mo):
    mo.md(r"""
    ## LLM-generated HYBRID prompts with strict compositional constraint
    """)
    return


@app.cell
def gen_hybrid_llm(json, llm_client, random, time):
    from benchmark.semantic_generation import (
        generate_hybrid_llm_batch as _generate_hybrid_llm_batch,
        SYMBOLIC_DIMS as _SYMBOLIC_DIMS,
        SEMANTIC_DIMS as _SEMANTIC_DIMS,
    )

    def generate_hybrid_llm_batch(n=25, symbolic_dim=None, semantic_dim=None, domain=None):
        return _generate_hybrid_llm_batch(
            llm_client, n=n, symbolic_dim=symbolic_dim, semantic_dim=semantic_dim,
            domain=domain, random_module=random,
        )

    print("Testing LLM HYBRID generation...")
    _test = generate_hybrid_llm_batch(n=3, symbolic_dim=_SYMBOLIC_DIMS[0], semantic_dim=_SEMANTIC_DIMS[0])
    for _p in _test[:3]:
        print(f"  - {_p}")
    return (generate_hybrid_llm_batch,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate the complete dataset with template + LLM generation for all categories
    """)
    return


@app.cell
def orchestrate(
    BEHAVIORS,
    SCENARIOS,
    SEMANTIC_CATS,
    STAGES,
    STRUCTURES,
    apply_noise,
    generate_count_prompt,
    generate_hybrid_llm_batch,
    generate_hybrid_prompt,
    generate_metadata_prompt,
    generate_semantic_batch,
    generate_time_prompt,
    generate_unsupported_llm_batch,
    generate_unsupported_prompt,
    pd,
    pick_noise_level,
    random,
    time,
):
    print("=" * 80)
    print("FULL DATASET GENERATION")
    print("=" * 80)

    dataset = []

    # ── SYMBOLIC_TIME (600 templates) ────────────────────────────────────
    print("\n[1/8] SYMBOLIC_TIME (600 templates)...")
    for _ in range(600):
        dataset.append(generate_time_prompt())
    print(f"  Done: 600")

    # ── SYMBOLIC_METADATA (600 templates) ────────────────────────────────
    print("[2/8] SYMBOLIC_METADATA (600 templates)...")
    for _ in range(600):
        dataset.append(generate_metadata_prompt())
    print(f"  Done: 600")

    # ── SYMBOLIC_COUNT (600 templates) ───────────────────────────────────
    print("[3/8] SYMBOLIC_COUNT (600 templates)...")
    for _ in range(600):
        dataset.append(generate_count_prompt())
    print(f"  Done: 600")

    # ── SEMANTIC (300 per sub-category × 6 = 1800, LLM) ─────────────────
    print("[4/8] SEMANTIC (1800 via scenario-conditioned gpt-5.2)...")
    for cat_name in SEMANTIC_CATS:
        print(f"  {cat_name}...")
        cat_prompts = []

        # Systematic scenario conditioning: cycle through dimensions
        for batch_i in range(12):
            scenario = SCENARIOS[batch_i % len(SCENARIOS)]
            stage = STAGES[batch_i % len(STAGES)]
            behavior = BEHAVIORS[batch_i % len(BEHAVIORS)]
            structure = STRUCTURES[batch_i % len(STRUCTURES)]

            try:
                batch = generate_semantic_batch(
                    cat_name, n=30,
                    scenario=scenario, stage=stage,
                    behavior=behavior, structure=structure,
                )
                cat_prompts.extend(batch)
                print(f"    Batch {batch_i+1}/12: {len(batch)} (scenario={scenario[:15]}...)")
                time.sleep(1)
            except Exception as e:
                print(f"    Batch {batch_i+1} error: {e}")

        # Add to dataset (cap at 300, apply noise)
        for p_text in cat_prompts[:300]:
            noise = pick_noise_level()
            p_text = apply_noise(p_text, noise)
            dataset.append({
                "prompt": p_text, "category": cat_name,
                "subcategory": "llm_generated", "style": "question" if "?" in p_text else "instruction",
                "noise": noise, "params": None,
            })
        print(f"    Total: {min(len(cat_prompts), 300)}")

    # ── HYBRID — templates (600) ─────────────────────────────────────────
    print("[5/8] HYBRID templates (600)...")
    for _ in range(600):
        dataset.append(generate_hybrid_prompt())
    print(f"  Done: 600 templates")

    # ── HYBRID — LLM (300, 12 batches of 25) ────────────────────────────
    print("[6/8] HYBRID LLM (300 via gpt-5.2)...")
    _SYMBOLIC_DIMS = [
        "response time threshold (minutes)", "message count threshold (turns)",
        "channel type (email/chat/phone)", "ticket tag (urgent/vip/escalated)",
        "internal notes presence", "resolution time",
    ]
    _SEMANTIC_DIMS = [
        "empathy and emotional acknowledgment", "professional tone and courtesy",
        "solution quality and helpfulness", "greeting appropriateness",
        "closing and follow-up offer", "comprehension of customer issue",
    ]
    hybrid_llm_prompts = []
    for batch_i in range(12):
        try:
            batch = generate_hybrid_llm_batch(
                n=25,
                symbolic_dim=_SYMBOLIC_DIMS[batch_i % len(_SYMBOLIC_DIMS)],
                semantic_dim=_SEMANTIC_DIMS[batch_i % len(_SEMANTIC_DIMS)],
            )
            hybrid_llm_prompts.extend(batch)
            print(f"    Batch {batch_i+1}/12: {len(batch)}")
            time.sleep(1)
        except Exception as e:
            print(f"    Batch {batch_i+1} error: {e}")
    for p_text in hybrid_llm_prompts[:300]:
        noise = pick_noise_level()
        p_text = apply_noise(p_text, noise)
        dataset.append({
            "prompt": p_text, "category": "HYBRID",
            "subcategory": "llm_generated", "style": "question" if "?" in p_text else "instruction",
            "noise": noise, "params": None,
        })
    print(f"  Done: {min(len(hybrid_llm_prompts), 300)} LLM-generated")

    # ── UNSUPPORTED — templates (600) ────────────────────────────────────
    print("[7/8] UNSUPPORTED templates (600)...")
    for _ in range(600):
        dataset.append(generate_unsupported_prompt())
    print(f"  Done: 600 templates")

    # ── UNSUPPORTED — LLM (200, 8 batches of 25) ────────────────────────
    print("[8/8] UNSUPPORTED LLM (200 via gpt-5.2)...")
    _UNAVAIL_TYPES = [
        "cross-ticket data (customer's other interactions)",
        "post-conversation outcomes (satisfaction score, follow-up)",
        "external system records (billing, subscription, purchase history)",
        "predictive/hypothetical (future behavior, churn likelihood)",
        "comparative data (agent vs team performance, benchmarks)",
        "customer profile data (lifetime value, account tier, preferences)",
    ]
    _DOMAINS = [
        "e-commerce refund", "SaaS subscription", "airline complaint",
        "banking dispute", "telecom service issue", "insurance claim",
        "food delivery complaint", "hotel booking issue",
    ]
    unsupported_llm_prompts = []
    for batch_i in range(8):
        try:
            batch = generate_unsupported_llm_batch(
                n=25,
                unavail_type=_UNAVAIL_TYPES[batch_i % len(_UNAVAIL_TYPES)],
                domain=_DOMAINS[batch_i % len(_DOMAINS)],
            )
            unsupported_llm_prompts.extend(batch)
            print(f"    Batch {batch_i+1}/8: {len(batch)}")
            time.sleep(1)
        except Exception as e:
            print(f"    Batch {batch_i+1} error: {e}")
    for p_text in unsupported_llm_prompts[:200]:
        noise = pick_noise_level()
        p_text = apply_noise(p_text, noise)
        dataset.append({
            "prompt": p_text, "category": "UNSUPPORTED",
            "subcategory": "llm_generated", "style": "question" if "?" in p_text else "instruction",
            "noise": noise, "params": None,
        })
    print(f"  Done: {min(len(unsupported_llm_prompts), 200)} LLM-generated")

    # Shuffle
    random.shuffle(dataset)
    df_raw = pd.DataFrame(dataset)

    print(f"\n{'=' * 80}")
    print(f"Generated: {len(df_raw):,} prompts")
    print(f"\nCategory distribution:")
    print(df_raw["category"].value_counts().to_string())
    return (df_raw,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 1: exact string deduplication only.

    Near-exact normalization (lowercasing, stripping punctuation) is intentionally
    NOT applied because it would conflate prompts that differ only by noise injection
    (which is a desired property) or by numeric parameters (which represent different
    evaluation thresholds). Semantic dedup handles true paraphrases instead.
    """)
    return


@app.cell
def dedup_string(df_raw):
    from benchmark.deduplication import dedup_exact

    print("=" * 80)
    print("DEDUPLICATION: Exact string matches")
    print("=" * 80)

    df_string_deduped = dedup_exact(df_raw)
    return (df_string_deduped,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 2: Semantic deduplication using embedding cosine similarity > 0.99.

    Threshold 0.99 removes only near-paraphrases (same meaning, trivially different wording).
    Lower thresholds (e.g. 0.95) are too aggressive — they remove prompts that differ
    meaningfully (e.g. different thresholds, different scenarios).
    """)
    return


@app.cell
def dedup_semantic(df_string_deduped, np):
    from benchmark.deduplication import dedup_semantic as _dedup_semantic

    print("=" * 80)
    print("DEDUPLICATION: Semantic (embedding cosine > 0.99)")
    print("=" * 80)

    df_deduped = _dedup_semantic(df_string_deduped, threshold=0.99)
    return (df_deduped,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Compute validation metrics for the synthetic dataset.
    """)
    return


@app.cell
def validate_dataset(df_deduped, re):
    from benchmark.validation import validate

    print("=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    validate(df_deduped)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stratified train/val/test split (70/15/15) and export
    """)
    return


@app.cell
def split_export(Path, SEED, datetime, df_deduped, hashlib, json, pd):
    print("=" * 80)
    print("STRATIFIED SPLIT & EXPORT")
    print("=" * 80)

    # Add top_category column (6 categories for router)
    _df = df_deduped.copy()
    _df["top_category"] = _df["category"].apply(
        lambda c: "SEMANTIC" if c.startswith("SEMANTIC_") else c
    )

    train_parts, val_parts, test_parts = [], [], []

    for _cat in _df["top_category"].unique():
        _sub = _df[_df["top_category"] == _cat].sample(frac=1, random_state=SEED)
        _n = len(_sub)
        _nt = int(_n * 0.70)
        _nv = int(_n * 0.15)
        train_parts.append(_sub.iloc[:_nt])
        val_parts.append(_sub.iloc[_nt:_nt + _nv])
        test_parts.append(_sub.iloc[_nt + _nv:])

    df_train = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_val = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_test = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\n  Train: {len(df_train):,}")
    print(f"  Val:   {len(df_val):,}")
    print(f"  Test:  {len(df_test):,}")

    print(f"\n  Per top_category:")
    for _cat in sorted(_df["top_category"].unique()):
        tr = (df_train["top_category"] == _cat).sum()
        va = (df_val["top_category"] == _cat).sum()
        te = (df_test["top_category"] == _cat).sum()
        print(f"    {_cat:<20} train={tr:>4}  val={va:>4}  test={te:>4}")

    # Save
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _hash = hashlib.md5(df_deduped["prompt"].str.cat().encode()).hexdigest()[:8]
    _dir = str(Path(__file__).parent.parent / "data")
    _base = f"{_dir}/synthetic_final_{_ts}_{_hash}"

    df_train.to_csv(f"{_base}_train.csv", index=False)
    df_val.to_csv(f"{_base}_val.csv", index=False)
    df_test.to_csv(f"{_base}_test.csv", index=False)
    _df.to_csv(f"{_base}_full.csv", index=False)

    metadata = {
        "generated_at": _ts,
        "data_hash": _hash,
        "seed": SEED,
        "total": len(_df),
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "top_categories": _df["top_category"].value_counts().to_dict(),
        "sub_categories": _df["category"].value_counts().to_dict(),
        "style_dist": _df["style"].value_counts().to_dict(),
        "language": "en",
        "model": "gpt-5.2",
        "dedup_stages": ["exact_string", "semantic_cosine_0.99_on_llm_generated_only"],
        "generation_strategies": [
            "template_deterministic_symbolic",
            "scenario_conditioned_llm_semantic",
            "template_hybrid",
            "llm_hybrid",
            "template_unsupported",
            "llm_unsupported",
        ],
        "noise_levels": _df["noise"].value_counts().to_dict(),
        "version": "final",
    }
    with open(f"{_base}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Files saved: {_base}_*.csv")
    return


if __name__ == "__main__":
    app.run()
