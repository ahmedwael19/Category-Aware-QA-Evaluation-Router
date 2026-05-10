"""Secondary-generator validation.

Generate a small batch of SEMANTIC, HYBRID, and UNSUPPORTED prompts with a
Claude model (Sonnet or Opus) using the same scenario-conditioning template
as the GPT-5.2 generator, then evaluate the calibrated TF-IDF + LR router
(the one shipped under `models/router_model_calibrated.joblib`) on them.
The router's accuracy here bounds how much of its in-distribution performance
transfers across LLM generator families rather than riding on stylistic
fingerprints of a single generator.

Usage:
    python -m scripts.validation.secondary_generator --model sonnet
    python -m scripts.validation.secondary_generator --model opus
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

import joblib
import numpy as np
import pandas as pd

from benchmark.semantic_generation import (
    SCENARIOS, STAGES, BEHAVIORS, STRUCTURES,
    SEMANTIC_CATS, PROMPT_TEMPLATE,
    UNSUPPORTED_LLM_PROMPT, UNAVAIL_TYPES,
    HYBRID_LLM_PROMPT,
)
from config import (
    MODEL_CALIBRATED, TFIDF_VECTORIZER, LABEL_ENCODER,
    RESULTS_VALIDATION,
)
from thesis_router import get_anthropic_client

random.seed(42)

MODEL_IDS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-7",
}


def call_claude(client, model_id: str, system_prompt_text: str, max_tokens: int = 4096) -> str:
    resp = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": system_prompt_text}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def generate_batch(client, model_id: str, system_prompt: str, n: int) -> list[str]:
    text = call_claude(client, model_id, system_prompt, max_tokens=4096)
    if not text:
        return []
    m = re.search(r'\{[^{}]*?"prompts"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("prompts", [])
        except json.JSONDecodeError:
            pass
    m = re.search(r'"prompts"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        prompts = re.findall(r'"((?:[^"\\]|\\.)*?)"', m.group(1))
        if prompts:
            return prompts
    print(f"  [warn] no prompts parsed (first 200 chars): {text[:200]!r}")
    return []


def gen_semantic_sample(client, model_id: str, n_per_subcat: int = 4) -> list[dict]:
    out = []
    for cat, info in SEMANTIC_CATS.items():
        prompt_text = PROMPT_TEMPLATE.format(
            forbidden=info["forbidden"],
            category=cat,
            desc=info["desc"],
            scenario=random.choice(SCENARIOS),
            stage=random.choice(STAGES),
            behavior=random.choice(BEHAVIORS),
            structure=random.choice(STRUCTURES),
            n=n_per_subcat,
        )
        print(f"  Generating {n_per_subcat} for {cat}...")
        for p in generate_batch(client, model_id, prompt_text, n_per_subcat):
            out.append({
                "prompt": p,
                "top_category": "SEMANTIC",
                "sub_category": cat,
                "generator": model_id,
            })
    return out


def gen_unsupported_sample(client, model_id: str, n: int = 12) -> list[dict]:
    prompt_text = UNSUPPORTED_LLM_PROMPT.format(
        unavail_type=random.choice(UNAVAIL_TYPES),
        domain=random.choice(SCENARIOS),
        n=n,
    )
    print(f"  Generating {n} UNSUPPORTED...")
    return [
        {"prompt": p, "top_category": "UNSUPPORTED", "sub_category": "UNSUPPORTED", "generator": model_id}
        for p in generate_batch(client, model_id, prompt_text, n)
    ]


def gen_hybrid_sample(client, model_id: str, n: int = 12) -> list[dict]:
    prompt_text = HYBRID_LLM_PROMPT.format(
        symbolic_dim=random.choice(["response time", "tag presence", "turn count"]),
        semantic_dim=random.choice(["empathy", "tone", "solution quality"]),
        domain=random.choice(SCENARIOS),
        n=n,
    )
    print(f"  Generating {n} HYBRID...")
    return [
        {"prompt": p, "top_category": "HYBRID", "sub_category": "HYBRID", "generator": model_id}
        for p in generate_batch(client, model_id, prompt_text, n)
    ]


def mask_digits(text: str) -> str:
    return re.sub(r"\d+", "<NUM>", text)


def run_router(df: pd.DataFrame) -> pd.DataFrame:
    tfidf = joblib.load(str(TFIDF_VECTORIZER))
    model = joblib.load(str(MODEL_CALIBRATED))
    le = joblib.load(str(LABEL_ENCODER))

    X = tfidf.transform([mask_digits(p) for p in df["prompt"]])
    preds_idx = model.predict(X)
    preds = le.inverse_transform(preds_idx) if isinstance(preds_idx[0], (np.integer, int)) else preds_idx
    probs = model.predict_proba(X)

    df = df.copy()
    df["predicted_category"] = preds
    df["max_confidence"] = probs.max(axis=1)
    df["correct"] = df["predicted_category"] == df["top_category"]
    return df


def main():
    from thesis_router import rebuild_baselines

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_IDS), required=True,
                    help="Claude model family to use as the secondary generator")
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    RESULTS_VALIDATION.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_VALIDATION / f"secondary_generator_{args.model}.csv"
    out_json = RESULTS_VALIDATION / f"secondary_generator_{args.model}.json"

    if not rebuild_baselines() and out_csv.exists() and out_json.exists():
        print(f"[phase4.secondary_gen:{args.model}] reproduce mode — committed "
              f"{out_csv.name} and {out_json.name} present; skipping re-run "
              "(set REBUILD_BASELINES=1 to regenerate)")
        return

    client = get_anthropic_client()

    print(f"Secondary-generator validation — {model_id}")
    print("=" * 72)

    rows: list[dict] = []
    rows.extend(gen_semantic_sample(client, model_id, n_per_subcat=4))
    rows.extend(gen_unsupported_sample(client, model_id, n=12))
    rows.extend(gen_hybrid_sample(client, model_id, n=12))

    if not rows:
        print("\nERROR: no prompts generated. Check API key and network.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\nGenerated {len(df)} prompts across {df['top_category'].nunique()} top categories.")
    print(df["top_category"].value_counts().to_string())

    df = run_router(df)
    acc = df["correct"].mean()
    print(f"\nOverall router accuracy: {acc:.3f}")
    for cat in sorted(df["top_category"].unique()):
        sub = df[df["top_category"] == cat]
        print(f"  {cat:<22}  n={len(sub):>3}  acc={sub['correct'].mean():.3f}")

    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "generator": model_id,
            "n_total": int(len(df)),
            "overall_accuracy": float(acc),
            "per_category_accuracy": {
                cat: {
                    "n": int((df["top_category"] == cat).sum()),
                    "accuracy": float(df[df["top_category"] == cat]["correct"].mean()),
                }
                for cat in sorted(df["top_category"].unique())
            },
            "notes": (
                "Prompts generated with the selected Claude model using the "
                "identical scenario-conditioning prompt template as the "
                "GPT-5.2 pipeline. Router is the calibrated TF-IDF + LR "
                "model trained only on GPT-5.2-generated data."
            ),
        }, f, indent=2)

    print(f"\nWrote {out_csv.name} and {out_json.name}")


if __name__ == "__main__":
    main()
