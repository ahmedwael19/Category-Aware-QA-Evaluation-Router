"""Reasoning-model comparison: o1, o3-mini, and GPT-4o on the same 50 symbolic
prompts, 5 repetitions each, measuring accuracy, consistency, tokens, and
latency.

For reasoning models, `max_completion_tokens` covers both the hidden reasoning
trace and the visible output; 4096 is sized so the final answer is not truncated
by the reasoning budget.

Run: python -m scripts.baselines.reasoning_model_comparison
"""

import sys
import time

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR
from evaluation.answer_parser import parse_llm_answer
from evaluation.system_runner import format_conversation as format_conv
from thesis_router import get_openai_client


EVAL_SYSTEM = """You are evaluating a customer support conversation against a quality criteria prompt.
Answer ONLY "yes" or "no". Nothing else."""

EVAL_USER = """CONVERSATION:
{conversation}

EVALUATION PROMPT: {prompt}

Does this conversation satisfy the evaluation criteria? Answer only "yes" or "no"."""

# Reasoning models charge for hidden reasoning tokens inside
# `max_completion_tokens`; standard models only need room for "yes" / "no".
MODELS = {
    "o3-mini": {"model": "o3-mini", "reasoning": True, "max_tokens": 4096},
    "o1": {"model": "o1", "reasoning": True, "max_tokens": 4096},
    "gpt-4o": {"model": "gpt-4o", "reasoning": False, "max_tokens": 10},
}

N_RUNS = 5


def _reasoning_tokens(usage) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def _call(client, config, conv_text, prompt):
    messages = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": EVAL_USER.format(conversation=conv_text, prompt=prompt)},
    ]
    kwargs = {
        "model": config["model"],
        "messages": messages,
        "max_completion_tokens": config["max_tokens"],
    }
    if not config["reasoning"]:
        kwargs["temperature"] = 0.0
    return client.chat.completions.create(**kwargs)


def main() -> None:
    from thesis_router import rebuild_baselines

    out_dir = RESULTS_DIR / "reasoning"
    raw_path = out_dir / "raw.csv"
    summary_path = out_dir / "summary.csv"

    if not rebuild_baselines():
        if raw_path.exists() and summary_path.exists():
            print("[phase4.reasoning] reproduce mode — committed "
                  "results/reasoning/{raw,summary}.csv present; skipping re-run "
                  "(set REBUILD_BASELINES=1 to regenerate)")
            return
        print("[phase4.reasoning] reproduce mode, but committed results missing — "
              "proceeding as if REBUILD_BASELINES=1")

    client = get_openai_client()

    eval_df = pd.read_csv(DATA_DIR / "evaluation_with_llm.csv")
    sym = eval_df[eval_df["ground_truth_source"] == "deterministic"]
    sample = sym.sample(n=min(50, len(sym)), random_state=42)
    n_prompts = len(sample)
    total_api_calls = len(MODELS) * N_RUNS * n_prompts
    print(f"Loaded {n_prompts} symbolic prompts")
    print(f"  Categories: {sample['category'].value_counts().to_dict()}")
    print(f"Total API calls planned: {total_api_calls} "
          f"({len(MODELS)} models x {N_RUNS} runs x {n_prompts} prompts)")

    all_rows: list[dict] = []
    global_call_num = 0
    experiment_start = time.time()

    for model_idx, (model_name, config) in enumerate(MODELS.items(), 1):
        model_start = time.time()
        print(f"\n[{model_idx}/{len(MODELS)}] Model: {model_name}  "
              f"(reasoning={config['reasoning']}, max_tokens={config['max_tokens']})")

        test_row = sample.iloc[0]
        try:
            test_resp = _call(client, config, format_conv(test_row["conversation_json"]), test_row["prompt"])
            test_answer = parse_llm_answer(test_resp.choices[0].message.content)
            tu = test_resp.usage
            print(f"  Test OK: answer='{test_answer}'  "
                  f"prompt_tok={tu.prompt_tokens}  completion_tok={tu.completion_tokens}  "
                  f"reasoning_tok={_reasoning_tokens(tu)}")
        except Exception as e:
            print(f"  FATAL: {model_name} not available: {e}. SKIPPING.")
            continue

        for run in range(N_RUNS):
            run_start = time.time()
            run_correct = run_tokens_total = run_latency_total = run_errors = 0

            for prompt_num, (_, row) in enumerate(sample.iterrows(), 1):
                global_call_num += 1
                prompt_start = time.time()
                try:
                    resp = _call(client, config, format_conv(row["conversation_json"]), row["prompt"])
                    raw = resp.choices[0].message.content
                    answer = parse_llm_answer(raw)
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    reasoning_tokens = _reasoning_tokens(usage)
                    latency_ms = (time.time() - prompt_start) * 1000
                    if answer == "invalid":
                        run_errors += 1
                        print(f"    WARN [{global_call_num}/{total_api_calls}] "
                              f"idx={row['idx']}: parsed 'invalid' from: '{raw[:80]}'")
                except Exception as e:
                    latency_ms = (time.time() - prompt_start) * 1000
                    run_errors += 1
                    print(f"    ERROR [{global_call_num}/{total_api_calls}] idx={row['idx']}: {e}")
                    answer = "error"
                    prompt_tokens = completion_tokens = reasoning_tokens = 0

                is_correct = int(answer == row["ground_truth"])
                run_correct += is_correct
                run_tokens_total += prompt_tokens + completion_tokens
                run_latency_total += latency_ms

                all_rows.append({
                    "model": model_name,
                    "run": run,
                    "idx": row["idx"],
                    "category": row["category"],
                    "prompt": row["prompt"][:80],
                    "answer": answer,
                    "ground_truth": row["ground_truth"],
                    "correct": is_correct,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "latency_ms": round(latency_ms, 1),
                })

                if prompt_num % 10 == 0 or prompt_num == n_prompts:
                    elapsed_total = time.time() - experiment_start
                    rate = global_call_num / elapsed_total if elapsed_total > 0 else 0
                    eta_min = ((total_api_calls - global_call_num) / rate / 60) if rate > 0 else 0.0
                    print(f"    [{model_name}] run {run+1}/{N_RUNS}  "
                          f"{prompt_num}/{n_prompts}  acc={run_correct/prompt_num:.0%}  "
                          f"[{global_call_num}/{total_api_calls}]  ETA {eta_min:.1f}min")
                    sys.stdout.flush()

            print(f"  >> Run {run+1} DONE: acc={run_correct/n_prompts:.1%}  "
                  f"tokens/prompt={run_tokens_total/n_prompts:.0f}  "
                  f"latency/prompt={run_latency_total/n_prompts:.0f}ms  "
                  f"errors={run_errors}  wall={time.time()-run_start:.0f}s")
            sys.stdout.flush()
            time.sleep(1)

        model_rows = [r for r in all_rows if r["model"] == model_name]
        model_acc = np.mean([r["correct"] for r in model_rows]) if model_rows else 0
        print(f"\n  >> {model_name} COMPLETE: overall_acc={model_acc:.1%}  "
              f"wall={(time.time()-model_start)/60:.1f}min")

    print(f"\nAll API calls complete: {global_call_num} in "
          f"{(time.time()-experiment_start)/60:.1f} minutes")

    df = pd.DataFrame(all_rows)
    (RESULTS_DIR / "reasoning").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "reasoning" / "raw.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'reasoning' / 'raw.csv'} ({len(df)} rows)")

    summary_rows = []
    for model_name in df["model"].unique():
        m = df[df["model"] == model_name]
        run_accs = m.groupby("run")["correct"].mean()
        inconsistent = sum(1 for idx in m["idx"].unique() if m[m["idx"] == idx]["answer"].nunique() > 1)
        summary_rows.append({
            "model": model_name,
            "mean_accuracy": round(run_accs.mean(), 4),
            "std_accuracy": round(run_accs.std(), 4),
            "inconsistent_prompts": inconsistent,
            "inconsistent_pct": round(inconsistent / m["idx"].nunique(), 4),
            "mean_total_tokens": round(m["total_tokens"].mean(), 1),
            "mean_reasoning_tokens": round(m["reasoning_tokens"].mean(), 1),
            "mean_latency_ms": round(m["latency_ms"].mean(), 1),
            "p50_latency_ms": round(m["latency_ms"].median(), 1),
            "p95_latency_ms": round(m["latency_ms"].quantile(0.95), 1),
        })

    # GPT-4o-mini reference from the existing reliability + evaluation artefacts.
    rel = pd.read_csv(RESULTS_DIR / "e2e" / "reliability.csv")
    rel_t0 = rel[rel["temperature"] == 0.0]
    mini_run_accs = rel_t0.groupby("run").apply(lambda g: (g["llm_answer"] == g["ground_truth"]).mean())
    mini_inconsistent = sum(
        1 for idx in rel_t0["idx"].unique() if rel_t0[rel_t0["idx"] == idx]["llm_answer"].nunique() > 1
    )
    eval_full = pd.read_csv(DATA_DIR / "evaluation_with_llm.csv")
    eval_sym_sample = eval_full[eval_full["idx"].isin(sample["idx"])]
    mini_tokens = eval_sym_sample["system_a_binary_tokens"].mean()
    mini_latency = eval_sym_sample["system_a_binary_latency_ms"]

    summary_rows.append({
        "model": "gpt-4o-mini (temp=0)",
        "mean_accuracy": round(mini_run_accs.mean(), 4),
        "std_accuracy": round(mini_run_accs.std(), 4),
        "inconsistent_prompts": mini_inconsistent,
        "inconsistent_pct": round(mini_inconsistent / rel_t0["idx"].nunique(), 4),
        "mean_total_tokens": round(mini_tokens, 1),
        "mean_reasoning_tokens": 0,
        "mean_latency_ms": round(mini_latency.mean(), 1),
        "p50_latency_ms": round(mini_latency.median(), 1),
        "p95_latency_ms": round(mini_latency.quantile(0.95), 1),
    })

    summary_rows.append({
        "model": "deterministic",
        "mean_accuracy": 1.0, "std_accuracy": 0.0,
        "inconsistent_prompts": 0, "inconsistent_pct": 0.0,
        "mean_total_tokens": 0, "mean_reasoning_tokens": 0,
        "mean_latency_ms": 0, "p50_latency_ms": 0, "p95_latency_ms": 0,
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "reasoning" / "summary.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'reasoning' / 'summary.csv'}")

    print(f"\n{'='*90}")
    print(f"REASONING MODEL COMPARISON — SYMBOLIC PROMPTS (N={n_prompts}, {N_RUNS} runs)")
    print(f"{'='*90}")
    print(f"{'Model':<22} {'Accuracy':>10} {'Std':>8} {'Incons.':>8} "
          f"{'Tokens':>8} {'Reason':>8} {'p50 ms':>8} {'p95 ms':>8}")
    print("-" * 90)
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<22} {r['mean_accuracy']:>9.1%} {r['std_accuracy']:>8.3f} "
              f"{r['inconsistent_pct']:>7.1%} {r['mean_total_tokens']:>8.0f} "
              f"{r['mean_reasoning_tokens']:>8.0f} {r['p50_latency_ms']:>8.0f} "
              f"{r['p95_latency_ms']:>8.0f}")


if __name__ == "__main__":
    main()
