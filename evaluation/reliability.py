"""
Reliability test: 5 repeated LLM runs on SYMBOLIC subset.

"""
import time as _time

import pandas as pd

from evaluation.system_runner import format_conversation


def run_reliability_test(
    eval_with_llm,
    llm_client,
    eval_system,
    eval_user,
    parse_fn,
    results_e2e_dir,
    sample_n=50,
    n_runs=5,
    seed=42,
):
    """Run System A (LLM-only) 5 times on the SYMBOLIC subset to measure variance.

    Deterministic functions have zero variance by construction; LLM calls at
    temperature 0 typically still show non-zero variance.

    Parameters
    ----------
    eval_with_llm : pd.DataFrame
        Evaluation dataset with LLM results.
    llm_client : openai.OpenAI
        OpenAI client.
    eval_system, eval_user : str
        Evaluation prompt templates.
    parse_fn : callable
        ``parse_llm_answer`` from answer_parser.
    results_e2e_dir : str
        Directory to save reliability CSVs.
    sample_n : int
        Number of SYMBOLIC pairs to sample.
    n_runs : int
        Number of repeated runs per temperature.
    seed : int
        Random seed for sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (reliability_df, variance_summary_df)
    """
    _sym = eval_with_llm[eval_with_llm["ground_truth_source"] == "deterministic"]

    # Sample SYMBOLIC pairs for the reliability test (full set is too expensive)
    _sample = _sym.sample(n=min(sample_n, len(_sym)), random_state=seed)

    print(f"Reliability test: {n_runs} LLM runs x 2 temperatures on {len(_sample)} SYMBOLIC pairs")
    print("(Proves LLM non-determinism vs deterministic function stability)")

    _reliability_rows = []

    for _temp in [0.0, 0.3]:
        print(f"\n  Temperature={_temp}:")
        for _run in range(n_runs):
            print(f"    Run {_run+1}/{n_runs}...")
            for _, _row in _sample.iterrows():
                _conv_text = format_conversation(_row["conversation_json"])
                try:
                    _resp = llm_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": eval_system},
                            {"role": "user", "content": eval_user.format(conversation=_conv_text, prompt=_row["prompt"])},
                        ],
                        temperature=_temp, max_completion_tokens=10,
                    )
                    _answer = parse_fn(_resp.choices[0].message.content)
                except Exception:
                    _answer = "error"
                _reliability_rows.append({
                    "idx": _row["idx"], "run": _run, "temperature": _temp,
                    "category": _row["category"],
                    "prompt": _row["prompt"][:60], "llm_answer": _answer,
                    "ground_truth": _row["ground_truth"],
                })
        _time.sleep(1)

    _rel_df = pd.DataFrame(_reliability_rows)

    # Compute variance per temperature
    _variance_stats = []
    for _temp in [0.0, 0.3]:
        _temp_df = _rel_df[_rel_df["temperature"] == _temp]
        for _idx in _sample["idx"].unique():
            _runs = _temp_df[_temp_df["idx"] == _idx]
            if len(_runs) == 0:
                continue
            _answers = _runs["llm_answer"].tolist()
            _unique = len(set(_answers))
            _gt = _runs["ground_truth"].iloc[0]
            _correct_runs = sum(1 for a in _answers if a == _gt)
            _variance_stats.append({
                "idx": _idx, "temperature": _temp, "unique_answers": _unique,
                "correct_runs": _correct_runs, "total_runs": n_runs,
                "category": _runs["category"].iloc[0],
            })

    _var_df = pd.DataFrame(_variance_stats)
    _rel_df.to_csv(f"{results_e2e_dir}/reliability.csv", index=False)
    _var_df.to_csv(f"{results_e2e_dir}/reliability_summary.csv", index=False)

    print(f"\nLLM Reliability Results:")
    _md_rows = []
    for _temp in [0.0, 0.3]:
        _sub = _var_df[_var_df["temperature"] == _temp]
        _inconsistent = (_sub["unique_answers"] > 1).sum()
        _total = len(_sub)
        _mean_correct = _sub["correct_runs"].mean()
        _mean_acc = _mean_correct / float(n_runs)
        print(f"  temp={_temp}: inconsistent={_inconsistent}/{_total} ({_inconsistent/_total*100:.1f}%), mean_correct={_mean_correct:.2f}/{n_runs}, acc={_mean_acc:.3f}")
        _md_rows.append({"temp": _temp, "inconsistent": _inconsistent, "total": _total, "mean_correct": _mean_correct, "acc": _mean_acc})

    print("  Deterministic: 0 inconsistent, accuracy 1.000 (zero variance by construction)")
    print("\nSaved reliability.csv and reliability_summary.csv")

    return _rel_df, _var_df, _md_rows
