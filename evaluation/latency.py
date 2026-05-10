"""
Uncached latency measurement: fresh API calls for true latency.
"""
import time as timer

import numpy as np
import pandas as pd

from evaluation.system_runner import (
    _deterministic_eval,
    _reconstruct_conv,
    format_conversation,
)


def measure_uncached_latency(
    eval_with_llm,
    llm_client,
    routers,
    tau,
    check_fns,
    eval_system,
    eval_user,
    Message,
    Conversation,
    datetime,
    results_e2e_dir,
):
    """Measure true uncached latency on all evaluation pairs.

    Parameters
    ----------
    eval_with_llm : pd.DataFrame
        Evaluation dataset with LLM results.
    llm_client : openai.OpenAI
        OpenAI client.
    routers : dict
        Router dict.
    tau : float
        Fallback confidence threshold.
    check_fns : dict
        Deterministic check functions.
    eval_system, eval_user : str
        Evaluation prompt templates.
    Message, Conversation : classes
        Data model classes.
    datetime : module
        datetime.datetime for timestamp parsing.
    results_e2e_dir : str
        Directory to save latency.csv.

    Returns
    -------
    pd.DataFrame
        Latency statistics DataFrame.
    """
    print("=" * 80)
    print(f"UNCACHED LATENCY MEASUREMENT (ALL {len(eval_with_llm)} pairs)")
    print("=" * 80)

    print("\nSystem A (LLM-only, fresh calls)...")
    _llm_latencies = []
    _llm_errors = 0
    for _i, (_, _row) in enumerate(eval_with_llm.iterrows()):
        _conv_text = format_conversation(_row["conversation_json"])
        _t0 = timer.time()
        try:
            llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": eval_system},
                    {"role": "user", "content": eval_user.format(conversation=_conv_text, prompt=_row["prompt"])},
                ],
                temperature=0.0, max_completion_tokens=10,
            )
            _llm_latencies.append((timer.time() - _t0) * 1000)
        except Exception as _exc:
            _llm_errors += 1
            print(f"  [error {_i+1}] {type(_exc).__name__}: {_exc}")
        if (_i + 1) % 50 == 0:
            print(f"  {_i+1}/{len(eval_with_llm)}")
    if _llm_errors:
        print(f"  {_llm_errors} API errors excluded from latency stats")

    # Measure router-based latency (deterministic part is real, LLM part uses measured average)
    _prompts = eval_with_llm["prompt"].tolist()
    _llm_mean = np.mean(_llm_latencies)

    _latency_rows = []

    # LLM-only stats
    _latency_rows.append({
        "system": "LLM-only",
        "mean_ms": round(np.mean(_llm_latencies), 1),
        "p50_ms": round(np.median(_llm_latencies), 1),
        "p95_ms": round(np.percentile(_llm_latencies, 95), 1),
        "p99_ms": round(np.percentile(_llm_latencies, 99), 1),
        "min_ms": round(np.min(_llm_latencies), 1),
        "max_ms": round(np.max(_llm_latencies), 1),
    })
    print(f"  LLM-only: mean={_latency_rows[-1]['mean_ms']}ms p50={_latency_rows[-1]['p50_ms']}ms p95={_latency_rows[-1]['p95_ms']}ms")

    # Each router
    for _rkey, _router in routers.items():
        _predicted, _pred_confs = _router["classify"](_prompts)
        _tau = tau if _router.get("calibrated", False) else 0.0
        _router_latencies = []

        for _i, (_, _row) in enumerate(eval_with_llm.iterrows()):
            _pred = _predicted[_i]
            if _pred_confs[_i] < _tau:
                _pred = "SEMANTIC"  # fallback
            if _pred in ("SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"):
                _t0 = timer.time()
                _conv = _reconstruct_conv(_row["conversation_json"], Message, Conversation, datetime)
                _deterministic_eval(_conv, _pred, _row["params"], check_fns, invalid_answer="no")
                _lat = (timer.time() - _t0) * 1000
            elif _pred == "UNSUPPORTED":
                _lat = 0.01  # instant reject
            else:
                _lat = _llm_latencies[_i]  # same LLM call as System A
            _router_latencies.append(_lat)

        _latency_rows.append({
            "system": f"Router ({_router['name']})",
            "mean_ms": round(np.mean(_router_latencies), 1),
            "p50_ms": round(np.median(_router_latencies), 1),
            "p95_ms": round(np.percentile(_router_latencies, 95), 1),
            "p99_ms": round(np.percentile(_router_latencies, 99), 1),
            "min_ms": round(np.min(_router_latencies), 1),
            "max_ms": round(np.max(_router_latencies), 1),
        })
        print(f"  {_router['name']}: mean={_latency_rows[-1]['mean_ms']}ms p50={_latency_rows[-1]['p50_ms']}ms p95={_latency_rows[-1]['p95_ms']}ms")

    _lat_df = pd.DataFrame(_latency_rows)
    _lat_df.to_csv(f"{results_e2e_dir}/latency.csv", index=False)
    print(f"\nSaved latency.csv")

    return _lat_df
