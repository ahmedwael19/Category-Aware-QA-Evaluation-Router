"""
Full system comparison: Router + Deterministic vs LLM-only.
"""
import json
import time as _time

import numpy as np
import pandas as pd

from evaluation.system_runner import (
    _deterministic_eval,
    _reconstruct_conv,
    format_conversation,
)


def run_full_comparison(
    eval_with_llm,
    routers,
    tau,
    check_fns,
    parse_fn,
    make_cache_key_fn,
    llm_client,
    eval_system,
    eval_user,
    Message,
    Conversation,
    datetime,
    cache_path,
    results_e2e_dir,
    invalid_answer="invalid",
    max_retries=3,
    cache_only=False,
):
    """Run all router-based systems on the full evaluation dataset.

    For each (prompt, conversation) pair and each router:
    1. Router classifies the prompt
    2. If SYMBOLIC -> deterministic function (0 tokens, <1ms)
    3. If SEMANTIC/HYBRID -> GPT-4o-mini (same model as System A)
    4. If UNSUPPORTED -> reject (answer = "reject")
    5. Compare answer to ground truth

    Parameters
    ----------
    eval_with_llm : pd.DataFrame
        Evaluation dataset with LLM baseline results.
    routers : dict
        Router dict with keys like 'tfidf_lr'.
    tau : float
        Fallback confidence threshold.
    check_fns : dict
        Deterministic check functions.
    parse_fn : callable
        ``parse_llm_answer`` from answer_parser.
    make_cache_key_fn : callable
        Cache key generator.
    llm_client : openai.OpenAI
        OpenAI client.
    eval_system, eval_user : str
        Evaluation prompt templates.
    Message, Conversation : classes
        Data model classes.
    datetime : module
        datetime.datetime for timestamp parsing.
    cache_path : str
        Path to LLM response cache JSON file.
    results_e2e_dir : str
        Directory to save full_comparison.csv.
    invalid_answer : str
        Label for invalid responses.
    max_retries : int
        Max LLM call retries.

    Returns
    -------
    pd.DataFrame
        system_results DataFrame with all router results.
    """
    import time as timer

    # Load LLM cache
    try:
        with open(cache_path) as _f:
            _llm_cache = json.load(_f)
    except (FileNotFoundError, json.JSONDecodeError):
        _llm_cache = {}

    def _call_llm_cached(prompt_text, conv_text, cache_key):
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]
        if cache_only:
            raise RuntimeError(
                f"cache miss for gpt-4o-mini/{cache_key!r} in reproduce mode; "
                "set REBUILD_E2E=1 to allow fresh API calls"
            )
        for _attempt in range(max_retries):
            try:
                _resp = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": eval_system},
                        {"role": "user", "content": eval_user.format(conversation=conv_text, prompt=prompt_text)},
                    ],
                    temperature=0.0, max_completion_tokens=10,
                )
                _raw = _resp.choices[0].message.content
                _answer = parse_fn(_raw)
                _tokens = _resp.usage.total_tokens if _resp.usage else 0
                if _answer == invalid_answer and _attempt < max_retries - 1:
                    _time.sleep(1)
                    continue
                _result = {"answer": _answer, "tokens": _tokens}
                _llm_cache[cache_key] = _result
                return _result
            except Exception as _e:
                print(f"    LLM error: {_e}")
                _time.sleep(2)
        return {"answer": invalid_answer, "tokens": 0}

    # ── Run each router-based system ──────────────────────────────────
    _all_system_results = []
    _prompts = eval_with_llm["prompt"].tolist()

    for _rkey, _router in routers.items():
        print(f"\nSystem B ({_router['name']})...")
        _predicted_cats, _confs = _router["classify"](_prompts)
        _tau = tau if _router.get("calibrated", False) else 0.0

        _system_answers = []
        _system_tokens = []
        _system_latencies = []
        _n_fallback = 0

        for _i, (_, _row) in enumerate(eval_with_llm.iterrows()):
            _pred_cat = _predicted_cats[_i]
            _conf = _confs[_i]
            _conv_json = _row["conversation_json"]

            # Apply confidence fallback: low confidence -> send to LLM (safe default)
            if _conf < _tau:
                _pred_cat = "SEMANTIC"  # fallback to LLM path
                _n_fallback += 1

            # Route based on (possibly overridden) category
            if _pred_cat in ("SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"):
                _t0 = timer.time()
                _conv = _reconstruct_conv(_conv_json, Message, Conversation, datetime)
                _answer = _deterministic_eval(_conv, _pred_cat, _row["params"], check_fns)
                _latency = (timer.time() - _t0) * 1000
                _tokens = 0
            elif _pred_cat == "UNSUPPORTED":
                _answer = "reject"
                _latency = 0.0
                _tokens = 0
            else:
                # SEMANTIC, HYBRID, or fallback -> LLM evaluation
                _conv_text = format_conversation(_conv_json)
                _result = _call_llm_cached(
                    _row["prompt"], _conv_text,
                    make_cache_key_fn("gpt-4o-mini", _row["prompt"], _row["conversation_json"], "baseline_binary"),
                )
                _latency = _row["system_a_binary_latency_ms"]
                _answer = _result["answer"]
                _tokens = _result["tokens"]

            _system_answers.append(_answer)
            _system_tokens.append(_tokens)
            _system_latencies.append(_latency)

        # Add to results
        _gt = eval_with_llm["ground_truth"].tolist()
        _correct = [1 if _system_answers[_j] == _gt[_j] else 0 for _j in range(len(_gt))]

        for _j in range(len(eval_with_llm)):
            _all_system_results.append({
                "idx": _j,
                "router": _router["name"],
                "prompt": eval_with_llm.iloc[_j]["prompt"],
                "predicted_category": _predicted_cats[_j],
                "true_category": eval_with_llm.iloc[_j]["category"],
                "answer": _system_answers[_j],
                "ground_truth": _gt[_j],
                "correct": _correct[_j],
                "tokens": _system_tokens[_j],
                "latency_ms": round(_system_latencies[_j], 2),
            })

        _acc = sum(_correct) / len(_correct)
        _total_tokens = sum(_system_tokens)
        _mean_latency = np.mean(_system_latencies)
        print(f"  Accuracy: {_acc:.3f} | Tokens: {_total_tokens:,} | Latency: {_mean_latency:.1f}ms/prompt | Fallback: {_n_fallback}")

    # Save cache
    with open(cache_path, "w") as _f:
        json.dump(_llm_cache, _f)

    system_results = pd.DataFrame(_all_system_results)
    system_results.to_csv(f"{results_e2e_dir}/full_comparison.csv", index=False)
    print(f"\nSaved full_comparison.csv ({len(system_results)} rows)")

    return system_results
