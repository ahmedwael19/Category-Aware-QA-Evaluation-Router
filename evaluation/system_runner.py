"""
Run evaluation systems: deterministic (router-based) and LLM baselines.
"""
import json
import time as _time

import numpy as np
import pandas as pd


def format_conversation(conv_json_str):
    """Format a conversation JSON into a readable transcript for the LLM."""
    _d = json.loads(conv_json_str)
    _lines = []
    for _m in _d["messages"]:
        _role = _m["role"].upper()
        _pub = "" if _m["is_public"] else " [INTERNAL NOTE]"
        _ts = _m["timestamp"][:19]
        _lines.append(f"[{_ts}] {_role}{_pub}: {_m['text']}")
    _meta = f"Channel: {_d.get('channel', 'unknown')} | Tags: {', '.join(_d.get('tags', []))}"
    return _meta + "\n\n" + "\n".join(_lines)


def _reconstruct_conv(conv_json_str, Message, Conversation, datetime):
    """Reconstruct Conversation from JSON."""
    _d = json.loads(conv_json_str)
    _msgs = [
        Message(
            role=m["role"], text=m["text"],
            timestamp=datetime.fromisoformat(m["timestamp"]),
            is_public=m["is_public"], channel=m["channel"],
        )
        for m in _d["messages"]
    ]
    return Conversation(
        messages=_msgs, tags=_d.get("tags", []),
        channel=_d.get("channel", "chat"),
        resolution_time_minutes=_d.get("resolution_time_minutes"),
    )


def _deterministic_eval(conv, category, params, check_fns, invalid_answer="invalid"):
    """Run the deterministic function for a given category.

    Used in the full_comparison path where misrouted non-symbolic prompts
    may arrive with missing params.
    """
    check_response_time = check_fns["check_response_time"]
    check_message_count = check_fns["check_message_count"]
    check_tag_present = check_fns["check_tag_present"]
    check_channel = check_fns["check_channel"]
    check_internal_notes_exist = check_fns["check_internal_notes_exist"]

    # Guard: if params is missing/NaN (misrouted non-symbolic prompt), can't evaluate
    if params is None or (isinstance(params, float) and np.isnan(params)):
        return invalid_answer
    if category == "SYMBOLIC_TIME":
        _p = json.loads(params) if isinstance(params, str) else params
        if not isinstance(_p, dict) or "threshold" not in _p:
            return invalid_answer
        _threshold = float(_p["threshold"])
        _unit = _p.get("unit", "minutes")
        _operator = _p.get("operator")
        if _operator is None:
            raise ValueError(f"SYMBOLIC_TIME params missing explicit operator: {_p}")
        if _unit in ("seconds", "sec"):
            _threshold = _threshold / 60.0
        elif _unit in ("hours", "hrs"):
            _threshold = _threshold * 60.0
        return check_response_time(conv, _threshold, _operator)
    elif category == "SYMBOLIC_COUNT":
        _p = json.loads(params) if isinstance(params, str) else params
        if not isinstance(_p, dict) or "threshold" not in _p:
            return invalid_answer
        _operator = _p.get("operator")
        if _operator is None:
            raise ValueError(f"SYMBOLIC_COUNT params missing explicit operator: {_p}")
        return check_message_count(conv, int(_p["threshold"]), _operator)
    elif category == "SYMBOLIC_METADATA":
        _p = json.loads(params) if isinstance(params, str) else params
        _subtype = _p.get("subtype", "tag")
        if _subtype == "tag":
            return check_tag_present(conv, _p.get("value", ""))
        elif _subtype == "channel":
            return check_channel(conv, _p.get("value", ""))
        elif _subtype == "notes":
            return check_internal_notes_exist(conv)
    return "no"


def run_deterministic_eval(
    eval_dataset,
    routers,
    tau,
    check_fns,
    Message,
    Conversation,
    datetime,
):
    """Run router-based evaluation on the SYMBOLIC subset.

    Verifies deterministic functions reproduce ground truth, then evaluates
    each router's routing accuracy.

    Parameters
    ----------
    eval_dataset : pd.DataFrame
        Full evaluation dataset.
    routers : dict
        Router dict with keys like 'tfidf_lr', each having 'classify', 'name', 'calibrated'.
    tau : float
        Fallback confidence threshold.
    check_fns : dict
        Deterministic check functions.
    Message, Conversation : classes
        Data model classes from evaluation.deterministic.
    datetime : module
        datetime.datetime for timestamp parsing.

    Returns
    -------
    list[dict]
        Router accuracy results.
    """
    import time as timer

    _sym = eval_dataset[eval_dataset["ground_truth_source"] == "deterministic"].copy()
    print(f"Evaluating on {len(_sym)} SYMBOLIC evaluation pairs")

    # ── Verify deterministic functions reproduce ground truth ──────────
    _prompts = _sym["prompt"].tolist()
    _correct_count = 0
    for _, _row in _sym.iterrows():
        _conv = _reconstruct_conv(_row["conversation_json"], Message, Conversation, datetime)
        _result = _deterministic_eval(_conv, _row["category"], _row["params"], check_fns)
        if _result == _row["ground_truth"]:
            _correct_count += 1
        else:
            print(f"  MISMATCH: expected={_row['ground_truth']} got={_result} prompt={_row['prompt'][:60]}")

    print(f"\nDeterministic function verification: {_correct_count}/{len(_sym)} correct ({_correct_count/len(_sym)*100:.1f}%)")
    assert _correct_count == len(_sym), f"Deterministic functions must be 100% correct! Got {_correct_count}/{len(_sym)}"

    # ── Evaluate each router on SYMBOLIC prompts ──────────────────────
    print(f"\n{'='*80}")
    print("ROUTER ACCURACY ON SYMBOLIC EVALUATION PAIRS")
    print(f"{'='*80}")

    _router_results = []
    for _router_key, _router in routers.items():
        _t0 = timer.time()
        _predicted_cats, _confs = _router["classify"](_prompts)
        _route_time = timer.time() - _t0

        # Apply fallback: low confidence -> treat as SEMANTIC (send to LLM, safe default)
        _tau = tau if _router.get("calibrated", False) else 0.0
        _n_fallback = 0
        _final_cats = []
        for _j in range(len(_predicted_cats)):
            if _confs[_j] < _tau:
                _final_cats.append("SEMANTIC")  # fallback to LLM
                _n_fallback += 1
            else:
                _final_cats.append(_predicted_cats[_j])

        # Routing accuracy
        _total = len(_sym)
        _correct = sum(1 for _i, (_, _row) in enumerate(_sym.iterrows()) if _final_cats[_i] == _row["category"])

        _router_results.append({
            "router": _router["name"],
            "accuracy": round(_correct / _total, 4),
            "route_time_ms": round(_route_time / _total * 1000, 4),
            "n_pairs": _total,
            "n_correct": _correct,
            "n_misrouted": _total - _correct,
            "n_fallback": _n_fallback,
            "tau": _tau,
        })
        print(f"  {_router['name']:<20} acc={_correct/_total:.3f} ({_correct}/{_total})  fallback={_n_fallback}  time={_route_time/_total*1000:.3f}ms/prompt")

    # ── Reliability test: 5 repeated runs ─────────────────────────────
    print(f"\n{'='*80}")
    print("RELIABILITY TEST: 5 repeated runs on SYMBOLIC pairs")
    print(f"{'='*80}")

    _reliability = []
    for _, _row in _sym.iterrows():
        _conv = _reconstruct_conv(_row["conversation_json"], Message, Conversation, datetime)
        _results_per_run = []
        for _run in range(5):
            _result = _deterministic_eval(_conv, _row["category"], _row["params"], check_fns)
            _results_per_run.append(_result)
        _variance = len(set(_results_per_run))  # 1 = all same, >1 = variance
        _reliability.append({"prompt": _row["prompt"][:60], "category": _row["category"], "unique_answers": _variance})

    _zero_variance = sum(1 for r in _reliability if r["unique_answers"] == 1)
    print(f"  Deterministic: {_zero_variance}/{len(_reliability)} pairs with zero variance ({_zero_variance/len(_reliability)*100:.1f}%)")
    assert _zero_variance == len(_reliability), "Deterministic functions MUST have zero variance!"
    print("  CONFIRMED: 100% deterministic -- zero variance across all runs")

    return _router_results


def run_llm_evaluation(
    eval_dataset,
    llm_client,
    gt_system,
    gt_user,
    eval_system,
    eval_user,
    eval_reject_system,
    eval_reject_user,
    parse_fn,
    make_cache_key_fn,
    cache_path,
    invalid_answer="invalid",
    max_retries=3,
    reject_answer="reject",
    save_interval=25,
    cache_only=False,
):
    """Generate ground truth (GPT-5.2) and baseline answers (GPT-4o-mini).

    For each evaluation pair:
    - GPT-5.2 judges the ground truth for SEMANTIC/HYBRID prompts
    - GPT-4o-mini evaluates ALL prompts (System A baseline)
    - Results saved incrementally

    Parameters
    ----------
    eval_dataset : pd.DataFrame
        Evaluation dataset from build_evaluation_dataset.
    llm_client : openai.OpenAI
        OpenAI client.
    gt_system, gt_user : str
        Ground truth prompt templates (GPT-5.2).
    eval_system, eval_user : str
        Binary evaluation prompt templates (GPT-4o-mini).
    eval_reject_system, eval_reject_user : str
        Ternary evaluation prompt templates (GPT-4o-mini with reject).
    parse_fn : callable
        ``parse_llm_answer(raw_text)`` from answer_parser.
    make_cache_key_fn : callable
        ``make_cache_key(model, prompt, conv_json, role)`` for caching.
    cache_path : str
        Path to LLM response cache JSON file.
    invalid_answer : str
        Label for invalid/unparseable responses.
    max_retries : int
        Max LLM call retries.
    reject_answer : str
        Expected answer for UNSUPPORTED prompts.
    save_interval : int
        Save cache every N calls.

    Returns
    -------
    pd.DataFrame
        eval_with_llm DataFrame with all LLM results.
    """
    import time as timer

    # Load cache
    try:
        with open(cache_path) as _f:
            _cache = json.load(_f)
        print(f"Loaded LLM cache: {len(_cache)} entries")
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = {}
        print("No LLM cache found, starting fresh")

    def _call_llm(model, system_prompt, user_prompt, cache_key):
        """Call LLM with caching, retries, and answer normalization.

        When `cache_only=True`, cache misses raise `RuntimeError` instead of
        hitting the API — used in reproduce mode so the notebook errors
        loudly if the committed cache does not cover the current evaluation
        dataset.
        """
        if cache_key in _cache:
            return _cache[cache_key]
        if cache_only:
            raise RuntimeError(
                f"cache miss for {model}/{cache_key!r} in reproduce mode; "
                "set REBUILD_E2E=1 to allow fresh API calls"
            )

        for _attempt in range(max_retries):
            try:
                _resp = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=10,
                )
                _raw = _resp.choices[0].message.content
                _answer = parse_fn(_raw)
                _tokens = _resp.usage.total_tokens if _resp.usage else 0

                # If invalid, retry (don't cache invalid)
                if _answer == invalid_answer and _attempt < max_retries - 1:
                    print(f"    Invalid response '{_raw}', retrying...")
                    _time.sleep(1)
                    continue

                _result = {"answer": _answer, "tokens": _tokens, "raw": _raw.strip()[:50]}
                _cache[cache_key] = _result
                return _result
            except Exception as _e:
                print(f"    LLM error (attempt {_attempt+1}): {_e}")
                _time.sleep(2)
        _result = {"answer": invalid_answer, "tokens": 0, "raw": "error_after_retries"}
        _cache[cache_key] = _result
        return _result

    # ── Process all evaluation pairs ──────────────────────────────────
    _results = []
    _total = len(eval_dataset)

    print(f"Processing {_total} evaluation pairs...")

    for _i, (_, _row) in enumerate(eval_dataset.iterrows()):
        _conv_text = format_conversation(_row["conversation_json"])
        _prompt = _row["prompt"]
        _cat = _row["category"]

        # GPT-5.2 ground truth (only for SEMANTIC/HYBRID where ground truth is pending)
        _gt = _row["ground_truth"]
        _gt_tokens = 0
        # pandas converts None to NaN -- check with pd.isna, not `is None`
        if _row["ground_truth_source"] == "gpt5.2_judge" and (pd.isna(_gt) or _gt is None):
            _gt_result = _call_llm(
                "gpt-5.2", gt_system,
                gt_user.format(conversation=_conv_text, prompt=_prompt),
                make_cache_key_fn("gpt-5.2", _prompt, _row["conversation_json"], "gt"),
            )
            _gt = _gt_result["answer"]
            _gt_tokens = _gt_result.get("tokens", 0)

        # GPT-4o-mini baseline A1: binary (yes/no only)
        _t0 = timer.time()
        _baseline_result = _call_llm(
            "gpt-4o-mini", eval_system,
            eval_user.format(conversation=_conv_text, prompt=_prompt),
            make_cache_key_fn("gpt-4o-mini", _prompt, _row["conversation_json"], "baseline_binary"),
        )
        _baseline_time = timer.time() - _t0

        # GPT-4o-mini baseline A2: ternary (yes/no/reject)
        _t1 = timer.time()
        _reject_result = _call_llm(
            "gpt-4o-mini", eval_reject_system,
            eval_reject_user.format(conversation=_conv_text, prompt=_prompt),
            make_cache_key_fn("gpt-4o-mini", _prompt, _row["conversation_json"], "baseline_reject"),
        )
        _reject_time = timer.time() - _t1

        _results.append({
            "idx": _i,
            "prompt": _prompt,
            "category": _cat,
            "ground_truth": _gt,
            "ground_truth_source": _row["ground_truth_source"],
            "scenario": _row["scenario"],
            "system_a_binary_answer": _baseline_result["answer"],
            "system_a_binary_tokens": _baseline_result["tokens"],
            "system_a_binary_latency_ms": round(_baseline_time * 1000, 1),
            "system_a_binary_correct": 1 if _baseline_result["answer"] == _gt else 0,
            "system_a_reject_answer": _reject_result["answer"],
            "system_a_reject_tokens": _reject_result["tokens"],
            "system_a_reject_latency_ms": round(_reject_time * 1000, 1),
            "system_a_reject_correct": 1 if _reject_result["answer"] == _gt else 0,
            "gt_tokens": _gt_tokens,
            "conversation_json": _row["conversation_json"],
            "params": _row.get("params"),
        })

        if (_i + 1) % save_interval == 0:
            with open(cache_path, "w") as _f:
                json.dump(_cache, _f)
            print(f"  Processed {_i+1}/{_total} ({(_i+1)/_total*100:.0f}%)")

    # Final cache save
    with open(cache_path, "w") as _f:
        json.dump(_cache, _f)

    eval_with_llm = pd.DataFrame(_results)
    print(f"\nCompleted LLM evaluation ({len(eval_with_llm)} rows)")

    # Summary
    _sym_mask = eval_with_llm["ground_truth_source"] == "deterministic"
    _sem_mask = eval_with_llm["ground_truth_source"] == "gpt5.2_judge"
    _unsup_mask = eval_with_llm["ground_truth_source"] == "rejection"

    print(f"\nSystem A (LLM-only, binary yes/no):")
    print(f"  SYMBOLIC: {eval_with_llm.loc[_sym_mask, 'system_a_binary_correct'].mean():.3f} ({_sym_mask.sum()} pairs)")
    print(f"  SEMANTIC/HYBRID: {eval_with_llm.loc[_sem_mask, 'system_a_binary_correct'].mean():.3f} ({_sem_mask.sum()} pairs)")
    print(f"  UNSUPPORTED: {eval_with_llm.loc[_unsup_mask, 'system_a_binary_correct'].mean():.3f} ({_unsup_mask.sum()} pairs)")
    print(f"  Overall: {eval_with_llm['system_a_binary_correct'].mean():.3f}")

    print(f"\nSystem A (LLM-only, with reject option):")
    print(f"  SYMBOLIC: {eval_with_llm.loc[_sym_mask, 'system_a_reject_correct'].mean():.3f} ({_sym_mask.sum()} pairs)")
    print(f"  SEMANTIC/HYBRID: {eval_with_llm.loc[_sem_mask, 'system_a_reject_correct'].mean():.3f} ({_sem_mask.sum()} pairs)")
    print(f"  UNSUPPORTED: {eval_with_llm.loc[_unsup_mask, 'system_a_reject_correct'].mean():.3f} ({_unsup_mask.sum()} pairs)")
    print(f"  Overall: {eval_with_llm['system_a_reject_correct'].mean():.3f}")

    return eval_with_llm
