import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    import re
    import random
    import time as timer
    from pathlib import Path
    from datetime import datetime, timedelta
    from dataclasses import dataclass, asdict, field

    return (
        Path, asdict, dataclass, datetime, field, json, mo, np, pd, random, re,
        timedelta, timer,
    )


@app.cell
def evaluation_policy():
    """Label policy and per-category evaluation routing."""
    VALID_ANSWERS = {"yes", "no"}
    REJECT_ANSWER = "reject"

    DETERMINISTIC_CATEGORIES = {"SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"}
    LLM_CATEGORIES = {"SEMANTIC", "HYBRID"}
    REJECT_CATEGORIES = {"UNSUPPORTED"}

    INVALID_ANSWER = "invalid"
    MAX_RETRIES = 3

    print("Evaluation policies defined:")
    print(f"  Valid answers: {VALID_ANSWERS}")
    print(f"  Deterministic categories: {DETERMINISTIC_CATEGORIES}")
    print(f"  LLM categories: {LLM_CATEGORIES}")
    print(f"  Reject categories: {REJECT_CATEGORIES}")
    return (
        DETERMINISTIC_CATEGORIES, INVALID_ANSWER, LLM_CATEGORIES,
        MAX_RETRIES, REJECT_ANSWER, REJECT_CATEGORIES, VALID_ANSWERS,
    )


@app.cell
def shared_utils():
    """Answer parser and cache-key helper."""
    import hashlib as _hashlib
    from evaluation.answer_parser import parse_llm_answer

    def make_cache_key(model, prompt, conv_json, role):
        """Content-based cache key for LLM response caching."""
        _content = f"{prompt}:{conv_json}"
        _h = _hashlib.md5(_content.encode()).hexdigest()[:12]
        return f"{role}_{model}_{_h}"

    print("Shared utils ready: parse_llm_answer, make_cache_key")
    return make_cache_key, parse_llm_answer


@app.cell
def title(mo):
    mo.md("""
    # End-to-End Evaluation Pipeline

    This notebook implements:
    1. **Conversation data model** — simple dataclasses representing QA conversations
    2. **Deterministic evaluation functions** — time_check, tag_check, count_check
    3. **Synthetic conversation generation** — conversations with controlled properties
    4. **Ground truth computation** — deterministic for SYMBOLIC, GPT-5.2 for SEMANTIC
    5. **System comparison** — LLM-only vs Router+Deterministic vs Keyword router
    6. **Statistical analysis** — McNemar's test, bootstrap CIs, reliability (variance)
    """)
    return


# ── PART 1: Data Model ────────────────────────────────────────────────────
@app.cell
def data_model():
    """Conversation data model."""
    from evaluation.deterministic import Conversation, Message
    print("Data model imported: Message, Conversation")
    return Conversation, Message


# ── PART 2: Deterministic Evaluation Functions ────────────────────────────
@app.cell
def deterministic_functions():
    """Deterministic evaluation functions."""
    from evaluation.deterministic import (
        check_channel, check_internal_notes_exist, check_message_count,
        check_response_time, check_tag_present,
        count_messages, first_response_time_minutes,
    )
    print("Deterministic functions loaded")
    return (
        check_channel, check_internal_notes_exist, check_message_count,
        check_response_time, check_tag_present,
        count_messages, first_response_time_minutes,
    )


# ── PART 3: Conversation Generator ────────────────────────────────────────
@app.cell
def conversation_generator():
    """Synthetic conversation generator."""
    from evaluation.conversation_generator import generate_conversation
    print("Conversation generator loaded")
    return (generate_conversation,)


# ── PART 4: Evaluation Dataset Construction ───────────────────────────────
@app.cell
def build_eval_dataset(pd):
    """Load or rebuild the 305-pair evaluation benchmark.

    Default (reproduce) mode loads the committed `data/evaluation_dataset.csv`
    produced for the thesis so results match exactly. Set REBUILD_EVAL=1 to
    regenerate the dataset from the test split and the full synthetic pool
    under the polarity-admission rule (see evaluation/dataset_builder.py).
    """
    from config import DATASET_FULL, DATASET_TEST, EVAL_DATASET, SEED
    from thesis_router import rebuild_eval_dataset as _rebuild_eval_dataset

    if _rebuild_eval_dataset():
        print("[phase3.eval] REBUILD_EVAL=1 — rebuilding evaluation_dataset.csv")
        from evaluation.dataset_builder import build_evaluation_dataset
        _df_test = pd.read_csv(DATASET_TEST)
        _df_full = pd.read_csv(DATASET_FULL)
        print(f"  Test split: {len(_df_test)} prompts  |  full pool: {len(_df_full)} prompts")
        eval_dataset = build_evaluation_dataset(
            df_test=_df_test,
            df_full=_df_full,
            out_path=EVAL_DATASET,
            seed=SEED,
        )
    else:
        print("[phase3.eval] reproduce mode — loading committed evaluation_dataset.csv "
              "(set REBUILD_EVAL=1 to rebuild)")
        eval_dataset = pd.read_csv(EVAL_DATASET)
        print(f"  Loaded {len(eval_dataset)} evaluation pairs")
    return (eval_dataset,)


@app.cell
def dataset_balance(eval_dataset, mo, pd):
    """Check ground truth balance for SYMBOLIC categories."""
    _sym = eval_dataset[eval_dataset["ground_truth_source"] == "deterministic"]
    _balance = _sym.groupby(["category", "ground_truth"]).size().unstack(fill_value=0)

    mo.md(f"""
    ### Ground Truth Balance (SYMBOLIC only)

    {_balance.to_markdown()}

    A balanced dataset ensures that a system predicting all "yes" or all "no" cannot achieve high accuracy.
    """)
    return


# ── PART 5: Train ALL Routers (from saved training data) ──────────────────
@app.cell
def train_routers(Path, json, np, pd, re):
    """Train all router models fresh from the saved training data.

    We train here rather than loading joblib files to:
    1. Avoid coupling to specific saved model versions
    2. Guarantee exact same training procedure as the benchmark
    3. Include ALL models, not just the ones that happened to be saved
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sentence_transformers import SentenceTransformer

    _dir = Path(__file__).parent.parent
    _data_dir = _dir / "data"
    _results_dir = _dir / "results"

    # Load training data
    _train_csv = sorted(_data_dir.glob("synthetic_final_*_train.csv"))[-1]
    _train_df = pd.read_csv(_train_csv)
    _num_pat = re.compile(r'\d+')
    _train_prompts = _train_df["prompt"].apply(lambda t: _num_pat.sub("<NUM>", t))
    _train_labels = _train_df["top_category"]

    print("Training all routers from saved training data...")

    # TF-IDF + LR
    _tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    _X_train_tfidf = _tfidf_vec.fit_transform(_train_prompts)
    _tfidf_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    _tfidf_lr.fit(_X_train_tfidf, _train_labels)
    print("  TF-IDF + LR trained")

    # TF-IDF + SVM
    _tfidf_svm = LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
    _tfidf_svm.fit(_X_train_tfidf, _train_labels)
    print("  TF-IDF + SVM trained")

    # Embedding + LR
    _emb_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    _emb_path = _dir / "models" / "embeddings" / "train.npy"
    if _emb_path.exists():
        _emb_train = np.load(_emb_path)
        print("  Loaded cached training embeddings")
    else:
        _emb_train = _emb_model.encode(
            [f"query: {t}" for t in _train_df["prompt"]], batch_size=64, normalize_embeddings=True,
        )
        print("  Computed training embeddings")
    _emb_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    _emb_lr.fit(_emb_train, _train_labels)
    print("  Emb + LR trained")

    # Ensemble (calibrated TF-IDF + calibrated Emb, averaged)
    _cal_tfidf = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42),
        method="sigmoid", cv=3,
    )
    _cal_tfidf.fit(_X_train_tfidf, _train_labels)
    _cal_emb = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42),
        method="sigmoid", cv=3,
    )
    _cal_emb.fit(_emb_train, _train_labels)
    print("  Ensemble (calibrated TF-IDF + Emb) trained")

    # Keyword classifier
    from router.keyword_baseline import keyword_router

    # ── Determine fallback threshold τ from validation set ───────────
    _val_csv = sorted(_data_dir.glob("synthetic_final_*_val.csv"))[-1]
    _val_df = pd.read_csv(_val_csv)
    _val_prompts_masked = _val_df["prompt"].apply(lambda t: _num_pat.sub("<NUM>", t))
    _val_labels = _val_df["top_category"]

    # Calibrate TF-IDF+LR on val set and find τ that gives ≥99% accuracy on routed prompts
    _val_X = _tfidf_vec.transform(_val_prompts_masked)
    _val_probs = _cal_tfidf.predict_proba(_val_X)
    _val_preds = _cal_tfidf.classes_[np.argmax(_val_probs, axis=1)]
    _val_confs = np.max(_val_probs, axis=1)

    _best_tau = 0.5
    for _tau_candidate in np.arange(0.5, 0.95, 0.05):
        _mask = _val_confs >= _tau_candidate
        if _mask.sum() > 0:
            _acc = (_val_preds[_mask] == _val_labels.values[_mask]).mean()
            _cov = _mask.mean()
            if _acc >= 0.99 and _cov >= 0.80:
                _best_tau = _tau_candidate
    TAU = round(_best_tau, 2)
    print(f"\n  Fallback threshold τ = {TAU} (determined on validation set)")

    # ── Router packaging ──────────────────────────────────────────────
    # Each router returns (predictions, confidences).
    # In the end-to-end eval, if confidence < τ → fallback to LLM.

    def _classify_tfidf_lr(prompts):
        _X = _tfidf_vec.transform([_num_pat.sub("<NUM>", p) for p in prompts])
        _probs = _cal_tfidf.predict_proba(_X)
        _preds = _cal_tfidf.classes_[np.argmax(_probs, axis=1)]
        _confs = np.max(_probs, axis=1)
        return _preds, _confs

    def _classify_tfidf_svm(prompts):
        _X = _tfidf_vec.transform([_num_pat.sub("<NUM>", p) for p in prompts])
        _preds = _tfidf_svm.predict(_X)
        # SVM has no predict_proba — set confidence to 1.0 (no fallback)
        return _preds, np.ones(len(prompts))

    def _classify_emb_lr(prompts):
        _embs = _emb_model.encode([f"query: {p}" for p in prompts], batch_size=64, normalize_embeddings=True)
        _probs = _cal_emb.predict_proba(_embs)
        _preds = _cal_emb.classes_[np.argmax(_probs, axis=1)]
        _confs = np.max(_probs, axis=1)
        return _preds, _confs

    def _classify_ensemble(prompts):
        _X_tf = _tfidf_vec.transform([_num_pat.sub("<NUM>", p) for p in prompts])
        _embs = _emb_model.encode([f"query: {p}" for p in prompts], batch_size=64, normalize_embeddings=True)
        _probs = (_cal_tfidf.predict_proba(_X_tf) + _cal_emb.predict_proba(_embs)) / 2
        _preds = _cal_tfidf.classes_[np.argmax(_probs, axis=1)]
        _confs = np.max(_probs, axis=1)
        return _preds, _confs

    def _classify_keyword(prompts):
        _preds = np.array([keyword_router(p) for p in prompts])
        return _preds, np.ones(len(prompts))  # no confidence for keyword

    routers = {
        "tfidf_lr": {"classify": _classify_tfidf_lr, "name": "TF-IDF+LR", "calibrated": True},
        "tfidf_svm": {"classify": _classify_tfidf_svm, "name": "TF-IDF+SVM", "calibrated": False},
        "emb_lr": {"classify": _classify_emb_lr, "name": "Emb+LR", "calibrated": True},
        "ensemble": {"classify": _classify_ensemble, "name": "Ensemble", "calibrated": True},
        "keyword": {"classify": _classify_keyword, "name": "Keyword", "calibrated": False},
    }

    print(f"All routers ready: {', '.join(r['name'] for r in routers.values())}")
    print(f"Calibrated routers use fallback threshold τ = {TAU}")
    return TAU, routers


# ── PART 6: System Comparison (deterministic eval dataset only — no LLM calls) 
@app.cell
def system_comparison(
    Conversation, Message,
    check_channel, check_internal_notes_exist, check_message_count,
    check_response_time, check_tag_present,
    datetime, eval_dataset, json, mo, np, pd, routers, TAU, timer,
):
    """Compare router accuracy on the SYMBOLIC subset (deterministic ground truth)."""
    from pathlib import Path as _Path
    from evaluation.system_runner import run_deterministic_eval

    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")

    _check_fns = {
        "check_response_time": check_response_time,
        "check_tag_present": check_tag_present,
        "check_channel": check_channel,
        "check_internal_notes_exist": check_internal_notes_exist,
        "check_message_count": check_message_count,
    }

    _router_results = run_deterministic_eval(
        eval_dataset=eval_dataset,
        routers=routers,
        tau=TAU,
        check_fns=_check_fns,
        Message=Message,
        Conversation=Conversation,
        datetime=datetime,
    )

    _sym = eval_dataset[eval_dataset["ground_truth_source"] == "deterministic"]
    pd.DataFrame(_router_results).to_csv(f"{_results_e2e}/routing_accuracy.csv", index=False)
    print(f"\nSaved results_e2e_routing_accuracy.csv")

    mo.md(f"""
    ### Symbolic Routing Analysis (deterministic subset, N={len(_sym)})

    Correct routing = correct evaluation (deterministic functions are 100% correct by construction).

    | Router | Accuracy | Misrouted | Time (ms/prompt) |
    |--------|----------|-----------|-----------------|
    {"".join(f"| {r['router']} | {r['accuracy']:.3f} | {r['n_misrouted']} | {r['route_time_ms']:.3f} |{chr(10)}" for r in _router_results)}

    **Reliability**: Deterministic functions achieve **zero variance** across 5 repeated runs on all {len(_sym)} pairs.
    This establishes the deterministic reliability result for symbolic evaluation; LLM reliability is evaluated separately below.
    """)
    return


# ── PART 7: GPT-5.2 Ground Truth + LLM-only Baseline ──────────────────────
@app.cell
def setup_llm():
    """Construct the OpenAI client.

    In reproduce mode (REBUILD_E2E unset) no LLM calls are issued — the
    committed caches + results under results/e2e/ are authoritative. We still
    build the client because downstream cells expect the `llm_client` symbol,
    but any cache miss in reproduce mode is treated as a hard error.
    """
    from thesis_router import get_openai_client, rebuild_e2e as _rebuild_e2e
    if _rebuild_e2e():
        print("[phase3.e2e] REBUILD_E2E=1 — fresh API calls will be made on cache misses")
        llm_client = get_openai_client()
    else:
        print("[phase3.e2e] reproduce mode — cache-only; cache misses will error "
              "(set REBUILD_E2E=1 to allow fresh API calls)")
        # Build a no-op placeholder so downstream cells have `llm_client`. We don't
        # expect any of the downstream code to actually call it; they're gated on
        # _rebuild_e2e() or load their inputs from committed CSVs.
        llm_client = None
    return (llm_client,)


@app.cell
def prompt_templates():
    """Prompt templates."""
    from evaluation.prompt_templates import (
        EVAL_REJECT_SYSTEM, EVAL_REJECT_USER,
        EVAL_SYSTEM, EVAL_USER,
        GT_SYSTEM, GT_USER,
        VERSION as PROMPT_TEMPLATES_VERSION,
    )
    print(f"Prompt templates imported ({PROMPT_TEMPLATES_VERSION})")
    return EVAL_REJECT_SYSTEM, EVAL_REJECT_USER, EVAL_SYSTEM, EVAL_USER, GT_SYSTEM, GT_USER, PROMPT_TEMPLATES_VERSION


@app.cell
def llm_evaluation(
    EVAL_REJECT_SYSTEM, EVAL_REJECT_USER, EVAL_SYSTEM, EVAL_USER, GT_SYSTEM, GT_USER,
    INVALID_ANSWER, MAX_RETRIES, REJECT_ANSWER,
    eval_dataset, json, llm_client, make_cache_key, mo, np, parse_llm_answer, pd, timer, datetime,
):
    """Generate GPT-5.2 ground truth + GPT-4o-mini baseline answers.

    Reproduce mode (default) loads the committed `data/evaluation_with_llm.csv`
    without hitting the API. Rebuild mode (REBUILD_E2E=1) regenerates it,
    using the committed cache where available and making fresh calls for
    cache misses.
    """
    from pathlib import Path as _Path
    from thesis_router import rebuild_e2e as _rebuild_e2e

    _dir = _Path(__file__).parent.parent
    _data_dir = str(_dir / "data")
    _results_e2e = str(_dir / "results" / "e2e")
    _cache_path = f"{_results_e2e}/llm_cache.json"
    _eval_llm_csv = f"{_data_dir}/evaluation_with_llm.csv"

    if not _rebuild_e2e():
        print("[phase3.llm_eval] reproduce mode — loading committed "
              "data/evaluation_with_llm.csv")
        eval_with_llm = pd.read_csv(_eval_llm_csv)
    else:
        from evaluation.system_runner import run_llm_evaluation
        eval_with_llm = run_llm_evaluation(
            eval_dataset=eval_dataset,
            llm_client=llm_client,
            gt_system=GT_SYSTEM,
            gt_user=GT_USER,
            eval_system=EVAL_SYSTEM,
            eval_user=EVAL_USER,
            eval_reject_system=EVAL_REJECT_SYSTEM,
            eval_reject_user=EVAL_REJECT_USER,
            parse_fn=parse_llm_answer,
            make_cache_key_fn=make_cache_key,
            cache_path=_cache_path,
            invalid_answer=INVALID_ANSWER,
            max_retries=MAX_RETRIES,
            reject_answer=REJECT_ANSWER,
        )

        eval_with_llm.to_csv(_eval_llm_csv, index=False)
        print(f"\nSaved evaluation_with_llm.csv ({len(eval_with_llm)} rows)")

    return (eval_with_llm,)


# ── PART 8: Full System Comparison (Router + Deterministic vs LLM-only) ───
@app.cell
def full_comparison(
    Conversation, Message,
    EVAL_SYSTEM, EVAL_USER, INVALID_ANSWER, MAX_RETRIES,
    check_channel, check_internal_notes_exist, check_message_count,
    check_response_time, check_tag_present,
    datetime, eval_with_llm, json, llm_client, make_cache_key, mo, np, parse_llm_answer, pd, routers, TAU, timer,
):
    """Run router-based systems on the full evaluation dataset.

    Reproduce mode loads the committed `results/e2e/full_comparison.csv`;
    rebuild mode re-runs every router, using the LLM cache where available
    and (only if REBUILD_E2E=1) making fresh calls for cache misses.
    """
    from pathlib import Path as _Path
    from thesis_router import rebuild_e2e as _rebuild_e2e

    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")
    _cache_path = f"{_results_e2e}/llm_cache.json"
    _out_csv = f"{_results_e2e}/full_comparison.csv"

    if not _rebuild_e2e():
        print("[phase3.full_comparison] reproduce mode — loading committed "
              "results/e2e/full_comparison.csv")
        system_results = pd.read_csv(_out_csv)
    else:
        from evaluation.comparison import run_full_comparison
        _check_fns = {
            "check_response_time": check_response_time,
            "check_tag_present": check_tag_present,
            "check_channel": check_channel,
            "check_internal_notes_exist": check_internal_notes_exist,
            "check_message_count": check_message_count,
        }
        system_results = run_full_comparison(
            eval_with_llm=eval_with_llm,
            routers=routers,
            tau=TAU,
            check_fns=_check_fns,
            parse_fn=parse_llm_answer,
            make_cache_key_fn=make_cache_key,
            llm_client=llm_client,
            eval_system=EVAL_SYSTEM,
            eval_user=EVAL_USER,
            Message=Message,
            Conversation=Conversation,
            datetime=datetime,
            cache_path=_cache_path,
            results_e2e_dir=_results_e2e,
            invalid_answer=INVALID_ANSWER,
            max_retries=MAX_RETRIES,
        )
        print(f"\nSaved results_e2e_full_comparison.csv ({len(system_results)} rows)")

    return (system_results,)


# ── PART 9: Reliability Test (5 repeated LLM runs on SYMBOLIC subset) ─────
@app.cell
def reliability_test(
    EVAL_SYSTEM, EVAL_USER,
    eval_with_llm, json, llm_client, mo, np, parse_llm_answer, pd, timer,
):
    """Run LLM-only 5 times on SYMBOLIC subset to measure variance.

    Skipped in reproduce mode — the committed results/e2e/reliability.csv
    and reliability_summary.csv are authoritative. Set REBUILD_E2E=1 to
    re-run (10 × symbolic-N ≈ 500 fresh API calls).
    """
    from pathlib import Path as _Path
    from thesis_router import rebuild_e2e as _rebuild_e2e

    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")

    if not _rebuild_e2e():
        print("[phase3.reliability] skipped in reproduce mode; committed "
              "results/e2e/reliability{,_summary}.csv are authoritative")
    else:
        from evaluation.reliability import run_reliability_test
        _rel_df, _var_df, _md_rows = run_reliability_test(
            eval_with_llm=eval_with_llm,
            llm_client=llm_client,
            eval_system=EVAL_SYSTEM,
            eval_user=EVAL_USER,
            parse_fn=parse_llm_answer,
            results_e2e_dir=_results_e2e,
        )
        _sample_n = len(_var_df[_var_df["temperature"] == 0.0])
        _ = mo.md(f"""
        ### Reliability Comparison (5 repeated runs on {_sample_n} SYMBOLIC prompts)

        | System | Temp | Inconsistent | Mean Correct (/5) | Accuracy |
        |--------|------|-------------|-------------------|----------|
        | LLM (GPT-4o-mini) | 0.0 | {_md_rows[0]['inconsistent']}/{_md_rows[0]['total']} ({_md_rows[0]['inconsistent']/_md_rows[0]['total']*100:.1f}%) | {_md_rows[0]['mean_correct']:.2f} | {_md_rows[0]['acc']:.3f} |
        | LLM (GPT-4o-mini) | 0.3 | {_md_rows[1]['inconsistent']}/{_md_rows[1]['total']} ({_md_rows[1]['inconsistent']/_md_rows[1]['total']*100:.1f}%) | {_md_rows[1]['mean_correct']:.2f} | {_md_rows[1]['acc']:.3f} |
        | **Deterministic** | — | 0/{_sample_n} (0.0%) | 5.00 | 1.000 |
        """)
    return


# ── PART 9b: Uncached Latency Measurement (ALL pairs, fresh API calls) ────
@app.cell
def uncached_latency(
    Conversation, Message,
    EVAL_SYSTEM, EVAL_USER,
    check_channel, check_internal_notes_exist, check_message_count,
    check_response_time, check_tag_present,
    datetime, eval_with_llm, json, llm_client, mo, np, pd, routers, TAU, timer,
):
    """Measure uncached latency on ALL evaluation pairs.

    Skipped in reproduce mode — the committed results/e2e/latency.csv and
    latency_measured.json are authoritative. Set REBUILD_E2E=1 to re-run
    (one fresh API call per evaluation pair).
    """
    from pathlib import Path as _Path
    from thesis_router import rebuild_e2e as _rebuild_e2e

    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")

    if not _rebuild_e2e():
        print("[phase3.latency] skipped in reproduce mode; committed "
              "results/e2e/latency{.csv,_measured.json} are authoritative")
    else:
        from evaluation.latency import measure_uncached_latency
        _check_fns = {
            "check_response_time": check_response_time,
            "check_tag_present": check_tag_present,
            "check_channel": check_channel,
            "check_internal_notes_exist": check_internal_notes_exist,
            "check_message_count": check_message_count,
        }
        _lat_df = measure_uncached_latency(
            eval_with_llm=eval_with_llm,
            llm_client=llm_client,
            routers=routers,
            tau=TAU,
            check_fns=_check_fns,
            eval_system=EVAL_SYSTEM,
            eval_user=EVAL_USER,
            Message=Message,
            Conversation=Conversation,
            datetime=datetime,
            results_e2e_dir=_results_e2e,
        )
        _ = mo.md(f"""
        ### Uncached Latency (fresh API calls on all {len(eval_with_llm)} pairs)

        {_lat_df.to_markdown(index=False)}

        Deterministic evaluations complete in <1ms. LLM calls dominate latency.
        Router systems are faster because symbolic prompts skip the LLM entirely.
        """)
    return


# ── PART 10: Statistical Analysis ─────────────────────────────────────────
@app.cell
def statistical_analysis(
    eval_with_llm, json, mo, np, pd, system_results,
):
    """McNemar's test, Cohen's h effect size, bootstrap CIs (overall + per-category)."""
    from pathlib import Path as _Path
    from evaluation.statistics import compute_all_statistics

    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")

    _stats = compute_all_statistics(
        eval_with_llm=eval_with_llm,
        system_results=system_results,
        results_e2e_dir=_results_e2e,
    )

    _summary_df = _stats["summary_df"]

    mo.md(f"""
    ### End-to-End Results

    {_summary_df.to_markdown(index=False)}

    **McNemar's test** compares each router system against the LLM-only baseline.
    Significance levels: * p<0.05, ** p<0.01, *** p<0.001
    """)
    return


# ── PART 11: Error Analysis ───────────────────────────────────────────────
@app.cell
def error_analysis_e2e(
    eval_with_llm, json, mo, np, pd, system_results,
):
    """Qualitative error analysis for the end-to-end systems."""
    from pathlib import Path as _Path

    _dir = str(_Path(__file__).parent.parent)
    _data_dir = str(_Path(__file__).parent.parent / "data")
    _results_e2e = str(_Path(__file__).parent.parent / "results" / "e2e")

    # ── System A (LLM-only) errors on SYMBOLIC prompts ────────────────
    _sym_a = eval_with_llm[eval_with_llm["ground_truth_source"] == "deterministic"]
    _sym_a_errors = _sym_a[_sym_a["system_a_binary_correct"] == 0]

    _md = "### Error Analysis\n\n"
    _md += f"**System A (LLM-only) errors on SYMBOLIC prompts**: {len(_sym_a_errors)}/{len(_sym_a)} ({len(_sym_a_errors)/len(_sym_a)*100:.1f}%)\n\n"

    if len(_sym_a_errors) > 0:
        _md += "| Category | Scenario | Ground Truth | LLM Answer | Prompt (truncated) |\n"
        _md += "|---|---|---|---|---|\n"
        for _, _row in _sym_a_errors.head(15).iterrows():
            _p = _row["prompt"][:60].replace("|", "\\|")
            _md += f"| {_row['category']} | {_row['scenario']} | {_row['ground_truth']} | {_row['system_a_binary_answer']} | {_p} |\n"

    # ── System A errors on UNSUPPORTED (hallucination) ────────────────
    _unsup = eval_with_llm[eval_with_llm["ground_truth_source"] == "rejection"]
    _unsup_binary = _unsup[_unsup["system_a_binary_answer"] != "reject"]
    _unsup_reject = _unsup[_unsup["system_a_reject_answer"] != "reject"]
    _md += f"\n**System A on UNSUPPORTED**:\n"
    _md += f"- Binary (yes/no only): {len(_unsup_binary)}/{len(_unsup)} answered instead of rejecting (0% rejection by design)\n"
    _md += f"- With reject option: {len(_unsup_reject)}/{len(_unsup)} failed to reject ({len(_unsup) - len(_unsup_reject)}/{len(_unsup)} correctly rejected)\n"

    # ── Per-category error rates for best router ──────────────────────
    _best_router = system_results.groupby("router")["correct"].mean().idxmax()
    _best_sub = system_results[system_results["router"] == _best_router]

    _md += f"\n**Best router ({_best_router}) errors by category**:\n\n"
    _md += "| Category | N | Errors | Error Rate |\n|---|---|---|---|\n"
    for _cat in sorted(_best_sub["true_category"].unique()):
        _cat_sub = _best_sub[_best_sub["true_category"] == _cat]
        _n = len(_cat_sub)
        _errors = (_cat_sub["correct"] == 0).sum()
        _md += f"| {_cat} | {_n} | {_errors} | {_errors/_n*100:.1f}% |\n"

    # ── Misrouting analysis: where does the router send wrong? ────────
    _misrouted = _best_sub[_best_sub["predicted_category"] != _best_sub["true_category"]]
    if len(_misrouted) > 0:
        _md += f"\n**Misrouting patterns ({_best_router})**: {len(_misrouted)} misrouted prompts\n\n"
        _md += "| True → Predicted | Count |\n|---|---|\n"
        _confusion = _misrouted.groupby(["true_category", "predicted_category"]).size().reset_index(name="count")
        _confusion = _confusion.sort_values("count", ascending=False)
        for _, _row in _confusion.head(10).iterrows():
            _md += f"| {_row['true_category']} → {_row['predicted_category']} | {_row['count']} |\n"

    # ── Save error examples ───────────────────────────────────────────
    _error_examples = []
    # LLM failures on symbolic
    for _, _row in _sym_a_errors.iterrows():
        _error_examples.append({
            "system": "LLM-only", "category": _row["category"],
            "scenario": _row["scenario"], "ground_truth": _row["ground_truth"],
            "system_answer": _row["system_a_binary_answer"],
            "prompt": _row["prompt"][:100], "error_type": "symbolic_llm_failure",
        })
    # Router misrouting
    for _, _row in _misrouted.iterrows():
        _error_examples.append({
            "system": f"Router ({_best_router})", "category": _row["true_category"],
            "scenario": "misrouted", "ground_truth": _row["ground_truth"],
            "system_answer": _row["answer"],
            "prompt": _row["prompt"][:100] if "prompt" in _row else "",
            "error_type": f"misroute_{_row['true_category']}_to_{_row['predicted_category']}",
        })

    pd.DataFrame(_error_examples).to_csv(f"{_results_e2e}/error_analysis.csv", index=False)
    print(f"Saved results_e2e_error_analysis.csv ({len(_error_examples)} error cases)")

    mo.md(_md)
    return


if __name__ == "__main__":
    app.run()
