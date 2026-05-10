#!/usr/bin/env python3
"""Rerun LLM baselines and router comparison on the regenerated evaluation set.

Steps:
  1. Load evaluation_dataset.csv (post-operator-fix).
  2. For each of the 305 rows, call GPT-4o-mini binary and reject prompts.
     Reuse cached answers from results/e2e/llm_cache.json where possible.
  3. Call GPT-4o binary and reject on the same set (reuse llm_cache_gpt4o.json).
  4. Dispatch via TF-IDF+SVM router; execute symbolic rows deterministically
     (with explicit operators), otherwise reuse the GPT-4o-mini binary answer.
  5. Save per-system outcome CSVs under results/e2e/.

Caches: extends existing llm_cache.json and llm_cache_gpt4o.json. Cache key
        matches the key schema in notebooks/03_evaluation_pipeline.py.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time as timer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_E2E
from evaluation.answer_parser import parse_llm_answer
from evaluation.deterministic import (
    Conversation, Message,
    check_response_time, check_message_count,
    check_tag_present, check_channel, check_internal_notes_exist,
)
from evaluation.prompt_templates import (
    EVAL_SYSTEM, EVAL_USER, EVAL_REJECT_SYSTEM, EVAL_REJECT_USER,
)
from evaluation.system_runner import format_conversation
from thesis_router import get_openai_client

EVAL = DATA_DIR / "evaluation_dataset.csv"
CACHE_MINI = RESULTS_E2E / "llm_cache.json"
CACHE_4O = RESULTS_E2E / "llm_cache_gpt4o.json"
OUT_E2E = RESULTS_E2E


def cache_key_mini(role: str, model: str, prompt: str, conv_json: str) -> str:
    h = hashlib.md5(f"{prompt}:{conv_json}".encode()).hexdigest()[:12]
    return f"{role}_{model}_{h}"


def cache_key_4o(model: str, prompt: str, conv_json: str, role: str) -> str:
    h = hashlib.sha256(f"{model}|{prompt}|{conv_json}|{role}".encode()).hexdigest()[:20]
    return f"{model}:{role}:{h}"


def load_cache(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=1)


def call_llm(client, model: str, system_prompt: str, user_prompt: str) -> dict:
    t0 = timer.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    lat = (timer.time() - t0) * 1000
    content = resp.choices[0].message.content or ""
    return {"answer": content.strip(), "tokens": resp.usage.total_tokens, "latency_ms": round(lat, 1)}


def reconstruct_conv(conv_json: str) -> Conversation:
    d = json.loads(conv_json)
    msgs = [
        Message(
            role=m["role"], text=m["text"],
            timestamp=_dt.datetime.fromisoformat(m["timestamp"]),
            is_public=m.get("is_public", True),
            channel=m.get("channel", d.get("channel", "")),
        )
        for m in d["messages"]
    ]
    return Conversation(
        messages=msgs, tags=d.get("tags", []), channel=d.get("channel", ""),
        resolution_time_minutes=d.get("resolution_time_minutes"),
    )


def deterministic_gt(row: pd.Series) -> str:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
    conv = reconstruct_conv(row["conversation_json"])
    if row["category"] == "SYMBOLIC_TIME":
        thr = float(params["threshold"])
        unit = params.get("unit", "minutes")
        if unit in ("seconds", "sec", "secs"):
            thr /= 60.0
        elif unit in ("hours", "hrs", "hr"):
            thr *= 60.0
        elif unit in ("days", "day"):
            thr *= 1440.0
        return check_response_time(conv, thr, params["operator"])
    if row["category"] == "SYMBOLIC_COUNT":
        return check_message_count(conv, int(params["threshold"]), params["operator"])
    if row["category"] == "SYMBOLIC_METADATA":
        sub = params.get("subtype", "tag")
        if sub == "tag":
            return check_tag_present(conv, params.get("value", ""))
        if sub == "channel":
            return check_channel(conv, params.get("value", ""))
        if sub == "notes":
            return check_internal_notes_exist(conv)
    raise ValueError(row["category"])


def main() -> None:
    from thesis_router import rebuild_e2e

    rescored_csv = OUT_E2E / "baselines_rescored.csv"
    summary_json = OUT_E2E / "e2e_rescored_summary.json"
    if not rebuild_e2e() and rescored_csv.exists() and summary_json.exists():
        print(f"[phase3.rerun_baselines] reproduce mode — committed "
              f"{rescored_csv.name} and {summary_json.name} present; "
              "skipping re-run (set REBUILD_E2E=1 to regenerate)")
        return

    ed = pd.read_csv(EVAL).reset_index(drop=True)
    print(f"Rows: {len(ed)}")

    cache_mini = load_cache(CACHE_MINI)
    cache_4o = load_cache(CACHE_4O)
    client = get_openai_client()

    # 1. GPT-4o-mini binary + reject + GPT-4o binary + reject on all 305 rows.
    for role, system_p, user_t, model, cache, key_fn in [
        ("baseline_binary", EVAL_SYSTEM, EVAL_USER, "gpt-4o-mini", cache_mini,
         lambda p, c: cache_key_mini("baseline_binary", "gpt-4o-mini", p, c)),
        ("baseline_reject", EVAL_REJECT_SYSTEM, EVAL_REJECT_USER, "gpt-4o-mini", cache_mini,
         lambda p, c: cache_key_mini("baseline_reject", "gpt-4o-mini", p, c)),
        ("binary_gpt4o", EVAL_SYSTEM, EVAL_USER, "gpt-4o", cache_4o,
         lambda p, c: cache_key_4o("gpt-4o", p, c, "binary_gpt4o")),
        ("reject_gpt4o", EVAL_REJECT_SYSTEM, EVAL_REJECT_USER, "gpt-4o", cache_4o,
         lambda p, c: cache_key_4o("gpt-4o", p, c, "reject_gpt4o")),
    ]:
        n_hit = n_new = 0
        for i, r in ed.iterrows():
            k = key_fn(r["prompt"], r["conversation_json"])
            if k in cache:
                n_hit += 1
                continue
            conv_text = format_conversation(r["conversation_json"])
            user_p = user_t.format(conversation=conv_text, prompt=r["prompt"])
            out = call_llm(client, model, system_p, user_p)
            cache[k] = out
            n_new += 1
            if (n_new % 20) == 0:
                print(f"  {role}: {n_new} new calls so far")
        print(f"{role} [{model}]: cache hits={n_hit}, fresh={n_new}")
    save_cache(cache_mini, CACHE_MINI)
    save_cache(cache_4o, CACHE_4O)

    # 2. Score baselines against new GT
    rows_out = []
    for i, r in ed.iterrows():
        mini_bin_k = cache_key_mini("baseline_binary", "gpt-4o-mini", r["prompt"], r["conversation_json"])
        mini_rej_k = cache_key_mini("baseline_reject", "gpt-4o-mini", r["prompt"], r["conversation_json"])
        bin4_k = cache_key_4o("gpt-4o", r["prompt"], r["conversation_json"], "binary_gpt4o")
        rej4_k = cache_key_4o("gpt-4o", r["prompt"], r["conversation_json"], "reject_gpt4o")
        rows_out.append({
            "idx": i, "prompt": r["prompt"], "category": r["category"],
            "ground_truth": r["ground_truth"],
            "mini_bin_answer": parse_llm_answer(cache_mini[mini_bin_k]["answer"]),
            "mini_bin_tokens": cache_mini[mini_bin_k]["tokens"],
            "mini_bin_latency": cache_mini[mini_bin_k].get("latency_ms", 0),
            "mini_rej_answer": parse_llm_answer(cache_mini[mini_rej_k]["answer"]),
            "mini_rej_tokens": cache_mini[mini_rej_k]["tokens"],
            "mini_rej_latency": cache_mini[mini_rej_k].get("latency_ms", 0),
            "gpt4o_bin_answer": parse_llm_answer(cache_4o[bin4_k]["answer"]),
            "gpt4o_bin_tokens": cache_4o[bin4_k]["tokens"],
            "gpt4o_rej_answer": parse_llm_answer(cache_4o[rej4_k]["answer"]),
            "gpt4o_rej_tokens": cache_4o[rej4_k]["tokens"],
        })
    baseline_df = pd.DataFrame(rows_out)
    baseline_df["mini_bin_correct"] = (baseline_df.mini_bin_answer == baseline_df.ground_truth).astype(int)
    baseline_df["mini_rej_correct"] = (baseline_df.mini_rej_answer == baseline_df.ground_truth).astype(int)
    baseline_df["gpt4o_bin_correct"] = (baseline_df.gpt4o_bin_answer == baseline_df.ground_truth).astype(int)
    baseline_df["gpt4o_rej_correct"] = (baseline_df.gpt4o_rej_answer == baseline_df.ground_truth).astype(int)
    baseline_df.to_csv(OUT_E2E / "baselines_rescored.csv", index=False)

    print("\n== Baseline accuracies (vs new GT) ==")
    print(f"GPT-4o-mini binary: {baseline_df.mini_bin_correct.mean():.4f}  ({baseline_df.mini_bin_correct.sum()}/305)")
    print(f"GPT-4o-mini reject: {baseline_df.mini_rej_correct.mean():.4f}  ({baseline_df.mini_rej_correct.sum()}/305)")
    print(f"GPT-4o     binary : {baseline_df.gpt4o_bin_correct.mean():.4f}  ({baseline_df.gpt4o_bin_correct.sum()}/305)")
    print(f"GPT-4o     reject : {baseline_df.gpt4o_rej_correct.mean():.4f}  ({baseline_df.gpt4o_rej_correct.sum()}/305)")

    # 3. Router comparison (TF-IDF+SVM, oracle config)
    # Retrain SVM on the training split — the notebook trained it in-memory and
    # never saved it. Vectorizer + encoder are the same joblibs as thesis.
    from config import DATASET_TRAIN, LABEL_ENCODER, TFIDF_VECTORIZER
    from sklearn.svm import LinearSVC
    vec = joblib.load(TFIDF_VECTORIZER)
    enc = joblib.load(LABEL_ENCODER)
    train = pd.read_csv(DATASET_TRAIN)
    X_tr = vec.transform(train["prompt"].tolist())
    y_tr = enc.transform(train["top_category"].tolist())
    svm = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    svm.fit(X_tr, y_tr)
    X = vec.transform(ed["prompt"].tolist())
    preds = enc.inverse_transform(svm.predict(X))
    ed["svm_predicted"] = preds

    router_correct = []
    router_answers = []
    for i, r in ed.iterrows():
        pred = r["svm_predicted"]
        if pred in ("SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"):
            if pred == r["category"]:
                try:
                    ans = deterministic_gt(r)
                except Exception:
                    ans = "invalid"
            else:
                # Misrouted symbolic: router assigned wrong symbolic type.
                # We cannot execute with the wrong params schema; count as wrong.
                ans = "invalid"
        elif pred == "UNSUPPORTED":
            ans = "reject"
        else:
            mini_bin_k = cache_key_mini("baseline_binary", "gpt-4o-mini", r["prompt"], r["conversation_json"])
            ans = parse_llm_answer(cache_mini[mini_bin_k]["answer"])
        router_answers.append(ans)
        router_correct.append(int(ans == r["ground_truth"]))
    ed["svm_answer"] = router_answers
    ed["svm_correct"] = router_correct
    print(f"\nRouter (TF-IDF+SVM oracle): {np.mean(router_correct):.4f}  ({sum(router_correct)}/305)")

    # 4. Deployed router: regex extraction on symbolic + LLM fallback
    def regex_ok(row) -> bool:
        """Return True if threshold/operator info appears literally in prompt text."""
        if not str(row["category"]).startswith("SYMBOLIC"):
            return False
        if row["category"] == "SYMBOLIC_METADATA":
            # METADATA: value must appear in prompt text (case-insensitive)
            params = json.loads(row["params"])
            val = str(params.get("value", "")).lower()
            return val in row["prompt"].lower() if val else False
        params = json.loads(row["params"])
        thr = str(params["threshold"])
        # threshold must be in the prompt text
        return thr in row["prompt"]

    ed["extract_ok"] = ed.apply(regex_ok, axis=1)

    deployed_ans = []
    deployed_correct = []
    for i, r in ed.iterrows():
        pred = r["svm_predicted"]
        if pred in ("SYMBOLIC_TIME", "SYMBOLIC_COUNT", "SYMBOLIC_METADATA"):
            if r["extract_ok"] and pred == r["category"]:
                try:
                    ans = deterministic_gt(r)
                except Exception:
                    # Fallback to LLM if deterministic dispatch fails
                    mini_bin_k = cache_key_mini("baseline_binary", "gpt-4o-mini", r["prompt"], r["conversation_json"])
                    ans = parse_llm_answer(cache_mini[mini_bin_k]["answer"])
            else:
                # extraction failed -> LLM binary
                mini_bin_k = cache_key_mini("baseline_binary", "gpt-4o-mini", r["prompt"], r["conversation_json"])
                ans = parse_llm_answer(cache_mini[mini_bin_k]["answer"])
        elif pred == "UNSUPPORTED":
            ans = "reject"
        else:
            mini_bin_k = cache_key_mini("baseline_binary", "gpt-4o-mini", r["prompt"], r["conversation_json"])
            ans = parse_llm_answer(cache_mini[mini_bin_k]["answer"])
        deployed_ans.append(ans)
        deployed_correct.append(int(ans == r["ground_truth"]))
    ed["deployed_answer"] = deployed_ans
    ed["deployed_correct"] = deployed_correct
    print(f"Router (TF-IDF+SVM deployed): {np.mean(deployed_correct):.4f}  ({sum(deployed_correct)}/305)")

    # Save full per-prompt comparison
    out_full = ed[["conversation_idx", "prompt", "category", "ground_truth", "svm_predicted",
                   "svm_answer", "svm_correct", "extract_ok", "deployed_answer", "deployed_correct"]]
    out_full.to_csv(OUT_E2E / "e2e_rescored.csv", index=False)

    # 5. Bootstrap CIs
    np.random.seed(42)
    B = 5000
    def boot_ci(arr):
        n = len(arr)
        accs = np.empty(B)
        for b in range(B):
            idx = np.random.randint(0, n, size=n)
            accs[b] = np.asarray(arr)[idx].mean()
        return round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)

    summary = {
        "n": 305,
        "llm_only_binary_mini": {
            "acc": round(float(baseline_df.mini_bin_correct.mean()), 4),
            "ci": boot_ci(baseline_df.mini_bin_correct.values),
            "tokens": int(baseline_df.mini_bin_tokens.sum()),
        },
        "llm_only_reject_mini": {
            "acc": round(float(baseline_df.mini_rej_correct.mean()), 4),
            "ci": boot_ci(baseline_df.mini_rej_correct.values),
            "tokens": int(baseline_df.mini_rej_tokens.sum()),
        },
        "llm_only_binary_gpt4o": {
            "acc": round(float(baseline_df.gpt4o_bin_correct.mean()), 4),
            "ci": boot_ci(baseline_df.gpt4o_bin_correct.values),
            "tokens": int(baseline_df.gpt4o_bin_tokens.sum()),
        },
        "llm_only_reject_gpt4o": {
            "acc": round(float(baseline_df.gpt4o_rej_correct.mean()), 4),
            "ci": boot_ci(baseline_df.gpt4o_rej_correct.values),
            "tokens": int(baseline_df.gpt4o_rej_tokens.sum()),
        },
        "router_oracle": {
            "acc": round(float(np.mean(router_correct)), 4),
            "ci": boot_ci(np.array(router_correct)),
        },
        "router_deployed": {
            "acc": round(float(np.mean(deployed_correct)), 4),
            "ci": boot_ci(np.array(deployed_correct)),
        },
    }
    with open(OUT_E2E / "e2e_rescored_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_E2E}/e2e_rescored_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
