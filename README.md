# Category-Aware QA Evaluation Router

Code, data, and figures for the MSc thesis *A Hybrid Evaluation Framework for QA Systems: Classifier-Driven Routing with Category-Specific Evaluation* (University
of Tartu, 2026).

The router classifies each QA prompt into one of six categories (SYMBOLIC_TIME,
SYMBOLIC_COUNT, SYMBOLIC_METADATA, SEMANTIC, HYBRID, UNSUPPORTED) and dispatches
symbolic prompts to a deterministic checker and semantic prompts to an LLM
judge. On a 305-pair benchmark the routed system reaches **96.1% accuracy
[93.8, 98.0]** against **64.6% [59.3, 69.8]** for the LLM-only baseline.

## Quick start

```bash
git clone <this repo> thesis-router
cd thesis-router
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make all
```

No API keys required in this path. `make all` re-derives every number cited in
the thesis from the committed `data/`, `models/`, and `results/` artefacts.

## What you should see

`make phase3` (or `make all`) prints, among other lines:

```
  LLM-only (binary)              acc=0.646 CI=[0.593, 0.698]
  Router (TF-IDF+LR)             acc=0.918 CI=[0.885, 0.948]
  Router (TF-IDF+SVM)            acc=0.961 CI=[0.938, 0.980]
  Router (Emb+LR)                acc=0.908 CI=[0.872, 0.941]
  Router (Ensemble)              acc=0.911 CI=[0.879, 0.941]
  Router (Keyword)               acc=0.800 CI=[0.754, 0.843]
```

Deviation from these values in reproduce mode means something regressed — open
an issue.

## Why route at all: the tool-augmented bound

A natural objection to routing is "just give the LLM the same tools the router
dispatches to". The tool-augmented sweep (`scripts/baselines/tool_augmented_llm_sweep.py`)
tests exactly that: every model is given the five deterministic oracle tools
(timestamp arithmetic, message counting, metadata lookup) with ground-truth
implementations, then asked to answer the 175 symbolic pairs.

| Model | Accuracy on 175 symbolic pairs | Tokens/prompt | p50 latency |
| --- | --- | --- | --- |
| Deterministic dispatch (router target) | 1.000 | 0 LLM tokens | <1 ms |
| o3-mini (reasoning) | 0.914 | 1881 | 3962 ms |
| o1 (reasoning) | 0.897 | 1920 | 5224 ms |
| Claude Sonnet 4.5 | 0.874 | 3058 | 4534 ms |
| Claude Opus 4.7 | 0.851 | 4002 | 6859 ms |
| GPT-4o | 0.851 | 1713 | 1063 ms |
| GPT-4o-mini | 0.817 | 2323 | 1572 ms |

Every residual error is an *application* failure — the model saw the
ground-truth tool output and still answered wrong. This bounds from below the
accuracy cost of keeping the LLM in the symbolic path, independent of
computation-capability improvements.

Numbers are reproduced from `results/independent_oracle/tool_augmented_comparison.csv`.

## Rebuild modes

Every phase has two modes. The default (all flags unset) reproduces the
thesis-cited numbers from committed artefacts. Setting a `REBUILD_*` flag
regenerates that phase from source.

| Flag | Phase | Regenerates | Deterministic | Cost |
| --- | --- | --- | --- | --- |
| `REBUILD_SYNTHETIC=1` | 1 | `data/synthetic_final_*.csv` via GPT-5.2 | no (LLM sampling) | ~$20–50 |
| `REBUILD_ROUTER=1` | 2 | `models/` + `results/router/` | yes (seed=42) | free |
| `REBUILD_EVAL=1` | 3 | `data/evaluation_dataset.csv` | yes (seed=42) | free |
| `REBUILD_E2E=1` | 3 | `data/evaluation_with_llm.csv`, `results/e2e/*` | cache-backed | ~$1–3 |
| `REBUILD_BASELINES=1` | 4 | reasoning-model + tool-augmented + secondary-generator | model-dependent | ~$80–115 |
| `REBUILD_ALL=1` | 1–4 | all of the above | — | ~$100–200 |

Rebuild modes that touch LLMs require `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY`
for Phase 4's Claude-family baselines). Copy `.env.example` to `.env` and fill
in the keys; `python-dotenv` picks them up automatically.

```bash
REBUILD_ROUTER=1 make phase2            # retrain all 10 classifiers
REBUILD_EVAL=1 REBUILD_E2E=1 make phase3   # rebuild eval set + rerun judge
REBUILD_ALL=1 make all                  # full rebuild from source
```

If `gpt-5.2` is unavailable, substitute a reachable model by editing
`GENERATION_MODEL` in `config.py` before running Phase 1.

## Code map

| Where | What |
| --- | --- |
| `config.py` | All thresholds, model IDs, and paths. Start here. |
| `thesis_router/llm_client.py` | OpenAI and Anthropic client factories (reads `.env`). |
| `thesis_router/rebuild.py` | `REBUILD_*` flag parser. Single source of truth for mode detection. |
| `benchmark/` | Synthetic benchmark construction (Phase 1). |
| `router/classifiers.py` | The 10 classifier training functions. |
| `router/benchmarking.py` | Phase 2 orchestration: trains all 10 configs, computes CIs, generalization gap. |
| `router/calibration.py`, `router/confidence.py` | Platt scaling, τ-based abstention. |
| `evaluation/polarity_parser.py` | Strict polarity-admission rule. Cited in thesis Ch1, Ch3, Ch4, Ch5, Appendix 7. |
| `evaluation/dataset_builder.py` | Builds the 305-pair benchmark under the polarity rule. |
| `evaluation/deterministic.py` | Symbolic checkers (time / count / metadata). |
| `evaluation/reliability.py`, `evaluation/latency.py` | Phase 3 reliability and latency measurement. |
| `notebooks/0{1,2,3}_*.py` | Marimo notebooks — the narrative version of phases 1–3. |
| `cli/run_phase{1,2,3,4}_*.py` | Plain-Python entry points used by the Makefile. |
| `scripts/baselines/` | Phase 4: reasoning-model and tool-augmented sweeps. |
| `scripts/validation/` | Synthetic-data validation, oracle audits, final-benchmark gate, secondary generators. |
| `scripts/analyses/` | τ-sensitivity, κ computation, hyperparameter tuning, parameter-extraction audit. |
| `results/` | All committed artefacts, grouped by producing-phase / script family. |

## Key thresholds

Every threshold below is a named constant in `config.py`; the thesis justifies
each one.

| Constant | Value | Gates |
| --- | --- | --- |
| `SEED` | 42 | All randomised training/splitting. |
| `SPLIT_TRAIN` / val / test | 70 / 15 / 15 | Stratified split; validation fold sized for stable Platt scaling. |
| `KEYWORD_LEAKAGE_THRESHOLD` | 0.25 | Reject a prompt if any single token predicts more than 25% of one category. |
| `KEYWORD_BASELINE_REJECTION` | 0.80 | Reject the entire dataset if a regex baseline exceeds this Macro-F1 — guards against trivial benchmarks. |
| `SEMANTIC_DEDUP_COSINE` | 0.99 | Drop near-identical LLM paraphrases within each category. |
| `TAU_DEFAULT` | 0.85 | Router-confidence cutoff below which a prompt is abstained (selected by validation-set sweep on routed accuracy × coverage). |
| `BOOTSTRAP_N` | 5000 | Iterations used for the 95% CIs shown above. |

## Layout

```
config.py              paths, thresholds, model IDs
thesis_router/         LLM clients + REBUILD_* flags
benchmark/             synthetic benchmark construction
router/                router training, calibration, evaluation
evaluation/            deterministic checkers, polarity parser, end-to-end harness
notebooks/             Marimo notebooks for phases 1–3
cli/                   plain-Python entry points for phases 1–3
scripts/               baselines, validation, analyses, artefact generation
data/ models/ results/ figures/    committed artefacts
```

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@mastersthesis{soliman2026router,
  title  = {A Hybrid Evaluation Framework for QA Systems: Classifier-Driven Routing with Category-Specific Evaluation},
  author = {Soliman, Ahmed},
  school = {University of Tartu},
  year   = {2026},
}
```
