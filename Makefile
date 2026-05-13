.PHONY: help setup phase1 phase2 phase3 phase4 figures all clean

help:
	@echo "Targets:"
	@echo "  setup      Install pinned dependencies and the editable package"
	@echo "  phase1     Phase 1: synthetic benchmark (reproduce summary; rebuild with REBUILD_SYNTHETIC=1)"
	@echo "  phase2     Phase 2: router training (reproduce summary; rebuild with REBUILD_ROUTER=1)"
	@echo "  phase3     Phase 3: end-to-end evaluation (reproduce from committed results; rebuild with REBUILD_EVAL=1 REBUILD_E2E=1)"
	@echo "  phase4     Phase 4: reasoning + tool-augmented baselines (reproduce; rebuild with REBUILD_BASELINES=1)"
	@echo "  figures    Regenerate every thesis figure from results/"
	@echo "  all        phase1 through phase4 plus figures"
	@echo "  clean      Remove __pycache__ and __marimo__ caches"
	@echo ""
	@echo "Rebuild flags (default: unset -> reproduce from committed artefacts):"
	@echo "  REBUILD_SYNTHETIC=1   regenerate data/synthetic_final_*.csv (non-deterministic; needs OPENAI_API_KEY)"
	@echo "  REBUILD_ROUTER=1      retrain models/ and results/router/ (deterministic)"
	@echo "  REBUILD_EVAL=1        rebuild data/evaluation_dataset.csv (deterministic)"
	@echo "  REBUILD_E2E=1         re-run end-to-end evaluation (cache-backed; needs OPENAI_API_KEY)"
	@echo "  REBUILD_BASELINES=1   re-run reasoning + tool-augmented sweeps (needs OPENAI_API_KEY and ANTHROPIC_API_KEY)"
	@echo "  REBUILD_ALL=1         shortcut for all of the above"
	@echo ""
	@echo "Typical workflows:"
	@echo "  make all                 # out of the box: reproduces thesis numbers, zero API calls"
	@echo "  REBUILD_EVAL=1 make phase3    # rebuild the eval dataset and regenerate results from cache"
	@echo "  REBUILD_ALL=1 make all        # full rebuild from source (budget: ~\$100-200)"

setup:
	pip install -r requirements.txt
	pip install -e .

phase1:
	python -m cli.run_phase1_generation

phase2:
	python -m cli.run_phase2_training

phase3:
	python -m cli.run_phase3_evaluation
	python -m scripts.baselines.rerun_llm_baselines
	python -m scripts.analyses.answerable_only
phase4:
	python -m cli.run_phase4_baselines

figures:
	python -m scripts.artifacts.generate_all_figures

all: phase1 phase2 phase3 phase4 figures

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name __marimo__ -exec rm -rf {} +
