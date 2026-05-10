"""Phase 3 entry point: end-to-end evaluation on the 305-pair benchmark.

Two independent rebuild toggles control this phase:

  REBUILD_EVAL=1  Rebuild the 305-pair benchmark (data/evaluation_dataset.csv)
                  from the test split + full pool under the polarity-admission
                  rule. Default: load the committed benchmark.

  REBUILD_E2E=1   Re-run the LLM evaluation (GPT-5.2 ground truth, GPT-4o-mini
                  baselines, router comparison, reliability, uncached latency).
                  Default: load committed results from results/e2e/ and only
                  recompute the deterministic statistics (McNemar, bootstrap
                  CIs, per-category accuracy, error analysis).

Set both for a full rebuild.
"""

from pathlib import Path

from cli._notebook_runner import run_notebook
from thesis_router import rebuild_e2e, rebuild_eval_dataset


def main() -> None:
    if rebuild_eval_dataset():
        print("[phase3] REBUILD_EVAL=1   — rebuilding data/evaluation_dataset.csv")
    else:
        print("[phase3] reproduce mode   — loading committed evaluation_dataset.csv "
              "(set REBUILD_EVAL=1 to rebuild)")
    if rebuild_e2e():
        print("[phase3] REBUILD_E2E=1    — re-running LLM evaluation + comparison + reliability + latency")
    else:
        print("[phase3] reproduce mode   — loading committed results/e2e/* artefacts "
              "(set REBUILD_E2E=1 to re-run)")
    run_notebook(Path(__file__).resolve().parent.parent / "notebooks" / "03_evaluation_pipeline.py")


if __name__ == "__main__":
    main()
