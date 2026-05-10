"""Phase 2 entry point: train and calibrate the router classifiers.

Default (reproduce) mode: print a summary of the committed training results
(models/ and results/router/) without retraining. Fast (< 1s). The committed
artifacts are deterministic given config.SEED=42 — retraining produces the
same models.

Rebuild mode: set `REBUILD_ROUTER=1` to retrain all 10 classifiers from the
train/val/test splits. Overwrites `models/` and `results/router/`.
"""

import json
import sys
from pathlib import Path

from config import MODELS_DIR, RESULTS_DIR
from thesis_router import rebuild_router


def _reproduce() -> None:
    print("[phase2] reproduce mode — loading committed training results "
          "(set REBUILD_ROUTER=1 to retrain)")
    metadata_path = RESULTS_DIR / "router" / "training_metadata.json"
    if not metadata_path.exists():
        print(f"  ERROR: {metadata_path} missing — re-run with REBUILD_ROUTER=1")
        sys.exit(1)
    metadata = json.loads(metadata_path.read_text())

    print(f"\n  Models under {MODELS_DIR}:")
    for jl in sorted(MODELS_DIR.glob("*.joblib")):
        print(f"    {jl.name} ({jl.stat().st_size // 1024} KB)")

    benchmark_path = RESULTS_DIR / "router" / "per_class_f1.csv"
    if benchmark_path.exists():
        print(f"\n  results/router/per_class_f1.csv:")
        print("  " + "\n  ".join(benchmark_path.read_text().splitlines()[:11]))

    best = metadata.get("best_on_holdout") or metadata.get("best")
    if best:
        print(f"\n  Best model on holdout: {best}")


def _rebuild() -> None:
    print("[phase2] REBUILD_ROUTER=1 — retraining all 10 classifiers")
    from cli._notebook_runner import run_notebook
    run_notebook(Path(__file__).resolve().parent.parent / "notebooks" / "02_router_training.py")


def main() -> None:
    if rebuild_router():
        _rebuild()
    else:
        _reproduce()


if __name__ == "__main__":
    main()
