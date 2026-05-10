"""Phase 1 entry point: regenerate the synthetic benchmark via GPT-5.2.

Default (reproduce) mode: print a summary of the committed synthetic dataset
without making any API calls. The committed `data/synthetic_final_*.csv`
files are the exact snapshot the thesis was trained on.

Rebuild mode: set `REBUILD_SYNTHETIC=1` to re-run the full synthetic data
generation pipeline (LLM calls via OpenAI; non-deterministic by nature). The
output will have different prompts and different conversation IDs from the
committed version.
"""

import json
import sys
from pathlib import Path

from config import DATA_DIR, DATASET_METADATA
from thesis_router import rebuild_synthetic


def _reproduce() -> None:
    print("[phase1] reproduce mode — summarising committed synthetic dataset "
          "(set REBUILD_SYNTHETIC=1 to regenerate)")
    if not DATASET_METADATA.exists():
        print(f"  ERROR: {DATASET_METADATA} missing — re-run with REBUILD_SYNTHETIC=1")
        sys.exit(1)

    meta = json.loads(DATASET_METADATA.read_text())
    print(f"\n  {DATASET_METADATA.name}:")
    for k in ("generated_at", "model", "total_prompts", "seed"):
        if k in meta:
            print(f"    {k}: {meta[k]}")

    print(f"\n  Committed CSVs:")
    for csv in sorted(DATA_DIR.glob("synthetic_final_*.csv")):
        print(f"    {csv.name} ({csv.stat().st_size // 1024} KB)")


def _rebuild() -> None:
    print("[phase1] REBUILD_SYNTHETIC=1 — regenerating via GPT-5.2 (non-deterministic)")
    from cli._notebook_runner import run_notebook
    run_notebook(Path(__file__).resolve().parent.parent / "notebooks" / "01_synthetic_data_generation.py")


def main() -> None:
    if rebuild_synthetic():
        _rebuild()
    else:
        _reproduce()


if __name__ == "__main__":
    main()
