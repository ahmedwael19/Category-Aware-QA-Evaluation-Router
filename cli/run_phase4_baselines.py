"""Phase 4 entry point: reasoning-model + tool-augmented LLM baselines.

  REBUILD_BASELINES=1   Re-run every Phase 4 experiment:
                        - reasoning_model_comparison (o1, o3-mini, gpt-4o × 5 runs)
                        - reasoning_model_posthoc (prompt-length-adjusted stats)
                        - tool_augmented_llm_sweep (6 models on 175 symbolic pairs)
                        - secondary_generator (sonnet + opus cross-generator check)

Default (reproduce): print what is committed and exit. Phase 4 is
non-deterministic and budget-bearing (~$80-115 for a full rebuild), so
rebuild is opt-in per phase.

Set REBUILD_BASELINES=1 to re-run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import RESULTS_DIR, RESULTS_VALIDATION
from thesis_router import rebuild_baselines

_ROOT = Path(__file__).resolve().parent.parent


_STEPS: list[tuple[str, list[str]]] = [
    ("reasoning-model comparison", ["python", "-m", "scripts.baselines.reasoning_model_comparison"]),
    ("reasoning-model post-hoc",   ["python", "-m", "scripts.baselines.reasoning_model_posthoc"]),
    ("tool-augmented sweep",       ["python", "-m", "scripts.baselines.tool_augmented_llm_sweep"]),
    ("secondary generator (sonnet)", ["python", "-m", "scripts.validation.secondary_generator", "--model", "sonnet"]),
    ("secondary generator (opus)",   ["python", "-m", "scripts.validation.secondary_generator", "--model", "opus"]),
]


def _reproduce() -> None:
    print("[phase4] reproduce mode — loading committed artefacts "
          "(set REBUILD_BASELINES=1 to re-run)")
    for label, path in [
        ("reasoning",        RESULTS_DIR / "reasoning" / "summary.csv"),
        ("tool-augmented",   RESULTS_DIR / "independent_oracle" / "tool_augmented_comparison.csv"),
        ("secondary (sonnet)", RESULTS_VALIDATION / "secondary_generator_sonnet.json"),
        ("secondary (opus)",   RESULTS_VALIDATION / "secondary_generator_opus.json"),
    ]:
        marker = "ok" if path.exists() else "MISSING"
        print(f"  {label:<22} {marker:<8} {path.relative_to(_ROOT)}")


def _rebuild() -> None:
    print("[phase4] REBUILD_BASELINES=1 — re-running every Phase 4 experiment "
          "(~$80-115, requires OPENAI_API_KEY and ANTHROPIC_API_KEY)")
    for label, cmd in _STEPS:
        print(f"\n[phase4] → {label}")
        result = subprocess.run(cmd, cwd=_ROOT)
        if result.returncode != 0:
            print(f"[phase4] {label} failed (exit {result.returncode}); aborting")
            sys.exit(result.returncode)


def main() -> None:
    if rebuild_baselines():
        _rebuild()
    else:
        _reproduce()


if __name__ == "__main__":
    main()
