"""Reproduce vs rebuild mode flags for the thesis pipeline.

Every phase has a committed output (the exact artifact produced for the
thesis) and a source (inputs + code that could regenerate it). By default
every phase loads its committed artifact — reviewers get thesis-matching
numbers without spending API budget or burning CPU.

Setting a `REBUILD_*` environment variable switches that phase to rebuild
mode:

  REBUILD_SYNTHETIC   Phase 1: regenerate the synthetic dataset via gpt-5.2
                      (non-deterministic; committed data/synthetic_final_*.csv
                      will be overwritten)
  REBUILD_ROUTER      Phase 2: retrain all 10 router classifiers from the
                      committed train/val/test splits (deterministic given
                      config.SEED=42)
  REBUILD_EVAL        Phase 3: rebuild data/evaluation_dataset.csv from the
                      test split + full pool under the polarity-admission
                      rule (deterministic given seed)
  REBUILD_E2E         Phase 3: re-run the end-to-end evaluation against the
                      current evaluation_dataset.csv (uses LLM cache when
                      available)
  REBUILD_BASELINES   Phase 4: re-run reasoning-model comparison and the
                      tool-augmented sweep (non-deterministic; reasoning
                      models are inconsistent by design)

  REBUILD_ALL         Convenience: set all five of the above to true

Flags are read as truthy strings: "1", "true", "yes", "on" (case-insensitive).
Anything else — including unset — is false.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    if os.environ.get("REBUILD_ALL", "").strip().lower() in _TRUTHY:
        return True
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def rebuild_synthetic() -> bool:
    return _flag("REBUILD_SYNTHETIC")


def rebuild_router() -> bool:
    return _flag("REBUILD_ROUTER")


def rebuild_eval_dataset() -> bool:
    return _flag("REBUILD_EVAL")


def rebuild_e2e() -> bool:
    return _flag("REBUILD_E2E")


def rebuild_baselines() -> bool:
    return _flag("REBUILD_BASELINES")
