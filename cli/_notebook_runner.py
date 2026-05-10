"""Run a marimo notebook's cells headlessly in dependency order."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def run_notebook(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"_notebook_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load notebook at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.run()
