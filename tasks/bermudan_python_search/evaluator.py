#!/usr/bin/env python3
"""Shared trusted evaluator entrypoint for the open Python Bermudan task.

The implementation deliberately lives in the historical
``bermudan_optimal_stopping`` evaluator so both task tracks use one financial
kernel and one scoring implementation.  This thin loader keeps the task
directory self-contained for :class:`harness.Task` and re-exports the shared
symbols for direct tests/imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SHARED_PATH = (
    _REPOSITORY_ROOT
    / "tasks"
    / "bermudan_optimal_stopping"
    / "evaluator.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "openhyra_shared_bermudan_evaluator", _SHARED_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure
    raise ImportError(f"could not load shared evaluator: {_SHARED_PATH}")
_SHARED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SHARED
_SPEC.loader.exec_module(_SHARED)

# Keep direct imports from this task compatible with imports from the shared
# evaluator.  Private helpers are included intentionally: existing smoke
# tests use a few underscore-prefixed protocol utilities.
for _name, _value in vars(_SHARED).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

# The shared evaluator keeps the historical Feature IR constants so archived
# callers remain compatible.  This directory is the explicit Python search
# track, however, and its direct CLI/import surface should construct requests
# for the AlgorithmBundle protocol by default.  Override only the task-facing
# aliases; the shared supported-name/protocol sets and financial kernel stay
# unchanged.
_SHARED.TASK_NAME = "bermudan_python_search"
_SHARED.TASK_PROTOCOL = _SHARED.ALGORITHM_BUNDLE_PROTOCOL
TASK_NAME = _SHARED.TASK_NAME
TASK_PROTOCOL = _SHARED.TASK_PROTOCOL


if __name__ == "__main__":
    _SHARED.main()
