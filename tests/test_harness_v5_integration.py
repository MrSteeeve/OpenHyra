import inspect
import pytest
from unittest.mock import MagicMock, patch

def test_run_pipeline_accepts_v5_bridge():
    from harness import run_pipeline
    sig = inspect.signature(run_pipeline)
    assert "v5_bridge" in sig.parameters

def test_commit_accepts_v5_bridge():
    from harness import _commit_candidate_result
    sig = inspect.signature(_commit_candidate_result)
    assert "v5_bridge" in sig.parameters

def test_v5_flag_in_argparse():
    import argparse
    from harness import main
    # Just verify --v5 is accepted by inspecting source
    import harness
    assert "--v5" in inspect.getsource(harness.main)

def test_v5_import_fallback():
    # V5Bridge import should not crash even if module is missing
    from harness import V5Bridge
    # It's either the real class or None
    assert V5Bridge is None or callable(V5Bridge)
