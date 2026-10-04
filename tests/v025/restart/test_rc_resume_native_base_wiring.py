"""RC-DEFAULTS C1 (review-s2small, adopted verbatim): the real checkpoint-resume path calls
_require_native_base_state once with both domains (deletion of the call fails this test)."""
import os

import pytest

from gpuwrf.integration import nested_pipeline as pipeline
from tests.v025.restart.test_restart_store import initialized
from tests.v025.restart import test_restart_store as restart_tests


def test_checkpoint_resume_runs_native_base_state_check(initialized, tmp_path, monkeypatch):
    if os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1":
        pytest.skip("synthetic restart_store checkpoint has no native-RK base_state: the fast defaults refuse it by "
                    "design (test_release_default_caveats); release-path resume = LW10 GPU restart arm 642/642 exact")
    calls = []
    original = pipeline._require_native_base_state

    def checked(carries, source):
        calls.append((set(carries), source))
        return original(carries, source)

    monkeypatch.setattr(pipeline, '_require_native_base_state', checked)
    restart_tests.test_production_segment_hooks_resume_clock_and_history(
        initialized, tmp_path, monkeypatch)
    assert len(calls) == 1
    assert calls[0][0] == {'d01', 'd02'}
    assert calls[0][1] is not None
