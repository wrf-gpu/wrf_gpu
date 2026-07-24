"""Static fail-closed gates for the CPU stop-after-alarm probe repair."""

from __future__ import annotations

import inspect

from scripts import v0234_final_ni_fable5_candidate_proof as aggregate
from scripts import v0234_final_ni_fable5_cpu_ab_probe as probe


def test_private_stop_occurs_only_after_completed_observation_append() -> None:
    source = inspect.getsource(probe.main)
    append_at = source.index("observations.append(")
    guard_at = source.index("if args.stop_after_alarm:", append_at)
    raise_at = source.index("raise _AlarmCaptured", guard_at)
    scheduler_at = source.index("result = run_operational_domain_tree(")
    catch_at = source.index("except _AlarmCaptured:", scheduler_at)
    assert append_at < guard_at < raise_at < scheduler_at < catch_at
    assert "if args.stop_after_alarm and not stopped_at_alarm:" in source
    assert 'raise RuntimeError("stop-after-alarm requested' in source


def test_supplemental_control_is_separate_and_only_falsified_arm_uses_it() -> None:
    assert aggregate.ARM_DIRS == {
        "falsified": "falsified-stop200",
        "corrected": "corrected",
    }
    assert aggregate.CORRECTION_PARENT == "60659a2e02e33153d87256ee78bf1380f5eabe76"
    assert aggregate.CORRECTED_CPU_COMMIT == "8fde9c941ba31f292d9921455dde9ea6e8387fc5"
