"""A failed stage must suppress every later GPU stage (manager blocker, `main 0eadb2d2`).

The W1b window driver was a shell script in a scratch directory:

    rc1=$?; echo "STAGE1 rc=$rc1"
    [ $rc1 -ne 0 ] && { echo "STAGE1 non-zero -- see the json"; }   # no exit

An earlier revision had `exit $rc1`; it was dropped during a rewrite. Nothing
caught it because the driver lived outside the repository and outside this suite.
When Stage 1 hit the frozen 600 s early stop and returned rc=1, the shell logged
it and launched `nsys` anyway — GPU spent on a capture that could not be scored,
and a separate kill needed to stop it.

These tests run on CPU with stub stages and launch nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import run_window as rw  # noqa: E402


def _stages(*names: str) -> list[rw.Stage]:
    return [rw.Stage(name=n, command=["true", n]) for n in names]


def _runner(script):
    """Build a runner whose behaviour is keyed off the stage's command tail."""
    executed: list[str] = []

    def runner(command, *, cwd, env, timeout):
        name = command[-1]
        executed.append(name)
        return script.get(name, (0, "")), ""

    def unwrap(command, *, cwd, env, timeout):
        name = command[-1]
        executed.append(name)
        result = script.get(name, 0)
        if isinstance(result, Exception):
            raise result
        return result, f"[stub {name}]"

    unwrap.executed = executed  # type: ignore[attr-defined]
    return unwrap


# --------------------------------------------------------------------------- #
# the regression                                                               #
# --------------------------------------------------------------------------- #
def test_a_failed_stage_suppresses_every_later_stage(tmp_path):
    """The exact W1b failure: stage 1 non-zero must not be followed by nsys."""
    runner = _runner({"arm": 1})
    result = rw.run_window(
        label="baseline-census", stages=_stages("arm", "nsys", "ncu"),
        cwd=tmp_path, env={}, runner=runner,
    )
    assert [s.status for s in result.stages] == [rw.FAILED, rw.SUPPRESSED, rw.SUPPRESSED]
    assert runner.executed == ["arm"], "a later GPU stage was executed after a failure"
    assert result.ok is False
    assert result.first_failure.name == "arm"


def test_suppressed_stages_are_named_in_the_result(tmp_path):
    runner = _runner({"arm": 1})
    result = rw.run_window(
        label="w", stages=_stages("arm", "nsys", "ncu"),
        cwd=tmp_path, env={}, runner=runner,
    )
    assert result.as_dict()["gpu_stages_suppressed"] == ["nsys", "ncu"]


def test_all_ok_runs_everything(tmp_path):
    runner = _runner({})
    result = rw.run_window(
        label="w", stages=_stages("arm", "nsys", "ncu"),
        cwd=tmp_path, env={}, runner=runner,
    )
    assert result.ok is True
    assert runner.executed == ["arm", "nsys", "ncu"]


def test_a_middle_stage_failure_suppresses_only_what_follows(tmp_path):
    runner = _runner({"nsys": 2})
    result = rw.run_window(
        label="w", stages=_stages("arm", "nsys", "ncu"),
        cwd=tmp_path, env={}, runner=runner,
    )
    assert [s.status for s in result.stages] == [rw.OK, rw.FAILED, rw.SUPPRESSED]
    assert runner.executed == ["arm", "nsys"]


# --------------------------------------------------------------------------- #
# exit code 0 is not enough                                                    #
# --------------------------------------------------------------------------- #
def test_a_stage_that_exits_zero_but_records_early_stop_still_stops(tmp_path):
    """Trusting the exit code alone repeats the 'report over gate' mistake.

    `run_gpu_arm.py` can complete cleanly having written `status: EARLY_STOP`
    into its result file — which is exactly what happened in W1b.
    """
    artifact = tmp_path / "arm.json"
    artifact.write_text(json.dumps({"arm": {"status": "EARLY_STOP"}}))
    stages = [
        rw.Stage(name="arm", command=["true", "arm"],
                 verdict_from=artifact, verdict_key=("arm", "status")),
        rw.Stage(name="nsys", command=["true", "nsys"]),
    ]
    runner = _runner({})   # arm exits 0
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner)
    assert result.stages[0].status == rw.EARLY_STOP
    assert result.stages[1].status == rw.SUPPRESSED
    assert runner.executed == ["arm"]


# The test that used to live here asserted the OPPOSITE of this block:
# "a missing or unparseable artifact falls back to the exit code" -> OK. That
# pinned a fail-OPEN as if it were intended behaviour, in the one function whose
# job is to fail closed. Declaring `verdict_from` is a promise; breaking it is a
# stop, not a pass.
def test_a_declared_artifact_that_is_missing_stops_the_window(tmp_path):
    stages = [
        rw.Stage(name="arm", command=["true", "arm"],
                 verdict_from=tmp_path / "absent.json", verdict_key=("arm", "status")),
        rw.Stage(name="nsys", command=["true", "nsys"]),
    ]
    runner = _runner({})   # arm exits 0
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner)
    assert result.stages[0].status == rw.FAILED
    assert "does not exist" in result.stages[0].detail
    assert result.stages[1].status == rw.SUPPRESSED
    assert runner.executed == ["arm"]


def test_a_declared_artifact_that_is_malformed_stops_the_window(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    stages = [rw.Stage(name="arm", command=["true", "arm"],
                       verdict_from=bad, verdict_key=("arm", "status"))]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({}))
    assert result.stages[0].status == rw.FAILED
    assert "malformed" in result.stages[0].detail


def test_a_declared_artifact_missing_the_key_stops_the_window(tmp_path):
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"arm": {"something_else": "OK"}}))
    stages = [rw.Stage(name="arm", command=["true", "arm"],
                       verdict_from=artifact, verdict_key=("arm", "status"))]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({}))
    assert result.stages[0].status == rw.FAILED
    assert "has no" in result.stages[0].detail


@pytest.mark.parametrize("value", [None, 0, 1, True, ["OK"], {"status": "OK"}])
def test_a_non_string_verdict_stops_the_window(tmp_path, value):
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"arm": {"status": value}}))
    stages = [rw.Stage(name="arm", command=["true", "arm"],
                       verdict_from=artifact, verdict_key=("arm", "status"))]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({}))
    assert result.stages[0].status == rw.FAILED


@pytest.mark.parametrize("unknown", ["ok", "Ok", "PASSED", "DONE", "", "GREEN"])
def test_an_unrecognised_status_stops_the_window(tmp_path, unknown):
    """An unknown status is a stop, not a pass. Case-sensitive on purpose."""
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"arm": {"status": unknown}}))
    stages = [
        rw.Stage(name="arm", command=["true", "arm"],
                 verdict_from=artifact, verdict_key=("arm", "status")),
        rw.Stage(name="nsys", command=["true", "nsys"]),
    ]
    runner = _runner({})
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner)
    assert result.stages[0].status == rw.FAILED
    assert "not a recognised status" in result.stages[0].detail
    assert result.stages[1].status == rw.SUPPRESSED
    assert runner.executed == ["arm"]


def test_an_explicit_recognised_ok_passes(tmp_path):
    """The only way a declared artifact lets the window continue."""
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"arm": {"status": "OK"}}))
    stages = [
        rw.Stage(name="arm", command=["true", "arm"],
                 verdict_from=artifact, verdict_key=("arm", "status")),
        rw.Stage(name="nsys", command=["true", "nsys"]),
    ]
    runner = _runner({})
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner)
    assert result.ok is True
    assert runner.executed == ["arm", "nsys"]


def test_artifact_ok_but_nonzero_exit_still_stops(tmp_path):
    """Both signals must agree; an artifact cannot vouch for a crashed process."""
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"arm": {"status": "OK"}}))
    stages = [
        rw.Stage(name="arm", command=["true", "arm"],
                 verdict_from=artifact, verdict_key=("arm", "status")),
        rw.Stage(name="nsys", command=["true", "nsys"]),
    ]
    runner = _runner({"arm": 3})
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner)
    assert result.stages[0].status == rw.FAILED
    assert result.stages[1].status == rw.SUPPRESSED


def test_no_declared_artifact_means_the_exit_code_governs(tmp_path):
    """Stages that declare nothing are still judged by rc, as before."""
    result = rw.run_window(label="w", stages=_stages("arm"), cwd=tmp_path, env={},
                           runner=_runner({}))
    assert result.stages[0].status == rw.OK


# --------------------------------------------------------------------------- #
# Stage.gpu must mean something                                                #
# --------------------------------------------------------------------------- #
def test_suppression_labels_distinguish_gpu_from_cpu_stages(tmp_path):
    """`Stage.gpu` was declared and never read; the label overstated what was stopped."""
    stages = [
        rw.Stage(name="arm", command=["true", "arm"], gpu=True),
        rw.Stage(name="reduce", command=["true", "reduce"], gpu=False),
        rw.Stage(name="nsys", command=["true", "nsys"], gpu=True),
    ]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({"arm": 1})).as_dict()
    assert result["gpu_stages_suppressed"] == ["nsys"]
    assert result["cpu_stages_suppressed"] == ["reduce"]


def test_every_stage_result_carries_its_gpu_flag(tmp_path):
    stages = [rw.Stage(name="a", command=["true", "a"], gpu=False),
              rw.Stage(name="b", command=["true", "b"], gpu=True)]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({}))
    assert [s.gpu for s in result.stages] == [False, True]
    assert [s["gpu"] for s in result.as_dict()["stages"]] == [False, True]


# --------------------------------------------------------------------------- #
# launch failures and timeouts also stop the window                            #
# --------------------------------------------------------------------------- #
def test_a_timeout_stops_the_window(tmp_path):
    def runner(command, *, cwd, env, timeout):
        if command[-1] == "arm":
            raise subprocess.TimeoutExpired(cmd="arm", timeout=1.0)
        return 0, ""

    result = rw.run_window(
        label="w", stages=_stages("arm", "nsys"), cwd=tmp_path, env={}, runner=runner,
    )
    assert result.stages[0].status == rw.EARLY_STOP
    assert result.stages[1].status == rw.SUPPRESSED


def test_a_launch_exception_stops_the_window(tmp_path):
    runner = _runner({"arm": FileNotFoundError("no such binary")})
    result = rw.run_window(
        label="w", stages=_stages("arm", "nsys"), cwd=tmp_path, env={}, runner=runner,
    )
    assert result.stages[0].status == rw.ERROR
    assert "FileNotFoundError" in result.stages[0].detail
    assert result.stages[1].status == rw.SUPPRESSED


# --------------------------------------------------------------------------- #
# there is no override                                                         #
# --------------------------------------------------------------------------- #
def test_there_is_no_continue_anyway_option():
    """The failure mode was a driver that continued anyway; don't ship the flag."""
    import inspect

    signature = inspect.signature(rw.run_window)
    forbidden = {"force", "continue_on_error", "ignore_failures", "keep_going"}
    assert not (forbidden & set(signature.parameters))


def test_the_result_states_the_suppression_rule(tmp_path):
    result = rw.run_window(label="w", stages=_stages("arm"), cwd=tmp_path, env={},
                           runner=_runner({}))
    assert "never executed after a failure" in result.as_dict()["suppression_rule"]


# --------------------------------------------------------------------------- #
# global window deadline                                                       #
# --------------------------------------------------------------------------- #
def test_a_plan_that_cannot_fit_its_own_cap_is_refused():
    """Regression for my Step-1 plan: 700+700+300 = 1700 s inside a '1500 s cap'.

    Nobody caught the arithmetic because nothing checked it -- the cap lived in
    prose. A plan that cannot fit its own cap is not a capped plan, and the cap
    is exactly what the coordinating managers are promised.
    """
    stages = [
        rw.Stage("a", ["true", "a"], timeout_seconds=700),
        rw.Stage("b", ["true", "b"], timeout_seconds=700),
        rw.Stage("c", ["true", "c"], timeout_seconds=300),
    ]
    with pytest.raises(rw.WindowBudgetError, match="exceed the window deadline"):
        rw.validate_budget(stages, deadline_seconds=1500)


def test_overhead_counts_against_the_deadline():
    """Setup and GPU return are part of the window, not free."""
    stages = [rw.Stage("a", ["true", "a"], timeout_seconds=1400)]
    with pytest.raises(rw.WindowBudgetError):
        rw.validate_budget(stages, deadline_seconds=1500, overhead_seconds=180)


def test_a_fitting_plan_reports_its_headroom():
    stages = [
        rw.Stage("a", ["true", "a"], timeout_seconds=600),
        rw.Stage("b", ["true", "b"], timeout_seconds=400),
    ]
    budget = rw.validate_budget(stages, deadline_seconds=1500, overhead_seconds=180)
    assert budget["headroom_seconds"] == 320


def test_run_window_refuses_an_over_budget_plan_before_running_anything(tmp_path):
    runner = _runner({})
    stages = [rw.Stage("a", ["true", "a"], timeout_seconds=1000),
              rw.Stage("b", ["true", "b"], timeout_seconds=1000)]
    with pytest.raises(rw.WindowBudgetError):
        rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                      runner=runner, deadline_seconds=1500)
    assert runner.executed == [], "a stage ran despite the plan being over budget"


def test_an_overrun_clamps_the_next_stage_and_the_window_ends_not_ok(tmp_path, monkeypatch):
    """The runtime check catches what the up-front check cannot: an OVERRUN.

    This test previously asserted the next stage was SKIPPED. That encoded the
    dead-code version of the clamp: skipping whenever `budget > remaining` is the
    same condition as clamping, so the clamp could never fire. Corrected
    semantics -- a stage with time left RUNS with a clamped timeout (so a real
    subprocess is killed at the window boundary rather than past it), and only a
    stage with no time left is skipped.

    Both stages declare 400 s (800 <= 1000, so the plan validates); stage `a`
    burns 700 s, leaving 300 s for `b`.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(rw.time, "perf_counter", lambda: clock["t"])
    seen: list[float] = []

    def runner(command, *, cwd, env, timeout):
        seen.append(timeout)
        clock["t"] += 700.0          # overruns its declared 400 s budget
        return 0, ""

    stages = [rw.Stage(n, ["true", n], timeout_seconds=400) for n in ("a", "b")]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=runner, deadline_seconds=1000)
    assert seen == [400.0, 300.0], "the second stage was not clamped to the remaining budget"
    # the stub ignores its timeout, so the window overruns and must not report OK
    assert result.ok is False
    assert result.stages[-1].name == "window_deadline"


def test_no_deadline_means_no_global_check(tmp_path):
    """Backwards compatible: existing callers without a deadline are unaffected."""
    stages = [rw.Stage(n, ["true", n], timeout_seconds=9999) for n in ("a", "b")]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=_runner({}))
    assert result.ok is True


# --------------------------------------------------------------------------- #
# global cap: four fail-open holes, all real                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, 0, -1, float("inf"), float("nan")])
def test_a_stage_without_a_real_budget_cannot_be_scheduled(bad):
    """None/0/negative/non-finite all counted as ZERO in the budget sum before."""
    stages = [rw.Stage("a", ["true", "a"], timeout_seconds=bad)]
    with pytest.raises(rw.WindowBudgetError):
        rw.validate_budget(stages, deadline_seconds=1500)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_a_nonsense_deadline_is_refused(bad):
    stages = [rw.Stage("a", ["true", "a"], timeout_seconds=10)]
    with pytest.raises(rw.WindowBudgetError):
        rw.validate_budget(stages, deadline_seconds=bad)


def test_overhead_is_reserved_at_runtime_not_only_checked_upfront(tmp_path, monkeypatch):
    """GPU return has to fit inside the window too."""
    clock = {"t": 0.0}
    monkeypatch.setattr(rw.time, "perf_counter", lambda: clock["t"])

    def runner(command, *, cwd, env, timeout):
        clock["t"] += 500.0          # every stage overruns its 300 s budget
        return 0, ""

    stages = [rw.Stage(n, ["true", n], timeout_seconds=300) for n in ("a", "b", "c")]
    # 900 s of stages + 200 s overhead fits 1000 s up front is FALSE, so use 1150:
    # 900 + 200 = 1100 <= 1150 validates. Each stage then OVERRUNS to 500 s, so
    # after two stages 1000 s are gone and only 150 s remain of the 950 s that is
    # left once the 200 s overhead is reserved.
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=runner, deadline_seconds=1150, overhead_seconds=200)
    assert result.stages[2].status == rw.EARLY_STOP
    assert "reserving" in result.stages[2].detail


def test_the_child_timeout_is_clamped_to_the_remaining_budget(tmp_path, monkeypatch):
    """A stage must not be allowed to overrun the window just because its own budget fits."""
    clock = {"t": 0.0}
    monkeypatch.setattr(rw.time, "perf_counter", lambda: clock["t"])
    seen: list[float] = []

    def runner(command, *, cwd, env, timeout):
        seen.append(timeout)
        clock["t"] += 600.0          # overruns its 400 s budget
        return 0, ""

    stages = [rw.Stage(n, ["true", n], timeout_seconds=400) for n in ("a", "b")]
    # 800 s of budget validates against 850 s. Stage `a` then overruns to 600 s,
    # leaving 250 s -- less than stage `b`'s own 400 s budget, so `b` must be
    # handed 250 s, not 400 s. Before the fix the child received 400 s and could
    # push the window to 1000 s.
    rw.run_window(label="w", stages=stages, cwd=tmp_path, env={}, runner=runner,
                  deadline_seconds=850, overhead_seconds=0)
    assert seen[0] == 400
    assert seen[1] == pytest.approx(250.0), "child got its own budget, not the remaining one"


def test_a_final_stage_overrun_does_not_return_ok(tmp_path, monkeypatch):
    """Nothing re-checked the deadline after the last stage."""
    clock = {"t": 0.0}
    monkeypatch.setattr(rw.time, "perf_counter", lambda: clock["t"])

    def runner(command, *, cwd, env, timeout):
        clock["t"] += 900.0          # blows past its 400 s budget
        return 0, ""

    stages = [rw.Stage("only", ["true", "only"], timeout_seconds=400)]
    result = rw.run_window(label="w", stages=stages, cwd=tmp_path, env={},
                           runner=runner, deadline_seconds=500)
    assert result.ok is False, "a window that overran its cap reported success"
    assert result.stages[-1].name == "window_deadline"
    assert result.stages[-1].status == rw.EARLY_STOP
