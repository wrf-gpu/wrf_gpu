"""Test-suite-wide hooks.

Honest CPU/GPU test hygiene
===========================

This project is a GPU-native model: the prognostic ``State`` constructors in
``gpuwrf.contracts.state`` deliberately *refuse* to allocate on CPU and raise

    RuntimeError("State.zeros requires a GPU device; no JAX GPU backend is visible")

so that a forecast can never silently run on the wrong device. As a result, the
M3/M4/M6 dycore + coupled-step tests that build a real ``State`` cannot execute
on a CPU-only checkout; on a GPU box (where the JAX GPU backend is visible) the
guard never fires and these same tests run and assert normally.

The hook below converts *only* that one specific GPU-required ``RuntimeError``
into a SKIP. It is intentionally narrow:

* It keys on the exact guard message, which can ONLY be produced when no JAX GPU
  backend is visible -- i.e. it is itself the "no GPU here" signal. On a GPU run
  the constructor succeeds and this branch is never reached.
* It does NOT touch any other exception. A real ``AssertionError`` (wrong number,
  contract drift, missing-citation, oracle tolerance, etc.) still FAILS. This
  preserves every genuine correctness/regression signal -- it does not mask bugs.
* Tests that do not hit the GPU guard (pure-Python / numpy / source-audit tests
  living in the same file) are unaffected and keep passing or failing as before.

This replaces what would otherwise be dozens of file-level ``skipif`` markers
that would also (wrongly) skip the many CPU-runnable tests sharing those files.
"""

from __future__ import annotations

import pytest

# The exact substrings emitted by gpuwrf.contracts.state._gpu_device() when no
# JAX GPU backend is visible. These markers are produced ONLY by that guard.
_GPU_REQUIRED_MARKERS = (
    "requires a GPU device",
    "no JAX GPU backend is visible",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when in ("setup", "call") and report.failed:
        excinfo = call.excinfo
        if (
            excinfo is not None
            and isinstance(excinfo.value, RuntimeError)
            and any(marker in str(excinfo.value) for marker in _GPU_REQUIRED_MARKERS)
        ):
            first_line = str(excinfo.value).splitlines()[0]
            report.outcome = "skipped"
            # longrepr must be a (path, lineno, reason) tuple so the terminal
            # reporter can fold the skip with -rs.
            report.longrepr = (
                str(item.fspath),
                0,
                "Skipped: GPU-required test on a CPU-only run "
                f"(no JAX GPU backend; runs on the GPU backend): {first_line}",
            )
