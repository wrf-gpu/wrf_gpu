"""v0.21 A4: permanent steep-terrain finite-state stability gate.

This is an opt-in GPU gate. A normal CPU pytest run skips it; the release/CI
GPU lane must set ``GPUWRF_RUN_V021_STEEP_TERRAIN_GATE=1`` and launch pytest
through ``scripts/with_gpu_lock.sh`` so the shared workstation GPU is serialized.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest


RUN_ENV = "GPUWRF_RUN_V021_STEEP_TERRAIN_GATE"
FIXTURE_ENV = "GPUWRF_STEEP_TERRAIN_FIXTURE"
FIXTURE_DIR = Path(
    os.environ.get(FIXTURE_ENV, "<DATA_ROOT>/wrf_downscale/fixture_montblanc_256")
)

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


requires_nvidia_gpu = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None,
    reason="v0.21 steep-terrain gate requires an NVIDIA GPU host",
)
requires_opt_in = pytest.mark.skipif(
    not _env_truthy(RUN_ENV),
    reason=(
        f"set {RUN_ENV}=1 and run through scripts/with_gpu_lock.sh "
        "to execute the steep-terrain GPU stability gate"
    ),
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.steep_terrain_gate,
    requires_nvidia_gpu,
    requires_opt_in,
]


def _require_fixture() -> None:
    required = (
        "namelist.input",
        "wrfbdy_d01",
        "wrfinput_d01",
        "wrfinput_d02",
    )
    missing = [name for name in required if not (FIXTURE_DIR / name).exists()]
    if missing:
        pytest.fail(
            f"Mont Blanc steep-terrain fixture is incomplete at {FIXTURE_DIR}: "
            f"missing {', '.join(missing)}"
        )


def _require_locked_gpu_preflight() -> None:
    from gpuwrf.runtime.gpu_preflight import GpuPreflightError, run_nested_gpu_preflight

    try:
        run_nested_gpu_preflight()
    except GpuPreflightError as exc:
        pytest.fail(
            "v0.21 steep-terrain gate must be run under the GPU lock with enough "
            f"free VRAM; launch via scripts/with_gpu_lock.sh. Preflight: {exc}",
            pytrace=False,
        )


def test_montblanc_two_domain_short_run_stays_finite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _require_fixture()
    _require_locked_gpu_preflight()

    scratch_dir = tmp_path / "scratch"
    output_dir = tmp_path / "wrfout"
    proof_dir = tmp_path / "proofs"
    scratch_dir.mkdir()
    monkeypatch.setenv("GPUWRF_FINITE_CHECK", "1")
    monkeypatch.setenv("GPUWRF_SCRATCH", str(scratch_dir))
    monkeypatch.setenv("GPUWRF_TMPDIR", str(scratch_dir))
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from gpuwrf.integration import nested_pipeline
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.runtime import finite_state_guard

    guarded_domains: set[str] = set()

    def recording_guard(
        state: Any,
        *,
        domain: str,
        step: int,
        sim_time_s: float | None = None,
        enabled: bool | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        guarded_domains.add(str(domain))
        finite_state_guard.assert_state_finite_at_boundary(
            state,
            domain=domain,
            step=step,
            sim_time_s=sim_time_s,
            enabled=enabled,
            environ=environ,
        )

    monkeypatch.setattr(nested_pipeline, "assert_state_finite_at_boundary", recording_guard)

    payload = nested_pipeline.execute_nested_pipeline(
        NestedPipelineConfig(
            input_dir=FIXTURE_DIR,
            output_dir=output_dir,
            proof_dir=proof_dir,
            hours=1,
            max_dom=2,
            scratch_dir=scratch_dir,
            feedback=False,
        )
    )

    assert payload["domains"] == ["d01", "d02"]
    assert {"d01", "d02"} <= guarded_domains
    assert payload["all_domains_finite"] is True
    assert payload["per_domain"]["d01"]["final_state_finite"] is True
    assert payload["per_domain"]["d02"]["final_state_finite"] is True
