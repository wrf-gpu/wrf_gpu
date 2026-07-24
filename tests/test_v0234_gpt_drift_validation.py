from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_gpt_drift_validation.py"
SPEC = importlib.util.spec_from_file_location("v0234_gpt_drift_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_stats_are_all_cell_signed_gpu_minus_cpu() -> None:
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]])
    gpu = np.array([[2.0, 1.0], [5.0, 2.0]])
    stats = MODULE._paired_stats(gpu, cpu)
    assert stats["finite_pair_fraction"] == 1.0
    assert stats["rmse"] == np.sqrt(2.5)
    assert stats["mean_signed_drift"] == 0.0
    assert stats["max_abs"] == 2.0


def test_paired_stats_do_not_mask_nonfinite_cells() -> None:
    cpu = np.array([1.0, 2.0])
    gpu = np.array([1.0, np.nan])
    stats = MODULE._paired_stats(gpu, cpu)
    assert stats["finite_pair_count"] == 1
    assert stats["finite_pair_fraction"] == 0.5


def test_namespace_requires_exact_root_nonce_and_absence(tmp_path: Path) -> None:
    nonce = "a" * 64
    wrong = MODULE._validate_namespace(tmp_path / nonce[:16], nonce, require_absent=True)
    assert not wrong["ok"]
    canonical = Path("<DATA_ROOT>/wrf_gpu2") / f"v0234_test_{nonce[:16]}"
    result = MODULE._validate_namespace(canonical, nonce, require_absent=True)
    assert result["ok"] == (not canonical.exists())


def test_derived_speed_ceiling_is_component_norm() -> None:
    assert MODULE.DERIVED_WSPD_CEILING == np.sqrt(1.5**2 + 1.5**2)


def test_24h_mode_has_one_exact_endpoint_and_72h_retains_both() -> None:
    assert MODULE._required_exact_leads(24) == (24,)
    assert MODULE._required_exact_leads(72) == (24, 72)


def test_score_parser_accepts_explicit_24h_mode(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        [
            "score",
            "--truth-dir",
            str(tmp_path / "truth"),
            "--candidate-dir",
            str(tmp_path / "candidate"),
            "--hours",
            "24",
            "--out",
            str(tmp_path / "score.json"),
        ]
    )
    assert args.hours == 24


def test_namelist_gate_binds_hourly_output_and_30min_radiation(tmp_path: Path) -> None:
    namelist = tmp_path / "namelist.input"
    namelist.write_text(
        """
run_hours = 72,
max_dom = 2,
feedback = 0,
history_interval = 60, 60,
sf_surface_physics = 4, 4,
bl_pbl_physics = 5, 5,
sf_sfclay_physics = 5, 5,
ra_lw_physics = 4, 4,
ra_sw_physics = 4, 4,
radt = 30, 30,
gwd_opt = 1,
"""
    )
    result = MODULE._namelist_checks(namelist)
    assert result["ok"]
    assert result["checks"]["history_hourly"]
    assert result["checks"]["radiation_cadence_30min_both_domains"]


def test_production_environment_keeps_boundary_default_unset_and_output_full(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "0")
    monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", "1")
    env = MODULE._production_environment(
        tmp_path / "repo", tmp_path / "WRF", tmp_path / "cache"
    )
    assert "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE" not in env
    assert "GPUWRF_TRAINING_OUTPUT_SUBSET" not in env
    assert env["GPUWRF_NESTED_FUSE"] == "0"
    assert "GPUWRF_NESTED_ASYNC_OUTPUT" not in env
    assert env["GPUWRF_NEST_OUTPUT_PIPELINE"] == "0"
    assert env["GPUWRF_PREPARED_RUNTIME_REUSE"] == "1"


def test_cpu_dry_run_does_not_materialize_fresh_gpu_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "must-remain-absent"

    def fake_run(command, **kwargs):
        assert "GPUWRF_JAX_CACHE_DIR" not in kwargs["env"]
        assert "JAX_COMPILATION_CACHE_DIR" not in kwargs["env"]
        assert not cache_dir.exists()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"dry_run":true,"run_type":"nested_live",'
                '"init_mode":"standalone_native_init_nested",'
                '"effective_max_dom":2,"effective_hours":24,"feedback":false}'
            ),
            stderr="",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = MODULE._cli_dry_run(
        tmp_path / "repo",
        tmp_path / "input",
        tmp_path / "namespace",
        tmp_path / "WRF",
        cache_dir,
        24,
    )
    assert result["ok"]
    assert not cache_dir.exists()


def test_pass_fds_preserves_an_inherited_descriptor(tmp_path: Path) -> None:
    authority = tmp_path / "canonical.lock"
    fd = os.open(authority, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.set_inheritable(fd, True)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; print(os.readlink('/proc/self/fd/'+sys.argv[1]))",
                str(fd),
            ],
            pass_fds=(fd,),
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        assert Path(child.stdout.strip()) == authority
    finally:
        os.close(fd)


def test_missing_pass_fds_closes_inherited_descriptor_fail_closed(tmp_path: Path) -> None:
    authority = tmp_path / "canonical.lock"
    fd = os.open(authority, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.set_inheritable(fd, True)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys; p='/proc/self/fd/'+sys.argv[1]; "
                    "expected=sys.argv[2]; "
                    "actual=(os.readlink(p) if os.path.exists(p) else 'CLOSED'); "
                    "raise SystemExit(1 if actual == expected else 0)"
                ),
                str(fd),
                str(authority),
            ],
            check=False,
        )
        assert child.returncode == 0
    finally:
        os.close(fd)
