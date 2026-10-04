"""v0.3 release-default caveats fail loudly before stepping (release notes, RC-DEFAULTS condition 3).

Moving nests and pre-native checkpoints under the native RK path; CARRY_REAL_ALL without DYN_REAL_ALL.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwrf.integration.nested_pipeline import _require_native_base_state
from gpuwrf.nesting.moving_driver import MovingNestConfig, MovingNestError, validate_moving_nest_config


def _moving_config() -> MovingNestConfig:
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(MovingNestConfig)}
    kwargs = {}
    for name, f in fields.items():
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        kwargs[name] = {"child": "d02", "mode": "prescribed"}.get(name, 1)
    return MovingNestConfig(**kwargs)


def test_moving_nest_refused_under_native_rk(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    with pytest.raises(MovingNestError, match="GPUWRF_FAST_DEFAULTS=0"):
        validate_moving_nest_config(_moving_config())


def test_moving_nest_check_is_inert_on_the_legacy_path(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "0")
    try:
        validate_moving_nest_config(_moving_config())
    except MovingNestError as exc:  # other config checks may still apply; never the release caveat
        assert "fast defaults" not in str(exc)


def test_legacy_checkpoint_refused_under_native_rk(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    carries = {"d01": SimpleNamespace(base_state=object()), "d02": SimpleNamespace(base_state=None)}
    with pytest.raises(ValueError, match=r"no native-RK base_state for \['d02'\].*GPUWRF_FAST_DEFAULTS=0"):
        _require_native_base_state(carries, "/ckpt/gen-3")


def test_fast_default_checkpoint_and_legacy_path_pass(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    _require_native_base_state({"d01": SimpleNamespace(base_state=object())}, "x")
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "0")
    _require_native_base_state({"d01": SimpleNamespace(base_state=None)}, "x")


def _namelist_from_small_grid(monkeypatch):
    import importlib.util
    from pathlib import Path

    import jax

    from gpuwrf.contracts import state as state_contract
    from gpuwrf.runtime import operational_mode as op

    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])  # CPU placement (E126)
    root = Path(__import__("gpuwrf").__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("lw_fixture", root / "tests/test_rrtm_lw_operational_wiring.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return op.OperationalNamelist.from_grid(fixture._grid(ny=3, nx=3, nz=8))


@pytest.mark.parametrize("dyn_real_all,carry_fp32,refused", (("0", "1", True), ("1", "1", False), ("0", "0", False)))
def test_real_carry_without_real_dycore_refused(monkeypatch, dyn_real_all, carry_fp32, refused):
    """b-carry G10: CARRY_REAL_ALL runs only paired with DYN_REAL_ALL (inert without the REAL carry)."""
    for name in ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GPUWRF_DYN_CARRY_FP32", carry_fp32)
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "1")
    monkeypatch.setenv("GPUWRF_DYN_REAL_ALL", dyn_real_all if carry_fp32 == "1" else "0")
    if refused:
        with pytest.raises(ValueError, match="CARRY_REAL_ALL requires GPUWRF_DYN_REAL_ALL=1.*GPUWRF_FAST_DEFAULTS=0"):
            _namelist_from_small_grid(monkeypatch)
    else:
        _namelist_from_small_grid(monkeypatch)
