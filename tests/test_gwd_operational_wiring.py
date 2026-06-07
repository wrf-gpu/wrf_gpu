"""CPU regression tests for the operational GWD (gwd_opt=1) statics-wiring.

These guard the wiring that makes ``gwd_opt=1`` actually run in a real-case
forecast: the geo_em sub-grid orography loader, the ``gwd_opt`` namelist-group
resolution (WRF places it in ``&dynamics``), and the ``OperationalNamelist``
plumbing of ``gwd_opt`` / ``gwdo_statics``.  They do NOT need a GPU (the full
coupled GPU forecast is proofs/gwd/gwd_coupled_validation.py).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from gpuwrf.coupling.physics_couplers import build_gwdo_statics_from_wrf_fields
from gpuwrf.io.gwdo_static import load_gwdo_statics


class _FakeProjection:
    dx_m = 9000.0
    dy_m = 9000.0


class _FakeGrid:
    ny = 4
    nx = 5
    projection = _FakeProjection()


class _FakeRun:
    """Minimal Gen2Run stand-in for the loader's field accessor."""

    def __init__(self, fields, *, raise_missing=True):
        self._fields = fields
        self._raise_missing = raise_missing

    def history_files(self, domain):  # noqa: D401 -- accessor stub
        return ["fake_wrfout"]

    def wrfinput_file(self, domain):
        return "fake_wrfinput"

    def load(self, domain, name, time=0, lazy=False):
        if name in self._fields:
            return self._fields[name]
        raise KeyError(name)

    def load_wrfinput(self, domain, name, lazy=False):
        if name in self._fields:
            return self._fields[name]
        raise KeyError(name)


def _full_fields(ny=4, nx=5, var_max=300.0, con_max=2.0):
    import numpy as np

    var = np.zeros((ny, nx), dtype="float32")
    var[1, 1] = var_max  # a single "mountain" column
    con = np.zeros((ny, nx), dtype="float32")
    con[1, 1] = con_max
    fields = {"VAR": var, "CON": con}
    for name in ("OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4"):
        fields[name] = np.zeros((ny, nx), dtype="float32")
    return fields


def test_loader_builds_statics_when_var_present():
    run = _FakeRun(_full_fields())
    statics, meta = load_gwdo_statics(run, "d01", grid=_FakeGrid(), metrics=None)
    assert statics is not None
    assert meta["status"] == "built"
    assert meta["var_abs_max"] == pytest.approx(300.0, rel=1e-5)
    # flattened to (B,)=ny*nx and dx threaded through
    assert statics.var.shape == (20,)
    assert float(jnp.max(statics.dxmeter)) == pytest.approx(9000.0)
    assert float(jnp.max(statics.var)) == pytest.approx(300.0, rel=1e-5)


def test_loader_fails_closed_when_var_absent():
    # No VAR field at all -> statics None (GWD has nothing to act on).
    run = _FakeRun({})
    statics, meta = load_gwdo_statics(run, "d01", grid=_FakeGrid(), metrics=None)
    assert statics is None
    assert meta["status"] == "absent"


def test_loader_fails_closed_when_var_identically_zero():
    import numpy as np

    fields = _full_fields()
    fields["VAR"] = np.zeros((4, 5), dtype="float32")
    run = _FakeRun(fields)
    statics, meta = load_gwdo_statics(run, "d01", grid=_FakeGrid(), metrics=None)
    assert statics is None
    assert meta["status"] == "zero_var"


def test_gwd_opt_resolves_from_dynamics_group():
    # WRF places gwd_opt in &dynamics; the wiring must read it there.
    from gpuwrf.integration.nested_pipeline import _domain_gwd_opt

    class _Run:
        namelist = {"dynamics": {"gwd_opt": 1}, "physics": {}}

    assert _domain_gwd_opt(_Run(), "d01") == 1
    assert _domain_gwd_opt(_Run(), "d03") == 1

    class _RunPhysics:
        # fallback: some namelists place it in &physics
        namelist = {"dynamics": {}, "physics": {"gwd_opt": 1}}

    assert _domain_gwd_opt(_RunPhysics(), "d01") == 1

    class _RunOff:
        namelist = {"dynamics": {"gwd_opt": 0}, "physics": {}}

    assert _domain_gwd_opt(_RunOff(), "d01") == 0


def test_from_grid_plumbs_gwd_fields_and_default_is_noop():
    # Build a real GridSpec from a tiny synthetic grid using a Gen2 case is not
    # available offline; instead verify the dataclass field plumbing directly.
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    # default: gwd off
    sig_defaults = OperationalNamelist.__dataclass_fields__
    assert sig_defaults["gwd_opt"].default == 0
    assert sig_defaults["gwdo_statics"].default is None

    # from_grid must accept the two kwargs (signature plumbing)
    import inspect

    params = inspect.signature(OperationalNamelist.from_grid).parameters
    assert "gwd_opt" in params
    assert "gwdo_statics" in params


def test_build_statics_directional_fields_round_trip():
    import numpy as np

    ny, nx = 3, 3
    var = np.full((ny, nx), 100.0, dtype="float32")
    con = np.full((ny, nx), 1.0, dtype="float32")
    oa = [np.full((ny, nx), 0.1 * (i + 1), dtype="float32") for i in range(4)]
    ol = [np.full((ny, nx), 0.2 * (i + 1), dtype="float32") for i in range(4)]
    statics = build_gwdo_statics_from_wrf_fields(
        var, con, *oa, *ol, dx_m=3000.0
    )
    assert statics.var.shape == (ny * nx,)
    assert float(jnp.max(statics.oc1)) == pytest.approx(1.0, rel=1e-5)
    assert float(jnp.max(statics.ol4)) == pytest.approx(0.8, rel=1e-5)
    assert float(jnp.max(statics.dxmeter)) == pytest.approx(3000.0)
