"""In-step WRF history surface diagnostics (GPUWRF_HISTORY_INSTEP_DIAG, default OFF).

WRF writes history before ``solve`` (module_integrate.F:375 vs :393), so a frame's
T2/Q2/U10/V10/HFX/LH/PSFC are the last step's surface_driver values. These tests pin
the carry container, the producer return, the M9/writer read path and restart.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import (
    HistoryDiagnostics,
    history_instep_diag_enabled,
    seed_history_diagnostics,
)
from test_v090_noahmp_t2_overwrite import HAVE_TABLES, _build_operational

SHAPE = (1, 5)


def _history(*, hfx: bool = True) -> HistoryDiagnostics:
    values = {name: jnp.full(SHAPE, float(i + 1), jnp.float32) for i, name in enumerate(HistoryDiagnostics._fields)}
    if not hfx:
        values["hfx"] = None
    return HistoryDiagnostics(**values)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("GPUWRF_HISTORY_INSTEP_DIAG", raising=False)
    assert not history_instep_diag_enabled()
    monkeypatch.setenv("GPUWRF_HISTORY_INSTEP_DIAG", "1")
    assert history_instep_diag_enabled()


def test_seed_is_real_and_write_casts_explicitly():
    seed = seed_history_diagnostics(SimpleNamespace(t_skin=jnp.ones((2, 3), jnp.float64)))
    assert all(leaf.dtype == jnp.float32 and leaf.shape == (2, 3) for leaf in seed)
    written = seed.write({name: np.full((2, 3), 300.0 + 1e-9) for name in seed._fields})
    assert all(leaf.dtype == jnp.float32 and leaf.shape == (2, 3) for leaf in written)
    assert np.asarray(written.t2).tobytes() == np.full((2, 3), 300.0 + 1e-9).astype(np.float32).tobytes()


def test_seed_reuses_state_hfx_when_state_carries_it():
    class _StateWithHfx:
        __slots__ = ("t_skin", "hfx")

    state = _StateWithHfx()
    state.t_skin = jnp.ones((2, 3))
    state.hfx = jnp.zeros((2, 3))
    seed = seed_history_diagnostics(state)
    assert seed.hfx is None
    assert seed.write({name: jnp.zeros((2, 3)) for name in seed._fields}).hfx is None


def _surface_state():
    state, land, static, rad, clock = _build_operational()
    zero = jnp.zeros(SHAPE)
    state = state.replace(theta_flux=zero, qv_flux=zero, tau_u=zero, tau_v=zero, rhosfc=zero, fltv=zero)
    return state, land, static, rad, clock


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_surface_step_history_is_read_only_and_maps_wrf_fields(monkeypatch):
    import gpuwrf.physics.noahmp_coupler as coupler
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view, noahmp_surface_step
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    seen = []
    real_step = coupler.noah_mp_step

    def _recording_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        seen.append(out[1])
        return out

    monkeypatch.setattr(coupler, "noah_mp_step", _recording_step)
    state, land, static, rad, clock = _surface_state()
    off_state, off_land = noahmp_surface_step(state, land, static, 90.0, radiation=rad, clock=clock)
    on_state, on_land, fields = noahmp_surface_step(
        state, land, static, 90.0, radiation=rad, clock=clock, history=True)

    for name in ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "t_skin", "roughness_m"):
        assert np.asarray(getattr(off_state, name)).tobytes() == np.asarray(getattr(on_state, name)).tobytes(), name
    for a, b in zip(jax.tree.leaves(off_land), jax.tree.leaves(on_land), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()

    assert set(fields) == {"hfx", "lh", "t2", "q2", "u10", "v10", "psfc"}
    diag = surface_layer_with_diagnostics(_build_column_view(state, None))
    for name in ("u10", "v10"):
        assert np.asarray(fields[name]).tobytes() == np.asarray(getattr(diag, name)).tobytes(), name
    # WRF: land T2/Q2 = Noah-MP FVEG blend (B47 nm.q2), water = sfclay.
    water = np.asarray((state.xland - 1.5) >= 0.0)
    nm = seen[-1]
    for name in ("t2", "q2"):
        got = np.asarray(fields[name])
        assert np.array_equal(got[water], np.asarray(getattr(diag, name), np.float64)[water]), name
        assert np.array_equal(got[~water], np.asarray(getattr(nm, name), np.float64)[~water]), name
    # No grid metrics: the hook's documented psfc fallback is the lowest-level pressure.
    assert np.asarray(fields["psfc"]).tobytes() == np.asarray(state.p, np.float64)[0].tobytes()
    assert all(np.isfinite(np.asarray(v)).all() for v in fields.values())


def _radiation():
    names = ("swnorm", "swdown", "glw", "coszen", "swup", "sw_toa_down", "sw_toa_up",
             "glw_up", "lw_toa_down", "lw_toa_up")
    return SimpleNamespace(**{name: jnp.full(SHAPE, 10.0 + i) for i, name in enumerate(names)})


def _forbid_resolve(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("post-step surface re-solve ran")

    for name in ("surface_layer_diagnostics", "overlay_noahmp_land_diagnostics", "rrtmg_radiation_diagnostics"):
        monkeypatch.setattr(op, name, _boom)
    monkeypatch.setattr(op, "_history_pblh", lambda state, grid: jnp.full(SHAPE, 7.0))


def test_m9_history_branch_reads_carry_without_resolve(monkeypatch):
    _forbid_resolve(monkeypatch)
    history = _history()
    state = SimpleNamespace(t_skin=jnp.full(SHAPE, 300.0))
    namelist = SimpleNamespace(use_noahmp=True, ra_sw_physics=4, ra_lw_physics=4, slope_rad=0, grid=None, metrics=None)
    m9 = op.compute_m9_diagnostics(
        state, namelist, 3600.0, noahmp_land=object(),
        noahmp_rad=(jnp.full(SHAPE, 2.0), jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.5)),
        radiation_diagnostics=_radiation(), history_diagnostics=history,
    )
    for attr in ("t2", "u10", "v10", "lh", "psfc", "hfx"):
        assert getattr(m9, attr) is getattr(history, attr), attr
    assert m9.tsk is state.t_skin
    assert float(np.asarray(m9.pblh)[0, 0]) == 7.0
    assert float(np.asarray(m9.swdown)[0, 0]) == 2.0  # held Noah-MP SOLDN, unchanged path


def test_m9_history_uses_state_hfx_when_container_has_none(monkeypatch):
    _forbid_resolve(monkeypatch)
    state = SimpleNamespace(t_skin=jnp.full(SHAPE, 300.0), hfx=jnp.full(SHAPE, 55.0))
    namelist = SimpleNamespace(use_noahmp=True, ra_sw_physics=4, ra_lw_physics=4, slope_rad=0, grid=None, metrics=None)
    m9 = op.compute_m9_diagnostics(
        state, namelist, 3600.0, noahmp_rad=None, radiation_diagnostics=_radiation(),
        history_diagnostics=_history(hfx=False),
    )
    assert m9.hfx is state.hfx


def test_writer_history_path_supplies_q2_hfx_lh(monkeypatch):
    from gpuwrf.integration import nested_pipeline as npl

    _forbid_resolve(monkeypatch)
    monkeypatch.setattr(op, "build_clock_base", lambda namelist: None)
    history = _history()
    namelist = SimpleNamespace(use_noahmp=True, ra_sw_physics=4, ra_lw_physics=4, slope_rad=0, grid=None,
                               metrics=None, time_utc=datetime(2026, 7, 26, tzinfo=timezone.utc))
    out = npl._history_surface_diagnostics_for_output(
        SimpleNamespace(t_skin=jnp.full(SHAPE, 300.0)), namelist, namelist.time_utc,
        lead_seconds=3600.0, noahmp_land=object(),
        noahmp_rad=(jnp.full(SHAPE, 2.0), jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.5)),
        radiation_diagnostics=_radiation(), history=history, variable_subset=None, output_radiation_work=None,
    )
    expected = {"T2": history.t2, "Q2": history.q2, "U10": history.u10, "V10": history.v10,
                "PSFC": history.psfc, "HFX": history.hfx, "LH": history.lh}
    for name, value in expected.items():
        assert np.asarray(out[name]).tobytes() == np.asarray(value).tobytes(), name
    assert {"TSK", "PBLH", "SWDOWN", "GLW", "LWUPT"} <= set(out)


def test_wrfrst_roundtrips_history_container_exactly(tmp_path: Path):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v0110_wrfrst_netcdf import _state

    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    seed = seed_history_diagnostics(state)
    rng = np.random.default_rng(3)
    history = seed.write({name: rng.normal(300.0, 5.0, np.shape(seed.t2)) for name in seed._fields})
    carry = initial_operational_carry(state).replace(history_diagnostics=history)
    path = tmp_path / "wrfrst_history"
    write_wrfrst_carry(carry, grid, {}, path, valid_time="2026-06-03_00:20:00",
                       run_start="2026-06-03_00:00:00", step_index=2)
    restored, _ = read_wrfrst_carry(path)
    assert isinstance(restored.history_diagnostics, HistoryDiagnostics)
    for name in HistoryDiagnostics._fields:
        a, b = np.asarray(getattr(history, name)), np.asarray(getattr(restored.history_diagnostics, name))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name

    off = initial_operational_carry(state)
    path_off = tmp_path / "wrfrst_off"
    write_wrfrst_carry(off, grid, {}, path_off, valid_time="2026-06-03_00:20:00",
                       run_start="2026-06-03_00:00:00", step_index=2)
    assert read_wrfrst_carry(path_off)[0].history_diagnostics is None


class _StopAtPbl(Exception):
    pass


def _step_write_harness(monkeypatch, *, with_container: bool):
    """Drive the real step through the Noah-MP surface slot with a stubbed surface step."""
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class _Surface:
        t_skin: object
        xland: object

        def replace(self, **kw):
            return replace(self, **kw)

    @dataclass(frozen=True)
    class _Carry:
        state: object
        noahmp_land: object
        radiation_diagnostics: object
        noahmp_rad: object = None
        rthraten: object = None
        census: object = None
        history_diagnostics: object = None

        def replace(self, **kw):
            updates.append(kw)
            return replace(self, **kw)

    class _Namelist(SimpleNamespace):
        def __getattr__(self, name):
            if name == "bl_pbl_physics":
                raise _StopAtPbl()
            raise AttributeError(name)

    updates, calls = [], []
    z = jnp.zeros(SHAPE)
    state = _Surface(jnp.full(SHAPE, 290.0), jnp.ones(SHAPE))
    rad = SimpleNamespace(swnorm=z, glw=z, coszen=z)
    container = seed_history_diagnostics(state) if with_container else None
    carry = _Carry(state, object(), rad, history_diagnostics=container)
    fields = {name: jnp.full(SHAPE, 100.0 + i, jnp.float64)
              for i, name in enumerate(("hfx", "lh", "t2", "q2", "u10", "v10", "psfc"))}

    def _surface(st, land, *a, history=False, **kw):
        calls.append(history)
        return (st, land, dict(fields)) if history else (st, land)

    namelist = _Namelist(lower_boundary=None, noahmp_static=None, dt_s=54.0, boundary_config=None,
                         run_physics=True, rad_rk_tendf=0, mp_physics=0, sf_sfclay_physics=5, cu_physics=0,
                         use_noahmp=True, radiation_interval_s=1800.0, radiation_cadence_steps=33, grid=None,
                         time_utc=datetime(2026, 7, 26, tzinfo=timezone.utc), radiation_static=None,
                         topo_shading=0, slope_rad=0, topo_shadow_length_m=25000.0,
                         noahmp_julian=206.0, noahmp_yearlen=365.0)
    monkeypatch.setitem(op.SFCLAY_SCAN_ADAPTERS, 5, lambda st, *a, **kw: st)
    monkeypatch.setattr(op, "_noahmp_params", lambda nml: (None, None))
    monkeypatch.setattr(op, "_lane_noahmp_static", lambda nml, clock_base=None: None)
    monkeypatch.setattr(op, "noahmp_surface_step", _surface)
    with pytest.raises(_StopAtPbl):
        op._physics_boundary_step_with_limiter_diagnostics(
            carry, namelist, jnp.asarray(2, jnp.int32), run_radiation=False)
    written = [u["history_diagnostics"] for u in updates if "history_diagnostics" in u]
    return calls, written, fields


def test_step_writes_history_container_from_surface_fields(monkeypatch):
    calls, written, fields = _step_write_harness(monkeypatch, with_container=True)
    assert calls == [True]
    assert len(written) == 1
    for name in HistoryDiagnostics._fields:
        got = getattr(written[0], name)
        if got is None:
            continue
        assert got.dtype == jnp.float32
        assert np.asarray(got).tobytes() == np.asarray(fields[name]).astype(np.float32).tobytes(), name


def test_step_without_container_requests_no_history(monkeypatch):
    calls, written, _ = _step_write_harness(monkeypatch, with_container=False)
    assert calls == [False]
    assert written == []


# ---------------------------------------------------------------------------
# B49: WRF Noah-MP-path water T2/Q2 (module_surface_driver.F:3446-3458)
# ---------------------------------------------------------------------------
_WRF_SURFACE_DRIVER = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_surface_driver.F")
_GFORTRAN = Path(os.environ.get("GF", os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")))


def _water_rows() -> np.ndarray:
    """Realistic ocean rows (TSK, PSFC, HFX, QFX, QSFC, CHS2, CQS2) plus threshold edges."""
    rng = np.random.default_rng(49)
    n = 64
    rows = np.stack([
        rng.uniform(285.0, 302.0, n), rng.uniform(95000.0, 103000.0, n), rng.uniform(-60.0, 160.0, n),
        rng.uniform(-2e-5, 2.5e-4, n), rng.uniform(0.008, 0.026, n), rng.uniform(1e-3, 2.5e-2, n),
        rng.uniform(1e-3, 2.5e-2, n)], axis=1)
    edges = np.array([[295.0, 101000.0, 20.0, 5e-5, 0.018, c, q]
                      for c, q in ((0.0, 0.0), (5e-6, 2e-6), (1e-5, 1e-5), (1.2e-5, 9.9e-6), (0.02, 0.0))])
    return np.vstack([rows, edges]).astype(np.float32)


def _pristine_water_2m(rows: np.ndarray, tmp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compile the VERBATIM pristine lines 3447-3458 (WRF REAL) and run them on ``rows``."""
    lines = _WRF_SURFACE_DRIVER.read_text().splitlines()[3446:3458]
    assert "IVGTYP(I,J) == ISWATER" in lines[0] and "TH2(I,J) = T2(I,J)" in lines[-1], "pristine lines moved"
    src = "\n".join([
        "program b49_oracle", "  implicit none", "  integer :: n, i, j",
        "  integer, parameter :: ISWATER = 17, ISICE = 15", "  real, parameter :: XICE_THRESHOLD = 0.5",
        "  real :: R_d, CP, RCP",
        "  real, allocatable, dimension(:,:) :: TSK, PSFC, HFX, QFX, QSFC, CHS2, CQS2, XICE, T2, Q2, TH2",
        "  integer, allocatable :: IVGTYP(:,:)",
        "  R_d = 287.", "  CP = 7.*R_d/2.", "  RCP = R_d/CP",
        "  open(10, file='in.bin', access='stream', form='unformatted')", "  read(10) n",
        "  allocate(TSK(n,1), PSFC(n,1), HFX(n,1), QFX(n,1), QSFC(n,1), CHS2(n,1), CQS2(n,1))",
        "  allocate(XICE(n,1), T2(n,1), Q2(n,1), TH2(n,1), IVGTYP(n,1))",
        "  read(10) TSK, PSFC, HFX, QFX, QSFC, CHS2, CQS2", "  IVGTYP = ISWATER", "  XICE = 0.", "  j = 1",
        "  DO i = 1, n", *lines, "              ENDIF", "  END DO",
        "  open(11, file='out.bin', access='stream', form='unformatted')", "  write(11) T2, Q2",
        "end program b49_oracle"])
    (tmp / "b49.f90").write_text(src + "\n")
    subprocess.run([str(_GFORTRAN), "-O0", "-ffree-line-length-none", "-o", str(tmp / "b49"), str(tmp / "b49.f90")],
                   check=True, cwd=tmp)
    with (tmp / "in.bin").open("wb") as f:
        f.write(np.int32(len(rows)).tobytes())
        for column in rows.T:
            f.write(np.ascontiguousarray(column, np.float32).tobytes())
    subprocess.run([str(tmp / "b49")], check=True, cwd=tmp)
    out = np.fromfile(tmp / "out.bin", dtype=np.float32)
    return out[: len(rows)], out[len(rows):]


def test_water_2m_matches_pristine_wrf_oracle(tmp_path, monkeypatch):
    from gpuwrf.physics.noahmp_coupler import _wrf_water_2m

    if not _GFORTRAN.exists():
        pytest.fail(f"pristine oracle compiler missing: {_GFORTRAN} (set GF/FC)")
    rows = _water_rows()
    t2_ref, q2_ref = _pristine_water_2m(rows, tmp_path)
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")  # WRF REAL arithmetic
    t2, q2 = _wrf_water_2m(*(rows[:, k] for k in range(7)))
    assert np.asarray(t2).dtype == np.float32
    assert np.asarray(t2).tobytes() == t2_ref.tobytes()
    assert np.asarray(q2).tobytes() == q2_ref.tobytes()
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "0")  # fp64 interface path
    t2, q2 = _wrf_water_2m(*(rows[:, k].astype(np.float64) for k in range(7)))
    # Row -3 sits exactly on the REAL 1.E-5 threshold: f32(1e-5) widened to f64 is
    # below the f64 literal, so only the REAL path can reproduce WRF's branch there.
    keep = np.ones(len(rows), bool)
    keep[-3] = False
    np.testing.assert_allclose(np.asarray(t2)[keep], t2_ref[keep], rtol=2e-6, atol=0)
    np.testing.assert_allclose(np.asarray(q2)[keep], q2_ref[keep], rtol=2e-5, atol=1e-8)
    assert np.array_equal(np.asarray(t2)[-5:-3], rows[-5:-3, 0])  # CHS2 < 1e-5 -> TSK


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_surface_step_history_uses_wrf_water_form_on_iswater(monkeypatch):
    import gpuwrf.physics.noahmp_coupler as coupler
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view, noahmp_surface_step

    seen = []
    real_sfclay = coupler.surface_layer_with_diagnostics

    def _recording(*args, **kwargs):
        out = real_sfclay(*args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(coupler, "surface_layer_with_diagnostics", _recording)
    state, land, static, rad, clock = _surface_state()
    iswater = int(static.parameters.iswater)
    # columns 3/4 are water; column 3 ISWATER (bulk form), column 4 another water class.
    static = static.replace(ivgtyp=static.ivgtyp.at[0, 3].set(iswater))
    _s, _l, fields = noahmp_surface_step(state, land, static, 90.0, radiation=rad, clock=clock, history=True)
    diag = seen[-1]
    view = _build_column_view(state, None)
    qfx = np.asarray(diag.fluxes.rhosfc) * np.asarray(diag.fluxes.qv_flux)
    t2_ref, q2_ref = coupler._wrf_water_2m(view.t_skin, view.psfc, diag.hfx, qfx, diag.qsfc, diag.chs2, diag.cqs2)
    assert np.asarray(fields["t2"])[0, 3] == np.asarray(t2_ref)[0, 3]
    assert np.asarray(fields["q2"])[0, 3] == np.asarray(q2_ref)[0, 3]
    assert np.asarray(fields["t2"])[0, 4] == np.asarray(diag.t2, np.float64)[0, 4]
    assert np.asarray(fields["t2"])[0, 3] != np.asarray(diag.t2, np.float64)[0, 3]


def test_chs2_cqs2_match_pristine_r8_mynn_oracle(tmp_path, monkeypatch):
    """Production fp64 CHS2/CQS2 vs the REAL*8 pristine SFCLAY1D_mynn on real PROD d01 columns.

    Independent of the port: byte-identical pristine module_sf_mynn.F built with
    -fdefault-real-8 (proofs/v090/build_oracle.sh DOUBLE=1), fed the production column
    view of the BP43 PROD carry plus CPU-WRF warm inputs (b-phys mynn_sl_oracle_gate flow).
    CQS2 must use WRF's flux-loop PSIQ2 = ln((2+z_q)/z_q) - PSIH2 (module_sf_mynn.F:1024),
    not the first-loop ln((2+ZNT)/z_q) form (:976): rel 2e-5 (water) / 3-5 % (land) without it.
    """
    import importlib.util
    import pickle

    import netCDF4
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    inputs = Path("<USER_HOME>/wrf_gpu2_lanes/b-phys/BP43/inputs.pkl")
    wrfout = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/"
                  "attempt_001/cpu_case/wrfout_d01_2026-07-26_13:00:18")
    if not inputs.exists() or not wrfout.exists():
        pytest.skip("real PROD operands (BP43 carry / CPU-WRF 13z frame) unavailable")
    assert _GFORTRAN.exists(), f"pristine oracle compiler missing: {_GFORTRAN} (set GF/FC)"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GF=str(_GFORTRAN), OUT_DIR=str(tmp_path), DOUBLE="1")
    subprocess.run(["bash", str(root / "proofs/v090/build_oracle.sh")], check=True, capture_output=True, text=True,
                   env=env)
    exe = tmp_path / "mynn_oracle_r8"
    spec = importlib.util.spec_from_file_location("mynn_gate", root / "tests/v025/b_phys/mynn_sl_oracle_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    def oracle_run(cols, dx):
        # the pristine driver holds at most N=512 columns per call
        n = len(cols["u"])
        parts = [gate.run_oracle(exe, {k: v[i:i + 512] for k, v in cols.items()}, dx, itimestep=2, real4=False)
                 for i in range(0, n, 512)]
        return {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}

    with inputs.open("rb") as f:
        state, grid, *_ = pickle.load(f)["d01"]
    view = _build_column_view(state, grid)
    with netCDF4.Dataset(wrfout) as nc:
        frame = {k: np.asarray(nc.variables[k][0], np.float64) for k in ("HFX", "QFX", "PBLH", "UST", "Q2")}
    dx = float(grid.projection.dx_m)
    warm = dict(hfx=frame["HFX"], qfx=frame["QFX"], pblh=frame["PBLH"], ust=frame["UST"],
                mol=np.zeros_like(frame["HFX"]), qsfc=-np.ones_like(frame["HFX"]))
    cols = gate.columns_from_view(view, warm)
    pre = oracle_run(cols, dx)
    land = cols["xland"] < 1.5
    q2 = frame["Q2"].reshape(-1)
    cols["mol"] = pre["mol"]
    cols["qsfc"] = np.where(land, q2 / (1.0 + q2), pre["qsfc"])
    for k in ("mol", "hfx", "qfx", "qsfc", "pblh"):
        cols[k] = cols[k].astype(np.float32).astype(np.float64)
    oracle = oracle_run(cols, dx)
    shape = view.xland.shape
    leaves = {k: jnp.asarray(cols[k].reshape(shape)) for k in ("mol", "hfx", "qfx", "qsfc", "pblh")}
    wired = state.replace(ustar=jnp.asarray(cols["ust"].reshape(shape)), **leaves)
    # R8 oracle of the f64 entry: pin it under the v0.3 release defaults too (SFCLAY_NATIVE_REAL=1 runs
    # the WRF-REAL entry, gated separately by tests/v025/b_phys/test_sfclay_native_real.py).
    import gpuwrf.physics.fp32.surface_layer_real as sfclay_real
    monkeypatch.setattr(sfclay_real, "_NATIVE_REAL", False)
    diag = surface_layer_with_diagnostics(_build_column_view(wired, grid), first_timestep=False)
    assert np.asarray(diag.cqs2).dtype == np.float64
    assert (~land).sum() > 1000 and land.sum() > 1000
    for name in ("chs2", "cqs2"):
        got = np.asarray(getattr(diag, name), np.float64).reshape(-1)
        rel = np.abs(got - oracle[name]) / np.maximum(np.abs(oracle[name]), 1e-30)
        assert float(rel.max()) <= 1e-10, f"{name} rel {rel.max():.3e} vs pristine R8 SFCLAY1D_mynn"
