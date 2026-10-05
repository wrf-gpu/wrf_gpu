"""Noah-MP snow-film reset and the QSNOWXY/QRAINXY history rates (b-noahmp NF12, v0.3.1, alisios §3.3).

WRF: NOAHMP_SFLX ends with IF (SNOWH <= 1.E-6 .OR. SNEQV <= 1.E-6) SNOWH = SNEQV = 0 (module_sf_noahmplsm.F:1067-1070),
so canopy-ice unloading (ICEDRIP = CANICE*(FV+FT), :1489-1491) onto snow-free ground never leaves a snowpack. The port
had SNOWH2O's SNOWH <= 1e-8 reset only: a frost-unloading film survived (CPU-WRF SNOW == 0).
QSNOWXY/QRAINXY are PRECIP_HEAT's ground snow/rain rates QSNOW/QRAIN (:1547-1548) written by the driver
(module_sf_noahmpdrv.F:1258-1259); the writer carried a column sum of the atmospheric QSNOW/QRAIN mixing ratios.

Oracle: proofs/noahmp/noahmp_offline_driver.F90 (NOAHMP_SCHED=1, NOAHMP_WIRE_SOIL/FICEOLD=1) on 11 real WN3 land
columns (proofs/noahmp/snow_film/build_columns.py): 0120 d02 Teide frost cells, 6 h from 2026-01-23_18 (dt 18 s), and
0227 d03 rain/frost cells, 16 h from 2026-02-28_12 (dt 6 s); state = the CPU-WRF history frame, forcing = the hourly
CPU-WRF frames (albedo_schedule.py). Pristine SNEQV is 0 on every step while QSNOW > 0 on thousands of them.
Bounds (E100) = the WRF-only envelope of each case (TOL below): pristine Noah-MP rebuilt -O0 / -O3 -mfma, 1-ulp
forcing arms and their combination. QSNOW flips between 0 and <= 1.084e-12 mm/s while CANICE sits at round-off
(deposition/unloading alternating every step) -> zero/non-zero mismatches are allowed only below FLOOR.
SNEQV/SNOWH must be EXACTLY 0 wherever pristine has 0 (no WRF variant moves them there); onset/melt-out steps exact.
Pack edges: onset_0227 (light -> moderate snowfall at 271 K, dt 6 s) and melt_0227 (2 mm pack melting out),
proofs/noahmp/snow_film/build_edges.py; the pristine census is asserted (non-vacuity, E186).
Release path (GPUWRF_NOAHMP_NATIVE_REAL=1); each hunk has a deletion mutant that must fail (E39).
"""
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE))
from albedo_schedule import FIELDS, step_forcing  # noqa: E402

SETS = {"teide_0120": "snow_film", "rain_0227": "snow_film", "onset_0227": "snow_film_edges", "melt_0227": "snow_film_edges"}
CASES = tuple(SETS)
# Bounds = the WRF-only envelope per case (NF12/wrf_envelopes.json): max |pristine variant - pristine -O2| over the
# pristine module_sf_noahmplsm built -O0 and -O3 -mfma -ffp-contract=fast and 9 single-input +-1 ulp forcing arms
# (SFCTMP/Q2/LWDN +-, SOLDN/PRCPNONC/PSFC +); never derived from the port error. Combined arms: the -O3 -mfma build under each
# forcing arm. Exact-zero cells (TOL 0.0) are exact in every WRF variant.
TOL = {
    "teide_0120": {"qsnow": 4.46e-09, "qrain": 6.41e-10, "sneqv": 0.0, "snowh": 0.0},
    "rain_0227": {"qsnow": 9.72e-10, "qrain": 1.21e-09, "sneqv": 0.0, "snowh": 0.0},
    "onset_0227": {"qsnow": 1.18e-08, "qrain": 0.0, "sneqv": 7.08e-08, "snowh": 7.9e-10},
    "melt_0227": {"qsnow": 0.0, "qrain": 1.1e-09, "sneqv": 1.36e-05, "snowh": 1.38e-07},
}   # mm/s (QSNOW/QRAIN), mm (SNEQV), m (SNOWH); worst arms in NF12/wrf_envelopes.json
FLOOR = 1.09e-12   # mm/s; pristine QSNOW zero/non-zero flips at CANICE round-off reach 1.084e-12


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def native():
    import jax
    mp = pytest.MonkeyPatch()
    mp.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    jax.clear_caches()
    yield
    mp.undo()
    jax.clear_caches()


@pytest.fixture(scope="module")
def cases(native):
    import jax.numpy as jnp
    snow = _module("snow_wrf_order_oracle", HERE / "test_snow_wrf_order_oracle.py")
    out = {}
    for name, base in SETS.items():
        spec = json.loads((FIXTURES / f"{base}.json").read_text())[name]
        rec = np.load(FIXTURES / f"{base}_steps.npz")
        cols = spec["columns"]
        c = snow._build(cols)
        c["land"] = c["land"].replace(albedo=jnp.asarray(
            np.asarray([col["albedo0"] for col in cols], np.float32).astype(np.float64).reshape(1, -1)))
        c["sched"] = np.stack([step_forcing(col["hourly"], col["dt"], spec["nsteps"]) for col in cols], 0)
        c["ref"] = {k: rec[f"{name}_{k}"].astype(np.float64) for k in ("qsnow", "qrain", "sneqv", "snowh")}
        c["nsteps"], c["names"] = spec["nsteps"], [col["name"] for col in cols]
        out[name] = c
    return out


def _run(c, nsteps):
    """noah_mp_step(history=True) (jitted, current module globals); SNEQV/SNOWH and QSNOWXY/QRAINXY after each step."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.noahmp import noahmp_driver as drv

    def one(ls, fo):
        land, fluxes = drv.noah_mp_step(ls, fo, c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"], history=True)
        return land, (land.sneqv, land.snowh, fluxes.history["QSNOWXY"], fluxes.history["QRAINXY"])
    step = jax.jit(one)
    names = {"q2": "qair"}
    land, out = c["land"], np.zeros((4, c["sched"].shape[0], nsteps))
    for k in range(nsteps):
        fo = c["forcing"]._replace(**{names.get(f, f): jnp.asarray(c["sched"][:, k, j].astype(np.float64).reshape(1, -1))
                                      for j, f in enumerate(FIELDS)})
        land, rows = step(land, fo)
        out[:, :, k] = np.asarray(jax.device_get(rows), np.float64)[:, 0]
    return dict(zip(("sneqv", "snowh", "qsnow", "qrain"), out))


def _edges(sneqv, sneqv0):
    """Per column: first step with snow on snow-free ground (onset) and first step without snow after snow (melt-out)."""
    z = np.concatenate([np.asarray(sneqv0).reshape(-1, 1) == 0, sneqv == 0], 1)
    onset = [int(np.flatnonzero(zz[:-1] & ~zz[1:])[0]) if (zz[:-1] & ~zz[1:]).any() else -1 for zz in z]
    gone = [int(np.flatnonzero(~zz[:-1] & zz[1:])[0]) if (~zz[:-1] & zz[1:]).any() else -1 for zz in z]
    return onset, gone


def _worst(c, port):
    n = port["sneqv"].shape[1]
    ref = {k: v[:, :n] for k, v in c["ref"].items()}
    zero = ref["sneqv"] == 0.0
    sneqv0 = np.asarray(c["land"].sneqv)[0]
    return {"film_steps": int(((port["sneqv"] != 0) | (port["snowh"] != 0))[zero].sum()),
            "snow_pattern": int(((port["sneqv"] == 0) != zero).sum()),
            "edges_port": _edges(port["sneqv"], sneqv0), "edges_ref": _edges(ref["sneqv"], sneqv0),
            **{k: float(np.abs(port[k] - ref[k]).max()) for k in ("qsnow", "qrain", "sneqv", "snowh")},
            "qsnow_zero_mismatch": int(((ref["qsnow"] > FLOOR) & (port["qsnow"] == 0)).sum()
                                       + ((port["qsnow"] > FLOOR) & (ref["qsnow"] == 0)).sum())}


def _fails(w, name):
    return (w["film_steps"] > 0 or w["snow_pattern"] > 0 or w["qsnow_zero_mismatch"] > 0
            or w["edges_port"] != w["edges_ref"] or any(w[k] > t for k, t in TOL[name].items()))


@pytest.fixture(scope="module")
def runs(cases):
    return {name: _run(cases[name], cases[name]["nsteps"]) for name in CASES}


def test_pristine_records_exercise_every_film_edge(cases):
    """Non-vacuity (E186), on the PRISTINE records: the reset branch fires (QSNOW > 0 while SNEQV stays 0) on the frost and
    snowfall-onset sets, every onset column starts a pack, every melt column melts out through the thresholds."""
    for name, c in cases.items():
        ref, ncol = c["ref"], c["ref"]["sneqv"].shape[0]
        resets = int(((ref["qsnow"] > FLOOR) & (ref["sneqv"] == 0)).sum())
        onset, gone = _edges(ref["sneqv"], np.asarray(c["land"].sneqv)[0])
        if name in ("teide_0120", "rain_0227"):
            assert resets > 100, (name, resets)
        if name == "onset_0227":
            assert resets > 1000 and all(k > 0 for k in onset), (resets, onset)
        if name == "melt_0227":
            assert all(k > 0 for k in gone) and float(ref["sneqv"][:, 0].min()) > 1.0, gone   # gradual melt-out


@pytest.mark.parametrize("name", CASES)
def test_snow_film_and_ground_rates_match_pristine_wrf(cases, runs, name):
    """Every step: SNEQV/SNOWH == 0 EXACTLY wherever pristine SNEQV == 0 (and nowhere else), onset/melt-out on the same step,
    QSNOWXY/QRAINXY/SNEQV/SNOWH within the WRF-only envelope of that case."""
    w = _worst(cases[name], runs[name])
    assert w["film_steps"] == 0 and w["snow_pattern"] == 0 and w["qsnow_zero_mismatch"] == 0, w
    assert w["edges_port"] == w["edges_ref"], w
    for key, tol in TOL[name].items():
        assert w[key] <= tol, (name, key, w)


def test_cpu_wrf_hourly_ground_rates_are_reproduced(cases, runs):
    """CPU-WRF's own hourly frames on the Teide frost cells (the end-of-hour step of the schedule): SNOW == 0 exactly, and
    wherever pristine-on-schedule unloads canopy ice, CPU-WRF's QSNOWXY is the same quantity (0.2-2x; the hourly-linear
    forcing delays frost onset by up to 1 h vs CPU-WRF, so this is a plausibility check, not the fidelity bound)."""
    fx = json.loads((FIXTURES / "snow_film.json").read_text())
    c, port = cases["teide_0120"], runs["teide_0120"]
    per_hour = int(round(3600 / c["dt"]))
    checked = 0
    for i, col in enumerate(fx["teide_0120"]["columns"]):
        for h in range(1, len(col["cpu_hourly"]["SNOW"])):
            k = h * per_hour - 1
            cpu = col["cpu_hourly"]["QSNOWXY"][h]
            if col["cpu_hourly"]["SNOW"][h] == 0.0:
                assert port["sneqv"][i, k] == 0.0 and port["snowh"][i, k] == 0.0, (col["name"], h)
            if c["ref"]["qsnow"][i, k] > FLOOR:
                assert cpu > 0.0 and 0.2 <= port["qsnow"][i, k] / cpu <= 2.0, (col["name"], h, cpu, port["qsnow"][i, k])
                checked += 1
    assert checked >= 15, checked


def _mutate(monkeypatch, which):
    from gpuwrf.physics.noahmp import noahmp_driver as drv
    src = textwrap.dedent(inspect.getsource(drv.noah_mp_step))
    if which == "film":    # drop the NOAHMP_SFLX end-of-step reset
        start = src.index("    film = (land_state.snowh <= 1.0e-6)")
        end = src.index("\n", src.index("sneqv=jnp.where(film,")) + 1
        src = src[:start] + src[end:]
    else:                  # QSNOWXY = the PRECIP_HEAT input snowfall instead of the ground rate
        old = '"QSNOWXY": precip.qsnow,'
        assert src.count(old) == 1
        src = src.replace(old, '"QSNOWXY": forcing.prcpsnow,')
    ns = dict(vars(drv))
    exec(compile(src, drv.__file__, "exec"), ns)
    monkeypatch.setattr(drv, "noah_mp_step", ns["noah_mp_step"])


@pytest.mark.parametrize("which,name,nsteps", [("film", "teide_0120", 1200), ("film", "onset_0227", 4800),
                                                ("rate", "teide_0120", 1200)])
def test_each_hunk_is_deletion_sensitive(cases, monkeypatch, which, name, nsteps):
    import jax
    _mutate(monkeypatch, which)
    jax.clear_caches()
    try:
        w = _worst(cases[name], _run(cases[name], nsteps))
    finally:
        jax.clear_caches()
    assert _fails(w, name), (which, w)


def test_coupler_exports_ground_rates_to_the_writer_stream(monkeypatch):
    """noahmp_surface_adapter(land_history=True) carries PRECIP_HEAT QSNOW/QRAIN as QSNOWXY/QRAINXY on land (water keeps
    WRF's init 0); the writer takes them from the land history (no mixing-ratio column-sum proxy any more)."""
    import jax.numpy as jnp
    sys.path.insert(0, str(HERE.parents[2] / "tests"))
    from test_noahmp_coupler import _build
    import gpuwrf.physics.noahmp.noahmp_driver as driver
    from gpuwrf.io import wrfout_writer as w
    from gpuwrf.io.land_history import LAND_HISTORY_FIELDS
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing, noahmp_surface_adapter
    state, land, static, rad, clock = _build()
    fo = assemble_noahmp_forcing(state, static, rad, clock, 90.0)
    rate = jnp.linspace(1e-4, 5e-4, fo.sfctmp.size).reshape(fo.sfctmp.shape).astype(fo.sfctmp.dtype)
    fo = fo._replace(prcpnonc=rate, prcpsnow=0.5 * rate, sfctmp=jnp.full_like(fo.sfctmp, 275.5))  # Jordan FPICE 0.6
    captured = []
    original = driver.noahmp_precip_heat
    monkeypatch.setattr(driver, "noahmp_precip_heat", lambda *a, **k: captured.append(original(*a, **k)) or captured[-1])
    _, _, _, fields = noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=90, forcing=fo, land_history=True)
    h, precip = fields["land_history"], captured[0][0]
    is_land = np.asarray(state.xland < 1.5)
    assert is_land.any() and (~is_land).any()
    for name, value in (("QSNOWXY", precip.qsnow), ("QRAINXY", precip.qrain)):
        np.testing.assert_array_equal(h[name], np.where(is_land, value, 0), err_msg=name)
        assert float(np.asarray(h[name])[is_land].min()) > 0, name
        assert name in LAND_HISTORY_FIELDS and name in w.WRFOUT_VARIABLE_SPECS
    assert w._full_derived_value("QSNOWXY", {"QSNOW": np.ones((3, 2, 2))}, grid=None, namelist=None, shape=(2, 2),
                                 dtype=np.float32, run_start=None, lead_hours=0.0) is None


def test_pre_qsnowxy_checkpoint_history_is_upgraded():
    """A checkpoint packed before QSNOWXY/QRAINXY restores the full family: old leaves byte-identical, new rates 0 (the
    next Noah step rewrites them); accumulators are never invented."""
    import jax.numpy as jnp
    from gpuwrf.runtime.history_accumulators import (ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS, PackedFields, as_packed)
    old_names = tuple(n for n in LAND_FLUX_FIELDS if n not in ("QSNOWXY", "QRAINXY"))
    old = PackedFields(old_names, jnp.arange(len(old_names) * 4, dtype=jnp.float32).reshape(len(old_names), 2, 2) + 1)
    new = as_packed(old, LAND_FLUX_FIELDS)
    assert new.names == LAND_FLUX_FIELDS and new.data.dtype == jnp.float32
    for name in old_names:
        assert np.asarray(new[name]).tobytes() == np.asarray(old[name]).tobytes(), name
    for name in ("QSNOWXY", "QRAINXY"):
        assert not np.asarray(new[name]).any()
    energy = PackedFields(ENERGY_ACCUMULATORS[1:], jnp.ones((len(ENERGY_ACCUMULATORS) - 1, 2, 2), jnp.float32))
    assert as_packed(energy, ENERGY_ACCUMULATORS).names == ENERGY_ACCUMULATORS[1:]
