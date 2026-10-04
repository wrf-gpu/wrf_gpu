"""Noah-MP grid ALBEDO and snow age in WRF driver order (b-noahmp NF10).

WRF: the driver keeps the previous ALBEDO while NOAHMP_SFLX returns SALB = -999.9 (SWDOWN = 0;
module_sf_noahmpdrv.F:1230-1232), the first step starts from the landuse_init ALBEDO (physics_init.F:1958-1965),
and SNOW_AGE / SNOWALB_CLASS run once per step, inside ENERGY's ALBEDO (module_sf_noahmplsm.F:2925-2945).
The port stored SALB raw (-999.9 on every land cell at night in wrfout; RRTMG fell back to the snow-free LANDUSE
table at the first sunrise call), seeded ALBOLD (0.65) as the grid albedo, and aged TAUSS/ALBOLD a second time in
SNOWWATER (2x snow age, 2x daylight ALBOLD decay, fresh-snow increments twice).

Oracle 1 (seed): the carry seed == the CPU-WRF lead-zero ALBEDO, every cell (Swiss d01 from the shipped example;
WN3 0227 d01-d03 and PROD d01/d02 where mounted).
Oracle 2 (24 h): proofs/noahmp/noahmp_offline_driver.F90 with NOAHMP_SCHED=1 (per-step forcing + the driver line
verbatim; NOAHMP_WIRE_SOIL/FICEOLD=1) on 16 real columns: Swiss d01 4 snow + 2 snow-free (dt 18 s), WN3 0227 d01
and PROD d01 5 each (dt 54 s); state and ALBEDO entering step 1 = CPU-WRF lead zero, forcing = the 25 hourly CPU-WRF
frames (albedo_schedule.py), night -> sunrise -> day -> sunset -> night. Every step: ALBEDO, TAUSS, ALBOLD.
Bounds: pristine WRF moves Swiss snow ALBEDO by up to 4.6e-3 at low sun (SWDOWN < 50 W/m2) under a 1-ulp SFCTMP
change, 6e-8 above (E100); measured port: low sun 1.1e-3, full sun 4.8e-7, night 1.4e-6, TAUSS/ALBOLD see TOL.
Release path (GPUWRF_NOAHMP_NATIVE_REAL=1). Each hunk has a deletion mutant that must fail (E39).
"""
import importlib.util
import inspect
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE))
from albedo_schedule import FIELDS, step_forcing  # noqa: E402

LOW_SUN = 50.0  # W/m2
TOL = {"albedo_low": 5e-3, "albedo": 1e-4, "night": 5e-3, "tauss": 1e-4, "albold": 1e-4}
CPU = {"swiss": Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_00:00:00")}
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
D5 = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/"
          "attempt_001/cpu_case")
SEEDS = [("swiss", ROOT / "examples/switzerland_d01", "d01", None)]
SEEDS += [("wn3_0227", WN3, d, WN3 / f"wrfout_{d}_2026-02-28_00:00:00") for d in ("d01", "d02", "d03")]
SEEDS += [("prod", PROD, d, D5 / f"wrfout_{d}_2026-07-26_00:00:00") for d in ("d01", "d02")]


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
    fx = json.loads((FIXTURES / "albedo_hold.json").read_text())
    rec = np.load(FIXTURES / "albedo_hold_steps.npz")
    out = {}
    for name, spec in fx.items():
        cols = spec["columns"]
        c = snow._build(cols)
        c["land"] = c["land"].replace(albedo=jnp.asarray(
            np.asarray([col["albedo0"] for col in cols], np.float32).astype(np.float64).reshape(1, -1)))
        c["sched"] = np.stack([step_forcing(col["hourly"], col["dt"], spec["nsteps"]) for col in cols], 0)
        c["ref"] = {k: rec[f"{name}_{k}"] for k in ("salb", "albedo", "tauss", "albold")}
        c["nsteps"], c["names"] = spec["nsteps"], [col["name"] for col in cols]
        out[name] = c
    return out


def _run(c, nsteps):
    """noah_mp_step (jitted, current module globals) with the per-step forcing; ALBEDO/TAUSS/ALBOLD after each step."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.noahmp import noahmp_driver as drv
    step = jax.jit(lambda ls, fo: drv.noah_mp_step(ls, fo, c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"])[0])
    names = {"q2": "qair"}
    land, out = c["land"], {k: np.zeros((c["sched"].shape[0], nsteps)) for k in ("albedo", "tauss", "albold")}
    for k in range(nsteps):
        fo = c["forcing"]._replace(**{names.get(f, f): jnp.asarray(c["sched"][:, k, j].astype(np.float64).reshape(1, -1))
                                      for j, f in enumerate(FIELDS)})
        land = step(land, fo)
        for key in out:
            out[key][:, k] = np.asarray(getattr(land, key), np.float64)[0]
    return out


def _worst(c, port):
    n = port["albedo"].shape[1]
    ref = {k: v[:, :n].astype(np.float64) for k, v in c["ref"].items()}
    night = ref["salb"] < -999
    low = ~night & (c["sched"][:, :n, FIELDS.index("soldn")] < LOW_SUN)
    d = np.abs(port["albedo"] - ref["albedo"])
    first = np.cumsum(~night, 1) == 0
    pick = lambda m: float(d[m].max()) if m.any() else 0.0  # noqa: E731
    return {"albedo_low": pick(low), "albedo": pick(~night & ~low), "night": pick(night),
            "tauss": float(np.abs(port["tauss"] - ref["tauss"]).max()), "albold": float(np.abs(port["albold"] - ref["albold"]).max()),
            "first_night_exact": bool((port["albedo"][first] == ref["albedo"][first]).all()),
            "sentinel": int((port["albedo"] < -999).sum())}


def _fails(w):
    return (not w["first_night_exact"]) or w["sentinel"] > 0 or any(w[k] > t for k, t in TOL.items())


@pytest.mark.parametrize("name", ["swiss", "wn3_0227", "prod"])
def test_albedo_and_snow_age_match_pristine_wrf_driver_over_24h(cases, name):
    w = _worst(cases[name], _run(cases[name], cases[name]["nsteps"]))
    assert w["sentinel"] == 0 and w["first_night_exact"], w
    for key, tol in TOL.items():
        assert w[key] <= tol, (name, key, w)


# The second SNOW_AGE/SNOWALB_CLASS of FINAL 1fece48da (snow.py), for the deletion mutant.
FINAL_SNOW_AGING = """
from gpuwrf.physics.noahmp.precision import real_scalar
def _snow_age(dt, tg, sneqvo, sneqv, tauss):
    dela0 = dt / 1.0e6
    tg_safe = jnp.where(tg > 0.0, tg, TFRZ)
    arg = 5000.0 * (1.0 / TFRZ - 1.0 / tg_safe)
    tage = jnp.exp(arg) + jnp.exp(jnp.minimum(0.0, 10.0 * arg)) + 0.3
    sge = (tauss + dela0 * tage) * (1.0 - jnp.maximum(0.0, sneqv - sneqvo) / 1.00)
    return jnp.where(sneqv <= 0.0, jnp.asarray(0.0, tauss.dtype), jnp.maximum(0.0, sge)), None
def _snowalb_class(qsnow, dt, albold):
    alb = 0.55 + (albold - 0.55) * jnp.exp(real_scalar(-0.01 * dt / 3600.0))
    return jnp.where(qsnow > 0.0, alb + jnp.minimum(qsnow, 1.00 / dt) * (0.84 - alb) / (1.00 / dt), alb)
"""


def _mutate(monkeypatch, which):
    from gpuwrf.physics.noahmp import energy, noahmp_driver as drv, snow
    if which == "hold":    # carry ALBEDO = raw SALB (-999.9 at night)
        mod, fn, old, new = energy, "noahmp_energy_canopy", \
            "albedo=jnp.where(rad.albedo > -999.0, rad.albedo, land_state.albedo.astype(rad.albedo.dtype)),", "albedo=rad.albedo,"
        target = drv
    else:                  # SNOWWATER ages TAUSS/ALBOLD again
        mod, fn, target = snow, "noahmp_snow", drv
        old = "        sneqvo=sneqvo,\n"
        new = ("        sneqvo=sneqvo,\n        tauss=_snow_age(dt, land_state.tg.astype(dtype), sneqvo, sneqv_n, land_state.tauss.astype(dtype))[0],\n"
               "        albold=jnp.where(forcing.cosz > 0.0, _snowalb_class(qsnow, dt, land_state.albold.astype(dtype)), land_state.albold.astype(dtype)),\n")
    src = textwrap.dedent(inspect.getsource(getattr(mod, fn)))
    assert src.count(old) == 1, which
    src = src.replace(old, new)
    if src.startswith("@partial"):
        src = src[src.index("\ndef ") + 1:]   # re-jitted by the caller's jax.jit
    ns = dict(vars(mod))
    if which == "age":
        exec(FINAL_SNOW_AGING, ns)
    exec(compile(src, mod.__file__, "exec"), ns)
    monkeypatch.setattr(target, fn, ns[fn])


@pytest.mark.parametrize("which,name,nsteps", [("hold", "wn3_0227", 4), ("hold", "swiss", 4), ("age", "swiss", 200)])
def test_each_hunk_is_deletion_sensitive(cases, monkeypatch, which, name, nsteps):
    import jax
    _mutate(monkeypatch, which)
    jax.clear_caches()
    try:
        w = _worst(cases[name], _run(cases[name], nsteps))
    finally:
        jax.clear_caches()
    assert _fails(w), (which, w)


def test_snow_free_columns_bitwise_except_albold(cases, monkeypatch):
    """The snow-age fix moves only ALBOLD on snow-free columns (SNOWALB_CLASS runs whenever COSZ > 0, FSNO = 0 keeps it
    out of the ground albedo): every other carry leaf is bitwise unchanged over a sunrise window (eager)."""
    import jax
    c = cases["swiss"]
    keep = [i for i, n in enumerate(c["names"]) if "nosnow" in n]
    leaves = lambda r: [np.asarray(x)[..., keep] for x in jax.tree_util.tree_leaves(r.replace(albold=r.albold * 0))]  # noqa: E731
    from gpuwrf.physics.noahmp import noahmp_driver as drv

    def steps():
        land = c["land"]
        names = {"q2": "qair"}
        import jax.numpy as jnp
        with jax.disable_jit():
            for k in range(1400, 1404):   # 07:00Z sunrise in the Swiss schedule (daylight from k = 1401)
                fo = c["forcing"]._replace(**{names.get(f, f): jnp.asarray(c["sched"][:, k, j].astype(np.float64).reshape(1, -1))
                                              for j, f in enumerate(FIELDS)})
                land = drv.noah_mp_step(land, fo, c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"])[0]
        return land
    new = steps()
    _mutate(monkeypatch, "age")
    old = steps()
    assert float(np.asarray(c["sched"][keep][:, 1400:1404, FIELDS.index("cosz")]).max()) > 0  # daylight reached
    for a, b in zip(leaves(new), leaves(old), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(np.asarray(new.albold)[..., keep], np.asarray(old.albold)[..., keep])


@pytest.mark.parametrize("name,run_dir,domain,cpu", SEEDS, ids=[f"{s[0]}_{s[2]}" for s in SEEDS])
def test_carry_albedo_seed_is_the_wrf_lead_zero_albedo(name, run_dir, domain, cpu):
    import jax
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    if cpu is None:
        reference = np.load(FIXTURES / "albedo_hold_steps.npz")["swiss_t0_albedo"]
    else:
        if not (cpu.exists() and run_dir.exists()):
            pytest.skip("case not mounted")
        with Dataset(cpu) as ds:
            reference = np.asarray(ds["ALBEDO"][0])
    land, _, meta = build_noahmp_land_state(run_dir, domain)
    np.testing.assert_array_equal(np.asarray(jax.device_get(land.albedo)).astype(np.float32), reference)
    assert "landuse_init" in meta["cold_init_provenance"]["albedo"]


def test_seed_deletion_mutant_fails():
    """Deleting the seed hunk (carry ALBEDO = ALBOLD 0.65) breaks the Swiss lead-zero equality."""
    import jax
    from gpuwrf.io import noahmp_land_init as init
    src = textwrap.dedent(inspect.getsource(init._build_noahmp_land_state))
    start = src.index("    from gpuwrf.io.land_history import wrf_initial_albedo\n")
    end = src.index("    provenance[\"albedo\"]")
    src = src[:start] + "    albedo = albold\n" + src[end:]
    ns = dict(vars(init))
    exec(compile(src, init.__file__, "exec"), ns)
    from gpuwrf.io.gen2_accessor import Gen2Run
    run = Gen2Run(ROOT / "examples/switzerland_d01")
    with run.input_read_scope("d01"):
        land, _, _ = ns["_build_noahmp_land_state"](ROOT / "examples/switzerland_d01", "d01", _run=run)
    reference = np.load(FIXTURES / "albedo_hold_steps.npz")["swiss_t0_albedo"]
    assert not np.array_equal(np.asarray(jax.device_get(land.albedo)).astype(np.float32), reference)


def test_rrtmg_ignores_surface_albedo_at_night_only():
    """The radiation input changes only where the carry ALBEDO was the night value: RRTMG at coszen <= 0 is
    bitwise independent of the surface albedo, so the effective window is the first sunrise call (and t0)."""
    topo = _module("rrtmg_topo", ROOT / "tests/test_rrtmg_topographic_coupling.py")
    import jax.numpy as jnp
    from gpuwrf.coupling.physics_couplers import build_radiation_static_from_wrf_fields, rrtmg_radiation_diagnostics
    grid = topo._grid()
    ny, nx = grid.ny, grid.nx
    lat, lon = jnp.full((ny, nx), 28.3), jnp.full((ny, nx), -16.4)
    static = build_radiation_static_from_wrf_fields(lat, lon, jnp.zeros((ny, nx)), dx_m=grid.projection.dx_m,
                                                    dy_m=grid.projection.dy_m, msftx=grid.metrics.msftx,
                                                    msfty=grid.metrics.msfty, sina=grid.metrics.sina, cosa=grid.metrics.cosa)

    def diag(hour, albedo):
        land = SimpleNamespace(albedo=jnp.full((ny, nx), albedo, dtype=jnp.float64), emiss=jnp.full((ny, nx), 0.93))
        return rrtmg_radiation_diagnostics(topo._state(grid), grid, time_utc=datetime(2026, 2, 28, hour, tzinfo=timezone.utc),
                                           lead_seconds=0.0, radiation_static=static, topo_shading=0, slope_rad=0,
                                           land_state=land)
    fields = lambda d: {k: np.asarray(v) for k, v in d._asdict().items() if k != "surface_albedo" and v is not None}  # noqa: E731
    night_held, night_sentinel = fields(diag(0, 0.61)), fields(diag(0, -999.9))
    assert float(np.asarray(diag(0, 0.61).coszen).max()) <= 0
    for k in night_held:
        np.testing.assert_array_equal(night_held[k], night_sentinel[k], err_msg=k)
    day_held, day_sentinel = fields(diag(13, 0.61)), fields(diag(13, -999.9))
    assert not np.array_equal(day_held["sw_toa_up"], day_sentinel["sw_toa_up"])
