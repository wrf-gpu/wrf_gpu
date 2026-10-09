"""Pristine-WRF one-step oracle on REAL mixed-phase columns for GPUWRF_THOMPSON_MIXED_PHASE_WRF (B55, v0.3.3).

Fixture (proofs/thompson/mixed_phase/build_fixture.py): 384 CPU-WRF history columns with riming, rain freezing or
rain-ice collection somewhere in the column (Swiss 2023-01-15 d01/d02 06/12/18Z, Swiss 2024-11-24 d01/d02, WN3 0227
d02/d03; dt 18 s and 6 s) and ONE pristine mp_gt_driver step (module_mp_thompson.F mp8, tables = the .dat files
CPU-WRF reads). The native C24 full column runs with the flag ON (interpret Pallas on CPU).
Gate on the one-step INCREMENTS, per column and field: excess = (max|d_port - d_WRF| - 2 ulp(max|WRF|) - R1 -
1e-4 * fixture max|d_WRF|) / column max|d_WRF| <= TOL (1e-2; Ni 5e-2: one cell where cloud ice sediments out of the
top ice layer at 207 K, sedimentation, not a source term); surface precipitation alike. Non-finite values fail.
The staged OFF path and every WRF element of the single-pass stage are deletion-sensitive (E39): wet-bulb twet, the
warm rain-snow and rain-graupel branches, the r_s(1)/r_g(1) guards, warm sublimation only while not melting, the
melt enhancement, Hallett-Mossop, the rime split only in the cold block, the melting-snow fall-speed blend,
tcg_racg at WRF's idx_bg1 offset, the cloud-freezing planes at WRF's INTEGER-nic1 idx_n, cloud freezing only below
0 C (rate-level: on the real tables a warm cell's -1 C plane freezes < R1, invisible in any column output).
"""
import contextlib
import inspect
import os
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics import thompson_tables as tt

FIXTURE = Path(__file__).with_name("fixtures") / "mixed_phase_columns.npz"
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0",
           GPUWRF_THOMPSON_IMPLICIT_SED="0", GPUWRF_THOMPSON_MIXED_PHASE_WRF="1")
FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")
TOL = {**{v: 1e-2 for v in FIELDS}, "ni": 5e-2, "precip": 1e-2}
R1_FLOOR = {v: 1e-12 for v in ("qc", "qr", "qi", "qs", "qg")}
SOURCE_MUTANTS = {  # name -> [(function, fixed source, the same source with the WRF element removed), ...]
    "no_wet_bulb": [("_wrf_mixed_phase_sources",
                     "    twet = _wrf_twet(temp, p, qv, qvs, _wrf_melting_band(tempc) & (l_qs | l_qg))\n",
                     "    twet = temp\n")],
    "no_warm_rain_snow": [("_wrf_rain_snow_graupel_rates", "    racs = rain & (rs >= 1.0e-6)\n",
                           "    racs = rain & (rs >= 1.0e-6) & (twet < T_0)\n")],
    "no_warm_rain_graupel": [("_wrf_rain_snow_graupel_rates", "    racg = rain & (rg >= 1.0e-6)\n",
                              "    racg = rain & (rg >= 1.0e-6) & (twet < T_0)\n")],
    "trace_snow_graupel_collection": [("_wrf_rain_snow_graupel_rates", "    racs = rain & (rs >= 1.0e-6)\n",
                                       "    racs = rain & (state.qs > R1)\n"),
                                      ("_wrf_rain_snow_graupel_rates", "    racg = rain & (rg >= 1.0e-6)\n",
                                       "    racg = rain & (state.qg > R1)\n")],
    "no_rain_snow_rebalance": [("_wrf_mixed_phase_sources", "    rebal = twet > T_0\n", "    rebal = twet > 1.0e9\n")],
    "sublimation_while_melting": [("_wrf_mixed_phase_sources",
                                   "    sde_w = warm & l_qs & ~(prr_sml > 0.0) & (ssati < 0.0)\n",
                                   "    sde_w = warm & l_qs & (ssati < 0.0)\n"),
                                  ("_wrf_mixed_phase_sources",
                                   "    gde_w = warm & l_qg & ~(prr_gml > 0.0) & (ssati < 0.0)\n",
                                   "    gde_w = warm & l_qg & (ssati < 0.0)\n")],
    "no_melt_enhancement": [("_wrf_mixed_phase_sources",
                             "prr_sml + 4218.0 * _OLFUS32 * (twet - T_0) * (prr_rcs + prs_scw)", "prr_sml")],
    "no_hallett_mossop": [("_wrf_mixed_phase_sources", "    hm = cold & (prg_gcw > EPS) & (tempc > -8.0)\n",
                           "    hm = cold & (prg_gcw > EPS) & (tempc > 1.0e9)\n")],
    "no_melting_snow_speed": [("_wrf_snow_speed",
                               "jnp.where(vts_boost < 0.0, vts * sr + (1.0 - sr) * vtr.astype(vts.dtype), vts * vts_boost)",
                               "vts * jnp.maximum(vts_boost, 1.0)")],
    "rime_split_above_0c": [("_wrf_mixed_phase_sources", "    prs_sde_c = jnp.where(cold & l_qs, prs_sde_c, 0.0)\n",
                             "    prs_sde_c = jnp.where(l_qs, prs_sde_c, 0.0)\n"),
                            ("_wrf_mixed_phase_sources",
                             "    riming_dom = cold & (prs_scw > 2.0 * prs_sde_c) & (prs_sde_c > EPS)\n",
                             "    riming_dom = (prs_scw > 2.0 * prs_sde_c) & (prs_sde_c > EPS)\n")],
    "cloud_freezing_above_0c": [("_wrf_mixed_phase_sources", "    pri_wfz = jnp.where(cold, pri_wfz, 0.0).astype(f64)\n",
                                 "    pri_wfz = pri_wfz.astype(f64)\n"),
                                ("_wrf_mixed_phase_sources", "    pni_wfz = jnp.where(cold, pni_wfz, 0.0).astype(f64)\n",
                                 "    pni_wfz = pni_wfz.astype(f64)\n")],
}
TABLE_MUTANTS = ("tcg_racg_without_idx_bg1_offset", "qcfz_real_nic1_planes")
MUTANTS = (*SOURCE_MUTANTS, *TABLE_MUTANTS, "staged_off_path")
# Elements no real column exercises above the gate floors (sweep: 0 failures) carry their own deletion-sensitive
# test below: the :2946 re-balance (snow limiter never binds) and the cold-block gate of droplet freezing.
TARGETED = ("no_rain_snow_rebalance", "cloud_freezing_above_0c")
COLUMN_GATE_MUTANTS = tuple(m for m in MUTANTS if m not in TARGETED)
# The 3 worst columns of every mutant in the full 384-column sweep (proofs/thompson/mixed_phase/mutant_sweep.json).
MUTANT_COLUMNS = (
    "swiss_20230115_d02_06_rime_y160_x171",
    "swiss_20230115_d01_18_rime_y16_x76",
    "swiss_20230115_d01_18_rime_y35_x98",
    "swiss_20230115_d02_12_bigg_y136_x94",
    "swiss_20230115_d02_12_bigg_y133_x99",
    "swiss_20230115_d02_12_rime_y141_x94",
    "swiss_20230115_d01_12_bigg_y117_x112",
    "swiss_20230115_d01_12_rci_y117_x112",
    "swiss_20230115_d01_12_rime_y122_x117",
    "swiss_20230115_d01_18_bigg_y14_x118",
    "swiss_20230115_d01_18_bigg_y27_x126",
    "swiss_20230115_d01_12_bigg_y31_x12",
    "swiss_20230115_d01_18_bigg_y35_x117",
    "swiss_20230115_d02_12_rime_y106_x27",
    "swiss_20230115_d02_12_rime_y106_x28",
    "swiss_20230115_d01_12_rci_y46_x33",
    "swiss_20230115_d02_12_rci_y24_x0",
    "swiss_20230115_d01_12_rci_y43_x22",
    "swiss_20230115_d01_12_rime_y120_x99",
    "swiss_20230115_d01_12_rime_y102_x80",
    "swiss_20230115_d01_12_rime_y83_x95",
    "swiss_20230115_d01_12_rci_y40_x21",
    "swiss_20230115_d01_12_rime_y114_x42",
    "swiss_20230115_d01_12_rime_y118_x38",
    "swiss_20230115_d01_12_bigg_y105_x55",
)


def _mutant_tables(name):
    wrf = tt.load_wrf_cold_collection_tables()
    if name == "tcg_racg_without_idx_bg1_offset":
        with np.load(tt.MIXED_PHASE_ASSET) as loaded:
            raw = np.ascontiguousarray(loaded["tcg_racg"])
        return wrf._replace(tcg_racg=jnp.asarray(raw, jnp.float64))
    v1 = tt.load_cold_collection_tables()
    return wrf._replace(tpi_qcfz=v1.tpi_qcfz, tni_qcfz=v1.tni_qcfz)


@contextlib.contextmanager
def _env(mutant=None):
    env = dict(C24, GPUWRF_THOMPSON_MIXED_PHASE_WRF="0" if mutant == "staged_off_path" else "1")
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    sources = {}
    for fn, fixed, removed in SOURCE_MUTANTS.get(mutant, []):
        src = sources.get(fn) or textwrap.dedent(inspect.getsource(getattr(tc, fn)))
        assert src.count(fixed) == 1, (mutant, fixed)
        sources[fn] = src.replace(fixed, removed)
    swaps = {}
    for fn, src in sources.items():
        namespace = dict(vars(tc))
        exec(compile(src, tc.__file__, "exec"), namespace)
        swaps[fn] = namespace[fn]
    if mutant in TABLE_MUTANTS:
        tables = _mutant_tables(mutant)
        swaps["_mixed_phase_cold_tables"] = lambda: tables
    originals = {fn: getattr(tc, fn) for fn in swaps}
    jax.clear_caches()
    try:
        for fn, f in swaps.items():
            setattr(tc, fn, f)
        yield
    finally:
        for fn, f in originals.items():
            setattr(tc, fn, f)
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        jax.clear_caches()


def _run(z, idx, mutant=None):
    f = lambda name: jnp.asarray(z[f"in_{name}"][idx], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"),
                                   Nr=f("nr"), Ns=zero, Ng=zero, T=T, p=p,
                                   rho=tc.density_from_pressure_temperature(p, T, qv), dz=f("dz8w"), w=f("w"))
    with _env(mutant):
        out, ppt = full.full_column(state, float(z["dt"][idx[0]]), interpret=True)
    g = lambda a: np.asarray(a, np.float64)  # noqa: E731
    port = {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qi": g(out.qi), "qs": g(out.qs), "qg": g(out.qg),
            "ni": g(out.Ni), "nr": g(out.Nr),
            "th": g(np.asarray(out.T, np.float32) / np.asarray(z["in_pii"][idx], np.float32))}
    port["precip"] = sum(g(ppt[key]) for key in ("rain", "snow", "graupel", "ice"))
    return port


def _validate_fixture(z):
    """E200: every fixture input and WRF reference value must be finite BEFORE any reduction -- a non-finite value in
    ANY column (selected or not) would otherwise turn the fixture-wide field-scale floor into NaN/inf."""
    bad = [k for k in sorted(z) if k.startswith(("in_", "wrf_")) and not np.isfinite(np.asarray(z[k], np.float64)).all()]
    if bad:
        raise ValueError(f"non-finite fixture values in {bad}")


def _scales(z):
    """Fixture-wide max|d_WRF| per field (the E100 field-scale floor) and max WRF surface precip."""
    _validate_fixture(z)
    out = {v: float(np.abs(z[f"wrf_{v}"].astype(np.float64) - z[f"in_{v}"]).max()) for v in FIELDS}
    out["precip"] = float(z["wrf_rainncv"].astype(np.float64).max())
    return out


def _excess(port, z, idx, scales):
    """Per (column, field) excess in units of the column max|d_WRF| (gate: <= TOL). Non-finite -> inf first."""
    res = {}
    for j, i in enumerate(idx):
        name = str(z["names"][i])
        for v in FIELDS:
            x, w = (np.asarray(z[f"{s}_{v}"][i], np.float64) for s in ("in", "wrf"))
            p = np.asarray(port[v][j], np.float64)
            if not (np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(p).all()):
                res[(name, v)] = np.inf
                continue
            dw, dp = w - x, p - x
            floor = 2.0 * float(np.spacing(np.float32(np.abs(w).max()))) + R1_FLOOR.get(v, 0.0) + 1e-4 * scales[v]
            res[(name, v)] = max(0.0, np.abs(dp - dw).max() - floor) / max(np.abs(dw).max(), 1e-30)
        w, p = float(z["wrf_rainncv"][i]), float(port["precip"][j])
        if not (np.isfinite(w) and np.isfinite(p)):
            res[(name, "precip")] = np.inf
            continue
        floor = 2.0 * float(np.spacing(np.float32(w))) + 1e-4 * scales["precip"]
        res[(name, "precip")] = max(0.0, abs(p - w) - floor) / max(w, 1e-30)
    return res


def failures(z, mutant=None, idx=None):
    scales, bad = _scales(z), []
    sel = np.arange(len(z["names"])) if idx is None else np.asarray(idx)
    for dt in sorted(set(z["dt"][sel].tolist())):
        part = sel[z["dt"][sel] == dt]
        for (name, v), e in _excess(_run(z, part, mutant), z, part, scales).items():
            if e > TOL[v]:
                bad.append((name, v, e))
    return bad


@pytest.fixture(scope="module")
def z():
    return dict(np.load(FIXTURE))


def test_fixture_is_real_mixed_phase(z):
    """E154 census: supercooled cloud + snow/graupel (riming), supercooled rain (freezing), a melting layer with
    snow, and WRF moving snow, graupel, rain, ice number and theta in the step."""
    T = z["in_th"].astype(np.float64) * z["in_pii"]
    cold, warm = T < 273.15, T > 273.15
    assert (cold & (z["in_qc"] > 1e-5) & (z["in_qs"] + z["in_qg"] > 1e-6)).any(-1).sum() >= 100      # riming
    assert (cold & (z["in_qr"] > 1e-6)).any(-1).sum() >= 100                                       # rain freezing
    assert (warm & (z["in_qs"] > 1e-6)).any(-1).sum() >= 50                                        # melting snow
    assert set(z["dt"].tolist()) == {6.0, 18.0} and len(z["names"]) == 384
    for v in ("qs", "qg", "qr", "ni", "th"):
        assert (np.abs(z[f"wrf_{v}"].astype(np.float64) - z[f"in_{v}"]) > 0).any(-1).sum() >= 100, v


def test_mixed_phase_matches_pristine_wrf(z):
    assert failures(z) == []


@pytest.mark.parametrize("mutant", COLUMN_GATE_MUTANTS)
def test_each_wrf_element_is_deletion_sensitive(z, mutant):
    idx = [int(np.flatnonzero(z["names"] == name)[0]) for name in MUTANT_COLUMNS]
    assert failures(z, mutant, idx), mutant


def test_wrf_tables_reproduce_the_cpu_wrf_reads():
    """tcg_racg carries WRF's idx_bg1 read offset (v1 cold tables already do) and the qcfz planes are WRF's
    INTEGER-nic1 idx_n = 66 planes, not v1's idx_n = 59 (proofs/thompson/mixed_phase/tables_receipt.json)."""
    raw = np.load(tt.MIXED_PHASE_ASSET)["tcg_racg"].reshape(-1, order="F")
    eff = tt.mixed_phase_numpy_tables()
    np.testing.assert_array_equal(eff["tcg_racg"].reshape(-1, order="F")[:-tt.IDX_BG1_OFFSET], raw[tt.IDX_BG1_OFFSET:])
    assert not eff["tcg_racg"].reshape(-1, order="F")[-tt.IDX_BG1_OFFSET:].any()
    v1 = np.load(tt.COLD_TABLE_ASSET)
    for name in ("tpi_qcfz", "tni_qcfz"):
        assert eff[name].shape == v1[name].shape and not np.array_equal(eff[name], v1[name])
    wrf = tt.load_wrf_cold_collection_tables()
    for name in ("tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"):
        np.testing.assert_array_equal(np.asarray(getattr(wrf, name)), v1[name])


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("field", (*FIELDS, "precip"))
def test_nonfinite_candidate_fails_its_field(z, field, bad):
    """Gate function alone: WRF's own values pass with zero excess; one non-finite value fails exactly that field."""
    idx = np.flatnonzero(z["dt"] == 18.0)[:12]
    port = {v: np.array(z[f"wrf_{v}"][idx], np.float64) for v in FIELDS}
    port["precip"] = np.array(z["wrf_rainncv"][idx], np.float64)
    scales = _scales(z)
    assert max(_excess(port, z, idx, scales).values()) == 0.0
    j = len(idx) // 2
    if field == "precip":
        port["precip"][j] = bad
    else:
        port[field][j, 7] = bad
    ex = _excess(port, z, idx, scales)
    name = str(z["names"][idx[j]])
    assert ex[(name, field)] == np.inf
    assert all(e == 0.0 for key, e in ex.items() if key != (name, field))


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("key", ("in_qs", "in_th", "in_pii", "wrf_qs", "wrf_ni", "wrf_th", "wrf_rainncv"))
def test_nonfinite_fixture_value_rejects_the_gate(z, key, bad):
    """E200 (review-s2small 03:05Z): one non-finite input or WRF reference value in the LAST column -- outside the
    selected subset -- must reject the gate before the field-scale floors exist (positive control: on the clean
    fixture, WRF's own values with a 1 g/kg snow error in one cell fail)."""
    idx = np.flatnonzero(z["dt"] == 18.0)[:12]
    assert len(z["names"]) - 1 not in idx
    poisoned = dict(z)
    poisoned[key] = np.array(z[key], np.float64)
    if poisoned[key].ndim == 1:
        poisoned[key][-1] = bad
    else:
        poisoned[key][-1, 7] = bad
    with pytest.raises(ValueError, match="non-finite fixture values"):
        _scales(poisoned)
    with pytest.raises(ValueError, match="non-finite fixture values"):
        failures(poisoned, idx=idx)                      # rejected before any column is run
    port = {v: np.array(z[f"wrf_{v}"][idx], np.float64) for v in FIELDS}
    port["precip"] = np.array(z["wrf_rainncv"][idx], np.float64)
    port["qs"][0, 7] += 1.0e-3                                          # a 1 g/kg snow error in one cell
    assert max(_excess(port, z, idx, _scales(z)).values()) > TOL["qs"]  # the clean gate sees the perturbation


def _column_state(d, idx):
    f = lambda v: jnp.asarray(d[f"in_{v}"][idx], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    return tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"),
                                  Nr=f("nr"), Ns=zero, Ng=zero, T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p, T, qv), dz=f("dz8w"), w=f("w"))


def _exit_run(state, dt, exit_on):
    """Full column, flag ON; only the SOURCE early exits toggle (the sedimentation/warm-rain exits stay off: on CPU
    interpret they move qg/Ng by 1 ulp with the flag on AND off, a pre-existing XLA:CPU fusion effect)."""
    env = dict(C24, GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1" if exit_on else "0",
               GPUWRF_THOMPSON_FULL_COLUMN_WARM_EXIT="0", GPUWRF_THOMPSON_FULL_COLUMN_SED_EXIT="0")
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    jax.clear_caches()
    try:
        out, ppt = full.full_column(state, dt, interpret=True)
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        jax.clear_caches()
    values = [np.asarray(getattr(out, k)) for k in tc.ThompsonColumnState.__slots__] + [np.asarray(ppt[k]) for k in sorted(ppt)]
    assert all(np.isfinite(value).all() for value in values), "non-finite exit-gate output"
    return values


TRIGGERS = ("cold_cloud", "cold_rain", "ice", "snow", "graupel", "nucleation")


def _trigger_columns(z):
    """One column per activity term of _wrf_mixed_phase_activity at 255 K, everything else ice-free and ice-subsaturated."""
    i = int(np.flatnonzero(z["dt"] == 18.0)[0])
    nz = z["in_qv"].shape[1]
    d = {k: np.array(v[[i] * 6]) for k, v in z.items() if k.startswith("in_")}
    T = np.full((6, nz), 255.0, np.float32)
    d["in_th"] = (T / d["in_pii"]).astype(np.float32)
    qvi = np.asarray(tc.saturation_mixing_ratio_ice(jnp.asarray(d["in_p"]), jnp.asarray(T)))
    d["in_qv"] = (0.9 * qvi).astype(np.float32)
    for v in ("qc", "qr", "qi", "qs", "qg", "ni", "nr"):
        d[f"in_{v}"] = np.zeros((6, nz), np.float32)
    k = slice(10, 14)
    d["in_qc"][0, k] = 2e-4
    d["in_qr"][1, k], d["in_nr"][1, k] = 2e-4, 1e3
    d["in_qi"][2, k], d["in_ni"][2, k] = 1e-5, 1e4
    d["in_qs"][3, k] = 1e-4
    d["in_qg"][4, k] = 1e-4
    d["in_qv"][5, k] = (1.4 * qvi[5, k]).astype(np.float32)          # ssati 0.4: nucleation
    return d


def test_source_early_exits_are_bitwise(z):
    """The EARLY_EXIT kernel branches of the single pass (cold block / warm rates skipped where
    _wrf_mixed_phase_activity proves them zero) are bitwise equal to the full pass: real mixed-phase columns (full
    branch), their warm-only and dry variants, and one column per activity term."""
    idx = np.flatnonzero(z["dt"] == 18.0)[:12]
    warm = {k: np.array(v) for k, v in z.items()}
    for v in ("qi", "qs", "qg", "ni"):
        warm[f"in_{v}"] = np.zeros_like(z[f"in_{v}"])
    warm["in_th"] = np.maximum(z["in_th"], np.float32(276.0) / z["in_pii"]).astype(np.float32)
    qvs = np.asarray(tc.saturation_mixing_ratio_liquid(jnp.asarray(z["in_p"]), jnp.asarray(warm["in_th"] * z["in_pii"])))
    warm["in_qv"] = np.minimum(z["in_qv"], 0.95 * qvs).astype(np.float32)
    dry = {k: np.array(v) for k, v in warm.items()}
    for v in ("qc", "qr", "nr"):
        dry[f"in_{v}"] = np.zeros_like(z[f"in_{v}"])
    for name, d, sel in (("real", z, idx), ("warm_only", warm, idx), ("dry", dry, idx),
                         ("triggers", _trigger_columns(z), np.arange(6))):
        state = _column_state(d, sel)
        on, off = _exit_run(state, 18.0, True), _exit_run(state, 18.0, False)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(on, off)), name


ACTIVITY_MUTANTS = {
    "cold_cloud": ("ice | (cold & (l_qc | l_qr)) | nuc\n", "ice | (cold & l_qr) | nuc\n"),
    "cold_rain": ("ice | (cold & (l_qc | l_qr)) | nuc\n", "ice | (cold & l_qc) | nuc\n"),
    "ice": ("ice = (state.qi > R1) | (state.qs > R1) | (state.qg > R1)", "ice = (state.qs > R1) | (state.qg > R1)"),
    "snow": ("ice = (state.qi > R1) | (state.qs > R1) | (state.qg > R1)", "ice = (state.qi > R1) | (state.qg > R1)"),
    "graupel": ("ice = (state.qi > R1) | (state.qs > R1) | (state.qg > R1)", "ice = (state.qi > R1) | (state.qs > R1)"),
    "nucleation": ("ice | (cold & (l_qc | l_qr)) | nuc\n", "ice | (cold & (l_qc | l_qr))\n"),
}


@pytest.fixture(scope="module")
def trigger_reference(z):
    state = _column_state(_trigger_columns(z), np.arange(6))
    return state, _exit_run(state, 18.0, False)


@pytest.mark.parametrize("term", TRIGGERS)
def test_each_activity_term_is_load_bearing(trigger_reference, term, monkeypatch):
    """Dropping one term of the activity predicate sends its trigger column through a zero branch: that column (and
    only that one) must leave the bitwise identity with the full pass."""
    state, ref = trigger_reference
    fixed, removed = ACTIVITY_MUTANTS[term]
    src = textwrap.dedent(inspect.getsource(tc._wrf_mixed_phase_activity))
    assert src.count(fixed) == 1, term
    namespace = dict(vars(tc))
    exec(compile(src.replace(fixed, removed), tc.__file__, "exec"), namespace)
    monkeypatch.setattr(tc, "_wrf_mixed_phase_activity", namespace["_wrf_mixed_phase_activity"])
    out = _exit_run(state, 18.0, True)
    changed = [c for c in range(6) if any(np.asarray(a)[c].tobytes() != np.asarray(b)[c].tobytes()
                                          for a, b in zip(out[:len(tc.ThompsonColumnState.__slots__)], ref))]
    assert changed == [TRIGGERS.index(term)], (term, [TRIGGERS[c] for c in changed])


def test_flag_requires_native_real_and_explicit_sedimentation(monkeypatch):
    monkeypatch.setenv("GPUWRF_THOMPSON_MIXED_PHASE_WRF", "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "0")
    state = tc.ThompsonColumnState(*(jnp.ones((1, 4), jnp.float32),) * 8, T=jnp.full((1, 4), 270.0, jnp.float32),
                                   p=jnp.full((1, 4), 8.0e4, jnp.float32), rho=jnp.ones((1, 4), jnp.float32))
    with pytest.raises(ValueError, match="NATIVE_REAL"):
        tc._wrf_mixed_phase_sources(state, 18.0, tc.THOMPSON_TABLES, None)
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_SED_PREP_FUSED", "1")
    with pytest.raises(ValueError, match="explicit sedimentation"):
        tc._wrf_mixed_phase_sources(state, 18.0, tc.THOMPSON_TABLES, None)


def _melting_cell(nz=6, temp=286.0, qr=1.0e-2, qs=5.0e-6):
    """Synthetic-but-real-structured melting layer: saturated (satw 0.999, so twet = T), heavy rain, thin snow."""
    T, p = jnp.full((1, nz), temp, jnp.float32), jnp.full((1, nz), 9.0e4, jnp.float32)
    qv = (0.999 * tc.saturation_mixing_ratio_liquid(p, T)).astype(jnp.float32)
    f = lambda v: jnp.full((1, nz), v, jnp.float32)  # noqa: E731
    return tc.ThompsonColumnState(qv=qv, qc=f(0.0), qr=f(qr), qi=f(0.0), qs=f(qs), qg=f(0.0), Ni=f(0.0), Nr=f(3.0e3),
                                  Ns=f(0.0), Ng=f(0.0), T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                                  dz=f(200.0), w=f(0.0))


@pytest.mark.parametrize("mutant", (None, "no_rain_snow_rebalance"))
def test_rain_snow_rebalance_conserves_water_when_the_snow_limiter_binds(mutant):
    """WRF :2943-2951: the rain and snow limiters scale prr_rcs and prs_rcs separately; above 0 C (twet) WRF then
    re-balances them so the melt-collision exchange conserves water. In this melting cell the snow limiter binds
    (all snow goes in one 18 s step); the 384 real columns never bind it, so the element gets its own check."""
    state = _melting_cell()
    with _env(mutant):
        out, _melt, _boost = tc._wrf_mixed_phase_sources(tc._real_state(state), 18.0, tc.THOMPSON_TABLES,
                                                         tc._mixed_phase_cold_tables())
    total = lambda s: sum(np.asarray(getattr(s, k), np.float64) for k in ("qv", "qc", "qr", "qi", "qs", "qg"))  # noqa: E731
    assert np.allclose(np.asarray(out.qs), 0.0, atol=1e-12)            # the snow limiter bound
    change = float(np.abs(total(out) - total(state)).max())
    bound = 2.0 * float(np.spacing(np.float32(1.0e-2)))                   # 2 ulp of the rain mixing ratio
    assert (change <= bound) == (mutant is None), change


@pytest.mark.parametrize("mutant", (None, "cloud_freezing_above_0c"))
def test_droplet_freezing_is_gated_to_the_cold_block(z, mutant, monkeypatch):
    """Droplet freezing is a cold-block process (:2554, :2606-2616). A warm cell reads the -1 C qcfz plane, whose frozen
    mass (<= 1.2e-16 kg m-3 per step) is below R1, so on the real tables the gate never shows in a column output; a
    freezing producer that freezes half of every droplet population exposes it: no cell above 0 C may gain ice."""
    real, calls = tc._cloud_water_freezing_rates, []

    def eager(state, dt, cold_tables):
        calls.append(dt)
        cf, cn, _pri, _pni = real(state, dt, cold_tables)
        cloudy = state.qc > tc.R1
        return cf, cn, jnp.where(cloudy, 0.5 * state.qc * state.rho / dt, 0.0), jnp.where(cloudy, 1.0e6 / dt, 0.0)

    monkeypatch.setattr(tc, "_cloud_water_freezing_rates", eager)
    i = int(np.argmax(z["in_qc"].max(-1)))
    f = lambda name: jnp.asarray(z[f"in_{name}"][[i]], jnp.float32)  # noqa: E731
    p, qv, zero = f("p"), f("qv"), jnp.zeros_like(f("qv"))
    T = jnp.full_like(p, 281.0)
    state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=zero, qs=zero, qg=zero, Ni=zero, Nr=f("nr"),
                                   Ns=zero, Ng=zero, T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                                   dz=f("dz8w"), w=f("w"))
    assert int((np.asarray(state.qc) > 1e-5).sum()) >= 3
    with _env(mutant):
        out, _melt, _boost = tc._wrf_mixed_phase_sources(tc._real_state(state), float(z["dt"][i]), tc.THOMPSON_TABLES,
                                                         tc._mixed_phase_cold_tables())
    assert calls, "the freezing producer was not called"
    created = float(np.abs(np.asarray(out.Ni)).max()) + float(np.abs(np.asarray(out.qi)).max())
    assert np.isfinite(created)
    assert (created == 0.0) == (mutant is None), created


@pytest.mark.parametrize("exit_on", (False, True))
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("field", (*tc.ThompsonColumnState.__slots__, "rain", "snow", "graupel", "ice", "cloudw"))
def test_exit_gate_rejects_each_nonfinite_output(z, monkeypatch, field, bad, exit_on):
    """Each State/precip output must be finite before any bitwise comparison, in either exit context."""
    _validate_fixture(z)
    state = _column_state(z, np.array([0]))
    out = state
    ppt = {name: np.zeros((1,), np.float64) for name in ("rain", "snow", "graupel", "ice", "cloudw")}
    if field in tc.ThompsonColumnState.__slots__:
        values = np.asarray(getattr(state, field)).copy()
        values.flat[0] = bad
        out = state.replace(**{field: values})
    else:
        ppt[field][0] = bad
    monkeypatch.setattr(full, "full_column", lambda *args, **kwargs: (out, ppt))
    with pytest.raises(AssertionError, match="non-finite exit-gate output"):
        _exit_run(state, 18.0, exit_on)
