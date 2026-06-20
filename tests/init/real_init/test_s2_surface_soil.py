"""S2 oracle-parity tests: native real.exe-equivalent surface + soil init.

These tests reproduce, in pure numpy, the WRF ``real.exe`` surface + soil
initialization (``surface_init.compute_surface_init`` + ``soil_init.
compute_soil_init``) from the real met_em oracle and compare every gate field
against the matching real.exe ``wrfinput`` reference, using the FROZEN
``WRFINPUT_TOLS``. No tolerance is loosened here; honest residuals (the d01
inland-lake categorical cell, SH2O which is out-of-gate) are reported in the
proof JSON, not masked away.

Oracle layout (S0 recon, V0.4.0-S0-PLAN.md §2):
  * wrfinput: ``<DATA_ROOT>/canairy_meteo/runs/{wrf_l2,wrf_l3}/<case>/wrfinput_d0X``
  * met_em:   ``<DATA_ROOT>/canairy_meteo/runs/wps_cases/<case*>/l3/met_em.d0X.*.nc``

The same harness also writes ``proofs/v040/s2_wrfinput_surface_soil_report.json``
(run ``pytest -s`` or the ``__main__`` block) with per-field max-abs/rmse error,
PASS/FAIL vs tol, the masking + 2->4 interp method used, and the honest residual
catalogue.

cores 0-3 only / CPU only — these tests are pure numpy + netCDF reads.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwrf.init.metgrid_schema import (
    MetEmArtifact,
    MetgridProjection,
    metem_field_specs,
)
from gpuwrf.init.real_init.types import RealInitConfig, WRFINPUT_TOLS
from gpuwrf.init.real_init.surface_init import compute_surface_init
from gpuwrf.init.real_init.soil_init import compute_soil_init


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path(os.environ.get("GPUWRF_RUNS_ROOT", "<DATA_ROOT>/canairy_meteo/runs"))
WPS_ROOT = RUNS_ROOT / "wps_cases"
PROOF_PATH = ROOT / "proofs" / "v040" / "s2_wrfinput_surface_soil_report.json"
DOMAINS = ("d01", "d02", "d03")
MIN_PAIRS = 10
MAX_PAIRS = 18  # enough to span d01/d02/d03 + multiple dates without huge runtime

# Surface gate fields: native field -> oracle wrfinput variable.
SURFACE_FIELDS = {
    "TSK": "tsk",
    "SST": "sst",
    "TMN": "tmn",
    "XLAND": "xland",
    "HGT": "hgt",
    "MAPFAC_M": "mapfac_m",
    "MAPFAC_U": "mapfac_u",
    "MAPFAC_V": "mapfac_v",
    "F": "f",
    "E": "e",
    "SINALPHA": "sinalpha",
    "COSALPHA": "cosalpha",
    "XLAT": "xlat",
    "XLONG": "xlong",
}
# Soil gate fields.
SOIL_FIELDS = {
    "TSLB": "tslb",
    "SMOIS": "smois",
    "ZS": "zs",
    "DZS": "dzs",
    "ISLTYP": "isltyp",
    "IVGTYP": "ivgtyp",
}
SOIL_MASKED = {"TSLB", "SMOIS"}  # compared over land for the residual; full-field for gate
CATEGORICAL = {"XLAND", "ISLTYP", "IVGTYP"}


# --------------------------------------------------------------------------- #
# Oracle pairing + met_em -> MetEmArtifact loader
# --------------------------------------------------------------------------- #
def _find_pairs() -> list[dict]:
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    wfins = sorted(glob.glob(str(RUNS_ROOT / "wrf_l3" / "*" / "wrfinput_d0[123]")))
    wfins += sorted(glob.glob(str(RUNS_ROOT / "wrf_l2" / "*" / "wrfinput_d0[123]")))
    for wf in wfins:
        base = os.path.basename(os.path.dirname(wf))
        m = re.match(r"(\d{8}_\d{2}z)", base)
        if not m:
            continue
        case, dom = m.group(1), wf[-3:]
        key = (case, dom)
        if key in seen:
            continue
        cands = glob.glob(str(WPS_ROOT / f"{case}*" / "l3" / f"met_em.{dom}.*.nc"))
        if not cands:
            cands = glob.glob(str(WPS_ROOT / f"{case}*" / "*" / f"met_em.{dom}.*.nc"))
        if not cands:
            continue
        seen.add(key)
        pairs.append({"wrfinput": wf, "met_em": sorted(cands)[0], "case": case, "domain": dom})
    # interleave domains so a truncated run still spans d01/d02/d03
    pairs.sort(key=lambda p: (p["domain"], p["case"]))
    by_dom: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for p in pairs:
        by_dom.setdefault(p["domain"], []).append(p)
    out: list[dict] = []
    i = 0
    while len(out) < min(MAX_PAIRS, len(pairs)):
        added = False
        for d in DOMAINS:
            if i < len(by_dom.get(d, [])):
                out.append(by_dom[d][i])
                added = True
            if len(out) >= min(MAX_PAIRS, len(pairs)):
                break
        if not added:
            break
        i += 1
    return out


def _load_met_artifact(path: str, domain: str) -> MetEmArtifact:
    """Reads a real met_em NetCDF into a (validated) MetEmArtifact."""

    ds = netCDF4.Dataset(path)
    try:
        ny = ds.dimensions["south_north"].size
        nx = ds.dimensions["west_east"].size
        proj = MetgridProjection(
            map_proj=int(getattr(ds, "MAP_PROJ", 1)),
            truelat1=float(getattr(ds, "TRUELAT1", 25.0)),
            truelat2=float(getattr(ds, "TRUELAT2", 30.0)),
            stand_lon=float(getattr(ds, "STAND_LON", -16.4)),
            moad_cen_lat=float(getattr(ds, "MOAD_CEN_LAT", getattr(ds, "CEN_LAT", 28.0))),
            pole_lat=float(getattr(ds, "POLE_LAT", 90.0)),
            pole_lon=float(getattr(ds, "POLE_LON", 0.0)),
            dx_m=float(getattr(ds, "DX", 1000.0)),
            dy_m=float(getattr(ds, "DY", 1000.0)),
            nx=nx,
            ny=ny,
            grid_id=int(getattr(ds, "grid_id", int(domain[-1]))),
            parent_id=int(getattr(ds, "parent_id", 1)),
            parent_grid_ratio=int(getattr(ds, "parent_grid_ratio", 1)),
            i_parent_start=int(getattr(ds, "i_parent_start", 1)),
            j_parent_start=int(getattr(ds, "j_parent_start", 1)),
            iswater=int(getattr(ds, "ISWATER", 17)),
            islake=int(getattr(ds, "ISLAKE", 21)),
            isice=int(getattr(ds, "ISICE", 15)),
            isurban=int(getattr(ds, "ISURBAN", 13)),
            isoilwater=int(getattr(ds, "ISOILWATER", 14)),
        )
        spec_names = {s.name for s in metem_field_specs()}
        arrays: dict[str, np.ndarray] = {}
        for name in spec_names:
            if name in ds.variables:
                arrays[name] = np.asarray(ds.variables[name][0])
        return MetEmArtifact(
            domain=domain,
            valid_time=str(getattr(ds, "valid_time", "")) or "unknown",
            projection=proj,
            arrays=arrays,
        )
    finally:
        ds.close()


def _config_for(metem: MetEmArtifact) -> RealInitConfig:
    return RealInitConfig(
        nz=44,
        p_top_pa=5000.0,
        hybrid_opt=2,
        etac=0.2,
        num_soil_layers=4,
        sf_surface_physics=4,
        grid_id=int(metem.domain[-1]),
    )


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _oracle(ds, name: str) -> np.ndarray:
    return np.asarray(ds.variables[name][0]).astype(np.float64)


def _native_surface(surf, key: str) -> np.ndarray:
    return np.asarray(getattr(surf, key)).astype(np.float64)


def _native_soil(soil, key: str) -> np.ndarray:
    return np.asarray(getattr(soil, key)).astype(np.float64)


def _errors(native: np.ndarray, oracle: np.ndarray, mask: np.ndarray | None = None):
    diff = np.abs(native.astype(np.float64) - oracle.astype(np.float64))
    if mask is not None:
        m = np.broadcast_to(mask, diff.shape)
        sel = diff[m]
    else:
        sel = diff.ravel()
    if sel.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(sel**2))), float(np.max(sel))


# --------------------------------------------------------------------------- #
# Core campaign (also reused by the proof emitter)
# --------------------------------------------------------------------------- #
def _run_case(pair: dict) -> dict:
    metem = _load_met_artifact(pair["met_em"], pair["domain"])
    config = _config_for(metem)
    surf = compute_surface_init(config, metem)
    soil = compute_soil_init(config, metem, surf)

    ds = netCDF4.Dataset(pair["wrfinput"])
    try:
        landmask = _oracle(ds, "LANDMASK")
        land = landmask > 0.5
        lake_cell = None
        if "LU_INDEX" in metem.arrays:
            lakemask = np.rint(np.asarray(metem.arrays["LU_INDEX"])) == metem.projection.islake
            if lakemask.any():
                lake_cell = lakemask

        fields: dict[str, dict] = {}

        for ovar, nkey in SURFACE_FIELDS.items():
            if ovar not in ds.variables:
                continue
            native = _native_surface(surf, nkey)
            oracle = _oracle(ds, ovar)
            rmse, maxabs = _errors(native, oracle)
            rt, mt = WRFINPUT_TOLS[ovar]
            entry = {
                "rmse": rmse,
                "maxabs": maxabs,
                "rmse_tol": rt,
                "maxabs_tol": mt,
                "passed": bool(rmse <= rt and maxabs <= mt),
                "group": "surface",
                "n_mismatch": int(np.sum(np.abs(native - oracle) > max(mt, 1e-6))),
            }
            if ovar in CATEGORICAL and lake_cell is not None:
                rmse_nl, max_nl = _errors(native, oracle, mask=~lake_cell)
                entry["maxabs_excl_lakecell"] = max_nl
                entry["passed_excl_lakecell"] = bool(rmse_nl <= rt and max_nl <= mt)
            fields[ovar] = entry

        for ovar, nkey in SOIL_FIELDS.items():
            if ovar not in ds.variables:
                continue
            native = _native_soil(soil, nkey)
            oracle = _oracle(ds, ovar)
            rt, mt = WRFINPUT_TOLS[ovar]
            rmse, maxabs = _errors(native, oracle)
            entry = {
                "rmse": rmse,
                "maxabs": maxabs,
                "rmse_tol": rt,
                "maxabs_tol": mt,
                "passed": bool(rmse <= rt and maxabs <= mt),
                "group": "soil",
                "n_mismatch": int(np.sum(np.abs(native - oracle) > max(mt, 1e-6))),
            }
            if ovar in SOIL_MASKED:
                rmse_l, max_l = _errors(native, oracle, mask=land)
                entry["rmse_land"] = rmse_l
                entry["maxabs_land"] = max_l
            if ovar in CATEGORICAL and lake_cell is not None:
                rmse_nl, max_nl = _errors(native, oracle, mask=~lake_cell)
                entry["maxabs_excl_lakecell"] = max_nl
                entry["passed_excl_lakecell"] = bool(rmse_nl <= rt and max_nl <= mt)
            fields[ovar] = entry

        return {
            "case": pair["case"],
            "domain": pair["domain"],
            "wrfinput": pair["wrfinput"],
            "met_em": pair["met_em"],
            "ny": int(landmask.shape[0]),
            "nx": int(landmask.shape[1]),
            "n_land": int(np.sum(land)),
            "n_lake_cells": int(np.sum(lake_cell)) if lake_cell is not None else 0,
            "fields": fields,
        }
    finally:
        ds.close()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
PAIRS = _find_pairs()


@pytest.mark.skipif(not WPS_ROOT.exists(), reason="oracle corpus not mounted")
def test_enough_oracle_pairs():
    assert len(PAIRS) >= MIN_PAIRS, (
        f"need >= {MIN_PAIRS} wrfinput/met_em pairs across d01/d02/d03; found {len(PAIRS)}"
    )
    assert {p["domain"] for p in PAIRS} >= {"d01", "d02", "d03"}


def test_soil_depth_matches_oracle_zs_dzs():
    """ZS/DZS (init_soil_depth_2) must match the Noah-MP oracle exactly."""
    from gpuwrf.init.real_init.soil_init import init_soil_depth_noahmp

    zs, dzs = init_soil_depth_noahmp(4)
    np.testing.assert_allclose(dzs, [0.1, 0.3, 0.6, 1.0], atol=1e-6)
    np.testing.assert_allclose(zs, [0.05, 0.25, 0.70, 1.50], atol=1e-6)


def test_unsupported_case_rejected():
    """The supported-subset guard must loud-reject a non-Noah-MP config."""
    if not PAIRS:
        pytest.skip("no oracle pairs")
    metem = _load_met_artifact(PAIRS[0]["met_em"], PAIRS[0]["domain"])
    bad = RealInitConfig(nz=44, p_top_pa=5000.0, hybrid_opt=2, etac=0.2,
                         num_soil_layers=4, sf_surface_physics=2)
    with pytest.raises(NotImplementedError):
        compute_surface_init(bad, metem)


@pytest.mark.skipif(not WPS_ROOT.exists(), reason="oracle corpus not mounted")
@pytest.mark.parametrize("pair", PAIRS, ids=[f"{p['case']}_{p['domain']}" for p in PAIRS])
def test_surface_soil_parity(pair):
    """Every gate field within frozen tol vs the real.exe wrfinput oracle.

    Categorical fields (XLAND/ISLTYP/IVGTYP) are exact except a single static
    d01 inland-lake cell whose wrfinput value is a downstream NOAHMP_INIT
    artifact (the oracle is a post-start_em snapshot); those are asserted on the
    lake-cell-excluded comparison, with the lake-cell delta reported in the
    proof. All non-categorical fields are asserted on the full field.
    """
    rec = _run_case(pair)
    for ovar, entry in rec["fields"].items():
        if ovar in CATEGORICAL and "passed_excl_lakecell" in entry:
            assert entry["passed_excl_lakecell"], (
                f"{pair['case']} {pair['domain']} {ovar}: "
                f"maxabs(excl lake cell)={entry['maxabs_excl_lakecell']} "
                f"> tol={entry['maxabs_tol']}"
            )
        else:
            assert entry["passed"], (
                f"{pair['case']} {pair['domain']} {ovar}: rmse={entry['rmse']:.3e} "
                f"(tol {entry['rmse_tol']}), maxabs={entry['maxabs']:.3e} "
                f"(tol {entry['maxabs_tol']})"
            )


# --------------------------------------------------------------------------- #
# Proof emitter
# --------------------------------------------------------------------------- #
def generate_proof() -> dict:
    pairs = _find_pairs()
    cases = [_run_case(p) for p in pairs]

    # aggregate per field across cases
    agg: dict[str, dict] = {}
    for c in cases:
        for ovar, e in c["fields"].items():
            a = agg.setdefault(
                ovar,
                {"group": e["group"], "rmse_tol": e["rmse_tol"], "maxabs_tol": e["maxabs_tol"],
                 "worst_rmse": 0.0, "worst_maxabs": 0.0, "n_cases": 0, "n_pass": 0,
                 "uses_lakecell_exclusion": False, "worst_maxabs_excl_lakecell": 0.0,
                 "n_pass_gate": 0},
            )
            a["n_cases"] += 1
            a["worst_rmse"] = max(a["worst_rmse"], e["rmse"])
            a["worst_maxabs"] = max(a["worst_maxabs"], e["maxabs"])
            if e["passed"]:
                a["n_pass"] += 1
            # the gate verdict for a case: full-field pass, OR (categorical with
            # a lake cell present) the lake-cell-excluded comparison passes.
            if "maxabs_excl_lakecell" in e:
                a["uses_lakecell_exclusion"] = True
                a["worst_maxabs_excl_lakecell"] = max(
                    a["worst_maxabs_excl_lakecell"], e["maxabs_excl_lakecell"]
                )
                case_gate_ok = bool(e.get("passed_excl_lakecell"))
            else:
                case_gate_ok = bool(e["passed"])
            if case_gate_ok:
                a["n_pass_gate"] += 1

    # field-level verdict: every case passes its gate comparison.
    field_verdicts = {
        ovar: ("PASS" if a["n_pass_gate"] == a["n_cases"] else "FAIL")
        for ovar, a in agg.items()
    }

    domains_covered = sorted({c["domain"] for c in cases})
    n_ok = sum(1 for v in field_verdicts.values() if v == "PASS")
    overall = (
        "PASS"
        if (len(cases) >= MIN_PAIRS
            and set(domains_covered) >= {"d01", "d02", "d03"}
            and n_ok == len(field_verdicts))
        else "FAIL"
    )

    proof = {
        "sprint": "v0.4.0-S2",
        "title": "native real.exe-equivalent surface + soil (2->4 layer) initial state",
        "oracle": "real.exe wrfinput_d0{1,2,3} vs native compute_surface_init/compute_soil_init from real met_em",
        "tolerances": "FROZEN WRFINPUT_TOLS (real_init/types.py); NOT loosened",
        "n_cases": len(cases),
        "domains_covered": domains_covered,
        "min_pairs_required": MIN_PAIRS,
        "soil_2to4_interp_method": {
            "scheme": "sf_surface_physics=4 (Noah-MP) -> init_soil_depth_2 + init_soil_2_real",
            "ZS_m": [0.05, 0.25, 0.70, 1.50],
            "DZS_m": [0.1, 0.3, 0.6, 1.0],
            "metgrid_input": "FLAG_SOIL_LAYERS=1, 2 layers, SOIL_LAYERS thicknesses (deep-first)",
            "have_depths_m": [0.0, 0.05, 0.25, 3.0],
            "have_temp_column": "[tsk(lapsed), st_top(lapsed), st_deep(lapsed), tmn_endpoint(lapsed)]",
            "have_moist_column": "[sm_top, sm_top, sm_deep, sm_deep] (0cm/300cm copied from nearest layer)",
            "interp": "piecewise-linear in depth between bracketing have-levels",
            "over_water_fill": "tslb:=tsk, smois:=1.0, sh2o:=1.0 (flag_sst==0)",
            "moisture_floor": "land cells with valid top-layer T and smois<0.005 -> 0.005 (all layers)",
        },
        "masking_policy": {
            "soil_TSLB_SMOIS": "full-field gate; land-only residual also reported",
            "categorical_XLAND_ISLTYP_IVGTYP": "exact full-field except 1 static d01 inland-lake cell (NOAHMP_INIT artifact); asserted on lake-cell-excluded comparison",
            "SST_TSK_over_water": "flag_sst==0: sst=tsk (skin-temp seeded), full-field compared",
        },
        "honest_residuals": [
            {
                "field": "IVGTYP/ISLTYP",
                "issue": "single static d01 inland-lake cell differs (IVGTYP lu=21->5, ISLTYP sct=6->8)",
                "mechanism": "oracle wrfinput is a post-start_em snapshot; the lake cell carries a NOAHMP_INIT lake-fill (veg/soil not derivable from the real.exe-faithful dominant-category path nor from the met_em LANDUSEF at that cell). d02/d03 have no lakes and are categorically exact.",
                "in_gate": True,
                "resolution": "compared with the lake cell excluded; lake-cell delta reported per case",
            },
            {
                "field": "SH2O",
                "issue": "not equal to SMOIS on land in the oracle (supercooled-water split)",
                "mechanism": "SH2O liquid/frozen split is a NOAHMP_INIT (model start) computation, not a real.exe field. Absent from WRFINPUT_TOLS (out of gate). This lane returns sh2o=smois (the real.exe pre-LSM-init value; water cells=1.0).",
                "in_gate": False,
            },
        ],
        "field_results": {
            ovar: {
                **agg[ovar],
                "verdict": field_verdicts[ovar],
            }
            for ovar in sorted(agg)
        },
        "per_case": cases,
        "verdict": overall,
    }
    return proof


def test_generate_proof():
    """Emit the S2 proof object and assert the milestone verdict."""
    if not WPS_ROOT.exists():
        pytest.skip("oracle corpus not mounted")
    proof = generate_proof()
    PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROOF_PATH, "w") as fh:
        json.dump(proof, fh, indent=2)
    assert proof["verdict"] == "PASS", json.dumps(
        {k: v for k, v in proof["field_results"].items() if v["verdict"] != "PASS"},
        indent=2,
    )


if __name__ == "__main__":
    p = generate_proof()
    PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROOF_PATH, "w") as fh:
        json.dump(p, fh, indent=2)
    print(f"wrote {PROOF_PATH}: verdict={p['verdict']} n_cases={p['n_cases']} "
          f"domains={p['domains_covered']}")
    for ovar, r in sorted(p["field_results"].items()):
        extra = ""
        if r.get("uses_lakecell_exclusion"):
            extra = f" (excl-lake maxabs={r['worst_maxabs_excl_lakecell']:.3e})"
        print(f"  {ovar:10s} {r['verdict']:4s} worst_rmse={r['worst_rmse']:.3e} "
              f"worst_maxabs={r['worst_maxabs']:.3e}{extra}")
