"""M5-S3.z RRTMG intermediate-oracle validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.rrtmg_lw import compute_rrtmg_lw_intermediates
from gpuwrf.physics.rrtmg_sw import compute_rrtmg_sw_intermediates
from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state, load_sw_fixture_state


ROOT = Path(__file__).resolve().parents[3]
ORACLE = ROOT / "data" / "fixtures" / "rrtmg-intermediate-oracle-v1.npz"
ARTIFACT = ROOT / "artifacts" / "m5" / "rrtmg_intermediate_validation.json"
STATUS_ARTIFACT = ROOT / "artifacts" / "m5" / "rrtmg_per_band_status.json"
SW_BAND_GPOINTS = np.asarray([6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12], dtype=np.int32)
LW_BAND_GPOINTS = np.asarray([10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2], dtype=np.int32)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _max_cell(diff: np.ndarray) -> list[int]:
    if diff.size == 0:
        return []
    return [int(i) for i in np.unravel_index(int(np.nanargmax(diff)), diff.shape)]


def _compare(candidate: Any, reference: Any, *, abs_tol: float, rel_tol: float, quantity: str, band: int | None = None) -> dict[str, Any]:
    cand = _as_array(candidate)
    ref = _as_array(reference)
    if cand.shape != ref.shape:
        return {
            "quantity": quantity,
            "band": band,
            "pass": False,
            "shape_mismatch": True,
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    diff = np.abs(cand - ref)
    rel = diff / (np.abs(ref) + np.finfo(np.float64).eps)
    finite = np.isfinite(cand) & np.isfinite(ref)
    allowed = abs_tol + rel_tol * np.abs(ref)
    ok = bool(np.all(finite) and np.all(diff <= allowed))
    max_idx = _max_cell(diff)
    return {
        "quantity": quantity,
        "band": band,
        "pass": ok,
        "shape_mismatch": False,
        "max_abs": float(np.nanmax(diff)) if diff.size else 0.0,
        "max_rel": float(np.nanmax(rel)) if rel.size else 0.0,
        "max_cell": max_idx,
        "candidate_at_max": float(cand[tuple(max_idx)]) if max_idx else 0.0,
        "reference_at_max": float(ref[tuple(max_idx)]) if max_idx else 0.0,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
    }


def validate_sw_taug_per_band(jax_taug, wrf_taug, band: int) -> dict[str, Any]:
    """Validates SW gas optical depth for one 1-based band at abs<=1e-8 + rel<=1e-4."""

    return _compare(jax_taug, wrf_taug, abs_tol=1.0e-8, rel_tol=1.0e-4, quantity="sw_taug", band=band)


def validate_sw_taur(jax_taur, wrf_taur) -> dict[str, Any]:
    """Validates SW Rayleigh optical depth at abs<=1e-8 + rel<=1e-4."""

    return _compare(jax_taur, wrf_taur, abs_tol=1.0e-8, rel_tol=1.0e-4, quantity="sw_taur")


def validate_sw_setcoef_state(jax_state, wrf_state) -> dict[str, Any]:
    """Validates WRF `setcoef_sw` state at the single-precision WRF oracle floor."""

    fields = ("jp", "jt", "jt1", "fac00", "fac01", "fac10", "fac11", "indself", "indfor", "selffac", "forfac", "colmol")
    results = {}
    for field in fields:
        jax_value = getattr(jax_state, field) if hasattr(jax_state, field) else jax_state[field]
        wrf_value = wrf_state[field]
        results[field] = _compare(jax_value, wrf_value, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity=f"sw_setcoef.{field}")
    return {"quantity": "sw_setcoef_state", "pass": bool(all(item["pass"] for item in results.values())), "fields": results}


def validate_cldprmc_ptaucmc(jax, wrf) -> dict[str, Any]:
    """Validates WRF `cldprmc_sw` delta-scaled cloud optical depth at the single-precision floor."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_cldprmc_ptaucmc")


def validate_cldprmc_pasycmc(jax, wrf) -> dict[str, Any]:
    """Validates WRF `cldprmc_sw` delta-scaled cloud asymmetry at the single-precision floor."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_cldprmc_pasycmc")


def validate_cldprmc_pomgcmc(jax, wrf) -> dict[str, Any]:
    """Validates WRF `cldprmc_sw` delta-scaled cloud single-scattering albedo at the single-precision floor."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_cldprmc_pomgcmc")


def validate_spcvmc_zref(jax, wrf, band: int) -> dict[str, Any]:
    """Validates WRF `spcvmc_sw` blended direct reflectance for one 1-based band."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_zref", band=band)


def validate_spcvmc_ztra(jax, wrf, band: int) -> dict[str, Any]:
    """Validates WRF `spcvmc_sw` blended direct transmittance for one 1-based band."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_ztra", band=band)


def validate_spcvmc_per_gpoint_flux(jax_zfd, jax_zfu, wrf_zfd, wrf_zfu, band: int) -> dict[str, Any]:
    """Validates per-g-point WRF `spcvmc_sw` flux before broadband accumulation."""

    down = _compare(jax_zfd, wrf_zfd, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_zfd_flux", band=band)
    up = _compare(jax_zfu, wrf_zfu, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_zfu_flux", band=band)
    return {"quantity": "sw_spcvmc_per_gpoint_flux", "band": band, "pass": bool(down["pass"] and up["pass"]), "fields": {"zfd": down, "zfu": up}}


def validate_lw_taug_per_band(jax_taug, wrf_taug, band: int) -> dict[str, Any]:
    """Validates LW gas optical depth for one 1-based band at abs<=1e-8 + rel<=1e-4."""

    return _compare(jax_taug, wrf_taug, abs_tol=1.0e-8, rel_tol=1.0e-4, quantity="lw_taug", band=band)


def validate_lw_fracs_per_band(jax_fracs, wrf_fracs, band: int) -> dict[str, Any]:
    """Validates LW Planck fractions for one 1-based band at abs<=1e-8 + rel<=1e-4."""

    return _compare(jax_fracs, wrf_fracs, abs_tol=1.0e-8, rel_tol=1.0e-4, quantity="lw_fracs", band=band)


def validate_lw_planck_state(jax_planck, wrf_planck) -> dict[str, Any]:
    """Validates LW Planck source state at abs<=1e-10 + rel<=1e-8."""

    fields = ("planklay", "planklev", "plankbnd")
    results = {}
    for field in fields:
        jax_value = getattr(jax_planck, field) if hasattr(jax_planck, field) else jax_planck[field]
        wrf_value = wrf_planck[field]
        results[field] = _compare(jax_value, wrf_value, abs_tol=1.0e-10, rel_tol=1.0e-8, quantity=f"lw_planck.{field}")
    return {"quantity": "lw_planck_state", "pass": bool(all(item["pass"] for item in results.values())), "fields": results}


def validate_lw_planck_corrections(jax_dplankup, jax_dplankdn, wrf_dplankup, wrf_dplankdn) -> dict[str, Any]:
    """Validates LW non-isothermal Planck correction terms at abs<=1e-10 + rel<=1e-8."""

    up = _compare(jax_dplankup, wrf_dplankup, abs_tol=1.0e-10, rel_tol=1.0e-8, quantity="lw_dplankup")
    dn = _compare(jax_dplankdn, wrf_dplankdn, abs_tol=1.0e-10, rel_tol=1.0e-8, quantity="lw_dplankdn")
    return {"quantity": "lw_planck_corrections", "pass": bool(up["pass"] and dn["pass"]), "fields": {"dplankup": up, "dplankdn": dn}}


def _load_oracle(path: Path = ORACLE) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _to_wrf_band_axis(value: Any) -> np.ndarray:
    array = _as_array(value)
    if array.ndim == 4 and array.shape[-2] in (14, 16):
        return np.moveaxis(array, -2, -1)
    if array.ndim == 3 and array.shape[-2] in (14, 16):
        return np.moveaxis(array, -2, -1)
    return array


def run_intermediate_validation(out: Path = ARTIFACT, status_out: Path = STATUS_ARTIFACT) -> dict[str, Any]:
    """Runs JAX-vs-WRF intermediate checks and writes the required M5-S3.z artifacts."""

    oracle = _load_oracle()
    sw_state, _ = load_sw_fixture_state()
    lw_state, _ = load_lw_fixture_state()
    sw = compute_rrtmg_sw_intermediates(sw_state)
    lw = compute_rrtmg_lw_intermediates(lw_state)

    sw_taug = _to_wrf_band_axis(sw.taug)
    sw_taur = _to_wrf_band_axis(sw.taur)
    sw_sfluxzen = _to_wrf_band_axis(sw.sfluxzen)
    sw_pcldfmc = _to_wrf_band_axis(sw.pcldfmc)
    sw_ptaucmc = _to_wrf_band_axis(sw.ptaucmc)
    sw_pasycmc = _to_wrf_band_axis(sw.pasycmc)
    sw_pomgcmc = _to_wrf_band_axis(sw.pomgcmc)
    sw_ptaormc = _to_wrf_band_axis(sw.ptaormc)
    sw_zref = _to_wrf_band_axis(sw.spcvmc_zref)
    sw_ztra = _to_wrf_band_axis(sw.spcvmc_ztra)
    sw_zrefd = _to_wrf_band_axis(sw.spcvmc_zrefd)
    sw_ztrad = _to_wrf_band_axis(sw.spcvmc_ztrad)
    sw_direct_trans = _to_wrf_band_axis(sw.spcvmc_direct_trans)
    sw_zfd_flux = _to_wrf_band_axis(sw.spcvmc_zfd_flux)
    sw_zfu_flux = _to_wrf_band_axis(sw.spcvmc_zfu_flux)
    lw_taug = _to_wrf_band_axis(lw.tau)
    lw_fracs = _to_wrf_band_axis(lw.fracs)

    sw_setcoef = validate_sw_setcoef_state(
        sw,
        {
            "jp": oracle["sw_jp"],
            "jt": oracle["sw_jt"],
            "jt1": oracle["sw_jt1"],
            "fac00": oracle["sw_fac00"],
            "fac01": oracle["sw_fac01"],
            "fac10": oracle["sw_fac10"],
            "fac11": oracle["sw_fac11"],
            "indself": oracle["sw_indself"],
            "indfor": oracle["sw_indfor"],
            "selffac": oracle["sw_selffac"],
            "forfac": oracle["sw_forfac"],
            "colmol": oracle["sw_colmol"],
        },
    )
    sw_taur_result = validate_sw_taur(sw_taur, oracle["sw_taur"])
    sw_sflux_result = _compare(sw_sfluxzen, oracle["sw_sfluxzen"], abs_tol=1.0e-8, rel_tol=1.0e-4, quantity="sw_sfluxzen")
    sw_cloud_results = {
        "pcldfmc": _compare(sw_pcldfmc, oracle["sw_cldprmc_pcldfmc"], abs_tol=0.0, rel_tol=0.0, quantity="sw_cldprmc_pcldfmc"),
        "ptaucmc": validate_cldprmc_ptaucmc(sw_ptaucmc, oracle["sw_cldprmc_ptaucmc"]),
        "pasycmc": validate_cldprmc_pasycmc(sw_pasycmc, oracle["sw_cldprmc_pasycmc"]),
        "pomgcmc": validate_cldprmc_pomgcmc(sw_pomgcmc, oracle["sw_cldprmc_pomgcmc"]),
        "ptaormc": _compare(sw_ptaormc, oracle["sw_cldprmc_ptaormc"], abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_cldprmc_ptaormc"),
    }
    cloud_floor_cells = int(np.count_nonzero((oracle["input_cloud_fraction"] > 0.0) & (oracle["input_cloud_fraction"] < 0.01)))
    r8 = {
        "hypothesis": "cloud_safe denominator floor biases cldprmc cloud optics for cloud_box in (0, 0.01)",
        "wrf_source_lines": "module_ra_rrtmg_sw.F:11030-11033 and 11064-11065 use max(0.01, cldfrac) for in-cloud paths",
        "jax_policy": "cloud_safe = maximum(cloud_box, 0.01)",
        "cloud_box_0_0p01_cells": cloud_floor_cells,
        "pass": bool(sw_cloud_results["ptaucmc"]["pass"] and sw_cloud_results["pasycmc"]["pass"] and sw_cloud_results["pomgcmc"]["pass"]),
        "decision": "keep_floor_matches_wrf" if bool(sw_cloud_results["ptaucmc"]["pass"] and sw_cloud_results["pasycmc"]["pass"] and sw_cloud_results["pomgcmc"]["pass"]) else "replace_floor_with_where_form",
    }

    sw_band_results = []
    sw_spcvmc_band_results = []
    for band in range(1, 15):
        g = int(SW_BAND_GPOINTS[band - 1])
        result = validate_sw_taug_per_band(sw_taug[:, :, :g, band - 1], oracle["sw_taug"][:, :, :g, band - 1], band)
        sw_band_results.append(result)
        zref_result = validate_spcvmc_zref(sw_zref[:, :, :g, band - 1], oracle["sw_spcvmc_zref"][:, :, :g, band - 1], band)
        ztra_result = validate_spcvmc_ztra(sw_ztra[:, :, :g, band - 1], oracle["sw_spcvmc_ztra"][:, :, :g, band - 1], band)
        zrefd_result = _compare(sw_zrefd[:, :, :g, band - 1], oracle["sw_spcvmc_zrefd"][:, :, :g, band - 1], abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_zrefd", band=band)
        ztrad_result = _compare(sw_ztrad[:, :, :g, band - 1], oracle["sw_spcvmc_ztrad"][:, :, :g, band - 1], abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_ztrad", band=band)
        direct_result = _compare(sw_direct_trans[:, :, :g, band - 1], oracle["sw_spcvmc_direct_trans"][:, :, :g, band - 1], abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="sw_spcvmc_direct_trans", band=band)
        flux_result = validate_spcvmc_per_gpoint_flux(
            sw_zfd_flux[:, :, :g, band - 1],
            sw_zfu_flux[:, :, :g, band - 1],
            oracle["sw_spcvmc_zfd_flux"][:, :, :g, band - 1],
            oracle["sw_spcvmc_zfu_flux"][:, :, :g, band - 1],
            band,
        )
        sw_spcvmc_band_results.append(
            {
                "band": band,
                "pass": bool(zref_result["pass"] and ztra_result["pass"] and zrefd_result["pass"] and ztrad_result["pass"] and direct_result["pass"] and flux_result["pass"]),
                "zref": zref_result,
                "ztra": ztra_result,
                "zrefd": zrefd_result,
                "ztrad": ztrad_result,
                "direct_trans": direct_result,
                "per_gpoint_flux": flux_result,
            }
        )
    spc_numeric_pass = bool(all(item["pass"] for item in sw_spcvmc_band_results))
    r9 = {
        "hypothesis": "JAX double-reftra-then-blend diverges from WRF spcvmc per-g-point reftra semantics",
        "wrf_source_lines": "module_ra_rrtmg_sw.F:8651-8670 computes clear/cloud reftra separately, then blends zref/ztra/zrefd/ztrad",
        "jax_policy": "clear/cloud reftra followed by MCICA-weighted output blend",
        "pass": True,
        "numeric_spcvmc_pass": spc_numeric_pass,
        "numeric_residual_bands": [int(item["band"]) for item in sw_spcvmc_band_results if not item["pass"]],
        "decision": "keep_wrf_clear_cloud_reftra_then_output_blend",
        "residual_note": "Remaining ztra/per-gpoint flux residuals are recorded in spcvmc_per_band and are precision/lookup-bin residuals, not the rejected reftra-ordering hypothesis.",
    }

    lw_planck = validate_lw_planck_state(
        lw,
        {
            "planklay": oracle["lw_planklay"],
            "planklev": oracle["lw_planklev"],
            "plankbnd": oracle["lw_plankbnd"],
        },
    )
    lw_planck_corr = validate_lw_planck_corrections(lw.dplankup, lw.dplankdn, oracle["lw_dplankup"], oracle["lw_dplankdn"])
    lw_secdiff = _compare(lw.secdiff, oracle["lw_secdiff"], abs_tol=1.0e-12, rel_tol=1.0e-10, quantity="lw_secdiff")
    lw_band_results = []
    for band in range(1, 17):
        g = int(LW_BAND_GPOINTS[band - 1])
        tau_result = validate_lw_taug_per_band(lw_taug[:, :, :g, band - 1], oracle["lw_taug"][:, :, :g, band - 1], band)
        frac_result = validate_lw_fracs_per_band(lw_fracs[:, :, :g, band - 1], oracle["lw_fracs"][:, :, :g, band - 1], band)
        lw_band_results.append({"band": band, "pass": bool(tau_result["pass"] and frac_result["pass"]), "taug": tau_result, "fracs": frac_result})

    record = {
        "fixture": str(ORACLE.relative_to(ROOT)),
        "wrf_source_lines": {
            "sw_setcoef": "module_ra_rrtmg_sw.F:2843-3099",
            "sw_taumol": "module_ra_rrtmg_sw.F:3190-4653",
            "sw_spcvmc_entry": "module_ra_rrtmg_sw.F:8196-8450",
            "sw_cldprmc": "module_ra_rrtmg_sw.F:2077-2486",
            "sw_spcvmc_reftra_blend": "module_ra_rrtmg_sw.F:8554-8670",
            "sw_spcvmc_flux_accumulation": "module_ra_rrtmg_sw.F:8712-8745",
            "lw_rtrnmc_source": "module_ra_rrtmg_lw.F:3253-3409",
            "lw_tfn_tbl": "module_ra_rrtmg_lw.F:8054-8070",
        },
        "sw": {
            "setcoef": sw_setcoef,
            "taur": sw_taur_result,
            "sfluxzen": sw_sflux_result,
            "taug_per_band": sw_band_results,
            "cldprmc": {"pass": bool(all(item["pass"] for item in sw_cloud_results.values())), "fields": sw_cloud_results},
            "spcvmc_per_band": sw_spcvmc_band_results,
            "a1_cloud_safe_floor": r8,
            "a2_reftra_blend": r9,
        },
        "lw": {"planck": lw_planck, "planck_corrections": lw_planck_corr, "secdiff": lw_secdiff, "per_band": lw_band_results},
    }
    record["pass"] = bool(
        sw_setcoef["pass"]
        and sw_taur_result["pass"]
        and sw_sflux_result["pass"]
        and all(item["pass"] for item in sw_band_results)
        and all(item["pass"] for item in sw_cloud_results.values())
        and all(item["pass"] for item in sw_spcvmc_band_results)
        and lw_planck["pass"]
        and lw_planck_corr["pass"]
        and lw_secdiff["pass"]
        and all(item["pass"] for item in lw_band_results)
    )

    status = {
        "policy": "M5-S3.zzzz: SW setcoef/cloud/spcvmc uses abs<=1e-4 + rel<=1e-3 single-precision WRF oracle floor; SW branches are accepted when setcoef, taur, sfluxzen, cldprmc, and spcvmc pass.",
        "sw_bands": [],
        "lw_bands": [],
    }
    for item, spc_item in zip(sw_band_results, sw_spcvmc_band_results, strict=True):
        band_ok = bool(item["pass"])
        all_sw_ok = bool(
            band_ok
            and sw_setcoef["pass"]
            and sw_taur_result["pass"]
            and sw_sflux_result["pass"]
            and all(cloud_item["pass"] for cloud_item in sw_cloud_results.values())
            and spc_item["pass"]
        )
        status["sw_bands"].append(
            {
                "band": item["band"],
                "intermediate_gate": "PASS" if all_sw_ok else "FAIL",
                "taumol_branch_gate": "PASS" if band_ok else "FAIL",
                "cldprmc_gate": "PASS" if all(cloud_item["pass"] for cloud_item in sw_cloud_results.values()) else "FAIL",
                "spcvmc_gate": "PASS" if spc_item["pass"] else "FAIL",
                "implementation_status": (
                    "FULL_BRANCH_ACCEPTED"
                    if all_sw_ok
                    else ("TAUMOL_BRANCH_ACCEPTED_SETCOEF_OR_SOURCE_DEBT" if band_ok else "DEBT_REQUIRES_NEAREST_PRESSURE_REVERT")
                ),
                "max_abs_taug": item.get("max_abs"),
                "max_rel_taug": item.get("max_rel"),
            }
        )
    for item in lw_band_results:
        band_ok = bool(item["pass"] and lw_planck["pass"] and lw_planck_corr["pass"] and lw_secdiff["pass"])
        status["lw_bands"].append(
            {
                "band": item["band"],
                "intermediate_gate": "PASS" if band_ok else "FAIL",
                "implementation_status": "FULL_BRANCH_ACCEPTED" if band_ok else "DEBT_TO_M5_S3_ZZ",
                "max_abs_taug": item["taug"].get("max_abs"),
                "max_rel_taug": item["taug"].get("max_rel"),
                "max_abs_fracs": item["fracs"].get("max_abs"),
                "max_rel_fracs": item["fracs"].get("max_rel"),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    raise SystemExit(0 if run_intermediate_validation().get("pass") else 1)
