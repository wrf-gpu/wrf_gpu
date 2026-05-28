"""M5-S3.z RRTMG intermediate-oracle validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
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


def validate_lw_cldprmc_taucmc(jax, wrf) -> dict[str, Any]:
    """Validates LW `cldprmc` cloud optical depth at the WRF r4 oracle floor."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_cldprmc_taucmc")


def validate_lw_cldprmc_cldfmc(jax, wrf) -> dict[str, Any]:
    """Validates LW MCICA cloud mask at the WRF r4 oracle floor."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_cldprmc_cldfmc")


def validate_lw_rtrnmc_per_gpoint_flux(jax_zfd, jax_zfu, wrf_zfd, wrf_zfu, band: int) -> dict[str, Any]:
    """Validates one LW band's `rtrnmc` per-g-point down/up flux outputs."""

    zfd = _compare(jax_zfd, wrf_zfd, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_rtrnmc_zfd_per_gpoint", band=band)
    zfu = _compare(jax_zfu, wrf_zfu, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_rtrnmc_zfu_per_gpoint", band=band)
    return {"quantity": "lw_rtrnmc_per_gpoint_flux", "band": band, "pass": bool(zfd["pass"] and zfu["pass"]), "fields": {"zfd": zfd, "zfu": zfu}}


def validate_lw_rtrnmc_source_recurrence(jax_pfracs, wrf_pfracs, band: int) -> dict[str, Any]:
    """Validates one LW band's Planck fractions entering the `rtrnmc` recurrence."""

    return _compare(jax_pfracs, wrf_pfracs, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_rtrnmc_pfracs", band=band)


def validate_lw_rtrnmc_tfn_tbl(jax, wrf) -> dict[str, Any]:
    """Validates LW `rtrnmc` source-correction lookup outputs."""

    return _compare(jax, wrf, abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_rtrnmc_tfn_tbl_output")


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
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        sw_state, _ = load_sw_fixture_state()
        lw_state, _ = load_lw_fixture_state()
        sw = compute_rrtmg_sw_intermediates(sw_state)
        lw = compute_rrtmg_lw_intermediates(lw_state)

    sw_taug = _to_wrf_band_axis(sw.taug)
    sw_taur = _to_wrf_band_axis(sw.taur)
    sw_sfluxzen = _to_wrf_band_axis(sw.sfluxzen)
    lw_taug = _to_wrf_band_axis(lw.tau)
    lw_fracs = _to_wrf_band_axis(lw.fracs)
    lw_cldfmc = _to_wrf_band_axis(lw.cldprmc_cldfmc)
    lw_taucmc = _to_wrf_band_axis(lw.cldprmc_taucmc)
    lw_rtrnmc_pfracs = _to_wrf_band_axis(lw.rtrnmc_pfracs)
    lw_rtrnmc_tfn = _to_wrf_band_axis(lw.rtrnmc_tfn_tbl_output)
    lw_rtrnmc_zfd = _to_wrf_band_axis(lw.rtrnmc_zfd_per_gpoint)
    lw_rtrnmc_zfu = _to_wrf_band_axis(lw.rtrnmc_zfu_per_gpoint)

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

    sw_band_results = []
    for band in range(1, 15):
        g = int(SW_BAND_GPOINTS[band - 1])
        result = validate_sw_taug_per_band(sw_taug[:, :, :g, band - 1], oracle["sw_taug"][:, :, :g, band - 1], band)
        sw_band_results.append(result)

    lw_planck = validate_lw_planck_state(
        lw,
        {
            "planklay": oracle["lw_planklay"],
            "planklev": oracle["lw_planklev"],
            "plankbnd": oracle["lw_plankbnd"],
        },
    )
    lw_planck_corr = validate_lw_planck_corrections(lw.dplankup, lw.dplankdn, oracle["lw_dplankup"], oracle["lw_dplankdn"])
    lw_secdiff = _compare(lw.secdiff, oracle["lw_secdiff"], abs_tol=1.0e-6, rel_tol=1.0e-6, quantity="lw_secdiff")
    lw_band_results = []
    lw_cldprmc_band_results = []
    lw_rtrnmc_band_results = []
    for band in range(1, 17):
        g = int(LW_BAND_GPOINTS[band - 1])
        tau_result = validate_lw_taug_per_band(lw_taug[:, :, :g, band - 1], oracle["lw_taug"][:, :, :g, band - 1], band)
        frac_result = validate_lw_fracs_per_band(lw_fracs[:, :, :g, band - 1], oracle["lw_fracs"][:, :, :g, band - 1], band)
        lw_band_results.append({"band": band, "pass": bool(tau_result["pass"] and frac_result["pass"]), "taug": tau_result, "fracs": frac_result})
        cldfmc_result = validate_lw_cldprmc_cldfmc(lw_cldfmc[:, :, :g, band - 1], oracle["lw_cldprmc_cldfmc"][:, :, :g, band - 1])
        taucmc_result = validate_lw_cldprmc_taucmc(lw_taucmc[:, :, :g, band - 1], oracle["lw_cldprmc_taucmc"][:, :, :g, band - 1])
        cldfmc_result["band"] = band
        taucmc_result["band"] = band
        lw_cldprmc_band_results.append(
            {"band": band, "pass": bool(cldfmc_result["pass"] and taucmc_result["pass"]), "cldfmc": cldfmc_result, "taucmc": taucmc_result}
        )
        pfracs_result = validate_lw_rtrnmc_source_recurrence(
            lw_rtrnmc_pfracs[:, :, :g, band - 1], oracle["lw_rtrnmc_pfracs"][:, :, :g, band - 1], band
        )
        tfn_band_result = _compare(
            lw_rtrnmc_tfn[:, :, :g, band - 1],
            oracle["lw_rtrnmc_tfn_tbl_output"][:, :, :g, band - 1],
            abs_tol=1.0e-4,
            rel_tol=1.0e-3,
            quantity="lw_rtrnmc_tfn_tbl_output",
            band=band,
        )
        flux_result = validate_lw_rtrnmc_per_gpoint_flux(
            lw_rtrnmc_zfd[:, :, :g, band - 1],
            lw_rtrnmc_zfu[:, :, :g, band - 1],
            oracle["lw_rtrnmc_zfd_per_gpoint"][:, :, :g, band - 1],
            oracle["lw_rtrnmc_zfu_per_gpoint"][:, :, :g, band - 1],
            band,
        )
        lw_rtrnmc_band_results.append(
            {
                "band": band,
                "pass": bool(pfracs_result["pass"] and tfn_band_result["pass"] and flux_result["pass"]),
                "pfracs": pfracs_result,
                "tfn_tbl": tfn_band_result,
                "per_gpoint_flux": flux_result,
            }
        )

    lw_rtrnmc_tfn_all = validate_lw_rtrnmc_tfn_tbl(lw_rtrnmc_tfn, oracle["lw_rtrnmc_tfn_tbl_output"])
    lw_rtrnmc_plansum = _compare(lw.rtrnmc_plansum, oracle["lw_rtrnmc_plansum"], abs_tol=1.0e-4, rel_tol=1.0e-3, quantity="lw_rtrnmc_plansum")

    record = {
        "fixture": str(ORACLE.relative_to(ROOT)),
        "wrf_source_lines": {
            "sw_setcoef": "module_ra_rrtmg_sw.F:2843-3099",
            "sw_taumol": "module_ra_rrtmg_sw.F:3190-4653",
            "sw_spcvmc_entry": "module_ra_rrtmg_sw.F:8196-8450",
            "lw_setcoef": "module_ra_rrtmg_lw.F:3556-3921",
            "lw_taumol": "module_ra_rrtmg_lw.F:4824-7942",
            "lw_o3_profile": "module_ra_rrtmg_lw.F:12398-12418,12842-13035",
            "lw_top_buffer_temperature": "module_ra_rrtmg_lw.F:12329-12378",
            "lw_cldprmc": "module_ra_rrtmg_lw.F:2738-3027",
            "lw_rtrnmc_source": "module_ra_rrtmg_lw.F:3253-3409",
            "lw_tfn_tbl": "module_ra_rrtmg_lw.F:8054-8070",
        },
        "sw": {"setcoef": sw_setcoef, "taur": sw_taur_result, "sfluxzen": sw_sflux_result, "taug_per_band": sw_band_results},
        "lw": {"planck": lw_planck, "planck_corrections": lw_planck_corr, "secdiff": lw_secdiff, "per_band": lw_band_results},
        "lw_cldprmc": {"pass": bool(all(item["pass"] for item in lw_cldprmc_band_results)), "per_band": lw_cldprmc_band_results},
        "lw_rtrnmc": {
            "pass": bool(lw_rtrnmc_tfn_all["pass"] and lw_rtrnmc_plansum["pass"] and all(item["pass"] for item in lw_rtrnmc_band_results)),
            "tfn_tbl": lw_rtrnmc_tfn_all,
            "plansum": lw_rtrnmc_plansum,
            "per_band": lw_rtrnmc_band_results,
        },
    }
    record["pass"] = bool(
        sw_setcoef["pass"]
        and sw_taur_result["pass"]
        and sw_sflux_result["pass"]
        and all(item["pass"] for item in sw_band_results)
        and lw_planck["pass"]
        and lw_planck_corr["pass"]
        and lw_secdiff["pass"]
        and all(item["pass"] for item in lw_band_results)
        and all(item["pass"] for item in lw_cldprmc_band_results)
        and lw_rtrnmc_tfn_all["pass"]
        and lw_rtrnmc_plansum["pass"]
        and all(item["pass"] for item in lw_rtrnmc_band_results)
    )

    status = {
        "policy": "M5-S3.zzz: SW setcoef and LW scalar helpers use WRF r4 oracle floors where applicable; LW taumol branches are accepted per band only when taug and fracs pass abs<=1e-8 + rel<=1e-4.",
        "sw_bands": [],
        "lw_bands": [],
        "lw_cldprmc_bands": [],
        "lw_rtrnmc_bands": [],
    }
    for item in sw_band_results:
        band_ok = bool(item["pass"])
        all_sw_ok = bool(band_ok and sw_setcoef["pass"] and sw_taur_result["pass"] and sw_sflux_result["pass"])
        status["sw_bands"].append(
            {
                "band": item["band"],
                "intermediate_gate": "PASS" if all_sw_ok else "FAIL",
                "taumol_branch_gate": "PASS" if band_ok else "FAIL",
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
        taug_ok = bool(item["taug"]["pass"])
        fracs_ok = bool(item["fracs"]["pass"])
        band_ok = bool(taug_ok and fracs_ok and lw_planck["pass"] and lw_planck_corr["pass"])
        status["lw_bands"].append(
            {
                "band": item["band"],
                "intermediate_gate": "PASS" if band_ok else "FAIL",
                "taug_gate": "PASS" if taug_ok else "FAIL",
                "fracs_gate": "PASS" if fracs_ok else "FAIL",
                "implementation_status": "FULL_BRANCH_ACCEPTED" if band_ok else "FALLBACK_NEAREST_PRESSURE",
                "max_abs_taug": item["taug"].get("max_abs"),
                "max_rel_taug": item["taug"].get("max_rel"),
                "max_abs_fracs": item["fracs"].get("max_abs"),
                "max_rel_fracs": item["fracs"].get("max_rel"),
            }
        )
    for cld_item, rt_item in zip(lw_cldprmc_band_results, lw_rtrnmc_band_results, strict=True):
        band_ok = bool(cld_item["pass"] and rt_item["pass"])
        status["lw_cldprmc_bands"].append(
            {
                "band": cld_item["band"],
                "intermediate_gate": "PASS" if cld_item["pass"] else "FAIL",
                "cldfmc_gate": "PASS" if cld_item["cldfmc"]["pass"] else "FAIL",
                "taucmc_gate": "PASS" if cld_item["taucmc"]["pass"] else "FAIL",
                "implementation_status": "FULL_BRANCH_ACCEPTED" if band_ok else "FALLBACK_NEAREST_PRESSURE",
                "debt": None if band_ok else "LW cldprmc/rtrnmc branch did not satisfy intermediate oracle; keep nearest-pressure fallback for failing quantity.",
                "max_abs_cldfmc": cld_item["cldfmc"].get("max_abs"),
                "max_abs_taucmc": cld_item["taucmc"].get("max_abs"),
            }
        )
        status["lw_rtrnmc_bands"].append(
            {
                "band": rt_item["band"],
                "intermediate_gate": "PASS" if rt_item["pass"] else "FAIL",
                "source_recurrence_gate": "PASS" if rt_item["pfracs"]["pass"] else "FAIL",
                "tfn_tbl_gate": "PASS" if rt_item["tfn_tbl"]["pass"] else "FAIL",
                "per_gpoint_flux_gate": "PASS" if rt_item["per_gpoint_flux"]["pass"] else "FAIL",
                "implementation_status": "FULL_BRANCH_ACCEPTED" if band_ok else "FALLBACK_NEAREST_PRESSURE",
                "max_abs_pfracs": rt_item["pfracs"].get("max_abs"),
                "max_abs_tfn_tbl": rt_item["tfn_tbl"].get("max_abs"),
                "max_abs_zfd": rt_item["per_gpoint_flux"]["fields"]["zfd"].get("max_abs"),
                "max_abs_zfu": rt_item["per_gpoint_flux"]["fields"]["zfu"].get("max_abs"),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    raise SystemExit(0 if run_intermediate_validation().get("pass") else 1)
