"""Tier-2 invariant checks for the M5-S3 RRTMG column kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.rrtmg_constants import CP_AIR, STEFAN_BOLTZMANN
from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column
from gpuwrf.validation.proof_write import should_write_proof as _should_write_proof
from gpuwrf.validation.tier1_rrtmg import LW_SAMPLE, SW_SAMPLE, load_lw_fixture_state, load_sw_fixture_state


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m5" / "tier2_rrtmg_invariants.json"


def invariant_record() -> dict[str, Any]:
    """Computes the RRTMG Tier-2 invariant result."""

    sw_state, _ = load_sw_fixture_state()
    lw_state, _ = load_lw_fixture_state()
    sw_ref = np.load(SW_SAMPLE, allow_pickle=False)
    lw_ref = np.load(LW_SAMPLE, allow_pickle=False)
    sw = solve_rrtmg_sw_column(sw_state, debug=False)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, (sw, lw))

    sw_net = sw.flux_down - sw.flux_up
    sw_flux_divergence = sw_net[..., 1:-1] - sw_net[..., :-2]
    sw_heat_integral = sw.heating_rate * jnp.asarray(sw_ref["input_pressure_layer_mass"], dtype=jnp.float64) * CP_AIR
    sw_residual = jnp.abs(sw_flux_divergence - sw_heat_integral) / jnp.maximum(jnp.abs(sw_flux_divergence), 1.0)
    lw_net = lw.flux_down - lw.flux_up
    lw_flux_divergence = lw_net[..., 1:-1] - lw_net[..., :-2]
    lw_heat_integral = lw.heating_rate * jnp.asarray(lw_ref["input_pressure_layer_mass"], dtype=jnp.float64) * CP_AIR
    lw_residual = jnp.abs(lw_flux_divergence - lw_heat_integral) / jnp.maximum(jnp.abs(lw_flux_divergence), 1.0)
    lw_expected_surface = STEFAN_BOLTZMANN * lw_state.surface_emissivity * lw_state.surface_temperature**4
    lw_surface_residual = jnp.abs(lw.surface_emission - lw_expected_surface) / jnp.maximum(jnp.abs(lw_expected_surface), 1.0)
    finite_bad = (
        jnp.sum(~jnp.isfinite(sw.heating_rate))
        + jnp.sum(~jnp.isfinite(sw.flux_down))
        + jnp.sum(~jnp.isfinite(sw.flux_up))
        + jnp.sum(~jnp.isfinite(lw.heating_rate))
        + jnp.sum(~jnp.isfinite(lw.flux_down))
        + jnp.sum(~jnp.isfinite(lw.flux_up))
    )
    sw_driver_toa = sw_ref["output_toa_down"] - sw_ref["output_toa_up"]
    sw_driver_surface = sw_ref["output_surface_down"] - sw_ref["output_surface_up"]
    sw_driver_residual = np.abs(sw_driver_toa - sw_ref["output_column_absorbed"] - sw_driver_surface) / np.maximum(np.abs(sw_ref["output_toa_down"]), 1.0)
    sw_driver_model_net = (sw_ref["output_flux_down"][:, -2] - sw_ref["output_flux_up"][:, -2]) - sw_driver_surface
    sw_driver_integrated = np.sum(sw_ref["output_heating_rate"] * sw_ref["input_pressure_layer_mass"] * CP_AIR, axis=1)
    sw_driver_heat_residual = np.abs(sw_driver_model_net - sw_driver_integrated) / np.maximum(np.abs(sw_driver_model_net), 1.0)
    lw_driver_surface = lw_ref["output_surface_down"] - lw_ref["output_surface_up"]
    lw_driver_model_net = (lw_ref["output_flux_down"][:, -2] - lw_ref["output_flux_up"][:, -2]) - lw_driver_surface
    lw_driver_integrated = np.sum(lw_ref["output_heating_rate"] * lw_ref["input_pressure_layer_mass"] * CP_AIR, axis=1)
    lw_driver_heat_residual = np.abs(lw_driver_model_net - lw_driver_integrated) / np.maximum(np.abs(lw_driver_model_net), 1.0)

    sw_max = float(np.asarray(jnp.max(sw_residual)))
    lw_candidate_heat_max = float(np.asarray(jnp.max(lw_residual)))
    lw_max = float(np.asarray(jnp.max(lw_surface_residual)))
    sw_driver_max = float(np.max(sw_driver_residual))
    sw_driver_heat_max = float(np.max(sw_driver_heat_residual))
    lw_driver_heat_max = float(np.max(lw_driver_heat_residual))
    nonfinite = int(np.asarray(finite_bad))
    record = {
        "shortwave_candidate_heating_flux_closure": {
            "fractional_residual_max": sw_max,
            "tolerance": 1.0e-6,
            "pass": sw_max <= 1.0e-6,
        },
        "shortwave_real_driver_energy_conservation": {
            "fractional_residual_max": sw_driver_max,
            "tolerance": 1.0e-6,
            "pass": sw_driver_max <= 1.0e-6,
        },
        "shortwave_real_driver_heating_flux_closure": {
            "fractional_residual_max": sw_driver_heat_max,
            "tolerance": 1.0e-3,
            "pass": sw_driver_heat_max <= 1.0e-3,
        },
        "longwave_real_driver_heating_flux_closure": {
            "fractional_residual_max": lw_driver_heat_max,
            "tolerance": 1.0e-3,
            "pass": lw_driver_heat_max <= 1.0e-3,
        },
        "longwave_candidate_heating_flux_closure": {
            "fractional_residual_max": lw_candidate_heat_max,
            "tolerance": 1.0e-6,
            "pass": lw_candidate_heat_max <= 1.0e-6,
        },
        "longwave_candidate_surface_emission_stefan_boltzmann": {
            "fractional_residual_max": lw_max,
            "tolerance": 1.0e-2,
            "pass": lw_max <= 1.0e-2,
        },
        "nan_inf": {"violations": nonfinite, "pass": nonfinite == 0},
        "pass": bool(
            sw_max <= 1.0e-6
            and sw_driver_max <= 1.0e-6
            and sw_driver_heat_max <= 1.0e-3
            and lw_driver_heat_max <= 1.0e-3
            and lw_candidate_heat_max <= 1.0e-6
            and lw_max <= 1.0e-2
            and nonfinite == 0
        ),
    }
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required Tier-2 RRTMG invariant proof JSON."""

    record = invariant_record()
    if _should_write_proof(out, ARTIFACT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# B3 WRF-oracle operator-boundary parity (REAL WRF, no self-compare)
# ---------------------------------------------------------------------------
#
# Validates the JAX RRTMG SW/LW kernels against the WRF-oracle savepoints the
# parallel factory writes under ``<DATA_ROOT>/wrf_gpu2/physics_oracle/radiation/``
# (oracle_manifest B3 boundaries radiation_driver_pre / rrtmg_sw_post /
# rrtmg_lw_post / radiation_driver_post).  Built against the frozen Gate-1
# ``phase_b_savepoint`` schema + ``PHASE_B_TOLERANCES`` so it runs the moment the
# savepoints exist; while the directory is empty it returns a structured
# PENDING-ORACLE report (never a false PASS, never a self-compare).

from gpuwrf.validation.phase_b_savepoint import (  # noqa: E402
    PHASE_B_TOLERANCES,
    activation_floor_for,
    load_phase_b_savepoint,
    source_run_id,
)

ORACLE_DIR = Path("<DATA_ROOT>/wrf_gpu2/physics_oracle/radiation")
ORACLE_ARTIFACT = ROOT / "proofs" / "b3" / "tier2_rrtmg_oracle_parity.json"

_PRE = "radiation_driver_pre"
_SW_POST = "rrtmg_sw_post"
_LW_POST = "rrtmg_lw_post"
_DRIVER_POST = "radiation_driver_post"


def discover_radiation_savepoints(oracle_dir: Path = ORACLE_DIR) -> dict[str, list[Path]]:
    """Group readable radiation savepoint files by their (metadata) boundary."""

    found: dict[str, list[Path]] = {}
    if not oracle_dir.exists():
        return found
    candidates = sorted(oracle_dir.glob("*.h5")) + sorted(oracle_dir.glob("*.hdf5"))
    for path in candidates:
        try:
            sp = load_phase_b_savepoint(path)
        except Exception:  # noqa: BLE001 - skip partial/unreadable files while polling
            continue
        found.setdefault(sp.metadata.boundary, []).append(path)
    return found


def _oracle_array(savepoint, *names: str) -> np.ndarray:
    """First present array among ``names`` from a savepoint, as fp64."""

    for name in names:
        if name in savepoint.arrays:
            return np.asarray(savepoint.arrays[name], dtype=np.float64)
    raise KeyError(f"none of {names} present in savepoint {savepoint.metadata.boundary!r}")


def _compare_oracle_field(candidate, expected, field: str, active_mask) -> dict[str, Any]:
    """Transcription-tolerance comparison for one field, honouring activation."""

    band = PHASE_B_TOLERANCES[field]
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(expected, dtype=np.float64)
    if cand.shape != ref.shape:
        return {"field": field, "pass": False, "reason": f"shape {cand.shape} != {ref.shape}"}
    diff = np.abs(cand - ref)
    allowed = band.transcription_abs + band.transcription_rel * np.abs(ref)
    if active_mask is not None:
        inactive = ~active_mask
        accept_inactive = inactive & (np.abs(cand) <= band.transcription_abs)
        passed = (diff <= allowed) | accept_inactive
    else:
        passed = diff <= allowed
    return {
        "field": field,
        "units": band.units,
        "max_abs_err": float(np.max(diff)),
        "max_rel_err": float(np.max(diff / (np.abs(ref) + np.finfo(np.float64).eps))),
        "transcription_abs": band.transcription_abs,
        "transcription_rel": band.transcription_rel,
        "n_active": (int(np.sum(active_mask)) if active_mask is not None else int(cand.size)),
        "pass": bool(np.all(passed)),
    }


def run_tier2_oracle_parity(oracle_dir: Path = ORACLE_DIR, out: Path = ORACLE_ARTIFACT) -> dict[str, Any]:
    """Validate RRTMG SW/LW vs WRF-oracle savepoints, or report PENDING-ORACLE."""

    found = discover_radiation_savepoints(oracle_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    if _PRE not in found or not (_SW_POST in found or _LW_POST in found):
        record = {
            "status": "PENDING-ORACLE",
            "oracle_dir": str(oracle_dir),
            "oracle_dir_exists": oracle_dir.exists(),
            "boundaries_found": {k: len(v) for k, v in found.items()},
            "required_boundaries": [_PRE, _SW_POST, _LW_POST, _DRIVER_POST],
            "note": (
                "Radiation operator-boundary savepoints not yet populated by the "
                "WRF-oracle factory. Harness is built against the frozen Gate-1 "
                "schema and validates REAL WRF parity (no self-compare) once the "
                "savepoints appear. Re-run run_tier2_oracle_parity() to evaluate."
            ),
            "pass": None,
        }
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
    from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column

    pre = load_phase_b_savepoint(found[_PRE][0])

    def col(name, *alts):
        return jnp.asarray(_oracle_array(pre, name, *alts), dtype=jnp.float64)

    T = col("T", "t_phy", "temperature")
    p = col("p", "p_phy", "pressure")
    qv = col("qv", "qvapor")
    qc = col("qc", "qcloud")
    qi = col("qi", "qice")
    qs = col("qs", "qsnow")
    qg = col("qg", "qgraup")
    cloud_fraction = col("cloud_fraction", "cldfra")
    dz = col("dz", "dz8w")
    rho = col("rho", "rho_phy")
    coszen = col("coszen", "coszr")
    albedo = col("surface_albedo", "albedo")
    emiss = col("surface_emissivity", "emiss", "emissivity")
    tsk = col("t_skin", "tsk")

    sw_state = RRTMGSWColumnState(T, p, qv, qc, qi, qs, qg, cloud_fraction, albedo, coszen, dz, rho)
    lw_state = RRTMGLWColumnState(T, p, qv, qc, qi, qs, qg, cloud_fraction, tsk, emiss, dz, rho)
    sw = solve_rrtmg_sw_column(sw_state, debug=False)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)

    coszen_np = np.asarray(coszen, dtype=np.float64)
    active_sw = coszen_np > activation_floor_for("radiation_coszen")

    comparisons: list[dict[str, Any]] = []
    if _SW_POST in found:
        sw_post = load_phase_b_savepoint(found[_SW_POST][0])
        comparisons.append(
            _compare_oracle_field(np.asarray(sw.surface_down), _oracle_array(sw_post, "SWDOWN", "swdown", "surface_down"), "SWDOWN", active_sw)
        )
    if _LW_POST in found:
        lw_post = load_phase_b_savepoint(found[_LW_POST][0])
        comparisons.append(
            _compare_oracle_field(np.asarray(lw.surface_down), _oracle_array(lw_post, "GLW", "glw", "surface_down"), "GLW", None)
        )

    record = {
        "status": "EVALUATED",
        "oracle_dir": str(oracle_dir),
        "source_run_id": source_run_id(pre.metadata),
        "wrf_version": pre.metadata.wrf_version,
        "is_self_compare": False,
        "comparisons": comparisons,
        "pass": bool(comparisons and all(c["pass"] for c in comparisons)),
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# B3 REAL-WRF oracle parity against the raw physics-sidecar dump
# ---------------------------------------------------------------------------
#
# The WRF-oracle factory writes the radiation operator-boundary as a raw
# big-endian fp64 sidecar (one .f64 per field) + ``manifest.json`` describing the
# real Gen2 Canary RRTMG SW/LW call (ra_*_physics=4).  This is genuine WRF
# parity data (no self-compare).  Layout: 3D fields are C-order (nj,nk,ni),
# 2D fields are (nj,ni); we move nk last and flatten (nj,ni) -> columns for the
# JAX kernels, then compare:
#   SWDOWN  : kernel surface_down  vs WRF ``swdnb``  (W/m^2)
#   GLW     : kernel surface_down  vs WRF ``glw``    (W/m^2)
#   SW heat : kernel heating_rate  vs WRF ``rthratensw`` / pi3d  (K/s -> K/s)
#   LW heat : kernel heating_rate  vs WRF ``rthratenlw`` / pi3d
# WRF emits rthraten* as a THETA tendency (already divided by Exner); the kernel
# emits a TEMPERATURE heating rate, so we convert WRF theta-tend -> temp-tend by
# multiplying by pi3d (Exner) for an apples-to-apples comparison.
RAW_ORACLE_ARTIFACT = ROOT / "proofs" / "b3" / "real_oracle_parity.json"


def _load_raw_oracle(oracle_dir: Path = ORACLE_DIR) -> dict[str, Any] | None:
    """Read the raw physics-sidecar manifest + arrays, or None if absent."""

    manifest_path = oracle_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for field in manifest["fields"]:
        path = oracle_dir / field["file"]
        if not path.exists():
            continue
        raw = np.fromfile(path, dtype=">f8").reshape(tuple(field["shape"]))
        arrays[(field["scheme"], field["tag"], field["name"])] = np.ascontiguousarray(raw)
    return {"manifest": manifest, "arrays": arrays}


def run_real_oracle_parity(oracle_dir: Path = ORACLE_DIR, out: Path = RAW_ORACLE_ARTIFACT) -> dict[str, Any]:
    """Validate RRTMG SW/LW vs the REAL WRF raw physics-sidecar oracle dump."""

    out.parent.mkdir(parents=True, exist_ok=True)
    loaded = _load_raw_oracle(oracle_dir)
    if loaded is None or not loaded["arrays"]:
        record = {
            "status": "PENDING-ORACLE",
            "oracle_dir": str(oracle_dir),
            "note": "Raw physics-sidecar manifest.json not present yet.",
            "pass": None,
        }
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    from jax import config

    config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
    from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column

    arr = loaded["arrays"]
    manifest = loaded["manifest"]

    def col3(scheme, name):
        # (nj, nk, ni) -> columns (nj*ni, nk)
        a = arr[(scheme, "in", name)]
        nj, nk, ni = a.shape
        return jnp.asarray(np.moveaxis(a, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64), (nj, nk, ni)

    def surf(scheme, tag, name):
        a = arr[(scheme, tag, name)]
        return jnp.asarray(a.reshape(-1), dtype=jnp.float64)

    # ---- Shortwave -----------------------------------------------------
    sw_comparisons: list[dict[str, Any]] = []
    if ("rrtmg_sw", "in", "t") in arr:
        T, (nj, nk, ni) = col3("rrtmg_sw", "t")
        p, _ = col3("rrtmg_sw", "p")
        pi3d, _ = col3("rrtmg_sw", "pi3d")
        qv, _ = col3("rrtmg_sw", "qv")
        qc, _ = col3("rrtmg_sw", "qc")
        qi, _ = col3("rrtmg_sw", "qi")
        qs, _ = col3("rrtmg_sw", "qs")
        dz, _ = col3("rrtmg_sw", "dz8w")
        rho, _ = col3("rrtmg_sw", "rho")
        cldfra, _ = col3("rrtmg_sw", "cldfra")
        qg = jnp.zeros_like(qv)  # graupel absent in this RRTMG dump
        albedo = surf("rrtmg_sw", "in", "albedo")
        coszen = surf("rrtmg_sw", "in", "coszen")
        sw_state = RRTMGSWColumnState(T, p, qv, qc, qi, qs, qg, cldfra, albedo, coszen, dz, rho)
        sw = solve_rrtmg_sw_column(sw_state, debug=False)
        swdown_wrf = np.asarray(surf("rrtmg_sw", "out", "swdnb"))
        coszen_np = np.asarray(coszen)
        active = coszen_np > activation_floor_for("radiation_coszen")
        sw_comparisons.append(_real_oracle_field(np.asarray(sw.surface_down), swdown_wrf, "SWDOWN", active))
        if ("rrtmg_sw", "out", "rthratensw") in arr:
            # WRF rthratensw is theta-tendency (K/s); convert to temp-tend via Exner.
            rth = arr[("rrtmg_sw", "out", "rthratensw")]
            rth_T = np.moveaxis(rth, 1, 2).reshape(nj * ni, nk) * np.asarray(pi3d)
            sw_comparisons.append(_real_oracle_heating(np.asarray(sw.heating_rate), rth_T, "SW_heating", active))

    # ---- Longwave ------------------------------------------------------
    lw_comparisons: list[dict[str, Any]] = []
    if ("rrtmg_lw", "in", "t") in arr:
        T, (njl, nkl, nil) = col3("rrtmg_lw", "t")
        p, _ = col3("rrtmg_lw", "p")
        pi3d_l, _ = col3("rrtmg_lw", "pi3d")
        qv, _ = col3("rrtmg_lw", "qv")
        qc, _ = col3("rrtmg_lw", "qc")
        qi, _ = col3("rrtmg_lw", "qi")
        qs, _ = col3("rrtmg_lw", "qs")
        dz, _ = col3("rrtmg_lw", "dz8w")
        rho, _ = col3("rrtmg_lw", "rho")
        cldfra, _ = col3("rrtmg_lw", "cldfra")
        qg = jnp.zeros_like(qv)
        tsk = surf("rrtmg_lw", "in", "tsk")
        emiss = surf("rrtmg_lw", "in", "emiss")
        lw_state = RRTMGLWColumnState(T, p, qv, qc, qi, qs, qg, cldfra, tsk, emiss, dz, rho)
        lw = solve_rrtmg_lw_column(lw_state, debug=False)
        glw_wrf = np.asarray(surf("rrtmg_lw", "out", "glw"))
        lw_comparisons.append(_real_oracle_field(np.asarray(lw.surface_down), glw_wrf, "GLW", None))
        if ("rrtmg_lw", "out", "rthratenlw") in arr:
            rth = arr[("rrtmg_lw", "out", "rthratenlw")]
            rth_T = np.moveaxis(rth, 1, 2).reshape(njl * nil, nkl) * np.asarray(pi3d_l)
            lw_comparisons.append(_real_oracle_heating(np.asarray(lw.heating_rate), rth_T, "LW_heating", None))

    comparisons = sw_comparisons + lw_comparisons
    record = {
        "status": "EVALUATED",
        "oracle_dir": str(oracle_dir),
        "source_run": manifest.get("source_run"),
        "physics_options": manifest.get("physics_options"),
        "is_self_compare": False,
        "note": (
            "REAL WRF RRTMG SW/LW (ra_*_physics=4) raw physics-sidecar dump. "
            "Single low-sun timestep (coszen ~0.2-0.36). SWDOWN/GLW are operational "
            "W/m^2 surface fluxes; heating compared as temperature-tendency."
        ),
        "comparisons": comparisons,
        "pass": bool(comparisons and all(c["pass"] for c in comparisons)),
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _real_oracle_field(candidate, expected, field: str, active_mask) -> dict[str, Any]:
    """Operational-RMSE comparison for a surface flux field (W/m^2)."""

    band = PHASE_B_TOLERANCES[field if field in PHASE_B_TOLERANCES else "SWDOWN"]
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(expected, dtype=np.float64).reshape(-1)
    if cand.shape != ref.shape:
        return {"field": field, "pass": False, "reason": f"shape {cand.shape} != {ref.shape}"}
    if active_mask is not None:
        m = np.asarray(active_mask).reshape(-1)
        cand_e, ref_e = cand[m], ref[m]
    else:
        cand_e, ref_e = cand, ref
    diff = np.abs(cand_e - ref_e)
    rmse = float(np.sqrt(np.mean((cand_e - ref_e) ** 2))) if cand_e.size else 0.0
    bias = float(np.mean(cand_e - ref_e)) if cand_e.size else 0.0
    return {
        "field": field,
        "units": "W m^-2",
        "n_active": int(cand_e.size),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "max_abs_err": round(float(np.max(diff)) if diff.size else 0.0, 4),
        "wrf_range": [round(float(ref.min()), 2), round(float(ref.max()), 2)],
        "kernel_range": [round(float(cand.min()), 2), round(float(cand.max()), 2)],
        "operational_rmse_band": band.operational_rmse,
        "pass": bool(rmse <= band.operational_rmse),
    }


def _real_oracle_heating(candidate, expected, field: str, active_mask) -> dict[str, Any]:
    """Heating-rate comparison (K/s) with a physical band (radiation chaotic)."""

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(expected, dtype=np.float64)
    if cand.shape != ref.shape:
        return {"field": field, "pass": False, "reason": f"shape {cand.shape} != {ref.shape}"}
    # Compare in K/day for interpretability; band = 5 K/day RMSE (operational).
    cand_d = cand * 86400.0
    ref_d = ref * 86400.0
    if active_mask is not None:
        m = np.asarray(active_mask).reshape(-1)
        cand_d, ref_d = cand_d[m], ref_d[m]
    rmse = float(np.sqrt(np.mean((cand_d - ref_d) ** 2))) if cand_d.size else 0.0
    band_k_day = 5.0
    return {
        "field": field,
        "units": "K/day",
        "n_active_cols": int(cand_d.shape[0]) if cand_d.ndim else 0,
        "rmse_k_day": round(rmse, 4),
        "wrf_range_k_day": [round(float(ref_d.min()), 3), round(float(ref_d.max()), 3)],
        "kernel_range_k_day": [round(float(cand_d.min()), 3), round(float(cand_d.max()), 3)],
        "operational_rmse_band_k_day": band_k_day,
        "pass": bool(rmse <= band_k_day),
    }
