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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
