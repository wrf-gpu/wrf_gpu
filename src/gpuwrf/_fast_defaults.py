"""Release defaults: the validated fp32 fast paths are ON unless the operator opts out.

Applied once at ``import gpuwrf`` BEFORE any submodule reads its import-time
flag, so a run with NO flag file resolves exactly the switch set of
``scripts/bench/flags_lw9.env`` (+ GPUWRF_SPEC_RING_SELECT and the
REAL_ALL pair): same environment,
hence the same AOT cheap keys and the same traced programs as a flags-file run.

Opt-outs: any explicitly set value wins (``GPUWRF_DYN_FP32=0`` keeps the legacy
path for that family); ``GPUWRF_FAST_DEFAULTS=0`` leaves the environment
untouched (all legacy defaults, the pre-v0.3 behaviour).

No imports beyond ``os``: this must run before JAX is imported.
"""

from __future__ import annotations

import os

# Exact resolved switch set of the LW9 release candidate (integrate LW9,
# flags_lw9.env sha 84f310de) plus the in-place ring select (b-carry 48824c974)
# and the WRF REAL dycore + carry pair (S2-DYN / b-carry G10: -9.8 % ordinary root),
# plus the merged LW10 physics levers (MYNN plume sums / tridiag REAL, LW band sums, KISS kernel).
# plus the b-core acoustic (A1c split w, A2 mass blocks, masked uv) and PD species-loop kernels (LW11).
# plus the fused large-step u/v PGF + Coriolis + curvature kernel on nested domains and the rhs_ph REAL
# stencil kernel (b-diff #15, LW12).
# plus the v0.3.1 b-core levers: blocked acoustic w recurrence (A1-SL, bitwise) and the Noah-MP layer lists +
# column kernels (#14, Noah-gate equivalent).
FAST_PATH_DEFAULTS: dict[str, str] = {
    # dynamics
    "GPUWRF_DYN_FP32": "1",
    "GPUWRF_DYN_ADVECTION_FP32": "1",
    "GPUWRF_DYN_DIFFUSION_FP32": "1",
    "GPUWRF_DYN_RK_FP32": "1",
    "GPUWRF_DYN_CARRY_FP32": "1",
    "GPUWRF_DYN_REAL_ALL": "1",  # S2-DYN; paired with CARRY_REAL_ALL (b-carry G10)
    "GPUWRF_CARRY_REAL_ALL": "1",
    "GPUWRF_DYN_PD_FP32": "1",
    "GPUWRF_ACOUSTIC_ALIAS": "1",
    "GPUWRF_ACOUSTIC_W2PASS": "3",  # b-core A1c split w phase (1dc7f2dd6; supersedes A1 =1)
    "GPUWRF_ACOUSTIC_MASS_BLOCK": "1",  # b-core A2 (b774ee8eb)
    "GPUWRF_ACOUSTIC_UV_MASKED": "1",  # b-core (1dc7f2dd6)
    "GPUWRF_DYN_PD_SPECIES_LOOP": "1",  # b-core #16 (6aac4bd5c)
    "GPUWRF_ACOUSTIC_W_RECUR_SL": "1",  # b-core A1-SL (8ee208a92): blocked w recurrence, bitwise (W1b/W1c)
    "GPUWRF_BOUNDARY_FP32": "1",
    "GPUWRF_SPEC_RING_SELECT": "1",
    "GPUWRF_DYN_GLUE_FUSED": "momuvn_rhsph_uvn_pin",  # b-diff #15 rhs_ph stencil + nested-only fused u/v, pinned; BD82 + nested-only u/v momentum advection
    "GPUWRF_LAYOUT_PIN": "1",  # b-core v0.3.2 P2: nested-only row-major layout pins, bitwise 0/343, d02 transposes 441 -> 223
    # land surface / surface layer / GWDO
    "GPUWRF_NOAHMP_NATIVE_REAL": "1",
    "GPUWRF_NOAHMP_ITERATION_BARRIER": "1",
    "GPUWRF_NOAHMP_LAYER_SELECT": "1",
    "GPUWRF_NOAHMP_LAYER_LISTS": "1",  # b-core #14 (2cab1a234): snow-water column + SOILWATER on layer lists
    "GPUWRF_NOAHMP_COLUMN_KERNELS": "1",  # b-core #14: canopy/bare Newton, SOILWATER, snow-water column kernels
    "GPUWRF_SFCLAY_NATIVE_REAL": "1",
    "GPUWRF_GWDO_NATIVE_REAL": "1",
    # PBL
    "GPUWRF_EDMF_FUSED_PLUME": "1",
    "GPUWRF_MYNN_FP32_PLUME": "1",
    "GPUWRF_MYNN_FP32_COLUMNS": "1",
    "GPUWRF_MYNN_FP32_BOULAC": "0",
    "GPUWRF_MYNN_CLOUDMIX": "1",
    "GPUWRF_MYNN_PLUME_SUMS": "1",  # b-phys a5f4c0126 (compile LW10b 13.2 gate PASS)
    "GPUWRF_MYNN_TRIDIAG_REAL": "1",
    # cumulus
    "GPUWRF_KF_RESIDENT_TABLES": "1",
    "GPUWRF_KF_COLUMN_FP32": "1",
    # radiation
    "GPUWRF_MCICA_JUMPAHEAD": "1",
    "GPUWRF_MCICA_LEGACY_FP64": "0",
    "GPUWRF_RRTMG_LW_FUSED_TRANSFER": "1",
    "GPUWRF_RRTMG_SW_FUSED_QUADRATURE": "1",
    "GPUWRF_RRTMG_COLUMN_TILE_COLS": "4096",
    "GPUWRF_RRTMG_LW_BAND_SUMS": "1",  # b-phys a5f4c0126
    "GPUWRF_MCICA_KISS_KERNEL": "1",  # BP61 (stream bitwise vs jump-ahead)
    "GPUWRF_RRTMG_SW_BAND_SUMS": "1",  # lever-phys LP02: fused SW band kernel (GPU tier1 + LP02 receipts)
    # microphysics
    "GPUWRF_THOMPSON_FP32": "0",
    "GPUWRF_THOMPSON_NATIVE_REAL": "1",
    "GPUWRF_THOMPSON_FULL_COLUMN": "1",
    "GPUWRF_THOMPSON_COLUMN_SED": "1",
    "GPUWRF_THOMPSON_SED_FP32": "1",
    "GPUWRF_THOMPSON_SED_PREP_FUSED": "0",
    "GPUWRF_THOMPSON_COLUMN_LAYOUT": "1",
    "GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT": "1",
    "GPUWRF_THOMPSON_IMPLICIT_SED": "0",
    # output
    "GPUWRF_HISTORY_INSTEP_DIAG": "1",
    # dispatch: donate the carry into the advance executable (D13: wrfout 6/6 bytes OFF==ON, restart exact)
    "GPUWRF_CARRY_DONATE": "1",
    # run-mode values the flag file pins (kept identical for cheap-key parity)
    "GPUWRF_CENSUS": "0",
    "JAX_ENABLE_X64": "true",
    "JAX_PALLAS_USE_MOSAIC_GPU": "false",
}

FAST_DEFAULTS_STATUS: dict[str, object] = {}


def apply_fast_path_defaults() -> dict[str, object]:
    """Set every unset release switch; never overwrite an operator value."""

    FAST_DEFAULTS_STATUS.clear()
    if os.environ.get("GPUWRF_FAST_DEFAULTS", "1").strip().lower() in {"0", "false", "no", "off"}:
        FAST_DEFAULTS_STATUS.update(enabled=False, applied={}, kept={})
        return FAST_DEFAULTS_STATUS
    applied: dict[str, str] = {}
    kept: dict[str, str] = {}
    for name, value in FAST_PATH_DEFAULTS.items():
        if name in os.environ:
            kept[name] = os.environ[name]
        else:
            os.environ[name] = value
            applied[name] = value
    FAST_DEFAULTS_STATUS.update(enabled=True, applied=applied, kept=kept)
    return FAST_DEFAULTS_STATUS
