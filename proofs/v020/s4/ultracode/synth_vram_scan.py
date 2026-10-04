#!/usr/bin/env python3
"""Synthetic peak-VRAM scan of the operational dynamics core (physics OFF).

Goal (v0.20 S4 / ultracode)
---------------------------
Measure the PEAK GPU VRAM of the operational dynamics core at a chosen grid
size, under two precision regimes:

  * ``fp64``       -- every prognostic forced to float64 (the correctness path).
  * ``aggressive`` -- the ADR-007 fp32-gated DEFAULT_DTYPES matrix
                      (u/v/theta/qv + the moisture family fp32; w/mu/p/ph fp64).

PHYSICS IS OFF (dynamics-only): ``run_physics=False`` and ``run_boundary=False``
so no RRTMG / MYNN / Thompson run and no boundary files are needed.  The point
is to find the grid size where the fp64 carry OOMs but the fp32-gated carry
still fits -- i.e. where the ~halved storage of the gated fields buys capability.

The grid is built directly (not from a fixture) so an arbitrary (nz, ny, nx)
can be requested.  It reuses the idealized.py machinery: ``DycoreMetrics.flat``
plus the PURE-SIGMA (hybrid_opt=0) vertical-coefficient override that
idealized.py applies so the ``calc_coef_w`` tridiagonal is non-singular.  The
state is a resting theta=300 hydrostatic column.  PHYSICAL REALISM IS NOT
REQUIRED -- NaNs are acceptable; we measure memory, not correctness.  The only
hard requirement is that ``run_forecast_operational(run_physics=False)``
compiles and runs the requested steps WITHOUT a shape/dtype error.

Memory is read from JAX's in-process device counter
(``jax.local_devices()[0].memory_stats()['peak_bytes_in_use']``), the reliable
in-process peak; nvidia-smi is NOT used.  On CPU ``memory_stats`` is typically
unavailable, so ``peak_vram_mib`` is ``None`` there -- expected.

Usage
-----
    PYTHONPATH=src python proofs/v020/s4/ultracode/synth_vram_scan.py \
        --nz 50 --ny 300 --nx 300 --regime fp64 --steps 2 --out scan.json

A single grid size + single regime per invocation (drive a sweep externally).
"""

from __future__ import annotations

# Must precede any jax.numpy use so x64 is enabled at import (fp64 leaves).
from gpuwrf._x64_config import configure_jax_x64

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    run_forecast_operational,
)

configure_jax_x64()

# WRF dry-air constants (match idealized.py so the resting column is in discrete
# hydrostatic balance with the acoustic solver; values are not load-bearing for a
# memory measurement, but a well-posed column avoids a degenerate solve).
R_DRY_AIR = 287.0
CP_DRY_AIR = 1004.0
CV_DRY_AIR = CP_DRY_AIR - R_DRY_AIR
P0_PA = 100000.0
THETA0_K = 300.0
GRAVITY_M_S2 = 9.81
P_SURFACE_PA = 100000.0
P_TOP_PA = 5000.0
DOMAIN_HEIGHT_M = 20000.0
DX_M = 3000.0


def _build_grid(nz: int, ny: int, nx: int) -> GridSpec:
    """Build a self-consistent GridSpec at an arbitrary (nz, ny, nx).

    Mirrors ``idealized._make_grid`` but generalised to ny > 1.  The vertical
    coefficients are overridden to PURE SIGMA (hybrid_opt=0: c1h=c1f=1, c2h=c2f=0,
    c3=c1, c4=0) exactly as idealized.py does, so the top-face dry mass is mut
    (nonzero) and the ``calc_coef_w`` tridiagonal is non-singular.  Everything
    else (map factors=1, terrain=0, no Coriolis) comes straight from
    ``DycoreMetrics.flat``.
    """

    if nz < 3:
        raise ValueError("nz must be >= 3 (DycoreMetrics.flat needs nz>=3 cf coeffs)")
    if ny < 1 or nx < 1:
        raise ValueError("ny and nx must be >= 1")

    # Normal eta ordering 1 -> 0 (surface -> top).  fp64 throughout (GridSpec
    # enforces fp64 for eta_levels / terrain_height).
    eta_levels = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)

    projection = Projection("lambert", 0.0, 0.0, float(DX_M), float(DX_M), nx, ny)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    terrain = TerrainProvenance(
        source_path="synthetic:vram-scan",
        sha256="synthetic-vram-scan",
        shape=(ny, nx),
        units="m",
        projection_transform="flat",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, float(DOMAIN_HEIGHT_M), eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="linear",
        restart_compatible=False,
    )

    base = DycoreMetrics.flat(
        ny=ny,
        nx=nx,
        nz=nz,
        eta_levels=eta_levels,
        top_pressure_pa=float(P_TOP_PA),
        provenance="synthetic-vram-scan",
    )
    # PURE-SIGMA override (idealized.py:416-440).  DycoreMetrics.flat builds a
    # HYBRID c1f=eta which makes the top-face dry mass (c1f[nz]*mut+c2f[nz])
    # vanish -> singular calc_coef_w (gamma=inf).  Pure sigma keeps the top-face
    # mass = mut (nonzero).  These overrides are nz-shaped and ny-independent, so
    # they apply unchanged for ny > 1.
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=base.msftx, msfty=base.msfty, msfux=base.msfux, msfuy=base.msfuy,
        msfvx=base.msfvx, msfvy=base.msfvy,
        c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
        dn=base.dn, dnw=base.dnw, rdn=base.rdn, rdnw=base.rdnw,
        cf1=base.cf1, cf2=base.cf2, cf3=base.cf3, fnm=base.fnm, fnp=base.fnp,
        dzdx=base.dzdx, dzdy=base.dzdy, dzdx_u=base.dzdx_u, dzdy_v=base.dzdy_v,
        f=base.f, e=base.e, sina=base.sina, cosa=base.cosa,
        p_top=base.p_top, provenance="synthetic-vram-scan-pure-sigma",
    )

    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta_levels,
        terrain_height=terrain_height,
        metrics=metrics,
        halo_width=2,
        staggering="c-grid",
    )


def _zeros_state(grid: GridSpec, *, mp_physics: int = 8) -> State:
    """Allocate a zeroed State on the FIRST LOCAL device (GPU if present, else CPU).

    ``State.zeros`` hard-requires a GPU (state.py:_gpu_device), which makes the
    mandatory CPU validation impossible.  This replicates ``State.zeros`` exactly
    -- the same ``_state_field_shapes`` leaf set, the same per-field
    DEFAULT_DTYPES dtype -- but places leaves on ``jax.local_devices()[0]`` so the
    same code path runs on CPU (no GPU) and GPU (one visible).  No source file is
    modified; this is the only deviation from calling ``State.zeros`` directly.
    """

    device = jax.local_devices()[0]
    fields = {
        field: jax.device_put(
            jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device
        )
        for field, shape in _state_field_shapes(grid, mp_physics=mp_physics).items()
    }
    return State(**fields)


def _zeros_tendencies(grid: GridSpec) -> Tendencies:
    """Allocate zero tendency buffers on the first local device (CPU or GPU).

    ``Tendencies.zeros`` also hard-requires a GPU; replicate its staggered-shape
    buffer set on ``jax.local_devices()[0]`` so ``from_grid`` need not call it.
    """

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    device = jax.local_devices()[0]

    def z(shape: tuple[int, ...], field: str) -> jax.Array:
        return jax.device_put(jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device)

    return Tendencies(
        u=z((nz, ny, nx + 1), "u"),
        v=z((nz, ny + 1, nx), "v"),
        w=z((nz + 1, ny, nx), "w"),
        theta=z((nz, ny, nx), "theta"),
        qv=z((nz, ny, nx), "qv"),
        p=z((nz, ny, nx), "p"),
        ph=z((nz + 1, ny, nx), "ph"),
        mu=z((ny, nx), "mu"),
    )


def _resting_state(grid: GridSpec) -> State:
    """Seed a resting theta=300 hydrostatic atmosphere on the zero state.

    All-fp64 seed; precision is then enforced by the caller via
    ``_enforce_operational_precision``.  ``mu_total`` MUST be positive --
    ``run_forecast_operational`` asserts ``max|mu_total| > 0`` before the scan
    (operational_mode._assert_nonzero_initial_mu_total).  Values target discrete
    hydrostatic balance with a constant theta=300 column so the acoustic /
    vertical solve is well-posed; NaNs would still be acceptable for a memory
    measurement, but a balanced column keeps the run clean.
    """

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    state = _zeros_state(grid)

    eta = np.asarray(grid.eta_levels, dtype=np.float64)  # (nz+1,) 1 -> 0
    eta_mass = 0.5 * (eta[:-1] + eta[1:])  # (nz,)

    mu_val = float(P_SURFACE_PA - P_TOP_PA)  # column dry mass (Pa), > 0
    mu_total = np.full((ny, nx), mu_val, dtype=np.float64)

    # Pure-sigma hydrostatic mass-level pressure: p(k) = eta_mass(k)*mu + p_top.
    p_mass_col = eta_mass * mu_val + float(P_TOP_PA)  # (nz,)
    p_total = np.broadcast_to(p_mass_col[:, None, None], (nz, ny, nx)).copy()

    theta = np.full((nz, ny, nx), THETA0_K, dtype=np.float64)

    # Hydrostatic geopotential on faces: ph(1)=0 at the surface, integrate with
    # the pure-sigma signed-dnw recurrence ph(k+1)=ph(k) - dnw(k)*mu*alpha(k),
    # alpha = (R/p0)*theta*(p/p0)^(-cv/cp) (dry EOS).  dnw is WRF-signed (negative
    # for the normal eta ordering), so -dnw*mu*alpha > 0 and ph increases upward.
    dnw = eta[1:] - eta[:-1]  # (nz,) negative
    alpha = (R_DRY_AIR / P0_PA) * theta[:, 0, 0] * (p_mass_col / P0_PA) ** (-CV_DRY_AIR / CP_DRY_AIR)
    ph_col = np.zeros(nz + 1, dtype=np.float64)
    for k in range(nz):
        ph_col[k + 1] = ph_col[k] - dnw[k] * mu_val * alpha[k]
    ph_total = np.broadcast_to(ph_col[:, None, None], (nz + 1, ny, nx)).copy()

    # Seed the authoritative TOTAL leaves; State.replace keeps the matching
    # perturbation as a delta (here perturbation stays 0 since the zeros-state
    # totals were 0 and we set the absolute totals -> perturbation = total).
    # That is harmless: we only need positive, finite, hydrostatically-consistent
    # totals for the solver and a positive mu_total for the entry assertion.
    state = state.replace(
        theta=jnp.asarray(theta),
        mu_total=jnp.asarray(mu_total),
        p_total=jnp.asarray(p_total),
        ph_total=jnp.asarray(ph_total),
    )
    return state


def _state_dtype_fraction_fp32(state: State) -> float:
    """Fraction of prognostic State BYTES carried in float32 (vs all leaves)."""

    fp32_bytes = 0
    total_bytes = 0
    for field in STATE_FIELD_ORDER:
        value = getattr(state, field, None)
        if value is None:
            continue
        nbytes = int(value.size) * int(value.dtype.itemsize)
        total_bytes += nbytes
        if value.dtype == jnp.float32:
            fp32_bytes += nbytes
    if total_bytes == 0:
        return 0.0
    return fp32_bytes / total_bytes


def _peak_vram_mib() -> float | None:
    """In-process peak device bytes -> MiB, or None if unavailable (e.g. CPU)."""

    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    if peak is None:
        return None
    return float(peak) / (1024.0 * 1024.0)


def run_one(nz: int, ny: int, nx: int, regime: str, steps: int) -> dict:
    """Build, seed, enforce precision, and run ``steps`` dynamics-only steps."""

    force_fp64 = regime == "fp64"
    cols = int(ny) * int(nx)
    dt_s = 6.0  # small dt; CFL is irrelevant (NaNs OK, we measure memory).
    hours = float(steps) * dt_s / 3600.0

    record: dict = {
        "regime": regime,
        "nz": int(nz),
        "ny": int(ny),
        "nx": int(nx),
        "cols": cols,
        "steps": int(steps),
        "dt_s": dt_s,
        "hours": hours,
        "force_fp64": force_fp64,
        "oom": False,
        "peak_vram_mib": None,
        "state_dtype_fraction_fp32": None,
        "ok": False,
        "error": None,
    }

    try:
        grid = _build_grid(nz, ny, nx)
        state = _resting_state(grid)
        # Apply the chosen precision regime to the carried state.  force_fp64=True
        # -> every prognostic fp64; force_fp64=False -> ADR-007 fp32-gated matrix
        # (the just-fixed _cast=False path genuinely downcasts u/v/theta/qv+moist).
        state = _enforce_operational_precision(state, force_fp64=force_fp64)
        record["state_dtype_fraction_fp32"] = _state_dtype_fraction_fp32(state)

        # Pass tendencies + metrics EXPLICITLY: from_grid otherwise calls
        # Tendencies.zeros (GPU-hard-required, breaks CPU validation) and rebuilds
        # metrics with DycoreMetrics.flat -- which would REPLACE our pure-sigma
        # override with the singular HYBRID c1f=eta and break the vertical solve.
        namelist = OperationalNamelist.from_grid(
            grid,
            tendencies=_zeros_tendencies(grid),
            metrics=grid.metrics,
            dt_s=dt_s,
            acoustic_substeps=10,
            radiation_cadence_steps=999_999,  # never triggers radiation
            use_vertical_solver=True,
            disable_guards=True,
            force_fp64=force_fp64,
        )
        # run_physics / run_boundary are NOT from_grid kwargs; set them on the
        # frozen namelist the same way idealized.py does, to isolate the dynamics
        # core (no RRTMG/MYNN/Thompson; no boundary files).  namelist.force_fp64
        # is the in-scan control: _initial_carry_for_run re-enforces precision
        # from it, so the actual scan-resident carry matches the regime.
        namelist = dataclasses.replace(namelist, run_physics=False, run_boundary=False)

        result = run_forecast_operational(state, namelist, hours)
        jax.block_until_ready(result)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- classify OOM vs other failures
        text = f"{type(exc).__name__}: {exc}"
        record["error"] = text
        lowered = text.lower()
        is_oom = (
            "resource_exhausted" in lowered
            or "out of memory" in lowered
            or "outofmemory" in lowered
            or "out-of-memory" in lowered
            or "ran out of memory" in lowered
            or type(exc).__name__ in ("XlaRuntimeError",)
            and "resource_exhausted" in lowered
        )
        if is_oom:
            record["oom"] = True
        # Record the in-process peak even on failure (it may be the OOM point).
        traceback.print_exc()

    # Read the peak AFTER execution (success or OOM) -- this is the in-process
    # high-water mark and survives a caught RESOURCE_EXHAUSTED.
    record["peak_vram_mib"] = _peak_vram_mib()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--regime", choices=("fp64", "aggressive"), required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    args = parser.parse_args(argv)

    record = run_one(args.nz, args.ny, args.nx, args.regime, args.steps)

    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}", file=sys.stderr)

    # Exit 0 on a clean run OR a cleanly-recorded OOM (both are valid data
    # points); non-zero only on an unexpected (non-OOM) failure so a sweep can
    # tell a measured OOM from a broken grid.
    if record["ok"] or record["oom"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
