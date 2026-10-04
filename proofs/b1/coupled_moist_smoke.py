"""B1 coupled moist-column smoke: thompson_adapter on a cloudy State.

Proves (no oracle required, runs today):
  * fp64 preserved on every written field,
  * all hydrometeor + number-concentration fields evolve (no silent no-op) on a
    MOIST/cloudy column pack (the known failure to avoid),
  * precip accumulators increment via per-step += (Gate-1 decision #3),
  * water mass + surface precip closure across the step (sedimentation conserves
    mass to the surface).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.coupling.physics_couplers import thompson_adapter

GRAVITY = 9.80665
P0 = 100000.0
RD_CP = 287.0 / 1004.0
WRITTEN = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")
ACCUMS = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "proofs" / "b1" / "coupled_moist_smoke.json"


def _grid(nz: int, ny: int, nx: int) -> GridSpec:
    projection = Projection("lambert", 30.0, -90.0, 3000.0, 3000.0, ny, nx)
    terrain = TerrainProvenance(
        source_path="analytic://b1-coupled-moist-smoke",
        sha256="analytic-b1-coupled-moist-smoke",
        shape=(ny, nx),
        units="m",
        projection_transform="flat",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", ("w", "ph", "theta"), 1, "linear", True)
    return GridSpec(projection, terrain, vertical, bc, eta, jnp.zeros((ny, nx), dtype=jnp.float64))


def _moist_state(grid: GridSpec) -> State:
    shapes = _state_field_shapes(grid)
    arrays = {f: jnp.zeros(s, dtype=jnp.float64) for f, s in shapes.items()}
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    # Hydrostatic-ish pressure column, decreasing with height.
    p_col = jnp.linspace(92000.0, 25000.0, nz, dtype=jnp.float64)
    p = jnp.broadcast_to(p_col[:, None, None], (nz, ny, nx))
    # Geopotential interfaces: ~250 m layers.
    z_faces = jnp.arange(nz + 1, dtype=jnp.float64) * 250.0
    ph = jnp.broadcast_to((GRAVITY * z_faces)[:, None, None], (nz + 1, ny, nx))
    # Temperature ~ surface 288 K, lapse to colder aloft; potential temperature.
    T_col = jnp.linspace(285.0, 230.0, nz, dtype=jnp.float64)
    exner = (p_col / P0) ** RD_CP
    theta_col = T_col / exner
    theta = jnp.broadcast_to(theta_col[:, None, None], (nz, ny, nx))

    # MOIST cloudy pack: vapour near saturation in lower levels, cloud + rain
    # in mid levels, ice/snow/graupel in cold upper levels.
    zk = jnp.arange(nz, dtype=jnp.float64)[:, None, None]
    qv = (6.0e-3 * jnp.exp(-zk / 8.0) + 1.0e-4) * jnp.ones((nz, ny, nx))
    warm = (zk < nz * 0.5).astype(jnp.float64)
    cold = (zk >= nz * 0.4).astype(jnp.float64)
    qc = 1.2e-3 * warm * jnp.ones((nz, ny, nx))
    qr = 6.0e-4 * warm * jnp.ones((nz, ny, nx))
    qi = 2.5e-4 * cold * jnp.ones((nz, ny, nx))
    qs = 4.0e-4 * cold * jnp.ones((nz, ny, nx))
    qg = 1.5e-4 * cold * jnp.ones((nz, ny, nx))

    arrays.update(
        p=p, p_total=p, p_perturbation=jnp.zeros_like(p),
        ph=ph, ph_total=ph, ph_perturbation=jnp.zeros_like(ph),
        theta=theta, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
        Ni=1.0e4 * cold * jnp.ones((nz, ny, nx)),
        Nr=1.0e3 * warm * jnp.ones((nz, ny, nx)),
        Ns=5.0e2 * cold * jnp.ones((nz, ny, nx)),
        Ng=1.0e2 * cold * jnp.ones((nz, ny, nx)),
    )
    mu = jnp.ones((ny, nx), dtype=jnp.float64) * 90000.0
    arrays.update(mu=mu, mu_total=mu, mu_perturbation=jnp.zeros_like(mu))
    return State(**arrays)


def main() -> dict:
    nz, ny, nx = 32, 3, 3
    grid = _grid(nz, ny, nx)
    state = _moist_state(grid)
    dt = 60.0

    out = thompson_adapter(state, dt, grid)

    # fp64 on every written field.
    fp64 = {f: str(getattr(out, f).dtype) for f in WRITTEN + ACCUMS}
    all_fp64 = all(v == "float64" for v in fp64.values())

    # All moisture + number fields evolve.
    deltas = {}
    for f in WRITTEN:
        deltas[f] = float(jnp.max(jnp.abs(getattr(out, f) - getattr(state, f))))
    evolved = {f: deltas[f] > 0.0 for f in WRITTEN}
    all_evolved = all(evolved.values())

    # Precip accumulators incremented from zero (per-step +=).
    accum = {f: float(jnp.max(getattr(out, f))) for f in ACCUMS}
    accum_incremented = any(accum[f] > 0.0 for f in ACCUMS)

    # Finite everywhere.
    finite = all(bool(jnp.all(jnp.isfinite(getattr(out, f)))) for f in WRITTEN + ACCUMS)

    # Water closure: column water mass change + surface precip ~ 0 per column.
    # Use a FIXED reference air density (the input column rho) and FIXED dz so the
    # budget isolates water mass transport from the thermodynamic density change
    # that latent heating legitimately induces (the air-mass coordinate is fixed
    # within a microphysics step in WRF; only water mass and heat move).
    rho_ref = (0.622 * state.p) / (287.04 * (state.theta * (state.p / P0) ** RD_CP) * (state.qv + 0.622))
    dz_ref = jnp.maximum((state.ph[1:] - state.ph[:-1]) / GRAVITY, 1.0)

    def col_water_mass(s):
        qtot = s.qv + s.qc + s.qr + s.qi + s.qs + s.qg
        return jnp.sum(qtot * rho_ref * dz_ref, axis=0)  # (ny,nx) kg/m^2

    mass0 = col_water_mass(state)
    mass1 = col_water_mass(out)
    # Per-step precip increments (mm == kg/m^2) for this single step.
    precip_total = out.rain_acc + out.snow_acc + out.graupel_acc + out.ice_acc
    closure = jnp.abs((mass1 - mass0) + precip_total)
    closure_rel = float(jnp.max(closure / jnp.maximum(mass0, 1e-30)))

    record = {
        "scenario": "moist cloudy column pack (qc/qr warm + qi/qs/qg cold), 32 levels",
        "dt_s": dt,
        "fp64_all_written_fields": all_fp64,
        "field_dtypes": fp64,
        "all_moisture_number_fields_evolve": all_evolved,
        "per_field_max_abs_delta": deltas,
        "field_evolved": evolved,
        "precip_accumulators_incremented": accum_incremented,
        "precip_accum_max_mm": accum,
        "all_finite": finite,
        "water_closure_max_rel_residual": closure_rel,
        "water_closure_pass": closure_rel < 1e-3,
        "pass": bool(all_fp64 and all_evolved and accum_incremented and finite and closure_rel < 1e-3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    rec = main()
    print(json.dumps(rec, indent=2, sort_keys=True))
