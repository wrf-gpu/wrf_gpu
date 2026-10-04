"""B1 oracle-harness MECHANICS self-test (NOT a physics-parity claim).

This proves the WRF-oracle parity harness wiring is correct and ready for the
real savepoints: it builds an oracle-shaped (k, j, i) array dict, runs
``compare_against_oracle``, and checks that the harness

  * reshapes WRF (k, j, i) -> (n_columns, n_levels) correctly,
  * applies the inactive-physical moist mask,
  * applies the frozen Phase-B transcription tolerance ladder per field, and
  * computes the water-mass + precip closure.

To validate ONLY the harness mechanics (orientation, masking, tolerances,
closure) we feed the kernel's own one-step output back in as the "post" arrays;
parity therefore trivially passes and the closure is the genuine kernel water
budget.  This is explicitly a mechanics check — REAL physics parity requires the
WRF oracle (run_oracle_parity, PENDING-ORACLE until the factory populates).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column_with_precip,
)
from gpuwrf.validation.tier1_thompson import compare_against_oracle, _ORACLE_TO_KERNEL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "proofs" / "b1" / "oracle_harness_mechanics.json"
P0 = 100000.0
RD_CP = 287.0 / 1004.0


def _oracle_arrays(nz: int, ny: int, nx: int, dt: float):
    """Build WRF-oracle-shaped (k, j, i) pre/post array dicts + dt."""

    rng = np.random.default_rng(7)
    # moist cloudy profile, varying per column
    p_col = np.linspace(92000.0, 25000.0, nz)
    T_col = np.linspace(286.0, 232.0, nz)
    pii_col = (p_col / P0) ** RD_CP
    th_col = T_col / pii_col

    def field3d(profile, jitter):
        base = np.broadcast_to(profile[:, None, None], (nz, ny, nx)).copy()
        return base * (1.0 + jitter * rng.standard_normal((nz, ny, nx)))

    zk = np.arange(nz)[:, None, None]
    warm = (zk < nz * 0.5).astype(np.float64)
    cold = (zk >= nz * 0.4).astype(np.float64)
    pre = {
        "p": np.broadcast_to(p_col[:, None, None], (nz, ny, nx)).copy(),
        "th": np.broadcast_to(th_col[:, None, None], (nz, ny, nx)).copy(),
        "pii": np.broadcast_to(pii_col[:, None, None], (nz, ny, nx)).copy(),
        "qv": field3d(6e-3 * np.exp(-np.arange(nz) / 8.0) + 1e-4, 0.05),
        "qc": 1.2e-3 * warm * (1.0 + 0.1 * rng.standard_normal((nz, ny, nx))),
        "qr": 6e-4 * warm * (1.0 + 0.1 * rng.standard_normal((nz, ny, nx))),
        "qi": 2.5e-4 * cold * (1.0 + 0.1 * rng.standard_normal((nz, ny, nx))),
        "qs": 4e-4 * cold * (1.0 + 0.1 * rng.standard_normal((nz, ny, nx))),
        "qg": 1.5e-4 * cold * (1.0 + 0.1 * rng.standard_normal((nz, ny, nx))),
        "ni": 1e4 * cold,
        "nr": 1e3 * warm,
        "dz8w": np.full((nz, ny, nx), 250.0),
        "w": np.zeros((nz, ny, nx)),
    }
    pre = {k: np.maximum(v, 0.0) if k not in ("th", "pii", "p", "w") else v for k, v in pre.items()}

    # "post" = kernel's own one-step output reshaped back to (k, j, i).
    def to_cols(name):
        a = np.moveaxis(np.asarray(pre[name], dtype=np.float64), 0, -1)
        return jnp.asarray(a.reshape(-1, a.shape[-1]))

    T = np.moveaxis(pre["th"] * pre["pii"], 0, -1).reshape(-1, nz)
    qv = to_cols("qv")
    rho = density_from_pressure_temperature(to_cols("p"), jnp.asarray(T), qv)
    column = ThompsonColumnState(
        qv=qv, qc=to_cols("qc"), qr=to_cols("qr"), qi=to_cols("qi"), qs=to_cols("qs"),
        qg=to_cols("qg"), Ni=to_cols("ni"), Nr=to_cols("nr"), T=jnp.asarray(T), p=to_cols("p"),
        rho=rho, Ns=jnp.zeros_like(qv), Ng=jnp.zeros_like(qv), dz=to_cols("dz8w"), w=to_cols("w"),
    )
    out, _precip = step_thompson_column_with_precip(column, dt)
    post = dict(pre)  # copy structure
    for oracle_name, kernel_name in _ORACLE_TO_KERNEL.items():
        cols = np.asarray(getattr(out, kernel_name), dtype=np.float64).reshape(ny, nx, nz)
        post[oracle_name] = np.moveaxis(cols, -1, 0)  # back to (k, j, i)
    return pre, post


def main() -> dict:
    nz, ny, nx, dt = 32, 4, 4, 60.0
    pre, post = _oracle_arrays(nz, ny, nx, dt)
    record = compare_against_oracle(pre, post, dt)
    record["note"] = (
        "MECHANICS self-test only: post=kernel-own-output, so per-field parity is "
        "trivially exact; this validates harness reshape/mask/tolerance/closure "
        "wiring, NOT physics. Real physics parity = run_oracle_parity (PENDING-ORACLE)."
    )
    record["harness_mechanics_pass"] = bool(
        record["per_field"]["qv"]["pass"]
        and record["per_field"]["qg"]["pass"]
        and record["per_field"]["ni"]["pass"]
        and record["water_closure_pass"]
        and record["moist_cells"] > 0
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    rec = main()
    print(json.dumps({k: rec[k] for k in ("harness_mechanics_pass", "moist_cells", "n_columns", "water_closure_max_rel_residual", "water_closure_pass")}, indent=2))
