#!/usr/bin/env python3
"""Small idealized proof for F1 diff_opt=2 / km_opt=2,3,5 turbulence.

Runs on the active JAX backend.  Use the repository GPU lock for the real proof:

  scripts/with_gpu_lock.sh --label gpt-f1-3d-tke -- \
    env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python proofs/v022/feature_push/f1_3d_tke_smag_oracle.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "dynamics"))

from gpuwrf.dynamics.advection import apply_halo, halo_spec  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    _augment_large_step_tendencies,
    _diffopt2_turbulence_fields,
    _tke_coupled_tendency,
)
from test_diffopt2_tke_smag3d import (  # noqa: E402
    C_S_DEFAULT,
    PRANDTL,
    _build_grid,
    _build_state,
    _namelist,
    _smag3d_oracle,
)


REFERENCE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F")
OUT = Path(__file__).with_name("f1_3d_tke_smag_oracle.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _range(arr) -> dict[str, float]:
    a = np.asarray(arr)
    return {"min": float(np.min(a)), "max": float(np.max(a)), "mean": float(np.mean(a))}


def main() -> int:
    grid = _build_grid(ny=6, nx=8, nz=4, dx=1000.0)
    state = _build_state(grid)
    haloed = apply_halo(state, halo_spec(grid))
    ph = haloed.ph_total
    dz = jnp.maximum(jnp.mean((ph[1:] - ph[:-1]) / 9.81), jnp.asarray(1.0, dtype=ph.dtype))
    mass_h = grid.metrics.c1h[:, None, None] * haloed.mu_total[None, :, :] + grid.metrics.c2h[:, None, None]
    baseline = _augment_large_step_tendencies(
        haloed,
        _namelist(grid, diff_opt=0, km_opt=0).tendencies,
        _namelist(grid, diff_opt=0, km_opt=0),
        rk_step=3,
    )

    results: dict[str, object] = {}
    all_pass = True
    for km_opt in (2, 3, 5):
        nl = _namelist(grid, diff_opt=2, km_opt=km_opt)
        turb = _diffopt2_turbulence_fields(haloed, nl, dz=dz)
        tend = _augment_large_step_tendencies(haloed, nl.tendencies, nl, rk_step=3)
        field_deltas = {}
        finite = True
        nonzero = True
        for field in ("u", "v", "w", "theta"):
            delta = np.asarray(getattr(tend, field) - getattr(baseline, field))
            field_deltas[field] = {
                "max_abs": float(np.max(np.abs(delta))),
                "l2": float(np.sqrt(np.mean(delta * delta))),
            }
            finite = finite and bool(np.all(np.isfinite(delta)))
            nonzero = nonzero and bool(np.max(np.abs(delta)) > 0.0)

        item: dict[str, object] = {
            "finite_tendency_deltas": finite,
            "nonzero_tendency_deltas": nonzero,
            "field_delta_norms": field_deltas,
            "bn2": _range(turb.bn2),
            "xkmh": _range(turb.xkmh),
            "xkmv": _range(turb.xkmv),
            "xkhh": _range(turb.xkhh),
            "xkhv": _range(turb.xkhv),
        }
        if km_opt == 3:
            exp = _smag3d_oracle(
                np.asarray(turb.d11),
                np.asarray(turb.d22),
                np.asarray(turb.d33),
                np.asarray(turb.d12),
                np.asarray(turb.d13),
                np.asarray(turb.d23),
                np.asarray(turb.bn2),
                dx=1000.0,
                dy=1000.0,
                dz=float(dz),
                dt=6.0,
                c_s=C_S_DEFAULT,
                pr=PRANDTL,
                mix_upper_bound=0.1,
            )
            got = (turb.xkmh, turb.xkmv, turb.xkhh, turb.xkhv)
            item["wrf_formula_oracle_max_abs"] = float(
                max(np.max(np.abs(np.asarray(g) - e)) for g, e in zip(got, exp))
            )
        if km_opt in (2, 5):
            qke_t = _tke_coupled_tendency(haloed, nl, turb, mass_h=mass_h, dz=dz)
            qke_new = (mass_h * haloed.qke + 6.0 * qke_t) / mass_h
            qke_new = jnp.minimum(jnp.maximum(qke_new, 0.0), float(nl.tke_upper_bound))
            item["qke_tendency"] = _range(qke_t)
            item["qke_after_one_dt"] = _range(qke_new)
            item["qke_finite_and_bounded"] = bool(
                np.all(np.isfinite(np.asarray(qke_new)))
                and float(jnp.min(qke_new)) >= 0.0
                and float(jnp.max(qke_new)) <= float(nl.tke_upper_bound)
            )

        item_pass = bool(finite and nonzero)
        if km_opt in (2, 5):
            item_pass = item_pass and bool(item["qke_finite_and_bounded"])
        if km_opt == 3:
            item_pass = item_pass and float(item["wrf_formula_oracle_max_abs"]) < 1.0e-8
        item["pass"] = item_pass
        all_pass = all_pass and item_pass
        results[f"km_opt_{km_opt}"] = item

    proof = {
        "schema": "gpuwrf.v022.f1_3d_tke_smag_oracle.v1",
        "status": "pass" if all_pass else "fail",
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "wrf_reference": str(REFERENCE),
        "wrf_reference_sha256": _sha256(REFERENCE),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "dx_m": grid.projection.dx_m},
        "landed": [
            "WRF 3-D Smagorinsky km_opt=3 coefficients",
            "WRF TKE km_opt=2 coefficients and dry TKE RHS/update",
            "SMS km_opt=5 coefficient blend using WRF pthl/pu functions",
            "diff_opt=2 runtime branch for km_opt=2/3/5",
            "namelist/catalog recognition for km_opt=2/3/5",
        ],
        "known_scaffold": [
            "momentum diffusion uses conservative variable-K staggered scalar flux divergence; exact WRF deformation-stress tensor parity remains to port",
            "dry BN2 reduction is used; moist calculate_N2 parity/oracle remains to thread",
            "km_opt=5 implicit vertical TKE dissipation/nonlocal flux extras are represented by the explicit dry idealized reduction",
        ],
        "results": results,
    }
    OUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
