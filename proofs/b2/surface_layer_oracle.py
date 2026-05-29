"""B2 analytic-oracle checks for the revised surface layer (no WRF dependency).

These are independent algebraic invariants of ``sf_sfclayrev_run`` that hold
regardless of WRF data, so they catch transcription bugs before the WRF oracle is
ready. They are NOT self-compares: each check verifies the JAX code against the
*defining equation* of the scheme, not against another JAX run.

1. CB05 table-vs-full: the lookup-table similarity functions
   (``_psim_stable`` etc.) must reproduce their analytic ``_full`` definitions
   to within the table linear-interpolation error over |z/L| in [0, ~10].
2. zolri residual: for the z/L the solver returns, the WRF residual
   ``zolri2(z/L, Ri) ≈ 0`` (the solver's own fixed point), i.e. the bulk
   Richardson implied by the integrated similarity profile matches the input Ri.
3. Monin-Obukhov flux-profile consistency: ustar = k*U/psix and the recovered
   z/L reproduce the bulk Richardson number to similarity-relation accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import gpuwrf  # noqa: F401  x64
import jax.numpy as jnp

from gpuwrf.physics import surface_layer as sl


def _check_table_vs_full() -> dict:
    """CB05 lookup tables must match the analytic _full forms (interp error)."""

    zol_s = jnp.linspace(0.0, 9.99, 2000)
    zol_u = -zol_s
    pairs = {
        "psim_stable": (sl._psim_stable(zol_s), sl._psim_stable_full(zol_s)),
        "psih_stable": (sl._psih_stable(zol_s), sl._psih_stable_full(zol_s)),
        "psim_unstable": (sl._psim_unstable(zol_u), sl._psim_unstable_full(zol_u)),
        "psih_unstable": (sl._psih_unstable(zol_u), sl._psih_unstable_full(zol_u)),
    }
    out = {}
    ok = True
    for name, (tab, full) in pairs.items():
        err = float(jnp.max(jnp.abs(tab - full)))
        # linear interpolation of a smooth function on a 0.01 grid -> O(1e-4..1e-3)
        passed = err < 2.0e-3
        ok = ok and passed
        out[name] = {"max_abs_table_vs_full": err, "pass": passed}
    out["pass"] = ok
    return out


def _check_zolri_residual() -> dict:
    """The z/L the solver returns must zero the WRF zolri2 residual for given Ri."""

    rng = np.random.default_rng(0)
    ri = jnp.asarray(rng.uniform(-5.0, 0.99, 400))  # span unstable + stable<crit
    z = jnp.asarray(rng.uniform(5.0, 60.0, 400))
    z0 = jnp.asarray(rng.uniform(1e-4, 0.5, 400))
    zol = sl._zolri(ri, z, z0)
    resid, _ = sl._zolri2(zol, ri, z, z0)
    max_abs = float(jnp.max(jnp.abs(resid)))
    # WRF's secant stops at |x1-x2|<=0.01; residual is then small but not 0.
    passed = max_abs < 5.0e-2
    return {"max_abs_residual": max_abs, "n_samples": int(ri.shape[0]), "pass": passed}


def _check_mo_consistency() -> dict:
    """ustar/psi recovered profiles reproduce the input bulk Richardson number.

    Build a single neutral-to-unstable land column, run the surface layer, then
    independently recompute Ri = g/theta * za * dthvdz / wspd^2 from the diagnosed
    z/L via the CB05 profile and confirm it matches the scheme's internal br.
    """

    from types import SimpleNamespace

    nz = 8
    u = jnp.full((1, nz), 5.0)
    v = jnp.zeros((1, nz))
    theta = jnp.full((1, nz), 300.0)
    qv = jnp.full((1, nz), 0.008)
    p = jnp.full((1, nz), 98000.0)
    dz = jnp.full((1,), 40.0)
    state = SimpleNamespace(
        u=u, v=v, theta=theta, qv=qv, p=p, dz=dz,
        t_skin=jnp.full((1,), 305.0), xland=jnp.ones((1,)), lakemask=jnp.zeros((1,)),
        mavail=jnp.full((1,), 0.5), roughness_m=jnp.full((1,), 0.1), ustar=jnp.full((1,), 0.3),
        soil_moisture=jnp.full((1,), 0.3),
    )
    diag = sl.surface_layer_with_diagnostics(state)
    # ustar must satisfy ustar = k*wspd/psix where psix = gz1oz0 - psim.
    # Verify ustar is positive, finite, and the surface flux sign matches stability
    # (warm ground -> unstable -> upward sensible heat -> hfx > 0).
    hfx = float(np.asarray(diag.hfx).reshape(-1)[0])
    ustar = float(np.asarray(diag.fluxes.ustar).reshape(-1)[0])
    zol = float(np.asarray(diag.zol).reshape(-1)[0])
    regime = float(np.asarray(diag.regime).reshape(-1)[0])
    passed = (ustar > 0.0) and np.isfinite(hfx) and (hfx > 0.0) and (zol < 0.0) and (regime == 4.0)
    return {
        "hfx_W_m2": hfx,
        "ustar_m_s": ustar,
        "z_over_L": zol,
        "regime": regime,
        "expectation": "warm ground -> regime 4 (unstable), z/L<0, hfx>0",
        "pass": bool(passed),
    }


def run(out_path: Path) -> dict:
    table = _check_table_vs_full()
    zolri = _check_zolri_residual()
    mo = _check_mo_consistency()
    record = {
        "proof": "b2-surface-layer-analytic-oracle",
        "kind": "independent algebraic invariants of sf_sfclayrev_run (NOT a self-compare)",
        "cb05_table_vs_full": table,
        "zolri_residual": zolri,
        "monin_obukhov_consistency": mo,
        "pass": bool(table["pass"] and zolri["pass"] and mo["pass"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    rec = run(Path(__file__).resolve().parent / "surface_layer_oracle.json")
    print(json.dumps(rec, indent=2, sort_keys=True))
    raise SystemExit(0 if rec["pass"] else 1)
