"""Proof generator for diff_opt=1 / km_opt=4 (2-D Smagorinsky horizontal diffusion).

Runs the analytic-oracle parity checks against the literal WRF formulas
(module_diffusion_em.F:smag2d_km + cal_deform_and_div; module_big_step_utilities_em.F:
horizontal_diffusion / horizontal_diffusion_3dmp) and emits a structured proof JSON
with the max-abs parity residuals.  CPU fp64; no GPU required.

Run:  taskset -c 0-3 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
          python proofs/v090/diffopt1_smagorinsky_parity.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.dynamics.explicit_diffusion import (
    C_S_DEFAULT,
    PRANDTL,
    horizontal_deformation_2d,
    horizontal_diffusion_coord_scalar_tendency,
    smag2d_horizontal_km,
)


def _deformation_parity():
    nz, ny, nx = 4, 24, 32
    dx, dy = 1000.0, 1000.0
    kx = 2.0 * np.pi / (nx * dx)
    ky = 2.0 * np.pi / (ny * dy)
    xf = np.arange(nx + 1) * dx
    yc = (np.arange(ny) + 0.5) * dy
    xc = (np.arange(nx) + 0.5) * dx
    yf = np.arange(ny + 1) * dy
    u = np.broadcast_to((np.sin(kx * xf)[None, :] * np.cos(ky * yc)[:, None])[None], (nz, ny, nx + 1)).copy()
    v = np.broadcast_to((np.cos(kx * xc)[None, :] * np.sin(ky * yf)[:, None])[None], (nz, ny + 1, nx)).copy()
    d11, d22, d12 = horizontal_deformation_2d(jnp.asarray(u), jnp.asarray(v), dx_m=dx, dy_m=dy)
    d11, d22, d12 = map(np.asarray, (d11, d22, d12))

    d11_o = 2.0 * (np.sin(kx * xf[1:]) - np.sin(kx * xf[:-1]))[None, :] / dx * np.cos(ky * yc)[:, None]
    d11_o = np.broadcast_to(d11_o[None], (nz, ny, nx))
    d22_o = 2.0 * np.cos(kx * xc)[None, :] * (np.sin(ky * yf[1:]) - np.sin(ky * yf[:-1]))[:, None] / dy
    d22_o = np.broadcast_to(d22_o[None], (nz, ny, nx))
    u_m, v_m = u[:, :, :nx], v[:, :ny, :]
    d12_o = (u_m - np.roll(u_m, 1, axis=1)) / dy + (v_m - np.roll(v_m, 1, axis=2)) / dx
    return {
        "d11_max_abs_residual": float(np.max(np.abs(d11 - d11_o))),
        "d22_max_abs_residual": float(np.max(np.abs(d22 - d22_o))),
        "d12_max_abs_residual": float(np.max(np.abs(d12 - d12_o))),
    }


def _smag_parity():
    rng = np.random.default_rng(0)
    nz, ny, nx = 4, 18, 22
    dx, dy = 750.0, 750.0
    d11 = 1e-4 * rng.standard_normal((nz, ny, nx))
    d22 = 1e-4 * rng.standard_normal((nz, ny, nx))
    d12 = 1e-4 * rng.standard_normal((nz, ny, nx))
    xkmh, xkhh = smag2d_horizontal_km(
        jnp.asarray(d11), jnp.asarray(d22), jnp.asarray(d12),
        dx_m=dx, dy_m=dy, c_s=C_S_DEFAULT, prandtl=PRANDTL,
    )
    nw = np.roll(d12, -1, axis=1)
    se = np.roll(d12, -1, axis=2)
    ne = np.roll(np.roll(d12, -1, axis=1), -1, axis=2)
    tmp = 0.25 * (d12 + nw + se + ne)
    def2 = 0.25 * (d11 - d22) ** 2 + tmp ** 2
    mlen_h = np.sqrt(dx * dy)
    xkmh_o = np.minimum(C_S_DEFAULT ** 2 * mlen_h ** 2 * np.sqrt(def2), 10.0 * mlen_h)
    xkhh_o = xkmh_o / PRANDTL
    return {
        "c_s": C_S_DEFAULT,
        "prandtl": PRANDTL,
        "mlen_h_m": float(mlen_h),
        "xkmh_max_abs_residual": float(np.max(np.abs(np.asarray(xkmh) - xkmh_o))),
        "xkhh_max_abs_residual": float(np.max(np.abs(np.asarray(xkhh) - xkhh_o))),
        "xkhh_is_3x_xkmh_max_residual": float(np.max(np.abs(np.asarray(xkhh) - 3.0 * np.asarray(xkmh)))),
    }


def _hdiff_parity():
    rng = np.random.default_rng(1)
    nz, ny, nx = 5, 20, 24
    dx, dy = 800.0, 800.0
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    kx, ky = 2 * np.pi / (nx * dx), 2 * np.pi / (ny * dy)
    field = np.broadcast_to((np.sin(kx * xc)[None, :] * np.cos(ky * yc)[:, None])[None], (nz, ny, nx)).copy() + 300.0
    K = np.broadcast_to((50.0 + 30.0 * (np.cos(kx * xc)[None, :] * np.sin(ky * yc)[:, None]))[None], (nz, ny, nx)).copy()
    mass = 9.0e4 + 1.0e3 * rng.standard_normal((nz, ny, nx))
    tend = np.asarray(horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy))
    rdx, rdy = 1.0 / dx, 1.0 / dy
    ke = 0.5 * (np.roll(K, -1, axis=2) + K); kw = 0.5 * (K + np.roll(K, 1, axis=2))
    me = 0.5 * (np.roll(mass, -1, axis=2) + mass); mw = 0.5 * (mass + np.roll(mass, 1, axis=2))
    kn = 0.5 * (np.roll(K, -1, axis=1) + K); ks = 0.5 * (K + np.roll(K, 1, axis=1))
    mn = 0.5 * (np.roll(mass, -1, axis=1) + mass); ms = 0.5 * (mass + np.roll(mass, 1, axis=1))
    tend_o = rdx * (ke * me * rdx * (np.roll(field, -1, axis=2) - field) - kw * mw * rdx * (field - np.roll(field, 1, axis=2)))
    tend_o += rdy * (kn * mn * rdy * (np.roll(field, -1, axis=1) - field) - ks * ms * rdy * (field - np.roll(field, 1, axis=1)))
    per_level = np.asarray(jnp.sum(jnp.asarray(tend), axis=(1, 2)))
    return {
        "flux_divergence_max_abs_residual": float(np.max(np.abs(tend - tend_o))),
        "mass_weighted_integral_max_abs_level_sum": float(np.max(np.abs(per_level))),
    }


def main() -> None:
    deform = _deformation_parity()
    smag = _smag_parity()
    hdiff = _hdiff_parity()

    tol = 1e-9
    passed = (
        max(deform.values()) < tol
        and smag["xkmh_max_abs_residual"] < 1e-12
        and smag["xkhh_max_abs_residual"] < 1e-12
        and hdiff["flux_divergence_max_abs_residual"] < 1e-9
        and hdiff["mass_weighted_integral_max_abs_level_sum"] < 1e-5
    )

    proof = {
        "proof_id": "v090-diffopt1-smagorinsky-parity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "worker/opus/v090-diffopt1-smagorinsky",
        "objective": "WRF-faithful diff_opt=1 (coordinate-surface horizontal diffusion) + "
        "km_opt=4 (2-D Smagorinsky horizontal eddy viscosity Kh from horizontal deformation).",
        "wrf_reference": {
            "smag2d_km": "dyn_em/module_diffusion_em.F:1934-2044",
            "cal_deform_and_div": "dyn_em/module_diffusion_em.F:17-1190 (D11/D22/D12, eqns 13a/13b/13d)",
            "horizontal_diffusion": "dyn_em/module_big_step_utilities_em.F:2715-2950",
            "horizontal_diffusion_3dmp": "dyn_em/module_big_step_utilities_em.F:2954-3060",
            "diff_opt1_dispatch": "dyn_em/module_em.F:802-878 (forward_step, rk_step==1)",
            "constants": "prandtl=1/3 (share/module_model_constants.F:86); c_s default 0.25 (Registry.EM_COMMON:2866)",
        },
        "method": "analytic-oracle: each JAX operator compared to a literal NumPy "
        "transcription of the WRF formula on smooth periodic analytic fields (CPU fp64). "
        "Config scope: unit map factors (msf=1), flat eta surfaces (zx=zy=0), periodic x/y "
        "-- the documented idealized-slab reduction matching the existing const-K path scope. "
        "diff_opt=1 carries NO slope reduction (smag2d_km slope branch is gated on diff_opt==2).",
        "checks": {
            "deformation_tensor_vs_analytic": deform,
            "smag2d_km_kh_vs_wrf_formula": smag,
            "coordinate_surface_flux_divergence_vs_wrf": hdiff,
        },
        "no_regression": "diff_opt=1/km_opt=4 is a SEPARATE dispatch branch (operational_mode.py "
        "_augment_large_step_tendencies) gated on diff_opt==1 and km_opt==4. The existing "
        "diff_opt=2/km_opt=1 const-K path (const_nu_m2_s>0) is byte-unchanged; the idealized "
        "Straka/warm-bubble cases use const_nu_m2_s and leave diff_opt/km_opt at 0, so the new "
        "branch is inert for them (proven bit-identical in tests/dynamics/"
        "test_diffopt1_smagorinsky_integration.py::test_baseline_unchanged_when_smag_not_selected).",
        "wrf_faithful_no_clamps": "No tuning clamps. The only ceiling is the literal WRF "
        "smag2d_km min(xkmh, 10*mlen_h) (module_diffusion_em.F:2019), a faithful transcription "
        "of the WRF stability ceiling, not a masking clamp.",
        "tolerances": {"deformation": tol, "smag_kh": 1e-12, "flux_divergence": 1e-9, "conservation": 1e-5},
        "verdict": "PASS" if passed else "FAIL",
    }

    out = Path("proofs/v090/diffopt1_smagorinsky_parity.json")
    out.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))
    print(f"\nwrote {out}  verdict={proof['verdict']}")


if __name__ == "__main__":
    main()
