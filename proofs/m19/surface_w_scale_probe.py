"""Compute the exact surface-w unit-scale factors at t=0 (NO forecast, NO GPU loop).

Determines: mass_w_stage[0] = c1f[0]*muts + c2f[0] (the divisor in
small_step_finish:53), the coupled-vs-physical wind ratio (mass_u/msf), and
what surface w the BC produces for decoupled (u_1) vs coupled (u) winds over
the peak-terrain column.  This is the algebra behind the contradiction.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import jax


def main() -> int:
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig

    cfg = DailyPipelineConfig(
        run_id="20260521_18z_l2_72h_20260522T133443Z",
        run_root=Path("/mnt/data/canairy_meteo/runs/wrf_l2"),
        domain="d02",
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=180,
    )
    case, run_dir = _build_real_case(cfg)
    s = case.state
    nl = case.namelist

    def g(x):
        return np.asarray(jax.device_get(x))

    # Build DycoreMetrics-style fields the same way small_step_prep does.
    # c1f/c2f are vertical coordinate coefficients; muts/mut ~ dry mass (Pa).
    metrics = nl.metrics
    candidates = {}
    for name in ("c1f", "c2f", "c1h", "c2h", "msfty", "msfux", "msftx"):
        if hasattr(metrics, name) and getattr(metrics, name) is not None:
            candidates[name] = g(getattr(metrics, name))
    print("found metric fields:", list(candidates.keys()), flush=True)

    mu_total = g(s.mu_total) if hasattr(s, "mu_total") else g(s.mu)
    print(f"mu_total range [{mu_total.min():.4g}, {mu_total.max():.4g}] Pa  (dry mass)", flush=True)

    c1f = candidates.get("c1f")
    c2f = candidates.get("c2f")
    if c1f is not None and c2f is not None:
        muts = mu_total  # at t=0 reference==state so muts~=mut~=mu_total(+base?)
        mass_w_stage_k0 = c1f[0] * muts + c2f[0]
        print(f"c1f[0]={c1f[0]:.6g}  c2f[0]={c2f[0]:.6g}", flush=True)
        print(f"mass_w_stage[0] = c1f[0]*muts + c2f[0] range "
              f"[{np.min(mass_w_stage_k0):.4g}, {np.max(mass_w_stage_k0):.4g}] Pa", flush=True)
        print(f"  -> divisor in small_step_finish:53 is O(1e5) = {np.mean(mass_w_stage_k0):.4g}", flush=True)

    c1h = candidates.get("c1h"); c2h = candidates.get("c2h")
    msfty = candidates.get("msfty")
    if c1h is not None and msfty is not None:
        mass_u = c1h[0] * mu_total + c2h[0]
        coupled_ratio = mass_u / msfty
        print(f"\ncoupled-wind ratio mass_u/msf = (c1h[0]*mu+c2h[0])/msfty range "
              f"[{np.min(coupled_ratio):.4g}, {np.max(coupled_ratio):.4g}]", flush=True)
        print(f"  -> uv_state.u (coupled) is ~{np.mean(coupled_ratio):.4g}x the physical u_1 (m/s)", flush=True)

    # physical surface winds (state.u/v are physical at t=0 init)
    u_phys = g(s.u); v_phys = g(s.v)
    print(f"\nphysical u range [{u_phys.min():.3g},{u_phys.max():.3g}] m/s "
          f"v range [{v_phys.min():.3g},{v_phys.max():.3g}] m/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
