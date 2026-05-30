"""De-risk the normal-component boundary fix WITHOUT touching production code.

Monkey-patches an EXPERIMENTAL stronger spec+relax of the NORMAL wind component
(U at W/E, V at S/N) over `apply_lateral_boundaries`, runs a SHORT forecast, and
checks whether the boundary spike is suppressed and the run stays finite. This is
a feasibility probe for the follow-up fix, not a production change.

The experimental rule: after the normal relaxation, additionally hard-pin the
NORMAL component's full spec+relax zone (b_dist = 0 .. relax_zone-1) toward the
interpolated boundary value with a strong weight, so the end-of-step correction
overpowers the acoustic pumping the diagnostic localized. Tangential component is
left exactly as production.

USAGE
  PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 OMP_NUM_THREADS=2 \
    taskset -c 0-3 python proofs/wind/normal_bdy_fix_probe.py --lead-h 0.5 --pin 0.7
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-h", type=float, default=0.5)
    ap.add_argument("--pin", type=float, default=0.7,
                    help="strong relaxation weight for the NORMAL component spec+relax zone")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.coupling import boundary_apply as BA
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented

    prod_apply = BA.apply_lateral_boundaries

    def strong_normal(field_name, field, boundary, lead_seconds, dt_s, config):
        """Extra strong pin of the NORMAL component over its spec+relax zone."""
        forcing = BA.interpolate_boundary_leaf(boundary, lead_seconds, config.update_cadence_s)
        z_len, y_len, x_len = field.shape
        out = field
        nrel = int(config.relax_zone)
        if field_name == "u":   # normal at W/E
            for b in range(nrel):
                wstrip = BA._strip(forcing, "W", b, z_len, y_len)
                estrip = BA._strip(forcing, "E", b, z_len, y_len)
                col_w = b
                col_e = x_len - 1 - b
                out = out.at[:, :, col_w].set((1 - args.pin) * out[:, :, col_w] + args.pin * wstrip)
                out = out.at[:, :, col_e].set((1 - args.pin) * out[:, :, col_e] + args.pin * estrip)
        elif field_name == "v":  # normal at S/N
            for b in range(nrel):
                sstrip = BA._strip(forcing, "S", b, z_len, x_len)
                nstrip = BA._strip(forcing, "N", b, z_len, x_len)
                row_s = b
                row_n = y_len - 1 - b
                out = out.at[:, row_s, :].set((1 - args.pin) * out[:, row_s, :] + args.pin * sstrip)
                out = out.at[:, row_n, :].set((1 - args.pin) * out[:, row_n, :] + args.pin * nstrip)
        return out

    def patched_apply(state, lead_seconds, dt_s, config=BA.DEFAULT_BOUNDARY_CONFIG):
        st = prod_apply(state, lead_seconds, dt_s, config)
        u = strong_normal("u", st.u, st.u_bdy, lead_seconds, dt_s, config)
        v = strong_normal("v", st.v, st.v_bdy, lead_seconds, dt_s, config)
        return st.replace(u=u, v=v)

    BA.apply_lateral_boundaries = patched_apply
    # operational_mode imported the symbol; patch there too.
    import gpuwrf.dynamics.core.coupled as CC
    CC.apply_lateral_boundaries = patched_apply

    cfg = DailyPipelineConfig(
        run_id="20260509_18z_l2_72h_20260511T190519Z",
        run_root=Path("/mnt/data/canairy_meteo/runs/wrf_l2"), domain="d02",
        dt_s=10.0, acoustic_substeps=10, radiation_cadence_steps=180)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=True, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    fs = run_forecast_operational_segmented(case.state, nl, float(args.lead_h), segment_steps=180)
    jax.block_until_ready(fs.theta)
    v = np.array(fs.v); u = np.array(fs.u)
    print(f"[pin={args.pin} lead={args.lead_h}h]")
    print(" staggered v[z=0] S rows0-4:", np.round(v[0, :5, :].mean(axis=1), 2))
    print(" staggered v[z=0] N rows-5..-1:", np.round(v[0, -5:, :].mean(axis=1), 2))
    print(" staggered u[z=0] W cols0-4:", np.round(u[0, :, :5].mean(axis=0), 2))
    print(" staggered u[z=0] E cols-5..-1:", np.round(u[0, :, -5:].mean(axis=0), 2))
    print(" finite:", bool(np.isfinite(v).all() and np.isfinite(u).all()),
          " v|max|=%.2f u|max|=%.2f theta_finite=%s"
          % (np.abs(v).max(), np.abs(u).max(), bool(np.isfinite(np.array(fs.theta)).all())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
