#!/usr/bin/env python
"""V0.14 venting root-cause: per-boundary-distance NORMAL-wind bias oracle.

The depth-8 interior control-surface excess outflux (-26.5 Pa/cell/h at h37) is
== a coherent normal-wind bias at the four depth-8 faces.  This oracle localises
WHERE that bias is born by measuring, hour-by-hour (h36/h37/h38), the GPU-vs-CPU
NORMAL-wind difference and the per-face flux contribution as a function of
boundary distance ``b_dist`` (0 = outermost row/column, 1..3 = relax zone,
8 = control surface, deeper = free interior).

NORMAL component per face:
  * W face: u at column b_dist            (x-normal)
  * E face: u at column nx_u-1-b_dist
  * S face: v at row    b_dist            (y-normal)
  * N face: v at row    ny_v-1-b_dist

For each (hour, face, b_dist) we report:
  * gpu_minus_cpu normal-wind mean (m/s) over the tangential extent (the bias)
  * the depth-8-equivalent flux contribution of that ONE face strip
    (same kernel as switzerland_hpg_native_face_fix.budget_between.outflux),
    GPU and CPU, and their difference (Pa/cell/h equiv).

Pure netCDF/CPU.  No GPU, no JAX.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_HPG = importlib.util.spec_from_file_location(
    "hpg", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG)
_HPG.loader.exec_module(hpg)  # type: ignore[union-attr]

CPU = hpg.CPU
PROBE = hpg.PROBE_ROOT
GPU = PROBE / "gpu_output_phys_tendf"  # the current production-config forecast
OUT_JSON = ROOT / "proofs/v014/switzerland_bdy_normal_wind_oracle.json"

DEPTH = 8
MAXB = 16  # how many b_dist strips to probe


def _state(base: Path, hour: int):
    s = hpg.load_budget_state(base, hour)
    return s


def normal_bias_and_flux(cpu_s, gpu_s, depth: int = DEPTH):
    """For each face and b_dist, the GPU-CPU normal-wind mean bias + the
    per-face flux at THAT b_dist (using the depth-8 tangential window)."""
    c1 = cpu_s["c1h"]
    c2 = cpu_s["c2h"]
    wk = -cpu_s["dnw"]
    ny, nx = cpu_s["mu"].shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth
    ncell = (j1 - j0) * (i1 - i0)
    dx = cpu_s["dx"]
    nz = wk.shape[0]

    def face_flux(s, face: str, b: int) -> float:
        mu, u, v = s["mu"], s["u"], s["v"]
        if face in ("W", "E"):
            col = b if face == "W" else (nx - 1 - b)  # mass-cell column index for u-face col
            # u index for the face at mass-cell column `col` on W side is `col`;
            # on E side the budget uses i1 = nx-depth as the u index.  Keep it
            # simple+consistent: flux through the u-face at u-column index `uidx`.
            uidx = b if face == "W" else (u.shape[2] - 1 - b)
            # muf at this u-face = 0.5*(mu[col-1]+mu[col]) with col=uidx; clamp.
            cm = min(max(uidx, 1), nx - 1)
            muf = 0.5 * (mu[j0:j1, cm - 1] + mu[j0:j1, cm])
            mul = c1[:, None] * muf[None, :] + c2[:, None]
            f = (u[:, j0:j1, uidx] * mul * wk[:, None] / s["muy"][j0:j1, uidx][None, :]).sum()
            return float(f)
        else:
            vidx = b if face == "S" else (v.shape[1] - 1 - b)
            rm = min(max(vidx, 1), ny - 1)
            muf = 0.5 * (mu[rm - 1, i0:i1] + mu[rm, i0:i1])
            mul = c1[:, None] * muf[None, :] + c2[:, None]
            f = (v[:, vidx, i0:i1] * mul * wk[:, None] / s["mvx"][vidx, i0:i1][None, :]).sum()
            return float(f)

    def normal_mean(s, face: str, b: int) -> float:
        """tangential+vertical mean of the NORMAL wind at this strip."""
        u, v = s["u"], s["v"]
        if face in ("W", "E"):
            uidx = b if face == "W" else (u.shape[2] - 1 - b)
            return float(u[:, j0:j1, uidx].mean())
        else:
            vidx = b if face == "S" else (v.shape[1] - 1 - b)
            return float(v[:, vidx, i0:i1].mean())

    rows = []
    for face in ("W", "E", "S", "N"):
        for b in range(MAXB):
            cw = normal_mean(cpu_s, face, b)
            gw = normal_mean(gpu_s, face, b)
            cf = face_flux(cpu_s, face, b)
            gf = face_flux(gpu_s, face, b)
            # convert a single-face flux to Pa/cell/h equiv (matches budget scale)
            def topa(fv: float) -> float:
                return fv * 3600.0 / dx / ncell
            rows.append({
                "face": face,
                "b_dist": b,
                "cpu_normal_mean": cw,
                "gpu_normal_mean": gw,
                "gpu_minus_cpu_normal_mean": gw - cw,
                "cpu_flux_pa_cell_h": topa(cf),
                "gpu_flux_pa_cell_h": topa(gf),
                "flux_excess_pa_cell_h": topa(gf - cf),
            })
    return rows, ncell


def main() -> int:
    out = {
        "schema": "v014_switzerland_bdy_normal_wind_oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": str(CPU),
        "gpu": str(GPU),
        "depth": DEPTH,
        "note": (
            "Sign convention: flux_excess is the GPU-minus-CPU single-face mass "
            "flux in budget units; net_influx = -(fe-fw + fn-fs). At the W/S "
            "faces, positive normal wind = INFLOW; at E/N, positive = OUTFLOW. A "
            "coherent OUTWARD normal-wind bias on all four faces == venting."
        ),
    }
    for hour in (36, 37, 38):
        if not hpg.fn(GPU, hour).exists() or not hpg.fn(CPU, hour).exists():
            out[f"h{hour}"] = {"available": False}
            continue
        cpu_s = _state(CPU, hour)
        gpu_s = _state(GPU, hour)
        rows, ncell = normal_bias_and_flux(cpu_s, gpu_s)
        out[f"h{hour}"] = {"available": True, "ncell": ncell, "rows": rows}
        # summary: control-surface (b=DEPTH) net flux excess and the OUTWARD
        # normal-wind bias averaged over the four control-surface faces.
        cs = {r["face"]: r for r in rows if r["b_dist"] == DEPTH}
        # outward = +E +N -W -S
        outward_bias = (
            cs["E"]["gpu_minus_cpu_normal_mean"]
            + cs["N"]["gpu_minus_cpu_normal_mean"]
            - cs["W"]["gpu_minus_cpu_normal_mean"]
            - cs["S"]["gpu_minus_cpu_normal_mean"]
        ) / 4.0
        net_flux_excess = (
            cs["E"]["flux_excess_pa_cell_h"] - cs["W"]["flux_excess_pa_cell_h"]
            + cs["N"]["flux_excess_pa_cell_h"] - cs["S"]["flux_excess_pa_cell_h"]
        )
        out[f"h{hour}"]["control_surface_summary"] = {
            "mean_outward_normal_wind_bias_mps": outward_bias,
            "net_outflux_excess_pa_cell_h": net_flux_excess,
            "per_face_outward_bias": {
                "W_inward_is_neg": -cs["W"]["gpu_minus_cpu_normal_mean"],
                "E": cs["E"]["gpu_minus_cpu_normal_mean"],
                "S_inward_is_neg": -cs["S"]["gpu_minus_cpu_normal_mean"],
                "N": cs["N"]["gpu_minus_cpu_normal_mean"],
            },
        }
    OUT_JSON.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    # console digest
    for hour in (36, 37, 38):
        h = out.get(f"h{hour}", {})
        if not h.get("available"):
            continue
        s = h["control_surface_summary"]
        print(f"h{hour}: mean_outward_normal_wind_bias = {s['mean_outward_normal_wind_bias_mps']:+.4f} m/s, "
              f"net_outflux_excess = {s['net_outflux_excess_pa_cell_h']:+.2f} Pa/cell/h")
        print(f"   per-face outward bias (m/s): "
              f"W={s['per_face_outward_bias']['W_inward_is_neg']:+.4f} "
              f"E={s['per_face_outward_bias']['E']:+.4f} "
              f"S={s['per_face_outward_bias']['S_inward_is_neg']:+.4f} "
              f"N={s['per_face_outward_bias']['N']:+.4f}")
        # b_dist profile of mean normal bias per face
        for face in ("W", "E", "S", "N"):
            prof = [r["gpu_minus_cpu_normal_mean"] for r in h["rows"] if r["face"] == face][:MAXB]
            sprof = " ".join(f"{v:+.3f}" for v in prof)
            print(f"   {face} bias by b_dist 0..{MAXB-1}: {sprof}")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
