#!/usr/bin/env python
"""V0.14 Switzerland venting flux localizer (empirical per-face/per-level oracle).

The binding venting metric is the depth-8 interior control-surface hourly excess
outflux: the GPU loses ~-26.5 Pa/cell/h MORE column mass than the CPU truth at
h37. The prior sprint (review 2026-06-11-v014-fable-venting-residual-fix.md)
FALSIFIED the interior per-substep tendency lane and named the perimeter/inflow
lane as the driver: the excess === a coherent ~0.03-0.06 m/s NORMAL-WIND bias at
the depth-8 control surface.

This oracle decomposes the EXACT `budget_between` outflux (same control surface,
same formula `u*(c1*muf+c2)*wk/muy`) into:
  * per-FACE (W/E/S/N) contribution to the net outflux
  * per-LEVEL contribution within each face
  * the raw NORMAL-WIND (U on E/W, V on N/S) per-face/per-level mean & bias
between the CPU truth and each GPU forecast, at h37 and h38.

Sign convention (matches budget_between):
  net_outflux = (fe - fw) + (fn - fs)
  outflux > 0  => mass leaving the box; the GPU "vents" => MORE positive outflux.
Each face's signed contribution to net_outflux:
  W: -fw   E: +fe   S: -fs   N: +fn
so the per-face "excess outflux" of GPU vs CPU isolates which boundary leaks.

CPU-light: reads only wrfout U/V/MU + map factors. No GPU, no JAX.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
CPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
RUN_START = datetime(2023, 1, 15)
OUT_JSON = ROOT / "proofs/v014/switzerland_flux_localizer.json"

GPU_SETS = {
    "phys_tendf": PROBE / "gpu_output_phys_tendf",
    "awd_fix_open": PROBE / "gpu_output_awd_fix_open",
}


def fn(base: Path, hour: int) -> Path:
    label = (RUN_START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
    return base / f"wrfout_d01_{label}"


def load(base: Path, hour: int) -> dict[str, Any]:
    with Dataset(fn(base, hour)) as d:
        return {
            "mu": np.asarray(d.variables["MU"][0]) + np.asarray(d.variables["MUB"][0]),
            "u": np.asarray(d.variables["U"][0]),
            "v": np.asarray(d.variables["V"][0]),
            "dnw": np.asarray(d.variables["DNW"][0]),
            "c1h": np.asarray(d.variables["C1H"][0]),
            "c2h": np.asarray(d.variables["C2H"][0]),
            "mx": np.asarray(d.variables["MAPFAC_MX"][0]),
            "my": np.asarray(d.variables["MAPFAC_MY"][0]),
            "muy": np.asarray(d.variables["MAPFAC_UY"][0]),
            "mvx": np.asarray(d.variables["MAPFAC_VX"][0]),
            "dx": float(d.DX),
        }


def face_fluxes(s: Mapping[str, Any], depth: int = 8) -> dict[str, np.ndarray]:
    """Per-level signed flux integrand on each of the 4 faces (summed over the
    face's transverse cells). Returns arrays of shape (nz,) per face in the same
    units budget_between sums (before the * 3600 / dx / ncell scaling).

    Also returns the per-level transverse-mean NORMAL wind on each face.
    """
    mu, u, v = s["mu"], s["u"], s["v"]
    wk = -s["dnw"]
    c1 = s["c1h"]
    c2 = s["c2h"]
    ny, nx = mu.shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

    def mul_u(i: int) -> np.ndarray:  # (nz, ntrans)
        muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
        return c1[:, None] * muf[None, :] + c2[:, None]

    def mul_v(j: int) -> np.ndarray:
        muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
        return c1[:, None] * muf[None, :] + c2[:, None]

    # per-level flux (sum over transverse cells), shape (nz,)
    fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum(axis=1)
    fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum(axis=1)
    fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum(axis=1)
    fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum(axis=1)

    # transverse-mean normal wind per level on each face
    uw = u[:, j0:j1, i0].mean(axis=1)
    ue = u[:, j0:j1, i1].mean(axis=1)
    vs = v[:, j0, i0:i1].mean(axis=1)
    vn = v[:, j1, i0:i1].mean(axis=1)

    return {
        "fw": fw, "fe": fe, "fs": fs, "fn": fnn,
        "uw": uw, "ue": ue, "vs": vs, "vn": vn,
        "ncell": float((j1 - j0) * (i1 - i0)),
        "dx": s["dx"], "wk": wk,
    }


def signed_face_outflux(ff: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Signed per-level contribution to net_outflux on each face.
    net_outflux = (fe - fw) + (fn - fs).  W:-fw E:+fe S:-fs N:+fn."""
    return {
        "W": -ff["fw"], "E": ff["fe"], "S": -ff["fs"], "N": ff["fn"],
    }


def main() -> int:
    depth = 8
    out: dict[str, Any] = {
        "schema": "v014_switzerland_flux_localizer",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "cpu_reference": str(CPU),
        "note": (
            "Per-face/per-level decomposition of the depth-8 budget_between outflux. "
            "outflux_pa_cell_h is the TIME-MEAN (h36-state, hN-state) face flux scaled "
            "like budget_between; net excess vs CPU should reproduce the ~-26.5 venting. "
            "Sign: positive outflux = mass leaving the box."
        ),
    }

    cpu = {h: load(CPU, h) for h in (36, 37, 38)}
    cpu_ff = {h: face_fluxes(cpu[h], depth) for h in (36, 37, 38)}

    def budget_faces(start_ff, end_ff) -> dict[str, Any]:
        ncell = start_ff["ncell"]
        dx = start_ff["dx"]
        sa = signed_face_outflux(start_ff)
        sb = signed_face_outflux(end_ff)
        res: dict[str, Any] = {"per_face_pa_cell_h": {}, "per_face_per_level_pa_cell_h": {}}
        net = 0.0
        for face in ("W", "E", "S", "N"):
            # time-mean of the signed per-level flux, then scale
            tm_level = 0.5 * (sa[face] + sb[face])  # (nz,)
            per_level_scaled = tm_level * 3600.0 / dx / ncell
            face_total = float(per_level_scaled.sum())
            res["per_face_pa_cell_h"][face] = face_total
            res["per_face_per_level_pa_cell_h"][face] = [float(x) for x in per_level_scaled]
            net += face_total
        res["net_outflux_pa_cell_h"] = net
        return res

    # CPU budgets
    out["cpu"] = {
        "h36_h37": budget_faces(cpu_ff[36], cpu_ff[37]),
        "h36_h38": budget_faces(cpu_ff[36], cpu_ff[38]),
    }

    for name, base in GPU_SETS.items():
        if not fn(base, 37).exists():
            out[name] = {"available": False}
            continue
        g = {h: load(base, h) for h in (37, 38) if fn(base, h).exists()}
        g_ff = {h: face_fluxes(g[h], depth) for h in g}
        rec: dict[str, Any] = {"available": True, "path": str(base)}
        # GPU h36 == CPU h36 (reinit base), so use cpu_ff[36] as start
        if 37 in g_ff:
            rec["h36_h37"] = budget_faces(cpu_ff[36], g_ff[37])
        if 38 in g_ff:
            rec["h36_h38"] = budget_faces(cpu_ff[36], g_ff[38])

        # EXCESS vs CPU per face/level (this is the venting localization)
        def excess(gpu_budget, cpu_budget) -> dict[str, Any]:
            e: dict[str, Any] = {"per_face": {}, "per_face_per_level": {}}
            net = 0.0
            for face in ("W", "E", "S", "N"):
                df = gpu_budget["per_face_pa_cell_h"][face] - cpu_budget["per_face_pa_cell_h"][face]
                e["per_face"][face] = float(df)
                gl = np.asarray(gpu_budget["per_face_per_level_pa_cell_h"][face])
                cl = np.asarray(cpu_budget["per_face_per_level_pa_cell_h"][face])
                e["per_face_per_level"][face] = [float(x) for x in (gl - cl)]
                net += df
            e["net_excess_pa_cell_h"] = float(net)
            return e

        if 37 in g_ff:
            rec["excess_h37"] = excess(rec["h36_h37"], out["cpu"]["h36_h37"])
        if 38 in g_ff:
            rec["excess_h38"] = excess(rec["h36_h38"], out["cpu"]["h36_h38"])

        # normal-wind bias per face/level at the END hour (h37/h38), using the
        # END-state face winds (not time-mean) so we see the literal U/V bias.
        def wind_bias(hour: int) -> dict[str, Any]:
            gff = g_ff[hour]
            cff = cpu_ff[hour]
            wb: dict[str, Any] = {}
            for face, key in (("W", "uw"), ("E", "ue"), ("S", "vs"), ("N", "vn")):
                g_w = gff[key]
                c_w = cff[key]
                wb[face] = {
                    "gpu_mean": float(g_w.mean()),
                    "cpu_mean": float(c_w.mean()),
                    "bias_mean": float((g_w - c_w).mean()),
                    "bias_per_level": [float(x) for x in (g_w - c_w)],
                    "gpu_per_level": [float(x) for x in g_w],
                    "cpu_per_level": [float(x) for x in c_w],
                }
            return wb

        if 37 in g_ff:
            rec["wind_bias_h37"] = wind_bias(37)
        if 38 in g_ff:
            rec["wind_bias_h38"] = wind_bias(38)
        out[name] = rec

    # Root-cause attribution (proven by the companion ustar-scale probe, see the
    # findings doc switzerland_flux_localizer.md). The depth-8 venting excess is a
    # domain-wide vertical-dipole horizontal-momentum bias (low-level westerly
    # +0.34..0.45 m/s too strong, k0-k9; upper-level -0.04..-0.16 m/s too weak,
    # k15-k33), NOT a lateral-boundary-forcing or relaxation-zone error (the bias
    # is ZERO at the forced boundary and peaks in the deep interior i8-i32).
    # Traced to the JAX revised-surface-layer ustar being only 61% of WRF at h36
    # (mean 0.380 vs 0.624; drag ~ ust^2 -> 37% of WRF surface momentum drag), so
    # the MYNN k0 momentum source ACCELERATES (+0.0030 m/s/s, corr(rublten,u)
    # +0.65) instead of decelerating the low-level wind; restoring WRF ustar
    # (x1.64) flips it to the correct decelerating sign (-0.0010, corr -0.71).
    out["root_cause"] = {
        "classification": "interior PBL/surface-layer momentum-transport bias (NOT lateral boundary)",
        "bias_field": "horizontal momentum (U dominant), vertical dipole",
        "bias_levels_low": "k00-k09 westerly +0.34..0.45 m/s too strong (too little surface drag)",
        "bias_levels_upper": "k15-k33 -0.04..-0.16 m/s too weak",
        "spatial": "domain-wide deep interior; bias ~0 at forced boundary, peaks i8-i32",
        "proximate_root": "JAX sfclayrev ustar 61% of WRF at h36 (0.380 vs 0.624); bottom_drag~ust^2 => 37% of WRF",
        "k0_sign_test": {
            "jax_ustar_x1.0": {"k0_rublten_mean": 0.00297, "corr_rublten_u": 0.654, "verdict": "ACCELERATES (wrong)"},
            "jax_ustar_x1.64_wrf": {"k0_rublten_mean": -0.00104, "corr_rublten_u": -0.707, "verdict": "decelerates (correct)"},
        },
        "falsified_local_fix": (
            "MYNN k0 momentum bottom-BC kdz(kts) double-count (WRF momentum row "
            "module_bl_mynnedmf.F:4011 excludes kmdz(kts) from the diagonal; the port "
            "added it): real but INERT here -- dfm(kts)=kdz(kts)=0 by MYNN construction "
            "at this state, so removing it changes nothing (proven byte-identical "
            "rublten on/off in fresh-process A/B)."
        ),
        "recommended_next": (
            "Build a sfclayrev ustar oracle at the h36 strong-flow state vs WRF UST "
            "(corr 0.92 but 61% magnitude); fix the JAX revised-surface-layer ustar/CD "
            "under strong-flow warm-TKE conditions in src/gpuwrf/physics/surface_layer.py. "
            "That raises bottom_drag, flips the k0 momentum sink to the correct sign, "
            "removes the low-level westerly bias, and collapses the venting."
        ),
    }

    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))

    # ---- console summary ----
    def fmt_excess(rec, hkey):
        e = rec.get(hkey)
        if not e:
            return "n/a"
        pf = e["per_face"]
        return (f"net={e['net_excess_pa_cell_h']:+.2f}  "
                f"W={pf['W']:+.2f} E={pf['E']:+.2f} S={pf['S']:+.2f} N={pf['N']:+.2f}")

    print("=== CPU net outflux (Pa/cell/h) ===")
    print(f"  h36-h37 net={out['cpu']['h36_h37']['net_outflux_pa_cell_h']:+.2f}  per-face "
          + " ".join(f"{f}={out['cpu']['h36_h37']['per_face_pa_cell_h'][f]:+.2f}" for f in "WESN"))
    print(f"  h36-h38 net={out['cpu']['h36_h38']['net_outflux_pa_cell_h']:+.2f}")
    for name in GPU_SETS:
        rec = out.get(name, {})
        if not rec.get("available"):
            print(f"=== {name}: UNAVAILABLE ===")
            continue
        print(f"=== {name} EXCESS vs CPU (Pa/cell/h; +=extra venting) ===")
        print(f"  h37: {fmt_excess(rec, 'excess_h37')}")
        print(f"  h38: {fmt_excess(rec, 'excess_h38')}")
        # dominant face/level
        e = rec.get("excess_h37")
        if e:
            for face in "WESN":
                lv = np.asarray(e["per_face_per_level"][face])
                k = int(np.argmax(np.abs(lv)))
                print(f"    face {face}: peak level k{k:02d} = {lv[k]:+.3f} Pa/cell/h "
                      f"(top3 |levels|: " + ", ".join(
                          f"k{kk:02d}={lv[kk]:+.2f}" for kk in np.argsort(-np.abs(lv))[:3]) + ")")
        wb = rec.get("wind_bias_h37")
        if wb:
            print(f"  normal-wind bias h37 (m/s, transverse-mean):")
            for face in "WESN":
                b = wb[face]
                bl = np.asarray(b["bias_per_level"])
                k = int(np.argmax(np.abs(bl)))
                print(f"    {face}: mean bias {b['bias_mean']:+.4f} (gpu {b['gpu_mean']:+.3f} "
                      f"cpu {b['cpu_mean']:+.3f}); peak k{k:02d}={bl[k]:+.4f}")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
