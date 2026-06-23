#!/usr/bin/env python3
"""v020_cfl_probe.py — per-domain realized-CFL probe for the v0.20.0 P4 timestep work.

Reads a warm wrfout snapshot per domain (d01..dN) and the namelist, and computes the
REALIZED Courant numbers at the configured dt / acoustic-substep cadence:

  * horizontal advective CFL  = max(|U|/dx, |V|/dy) * dt
  * vertical advective CFL    = max(|W|) * dt / min(dz)        (terrain-induced w)
  * horizontal acoustic CFL   = c_s * dt_acoustic / dx,  dt_acoustic = dt / n_sound
                                (c_s = sqrt(gamma * Rd * T), gamma=1.4)
  * boundary-ring vs interior split for each, so steep-terrain edge erosion is visible

This is the T0 "per-domain CFL headroom probe" from OPUS_P34_PLAN §4.2 / the roadmap
§8.3 landmine list. It is a PREDICTOR for the P4 dt/n_sound ladder: it says how much
headroom each domain has BEFORE any expensive forecast, and which (steep 1 km) nest
binds first.

PURE numpy + netCDF4 — NO GPU, NO gpuwrf import. Safe to run anytime (it only reads
wrfout files). It is invoked by scripts/v020_s0_instrument.sh as the CPU post-step of
the warm run, but also stands alone on any existing wrfout set (which is how the
CPU-dry-run validates it without the GPU).

Usage:
  python v020_cfl_probe.py --run-dir DIR [--namelist FILE] [--max-dom N]
                           [--time LAST|FIRST|<index>] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover - dependency guard
    print(f"v020_cfl_probe: netCDF4 import failed: {exc}", file=sys.stderr)
    sys.exit(3)

GAMMA = 1.4
RD = 287.0
G = 9.81
RING = 5  # boundary-ring width (cells) for the edge/interior split


# --------------------------------------------------------------------------- #
# namelist parsing (tiny, dependency-free: just the fields P4 needs)          #
# --------------------------------------------------------------------------- #
def _nml_list(text: str, key: str) -> list[float] | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n/]+)", text, re.M | re.I)
    if not m:
        return None
    vals = []
    for tok in m.group(1).replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals or None


# nested_pipeline.py hardcodes acoustic_substeps=10 in the nested path; mirror that
# as the default when the namelist omits the field (the all-7 namelist does).
DEFAULT_ACOUSTIC_SUBSTEPS = 10


def parse_namelist(path: Path) -> dict:
    text = path.read_text(errors="replace")
    ts = _nml_list(text, "time_step")
    dx = _nml_list(text, "dx")
    dy = _nml_list(text, "dy")
    ratio = _nml_list(text, "parent_time_step_ratio")
    parent_id = _nml_list(text, "parent_id")
    nsound = _nml_list(text, "acoustic_substeps") or _nml_list(text, "time_step_sound")
    maxd = _nml_list(text, "max_dom")
    return {
        "time_step": ts[0] if ts else None,
        "dx": dx,
        "dy": dy,
        "parent_time_step_ratio": ratio,
        "parent_id": [int(p) for p in parent_id] if parent_id else None,
        "acoustic_substeps": (int(nsound[0]) if nsound else DEFAULT_ACOUSTIC_SUBSTEPS),
        "acoustic_substeps_source": ("namelist" if nsound else "runtime_default_10"),
        "max_dom": int(maxd[0]) if maxd else None,
    }


def domain_dt(nml: dict, dom_idx: int) -> float | None:
    """dt for domain dom_idx (1-based), walking the WRF parent_id chain.

    Each child dt = parent_dt / parent_time_step_ratio[child]. WRF nests form a TREE
    (parent_id), not a serial chain, so a star of d03..d09 all parented to d02 each get
    dt = d02_dt / their ratio — NOT the root dt divided by every preceding domain. Walk
    the parent_id chain to the root.
    """
    ts = nml.get("time_step")
    if ts is None:
        return None
    ratios = nml.get("parent_time_step_ratio") or []
    parents = nml.get("parent_id")
    # serial fallback if parent_id absent
    if not parents:
        dt = float(ts)
        for k in range(1, dom_idx):
            if k < len(ratios):
                dt /= max(1.0, ratios[k])
        return dt
    # walk parent_id (1-based) up to the root, dividing by each level's ratio
    dt = float(ts)
    cur = dom_idx
    guard = 0
    while cur >= 1 and guard < len(parents) + 2:
        guard += 1
        idx0 = cur - 1
        if idx0 < 0 or idx0 >= len(parents):
            break
        parent = parents[idx0]
        if parent == cur or parent < 1:  # root (parent_id==self or 1 at root)
            break
        r = ratios[idx0] if idx0 < len(ratios) else 1.0
        dt /= max(1.0, r)
        cur = parent
    return dt


# --------------------------------------------------------------------------- #
# field reading + CFL math                                                     #
# --------------------------------------------------------------------------- #
def _read(ds: Dataset, name: str, t_idx: int) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    v = ds.variables[name]
    arr = v[t_idx] if "Time" in v.dimensions else v[:]
    return np.asarray(arr, dtype=np.float64)


def _unstagger(a: np.ndarray, axis: int) -> np.ndarray:
    sl_lo = [slice(None)] * a.ndim
    sl_hi = [slice(None)] * a.ndim
    sl_lo[axis] = slice(0, -1)
    sl_hi[axis] = slice(1, None)
    return 0.5 * (a[tuple(sl_lo)] + a[tuple(sl_hi)])


def _ring_split(field2d_or_3d: np.ndarray) -> tuple[float, float]:
    """Return (interior_max, boundary_ring_max) of |field| over the horizontal plane.

    Works on (z,y,x) or (y,x); reduces the vertical first.
    """
    a = np.abs(field2d_or_3d)
    if a.ndim == 3:
        a = a.max(axis=0)
    ny, nx = a.shape
    interior = a[RING : ny - RING, RING : nx - RING] if ny > 2 * RING and nx > 2 * RING else a
    ring_mask = np.ones_like(a, dtype=bool)
    if ny > 2 * RING and nx > 2 * RING:
        ring_mask[RING : ny - RING, RING : nx - RING] = False
    ring = a[ring_mask]
    return (float(interior.max()) if interior.size else 0.0,
            float(ring.max()) if ring.size else 0.0)


def column_dz(ph: np.ndarray | None, phb: np.ndarray | None) -> float:
    """Min layer thickness (m) from geopotential (PH+PHB)/g; fallback to a typical 100 m."""
    if ph is None or phb is None:
        return 100.0
    z = (ph + phb) / G  # (z_stag, y, x)
    dz = np.diff(z, axis=0)
    dz = dz[np.isfinite(dz) & (dz > 0)]
    return float(dz.min()) if dz.size else 100.0


def probe_domain(path: Path, dt: float, dx: float, dy: float, n_sound: int | None,
                 t_idx: int) -> dict:
    ds = Dataset(str(path))
    try:
        U = _read(ds, "U", t_idx)  # (z, y, x_stag)
        V = _read(ds, "V", t_idx)  # (z, y_stag, x)
        W = _read(ds, "W", t_idx)  # (z_stag, y, x)
        T = _read(ds, "T", t_idx)  # perturbation potential temperature
        P = _read(ds, "P", t_idx)
        PB = _read(ds, "PB", t_idx)
        PH = _read(ds, "PH", t_idx)
        PHB = _read(ds, "PHB", t_idx)
    finally:
        ds.close()

    out: dict = {"dt_s": dt, "dx_m": dx, "dy_m": dy, "n_sound": n_sound}

    # horizontal advective CFL
    if U is not None and V is not None:
        Uc = _unstagger(U, axis=2)
        Vc = _unstagger(V, axis=1)
        u_int, u_ring = _ring_split(Uc)
        v_int, v_ring = _ring_split(Vc)
        cfl_h_int = max(u_int / dx, v_int / dy) * dt
        cfl_h_ring = max(u_ring / dx, v_ring / dy) * dt
        out["max_wind_ms"] = {"interior": max(u_int, v_int), "ring": max(u_ring, v_ring)}
        out["cfl_horiz_advective"] = {"interior": cfl_h_int, "ring": cfl_h_ring}

    # vertical advective CFL
    if W is not None:
        dz_min = column_dz(PH, PHB)
        w_int, w_ring = _ring_split(W)
        out["min_dz_m"] = dz_min
        out["max_w_ms"] = {"interior": w_int, "ring": w_ring}
        out["cfl_vert_advective"] = {
            "interior": w_int * dt / dz_min,
            "ring": w_ring * dt / dz_min,
        }

    # acoustic CFL (sound speed from absolute temperature)
    if T is not None and P is not None and PB is not None and n_sound:
        theta = T + 300.0  # WRF stores theta perturbation about 300 K
        p_abs = P + PB
        # absolute T from potential temperature: T = theta * (p/p0)^(Rd/cp)
        T_abs = theta * (p_abs / 1.0e5) ** (RD / 1004.0)
        c_s = np.sqrt(GAMMA * RD * np.clip(T_abs, 150.0, 350.0))
        dt_ac = dt / max(1, n_sound)
        cmax = float(np.nanmax(c_s))
        out["sound_speed_ms"] = cmax
        out["dt_acoustic_s"] = dt_ac
        out["cfl_acoustic_horiz"] = cmax * dt_ac / dx

    return out


def headroom_verdict(d: dict) -> dict:
    """Express remaining margin to the standard stability limits (advective CFL<1,
    acoustic split-explicit tolerates ~0.5-1.0). Reports the factor by which dt
    could rise before each limit binds — the P4 ladder's predictor."""
    v = {}
    ch = d.get("cfl_horiz_advective", {})
    if ch:
        worst = max(ch.get("interior", 0.0), ch.get("ring", 0.0))
        v["horiz_advective_dt_headroom_x"] = (1.0 / worst) if worst > 0 else float("inf")
    cv = d.get("cfl_vert_advective", {})
    if cv:
        worst = max(cv.get("interior", 0.0), cv.get("ring", 0.0))
        v["vert_advective_dt_headroom_x"] = (1.0 / worst) if worst > 0 else float("inf")
    ca = d.get("cfl_acoustic_horiz")
    if ca:
        # acoustic dt can rise to CFL~0.5 (conservative split-explicit target)
        v["acoustic_dt_headroom_x"] = (0.5 / ca) if ca > 0 else float("inf")
    return v


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def discover(run_dir: Path, dom: str) -> Path | None:
    cands = sorted(run_dir.glob(f"wrfout_{dom}_*"))
    return cands[-1] if cands else None


def resolve_time_index(path: Path, spec: str) -> int:
    ds = Dataset(str(path))
    try:
        nt = ds.dimensions["Time"].size if "Time" in ds.dimensions else 1
    finally:
        ds.close()
    if spec.upper() == "LAST":
        return nt - 1
    if spec.upper() == "FIRST":
        return 0
    return max(0, min(int(spec), nt - 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="directory with wrfout_d0N_* files (a warm GPU output dir)")
    ap.add_argument("--namelist", type=Path, default=None,
                    help="namelist.input (defaults to <run-dir>/namelist.input)")
    ap.add_argument("--max-dom", type=int, default=None)
    ap.add_argument("--time", default="LAST", help="LAST|FIRST|<index>")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    nml_path = args.namelist or (run_dir / "namelist.input")
    if not nml_path.is_file():
        print(f"v020_cfl_probe: namelist not found: {nml_path}", file=sys.stderr)
        return 2
    nml = parse_namelist(nml_path)
    max_dom = args.max_dom or nml.get("max_dom") or 1

    report = {"run_dir": str(run_dir), "namelist": str(nml_path), "max_dom": max_dom,
              "config": nml, "domains": {}}
    found = 0
    for i in range(1, max_dom + 1):
        dom = f"d{i:02d}"
        path = discover(run_dir, dom)
        dx = (nml["dx"][i - 1] if nml.get("dx") and i - 1 < len(nml["dx"]) else None)
        dy = (nml["dy"][i - 1] if nml.get("dy") and i - 1 < len(nml["dy"]) else dx)
        dt = domain_dt(nml, i)
        if path is None or dx is None or dt is None:
            report["domains"][dom] = {"status": "MISSING",
                                      "wrfout": str(path) if path else None,
                                      "dx_m": dx, "dt_s": dt}
            continue
        t_idx = resolve_time_index(path, args.time)
        dd = probe_domain(path, dt=dt, dx=dx, dy=dy,
                          n_sound=nml.get("acoustic_substeps"), t_idx=t_idx)
        dd["status"] = "OK"
        dd["wrfout"] = str(path)
        dd["time_index"] = t_idx
        dd["headroom"] = headroom_verdict(dd)
        report["domains"][dom] = dd
        found += 1

    # binding (most-constrained) domain across all advective+acoustic CFLs
    binding = None
    worst_head = float("inf")
    for dom, dd in report["domains"].items():
        if dd.get("status") != "OK":
            continue
        for hk, hv in dd.get("headroom", {}).items():
            if hv < worst_head:
                worst_head = hv
                binding = {"domain": dom, "constraint": hk, "dt_headroom_x": hv}
    report["binding_constraint"] = binding
    report["n_domains_probed"] = found

    payload = json.dumps(report, indent=2, default=lambda o: None) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)

    # human summary to stderr
    print("\n=== per-domain CFL headroom (dt-multiplier until each limit binds) ===",
          file=sys.stderr)
    for dom, dd in report["domains"].items():
        if dd.get("status") != "OK":
            print(f"  {dom}: {dd.get('status')}", file=sys.stderr)
            continue
        h = dd.get("headroom", {})
        ch = dd.get("cfl_horiz_advective", {})
        ca = dd.get("cfl_acoustic_horiz")
        print(f"  {dom}: dt={dd['dt_s']:.3f}s dx={dd['dx_m']:.0f}m "
              f"CFL_h={max(ch.get('interior',0),ch.get('ring',0)):.3f} "
              f"CFL_ac={ca if ca is not None else float('nan'):.3f} "
              f"-> dt_headroom: h={h.get('horiz_advective_dt_headroom_x',float('nan')):.2f}x "
              f"v={h.get('vert_advective_dt_headroom_x',float('nan')):.2f}x "
              f"ac={h.get('acoustic_dt_headroom_x',float('nan')):.2f}x", file=sys.stderr)
    if binding:
        print(f"  BINDING: {binding['domain']} via {binding['constraint']} "
              f"= {binding['dt_headroom_x']:.2f}x dt headroom", file=sys.stderr)
    return 0 if found > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
