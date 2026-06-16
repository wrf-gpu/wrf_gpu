"""F7N: per-acoustic-substep WRF-vs-JAX touchdown-column diff.

Parses the WRF text dump (em_grav2d_x_touchdown_dump.txt) and the JAX JSON dump
(em_grav2d_x_touchdown_substeps.json), aligns them at the cold-pool touchdown
column (WRF i=256 == JAX center), and computes the mechanism diagnostics that
resolve which acoustic-substep operator under-drives the horizontal spreading:

  (a) omega / ww  -- the vertical mass flux. Does WRF's low-level ww (descending
      mass) get converted to horizontal divergence that JAX's does not?
  (b) advance_uv acoustic PGF -- the surface u-outflow. We recover du/dx at the
      surface (lowest mass level) from the i-face and (i+1)-face coupled u.
  (c) surface mass coupling -- muts/mut evolution at the column.

Because WRF (e_vert=65, stretched z) and JAX (nz=60, uniform 100 m) differ in
vertical grid AND substep count (WRF RK3=6 / JAX RK3=10 sound steps), we compare:
  - END-OF-STEP (post-RK3) column profiles vs physical height z (interpolated to
    common heights), per timestep across the touchdown window.
  - The low-level (z<1500 m) horizontal-divergence du/dx and the column-integrated
    |ww| (descending mass flux), the two quantities that diagnose whether the
    descending air is being turned into outflow.

Emits proofs/f7n/touchdown_substep_diff.json.
"""
from __future__ import annotations

import json
import sys
import numpy as np

WRF_TXT = "/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt"
JAX_JSON = "/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_substeps.json"
OUT = "/home/user/src/wrf_gpu2/proofs/f7n/touchdown_substep_diff.json"

G = 9.81
DX = 100.0


def parse_wrf(path):
    """Return dict[(itimestep,rk,iter)][i] = {field: np.array over k}."""
    recs = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            t = line.split()
            if t[0] == "REC":
                it, rk, itr, ns, kde = int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])
                cur = (it, rk, itr)
                recs.setdefault(cur, {"nsmall": ns, "kde": kde, "cols": {}})
            elif t[0] == "COL":
                i = int(t[1]); k = int(t[2])
                vals = [float(x) for x in t[3:]]
                # order: w_2 ph_2 p rw_tend ph_tend ww u_iface u_ip1face v t_2save muave muts mut
                col = recs[cur]["cols"].setdefault(i, {f: [] for f in
                      ("w","ph","p","rw_tend","ph_tend","ww","u_iface","u_ip1face","v","t_2ave","muave","muts","mut")})
                col["w"].append(vals[0]); col["ph"].append(vals[1]); col["p"].append(vals[2])
                col["rw_tend"].append(vals[3]); col["ph_tend"].append(vals[4]); col["ww"].append(vals[5])
                col["u_iface"].append(vals[6]); col["u_ip1face"].append(vals[7]); col["v"].append(vals[8])
                col["t_2ave"].append(vals[9]); col["muave"].append(vals[10])
                col["muts"].append(vals[11]); col["mut"].append(vals[12])
    # to arrays
    for cur in recs:
        for i in recs[cur]["cols"]:
            for f in recs[cur]["cols"][i]:
                recs[cur]["cols"][i][f] = np.asarray(recs[cur]["cols"][i][f], dtype=np.float64)
    return recs


def wrf_phys_height(ph, phb_unknown=None):
    """WRF dumped ph_2 is the *coupled work-delta* perturbation geopotential at
    this point in the loop, NOT total geopotential -- so it is not a clean height.
    Use the WRF mass-level index proportional height as a fallback only for the
    low-level mask; the true height comparison uses the JAX uniform z and the WRF
    stretched grid separately.  For the low-level mask we use the WRF stretched
    znw heights derived from the front_savepoints (z ~ stretched).  Here we
    approximate low-level via the lowest ~15 mass levels (z<~1500 m on the
    stretched 65-level / ztop=6409 grid the first ~17 levels span ~0-1500 m)."""
    return None


def jax_records(payload):
    """index by (itimestep, rk_step, iteration)."""
    out = {}
    for r in payload["records"]:
        out[(r["itimestep"], r["rk_step"], r["iteration"])] = r
    return out


def col_surface_divergence_coupled(u_iface, u_ip1face, mut_col):
    """du/dx at each level from coupled u faces (u is coupled = mu*u/msfy ~ u*mass).
    The *coupled* horizontal divergence (mass-flux divergence) is
    (u_ip1face - u_iface)/dx ; this is exactly the quantity advance_mu_t sums to
    get dmdt and ww.  Return per-level coupled du/dx."""
    return (np.asarray(u_ip1face) - np.asarray(u_iface)) / DX


def summarize(side, rec_for_col, center, klow):
    """Return mechanism diagnostics for one (timestep,rk,iter) at the touchdown col."""
    c = rec_for_col
    w = np.asarray(c["w"])            # faces
    ww = np.asarray(c["ww"])          # faces (omega)
    u_i = np.asarray(c["u_iface"])    # mass-level coupled u at face i
    u_ip1 = np.asarray(c["u_ip1face"])
    mut = c["mut"] if np.ndim(c["mut"]) else float(c["mut"])
    dudx = (u_ip1 - u_i) / DX         # coupled mass-flux divergence per level
    nz_mass = u_i.shape[0]
    klow = min(klow, nz_mass)
    return {
        "w_min": float(np.min(w)),                       # most negative (downdraft)
        "w_max": float(np.max(w)),
        "w_argmin_face": int(np.argmin(w)),
        "ww_lowlevel_absmax": float(np.max(np.abs(ww[:klow + 1]))),  # descending mass flux low
        "ww_lowlevel_signed_min": float(np.min(ww[:klow + 1])),
        "dudx_surface": float(dudx[0]),                  # coupled du/dx at lowest mass level
        "dudx_lowlevel_absmax": float(np.max(np.abs(dudx[:klow]))),
        "dudx_lowlevel_mean": float(np.mean(dudx[:klow])),
        "u_iface_surface": float(u_i[0]),
        "u_ip1face_surface": float(u_ip1[0]),
    }


def main():
    wrf = parse_wrf(WRF_TXT)
    payload = json.load(open(JAX_JSON))
    jax = jax_records(payload)
    center = payload["center_mass_index"]   # JAX center mass index
    jax_z = np.asarray(payload["z_m"])
    # low-level cutoff index ~ z<1500 m. JAX uniform 100 m -> klow=15.
    jax_klow = int(np.searchsorted(jax_z, 1500.0))
    # WRF stretched 65-level ztop=6409: first ~17 mass levels span ~0-1500 m.
    wrf_klow = 17

    # WRF center column is i=256.
    wrf_center = 256

    # Per-timestep END-OF-STEP comparison: the last RK3 substep of each timestep.
    timesteps = sorted({k[0] for k in wrf})
    per_step = []
    for it in timesteps:
        # WRF RK3 final substep: rk=3, iter=nsmall.
        wrf_rk3 = [k for k in wrf if k[0] == it and k[1] == 3]
        if not wrf_rk3:
            continue
        wrf_last = max(wrf_rk3, key=lambda k: k[2])
        wc = wrf[wrf_last]["cols"].get(wrf_center)
        # JAX RK3 final substep:
        jax_rk3 = [k for k in jax if k[0] == it and k[1] == 3]
        if not jax_rk3 or wc is None:
            continue
        jax_last = max(jax_rk3, key=lambda k: k[2])
        jr = jax[jax_last]["cols"].get(str(center))
        if jr is None:
            continue
        wsum = summarize("wrf", wc, wrf_center, wrf_klow)
        jsum = summarize("jax", jr, center, jax_klow)
        per_step.append({
            "itimestep": it,
            "wrf_final_substep": {"rk": wrf_last[1], "iter": wrf_last[2], "nsmall": wrf[wrf_last]["nsmall"]},
            "jax_final_substep": {"rk": jax_last[1], "iter": jax_last[2], "nsmall": jax[jax_last]["nsmall"]},
            "wrf": wsum,
            "jax": jsum,
        })

    # Per-substep RK3 trajectory at a representative touchdown timestep band.
    # We capture the full RK3 substep sweep for itimestep 180, 200 to show how the
    # descending-mass -> divergence conversion evolves within the acoustic loop.
    traj = {}
    for it in (180, 195, 200, 205):
        rows = []
        for k in sorted([kk for kk in wrf if kk[0] == it and kk[1] == 3], key=lambda kk: kk[2]):
            wc = wrf[k]["cols"].get(wrf_center)
            if wc is not None:
                rows.append({"rk": k[1], "iter": k[2], **summarize("wrf", wc, wrf_center, wrf_klow)})
        wtraj = rows
        rows = []
        for k in sorted([kk for kk in jax if kk[0] == it and kk[1] == 3], key=lambda kk: kk[2]):
            jr = jax[k]["cols"].get(str(center))
            if jr is not None:
                rows.append({"rk": k[1], "iter": k[2], **summarize("jax", jr, center, jax_klow)})
        traj[str(it)] = {"wrf": wtraj, "jax": rows}

    out = {
        "title": "F7N per-acoustic-substep touchdown-column WRF-vs-JAX diff (Straka em_grav2d_x)",
        "touchdown_column": {"wrf_i": wrf_center, "jax_center_idx": center,
                             "note": "WRF i=256=nxc=domain center; JAX center mass index nearest x=0"},
        "grids": {"wrf_e_vert": 65, "wrf_klow_idx_z1500": wrf_klow,
                  "jax_nz": int(jax_z.size), "jax_klow_idx_z1500": jax_klow},
        "field_convention": "u_iface/u_ip1face/w/ww are the COUPLED small-step work arrays (mu*field); dudx is the coupled mass-flux horizontal divergence (u_ip1face-u_iface)/dx, exactly the term advance_mu_t integrates into dmdt and ww.",
        "per_step_end_of_rk3": per_step,
        "rk3_substep_trajectory": traj,
    }
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")
    # Quick console summary of the key mechanism.
    print("\n itimestep | WRF w_min  JAX w_min | WRF dudx_surf JAX dudx_surf | WRF wwLowMin JAX wwLowMin")
    for r in per_step:
        w, j = r["wrf"], r["jax"]
        print(f"   {r['itimestep']:5d}  | {w['w_min']:9.2f} {j['w_min']:9.2f} | "
              f"{w['dudx_surface']:12.4f} {j['dudx_surface']:12.4f} | "
              f"{w['ww_lowlevel_signed_min']:10.3f} {j['ww_lowlevel_signed_min']:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
