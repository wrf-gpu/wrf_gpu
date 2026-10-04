"""MYNN surface layer vs pristine module_sf_mynn.F on real PROD columns (CPU).

Columns: the production Noah-MP-path column view (`noahmp_surface_hook.
_build_column_view`) of the saved PROD carry (d01 9 km, d02 3 km). Warm-step
inputs WRF passes to SFCLAY1D_mynn (HFX/QFX/PBLH/UST from a CPU-WRF history
frame on the same grid; MOL/QSFC from an oracle pre-pass, land QSFC from the
LSM-like Q2) are fed IDENTICALLY to the Fortran oracle (REAL*4 build of the
byte-identical pristine module) and to the port. `--drop` removes the named
inputs from the port only (deletion sensitivity: the gate must then FAIL).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np

IN_COLS = ["u", "v", "t1d", "qv", "p1d", "dz8w", "rho", "u1d2", "v1d2", "dz2w",
           "mavail", "pblh", "xland", "tsk", "psfcpa", "qcg", "snowh",
           "znt", "ust", "mol", "qsfc", "hfx", "qfx"]
OUT_COLS = ["ust", "mol", "rmol", "zol", "regime", "psim", "psih", "br",
            "flhc", "flqc", "hfx", "qfx", "lh", "qsfc", "qgh",
            "chs", "chs2", "cqs2", "ch", "wspd", "gz1oz0",
            "u10", "v10", "th2", "t2", "q2", "cpm", "wstar", "qstar", "znt"]
# Frozen v090 thresholds (tests/test_v090_mynnsl_oracle_parity.py): relative max, absolute escape 1e-6.
THR = {"ust": 0.01, "hfx": 0.02, "lh": 0.03, "qsfc": 0.01, "br": 0.05, "zol": 0.05,
       "mol": 0.05, "psim": 0.03, "psih": 0.03, "rmol": 0.05, "u10": 0.01, "v10": 0.01,
       "t2": 0.001, "th2": 0.001, "regime": 0.0, "znt": 1e-3}
LANE_CPUS = "8,9,12,13,24,25,28,29"


def run_oracle(exe, cols, dx, itimestep, real4=True):
    n = len(cols["u"])
    cast = np.float32 if real4 else np.float64
    lines = [f"{n} {itimestep} 1 0 0 0 {dx!r}"]
    for i in range(n):
        lines.append(" ".join(repr(float(cast(cols[k][i]))) for k in IN_COLS))
    proc = subprocess.run(["taskset", "-c", LANE_CPUS, str(exe)], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, check=True)
    out = {k: np.zeros(n) for k in OUT_COLS}
    for row in proc.stdout.splitlines():
        if not row or row.startswith("#"):
            continue
        p = row.split()
        i = int(p[0]) - 1
        for j, k in enumerate(OUT_COLS):
            out[k][i] = float(p[1 + j])
    return out


def columns_from_view(view, extras):
    s = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    surf = lambda a: s(a)[..., 0] if s(a).ndim == 3 else s(a)  # noqa: E731
    lvl1 = lambda a: s(a)[..., 1] if s(a).ndim == 3 else s(a)  # noqa: E731
    cols = dict(
        u=surf(view.u), v=surf(view.v), t1d=surf(view.t_air), qv=np.maximum(surf(view.qv), 0.0),
        p1d=surf(view.p), dz8w=surf(view.dz), rho=surf(view.rho), u1d2=lvl1(view.u), v1d2=lvl1(view.v),
        dz2w=lvl1(view.dz), mavail=s(view.mavail), xland=s(view.xland), tsk=s(view.t_skin),
        psfcpa=s(view.psfc), qcg=np.zeros_like(s(view.xland)), snowh=np.zeros_like(s(view.xland)),
        znt=s(view.roughness_m))
    cols.update({k: np.asarray(v, dtype=np.float64) for k, v in extras.items()})
    return {k: v.reshape(-1) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True, help="pickle {domain: (state, grid, clock, static, land, manifest)}")
    ap.add_argument("--wrfout", type=Path, nargs=2, required=True, help="CPU-WRF history frames d01 d02 (same grid)")
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--real8", action="store_true", help="oracle is the REAL*8 build: pass f64 inputs")
    ap.add_argument("--thr-scale", type=float, default=1.0, help="multiply frozen relative thresholds")
    ap.add_argument("--via-state", action="store_true",
                    help="B39 wiring: put the warm inputs into the State leaves and build the view with the production "
                         "_build_column_view (no explicit extras); --drop then zeroes those State leaves")
    ap.add_argument("--save-npz", type=Path, help="store port/oracle arrays per domain for offline comparison")
    ap.add_argument("--patch-wrf-constants", action="store_true",
                    help="EXPERIMENT: rebind surface_layer's constants to WRF module_model_constants values")
    args = ap.parse_args()
    assert Path("/tmp/wrf_gpu2_quiet").exists() is False, "QUIET"
    import jax
    import jax.numpy as jnp
    import netCDF4
    assert jax.devices()[0].platform == "cpu"
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    if args.patch_wrf_constants:
        import gpuwrf.physics.surface_layer as SL
        SL.CP_D, SL.EP1, SL.EP2 = 1004.5, 461.6 / 287.0 - 1.0, 287.0 / 461.6
        SL.R_D_OVER_CP = 287.0 / 1004.5

    data = pickle.load(args.inputs.open("rb"))
    report = dict(inputs=str(args.inputs), wrfout=[str(p) for p in args.wrfout], drop=args.drop,
                  oracle=str(args.oracle), real8=args.real8, thr_scale=args.thr_scale,
                  patch_wrf_constants=args.patch_wrf_constants, via_state=args.via_state,
                  thresholds=THR, domains={})
    for domain, wrfout in zip(("d01", "d02"), args.wrfout):
        state, grid, *_ = data[domain]
        for name in ("mol", "hfx", "qfx", "qsfc", "pblh"):  # pre-B39 pickles lack the slots
            try:
                getattr(state, name)
            except AttributeError:
                object.__setattr__(state, name, jnp.zeros_like(state.xland, dtype=jnp.float32))
        view = _build_column_view(state, grid)
        nc = netCDF4.Dataset(wrfout)
        frame = lambda name: np.asarray(nc.variables[name][0], dtype=np.float64)  # noqa: E731
        dx = float(grid.projection.dx_m)
        assert abs(dx - float(nc.getncattr("DX"))) < 1e-3
        warm = dict(hfx=frame("HFX"), qfx=frame("QFX"), pblh=frame("PBLH"), ust=frame("UST"),
                    mol=np.zeros_like(frame("HFX")), qsfc=-np.ones_like(frame("HFX")))
        cols = columns_from_view(view, warm)
        pre = run_oracle(args.oracle, cols, dx, itimestep=2, real4=not args.real8)
        # Warm-step carry WRF hands to the next call: MOL from sfclay, land QSFC from the
        # LSM (here the CPU-WRF 2 m specific humidity), water QSFC recomputed by sfclay.
        q2 = frame("Q2").reshape(-1)
        land = cols["xland"] < 1.5
        cols["mol"] = pre["mol"]
        cols["qsfc"] = np.where(land, q2 / (1.0 + q2), pre["qsfc"])
        if args.via_state:
            # State leaves are WRF REAL: the oracle sees the same REAL values.
            for k in ("mol", "hfx", "qfx", "qsfc", "pblh"):
                cols[k] = cols[k].astype(np.float32).astype(np.float64)
        oracle = run_oracle(args.oracle, cols, dx, itimestep=2, real4=not args.real8)

        if args.via_state:
            shape2 = view.xland.shape
            leaves = {k: jnp.asarray(cols[k].reshape(shape2)) for k in ("mol", "hfx", "qfx", "qsfc", "pblh")}
            for name in args.drop:
                leaves[name] = jnp.zeros(shape2)
            wired = state.replace(ustar=jnp.asarray(cols["ust"].reshape(shape2)), **leaves)
            port_view = _build_column_view(wired, grid)
            assert port_view.dx_m == dx
        else:
            passed = {k: jnp.asarray(cols[k].reshape(view.xland.shape)) for k in ("mol", "hfx", "qfx", "qsfc", "pblh")}
            passed["ustar"] = jnp.asarray(cols["ust"].reshape(view.xland.shape))
            passed["dx_m"] = dx
            for name in args.drop:
                passed.pop(name)
            port_view = view._replace(ustar=passed.pop("ustar")) if "ustar" in passed else view
            port_view = _ViewWithExtras(port_view, passed)
        diag = surface_layer_with_diagnostics(port_view, first_timestep=False)
        flat = lambda a: np.asarray(a, dtype=np.float64).reshape(-1)  # noqa: E731
        port = dict(ust=flat(diag.fluxes.ustar), mol=flat(diag.mol), rmol=flat(diag.rmol), zol=flat(diag.zol),
                    regime=flat(diag.regime), psim=flat(diag.psim), psih=flat(diag.psih), br=flat(diag.br),
                    hfx=flat(diag.hfx), lh=flat(diag.lh), qsfc=flat(diag.qsfc), u10=flat(diag.u10),
                    v10=flat(diag.v10), th2=flat(diag.th2), t2=flat(diag.t2), znt=flat(diag.znt))
        if args.save_npz:
            np.savez_compressed(args.save_npz.with_name(f"{args.save_npz.stem}_{domain}.npz"),
                                **{f"port_{k}": v for k, v in port.items()},
                                **{f"oracle_{k}": oracle[k] for k in port},
                                xland=cols["xland"], port_dtype=str(np.asarray(diag.hfx).dtype))
        fields = {}
        for k, thr in THR.items():
            a, b = port[k], oracle[k]
            absd = np.abs(a - b)
            rel = absd / np.maximum(np.abs(b), 1e-12)
            bad = (rel > thr * args.thr_scale) & (absd >= 1e-6)
            fields[k] = dict(relmax=float(rel.max()), absmax=float(absd.max()), n_fail=int(bad.sum()),
                             finite=bool(np.isfinite(a).all()), rel_p999=float(np.quantile(rel, 0.999)),
                             worst=[dict(i=int(i), port=float(a[i]), oracle=float(b[i]), xland=float(cols["xland"][i]),
                                         wspd=float(oracle["wspd"][i]), br=float(oracle["br"][i]))
                                    for i in np.argsort(-np.where(bad, absd, 0))[:3] if bad[i]])
        report["domains"][domain] = dict(
            n_columns=int(cols["u"].size), dx=dx, wrfout=str(wrfout), port_dtype=str(np.asarray(diag.hfx).dtype),
            wstar_oracle_max=float(oracle["wstar"].max()), n_land=int(land.sum()),
            fields=fields, n_fail=sum(f["n_fail"] for f in fields.values()))
        print(domain, report["domains"][domain]["n_fail"],
              {k: (round(v["relmax"], 5), v["n_fail"]) for k, v in fields.items()})
    report["pass"] = all(d["n_fail"] == 0 for d in report["domains"].values())
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if report["pass"] else "FAIL")


class _ViewWithExtras:
    """Attribute view: the production column view plus explicitly passed WRF inputs."""

    def __init__(self, view, extras):
        self._view, self._extras = view, extras

    def __getattr__(self, name):
        if name in self._extras:
            return self._extras[name]
        return getattr(self._view, name)


if __name__ == "__main__":
    os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
    main()
