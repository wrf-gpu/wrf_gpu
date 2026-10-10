#!/usr/bin/env python3
"""RUC LSM (sf_surface_physics=3) JAX port vs pristine-WRF oracle, step by step.

Oracle: ``proofs/v034/oracle/ruclsm`` (unmodified WRF ``LSMRUC`` driven per regime,
12 steps, full state dumped after every step, fp64 and REAL=4 fp32 builds).
Savepoints: ``proofs/v034/savepoints/ruclsm/{fp64,fp32}/ruclsm_v2_{fp64,fp32}.json``.

Usage::

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src python proofs/v034/ruclsm_parity.py \
        [--precision fp64|fp32|both] [--out metrics.json]

Each regime runs as one column with its own configuration; every one of the 12
steps is compared on all 43 dumped fields (scalars + soil profiles).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAVEPOINTS = ROOT / "proofs" / "v034" / "savepoints" / "ruclsm"

STATE_IN = {
    # state key -> oracle input name
    **{k: k.upper() for k in (
        "soilt", "soilt1", "tsnav", "snow", "snowh", "snowc", "canwat", "alb", "emiss", "znt",
        "z0", "lai", "mavail", "vegfra", "snoalb", "qvg", "qsg", "qcg", "dew", "qsfc", "chklowq",
        "hfx", "qfx", "lh", "grdflx", "sfcrunoff", "udrunoff", "acrunoff", "sfcexc", "sfcevp",
        "smavail", "smmax", "snowfallac", "acsnow", "snom", "rhosnf", "precipfr", "tso",
        "soilmois", "sh2o", "smfr3d", "keepfr3dflag")},
}
FORCING_IN = {k: k.upper() for k in (
    "t3d", "qv3d", "qc3d", "p8w", "rho3d", "z3d", "glw", "gsw", "chs", "flqc", "flhc", "rainbl",
    "rainncv", "snowncv", "graupelncv", "frzfrac")}
STATIC_IN = {k: k.upper() for k in ("ivgtyp", "isltyp", "xland", "xice", "tbot", "shdmin", "shdmax", "albbck")}
COMPARED = (
    "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg", "dew", "qsfc", "hfx", "qfx", "lh", "grdflx",
    "sfcrunoff", "udrunoff", "acrunoff", "sfcexc", "sfcevp", "smavail", "smmax", "snowfallac",
    "acsnow", "snom", "rhosnf", "precipfr", "chklowq", "snow", "snowh", "snowc", "canwat", "alb",
    "emiss", "znt", "z0", "lai", "mavail", "vegfra", "snoalb", "tso", "soilmois", "sh2o", "smfr3d",
    "keepfr3dflag",
)


def load(precision: str) -> dict:
    return json.loads((SAVEPOINTS / precision / f"ruclsm_v2_{precision}.json").read_text())


def regime_config(reg: dict, scalars: dict, dtype: str):
    from gpuwrf.physics.ruclsm import RucConfig

    inp = reg["inputs"]
    return RucConfig(
        dt=float(inp["DT"]), zs=tuple(float(z) for z in inp["ZS"]), dtype=dtype,
        frpcpn=bool(int(inp["FRPCPN"])), myj=bool(int(inp["MYJ"])),
        mosaic_lu=int(inp["MOSAIC_LU"]), mosaic_soil=int(inp["MOSAIC_SOIL"]),
        rdlai2d=bool(int(inp["RDLAI2D"])), iswater=int(inp["ISWATER"]), isice=int(inp["ISICE"]),
        xice_threshold=float(inp["XICE_THRESHOLD"]), cp=float(scalars["CONST_CP"]),
        rovcp=float(scalars["CONST_ROVCP"]), g0=float(scalars["CONST_G0"]),
        lv=float(scalars["CONST_LV"]), stbolt=float(scalars["CONST_STBOLT"]),
    )


def _arr(v, dtype, prof=False):
    a = np.asarray(v, dtype=dtype)
    return a[None, :] if prof else a.reshape(1)


def run_regime(reg: dict, scalars: dict, precision: str, *, jit: bool = True):
    """Run the JAX port over the oracle steps of one regime; return per-step outputs."""

    import jax
    from gpuwrf.physics import ruclsm as r

    dtype = "float64" if precision == "fp64" else "float32"
    cfg = regime_config(reg, scalars, dtype)
    tab = r.load_ruc_tables(scalars.get("MMINLURUC", "USGS-RUC"))
    inp = reg["inputs"]
    prof = {"tso", "soilmois", "sh2o", "smfr3d", "keepfr3dflag"}
    state = {k: _arr(inp[n], dtype, k in prof) for k, n in STATE_IN.items()}
    forcing = {k: _arr(inp[n], dtype) for k, n in FORCING_IN.items()}
    static = {k: _arr(inp[n], dtype) for k, n in STATIC_IN.items()}
    static["ivgtyp"] = static["ivgtyp"].astype(np.int32)
    static["isltyp"] = static["isltyp"].astype(np.int32)
    extra = {}
    if cfg.mosaic_lu == 1:
        extra["landusef"] = np.asarray(inp["LANDUSEF"], dtype)[None, :]
    if cfg.mosaic_soil == 1:
        extra["soilctop"] = np.asarray(inp["SOILCTOP"], dtype)[None, :]

    def step(state, ktau):
        return r.lsmruc_step(state, forcing, static, cfg, tab, ktau, **extra)

    fn = jax.jit(step) if jit else step
    ktau0 = int(inp.get("KTAU_START", 1))
    out = []
    for n in range(len(reg["steps"])):
        res = fn(state, ktau0 + n)
        out.append({k: np.asarray(v) for k, v in res.items()})
        state = {k: res[k] for k in state}
    return out


def compare(reg: dict, out: list) -> dict:
    worst = {}
    for n, ref in enumerate(reg["steps"]):
        got = out[n]
        for k in COMPARED:
            g = np.asarray(got[k], dtype=np.float64).reshape(-1)
            o = np.asarray(ref[k.upper()], dtype=np.float64).reshape(-1)
            err = float(np.max(np.abs(g - o)))
            scale = float(np.max(np.abs(o)))
            rel = err / scale if scale > 0 else err
            w = worst.get(k)
            if w is None or rel > w["rel"]:
                worst[k] = {"rel": rel, "abs": err, "step": n + 1}
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precision", default="both", choices=("fp64", "fp32", "both"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--regimes", default=None, help="comma list of 1-based regime indices")
    ap.add_argument("--metrics", default=None, help="write the fp64 summary metrics JSON here")
    args = ap.parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    precisions = ("fp64", "fp32") if args.precision == "both" else (args.precision,)
    sel = None if args.regimes is None else {int(x) for x in args.regimes.split(",")}
    report = {}
    for prec in precisions:
        data = load(prec)
        rows = {}
        for i, reg in enumerate(data["regimes"], start=1):
            if sel is not None and i not in sel:
                continue
            if reg["status"] != "ok":
                rows[reg["name"]] = {"status": reg["status"], "skipped": "WRF fatal (undefined behaviour)"}
                continue
            out = run_regime(reg, data["scalars"], prec)
            unsupported = bool(np.any(out[0]["unsupported"]))
            worst = compare(reg, out)
            top = max(worst.items(), key=lambda kv: kv[1]["rel"])
            rows[reg["name"]] = {
                "index": i, "unsupported_flag": unsupported,
                "vilka_ok": bool(all(np.all(o["vilka_ok"]) for o in out)),
                "worst_field": top[0], "worst_rel": top[1]["rel"], "worst_abs": top[1]["abs"],
                "worst_step": top[1]["step"], "fields": worst,
            }
            print(f"[{prec}] {i:2d} {reg['name']:34s} unsup={unsupported!s:5s} worst {top[0]:12s} "
                  f"rel {top[1]['rel']:.3e} abs {top[1]['abs']:.3e} @step {top[1]['step']}", flush=True)
        report[prec] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    if args.metrics and "fp64" in report:
        rows = {k: v for k, v in report["fp64"].items() if "fields" in v and not v["unsupported_flag"]}
        fields = {}
        for row in rows.values():
            for name, w in row["fields"].items():
                cur = fields.get(name)
                if cur is None or w["rel"] > cur["max_rel"]:
                    fields[name] = {"max_rel": w["rel"], "max_abs": w["abs"], "pass": w["rel"] <= 1.0e-9}
        worst_rel = max(f["max_rel"] for f in fields.values())
        summary = {
            "schema": "ruclsm-v2-fp64-step-parity",
            "oracle": "proofs/v034/savepoints/ruclsm/fp64/ruclsm_v2_fp64.json",
            "all_green": all(f["pass"] for f in fields.values()),
            "rel_tol": 1.0e-9,
            "n_columns": len(rows),
            "n_steps": 12,
            "worst_rel": worst_rel,
            "worst_abs": max(f["max_abs"] for f in fields.values()),
            "unsupported_flagged": sorted(k for k, v in report["fp64"].items() if v.get("unsupported_flag")),
            "wrf_fatal_skipped": sorted(k for k, v in report["fp64"].items() if "skipped" in v),
            "fields": fields,
        }
        Path(args.metrics).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
