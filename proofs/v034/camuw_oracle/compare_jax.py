#!/usr/bin/env python3
"""Compare gpuwrf.physics.bl_camuw against the v034 pristine-WRF CAM-UW oracle (CPU only).

Metric per output field: max |jax - wrf| in units of the WRF REAL ulp at the oracle value
(spacing(np.float32(wrf))) plus max abs/rel error. Writes camuw_jax_vs_wrf_v034.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.bl_camuw import camuw_column

HERE = Path(__file__).resolve().parent
OUTS = {"rublten": "RUBLTEN", "rvblten": "RVBLTEN", "rthblten": "RTHBLTEN", "rqvblten": "RQVBLTEN",
        "rqcblten": "RQCBLTEN", "rqiblten": "RQIBLTEN", "rqniblten": "RQNIBLTEN", "tke_pbl": "TKE_PBL",
        "kvm3d": "KVM", "kvh3d": "KVH", "smaw3d": "SMAW", "turbtype3d": "TURBTYPE", "pblh": "PBLH",
        "kpbl": "KPBL", "tpert": "TPERT", "qpert": "QPERT", "wpert": "WPERT", "tauresx2d": "TAURESX",
        "tauresy2d": "TAURESY"}


def inputs(rec, dt):
    i = rec["in"]
    a = lambda k: np.asarray(i[k], np.float32)  # noqa: E731
    return dict(dt=np.float32(dt), u_phy=a("U"), v_phy=a("V"), th_phy=a("TH"), rho=a("RHO"), qv=a("QV"),
                qc=a("QC"), qi=a("QI"), qnc=a("QNC"), qni=a("QNI"), p_phy=a("P"), p8w=a("P8W"), z=a("Z"),
                z_at_w=a("ZW"), t_phy=a("T"), cldfra=a("CLDFRA"), rthratenlw=a("RTHRATENLW"),
                exner=a("EXNER"), wsedl=a("WSEDL"), hfx=np.float32(i["HFX"]), qfx=np.float32(i["QFX"]),
                ustar=np.float32(i["UST"]), ht=np.float32(i["HT"]), kvm3d=a("KVM"), kvh3d=a("KVH"),
                tauresx2d=np.float32(i["TAURESX"]), tauresy2d=np.float32(i["TAURESY"]),
                first_step=np.bool_(rec["step"] == 1))


def run(oracle_path):
    d = json.load(open(oracle_path))
    recs = d["records"]
    dt = 60.0
    ins = [inputs(r, dt) for r in recs]
    keys = list(ins[0].keys())
    batch = {k: np.stack([x[k] for x in ins]) for k in keys}
    f = jax.jit(jax.vmap(lambda kw: camuw_column(**kw)))
    out = jax.tree_util.tree_map(np.asarray, f(batch))
    report = {"fields": {}, "records": []}
    for j, r in enumerate(recs):
        rr = {"col": r["col"], "name": r["name"], "step": r["step"], "fields": {}}
        for k, ok in OUTS.items():
            got = np.atleast_1d(out[k][j]).astype(np.float64)
            exp = np.atleast_1d(np.asarray(r["out"][ok], np.float64))
            err = np.abs(got - exp)
            ulp = np.spacing(np.abs(exp).astype(np.float32)).astype(np.float64)
            ulps = err / ulp
            scale = max(float(np.max(np.abs(exp))), 1e-30)
            rr["fields"][k] = {"max_abs": float(np.max(err)), "max_ulp": float(np.max(ulps)),
                               "max_rel_scale": float(np.max(err) / scale), "argmax": int(np.argmax(err))}
            agg = report["fields"].setdefault(k, {"max_abs": 0.0, "max_ulp": 0.0, "max_rel_scale": 0.0, "worst": None})
            if rr["fields"][k]["max_rel_scale"] > agg["max_rel_scale"]:
                agg["worst"] = f"{r['name']}#{r['step']}"
            for m in ("max_abs", "max_ulp", "max_rel_scale"):
                agg[m] = max(agg[m], rr["fields"][k][m])
        report["records"].append(rr)
    return report


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    rep = run(sys.argv[1] if len(sys.argv) > 1 else HERE / "camuw_oracle_v034.json")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "camuw_jax_vs_wrf_v034.json"
    json.dump(rep, open(dst, "w"), indent=1)
    for k, v in rep["fields"].items():
        print(f"{k:11s} max_rel={v['max_rel_scale']:.3e} max_ulp={v['max_ulp']:.3g} max_abs={v['max_abs']:.3e} worst={v['worst']}")
