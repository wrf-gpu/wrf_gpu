"""GWDO oracle comparison: JAX port vs pristine-WRF ``bl_gwdo_run`` savepoint.

Reads the Fortran oracle dump (built by ``/tmp/gwdo_oracle/oracle_driver`` from a
pristine WRF v4 ``phys/physics_mmm/bl_gwdo.F90`` compiled at fp32 ``kind_phys``)
and runs the JAX :func:`gpuwrf.physics.gwd_gwdo.gwdo_columns` on the IDENTICAL
inputs, asserting per-level agreement on the wind tendencies and the integrated
surface stress within an fp32 tolerance.

Usage:
    JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/gwd/compare_oracle.py [oracle.txt]

Emits ``proofs/gwd/gwdo_oracle_gate.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpuwrf.physics.gwd_gwdo import GWDOColumnState, GWDOStatics, gwdo_columns


def parse_oracle(path: Path):
    lines = path.read_text().splitlines()
    it = iter(lines)
    next(it)  # header
    kte, ncol = (int(x) for x in next(it).split())
    cols = []
    for _ in range(ncol):
        line = next(it)
        assert line.startswith("# COL"), line
        next(it)  # inputs header
        inp = np.array([[float(x) for x in next(it).split()] for _ in range(kte)])
        prsi_top = float(next(it).split()[-1])
        next(it)  # statics header
        stat = np.array([float(x) for x in next(it).split()])
        next(it)  # outputs header
        out = np.array([[float(x) for x in next(it).split()] for _ in range(kte)])
        dline = next(it).split()
        dusfcg, dvsfcg = float(dline[-2]), float(dline[-1])
        cols.append(dict(inp=inp, prsi_top=prsi_top, stat=stat, out=out,
                         dusfcg=dusfcg, dvsfcg=dvsfcg))
    return kte, ncol, cols


def main():
    oracle_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gwdo_oracle/oracle_out.txt")
    kte, ncol, cols = parse_oracle(oracle_path)

    # assemble batched JAX inputs from the oracle's own inputs
    uproj = np.stack([c["inp"][:, 1] for c in cols])
    vproj = np.stack([c["inp"][:, 2] for c in cols])
    t1 = np.stack([c["inp"][:, 3] for c in cols])
    q1 = np.stack([c["inp"][:, 4] for c in cols])
    prsl = np.stack([c["inp"][:, 5] for c in cols])
    prsi_low = np.stack([c["inp"][:, 6] for c in cols])  # prsi(k=1..kte)
    prslk = np.stack([c["inp"][:, 7] for c in cols])
    zl = np.stack([c["inp"][:, 8] for c in cols])
    prsi = np.concatenate([prsi_low, np.array([[c["prsi_top"]] for c in cols])], axis=1)

    stat = np.stack([c["stat"] for c in cols])  # (ncol, 13)
    col = GWDOColumnState(
        uproj=jnp.asarray(uproj), vproj=jnp.asarray(vproj), t1=jnp.asarray(t1),
        q1=jnp.asarray(q1), prsl=jnp.asarray(prsl), prsi=jnp.asarray(prsi),
        prslk=jnp.asarray(prslk), zl=jnp.asarray(zl),
    )
    statics = GWDOStatics(
        var=jnp.asarray(stat[:, 0]), oc1=jnp.asarray(stat[:, 1]),
        oa1=jnp.asarray(stat[:, 2]), oa2=jnp.asarray(stat[:, 3]),
        oa3=jnp.asarray(stat[:, 4]), oa4=jnp.asarray(stat[:, 5]),
        ol1=jnp.asarray(stat[:, 6]), ol2=jnp.asarray(stat[:, 7]),
        ol3=jnp.asarray(stat[:, 8]), ol4=jnp.asarray(stat[:, 9]),
        sina=jnp.asarray(stat[:, 10]), cosa=jnp.asarray(stat[:, 11]),
        dxmeter=jnp.asarray(stat[:, 12]),
    )
    out = gwdo_columns(col, statics, 60.0)

    ru = np.asarray(out.rublten)
    rv = np.asarray(out.rvblten)
    dtx = np.asarray(out.dtaux3d)
    dty = np.asarray(out.dtauy3d)
    dus = np.asarray(out.dusfcg)
    dvs = np.asarray(out.dvsfcg)

    report = {"kte": kte, "ncol": ncol, "tol_note": "fp32 oracle; rel/abs blended", "columns": []}
    all_pass = True
    for i, c in enumerate(cols):
        o_ru = c["out"][:, 1]
        o_rv = c["out"][:, 2]
        o_dtx = c["out"][:, 3]
        o_dty = c["out"][:, 4]
        # blended tolerance: physics tendencies are tiny (1e-4..1e-2); use an
        # absolute floor (fp32 noise ~1e-7 of the wind*stress scale) + relative.
        def err(a, b):
            scale = max(np.abs(b).max(), 1e-12)
            return float(np.max(np.abs(a - b))), float(np.max(np.abs(a - b)) / scale)

        e_ru = err(ru[i], o_ru)
        e_rv = err(rv[i], o_rv)
        e_dtx = err(dtx[i], o_dtx)
        e_dty = err(dty[i], o_dty)
        e_dus = (abs(float(dus[i] - c["dusfcg"])), )
        e_dvs = (abs(float(dvs[i] - c["dvsfcg"])), )
        # pass if abs err small OR rel err small (fp32 nonlinear scheme)
        col_ok = (
            (e_ru[0] < 1e-6 or e_ru[1] < 2e-3)
            and (e_rv[0] < 1e-6 or e_rv[1] < 2e-3)
            and (e_dtx[0] < 1e-6 or e_dtx[1] < 2e-3)
            and (e_dty[0] < 1e-6 or e_dty[1] < 2e-3)
            and (e_dus[0] < max(1e-4, abs(c["dusfcg"]) * 3e-3))
            and (e_dvs[0] < max(1e-4, abs(c["dvsfcg"]) * 3e-3))
        )
        all_pass = all_pass and col_ok
        report["columns"].append({
            "col": i + 1, "var": float(c["stat"][0]),
            "rublten_abs_rel": e_ru, "rvblten_abs_rel": e_rv,
            "dtaux3d_abs_rel": e_dtx, "dtauy3d_abs_rel": e_dty,
            "dusfcg": [float(dus[i]), c["dusfcg"], e_dus[0]],
            "dvsfcg": [float(dvs[i]), c["dvsfcg"], e_dvs[0]],
            "pass": col_ok,
        })

    report["verdict"] = "PASS" if all_pass else "FAIL"
    out_path = Path(__file__).resolve().parent / "gwdo_oracle_gate.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    print("VERDICT:", report["verdict"])
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
