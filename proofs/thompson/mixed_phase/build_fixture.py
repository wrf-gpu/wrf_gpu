"""Freeze the 384 REAL mixed-phase columns and their pristine one-step outputs for GPUWRF_THOMPSON_MIXED_PHASE_WRF.

Columns (build_columns.py, seeded): CPU-WRF history states of Swiss 2023-01-15 (d01/d02, 06/12/18Z) and 2024-11-24
(d01/d02) and WN3 0227 (d02/d03), kinds rime / bigg / rci (riming, rain freezing, rain-ice collection somewhere in the
column). WRF reference: ONE pristine mp_gt_driver step (module_mp_thompson.F mp8, gfortran WRF FCOPTIM build, the
LL01 driver proofs/thompson/drizzle_oracle/thompson_column_driver.F90 + run_oracle.py; tables = the .dat files
CPU-WRF reads, sha256 441bd836 / 910fb31d / 05497b0e).
usage: build_fixture.py <columns.npz> <columns.json> <oracle.npz> <fixture.npz>
"""
import json, sys
import numpy as np
z, meta, o = dict(np.load(sys.argv[1])), json.load(open(sys.argv[2]))["columns"], dict(np.load(sys.argv[3]))
ins = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
outs = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "rainncv", "snowncv", "graupelncv")
names = np.array(["%s_%s_%s_%s_y%d_x%d" % (m["case"], m["dom"], m["time"][11:13], m["kind"], m["y"], m["x"]) for m in meta])
np.savez_compressed(sys.argv[4], names=names, dt=z["dt"], kind=np.array([m["kind"] for m in meta]),
                    case=np.array([m["case"] for m in meta]),
                    **{f"in_{v}": z[f"in_{v}"] for v in ins}, **{f"wrf_{v}": o[f"wrf_{v}"] for v in outs})
print(len(names), "columns:", dict(zip(*np.unique(z["dt"], return_counts=True))))
