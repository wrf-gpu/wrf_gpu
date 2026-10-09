"""Run the mixed-phase oracle gate (tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py) and every deletion mutant
on all 384 fixture columns; record per mutant the failing (column, field, excess). usage: mutant_sweep.py <out.json>"""
import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("mp_oracle", ROOT / "tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
z = dict(np.load(m.FIXTURE))
out = {}
for mutant in (None, *m.MUTANTS):
    t = time.time()
    bad = m.failures(z, mutant)
    out[mutant or "candidate"] = {"n_fail": len(bad), "seconds": round(time.time() - t, 1),
                                  "worst": sorted(([n, v, float(e)] for n, v, e in bad), key=lambda r: -r[2])[:12],
                                  "columns": sorted({n for n, _v, _e in bad})}
    print(mutant or "candidate", len(bad), out[mutant or "candidate"]["worst"][:2], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
