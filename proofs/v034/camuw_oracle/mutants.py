#!/usr/bin/env python3
"""Re-runnable CAM-UW mutant harness (E202: a kill = the parity GATE fails, not a crash).

Each mutant edits a TEMP copy of src/gpuwrf/physics/bl_camuw.py (the worktree file is never touched),
imports it as a separate module and applies the exact gate of tests/test_v034_camuw_oracle_parity.py
(4 REAL ulp of per-record field scale; exact pblh/kpbl/turbtype) on both oracle sets.
Usage: JAX_PLATFORMS=cpu PYTHONPATH=src python3 proofs/v034/camuw_oracle/mutants.py OUT.json
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_v034_camuw_oracle_parity import EXACT, OUTS, SETS, ULP_TOL, _inputs  # noqa: E402

MUTANTS = {
    "control": None,
    "no_ke_dissipation_heating": ("dse = dse + dtk", "dse = dse + 0.0 * dtk"),
    "tauresx_sign": ("tauresx = taux + tautmsx + tauresx - tauimpx", "tauresx = taux + tautmsx + tauresx + tauimpx"),
    "no_specific_humidity_conversion": ("mult = 1.0 / (1.0 + jnp.asarray(qv32, F64))", "mult = 1.0 + 0.0 * jnp.asarray(qv32, F64)"),
    "ricrit_0.25": ("RICRIT = 0.19\n", "RICRIT = 0.25\n"),
    "ricrit_0.20": ("RICRIT = 0.19\n", "RICRIT = 0.20\n"),
    "estbl_zero": ("ESTBL = _estbl_table()\n", "ESTBL = tuple(0.0 for _ in _estbl_table())\n"),
}


def load(src_text, name):
    d = tempfile.mkdtemp(prefix="camuw_mut_")
    p = Path(d) / f"{name}.py"
    p.write_text(src_text)
    spec = importlib.util.spec_from_file_location(f"camuw_mut_{name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gate(mod):
    for oracle_set in SETS:
        recs = json.load(open(HERE / oracle_set))["records"]
        batch = {k: np.stack([_inputs(r)[k] for r in recs]) for k in _inputs(recs[0])}
        out = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(lambda kw: mod.camuw_column(**kw)))(batch))
        for j, r in enumerate(recs):
            for k, ok in OUTS.items():
                got = np.atleast_1d(out[k][j]).astype(np.float64)
                exp = np.atleast_1d(np.asarray(r["out"][ok], np.float64))
                where = f"{oracle_set}:{r['name']}#{r['step']}:{k}"
                if not np.all(np.isfinite(got)):
                    return f"GATE-FAIL nonfinite {where}"
                if k in EXACT:
                    if not np.array_equal(got, exp):
                        return f"GATE-FAIL exact {where}"
                    continue
                scale = float(np.max(np.abs(exp)))
                ulp = float(np.spacing(np.float32(scale))) if scale > 0 else 0.0
                err = float(np.max(np.abs(got - exp)))
                if err > (ULP_TOL * ulp if scale > 0 else 1e-30):
                    return f"GATE-FAIL {where} err={err:.3e} scale={scale:.3e}"
    return "PASS"


def main(dst):
    base = (ROOT / "src/gpuwrf/physics/bl_camuw.py").read_text()
    res = {}
    for name, edit in MUTANTS.items():
        text = base
        if edit is not None:
            assert base.count(edit[0]) == 1, name
            text = base.replace(edit[0], edit[1])
        verdict = gate(load(text, name))
        killed = verdict.startswith("GATE-FAIL")
        res[name] = {"verdict": verdict, "killed": killed}
        print(name, "KILLED" if killed else "SURVIVED", verdict, flush=True)
    assert not res["control"]["killed"], "control must pass"
    json.dump(res, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
