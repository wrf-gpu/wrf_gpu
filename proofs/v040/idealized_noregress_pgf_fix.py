"""Opus v040 round — idealized no-regression for the WRF-faithful PGF (al+dpn) fix.

Proves the al + dpn-top fixes leave the periodic idealized path BIT-IDENTICAL by
running the actual operational dycore (run_boundary=False, top_lid=True,
use_flux_advection=True -- the F7 idealized config) for a short segment with the
LIVE (fixed) `_absolute_diagnostics` vs the pre-fix one from base commit 7170e1a.

The corrections are proportional to mu' (al) and terrain slope (dpn enters via the
horizontal geopotential gradient php_r-php_l); both vanish on the flat-terrain
hydrostatically-balanced idealized init, so the result is bit-identical -- the
required BC-conditional behaviour, achieved structurally rather than by a flag.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)

import gpuwrf.dynamics.core.rk_addtend_dry as rk  # noqa: E402
from gpuwrf.ic_generators.idealized import (  # noqa: E402
    _initial_carry,
    _run_segment,
    build_density_current_setup,
    build_warm_bubble_setup,
)

BASE = "7170e1a"


def _old_absdiag():
    src = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:src/gpuwrf/dynamics/core/rk_addtend_dry.py"], text=True
    )
    m = types.ModuleType("old_rk_idealized")
    sys.modules[m.__name__] = m
    exec(compile(src, f"{BASE}:rk_addtend_dry.py", "exec"), m.__dict__)
    return m._absolute_diagnostics


def _segment(setup, steps, absdiag):
    rk._absolute_diagnostics = absdiag
    c = _run_segment(_initial_carry(setup.state), setup.namelist, start_step=0, steps=steps)
    return np.asarray(c.state.theta), np.asarray(c.state.u), np.asarray(c.state.w)


def main() -> int:
    new_ad = rk._absolute_diagnostics
    old_ad = _old_absdiag()
    cases = {"warm_bubble": (build_warm_bubble_setup(require_gpu=False), 30),
             "straka_density_current": (build_density_current_setup(require_gpu=False), 40)}
    out = {"schema": "v040_idealized_noregress_pgf_fix.v1", "created_by": "Opus 4.8 MAX",
           "base_commit": BASE, "config": "run_boundary=False, top_lid=True, use_flux_advection=True (F7 idealized)",
           "cases": {}}
    overall = True
    for name, (setup, steps) in cases.items():
        rk._absolute_diagnostics = new_ad
        tn, un, wn = _segment(setup, steps, new_ad)
        rk._absolute_diagnostics = old_ad
        to, uo, wo = _segment(setup, steps, old_ad)
        rk._absolute_diagnostics = new_ad
        dth = float(np.max(np.abs(tn - to))); du = float(np.max(np.abs(un - uo))); dw = float(np.max(np.abs(wn - wo)))
        bit_identical = (dth == 0.0 and du == 0.0 and dw == 0.0)
        overall = overall and bit_identical and bool(np.all(np.isfinite(tn)))
        out["cases"][name] = {"steps": steps, "max_abs_dtheta": dth, "max_abs_du": du, "max_abs_dw": dw,
                              "bit_identical": bit_identical, "finite": bool(np.all(np.isfinite(tn)))}
    out["overall_bit_identical_periodic_path"] = bool(overall)
    p = Path(__file__).resolve().parent / "idealized_noregress_pgf_fix.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    print("WROTE", p)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
