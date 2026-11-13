"""Nested GPU-vs-CPU-WRF comparator verdicts (critic FIX v50-stoch finding 8).

w41 pattern_compare.py exited 0 for a FAIL and passed one OFF pair with zero pattern fields; the nested
comparator needs hours+1 pairs per domain, the full stochastic inventory, zero-reference = zero, and
exits 0/1/2.
"""
import importlib.util
from pathlib import Path

import netCDF4
import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nested_compare", REPO / "proofs/v50_stoch/nested_compare.py")
nc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nc)


def _frame(path, hour, *, rstoch=0., seed=7, drop=(), second=0):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1); ds.createDimension("DateStrLen", 19)
        ds.createDimension("y", 3); ds.createDimension("x", 4); ds.createDimension("s", 8)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times.set_auto_chartostring(False)
        times[0, :] = np.array(list(f"2026-02-28_{hour:02d}:00:{second:02d}"), "S1")
        for name in (*nc.PATTERNS, *nc.SPECTRAL):
            if name not in drop:
                ds.createVariable(name, "f4", ("Time", "y", "x"))[:] = rstoch if name == "RSTOCH" else 0.
        for name in nc.SEEDS:
            ds.createVariable(name, "i4", ("Time", "s"))[:] = seed


def _case(root, hours, **kw):
    root.mkdir()
    for d in ("d01", "d02"):
        for h in range(hours + 1):
            _frame(root / f"wrfout_{d}_2026-02-28_{h:02d}:00:00", h, **(kw.get(d, {}) if h == 0 else {}))
    return root


def test_verdicts_and_exit_codes(tmp_path):
    args = lambda cpu, gpu, n, hours: [str(cpu), str(gpu), str(tmp_path / f"{n}.json"), "--domains", "d01", "d02",
                                       "--hours", str(hours)]
    cpu = _case(tmp_path / "cpu", 2, d02={"rstoch": .05})
    same = _case(tmp_path / "same", 2, d02={"rstoch": .05})
    assert nc.main(args(cpu, same, "pass", 2)) == 0
    zero_child = _case(tmp_path / "zero", 2)                      # port wrote a zero child lead-zero pattern
    assert nc.main(args(cpu, zero_child, "zero", 2)) == 1
    nonzero_on_zero = _case(tmp_path / "nz", 2, d01={"rstoch": 1e-3}, d02={"rstoch": .05})
    assert nc.main(args(cpu, nonzero_on_zero, "nz", 2)) == 1      # zero reference requires zero
    seeds = _case(tmp_path / "seeds", 2, d01={"seed": 8}, d02={"rstoch": .05})
    assert nc.main(args(cpu, seeds, "seeds", 2)) == 1
    assert nc.main(args(cpu, same, "short", 3)) == 2               # one pair per domain is never a pass
    missing = _case(tmp_path / "missing", 2, d02={"rstoch": .05, "drop": ("SPFORCC2",)})
    assert nc.main(args(cpu, missing, "missing", 2)) == 2


def test_delta1_inventory_and_clock_are_exact(tmp_path):
    """Critic Delta 1 finding 4: extra frames, shifted seconds and undeclared domains used to PASS."""
    args = lambda cpu, gpu, n: [str(cpu), str(gpu), str(tmp_path / f"{n}.json"), "--domains", "d01", "d02", "--hours", "2"]
    cpu = _case(tmp_path / "cpu", 2)
    extra = _case(tmp_path / "extra", 2)
    _frame(extra / "wrfout_d01_2026-02-28_03:00:00", 3)
    assert nc.main(args(cpu, extra, "extra")) == 1
    shifted = _case(tmp_path / "shifted", 2)
    (shifted / "wrfout_d02_2026-02-28_01:00:00").unlink()
    _frame(shifted / "wrfout_d02_2026-02-28_01:00:18", 1, second=18)
    assert nc.main(args(cpu, shifted, "shifted")) == 1
    undeclared = _case(tmp_path / "undeclared", 2)
    _frame(undeclared / "wrfout_d03_2026-02-28_00:00:00", 0)
    assert nc.main(args(cpu, undeclared, "undeclared")) == 1
    assert nc.main(args(cpu, _case(tmp_path / "same", 2), "same")) == 0


def test_spforc_pairs_use_the_joint_mode_amplitude(tmp_path):
    """GPU arm 11.11: d01 SPFORCS3 |d| 2.8e-9 on a 9.3e-5 sine part of a 2.3e-3 mode failed per-field scaling."""
    def case(root, s3):
        _case(root, 1, d02={"rstoch": .05})
        for p in sorted(root.glob("wrfout_d01_*")):
            with netCDF4.Dataset(p, "a") as ds:
                ds["SPFORCC3"][:] = np.float32(2.3e-3); ds["SPFORCS3"][:] = np.float32(s3)
        return root
    cpu = case(tmp_path / "cpu", 9.3e-5)
    args = lambda gpu, n: [str(cpu), str(gpu), str(tmp_path / f"{n}.json"), "--domains", "d01", "d02", "--hours", "1"]
    assert nc.main(args(case(tmp_path / "rounding", 9.3e-5 + 2.8e-9), "rounding")) == 0        # mode-relative 1.2e-6
    assert nc.main(args(case(tmp_path / "real", 9.3e-5 + 2.3e-6), "real")) == 1                 # mode-relative 1e-3
