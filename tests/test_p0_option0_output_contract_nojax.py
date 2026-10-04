"""P0 option-0 history contract, JAX-free (manager da6450c85).

WRF: T = dry theta - T0 under both use_theta_m values; THM = theta_m - T0 (1) or dry theta - T0 (0); integer global
USE_THETA_M (Registry.EM_COMMON:209-211, share/output_wrf.F:679).  These tests never import gpuwrf (whose package
init imports JAX): the numpy-only convention helper is loaded by file path, and the writer wiring is checked on its
source (AST).  The real-writer both-option / subset+full / wrfinput round-trip tests live in
tests/test_p0_option0_writer_paths.py (JAX import; run only under a reviewed package).
Run: python3 -m pytest -q -p no:cacheprovider tests/test_p0_option0_output_contract_nojax.py   (or python3 <file>)
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
IO = ROOT / "src/gpuwrf/io"


def _helper():
    spec = importlib.util.spec_from_file_location("p0_wrfout_theta_convention", IO / "wrfout_theta_convention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_use_theta_m_accepts_only_integral_zero_or_one():
    h = _helper()
    assert h.parse_use_theta_m(None) == 1
    assert h.parse_use_theta_m(1) == 1 and h.parse_use_theta_m(0) == 0
    assert h.parse_use_theta_m(np.int32(0)) == 0 and h.parse_use_theta_m(np.int64(1)) == 1
    for bad in (2, -1, True, False, 0.5, "1", "0", [1]):
        try:
            h.parse_use_theta_m(bad)
        except ValueError as exc:
            assert "WRFOUT_USE_THETA_M_INVALID" in str(exc)
        else:
            raise AssertionError(f"accepted {bad!r}")


def test_thm_source_option1_is_the_same_array_option0_is_dry():
    h = _helper()
    rv = 461.6 / 287.0
    theta_m = np.array([300.5, 301.25], dtype=np.float32)
    qv = np.array([0.012, 0.0], dtype=np.float32)
    theta_dry = theta_m / (1.0 + rv * np.maximum(qv, 0.0))
    assert h.thm_source(theta_m, theta_dry, 1) is theta_m                 # option 1 bitwise unchanged
    assert h.thm_source(theta_m, theta_dry, 0) is theta_dry               # option 0: THM == T (dry)
    thm0 = h.thm_source(theta_m, theta_dry, 0) - 300.0
    t = theta_dry - 300.0
    assert np.array_equal(thm0, t)
    try:
        h.thm_source(theta_m, theta_dry, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("thm_source accepted use_theta_m=2")


def _writer_ast():
    return ast.parse((IO / "wrfout_writer.py").read_text())


def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_both_writer_paths_route_thm_through_the_option_and_keep_t_dry():
    tree = _writer_ast()
    want = "thm_source(theta, theta_dry, parse_use_theta_m(_lookup(namelist, 'use_theta_m', 1))) - P0_THETA_OFFSET_K"
    for name in ("_build_output_fields", "_build_subset_output_fields"):
        src = ast.unparse(_fn(tree, name))
        assert want in src, name
        assert "theta - P0_THETA_OFFSET_K" not in src.replace("theta_dry - P0_THETA_OFFSET_K", ""), name   # no raw THM
        assert "theta_dry - P0_THETA_OFFSET_K" in src, name                                           # T stays dry


def test_wrfout_writes_use_theta_m_attribute_and_restart_caller_is_untouched():
    tree = _writer_ast()
    attrs = ast.unparse(_fn(tree, "_write_global_attrs"))
    assert "attrs['USE_THETA_M'] = np.int32(parse_use_theta_m(use_theta_m))" in attrs
    calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_write_global_attrs"]
    assert len(calls) == 1 and "use_theta_m=parse_use_theta_m(_lookup(prepared.namelist, 'use_theta_m', 1))" in calls[0]
    rst = (IO / "wrfrst_netcdf.py").read_text()                   # M3R owns restart: no partial THM/attr patch here
    assert "use_theta_m" not in rst and "USE_THETA_M" not in rst


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"PASS={len(fns)} FAIL=0")
    sys.exit(0)
