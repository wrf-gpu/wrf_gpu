"""Actual MYNN probe loaders accept omitted optional pi and still require science leaves."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from mynn_wrf_gates import inputs
from test_mynn_dheat import batch, FIXTURE


def column_arrays(with_exner, with_ni=False):
    state, flux = batch(json.loads(FIXTURE.read_text())["rows"][0])
    state = state.replace(**{n: getattr(state, n)[:4] for n in state.__slots__
                             if getattr(state, n) is not None})
    if with_exner:
        state = state.replace(exner=P._exner_from_pressure(state.p))
    if with_ni:
        from test_mynn_qni import DATA
        state = state.replace(ni=jnp.asarray(DATA["rows"][0]["input"]["ni"],state.qv.dtype))
    arrays = {"state_" + n: np.asarray(getattr(state, n)) for n in state.__slots__
              if getattr(state, n) is not None}
    arrays.update({"flux_" + n: np.asarray(getattr(flux, n))[:4]
                   for n in flux._fields if getattr(flux, n) is not None})
    return state, arrays


@pytest.mark.parametrize("with_exner", [False, True])
@pytest.mark.parametrize("with_ni", [False, True])
def test_real_mean_gate_loader_missing_or_present_optional_pi(tmp_path, with_exner, with_ni):
    state, arrays = column_arrays(with_exner,with_ni)
    (tmp_path / "BP28").mkdir()
    (tmp_path / "columns_p0_night").mkdir()
    (tmp_path / "BP28/manifest.json").write_text(json.dumps({"night_d03": {"columns": [0, 3]}}))
    np.savez(tmp_path / "columns_p0_night/d03_whole.npz", **arrays)
    got, _, _ = inputs(tmp_path, "night", "d03", dtype=jnp.float32)
    for n in state.__slots__:
        before = getattr(state, n)
        if before is None:
            assert getattr(got, n) is None
        else:
            np.testing.assert_array_equal(getattr(got, n), np.asarray(before)[[0, 3]])
    del arrays["state_theta"]
    np.savez(tmp_path / "columns_p0_night/d03_whole.npz", **arrays)
    with pytest.raises(KeyError, match="state_theta"):
        inputs(tmp_path, "night", "d03")


@pytest.mark.parametrize("with_exner", [False, True])
@pytest.mark.parametrize("with_ni", [False, True])
def test_actual_whole_probe_load_and_save_statements(tmp_path, with_exner, with_ni):
    state, arrays = column_arrays(with_exner,with_ni)
    source = Path(__file__).with_name("mynn_whole_probe.py")
    tree = ast.parse(source.read_text())
    loader = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets))
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "savez"]
    assert len(saves) == 1
    namespace = dict(P=P, arrays={n: jnp.asarray(v) for n, v in arrays.items()}, np=np,
                     args=SimpleNamespace(output=tmp_path))
    exec(compile(ast.Module([loader], []), str(source), "exec"), namespace)
    got = namespace["state"]
    assert (got.exner is None) == (state.exner is None)
    namespace["result"] = (got, jnp.ones((4,), jnp.float32))
    exec(compile(ast.Module(saves, []), str(source), "exec"), namespace)
    with np.load(tmp_path / "outputs.npz", allow_pickle=False) as output:
        assert ("exner" in output.files) == with_exner
        assert ("ni" in output.files) == with_ni
        for name in got.__slots__:
            value = getattr(got, name)
            if value is None:
                assert name not in output.files
            else:
                assert output[name].dtype != object
                np.testing.assert_array_equal(output[name], np.asarray(value))
    del namespace["arrays"]["state_p"]
    with pytest.raises(KeyError, match="state_p"):
        exec(compile(ast.Module([loader], []), str(source), "exec"), namespace)
