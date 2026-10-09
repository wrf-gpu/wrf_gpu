"""Pristine REAL4 DMP shallow-cloud KTOP bound on 32 real RC night/onset columns."""
from pathlib import Path
import ast
import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_sgs_cloud as G
from gpuwrf.runtime import aot_cheap_key as K

FLAG = "GPUWRF_MYNN_DMP_KTOP_BOUND"
FIXTURE = Path(__file__).parent / "fixtures/mynn_dmp_ktop_wrf.json"
PACKET = json.loads(FIXTURE.read_text())
FIELDS = tuple(PACKET["rows"][0]["input"]) + ("reference_qc_bl", "reference_cldfra_bl")


@pytest.fixture(autouse=True)
def reset_flag_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def gate(candidate, reference, inputs):
    for side, fields in (("candidate", candidate), ("reference", reference), ("input", inputs)):
        for name, value in fields.items():
            assert np.isfinite(value).all(), f"{side} {name} nonfinite"
    assert candidate.keys() == reference.keys()
    for name in candidate:
        np.testing.assert_allclose(candidate[name], reference[name],
            atol=PACKET["frozen_bounds"][name + "_abs"],
            rtol=PACKET["frozen_bounds"][name + "_rel"], err_msg=name)


@pytest.mark.parametrize("row", PACKET["rows"], ids=lambda r: f"tau{r['tau']}")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["native", "retained"])
def test_original_real4_night_onset_bound(monkeypatch, row, dtype):
    monkeypatch.setattr(G, "_DMP_KTOP_BOUND", True)
    jax.clear_caches()
    x = {n: jnp.asarray(v, dtype) for n, v in row["input"].items()}
    fn = jax.jit(G.dmp_shallow_cu_overwrite)
    q, c = fn(**x)
    gate({"qc_bl": np.asarray(q), "cldfra_bl": np.asarray(c)},
         {n: np.asarray(v, np.float32) for n, v in row["reference"].items()}, x)
    # Original CB diagnoses above KTOP must be held, including nonzero Sc.
    nz = x["qc_bl"].shape[-1]
    top = np.max(np.where(np.asarray(x["edmf_a"]) > 0, np.arange(nz) + 1, 0), axis=-1)
    mask = np.arange(nz)[None, :] >= top[:, None]
    np.testing.assert_array_equal(np.asarray(q)[mask], np.asarray(x["qc_bl"])[mask])
    np.testing.assert_array_equal(np.asarray(c)[mask], np.asarray(x["cldfra_bl"])[mask])


def test_off_is_original_overwrite(monkeypatch):
    monkeypatch.setattr(G, "_DMP_KTOP_BOUND", False)
    jax.clear_caches()
    tree = ast.parse(inspect.getsource(G.dmp_shallow_cu_overwrite))
    fn = tree.body[0]
    fn.body = [node for node in fn.body if not (
        isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == "_DMP_KTOP_BOUND")]
    namespace = dict(G.__dict__)
    exec(compile(ast.fix_missing_locations(tree), "pre-BP98-overwrite", "exec"), namespace)
    old = namespace["dmp_shallow_cu_overwrite"]
    for row in PACKET["rows"]:
        x = {n: jnp.asarray(v, jnp.float32) for n, v in row["input"].items()}
        a = jax.jit(G.dmp_shallow_cu_overwrite)(**x)
        b = jax.jit(old)(**x)
        for candidate, reference in zip(a, b):
            assert np.isfinite(candidate).all() and np.isfinite(reference).all()
            np.testing.assert_array_equal(candidate, reference)


def test_resolved_import_flag_is_in_cheap_key(monkeypatch):
    assert ("gpuwrf.physics.mynn_sgs_cloud", "_DMP_KTOP_BOUND") in K.IMPORT_TIME_ENV_CONSTANTS
    monkeypatch.setattr(G, "_DMP_KTOP_BOUND", False)
    key0 = K.module_const_env_hash()
    monkeypatch.setenv(FLAG, "1")
    assert K.module_const_env_hash() == key0, "live env must not replace imported authority"
    monkeypatch.setattr(G, "_DMP_KTOP_BOUND", True)
    assert K.module_const_env_hash() != key0, "resolved HLO flag omitted from cheap key"


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_gate_rejects_nonfinite_input_or_reference(field, bad):
    row = PACKET["rows"][0]
    x = {n: np.asarray(v, np.float32) for n, v in row["input"].items()}
    ref = {n: np.asarray(v, np.float32) for n, v in row["reference"].items()}
    candidate = {n: v.copy() for n, v in ref.items()}
    if field.startswith("reference_"):
        ref[field.removeprefix("reference_")].flat[0] = bad
    else:
        x[field].flat[0] = bad
    with pytest.raises(AssertionError, match="nonfinite"):
        gate(candidate, ref, x)


@pytest.mark.parametrize("field", ["qc_bl", "cldfra_bl"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_gate_rejects_nonfinite_candidate(field, bad):
    row = PACKET["rows"][0]
    ref = {n: np.asarray(v, np.float32) for n, v in row["reference"].items()}
    candidate = {n: v.copy() for n, v in ref.items()}
    candidate[field].flat[0] = bad
    with pytest.raises(AssertionError, match="candidate .* nonfinite"):
        gate(candidate, ref, {})
