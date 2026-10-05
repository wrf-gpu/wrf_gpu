"""Install prefixes must not fragment serialized Pallas programs."""
import io

import jax
import numpy as np
import pytest

from gpuwrf.runtime import compile_cache as cache


@pytest.fixture(autouse=True)
def restore_source_config(monkeypatch):
    old = jax.config.jax_hlo_source_file_canonicalization_regex
    monkeypatch.delenv("JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX", raising=False)
    jax.config.update("jax_hlo_source_file_canonicalization_regex", None)
    yield
    jax.config.update("jax_hlo_source_file_canonicalization_regex", old)


def test_supported_default_preserves_operator_override(monkeypatch):
    status = cache._configure_source_paths()
    assert status == {"source": "gpuwrf", "regex": cache._SOURCE_PATH_REGEX}
    jax.config.update("jax_hlo_source_file_canonicalization_regex", "custom-prefix")
    assert cache._configure_source_paths() == {"source": "operator", "regex": "custom-prefix"}
    jax.config.update("jax_hlo_source_file_canonicalization_regex", None)
    monkeypatch.setenv("JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX", "")
    assert cache._configure_source_paths()["source"] == "operator"
    assert jax.config.jax_hlo_source_file_canonicalization_regex is None


def _triton_bytes(source, *, increment=1, blank_lines=0):
    pl = pytest.importorskip("jax.experimental.pallas")
    plt = pytest.importorskip("jax.experimental.pallas.triton")
    lowering = pytest.importorskip("jax._src.pallas.triton.lowering")
    namespace = {}
    code = "\n" * blank_lines + (
        "def kernel(x_ref, y_ref):\n"
        f"    y_ref[...] = x_ref[...] * 2.0 + {increment}.0\n"
    )
    exec(compile(code, source, "exec"), namespace)
    fn = pl.pallas_call(namespace["kernel"],
                        out_shape=jax.ShapeDtypeStruct((128,), np.float32),
                        grid=(), compiler_params=plt.CompilerParams())
    closed = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((128,), np.float32))

    def find(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                return eqn.params
            for child in eqn.params.values():
                inner = getattr(child, "jaxpr", child)
                if hasattr(inner, "eqns"):
                    found = find(inner)
                    if found is not None:
                        return found
        return None

    params = find(closed.jaxpr)
    assert params is not None
    result = lowering.lower_jaxpr_to_triton_module(
        params["jaxpr"], params["grid_mapping"], "cuda", 120)
    buf = io.BytesIO()
    result.module.operation.write_bytecode(buf)
    return buf.getvalue()


def test_real_triton_bytecode_reuses_install_path_but_keeps_program_identity():
    first = "/install-a/src/gpuwrf/kernels/example.py"
    second = "/install-b/site-packages/gpuwrf/kernels/example.py"
    assert _triton_bytes(first) != _triton_bytes(second)  # Genuine miss control.
    assert cache._configure_source_paths()["source"] == "gpuwrf"
    canonical = _triton_bytes(first)
    assert canonical == _triton_bytes(second)
    assert first.encode() not in canonical and second.encode() not in canonical
    assert b"/gpuwrf/kernels/example.py" in canonical
    assert canonical != _triton_bytes(second, increment=2)  # Changed computation.
    assert canonical != _triton_bytes(second, blank_lines=1)  # Keep source lines.
    assert canonical != _triton_bytes("/install-b/site-packages/gpuwrf/kernels/other.py")
