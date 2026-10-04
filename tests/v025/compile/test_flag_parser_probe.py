"""GPU flag validation must not initialize a CUDA context during imports."""
import pytest

from gpuwrf.runtime import xla_autotune


@pytest.mark.parametrize("flag,expected", [
    ("--xla_gpu_enable_command_buffer=+CONDITIONAL "
     "--xla_enable_command_buffers_during_profiling=true", True),
    ("--xla_gpu_force_compilation_parallelism=2", True),
    ("--gpuwrf_deliberately_invalid_flag=1", False),
])
def test_shared_parser_validates_gpu_flags_with_cuda_hidden(monkeypatch, flag, expected):
    # The subprocess itself asserts CPU devices. A legacy CUDA pin must not
    # override the child pin or cause the parent's CUDA context to be inherited.
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ok, detail = xla_autotune.probe_flag_supported(flag)
    assert ok is expected, detail
    if not expected:
        assert "Unknown flag" in detail
