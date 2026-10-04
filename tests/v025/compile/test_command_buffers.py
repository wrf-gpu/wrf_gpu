"""CPU unit gates for package-wide GPU capture flag policy."""
from gpuwrf.runtime import xla_autotune as xa


def prepare(monkeypatch, flags="", gpu=True):
    monkeypatch.setenv("XLA_FLAGS", flags)
    monkeypatch.setenv("GPUWRF_XLA_CONDITIONAL_GRAPHS", "1")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_PROBE", "1")
    monkeypatch.setattr(xa, "_gpu_is_target", lambda: (gpu, "test-target"))
    calls = []
    monkeypatch.setattr(xa, "probe_flag_supported", lambda flags: (calls.append(flags) or True, "accepted"))
    return calls


def test_cpu_pin_does_not_probe_or_modify_flags(monkeypatch):
    calls = prepare(monkeypatch, "--existing=value", gpu=False)
    result = xa.configure_command_buffers(default_on=True)
    assert not result["enabled"]
    assert not calls
    assert xa.os.environ["XLA_FLAGS"] == "--existing=value"


def test_additive_capture_and_profile_share_one_probe(monkeypatch):
    calls = prepare(monkeypatch)
    result = xa.configure_command_buffers()
    assert result["enabled"]
    assert calls == ["--xla_gpu_enable_command_buffer=+CONDITIONAL --xla_enable_command_buffers_during_profiling=true"]
    assert xa.os.environ["XLA_FLAGS"] == calls[0]
    again = xa.configure_command_buffers()
    assert again["enabled"] and again["injected_flags"] == []
    assert len(calls) == 1


def test_explicit_empty_capture_is_respected(monkeypatch):
    calls = prepare(monkeypatch, "--xla_gpu_enable_command_buffer=")
    result = xa.configure_command_buffers(default_on=True)
    assert not result["enabled"]
    assert calls == ["--xla_enable_command_buffers_during_profiling=true"]
    assert xa.os.environ["XLA_FLAGS"].startswith("--xla_gpu_enable_command_buffer= ")


def test_operator_flag_values_are_preserved(monkeypatch):
    flags = "--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL --xla_enable_command_buffers_during_profiling=false"
    calls = prepare(monkeypatch, flags)
    result = xa.configure_command_buffers()
    assert result["enabled"] and not calls
    assert xa.os.environ["XLA_FLAGS"] == flags


def test_build_rejection_leaves_environment_unchanged(monkeypatch):
    prepare(monkeypatch, "--existing=value")
    monkeypatch.setattr(xa, "probe_flag_supported", lambda flags: (False, "unsupported"))
    result = xa.configure_command_buffers(default_on=True)
    assert not result["enabled"] and result["reason"] == "unsupported-flags"
    assert xa.os.environ["XLA_FLAGS"] == "--existing=value"


def test_explicit_opt_out_wins_over_product_default(monkeypatch):
    calls = prepare(monkeypatch)
    monkeypatch.setenv("GPUWRF_XLA_CONDITIONAL_GRAPHS", "0")
    result = xa.configure_command_buffers(default_on=True)
    assert not result["enabled"] and not calls
    assert xa.os.environ["XLA_FLAGS"] == ""
