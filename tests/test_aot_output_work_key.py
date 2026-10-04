"""Output persistence placement must reuse the same numeric step artifact."""
from gpuwrf.runtime import aot_cheap_key as ck
from test_aot_cheap_key import _build_call, _cheap_key


def test_async_netcdf_persistence_reuses_actual_call_key(monkeypatch):
    carry, namelist, clock = _build_call()
    keys = []
    for setting in (None, "0", "1", "false", "true"):
        if setting is None:
            monkeypatch.delenv("GPUWRF_NESTED_ASYNC_OUTPUT", raising=False)
        else:
            monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", setting)
        keys.append(_cheap_key(carry, namelist, clock))
    assert len(set(keys)) == 1, "host writer placement caused a warm numeric-artifact miss"
    assert ck.trace_env_is_inert("GPUWRF_NESTED_ASYNC_OUTPUT")
    # Keep memory-plan policy separate from proven numeric work equivalence.
    assert not ck.is_process_infra_env("GPUWRF_NESTED_ASYNC_OUTPUT")


def test_output_exclusion_preserves_physics_flag_keying(monkeypatch):
    carry, namelist, clock = _build_call()
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", "0")
    monkeypatch.delenv("GPUWRF_MOIST_CQW", raising=False)
    baseline = _cheap_key(carry, namelist, clock)
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", "1")
    assert _cheap_key(carry, namelist, clock) == baseline
    monkeypatch.setenv("GPUWRF_MOIST_CQW", "0")
    assert _cheap_key(carry, namelist, clock) != baseline
