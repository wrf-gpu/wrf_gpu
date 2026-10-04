"""CPU-only policy guards; real executable/live-byte fixtures from VR10/VR14."""
from types import SimpleNamespace
import math

import pytest

from gpuwrf.runtime import gpu_allocator as policy


def stats(arguments, outputs, temps):
    return {"argument_size_in_bytes": arguments, "output_size_in_bytes": outputs,
            "temp_size_in_bytes": temps, "alias_size_in_bytes": 0,
            "generated_code_size_in_bytes": 0}


PROD = {"d01": [stats(531276732, 513963976, 588921920)],
        "d02": [stats(499828580, 438768768, 1033365576)]}
PROD_LIVE = 4971842764


def test_actual_prod_live_floor_and_domain_variants():
    budget = policy.pool_budget(PROD, PROD_LIVE)
    assert budget >= PROD_LIVE * 1.20 + policy.HEADROOM_BYTES
    variants = {key: value + value for key, value in PROD.items()}
    assert policy.pool_budget(variants, PROD_LIVE) == budget
    assert budget < 0.20 * 33664860160


def test_larger_wn3_measured_live_floor_fits_with_margin():
    # Manager's WN3 three-nest flags-OFF live highwater; geometry adds one domain.
    programs = dict(PROD, d03=[stats(200000000, 180000000, 1000000000)])
    live = int(10.6 * policy.GIB)
    budget = policy.pool_budget(programs, live)
    assert budget > live * 1.20
    assert budget + policy.HEADROOM_BYTES < 33664860160 - 4 * policy.GIB


def test_plan_rejects_wrong_case_missing_domain_and_edited_budget(tmp_path):
    programs = {key: {"key": values[0]} for key, values in PROD.items()}
    path = tmp_path / "plan.json"
    saved = policy.write_plan(path, "case", 2, programs, PROD_LIVE)
    assert policy.read_plan(path, "case", 2) == saved
    with pytest.raises(ValueError):
        policy.read_plan(path, "different-source-flags-geometry", 2)
    with pytest.raises(ValueError):
        policy.write_plan(path, "case", 3, programs, PROD_LIVE)
    import json
    saved["budget_bytes"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        policy.read_plan(path, "case", 2)


def test_oversized_plan_has_no_fraction():
    plan = {"budget_bytes": 8 * policy.GIB, "programs": PROD}
    with pytest.raises(ValueError, match="free GPU memory"):
        policy.planned_fraction(plan, 32 * policy.GIB, 4 * policy.GIB)


def test_admission_reserves_outside_pool_headroom():
    # VR19: 824 MiB outside the pool -> at least 1 GiB, or 512 MiB + resident code.
    small = {"budget_bytes": 6 * policy.GIB, "programs": PROD}
    with pytest.raises(ValueError, match="free GPU memory"):
        policy.planned_fraction(small, 32 * policy.GIB, 6 * policy.GIB + 1000 * 1024**2)
    assert policy.planned_fraction(small, 32 * policy.GIB, 7 * policy.GIB + 1)
    code = {key: [dict(value[0], generated_code_size_in_bytes=policy.GIB)] for key, value in PROD.items()}
    with pytest.raises(ValueError, match="free GPU memory"):
        policy.planned_fraction({"budget_bytes": 6 * policy.GIB, "programs": code}, 32 * policy.GIB, 8 * policy.GIB)


def test_live_peak_above_applied_pool_warns_and_refreshes(monkeypatch, tmp_path, capsys):
    import jax
    path = tmp_path / "plan.json"
    path.write_text("stale")
    records = {key: {"real": value[0]} for key, value in PROD.items()}
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": PROD_LIVE})])
    for incomplete in (True, False):
        monkeypatch.setattr(policy, "_SESSION", {"key": "case", "path": path, "domains": 2, "programs": records,
                                                 "incomplete": incomplete, "applied_budget": PROD_LIVE - 1})
        policy.finish_cli_pool()
        assert "exceeded the C-auto pool" in capsys.readouterr().err
        assert path.exists() is (not incomplete)
    assert policy.read_plan(path, "case", 2)["budget_bytes"] > PROD_LIVE


def test_geometry_flags_source_each_change_key():
    key = policy.case_key("source", {"native": "1"}, {"d01": [120, 70, 44]})
    assert key != policy.case_key("newsource", {"native": "1"}, {"d01": [120, 70, 44]})
    assert key != policy.case_key("source", {"native": "0"}, {"d01": [120, 70, 44]})
    assert key != policy.case_key("source", {"native": "1"}, {"d01": [121, 70, 44]})


def test_metadata_failure_does_not_raise_or_execute(monkeypatch):
    monkeypatch.setattr(policy, "_SESSION", {"programs": {}, "incomplete": False})
    class Broken:
        def memory_analysis(self):
            raise RuntimeError("statistics unavailable")
    policy.record_executable("d01", Broken(), "key")
    assert policy._SESSION["incomplete"]


@pytest.mark.parametrize("override", [
    {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"},
    {"XLA_CLIENT_MEM_FRACTION": "0.31"},
    {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.31"},
    {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"},
])
def test_operator_overrides_bypass_auto_budget(monkeypatch, override):
    monkeypatch.setattr(policy, "_case_description", lambda args: ({"d01": [120, 70, 44]}, 1))
    monkeypatch.setattr(policy, "_source_fingerprint", lambda: "source")
    monkeypatch.setattr(policy, "_pool_device_memory", lambda: pytest.fail("must not query GPU for override"))
    env = dict(override)
    policy.configure_cli_pool(SimpleNamespace(), env)
    assert all(env[key] == value for key, value in override.items())


def test_cold_missing_plan_uses_demand_without_gpu_query(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(policy, "_case_description", lambda args: ({"d01": [120, 70, 44]}, 1))
    monkeypatch.setattr(policy, "_source_fingerprint", lambda: "source")
    monkeypatch.setattr(policy, "_pool_device_memory", lambda: pytest.fail("missing plan must not query GPU"))
    env = {"GPUWRF_JAX_CACHE_DIR": str(tmp_path)}
    policy.configure_cli_pool(SimpleNamespace(), env)
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and "_GPUWRF_C_AUTO_HEADROOM" not in env
    assert "recording this case" in capsys.readouterr().err


def test_warm_verified_plan_selects_case_fraction_and_reexec_recreates_session(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "_case_description", lambda args: ({"d01": [120, 70, 44]}, 2))
    monkeypatch.setattr(policy, "_source_fingerprint", lambda: "source")
    monkeypatch.setattr(policy, "_pool_device_memory", lambda: (33664860160, 28 * policy.GIB))
    cold = {"GPUWRF_JAX_CACHE_DIR": str(tmp_path)}
    policy.configure_cli_pool(SimpleNamespace(), cold)
    session = policy._SESSION
    records = {key: {"real": value[0]} for key, value in PROD.items()}
    saved = policy.write_plan(session["path"], session["key"], 2, records, PROD_LIVE)
    warm = {"GPUWRF_JAX_CACHE_DIR": str(tmp_path)}
    policy.configure_cli_pool(SimpleNamespace(), warm)
    assert warm["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert float(warm["XLA_CLIENT_MEM_FRACTION"]) * 33664860160 >= saved["budget_bytes"]
    assert float(warm["XLA_CLIENT_MEM_FRACTION"]) < 0.20
    assert warm["_GPUWRF_C_AUTO_HEADROOM"] == str(policy.outside_headroom(saved)) == str(policy.GIB)
    assert not policy.configure_cli_pool(SimpleNamespace(), warm)
    assert policy._SESSION["applied_budget"] == saved["budget_bytes"]


def test_oom_invalidates_only_case_plan_for_next_demand_launch(monkeypatch, tmp_path, capsys):
    path = tmp_path / "this-case.json"
    path.write_text("old")
    sibling = tmp_path / "other-case.json"
    sibling.write_text("keep")
    monkeypatch.setattr(policy, "_SESSION", {"path": path, "applied_budget": 6 * policy.GIB})
    policy.failed_cli_pool(RuntimeError("RESOURCE_EXHAUSTED: Out of memory"))
    assert not path.exists() and sibling.read_text() == "keep"
    assert "next CLI launch uses demand" in capsys.readouterr().err


def test_unused_prefetch_allowance_is_one_variant_per_domain():
    records = {"d01": [dict(stats(500, 500, 500), generated_code_size_in_bytes=120),
                       dict(stats(500, 500, 900), generated_code_size_in_bytes=310)],
               "d02": [dict(stats(500, 500, 500), generated_code_size_in_bytes=40)]}
    assert policy.speculative_allowance(records) == 310 + 40
    assert policy.pool_budget(records, 0) == math.ceil((1900 + 1500 + 350) * policy.SAFETY_FACTOR) + policy.HEADROOM_BYTES


def test_device_memory_uses_cuda_visible_total(monkeypatch):
    # RTX 5090: NVML 32607 MiB total, 502 MiB driver reservation = CUDA total 33664860160 B.
    out = SimpleNamespace(stdout="32607, 502, 27019\n")
    monkeypatch.setattr(policy.subprocess, "run", lambda *a, **k: out)
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD", raising=False)
    total, free = policy._pool_device_memory()
    # NVML rounds to MiB: never above CUDA's total, so the fraction rounds up.
    assert 33664860160 - 1024**2 < total <= 33664860160 and free == 27019 * 1024**2


def test_successful_gpu_run_saves_plan_without_dev_lock(monkeypatch, tmp_path):
    import jax
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": PROD_LIVE})
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD", raising=False)
    path = tmp_path / "plan.json"
    records = {key: {"real": value[0]} for key, value in PROD.items()}
    monkeypatch.setattr(policy, "_SESSION", {"key": "case", "path": path, "domains": 2,
                                             "programs": records, "incomplete": False})
    policy.finish_cli_pool()
    assert policy.read_plan(path, "case", 2)["live_peak_bytes"] == PROD_LIVE


def test_cpu_or_incomplete_run_saves_nothing(monkeypatch, tmp_path):
    import jax
    path = tmp_path / "plan.json"
    records = {key: {"real": value[0]} for key, value in PROD.items()}
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(policy, "_SESSION", {"key": "case", "path": path, "domains": 2,
                                             "programs": records, "incomplete": False})
    policy.finish_cli_pool()
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": PROD_LIVE})])
    monkeypatch.setattr(policy, "_SESSION", {"key": "case", "path": path, "domains": 2,
                                             "programs": records, "incomplete": True})
    policy.finish_cli_pool()
    assert not path.exists()


def _case_key(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(policy, "_case_description", lambda args: ({"d01": [120, 70, 44]}, 1))
    monkeypatch.setattr(policy, "_source_fingerprint", lambda: "source")
    monkeypatch.setattr(policy, "_pool_device_memory", lambda: pytest.fail("missing plan must not query GPU"))
    policy.configure_cli_pool(SimpleNamespace(), {"GPUWRF_JAX_CACHE_DIR": str(tmp_path), **extra})
    return policy._SESSION["key"]


@pytest.mark.parametrize("name,value", [("GPUWRF_GPU_ARM_CPUS", "10,11,12,13,26,27,28,29"),
                                        ("GPUWRF_BENCH_FLAGS_ENV", "scripts/bench/flags_lw9.env"),
                                        ("GPUWRF_MIN_FREE_VRAM_GIB", "8")])
def test_process_infra_env_never_changes_case_key(monkeypatch, tmp_path, name, value):
    """Process placement / bench bookkeeping: same programs, same memory, same plan."""
    base = _case_key(monkeypatch, tmp_path, {})
    assert _case_key(monkeypatch, tmp_path, {name: value}) == base, f"{name} changed the case key"
    assert _case_key(monkeypatch, tmp_path, {"GPUWRF_MOIST_CQW": "0"}) != base


@pytest.mark.parametrize("name,value", [
    ("GPUWRF_NESTED_FUSE", "0"), ("GPUWRF_NESTED_DEFUSE_COMPILE", "1"),
    ("GPUWRF_ADVANCE_CHUNK_LOOP", "scan"), ("GPUWRF_TRAINING_OUTPUT_SUBSET", "1"),
    ("GPUWRF_NESTED_AOT", "0"), ("GPUWRF_PROFILE", "1"), ("GPUWRF_AOT_VERIFY", "1"),
])
def test_case_key_separates_program_shaping_env(monkeypatch, tmp_path, name, value):
    """HLO-keyed-elsewhere knobs still change which programs run and what they hold."""
    assert _case_key(monkeypatch, tmp_path, {name: value}) != _case_key(monkeypatch, tmp_path, {})
