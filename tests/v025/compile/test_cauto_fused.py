"""CPU fixtures exercise real fused compile/load telemetry and CLI plan reuse."""
import json
from types import SimpleNamespace

import pytest

from gpuwrf.runtime import gpu_allocator as policy


def memory(arguments, outputs, temps, code=0):
    return dict(argument_size_in_bytes=arguments, output_size_in_bytes=outputs,
                temp_size_in_bytes=temps, alias_size_in_bytes=0,
                generated_code_size_in_bytes=code)


class Executable:
    def __init__(self, record, loaded=False):
        self.stats = SimpleNamespace(**record)
        self.reads = 0
        if loaded:
            self.loaded_executable = SimpleNamespace(get_compiled_memory_stats=self.memory_analysis)

    def memory_analysis(self):
        self.reads += 1
        return self.stats

    def __call__(self, parent, children, *_):
        return parent, children


@pytest.fixture
def cli(monkeypatch, tmp_path):
    import jax
    monkeypatch.setattr(policy, "_case_description", lambda args: ({"domains": "three-nest-fixture"}, 3))
    monkeypatch.setattr(policy, "_source_fingerprint", lambda: "fused-fixture-source")
    monkeypatch.setattr(policy, "_pool_device_memory", lambda: (32 * policy.GIB, 30 * policy.GIB))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": 2 * policy.GIB})])
    monkeypatch.setattr(policy, "_SESSION", None)
    env = {"GPUWRF_JAX_CACHE_DIR": str(tmp_path)}
    policy.configure_cli_pool(SimpleNamespace(), env)
    return tmp_path, policy._SESSION


@pytest.mark.parametrize("loaded", [False, True], ids=["cold-compile", "warm-AOT-load"])
def test_real_fused_path_saves_three_domain_plan_then_selects_capped_pool(monkeypatch, cli, loaded):
    from gpuwrf.runtime import domain_tree as dt, aot_precompile as aotp, aot_executable as ax
    directory, session = cli
    d01 = memory(300_000_000, 250_000_000, 100_000_000, 10_000_000)
    fused = memory(600_000_000, 550_000_000, 900_000_000, 20_000_000)
    executable = Executable(fused, loaded=loaded)
    policy.record_executable("d01", Executable(d01), "parent-key")
    monkeypatch.setattr(dt, "_nested_aot_enabled", lambda: True)
    monkeypatch.setattr(dt, "_nested_aot_verify_enabled", lambda: False)
    monkeypatch.setattr(dt, "build_clock_base", lambda _: 0)
    monkeypatch.setattr(dt, "_build_fused_aux_namelist", lambda **_: object())
    monkeypatch.setattr(dt, "_aval_signature", lambda _: "shape-signature")
    monkeypatch.setattr(dt, "_fused_cascade_cheap_key", lambda *_: "fused-key")
    monkeypatch.setattr(dt, "_record_nested_aot_status", lambda *_: None)
    monkeypatch.setattr(dt, "_log_nested_aot_status", lambda *_: None)
    monkeypatch.setattr(aotp, "load_domain_blob", lambda *_a, **_k:
                        (executable if loaded else None, {}))
    monkeypatch.setattr(aotp, "_serialize_domain_blob", lambda *_a, **_k: {"aot_written": True})
    monkeypatch.setattr(ax, "hlo_sha256_from_lowered", lambda _: "fixture-hlo")
    lower = SimpleNamespace(compile=lambda *_: executable)
    monkeypatch.setattr(dt.jax, "jit", lambda _: SimpleNamespace(lower=lambda *_: lower))
    advance = dt._build_fused_cascade_program(
        parent_name="d02", parent_namelist=SimpleNamespace(), parent_cadence=1,
        child_names=("d03",), child_namelists=(SimpleNamespace(),),
        child_weights=(None,), child_bdy_widths=(5,), child_ratios=(3,), child_cadences=(1,))
    assert advance(1, (2,), 1, (1,)) == (1, (2,))
    assert advance(1, (2,), 2, (4,)) == (1, (2,))
    assert executable.reads == 1  # metadata is outside subsequent timesteps
    policy.finish_cli_pool()
    assert session["path"].exists(), "fused execution must persist a usable C-auto plan"
    saved = policy.read_plan(session["path"], session["key"], 3)
    assert saved["domain_programs"] == {"d01": ["d01"], "d02": ["fused/d02"], "d03": ["fused/d02"]}
    assert set(saved["programs"]) == {"d01", "fused/d02"}
    assert saved["budget_bytes"] == policy.pool_budget({"d01": [d01], "fused/d02": [fused]}, 2 * policy.GIB)
    assert saved["speculative_bytes"] == 30_000_000  # fused code counted once
    warm = {"GPUWRF_JAX_CACHE_DIR": str(directory)}
    policy.configure_cli_pool(SimpleNamespace(), warm)
    assert warm["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert warm["_GPUWRF_C_AUTO_BUDGET"] == str(saved["budget_bytes"])


def test_missing_fused_child_is_refused_with_loud_reason(cli, capsys):
    _, session = cli
    policy.record_executable("d01", Executable(memory(100, 100, 100)), "parent")
    policy.record_executable("fused/d02", Executable(memory(200, 200, 200)), "fused",
                             covered_domains=("d02",))
    policy.finish_cli_pool()
    assert not session["path"].exists()
    warning = capsys.readouterr().err
    assert "WARNING: C-auto case plan not saved" in warning and "d03" in warning


def test_statistics_failure_explains_incomplete_plan(cli, capsys):
    class Broken:
        def memory_analysis(self):
            raise RuntimeError("fixture statistics unavailable")
    _, session = cli
    policy.record_executable("fused/d02", Broken(), "broken", covered_domains=("d02", "d03"))
    policy.finish_cli_pool()
    assert not session["path"].exists()
    warning = capsys.readouterr().err
    assert "WARNING" in warning and "fused/d02: RuntimeError: fixture statistics unavailable" in warning
    assert "case plan not saved" in warning


def test_fused_plan_rejects_edited_or_out_of_case_coverage(tmp_path):
    records = {"d01": {"p": memory(100, 100, 100)},
               "fused/d02": {"f": dict(memory(200, 200, 200), domains=["d02", "d03"])}}
    path = tmp_path / "plan.json"
    saved = policy.write_plan(path, "case", 3, records, 0)
    saved["domain_programs"]["d03"] = []
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        policy.read_plan(path, "case", 3)
    records["fused/d02"]["f"]["domains"] = ["d02", "d04"]
    with pytest.raises(ValueError, match="unknown domain d04"):
        policy.write_plan(path, "case", 3, records, 0)


def test_legacy_unfused_two_domain_plan_remains_readable(tmp_path):
    records = {d: {"p": memory(100, 100, 100)} for d in ("d01", "d02")}
    path = tmp_path / "plan.json"
    saved = policy.write_plan(path, "case", 2, records, 0)
    del saved["domain_programs"]
    path.write_text(json.dumps(saved))
    assert policy.read_plan(path, "case", 2) == saved
