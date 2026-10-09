"""Coverage controls for the actual wave gate, no CUDA runtime required."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/wave_cubin_gate.py"
spec = importlib.util.spec_from_file_location("wave_cubin_gate_tested", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def put(path, data=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def fixture_cache(tmp_path):
    cache = tmp_path / "cache"
    put(cache / "jit__advance_chunk_fori-parent-cache")
    put(cache / "aot/tag/d01/root.xlaexec")
    put(cache / "aot/tag/fused_d02/child.xlaexec")
    put(cache / "aot/tag/init_solve_rrtmg_lw_column/aux.xlaexec")
    return cache


def record(entry, out, db):
    return {"path": str(entry.path), "program": entry.program, "source_kind": entry.kind,
            "elfs": [{"stack_over256": [], "nonleaf_over1k": [], "max_stack_B": 32}]}


def test_default_gate_includes_child_fused_program(tmp_path):
    cache = fixture_cache(tmp_path)
    result = gate.run_gate(cache, tmp_path / "out", scanner=record)
    assert result["coverage"]["aot_programs"] == ["d01", "fused_d02"]
    assert result["coverage"]["n_scanned"] == 3  # root JIT, root AOT, child AOT.
    assert result["hard_stack_gate"] == "PASS"
    assert not any("init_solve" in row["path"] for row in result["entries"])


@pytest.mark.parametrize("mutation", ["drop_result", "empty_elf", "wrong_program"])
def test_actual_gate_fails_if_child_is_missing_from_scan(tmp_path, mutation):
    cache = fixture_cache(tmp_path)
    def omit_child(entry, out, db):
        row = record(entry, out, db)
        if entry.program == "fused_d02":
            if mutation == "drop_result":
                row["path"] = "unrelated-parent-path"
            elif mutation == "empty_elf":
                row["elfs"] = []
            else:
                row["program"] = "d01"
        return row
    with pytest.raises(ValueError, match="unscanned stepping|absent from scan"):
        gate.run_gate(cache, tmp_path / "out", scanner=omit_child)
    assert not (tmp_path / "out/wave_gate.json").exists()


def test_declared_child_missing_blob_refuses_root_only_pass(tmp_path):
    cache = fixture_cache(tmp_path)
    child = cache / "aot/tag/fused_d02/child.xlaexec"
    child.unlink()
    put(child.with_suffix(".meta"))
    with pytest.raises(ValueError, match="missing AOT stepping blob"):
        gate.run_gate(cache, tmp_path / "out", scanner=record)


def test_run_proof_refuses_entire_child_directory_missing(tmp_path):
    cache = tmp_path / "cache"
    put(cache / "aot/tag/d01/root.xlaexec")
    proof = tmp_path / "run.json"
    proof.write_text(json.dumps({"domains": ["d01", "d02", "d03"],
        "metadata": {"nested_aot": {"domains": {"d01": {}, "fused/d02": {}}}}}))
    with pytest.raises(ValueError, match="fused_d02.*absent from scan|absent from scan.*fused_d02"):
        gate.run_gate(cache, tmp_path / "out", run_proof=proof, scanner=record)


def test_aot_alias_hardlinks_are_scanned_once_per_program(tmp_path):
    cache = fixture_cache(tmp_path)
    original = cache / "aot/tag/fused_d02/child.xlaexec"
    alias = original.with_name("k_cheap.xlaexec")
    alias.hardlink_to(original)
    found = [entry for entry in gate.discover(cache) if entry.program == "fused_d02"]
    assert len(found) == 1
    assert len(found[0].aliases) == 1


def test_legacy_domain_aot_blob_is_included(tmp_path):
    cache = tmp_path / "cache"
    put(cache / "aot/tag/d02.xlaexec")
    assert [(entry.kind, entry.program) for entry in gate.discover(cache)] == [("aot", "d02")]


def test_empty_cache_or_program_does_not_pass(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    with pytest.raises(ValueError, match="no stepping executable"):
        gate.run_gate(cache, tmp_path / "out", scanner=record)
    put(cache / "aot/tag/fused_d02/child.xlaexec", b"")
    with pytest.raises(ValueError, match="empty AOT"):
        gate.run_gate(cache, tmp_path / "out", scanner=record)


def test_hard_stack_failure_is_preserved(tmp_path):
    cache = fixture_cache(tmp_path)
    def high_stack(entry, out, db):
        row = record(entry, out, db)
        if entry.program == "fused_d02":
            row["elfs"][0]["stack_over256"] = [("child", 328)]
        return row
    assert gate.run_gate(cache, tmp_path / "out", scanner=high_stack)["hard_stack_gate"] == "FAIL"


def test_default_producer_inventory_catches_deleted_child_directory(tmp_path):
    wave = tmp_path / "wave"
    cache = wave / "cache/jax"
    put(cache / "aot/tag/d01/root.xlaexec")
    proof = wave / "runs/case/wrfout/proofs/nested_pipeline_run.json"
    put(proof, json.dumps({"metadata": {"nested_aot": {"domains": {"d01": {}, "fused/d02": {}}}}}).encode())
    with pytest.raises(ValueError, match="absent from scan"):
        gate.run_gate(cache, wave / "cubin", scanner=record)


@pytest.mark.parametrize("field", ["stack", "sass", "calls", "callees"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1, True])
def test_resource_gate_rejects_invalid_counts_before_comparison(field, bad):
    row = {"stack": 32, "sass": 100, "calls": 0, "callees": 0}
    row[field] = bad
    with pytest.raises(ValueError, match="invalid cubin resource count"):
        gate.checked_resource_rows([row])


def test_resource_gate_refuses_empty_disassembler_result():
    with pytest.raises(ValueError, match="no measured kernel resources"):
        gate.checked_resource_rows([])


def test_cli_hard_stack_failure_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(gate, "run_gate", lambda *a: {"hard_stack_gate": "FAIL", "conservative_flags": False,
        "n_entries": 1, "coverage": {"aot_programs": ["d01"]}})
    assert gate.main([str(tmp_path), str(tmp_path / "out")]) == 2
