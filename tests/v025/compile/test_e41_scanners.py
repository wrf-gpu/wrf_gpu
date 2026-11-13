"""Independent text/native fixtures and corruption controls for E41 tools."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/wave_cubin_gate.py"
spec = importlib.util.spec_from_file_location("e41_gate_tested", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
cubin = gate._scanner("cubin_scan")
hlo = gate._scanner("hlo_scan")
FIXTURES = Path(__file__).with_name("fixtures") / "e41"


def test_committed_cuobjdump_fixture():
    rows = cubin.scan_text((FIXTURES / "callgraph.resources.txt").read_text(),
                           (FIXTURES / "callgraph.sass.txt").read_text())
    assert [row["stack"] for row in rows] == [0, 2144]
    assert rows[0]["callee_rows"] == [{"function": "large_root", "target_pc": 64,
                                      "instructions": 4, "nested_calls": 0}]
    assert rows[0]["callgraph_status"] == "PASS"


def disassembly(ops, name="root"):
    return f"Function : {name}\n" + "\n".join(f"/*{i * 16:04x}*/ {op}" for i, op in enumerate(ops))


def test_large_root_calling_small_leaf_is_not_a_large_nonleaf_callee():
    ops = ["CALL.REL.NOINC 0x4b10 ;"] + ["NOP ;"] * 1200 + ["EXIT ;", "RET.REL.NODEC R20, 0x0 ;"]
    # Leaf starts after the root's EXIT; the unused gap is not callee work.
    ops[0] = f"CALL.REL.NOINC 0x{1202 * 16:x} ;"
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["sass"] > 1000 and result["calls"] == 1
    assert result["callee_rows"][0]["instructions"] == 1
    assert result["nonleaf_over1k"] == [] and result["callgraph_status"] == "PASS"


@pytest.mark.parametrize("n, verdict", [(1000, "PASS"), (1001, "FAIL")])
def test_nested_callee_sass_limit_is_strictly_greater_than_1000(n, verdict):
    target = (4 + n) * 16
    ops = ["CALL.REL.NOINC 0x40 ;", "EXIT ;", "NOP ;", "NOP ;",
           f"CALL.REL.NOINC 0x{target:x} ;"] + ["NOP ;"] * (n - 2) + ["RET.REL.NODEC R20, 0x0 ;", "RET.REL.NODEC R20, 0x0 ;"]
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callgraph_status"] == verdict
    assert result["callee_rows"][0]["instructions"] == n
    assert result["callee_rows"][0]["nested_calls"] == 1


def test_branch_skips_unreachable_instructions_in_callee():
    ops = ["CALL.REL.NOINC 0x20 ;", "EXIT ;", "BRA 0x3ed0 ;"] + ["NOP ;"] * 1002 + ["RET.REL.NODEC R20, 0x0 ;"]
    ops[2] = f"BRA 0x{(len(ops) - 1) * 16:x} ;"
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callee_rows"][0]["instructions"] == 2
    assert result["callgraph_status"] == "PASS"


@pytest.mark.parametrize("op", ["CALL.ABS.NOINC R20 ;", "CALL.REL.NOINC 0xffff ;"])
def test_unresolved_call_is_unknown(op):
    result = cubin.analyze_calls(cubin.parse_sass(disassembly([op, "EXIT ;"])))["root"]
    assert result["callgraph_status"] == "UNKNOWN"
    assert result["callgraph_unknown"]


def test_symbol_callee_is_resolved():
    sass = disassembly(["CALL.REL.NOINC `(leaf)` ;", "EXIT ;"]) + "\n" + disassembly(["RET ;"], "leaf")
    result = cubin.analyze_calls(cubin.parse_sass(sass))["root"]
    assert result["callgraph_status"] == "PASS"
    assert result["callee_rows"][0]["function"] == "leaf"


@pytest.mark.parametrize("predicate", ["@UP0", "@!UP3", "@PT", "@UPT", "@!P0"])
def test_uniform_predicates_from_actual_sm120_sass(predicate):
    result = cubin.analyze_calls(cubin.parse_sass(disassembly([f"{predicate} UMOV.64 UR6, UR8 ;", "EXIT ;"])))["root"]
    assert result["sass"] == 2 and result["callgraph_status"] == "PASS"


def test_direct_convergence_barrier_edges_bound_callee_instructions():
    ops = ["CALL.REL.NOINC 0x20 ;", "EXIT ;", "BSSY.RECONVERGENT B1, 0x60 ;",
           "@UP0 BRA 0x50 ;", "NOP ;", "BSYNC.RECONVERGENT B1 ;", "RET ;"]
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callgraph_status"] == "PASS"
    assert result["callee_rows"][0]["instructions"] == 5


def test_missing_convergence_barrier_target_remains_unknown():
    ops = ["CALL.REL.NOINC 0x20 ;", "EXIT ;", "BSYNC.RECONVERGENT B1 ;", "RET ;"]
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callgraph_status"] == "UNKNOWN"


def test_barrier_register_reuse_does_not_pull_other_callees_into_body():
    ops = ["CALL.REL.NOINC 0x40 ;", "CALL.REL.NOINC 0x70 ;", "EXIT ;", "NOP ;",
           "BSSY B1, 0x60 ;", "BSYNC B1 ;", "RET ;", "BSSY B1, 0x90 ;", "BSYNC B1 ;", "RET ;"]
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callgraph_status"] == "PASS"
    assert [row["instructions"] for row in result["callee_rows"]] == [3, 3]


def test_branch_operand_predicate_keeps_fallthrough_path():
    ops = ["CALL.REL.NOINC 0x20 ;", "EXIT ;", "BRA !P4, 0x50 ;",
           "NOP ;", "RET ;", "RET ;"]
    result = cubin.analyze_calls(cubin.parse_sass(disassembly(ops)))["root"]
    assert result["callgraph_status"] == "PASS" and result["callee_rows"][0]["instructions"] == 4


@pytest.mark.parametrize("resources", ["", "Function k:\n REG:8\n", "Function k:\n STACK:-1\n"])
def test_missing_stack_measurement_refuses(resources):
    with pytest.raises(ValueError, match="STACK"):
        cubin.parse_resources(resources)


def test_resource_sass_inventory_mismatch_refuses():
    with pytest.raises(ValueError, match="absent from"):
        cubin.scan_text("Function other:\n STACK:0\n", disassembly(["EXIT ;"]))


def test_committed_native_cubin_is_decoded_without_gpu():
    tool = shutil.which("cuobjdump")
    if not tool:
        pytest.skip("CUDA disassembler not installed; text parser fixtures still run")
    rows = cubin.scan(FIXTURES / "tiny.cubin", tool)
    assert len(rows) == 1 and rows[0]["kernel"] == "e41_fixture"
    assert rows[0]["stack"] == 0 and rows[0]["sass"] > 0
    assert rows[0]["callgraph_status"] == "PASS"


def test_tools_and_fixtures_work_from_clean_git_archive(tmp_path):
    root = SCRIPT.parent.parent
    if not (root / ".git").exists():
        pytest.skip("source checkout needed to build a clean git archive")
    paths = ["scripts/e41", "scripts/wave_cubin_gate.py", "scripts/gpu_smoke_e41.py",
             "tests/v025/compile/fixtures/e41"]
    archive = subprocess.run(["git", "archive", "HEAD", *paths], cwd=root,
                             check=True, capture_output=True, timeout=30).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    clean_fixtures = tmp_path / "tests/v025/compile/fixtures/e41"
    for name in ["tiny.cubin", "tiny.ptx", "callgraph.resources.txt", "callgraph.sass.txt", "transpose.optimized.hlo",
                 "g0_identity_convert.jaxcache.zst", "g0_add.jaxcache.zst"]:
        assert (clean_fixtures / name).is_file()
    source = """
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('archived_hlo', 'scripts/e41/hlo_scan.py')
hlo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hlo)
row, = hlo.scan_text(Path('tests/v025/compile/fixtures/e41/transpose.optimized.hlo').read_text())['transpose_hero_fusions']
assert row['n_instr'] == 5 and row['kind'] == 'kInput'
"""
    subprocess.run([sys.executable, "-c", source], cwd=tmp_path, check=True, timeout=30)


def test_hlo_fixture_counts_parameters_and_root_in_hlo_units():
    row, = hlo.scan_text((FIXTURES / "transpose.optimized.hlo").read_text())["transpose_hero_fusions"]
    assert (row["kind"], row["n_instr"], row["n_params"], row["n_transpose"]) == ("kInput", 5, 1, 1)


@pytest.mark.parametrize("n", [500, 501])
def test_hlo_counts_large_fusion_body_exactly(n):
    text = (FIXTURES / "transpose.optimized.hlo").read_text()
    extra = "\n".join(f"  %extra.{i} = f32[] negate(%zero)" for i in range(n - 5))
    text = text.replace("  ROOT %turned", extra + "\n  ROOT %turned")
    row, = hlo.scan_text(text)["transpose_hero_fusions"]
    assert row["n_instr"] == n


def test_loop_transpose_is_reported_with_its_actual_kind():
    text = (FIXTURES / "transpose.optimized.hlo").read_text().replace("kind=kInput", "kind=kLoop")
    assert hlo.scan_text(text)["transpose_hero_fusions"][0]["kind"] == "kLoop"


@pytest.mark.parametrize("text", ["", "ENTRY main {\n", "ENTRY main {\n ROOT x = f32[] fusion(), kind=kInput, calls=missing\n}\n"])
def test_missing_hlo_body_does_not_prove_zero(text):
    with pytest.raises(ValueError):
        hlo.scan_text(text)


def test_compact_aot_placeholder_does_not_prove_zero():
    with pytest.raises(ValueError, match="compact AOT metadata"):
        hlo.scan_text("ENTRY main {\n ROOT slim.zero = s32[] constant(0)\n}\n")


def test_cuda_elf_fixture_and_incidental_magic():
    binary = (FIXTURES / "tiny.cubin").read_bytes()
    assert list(gate._elfs(b"prefix\x7fELFgarbage" + binary + b"suffix")) == [binary]
    with pytest.raises(ValueError, match="invalid CUDA ELF"):
        list(gate._elfs(binary[:100]))
    corrupt = bytearray(binary)
    struct.pack_into("<Q", corrupt, 0x28, 0xFFFFFFFFFFFFFF00)
    with pytest.raises(ValueError, match="invalid CUDA ELF"):
        list(gate._elfs(bytes(corrupt)))


def gpu_blob(native, fmt):
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    if fmt == "riegeli":
        payload = asm._append_simple_chunk(ax._RIEGELI_SIGNATURE, [b"first", b"second", native, b"rest"])
    else:
        payload = ax._pack(4, 2, native)
    return b"\x00" + ax._pack(1, 2, payload)


@pytest.mark.parametrize("fmt", ["riegeli", "legacy"])
@pytest.mark.parametrize("kind", ["jax_cache", "aot"])
def test_actual_cache_formats_decode_native_record(tmp_path, monkeypatch, fmt, kind):
    import zstandard
    binary = (FIXTURES / "tiny.cubin").read_bytes()
    blob = gpu_blob(binary, fmt)
    path = tmp_path / "jit__advance_chunk_fori-fixture-cache"
    # JAX stores a four-byte compile-time prefix, not the obsolete eight bytes.
    path.write_bytes(zstandard.ZstdCompressor().compress(struct.pack("<I", 123) + blob) if kind == "jax_cache" else blob)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(gate, "_optimized_hlo", lambda *a: (FIXTURES / "transpose.optimized.hlo").read_text())
    scanner = gate._scanner
    class Native:
        @staticmethod
        def scan(path):
            assert path.read_bytes() == binary
            return [{"kernel": "fixture", "stack": 0, "sass": 3, "calls": 0, "callees": 0,
                     "nonleaf_over1k": [], "callgraph_unknown": []}]
    monkeypatch.setattr(gate, "_scanner", lambda name: Native if name == "cubin_scan" else scanner(name))
    rec = gate.scan_executable(gate.Executable(path, kind), out, tmp_path / "unused.db")
    assert rec["format"] == f"xla-gpu-{'thunks-riegeli' if fmt == 'riegeli' else 'legacy'}-v1"
    assert rec["elfs"][0]["kernels"] == 1
    assert rec["kinput_transpose_status"] == ("MEASURED_RICH_JIT_HLO" if kind == "jax_cache" else "MEASURED_RICH_AOT_HLO")
    assert not list(out.glob("*.cubin"))


def test_missing_riegeli_native_record_refuses():
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    payload = asm._append_simple_chunk(ax._RIEGELI_SIGNATURE, [b"first"])
    with pytest.raises(ValueError, match="missing Riegeli record 2"):
        gate._record(b"\x00" + ax._pack(1, 2, payload), 2, ax, asm)


@pytest.mark.parametrize("failure", ["callee", "hlo", "unknown_native", "unknown_hlo"])
def test_overall_e41_verdict_does_not_hide_nonstack_gates(tmp_path, failure):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "jit__advance_chunk_fori-fixture-cache").write_bytes(b"stub")
    def scan(entry, out, db):
        elf = {"stack_over256": [], "nonleaf_over1k": [], "callgraph_unknown": []}
        rec = {"path": str(entry.path), "elfs": [elf], "kinput_transpose_status": "MEASURED_RICH_JIT_HLO",
               "kinput_transpose_over500": []}
        if failure == "callee":
            elf["nonleaf_over1k"] = [{"instructions": 1001}]
        elif failure == "hlo":
            rec["kinput_transpose_over500"] = [("fusion", 501)]
        elif failure == "unknown_native":
            elf["callgraph_unknown"] = ["unresolved CALL"]
        else:
            rec["kinput_transpose_status"] = "UNKNOWN_AOT_METADATA"
        return rec
    result = gate.run_gate(cache, tmp_path / "out", scanner=scan)
    assert result["hard_stack_gate"] == "PASS"
    assert result["e41_gate"] == ("FAIL" if failure in {"callee", "hlo"} else "INCOMPLETE")


def test_refusal_removes_old_passing_verdict(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "wave_gate.json").write_text('{"e41_gate": "PASS"}')
    with pytest.raises(ValueError, match="does not exist"):
        gate.run_gate(tmp_path / "absent", out)
    assert not (out / "wave_gate.json").exists()


def test_default_discovery_includes_rich_fused_and_static_scan_jit(tmp_path):
    for name in ["jit_fused_jit-child-cache", "jit__advance_chunk_static_scan-root-cache", "jit_init_radiation-cache"]:
        (tmp_path / name).write_bytes(b"fixture")
    assert [entry.path.name for entry in gate.discover(tmp_path)] == [
        "jit__advance_chunk_static_scan-root-cache", "jit_fused_jit-child-cache"]


def test_smoke_entrypoint_uses_full_gate_and_preserves_failure(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("smoke_e41_tested", SCRIPT.with_name("gpu_smoke_e41.py"))
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    class FullGate:
        @staticmethod
        def run_gate(cache, out):
            assert cache == tmp_path / "cache"
            return {"e41_gate": "FAIL", "hard_stack_gate": "PASS", "conservative_flags": True,
                    "n_entries": 1, "entries": [{"elfs": [{"max_stack_B": 32}]}]}
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(smoke, "_gate_module", lambda: FullGate)
    assert smoke.main([str(tmp_path / "cache"), str(tmp_path)]) == 2
    assert json.loads((tmp_path / "e41.json").read_text())["e41_gate"] == "FAIL"
