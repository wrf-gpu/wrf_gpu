"""E41 NO_KERNEL class: a JAX-cache executable without native CUDA ELF passes only with a pure buffer-move HLO.

Fixtures are real cache files from the v50-gust G0 arm (GPU 2026-11-11 14:05Z, cache/g0/jax):
- g0_identity_convert.jaxcache.zst = jit_convert_element_type-1f92d308...-cache (s32[14,12] parameter -> copy, 0 ELF)
- g0_add.jaxcache.zst              = jit_add-a3b99082...-cache (one loop fusion, 1 ELF)
"""
import importlib.util
from collections import Counter
from pathlib import Path
import shutil
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/wave_cubin_gate.py"
spec = importlib.util.spec_from_file_location("e41_gate_no_kernel_tested", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
FIXTURES = Path(__file__).with_name("fixtures") / "e41"
IDENTITY = FIXTURES / "g0_identity_convert.jaxcache.zst"
ADD = FIXTURES / "g0_add.jaxcache.zst"
CONVERT_NAME = "jit_convert_element_type-1f92d308a67ddf22f87c07430dc982a808f2090560669a643951fd7fc07bca9e-cache"
ADD_NAME = "jit_add-a3b990826d388b0552d93d29143635135d533a4777fde07ea1facf8576ca90fd-cache"


def scan(path, tmp_path, kind="jax_cache"):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return gate.scan_executable(gate.Executable(path, kind), out, tmp_path / "unused.db")


def cache_with(tmp_path, *files):
    cache = tmp_path / "cache"
    cache.mkdir()
    for src, name in files:
        shutil.copyfile(src, cache / name)
    return cache


def identity_text():
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    import zstandard
    blob = zstandard.ZstdDecompressor().decompress(IDENTITY.read_bytes(), max_output_size=1 << 33)[4:]
    return gate._optimized_hlo(blob, ax, asm)


def test_real_kernel_less_blob_is_no_kernel(tmp_path):
    rec = scan(IDENTITY, tmp_path)
    assert rec["kernel_class"] == "NO_KERNEL" and rec["elfs"] == []
    assert rec["no_kernel_hlo_ops"] == {"copy": 1, "parameter": 1}
    assert rec["kinput_transpose_status"] == "NO_KERNEL_IDENTITY_HLO" and rec["kinput_transpose_over500"] == []


def test_text_and_proto_opcode_multisets_agree_on_real_blob():
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    import zstandard
    for fixture in (IDENTITY, ADD):
        blob = zstandard.ZstdDecompressor().decompress(fixture.read_bytes(), max_output_size=1 << 33)[4:]
        text_ops = gate.hlo_text_opcodes(gate._optimized_hlo(blob, ax, asm))
        assert text_ops == gate.hlo_proto_opcodes(gate._optimized_hlo_proto(blob, ax, asm), ax)
    assert "fusion" in text_ops  # the add fixture really carries a kernel


@pytest.mark.parametrize("op", ["fusion", "custom-call", "while", "conditional", "all-reduce", "sort", "add", "convert"])
def test_injected_kernel_op_in_hlo_text_refuses(op):
    text = identity_text()
    shape = "s32[14,12]{1,0}"
    injected = text.replace("  ROOT %copy.1", f"  %inj.1 = {shape} {op}(%args_0_.1), calls=%x\n  ROOT %copy.1")
    assert injected != text
    with pytest.raises(ValueError, match=r"kernel-bearing HLO ops \['" + op):
        gate.classify_no_kernel(injected)


def test_injected_fusion_through_full_scan_path_refuses(tmp_path, monkeypatch):
    text = identity_text().replace("  ROOT %copy.1", "  %f.1 = s32[14,12]{1,0} fusion(%args_0_.1), kind=kLoop, calls=%c\n  ROOT %copy.1")
    monkeypatch.setattr(gate, "_optimized_hlo", lambda *a: text)
    with pytest.raises(ValueError, match="no native CUDA ELF in executable"):
        scan(IDENTITY, tmp_path)


def test_injected_fusion_with_matching_proto_still_refuses():
    text = identity_text().replace("  ROOT %copy.1", "  %f.1 = s32[14,12]{1,0} fusion(%args_0_.1), kind=kLoop, calls=%c\n  ROOT %copy.1")
    with pytest.raises(ValueError, match=r"kernel-bearing HLO ops \['fusion'\]"):
        gate.classify_no_kernel(text, Counter({"parameter": 1, "copy": 1, "fusion": 1}))


def test_text_proto_disagreement_refuses():
    with pytest.raises(ValueError, match="text/proto opcode multisets differ"):
        gate.classify_no_kernel(identity_text(), Counter({"parameter": 1, "copy": 2}))


@pytest.mark.parametrize("bad", ["  garbage line without assignment", "  %x = s32[] ", "orphan line outside"])
def test_unparsed_hlo_lines_refuse(bad):
    text = identity_text()
    text = text.replace("  ROOT %copy.1", bad + "\n  ROOT %copy.1") if bad.startswith("  ") else text + "\n" + bad + "\n"
    with pytest.raises(ValueError, match="unparsed|opcode not found"):
        gate.hlo_text_opcodes(text)


def test_truncated_hlo_refuses():
    with pytest.raises(ValueError, match="truncated or empty"):
        gate.hlo_text_opcodes(identity_text().rsplit("}", 1)[0])


def test_tuple_shapes_parse():
    text = ("HloModule m\n\nENTRY %main (p: f32[2]) -> (f32[2], f32[2]) {\n  %p = f32[2]{0} parameter(0)\n"
            "  %t = (f32[2]{0}, /*index=1*/f32[2]{0}) tuple(%p, %p)\n  %g = f32[2]{0} get-tuple-element(%t), index=0\n"
            "  %c = s32[] constant(3)\n  %b = f32[2]{0} bitcast(%g)\n  ROOT %r = (f32[2]{0}, f32[2]{0}) tuple(%b, %g)\n}\n")
    assert gate.classify_no_kernel(text) == Counter({"tuple": 2, "parameter": 1, "get-tuple-element": 1, "constant": 1, "bitcast": 1})


def corrupt_elf_magic(src, dst):
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(src.read_bytes(), max_output_size=1 << 33)
    assert raw.count(b"\x7fELF\x02\x01\x01") >= 1
    dst.write_bytes(zstandard.ZstdCompressor().compress(raw.replace(b"\x7fELF\x02\x01\x01", b"\x7fELG\x02\x01\x01")))
    return dst


def test_no_elf_executable_with_fusion_hlo_still_fails(tmp_path):
    mutant = corrupt_elf_magic(ADD, tmp_path / ADD_NAME)
    with pytest.raises(ValueError, match=r"no native CUDA ELF in executable.*kernel-bearing HLO ops \[.*'fusion'"):
        scan(mutant, tmp_path)


def test_aot_stepping_blob_never_qualifies(tmp_path):
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(IDENTITY.read_bytes(), max_output_size=1 << 33)[4:]
    aot = tmp_path / "d01.xlaexec"
    aot.write_bytes(raw)
    with pytest.raises(ValueError, match="no native CUDA ELF in executable"):
        scan(aot, tmp_path, kind="aot")


def test_all_no_kernel_cache_is_fail(tmp_path):
    cache = cache_with(tmp_path, (IDENTITY, CONVERT_NAME))
    result = gate.run_gate(cache, tmp_path / "gate", "*")
    assert result["e41_gate"] == "FAIL" and result["no_measured_kernels"] is True
    assert result["n_no_kernel"] == 1 and result["no_kernel_hlo_ops"] == {"copy": 1, "parameter": 1}


def test_all_no_kernel_cache_cli_exits_nonzero(tmp_path, monkeypatch):
    cache = cache_with(tmp_path, (IDENTITY, CONVERT_NAME))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert gate.main([str(cache), str(tmp_path / "gate"), "*"]) == 2


def native_stub(monkeypatch):
    scanner = gate._scanner
    class Native:
        @staticmethod
        def scan(path):
            return [{"kernel": "loop_add_fusion", "stack": 0, "sass": 12, "calls": 0, "callees": 0,
                     "nonleaf_over1k": [], "callgraph_unknown": []}]
    monkeypatch.setattr(gate, "_scanner", lambda name: Native if name == "cubin_scan" else scanner(name))


def test_mixed_cache_passes_and_records_no_kernel(tmp_path, monkeypatch):
    native_stub(monkeypatch)  # the cubin disassembler is not part of this test; ELF extraction and HLO are real
    cache = cache_with(tmp_path, (IDENTITY, CONVERT_NAME), (ADD, ADD_NAME))
    result = gate.run_gate(cache, tmp_path / "gate", "*")
    assert result["e41_gate"] == "PASS" and result["n_entries"] == 2 and result["n_no_kernel"] == 1
    assert result["no_kernel_entries"] == [CONVERT_NAME] and result["no_measured_kernels"] is False
    kinds = {Path(r["path"]).name: r["kernel_class"] for r in result["entries"]}
    assert kinds == {CONVERT_NAME: "NO_KERNEL", ADD_NAME: "KERNEL"}


def test_mixed_cache_with_mutant_no_elf_fusion_refuses(tmp_path, monkeypatch):
    native_stub(monkeypatch)
    cache = cache_with(tmp_path, (IDENTITY, CONVERT_NAME))
    corrupt_elf_magic(ADD, cache / ADD_NAME)
    with pytest.raises(ValueError, match="kernel-bearing HLO ops"):
        gate.run_gate(cache, tmp_path / "gate", "*")
    assert not (tmp_path / "gate/wave_gate.json").exists()


def test_inconsistent_no_kernel_record_refuses(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "jit__advance_chunk_fori-x-cache").write_bytes(b"stub")
    def lying(entry, out, db):
        return {"path": str(entry.path), "source_kind": "jax_cache", "kernel_class": "NO_KERNEL",
                "no_kernel_hlo_ops": {"copy": 1}, "elfs": [{"stack_over256": [], "nonleaf_over1k": []}]}
    with pytest.raises(ValueError, match="inconsistent NO_KERNEL"):
        gate.run_gate(cache, tmp_path / "gate", scanner=lying)


def test_no_kernel_record_does_not_cover_aot_program(tmp_path):
    cache = tmp_path / "cache"
    (cache / "aot/tag/d01").mkdir(parents=True)
    (cache / "aot/tag/d01/root.xlaexec").write_bytes(b"stub")
    def no_kernel(entry, out, db):
        return {"path": str(entry.path), "program": entry.program, "source_kind": entry.kind,
                "kernel_class": "NO_KERNEL", "no_kernel_hlo_ops": {"copy": 1}, "elfs": []}
    with pytest.raises(ValueError, match="inconsistent NO_KERNEL"):
        gate.run_gate(cache, tmp_path / "gate", scanner=no_kernel)


def test_fixtures_are_not_gitignored():
    import subprocess
    root = SCRIPT.parent.parent
    if not (root / ".git").exists():
        pytest.skip("source checkout needed")
    run = subprocess.run(["git", "check-ignore", "-v", str(IDENTITY), str(ADD)], cwd=root, capture_output=True, text=True)
    assert run.returncode == 1, (run.stdout, run.stderr)  # 1 = none ignored (E229)


# scan_executable call-site wiring (Sol Delta E41NOKERNEL test gap): the classifier must receive the proto multiset of
# the SAME blob. Dropping the proto argument at the call site must fail these tests.
def _blob(fixture):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(fixture.read_bytes(), max_output_size=1 << 33)[4:]


def test_full_scan_refuses_clean_text_with_kernel_bearing_proto(tmp_path, monkeypatch):
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    add_proto = gate._optimized_hlo_proto(_blob(ADD), ax, asm)  # real proto carrying a fusion
    text = identity_text()
    monkeypatch.setattr(gate, "_optimized_hlo", lambda *a: text)  # _optimized_hlo reads the proto: pin the real clean text
    monkeypatch.setattr(gate, "_optimized_hlo_proto", lambda *a: add_proto)
    with pytest.raises(ValueError, match="no native CUDA ELF in executable.*text/proto opcode multisets differ"):
        scan(IDENTITY, tmp_path)


def test_full_scan_refuses_missing_proto_instructions(tmp_path, monkeypatch):
    text = identity_text()
    monkeypatch.setattr(gate, "_optimized_hlo", lambda *a: text)
    monkeypatch.setattr(gate, "_optimized_hlo_proto", lambda *a: b"")  # proto yields no instructions
    with pytest.raises(ValueError, match="no native CUDA ELF in executable.*text/proto opcode multisets differ"):
        scan(IDENTITY, tmp_path)


def test_full_scan_refuses_unreadable_proto(tmp_path, monkeypatch):
    def broken(*a):
        raise ValueError("proto record missing")
    text = identity_text()
    monkeypatch.setattr(gate, "_optimized_hlo", lambda *a: text)  # clean real text; only the proto read fails
    monkeypatch.setattr(gate, "_optimized_hlo_proto", broken)
    with pytest.raises(ValueError, match="no native CUDA ELF in executable.*proto record missing"):
        scan(IDENTITY, tmp_path)


def test_full_scan_passes_this_blobs_proto_multiset_to_the_classifier(tmp_path, monkeypatch):
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    seen, real = [], gate.classify_no_kernel
    def spy(text, proto_ops=None):
        seen.append(proto_ops)
        return real(text, proto_ops)
    monkeypatch.setattr(gate, "classify_no_kernel", spy)
    assert scan(IDENTITY, tmp_path)["kernel_class"] == "NO_KERNEL"
    want = gate.hlo_proto_opcodes(gate._optimized_hlo_proto(_blob(IDENTITY), ax, asm), ax)
    assert seen == [want] and want == Counter({"parameter": 1, "copy": 1})
