"""Nested NVTX ranges must be attributed by EXCLUSIVE time.

Manager review of `dd5ed7b8`, defect 6. XLA compile ranges nest three deep
(`XlaCompile` > `XlaPassPipeline` > `XlaPass`). Summing the inclusive durations
that `nvtx_sum` reports counts each child once per ancestor, so a naive
attribution can exceed 100% of the compile and still look plausible.

These tests pin the arithmetic, the fail-closed cross-check against nsys's own
column, and the 95% attribution validator.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import nvtx_exclusive as nx  # noqa: E402

MS = 1_000_000  # ns


def _rows(*specs):
    """(name, start_ms, duration_ms[, tid]) -> raw row dicts."""
    out = []
    for spec in specs:
        name, start, duration = spec[0], spec[1], spec[2]
        tid = spec[3] if len(spec) > 3 else "1"
        out.append({"name": name, "start_ns": start * MS, "duration_ns": duration * MS,
                    "pid": "100", "tid": tid, "range_id": None, "parent_id": None,
                    "nsys_non_child_ns": None})
    return out


# --------------------------------------------------------------------------- #
# the arithmetic                                                               #
# --------------------------------------------------------------------------- #
def test_a_parent_does_not_keep_its_childrens_time():
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaCompile:#module=jit_f#", 0, 100),
        ("TSL:XlaPass:#name=a#", 10, 30),
        ("TSL:XlaPass:#name=b#", 50, 40),
    ))
    assert rows[0]["exclusive_ns"] / MS == pytest.approx(30)   # 100 - 30 - 40
    assert rows[1]["exclusive_ns"] / MS == pytest.approx(30)
    assert rows[2]["exclusive_ns"] / MS == pytest.approx(40)


def test_grandchildren_are_not_subtracted_twice():
    """Three levels: only DIRECT children come off each range."""
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaCompile:#module=jit_f#", 0, 100),
        ("TSL:XlaPassPipeline:#name=opt#", 10, 60),
        ("TSL:XlaPass:#name=inner#", 20, 40),
    ))
    exclusive = [r["exclusive_ns"] / MS for r in rows]
    assert exclusive == pytest.approx([40, 20, 40])
    assert sum(exclusive) == pytest.approx(100), "exclusive times must partition the parent"


def test_exclusive_total_equals_the_root_span_not_the_inclusive_sum():
    """The defect being guarded: inclusive summing inflates the total."""
    raw = _rows(
        ("TSL:XlaCompile:#module=jit_f#", 0, 100),
        ("TSL:XlaPassPipeline:#name=opt#", 0, 100),
        ("TSL:XlaPass:#name=inner#", 0, 100),
    )
    inclusive_total = sum(r["duration_ns"] for r in raw) / MS
    exclusive_total = sum(r["exclusive_ns"] for r in nx.compute_exclusive(raw)) / MS
    assert inclusive_total == pytest.approx(300)
    assert exclusive_total == pytest.approx(100)


def test_sibling_threads_do_not_parent_each_other():
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaCompile:#module=jit_f#", 0, 100, "1"),
        ("TSL:XlaCompile:#module=jit_g#", 10, 50, "2"),
    ))
    assert rows[0]["exclusive_ns"] / MS == pytest.approx(100)
    assert rows[1]["exclusive_ns"] / MS == pytest.approx(50)


def test_a_range_ending_before_the_next_starts_is_not_a_parent():
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=a#", 0, 10),
        ("TSL:XlaPass:#name=b#", 10, 10),
    ))
    assert [r["exclusive_ns"] / MS for r in rows] == pytest.approx([10, 10])


def test_parent_ids_are_used_when_nsys_supplies_them():
    rows = [
        {"name": "TSL:XlaCompile:#module=jit_f#", "start_ns": 0, "duration_ns": 100 * MS,
         "pid": "1", "tid": "1", "range_id": "r1", "parent_id": "", "nsys_non_child_ns": None},
        {"name": "TSL:XlaPass:#name=a#", "start_ns": 5 * MS, "duration_ns": 60 * MS,
         "pid": "1", "tid": "1", "range_id": "r2", "parent_id": "r1",
         "nsys_non_child_ns": None},
    ]
    computed = nx.compute_exclusive(rows)
    assert computed[0]["exclusive_ns"] / MS == pytest.approx(40)


# --------------------------------------------------------------------------- #
# fail-closed cross-check                                                      #
# --------------------------------------------------------------------------- #
def test_agreement_with_nsys_own_column_is_accepted():
    rows = _rows(("TSL:XlaCompile:#module=jit_f#", 0, 100), ("TSL:XlaPass:#name=a#", 10, 30))
    rows[0]["nsys_non_child_ns"] = 70 * MS
    assert nx.compute_exclusive(rows)[0]["nsys_cross_check_drift"] == pytest.approx(0.0)


def test_disagreement_with_nsys_own_column_refuses_to_attribute():
    """Reconciling two sources by picking one silently is how bad numbers ship."""
    rows = _rows(("TSL:XlaCompile:#module=jit_f#", 0, 100), ("TSL:XlaPass:#name=a#", 10, 30))
    rows[0]["nsys_non_child_ns"] = 20 * MS   # nsys says 20 ms, we compute 70 ms
    with pytest.raises(ValueError, match="cross-check failed"):
        nx.compute_exclusive(rows)


# --------------------------------------------------------------------------- #
# family mapping                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,family", [
    ("TSL:XlaPass:#name=triton-autotuner,module=jit_f#", "autotune"),
    ("TSL:XlaPass:#name=gemm-algorithm-picker,module=jit_f#", "autotune"),
    ("TSL:XlaPass:#name=conv-algorithm-picker,module=jit_f#", "autotune"),
    ("TSL:XlaCompileGpuAsm", "codegen"),
    ("TSL:XlaOptimizeLlvmIr", "codegen"),
    ("TSL:XlaPassPipeline:#name=optimization,module=jit_f#", "pass_pipeline"),
    ("TSL:XlaPass:#name=call-inliner,module=jit_f#", "hlo_pass"),
    ("TSL:XlaCompile:#module=jit_f,program_id=1#", "compile_frame"),
    ("SomethingElseEntirely", "unknown"),
])
def test_ranges_map_to_the_frozen_families(name, family):
    assert nx.classify(name) == family


def test_an_autotuner_pass_is_autotune_not_a_generic_hlo_pass():
    """Ordering matters: an autotuner range also matches the XlaPass pattern."""
    assert nx.classify("TSL:XlaPass:#name=triton-autotuner,module=jit_f#") == "autotune"


@pytest.mark.parametrize("name,family", [
    # Every one of these was left `unknown` by the first version of the mapping and
    # found by inspecting the real W1b capture rather than by guessing.
    ("TSL:XlaCompileBackend:#module=jit_f#", "compile_frame"),
    ("TSL:XlaCreateGpuExecutable:#module=jit_f#", "compile_frame"),
    ("TSL:XlaModule:#module=jit_f#", "compile_frame"),
    ("TSL:XlaMemoryScheduler:#module=jit_f#", "buffer_and_schedule"),
    ("TSL:XlaBufferAssignment:#module=jit_f#", "buffer_and_schedule"),
    ("TSL:XlaCompileCudnnFusion:#module=jit_f#", "codegen"),
    ("TSL:XlaEmitLlvmIr:#module=jit_f#", "codegen"),
    ("TSL:Thunk:#name=fusion#", "runtime"),
])
def test_the_phases_found_in_the_real_capture_are_mapped(name, family):
    assert nx.classify(name) == family


def test_thunk_execution_is_kept_out_of_the_compile_denominator():
    """Mixing execution into a compile attribution silently shrinks every share."""
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=call-inliner#", 0, 100),
        ("TSL:Thunk:#name=fusion#", 200, 900),
    ))
    result = nx.attribute(rows)
    assert result["total_exclusive_compile_seconds"] == pytest.approx(0.100)
    assert result["non_compile_seconds"]["runtime"] == pytest.approx(0.900)
    assert result["family_share_of_compile"]["hlo_pass"] == pytest.approx(1.0)
    assert "runtime" not in result["family_share_of_compile"]


def test_execution_time_cannot_mask_an_unattributed_compile():
    """A huge Thunk must not dilute `unknown` below the 5% bar."""
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=a#", 0, 50),
        ("MysteryRange", 100, 50),
        ("TSL:Thunk:#name=fusion#", 200, 100_000),
    ))
    assert nx.attribute(rows)["status"] == "BLOCKED"


def test_pass_labels_are_extracted_for_reporting():
    assert nx.pass_label("TSL:XlaPass:#name=call-inliner,module=jit_f#") == "call-inliner"
    assert nx.pass_label("TSL:XlaCompileGpuAsm") == "TSL:XlaCompileGpuAsm"


# --------------------------------------------------------------------------- #
# the 95% validator                                                            #
# --------------------------------------------------------------------------- #
def test_a_clean_trace_is_attributed():
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaCompile:#module=jit_f#", 0, 100),
        ("TSL:XlaPass:#name=triton-autotuner#", 0, 90),
    ))
    result = nx.attribute(rows)
    assert result["status"] == "OK"
    assert result["by_family_seconds"]["autotune"] == pytest.approx(0.090)


def test_too_much_unknown_time_blocks_rather_than_scoring():
    rows = nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=call-inliner#", 0, 50),
        ("MysteryRange", 100, 50),
    ))
    result = nx.attribute(rows)
    assert result["status"] == "BLOCKED"
    assert result["unknown_share"] == pytest.approx(0.5)
    assert "not scored" in result["reason"]


def test_the_unknown_budget_is_the_pre_registered_five_percent():
    assert nx.MAX_UNKNOWN_SHARE == 0.05


def test_just_under_the_budget_passes_and_just_over_blocks():
    under = nx.attribute(nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=a#", 0, 960), ("Mystery", 1000, 40))))
    over = nx.attribute(nx.compute_exclusive(_rows(
        ("TSL:XlaPass:#name=a#", 0, 940), ("Mystery", 1000, 60))))
    assert under["status"] == "OK"
    assert over["status"] == "BLOCKED"


def test_an_empty_trace_blocks_instead_of_reporting_perfect_attribution():
    assert nx.attribute([])["status"] == "BLOCKED"


# --------------------------------------------------------------------------- #
# CSV shape                                                                    #
# --------------------------------------------------------------------------- #
CSV = textwrap.dedent("""\
    Generating NVTX Push/Pop Range Trace...

    Start (ns),Duration (ns),DurNonChild (ns),Name,PID,TID,RangeId,ParentId
    1000,100000000,70000000,TSL:XlaCompile:#module=jit_f#,100,200,r1,
    5000,30000000,30000000,TSL:XlaPass:#name=call-inliner#,100,200,r2,r1
    """)


def test_the_csv_preamble_does_not_defeat_the_parser():
    rows = nx.parse_pushpop_trace(CSV)
    assert len(rows) == 2
    assert rows[0]["name"].startswith("TSL:XlaCompile")
    assert rows[1]["parent_id"] == "r1"


def test_parsed_rows_flow_through_to_an_attribution():
    result = nx.attribute(nx.compute_exclusive(nx.parse_pushpop_trace(CSV)))
    assert result["status"] == "OK"
    assert result["total_exclusive_compile_seconds"] == pytest.approx(0.1)


def test_a_csv_without_a_header_is_rejected():
    with pytest.raises(ValueError, match="no nvtx_pushpop_trace header"):
        nx.parse_pushpop_trace("just some text\nand more\n")
