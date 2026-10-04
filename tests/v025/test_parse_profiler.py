"""Profiler reduction must work before the window, not after it (§8, §9, §16).

Fixtures below are shaped like real `nsys stats --report cuda_gpu_kern_sum
--format csv` and `ncu --csv` output, banner lines and all. If reduction only
gets exercised on the real capture, an unparseable trace costs a second
coordinated window and a second ask of two neighbouring managers.

The load-bearing behaviour under test is the fail-closed half: a counter the
card did not produce must surface as a BLOCKED denominator, never as a zero that
quietly flows into an arithmetic mean and makes the model look more efficient
than it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import parse_profiler as pp  # noqa: E402

# `nsys stats --format csv` emits PLAIN integers. An earlier version of this
# fixture used thousands-separated numbers copied from the human-readable
# "column" format; unquoted commas are not valid CSV and the fixture, not the
# parser, was wrong. Both forms are covered: this one, and a quoted
# thousands-separated variant below, since `--format table` piped through some
# tooling does produce the latter.
NSYS_CSV = """** CUDA GPU Kernel Summary (cuda_gpu_kern_sum)

Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
41.2,1236000000,4120,300000,295000,210000,480000,22000,fusion_rrtmg_lw_taumol_kernel
19.6,588000000,2000,294000,290000,240000,350000,18000,thompson_column_fusion_7
12.1,363000000,8400,43214,42000,30000,90000,4100,advect_scalar_flux_fusion
6.4,192000000,1200,160000,158000,120000,220000,9000,mynn_edmf_fusion_3
3.0,90000000,3600,25000,24800,20000,40000,2000,wrapped_thomas_solve_scan
1.1,33000000,600,55000,54000,50000,70000,3000,unnamed_kernel_0x7f
"""

NSYS_CSV_QUOTED_THOUSANDS = '''** CUDA GPU Kernel Summary (cuda_gpu_kern_sum)

Time (%),Total Time (ns),Instances,Avg (ns),Name
41.2,"1,236,000,000","4,120","300,000",fusion_rrtmg_lw_taumol_kernel
12.1,"363,000,000","8,400","43,214",advect_scalar_flux_fusion
'''

NCU_CSV = '''==PROF== Profiling "fusion_rrtmg_lw" - 1: 0%....50%....100%

"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_fadd_pred_on.sum","inst","1000000"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_fmul_pred_on.sum","inst","2000000"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_ffma_pred_on.sum","inst","3000000"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_dadd_pred_on.sum","inst","500000"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_dmul_pred_on.sum","inst","400000"
"0","1234","fusion_rrtmg_lw","sm__sass_thread_inst_executed_op_dfma_pred_on.sum","inst","100000"
"0","1234","fusion_rrtmg_lw","dram__bytes.sum","byte","8000000"
"0","1234","fusion_rrtmg_lw","lts__t_bytes.sum","byte","16000000"
"0","1234","fusion_rrtmg_lw","sm__warps_active.avg.pct_of_peak_sustained_active","%","15.6"
'''


# --------------------------------------------------------------------------- #
# nsys                                                                         #
# --------------------------------------------------------------------------- #
def test_nsys_summary_parses_past_the_banner():
    kernels = pp.parse_nsys_kernel_summary(NSYS_CSV)
    assert len(kernels) == 6
    first = kernels[0]
    assert first["name"] == "fusion_rrtmg_lw_taumol_kernel"
    assert first["device_time_ns"] == 1_236_000_000
    assert first["launches"] == 4_120


def test_launch_counts_sum_correctly():
    kernels = pp.parse_nsys_kernel_summary(NSYS_CSV)
    assert sum(k["launches"] for k in kernels) == 4120 + 2000 + 8400 + 1200 + 3600 + 600


def test_quoted_thousands_separated_numbers_also_parse():
    """Some tooling re-emits nsys numbers quoted with separators; handle both."""
    kernels = pp.parse_nsys_kernel_summary(NSYS_CSV_QUOTED_THOUSANDS)
    assert len(kernels) == 2
    assert kernels[0]["device_time_ns"] == 1_236_000_000
    assert kernels[0]["launches"] == 4_120


def test_nsys_output_feeds_the_attributor_directly():
    """The parser's shape must be exactly what attribute_device_time consumes."""
    import run_gpu_arm as rga

    result = rga.attribute_device_time(pp.parse_nsys_kernel_summary(NSYS_CSV))
    assert result["families"]["physics.radiation"]["launches"] == 4120
    assert "dycore.advection" in result["families"]
    # one deliberately unattributable kernel in the fixture
    assert result["families"]["unknown"]["launches"] == 600


def test_nsys_discriminator_is_computable_from_a_real_shaped_trace():
    """The manager's W1 question must fall out of the capture without extra work."""
    import run_gpu_arm as rga

    result = rga.attribute_device_time(pp.parse_nsys_kernel_summary(NSYS_CSV))
    physics = result["measured_physics_share_of_known"]
    dycore = result["measured_dycore_share_of_known"]
    assert physics is not None and dycore is not None
    assert physics + dycore == pytest.approx(1.0)


def test_nsys_garbage_raises_rather_than_returning_empty():
    with pytest.raises(pp.ProfilerParseError, match="no cuda_gpu_kern_sum header"):
        pp.parse_nsys_kernel_summary("total nonsense\nno header here\n")


def test_nsys_header_without_rows_is_an_error_not_an_empty_census():
    header = " Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
    with pytest.raises(pp.ProfilerParseError, match="zero kernels"):
        pp.parse_nsys_kernel_summary(header)


# --------------------------------------------------------------------------- #
# ncu                                                                          #
# --------------------------------------------------------------------------- #
def test_ncu_parses_all_requested_counters():
    parsed = pp.parse_ncu_csv(NCU_CSV)
    assert parsed["availability"]["flop_counters_complete"] is True
    assert parsed["availability"]["blocked_denominator"] is False
    assert parsed["per_kernel"]["fusion_rrtmg_lw"]["dram__bytes.sum"] == 8_000_000


def test_a_missing_counter_blocks_the_denominator():
    """§16: unavailable counters block F_measured; they never license an estimate."""
    trimmed = "\n".join(
        line for line in NCU_CSV.splitlines() if "dfma" not in line
    )
    parsed = pp.parse_ncu_csv(trimmed)
    assert parsed["availability"]["flop_counters_complete"] is False
    assert parsed["availability"]["blocked_denominator"] is True
    assert "sm__sass_thread_inst_executed_op_dfma_pred_on.sum" in parsed["availability"]["missing"]


def test_an_na_value_counts_as_missing_not_as_zero():
    """A zero here would silently shrink F_measured and flatter the model."""
    text = NCU_CSV.replace('"inst","100000"', '"inst","n/a"')
    parsed = pp.parse_ncu_csv(text)
    assert parsed["availability"]["blocked_denominator"] is True


def test_ncu_without_headers_raises():
    with pytest.raises(pp.ProfilerParseError, match="no ncu CSV header"):
        pp.parse_ncu_csv("junk\n")


# --------------------------------------------------------------------------- #
# F_measured                                                                   #
# --------------------------------------------------------------------------- #
def test_fma_counts_as_two_operations():
    parsed = pp.parse_ncu_csv(NCU_CSV)
    flops = pp.useful_flops(parsed["per_kernel"])
    assert flops["by_class"]["f32_fma"] == 6_000_000   # 3e6 instructions x 2
    assert flops["by_class"]["f64_fma"] == 200_000     # 1e5 instructions x 2


def test_fp32_and_fp64_are_reported_separately():
    """Merging them would erase the very gap v0.25 exists to close."""
    parsed = pp.parse_ncu_csv(NCU_CSV)
    flops = pp.useful_flops(parsed["per_kernel"])
    assert flops["f32_flops"] == 1_000_000 + 2_000_000 + 6_000_000
    assert flops["f64_flops"] == 500_000 + 400_000 + 200_000
    assert flops["total_flops"] == flops["f32_flops"] + flops["f64_flops"]
    assert flops["f64_fraction"] == pytest.approx(1_100_000 / 10_100_000)


# --------------------------------------------------------------------------- #
# whole-capture reduction                                                      #
# --------------------------------------------------------------------------- #
def test_summarise_capture_reduces_both_halves():
    summary = pp.summarise_capture(nsys_text=NSYS_CSV, ncu_text=NCU_CSV)
    assert summary["kernel_count"] == 6
    assert summary["F_measured"]["total_flops"] > 0


def test_summarise_capture_reports_blocked_rather_than_estimating():
    trimmed = "\n".join(line for line in NCU_CSV.splitlines() if "dfma" not in line)
    summary = pp.summarise_capture(nsys_text=NSYS_CSV, ncu_text=trimmed)
    assert summary["F_measured"]["status"] == "BLOCKED"
    assert "not permission to estimate" in summary["F_measured"]["reason"]


def test_summarise_capture_needs_something_to_summarise():
    with pytest.raises(pp.ProfilerParseError, match="nothing to summarise"):
        pp.summarise_capture()
