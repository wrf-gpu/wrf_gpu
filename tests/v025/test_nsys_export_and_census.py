"""The live exporter must be invoked, and the census must fail closed.

Manager review on main `c227cfc7`. Previously only `step1_pushpop.csv` came from
the real report; the census CSVs were written by the stub, so the exporter was
never exercised and a missing file read as an empty measurement. The census in
turn checked one share instead of two, could not tell initialisation copies from
timestep-loop copies, scored an empty transfer input as `OK`, inferred profiler
overhead from application work, and called the biggest allocation row "peak
VRAM".

The load-bearing test here runs the REAL exporter against the REAL W1b report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import baseline_census as bc  # noqa: E402
import nsys_export as nex  # noqa: E402
import nvtx_exclusive as nx  # noqa: E402
import parse_profiler as pp  # noqa: E402

W1B_REP = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline.nsys-rep")


# --------------------------------------------------------------------------- #
# the real exporter, on the real report                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not W1B_REP.is_file(), reason="W1b report not on this machine")
def test_the_real_exporter_runs_against_the_existing_w1b_report(tmp_path):
    """CPU-only: `nsys stats` reads a file, it does not open the device."""
    result = nex.export_all(W1B_REP, tmp_path)

    # Ground truth already on record: this capture was killed during init, so it
    # has NO CUDA kernel data and no projected forecast GPU ranges.
    assert result["exports"]["nvtx_pushpop_trace"]["status"] == "OK"
    assert result["exports"]["nvtx_pushpop_trace"]["data_rows"] > 100_000
    assert result["exports"]["cuda_gpu_kern_sum"]["status"] == "EMPTY"
    assert result["exports"]["cuda_gpu_trace"]["status"] == "OK"
    assert result["exports"]["cuda_gpu_trace"]["data_rows"] == 8
    assert result["exports"]["nvtx_gpu_proj_sum"]["status"] == "EMPTY"

    # An empty REQUIRED report must block the whole export.
    assert result["status"] == "BLOCKED"
    assert "cuda_gpu_kern_sum" in result["required_unusable"]


@pytest.mark.skipif(not W1B_REP.is_file(), reason="W1b report not on this machine")
def test_the_memset_count_matches_the_recorded_ground_truth(tmp_path):
    """Eight memsets, aggregated by nsys into one summary row."""
    nex.export_report(W1B_REP, "cuda_gpu_mem_time_sum",
                      tmp_path / "mem.csv")
    text = (tmp_path / "mem.csv").read_text()
    assert "[CUDA memset]" in text
    row = next(line for line in text.splitlines() if "[CUDA memset]" in line)
    assert row.split(",")[2] == "8", f"expected 8 memsets, row was {row!r}"


@pytest.mark.skipif(not W1B_REP.is_file(), reason="W1b report not on this machine")
def test_every_export_is_command_and_hash_recorded(tmp_path):
    result = nex.export_all(W1B_REP, tmp_path)
    assert len(result["source_rep_sha256"]) == 64
    for name, export in result["exports"].items():
        assert export["command"].startswith("nsys stats --report ")
        assert name in export["command"]
        assert len(export["sha256"]) == 64


def test_a_missing_report_file_raises(tmp_path):
    with pytest.raises(nex.ExportFailed, match="no nsys report"):
        nex.export_all(tmp_path / "absent.nsys-rep", tmp_path)


def test_an_empty_table_is_not_a_measurement_of_zero(tmp_path):
    def runner(command):
        return subprocess.CompletedProcess(list(command), 0, "Name,Total Time (ns)\n", "")
    record = nex.export_report(Path("/dev/null"), "cuda_gpu_kern_sum",
                               tmp_path / "k.csv", runner=runner)
    assert record["status"] == "EMPTY"
    assert "NOT a measurement of zero" in record["reason"]


def test_a_failing_nsys_invocation_is_reported_not_swallowed(tmp_path):
    def runner(command):
        return subprocess.CompletedProcess(list(command), 3, "", "boom")
    record = nex.export_report(Path("/dev/null"), "cuda_gpu_kern_sum",
                               tmp_path / "k.csv", runner=runner)
    assert record["status"] == "FAILED"
    assert "boom" in record["stderr_tail"]


def test_export_refuses_to_replace_a_stale_destination(tmp_path):
    target = tmp_path / "cuda_gpu_trace.csv"
    target.write_text("STALE\n")

    def runner(command):
        return subprocess.CompletedProcess(list(command), 9, "", "failed")

    with pytest.raises(nex.ExportFailed, match="target already exists"):
        nex.export_report(
            Path("/dev/null"),
            "cuda_gpu_trace",
            target,
            runner=runner,
            run_id="run-atomic",
        )
    assert target.read_text() == "STALE\n"


# --------------------------------------------------------------------------- #
# census: BOTH 95% bars                                                        #
# --------------------------------------------------------------------------- #
def _kernels():
    """Five families, ranked to broadly track the A6 proxy.

    The rank gate needs at least three families common with the static proxy, so
    a two-kernel fixture would make the census MISSING for a reason unrelated to
    whatever the test is actually about.
    """
    return [
        {"name": "rrtmg_lw_kernel", "device_time_ns": 600e6, "launches": 120},
        {"name": "kain_fritsch_cumulus_kernel", "device_time_ns": 300e6, "launches": 90},
        {"name": "thompson_microphysics_kernel", "device_time_ns": 150e6, "launches": 60},
        {"name": "mynn_pbl_kernel", "device_time_ns": 120e6, "launches": 50},
        {"name": "advect_u_flux_kernel", "device_time_ns": 90e6, "launches": 40},
    ]


def _scope(start=0.0, end=10.0):
    return {
        "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
        "status": "OK",
        "run_id": "candidate-run",
        "capture_run_id": "candidate-run",
        "source_rep_sha256": "b" * 64,
        "production_derived": True,
        "mechanically_verified": True,
        "boundary_kind": "integration",
        "boundary_source": "nvtx_pushpop_trace",
        "start_ns": start,
        "end_ns": end,
    }


def _production_reference():
    shares = {
        "physics.radiation": 5.0,
        "physics.cumulus": 4.0,
        "physics.microphysics": 3.0,
        "physics.pbl": 2.0,
        "dycore.advection": 1.0,
    }
    return {
        "case_role": "matched_short_two_domain_production",
        "run_id": "reference-run",
        "source_rep": {"path": "/proof/reference.nsys-rep", "sha256": "a" * 64},
        "workload_identity_sha256": "c" * 64,
        "non_nesting_families": shares,
        "rank_families": list(shares),
        "active_d01_schemes": ["mp=8", "ra=4", "pbl=5", "cu=1"],
        "cadence_events": ["ordinary", "radiation", "cumulus"],
    }


def _candidate_coverage():
    return {
        "run_id": "candidate-run",
        "capture_run_id": "candidate-run",
        "source_rep_sha256": "b" * 64,
        "workload_identity_sha256": "c" * 64,
        "executed_schemes": ["mp=8", "ra=4", "pbl=5", "cu=1"],
        "executed_cadence_events": ["ordinary", "radiation", "cumulus"],
    }


def _matched_profiler():
    identity = {
        "workload_identity_sha256": "c" * 64,
        "integration_scope_sha256": "d" * 64,
        "event_mix_sha256": "e" * 64,
        "timing_region": "integration-only",
    }
    return {
        "schema": "wrf_gpu2.v025.m0.profiler_matched_pair.v1",
        "profiled": {
            **identity, "run_id": "profiled-run", "seconds": 110.0,
            "instrumentation": "nsys",
        },
        "unprofiled": {
            **identity, "run_id": "bare-run", "seconds": 100.0,
            "instrumentation": "none",
        },
        "order": ["profiled-run", "bare-run"],
    }


def _vram_evidence():
    binding = {
        "run_id": "candidate-run",
        "forecast_pid": 1234,
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "device_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    return {
        "schema": "wrf_gpu2.v025.m0.vram_evidence.v1",
        "forecast_allocator": {
            **binding,
            "schema": "wrf_gpu2.v025.m0.forecast_allocator.v1",
            "emitter_process_role": "forecast_process",
            "instrumentation": {
                "enabled": True,
                "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
                "default_when_unset": "original-direct-call-no-range-no-sidecar",
            },
            "peak_bytes_in_use": 2 << 30,
            "peak_bytes_reserved": 3 << 30,
            "measurement_start_ns": 1000,
            "measurement_end_ns": 9000,
            "measurement_start_utc": "2026-07-28T00:00:01+00:00",
            "measurement_end_utc": "2026-07-28T00:00:09+00:00",
        },
        "lock_owner_total_residency": {
            **binding,
            "schema": "wrf_gpu2.v025.m0.lock_owner_total_residency.v1",
            "sampler_process_role": "lock_owner_parent",
            "jax_imported_by_sampler": False,
            "jax_import_check": {
                "method": "sys.modules-prefix-scan-before-baseline-or-stream",
                "status": "PASS",
                "loaded_modules": [],
            },
            "orphan_control": "linux-prctl-pdeathsig-sigterm-plus-parent-stop",
            "baseline_absolute_bytes": 1 << 30,
            "peak_absolute_bytes": 4 << 30,
            "peak_baseline_subtracted_bytes": 3 << 30,
            "sample_cadence_ms": 100.0,
            "samples": 200,
            "sampling_misses": 0,
            "sampling_quality": {
                "baseline_counted_as_stream_sample": False,
                "denominator": "stream_samples_plus_inferred_or_malformed_misses",
                "observed_miss_fraction": 0.0,
                "maximum_miss_fraction": 0.05,
                "status": "PASS",
            },
            "sampling_limitations": [
                "Polling can miss a residency spike shorter than sample_cadence_ms.",
                "The metric is invalid if inferred or malformed misses exceed 5%.",
                "A wholly sub-cadence competing context cannot be observed.",
            ],
            "observed_process_tree_pids": [1234],
            "unexpected_competing_contexts": [],
            "measurement_start_ns": 0,
            "measurement_end_ns": 10_000,
            "measurement_start_utc": "2026-07-28T00:00:00+00:00",
            "measurement_end_utc": "2026-07-28T00:00:10+00:00",
        },
    }


def test_the_census_reuses_the_frozen_two_share_taxonomy():
    """Not a second implementation: the one `validate_kernel_census.py` validates."""
    import run_gpu_arm as arm
    source = (REPO / "scripts/v025/baseline_census.py").read_text()
    assert "arm.attribute_device_time" in source
    assert hasattr(arm, "attribute_device_time")


def test_unattributed_LAUNCHES_block_even_when_device_time_is_fine():
    """The gap the one-share version could not see."""
    kernels = [
        {"name": "rrtmg_lw_kernel", "device_time_ns": 1e9, "launches": 10},
        # many launches, negligible time, unattributable name
        {"name": "zzz_mystery", "device_time_ns": 1e3, "launches": 500},
    ]
    census = bc.build(kernels=kernels)
    attribution = census["device_time_attribution"]
    assert attribution["attributed_device_time_share"] > 0.95
    assert attribution["attributed_launch_share"] < 0.95
    assert attribution["status"] == "FAILED"
    assert census["status"] == "BLOCKED"


def test_no_kernels_blocks_rather_than_reporting_an_empty_census():
    census = bc.build(kernels=None)
    assert census["device_time_attribution"]["status"] == "MISSING"
    assert census["status"] == "BLOCKED"


# --------------------------------------------------------------------------- #
# census: transfers must be timestep-scoped                                    #
# --------------------------------------------------------------------------- #
def test_a_missing_transfer_report_is_missing_not_ok():
    assert bc.transfer_gate(None, _scope())["status"] == "MISSING"


def test_an_empty_transfer_report_is_missing_not_ok():
    """The previous version scored this as a pass on the one gate §7 exists for."""
    gate = bc.transfer_gate([], _scope())
    assert gate["status"] == "MISSING"
    assert "equally consistent" in gate["needs"] or "positive evidence" in gate["needs"]


def test_transfers_without_a_timestep_window_are_missing():
    rows = [{"name": "[CUDA memcpy HtoD]", "start_ns": 5, "duration_ns": 1}]
    assert bc.transfer_gate(rows, None)["status"] == "MISSING"


def test_a_host_device_copy_inside_the_loop_fails_the_gate():
    rows = [{"name": "[CUDA memcpy HtoD]", "start_ns": 5, "duration_ns": 1}]
    gate = bc.transfer_gate(rows, _scope())
    assert gate["status"] == "FAILED"
    assert gate["host_device_in_loop"] == 1


def test_an_initialisation_copy_outside_the_loop_does_not_fail_the_gate():
    """Initialisation legitimately copies inputs to the device."""
    rows = [{"name": "[CUDA memcpy HtoD]", "start_ns": 1, "duration_ns": 1}]
    gate = bc.transfer_gate(rows, _scope(100, 200))
    assert gate["status"] == "OK"
    assert gate["host_device_in_loop"] == 0


def test_aggregate_htod_without_timestamps_can_never_pass():
    rows = [{"name": "[CUDA memcpy HtoD]", "instances": 7, "seconds": 1.0}]
    gate = bc.transfer_gate(rows, _scope())
    assert gate["status"] == "MISSING"
    assert "aggregate" in gate["reason"]


def test_dry_nvtx_scope_cannot_satisfy_the_production_transfer_gate():
    scope = _scope()
    scope["dry_run_stub"] = True
    gate = bc.transfer_gate(
        [{
            "name": "[CUDA memset]",
            "start_ns": 1.0,
            "duration_ns": 1.0,
            "source_memory_kind": None,
            "destination_memory_kind": None,
        }],
        scope,
    )
    assert gate["status"] == "MISSING"
    assert "dry-run fixture" in gate["reason"]


def test_live_cuda_trace_format_preserves_timestamps_and_memory_kinds():
    text = """Start (ns),Duration (ns),SrcMemKd,DstMemKd,Bytes (MB),Name
100,25,Host,Device,1.5,[CUDA memcpy HtoD]
"""
    rows = pp.parse_cuda_gpu_trace(text)
    assert rows == [{
        "name": "[CUDA memcpy HtoD]",
        "start_ns": 100.0,
        "duration_ns": 25.0,
        "end_ns": 125.0,
        "bytes": 1_500_000.0,
        "source_memory_kind": "Host",
        "destination_memory_kind": "Device",
        "device": None,
        "context": None,
        "stream": None,
    }]


def test_an_aggregate_summary_is_rejected_by_the_trace_parser():
    with pytest.raises(pp.ProfilerParseError, match="timestamped"):
        pp.parse_cuda_gpu_trace(
            "Time (%),Total Time (ns),Instances,Name\n100,1000,7,[CUDA memcpy HtoD]\n"
        )


def test_the_timestep_window_is_missing_without_markers():
    rows = [{"name": "TSL:Thunk", "start_ns": 0, "duration_ns": 1}]
    assert nx.timestep_window(rows)["status"] == "MISSING"


def test_the_timestep_window_picks_an_ordinary_step():
    rows = [{"name": "timestep_0", "start_ns": 0, "duration_ns": 900},
            {"name": "timestep_1", "start_ns": 1000, "duration_ns": 100},
            {"name": "timestep_2", "start_ns": 2000, "duration_ns": 110}]
    window = nx.timestep_window(rows)
    assert window["status"] == "OK"
    assert window["window_ns"][0] == 2000, "median duration, not the first or longest"


# --------------------------------------------------------------------------- #
# census: profiler and VRAM are MISSING, not inferred                          #
# --------------------------------------------------------------------------- #
def test_profiler_overhead_requires_a_matched_pair():
    gate = bc.profiler_gate(None)
    assert gate["status"] == "MISSING"
    assert "matched" in gate["reason"]
    assert "with and without nsys" in gate["needs"]


def test_profiler_overhead_is_a_difference_when_the_pair_exists():
    gate = bc.profiler_gate(_matched_profiler())
    assert gate["status"] == "OK"
    assert gate["overhead_fraction"] == pytest.approx(0.10)
    assert gate["class"] == "LOW"


def test_an_incomplete_pair_is_missing():
    assert bc.profiler_gate({"profiled_seconds": 110.0})["status"] == "MISSING"


def test_profiler_pair_rejects_mismatched_workload_identity():
    pair = _matched_profiler()
    pair["unprofiled"]["event_mix_sha256"] = "f" * 64
    assert bc.profiler_gate(pair)["status"] == "MISSING"


def test_profiler_pair_rejects_clean_first_order():
    pair = _matched_profiler()
    pair["order"] = list(reversed(pair["order"]))
    gate = bc.profiler_gate(pair)
    assert gate["status"] == "MISSING"
    assert "profiled-first" in gate["reason"]


def test_peak_vram_requires_total_residency():
    gate = bc.vram_gate(None)
    assert gate["status"] == "MISSING"
    assert "capture parent" in gate["needs"]


def test_residency_is_accepted_when_actually_measured():
    gate = bc.vram_gate(_vram_evidence())
    assert gate["status"] == "OK"
    assert gate["peak_resident_mib"] == pytest.approx(3072.0)


def test_vram_sidecars_require_a_sampler_interval_enclosing_the_forecast():
    evidence = _vram_evidence()
    evidence["lock_owner_total_residency"]["measurement_end_ns"] = 8000
    gate = bc.vram_gate(evidence)
    assert gate["status"] == "FAILED"
    assert "do not enclose" in gate["reason"]


def test_vram_sidecars_require_timestamps_not_only_peak_scalars():
    evidence = _vram_evidence()
    del evidence["forecast_allocator"]["measurement_start_ns"]
    gate = bc.vram_gate(evidence)
    assert gate["status"] == "MISSING"
    assert "incomplete" in gate["reason"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evidence: evidence["forecast_allocator"].update({"schema": "wrong"}),
        lambda evidence: evidence["lock_owner_total_residency"].update(
            {"jax_imported_by_sampler": True}
        ),
        lambda evidence: evidence["forecast_allocator"]["instrumentation"].update(
            {"enabled": False}
        ),
        lambda evidence: evidence["lock_owner_total_residency"].update(
            {"observed_process_tree_pids": []}
        ),
        lambda evidence: evidence["lock_owner_total_residency"].update(
            {"baseline_absolute_bytes": 2 << 30}
        ),
        lambda evidence: evidence["lock_owner_total_residency"].update(
            {"measurement_start_utc": "2026-07-28T01:00:00+01:00"}
        ),
    ],
)
def test_vram_schema_and_provenance_mutations_never_pass(mutate):
    evidence = _vram_evidence()
    mutate(evidence)
    assert bc.vram_gate(evidence)["status"] != "OK"


# --------------------------------------------------------------------------- #
# census: any missing sub-gate blocks the whole thing                          #
# --------------------------------------------------------------------------- #
def test_a_good_attribution_alone_does_not_make_the_census_ok():
    census = bc.build(kernels=_kernels())
    assert census["device_time_attribution"]["status"] == "OK"
    assert census["status"] == "BLOCKED"
    assert set(census["gates_not_ok"]) >= {"transfer_audit", "profiler_perturbation", "vram"}


def test_the_census_is_ok_only_when_every_required_gate_is_ok():
    census = bc.build(
        kernels=_kernels(),
        mem_rows=[{"name": "[CUDA memcpy Device-to-Device]",
                   "start_ns": 5, "duration_ns": 1}],
        integration_scope=_scope(),
        production_reference=_production_reference(),
        candidate_coverage=_candidate_coverage(),
        matched_profiler=_matched_profiler(),
        residency=_vram_evidence(),
    )
    assert census["gates_not_ok"] == []
    assert census["status"] == "OK"


@pytest.mark.parametrize("drop", ["transfer_audit", "profiler_perturbation", "vram"])
def test_dropping_any_single_gate_blocks_the_census(drop):
    kwargs = dict(
        kernels=_kernels(),
        mem_rows=[{"name": "[CUDA memcpy Device-to-Device]",
                   "start_ns": 5, "duration_ns": 1}],
        integration_scope=_scope(),
        production_reference=_production_reference(),
        candidate_coverage=_candidate_coverage(),
        matched_profiler=_matched_profiler(),
        residency=_vram_evidence(),
    )
    kwargs[{"transfer_audit": "mem_rows", "profiler_perturbation": "matched_profiler",
            "vram": "residency"}[drop]] = None
    census = bc.build(**kwargs)
    assert census["status"] == "BLOCKED"
    assert drop in census["gates_not_ok"]


# --------------------------------------------------------------------------- #
# top-family rank qualification                                                #
# --------------------------------------------------------------------------- #
def test_the_static_proxy_shares_are_derived_from_the_a6_census():
    if not bc.A6_CENSUS.exists():
        pytest.skip("A6 census not on this machine")
    proxy = bc.static_proxy_shares()
    assert proxy["physics.radiation"] == pytest.approx(0.3652, abs=5e-4)
    assert sum(proxy.values()) == pytest.approx(1.0)


def test_the_rank_gate_needs_at_least_three_common_families():
    families = {"physics.radiation": {"device_time_share": 1.0}}
    assert bc.a6_static_proxy_diagnostic(
        families, {"physics.radiation": 1.0}
    )["status"] == "MISSING"


def test_identical_orderings_give_perfect_rank_agreement():
    families = {f"f{i}": {"device_time_share": 1.0 / (i + 1)} for i in range(5)}
    proxy = {f"f{i}": 1.0 / (i + 1) for i in range(5)}
    gate = bc.a6_static_proxy_diagnostic(families, proxy)
    assert gate["status"] == "DIAGNOSTIC"
    assert gate["spearman"] == pytest.approx(1.0)


def test_a_reversed_ordering_fails_the_rank_gate():
    families = {f"f{i}": {"device_time_share": 1.0 / (i + 1)} for i in range(5)}
    proxy = {f"f{i}": float(i + 1) for i in range(5)}
    gate = bc.a6_static_proxy_diagnostic(families, proxy)
    assert gate["status"] == "DIAGNOSTIC"
    assert gate["spearman"] == pytest.approx(-1.0)


def test_rho_between_old_and_frozen_threshold_fails_production_gate():
    names = [f"f{i}" for i in range(6)]
    reference = _production_reference()
    reference["non_nesting_families"] = {
        name: float(6 - index) for index, name in enumerate(names)
    }
    reference["rank_families"] = names
    candidate_values = [4.0, 5.0, 6.0, 3.0, 2.0, 1.0]  # swap rank 1 and 3
    families = {
        name: {"device_time_share": value}
        for name, value in zip(names, candidate_values)
    }
    gate = bc.production_representativeness_gate(
        families, reference, _candidate_coverage()
    )
    assert gate["spearman"] == pytest.approx(0.7714285714285715)
    assert 0.70 < gate["spearman"] < 0.80
    assert gate["status"] == "FAILED"


def test_missing_reference_family_is_not_dropped_from_the_rank_or_coverage():
    reference = _production_reference()
    families = {
        name: {"device_time_share": share}
        for name, share in reference["non_nesting_families"].items()
        if name != "physics.radiation"
    }
    gate = bc.production_representativeness_gate(
        families, reference, _candidate_coverage()
    )
    assert "physics.radiation" in gate["missing_reference_families"]
    assert gate["status"] == "FAILED"


def test_production_rank_gate_rejects_dry_fixtures_and_mismatched_workloads():
    reference = _production_reference()
    families = {
        name: {"device_time_share": share}
        for name, share in reference["non_nesting_families"].items()
    }
    reference["dry_run_stub"] = True
    assert bc.production_representativeness_gate(
        families, reference, _candidate_coverage()
    )["status"] == "MISSING"

    reference.pop("dry_run_stub")
    candidate = _candidate_coverage()
    candidate["workload_identity_sha256"] = "d" * 64
    gate = bc.production_representativeness_gate(families, reference, candidate)
    assert gate["status"] == "MISSING"
    assert "same workload identity" in gate["needs"]
