"""Step 1 artifact plumbing: capture outcome -> trace + HLO -> H1/H2/H3.

The live boundary (coordination, lock lifecycle, preemption, H1's clock) is
covered by `test_step1_boundary.py`. This file covers what the ANALYSIS process
does with the artifacts a capture leaves behind, including every way they can be
broken.

Manager review on main `965fe281` is what put these here: nsys stdout was never
persisted, H3 received a directory where its parser needs a file, H2 received no
scoped denominator, and a `<measured>` placeholder stood in for a number.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import nvtx_exclusive as nx  # noqa: E402
import run_window as rw  # noqa: E402
import step1_driver as drv  # noqa: E402
import step1_stub as stub  # noqa: E402

RUN_ID = "dryrun-test-0001"

def _prepare(out: Path, *, capture_status: str = "OK", run_id: str = RUN_ID) -> Path:
    """Everything a successful capture would have left behind.

    Including the canned export tables: the analyser now ALWAYS invokes the real
    export path, so a fixture that only drops finished CSVs in place would be
    testing a code path that no longer exists.
    """
    out.mkdir(parents=True, exist_ok=True)
    stub.write_trace(out / "step1_pushpop.csv")
    stub.write_hlo_dump(out / "dump")
    stub.write_census_inputs(out)
    stub.write_canned_exports(out / "canned_exports")
    stub.write_scope_and_representativeness(out, run_id)
    (out / "step1_autotune_off.nsys-rep").write_bytes(b"STUB\n")
    (out / "capture.log").write_text("stub capture log\n")
    outcome = {
        "status": capture_status, "killed": False, "orphans": [],
        "window": drv.WINDOW_LABEL, "run_id": run_id, "capture_kind": "stub",
    }
    manifest = drv._capture_artifact_manifest(
        out_root=out, run_id=run_id, outcome=outcome
    )
    drv._atomic_json(out / "capture_artifacts.json", manifest)
    outcome["artifact_manifest"] = {
        "path": str(out / "capture_artifacts.json"),
        "sha256": drv.sha256_file(out / "capture_artifacts.json"),
    }
    drv._atomic_json(out / "capture_outcome.json", outcome)
    return out


@pytest.fixture
def analysed(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = _prepare(tmp_path / "run")
    monkeypatch.setenv("GPUWRF_STEP1_STUB_EXPORT", str(out / "canned_exports"))
    return drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True), out


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #
def test_no_command_contains_a_placeholder():
    for command in (drv.forecast_arm_command(out_root=Path("/tmp/x")),
                    drv.nsys_command(["true"], out_root=Path("/tmp/x")),
                    drv.analyse_command(out_root=Path("/tmp/x"), run_id=RUN_ID)):
        for token in command:
            assert "<" not in token and ">" not in token, f"placeholder {token!r}"


def test_the_arm_command_comes_from_the_proven_builder():
    command = drv.forecast_arm_command(out_root=Path("/tmp/x"))
    assert "--input-dir" in command
    assert command[command.index("--hours") + 1] == "1"


def test_the_forecast_arm_is_accepted_by_the_real_gpuwrf_parser():
    """The `--hours 1.0` class of failure, checked against the actual parser."""
    probe = ("import sys, json; from gpuwrf.cli import build_parser; "
             "args = build_parser().parse_args(sys.argv[1:]); "
             "print(json.dumps({'hours': args.hours}))")
    command = drv.forecast_arm_command(out_root=Path("/tmp/x"))
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run([sys.executable, "-c", probe, *command[3:]],
                          capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    if proc.returncode != 0 and "No module named" in proc.stderr:
        pytest.skip("gpuwrf not importable here")
    assert proc.returncode == 0, f"gpuwrf's parser REJECTED it:\n{proc.stderr[-600:]}"
    assert json.loads(proc.stdout.strip().splitlines()[-1])["hours"] == 1


def test_the_hours_argument_is_an_int_not_a_float():
    command = drv.forecast_arm_command(out_root=Path("/tmp/x"))
    assert "." not in command[command.index("--hours") + 1]


def test_the_stage_environment_is_the_autotune_off_arm():
    env = drv.stage_environment(out_root=Path("/tmp/x"))
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]
    assert "--xla_dump_to=/tmp/x/dump" in env["XLA_FLAGS"]
    assert env["GPUWRF_JAX_CACHE_DIR"] == "/tmp/x/cache"


def test_the_environment_reaches_the_child_process(tmp_path):
    """Applied, not printed -- observed from INSIDE the child that ran."""
    receipt_env = dict(os.environ)
    holder = tmp_path / "holder.txt"
    holder.write_text("holder=baseline-census pid=1 token=tok cmd=y\n")
    receipt_env.update({"GPUWRF_GPU_LOCK_HELD": "1", "GPUWRF_GPU_LOCK_TOKEN": "tok",
                        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
                        "GPUWRF_GPU_LOCK_LABEL": "baseline-census"})
    import step1_dryrun as dry
    receipt = dry.write_receipt(tmp_path / "receipt.json")
    out = tmp_path / "run"
    drv.run_capture(out_root=out, receipt_path=receipt, env=receipt_env,
                    run_id=RUN_ID, stub_stage=stub.stub_stage_command(
                        out, run_id=RUN_ID),
                    ledger_hint=tmp_path / "ledger.json")
    seen = json.loads((out / "stub_env.json").read_text())
    assert "--xla_gpu_autotune_level=0" in (seen.get("XLA_FLAGS") or "")
    assert seen.get("GPUWRF_JAX_CACHE") == "1"


# --------------------------------------------------------------------------- #
# the analysis produces total objects                                          #
# --------------------------------------------------------------------------- #
def test_the_analysis_completes(analysed):
    record, _ = analysed
    # PARTIAL, not OK: dry production-reference/scope fixtures are ineligible,
    # and the capture takes no matched profiler pair or residency series.
    assert record["status"] == "PARTIAL"
    assert record["verdicts_suppressed"] is False
    assert record["deliverables_not_ok"] == ["baseline_census"]


def test_all_three_verdicts_are_emitted_and_the_census_is_honest(analysed):
    record, _ = analysed
    for key in ("H1", "H2", "H3"):
        assert record[key]["verdict"] not in {None, "SUPPRESSED"}
    census = record["baseline_census"]
    assert census["status"] == "BLOCKED"
    assert set(census["gates_not_ok"]) == {
        "production_representativeness",
        "transfer_audit",
        "profiler_perturbation",
        "vram",
    }


def test_the_trace_is_hashed_into_the_manifest(analysed):
    record, _ = analysed
    manifest = record["artifact_manifest"]["pushpop_csv"]
    assert manifest["bytes"] > 0
    assert len(manifest["sha256"]) == 64


def test_the_object_is_written_and_marked_dry(analysed, tmp_path):
    record, _ = analysed
    path = tmp_path / "proofs" / "step1_attribution_dryrun.json"
    assert json.loads(path.read_text())["status"] == "PARTIAL"
    assert record["dry_run"] is True


def test_stub_capture_cannot_be_relabelled_as_production(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = _prepare(tmp_path / "run")
    monkeypatch.setenv("GPUWRF_STEP1_STUB_EXPORT", str(out / "canned_exports"))
    record = drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=False)
    assert record["capture_kind"] == "stub"
    assert record["production_evidence_eligible"] is False
    assert record["integration_scope"]["dry_run_stub"] is True
    assert record["baseline_census"]["transfer_audit"]["status"] == "MISSING"


# --------------------------------------------------------------------------- #
# H2: scoped to the dominant module with a matching denominator                #
# --------------------------------------------------------------------------- #
def test_the_dominant_module_is_selected_mechanically(analysed):
    record, _ = analysed
    dominant = record["artifact_manifest"]["dominant_module"]
    assert dominant["module"] == stub.MODULE
    assert len(dominant["ranked"]) > 1, "selection must be a real choice"


def test_h2_is_scoped_to_that_module(analysed):
    record, _ = analysed
    assert record["H2"]["scope"]["module"] == stub.MODULE


def test_h2_shares_are_shares_of_the_scoped_denominator(analysed):
    record, _ = analysed
    scoped = record["scoped_passes"]
    assert sum(scoped["pass_exclusive_seconds"].values()) <= scoped["module_compile_seconds"] + 1e-9
    assert "same rows" in scoped["denominator_basis"]


def test_a_smaller_modules_passes_do_not_leak_into_h2(analysed):
    """The stub's second module also has a `fusion` pass; its 6 s must not count."""
    record, _ = analysed
    assert record["scoped_passes"]["module_compile_seconds"] == pytest.approx(100.0)


def test_execution_time_is_excluded_from_the_module_denominator(analysed):
    record, _ = analysed
    assert record["scoped_passes"]["module_compile_seconds"] < 150.0


# --------------------------------------------------------------------------- #
# H3: exactly one HLO representation                                          #
# --------------------------------------------------------------------------- #
def test_h3_reads_one_file_not_a_directory(analysed):
    record, _ = analysed
    path = Path(record["H3"]["scope"]["hlo_file"].replace("<OUT_ROOT>", ""))
    assert path.name.endswith("after_optimizations.txt")


def test_h3_ignores_pass_snapshots_and_before_optimizations(tmp_path):
    out = _prepare(tmp_path / "run")
    manifest = json.loads((out / "capture_artifacts.json").read_text())
    chosen = drv.select_hlo_from_manifest(
        manifest, stub.MODULE, out_root=out, run_id=RUN_ID
    )
    assert "after_optimizations" in chosen.name
    others = [p.name for p in (out / "dump").iterdir() if p != chosen]
    assert any("after_fusion" in n for n in others), "the decoys must exist"
    assert any("before_optimizations" in n for n in others)


def test_h3_refuses_when_the_chosen_stage_is_absent(tmp_path):
    out = _prepare(tmp_path / "run")
    manifest = json.loads((out / "capture_artifacts.json").read_text())
    manifest["hlo_candidates"] = [
        entry for entry in manifest["hlo_candidates"]
        if stub.MODULE not in entry["relative_path"]
    ]
    with pytest.raises(drv.Step1Blocked, match="after_optimizations"):
        drv.select_hlo_from_manifest(
            manifest, stub.MODULE, out_root=out, run_id=RUN_ID
        )


def test_h3_refuses_a_missing_dump_directory(tmp_path):
    with pytest.raises(drv.Step1Blocked, match="manifest-selected"):
        drv.select_hlo_from_manifest(
            {"hlo_candidates": []}, stub.MODULE,
            out_root=tmp_path, run_id=RUN_ID,
        )


def test_h3_records_the_hash_of_the_file_it_read(analysed):
    record, _ = analysed
    assert len(record["H3"]["scope"]["hlo_sha256"]) == 64


def test_h3_ignores_a_stale_hlo_created_after_capture_manifest(tmp_path):
    out = _prepare(tmp_path / "run")
    manifest = json.loads((out / "capture_artifacts.json").read_text())
    stale = out / "dump" / f"module_0000.{stub.MODULE}.after_optimizations.txt"
    stale.write_text("HloModule STALE\n")
    chosen = drv.select_hlo_from_manifest(
        manifest, stub.MODULE, out_root=out, run_id=RUN_ID
    )
    assert chosen != stale
    assert "module_0001" in chosen.name


def test_manifest_selected_hlo_hash_change_is_refused(tmp_path):
    out = _prepare(tmp_path / "run")
    manifest = json.loads((out / "capture_artifacts.json").read_text())
    selected = drv.select_hlo_from_manifest(
        manifest, stub.MODULE, out_root=out, run_id=RUN_ID
    )
    selected.write_text("HloModule TAMPERED\n")
    with pytest.raises(drv.Step1Blocked, match="hash changed"):
        drv.select_hlo_from_manifest(
            manifest, stub.MODULE, out_root=out, run_id=RUN_ID
        )


# --------------------------------------------------------------------------- #
# mutation: every broken artifact suppresses verdicts                          #
# --------------------------------------------------------------------------- #
def _mutated(tmp_path, monkeypatch, mutate):
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = _prepare(tmp_path / "run")
    monkeypatch.setenv("GPUWRF_STEP1_STUB_EXPORT", str(out / "canned_exports"))
    mutate(out)
    return drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True)


def _assert_suppressed(record):
    assert record["status"] == "BLOCKED"
    assert record["verdicts_suppressed"] is True
    for key in ("H1", "H2", "H3"):
        assert record[key]["verdict"] == "SUPPRESSED"
    assert record["baseline_census"]["status"] == "SUPPRESSED"


def test_a_missing_hlo_dump_suppresses_every_verdict(tmp_path, monkeypatch):
    import shutil
    _assert_suppressed(_mutated(tmp_path, monkeypatch,
                                lambda out: shutil.rmtree(out / "dump")))


def test_an_empty_trace_suppresses_every_verdict(tmp_path, monkeypatch):
    """Mutate the EXPORT SOURCE: the exporter overwrites its own output.

    Editing `step1_pushpop.csv` in place would be silently undone -- which is
    itself worth pinning, because it is the difference between testing the
    analysis and testing a file the analysis discards.
    """
    _assert_suppressed(_mutated(
        tmp_path, monkeypatch,
        lambda out: (out / "canned_exports/nvtx_pushpop_trace.csv").write_text(
            stub.HEADER + "\n")))


def test_a_trace_without_module_tags_suppresses_every_verdict(tmp_path, monkeypatch):
    def mutate(out):
        path = out / "canned_exports/nvtx_pushpop_trace.csv"
        path.write_text(path.read_text().replace("module=", "mod="))
    _assert_suppressed(_mutated(tmp_path, monkeypatch, mutate))


def test_editing_the_exported_output_is_undone_by_the_exporter(tmp_path, monkeypatch):
    """Pins WHY the mutations above target the source: the export is authoritative."""
    monkeypatch.setattr(drv, "PROOFS", tmp_path / "proofs")
    out = _prepare(tmp_path / "run")
    monkeypatch.setenv("GPUWRF_STEP1_STUB_EXPORT", str(out / "canned_exports"))
    (out / "step1_pushpop.csv").write_text("GARBAGE\n")
    record = drv.run_analyse(out_root=out, run_id=RUN_ID, dry_run=True)
    assert record["status"] == "PARTIAL", "the exporter must have rewritten the trace"


def test_a_dump_missing_the_dominant_module_suppresses_every_verdict(tmp_path, monkeypatch):
    def mutate(out):
        for path in (out / "dump").iterdir():
            if stub.MODULE in path.name:
                path.unlink()
    _assert_suppressed(_mutated(tmp_path, monkeypatch, mutate))


def test_a_blocked_run_still_writes_its_object(tmp_path, monkeypatch):
    import shutil
    _mutated(tmp_path, monkeypatch, lambda out: shutil.rmtree(out / "dump"))
    path = tmp_path / "proofs" / "step1_attribution_dryrun.json"
    assert json.loads(path.read_text())["status"] == "BLOCKED"


def test_tampered_capture_manifest_suppresses_every_verdict(tmp_path, monkeypatch):
    def mutate(out):
        manifest = json.loads((out / "capture_artifacts.json").read_text())
        manifest["run_id"] = "another-run-id"
        (out / "capture_artifacts.json").write_text(json.dumps(manifest))

    record = _mutated(tmp_path, monkeypatch, mutate)
    _assert_suppressed(record)
    assert "manifest hash mismatch" in record["blocked_reason"]


# --------------------------------------------------------------------------- #
# T_off basis                                                                  #
# --------------------------------------------------------------------------- #
def _stub_rows():
    with tempfile.TemporaryDirectory() as tmp:
        text = stub.write_trace(Path(tmp) / "t.csv").read_text()
    return nx.compute_exclusive(nx.parse_pushpop_trace(text))


def test_the_readiness_measure_excludes_forecast_execution():
    rows = _stub_rows()
    readiness = nx.compile_readiness_seconds(rows)
    thunk = [r for r in rows if nx.classify(r["name"]) == "runtime"]
    last_execution_end = max(r["start_ns"] + r["duration_ns"] for r in thunk) / 1e9
    assert readiness["t_off_seconds"] < last_execution_end


def test_the_readiness_measure_keeps_startup():
    readiness = nx.compile_readiness_seconds(_stub_rows())
    assert readiness["startup_before_first_range_seconds"] > 0
    assert readiness["t_off_seconds"] > readiness["first_to_last_range_span_seconds"]


def test_the_placeholder_is_gone_from_every_command():
    for command in (drv.nsys_command(["true"], out_root=Path("/tmp/x")),
                    drv.analyse_command(out_root=Path("/tmp/x"), run_id=RUN_ID)):
        assert not any("<measured>" in token for token in command)


# --------------------------------------------------------------------------- #
# budget                                                                       #
# --------------------------------------------------------------------------- #
def test_the_budget_fits_and_is_validated():
    budget = drv.plan()["budget"]
    assert budget["stage_budget_seconds"] == 1100.0
    assert budget["overhead_seconds"] == 180.0
    assert budget["deadline_seconds"] == 1500.0
    assert budget["headroom_seconds"] == pytest.approx(220.0)


def test_an_infeasible_plan_is_refused_by_the_same_validator():
    with pytest.raises(rw.WindowBudgetError):
        rw.validate_budget([rw.Stage(name="x", command=["true"], timeout_seconds=2000.0)],
                           deadline_seconds=drv.DEADLINE_S, overhead_seconds=drv.OVERHEAD_S)


def test_the_h1_threshold_is_derived_from_the_discriminator():
    import h1_discriminator as h1
    assert drv.H1_DECISIVE_THRESHOLD_S == pytest.approx(h1.T_OFF_PARTIAL_MAX_S, abs=0.01)


def test_the_plan_publishes_the_two_ordered_commands():
    commands = drv.plan()["commands"]
    assert "with_gpu_lock.sh" in commands["1_locked_capture"]
    assert "--analyse" in commands["2_analyse_after_release"]
    assert drv.plan()["deliverables"] == ["H1", "H2", "H3", "baseline_census"]
