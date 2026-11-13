"""Extract OFF identity: no traced source touched, step program/key unchanged, OFF policy unchanged; CLI wiring."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


def _changed_files():
    base = subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", "main"], capture_output=True, text=True)
    if base.returncode:
        pytest.skip("git main ref unavailable")
    committed = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", base.stdout.strip(), "HEAD"],
                               capture_output=True, text=True, check=True).stdout.split()
    working = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.split()
    return {(ROOT / p).resolve() for p in committed + working}


def test_no_trace_closure_source_changed_and_extract_outside_closure():
    from gpuwrf.runtime import aot_cheap_key as ck
    import gpuwrf.extract.collector as collector
    scope = {Path(p).resolve() for p in ck._trace_reachable_source_files(ck._gpuwrf_package_root())}
    assert len(scope) > 100
    extract_files = set(Path(collector.__file__).resolve().parent.glob("*.py"))
    assert not (scope & extract_files)
    changed = _changed_files()
    assert not (scope & changed), sorted(scope & changed)
    # The hooks live in orchestration modules that are outside the closure by design.
    for rel in ("src/gpuwrf/integration/nested_pipeline.py", "src/gpuwrf/cli.py",
                "src/gpuwrf/runtime/restart_store.py"):
        assert (ROOT / rel).resolve() not in scope


def test_step_program_and_cheap_key_unchanged_by_extract(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_aot_cheap_key import _build_call
    import jax
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
    carry, nl, clock = _build_call()
    start = jax.numpy.asarray(1, jax.numpy.int32)
    kwargs = dict(n_steps=1, cadence=1)

    def observe():
        jax.clear_caches()
        key = ck.cheap_key(_advance_chunk_fori, (carry, nl, start, clock), kwargs, nl)
        hlo = _advance_chunk_fori.lower(carry, nl, start, clock, **kwargs).compiler_ir("hlo")
        return key, hashlib.sha256(hlo.as_hlo_text(print_large_constants=True).encode()).hexdigest(), \
            ck.source_fingerprint_hash()

    before = observe()
    # Turn the extract fully on in this process: import, request, collector thread.
    from gpuwrf.extract.collector import ExtractCollector, DomainPlan
    from gpuwrf.extract.request import load_request
    from datetime import datetime, timezone
    cfg = tmp_path / "x.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "r", "member": "m",
                               "issue_utc": "20260725_12z"}))
    col = ExtractCollector(load_request("v1", cfg, tmp_path / "x"),
                           run_start=datetime(2026, 7, 25, 18, tzinfo=timezone.utc), input_dir=tmp_path,
                           plans={"d01": DomainPlan("d01", 54.0, 3600.0, (0, 67))})
    after = observe()
    col.close()
    assert before[0] is not None and before == after


def test_restart_output_policy_absent_when_off(tmp_path):
    from gpuwrf.runtime.restart_store import _nested_output_policy
    from gpuwrf.extract.request import load_request
    bundles = {"d01": SimpleNamespace(namelist=SimpleNamespace(diagnostic_history_interval=60.0))}
    off = _nested_output_policy(SimpleNamespace(input_dir=tmp_path, emit_initial_history=True), bundles)
    assert "extract" not in off
    cfg = tmp_path / "x.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "r", "member": "m",
                               "issue_utc": "20260725_12z"}))
    request = load_request("v1", cfg, tmp_path / "x")
    on = _nested_output_policy(SimpleNamespace(input_dir=tmp_path, emit_initial_history=True, extract=request),
                               bundles)
    assert on["extract"] == request.policy()
    assert {k: v for k, v in on.items() if k != "extract"} == off


def test_cli_extract_flags(tmp_path, capsys, monkeypatch):
    if not (PROD / "namelist.input").is_file():
        pytest.skip("PROD case unavailable")
    from gpuwrf import cli
    monkeypatch.delenv("GPUWRF_BATCH_ENSEMBLE", raising=False)
    cfg = tmp_path / "x.json"
    cfg.write_text(json.dumps({"schema": "gpuwrf.extract.request.v1", "run_id": "r", "member": "m",
                               "issue_utc": "20260725_12z"}))
    common = ["run", "--input-dir", str(PROD), "--output-dir", str(tmp_path / "out"), "--max-dom", "2",
              "--hours", "3", "--dry-run"]
    extract = ["--extract-spec", "v1", "--extract-config", str(cfg), "--extract-dir", str(tmp_path / "ex")]
    parser = cli.build_parser()
    assert cli._cmd_run(parser.parse_args(common + extract[:2])) == 2
    assert "must be given together" in capsys.readouterr().err
    assert cli._cmd_run(parser.parse_args(common + extract)) == 2
    assert "--emit-initial-history" in capsys.readouterr().err
    rc = cli._cmd_run(parser.parse_args(common + extract + ["--emit-initial-history"]))
    out = capsys.readouterr()
    assert rc == 0, out.err
    plan = json.loads(out.out)
    assert plan["extract"]["spec"] == "v1" and plan["extract"]["run_id"] == "r"
    rc = cli._cmd_run(parser.parse_args(common))
    assert rc == 0 and "extract" not in json.loads(capsys.readouterr().out)
    assert not (tmp_path / "ex").exists()  # dry run binds nothing on disk
