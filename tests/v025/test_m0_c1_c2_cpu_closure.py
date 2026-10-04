"""CPU-only end-to-end and mutation proofs for the M0 C1/C2 closure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_c1_c2_cpu_proofs as proofs  # noqa: E402
import m0_postlock_census as postlock  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import build_m0_c1_c2_cpu_closure_evidence as closure_builder  # noqa: E402
import build_m0_c1_c2_cpu_residual_evidence as residual_builder  # noqa: E402
import build_m0_exact_boundary_evidence as exact_builder  # noqa: E402


def _fresh_json(code: str, *, extra_env: dict[str, str] | None = None) -> dict:
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if extra_env:
        environment.update(extra_env)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_terminal_status_is_reserved_for_combined_c1_c2_proof():
    assert exact_builder.STATUS == "CPU_C1_GREEN_C2_PENDING"
    assert closure_builder.STATUS == "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
    assert residual_builder.STATUS == closure_builder.STATUS
    assert residual_builder.C1_STATUS == "BLOCKED_RUNTIME_RESIDUAL"


def test_r8_residual_is_bound_without_becoming_a_c1_pass():
    residual = residual_builder.collect_r8_residual()
    assert (
        residual["status"]
        == "RESIDUAL_EXACT_COMPUTE_COMPLETE_PUBLICATION_REFUSED"
    )
    assert residual["accepted_as_c1_proof"] is False
    assert residual["wrfout_published"] is False
    assert residual["fresh_cpu_wrf_comparator_run"] is False
    assert residual["process_tree_peak_rss_bytes"] == 21_454_475_264
    assert residual["machine_identity"]["cpu_affinity"] == [4]
    assert (
        residual["machine_identity"]["environment"]["JAX_PLATFORMS"]
        == "cpu"
    )


def test_m0_core_and_deferred_are_separate_and_nothing_is_waived():
    partition = residual_builder.milestone_partition()
    assert partition["M0_CORE"]["status"] == "BLOCKED"
    assert partition["M0_CORE"]["may_open_m1"] is False
    assert (
        partition["M0_CORE"]["fields"]["c1_same_result_wrfout_runtime"]
        == "BLOCKED_R8_PUBLICATION_REFUSED"
    )
    assert partition["M0_DEFERRED"]["waived"] is False
    assert set(partition["M0_DEFERRED"]["fields"]) == {
        "instruction_counter_E_baseline_T_ceiling",
        "native_pallas_sm120",
        "ncu_registers_occupancy_stalls_dram_l2",
        "two_domain_rho_family_coverage_trace",
        "multi_size_vram_scaling_fit",
    }


def test_terminal_manifest_validator_rehashes_every_member(
    tmp_path, monkeypatch
):
    # Other modules in the monolithic pytest process intentionally import the
    # model. A separate test below proves the real closure import path is
    # accelerator-free; this test isolates content re-hashing.
    monkeypatch.setattr(closure_builder, "_accelerator_modules", lambda: [])
    c1_path = tmp_path / "c1.json"
    c2_path = tmp_path / "c2.json"
    closure_path = tmp_path / "closure.json"
    manifest_path = tmp_path / "manifest.json"
    junit_path = tmp_path / "tests.xml"
    closure_builder._atomic_json(c1_path, {"status": "PASS"})
    closure_builder._atomic_json(c2_path, {"status": "PASS"})
    junit_path.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" '
        'skipped="1"/></testsuites>\n',
        encoding="utf-8",
    )
    closure = {
        "schema": closure_builder.CLOSURE_SCHEMA,
        "status": closure_builder.STATUS,
        "device_touched": False,
        "device_queries": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "gpu_windows": {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"},
        "m0_complete": False,
        "source_contract": closure_builder.exact_contract.validate_sources(),
        "production_identity": {
            "src_gpuwrf_tree": closure_builder.PRODUCTION_TREE
        },
        "c1": {
            "status": "PASS",
            "artifact": closure_builder._artifact(c1_path),
        },
        "c2": {
            "status": "PASS",
            "artifact": closure_builder._artifact(c2_path),
        },
        "tests_v025": {
            "artifact": closure_builder._artifact(junit_path),
            "totals": {"tests": 3, "failures": 0, "errors": 0, "skipped": 1},
        },
    }
    closure["closure_sha256"] = closure_builder._canonical_sha256(closure)
    closure_builder._atomic_json(closure_path, closure)
    manifest = {
        "schema": closure_builder.MANIFEST_SCHEMA,
        "status": closure_builder.STATUS,
        "device_action": False,
        "files": [
            closure_builder._artifact(path)
            for path in (c1_path, c2_path, closure_path, junit_path)
        ],
    }
    manifest["manifest_payload_sha256"] = (
        closure_builder._canonical_sha256(manifest)
    )
    closure_builder._atomic_json(manifest_path, manifest)

    result = closure_builder.validate_terminal_artifacts(
        closure_path=closure_path,
        manifest_path=manifest_path,
    )
    assert result["status"] == "PASS"
    assert result["files_verified"] == 4

    c1_path.write_text('{"status":"BLOCKED"}\n', encoding="utf-8")
    with pytest.raises(closure_builder.ClosureError, match="C1 artifact"):
        closure_builder.validate_terminal_artifacts(
            closure_path=closure_path,
            manifest_path=manifest_path,
        )


def test_complete_census_fixture_recomputes_every_frozen_arithmetic():
    proof = _fresh_json(
        f"""
import json,sys
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_c1_c2_cpu_proofs as p
print(json.dumps(p.complete_census_proof()))
"""
    )
    assert proof["status"] == "PASS"
    assert proof["fixture_is_device_evidence"] is False
    assert all(proof["arithmetic"].values())
    assert proof["census"]["status"] == "OK"
    assert proof["census"]["gates_not_pass"] == []
    assert (
        proof["census"]["device_time_attribution"][
            "attributed_launch_share"
        ]
        >= 0.95
    )
    assert (
        proof["census"]["device_time_attribution"][
            "attributed_device_time_share"
        ]
        >= 0.95
    )
    partition = proof["census"]["measurement_partition"]
    assert (
        partition["m0_core_m1_blocking"]["fields"]
        == list(postlock.NSYS_M1_BLOCKING_FIELDS)
    )
    assert (
        partition["m0_deferred_m2"]["fields"]
        == list(postlock.NCU_M2_DEFERRED_FIELDS)
    )
    assert partition["m0_deferred_m2"]["status"] == "MISSING_UNTIL_M2"
    assert partition["m0_deferred_m2"]["waived"] is False


def test_ncu_deferred_partition_cannot_be_hidden_or_promoted_to_m0_core():
    result = _fresh_json(
        f"""
import json,sys
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_c1_c2_cpu_proofs as p
import m0_postlock_census as c
value=c.build_census(**p.complete_census_inputs())
value["measurement_partition"]["m0_deferred_m2"]["waived"]=True
value["census_sha256"]=c.canonical_sha256({{
 k:v for k,v in value.items() if k!="census_sha256"
}})
try:
 c.validate_census(value)
except Exception as exc:
 print(json.dumps({{"rejected":True,"error":str(exc)}}))
else:
 print(json.dumps({{"rejected":False}}))
"""
    )
    assert result["rejected"] is True
    assert "ncu-only" in result["error"]


def test_complete_census_attack_matrix_fails_closed():
    matrix = _fresh_json(
        f"""
import json,sys
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_c1_c2_cpu_proofs as p
print(json.dumps(p.census_attack_matrix()))
"""
    )
    assert matrix["total"] >= 15
    assert matrix["rejected"] == matrix["total"]
    assert matrix["status"] == "PASS"


def test_top_level_ok_cannot_hide_a_missing_required_gate():
    result = _fresh_json(
        f"""
import json,sys
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_c1_c2_cpu_proofs as p
import m0_postlock_census as c
value=c.build_census(**p.complete_census_inputs())
value["required_gates"]["host_ram"]["status"]="MISSING"
value["census_sha256"]=c.canonical_sha256({{
 k:v for k,v in value.items() if k!="census_sha256"
}})
try:
 c.validate_census(value)
except Exception as exc:
 print(json.dumps({{"rejected":True,"error":str(exc)}}))
else:
 print(json.dumps({{"rejected":False}}))
"""
    )
    assert result["rejected"] is True
    assert "inventory is stale" in result["error"]


def test_release_proof_requires_observed_return_and_unchanged_result(tmp_path):
    manager = tmp_path / "w2.json"
    release_path = tmp_path / "release.json"
    result = _fresh_json(
        f"""
import json,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_postlock_census as c
manager=Path({str(manager)!r})
release_path=Path({str(release_path)!r})
manager.write_text(json.dumps({{
 "status":"OK","window":"W2","run_id":"release-cpu-fixture"
}}))
now=time.monotonic_ns()
release=c.build_lock_release_proof(
 window_id="W2",run_id="release-cpu-fixture",
 wrapper_command=["CPU_DRY_WRAPPER"],
 wrapper_started_monotonic_ns=now-2,
 wrapper_returned_monotonic_ns=now-1,wrapper_returncode=0,
 window_result_path=manager,output_path=release_path,
 wrapper_started_at_utc=datetime.now(timezone.utc).isoformat(),
 wrapper_returned_at_utc=datetime.now(timezone.utc).isoformat())
errors=[]
try:
 c.validate_lock_release_proof(
  release_path=release_path,window_result_path=manager,
  analysis_started_monotonic_ns=release["wrapper_returned_monotonic_ns"],
  expected_window="W2")
except Exception as exc: errors.append(str(exc))
good=c.validate_lock_release_proof(
 release_path=release_path,window_result_path=manager,
 analysis_started_monotonic_ns=time.monotonic_ns(),expected_window="W2")
manager.write_text(manager.read_text()+" ")
try:
 c.validate_lock_release_proof(
  release_path=release_path,window_result_path=manager,
  analysis_started_monotonic_ns=time.monotonic_ns(),expected_window="W2")
except Exception as exc: errors.append(str(exc))
print(json.dumps({{"errors":errors,"good":good["status"]}}))
"""
    )
    assert result["good"] == "LOCK_WRAPPER_RETURNED"
    assert "strictly after" in result["errors"][0]
    assert "changed" in result["errors"][1]


def test_postlock_entry_refuses_inherited_lock_environment(
    tmp_path,
):
    result = _fresh_json(
        f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_postlock_census as c
try:
 c.analyze_w2(
  w2_result_path=Path({str(tmp_path / "not-read.json")!r}),
  release_path=Path({str(tmp_path / "not-read-release.json")!r}),
  output_path=Path({str(tmp_path / "not-written.json")!r}),
  export_root=Path({str(tmp_path / "not-created")!r}))
except Exception as exc:
 print(json.dumps({{"rejected":True,"error":str(exc)}}))
else:
 print(json.dumps({{"rejected":False}}))
""",
        extra_env={"GPUWRF_GPU_LOCK_HELD": "1"},
    )
    assert result["rejected"] is True
    assert "inherited" in result["error"]
    assert not (tmp_path / "not-written.json").exists()


def test_postlock_entry_refuses_accelerator_module_contamination(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", types.ModuleType("jax"))
    with pytest.raises(postlock.PostlockRefusal, match="accelerator roots"):
        postlock.assert_accelerator_free()


def test_importing_closure_proof_path_is_accelerator_free():
    code = f"""
import json
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
before = set(sys.modules)
import m0_c1_c2_cpu_proofs
import m0_host_rss_sampler
import m0_postlock_census
import build_m0_c1_c2_cpu_closure_evidence
import build_m0_c1_c2_cpu_residual_evidence
after = set(sys.modules)
print(json.dumps({{
    "jax": any(n == "jax" or n.startswith("jax.") for n in after - before),
    "jaxlib": any(n == "jaxlib" or n.startswith("jaxlib.") for n in after - before),
    "gpuwrf": any(n == "gpuwrf" or n.startswith("gpuwrf.") for n in after - before),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "jax": False,
        "jaxlib": False,
        "gpuwrf": False,
    }


def test_host_rss_sampler_cli_wraps_fresh_process_without_accelerator(
    tmp_path,
):
    output = tmp_path / "host-rss-cli.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "m0_host_rss_sampler.py"),
            "--output",
            str(output),
            "--run-id",
            "cpu-rss-cli",
            "--stage",
            "unit",
            "--cadence-ms",
            "5",
            "--",
            sys.executable,
            "-c",
            "import time; x=bytearray(2<<20); time.sleep(.03); assert x[0]==0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    payload = json.loads(output.read_text())
    assert summary["command_returncode"] == 0
    assert payload["status"] == "PASS"
    assert payload["run_id"] == "cpu-rss-cli"
    assert payload["peak_process_tree_rss_bytes"] > 0
    assert payload["device_action"] is False


def test_host_rss_sampler_records_machine_affinity_and_peak(tmp_path):
    output = tmp_path / "host-rss.json"
    payload = _fresh_json(
        f"""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_host_rss_sampler as h
s=h.ProcessTreeRssSampler(
 root_pid=os.getpid(),output_path=Path({str(output)!r}),cadence_ms=5)
s.start()
x=bytearray(2<<20)
time.sleep(.02)
p=s.finish()
assert x[0]==0
print(json.dumps(p))
"""
    )
    assert payload["status"] == "PASS"
    assert payload["hostname"]
    assert payload["cpu_affinity"]
    assert payload["peak_process_tree_rss_bytes"] > 0
    assert payload["sampler_ram_scaling"].startswith("O(")
    assert postlock.sha256_file(output)


def _running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = stat[stat.rfind(")") + 2 :].split()
    return bool(fields) and fields[0] != "Z"


def test_nonzero_child_exit_sweeps_sigterm_trapping_descendant(
    tmp_path,
):
    import wrf_source_authority as wsa

    authority = wsa.build_source_authority(
        namelist_path=Path(
            "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1/namelist.input"
        ),
        environ={
            variable: str(wsa.CANONICAL_ROOT)
            for variable in wsa.ROOT_ENV_VARS
        },
    )
    authority_path = tmp_path / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    grandchild_pid_path = tmp_path / "grandchild.pid"
    spawner = tmp_path / "nonzero_tree.py"
    spawner.write_text(
        "\n".join(
            [
                "import subprocess, signal, sys, time",
                "from pathlib import Path",
                "child = subprocess.Popen([",
                "    sys.executable, '-c',",
                "    'import signal,time;'",
                "    'signal.signal(signal.SIGTERM,lambda *_:None);'",
                "    'signal.signal(signal.SIGINT,lambda *_:None);'",
                "    'time.sleep(300)',",
                "])",
                "Path(sys.argv[1]).write_text(str(child.pid))",
                "time.sleep(0.1)",
                "raise SystemExit(19)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import json,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_window_parent as parent
started=time.monotonic()
try:
 parent._launch_authorized_child(
  window_id="W1",stage="nonzero-trapping-descendant",
  run_id=parent.WINDOWS["W1"]["run_id"],mode="clean",
  cache_path=Path({str(tmp_path / "cache")!r}),
  run_root=Path({str(tmp_path / "run")!r}),
  authorization={{
   "label":parent.WINDOWS["W1"]["label"],
   "receipt_fingerprint":"f"*64,
   "receipt_spent_at_utc":datetime.now(timezone.utc).isoformat()}},
  timeout_seconds=10.0,
  command_builder=lambda _h,_r:[
   sys.executable,{str(spawner)!r},{str(grandchild_pid_path)!r}])
except Exception as exc:
 print(json.dumps({{"error":str(exc),"seconds":time.monotonic()-started}}))
else:
 print(json.dumps({{"error":None,"seconds":time.monotonic()-started}}))
""",
        encoding="utf-8",
    )
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(driver)],
        cwd=REPO,
        env={
            **os.environ,
            "GPUWRF_GPU_LOCK_TOKEN": "cpu-only-token",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            wsa.AUTHORITY_ENV_VAR: str(authority_path),
            **wsa.child_environment_binding(authority),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "child returned" in result["error"]
    assert time.monotonic() - started < 6.0
    pid = int(grandchild_pid_path.read_text())
    deadline = time.monotonic() + 3.0
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _running(pid)


def test_cpu_dry_three_window_graph_remains_device_free_and_missing():
    payload = executor.dry_run_all()
    assert payload["status"] == "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
    assert payload["device_touched"] is False
    assert payload["gpu_evidence_status"] == "MISSING"
    assert payload["receipts_consumed"] == []


@pytest.mark.skipif(
    not proofs.W1B_REPORT.is_file(),
    reason="existing incomplete W1b report is unavailable",
)
def test_real_incomplete_w1b_runs_full_postlock_path_and_stays_blocked(
    tmp_path,
):
    result = _fresh_json(
        f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
import m0_c1_c2_cpu_proofs as p
print(json.dumps(p.run_incomplete_w1b_postlock(
 report_path=Path({str(proofs.W1B_REPORT)!r}),
 run_root=Path({str(tmp_path / "w1b-postlock")!r}))))
"""
    )
    assert result["status"] == "PASS"
    assert result["path_status"] == "BLOCKED_AS_REQUIRED"
    assert result["analysis"]["status"] == "BLOCKED"
    assert result["analysis"]["release_before_analysis"] is True
    assert result["analysis"]["accelerator_modules_imported"] == []
    assert not any(result["analysis"]["lock_environment_present"].values())
