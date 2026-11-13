"""scripts/branch_qualification.py: evidence gates, stop paths, admission and journaled cleanup (CPU, structural).

FIX1 for the v50-branch critic (2026-11-11): the old lane gate passed with empty arm/counter arrays, no GPU_DONE, no
backend assertion and an optional E41; it started consumers without forwarding TERM and had no window deadline or
continuous disk floor. Every test here fails if the corresponding check is removed (see HANDOFF_FIX1 mutants).
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("branch_qualification", ROOT / "scripts" / "branch_qualification.py")
bq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bq)
KEYS = ("cache_requests", "cache_hits", "cache_misses", "cache_disabled", "backend_compile_events", "fresh_compiles")
PLAN = {"variants": [{"name": "v1", "expect_differ": ["d03"]}, {"name": "v2", "expect_differ": ["d02", "d03"]}],
        "compile_allowed_arms": ["C"], "model": {"kind": "snapshot"}}
STREAMS = ("C", "R", "G2", "G3_v1", "G3_v2", "G3p_v1")
WARM = {"d01": {"loaded": True, "source": "aot_blob", "error": None},
        "fused/d02": {"loaded": True, "source": "aot_memory_cache", "error": None}}


def _counters(misses=0, fresh=0, disabled=0):
    return {"cache_requests": 10, "cache_hits": 10 - misses, "cache_misses": misses, "cache_disabled": disabled,
            "backend_compile_events": 10 - misses + fresh, "fresh_compiles": fresh}


def _evidence():
    arms = {a: {"arm": a, "status": "OK", "new_executables": ["x-cache"] * (5 if a == "C" else 0)} for a in bq.ARMS}
    return {
        "gpu_done": {"utc": "t", "stop": None},
        "arms": arms,
        "acct": {a: {"platform": "gpu", "runs": [_counters(misses=7 if a == "C" else 0)]} for a in bq.ACCT_ARMS},
        "series": {"G3": {"variants": [{"name": n, "status": "OK", "jax_counters": _counters()} for n in ("v1", "v2")]},
                   "G3p": {"variants": [{"name": "v1", "status": "OK", "platform": "gpu", "jax_counters": _counters()}]}},
        "payload_device": {k: "cuda:0" for k in STREAMS},
        "payload_aot": {k: copy.deepcopy(WARM) for k in STREAMS},
        "expected_programs": ["d01", "fused/d02"],
        "inventory": {"whole": ["f"], "branch": ["f"]},
        "stray_files": [],
        "cmp": {n: {"verdict": "PASS"} for n in ("C_inventory", "G1_null_resume", "G2_null_branch", "G3_v1", "G3_v2",
                                                 "G3x_inprocess_vs_process")},
        "e41": {"assessment": {"verdict": "WARN_DISCLOSED", "reasons": []}},
        "consumer": {},
    }


def test_complete_evidence_passes_and_e41_decision_is_disclosed():
    verdict, reasons, gates = bq.evaluate(_evidence(), PLAN, False)
    assert verdict == "PASS_E41_DISCLOSED" and not reasons and set(gates.values()) == {"PASS"}
    ev = _evidence()
    ev["e41"]["assessment"]["verdict"] = "PASS"
    assert bq.evaluate(ev, PLAN, False)[0] == "PASS"
    ev["payload_aot"]["C"] = {"d01": {"loaded": False, "source": "fallback:jit-compiled+aot-captured", "error": None}}
    assert bq.evaluate(ev, PLAN, False)[0] == "PASS"  # C may compile (E41-bound); every other arm must be warm


def _drop(path):
    def mutate(ev):
        node = ev
        for key in path[:-1]:
            node = node[key]
        node.pop(path[-1])
    return mutate


@pytest.mark.parametrize("name, gate, mutate", [
    ("no GPU_DONE", "gpu_done", lambda ev: ev.update(gpu_done=None)),
    ("empty evidence", "arms", lambda ev: ev.clear()),
    ("arm skipped", "arms", lambda ev: ev["arms"]["R2"].update(status="REFUSED")),
    ("arm missing", "arms", _drop(["arms", "G3p"])),
    ("cpu backend", "platform", lambda ev: ev["acct"]["G2"].update(platform="cpu")),
    ("payload on no GPU", "platform", lambda ev: ev["payload_device"].update(G3_v2="none")),
    ("no payload device", "platform", lambda ev: ev.update(payload_device={})),
    ("acct receipt missing", "receipts", lambda ev: ev["acct"].update(R1=None)),
    ("counter key missing", "receipts", _drop(["acct", "R2", "runs", 0, "fresh_compiles"])),
    ("empty counter arrays", "receipts", lambda ev: ev["acct"]["G2"].update(runs=[])),
    ("fresh compile without a cache request", "compile", lambda ev: ev["acct"]["G2"]["runs"][0].update(fresh_compiles=1)),
    ("cache-disabled event", "compile", lambda ev: ev["acct"]["R1"]["runs"][0].update(cache_disabled=1)),
    ("cache miss after C", "compile", lambda ev: ev["series"]["G3"]["variants"][1]["jax_counters"].update(cache_misses=1)),
    ("series variant missing", "receipts", lambda ev: ev["series"]["G3"]["variants"].pop()),
    ("process receipt missing", "receipts", lambda ev: ev["series"].update(G3p=None)),
    ("new cache entry in G3p", "compile", lambda ev: ev["arms"]["G3p"].update(new_executables=["new-cache"])),
    ("inventory missing", "inventory", lambda ev: ev["inventory"].update(branch=None)),
    ("stray file in a stream", "no_stray_files", lambda ev: ev.update(stray_files=["G2/wrfout_d01_x.bak"])),
    ("G3p child fresh compile", "compile", lambda ev: ev["series"]["G3p"]["variants"][0]["jax_counters"].update(fresh_compiles=1)),
    ("G3p child cache-disabled", "compile", lambda ev: ev["series"]["G3p"]["variants"][0]["jax_counters"].update(cache_disabled=1)),
    ("G3p child counters missing", "receipts", lambda ev: ev["series"]["G3p"]["variants"][0].update(jax_counters=None)),
    ("G3p child on cpu", "platform", lambda ev: ev["series"]["G3p"]["variants"][0].update(platform="cpu")),
    ("G2 program compiled", "aot_warm", lambda ev: ev["payload_aot"]["G2"]["d01"].update(
        loaded=False, source="fallback:jit-compiled+aot-captured")),
    ("G3p program load error", "aot_warm", lambda ev: ev["payload_aot"]["G3p_v1"]["fused/d02"].update(error="boom")),
    ("no AOT evidence for R", "aot_warm", lambda ev: ev["payload_aot"].pop("R")),
    # critic Delta 2 repro (critic_evidence/d2_missing_warm_program.json): G3p keeps d01, loses fused/d02
    ("warm stream misses a program", "aot_warm", lambda ev: ev["payload_aot"]["G3p_v1"].pop("fused/d02")),
    ("warm stream with an unexpected program", "aot_warm", lambda ev: ev["payload_aot"]["G2"].update(
        {"fused/d05": {"loaded": True, "source": "aot_blob", "error": None}})),
    ("expected program set missing", "aot_warm", lambda ev: ev.update(expected_programs=None)),
    ("comparison missing", "cmp:G1_null_resume", _drop(["cmp", "G1_null_resume"])),
    ("comparison failed", "cmp:G3_v2", lambda ev: ev["cmp"]["G3_v2"].update(verdict="FAIL")),
    ("E41 assessment missing", "e41", lambda ev: ev.update(e41={})),
    ("E41 assessment FAIL", "e41", lambda ev: ev["e41"]["assessment"].update(verdict="FAIL", reasons=["x"])),
])
def test_missing_or_failed_evidence_is_fail(name, gate, mutate):
    ev = copy.deepcopy(_evidence())
    mutate(ev)
    verdict, reasons, gates = bq.evaluate(ev, PLAN, False)
    assert verdict == "FAIL" and reasons, name
    assert gates[gate] == "FAIL", (name, gates)  # the specific gate refuses, not only a neighbour


def test_dry_run_demands_the_stub_and_cpu():
    ev = _evidence()
    assert bq.evaluate(ev, PLAN, True)[0] == "FAIL"  # GPU evidence in a dry run is wrong too
    for a in bq.ACCT_ARMS:
        ev["acct"][a]["platform"] = "cpu"
    ev["series"]["G3p"]["variants"][0]["platform"] = "cpu"
    ev["payload_device"] = {k: "STUB-DRYRUN" for k in ev["payload_device"]}
    assert bq.evaluate(ev, PLAN, True)[0] == "DRYRUN_PASS"


# ---- stop paths -------------------------------------------------------------------------------------------------
def _plan(tmp_path, **extra):
    return {"data_mount": str(tmp_path), "floors_gib": {"data": 0, "root": 0}, "projection_gib": {"data": 0, "root": 0},
            "own_cap_gib": 1000, "cpu_reserve_s": 0, "kill_grace_s": 2, "lease": str(tmp_path / "lease"),
            "nightly": None, **extra}


SLEEPER = [sys.executable, "-c", "import os, time; open(os.environ['PIDFILE'], 'w').write(str(os.getpid())); time.sleep(120)"]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_pidfile(path):
    for _ in range(100):
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("child never started")


def test_arm_budget_includes_teardown(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    env = {**os.environ, "PIDFILE": str(tmp_path / "pid")}
    t0 = time.time()
    row = bq.run_arm(SLEEPER, env, (), tmp_path, tmp_path / "log", 8, 2, _plan(tmp_path), tmp_path)
    assert row["status"] == "TIMEOUT" and time.time() - t0 <= 8 + 5.5  # cap incl. grace (+ one 5 s poll)
    assert not _alive(int((tmp_path / "pid").read_text()))


def test_continuous_disk_floor_stops_the_running_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    plan = _plan(tmp_path, floors_gib={"data": 0, "root": 10 ** 6})  # / can never satisfy this floor
    env = {**os.environ, "PIDFILE": str(tmp_path / "pid")}
    row = bq.run_arm(SLEEPER, env, (), tmp_path, tmp_path / "log", 120, 2, plan, tmp_path)
    assert row["status"] == "STOPPED_DISK" and row["wall_s"] < 20
    assert not _alive(int((tmp_path / "pid").read_text()))


def test_term_kills_the_arm_session_and_journals(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    env = {**os.environ, "PIDFILE": str(tmp_path / "pid")}
    proc = subprocess.Popen(SLEEPER, env=env, start_new_session=True)
    pid = _wait_pidfile(tmp_path / "pid")
    bq.CURRENT.update(proc=proc, root=root, arms=[], arm="G2")
    try:
        with pytest.raises(SystemExit) as stop:
            bq._terminate(signal.SIGTERM, None)
    finally:
        bq.CURRENT.update(proc=None, root=None, arms=None, arm=None)
    assert stop.value.code == 128 + signal.SIGTERM
    assert not _alive(pid)
    assert json.loads((root / "ARMS.json").read_text())[-1]["status"] == "STOPPED_SIGNAL"


# ---- admission --------------------------------------------------------------------------------------------------
def test_admission_refuses_quiet_hard_end_and_own_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    monkeypatch.setattr(bq, "QUIET", tmp_path / "quiet")
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("V50_DEADLINE_EPOCH", str(time.time() + 1000))
    assert bq.admit(_plan(tmp_path), root, 500)["ok"]
    with pytest.raises(RuntimeError, match="hard end"):
        bq.admit(_plan(tmp_path), root, 1200)  # would cross the window end: refused, not truncated
    with pytest.raises(RuntimeError, match="hard end"):
        bq.admit(_plan(tmp_path, cpu_reserve_s=600), root, 500)
    (tmp_path / "quiet").write_text("")
    with pytest.raises(RuntimeError, match="QUIET"):
        bq.admit(_plan(tmp_path), root, 10)
    (tmp_path / "quiet").unlink()
    (root / "big").write_bytes(b"x" * 2048)
    with pytest.raises(RuntimeError, match="own byte cap"):
        bq.admit(_plan(tmp_path, own_cap_gib=1e-6), root, 10)
    rows = [json.loads(line) for line in (root / "ADMISSION.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in rows] == [True, False, False, False, False]


def test_admission_takes_the_earlier_lease_expiry(tmp_path, monkeypatch):
    monkeypatch.delenv("V50_DRY_RUN", raising=False)
    monkeypatch.setenv("V50_DEADLINE_EPOCH", str(time.time() + 10000))
    lease = tmp_path / "lease"
    lease.write_text("expires-utc: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 100)) + "\n")
    assert bq.hard_end(_plan(tmp_path)) < time.time() + 200
    lease.unlink()
    with pytest.raises(RuntimeError, match="lease missing"):
        bq.hard_end(_plan(tmp_path))


def test_nightly_window_overlap_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("V50_DRY_RUN", raising=False)
    monkeypatch.setattr(bq, "QUIET", tmp_path / "quiet")
    monkeypatch.setenv("V50_DEADLINE_EPOCH", str(time.time() + 100000))
    (tmp_path / "lease").write_text("expires-utc: 2099-01-01T00:00:00Z\n")
    now = time.time()
    monkeypatch.setattr(bq, "nightly_window", lambda cfg, t: ([(now + 300, now + 4000)], {}))
    root = tmp_path / "root"
    root.mkdir()
    plan = _plan(tmp_path, nightly={"timer": "t"})
    with pytest.raises(RuntimeError, match="Nightly window"):
        bq.admit(plan, root, 600)


# ---- cleanup ----------------------------------------------------------------------------------------------------
def test_cleanup_refuses_without_result_and_journals_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    parent = tmp_path / "dry"
    root = parent / "gpuarm_20261111T000000Z_dry"
    for sub in ("C", "C_scratch", "G2_scratch", "R_scratch", "G3/v1/scratch", "cache/jax", "digests"):
        (root / sub).mkdir(parents=True)
    (root / "C" / "wrfout_d01_2026-02-28_01:00:18").write_bytes(b"frame")
    (root / "G2_scratch" / "transient.nc").write_bytes(b"s")
    (root / "digests" / "C.json").write_text(json.dumps({"source": str(root / "C"),
                                                        "frames": {"wrfout_d01_2026-02-28_01:00:18": {"file_sha": "ab"}}}))
    plan = {"run_parent": "/nonexistent", "dry_run_parent": str(parent), "evidence_dir": str(tmp_path / "evidence")}
    with pytest.raises(SystemExit, match="refused"):
        bq.cleanup(plan, root)
    assert (root / "C").exists()
    (root / "RESULT.json").write_text(json.dumps({"verdict": "FAIL", "e41": {"receipt": None}}))
    bq.cleanup(plan, root)
    for sub in ("C", "C_scratch", "G2_scratch", "R_scratch", "G3"):
        assert not (root / sub).exists(), sub
    assert (root / "cache").exists()  # no E41 receipt: the cache stays for an E41 rerun
    journal = [json.loads(line) for line in (root / "RAW_DELETE_JOURNAL.jsonl").read_text().splitlines()]
    frame = next(j for j in journal if j.get("path", "").endswith("01:00:18"))
    assert frame["sha256"] == "ab" and journal[0]["event"] == "START" and journal[-1]["event"] == "END"
    assert (tmp_path / "evidence" / root.name / "RESULT.json").exists()


def test_checked_root_refuses_foreign_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    plan = {"run_parent": "/x", "dry_run_parent": str(tmp_path)}
    for bad in (tmp_path / "gpuarm_20261111T000000Z", tmp_path / "other_dry", Path("/x") / "gpuarm_20261111T000000Z_dry"):
        with pytest.raises(SystemExit):
            bq.checked_root(plan, bad)
    assert bq.checked_root(plan, tmp_path / "gpuarm_20261111T000000Z_dry")


# ---- Delta 1 (critic 2026-11-11 02:15Z) ---------------------------------------------------------------------------
BUNDLE_PLAN = dict(PLAN, model={"kind": "bundle_consumer", "bundle": "/B", "layer": "/L", "env": {}})


def _consumer(compiled_new=0, after="BUNDLE_OK missing=0 changed=0 new=0 aliases=0 moved=False", rc=0):
    return {"rc": rc, "compiled_new": compiled_new, "receipt": "/L/runs/x.json", "bundle_check_after": after}


@pytest.mark.parametrize("arm, consumer", [("G2", _consumer(compiled_new=3)), ("R1", _consumer(after="BUNDLE_BROKEN x")),
                                           ("G3p", None), ("C", _consumer(rc=1))])
def test_bundle_mode_requires_consumer_evidence(arm, consumer):
    ev = _evidence()
    ev["consumer"] = {a: _consumer(compiled_new=7 if a == "C" else 0) for a in bq.ARMS}
    assert bq.evaluate(ev, BUNDLE_PLAN, False)[0] == "PASS_E41_DISCLOSED"  # C may compile (E41-bound)
    ev["consumer"][arm] = consumer
    verdict, _, gates = bq.evaluate(ev, BUNDLE_PLAN, False)
    assert verdict == "FAIL" and gates["bundle_consumer"] == "FAIL"


def test_consumer_line_is_parsed():
    text = ("x\nP check after: BUNDLE_OK missing=0 changed=0 new=0 aliases=0 moved=False\n"
            "BUNDLE_CONSUMER rc=0 compiled_new=0 aliases_new=0 runtime_new=12 other_new=0 layer_compiled=5381 receipt=/L/r.json\n")
    assert bq.consumer_evidence(text) == {"rc": 0, "compiled_new": 0, "receipt": "/L/r.json",
                                          "bundle_check_after": "BUNDLE_OK"}
    assert bq.consumer_evidence("no consumer here") is None


def test_bundle_env_is_exactly_the_layer_build_env(tmp_path, monkeypatch):
    monkeypatch.delenv("V50_DRY_RUN", raising=False)
    for k in ("GPUWRF_GPU_LOCK_FD", "GPUWRF_GPU_LOCK_HELD", "GPUWRF_GPU_LOCK_FILE", "GPUWRF_GPU_LOCK_HOLDER_FILE",
              "GPUWRF_GPU_LOCK_LABEL", "GPUWRF_GPU_LOCK_TOKEN"):
        monkeypatch.setenv(k, "x")
    built = {"GPUWRF_FAST_DEFAULTS": "1", "GPUWRF_JAX_CACHE_DIR": "/B/jax", "JAX_COMPILATION_CACHE_DIR": "/B/jax",
             "GPUWRF_XLA_AUTOTUNE_CACHE_DIR": "/B/autotune", "TRITON_CACHE_DIR": "/B/triton", "CUDA_CACHE_PATH": "/B/cuda"}
    plan = dict(BUNDLE_PLAN, model={"kind": "bundle_consumer", "bundle": "/B", "layer": "/L", "env": built})
    env = bq.child_env(plan, tmp_path)
    assert {k: env[k] for k in built} == built and env["PYTHONPATH"] == "/B/src/src" and env["JAX_PLATFORMS"] == "cuda"
    assert bq.consumer_prefix(plan) == ["/B/tools/bundle_consumer.sh", "--bundle", "/B", "--layer", "/L", "--"]
    assert bq.cache_dir(plan, tmp_path) == Path("/L/upper")


# E41 assessment ---------------------------------------------------------------------------------------------------
FUSION = "loop_add_compare_divide_power_select_fusion"


def _decisions(tmp_path):
    for name in ("opt3.md", "warn.md", "policy.md"):
        (tmp_path / name).write_text(name)
    return {"option3_incomplete": {"file": str(tmp_path / "opt3.md"), "sha256": bq.sha(tmp_path / "opt3.md"),
                                   "also": [{"file": str(tmp_path / "policy.md"), "sha256": bq.sha(tmp_path / "policy.md")}]},
            "structural_warn": {"file": str(tmp_path / "warn.md"), "sha256": bq.sha(tmp_path / "warn.md"),
                                "fusions": {FUSION: {"hlo": 564, "sass": 1142}, FUSION + ".1": {"hlo": 522, "sass": 1016}}}}


def _gate(out, *, verdict="FAIL", flagged=True, unknown=True, sass=(1096, 960), coverage="PASS", extra_flag=False,
          two_flagged=False, aot_elf="e", no_kernel=0, no_kernel_patch=None, only_no_kernel=False):
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    def entry(name, kind, status, flags, program=None):
        path = f"/L/upper/jax/{name}"
        elf = aot_elf if kind == "aot" else "e"
        entries.append({"entry": name, "path": path, "source_kind": kind, "program": program, "kinput_transpose_status": status,
                        "kinput_transpose_over500": flags, "elfs": [{"sha256": elf, "nonleaf_over1k": [], "callgraph_unknown": [],
                                                                     "stack_over256": []}]})
        label = bq.hashlib.sha256(path.encode()).hexdigest()[:12]
        rows = [{"kernel": "loop_add_fusion_1", "sass": 96, "calls": 0, "stack": 0}]
        if flags:
            rows += [{"kernel": FUSION, "sass": sass[0], "calls": 0, "stack": 0},
                     {"kernel": FUSION + "_1", "sass": sass[1], "calls": 0, "stack": 0}]
        (out / f"cubin_scan_{label}_e0.json").write_text(json.dumps(rows))
    flags = [[FUSION, 564], [FUSION + ".1", 522]] + ([["other_fusion", 600]] if extra_flag else [])
    entry("jit__advance_chunk_fori-a-cache", "jax_cache", "MEASURED_RICH_JIT_HLO", [])
    entry("jit_fused_jit-b-cache", "jax_cache", "MEASURED_RICH_JIT_HLO", flags if flagged else [])
    if two_flagged:
        entry("jit_fused_jit-c-cache", "jax_cache", "MEASURED_RICH_JIT_HLO", flags)
    if unknown:
        entry("k.xlaexec", "aot", "UNKNOWN_AOT_METADATA", None, "fused_d02")
    if only_no_kernel:
        entries.clear()
    for i in range(no_kernel):  # wave_cubin_gate NO_KERNEL record (eager identity convert)
        rec = {"entry": f"jit_convert_element_type-{i}-cache", "path": f"/L/upper/jax/jit_convert_element_type-{i}-cache",
               "source_kind": "jax_cache", "program": None, "kernel_class": "NO_KERNEL", "elfs": [],
               "no_kernel_hlo_ops": {"copy": 1, "parameter": 1}, "kinput_transpose_status": "NO_KERNEL_IDENTITY_HLO",
               "kinput_transpose_over500": []}
        if no_kernel_patch:
            rec.update(no_kernel_patch)
        entries.append(rec)
    gate = {"e41_gate": verdict, "hard_stack_gate": "PASS", "entries": entries,
            "no_measured_kernels": bool(entries) and all(e.get("kernel_class") == "NO_KERNEL" for e in entries),
            "coverage": {"status": coverage, "n_discovered": len(entries), "n_scanned": len(entries)}}
    (out / "wave_gate.json").write_text(json.dumps(gate))
    return out


def test_e41_bound_warn_and_incomplete_are_disclosed(tmp_path):
    dec = _decisions(tmp_path)
    res = bq.e41_assess(_gate(tmp_path / "a"), 2, dec)
    assert res["verdict"] == "WARN_DISCLOSED" and not res["reasons"] and len(res["disclosed"]) == 3
    res = bq.e41_assess(_gate(tmp_path / "b", verdict="INCOMPLETE", flagged=False), 2, dec)
    assert res["verdict"] == "PASS_DISCLOSED"
    res = bq.e41_assess(_gate(tmp_path / "c", verdict="PASS", flagged=False, unknown=False), 0, dec)
    assert res["verdict"] == "PASS"


@pytest.mark.parametrize("verdict, kwargs, rc, want", [
    ("PASS", {"flagged": False, "unknown": False}, 0, "PASS"),
    ("INCOMPLETE", {"flagged": False}, 2, "PASS_DISCLOSED"),
    ("FAIL", {}, 2, "WARN_DISCLOSED"),
])
def test_e41_no_kernel_entries_need_no_decision(tmp_path, verdict, kwargs, rc, want):
    res = bq.e41_assess(_gate(tmp_path / "n", verdict=verdict, no_kernel=6, **kwargs), rc, _decisions(tmp_path))
    assert res["verdict"] == want and not res["reasons"] and res["no_kernel"] == 6


@pytest.mark.parametrize("patch", [{"elfs": [{"sha256": "e", "nonleaf_over1k": [], "callgraph_unknown": [], "stack_over256": []}]},
                                   {"source_kind": "aot"}, {"kinput_transpose_status": "UNKNOWN_HLO_ERROR"},
                                   {"no_kernel_hlo_ops": {"copy": 1, "fusion": 1}}, {"no_kernel_hlo_ops": {}}])
def test_e41_malformed_no_kernel_record_is_fail(tmp_path, patch):
    res = bq.e41_assess(_gate(tmp_path / "m", verdict="INCOMPLETE", flagged=False, no_kernel=1, no_kernel_patch=patch), 2,
                        _decisions(tmp_path))
    assert res["verdict"] == "FAIL" and any("NO_KERNEL" in r for r in res["reasons"])


def test_e41_only_no_kernel_entries_is_fail(tmp_path):
    res = bq.e41_assess(_gate(tmp_path / "o", verdict="FAIL", only_no_kernel=True, no_kernel=3), 2, _decisions(tmp_path))
    assert res["verdict"] == "FAIL" and any("no measured kernels" in r for r in res["reasons"])


@pytest.mark.parametrize("name, kwargs, rc, decision_patch", [
    ("timeout rc", {}, 124, None),
    ("PASS with rc 2", {"verdict": "PASS", "flagged": False, "unknown": False}, 2, None),
    ("coverage FAIL", {"coverage": "FAIL"}, 2, None),
    ("SASS over the tolerance", {"sass": (1150, 960)}, 2, None),
    ("unknown fusion flagged", {"extra_flag": True}, 2, None),
    ("two flagged entries in one family", {"two_flagged": True}, 2, None),
    ("warn decision file changed", {}, 2, "structural_warn"),
    ("option3 decision missing", {"flagged": False, "verdict": "INCOMPLETE"}, 2, "option3_incomplete"),
    ("compact AOT ELF matches no rich entry", {"flagged": False, "verdict": "INCOMPLETE", "aot_elf": "other"}, 2, None),
    ("option3 base policy changed", {"flagged": False, "verdict": "INCOMPLETE"}, 2, "option3_also"),
])
def test_e41_unbound_or_unsuccessful_is_fail(tmp_path, name, kwargs, rc, decision_patch):
    dec = _decisions(tmp_path)
    if decision_patch == "option3_also":
        Path(dec["option3_incomplete"]["also"][0]["file"]).write_text("edited after pinning")
    elif decision_patch:
        Path(dec[decision_patch]["file"]).write_text("edited after pinning")
    res = bq.e41_assess(_gate(tmp_path / "g", **kwargs), rc, dec)
    assert res["verdict"] == "FAIL" and res["reasons"], name
    assert bq.e41_assess(tmp_path / "absent", 0, dec)["verdict"] == "FAIL"


# session lifetime, admission growth, hashing before deletion -------------------------------------------------------
def test_descendants_are_torn_down_after_the_leader_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    script = (f"import os, subprocess, sys; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
              f"open({str(tmp_path / 'child')!r}, 'w').write(str(p.pid)); os._exit(0)")
    row = bq.run_arm([sys.executable, "-c", script], dict(os.environ), (), tmp_path, tmp_path / "log", 60, 3,
                     _plan(tmp_path), tmp_path)
    child = int((tmp_path / "child").read_text())
    assert row["status"] == "FAILED" and child in row["leftover_pids"]
    assert not _alive(child)


def test_admission_projects_only_the_remaining_growth(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    monkeypatch.setattr(bq, "QUIET", tmp_path / "quiet")
    monkeypatch.setenv("V50_DEADLINE_EPOCH", str(time.time() + 10000))
    monkeypatch.setattr(bq, "own_bytes", lambda root: 5 * bq.GIB)
    root = tmp_path / "root"
    root.mkdir()
    plan = _plan(tmp_path, own_cap_gib=20, projection_gib={"data": 16, "root": 0})
    rec = bq.admit(plan, root, 10)  # 5 GiB written + 11 GiB still to come <= 20 GiB cap
    assert rec["ok"] and rec["remaining_growth_gib"] == 11
    monkeypatch.setattr(bq, "own_bytes", lambda root: 21 * bq.GIB)
    with pytest.raises(RuntimeError, match="own byte cap"):
        bq.admit(plan, root, 10)


def test_cleanup_hashes_undigested_raw_before_deleting(tmp_path, monkeypatch):
    monkeypatch.setenv("V50_DRY_RUN", "1")
    parent = tmp_path / "dry"
    root = parent / "gpuarm_20261111T000000Z_dry"
    (root / "G2").mkdir(parents=True)
    (root / "G2" / "wrfout_d01_2026-02-28_04:00:18").write_bytes(b"undigested frame")
    (root / "RESULT.json").write_text(json.dumps({"verdict": "FAIL", "e41": {"receipt": None}}))
    plan = {"run_parent": "/nonexistent", "dry_run_parent": str(parent), "evidence_dir": str(tmp_path / "evidence")}
    bq.cleanup(plan, root)
    rows = [json.loads(line) for line in (root / "RAW_DELETE_JOURNAL.jsonl").read_text().splitlines() if '"path"' in line]
    assert rows and all(r["sha256"] for r in rows)
    assert rows[0]["sha256"] == bq.hashlib.sha256(b"undigested frame").hexdigest() and rows[0]["hash_source"] == "hashed_at_delete"


def test_expected_programs_come_from_the_namelist():
    wn3 = " &domains\n max_dom = 3,\n parent_id = 0, 1, 2,\n /\n"
    nsrt = " &domains\n max_dom = 9,\n parent_id = 0, 1, 2, 2, 2, 2, 2, 2, 2,\n /\n"
    two_children = " &domains\n max_dom = 3,\n parent_id = 0, 1, 1,\n /\n"
    assert bq.expected_programs(wn3) == ["d01", "fused/d02"]
    assert bq.expected_programs(nsrt) == ["d01", "fused/d02"]
    assert bq.expected_programs(two_children) == ["d01", "fused/d02", "fused/d03"]


# ---- ALISIOS Nightly rule 2026-11-11T03:39Z ('Nightly fertig'), hardened per the critic Nightly-rule delta ----------
# Cases 1-27 are ported from the critic-PASSed reference (v50-evert gpu/test_nightly_gate.py, evert_tools 7face710).
from datetime import datetime as _dt, timezone as _tz

NOW = int(_dt(2026, 11, 11, 6, 0, tzinfo=_tz.utc).timestamp())  # scheduled issue = 20261110_18z
GOOD_STEPS = {k: {"status": "DONE"} for k in ("prep3", "prep9", "gpu", "extract", "cleanup")}
DEG_STEPS = dict(GOOD_STEPS, nest333={"status": "DEGRADED"})


def _st(verdict="DONE", done="2026-11-11T02:49:44+00:00", steps=None, issue="20261110_18z"):
    d = {"issue": issue, "verdict": verdict, "steps": GOOD_STEPS if steps is None else steps}
    if done is not None:
        d["done_utc"] = done
    return json.dumps(d)


NIGHTLY_CASES = [  # name, {issue_dir: text or None (dir without STATUS.json)}, expect_ok
    ("DONE + gpu DONE", {"20261110_18z": _st()}, True),
    ("DEGRADED + gpu DONE (tonight's shape)", {"20261110_18z": _st("DEGRADED", steps=DEG_STEPS)}, True),
    ("DEGRADED, done_utc missing", {"20261110_18z": _st("DEGRADED", done=None, steps=DEG_STEPS)}, False),
    ("DONE, done_utc empty", {"20261110_18z": _st(done="")}, False),
    ("DONE, done_utc garbage", {"20261110_18z": _st(done="yesterday")}, False),
    ("DONE, done_utc in the future", {"20261110_18z": _st(done="2026-11-12T00:00:00+00:00")}, False),
    ("FAILED + gpu DONE", {"20261110_18z": _st("FAILED")}, False),
    ("DEGRADED + gpu DEGRADED", {"20261110_18z": _st("DEGRADED", steps=dict(GOOD_STEPS, gpu={"status": "DEGRADED"}))}, False),
    ("DONE + gpu FAILED", {"20261110_18z": _st(steps=dict(GOOD_STEPS, gpu={"status": "FAILED"}))}, False),
    ("DONE + gpu step missing", {"20261110_18z": _st(steps={k: v for k, v in GOOD_STEPS.items() if k != "gpu"})}, False),
    ("DONE + gpu not a dict", {"20261110_18z": _st(steps=dict(GOOD_STEPS, gpu="DONE"))}, False),
    ("DONE + steps missing", {"20261110_18z": _st(steps={})}, False),
    ("RUNNING", {"20261110_18z": _st("RUNNING", done=None, steps=dict(GOOD_STEPS, gpu={"status": "RUNNING"}))}, False),
    ("DONE but a step RUNNING", {"20261110_18z": _st(steps=dict(GOOD_STEPS, upload={"status": "RUNNING"}))}, False),
    ("future-named issue (outside the 4-day scan) DONE but a step RUNNING", {"20261110_18z": _st(), "20261120_18z": _st(issue="20261120_18z", steps=dict(GOOD_STEPS, upload={"status": "RUNNING"}))}, False),
    ("DONE but a step lowercase 'running'", {"20261110_18z": _st(steps=dict(GOOD_STEPS, upload={"status": "running"}))}, False),
    ("older recent issue verdict lowercase 'running'", {"20261110_18z": _st(), "20261109_18z": _st("running", done=None, issue="20261109_18z")}, False),
    ("older recent issue step lowercase 'queued'", {"20261110_18z": _st(), "20261109_18z": _st(issue="20261109_18z", steps=dict(GOOD_STEPS, upload={"status": "queued"}))}, False),
    ("future-named issue step mixed-case 'Started'", {"20261110_18z": _st(), "20261120_18z": _st(issue="20261120_18z", steps=dict(GOOD_STEPS, upload={"status": "Started"}))}, False),
    ("lowercase 'done' step elsewhere stays not busy, 18z DONE ok", {"20261110_18z": _st(steps=dict(GOOD_STEPS, upload={"status": "done"}))}, True),
    ("lowercase 'degraded'", {"20261110_18z": _st("degraded", steps=DEG_STEPS)}, False),
    ("issue field mismatch", {"20261110_18z": _st(issue="20261109_18z")}, False),
    ("STATUS unreadable (bad JSON)", {"20261110_18z": "{not json"}, False),
    ("STATUS is a JSON list", {"20261110_18z": "[]"}, False),
    ("STATUS missing", {"20261110_18z": None}, False),
    ("only older issue DONE", {"20261109_18z": _st(issue="20261109_18z")}, False),
    ("newer 18z issue present, not finished", {"20261110_18z": _st(), "20261111_18z": _st("PREP", done=None, steps={}, issue="20261111_18z")}, False),
    ("newer 18z issue present, DEGRADED + gpu DONE",
     {"20261110_18z": _st("FAILED"), "20261111_18z": _st("DEGRADED", done="2026-11-11T05:00:00+00:00", steps=DEG_STEPS, issue="20261111_18z")}, True),
    ("newer 18z issue DIR without STATUS.json", {"20261110_18z": _st(), "20261111_18z": None}, False),
    ("newer 18z issue dir with bad STATUS.json", {"20261110_18z": _st(), "20261111_18z": "{"}, False),
    ("non-date *_18z dir ignored, 18z DONE ok", {"20261110_18z": _st(), "x9_18z": None, "2026111x_18z": None}, True),
    ("deeply nested STATUS.json", {"20261110_18z": "[" * 100000 + "]" * 100000}, False),
    ("12z SKIPPED alongside, 18z DEGRADED ok",
     {"20261110_18z": _st("DEGRADED", steps=DEG_STEPS), "20261110_12z": _st("SKIPPED", done=None, steps={}, issue="20261110_12z")}, True),
    # v50-branch additions (12z pass-over, busy states, tz-aware done_utc)
    ("newer 12z SKIPPED is passed over",
     {"20261110_18z": _st("DEGRADED", steps=DEG_STEPS), "20261111_12z": _st("SKIPPED", done=None, steps={}, issue="20261111_12z")}, True),
    ("newer 12z WAITING_SOURCE decides (not finished)",
     {"20261110_18z": _st(), "20261111_12z": _st("WAITING_SOURCE", done=None, steps={}, issue="20261111_12z")}, False),
    ("newer 12z dir without STATUS.json", {"20261110_18z": _st(), "20261111_12z": None}, False),
    ("scheduled 18z SKIPPED is never passed over",
     {"20261110_18z": _st("SKIPPED", done=None, steps={}), "20261109_18z": _st(issue="20261109_18z")}, False),
    ("older recent issue with a step RUNNING", {"20261110_18z": _st(), "20261109_12z": _st("SKIPPED", done=None,
     steps={"prep3": {"status": "RUNNING"}}, issue="20261109_12z")}, False),
    ("latest step QUEUED", {"20261110_18z": _st(steps=dict(GOOD_STEPS, upload={"status": "QUEUED"}))}, False),
    ("latest step STARTED", {"20261110_18z": _st(steps=dict(GOOD_STEPS, extract={"status": "STARTED"}))}, False),
    ("verdict PENDING", {"20261110_18z": _st("PENDING")}, False),
    ("done_utc without timezone", {"20261110_18z": _st(done="2026-11-11T02:49:44")}, False),
    ("done_utc as a number", {"20261110_18z": _st(done=5)}, False),
    ("no issue dirs at all", {}, False),
]


@pytest.mark.parametrize("name, files, expect", NIGHTLY_CASES, ids=[c[0] for c in NIGHTLY_CASES])
def test_nightly_gate(tmp_path, name, files, expect):
    root = tmp_path / "nightly"
    root.mkdir()
    for issue, text in files.items():
        (root / issue).mkdir()
        if text is not None:
            (root / issue / "STATUS.json").write_text(text)
    doc = bq.nightly_gate(root, NOW)
    assert doc["ok"] == expect, (name, doc["problems"])
    if not expect:
        assert doc["problems"], name


def test_scheduled_issue_floor():
    assert bq.scheduled_issue(NOW) == "20261110_18z"
    assert bq.scheduled_issue(_dt(2026, 11, 11, 1, 0, tzinfo=_tz.utc).timestamp()) == "20261109_18z"  # before 01:45Z
    assert bq.scheduled_issue(_dt(2026, 11, 11, 14, 59, tzinfo=_tz.utc).timestamp()) == "20261110_18z"


def test_nightly_window_is_local_canary_time_and_dst_correct(monkeypatch):
    monkeypatch.setattr(bq.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "@1794362733"})())
    cfg = {"timer": "t", "local_tz": "Atlantic/Canary", "protected_local": ["18:50", "06:00"]}
    overlaps = lambda t, budget: any(t < b and t + budget > a for a, b in bq.nightly_window(cfg, t)[0])  # noqa: E731
    winter_noon = _dt(2026, 11, 11, 11, 0, tzinfo=_tz.utc).timestamp()
    assert not overlaps(winter_noon, 4 * 3600)                 # 11-15Z free (WET = UTC)
    assert overlaps(winter_noon, int(7.9 * 3600))              # reaches 18:50Z
    assert overlaps(_dt(2026, 11, 11, 5, 30, tzinfo=_tz.utc).timestamp(), 60)  # before 06:00 local
    summer = _dt(2027, 6, 1, 17, 40, tzinfo=_tz.utc).timestamp()  # WEST: 18:40 local
    assert overlaps(summer, 15 * 60)                           # 17:40Z + 15 min reaches 18:50 local = 17:50Z
    assert not overlaps(_dt(2027, 6, 1, 5, 5, tzinfo=_tz.utc).timestamp(), 3600)  # 06:05 local free
    assert bq.nightly_window(cfg, winter_noon)[1]["timer_next_poll_utc"] == "2026-11-11T02:05:33Z"  # recorded only


def test_admission_refuses_when_the_nightly_is_not_finished(tmp_path, monkeypatch):
    monkeypatch.delenv("V50_DRY_RUN", raising=False)
    monkeypatch.setattr(bq, "QUIET", tmp_path / "quiet")
    monkeypatch.setenv("V50_DEADLINE_EPOCH", str(time.time() + 100000))
    monkeypatch.setattr(bq, "nightly_window", lambda cfg, t: ([(t + 50000, t + 60000)], {}))
    monkeypatch.setattr(bq, "lease_end", lambda plan: None)
    monkeypatch.setattr(bq.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "", "returncode": 0})())
    root = tmp_path / "root"
    root.mkdir()
    status_root = tmp_path / "nightly"
    sched = bq.scheduled_issue(time.time())
    (status_root / sched).mkdir(parents=True)
    st = json.loads(_st("DEGRADED", steps=dict(DEG_STEPS, gpu={"status": "FAILED"})))
    st.update(issue=sched)
    (status_root / sched / "STATUS.json").write_text(json.dumps(st))
    plan = _plan(tmp_path, nightly={"timer": "t", "status_root": str(status_root)})
    with pytest.raises(RuntimeError, match="Nightly not finished"):
        bq.admit(plan, root, 600)
    st["steps"]["gpu"] = {"status": "DONE"}
    (status_root / sched / "STATUS.json").write_text(json.dumps(st))
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HOLDER_FILE", str(tmp_path / "holder"))
    rec = bq.admit(plan, root, 600)
    assert rec["ok"] and rec["nightly_gate"]["latest_issue"] == sched
