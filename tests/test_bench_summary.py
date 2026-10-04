"""CPU test for scripts/bench_summary.py on a synthetic harness receipt (no GPU)."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("bench_summary", Path(__file__).parents[1] / "scripts/bench_summary.py")
bs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bs)


def _receipt(period: float) -> dict:
    segs = [{"k": k, "root_steps": 67, "t_start": 100.0 + (k - 1) * period, "t_return": 100.0 + k * period - 1}
            for k in (1, 2, 3)]
    t2 = segs[1]["t_start"]
    return {
        "rc": 0, "t_end": 100.0 + 3 * period, "segments": segs, "git_head": "h", "src_tree": "t", "jax_version": "x",
        "outputs": [{"d": "d02", "own_step": 1, "t_start": t2 + 10, "s": 4.0}],
        "compile_events": [{"t": t2 + 12, "event": "/jax/core/compile/backend_compile_duration", "s": 1.5}],
        "advance": [{"d": "d01", "s0": 1, "n": 1, "t": t2 + 1, "host_s": 20.0}],
        "force": [{"t": t2 + 2, "host_s": 1.0}],
        "derived": {"vram_peak_mib": 123}, "env": {},
    }


def test_arm_and_aa(tmp_path):
    h = 67 * 54.0 / 3600.0
    arms = []
    for i, period in enumerate((100.0, 104.0)):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_receipt(period)))
        arms.append(bs.arm_summary(p))
    a = arms[0]
    assert abs(a["s_per_fc_h_seg2plus"] - 100.0 / h) < 0.01
    assert abs(a["output_s_per_fc_h"] - 4.0 / h / 2) < 0.01  # output only in segment 2 of the 2 warm segments
    seg2 = a["segments"][1]
    assert seg2["output_compile_s_per_fc_h"] == round(1.5 / h, 3)
    assert abs(seg2["host_idle_est_s_per_fc_h"] - (100.0 - 20 - 1 - 4) / h) < 0.01
    aa = bs.aa_floor(arms)
    assert abs(aa["aa_floor_pct"] - 4.0) < 0.01 and aa["within_historical_floor"]


# --- bench_prod.sh --src: measure an arbitrary source tree (CPU only, no GPU/lock) ---
import os  # noqa: E402
import time  # noqa: E402
import subprocess  # noqa: E402

BENCH = Path(__file__).parents[1] / "scripts" / "bench_prod.sh"


def _git_repo_with_src(root: Path) -> Path:
    (root / "src" / "gpuwrf").mkdir(parents=True)
    (root / "src" / "gpuwrf" / "__init__.py").write_text("")
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    return root


def _print_src(tmp_path, repo, *args):
    env = dict(os.environ, BENCH_SNAP_ROOT=str(tmp_path / "snap"))
    return subprocess.run(["bash", str(BENCH), "--src", str(repo), "--print-src", "--tag", "T", *args],
                          capture_output=True, text=True, env=env)


def test_bench_prod_print_src_snapshots_rev(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD:src/gpuwrf"],
                          capture_output=True, text=True, check=True).stdout.strip()
    r = _print_src(tmp_path, repo)
    assert r.returncode == 0, r.stderr
    kv = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    assert Path(kv["SRC_REPO"]).resolve() == repo.resolve()
    assert kv["SRC_REV"] == rev
    assert kv["SRC_TREE"] == tree
    assert Path(kv["SNAP"]) == (tmp_path / "snap" / rev[:12]) and Path(kv["SNAP"]).is_dir()
    assert kv["PYTHONPATH"] == str(Path(kv["SNAP"]) / "src")  # imports only from the snapshot
    assert "wrf_gpu2_lanes" in kv["CACHE"]                   # disk lane root, not /tmp
    assert kv["DIRTY"] == "0"
    assert Path(kv["HARNESS"]).is_file() and kv["HARNESS"].endswith("bench/s0_nested_harness.py")


def test_bench_prod_rejects_bad_src(tmp_path):
    r = subprocess.run(["bash", str(BENCH), "--src", str(tmp_path / "nope"), "--print-src"],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_summary_records_src(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_receipt(100.0)))
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", "--src", "/some/src", "--src-tree", "deadbeef",
                  "--src-rev", "cafe", f"cold={p}", f"warm1={p}", f"warm2={p}"])
    assert rc == 0
    j = json.loads(out.read_text())
    assert j["src"] == {"path": "/some/src", "rev": "cafe", "tree_hash": "deadbeef"}
    assert j["tree"]["src_tree"] == "deadbeef"


def test_bench_prod_refuses_dirty_src(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    (repo / "src" / "gpuwrf" / "dirty_new.py").write_text("# untracked edit\n")
    r = _print_src(tmp_path, repo)
    assert r.returncode != 0
    assert "DIRTY=1" in r.stdout and "dirty src/gpuwrf" in r.stderr


def test_compile_attribution_no_grace():
    h = 67 * 54.0 / 3600.0
    rec = _receipt(100.0)
    rec["outputs"] = [{"d": "d02", "own_step": 1, "t_start": 200.0, "s": 4.0}]  # output interval [200, 204]
    # compile ends 0.25 s AFTER the output ends: interval [204.15, 204.25] must NOT be attributed to output
    rec["compile_events"] = [{"t": 204.25, "event": "/jax/core/compile/backend_compile_duration", "s": 0.1}]
    seg2 = [r for r in bs.segment_rows(rec) if r["k"] == 2][0]
    assert seg2["output_compile_s_per_fc_h"] == 0.0
    assert seg2["compile_total_s_per_fc_h"] == round(0.1 / h, 3)  # still counted as in-segment compile
    # a compile overlapping the output ([200.5, 202]) IS attributed
    rec["compile_events"] = [{"t": 202.0, "event": "/jax/core/compile/backend_compile_duration", "s": 1.5}]
    seg2 = [r for r in bs.segment_rows(rec) if r["k"] == 2][0]
    assert seg2["output_compile_s_per_fc_h"] == round(1.5 / h, 3)


def test_headline_rejected_on_mixed_trees(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_receipt(100.0)))
    r = _receipt(104.0)
    r["src_tree"] = "different_tree"
    b.write_text(json.dumps(r))
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", f"warm1={a}", f"warm2={b}"])
    j = json.loads(out.read_text())
    assert rc != 0 and j["ok"] is False and j["headline_valid"] is False
    assert j["s1_warm_s_per_fc_h"] is None and j["aa"] is None
    assert any("src trees" in x for x in j["reject_reasons"])


def test_headline_rejected_on_single_warm(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_receipt(100.0)))
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", f"warm1={a}"])
    j = json.loads(out.read_text())
    assert rc != 0 and j["s1_warm_s_per_fc_h"] is None
    assert any("warm arms" in x for x in j["reject_reasons"])


def test_extra_arm_excluded_from_headline(tmp_path):
    # a 6 h warm arm named 'extra6h' must not enter the A/A headline (only names starting 'warm' do)
    files = {}
    for name, period in (("warm1", 100.0), ("warm2", 104.0), ("extra6h", 90.0)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(_receipt(period)))
        files[name] = p
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", *[f"{n}={p}" for n, p in files.items()]])
    assert rc == 0  # warm1/warm2 present -> headline valid even with the extra arm
    j = json.loads(out.read_text())
    assert j["aa"] == {"a": list(v for k, v in j["arms"].items() if k == "warm1")[0]["s_per_fc_h_seg2plus"],
                       "b": list(v for k, v in j["arms"].items() if k == "warm2")[0]["s_per_fc_h_seg2plus"],
                       "aa_floor_pct": 4.0, "historical_floor_pct": 6.77, "within_historical_floor": True}
    assert "extra6h" in j["arms"]


def test_bench_prod_accepts_extra_hours(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--extra-hours", "6")
    assert r.returncode == 0, r.stderr


def test_bench_prod_refuses_prewrap(tmp_path):
    # per-arm locking: a global with_gpu_lock wrap would deadlock (re-flock) -> must refuse loudly
    repo = _git_repo_with_src(tmp_path / "repo")
    env = dict(os.environ, GPUWRF_GPU_LOCK_HELD="1", BENCH_SNAP_ROOT=str(tmp_path / "snap"))
    r = subprocess.run(["bash", str(BENCH), "--src", str(repo), "--arms", "1", "--tag", "T"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "each ARM acquires the lock" in r.stderr


def test_run_arm_helper_present():
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    assert helper.is_file() and os.access(helper, os.X_OK)


def test_census_arm_excluded_from_headline(tmp_path):
    files = {}
    for name, period in (("warm1", 100.0), ("warm2", 104.0), ("census", 90.0)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(_receipt(period)))
        files[name] = p
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", *[f"{n}={p}" for n, p in files.items()]])
    assert rc == 0  # census arm is not named 'warm*' -> excluded from the A/A headline
    j = json.loads(out.read_text())
    assert "census" in j["arms"] and j["aa"]["aa_floor_pct"] == 4.0


def test_bench_prod_accepts_census_arm(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--census-arm")
    assert r.returncode == 0, r.stderr


def test_census_attach_requires_manifest_for_enabled_arm(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    try:
        from gpuwrf.diagnostics import census_report
    except Exception as exc:  # heavy optional import (gpuwrf/jax); skip rather than fail the CPU suite
        import pytest
        pytest.skip(f"census_report import unavailable: {exc!r}")
    import pytest
    # non-census arm with no manifest -> skipped; census-enabled arm with no manifest -> error
    good = {"arms": {"warm1": {"receipt": str(tmp_path / "w1" / "run" / "receipt.json"), "env": {}}}}
    bp = tmp_path / "b1.json"
    bp.write_text(json.dumps(good))
    assert census_report.attach_bench(bp) == {}  # nothing to attach, no raise

    bad = {"arms": {"census": {"receipt": str(tmp_path / "c" / "run" / "receipt.json"),
                               "env": {"GPUWRF_CENSUS": "1"}}}}
    bp2 = tmp_path / "b2.json"
    bp2.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        census_report.attach_bench(bp2)


def test_run_arm_propagates_rc_and_returns(tmp_path):
    # run_arm.sh must run the command, then return its rc (drain loop disabled via threshold 0 in the test)
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"
    stub.write_text("s0_guard() { :; }\n")
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0")
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "", "", "--",
                        "bash", "-c", "exit 7"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 7


def test_run_arm_reexports_xla_flags(tmp_path):
    # s0_env.sh unsets XLA_FLAGS; run_arm.sh must re-export the requested flags AFTER sourcing it
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"
    stub.write_text("s0_guard() { :; }\nunset XLA_FLAGS\n")
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", XLA_FLAGS="stale-must-be-overwritten")
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "--xla_gpu_autotune_level=0", "", "",
                        "--", "bash", "-c", "printf %s \"$XLA_FLAGS\""], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and r.stdout == "--xla_gpu_autotune_level=0"


def test_bench_prod_accepts_xla_flags(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--xla-flags", "--xla_gpu_autotune_level=0")
    assert r.returncode == 0, r.stderr


def test_summary_records_xla_flags(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_receipt(100.0)))
    out = tmp_path / "bench.json"
    rc = bs.main(["--out", str(out), "--tag", "T", "--xla-flags=--xla_gpu_autotune_level=0",
                  f"warm1={p}", f"warm2={p}"])
    assert rc == 0 and json.loads(out.read_text())["xla_flags"] == "--xla_gpu_autotune_level=0"


def test_run_arm_quiet_file_lifecycle(tmp_path):
    # a quiet label must create $BENCH_QUIET_FILE while the arm runs and remove it on exit (trap)
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"
    stub.write_text("s0_guard() { :; }\n")
    qf = tmp_path / "quiet"
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(qf))
    inner = f'test -f "{qf}" && echo had-quiet=$(cat "{qf}")'
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "LABEL42", "",
                        "--", "bash", "-c", inner], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "had-quiet=LABEL42" in r.stdout      # present during the run (with the UTC stamp)
    assert not qf.exists()                       # removed by the EXIT trap


def test_bench_prod_accepts_quiet_label(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--quiet-label", "LW4 extra24h")
    assert r.returncode == 0, r.stderr


def test_bench_prod_accepts_warm2_env(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--warm2-env", "XLA_PYTHON_CLIENT_PREALLOCATE=false")
    assert r.returncode == 0, r.stderr


def test_run_arm_applies_arm_timeout_inside_lock(tmp_path):
    # the arm timeout kills a too-long measured command (rc 124), independent of any queue wait
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"; stub.write_text("s0_guard() { :; }\n")
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(tmp_path / "q"))
    t0 = time.time()
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "", "2",
                        "--", "sleep", "30"], capture_output=True, text=True, env=env, timeout=60)
    assert time.time() - t0 < 15 and r.returncode == 124


def test_bench_prod_accepts_single_lock_and_warm1_env(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--single-lock", "--warm1-env", "XLA_PYTHON_CLIENT_PREALLOCATE=true")
    assert r.returncode == 0, r.stderr


def test_bench_prod_accepts_arm_env(tmp_path):
    repo = _git_repo_with_src(tmp_path / "repo")
    r = _print_src(tmp_path, repo, "--arm-env", "3:XLA_PYTHON_CLIENT_MEM_FRACTION=0.42", "--single-lock")
    assert r.returncode == 0, r.stderr


def _stub_guard(tmp_path, ack_pause=True, log=None):
    import sys
    p = tmp_path / "guard.py"
    body = ""
    if log is not None:
        body = f"import pathlib; pathlib.Path({str(log)!r}).open('a').write(sys.argv[1]+'\\n')\n"
    p.write_text("import sys, json\n" + body +
                 f"print(json.dumps({{'pause_ack': {ack_pause}}} if sys.argv[1]=='pause' else {{'resume_ack': True}}))\n")
    return p


def test_run_arm_guard_pause_resume(tmp_path):
    import sys
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"; stub.write_text("s0_guard() { :; }\n")
    log = tmp_path / "guardcalls"
    guard = _stub_guard(tmp_path, True, log)
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(tmp_path / "q"),
               BENCH_COMPILE_GUARD=str(guard), BENCH_COMPILE_GUARD_PY=sys.executable)
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "LBL", "", "--",
                        "bash", "-c", f'test -f "{tmp_path / "q"}" && echo had-quiet'],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert log.read_text().split()[:2] == ["pause", "resume"]
    assert "had-quiet" in r.stdout and not (tmp_path / "q").exists()  # present during run, removed on exit


def test_run_arm_guard_pause_fails_closed(tmp_path):
    import sys
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"; stub.write_text("s0_guard() { :; }\n")
    guard = _stub_guard(tmp_path, ack_pause=False)
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(tmp_path / "q"),
               BENCH_COMPILE_GUARD=str(guard), BENCH_COMPILE_GUARD_PY=sys.executable, BENCH_GUARD_WAIT="2")
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "LBL", "", "--",
                        "bash", "-c", "true"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 7 and "not acked" in r.stderr and not (tmp_path / "q").exists()


def test_run_arm_guard_missing_continues(tmp_path):
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"; stub.write_text("s0_guard() { :; }\n")
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(tmp_path / "q"),
               BENCH_COMPILE_GUARD=str(tmp_path / "nope.py"))
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "LBL", "", "--",
                        "bash", "-c", f'test -f "{tmp_path / "q"}" && echo had-quiet'],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "compile_only_guard missing" in r.stderr and "had-quiet" in r.stdout


def test_run_arm_loads_optional_flags_env(tmp_path):
    # GPUWRF_BENCH_FLAGS_ENV (default no-op) sources a candidate flags file AFTER s0_env, before the command
    helper = Path(__file__).parents[1] / "scripts" / "bench" / "run_arm.sh"
    stub = tmp_path / "stub_env.sh"; stub.write_text("s0_guard() { :; }\nunset SOME_CANDIDATE_FLAG\n")
    flags = tmp_path / "flags.env"; flags.write_text("export SOME_CANDIDATE_FLAG=on\n")
    env = dict(os.environ, GPUWRF_LOCK_MIN_FREE_MIB="0", BENCH_QUIET_FILE=str(tmp_path / "q"),
               GPUWRF_BENCH_FLAGS_ENV=str(flags))
    r = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "", "", "--",
                        "bash", "-c", 'printf %s "$SOME_CANDIDATE_FLAG"'], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and r.stdout == "on"
    # missing flags file -> refuse (rc 2)
    env2 = dict(env, GPUWRF_BENCH_FLAGS_ENV=str(tmp_path / "nope.env"))
    r2 = subprocess.run(["bash", str(helper), str(stub), str(tmp_path), str(tmp_path), "", "", "", "--",
                         "bash", "-c", "true"], capture_output=True, text=True, env=env2, timeout=60)
    assert r2.returncode == 2 and "missing flags env" in r2.stderr
