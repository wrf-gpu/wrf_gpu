#!/usr/bin/env python3
"""Exact SEQ_SMOKE proof for night_seq_v11 + gate_template_v2 (lane v46-seqharden).

Every case runs the real sequencer/gate script as a subprocess with SEQ_SMOKE=1 / GATE_SMOKE=1, private log/cross/lease dirs
and tiny stub launchers (plus the real sealed SCREEN6 entry_A.sh as the step that must NOT start after a forced inputs FAIL).
No jax/gpuwrf import, no GPU, no real lock, no real cross, no real CPU lease.  Mutants of v11 must be killed by a named assertion.
Usage: run_smoke.py [--skip-timeout]   -> writes <run>/PROOF.json, exit 0 only if every assertion holds and every mutant is killed.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

LANE = Path("<USER_HOME>/wrf_gpu2_lanes/v46-seqharden")
STUBS = LANE / "smoke/stubs"
V11 = LANE / "night_seq_v11.sh"
GATE = LANE / "gate_template_v2.sh"
V9 = Path("<USER_HOME>/wrf_gpu2_lanes/manager/night_seq_v9.sh")
REAL_ENTRY = Path("<USER_HOME>/wrf_gpu2_lanes/v45-waveops/SCREEN6_SEALED_03/entry_A.sh")
REAL_CROSS = Path("<DATA_ROOT>/alisios/state/manager/cross")
RUN = LANE / "smoke" / ("run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
SHA_A = "a" * 64
STRIP = re.compile(r"^(SEQ_SMOKE|STEP_ONLY|PREFLIGHT_ONLY|STOP_ON_FAIL|HARD_END_UTC|ARM_ENV_MANIFEST|STEP_RESULTS_JSON|CROSS_DIR|NIGHT_SEQ_LOG_DIR|"
                   r"W1A_.*|GATE_.*|WRF_GPU2_.*|GA_.*)$")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc(off_s=0):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + off_s))


def mkplan(d, name, steps, **top):
    p = d / f"{name}.json"
    p.write_text(json.dumps(dict(steps=[dict(lane=l, path=str(path), sha256=sha(path), cap_min=cap, args=[], **extra)
                                        for (l, path, cap, *rest) in steps for extra in [rest[0] if rest else {}]], **top), indent=1))
    return p


def base_env(extra):
    env = {k: v for k, v in os.environ.items() if not STRIP.match(k)}
    env.update(JAX_PLATFORMS="cpu", LC_ALL="C")
    env.update(extra)
    return env


RAM_RETRIES = {}


def run_seq(case, plan, seq=V11, step_only=True, extra=None, timeout=1800):
    """Runs once; if the sequencer's own RAM admission (MemAvailable >= 32 GiB, v9 rule kept) skipped a step because of host load,
    the case is rerun ONCE in a fresh dir after 60 s and the retry is disclosed in PROOF.json (ram_retries)."""
    r = _run_seq(case, plan, seq, step_only, extra, timeout)
    if "RAM slots not free" in r["log"]:
        RAM_RETRIES[case] = r["dir"]
        time.sleep(60)
        r = _run_seq(case + "_ramretry", plan, seq, step_only, extra, timeout)
        r["case"] = case
    return r


def _run_seq(case, plan, seq, step_only, extra, timeout):
    d = RUN / case
    (d / "mark").mkdir(parents=True)
    (d / "cross").mkdir()
    env = base_env(dict(SEQ_SMOKE="1", NIGHT_SEQ_LOG_DIR=str(d), CROSS_DIR=str(d / "cross"), SMK_MARK=str(d / "mark"), **(extra or {})))
    if step_only:
        env["STEP_ONLY"] = "1"
    t0 = time.time()
    p = subprocess.run(["taskset", "-c", "24,25,28", "nice", "-n", "19", "bash", str(seq), str(plan), "20261026_12z", "STEP_NOTE.md"],
                       env=env, capture_output=True, text=True, timeout=timeout)
    logs = sorted(d.glob("night_seq_*.log"))
    log = logs[0].read_text() if logs else ""
    res = sorted(d.glob("night_seq_*.log.results.json"))
    smk = sorted(d.glob("SEQ_SMOKE_*.json"))
    note = d / "cross/STEP_NOTE.md"
    return dict(case=case, rc=p.returncode, wall_s=round(time.time() - t0, 1), log=log, stdout=p.stdout,
                markers=sorted(x.name for x in (d / "mark").iterdir()),
                results=json.loads(res[0].read_text()) if res else None,
                smoke=json.loads(smk[0].read_text()) if smk else None,
                note=note.read_text() if note.exists() else None, dir=str(d))


FAILS = []


def check(case, cond, what):
    if not cond:
        FAILS.append(f"{case}: {what}")
    return bool(cond)


def assert_r15_stop(r, entry_lane, entry_marker=None):
    c = r["case"]
    ok = check(c, r["rc"] == 10, f"rc {r['rc']} != 10")
    ok &= check(c, f"NOT-STARTED {entry_lane}" in r["log"], "entry not logged NOT-STARTED")
    ok &= check(c, f"RUN {entry_lane} " not in r["log"], "entry was RUN after inputs FAIL")
    if entry_marker:
        ok &= check(c, entry_marker not in r["markers"], "entry stub started (marker present)")
    ok &= check(c, r["results"] and r["results"]["verdict"] == "FAIL" and [s["status"] for s in r["results"]["steps"]] == ["FAIL", "NOT-STARTED"],
                "results statuses != [FAIL, NOT-STARTED]")
    ok &= check(c, r["note"] and "Zwischenschritt FAIL" in r["note"].splitlines()[0], "STEP note not labelled FAIL")
    ok &= check(c, r["note"] and "erster Fehler: ValueError: SCREEN6 explicit J2 result pin required" in r["note"], "STEP note lacks inputs first error")
    ok &= check(c, r["note"] and "NICHT GESTARTET" in r["note"], "STEP note lacks NOT-STARTED entry")
    ok &= check(c, r["smoke"] is not None and all(e["gpu_processes_in_step_pgid"] == 0 for e in r["smoke"]), "smoke GPU processes")
    return ok


def main():
    skip_timeout = "--skip-timeout" in sys.argv
    RUN.mkdir(parents=True)
    P = RUN / "plans"
    P.mkdir()
    S = {n: STUBS / f"{n}.sh" for n in ("inputs_fail", "inputs_ok", "entry_fail", "entry_ok", "slow")}
    out = dict(schema="v46-seqharden.seq_smoke.v1", utc_start=utc(), v11_sha256=sha(V11), gate_sha256=sha(GATE),
               v9_sha256=sha(V9), real_entry_sha256=sha(REAL_ENTRY), cases={}, mutants={})
    plan_r15 = mkplan(P, "r15_real_entry", [("v45-waveops-w1a-inputs", S["inputs_fail"], 10), ("v45-waveops-w1a", REAL_ENTRY, 26)])
    plan_b = mkplan(P, "inputs_fail_entry_ok", [("w-inputs", S["inputs_fail"], 10), ("w-entry", S["entry_ok"], 26)])
    plan_ok = mkplan(P, "all_ok", [("w-inputs", S["inputs_ok"], 10), ("w-entry", S["entry_ok"], 26)])
    plan_2f = mkplan(P, "both_fail", [("w-inputs", S["inputs_fail"], 10), ("w-entry", S["entry_fail"], 26)])
    req = {"W1A_EXTENSION_RESULT_SHA": "sha256", "W1A_SCREEN6_GPU_BUDGET_SHA": "sha256_or_NONE"}
    plan_req = mkplan(P, "required_env", [("w-inputs", S["inputs_ok"], 10), ("w-entry", S["entry_ok"], 26, {"required_env": {"W1A_SCREEN6_A_RESULT_SHA": "sha256"}})],
                      required_env=req)
    plan_slow = mkplan(P, "timeout", [("w-slow", S["slow"], 1), ("w-entry", S["entry_ok"], 26)])
    cases = out["cases"]

    def keep(r, ok, **kw):
        cases[r["case"]] = dict(ok=bool(ok), rc=r["rc"], wall_s=r["wall_s"], markers=r["markers"], dir=r["dir"],
                                results=r["results"], note=r["note"], **kw)

    # A. R15 replay with the REAL sealed entry_A.sh as step 2 (STEP_ONLY, exactly as the gate calls it)
    r = run_seq("A_r15_real_entry", plan_r15)
    keep(r, assert_r15_stop(r, "v45-waveops-w1a"))
    # B. same with an entry stub that leaves a start marker
    r = run_seq("B_inputs_fail_stop", plan_b)
    keep(r, assert_r15_stop(r, "w-entry", "entry_ok.started"))
    # C. all OK -> rc 0, note OK, both ran
    r = run_seq("C_all_ok", plan_ok)
    ok = check("C", r["rc"] == 0, f"rc {r['rc']}") & check("C", r["markers"] == ["entry_ok.started", "inputs_ok.started"], "markers") \
        & check("C", r["results"]["verdict"] == "OK", "verdict") & check("C", r["note"] and "Zwischenschritt OK" in r["note"], "note OK label")
    keep(r, ok)
    # D. independent-steps mode (no STEP_ONLY, no STOP_ON_FAIL): both run, rc 10, BOTH first errors listed (R15 'all errors')
    r = run_seq("D_both_fail_all_errors", plan_2f, step_only=False)
    errs = [e["first_error"] for e in (r["smoke"] or [])]
    ok = check("D", r["rc"] == 10, f"rc {r['rc']}") & check("D", r["markers"] == ["entry_fail.started", "inputs_fail.started"], "both must run") \
        & check("D", errs == ["ValueError: SCREEN6 explicit J2 result pin required", "ValueError: SCREEN6 prior budget ledger pin mismatch"], f"errors {errs}") \
        & check("D", "RESULT FAIL failed=w-inputs,w-entry," in r["log"], "RESULT line")
    keep(r, ok, first_errors=errs)
    # D2. same plan with STOP_ON_FAIL=1 (no STEP_ONLY) -> entry not started
    r = run_seq("D2_stop_on_fail_env", plan_2f, step_only=False, extra=dict(STOP_ON_FAIL="1"))
    ok = check("D2", r["rc"] == 10, f"rc {r['rc']}") & check("D2", r["markers"] == ["inputs_fail.started"], "entry must not run")
    keep(r, ok)
    # E. required_env pins (arming-env manifest)
    good = {"W1A_EXTENSION_RESULT_SHA": SHA_A, "W1A_SCREEN6_GPU_BUDGET_SHA": "NONE", "W1A_SCREEN6_A_RESULT_SHA": SHA_A}
    variants = {"E1_missing_all": {}, "E2_budget_empty": dict(good, W1A_SCREEN6_GPU_BUDGET_SHA=""),
                "E3_budget_lowercase_none": dict(good, W1A_SCREEN6_GPU_BUDGET_SHA="none"),
                "E4_step_pin_short": dict(good, W1A_SCREEN6_A_RESULT_SHA="abc"), "E5_all_good": good}
    expect = {"E1_missing_all": ["W1A_EXTENSION_RESULT_SHA: missing", "W1A_SCREEN6_GPU_BUDGET_SHA: missing", "W1A_SCREEN6_A_RESULT_SHA: missing"],
              "E2_budget_empty": ["W1A_SCREEN6_GPU_BUDGET_SHA: empty"], "E3_budget_lowercase_none": ["W1A_SCREEN6_GPU_BUDGET_SHA: malformed"],
              "E4_step_pin_short": ["W1A_SCREEN6_A_RESULT_SHA: malformed"], "E5_all_good": []}
    for name, envx in variants.items():
        r = run_seq(name, plan_req, extra=envx)
        if expect[name]:
            ok = check(name, r["rc"] == 2, f"rc {r['rc']} != 2") & check(name, r["markers"] == [], "a step started despite refusal") \
                & check(name, all(e in r["log"] for e in expect[name]), "expected REQENV REFUSE lines") \
                & check(name, SHA_A not in r["log"] and SHA_A not in r["stdout"], "pin value printed")
        else:
            ok = check(name, r["rc"] == 0, f"rc {r['rc']}") & check(name, r["log"].count("REQENV OK") == 3, "3 REQENV OK") \
                & check(name, SHA_A not in r["log"], "pin value printed")
        keep(r, ok)
    # F. time SKIP of step 1 -> stop, rc 10
    r = run_seq("F_time_skip_stop", plan_b, extra=dict(HARD_END_UTC=utc(90)))
    ok = check("F", r["rc"] == 10, f"rc {r['rc']}") & check("F", r["markers"] == [], "nothing may run") \
        & check("F", r["results"] and [s["status"] for s in r["results"]["steps"]] == ["SKIPPED", "NOT-STARTED"], "statuses")
    keep(r, ok)
    # G. TIMEOUT of step 1 -> group killed, entry not started
    if not skip_timeout:
        r = run_seq("G_timeout_stop", plan_slow, timeout=1800)
        ok = check("G", r["rc"] == 10, f"rc {r['rc']}") & check("G", r["markers"] == ["slow.started"], "entry must not run") \
            & check("G", r["results"] and [s["status"] for s in r["results"]["steps"]] == ["TIMEOUT", "NOT-STARTED"], "statuses") \
            & check("G", subprocess.run(["pgrep", "-f", str(S["slow"])], capture_output=True).returncode == 1, "slow stub survived")
        keep(r, ok)
    # H. BEFORE: night_seq_v9 behaviour on plan B (private copy, only CROSS/LOG/SMOKE_JSON paths redirected) -> rc 0 and entry RAN
    v9p = RUN / "night_seq_v9_private.sh"
    t = V9.read_text()
    subs = [("CROSS=<DATA_ROOT>/alisios/state/manager/cross\n", 'CROSS=${CROSS_DIR:?}\n'),
            ("LOG=<USER_HOME>/wrf_gpu2_lanes/manager/night_seq_", "LOG=${NIGHT_SEQ_LOG_DIR:?}/night_seq_"),
            ("SMOKE_JSON=<USER_HOME>/wrf_gpu2_lanes/manager/SEQ_SMOKE_", "SMOKE_JSON=${NIGHT_SEQ_LOG_DIR:?}/SEQ_SMOKE_")]
    for a, b in subs:
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    v9p.write_text(t)
    r = run_seq("H_v9_before", plan_b, seq=v9p)
    ok = check("H", r["rc"] == 0 and "entry_ok.started" in r["markers"], "v9 defect not reproduced")
    keep(r, ok, note_v9=r["note"], meaning="v9 starts the entry after inputs rc 1 and exits 0 (ALISIOS F1) — reproduced")
    # K. v11.1 critic fixes in REAL (non-smoke) mode on a private copy: F1 dispatch re-check / window-end clamp, F2 own-only cleanup, F3 modes
    out["real_mode"] = private_cases(P, S)
    # I. gate_template_v2 GATE_SMOKE
    out["gate"] = gate_cases(P, S, plan_r15, plan_ok, plan_req)
    # J. mutants of v11 (each must be killed by its named assertion, not by a setup error)
    v11t = V11.read_text()
    mutants = {
        "M1_no_stop_branch": ('  if [[ -n $STOPPED_AFTER ]]; then\n    log "NOT-STARTED', '  if false; then\n    log "NOT-STARTED'),
        "M2_exit0": ('  exit $SRC\nfi\nif [[ -n ${SEQ_SMOKE:-} ]]; then log "SEQ_SMOKE summary $SMOKE_JSON ($VERDICT)"', '  exit 0\nfi\nif [[ -n ${SEQ_SMOKE:-} ]]; then log "SEQ_SMOKE summary $SMOKE_JSON ($VERDICT)"'),
        "M3_first_error_const": ('print(" ".join(pick.split())[:240])', 'print("error")'),
        "M4_reqenv_bypass": ("(( rq != 0 )) && PF_BAD=1", "(( rq != 0 )) && true"),
        "M5_stop_only_explicit": ('[[ -n ${STEP_ONLY:-} || ${STOP_ON_FAIL:-0} == 1', '[[ ${STOP_ON_FAIL:-0} == 1'),
    }
    for name, (a, b) in mutants.items():
        assert v11t.count(a) == 1, (name, v11t.count(a))
        mp = RUN / f"mutant_{name}.sh"
        mp.write_text(v11t.replace(a, b))
        before = len(FAILS)
        if name == "M4_reqenv_bypass":
            r = run_seq(f"J_{name}", plan_req, seq=mp, extra=dict(good, W1A_SCREEN6_GPU_BUDGET_SHA=""))
            killed = r["rc"] != 2 or r["markers"] != []
            sig = f"rc {r['rc']}, markers {r['markers']}"
        else:
            r = run_seq(f"J_{name}", plan_b, seq=mp)
            assert_r15_stop(r, "w-entry", "entry_ok.started")
            new = FAILS[before:]
            del FAILS[before:]
            killed = bool(new) and r["log"] != ""  # setup errors (empty log) do not count as kills (E202)
            sig = new
        out["mutants"][name] = dict(killed=killed, signature=sig, rc=r["rc"])
        if not killed:
            FAILS.append(f"mutant {name} survived")
    out["utc_end"] = utc()
    out["fails"] = FAILS
    out["ram_retries"] = RAM_RETRIES
    out["verdict"] = "PASS" if not FAILS and all(c["ok"] for c in cases.values()) and all(v["ok"] for v in out["real_mode"].values()) and all(v["ok"] for v in out["gate"].values()) else "FAIL"
    # safety: nothing written into the real cross by this run
    out["real_cross_files_named_STEP_NOTE"] = sorted(str(p) for p in REAL_CROSS.glob("STEP_NOTE*"))
    (RUN / "PROOF.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(dict(verdict=out["verdict"], fails=FAILS, real_mode={k: v["ok"] for k, v in out["real_mode"].items()}, cases={k: (v["ok"], v["rc"]) for k, v in cases.items()},
                          gate={k: (v["ok"], v["rc"]) for k, v in out["gate"].items()}, mutants=out["mutants"]), indent=1, default=str))
    print("PROOF", RUN / "PROOF.json")
    return 0 if out["verdict"] == "PASS" else 1


def private_copy(dst, clock=None, sleep_fn=None, ram_fn=None, mutate=None, src=None):
    """Private non-smoke copy: only the window flag, pg files and real_utc path are redirected (plus optional fake clock/sleep/RAM)."""
    t = (src or V11).read_text()
    reps = [("touch /tmp/wrf_gpu2_gpu_window; trap 'rm -f /tmp/wrf_gpu2_gpu_window' EXIT", f"touch {dst.parent}/WINDOW_FLAG; trap 'rm -f {dst.parent}/WINDOW_FLAG' EXIT"),
            ("/tmp/night_seq_pg_", f"{dst.parent}/pg_"), ("REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh", "REAL=/PRIVATE_ABSENT_REAL_UTC")]
    if clock:
        reps.append(("now() { date -u +%s; }", "now() { cat %s; }" % clock))
    if ram_fn:
        a = t.index("ram_ready() {"); b = t.index("\n", a)
        t = t[:a] + ram_fn + t[b:]
    for a, b in reps:
        n = t.count(a)
        assert n >= 1, a
        t = t.replace(a, b)
    if sleep_fn:
        t = t.replace("now() { cat %s; }" % clock, "now() { cat %s; }\n%s" % (clock, sleep_fn), 1)
    for a, b in (mutate or []):
        assert t.count(a) == 1, ("mutant anchor", a[:60])
        t = t.replace(a, b)
    dst.write_text(t)
    return dst


def run_private(case, plan, seqcopy, d, extra=None, umask=0o022, timeout=1200):
    (d / "cross").mkdir(exist_ok=True)
    env = base_env(dict(NIGHT_SEQ_LOG_DIR=str(d / "log"), CROSS_DIR=str(d / "cross"), SMK_MARK=str(d / "mark"), STEP_ONLY="1",
                        WAIT_MODE="gpufree", PATH=f"{d}/bin:<USER_HOME>/miniconda3/bin:/usr/bin:/bin", FAKE_LOCKS=str(d / "locks"),
                        FAKE_GPUS=str(d / "gpus"), LOCK_WAIT_S="2", REAP_GRACE_S="2", **(extra or {})))
    t0 = time.time()
    p = subprocess.run(["taskset", "-c", "24,25,28", "nice", "-n", "19", "bash", str(seqcopy), str(plan), "PRIVATE", "NOTE.md"], env=env,
                       capture_output=True, text=True, timeout=timeout, umask=umask)
    logs = sorted((d / "log").glob("night_seq_*.log"))
    res = sorted((d / "log").glob("night_seq_*.log.results.json"))
    return dict(case=case, rc=p.returncode, wall_s=round(time.time() - t0, 1), log=logs[0].read_text() if logs else "",
                results=json.loads(res[0].read_text()) if res else None, markers=sorted(x.name for x in (d / "mark").iterdir()), dir=str(d))


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}").exists() and Path(f"/proc/{pid}/stat").read_text().split(")")[1].split()[0] != "Z"


def private_cases(P, S):
    res = {}

    def mk(case):
        d = RUN / case
        for sub in ("mark", "bin", "log"):
            (d / sub).mkdir(parents=True)
        (d / "locks").write_text(""); (d / "gpus").write_text("")
        (d / "bin/lslocks").write_text('#!/bin/bash\n# like lslocks: only live holders\nwhile read -r l; do p=$(awk \'{print $2}\' <<< "$l"); [[ -n $p ]] && kill -0 "$p" 2>/dev/null && echo "$l"; done < "${FAKE_LOCKS:?}"\n'); (d / "bin/nvidia-smi").write_text('#!/bin/bash\ncat "${FAKE_GPUS:?}"\n')
        for x in ("lslocks", "nvidia-smi"):
            (d / "bin" / x).chmod(0o755)
        return d

    def stub(d, name, body):
        f = d / f"{name}.sh"; f.write_text("#!/bin/bash\n" + body); return f

    def foreign():  # a foreign process whose cmdline looks like one of our lanes (v9 cleanup would have killed it)
        pr = subprocess.Popen(["bash", "-c", "exec -a <USER_HOME>/wrf_gpu2_lanes/v46-seqharden/FOREIGN_holder sleep 900"], start_new_session=True)
        time.sleep(0.3)
        return pr

    def k1(name, mutate=None):
        d = mk(name); clock = d / "clock"; start = int(time.time()); clock.write_text(str(start))
        plan = mkplan(d, "plan", [("p-one", S["inputs_ok"], 1), ("p-two", S["entry_ok"], 1)])
        end = start + 180
        cp = private_copy(d / "seq.sh", clock=clock, ram_fn="RAM_COUNT=0\nram_ready() { RAM_COUNT=$((RAM_COUNT+1)); (( RAM_COUNT >= 2 )); }",
                          sleep_fn='sleep() { if [[ $1 == 5 ]]; then echo %d > %s; fi; /usr/bin/sleep .02; }' % (end + 1, clock), mutate=mutate)
        return run_private(name, plan, cp, d, dict(HARD_END_UTC=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end))))
    r = k1("K1_ram_wait_crosses_end")
    ok = check("K1", r["rc"] == 10, f"rc {r['rc']}") & check("K1", r["markers"] == [], "a step was dispatched after the RAM wait crossed END") \
        & check("K1", "(re-check before dispatch)" in r["log"], "no dispatch re-check line") \
        & check("K1", r["results"] and [x["status"] for x in r["results"]["steps"]] == ["SKIPPED", "NOT-STARTED"], "statuses")
    res["K1_ram_wait_crosses_end"] = dict(ok=bool(ok), rc=r["rc"], markers=r["markers"], dir=r["dir"])

    def k2(name, mutate=None):
        d = mk(name); clock = d / "clock"; start = int(time.time()); clock.write_text(str(start))
        slow = stub(d, "slow_step", 'touch "${SMK_MARK:?}/slow.started"; sleep 600 & echo $! > %s/child.pid; wait\n' % d)
        plan = mkplan(d, "plan", [("p-slow", slow, 26), ("p-two", S["entry_ok"], 1)])
        cp = private_copy(d / "seq.sh", clock=clock, mutate=mutate,
                          sleep_fn='sleep() { local t; t=$(cat %s); case $1 in 1) echo $((t+1800)) > %s;; 2) echo $((t+400)) > %s;; esac; /usr/bin/sleep .05; }' % (clock, clock, clock))
        r = run_private(name, plan, cp, d, dict(HARD_END_UTC=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start + 2400))))
        cpid = int((d / "child.pid").read_text()) if (d / "child.pid").exists() else -1
        r["child_alive"] = cpid > 0 and alive(cpid)
        return r
    r = k2("K2_clock_jump_window_end")
    st = r["results"]["steps"] if r["results"] else []
    ok = check("K2", r["rc"] == 10, f"rc {r['rc']}") & check("K2", [x["status"] for x in st] == ["TIMEOUT", "NOT-STARTED"], "statuses") \
        & check("K2", bool(st) and st[0]["first_error"].startswith("stopped at window end"), "not stopped at the window end") \
        & check("K2", not r["child_alive"], "own child survived the timeout") & check("K2", "entry_ok.started" not in r["markers"], "second step ran")
    res["K2_clock_jump_window_end"] = dict(ok=bool(ok), rc=r["rc"], first_error=st[0]["first_error"] if st else None, dir=r["dir"])

    def k3(name, mutate=None):
        d = mk(name); fr = foreign()
        line = f"fake {fr.pid} FLOCK WRITE /PRIVATE/wrf_gpu2_gpu.lock"
        s1 = stub(d, "s1", f'touch "${{SMK_MARK:?}}/s1.started"; echo "{line}" > "${{FAKE_LOCKS:?}}"; echo {fr.pid} > "${{FAKE_GPUS:?}}"; exit 0\n')
        plan = mkplan(d, "plan", [("p-one", s1, 1), ("p-two", S["entry_ok"], 1)])
        r = run_private(name, plan, private_copy(d / "seq.sh", mutate=mutate), d)
        r["foreign_alive"] = fr.poll() is None
        fr.kill(); fr.wait()
        return r, fr.pid
    r, fp = k3("K3_foreign_holder_untouched")
    ok = check("K3", r["foreign_alive"], "foreign lock/GPU holder was signalled") & check("K3", r["rc"] == 10, f"rc {r['rc']}") \
        & check("K3", r["markers"] == ["s1.started"], "second step dispatched under a foreign lock holder") \
        & check("K3", f"GPU lock held by foreign pid {fp} after p-one" in r["log"] and "SKIP p-two: GPU lock held by foreign" in r["log"], "foreign holder not reported/skip")
    res["K3_foreign_holder_untouched"] = dict(ok=bool(ok), rc=r["rc"], foreign_alive=r["foreign_alive"], dir=r["dir"])

    def k4(name, mutate=None):
        d = mk(name)
        s1 = stub(d, "s1", 'touch "${SMK_MARK:?}/s1.started"\nsleep 600 & echo $! > %s/same_session.pid\nsetsid sleep 600 & echo $! > %s/own_session.pid\n'
                  'echo "fake $(cat %s/same_session.pid) FLOCK WRITE /PRIVATE/wrf_gpu2_gpu.lock" > "${FAKE_LOCKS:?}"\n/usr/bin/sleep 3; exit 0\n' % (d, d, d))
        plan = mkplan(d, "plan", [("p-one", s1, 1), ("p-two", S["entry_ok"], 1)])
        r = run_private(name, plan, private_copy(d / "seq.sh", mutate=mutate), d)
        pids = [int((d / f).read_text()) for f in ("same_session.pid", "own_session.pid")]
        r["left_alive"] = [p for p in pids if alive(p)]
        for p in r["left_alive"]:
            os.kill(p, 9)
        return r
    r = k4("K4_own_leftovers_reaped")
    ok = check("K4", r["left_alive"] == [], f"own leftovers alive {r['left_alive']}") & check("K4", r["rc"] == 0, f"rc {r['rc']}") \
        & check("K4", "left own processes alive" in r["log"], "no own-leftover reap line") & check("K4", r["markers"] == ["entry_ok.started", "s1.started"], "markers")
    res["K4_own_leftovers_reaped"] = dict(ok=bool(ok), rc=r["rc"], dir=r["dir"])

    # F3: caller umask 022 -> every file the sequencer writes in its log dir is 0600, cross note 0644
    d = mk("K5_file_modes")
    plan = mkplan(d, "plan", [("p-fail", S["inputs_fail"], 1), ("p-two", S["entry_ok"], 1)])
    r = run_private("K5_file_modes", plan, private_copy(d / "seq.sh"), d, umask=0o022)
    modes = {x.name: oct(x.stat().st_mode & 0o777) for x in (d / "log").iterdir() if x.is_file()}
    note = d / "cross/NOTE.md"
    ok = check("K5", modes and all(m == "0o600" for m in modes.values()), f"log modes {modes}") \
        & check("K5", note.exists() and oct(note.stat().st_mode & 0o777) == "0o644", "cross note mode")
    res["K5_file_modes"] = dict(ok=bool(ok), rc=r["rc"], modes=modes, dir=r["dir"])

    # mutants of the three fixes (each must be killed by the case's named assertion)
    muts = {
        "MF1_no_dispatch_recheck": (k1, "K1", [('  if (( rem < cap + 2 )); then log "SKIP $lane: ${rem} min left < cap $cap (re-check before dispatch)"', '  if false; then log "SKIP $lane: ${rem} min left < cap $cap (re-check before dispatch)"')],
                                    lambda r: "RUN p-one" in r["log"]),
        "MF1_no_end_clamp": (k2, "K2", [("(( deadline > END )) && deadline=$END;", "true;")],
                             lambda r: not (r["results"] and r["results"]["steps"][0]["first_error"].startswith("stopped at window end"))),
        "MF2_v9_foreign_kill": (k3, "K3", [('for hp in $(lock_holders); do log "WARN GPU lock held by foreign pid $hp after $lane (not started by this step; left untouched)"; done',
                                            'for hp in $(lock_holders); do log "WARN GPU lock held by foreign pid $hp after $lane (not started by this step; left untouched)"; kill -TERM $hp; done')],
                                lambda r: not r[0]["foreign_alive"]),
        "MF2_no_own_reap": (k4, "K4", [('-> TERM/KILL (own only)"; reap_owned "$OWN" "$lane"; fi', '-> TERM/KILL (own only)"; fi')], lambda r: r["left_alive"] != []),
        "MF3_no_umask": (None, "K5", [("umask 077  # v11.1 F3", "true  # v11.1 F3")], None),
    }
    for name, (fn, base, mutate, killed_if) in muts.items():
        if name == "MF3_no_umask":
            d = mk("KM_" + name)
            plan = mkplan(d, "plan", [("p-fail", S["inputs_fail"], 1)])
            run_private(name, plan, private_copy(d / "seq.sh", mutate=mutate), d, umask=0o022)
            killed = any(oct(x.stat().st_mode & 0o777) != "0o600" for x in (d / "log").iterdir() if x.is_file())
        else:
            r = fn("KM_" + name, mutate=mutate)
            killed = bool(killed_if(r))
        res["KM_" + name] = dict(ok=killed, mutant=True, killed=killed)
        out_m = "killed" if killed else "SURVIVED"
        if not killed:
            FAILS.append(f"real-mode mutant {name} {out_m}")
    return res


def gate_cases(P, S, plan_r15, plan_ok, plan_req):
    res = {}
    fake_lease = RUN / "fake_cpu_lease.sh"
    fake_lease.write_text("#!/bin/bash\n# fake cpu_lease for GATE_SMOKE: acquire <id> <cores> <exp> <desc> | release <id> <why>\n"
                          'f=${CPU_LEASE_DIR:?}/leases/$2-cpu.lease; mkdir -p "$(dirname "$f")"\n'
                          'case $1 in acquire) printf "id: %s\\ncores: %s\\nexpires-utc: %s\\ndesc: %s\\n" "$2" "$3" "$4" "$5" > "$f";; '
                          'release) rm -f "$f";; *) exit 2;; esac\n')
    fake_lease.chmod(0o755)

    def gate(case, plan, envx=None, title="W1a SCREEN6 smoke", gate_path=GATE):
        d = RUN / case
        for sub in ("mark", "cross", "gate", "lease"):
            (d / sub).mkdir(parents=True)
        (d / "cross/TO_WRF_GPU2_SMOKE_GO_20261031.md").write_text("GO smoke\nHARD_END_UTC=%s\n" % utc(5400))
        (d / "run.lease").write_text("expires-utc: %s\n" % utc(7200))
        conf = d / "gate.conf"
        conf.write_text("\n".join([
            f"GATE_TAG=smoke_{case}", f"TITLE='{title}'", "CROSS_FMT='TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_%s.md'", f"PLAN={plan}",
            "GO_GLOB='TO_WRF_GPU2_SMOKE_GO_20261031*.md'", f"NOT_BEFORE_UTC={utc(-60)}", f"LATEST_START_UTC={utc(2400)}",
            f"GPU_CAP_UTC={utc(5700)}", "FLOOR=0", "NEED=0", "ROOT_ADMIT=1", f"RUN_LEASE={d}/run.lease",
            f"CPU_LEASE_ID=smoke-{case}", "CPU_LEASE_CORES=24,25,28", "GA_GATE_ID=smoke", "QUIET_S=2", "QUIET_MAX_BUSY=1.01", "GATE_POLL_S=2", ""]))
        env = base_env(dict(GATE_SMOKE="1", CROSS_DIR=str(d / "cross"), GATE_DIR=str(d / "gate"), CPU_LEASE_BIN=str(fake_lease),
                            CPU_LEASE_DIR=str(d / "lease"), NIGHT_SEQ=str(V11), SMK_MARK=str(d / "mark")))
        env.update(envx or {})
        t0 = time.time()
        p = subprocess.run(["taskset", "-c", "24,25,28", "nice", "-n", "19", "bash", str(gate_path), str(conf)], env=env, capture_output=True, text=True, timeout=2400)
        files = {x.name: x.read_text() for x in (d / "cross").iterdir() if not x.name.startswith("TO_WRF_GPU2_SMOKE_GO")}
        outf = d / f"gate/gate_smoke_{case}.out"
        return dict(rc=p.returncode, wall_s=round(time.time() - t0, 1), files=files, out=outf.read_text() if outf.exists() else "",
                    markers=sorted(x.name for x in (d / "mark").iterdir()), lease_left=sorted(str(x) for x in (d / "lease").rglob("*.lease")), dir=str(d))

    g = gate("GA_r15_real_entry_STOP", plan_r15)
    stop = g["files"].get("TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_STOPPED.md", "")
    ok = check("GA", g["rc"] == 10, f"gate rc {g['rc']}") & check("GA", "START" in "".join(g["files"]), "START file") \
        & check("GA", "TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_DONE.md" not in g["files"], "DONE file written on failure") \
        & check("GA", "STOPPED — FAIL" in stop and "erster Fehler: ValueError: SCREEN6 explicit J2 result pin required" in stop, "STOP note lacks first error") \
        & check("GA", "v45-waveops-w1a: NICHT GESTARTET" in stop, "STOP note lacks NOT-STARTED entry") \
        & check("GA", "v0.4.5-RC" not in "".join(g["files"].values()) and "O72" not in "".join(g["files"].values()), "stale RW/O72 text") \
        & check("GA", g["lease_left"] == [], "CPU lease not released") \
        & check("GA", g["out"].count("gate armed") == 1, "duplicate log lines")
    res["GA_r15_real_entry_STOP"] = dict(ok=bool(ok), **g)
    g = gate("GB_all_ok_DONE", plan_ok)
    ok = check("GB", g["rc"] == 0, f"gate rc {g['rc']}") & check("GB", "TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_DONE.md" in g["files"], "DONE missing") \
        & check("GB", g["markers"] == ["entry_ok.started", "inputs_ok.started"], "markers") & check("GB", g["lease_left"] == [], "lease")
    res["GB_all_ok_DONE"] = dict(ok=bool(ok), **g)
    good = {"W1A_EXTENSION_RESULT_SHA": SHA_A, "W1A_SCREEN6_A_RESULT_SHA": SHA_A}
    g = gate("GC_arm_refuse_empty_budget", plan_req, dict(good, W1A_SCREEN6_GPU_BUDGET_SHA=""))
    ok = check("GC", g["rc"] == 2, f"gate rc {g['rc']}") & check("GC", g["files"] == {}, "cross files written at arming refusal") \
        & check("GC", "ARM REFUSE: PREFLIGHT REQENV REFUSE plan W1A_SCREEN6_GPU_BUDGET_SHA: empty" in g["out"], "refusal reason") \
        & check("GC", g["markers"] == [], "step started")
    res["GC_arm_refuse_empty_budget"] = dict(ok=bool(ok), **g)
    # pin present in the gate env but stripped by the real sequencer env (GPUWRF_ prefix) -> must refuse at arming
    plan_strip = mkplan(P, "required_env_stripped", [("w-inputs", S["inputs_ok"], 10)], required_env={"GPUWRF_SMOKE_PIN": "sha256"})
    g = gate("GD_arm_refuse_stripped_pin", plan_strip, dict(GPUWRF_SMOKE_PIN=SHA_A))
    ok = check("GD", g["rc"] == 2, f"gate rc {g['rc']}") & check("GD", "GPUWRF_SMOKE_PIN: missing" in g["out"], "stripped pin not detected") \
        & check("GD", g["files"] == {}, "cross files")
    res["GD_arm_refuse_stripped_pin"] = dict(ok=bool(ok), **g)
    # sequencer aborts in the window (e.g. wind300/queue wait, exit 4) without a results JSON -> STOP note must still name every planned step
    ab = RUN / "abort_seq.sh"
    ab.write_text(f'#!/bin/bash\n[[ -n ${{PREFLIGHT_ONLY:-}} ]] && exec bash {V11} "$@"\necho "$(date -u +%FT%TZ) ABORT: wind300 not finished before cutoff (smoke stand-in)"; exit 4\n')
    g = gate("GE_seq_abort_STOP", plan_ok, dict(NIGHT_SEQ=str(ab)))
    stop = g["files"].get("TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_STOPPED.md", "")
    ok = check("GE", g["rc"] == 10, f"gate rc {g['rc']}") & check("GE", "Sequenzer rc 4" in stop and "results-json-missing" in stop, "STOP head") \
        & check("GE", all(f"{l}: NICHT GELAUFEN (Sequenzer ohne Ergebnis: ABORT: wind300" in stop for l in ("w-inputs", "w-entry")), "STOP lacks per-step abort lines") \
        & check("GE", g["markers"] == [], "steps ran") & check("GE", g["lease_left"] == [], "lease")
    res["GE_seq_abort_STOP"] = dict(ok=bool(ok), **g)
    # gate mutants: STOP note truncated to the first step / gate exit forced 0 -> must be detected by the GA assertions
    gt = GATE.read_text()
    for name, (a, b) in {"GM1_stop_note_first_step_only": ("echo \"$stepl\" | sed 's/^/  /'", "echo \"$stepl\" | head -1 | sed 's/^/  /'"),
                         "GM2_gate_rc0": ("> \"$STOPF\"; chmod 0644 \"$STOPF\"; GRC=10", "> \"$STOPF\"; chmod 0644 \"$STOPF\"; GRC=0")}.items():
        assert gt.count(a) == 1, name
        mp = RUN / f"mutant_{name}.sh"; mp.write_text(gt.replace(a, b))
        g = gate(f"GM_{name}", plan_r15, gate_path=mp)
        stop = g["files"].get("TO_ALISIOS_MGR_FROM_WRF_GPU2_SMOKE_STOPPED.md", "")
        killed = g["out"] != "" and (g["rc"] != 10 or "v45-waveops-w1a: NICHT GESTARTET" not in stop)
        res[f"GM_{name}"] = dict(ok=killed, mutant=True, killed=killed, **g)
        if not killed:
            FAILS.append(f"gate mutant {name} survived")
    return res


if __name__ == "__main__":
    sys.exit(main())
