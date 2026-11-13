#!/usr/bin/env python3
"""Targeted test for critic cr-w44-launchers 02:19:34Z F2-RESIDUAL (night_seq_v11 v11.2).

R1 exact inline owned() Python on a fully FAKE /proc (no real proc reads): reused root pid / reused or unanchored session id must
   never seed ownership or growth; positive controls keep the legitimate root/session/descendants; zombies are not 'alive'.
R2 exact reap_owned shell function with fake owned/kill/sleep: a survivor after KILL -> rc 1 + REAP_LEFT; settled -> rc 0.
R3 real processes, private non-smoke copy (only window flag/pg files/real_utc redirected; KILL in reap_owned disabled in the
   survivor case): an own TERM-immune survivor makes the step FAIL ('teardown unresolved') and stops the chain WITHOUT STEP_ONLY
   (next step NOT-STARTED, rc 10, verdict FAIL); control with KILL enabled -> survivor reaped, rc 0.
Deletion sensitivity: R1/R2 also run against the pre-fix v11.1 source (night_seq_v11.v111.sh) and must FAIL there by assertion.
Usage: test_f2_residual.py -> <smoke>/f2res_<UTC>/PROOF.json, exit 0 only if all hold.
"""
import builtins, contextlib, io, json, os, subprocess, sys, time, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_smoke as H  # noqa: E402  (harness helpers: mkplan, private_copy, run_private, alive, sha)

LANE = H.LANE
NEW, OLD, OLD2, OLD3 = (LANE / "night_seq_v11.sh", LANE / "night_seq_v11.v111.sh", LANE / "night_seq_v11.v112.sh",
                         LANE / "night_seq_v11.v113.sh")
OUT = LANE / "smoke" / ("f2res_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))


def owned_code(src):
    s = src.read_text(); a = s.index("owned() { python3"); a = s.index("\n", a) + 1; b = s.index("\nPY\n}", a)
    return s[a:b]


def stat(pid, ppid, session, start, st="S"):
    f = [st, str(ppid), str(pid), str(session)] + ["0"] * 16; f[19] = str(start)
    return f"{pid} (FAKE) " + " ".join(f)


def run_owned(code, procs, state, mode, sid, root):
    fos = types.SimpleNamespace(listdir=lambda p: [str(x) for x in procs], getpid=lambda: 9999, getppid=lambda: 8888, getsid=lambda p: 7777)
    fsys = types.SimpleNamespace(argv=["owned", mode, str(sid), str(state), str(root)])
    ri = builtins.__import__
    def imp(n, *a, **k): return fos if n == "os" else fsys if n == "sys" else ri(n, *a, **k)
    def op(p, *a, **k):
        if str(p).startswith("/proc/"): return io.StringIO(procs[int(str(p).split("/")[2])])
        return builtins.open(p, *a, **k)
    out = io.StringIO(); rc = 0
    with contextlib.redirect_stdout(out):
        try: exec(compile(code, "exact_owned_inline", "exec"), {"__builtins__": dict(vars(builtins), __import__=imp, open=op)})
        except SystemExit as e: rc = e.code
    return out.getvalue().strip() if mode not in ("anchor", "verify") else rc


def r1(src, d):
    code = owned_code(src); res = {}
    def case(name, state_lines, procs, sid, root, want_alive, spawn_procs=None, want_anchor_rc=None):
        st = d / f"{src.stem}.{name}.state"; st.write_text("".join(f"{p} {s}\n" for p, s in state_lines))
        arc = run_owned(code, spawn_procs, st, "anchor", sid, root) if spawn_procs is not None else None
        run_owned(code, procs, st, "scan", sid, root)
        got = run_owned(code, procs, st, "alive", sid, root)
        ok = got == want_alive and (want_anchor_rc is None or arc == want_anchor_rc)
        res[name] = dict(ok=ok, alive=got, want=want_alive, anchor_rc=arc, state=st.read_text())
    # critic negatives: anchor 101/1000; now 101 is a REUSED pid (start 2000) with a foreign child 102
    case("N1_reused_session_leader", [(101, 1000)], {101: stat(101, 1, 101, 2000), 102: stat(102, 101, 101, 2001)}, 101, 101, "")
    case("N2_reused_root_other_sid", [(101, 1000)], {101: stat(101, 1, 202, 2000), 102: stat(102, 101, 202, 2001)}, 202, 101, "")
    # session gone, sid number reused by a foreign session leader (anchor 300/500, now 300/900 + member 301)
    case("N3_sid_reused_after_session_end", [(300, 500)], {300: stat(300, 1, 300, 900), 301: stat(301, 300, 300, 901)}, 300, 300, "")
    # cr-v46-terra 18:12:58Z: root ABSENT at spawn -> no anchor (rc 3); later pid 101 reused (start 2000) + foreign child 102 -> nothing
    case("N4_root_absent_at_spawn_then_reused", [], {101: stat(101, 1, 101, 2000), 102: stat(102, 101, 101, 2001)}, 101, 101, "",
         spawn_procs={}, want_anchor_rc=3)
    case("N5_no_anchor_scan_only", [], {101: stat(101, 1, 101, 2000), 102: stat(102, 101, 101, 2001)}, 101, 101, "")
    # positives
    case("P1_valid_root_session_descendants", [(101, 1000)],
         {101: stat(101, 1, 101, 1000), 102: stat(102, 101, 101, 1001), 103: stat(103, 1, 101, 1002), 104: stat(104, 103, 555, 1003)},
         101, 101, "101 102 103 104")
    case("P2_leader_exited_anchored_member_keeps_session", [(101, 1000), (105, 1005)],
         {105: stat(105, 1, 101, 1005), 106: stat(106, 1, 101, 1006)}, 101, 101, "105 106")
    case("P3_spawn_anchor_then_scan", [], {101: stat(101, 1, 101, 1000), 102: stat(102, 101, 101, 1001)}, 101, 101, "101 102",
         spawn_procs={101: stat(101, 1, 101, 1000)}, want_anchor_rc=0)
    case("P4_forked_setsid_child_leader", [(100, 990)],
         {100: stat(100, 50, 50, 990), 101: stat(101, 100, 101, 1000), 102: stat(102, 101, 101, 1001)}, 101, 100, "100 101 102")
    case("P5_zombie_not_alive", [(101, 1000), (102, 1001)], {101: stat(101, 1, 101, 1000), 102: stat(102, 101, 101, 1001, "Z")}, 101, 101, "101")
    return res


def r1v(src, d):
    """v11.4 verify: root must equal the spawn anchor in pid + starttime + session (rc 0), else rc 1."""
    code = owned_code(src); res = {}
    def case(name, spawn, later, want):
        st = d / f"{src.stem}.{name}.state"; st.write_text("")
        run_owned(code, spawn, st, "anchor", 0, 101)
        rc = run_owned(code, later, st, "verify", 0, 101)
        res[name] = dict(ok=rc == want, rc=rc, want=want)
    a = {101: stat(101, 1, 101, 1000)}
    case("V1_same_identity", a, {101: stat(101, 1, 101, 1000)}, 0)
    case("V2_reused_pid_new_start", a, {101: stat(101, 1, 101, 2000)}, 1)
    case("V3_same_start_other_session", a, {101: stat(101, 1, 555, 1000)}, 1)
    case("V4_no_anchor_then_reused", {}, {101: stat(101, 1, 101, 2000), 102: stat(102, 101, 101, 2001)}, 1)
    case("V5_root_gone", a, {}, 1)
    return res


def r3t(d):
    """v11.4 real chains, no STEP_ONLY: no anchor -> FAIL + stop + nothing signalled; timeout root KILL only when verified."""
    S = {"entry_ok": H.STUBS / "entry_ok.sh"}; out = {}
    def mk(c):
        for sub in ("mark", "bin", "log"): (c / sub).mkdir(parents=True)
        (c / "bin/lslocks").write_text("#!/bin/bash\n:\n"); (c / "bin/nvidia-smi").write_text("#!/bin/bash\n:\n")
        for x in ("lslocks", "nvidia-smi"): (c / "bin" / x).chmod(0o755)
    def go(name, step_body, cap, mutate, clock=False):
        c = d / name; mk(c); s1 = c / "s1.sh"
        s1.write_text('#!/bin/bash\ntouch "${SMK_MARK:?}/s1.started"\necho $$ > %s/root.pid\n%s' % (c, step_body % {"d": c}))
        plan = H.mkplan(c, "plan", [("p-one", s1, cap), ("p-two", S["entry_ok"], 1)])
        extra = {}; kw = {}
        if clock:
            ck = c / "clock"; t0 = int(time.time()); ck.write_text(str(t0))
            kw = dict(clock=ck, sleep_fn='sleep() { local t; t=$(cat %s); case $1 in 1) echo $((t+1800)) > %s;; 2) echo $((t+400)) > %s;; esac; /usr/bin/sleep .05; }' % (ck, ck, ck))
            extra["HARD_END_UTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + 2400))
        orig = H.base_env
        H.base_env = lambda e: {**orig(e), "STEP_ONLY": "", "REAP_SETTLE_S": "3", "SPAWN_GO_TIMEOUT_S": "5", **extra}
        try:
            r = H.run_private(name, plan, H.private_copy(c / "seq.sh", mutate=mutate, **kw), c, timeout=600)
        finally:
            H.base_env = orig
        pids = {f.stem: int(f.read_text()) for f in c.glob("*.pid")}
        r["alive_after"] = sorted(k for k, v in pids.items() if H.alive(v))
        for v in pids.values():  # our own test processes only
            if H.alive(v):
                try: os.kill(v, 9)
                except ProcessLookupError: pass
        st = r["results"]["steps"] if r["results"] else []
        r["st"] = [(x["lane"], x["status"]) for x in st]; r["ferr"] = st[0]["first_error"] if st else ""
        return r
    child = "sleep 600 & echo $! > %(d)s/child.pid\n"
    # T3: no anchor (anchor asked for a pid that does not exist) -> launcher never released (exit 97), FAIL closed, chain stops, nothing signalled
    r = go("T3_no_anchor_fail_closed", child + "/usr/bin/sleep 2; exit 0\n", 1,
           [('if owned anchor 0 "$OWN" "$pid"; then', 'if owned anchor 0 "$OWN" 999999999; then')])
    out["T3_no_anchor_fail_closed"] = dict(ok=r["rc"] == 10 and r["st"] == [("p-one", "FAIL"), ("p-two", "NOT-STARTED")]
        and r["ferr"].startswith("no ownership anchor at spawn") and "FAIL-CLOSED p-one" in r["log"] and r["alive_after"] == []
        and r["markers"] == [] and "rc=97" in r["log"], rc=r["rc"], steps=r["st"], first_error=r["ferr"], alive_after=r["alive_after"], dir=r["dir"])
    # T4 control: timeout (fake clock), root verified -> root + child killed, TIMEOUT, chain stops (STOP via timeout? no: STEP_ONLY off ->
    #    TIMEOUT alone does not stop; p-two runs) -> rc 10 because p-one TIMEOUT
    r = go("T4_timeout_verified_root_killed", child + "wait\n", 26, None, clock=True)
    out["T4_timeout_verified_root_killed"] = dict(ok=r["rc"] == 10 and r["st"][:1] == [("p-one", "TIMEOUT")] and r["alive_after"] == []
        and "NOT verified" not in r["log"], rc=r["rc"], steps=r["st"], first_error=r["ferr"], alive_after=r["alive_after"], dir=r["dir"])
    # T5: timeout, anchored identity does NOT match the root (state + anchor session falsified) -> nothing owned, root NOT signalled,
    #    step FAIL 'not verified', chain stops without STEP_ONLY
    r = go("T5_timeout_unverified_root_not_signalled", child + "wait\n", 26,
           [('fh.write(f"{root} {procs[root][2]}\\n")', 'fh.write(f"{root} 1\\n")'),
            ('{procs[root][1]}\\n")  # v11.4', '{procs[root][1]}9\\n")  # v11.4')], clock=True)
    out["T5_timeout_unverified_root_not_signalled"] = dict(ok=r["rc"] == 10 and r["st"] == [("p-one", "FAIL"), ("p-two", "NOT-STARTED")]
        and r["ferr"].startswith("root pid") and "not verified at timeout" in r["ferr"] and "identity NOT verified" in r["log"]
        and "root" in r["alive_after"] and "child" in r["alive_after"], rc=r["rc"], steps=r["st"], first_error=r["ferr"],
        alive_after=r["alive_after"], dir=r["dir"])
    return out


def r2(src):
    s = src.read_text(); a = s.index("reap_owned() {"); b = s.index("\n# GPU lock holders", a); reaper = s[a:b]
    env = {k: os.environ[k] for k in ("HOME", "PATH", "LANG") if k in os.environ}; env["LC_ALL"] = "C"
    res = {}
    def case(name, owned_fn, want_rc, want_left):
        sc = (owned_fn + '\nkill() { echo "kill $*" >> "$KLOG"; }; sleep() { :; }; log() { :; }; REAP_LEFT=""\n' + reaper +
              '\nreap_owned STATE STEP; rc=$?; printf "%s|%s" "$rc" "${REAP_LEFT:-}"')
        kl = OUT / f"{src.stem}.{name}.kills"; env["KLOG"] = str(kl)
        out = subprocess.run(["/bin/bash", "-c", sc], env=env, capture_output=True, text=True, timeout=30).stdout
        rc, left = out.split("|")
        res[name] = dict(ok=(rc == want_rc and left == want_left), rc=rc, left=left, kills=kl.read_text().splitlines() if kl.exists() else [])
    case("RA_survivor_after_KILL", "owned() { echo 999999999; }", "1", "999999999")
    case("RB_exits_after_TERM", 'owned() { [[ -s $KLOG ]] && return 0; echo 4242; }', "0", "")
    case("RC_exits_after_KILL", 'owned() { grep -q KILL "$KLOG" 2>/dev/null && return 0; echo 4242; }', "0", "")
    return res


def r3(d):
    S = {n: H.STUBS / f"{n}.sh" for n in ("entry_ok",)}
    out = {}
    for name, mutate, want in (("T1_own_survivor_KILL_disabled", [('kill -KILL $l 2>/dev/null\n  for', ': KILL-DISABLED-FOR-TEST\n  for')], "stop"),
                               ("T2_control_KILL_enabled", None, "ok")):
        c = d / name
        for sub in ("mark", "bin", "log"): (c / sub).mkdir(parents=True)
        (c / "locks").write_text(""); (c / "gpus").write_text("")
        (c / "bin/lslocks").write_text("#!/bin/bash\n:\n"); (c / "bin/nvidia-smi").write_text("#!/bin/bash\n:\n")
        for x in ("lslocks", "nvidia-smi"): (c / "bin" / x).chmod(0o755)
        s1 = c / "s1.sh"
        s1.write_text('#!/bin/bash\ntouch "${SMK_MARK:?}/s1.started"\n'
                      f"bash -c \"trap '' TERM; exec -a F2RES_SURVIVOR sleep 600\" & echo $! > {c}/survivor.pid\n/usr/bin/sleep 2; exit 0\n")
        plan = H.mkplan(c, "plan", [("p-one", s1, 1), ("p-two", S["entry_ok"], 1)])
        orig = H.base_env  # run_private hard-sets STEP_ONLY=1: clear it so the stop must come from the teardown FAIL itself
        H.base_env = lambda e: {**orig(e), "STEP_ONLY": "", "REAP_SETTLE_S": "3"}
        try:
            r = H.run_private(name, plan, H.private_copy(c / "seq.sh", mutate=mutate), c, timeout=600)
        finally:
            H.base_env = orig
        sp = int((c / "survivor.pid").read_text()); left = H.alive(sp)
        if left: os.kill(sp, 9)  # our own test process
        st = [(x["lane"], x["status"]) for x in (r["results"] or {}).get("steps", [])]
        p1 = (r["results"] or {}).get("steps", [{}])[0]
        if want == "stop":
            ok = (r["rc"] == 10 and st == [("p-one", "FAIL"), ("p-two", "NOT-STARTED")] and "teardown unresolved" in p1.get("first_error", "")
                  and "TEARDOWN UNRESOLVED p-one" in r["log"] and r["results"]["verdict"] == "FAIL" and left
                  and r["markers"] == ["s1.started"])
        else:
            ok = r["rc"] == 0 and st == [("p-one", "OK"), ("p-two", "OK")] and not left and "TEARDOWN UNRESOLVED" not in r["log"]
        out[name] = dict(ok=bool(ok), rc=r["rc"], steps=st, first_error=p1.get("first_error"), survivor_was_alive_after=left,
                         markers=r["markers"], wall_s=r["wall_s"], dir=str(c))
    return out


def main():
    if os.path.exists("/tmp/wrf_gpu2_quiet"):
        print("QUIET"); sys.exit(3)
    OUT.mkdir(parents=True)
    new = dict(R1=r1(NEW, OUT), R2=r2(NEW))
    old = dict(R1=r1(OLD, OUT), R2=r2(OLD))
    old2 = dict(R1=r1(OLD2, OUT))
    new["R1V"] = r1v(NEW, OUT); old3 = r1v(OLD3, OUT)
    new["R3T"] = r3t(OUT)
    new["R3"] = r3(OUT)
    neg = [k for k in old["R1"] if k.startswith("N") and k not in ("N4_root_absent_at_spawn_then_reused", "N5_no_anchor_scan_only")] + ["RA_survivor_after_KILL"]
    old_fail = {k: not (old["R1"].get(k) or old["R2"].get(k))["ok"] for k in neg}
    # v11.2 (f6c3) has no anchor mode and anchors on first sight in scan -> must fail the new critic case by assertion
    old_fail.update({f"v113.{k}": not old3[k]["ok"] for k in ("V2_reused_pid_new_start", "V3_same_start_other_session", "V4_no_anchor_then_reused")})
    old_fail.update({f"v112.{k}": not old2["R1"][k]["ok"] for k in ("N4_root_absent_at_spawn_then_reused", "N5_no_anchor_scan_only")})
    all_new = all(v["ok"] for g in new.values() for v in g.values())
    verdict = "PASS" if all_new and all(old_fail.values()) else "FAIL"
    proof = dict(schema="v46-seqharden.f2_residual.v1", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), verdict=verdict,
                 v11_sha256=H.sha(NEW), pre_fix_v111_sha256=H.sha(OLD), pre_fix_v112_sha256=H.sha(OLD2), pre_fix_v113_sha256=H.sha(OLD3), new=new, old_negatives_fail_by_assertion=old_fail,
                 scope="R1/R2 fully fake proc/kill/sleep; R3 own private processes only (lslocks/nvidia-smi stubbed empty), no GPU/real lock/cross/lease/foreign signals")
    (OUT / "PROOF.json").write_text(json.dumps(proof, indent=1))
    print(json.dumps(dict(verdict=verdict, proof=str(OUT / "PROOF.json"),
                          fails=[f"{g}.{k}" for g, v in new.items() for k, x in v.items() if not x["ok"]] + [k for k, v in old_fail.items() if not v])))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
