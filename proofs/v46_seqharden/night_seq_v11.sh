#!/bin/bash
# Manager night-window sequencer v11 (lane v46-seqharden, 2026-10-31; ALISIOS Sol F1 / FINDINGS E262). Derived from night_seq_v9 (manager/night_seq_v9.sh,
# all v3..v9 behaviour kept: PREFLIGHT, waits, window, per-step setsid group + timeout, lock/orphan cleanup, SEQ_SMOKE exact call).
# v11 deltas:
#  (1) STOP-ON-FAIL: with STEP_ONLY=1 (always) or STOP_ON_FAIL=1 / plan "stop_on_fail": true, the first step that ends rc!=0, times out, is
#      SKIPPED (time/ram/sha) or violates SEQ_SMOKE stops the sequence; every later planned step is logged 'NOT-STARTED <lane>' and never runs.
#  (2) OVERALL RC: exit 10 whenever any planned step did not end rc=0 (FAIL/TIMEOUT/SKIPPED/NOT-STARTED), also in STEP_ONLY and SEQ_SMOKE;
#      the STEP note / DONE file / smoke JSON carry the label FAIL and list EVERY failed step with its first error line (OK otherwise).
#  (3) REQUIRED ENV: plan top-level and per-step "required_env": {"NAME": "<type>"} (types: sha256, sha256_or_NONE, NONE, nonempty, utc, int,
#      or "re:<python regex>", full match) is checked in PREFLIGHT; a missing/empty/malformed pin refuses the plan (rc 2) before any wait or step.
#      Values are never printed. ARM_ENV_MANIFEST=<json> adds the same schema from a separate file (for already sealed plans).
#  (4) Results: <log>.results.json (and $STEP_RESULTS_JSON if set) = every planned step with status OK|FAIL|TIMEOUT|SKIPPED|NOT-STARTED, rc, first_error.
#  (5) SEQ_SMOKE never writes into the real ALISIOS cross (CROSS_DIR must be private; default <logdir>/seq_smoke_cross); v9 wrote the STEP note
#      of a STEP_ONLY smoke into the real cross. Log dir configurable: NIGHT_SEQ_LOG_DIR (default manager dir).
#  (6) setsid -w: the step's exit status is the child's even if setsid has to fork (E131).
# v11.1 (critic cr-w44-launchers FIX 2026-10-31T01:28Z): F1 time/lock re-check immediately before dispatch + deadline = min(END, cap+1 min);
# v11.4 (critic cr-v46-terra 20:11:24Z): timeout root KILL only after (pid,starttime,session) == spawn anchor; no anchor or unverified
#   root -> step FAIL + chain stops, nothing signalled (fail closed).
# v11.3 (critic cr-v46-terra 18:12:58Z): ownership anchor only at spawn (owned anchor); no anchor -> nothing owned, only reported.
# v11.2 (critic 02:19:34Z F2-residual): owned() pins root/session identity (first record = anchor) BEFORE discovery/growth; reaper
#   settles after KILL and an unresolved own survivor makes the step FAIL + stops the chain.
#   F2 cleanup only of processes the step started (own session + descendants recorded pid+starttime) — foreign lock/GPU holders are never
#   signalled, only reported (v9 TERM/KILLed every lock holder and our-lane GPU pids); bounded lslocks/nvidia-smi queries; F3 umask 077.
# Usage: night_seq_v11.sh <plan.json> <nightly_issue e.g. 20261023_12z> <done_cross_name>
# Exit: 0 all steps rc 0 | 2 plan/preflight refused | 3/4/5 window aborts (as v9) | 10 at least one planned step not OK.
set -u
export LC_ALL=C
umask 077  # v11.1 F3 (E245): logs/results/step logs/smoke JSON 0600; cross notes chmod 0644 explicitly
# v9 security (E245): drop every secret-looking variable before anything is started
for _v in $(compgen -e | grep -E 'TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$' | grep -vE '^(DBUS_|XDG_)'); do unset "$_v"; done
unset _v
PLAN=$1 ISSUE=$2 DONE_NAME=$3
REAL_CROSS=<DATA_ROOT>/alisios/state/manager/cross
LOGDIR=${NIGHT_SEQ_LOG_DIR:-<USER_HOME>/wrf_gpu2_lanes/manager}
mkdir -p "$LOGDIR"
LOG=$LOGDIR/night_seq_$(date -u +%Y%m%dT%H%M%SZ).log
if [[ -n ${SEQ_SMOKE:-} ]]; then
  CROSS=${CROSS_DIR:-$LOGDIR/seq_smoke_cross}; mkdir -p "$CROSS"
  if [[ $(realpath -m "$CROSS") == $(realpath -m "$REAL_CROSS") ]]; then echo "SEQ_SMOKE refuses the real ALISIOS cross dir as CROSS_DIR" | tee -a "$LOG"; exit 2; fi
else
  CROSS=${CROSS_DIR:-$REAL_CROSS}
fi
QCTL=<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009/rd-e2-massdays/gpu_queue_ctl
ST=<DATA_ROOT>/alisios/state/nightly_9nest/$ISSUE/STATUS.json
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
now() { date -u +%s; }
iso() { if [[ -x $REAL ]]; then $REAL --iso; else date -u +%FT%TZ; fi; }
log() { echo "$(iso) $*" | tee -a "$LOG"; }
STEPS=$(python3 -c 'import json,sys,shlex;d=json.load(open(sys.argv[1]));[print(s["lane"],s["path"],s["sha256"],s["cap_min"],shlex.join([str(a) for a in s.get("args",[])])) for s in d["steps"]]' "$PLAN") || { log "bad plan"; exit 2; }
PLAN_STOP=$(python3 -c 'import json,sys;print(1 if json.load(open(sys.argv[1])).get("stop_on_fail") is True else 0)' "$PLAN") || { log "bad plan"; exit 2; }
STOP=0; [[ -n ${STEP_ONLY:-} || ${STOP_ON_FAIL:-0} == 1 || $PLAN_STOP == 1 ]] && STOP=1
# v11 (3): required arming env (plan + optional manifest); prints REQENV lines, rc 1 on any refusal; never prints values
reqenv_check() { python3 - "$PLAN" "${ARM_ENV_MANIFEST:-}" <<'PY'
import json, os, re, sys
TYPES = {"sha256": r"[0-9a-f]{64}", "sha256_or_NONE": r"(?:[0-9a-f]{64}|NONE)", "NONE": r"NONE", "nonempty": r"\S(?:.*\S)?",
         "utc": r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "int": r"[0-9]+"}
SECRET = re.compile(r"TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$")
plan = json.load(open(sys.argv[1]))
specs = []  # (scope, name, type)
def take(scope, d):
    if d is None: return
    if not isinstance(d, dict): specs.append((scope, "<required_env>", None)); return
    for k, v in d.items(): specs.append((scope, k, v))
take("plan", plan.get("required_env"))
for s in plan.get("steps", []): take("step " + str(s.get("lane")), s.get("required_env"))
if sys.argv[2]:
    try: m = json.load(open(sys.argv[2]))
    except (OSError, ValueError) as e: print(f"REQENV REFUSE manifest {sys.argv[2]}: unreadable ({type(e).__name__})"); sys.exit(1)
    take("manifest", m.get("required_env") if isinstance(m, dict) else m)
bad = 0
for scope, name, typ in specs:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
        print(f"REQENV REFUSE {scope}: invalid env name/spec {name!r}"); bad = 1; continue
    if SECRET.search(name):
        print(f"REQENV REFUSE {scope} {name}: secret-like names are stripped by the sequencer and cannot be pins"); bad = 1; continue
    if not isinstance(typ, str) or not (typ in TYPES or typ.startswith("re:")):
        print(f"REQENV REFUSE {scope} {name}: unknown type {typ!r}"); bad = 1; continue
    pat = TYPES.get(typ) or typ[3:]
    try: rx = re.compile(pat)
    except re.error: print(f"REQENV REFUSE {scope} {name}: bad regex"); bad = 1; continue
    if name not in os.environ: print(f"REQENV REFUSE {scope} {name}: missing (expected {typ})"); bad = 1; continue
    v = os.environ[name]
    if v == "": print(f"REQENV REFUSE {scope} {name}: empty (expected {typ})"); bad = 1; continue
    if not rx.fullmatch(v): print(f"REQENV REFUSE {scope} {name}: malformed (expected {typ}, len {len(v)})"); bad = 1; continue
    print(f"REQENV OK {scope} {name} ({typ})")
sys.exit(bad)
PY
}
# v11 (2): first error line of a step log (first Python exception line, else first error-word line, else last non-empty line); redacted, 240 chars
first_error() { python3 - "$1" <<'PY'
import re, sys
try: lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError: print("(no step log)"); sys.exit()
exc = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Refusal|Failure)\b(:|$)")
word = re.compile(r"\b(REFUSE[DS]?|FATAL|FAIL(ED|URE)?|ERROR|[Ee]rror:|No such file|command not found|Permission denied|unbound variable|Killed)")
pick = next((l for l in lines if exc.match(l)), None) or next((l for l in lines if word.search(l)), None) \
       or next((l for l in reversed(lines) if l.strip()), None) or "(empty step log)"
pick = re.sub(r"([A-Za-z_]*(?:TOKEN|PASSW|SECRET|API_?KEY|CREDENTIAL)[A-Za-z_]*)=\S+", r"\1=<redacted>", pick, flags=re.I)
print(" ".join(pick.split())[:240])
PY
}
# v7 PREFLIGHT (before waiting for any window): every step must be runnable exactly as planned
PF_BAD=0
while read -r lane path sha cap args; do
  [[ -z $lane ]] && continue
  got=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1)
  [[ $got == "$sha" ]] || { log "PREFLIGHT REFUSE $lane: sha ${got:-missing} != plan $sha"; PF_BAD=1; continue; }
  bash -n "$path" 2>/dev/null || { log "PREFLIGHT REFUSE $lane: bash -n failed"; PF_BAD=1; }
  [[ $cap =~ ^[0-9]+$ ]] && (( cap >= 1 )) || { log "PREFLIGHT REFUSE $lane: cap_min $cap < 1 (must be >= the launcher's own cap)"; PF_BAD=1; }
  if grep -v '^[[:space:]]*#' "$path" | grep -q 'with_ram_slot'; then log "PREFLIGHT REFUSE $lane: launcher uses with_ram_slot (GPU steps need their own systemd scope)"; PF_BAD=1; fi
  if [[ -z $args ]] && grep -v '^[[:space:]]*#' "$path" | grep -qE '\$\{1:\?|^[A-Za-z_][A-Za-z0-9_]*="?\$\{?1([}:"]|$)'; then log "PREFLIGHT REFUSE $lane: launcher reads \$1 but the plan gives no args"; PF_BAD=1; fi
done <<< "$STEPS"
RQ=$(reqenv_check); rq=$?
[[ -n $RQ ]] && while IFS= read -r l; do log "PREFLIGHT $l"; done <<< "$RQ"
(( rq != 0 )) && PF_BAD=1
if (( PF_BAD )); then log "PREFLIGHT FAILED -> fix the plan/arming env; no window requested"; exit 2; fi
log "PREFLIGHT OK (stop_on_fail=$STOP)"
[[ -n ${PREFLIGHT_ONLY:-} ]] && exit 0
log "plan $(sha256sum "$PLAN" | cut -c1-16) steps: $(echo "$STEPS" | awk '{print $1"("$4")"}' | tr '\n' ' ')"
CUTOFF=$(date -u -d "$(date -u +%F) 03:00:00" +%s); (( CUTOFF < $(now) )) && CUTOFF=$(date -u -d "tomorrow 03:00:00" +%s)
if [[ -n ${HARD_END_UTC:-} ]]; then CUTOFF=$(date -u -d "$HARD_END_UTC" +%s); fi
# 1. nightly not running (v9 unchanged)
if [[ -z ${SEQ_SMOKE:-} && ${WAIT_MODE:-} != wind300 && ${WAIT_MODE:-} != gpufree ]]; then
while :; do
  v=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("verdict",""))' "$ST" 2>/dev/null || echo MISSING)
  svc=$(systemctl --user is-active alisios-nightly-9nest.service 2>/dev/null; systemctl is-active alisios-nightly-9nest.service 2>/dev/null)
  case "$v" in DONE|FAILED|DEGRADED|SKIPPED) log "nightly $ISSUE terminal: $v"; break;; esac
  if [[ $svc != *"active"* || $svc == *inactive*inactive* ]] && [[ $v == MISSING || $v == DEFERRED || $v == WAITING_SOURCE ]] && [[ -n ${ALLOW_NONTERMINAL:-} ]]; then log "nightly $ISSUE verdict $v, service inactive, ALLOW_NONTERMINAL set"; break; fi
  (( $(now) > CUTOFF - 1800 )) && { log "ABORT: nightly not terminal ($v) before cutoff-30min"; exit 3; }
  sleep 60
done
fi
# 2a. optional GO marker / not-before (v9 unchanged)
if [[ -n ${NOT_BEFORE_UTC:-} ]]; then nb=$(date -u -d "$NOT_BEFORE_UTC" +%s); while (( $(now) < nb )); do sleep 60; done; log "not-before $NOT_BEFORE_UTC reached"; fi
if [[ -n ${REQUIRE_GO_GLOB:-} ]]; then while ! compgen -G "$CROSS/$REQUIRE_GO_GLOB" >/dev/null; do (( $(now) > CUTOFF )) && { log "ABORT: no GO marker $REQUIRE_GO_GLOB"; exit 5; }; sleep 60; done; GOF=$(ls $CROSS/$REQUIRE_GO_GLOB | head -1); log "GO marker present: $GOF"
  heg=$(grep -oE 'HARD_END_UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z' "$GOF" | head -1 | cut -d= -f2)
  if [[ -n $heg ]]; then
    if [[ -z ${HARD_END_UTC:-} ]] || (( $(date -u -d "$heg" +%s) < $(date -u -d "$HARD_END_UTC" +%s) )); then HARD_END_UTC=$heg; fi
    log "hard end from GO file: $heg -> effective $HARD_END_UTC"
  else log "GO file has no HARD_END_UTC= line; effective hard end ${HARD_END_UTC:-none} + WINDOW_S"; fi
fi
# 2. ALISIOS queue paused / wind300 finished / gpufree (v9 unchanged)
t0=$(now)
if [[ -n ${SEQ_SMOKE:-} ]]; then
  log "SEQ_SMOKE: no waits"
elif [[ ${WAIT_MODE:-} == wind300 || ${WAIT_MODE:-} == gpufree ]]; then
  W3=<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009/rd-e2-massdays/wind300_gpu_v043_t30/STATUS.json
  while :; do
    ws=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(int(d.get("completed",0))>=int(d.get("maximum",300)), d.get("status"), d.get("completed"))' "$W3" 2>/dev/null || echo "False ? ?")
    holder=$(timeout 10 lslocks 2>/dev/null | grep -c wrf_gpu2_gpu.lock)
    if [[ ( $ws == True* || ${WAIT_MODE:-} == gpufree ) && $ws != *GPU_CASE* && $holder == 0 ]]; then log "window opens (WAIT_MODE=${WAIT_MODE:-}; wind300 $ws; lock holders 0)"; break; fi
    (( $(now) > CUTOFF )) && { log "ABORT: wind300 not finished before cutoff ($ws, holders $holder)"; exit 4; }
    sleep 30
  done
else
while :; do
  q=$(timeout 30 "$QCTL" status 2>/dev/null)
  cs=$(python3 -c 'import json,sys;d=json.loads(sys.stdin.read());c=d["control"];t=d["status"];st=t["status"]
st="STALE_"+st if t.get("utc","")<c.get("utc","") and st=="GPU_CASE" else st
print(c["state"],st)' <<<"$q" 2>/dev/null || echo "? ?")
  holder=$(timeout 10 lslocks 2>/dev/null | grep -c wrf_gpu2_gpu.lock)
  if [[ ${cs%% *} == PAUSED && ${cs##* } != GPU_CASE && $holder == 0 ]]; then log "queue $cs, no lock holder -> window opens"; break; fi
  (( $(now) - t0 > ${QUEUE_WAIT_S:-2700} )) && { log "ABORT: queue not paused within 45 min ($cs, holders $holder)"; exit 4; }
  sleep 30
done
fi
START=$(now); END=$(( START + ${WINDOW_S:-7200} )); (( END > CUTOFF )) && END=$CUTOFF
if [[ -n ${HARD_END_UTC:-} ]]; then he=$(date -u -d "$HARD_END_UTC" +%s); (( END > he )) && END=$he; fi
export GA_START_UTC=$(date -u -d @$START +%FT%TZ) GA_END_UTC=$(date -u -d @$END +%FT%TZ)
log "window $GA_START_UTC -> $GA_END_UTC"
if [[ -z ${SEQ_SMOKE:-} ]]; then touch /tmp/wrf_gpu2_gpu_window; trap 'rm -f /tmp/wrf_gpu2_gpu_window' EXIT; else export WRF_GPU2_SEQ_SMOKE=1; SMOKE_JSON=$LOGDIR/SEQ_SMOKE_$(basename "$PLAN" .json)_$(date -u +%Y%m%dT%H%M%SZ).json; echo '[]' > "$SMOKE_JSON"; fi
ram_ready() { local avail; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); (( avail >= 32 )); }
# v11.1 F2: process ownership = the step's own session (setsid) + every descendant seen while it ran, recorded as pid+starttime (no PID reuse)
owned() { python3 - "$@" <<'PY'
import os, sys
mode, sid, state = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root = int(sys.argv[4]) if len(sys.argv) > 4 else -1
skip = {os.getpid(), os.getppid()}
procs = {}
for d in os.listdir("/proc"):
    if not d.isdigit(): continue
    try: s = open(f"/proc/{d}/stat").read()
    except OSError: continue
    f = s[s.rindex(")") + 2:].split()
    procs[int(d)] = (int(f[1]), int(f[3]), f[19], f[0])  # ppid, session, starttime, state
# v11.2 F2-residual: the state file is the identity record; the FIRST line of a pid is its anchor (root/session leader recorded at spawn)
pairs, anchor = set(), {}
try:
    for l in open(state):
        p, st = l.split(); pairs.add((int(p), st)); anchor.setdefault(int(p), st)
except (OSError, ValueError): pass
def ident(p): return p in procs and (p, procs[p][2]) in pairs
alive_known = {p for p in procs if ident(p)}
if mode == "anchor":  # v11.3: ONLY at spawn; no live root -> no anchor at all (rc 3, caller reports; nothing is ever owned)
    if root not in procs: raise SystemExit(3)
    if (root, procs[root][2]) not in pairs:
        with open(state, "a") as fh: fh.write(f"{root} {procs[root][2]}\n")
    with open(state + ".anchor", "w") as fh: fh.write(f"{root} {procs[root][2]} {procs[root][1]}\n")  # v11.4: pid starttime session
elif mode == "verify":  # v11.4: root still IS the spawned process: pid + starttime + session equal the spawn anchor, else rc 1
    try: ap, ast, ass = open(state + ".anchor").read().split()
    except (OSError, ValueError): raise SystemExit(1)
    if not (int(ap) == root and root in procs and procs[root][2] == ast and str(procs[root][1]) == ass): raise SystemExit(1)
elif mode == "scan":
    # root: only with the identity anchored at spawn; scan never anchors, so a missing/reused/stale root never seeds growth
    root_ok = root in anchor and root in procs and anchor[root] == procs[root][2]
    # session: never our own/init; valid only while the anchored leader is alive OR an anchored member is alive in it
    # (an existing session member keeps the sid number allocated -> no reuse); a reused/unanchored sid is never matched
    sid_ok = sid > 1 and sid != os.getsid(0) and (
        (sid in anchor and ident(sid) and procs[sid][1] == sid) or
        any(procs[p][1] == sid for p in alive_known))
    own = set(alive_known) | ({root} if root_ok else set())
    if sid_ok: own |= {p for p, v in procs.items() if v[1] == sid}
    grow = True
    while grow:  # descendants of validated members only; a child never starts before its parent
        grow = False
        for p, (pp, ss, st, z) in procs.items():
            if p not in own and pp in own and int(st) >= int(procs[pp][2]): own.add(p); grow = True
    with open(state, "a") as fh:
        for p in sorted(own - skip):
            if (p, procs[p][2]) not in pairs: fh.write(f"{p} {procs[p][2]}\n")
else:  # alive = recorded identities still running (zombies are exited: not signallable, hold nothing)
    print(" ".join(str(p) for p in sorted(alive_known - skip) if procs[p][3] not in ("Z", "X")))
PY
}
# v11.2 F2-residual: TERM, grace, KILL, then a bounded settlement check; own non-zombie survivors -> REAP_LEFT set + rc 1 (never assume kill = exit)
reap_owned() { local l i; REAP_LEFT=""; l=$(owned alive 0 "$1"); [[ -z $l ]] && return 0; kill -TERM $l 2>/dev/null; sleep ${REAP_GRACE_S:-10}
  l=$(owned alive 0 "$1"); [[ -z $l ]] && return 0; log "KILL own leftovers of $2: $l"; kill -KILL $l 2>/dev/null
  for (( i = 0; i < ${REAP_SETTLE_S:-10}; i++ )); do l=$(owned alive 0 "$1"); [[ -z $l ]] && return 0; sleep 1; done
  REAP_LEFT=$l; log "TEARDOWN UNRESOLVED $2: own pid(s) $l still alive ${REAP_SETTLE_S:-10} s after KILL"; return 1; }
# GPU lock holders NOT owned by the current step (bounded query); owned holders are reaped by reap_owned, foreign ones only reported
lock_holders() { local hp; for hp in $(timeout 10 lslocks 2>/dev/null | awk '/wrf_gpu2_gpu\.lock/{print $2}' | sort -u); do
  [[ -n ${OWN:-} && -f ${OWN:-} ]] && grep -q "^$hp " "$OWN" && [[ -n $(owned alive 0 "$OWN" | tr ' ' '\n' | grep -x "$hp") ]] && continue; echo "$hp"; done; }
RTSV=$LOG.steps.tsv; : > "$RTSV"
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$RTSV"; }  # lane status rc first_error steplog
smoke_record() { [[ -n ${SEQ_SMOKE:-} ]] || return 0
  python3 - "$SMOKE_JSON" "$@" <<'PY'
import json,sys
f,lane,path,sha,cap,args,rc,gpu,status,ferr=sys.argv[1:]
d=json.load(open(f)); d.append(dict(lane=lane,path=path,sha256=sha,cap_min=int(cap),args=args,rc=(int(rc) if rc not in ("","-") else None),
  status=status,first_error=ferr,gpu_processes_in_step_pgid=int(gpu or 0),exact_call='setsid -w bash <path> <args> via night_seq_v11 SEQ_SMOKE=1'))
json.dump(d,open(f,'w'),indent=1)
PY
}
STOPPED_AFTER=""; NFAIL=0; REAP_LEFT=""; FORCE_STOP=0
fail_step() { # lane status rc first_error
  NFAIL=$(( NFAIL + 1 )); log "FAIL $1 status=$2 rc=$3 first_error: $4"
  if (( STOP || ${FORCE_STOP:-0} )) && [[ -z $STOPPED_AFTER ]]; then STOPPED_AFTER=$1; log "STOP-ON-FAIL: no further planned step starts after $1"; fi; }
while read -r lane path sha cap args; do
  [[ -z $lane ]] && continue
  if [[ -n $STOPPED_AFTER ]]; then
    log "NOT-STARTED $lane (stop-on-fail after $STOPPED_AFTER)"; record "$lane" NOT-STARTED - "not started: stop-on-fail after $STOPPED_AFTER" -
    smoke_record "$lane" "$path" "$sha" "$cap" "$args" - 0 NOT-STARTED "stop-on-fail after $STOPPED_AFTER"; continue
  fi
  rem=$(( (END - $(now)) / 60 ))
  if (( rem < cap + 2 )); then log "SKIP $lane: ${rem} min left < cap $cap"; record "$lane" SKIPPED - "skipped: ${rem} min left < cap $cap + 2" -; fail_step "$lane" SKIPPED - "time"; continue; fi
  w=0; until ram_ready; do (( w++ > 60 )) && break; sleep 5; done
  if ! ram_ready; then log "SKIP $lane: RAM slots not free after 5 min"; record "$lane" SKIPPED - "skipped: MemAvailable < 32 GiB after 5 min" -; fail_step "$lane" SKIPPED - "ram"; continue; fi
  got=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1)
  if [[ $got != "$sha" ]]; then log "SKIP $lane: sha mismatch $got"; record "$lane" SKIPPED - "skipped: launcher sha ${got:-missing} != plan" -; fail_step "$lane" SKIPPED - "sha"; continue; fi
  # v11.1 F1: the RAM/sha waits above can take minutes -> re-check the window and the GPU lock IMMEDIATELY before dispatch
  if [[ -z ${SEQ_SMOKE:-} ]]; then  # smoke never depends on the real lock (a real session may hold it)
    lw=0; while [[ -n $(lock_holders) ]] && (( lw < ${LOCK_WAIT_S:-60} )); do sleep 5; lw=$(( lw + 5 )); done
    if [[ -n $(lock_holders) ]]; then log "SKIP $lane: GPU lock held by foreign pid(s) $(lock_holders | tr '\n' ' ')(left untouched)"; record "$lane" SKIPPED - "skipped: GPU lock held by a foreign process before dispatch" -; fail_step "$lane" SKIPPED - "lock"; continue; fi
  fi
  rem=$(( (END - $(now)) / 60 ))
  if (( rem < cap + 2 )); then log "SKIP $lane: ${rem} min left < cap $cap (re-check before dispatch)"; record "$lane" SKIPPED - "skipped: ${rem} min left < cap $cap + 2 at dispatch" -; fail_step "$lane" SKIPPED - "time"; continue; fi
  SLOG=$LOG.$lane
  OWN=$LOG.$lane.owned; : > "$OWN"
  log "RUN $lane $path ${args:+args [$args] }cap ${cap}min"
  rm -f /tmp/night_seq_pg_$lane; GO=$LOG.$lane.go; rm -f "$GO"; mkfifo -m 600 "$GO"
  # v11.4: the leader writes its pg file, then BLOCKS on a private fifo until the sequencer has anchored it (root alive + setsid() done at
  # anchor time, also for steps that would exit at once); without an anchor it is never released and exits 97 (launcher never runs)
  setsid -w bash -c "echo \$\$ > /tmp/night_seq_pg_$lane; exec 3<> '$GO'; read -r -t ${SPAWN_GO_TIMEOUT_S:-60} -u 3 _ || exit 97; exec 3<&-; exec bash '$path' $args" >> "$SLOG" 2>&1 &
  pid=$!; noanchor=0; for (( aw = 0; aw < 50; aw++ )); do [[ -s /tmp/night_seq_pg_$lane ]] && break; kill -0 $pid 2>/dev/null || break; sleep 0.1; done
  if owned anchor 0 "$OWN" "$pid"; then timeout 10 bash -c "echo go > '$GO'" 2>/dev/null
  else noanchor=1; log "FAIL-CLOSED $lane: no ownership anchor for root pid $pid at spawn -> launcher not released; nothing recorded or signalled"; fi
  rm -f "$GO"; sleep 1; pg=$(cat /tmp/night_seq_pg_$lane 2>/dev/null || ps -o sid= -p $pid 2>/dev/null | tr -d ' '); pg=${pg:-$pid}
  owned scan "$pg" "$OWN" "$pid"
  deadline=$(( $(now) + cap*60 + 60 )); (( deadline > END )) && deadline=$END; rc=""; endhit=0; it=0; tdleft=""; unverified=""  # F1: never past the window end
  while kill -0 $pid 2>/dev/null; do
    if (( $(now) > deadline )); then
      (( deadline == END )) && endhit=1
      log "TIMEOUT $lane ($( (( endhit )) && echo 'window end' || echo 'cap')) -> TERM own session $pg + owned descendants"
      owned scan "$pg" "$OWN" "$pid"; reap_owned "$OWN" "$lane"; tdleft=$REAP_LEFT
      # v11.4: the direct root KILL only after (pid, starttime, session) == spawn anchor; otherwise nothing is signalled (fail closed)
      if owned verify 0 "$OWN" "$pid"; then kill -KILL $pid 2>/dev/null; elif kill -0 $pid 2>/dev/null; then unverified=$pid; log "TIMEOUT $lane: root pid $pid identity NOT verified (pid,starttime,session vs spawn anchor) -> not signalled"; fi
      rc=124; break
    fi
    sleep 2; it=$(( it + 1 )); (( it % 5 == 0 )) && owned scan "$pg" "$OWN" "$pid"
  done
  [[ -z $rc ]] && { wait $pid; rc=$?; }
  # v11.1 F2: cleanup only of processes this step started (own session + recorded descendants, pid+starttime); foreign lock/GPU holders are never signalled
  owned scan "$pg" "$OWN" "$pid"
  left=$(owned alive "$pg" "$OWN")
  if [[ -n $left ]]; then log "WARN $lane left own processes alive: $left -> TERM/KILL (own only)"; reap_owned "$OWN" "$lane"; fi
  [[ -n ${REAP_LEFT:-} ]] && tdleft="$tdleft $REAP_LEFT"; REAP_LEFT=""
  for hp in $(lock_holders); do log "WARN GPU lock held by foreign pid $hp after $lane (not started by this step; left untouched)"; done
  status=OK; ferr=""; gpu_hit=0
  if [[ -n ${SEQ_SMOKE:-} ]]; then
    gpu_hit=$(for gp in $(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do grep -c "^$gp " "$OWN"; done | awk '{s+=$1} END{print s+0}')
    if (( ${gpu_hit:-0} > 0 )); then log "SEQ_SMOKE VIOLATION: $lane started a GPU process"; status=FAIL; ferr="SEQ_SMOKE violation: GPU process started by the step"; fi
  fi
  if [[ $rc == 124 ]]; then status=TIMEOUT; if (( endhit )); then ferr="stopped at window end $GA_END_UTC; log: $(first_error "$SLOG")"; else ferr="timeout after cap ${cap}+1 min; log: $(first_error "$SLOG")"; fi
  elif [[ $rc != 0 ]]; then status=FAIL; ferr=$(first_error "$SLOG"); fi
  # v11.2 F2-residual: own processes that survive TERM/KILL may still hold the GPU -> step FAIL and the chain stops (regardless of STOP)
  tdleft=$(echo $tdleft | tr ' ' '\n' | sort -un | sed 's/^/ /' | tr -d '\n')
  if [[ -n ${tdleft// /} ]]; then status=FAIL; ferr="teardown unresolved: own pid(s)$tdleft alive after TERM/KILL${ferr:+; $ferr}"; FORCE_STOP=1; fi
  # v11.4 fail closed: no spawn anchor or an unverifiable root at timeout -> step FAIL, chain stops, nothing was signalled
  if [[ -n $unverified ]]; then status=FAIL; ferr="root pid $unverified not verified at timeout (pid,starttime,session) - not signalled${ferr:+; $ferr}"; FORCE_STOP=1; fi
  if (( noanchor )); then status=FAIL; ferr="no ownership anchor at spawn - launcher not released, nothing recorded or signalled${ferr:+; $ferr}"; FORCE_STOP=1; fi
  smoke_record "$lane" "$path" "$sha" "$cap" "$args" "$rc" "${gpu_hit:-0}" "$status" "$ferr"
  log "END $lane rc=$rc"; record "$lane" "$status" "$rc" "$ferr" "$SLOG"
  [[ $status != OK ]] && fail_step "$lane" "$status" "$rc" "$ferr"
done <<< "$STEPS"
holder=$(timeout 10 lslocks 2>/dev/null | grep -c wrf_gpu2_gpu.lock)
log "all steps done; lock holders now $holder"
RESJ=$LOG.results.json
python3 - "$RTSV" "$RESJ" "$PLAN" "$STOP" "${STOPPED_AFTER:-}" "${SEQ_SMOKE:+smoke}" <<'PY' || { log "results JSON write FAILED"; NFAIL=$(( NFAIL + 1000 )); }
import json,sys,hashlib
tsv,out,plan,stop,after,mode=sys.argv[1:]
planned=[s["lane"] for s in json.load(open(plan))["steps"]]
rows=[l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
steps=[dict(lane=r[0],status=r[1],rc=(None if r[2]=="-" else int(r[2])),first_error=r[3],log=(None if r[4]=="-" else r[4])) for r in rows]
ok=[s["lane"] for s in steps]==planned and all(s["status"]=="OK" and s["rc"]==0 for s in steps)
json.dump(dict(schema="wrf_gpu2.night_seq.results.v11",plan=plan,plan_sha256=hashlib.sha256(open(plan,"rb").read()).hexdigest(),
  mode=mode or "run",stop_on_fail=stop=="1",stopped_after=after or None,verdict="OK" if ok else "FAIL",
  failed=[s["lane"] for s in steps if s["status"]!="OK"],steps=steps),open(out,"w"),indent=1)
PY
[[ -n ${STEP_RESULTS_JSON:-} ]] && cp "$RESJ" "$STEP_RESULTS_JSON"
NPLAN=$(echo "$STEPS" | sed '/^$/d' | wc -l); NROWS=$(wc -l < "$RTSV")
(( NROWS != NPLAN )) && { log "RESULT rows $NROWS != planned $NPLAN"; NFAIL=$(( NFAIL + 1 )); }
NNS=$(awk -F'\t' '$2=="NOT-STARTED"' "$RTSV" | wc -l)
if (( NFAIL > 0 || NNS > 0 )); then VERDICT=FAIL; SRC=10; else VERDICT=OK; SRC=0; fi
# per-step lines for notes: "- lane: STATUS rc=N — first error: ..."
step_lines() { awk -F'\t' '{ if ($2=="OK") printf "- %s: OK rc=%s\n",$1,$3; else if ($2=="NOT-STARTED") printf "- %s: NICHT GESTARTET (%s)\n",$1,$4; else printf "- %s: %s rc=%s — erster Fehler: %s\n",$1,$2,$3,$4 }' "$RTSV"; }
log "RESULT $VERDICT failed=$(awk -F'\t' '$2!="OK"&&$2!="NOT-STARTED"{printf "%s,",$1}' "$RTSV") not_started=$(awk -F'\t' '$2=="NOT-STARTED"{printf "%s,",$1}' "$RTSV") results=$RESJ"
if [[ -n ${STEP_ONLY:-} ]]; then
  { if [[ $VERDICT == OK ]]; then echo "# wrf_gpu2 → alisios_mgr, $(iso): Zwischenschritt OK (alle $NPLAN geplanten Schritte rc 0) — Slot läuft weiter (KEINE Rückgabe)"
    else echo "# wrf_gpu2 → alisios_mgr, $(iso): Zwischenschritt FAIL ($NFAIL fehlgeschlagen, $NNS nicht gestartet; Sequenzer rc $SRC) — Slot läuft weiter (KEINE Rückgabe)"; fi
    echo; step_lines; } > "$CROSS/$DONE_NAME"
  chmod 0644 "$CROSS/$DONE_NAME"
  log "STEP note $CROSS/$DONE_NAME written (no release; $VERDICT)"
  [[ -n ${SEQ_SMOKE:-} ]] && log "SEQ_SMOKE summary $SMOKE_JSON"
  exit $SRC
fi
if [[ -n ${SEQ_SMOKE:-} ]]; then log "SEQ_SMOKE summary $SMOKE_JSON ($VERDICT)"; exit $SRC; fi
{ echo "# wrf_gpu2 → alisios_mgr, $(iso): night window DONE — GPU free — Ergebnis $VERDICT"; echo
  echo "Window $GA_START_UTC → $GA_END_UTC (planned), closed early at $(iso)."
  echo "GPU lock holders after the window: $holder. GPU free for ALISIOS."; echo; echo "Steps:"; step_lines
} > "$CROSS/$DONE_NAME"
chmod 0644 "$CROSS/$DONE_NAME"
log "DONE file $CROSS/$DONE_NAME written ($VERDICT)"
exit $SRC
