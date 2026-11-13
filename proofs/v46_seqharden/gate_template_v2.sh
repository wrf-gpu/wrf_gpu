#!/bin/bash
# Gate template v2 (lane v46-seqharden, 2026-10-31; ALISIOS Sol F1 / FINDINGS E262 / V041_FOLLOWUPS 2026-10-29 template debt).
# Usage: gate_template_v2.sh <gate.conf>     (the conf is a bash file of plain assignments; it is the ONLY per-gate edit — no more sed-derived copies)
# Derived from gate_session_w1aq_r16_20261030.sh (c17d lineage R5..R16): same admission (trigger glob, not-before/latest-start, lock holders 0,
# DATA floor+NEED, ROOT_ADMIT, quiet GPU-arm cores, run-lease expiry, plan cap sum, PREFLIGHT, optional admission SHA pin, CPU lease in traps).
# v2 deltas:
#  - title/cross names come from the conf (TITLE, CROSS_FMT); no stale 'v0.4.5-RC RW1–RW4' text; the O72 line only if NEED_O72 is set.
#  - sequencer = night_seq_v11 (stop-on-fail under STEP_ONLY, rc 10 on any failed step, <log>.results.json with first error per step).
#  - ARMING CHECK: right at arming the gate runs night_seq_v11 PREFLIGHT_ONLY with exactly the env the real invocation gets → a plan-declared
#    required_env pin that is missing/empty/malformed (R15: empty budget pin instead of NONE, missing J2 pin) refuses at once (exit 2, no cross).
#  - STOP note names EVERY failed step with its status, rc and first error line, and every NOT-STARTED step; gate exit 10 (DONE 0, NOSTART 4).
#  - GATE_SMOKE=1: exact-call CPU smoke of the whole gate: night_seq SEQ_SMOKE=1, private CROSS_DIR (real cross refused), fake CPU_LEASE_BIN
#    required, lock holders only logged, QUIET_S seconds core sample. No GPU, no real lease, no real cross.
set -u
export LC_ALL=C
umask 077  # v2.1 F3 (E245): gate logs/invocation/results 0600; cross notes chmod 0644 explicitly
CONF=${1:?usage: gate_template_v2.sh <gate.conf>}
# shellcheck disable=SC1090
source "$CONF" || { echo "conf $CONF unreadable"; exit 2; }
for v in GATE_TAG TITLE CROSS_FMT PLAN GO_GLOB NOT_BEFORE_UTC LATEST_START_UTC GPU_CAP_UTC FLOOR NEED ROOT_ADMIT RUN_LEASE CPU_LEASE_ID CPU_LEASE_CORES GA_GATE_ID; do
  [[ -n ${!v:-} ]] || { echo "conf $CONF: $v unset"; exit 2; }; done
[[ $CROSS_FMT == *%s* ]] || { echo "conf: CROSS_FMT needs one %s (START/DONE/STOPPED/NOSTART/STEP)"; exit 2; }
for v in FLOOR NEED ROOT_ADMIT; do [[ ${!v} =~ ^[0-9]+$ ]] || { echo "conf: $v not an integer"; exit 2; }; done
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M=<USER_HOME>/wrf_gpu2_lanes/manager
REAL_CROSS=<DATA_ROOT>/alisios/state/manager/cross
SEQ=${NIGHT_SEQ:-$HERE/night_seq_v11.sh}
ISSUE=${ISSUE:-20261026_12z}
QUIET_CORES=${QUIET_CORES:-3,13,19,29}; QUIET_S=${QUIET_S:-60}; QUIET_MAX_BUSY=${QUIET_MAX_BUSY:-0.5}  # R16 rule: busy <= 50 % per core
GATE_DIR=${GATE_DIR:-$M}
CROSS=${CROSS_DIR:-$REAL_CROSS}
SMOKE=${GATE_SMOKE:-}
if [[ -n $SMOKE ]]; then
  [[ $(realpath -m "$CROSS") != $(realpath -m "$REAL_CROSS") ]] || { echo "GATE_SMOKE refuses the real ALISIOS cross dir"; exit 2; }
  [[ -n ${CPU_LEASE_BIN:-} && $(realpath -m "$CPU_LEASE_BIN") != $(realpath -m "$M/cpu_lease.sh") ]] || { echo "GATE_SMOKE needs a fake CPU_LEASE_BIN"; exit 2; }
  [[ $(realpath -m "$GATE_DIR") != $(realpath -m "$M") ]] || { echo "GATE_SMOKE needs a private GATE_DIR"; exit 2; }
fi
mkdir -p "$GATE_DIR" "$CROSS"
xf() { printf "$CROSS_FMT" "$1"; }
STARTF=$CROSS/$(xf START); DONEF=$CROSS/$(xf DONE); STOPF=$CROSS/$(xf STOPPED); NOSTARTF=$CROSS/$(xf NOSTART); STEPNOTE=$(xf STEP)
OUT=$GATE_DIR/gate_${GATE_TAG}.out
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
now() { date -u +%s; }
log() { local l; l="$(ts) $*"; echo "$l" >> "$OUT"; [[ /dev/stdout -ef $OUT ]] || echo "$l"; }  # R16 printed every line twice
nostart() { log "NOSTART: $*"; printf '# wrf_gpu2 → alisios_mgr, %s: %s KEIN START — %s; GPU nicht benutzt\n' "$(ts)" "$TITLE" "$*" > "$NOSTARTF"; chmod 0644 "$NOSTARTF"; exit 4; }
holders() { timeout 10 lslocks 2>/dev/null | grep -c wrf_gpu2_gpu.lock; }
rootfree() { echo $(( $(df --output=avail -B1 / | tail -1 | tr -dc '0-9') / 1073741824 )); }   # floor GiB
datafree() { echo $(( $(df --output=avail -B1 <DATA_ROOT> | tail -1 | tr -dc '0-9') / 1073741824 )); }
quiet_cores() { python3 - "$QUIET_CORES" "$QUIET_S" "$QUIET_MAX_BUSY" <<'PY'
import sys, time
cores=[int(c) for c in sys.argv[1].split(",")]; dt=float(sys.argv[2]); mx=float(sys.argv[3])
def rd():
    d={}
    for l in open("/proc/stat"):
        if l.startswith("cpu") and l[3].isdigit():
            f=l.split(); v=list(map(int,f[1:])); d[int(f[0][3:])]=(sum(v), v[3]+v[4])
    return d
a=rd(); time.sleep(dt); b=rd()
bad=[c for c in cores if (b[c][0]-a[c][0]) and 1-(b[c][1]-a[c][1])/(b[c][0]-a[c][0]) > mx]
raise SystemExit(1 if bad else 0)
PY
}
# exactly the env the sequencer gets in the real invocation (also used for the arming/start PREFLIGHT, so a pin stripped here is caught)
seq_env() { env -u SEQ_SMOKE -u PREFLIGHT_ONLY -u STEP_ONLY -u WRF_GPU2_SEQ_SMOKE -u W44R_LAUNCH_SMOKE -u LD_PRELOAD -u CUDA_ROOT -u CUDA_VISIBLE_DEVICES \
  $(compgen -e | grep -E '^(W44R_|GPUWRF_|XLA_|JAX_|TF_)' | sed 's/^/-u /') ${SMOKE:+SEQ_SMOKE=1 CROSS_DIR=$CROSS} NIGHT_SEQ_LOG_DIR=$GATE_DIR "$@"; }
utc_s() { date -u -d "$1" +%s 2>/dev/null || { echo "conf: bad UTC $1" >&2; kill -TERM $$; }; }
LAT=$(utc_s "$LATEST_START_UTC"); NB=$(utc_s "$NOT_BEFORE_UTC"); GPU_CAP=$(utc_s "$GPU_CAP_UTC")
[[ $LAT =~ ^[0-9]+$ && $NB =~ ^[0-9]+$ && $GPU_CAP =~ ^[0-9]+$ ]] || { echo "conf: bad UTC"; exit 2; }
HARD_END=${HARD_END:-$GPU_CAP_UTC}; he=$(date -u -d "$HARD_END" +%s)
(( he > GPU_CAP )) && { HARD_END=$GPU_CAP_UTC; he=$GPU_CAP; }   # an inherited later HARD_END cannot extend the conf cap
PLANSHA=$(sha256sum "$PLAN" | cut -d' ' -f1) || exit 2
log "gate armed (template v2, conf $(basename "$CONF") sha $(sha256sum "$CONF" | cut -c1-16)); plan $PLAN sha ${PLANSHA:0:16}; trigger $GO_GLOB; hard end $HARD_END${SMOKE:+; GATE_SMOKE}"
# ARMING CHECK (v2): plan runnable + every required_env pin present and well-formed under the real sequencer env
if ! seq_env PREFLIGHT_ONLY=1 WAIT_MODE=gpufree bash "$SEQ" "$PLAN" "$ISSUE" NONE > "$GATE_DIR/gate_${GATE_TAG}_arm_preflight.out" 2>&1; then
  grep -E 'REFUSE|FAILED|bad plan' "$GATE_DIR/gate_${GATE_TAG}_arm_preflight.out" | sed 's/^[^ ]* //' | while IFS= read -r l; do log "ARM REFUSE: $l"; done
  log "ARM REFUSED: fix plan/arming env (no cross written, gate not armed)"; exit 2
fi
log "arming preflight OK ($(grep -c 'REQENV OK' "$GATE_DIR/gate_${GATE_TAG}_arm_preflight.out") required env pins)"
if [[ -n ${ADM:-} ]]; then
  [[ -n ${ADMSHA_PIN_VAR:-} && -n ${!ADMSHA_PIN_VAR:-} ]] || { log "ARM REFUSED: ADM set but pin variable ${ADMSHA_PIN_VAR:-<unset>} empty"; exit 2; }
  [[ $(sha256sum "$ADM" 2>/dev/null | cut -d' ' -f1) == "${!ADMSHA_PIN_VAR}" ]] || { log "ARM REFUSED: admission $ADM missing or SHA != \$$ADMSHA_PIN_VAR"; exit 2; }
fi
while :; do
  t=$(now)
  (( t > LAT )) && nostart "Startfenster $LATEST_START_UTC ohne Trigger/freie GPU verstrichen"
  h=$(holders); [[ -n $SMOKE ]] && h=0
  go=$(compgen -G "$CROSS/$GO_GLOB" | head -1)
  [[ -z $go || $(basename "$go") == $GO_GLOB ]] || go=""
  if [[ -n $go && $h == 0 ]] && (( t >= NB )) && (( $(datafree) - NEED >= FLOOR )) && (( $(rootfree) >= ROOT_ADMIT )) && quiet_cores; then break; fi
  sleep ${GATE_POLL_S:-20}
done
heg=$(grep -oE "HARD_END_UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z" "$go" | head -1 | cut -d= -f2)
if [[ -n $heg ]] && (( $(date -u -d "$heg" +%s) < he )); then HARD_END=$heg; he=$(date -u -d "$heg" +%s); fi
log "window opens (trigger $(basename "$go"), lock holders $(holders)${SMOKE:+ [smoke: not required]}, hard end $HARD_END)"
exp=$(grep -m1 '^expires-utc:' "$RUN_LEASE" 2>/dev/null | cut -d' ' -f2)
[[ -n $exp ]] && (( $(date -u -d "$exp" +%s) >= he )) || nostart "Lease $(basename "$RUN_LEASE") fehlt/läuft vor HARD_END ab ($exp)"
free=$(datafree); (( free - NEED >= FLOOR )) || nostart "Disk: ${free} GiB frei (abgerundet) < $(( FLOOR + NEED )) GiB"
(( $(rootfree) >= ROOT_ADMIT )) || nostart "Root: $(rootfree) GiB frei (abgerundet) < ${ROOT_ADMIT} GiB"
sum=$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(sum(int(s["cap_min"])+2 for s in p["steps"]))' "$PLAN")
rem=$(( (he - $(now)) / 60 )); (( sum <= rem )) || nostart "Zeit: braucht $sum min, $rem übrig"
[[ $(sha256sum "$PLAN" | cut -d' ' -f1) == "$PLANSHA" ]] || nostart "Plan-SHA hat sich seit dem Scharfschalten geändert"
seq_env PREFLIGHT_ONLY=1 WAIT_MODE=gpufree bash "$SEQ" "$PLAN" "$ISSUE" NONE > "$GATE_DIR/gate_${GATE_TAG}_preflight.out" 2>&1 \
  || nostart "PREFLIGHT night_seq_v11 fehlgeschlagen: $(grep -m3 -E 'REFUSE|FAILED' "$GATE_DIR/gate_${GATE_TAG}_preflight.out" | sed 's/^[^ ]* //' | tr '\n' ';')"
(( $(now) <= LAT )) || nostart "Startfenster $LATEST_START_UTC während Preflight überschritten"
[[ -n $SMOKE || $(holders) == 0 ]] || nostart "GPU-Lock wurde vor dem Start belegt"
if [[ -n ${ADM:-} ]]; then
  [[ $(sha256sum "$ADM" | cut -d' ' -f1) == "${!ADMSHA_PIN_VAR}" ]] || nostart "Admission $(basename "$ADM") fehlt oder SHA weicht ab"
  [[ -n ${EXPORT_ADM_AS:-} ]] && export "$EXPORT_ADM_AS=${!ADMSHA_PIN_VAR}"
fi
INV=$GATE_DIR/gate_${GATE_TAG}_invocation_$(date -u +%Y%m%dT%H%M%SZ).out; : > "$INV"
RESULTS=$GATE_DIR/gate_${GATE_TAG}_results.json; rm -f "$RESULTS"
export GA_RW_GATE_ID=gate_${GATE_TAG}_$(date -u +%Y%m%dT%H%M%SZ)_$$ GA_GATE_ID CPU_LEASE_DIR=${CPU_LEASE_DIR:-$M}
{ echo "# wrf_gpu2 → alisios_mgr, $(ts): START $TITLE"
  echo "- Trigger $(basename "$go"), Lock-Halter $(holders), harter Endpunkt $HARD_END; Plan $(basename "$PLAN") sha256 ${PLANSHA:0:16}…, Summe Cap+Marge $sum min"
  echo "- Disk frei ${free} GiB (abgerundet, Boden $FLOOR + Reserve $NEED); Root frei $(rootfree) GiB (≥ $ROOT_ADMIT)"
  [[ -n ${NEED_O72:-} ]] && echo "- O72 läuft nur, wenn der Launcher bei seiner eigenen Prüfung >= $(( FLOOR + NEED_O72 )) GiB sieht"
  echo "- Sequenzer night_seq_v11 (Stopp nach dem ersten fehlgeschlagenen Schritt; rc ≠ 0 bei jedem Schrittfehler)"; } > "$STARTF"; chmod 0644 "$STARTF"
LEASED=0; ACQ_PID=""; RELEASE_FAILED=0
LEASE_FILE=${CPU_LEASE_DIR:-$M}/leases/${CPU_LEASE_ID}-cpu.lease
LBIN=${CPU_LEASE_BIN:-$M/cpu_lease.sh}
release_lease() {
  (( LEASED )) || return 0
  "$LBIN" release "$CPU_LEASE_ID" "${1:-gate exit}"; local r=$?
  [[ -e $LEASE_FILE ]] || LEASED=0
  if (( r != 0 )) || [[ -e $LEASE_FILE ]]; then RELEASE_FAILED=1; log "LEASE RELEASE FAILED rc=$r (lease file $( [[ -e $LEASE_FILE ]] && echo present || echo absent))"; fi
}
own_lease() { [[ -f $LEASE_FILE ]] && grep -q "gate pid $$)" "$LEASE_FILE"; }
on_signal() {
  log "signal received — terminal shutdown"
  if [[ -n $ACQ_PID ]]; then kill "$ACQ_PID" 2>/dev/null; wait "$ACQ_PID" 2>/dev/null; ACQ_PID=""; own_lease && LEASED=1; fi
  release_lease "gate signal"; trap - EXIT
  (( LEASED )) && log "WARNING: lease could not be released on signal"
  exit 130
}
on_exit() { release_lease "gate exit/abort"; (( LEASED )) && release_lease "gate exit retry"; }
trap on_exit EXIT
trap on_signal INT TERM HUP
"$LBIN" acquire "$CPU_LEASE_ID" "$CPU_LEASE_CORES" "$(date -u -d @$he +%Y-%m-%dT%H:%M:%SZ)" "$TITLE (night_seq_v11, gate pid $$)" &
ACQ_PID=$!; wait $ACQ_PID; arc=$?; ACQ_PID=""
if (( arc == 0 )) && own_lease; then LEASED=1
else own_lease && LEASED=1; release_lease "acquire failed rc $arc"; nostart "CPU-Lease konnte nicht angelegt werden (cpu_lease rc $arc)"; fi
(( $(now) <= LAT )) || { release_lease "latest start passed during lease wait"; nostart "Startfenster $LATEST_START_UTC während Lease-Wartezeit überschritten"; }
log "PART start (invocation log $INV)"
set -o pipefail
seq_env STEP_ONLY=1 WAIT_MODE=gpufree HARD_END_UTC=$HARD_END WINDOW_S=36000 STEP_RESULTS_JSON=$RESULTS bash "$SEQ" "$PLAN" "$ISSUE" "$STEPNOTE" 2>&1 | tee -a "$INV" > /dev/null
rc=${PIPESTATUS[0]}
set +o pipefail
release_lease "step ended rc ${rc}"
# verdict: results JSON (every planned step exactly once, status OK, rc 0) AND sequencer rc 0 AND END-line count cross-check
SUM=$(python3 - "$RESULTS" "$PLAN" "$INV" <<'PY'
import json, re, sys
res, plan, inv = sys.argv[1:]
planned = [s["lane"] for s in json.load(open(plan))["steps"]]
bad = []
try: r = json.load(open(res))
except (OSError, ValueError): r = None; bad.append("results-json-missing")
text = open(inv, errors="replace").read()
ends = re.findall(r"\bEND ([A-Za-z0-9_.-]+) rc=([0-9]+)\b", text)
lines = []
if r is None:
    why = [" ".join(l.split()[1:])[:240] for l in text.splitlines() if re.search(r"ABORT|REFUSE|FAILED|bad plan", l)][:3] or ["kein Sequenzer-Ergebnis"]
    lines = [f"- {l}: NICHT GELAUFEN (Sequenzer ohne Ergebnis: {'; '.join(why)})" for l in planned]
if r:
    got = [s["lane"] for s in r["steps"]]
    if got != planned: bad.append("steps!=plan")
    for s in r["steps"]:
        if s["status"] == "OK" and s["rc"] == 0: lines.append(f"- {s['lane']}: OK rc 0")
        elif s["status"] == "NOT-STARTED": bad.append(f"{s['lane']}:NOT-STARTED"); lines.append(f"- {s['lane']}: NICHT GESTARTET ({s['first_error']})")
        else: bad.append(f"{s['lane']}:{s['status']}(rc={s['rc']})"); lines.append(f"- {s['lane']}: {s['status']} rc {s['rc']} — erster Fehler: {s['first_error']}")
ran = [s["lane"] for s in (r["steps"] if r else []) if s["status"] not in ("NOT-STARTED", "SKIPPED")]
if sorted(l for l, _ in ends) != sorted(ran): bad.append("END-lines!=results")
print(" ".join(bad) or "none"); print("\n".join(lines))
PY
)
bad=$(head -1 <<< "$SUM"); stepl=$(tail -n +2 <<< "$SUM")
(( RELEASE_FAILED || LEASED )) && bad="${bad/#none/} cpu-lease-release-failed"
log "night_seq rc=$rc bad: $bad"
while IFS= read -r l; do [[ -n $l ]] && log "STEP $l"; done <<< "$stepl"
if [[ $rc == 0 && $bad == none ]]; then
  { echo "# wrf_gpu2 → alisios_mgr, $(ts): $TITLE DONE (alle geplanten Schritte genau einmal rc 0) — GPU frei"; echo "$stepl"
    echo "- Lock-Halter jetzt: $(holders)"; } > "$DONEF"; chmod 0644 "$DONEF"; GRC=0
else
  { echo "# wrf_gpu2 → alisios_mgr, $(ts): $TITLE STOPPED — FAIL (Sequenzer rc $rc; Gate rc 10) — KEIN sauberes Paketende, GPU frei"
    echo "- Abweichungen: $bad"; echo "- Alle Schritte (jeder Fehler mit erster Fehlerzeile):"; echo "$stepl" | sed 's/^/  /'
    echo "- Lock-Halter jetzt: $(holders)"; } > "$STOPF"; chmod 0644 "$STOPF"; GRC=10
fi
log "gate end rc=$GRC"
exit $GRC
