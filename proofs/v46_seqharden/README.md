# v46-seqharden — sequencer/gate hardening (ALISIOS Sol F1, FINDINGS E262)

Canonical files live in `<USER_HOME>/wrf_gpu2_lanes/v46-seqharden/` (lane dir); this directory carries byte-identical copies for review.
Live manager gates/sequencers were NOT edited.

| file | from | what changed |
|---|---|---|
| `night_seq_v11.sh` | `manager/night_seq_v9.sh` | stop-on-fail (STEP_ONLY always; STOP_ON_FAIL=1 or plan `stop_on_fail`), exit 10 on any non-OK planned step, `NOT-STARTED <lane>` lines, first error line per step, `<log>.results.json`, plan/manifest `required_env` PREFLIGHT (values never printed), SEQ_SMOKE refuses the real cross, `setsid -w`, stale pg-file removed before each step |
| `gate_template_v2.sh` | `manager/gate_session_w1aq_r16_20261030.sh` | conf-driven (title/cross names/windows/floors/leases; no sed-derived copies), O72 line only with NEED_O72, arming-time PREFLIGHT under the exact sequencer env (catches missing/empty/malformed and env-stripped pins), STOP note lists every step with status + first error (also when the sequencer aborts without results), gate exit 10 on STOP, single log lines, GATE_SMOKE mode |
| `gate_conf_example_R16.conf` | R16 | R16 expressed as a conf |
| `smoke/run_smoke.py` + `smoke/stubs/` | new | exact SEQ_SMOKE / GATE_SMOKE proof, v9 BEFORE reproduction, 5 sequencer + 2 gate mutants |

`required_env` types: `sha256`, `sha256_or_NONE`, `NONE`, `nonempty`, `utc`, `int`, `re:<regex>` (full match). Example (R15/R16 pins):
`"required_env": {"W1A_EXTENSION_RESULT_SHA": "sha256", "W1A_SCREEN6_GPU_BUDGET_SHA": "sha256_or_NONE"}`.
Behavioural deltas beyond F1 (disclose to critic): step poll 10 s → 2 s; SKIPPED (time/ram/sha) counts as a failure (stop + rc 10).

## v11.1 / v2.1 delta (critic cr-w44-launchers FIX 2026-10-31T01:28Z)
- F1: after the RAM/sha waits the sequencer re-checks the foreign GPU lock (non-smoke; wait LOCK_WAIT_S=60 s, then SKIP→stop) and the
  remaining window (rem < cap+2 → SKIP→stop) immediately before dispatch; per-step deadline = min(END, start+cap+1 min); a step hit by the
  window end is TIMEOUT with first_error "stopped at window end".
- F2: post-step/timeout cleanup signals ONLY processes the step started: its own setsid session plus every descendant seen while it ran
  (recorded pid+starttime every ~10 s, so PID reuse is impossible); session matching is disabled if the step did not get its own session.
  Foreign lock holders / GPU processes are never signalled, only logged (v9 TERM/KILLed every lock holder and any GPU pid whose cmdline
  looked like one of our lanes). lslocks/nvidia-smi queries bounded by timeout 10/15 s. Limit: a descendant that escapes into a new session
  AND whose parent exits within < 10 s between two scans is not tracked (documented, not seen in the tests).
- F3: umask 077 in both scripts → logs/results/step logs/smoke JSON/gate .out/invocation 0600; cross notes chmod 0644 explicitly (shared
  channel, contents redacted first-error lines only).
- Proof run_20261031T014432Z PASS: 13 seq + 7 gate + 10 real-mode (K1–K5 + 5 fix mutants) + 5 seq mutants; real-mode cases run a private
  non-smoke copy (only window flag / pg dir / real_utc path redirected; fake clock/RAM for K1/K2; lslocks/nvidia-smi stubbed on PATH).

## v11.2 delta (critic cr-w44-launchers F2-RESIDUAL 2026-10-31T02:19:34Z) — night_seq_v11 08c81531 → f6c3bcc7, gate unchanged (ec92c5fa)
- R1 ownership identity: owned() treats the FIRST state-file line of a pid as its anchor; the root (the `$!` process) is anchored by an
  extra scan immediately at spawn (before the 1 s pg wait). The root seeds ownership/growth only with its anchored starttime; the session id
  is matched only while its anchored leader is alive, or an already-recorded member (pid+starttime) is alive in it (a live member keeps the
  sid number allocated → no reuse); unanchored/reused sid or root → nothing. Growth only from validated members, child start ≥ parent start.
  'alive' excludes zombies.
- R2 teardown settlement: reap_owned = TERM, REAP_GRACE_S, KILL, then up to REAP_SETTLE_S (10) 1 s checks; own non-zombie survivors →
  rc 1, REAP_LEFT, log "TEARDOWN UNRESOLVED"; the step becomes FAIL ("teardown unresolved: own pid(s) …") and the chain stops even without
  STEP_ONLY/stop_on_fail (next steps NOT-STARTED, rc 10, results FAIL → gate STOP, no DONE).
- Targeted test smoke/test_f2_residual.py (proof F2_RESIDUAL_PROOF.json, run f2res_20261031T161129Z PASS): exact inline owned() on fake
  /proc — critic's N1 reused leader / N2 reused root + other sid, N3 sid reused after session end → nothing owned; positives P1–P5 (valid
  root/session/descendants, leader exited + anchored member, first sight, forked setsid child leader, zombie); exact reap_owned with fake
  owned/kill/sleep (survivor → rc 1, settled after TERM/KILL → 0); real own processes on a private non-smoke copy: TERM-immune survivor
  with KILL disabled → p-one FAIL "teardown unresolved", p-two NOT-STARTED, rc 10 without STEP_ONLY; control with KILL → rc 0, reaped.
  The negatives N1–N3 and the survivor case FAIL by assertion on the pre-fix v11.1 source (lane dir night_seq_v11.v111.sh = 08c81531).
- Not rerun (no rebuild): the full run_smoke.py proof (13 seq / 7 gate / 10 real-mode); the change touches only owned()/reap_owned and
  the teardown verdict. Known limit unchanged: an escaped-session descendant whose parent exits between two ~10 s scans is untracked.

## v11.3 delta (critic cr-v46-terra FIX 2026-10-31T18:12:58Z) — night_seq_v11 f6c3bcc7 → 03ea9a3f, gate unchanged (ec92c5fa)
- The ownership anchor is written ONLY by `owned anchor` immediately at spawn; `scan` never anchors (v11.2 anchored a root on first sight
  in any scan). Root absent at spawn → anchor rc 3, sequencer logs "WARN <lane>: root pid … gone at spawn -> no ownership anchor; nothing
  recorded or signalled for this step (report only)"; afterwards nothing is ever owned for that step. The unanchored 'sid == root' session
  clause is removed (a sid is matched only via its anchored leader or an anchored live member).
- Test f2res_20261031T192100Z PASS: new N4 = the critic's case (empty /proc at spawn → anchor rc 3; later reused 101/start 2000 + foreign
  child 102 → nothing owned, state file stays empty), N5 scan with no anchor → nothing; P3 now spawn-anchor then scan → 101 102; all v11.2
  cases carried (N1–N3, P1/P2/P4/P5, reaper RA–RC, real chain T1 rc 10 / T2 rc 0). N4/N5 FAIL by assertion on the v11.2 source
  (lane dir night_seq_v11.v112.sh = f6c3bcc7); N1–N3 + survivor still FAIL on v11.1.

## v11.4 delta (critic cr-v46-terra FIX 2026-10-31T20:11:24Z) — night_seq_v11 03ea9a3f → 7844cc80, gate unchanged (ec92c5fa)
- Spawn gate: the step's leader writes its pg file, then blocks on a private fifo (`<log>.<lane>.go`, 0600) until the sequencer has
  anchored it → root alive and setsid() done at anchor time, also for steps that would exit at once (bash reaps finished background jobs
  itself, so `$!` is NOT protected against PID reuse — found by the control test: v11.3-style anchoring failed a fast OK step).
  `owned anchor` writes pid+starttime into the state file and `pid starttime session` into `<state>.anchor`.
- No anchor → fail closed: the launcher is never released (leader exits 97 after SPAWN_GO_TIMEOUT_S, default 60), step FAIL
  "no ownership anchor at spawn - launcher not released, nothing recorded or signalled", chain stops (also without STEP_ONLY).
- Timeout: the direct root KILL runs only if `owned verify` confirms (pid, starttime, session) == spawn anchor; otherwise nothing is
  signalled, log "identity NOT verified", step FAIL "root pid … not verified at timeout", chain stops.
- Test f2res_20261031T220905Z PASS: R1V verify V1 same → 0, V2 reused pid, V3 other session, V4 no anchor + reused 101/102, V5 gone → 1;
  real chains without STEP_ONLY: T3 no anchor → launcher never ran (no marker, rc 97), FAIL + p-two NOT-STARTED, nothing signalled;
  T4 timeout with verified root → root + child killed (control); T5 timeout with falsified anchor → root and child NOT signalled (alive,
  then killed by the test), FAIL + NOT-STARTED, rc 10; all earlier cases carried (T2 control with a fast step now OK). V2–V4 FAIL by
  assertion on v11.3 (lane dir night_seq_v11.v113.sh = 03ea9a3f). Full run_smoke.py not rerun (spawn path exercised by 7 real chain runs).
