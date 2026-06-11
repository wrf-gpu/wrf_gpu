# Sprint Contract: v0.14 Switzerland Venting Residual Fix (Fable 5 medium)

Date: 2026-06-11 WEST
Owner: manager (Opus 4.8)
Assignee: Fable 5 medium, dedicated worktree
Branch: `worker/fable/v014-venting-residual-fix` (base = advance_w-fixed tip)

## Objective

Close the remaining Switzerland/Gotthard h36 strong-flow dry-mass **venting
residual** so that the 72h GPU field-parity/stability gate passes. The dominant
per-substep `(w,phi)` creator is already fixed and merged (`b14b5f17`:
`pg_buoy_w` carried-`grid%p`, w cosine-Coriolis + curvature, WRF open top). What
remains is a smaller interior residual that still produces excess hourly mass
outflux in the h36–h72 Alpine storm window.

This is the last open v0.14 correctness blocker. The endpoint is a real,
WRF-faithful fix proven against the WRF-native oracle — not a clamp, not a
tolerance change, not a station-score workaround.

## Compact evidence chain (what is already proven)

From the merged WRF-native intra-`advance_w` dump
(`proofs/v014/wrf_native_advance_w_dump.{py,json,md}`, dump data at
`/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps/`, WRF call
`21601 -> 21602`, the first-bad RK1 acoustic substep):

- `rw_tend` lane is **closed**: post-fix production `rw_tend` matches WRF-native
  total at 0.37% rel; stage `ph` error dropped 96.5% (substep gate
  `p` 1.126→0.369, `ph` 0.435→0.0151, `al` 6.75e-6).
- The 2h open-top short forecast from the h36 re-init is **stable** (PASS,
  max|W| 3.4 m/s); the 2026-05-30 open-top blow-up does NOT recur.
- BUT the **72h venting KI is NOT closed**: hourly venting excess only improved
  −28.8 → **−26.6 Pa/cell/h** (CPU reference outflux −74.5). The surviving
  residual tracks the interior coupled work–theta / EOS lane.

## Candidates to TEST (ranked clues, NOT assumptions)

Build your own ranked hypothesis ledger. These are the named residuals from the
dump; test and reject any that the proof does not support:

1. coupled work–theta `t_2` lane — 53.7% rel (small absolute) — now also
   carries the remaining stage `p` error through the EOS / `calc_p_rho`.
2. `ph_tend` contribution into the RHS / implicit `(w,phi)` solve — 13.9% rel.
3. `mu''` mass coupling — 15.5% rel.
4. `ww` (omega) — 51% rel of a 0.003-rms field.

The manager's suspicion is the EOS/theta coupling lane; do not preserve it if
the oracle points elsewhere. The endpoint is whole-task: find and fix the
WRF-anchored root if local and provable, or return the strongest falsifiable
narrowing with the exact next WRF term and a precise fix recommendation.

## Method (reuse the existing oracle — do not rebuild from scratch)

- WRF-native term oracle: `proofs/v014/wrf_native_advance_w_dump.py`
  (`--manifest --compare`, `--compare-rw`) against the dumped truth.
- Stage gate (GPU): `proofs/v014/switzerland_acoustic_substep_blocker.py
  --stage-compare --tag <name> --steps 1` → interior increment rmse `mu/p/ph/al`
  vs WRF call 21602.
- Short forecast venting budget (GPU): 2h open-top forecast from the h36
  re-init `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`;
  report depth-8 interior hourly excess in Pa/cell/h vs CPU −74.5.
- If a new WRF-native term must be dumped, the disposable additive
  instrumentation patch is `proofs/v014/wrf_native_advance_w_dump_wrf_patch.diff`
  (re-instrument a disposable WRF tree; never edit `/home/enric/src/wrf_pristine/WRF`
  in place).

## Constraints

- No clamps, no masking, no tolerance changes, no JAX-vs-JAX self-acceptance.
- No host/device transfer inside the timestep loop.
- Do NOT run the long 72h GPU gate — the manager runs it after merge.
- Do not touch `/home/enric/src/canairy_waves`. Do not use Hermes/`ask-hermes`.
- Work only in the assigned worktree; commit source + proofs on your branch.
- Keep instrumentation env-gated and off by default; nothing debug-only in the
  production hot path.

## Acceptance gate (manager merges on)

1. A proven local WRF-faithful source fix that **materially collapses** the h36
   hourly venting excess well below −26.6 Pa/cell/h (toward the CPU −74.5
   reference, i.e. excess → ~0), with the 2h short forecast still stable
   (no blow-up, finite); OR
2. an exact WRF-anchored proof that names the precise next mismatching term /
   unexposed input with a concrete fix recommendation, if no further local fix
   is provable this round.
- Stage-compare gate + short-forecast venting budget JSON committed under
  `proofs/v014/`.
- Review doc at `.agent/reviews/2026-06-11-v014-fable-venting-residual-fix.md`.
- `git diff --check` clean; no source files outside the venting lane changed.

## Completion marker

Print exactly:
`FABLE V014_VENTING_RESIDUAL_FIX DONE - see .agent/reviews/2026-06-11-v014-fable-venting-residual-fix.md`
