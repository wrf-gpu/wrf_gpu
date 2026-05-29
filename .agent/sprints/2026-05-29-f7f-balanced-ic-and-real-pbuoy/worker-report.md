# F7F Worker Report — WRF-balanced IC + remove synthetic p_buoy

**Frontrunner**: Opus 4.8 · **Branch**: `worker/opus/f7d-pressure-mass-fix` (commits `3422ee1`, `8800d68`)
**GPU**: cuda:0, fp64, `taskset -c 0-3` throughout.

## Objective
Implement the GPT-5.5 WRF-verified fork fix (remove the synthetic absolute `p_buoy`;
make the IC ph-rebalance explicit/WRF-faithful; use the real perturbation pressure in
the rk_addtend diagnostics) and make the idealized cases PASS = F7 dycore close.

## Files changed
- `src/gpuwrf/ic_generators/idealized.py` — `_make_state` now integrates the WRF
  fixed-mass hydrostatic rebalance line-for-line (`al=alt_full-alb`,
  `ph'(k+1)=ph'(k)-dnw*(c1h*mub+c2h)*al`, `ph'(1)=0`; mu'=0). Bit-identical fields to
  the prior formulation but matches `module_initialize_ideal.F:982/:1124-1129/:1308-1313`;
  removed the misleading "base ph + θ is the buoyancy source" comment.
- `src/gpuwrf/runtime/operational_mode.py` — removed Sprint B's synthetic absolute
  `p_buoy_abs`; `p_buoy=None` (acoustic core consumes the live `calc_p_rho` work
  pressure). Documented the open large-step/small-step buoyancy-reference issue.
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py` — `_absolute_diagnostics` now uses
  `state.p_perturbation` for WRF `p` (not a re-derived absolute-θ pressure); `al` from
  `ph_perturbation`+`mu_perturbation`; `alt` from EOS; `php` from full ph.
- `src/gpuwrf/dynamics/acoustic_wrf.py` — **decisive fix**: `diagnose_pressure_al_alt`
  (JAX `calc_p_rho_phi`) had **dropped the `rdnw*(ph'(k+1)-ph'(k))` geopotential term**
  and used base θ in the EOS. Restored the WRF form
  (`module_big_step_utilities_em.F:1029,:1083-1087`): `al` includes the geopotential
  term, EOS uses full θ.
- `src/gpuwrf/dynamics/core/acoustic.py` — comment update for the `p_for_buoy` fallback.
- `scripts/f7f_rwtend_after_fix.py` — AC1 + non-tautological AC2 (grid%p discriminator).

## Commands run (all `PYTHONPATH=src taskset -c 0-3`)
- `python -u scripts/f7d_rwtend_check.py` (pre-fix baseline: 0.6147 m/s², 9.40×, 744 Pa).
- `python -u scripts/f7f_rwtend_after_fix.py` (AC1/AC2).
- `python -u scripts/f7d_runaway_probe.py --case {warm_bubble,density_current}` (traces).
- `python -u scripts/f7a_oracles.py` (AC5 flat-rest/dipole/conservation).
- `python -m pytest tests/test_m6b1_… test_m6b4_… test_m6x_c2_pgf test_m6x_c2_acoustic test_m6x_vertical_acoustic_oracle` (AC5 subset; stash-bisected to prove the 2 m6b4 fails are pre-existing).
- `python -m gpuwrf.ic_generators.idealized --case all --proof-dir proofs/f7f` (AC3/AC4 verdicts+plots).

## Proof objects produced (`proofs/f7f/`)
- `rwtend_after_fix.json` — **AC1 PASS**, AC2 grid%p discriminator.
- `regression_recheck.json` — **AC5 PASS** (no new regression; 2 m6b4 fails pre-existing).
- `skamarock_warm_bubble.json` / `straka_density_current.json` (+`*_diagnostics.json`,
  `*_verdict.md`, `plots/`) — **AC3/AC4 honest FAIL** (still go non-finite).
- `ic_balance_proof.md` — WRF ph-rebalance derivation + the calc_p_rho_phi fix + before/after.
- `runaway_probe_*.json`, `runaway_pbuoy_warm.json` — supporting traces.

## Acceptance gate status
- **AC1 — frozen-buoyancy sanity: PASS.** Balanced IC: `max_abs(c1f·mu')=0`, direct
  stage-constant `pg_buoy_w` `max|rw_phys|=0.0 m/s²` (was 0.6147; 9.40×→0). The
  synthetic 9.4× over-forcing is gone.
- **AC2 — negative control: PASS (non-tautological).** The base-ph (no-rebalance) IC
  reproduces the historic `max|grid%p|≈750 Pa` artifact (≈ the 744 Pa the synthetic hack
  used), while the WRF-rebalanced IC gives a distinct, hydrostatically-consistent
  `grid%p≈1.51e3 Pa`; neutral base = 2.9e-11 Pa. The checker discriminates the two ICs.
- **AC3 — Skamarock warm bubble: FAIL.** No `0.615·t` linear runaway (was: 5.95 m/s@10s,
  NaN@80s; now max|w|~0.03@100s). But the thermal is weakly forced (centroid 2000→2027m,
  ~dead) and a top-boundary gravity-wave mode detonates ~190s.
- **AC4 — Straka density current: FAIL.** Cold bubble does not propagate
  (max|w|~0.001 at the bubble); a top-face mode (k=59/60) grows 0.06→3.5 m/s and goes
  NaN by ~30s. Front never forms.
- **AC5 — no regression: PASS.** flat-rest deltas exactly 0.0; analytic dipole
  w_abs_max=0.3243531513289849 (bit-identical to F7D); conservation drift=0.0; pytest
  21 pass / 2 fail (the 2 m6b4 self-compare-tautology fails are PRE-EXISTING — proven by
  stash-bisect, no test weakened/xfailed). The d02 operational-dt audit was not re-run
  (the balanced-ph/real-p_buoy change does not alter the d02 path; deferred).

## Unresolved risks / open decision (CRITICAL escalation signal)
The GPT-5.5 spec'd fix is verified-correct and eliminates the 9.4× over-forcing (AC1),
but the idealized cases still FAIL because of a deeper, pre-existing dycore issue the
spec did not name:

1. **Found and fixed** a real bug: `calc_p_rho_phi` dropped the `rdnw*(ph'(k+1)-ph'(k))`
   geopotential term, so the rebalanced bubble produced `grid%p=0` (dead). After the fix
   `grid%p=1.51e3 Pa`.
2. **Open**: feeding that correct `grid%p` into `pg_buoy_w` over-forces the other way
   (max|w|≈1.2·t, NaN@200s) because the in-solver `advance_w` buoyancy
   `c2a·alt·t_2ave − c1f·muave` does not subtract the same hydrostatic reference
   (`muave=0` when `mu'=0`), so `pg_buoy_w(grid%p)` double-counts the perturbation
   column weight. Diagnostically, `pg_buoy_w(grid%p)/analytic ≈ 19×` for the balanced
   bubble — i.e. the IC's ph'-rebalance (built via a midpoint-z hydrostatic integration)
   is **not discretely consistent** with the dycore's `calc_p_rho_phi` operator, so it is
   not in exact discrete hydrostatic balance and the two large terms do not cancel.

This is the binding open item: **reconcile the large-step `pg_buoy_w` pressure reference
with the small-step `advance_w` buoyancy reference, and balance the IC in the dycore's
own discrete operators** (not a separate midpoint-z integration). It is a
dycore-architecture coupling issue, not a coefficient/clamp — per the hard rules I did
NOT mask it. Recommend escalation: an empirical bisection of which discrete operator the
IC must be balanced against, plus a council review of the pg_buoy_w/advance_w buoyancy
split (the WRF design adds BOTH terms; their cancellation depends on shared references).

## Verdict
**F7F_PARTIAL.** AC1, AC2, AC5 PASS; AC3, AC4 FAIL (idealized cases still go
non-finite). Delivered: synthetic p_buoy removed, IC ph-rebalance made WRF-faithful,
real perturbation pressure in rk_addtend diagnostics, and a genuine `calc_p_rho_phi`
geopotential-term bug fixed. Precise residual + traces above and in `ic_balance_proof.md`.
