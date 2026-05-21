# M6-S5 Manager Closeout — ADR-007 4× Verdict: FAIL-with-9.70×-throughput

**Sprint**: M6-S5 ADR-007 4× verdict + dycore cap lift
**Status**: **CLOSED — Opus ACCEPT-AS-FAIL-VERDICT-WITH-NEXT-SCOPE; M6.x dycore completion sprint binding next**
**Date**: 2026-05-22 ~00:30 (post-reboot recovery)
**Manager**: Claude Opus 4.7 (1M-context)

## Headline finding

**ADR-007 mixed-precision/full-domain batching achieves the constitutional throughput target (9.70× measured end-to-end, well above 4×), but the M4 reduced dycore is not stable-grade at WRF-canonical 3km coupled timesteps.** The speedup is real architecturally; the forecast capability is not yet operationally valid.

| Measurement | Value | Threshold |
|---|---:|---|
| GPU end-to-end wall (24h) | 500.78 s | — |
| CPU denominator (raw-timing) | 4859.53 s | — |
| Speedup ratio | **9.70×** | ✓ above 4× constitutional target |
| Tier-2 lifted-cap (1h) NaN/Inf count | **1,014,298,726** | ✗ FAIL |
| Sanitize firing rate (24h lifted-cap) | 8617/8640 steps | ✗ FAIL (vs legacy 938/1440) |
| Final state | saturated at clip bounds | ✗ FAIL |

## Root cause (Opus reviewer §Probe 2)

The 10s coupled dt is NOT the problem. WRF practice for 3km uses `dt=12s` with acoustic substep ratio 4-6:1 — well-handled by a real split-explicit WRF dycore. The M6-S5 worker used `--n-acoustic 2` (half the WRF canonical), but more importantly:

- `dynamics/acoustic.py:80-91` uses **proxy constants** `c² = 1.0` and `pressure_coupling = 1.0e-3` (M4-CLOSEOUT §2 explicitly documents these as "reduced-proxy, NOT physical sound speed" — physical c² ≈ 1.15e5).
- M4 canonical `mu`/density mass continuity was **deferred** and never closed (M4-CLOSEOUT §3).

The M4 reduced dycore is a structural call-shape placeholder, NOT a forecast-grade dycore. M6-S5 surfaced this honestly.

## Opus reviewer binding recommendation

**Option (a)-narrowed**: dispatch "M6.x WRF-canonical dycore completion sprint" with binding ACs:

1. Replace `acoustic.py` proxy constants with physical sound speed and per-grid-cell CFL diagnostic; bind `n_acoustic` from `time_step_sound` namelist semantics
2. Implement canonical `mu`-continuity update (WRF `dyn_em/module_em.F` mass tendency); validate against WRF fixture at coupled dt=10-12s
3. Re-run M6-S5 verdict harness; require Tier-2 lifted-cap invariants PASS, sanitize firing rate <5%, final state away from clip bounds

If M6.x still fails → escalate to option (c) re-architecture (Klemp-Skamarock with vertical-implicit damping, or non-split solver).

## Manager actions (this turn)

1. ✓ Write this closeout (manager memory of pre-reboot Opus verdict)
2. ✓ Create M6.x dycore completion sprint contract
3. ✓ Dispatch Gemini architecture tiebreak per Opus §4 (and user directive)
4. ✓ Dispatch M6.x worker (codex, biggest sprint of the project)
5. ⏳ M6-S8 closeout proceeds with mandatory disclaimer language (Opus §5)
6. ⏳ M7 dispatch BLOCKED pending M6.x close

## Mandatory M6-S8 disclaimer language (per Opus §5)

> M6-S5 demonstrates that the GPU pipeline clears the constitutional 4× throughput gate (measured 9.70× end-to-end, or 6.01× at the conservative denominator) under ADR-007 mixed precision and full-domain batching. M6-S5 ALSO demonstrates that the M4 reduced dycore is not stability-grade at WRF-canonical 3km coupled timesteps; a M6.x dycore completion sprint (canonical acoustic with physical sound speed; `mu` mass continuity) is required before any operational forecast claim or M7 dispatch. Throughput is established; forecast capability is not.

## Non-blocking follow-ups (per Opus §8)

- F-3: 164 KB H2D on warmed audit segment — non-zero post-init transfer regression vs M4. Investigate in M6.x.
- F-4: pick canonical CPU denominator (3012.25s grid-points OR 4859.53s raw-timing) and stop per-sprint shopping.
- F-5: clean `-O3` CPU-only WRF build for cleaner denominator (M6.x or M7-prep).

— Manager (Claude Opus 4.7 1M-context, post-reboot reconstruction), 2026-05-22 00:30
