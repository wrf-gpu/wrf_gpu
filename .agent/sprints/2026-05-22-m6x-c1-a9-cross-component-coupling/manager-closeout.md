# c1-A9 Manager Closeout — Pair-Probe Semantics Wrong + Map-Factor Gap Surfaced

**Sprint**: c1-A9 Cross-Component Momentum Coupling Bisect
**Status**: **CLOSED — c1-A8 pair-probe semantics WRONG; full momentum step 188 residual implicates missing WRF map-factor support**
**Date**: 2026-05-22 ~05:50
**Worker**: codex gpt-5.5 xhigh

## Two important findings

### 1. c1-A8 pair-probe semantics were instrumentation-incorrect

c1-A8's `--disable-advection-u/v` flags only disabled the tendency, NOT the field as advecting velocity. So "u alone stable" reports were misleading — the `v` field was still actually advecting throughout.

c1-A9 added proper cross-term flags:
- `--disable-advection-u-vertical-by-w`
- `--disable-advection-v-vertical-by-w`
- `--disable-advection-w-horizontal-by-u`
- `--disable-advection-w-horizontal-by-v`

With these TRUE flags, pair probes (u+w with v fully removed) ARE stable through 360 steps.

### 2. Real residual: missing WRF map-factor support

c1-A9 audit found WRF `advect_v` vertical branch uses `msfvy/msfvx` map-factor correction (`module_advect_em.F:2813-2920`). c1's reduced `GridSpec` has no map-factor support. The full momentum coupling fails at step 188-190 because the map-factor-coupled cross-terms aren't represented.

Worker tried 2 WRF mass-flux-first variants — neither closed; one worsened (step 188 → 134).

## Worker recommendation

> "Manager/user should escalate before further model edits. The next likely decision is whether to add map-factor fields / ADR-backed metric support to the reduced grid, or to open a new sprint on full u/v/w kinetic-energy/reflexivity diagnostics with an explicit WRF fixture or analytic energy oracle."

## Manager response

**Dispatched Option B parallel test** (window 0:22): integration test of c1-A8 HEAD + sanitize ON + 24h Tier-4 RMSE vs Gen2. Per `[[feedback_validation_philosophy]]` (operational RMSE > bitwise parity), this could close M6.x with documented sanitize-firing as a known limitation.

**c1-A10 (map-factor extension) NOT dispatched** — would extend State pytree + break ADR-002 baseline; requires user authorization.

## Decision logic for user wake-up

| Option B result | Action |
|---|---|
| PASS (RMSE ≤ 1.5× Gen2-AIFS) | M6.x closes-with-known-issue; M6-S8 + M7-S0 unblock; map-factor extension queued as quality follow-up |
| PARTIAL (1.5-2.0×) | Surface to user; they decide accept vs continue |
| FAIL (>2.0×) | Continue c1 with map-factor extension (user authorization needed) OR escalate to c2 |

## Cumulative c1 state

- Acoustic: STABLE in isolation
- Advection scalar: mass-conservative flux form (c1-A2)
- Advection horizontal momentum: flux form (c1-A7) ✅
- Advection vertical w-momentum: eta-metric sign fixed (c1-A8) ✅
- Cross-component (u+w, v+w) WITH fixed flags: actually stable
- Full momentum coupled: step 188 residual due to missing map-factor support
- 1h coupled sanitize: 86.9%
- **Speedup: 44.33×** (well above 4× constitutional target)

## Branch state

- `worker/codex/m6x-c1-a9-cross-component` at `ae6921c` (instrumentation improvements + audit findings; NO production code change retained)
- All worker reports + closeouts committed to main

## What's actually working

The c1 dycore has substantial correct infrastructure:
- Klemp-Skamarock acoustic core (verified)
- Tridiagonal Thomas solver
- Mass-conservative scalar advection
- Mass-conservative momentum advection (horizontal + vertical-isolated)
- 44× speedup

What's missing:
- Map-factor support for fully-conserving cross-component momentum coupling
- (Possibly) WRF hybrid-eta coordinate (c1h/c2h/c1f/c2f) — not yet investigated

Both are substantive code additions, not surgical fixes.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 ~05:50
