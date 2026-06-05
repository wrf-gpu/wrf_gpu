# Release Notes — v0.10.0

- **Tag:** `v0.10.0`
- **Release commit:** resolve with `git rev-parse v0.10.0^{commit}` after the annotated tag is created. Source-freeze basis reviewed by the gap-critic: `246a3dc` (Wave-B3 complete revert; source frozen). Mandatory gap-critic record is committed at `563599e`.
- **Tag date:** 2026-06-05.
- **Release gate:** mandatory cross-model pre-release **gap-analysis critic** (GPT-5.5 xhigh) -> **verdict SHIP, 0 fix-now source blockers** ([`.agent/reviews/2026-06-05-gpt-v0100-gapcritic.md`](.agent/reviews/2026-06-05-gpt-v0100-gapcritic.md)).
- **Binding numbers:** every performance or fidelity figure below traces to committed proof objects under [`proofs/v0100/`](proofs/v0100/) and the v0.9.0 baseline proofs. Nothing is rounded upward, invented, or relaxed to manufacture a pass.

## What v0.10.0 is

v0.10.0 is the **optimized-kernel release** on top of v0.9.0. It keeps the validated
forecast trajectory and wrfout semantics unchanged relative to v0.9.0, while shipping
one faithful performance change: Thompson sedimentation's static substep cap defaults
from `NSED_MAX=64` to `NSED_MAX=16`.

The final net source delta vs `v0.9.0` is intentionally narrow:

- `src/gpuwrf/dynamics/core/acoustic.py` and `src/gpuwrf/runtime/operational_mode.py`:
  Wave-A bit-identical cleanups / source hooks with no release throughput claim.
- `src/gpuwrf/physics/thompson_column.py`: Thompson faithful sedimentation cap
  `64 -> 16`, with the env override retained.

Wave-B3 daily-wrapper source changes were reverted before release; `daily_pipeline.py`,
`wrfout_writer.py`, and `physics_couplers.py` have no net source diff vs v0.9.0.

## Honest gain

Thompson NSED16 is proven cap16==cap64 on the precip oracle and 24 h d02
hydrometeor/precip/skill checks:
[`wave_b1_nsed16_precip_oracle.json`](proofs/v0100/wave_b1_nsed16_precip_oracle.json),
[`wave_b1_nsed16_skill_24h.json`](proofs/v0100/wave_b1_nsed16_skill_24h.json), and
[`wave_b1_nsed16_conservation.json`](proofs/v0100/wave_b1_nsed16_conservation.json).

The warmed coupled d02 step improves **74.25 -> 64.76 ms**, a **12.78% reduction**
(**1.146x**) in [`wave_b1_nsed16_timing.json`](proofs/v0100/wave_b1_nsed16_timing.json).
Applied to the v0.9.0 conservative warm real-user d02 ratio, the end-to-end
real-user speedup vs 28-rank CPU-WRF rises from **~2.16x to ~2.47x warm**.

Kept separate: OC-A / compute-only ceiling numbers remain kernel-level context, not
the release headline. v0.10.0's release headline is the warm real-user estimate above
plus the measured warmed coupled-step proof.

## Bit identity

The release is **bit-identical-output to v0.9.0** on the validated forecast trajectory
and final wrfout semantics:

- Wave-A changes are value-preserving and gate bit-identical on idealized checks
  (`worst_reldiff=0.0`) in [`wave_a_gates.json`](proofs/v0100/wave_a_gates.json).
- Thompson NSED16 is bit-identical to cap64 for the release d02 trajectory and
  hydrometeor/precip fields, with zero precip/water deltas.
- Wave-B3 writer/output changes were reverted after the proof run that changed Q2
  output semantics; the final release writer is v0.9.0 source-equivalent.

Note: [`proofs/v0100/v0100_release_d02_vs_v090.json`](proofs/v0100/v0100_release_d02_vs_v090.json)
contains a stale pre-complete-revert Q2 interpretation. Treat it as historical
rejected evidence for B3, not as shipped v0.10.0 output evidence; see
[`proofs/v0100/v0100_release_d02_vs_v090_POST_REVERT_NOTE.md`](proofs/v0100/v0100_release_d02_vs_v090_POST_REVERT_NOTE.md).

## Lever dispositions

| Lever | Disposition | Evidence |
|---|---|---|
| Thompson NSED cap | **SHIPPED** | `NSED_MAX=16`; cap16==cap64 on precip oracle, 24 h d02 skill, hydrometeors, precip, and conservation; **12.78% / 1.146x** warmed coupled-step gain. |
| Acoustic unroll / fusion | **No release speed claim** | Hook added, default kept `1`; unroll>1 measured below the 1% exit gate on the coupled path and carried higher compile cost. |
| Acoustic carry split | **REVERTED / no-go for v0.10.0** | Bit-identical in idealized tests but warmed A/B was confounded and no clean benefit was proven. |
| Gated-fp32 | **NO-GO** | Current coupled path measured ~0% / negative while launch- and memory-bound; d03 1 km gated-fp32 remains unstable. |
| MYNN/PBL restructuring | **IRREDUCIBLE / no source change** | Profile and independent cross-check show ~95% genuine dependent closure compute; fusion/unroll attempts were <0.1%, negative, or not bit-identical. |
| Daily-wrapper / Wave-B3 | **REVERTED** | Historical B3 branch measured only **0.848%** daily-hour gain and changed Q2 output semantics; final source restores v0.9.0 writer behavior. |

## Faithful floor

On the committed evidence, a **2x warmed speedup is not WRF-faithfully achievable in
this release**. Every v0.10.0 Wave-A/B lever is shipped, rejected, reverted, or below
the 1% exit gate. The remaining launch/occupancy headroom requires a hand-fused-kernel
rewrite or equivalent architecture branch, not another low-risk release tweak.

## Known issues / carried forward

Full write-up: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). The current open issues
are carried forward unchanged by v0.10.0's bit-identical forecast/wrfout release:

- **d03 1 km gated-fp32 dynamics/qke instability:** still non-finite after forecast
  hour 1; qke->fp64 falsified the simple precision-overflow diagnosis; full fp64 is
  only short-window finite.
- **Autonomous long single-call daily-pipeline qke edge:** case-sensitive robustness
  issue on susceptible initial states; supported segmented d02 cadence remains the
  validated path.
- **Operational writer scope:** focused 64-variable wrfout remains the release
  contract; Wave-B3 writer-wrapper work is reverted and not shipped.
- **d02 U10 episodic residual:** v0.10.0 inherits the v0.9.0 bit-identical d02
  trajectory, including the documented 6/72 U10 evening-peak breaches.
- **Scope boundaries:** flat-slab diffusion, fail-closed unported schemes, and no
  scored n=15 TOST equivalence.

## What v0.10.0 does NOT claim

No powered TOST PASS; no d03 1 km validated path; no fp32 release path; no full WRF v4
physics catalog; no 2x warmed-kernel speedup; and no daily-wrapper B3 output change.
