# Manager Closeout Draft — c2-A1 JAX/WRF Dycore Architecture

Date: 2026-05-22
Author: worker-generated decision-gate draft

## Recommendation

Merge Decision: do not merge to main from this worker alone; send to mandatory reviewer/tester. Technical decision-gate recommendation is continue C implementation.

Continue C implementation.

The architecture is compatible with JAX/XLA representation: WRF map factors and hybrid coefficients load from the Gen2 `wrfinput_d02`, analytic fixtures run through JIT-safe helpers, disabled stabilizers are identity by default, the limiter preserves nonnegative scalar mass on an analytic fixture, and the outer/nested scan carry executes on GPU without host-callback primitives.

## Evidence

- AC1: `architecture.md`, ADR-002 patch proposal, and ADR-020 proposal created.
- AC2: `proofs/metrics.json` loads analytic flat metrics and WRF fixture map factors.
- AC3: `proofs/hybrid_eta.json` loads WRF `c1h/c2h/c3h/c4h/c1f/c2f/c3f/c4f/dn/dnw/rdn/rdnw` and passes analytic pressure oracle with max error 0.0.
- AC4: `pytest` stabilizer tests pass disabled identity and enabled finite-effect checks.
- AC5: `proofs/scan_transfer_audit.md` records outer and nested `lax.scan`, final carry leaves on GPU, and no host-callback primitives.
- AC6: `proofs/limiter_conservation.json` reports nonnegative output and relative mass error 0.0.
- AC7: `proofs/integration_warm_bubble.json` is finite analytic smoke only.

## Blocking Caveat

AC7 is not a real warm-bubble validation. The role prompt referenced `scripts/m6_warm_bubble_test.py`, but that file is absent in this worktree. Do not claim c2 physical stability or warm-bubble improvement from this closeout.

## Decision

Proceed to c2-A2 only as an implementation sprint with a first task to restore or rebuild the warm-bubble proof harness. Do not narrow to B throughput-only and do not rollback the architecture based on c2-A1 evidence.

## Next Sprint Gate

c2-A2 should not expand into real-case coupled forecast work until it has:

- WRF small-step pressure/geopotential/mu coupling implemented in `acoustic_wrf.py`
- warm-bubble harness present and committed
- stabilizer progression report: off, smdiv, hyperdiffusion, Rayleigh, limiter
- updated transfer audit after real acoustic math is inside the nested scan
