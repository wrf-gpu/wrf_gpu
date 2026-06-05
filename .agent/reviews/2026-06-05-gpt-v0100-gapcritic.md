# v0.10.0 Gap-Critic Review

Reviewer: GPT-5.5 xhigh / mandatory pre-release gap-analysis critic
Date: 2026-06-05
Branch: `worker/perf/v0100-kernel`
HEAD: `246a3dc1b24625124b830b849e39328c49f9ceaf`
Base checked: `016d993` (`v0.9.0`)

## Verdict

**SHIP.** I found no fix-now source blocker for tagging v0.10.0 as the optimized-kernel release. The final source delta is coherent and narrow: `git diff 016d993..HEAD -- src/` changes only `src/gpuwrf/dynamics/core/acoustic.py`, `src/gpuwrf/runtime/operational_mode.py`, and `src/gpuwrf/physics/thompson_column.py`. Wave-B3 wrapper changes are reverted in source; `src/gpuwrf/integration/daily_pipeline.py`, `src/gpuwrf/io/wrfout_writer.py`, and `src/gpuwrf/coupling/physics_couplers.py` have no net diff vs v0.9.0.

One documentation/proof caveat: `proofs/v0100/inefficiency_ledger.md` still contains stale prose saying Wave-B3 daily-wrapper changes shipped, and `proofs/v0100/v0100_release_d02_vs_v090.json` still contains a stale pre-complete-revert Q2 interpretation. The source diff and final two revert commits are decisive, so this is not a release blocker, but the README must not cite B3 as shipped.

## Completeness

The v0.10.0 scope is complete enough to ship. Every Wave-A/B lever in the super-plan scope is either removed, measured below the 1% gate, rejected for fidelity/bit-identity, or dispositioned as out of scope. The only shipped warmed-step gain is Thompson NSED16. Acoustic unroll default remains 1 after sub-1% coupled A/B; acoustic carry split is out; MYNN is dispositioned irreducible with profile and independent cross-check; fp32 is no-go; B3 daily-wrapper is reverted after <1% gain plus Q2 output change.

The inefficiency ledger is complete in coverage but stale in final B3 wording. Treat the B3 daily-wrapper entries as historical rejected evidence, not shipped release evidence.

## Correctness

The bit-identity composition is sound.

- Wave-A changes are value-preserving: `proofs/v0100/wave_a_gates.json` records idealized warm-bubble and Straka bit identity with `worst_reldiff=0.0`, and the shipped Wave-A edits are cast-skip, cqw reuse, concatenate replacements for edge pad/scatter construction, and an unroll hook defaulted to 1.
- Thompson NSED16 is proven equal to cap64 for the release trajectory: `wave_b1_nsed16_precip_oracle.json` has cap16-vs-cap64 surface precip bit-identical and zero clips; `wave_b1_nsed16_skill_24h.json` has T2/U10/V10 max delta 0; `wave_b1_nsed16_conservation.json` has qv/qc/qr/qi/qs/qg and RAINC/RAINNC bit-identical with zero precip/water deltas.
- Wave-B3 output-writer changes are reverted in the final source. Therefore forecast trajectory is v0.9.0-identical, and writer semantics are v0.9.0-identical by source diff, despite the stale Q2 proof file noted above.

The timing gain is supported: `proofs/v0100/wave_b1_nsed16_timing.json` compares fresh cap64 vs default cap16 processes, discards the first sample, and reports 74.2487 ms/step to 64.7622 ms/step: 12.7767% coupled gain, 1.14648x.

CPU pytest attempt: I ran `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src taskset -c 0-3 python -m pytest tests/ -q --ignore=tests/savepoint/test_dycore_100_steps.py --tb=short -ra`. It did not complete locally: historical backend/oracle build rows spawned long `nvc++` and `nvfortran` compiles that I terminated, then pytest stalled inside Python at 45% with no child process. Partial failures were in known backend/fixture/helper categories. Focused checks gave: `tests/test_async_wrfout_equiv.py` 3/3 pass; Thompson precip/tier2 checks pass with one GPU-required tier3 failure; source-contract checks show the existing v0.9.0-classified `test_m6b_fix_advance_mu_t_commit.py` string-test failures. I found **zero new real v0.10.0 failures**, but I cannot honestly mark the local CPU full-suite run as completed green.

## Efficiency

The WRF-faithful warmed-kernel floor claim is honest for this release scope. Thompson NSED16 is the only faithful >1% lever that survived the gates. MYNN/PBL is convincingly closed for v0.10.0: `wave_b2_mynn_profile.json`, `wave_b2_mynn_internal.json`, and `wave_b3_mynn_crosscheck.json` support the 95% closure-compute finding, with EDMF dependent vertical recurrence and XLA tridiagonal solves already on the faithful primitive path. The rejected MYNN fusion/unroll attempts were either <0.1%, negative, or not bit-identical.

I did not find a missed faithful >1% lever in the final three-file source delta. d03/L3 steep-terrain instability remains a v0.9.0 carry-over, not a v0.10.0 regression.

## Honest Framing

Approved README gain paragraph:

v0.10.0 is the optimized-kernel release. Relative to v0.9.0, it keeps the validated forecast and wrfout numerics unchanged while reducing Thompson sedimentation's faithful static substep cap from 64 to 16, a change proven cap16==cap64 on the precip oracle and 24h d02 hydrometeor, precipitation, and skill checks. The measured warmed coupled-step time improves from 74.25 ms to 64.76 ms on the L2 d02 kernel path, a 12.78% reduction (1.146x). Other candidate levers were below the 1% exit gate, negative, not bit-identical, or fidelity/precision-gated; based on the committed evidence, a 2x warmed speedup is not WRF-faithfully achievable in this release.

## Commands Run

- `git status --short --branch`
- `git diff --name-only 016d993..HEAD -- src`
- `git diff --stat 016d993..HEAD -- src`
- `git diff 016d993..HEAD -- src/gpuwrf/dynamics/core/acoustic.py src/gpuwrf/runtime/operational_mode.py src/gpuwrf/physics/thompson_column.py`
- `git diff 016d993..HEAD -- src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/io/wrfout_writer.py`
- `jq` reads of `proofs/v0100/wave_a_gates.json`, `wave_a_unroll_ab_verdict.json`, `wave_b1_nsed16_precip_oracle.json`, `wave_b1_nsed16_skill_24h.json`, `wave_b1_nsed16_conservation.json`, `wave_b1_nsed16_timing.json`, `thompson_nstep_histogram_graupel_wet.json`, `wave_b3_mynn_crosscheck.json`, `v0100_release_d02_vs_v090.json`, and `proofs/v090/release_trunk_greensuite.json`
- `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src taskset -c 0-3 python -m pytest tests/ -q --ignore=tests/savepoint/test_dycore_100_steps.py --tb=short -ra`
- `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src taskset -c 0-3 python -m pytest tests/test_async_wrfout_equiv.py -q --tb=short`
- Focused CPU pytest checks for operational/acoustic source contracts, Thompson precip/tier2/tier3, and selected unit tests

## Proof Objects Produced

- `.agent/reviews/2026-06-05-gpt-v0100-gapcritic.md`

## Unresolved Risks

- Full local CPU pytest did not complete; this is an environment/toolchain limitation of the historical suite in this sandboxed CPU-only run, not evidence of a v0.10.0 source regression.
- `proofs/v0100/inefficiency_ledger.md` and `proofs/v0100/v0100_release_d02_vs_v090.json` contain stale B3/Q2 wording after the final revert. The release README paragraph above avoids that overclaim.

## Next Decision Needed

Tag v0.10.0 with the README gain paragraph above. Optionally clean the stale B3 proof prose in a follow-up documentation-only commit, but do not re-open kernel scope for it.
