# v0.17 FP32 Memory/Compute Re-Audit

Status: independent audit, no new GPU run.  Sources are the v0.16/v0.17 proof
artifacts plus static code/shape review.  Machine-readable proof object:
`proofs/perf/v017/fp32_memory_compute_reaudit.json`.

## Verdict

The tested v0.17 fp32-dycore path did not test the memory architecture needed
for a 2x VRAM win.  It narrows part of the inner `AcousticCoreState`, but then
widens the outer timestep carry back to the original fp64 contract and keeps
`State` in the legacy total/alias representation.  Therefore the observed
0-2% VRAM delta is expected, not surprising.

The compute blocker is mostly real for the tested approaches, but it should be
stated more narrowly: launch packaging, CUDA graphs, and broad Pallas fusion
are falsified as clean 2-4x small-grid wins.  A hidden improvement is still
possible only if it removes actual device work or actual live bytes.  The best
remaining suspect is the same as the memory suspect: XLA liveness/materialized
fp64 total/base temporaries, not raw fp32 arithmetic throughput.

## What I Verified/Falsified

1. `FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md` is right that cancellation must be
   handled by representation, not by hoping fp32 stores a 1e5 absolute with
   enough low bits.  It is wrong or at least stale where it frames fp32 as an
   immediate VRAM unlock.  Double-single/EFT uses two fp32 words for one value:
   it can recover precision, but for that value it is 8 bytes again.  It is a
   local numerical tool, not a general memory-halving tool.

2. The v0.17 fp32-dycore code explicitly re-widens the finished stage to the
   reference carry dtype (`operational_mode.py:251-285`, `:2164-2168`).  The
   comment even says the persistent carry/State deliberately stays fp64.

3. The code still reconstructs base fields from total-minus-perturbation inside
   hot paths:
   - `p_base = state.p_total - state.p_perturbation`
   - `ph_base = state.ph_total - state.ph_perturbation`
   - `mu_base = state.mu_total - state.mu_perturbation`

   The audit found 5 `p_total-p_perturbation`, 8 `ph_total-ph_perturbation`,
   4 `mu_total-mu_perturbation`, 61 explicit fp64 casts, and dozens of
   total-field references in the reviewed dycore/runtime files.

4. `small_step_finish_wrf` returns the legacy representation again
   (`p`, `p_total`, `p_perturbation`, same for `ph` and `mu`).  So even if the
   inner acoustic loop is temporarily fp32, the outer state is not compact.

5. v016 HLO already showed the decisive memory fact at 65k columns:

| HLO field | fp64 GiB | mixed GiB | mixed/fp64 |
|---|---:|---:|---:|
| argument | 1.043 | 0.756 | 0.725 |
| output | 0.983 | 0.696 | 0.708 |
| alias | 0.983 | 0.696 | 0.708 |
| temp | 7.658 | 7.570 | 0.989 |

   The persistent args/outputs moved, but the transient temp arena did not.
   Therefore peak VRAM cannot fall by 2x from this change.

## Why fp32 Did Not Save VRAM

There are three independent blockers.

1. The actual v0.17 lever is an inner island, not persistent fp32 storage.
   Measured:

| ncol | speedup | fp64 peak GiB | fp32dyc peak GiB | fp64/fp32 VRAM |
|---:|---:|---:|---:|---:|
| 16,384 | 1.043x | 4.145 | 4.050 | 1.024x |
| 65,536 | 0.996x | 12.025 | 12.082 | 0.995x |
| 147,456 | OOM/missing | - | - | - |

   The debug carry-split variant also did not move peak VRAM meaningfully
   (1.006x at 16k, 1.001x at 65k).

2. The current `State` is already mixed precision for many fields.  At 147k
   columns, the source-shape audit estimates:

| representation | State GiB | ratio vs current State |
|---|---:|---:|
| all float64 State | 1.453 | 1.509 |
| current precision matrix State | 0.963 | 1.000 |
| drop `p/ph/mu` legacy+total aliases only | 0.765 | 0.795 |
| drop aliases + dynamic perturbations fp32 target | 0.691 | 0.718 |
| explicit fp64 `BaseState` needed somewhere | 0.196 | 0.203 |

   Removing total/alias fields and adding an explicit base state is almost
   neutral in resident bytes (`0.765 + 0.196 ~= 0.961 GiB`).  Its value is not
   raw storage reduction; its value is shrinking the high-frequency scan carry
   and preventing XLA from materializing many fp64 totals at once.

3. Peak is transient/liveness dominated.  The 65k HLO temp arena is already
   7.57 GiB.  At 147k the measured BouLac-ONZ peak is 18.57 GiB, consistent
   with a large live transient arena.  Saving 0.27 GiB in persistent State
   cannot halve an 18 GiB peak.

## What Would Be Required For a Real FP32 Memory Win

A real attempt must change the state/carry contract, not just add casts inside
the acoustic loop.

1. Make `BaseState(pb, phb, mub, t0, theta_base)` authoritative and fp64,
   resident outside the high-frequency timestep scan.

2. Introduce a compact dynamic carry that does not contain `p`, `p_total`,
   `ph`, `ph_total`, `mu`, or `mu_total`.  It should carry only perturbations
   and dynamic fields.

3. Store validated dynamic perturbations in fp32.  Reconstruct totals only
   inside local fp64 islands and never carry those totals across the outer scan.

4. Delete the fp32->fp64 outer restore boundary for the compact path.  The scan
   carry dtype invariant must be the compact dtype invariant, not the old fp64
   invariant.

5. Rewrite hot equations to consume `(base_fp64, perturb_fp32)` directly:

   `grad(base + pert) = grad(base) + grad(pert)`

   and equivalent pressure/geopotential/mass identities.  The point is to avoid
   `total - perturbation` and avoid materializing full-grid fp64 totals.

6. Use double-single or compensated summation only at the few cancellation or
   long-accumulation sites that still fail.  Do not use double-single as a
   general storage format, because it is two fp32 arrays and therefore gives no
   memory win for that value.

7. Gate the work with HLO memory analysis before any long stability campaign:
   `argument`, `output`, and especially `temp_size_in_bytes` must all drop.
   A path that reduces only args/outputs but leaves temp unchanged is another
   dead end.

## Compute Re-Audit

The old Phase-R interpretation "35 ms launch gap implies roughly 3x structural
headroom" is falsified as a clean wall-clock claim.

Evidence:

| test | result |
|---|---|
| R2 operator fusion | launch count collapsed, wall time did not improve |
| R4 CUDA graphs | safe but slower |
| no-op launch flood, 5281 raw launches | 6.23 ms median, not 35 ms |
| no-op launch flood, 5281 graph nodes | 1.99 ms median |

So launch overhead exists, but it is not the hidden 2-4x blocker.  The profiler
"gap" included dependency, scheduling, or device-work effects that are not
recoverable by wrapping the same work differently.

The remaining plausible hidden compute blocker is excessive live/materialized
work: repeated total reconstruction, fp64 island promotion, conversion churn,
and temporary arrays that turn a theoretically compact perturbation algorithm
into a large fp64 HLO program.  If a compact-state rewrite truly drops device
bytes and temp liveness, it can also improve wall clock.  If it only changes
storage dtypes while XLA still builds the same temp arena, wall clock will stay
near 1.0x.

## Realistic Estimates By Path

These are bounded estimates from the audited measurements, not guesses from the
design docs.

| path | expected VRAM | expected wall-clock | reason |
|---|---:|---:|---|
| current v0.17 fp32 dycore island | 0-3% better, sometimes worse | 0-5% | measured 1.024x VRAM at 16k, 0.995x at 65k; outer carry fp64 |
| carry-split/debug scatter-only tweaks | ~0% | -2% to +2% | measured essentially unchanged VRAM and slower/noisy speed |
| launch packaging / CUDA graphs only | ~0% | -10% to +10% | R2/R4/flood falsify 2-4x clean win |
| remove `p/ph/mu` aliases from high-frequency carry, base outside scan | peak maybe 0-10%; arg/out better | 0-10% | resident bytes mostly move from carry to base; useful only if HLO temp drops |
| compact perturbation State + no outer fp64 restore + local fp64 islands | 10-35% plausible peak reduction; >35% only if temp liveness collapses | 5-25% small grid, 10-50% large memory-bound grid | source State target is 0.718x current; HLO temp must follow for real peak |
| general double-single dycore | no memory win for DD values | likely slower | DD is two fp32 words plus more ops; use only surgically |
| algorithmic shape fixes like BouLac O(nz) | large, proven | neutral/slightly slower possible | d03 dense 3.127 GiB -> ONZ 1.732 GiB; bit-identical |

For a full 2x VRAM win from current v0.17, a compact rewrite must reduce the
large transient arena, not just the State leaves.  At 65k, halving only the
visible State cannot matter against a 7.57 GiB temp arena.  Halving the temp
arena itself would be a real breakthrough; the current fp32-dycore path did not
do that.

For a 2-4x wall-clock win on small grids, I would not bet on precision or
launch structure.  The path would need a demonstrable reduction in actual
device work/bytes, visible in Nsight as lower DRAM/L2 traffic or fewer heavy
HLO fusions, not just fewer launches.

## Recommended Next Proof Sprint

Do not start with stability.  First prove the memory model.

1. Build a compile-only `CompactState` prototype for one RK stage or one
   acoustic stage.  It may be numerically incomplete, but it must remove
   `p_total/ph_total/mu_total` from the carry.

2. Lower both old and compact paths and record:
   - `argument_size_in_bytes`
   - `output_size_in_bytes`
   - `alias_size_in_bytes`
   - `temp_size_in_bytes`
   - HLO dtype tokens and convert counts

3. Pass gate only if temp drops by at least 20-30% at 65k.  If only
   args/outputs drop, stop: it is another elegant dead end.

4. If the compile-only gate passes, run one small GPU timing with the lock and
   collect a transfer audit plus Nsight bytes.  The wall-clock estimate should
   be updated from measured byte reduction, not from fp32 theory.

5. Only after those gates start the expensive numerical stability campaign.

## Layer Answer

This is primarily a JAX/source representation problem, not an assembler problem.
The compiler cannot safely infer that a carried `p_total` plus
`p_perturbation` should be split, packed, rematerialized, or algebraically
reassociated without changing observable IEEE behavior.  We must express the
compact variables in the source-level state contract first.

XLA/compiler work may become relevant after that, specifically for liveness,
rematerialization, and preserving compensated/EFT sequences.  PTX/SASS is a
later optimization layer only if the compact HLO proof passes but generated
code is bad.  Changing registers directly cannot make one fp32 value preserve
both a 1e5 absolute and 1e-2 increments; that requires a different
representation.
