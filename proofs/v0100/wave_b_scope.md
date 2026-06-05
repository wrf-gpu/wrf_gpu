# v0.10.0 Wave-B Scope

## Outcome

Wave-A validation: **PASS** on behavior. The five shipped changes are
bit-identical/default-preserving and the carry-split is cleanly out of the active
path. I found only documentation cleanup issues: the unroll comment still says
the default will be bumped to 2, and the carry-split comment overstates an older
timing result that the ledger later qualified as compile-artifact-confounded.

Fresh validation:
- Idealized warm-bubble + Straka rerun PASS, bit-identical to Wave-A `u1`
  scalar verdicts (`worst_reldiff=0.0`):
  `proofs/v0100/wave_b_idealized_roundoff.json`.
- Short L2 d02 coupled finite check PASS:
  `proofs/v0100/wave_b_coupled_spot_fp64.json`, `75.91 ms/step`, finite and
  physically plausible.

## Corrected Bottleneck

Fresh 240-step cache-hit baseline on L2 d02, guards ON, fp64, unroll=1:
`74.35 ms/step` (`proofs/v0100/wave_b_timing_full_fp64_u1.json`). The first
repeat was a one-off cache miss (`551 ms/step`) and was discarded.

The step is **not acoustic-launch-bound**:
- unroll=2: `73.85 ms/step`, only `0.67%` faster, compile `136.0s -> 176.0s`.
- Keep acoustic unroll default at `1`.

Corrected in-context toggle breakdown:

| Component | Evidence | Delta |
|---|---:|---:|
| MYNN/PBL | full - no_pbl | `33.84 ms` / `45.5%` |
| Thompson | full - no_thompson | `20.71 ms` / `27.9%` |
| Dycore floor | dycore_only | `16.59 ms` / `22.3%` |
| Other physics/boundary residual | remainder | `3.21 ms` / `4.3%` |

This refutes the old acoustic-first plan and also corrects the Wave-A shorthand:
the largest measured share is now MYNN/PBL, with Thompson still a large concrete
lever.

## Precision

Gated-fp32 does **not** pay now:
- fp64: `74.35 ms/step`
- gated-fp32 (`force_fp64=False`): `74.57 ms/step`
- gain: `-0.30%`, finite/physical short run

Verdict: **NO-GO for Wave-B**. Also note the gated-fp32 trace emitted a JAX
scatter dtype warning (`float64` value into `float32`), so precision boundaries
still need cleanup before another precision sprint.

## Thompson Phase-4

Wet graupel evidence was gathered from the local WRF corpus:
`proofs/v0100/graupel_wet_candidates.json`.

Histogram on the top 8 graupel-wet files:
`proofs/v0100/thompson_nstep_histogram_graupel_wet.json`.

Merged wet columns:
- graupel: `2493` wet columns, max/P99/P99.9 nstep = `1/1/1`
- rain: `9122` wet columns, max/P99/P99.9 nstep = `2/2/2`
- ice/snow: max nstep = `1`
- clip counts at `NSED_MAX={16,32,48,64}`: all zero

Measured config-only A/B with `GPUWRF_THOMPSON_NSED=16`:
- baseline: `74.35 ms/step`
- `NSED=16`: `64.73 ms/step`
- gain: `9.62 ms`, `12.94%`, `1.149x`
- finite/physical short run

Verdict: **GO after precip-oracle + 24h skill/conservation gate**. This is the
top immediate implementation lever.

## Wave-B Scope

1. **Thompson `NSED_MAX=16` default, gated by precip oracle**
   - Measured coupled gain: `12.9%`.
   - Effort: S/M. Risk: medium fidelity, low/medium kernel stability.
   - Gate: wet histogram retained, precipitating Thompson oracle, d02 24h
     skill, water/conservation budget, zero clips.

2. **MYNN/PBL internal profile and restructuring plan**
   - Measured share: `33.84 ms` / `45.5%`.
   - Estimated gain: `11-23%` if 25-50% of the wall can be removed.
   - Effort: L. Risk: high kernel and fidelity risk.
   - Gate: first profile internals; then WRF MYNN/surface parity or diagnostic
     equivalence, 24h skill/conservation, no in-loop transfers.

3. **M9/RRTMG diagnostic reuse**
   - Existing Wave-A host proof: M9 diagnostics `2.99s` in a `30.36s` forecast
     hour; non-forecast host share `10.95%`.
   - Gain is daily wall, not warmed timestep wall.

Do **not** spend Wave-B on gated-fp32 or acoustic unroll default changes.

## Honest Ceiling

Current corrected warmed step: `74.35 ms`.

Measured immediate ceiling with only Thompson `NSED=16`: `64.73 ms` (`1.15x`).
Conservative Wave-B ceiling if `NSED=16` ships and MYNN/PBL removes 25-33% of
its measured wall: roughly `53-56 ms/step` (`1.32-1.40x`). Aggressive if about
half of MYNN/PBL is removed: `48-50 ms/step` (`1.49-1.55x`).

A 2x warmed coupled speedup is **not supported** by current evidence without a
high-risk MYNN/PBL rewrite and further architecture work.

## Commands Run

- `git diff 016d993..HEAD -- src/gpuwrf/runtime/operational_mode.py src/gpuwrf/dynamics/core/acoustic.py`
- `python -m gpuwrf.ic_generators.idealized --case all --proof-dir proofs/v0100/wave_b_idealized_spot`
- `python proofs/v0100/compare_idealized_snapshots.py proofs/v0100/idealized_after_u1 proofs/v0100/wave_b_idealized_spot --rtol 1e-12 --out proofs/v0100/wave_b_idealized_roundoff.json`
- `python proofs/v0100/wave_a_gate.py --hours 0.25 --out wave_b_coupled_spot_fp64.json`
- `python proofs/v0100/wave_b_timing_probe.py ...` for full fp64, gated-fp32, unroll=2, no_thompson, no_pbl, dycore_only, and `GPUWRF_THOMPSON_NSED=16`
- `python proofs/v0100/find_graupel_wet_wrfouts.py /mnt/data/canairy_meteo/runs --max-files 5000`
- `JAX_PLATFORMS=cpu python proofs/v0100/thompson_nstep_histogram.py --dt-s 10 --caps 16,32,48,64 ...`

## Unresolved Risks

- `NSED=16` still needs precip-oracle and 24h skill/conservation proof before
  becoming default.
- MYNN/PBL is now the largest measured share, but no implementation-level
  subprofile was produced in this pass.
- Gated-fp32 was finite short-run only and slower; do not treat it as a stability
  proof for production mixed precision.
