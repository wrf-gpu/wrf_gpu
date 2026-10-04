# v0.22 GPU Openers Synthesis

Date: 2026-06-26
Worktree: `<WORKTREE>`

## Executive Verdict

K2 is the clear win: on the Switzerland-128 single-domain 2h window, R4 (`dt=18`, `n_sound=7`) cuts forecast-hour wall from `34.67` to `19.25 s/fc-h`, a `1.80x` speedup (`-44.5%`). This is not a per-step speedup; per-step cost is flat by construction.

K1 remains pathological for O(nz) BouLac as a full-pipeline default. The fixed harness shows dense compiles cleanly, so the O(nz) failure is real and specific to the data-dependent first-crossing search.

R0 memory-fitting is no longer the right question for 3-domain fused compile RAM. On the sound harness, fused 3-domain cold compile peaked at `22.32 GiB` RSS, not the old roughly `60 GiB` premise. The practical issue is compile time, and the AOT blob path is validated.

## K2 dt/n_sound Ladder

Verdict: `PASS` for the short single-domain gate. Best rung: `R4`, `dt=18`, `n_sound=7`.

Key results:

- R1 baseline: `34.67 s/fc-h`, `0.09631 s/step`.
- R4: `19.25 s/fc-h`, `0.09624 s/step`.
- Forecast-hour speedup: `1.80x`; per-step ratio: `0.999x`.
- R4 operational tolerance vs R1 is green: `T2 RMSE 0.046 K`, `U10 0.150 m/s`, `V10 0.137 m/s`, `PSFC 30.3 Pa`.
- R4 finite/bounded/conserving: mass drift max `4.50e-4`; worst RMSE/band `0.505`.

Caveats:

- This was a 2h single-domain Switzerland-128 run only. Full K2 still needs 24h/72h skill and multi-day stability; PSFC bias growth around `29 Pa` should be watched.
- This did not test the real operational cost/risk: nested Canary domains over steep terrain.
- This GPU diagnostic did not exercise `Cz > 1`; worst GPU `Cz` was `0.726` at R4. The CPU-WRF anchor suggested higher vertical Courant, but this GPU wrfout-frame diagnostic did not prove GPU `Cz > 1` tolerance.

Artifacts:

- `proofs/v022/k2_dt_ladder/k2_gate_summary.json`
- `proofs/v022/k2_dt_ladder/runs/R*/result.json`
- `proofs/v022/k2_dt_ladder/runs/R4/compare_to_r1.json`

## K1 BouLac O(nz) Compile Pathology

Verdict: `PATHOLOGICAL`.

The killgate harness had a donation/reuse bug: it reused donated JAX arrays during warm calls and could inflate failures. I fixed the harness to build fresh real-case input trees per timed forecast call.

Fixed-harness evidence:

- Dense full-pipeline baseline completed cleanly:
  - cold compile plus run: `83.78 s`
  - estimated compile wall: `51.48 s`
  - warm min: `32.30 s`
  - peak RSS: `8.26 GiB`
- O(nz) full-pipeline compile did not complete:
  - manually stopped at `727 s`
  - already `14.1x` dense compile wall and `8.7x` dense cold+run wall
  - RSS reached `15.18 GiB`, `1.84x` dense
  - XLA emitted `Very slow compile`

Conclusion: dense is not the issue on the sound harness. O(nz) remains opt-in or needs a real restructure of the first-crossing search before defaulting it.

Artifacts:

- `proofs/v022/compile_pathology/k1_boulac_onz_verdict.json`
- `proofs/v022/compile_pathology/killgate_dense.json`
- `proofs/v022/compile_pathology/killgate_onz.json`
- `proofs/v022/compile_pathology/boulac_onz_killgate.py`

Note: the fp32-BouLac subgate was not run in this opener pass after manager scope shifted to R0 baseline and K2 nest ladder.

## R0 Compile Memory Flags

Verdict: `RAM_REDUCTION_PREMISE_MOOT`.

The first R0 run used the old default `<DATA_ROOT>/wrf_downscale/runs/20240901/cpu` and was invalid: its `wrfinput_d01` / `wrfbdy_d01` symlinks point to missing staging targets. I reran with valid inactive input `<DATA_ROOT>/wrf_downscale/runs/20250121/cpu`.

Valid baseline evidence:

- fused maxdom3 cold first call: `1121.76 s`
- peak compile RSS: `22858 MiB` / `22.32 GiB`
- warm peak RSS: `12035 MiB`
- warm steady: `28.78 s/root-step`
- AOT warm load: `5` `source=aot_blob` loads, `0` load failures, no fused re-lower
- bit identity: `REF_COMPARE ref_equal=True`
- digest: `d7908da589dfcd9787c44302f8b1cfed1d5b32cf510fb6d9a056e0c16bb0c37d`

This is far below the old roughly `60 GiB` compile-RAM premise. Combined with the K1 harness bug, the old number was at least partly a harness/path measurement artifact. Memory-fitting flags were stopped because there is no meaningful RAM problem to solve on this sound 3-domain harness. Compile time remains real, and the AOT blob is the practical fix.

Follow-up: re-measure the 9-nest fused cold-compile peak RSS on the sound harness before treating the old roughly `60 GiB` number as a real v0.22 constraint.

Artifacts:

- `proofs/v022/r0_flag_sweep/r0_baseline_verdict.json`
- `proofs/v022/r0_flag_sweep/results_maxdom3_20260626T_validinput_20250121.jsonl`
- `proofs/v022/r0_flag_sweep/logs/cap_baseline_md3_20260626T_validinput_20250121.log`
- `proofs/v022/r0_flag_sweep/logs/warm_baseline_md3_20260626T_validinput_20250121.log`

## Work Items

1. K2 nest dt-ladder is the next decisive gate: run the 3-domain Canary ladder and report per-domain Courant, especially inner d03 over steep terrain.
2. Keep K1 O(nz) opt-in unless the first-crossing search is restructured and retested.
3. Keep shipping AOT blobs for compile-time mitigation; memory-fitting flags are not a v0.22 priority on the current evidence.
4. Re-measure 9-nest fused peak RSS on the fixed harness before using the old roughly `60 GiB` figure in roadmap decisions.
