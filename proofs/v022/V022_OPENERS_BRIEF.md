# v0.22 GPU-OPENERS — frontrunner brief (manager 0:1 → GPT 0:3, 2026-06-26)

You are the v0.22 frontrunner (the user: GPT is the default frontrunner). Run the v0.22 GPU openers PROPERLY,
**supervised + sequential**, and synthesize honestly. A prior ultracode workflow attempt FAILED (synthesis
schema cap) and left methodology flaws — this brief fixes them. Working dir: `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile`.

## Context (read first)
- `.agent/decisions/V022-ROADMAP.md` — items **R0** (compile trilemma), **K1** (BouLac O(nz)), **K2** (dt/n_sound), step "RECOMMENDED v0.22 SEQUENCE".
- `proofs/v022/GPU_OPENER_PREP.md` — verified knobs/paths/flags (K2 rungs, R0 flag shortlist, RSS harness, AOT-blob path).
- `.agent/decisions/V022-EXECUTION-STATE.md` — live state + dispatch rule.

## HARD RULES
- Every GPU command goes through `scripts/with_gpu_lock.sh`. **Run ONE GPU job at a time** — do NOT launch
  multiple lock-waiters concurrently (that starved R0 for >1.5h last time). Supervise each to completion, collect
  its JSON, THEN start the next.
- Do NOT kill 0:2's corpus `wrf.exe` (currently `<DATA_ROOT>/wrf_downscale/runs/20250604/cpu/wrf.exe`). Exact-PID kills only if ever needed.
- Measure-first; flip NO default without a passing gate. North-star: wallclock above all + WRF-faithful (bit-identical OR the declared tiered/op-relaxed tiers).
- Synthesize in PLAIN JSON/markdown you write yourself — do NOT depend on a fragile structured-output tool.
- Report to manager `tmux 0:1.0` (send-keys) at each gate completion + at the end.

## SALVAGED partial results from the failed run (DO NOT redo what's done)
1. **K2 CFL anchor — DONE.** `proofs/v022/k2_dt_ladder/cfl_anchor_cpuwrf.json`: CPU-WRF Switzerland-128 trajectory,
   **vertical Courant Cz is the binding term** (Cz 0.83 @dt=10 → 1.49 @dt=18; C_total 1.11→2.00; min_dz~43m).
   CPU-WRF stays stable to dt=18 via its **implicit vertical (acoustic) solver**. The K2 harness is already written
   in `proofs/v022/k2_dt_ladder/` (`run_rung.py`, `scale_namelist.py`, `run_switzerland_sweep.sh`, `compare_to_r1.py`,
   `pick_best_rung.py`). **REMAINING K2 WORK = actually run the 4 GPU rungs** (R1 dt10/ns10, R2 12/9, R3 15/8, R4 18/7)
   on Switzerland-128, compare each wrfout to R1 (operational tolerance), check finite/stable. **THE key question:
   does OUR GPU vertical solver tolerate Cz>1 (is it implicit like WRF?) at dt=15/18, or go unstable?** Report best
   CFL-safe + stable rung + its s/step win vs R1. Validate the best rung on the 3-dom Canary only if time permits.
2. **Compile-pathology baseline — DONE, but the killgate cap was WRONG.** `proofs/v022/compile_pathology/gate_b_baseline.json`
   (in the MAIN checkout `<USER_HOME>/src/wrf_gpu2/proofs/v022/compile_pathology/`): Switzerland-128 full-operational
   **cold compile ≈ 768 s**, steady **158.7 ms/step**, finite. The O(nz)/dense BouLac killgate used a **300 s probe cap
   — far SHORTER than the 768 s normal compile** → its "STILL-PATHOLOGICAL" verdicts are meaningless. **RE-RUN gate (a)
   O(nz)-BouLac + dense with a proper cap (≥1200 s, ideally 1800 s)** to get a real CRACKED / PATHOLOGICAL distinction
   (PASS = O(nz) full-pipeline compile completes in a bounded time comparable to the 768 s dense baseline + tiered-identity
   vs frozen v0.14). Reuse `boulac_onz_killgate.py` (fix the timeout). Consolidate outputs into the worktree `proofs/v022/`.
3. **fp32-BouLac gate (b):** baseline (fp32 OFF) captured in `gate_b_baseline.json`; the fp32-ON run was interrupted —
   re-run it (`gate_b_harness.py --tag fp32boulac`) and record whether it compiles bounded or stalls.
4. **R0 — NEVER RAN (lock-starved).** `proofs/v022/r0_flag_sweep/run_r0_sweep.sh` + `extract_r0.py` exist.
   **FIRST probe each candidate flag against the installed jaxlib** (`python -c "import jax;print(jax.__version__)"` →
   0.10.0; test each flag is accepted — many in the shortlist may not exist). Then sweep memory-fitting flags on the
   fused cold compile; for each: cold wall, peak host RSS (getrusage ru_maxrss, see perstep_timing_driver.py), valid
   AOT blob (load + bit-identical), **runtime s/step (HARD-FAIL if >1-2% slower than fused baseline)**. PASS = a setting
   with materially lower peak RAM + zero runtime regression + valid blob; else a clean impossibility proof (ship-the-blob stays the answer).

## Deliverable
`proofs/v022/V022_OPENERS_SYNTHESIS.md` (you write it): per-gate verdict (CRACKED / PATHOLOGICAL / INCONCLUSIVE with
the real cap), K2 best safe rung + s/step win + the Cz>1 finding, R0 best setting or impossibility proof, and concrete
v0.22 work-items for the manager. Honest per the north-star — do NOT claim a single-card multiplier you did not measure.
Commit your artifacts. Notify 0:1 at each gate + at the end.
