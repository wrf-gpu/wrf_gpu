# F1 B>1 LOADER FIX — DIAGNOSIS-GATED (measure F1's real value, or prove it needs a deep rework)

**Frontrunner:** GPT. **Manager:** pane `0:1`. **Worktree:** you are in `v023-gpu` on `worker/opus/v023-integration` (HEAD has the GATE-2 fix `58cdbc3f` which DISABLED `GPUWRF_BATCH_ENSEMBLE>1` fail-closed). **GPU:** free; MANDATORY `scripts/with_gpu_lock.sh --label v023-f1fix -- <cmd>` on every GPU run.

## Context (the finding)
F1 value re-test on the CORRECT real Tenerife-only 3km/1km fixture (`<DATA_ROOT>/wrf_downscale/f1_retest/tenerife_20250121_3km1km/real_run`, d01 61×61@3km + d02 103×70@1km): **B=1 works** (4.278M warm cells/s, 9.4 GiB/case → B=3 would fit 32 GiB VRAM). **B=2 CRASHES at LOAD time** (before the timestep loop): `_load_batched_domains()` → `_assert_homogeneous_batch()` raises `ValueError: batch lane 1 d01: namelist/static treedef differs` — for IDENTICAL input lanes (same input_dir repeated). So F1's batching value is UNMEASURED (it doesn't even run at B>1). Report: `proofs/v023/f1_retest/F1_RETEST_REPORT.md`.

## STEP 1 — DIAGNOSE (cheap, CPU, report to 0:1 BEFORE the expensive fix)
Root-cause WHY two identical input lanes produce different d01 `namelist/static treedef`. Look at `_load_batched_domains` + `_assert_homogeneous_batch` (src/gpuwrf/runtime/domain_tree.py / integration/nested_pipeline.py) + how each lane's namelist/static-aux is built. Determine: is it (**BOUNDED**) a lane-canonicalization/determinism bug (e.g. per-lane object identity, unsorted dict, a Path vs str, a non-canonical static field) → a small fix; or (**DEEP**) F1's per-lane static-aux genuinely can't be batched without a runtime rework (the "default-inert F1 reimplementation" the gate2 agent flagged)?
**→ POST the root cause + a BOUNDED-vs-DEEP verdict + effort estimate to `0:1` (`scripts/tmux_submit.sh 0:1 '<msg>'`) and write `proofs/v023/f1_loaderfix/F1_LOADERFIX_DIAGNOSIS.md`. If DEEP → STOP there (do not over-invest); the manager decides fix-vs-defer.**

## STEP 2 — FIX (ONLY if BOUNDED)
Make identical lanes produce identical static treedefs so `_assert_homogeneous_batch` passes + B>1 loads. Re-enable `GPUWRF_BATCH_ENSEMBLE>1` (remove the `58cdbc3f` fail-closed gate) ONLY behind the now-working loader. **HARD CONSTRAINT: the DEFAULT path (`GPUWRF_BATCH_ENSEMBLE` unset or =1) MUST stay byte-identical to v0.22.2 — re-verify with the gate2 field-compare tool (`canary_state_capture.py` CANARY_DUMP_STATE=1 + `canary_state_compare.py` vs the fresh v0.22.2 NPZ at `v023-refcheck/.../v0222_cap_state.npz`; same-cache max_abs must be 0.0). Do NOT re-break GATE 2.** No masking/clamp.

## STEP 3 — RE-MEASURE (ONLY if STEP 2 done)
Re-run the B-sweep B∈{1,2,3} on the Tenerife fixture via `proofs/v023/gpu_gates/g3_batch_sweep_driver.py` (warm, `G3_ROOT_STEPS=400`) → per B: warm cells/s, peak VRAM, VRAM/case, **throughput multiplier vs B=1 serial (cases/GPU-hour)**. **THE verdict: does batching WIN on the correct starved Tenerife grid (≥~1.5× at B=2/3)?** Honest numbers.

## Deliverable / constraints
- `proofs/v023/f1_loaderfix/F1_LOADERFIX_REPORT.md` (diagnosis + fix + B-sweep verdict + GATE-2-still-green proof). `touch proofs/v023/f1_loaderfix/F1_LOADERFIX_DONE` last (or `..._QUESTION` if blocked/DEEP-stop).
- Report to `0:1` at each step boundary. GPU via with_gpu_lock (single GPU). No masking. Don't touch corpus / other lanes / windows 3-4. Honest — if it's DEEP, say so + stop; if batching loses even when fixed, say so. Begin STEP 1 now.
