# GATE-2 FIX BRIEF — localize + fix the REAL v0.23-default divergence (RELEASE-CRITICAL)

**Frontrunner:** GPT. **Manager:** pane `0:1`. **Type:** perf-core / bit-identity → release blocker. **GPU:** MANDATORY `scripts/with_gpu_lock.sh --label v023-gate2 -- <cmd>` on every GPU run (single GPU; you have priority over the f1-retest window).

## The finding (ground truth)
The v0.23 default path is NOT bit-identical to v0.22.2. The PROPER field-compare (NOT the stored digest — that gate is autotune-broken/false) of **v0.23-default (`1c77855c`) vs FRESH v0.22.2 (`53bd20bf`)**, both warm-cache NPZ, shows:
`max_abs_global = 0.44012, worst field d01/0005/5, 80/178 leaves differ`.
That is a REAL numerical change (NOT ~1e-13 autotune noise). The v0.23 line = F1 (`80287687`) + P-bundle (P6 `1d05d764`, P2 `6a26f145`, P4 `2a3079ab`, P1 `cb65c286`, P7b `68730ac1`). An earlier bisect flagged **F1@80287687 as the first-diverging commit → F1's "default inert" claim is the prime suspect** (its default path likely routes through batched/vmap code even when `GPUWRF_BATCH_ENSEMBLE` is unset).

## Tooling (already proven working)
- Capture: `proofs/v023/gpu_gates/canary_state_capture.py` with env `CANARY_DUMP_STATE=1 CANARY_FIXED_REPEATS=2 CANARY_MAXDOM=3 CANARY_SINGLE_SHAPE_TIMING=1 CANARY_FIXED_ROOT_STEPS=2 CANARY_INPUT_DIR=<DATA_ROOT>/wrf_downscale/runs/20250121/cpu`, plus per-worktree `PYTHONPATH=<wt>/src GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF JAX_PLATFORMS=cuda JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=<wt-cache> GPUWRF_CACHE=<wt-cache> GPUWRF_NESTED_AOT=1 GPUWRF_NESTED_PARALLEL_COMPILE=0 GPUWRF_NESTED_DEFUSE_COMPILE=0` and `-u GPUWRF_BATCH_ENSEMBLE -u GPUWRF_SINGLE_SCAN`. Writes `<tag>_state.npz` + `<tag>_state_meta.json` (prints `STATE_DUMP ... npz=<abs> meta=<abs>` — use the ABSOLUTE paths).
- Compare: `python proofs/v023/gpu_gates/canary_state_compare.py --ref-npz A --ref-meta Am --cand-npz B --cand-meta Bm --output out.json` → `max_abs_global` + per-field.
- FRESH v0.22.2 baseline already captured: `v023-refcheck/proofs/v023/gpu_gates/canary_state/v0222_cap_state.npz` (digest 8b5b4405). Warm caches: `v023-refcheck/proofs/v023/refcheck/cache`, `v023-defcheck/proofs/v023/defcheck/cache`.
- Worktrees: `v023-refcheck`@53bd20bf(v0.22.2) · `v023-f1`@80287687(F1-only) · `v023-canary-bisect/{p6,p2,p4}`@those commits · `v023-defcheck`@1c77855c(v0.23-default) · `v023-gpu`@`worker/opus/v023-integration` (commit the FIX here).

## Do this
1. **LOCALIZE:** capture F1-only state (worktree `v023-f1`@80287687, batch-flag UNSET, fresh cache) → field-compare vs the FRESH v0.22.2 NPZ. If `max_abs ≈ 0.44` → **F1 is the culprit** (default path not inert). If `≈0` → walk the P-bundle bisect worktrees (p6 → p2 → p4, each vs v0.22.2) to find the first that diverges. Report which commit + which fields.
2. **FIX (root-cause, no masking):**
   - If **F1**: make the default path (when `GPUWRF_BATCH_ENSEMBLE` is unset/`1`) route through the EXACT pre-F1 unbatched orchestration — byte-identical to 53bd20bf. The vmap/batched wrappers must be fully bypassed when B is not explicitly >1. (Look at `src/gpuwrf/runtime/domain_tree.py` + `integration/nested_pipeline.py` F1 edits.)
   - If a **P-item**: it wasn't actually bit-identical — fix it to be byte-identical on the default path, or gate it default-off. Do NOT widen a tolerance.
3. **VERIFY:** re-capture the v0.23-line default WITH the fix → field-compare vs FRESH v0.22.2 → **`max_abs_global` must be ≤ ~1e-11** (rounding/autotune floor). Also confirm all_finite + no s/step regression.
4. **COMMIT** the fix on `worker/opus/v023-integration` (clear message). Update `proofs/v023/gate2_fix/GATE2_FIX_REPORT.md` (localization result, the fix, before/after max_abs).

## Report / constraints
- Report the LOCALIZATION result to `0:1` FIRST (`scripts/tmux_submit.sh 0:1 '<msg>'`), then the FIX verdict. `touch proofs/v023/gate2_fix/GATE2_FIX_DONE` as the last step.
- No masking/clamp/nan_to_num. Honest — if the fix doesn't reach ≤1e-11, report why. Blocked → append `proofs/v023/gate2_fix/GATE2_FIX_QUESTION.md` + ping `0:1`. Begin now.
