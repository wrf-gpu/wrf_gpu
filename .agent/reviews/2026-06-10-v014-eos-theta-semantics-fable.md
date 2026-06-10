# V0.14 EOS / Theta Semantics — Fable Second-Pass Verdict

Date: 2026-06-10 WEST. Base `41468af4` + round-1 patch. CPU-only (no GPU used).

## Verdict: `FIXED` — round-1 patch NOT ratified as-is; replaced by a unified moist-theta_m convention. One GPU 1h rerun required before 72h gates.

**The manager's concern is CONFIRMED and the truth is worse than either side
assumed: the run was MIXED-convention.** Proven from the falsifier's own files
(`proofs/v014/eos_theta_semantics.{py,json,md}`):

- **d01 `State.theta` was DRY theta** (GPU d01 h1 `T` matches CPU dry `T` to
  +0.64 K at k0 and is −4.0 K off `THM`).
- **d02 interior `State.theta` was MOIST theta_m** (GPU d02 h1 `T` interior
  matches CPU `THM` to −0.38 K, +4.4 K off dry; the run's own pipeline metadata
  records `theta_m_conversion_applied=true`), while the **d02 boundary ring was
  forced DRY by the d01 parent** (band5 ≈ dry, −4.9 K vs THM) — a standing
  ~5 K theta discontinuity at the nest edge every parent step.
- The dycore EOS applied `qvf=1+0.608qv` to whatever `state.theta` held: on
  dry d01 that is vapor-light by ~1.0·qv (the −300 Pa PSFC family); on moist
  d02 it over-couples by ~0.61·qv — two different wrong EOS forms from one
  constant. Internal self-consistency inversion (each file's own discrete
  alpha_d, interior rmse, Pa): CPU dry-T needs 1.608 (22.3); GPU d01 dry-T fits
  0.608 (5.1); GPU d02 moist-T fits 0.608 (5.1). `qvf=1` and `1.608` both fail
  on the pre-fix GPU files by 330–554 Pa.
- Round-1's `1+rvovrd·qv` is the correct **dry-theta (use_theta_m=0)** WRF form
  — right for d01, **doubly wrong for moist d02** (θ_m·(1+1.608q) ≈
  dry·(1+1.608q)²). It also conflicted with the rest of the dycore:
  `rk_addtend_dry._absolute_diagnostics` and
  `operational_mode._acoustic_core_state` already used `qvf=1`, and the
  bit-exact live-nest `start_domain` transcription uses `qvf=1` with moist
  theta.

## Answers to the manager's four questions

1. **Dry or moist at h1?** Mixed (the bug): d01 dry; d02 interior moist θ_m,
   ring dry. The canonical v0.14 convention — anchored by the entire proven
   step-1 physics chain (NoahMP/MYNN decoupling, `conv_t_tendf_to_moist` at
   `operational_mode.py:3191`), the GPT consumer audit, and the bit-exact
   live-nest init — is **moist theta_m everywhere (WRF use_theta_m=1)**. This
   sprint makes that true in every lane.
2. **qvf per production caller:** with `State.theta = theta_m`, **qvf=1 in all
   production callers** (WRF `module_big_step_utilities_em.F` use_theta_m
   branch). `diagnose_pressure_al_alt` (both call sites) now passes no qv;
   `rk_addtend_dry.py:167` and `operational_mode.py:1152` already did. The
   helper keeps the round-1 `1+rvovrd·qv` form for `qv=`-passing callers, which
   is correct only for genuinely DRY theta inputs (use_theta_m=0 oracles).
   `1+0.608·qv` is wrong in every convention.
3. **Writer:** it wrote raw `state.theta − 300` as `T` — i.e. mislabeled moist
   THM as `T` on d02 (a fake +4.8 K low-level "warm bias" in any field-parity
   compare). Now: `T = theta_m/(1+rvovrd·qv) − 300` (WRF-compatible dry) and
   **`THM` is emitted** (`theta_m − 300`), matching WRF use_theta_m=1 wrfout.
   T2/TSK/flux fallbacks also use the dry view.
4. **Post-fix 1h falsifier signals** (decision rule before 72h gates):
   - GPU d02 `T` k0 bias vs CPU `T`: +3.3 K → drift scale (≲1 K); `THM`
     present and at the same scale vs CPU `THM`.
   - d02 boundary-band vs interior theta bias: same sign/scale (the −4.9 K
     ring discontinuity gone).
   - PSFC/P/MU/PH family: −313/−296 Pa biases collapse to the init-mode
     transient envelope (~±50–90 Pa class).
   - Persisting known separate classes: SWDOWN/COSZEN radiation-timing
     (~15–20 min), MUB/PB spec-band cells, init-mode envelope.

## Fix applied (smallest WRF-faithful, GPU-native; all init/ingest-time, no timestep-loop cost)

| File | Change |
|---|---|
| `src/gpuwrf/dynamics/acoustic_wrf.py` | `diagnose_pressure_al_alt`: both EOS calls drop `state.qv` → `qvf=1` (moist θ_m); helper docs state the dual convention; round-1 `RVOVRD` dry form kept for dry-theta callers. |
| `src/gpuwrf/integration/d02_replay.py` | IC (non-live-nest): dry wrfinput/wrfout `T` → θ_m when `use_theta_m=1` (live-nest child already converts via `adjust_tempqv`). wrfbdy LBC decode: keep the moist THM strips (was dividing to dry → the ring bug). History + nested-parent boundary loaders: recouple dry wrfout `T` with same-file QVAPOR. |
| `src/gpuwrf/io/wrfout_writer.py` | `T` decoupled dry; `THM` added (spec + minimum set); T2/TSK/flux fallbacks on dry view. |
| `tests/test_m7_netcdf_writer.py` | minimum-variable count 41 → 42 (THM). |

All conversions are elementwise, fuse in XLA, run at init/ingest/output only.

## Proof and validation

- `proofs/v014/eos_theta_semantics.{py,json,md}` — convention evidence,
  3-hypothesis EOS inversion, post-fix identities: helper moist==dry==analytic
  to 6.7e-16 rel; ingest end-to-end `build_replay_case(d01, standalone)` gives
  `State.theta == wrfinput THM` to 6.1e-5 K (vs dry: 5.47 K) with moist,
  IC-consistent wrfbdy theta leaves (3.1e-5 K); writer round trip `T`==dry
  to 3.1e-5 K (fp32), `THM`==θ_m exact.
- Commands:
  - `python -m py_compile src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/io/wrfout_writer.py tests/test_m7_netcdf_writer.py` ✓
  - `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/eos_theta_semantics.py` ✓
  - `python -m json.tool proofs/v014/eos_theta_semantics.json` ✓
  - acoustic subset (7 files): **24 passed, 3 skipped, 1 failed = the
    pre-existing qv=0 synthetic-fixture failure** (fails identically at HEAD;
    qv=0 makes this change an identity there).
  - `tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py
    tests/test_m6x_d02_replay_hang_debug.py tests/test_m7_netcdf_writer.py
    tests/test_async_wrfout_equiv.py` → 16 passed, 3 skipped.
  - `tests/test_auxhist_stream.py tests/test_auxhist_multistream.py` → 13 passed.
  - `git diff --check` clean.

## Unresolved risks / notes for the manager

1. GPU rerun pending (no GPU lock) — fix proven against truth files + CPU
   ingest, not yet end-to-end on device. JIT cache will recompile.
2. GPU-only golden fixtures (`tests/savepoint/fixtures/wrf_b6_100step/golden`)
   likely need regeneration (EOS qvf and theta convention changed).
3. d01 surface physics was silently cold-biased (~−4.7 K dry theta into
   moist-assuming adapters); the fix should IMPROVE replay/skill lanes but
   historical RMSE numbers will shift.
4. Provenance nuance: the live-nest child DOES read `wrfinput_d02`
   perturbation fields (T/QVAPOR/P/PH/MU at `d02_replay.py:1916ff`) — the
   earlier "wrfinput_d02 never read" memory line is wrong for perturbations
   (true only for the recomputed base state).
5. Minor, separate: dycore `CP_D = 1004.0` vs WRF `cp = 7·R_d/2 = 1004.5`
   (cpovcv 1.39972 vs 1.39986) — sub-Pa-to-few-Pa class, not this blocker.

## Next manager command (GPU lock required)

```bash
RUN_ROOT=/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
GPUWRF_RESOURCE_LOG_DIR="$RUN_ROOT/resources" GPUWRF_RESOURCE_LABEL=v014_short_field_h1_thetafix \
scripts/run_gpu_lowprio.sh -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
  --run-root /tmp/v0120_merged_run_root \
  --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
  --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 1 \
  --output-root "$RUN_ROOT/gpu_output" --proof-dir "$RUN_ROOT/proofs"
```
then `scripts/compare_wrfout_grid.py`; gate on the signals in answer 4.

## Handoff

- objective: settle theta/EOS semantics, replace-or-ratify round-1 — DONE
  (replaced; unified moist θ_m).
- files changed: `acoustic_wrf.py` (round-1 helper kept, callers → qvf=1),
  `d02_replay.py` (IC + 3 boundary loaders → moist), `wrfout_writer.py`
  (dry `T` + `THM`), `tests/test_m7_netcdf_writer.py` (count).
- proof objects: `proofs/v014/eos_theta_semantics.{py,json,md}`.
- next decision: manager runs the 1h GPU falsifier above; if theta/P families
  collapse per answer 4, start the 72h field-parity gates.
