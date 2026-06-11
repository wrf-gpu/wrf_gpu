# V0.14 Switzerland HPG Native-Face Fix (Fable)

Date: 2026-06-11
Worker: Fable xhigh, branch `worker/fable/v014-hpg-native-face-fix`
Sprint: `.agent/sprints/2026-06-11-v014-fable-hpg-native-face-fix/sprint-contract.md`

## Verdict

`EXACT_ROOT_NO_FIX` (for the venting blocker) — **with a committed,
native-face-proven WRF-faithfulness source fix** of the actual HPG-input
mismatch the sprint targeted.

Two separate findings, both WRF-anchored to native truth:

1. The targeted HPG native-face input mismatch was **found exactly and fixed in
   source**: the JAX runtime rebuilt every `calc_p_rho_phi` diagnostic
   (`al`/`alt`/`p`) with the `hypsometric_opt=1` LINEAR relation while WRF real
   cases run the v4 Registry DEFAULT `hypsometric_opt=2` LOG relation.  After
   the fix the large-step HPG per-face terms match WRF native truth to
   ~1.3–1.9e-4 relative (legacy: 1.6e-2 rel, max-abs 87 in `dpx`).
2. **The venting blocker is thereby REFUTED at the large-step HPG faces**: with
   the faces exact and the state arrays bit-identical, the h36→h37
   full-physics excess outflux collapses only **1.0 %** (old −28.62 → fixed
   −28.33 Pa/cell/h).  The sprint's premise ("the blocker is the HPG
   native-face inputs") is disproven by the native-face evidence itself.  The
   blocker lives downstream of `rk_tendency` — the acoustic-substep lane —
   and the WRF stage-boundary truth for that next comparison is ALREADY
   captured in this sprint's dumps.

## Root Cause Of The Face Mismatch (exact, fixed)

`module_big_step_utilities_em.F:1043-1062` (`hypsometric_opt=2`, Registry
default `2`, `Registry.EM_COMMON:2285`): WRF computes specific volume with the
LOG-pressure-thickness relation on the dry reference column; real.exe also
integrates the base `PHB` from `alb` with the SAME relation, so the carried
`alb` is the LOG base.  The JAX runtime used the linear (`opt=1`) form
everywhere.  Same-state anchors at the h36 CPU truth state (all in
`proofs/v014/switzerland_hpg_native_face_fix.json`):

| quantity | LOG form (`opt=2`) | LINEAR form (JAX legacy) |
|---|---|---|
| `alt` vs live EOS-implied `alt` (rel) | 1.4e-6 (fp32 roundoff) | 4.2e-4 mean / 6.2e-4 max, ONE-SIGNED |
| `al` vs live WRF `al` (rel, log-base alb) | **1.7e-6** | 2.8e-5 (bias partly cancels alt-vs-alb) |
| EOS `p` from `alt` vs file `P` | mean −0.005 Pa, rmse 0.30 | mean −19.5 Pa, max −52.9, terrain-modulated |

The bias grows with layer pressure thickness exactly as the WRF source comment
predicts (`p*dLOG(p) − dp = 1/12*(dp/p)^3…`), is horizontally modulated by
`muts` (terrain), and exactly reproduces GPT's unexplained
`alt_eos − (al+alb)` = 6.06e-4 measurement.  Every prior falsified variant
(`muts` cleanup, `alt=al+alb`, `p=EOS(θ,al+alb)`) stayed INSIDE the hypso-1
family — which is why nothing moved before.

**Load-bearing second-order finding:** subtracting a LINEAR-reconstructed
`alb` inside the `opt=2` `al` (first fix attempt) leaves the bias inside `al`
and corrupts the dominant `pb*al` face term ~15× worse than legacy.  The
committed fix computes BOTH total and base specific volume with the LOG form
(`al = LOG(total) − LOG(base)`), reproducing live WRF `al` to 1.7e-6 rel.

## WRF-Native Face Evidence

* Instrumented disposable WRF copy `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`
  (v4.7.1; HPG dump is env-gated, additive-only — recomputes identical subterm
  expressions after the untouched live loops).  Patch:
  `proofs/v014/switzerland_hpg_native_face_wrf_patch.diff`.  Gen2 truth
  binary's HPG + `calc_p_rho_phi` verified byte-identical to pristine 4.7.1.
* 36h30m 24-rank dmpar re-run of the exact d01 case
  (`/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face/run_wrf`,
  rc=0, `SUCCESS COMPLETE WRF`, wall 1486 s, cores 4–27).
* **Bit-exact trajectory reproduction**: the instrumented run's h36 wrfout
  equals the original Gen2-binary CPU truth h36 frame exactly
  (P/PH/MU/T/U/V/QVAPOR all rmse=0, max=0) — the dumps are native truth for
  the exact state every prior baseline and the JAX h36 re-init used.
* Dump mapping proven: call 1 `pb` bit-identical to wrfinput `PB`; call 21601
  (step 7201 RK1, first step after h36) `p`/`pb` bit-identical to the h36
  wrfout.  144 rank-dumps at calls 21601–21606 (steps 7201–7202 × RK1–3).

### Face comparison at RK1 of step 7201 (JAX minus WRF native)

State arrays `p/ph/pb/mu`: **bit-identical** (rmse = 0).

| term | legacy opt=1 rmse (max) | fixed opt=2 rmse (max) | WRF signal rmse |
|---|---:|---:|---:|
| `al` | 5.9e-4 (3.1e-3) | **5.4e-6** (5.3e-5) | — |
| `t1` (ph pair), x/y | 2.7e-5 | 2.7e-5 | 25.8 / 21.5 |
| `t2` (alt·Δp), x/y | 3.3e-3 / 2.0e-3 | 3.3e-3 / 2.0e-3 | 30.8 / 26.6 |
| `t3` (al·Δpb), x/y | **0.567 (11.5)** / 0.567 (10.9) | **4.0e-3 (0.21)** / 3.9e-3 | 30.0 / 29.1 |
| `t4` (non-hydro) | 6.4e-4 | 6.4e-4 | 28.3 / 27.4 |
| `dpx` / `dpy` (full) | **6.05 (86.9)** / 6.01 (89.0) | **0.070 (2.76)** / 0.056 (2.02) | 368 / 305 |
| `muu`/`muv` | 3.3e-3 (= mu fp32 ulp) | same | — |

The legacy face error (1.6e-2 rel, concentrated in `pb*al`) is removed (~86×);
the fixed faces sit at 1.3–1.9e-4 relative — fp32-roundoff class for these
chains.  Residual `alt` diff vs the live array is ~1e-4 rel (EOS-θ sourcing
chain), with face impact ≤1e-4 rel — three orders below the venting signal.

## Source Fix Summary (committed)

`hypsometric_opt` threaded as a WRF-faithful option (function default 1 =
legacy byte-unchanged, because idealized generators carry placeholder
`c3f/c4f` that make the LOG form singular; the REAL pipelines pass the WRF
Registry default 2):

* `src/gpuwrf/dynamics/acoustic_wrf.py::diagnose_pressure_al_alt` — opt-2:
  `al = LOG(total) − LOG(base)`, `alt = al + alb_log`, `p = EOS(θm, alt)`;
  consumed by the per-stage refresh `_refresh_grid_p_from_finished` and the
  `pg_buoy_w` stage pressure.
* `src/gpuwrf/dynamics/core/rk_addtend_dry.py::_absolute_diagnostics` +
  `large_step_horizontal_pgf(hypsometric_opt=…)` — opt-2 `al` for the `pb*al`
  branch; `muts = State.mu_total` (WRF dry total) in the opt-2 branch.
* `src/gpuwrf/runtime/operational_mode.py` —
  `OperationalNamelist.hypsometric_opt` (static aux, default 1) + `from_grid`
  + the three operational call sites.
* `src/gpuwrf/integration/daily_pipeline.py`, `nested_pipeline.py` — real-case
  namelists pass `hypsometric_opt=2`.
* New regression test `tests/test_v014_hypsometric_opt2.py` (analytic
  LOG-column oracle; log-base `alb` subtraction pinned; option threading
  pinned).  3/3 pass.

No clamps/masking; no host/device transfer in the timestep loop (pure algebra
swap on resident arrays, same operation count class — log/pow replace
divisions only inside the already-jitted diagnostics); no GPU memory increase
observed (probe peak ~ same as baseline probes).

## H36 Gate Result

30-step h36 dry probe (same config as the ec4d6769 baseline;
`hypso1_legacy` reproduces the baseline −6.7059 Pa bit-tight):

| variant | MU Δ steps 1→30 | PGF contribution vs zero-PGF |
|---|---:|---:|
| hypso1 legacy (= ec4d6769 baseline) | −6.7059 Pa | −34.05 Pa/cell/h |
| hypso2 fixed (log-base alb) | −6.0829 Pa | −26.59 Pa/cell/h |
| zero-PGF controls (opt 1 / 2) | −3.8684 / −3.8674 Pa | — |

Hourly full-physics budget gate, h36→h37, depth-8 control surface:

| run | net influx (Pa/cell/h) | excess vs CPU |
|---|---:|---:|
| CPU truth | −74.515 | — |
| old baseline (`gpu_output`) | −103.130 | −28.615 |
| fixed (`gpu_output_hpg_native_face_fix`) | −102.843 | −28.328 |

**Collapse fraction = 0.010 — gate NOT met.**  Fixed h37 state is finite; MU
bias −50.0 (old −54.4), PSFC bias −52.2 (old −56.6); U/V/T rmse essentially
unchanged.  Conclusion: with the HPG faces now exact, the venting persists ⇒
the blocker is NOT the large-step HPG inputs/operator.

## Files Changed

* `src/gpuwrf/dynamics/acoustic_wrf.py`
* `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
* `src/gpuwrf/runtime/operational_mode.py`
* `src/gpuwrf/integration/daily_pipeline.py`
* `src/gpuwrf/integration/nested_pipeline.py`
* `tests/test_v014_hypsometric_opt2.py` (new)
* `proofs/v014/switzerland_hpg_native_face_fix.py` (new)
* `proofs/v014/switzerland_hpg_native_face_fix.json` (new)
* `proofs/v014/switzerland_hpg_native_face_wrf_patch.diff` (new)
* `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`

WRF instrumentation lives ONLY in the disposable copy
`/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`;
`/home/enric/src/wrf_pristine/WRF` untouched this sprint (read-only anchor).

## Commands Run (essential)

```bash
git branch --show-current && git log -1 --oneline
# WRF instrumentation + build (disposable copy)
git -C /mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF diff dyn_em/module_big_step_utilities_em.F \
  > proofs/v014/switzerland_hpg_native_face_wrf_patch.diff
taskset -c 0-3 /mnt/data/wrf_gpu2/v014_post_rk_refresh/compile_hpg_native_face.sh   # rc=0
# 36h30m native truth run (24 ranks, dump window steps 7201-7202)
WRFGPU2_HPG=1 WRFGPU2_HPG_CALL_LO=21601 WRFGPU2_HPG_CALL_HI=21606 WRFGPU2_HPG_FIRST=3 \
  WRFGPU2_HPG_ROOT=.../hpg_dumps  taskset -c 4-27 mpirun --oversubscribe --bind-to none -np 24 ./wrf.exe
# probes / gates (GPU, serialized via scripts/run_gpu_lowprio.sh)
scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_hpg_native_face_fix.py --step-probe --steps 30
scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_hpg_native_face_fix.py --forecast-variant --hours 3
scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_hpg_native_face_fix.py --wrf-faces --ncall 21601
python proofs/v014/switzerland_hpg_native_face_fix.py     # analyzer -> JSON
# tests
pytest -q tests/test_v014_hypsometric_opt2.py             # 3 passed
pytest -q tests/test_daily_boundary_clock.py tests/test_m6_boundary_apply.py   # 9 passed
pytest -q tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6_horizontal_pressure_gradient_fix.py \
  tests/test_m6x_c2_acoustic.py   # 9 passed, 1 skipped, 1 PRE-EXISTING failure (also fails with fix stashed)
# hygiene
python -m py_compile proofs/v014/switzerland_hpg_native_face_fix.py
python -m json.tool proofs/v014/switzerland_hpg_native_face_fix.json >/tmp/switzerland_hpg_native_face_fix.validated.json
git diff --check
```

## Proof Objects / Run Roots / Resource CSVs

* Main proof: `proofs/v014/switzerland_hpg_native_face_fix.json`
  (same-state alt forms, step probe, hourly gate, full face table).
* Proof script: `proofs/v014/switzerland_hpg_native_face_fix.py`.
* WRF patch: `proofs/v014/switzerland_hpg_native_face_wrf_patch.diff`.
* Native truth run + dumps:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face/{run_wrf,hpg_dumps,run_h36}`
  — `hpg_dumps` holds 144+72 rank dumps incl. **RK2/RK3 stage-boundary live
  states (calls 21602–21606) = the ready-made truth for the next sprint**.
* Fixed GPU output:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_hpg_native_face_fix`
  (h37 frame; the 3h run was interrupted after h37 by a stray SIGTERM — h37 is
  the gate frame and is finite; not rerun because the 1 % collapse verdict
  cannot flip with more hours).
* Resource CSVs:
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/fable_hpg_*`
  (probe max GPU mem ~7.6 GiB class, in line with prior probes).

## Unresolved Risks

* The venting blocker remains open — now provably OUTSIDE the large-step HPG
  faces.  Remaining candidate lanes: acoustic substeps (`advance_uv` work-array
  p, `advance_mu`/`ww` divergence, vertical implicit solve), large-step
  momentum advection, boundary-relaxation interplay.
* The hypso fix changes real-pipeline behaviour everywhere (al/alt/p
  diagnostics); Canary/TOST re-validation should ride the next full validation
  wave.  Idealized paths are byte-unchanged by construction (default opt=1);
  `tests/test_m6x_pressure_diagnose_wiring.py::test_nonhydrostatic_carry_...`
  fails PRE-EXISTING (identically with the fix stashed).
* The fixed 3h forecast only produced h37 (stray SIGTERM); h38/h39 stability
  evidence not collected (h37 finite, fields sane).
* The remaining `alt` residual vs live (~1e-4 rel, EOS-θ sourcing chain) is
  three orders below the venting signal but not yet bit-closed.

## Next Manager Action

ONE concrete implementation target: **the acoustic-substep lane at step
7201**.  The WRF truth is already on disk — `hpg_dumps` calls 21602/21603
(RK2/RK3 of 7201) and 21604–21606 (7202) contain the live post-substep
`p/ph/al/alt/mu` at each RK-stage boundary, bit-anchored to the trajectory the
JAX h36 re-init starts from.  Dispatch a sprint that (a) exposes the JAX
operational per-stage states (debug hook on `_physics_boundary_step`'s RK
loop), (b) compares stage-boundary increments RK1→RK2→RK3 vs the dumps, and
(c) bisects the first diverging increment inside the acoustic core
(`advance_uv` / `advance_w` / `advance_mu` / `ww`).  No new WRF run or build
is required.
