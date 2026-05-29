# Sprint U Confirm-Close Findings

Reviewer: GPT-5.5 xhigh (Codex)
Date: 2026-05-29
Branch reviewed: `worker/opus/f7d-pressure-mass-fix` at `1b7836c`
Scope: final confirm-close of Sprint U remediation only, per `gpt-confirm-close-prompt.md`.

## Findings

### P0-1 NOT-CLOSED: real operational path is still not actually fp64

`daily_pipeline._build_real_case` now sets `force_fp64=True` in the namelist, but the real Canary state is not upcast. The proof object itself contradicts the "fp64" active-operator claim: `proofs/sprintU/real_case_smoke.json:12` says `"precision": "fp64"`, while the evolved real-case fields are still `theta=float32`, `u=float32`, and `v=float32` at `proofs/sprintU/real_case_smoke.json:66`, `proofs/sprintU/real_case_smoke.json:77`, and `proofs/sprintU/real_case_smoke.json:88`.

Root cause is concrete:

- `_enforce_operational_precision(..., force_fp64=True)` builds float64 updates at `src/gpuwrf/runtime/operational_mode.py:299`, `src/gpuwrf/runtime/operational_mode.py:305`.
- `State.replace` immediately casts each update back to the existing field dtype at `src/gpuwrf/contracts/state.py:576` through `src/gpuwrf/contracts/state.py:579`.
- `run_forecast_operational`, `run_forecast_operational_with_limiter_diagnostics`, and `run_forecast_operational_debug` also create their initial carry with `_enforce_operational_precision(state)` without passing `force_fp64=namelist.force_fp64` at `src/gpuwrf/runtime/operational_mode.py:1549`, `src/gpuwrf/runtime/operational_mode.py:1602`, and `src/gpuwrf/runtime/operational_mode.py:1655`.

Verification command:

```bash
PYTHONPATH=src taskset -c 0-3 python - <<'PY'
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
cfg=DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
case,_=_build_real_case(cfg)
forced=_enforce_operational_precision(case.state, force_fp64=True)
print('namelist.force_fp64', case.namelist.force_fp64)
for name in ('u','v','w','theta','p','ph','mu'):
    print(name, 'before', getattr(case.state,name).dtype, 'after_force', getattr(forced,name).dtype)
PY
```

Observed:

```text
namelist.force_fp64 True
u before float32 after_force float32
v before float32 after_force float32
w before float64 after_force float64
theta before float32 after_force float32
p before float64 after_force float64
ph before float64 after_force float64
mu before float64 after_force float64
```

Other P0-1 subclaims are substantially true: `_build_real_case` sets `use_flux_advection=True`, `diff_6th_opt=2`, WRF damping, and open top at `src/gpuwrf/integration/daily_pipeline.py:179` through `src/gpuwrf/integration/daily_pipeline.py:195`; `_augment_large_step_tendencies` takes the flux branch when `use_flux_advection` is true at `src/gpuwrf/runtime/operational_mode.py:1107`; `advect_w_flux` receives `top_lid=bool(namelist.top_lid)` at `src/gpuwrf/runtime/operational_mode.py:1139` through `src/gpuwrf/runtime/operational_mode.py:1142`; and `run_forecast_operational` scans `_physics_boundary_step` at `src/gpuwrf/runtime/operational_mode.py:1500` through `src/gpuwrf/runtime/operational_mode.py:1503`. But the "force_fp64" part was central to the remediation and is false for the real path.

Verdict: NOT-CLOSED.

### P0-2 NOT-CLOSED for the stated u/v/w WRF deformation operator

Sprint U added a useful flat-slab u/w deformation operator, and its unit tests pass. But the requested remediation was WRF deformation-tensor momentum diffusion for u/v/w, including `defor11/22/33/12/13/23` and `horizontal_diffusion_u/v/w_2`. The implementation is only the one-row u/w reduction:

- `wrf_deformation_momentum_tendency` takes only `u` and `w`, and returns only `(du, dw)` at `src/gpuwrf/dynamics/explicit_diffusion.py:302` through `src/gpuwrf/dynamics/explicit_diffusion.py:310`.
- Its own docstring scopes it to `ny=1` and lists only `D11`, `D33`, and `D13` at `src/gpuwrf/dynamics/explicit_diffusion.py:320` through `src/gpuwrf/dynamics/explicit_diffusion.py:332`.
- Runtime wiring explicitly leaves `v` on scalar flux-divergence when deformation mode is enabled at `src/gpuwrf/runtime/operational_mode.py:1206` through `src/gpuwrf/runtime/operational_mode.py:1216`.
- The analytic tests cover only constant-density u/w closed forms and no v/D22/D12/D23 path at `tests/dynamics/test_deformation_momentum_diffusion.py:1` through `tests/dynamics/test_deformation_momentum_diffusion.py:15`.
- The proof admits `v` uses scalar form for the one-row slab and that multi-row `D22/D12/D23` is deferred at `proofs/sprintU/momentum_diffusion_deformation.md:47` through `proofs/sprintU/momentum_diffusion_deformation.md:55`, and `proofs/sprintU/momentum_diffusion_deformation.md:81` through `proofs/sprintU/momentum_diffusion_deformation.md:86`.

WRF does have the missing pieces: `horizontal_diffusion_v_2` starts at `/home/enric/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F:3323`, its flat tendency assembly uses `titau2`/`titau1` at `/home/enric/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F:3503` through `/home/enric/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F:3508`, and `cal_deform_and_div` builds the full deformation tensor, not just `D11/D33/D13`.

The Straka deformation A/B proof is still meaningful for the one-row 2D density-current gate (`proofs/sprintU/straka_deformation_gate.json:1` through `proofs/sprintU/straka_deformation_gate.json:31`), but it does not close the stated u/v/w WRF deformation remediation.

Verdict: NOT-CLOSED as stated; closed only for the 2D one-row u/w subcase.

## Per-Item Verdicts

### P0-1: operational/real-case path unified with F7 dycore

Verdict: NOT-CLOSED.

Evidence:

- Config flags are present in `_build_real_case`: `use_flux_advection=True`, `force_fp64=True`, `diff_6th_opt=2`, `w_damping=1`, `damp_opt=3`, `zdamp=5000`, `dampcoef=0.2` at `src/gpuwrf/integration/daily_pipeline.py:179` through `src/gpuwrf/integration/daily_pipeline.py:195`.
- Operational scan uses `_physics_boundary_step` via `_scan_forecast_segment` at `src/gpuwrf/runtime/operational_mode.py:1500` through `src/gpuwrf/runtime/operational_mode.py:1503`.
- Flux-form branch is taken when the namelist says so at `src/gpuwrf/runtime/operational_mode.py:1107`, and the proof reports a nonzero flux-vs-primitive theta difference at `proofs/sprintU/real_case_smoke.json:113` through `proofs/sprintU/real_case_smoke.json:114`.
- However, the real path is not fp64 for `theta/u/v`; see finding P0-1 above. This directly contradicts `proofs/sprintU/operational_path_unification.md:89` through `proofs/sprintU/operational_path_unification.md:92`.

### P0-2: WRF deformation-tensor momentum diffusion wired

Verdict: NOT-CLOSED as stated.

Evidence:

- u/w flat-slab implementation exists and tests pass.
- v deformation is not implemented; runtime keeps scalar diffusion for v at `src/gpuwrf/runtime/operational_mode.py:1213` through `src/gpuwrf/runtime/operational_mode.py:1216`.
- Unit tests are FD closed-form self-consistency for constant-density u/w, not a WRF Fortran fixture for u/v/w. See `tests/dynamics/test_deformation_momentum_diffusion.py:30` through `tests/dynamics/test_deformation_momentum_diffusion.py:59`.
- The WRF source has separate v deformation diffusion and full deformation terms; Sprint U defers these.

### P0-3: canonical-WRF Straka array-level parity

Verdict: CLOSED for the submitted time-series parity gate, with documented limits.

Evidence:

- The WRF reference is a real `wrfout` path and exists: `proofs/m9/wrf_em_grav2d_x_front_savepoints.json` points to `/home/enric/src/wrf_pristine/run_grav2d/wrfout_d01_0001-01-01_00:00:00`; I verified it exists.
- The canonical namelist matches the claimed essentials: `time_step=1`, `e_vert=65`, `dx=100`, `dy=100`, `diff_opt=2`, `km_opt=1`, `damp_opt=0`, `khdif=75`, `kvdif=75`, `time_step_sound=6` at `/home/enric/src/wrf_pristine/WRF/test/em_grav2d_x/namelist.input.100m:29` through `/home/enric/src/wrf_pristine/WRF/test/em_grav2d_x/namelist.input.100m:73`.
- The proof compares against WRF rows, not JAX-vs-JAX, and reports worst max|w| relative diff 0.1188 and worst front diff 400 m at `proofs/sprintU/straka_canonical_parity.json:157` through `proofs/sprintU/straka_canonical_parity.json:188`.
- The 0.119 worst max|w| error occurs at 60 s, not the touchdown peak; touchdown is 0.0456 at 240 s and 0.0048 at 300 s in `proofs/sprintU/straka_canonical_parity.json:86` through `proofs/sprintU/straka_canonical_parity.json:121`. I do not see evidence there of the previous runaway operator error.
- Limit remains: this is diagnostic time-series parity, not per-cell field parity, as documented at `proofs/sprintU/straka_canonical_parity.md:61` through `proofs/sprintU/straka_canonical_parity.md:67`.

### P0-4/P0-5: CI close-gate asserts PASS

Verdict: CLOSED.

Evidence:

- New close-gate tests assert `result.verdict == "PASS"` and archived payload `verdict == "PASS"` at `tests/idealized/test_dycore_close_gate.py:47` through `tests/idealized/test_dycore_close_gate.py:76`.
- Existing idealized tests also assert `result.verdict == "PASS"` at `tests/idealized/test_warm_bubble.py:24` through `tests/idealized/test_warm_bubble.py:30`, and `tests/idealized/test_density_current.py:24` through `tests/idealized/test_density_current.py:30`.
- Marker is registered at `pyproject.toml:28`.
- Verification command passed:

```bash
PYTHONPATH=src taskset -c 0-3 pytest -q tests/idealized/test_dycore_close_gate.py -m close_gate
```

Observed:

```text
2 passed in 425.80s (0:07:05)
```

### P1-5: advect_w open-top face

Verdict: CLOSED.

Evidence:

- WRF source computes `vflux(i,ktf+1)` and lid pickup at `/home/enric/src/wrf_pristine/WRF/dyn_em/module_advect_em.F:6014` through `/home/enric/src/wrf_pristine/WRF/dyn_em/module_advect_em.F:6028`.
- JAX open-top branch sets `vflux[nz]` when `top_lid=False` and adds the pickup at `src/gpuwrf/dynamics/flux_advection.py:503` through `src/gpuwrf/dynamics/flux_advection.py:515`.
- `advect_w_flux` threads the flag at `src/gpuwrf/dynamics/flux_advection.py:333` through `src/gpuwrf/dynamics/flux_advection.py:375`.
- Tests pin rigid-lid zero and open-top WRF formula at `tests/dynamics/test_advect_w_topface.py:42` through `tests/dynamics/test_advect_w_topface.py:103`.

### P1-6: guards-off operational proof

Verdict: CLOSED for the stated dycore guards-off gate.

Evidence:

- Guards are actually bypassed when `disable_guards=True`: the theta/mass/moisture guard block is conditional at `src/gpuwrf/runtime/operational_mode.py:1429` through `src/gpuwrf/runtime/operational_mode.py:1439`, and boundary finite fallback is bypassed at `src/gpuwrf/runtime/operational_mode.py:1447` through `src/gpuwrf/runtime/operational_mode.py:1466`.
- The proof reports warm bubble full gate passes guards-off and real Canary dycore finite guards-off at `proofs/sprintU/guards_off_operational_proof.json:9` through `proofs/sprintU/guards_off_operational_proof.json:31`, and `proofs/sprintU/guards_off_operational_proof.json:80` through `proofs/sprintU/guards_off_operational_proof.json:87`.
- Grep found surviving `where`/`maximum` uses, but the relevant survivors are WRF-style safe denominators, pressure floors, diff_6th monotonic limiter, or physical WRF damping; I did not find a hidden theta/mass finite fallback that still runs when `disable_guards=True`.

## New Shortcuts / Scope Boundary Assessment

- New blocker: false precision proof in P0-1. This is not just a missing artifact; it is contradicted by the proof JSON and by a direct dtype audit.
- New blocker: P0-2 is narrower than the prompt's u/v/w WRF deformation requirement. The proof is honest about the one-row limitation, but the prompt did not list full v deformation as an acceptable remaining Phase-B gap.
- Not a blocker: canonical Straka tolerances are broad (25% max|w|, 2 km front), but achieved margins are much tighter and the touchdown peak does not show the previous runaway.
- Not a blocker: documented Phase-B gaps for terrain slope terms, map factors, lateral/nested boundaries, moist/scalar RK-bundle coupling, and per-cell WRF parity are honest as scope boundaries. The additional full v deformation gap should be added explicitly if management chooses to defer it.

## Commands Run

```bash
sed -n '1,220p' PROJECT_CONSTITUTION.md
sed -n '1,240p' AGENTS.md
sed -n '1,260p' .agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-confirm-close-prompt.md
sed -n '1,240p' .agent/skills/conducting-blind-review/SKILL.md
sed -n '1,240p' .agent/skills/validating-physics/SKILL.md
git diff --stat 7a17f3e..1b7836c
git diff --name-status 7a17f3e..1b7836c
git diff --check 7a17f3e..1b7836c
PYTHONPATH=src taskset -c 0-3 python - <<'PY' ... dtype audit ... PY
PYTHONPATH=src taskset -c 0-3 pytest -q tests/dynamics/test_advect_w_topface.py tests/dynamics/test_deformation_momentum_diffusion.py
PYTHONPATH=src taskset -c 0-3 pytest -q tests/idealized/test_dycore_close_gate.py -m close_gate
rg/nl/sed inspections of Sprint U source, tests, proofs, and pristine WRF `module_advect_em.F` / `module_diffusion_em.F`
```

Test outputs:

```text
tests/dynamics/test_advect_w_topface.py tests/dynamics/test_deformation_momentum_diffusion.py: 8 passed in 6.22s
tests/idealized/test_dycore_close_gate.py -m close_gate: 2 passed in 425.80s (0:07:05)
```

Note: the close-gate pytest rewrote three tracked proof artifacts as part of its archive behavior; I restored those test-generated changes so the only intended write from this review is this report.

## Final Decision

Do not declare Sprint U operational-ready. The dycore close gate is green, and several remediation pieces are real, but P0-1's real-case fp64 unification is false and P0-2 does not implement the stated u/v/w WRF deformation operator.

SPRINTU_CONFIRM_COMPLETE
**CLOSE-REJECTED-pending-P0-1-P0-2** confidence 9/10
