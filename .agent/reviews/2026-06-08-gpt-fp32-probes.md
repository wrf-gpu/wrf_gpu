# GPT FP32 Acoustic Probes

Date: 2026-06-08
Worker: GPT-5.5 xhigh (codex)
Branch: `worker/gpt/v014-fp32-probes`
Scope: CPU-only numerical probes and report. No source code changes. No GPU jobs.

## objective

Create CPU-only proof probes for the v0.14 mixed-fp32 acoustic lane:

- demonstrate absolute-total fp32 cancellation,
- demonstrate perturbation-form fp32 preservation,
- demonstrate a small recurrence sensitivity mechanism,
- identify first fp64 islands and plausible resident fp32 candidates,
- quantify resident acoustic/carry memory savings,
- recommend v0.13 pull-in vs v0.14-only.

## files changed

- `proofs/v014/fp32_acoustic_probes.py`
- `proofs/v014/fp32_acoustic_probes.json`
- `.agent/reviews/2026-06-08-gpt-fp32-probes.md`

No files under `src/` were modified.

## commands run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,260p' .agent/sprints/2026-06-08-v014-fp32-acoustic-derisk/sprint-contract.md`
- `find .agent/skills -maxdepth 2 -type f -name 'SKILL.md' -print`
- `sed -n '1,240p' .agent/skills/validating-physics/SKILL.md`
- `sed -n '1,220p' .agent/skills/designing-gpu-state/SKILL.md`
- `sed -n '1,200p' .agent/skills/reporting-to-human/SKILL.md`
- `sed -n '1,280p' .agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `sed -n '1,280p' .agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md`
- `nl -ba src/gpuwrf/dynamics/core/calc_p_rho.py | sed -n '1,280p'`
- `nl -ba src/gpuwrf/dynamics/core/small_step_prep.py | sed -n '1,420p'`
- `nl -ba src/gpuwrf/dynamics/core/small_step_finish.py | sed -n '1,320p'`
- `nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '1,720p'`
- `nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '720,980p'`
- `nl -ba src/gpuwrf/dynamics/core/advance_w.py | sed -n '1,360p'`
- `nl -ba src/gpuwrf/dynamics/core/advance_w.py | sed -n '360,760p'`
- `git status --short`
- `git branch --show-current`
- `find proofs/v014 -maxdepth 2 -type f -print`
- `find .agent/reviews -maxdepth 1 -type f -name '*fp32*' -print`
- `rg -n "fp32_acoustic|memory-savings|acoustic_precision|mixed_perturb" proofs .agent src/gpuwrf | head -n 200`
- `mkdir -p proofs/v014`
- `python proofs/v014/fp32_acoustic_probes.py`
- `python -m json.tool proofs/v014/fp32_acoustic_probes.json | sed -n '1,220p'`
- `python -m json.tool proofs/v014/fp32_acoustic_probes.json | rg -n "millipascal|fp32_ulp|p_error_ratio|ph_error_ratio|saving_mib|supports_v013|recommended_scope|final_linf|errors_vs|core_candidate_set|core_plus_prep"`
- `python proofs/v014/fp32_acoustic_probes.py`
- `sed -n '1,260p' .agent/reviews/2026-06-08-gpt-fp32-probes.md`
- `wc -l proofs/v014/fp32_acoustic_probes.py proofs/v014/fp32_acoustic_probes.json`
- `git diff --stat`
- `git diff --check`
- `python -m py_compile proofs/v014/fp32_acoustic_probes.py`
- `git status --short`
- `find proofs/v014 -maxdepth 2 -type f -print`
- `rm -rf proofs/v014/__pycache__`

## proof objects produced

- `proofs/v014/fp32_acoustic_probes.py`
- `proofs/v014/fp32_acoustic_probes.json`

The proof script imports NumPy only; it does not import JAX and does not touch CUDA/GPU.

Key probe results:

- Absolute-total fp32 pressure at `90100 Pa` has ULP `0.0078125 Pa`; a `0.001 Pa` acoustic update is recovered as `0.0 Pa`.
- Perturbation-form fp32 at `p' = 100 Pa` has ULP `7.62939453125e-06 Pa`; the same `0.001 Pa` update is recovered as `0.00099945068359375 Pa` with relative error `-5.493e-4`.
- In the small 1D acoustic recurrence, every reference pressure update is at most `0.000195 Pa`, only `0.02498` of an absolute-total fp32 ULP at `90000 Pa`.
- Final recurrence pressure max: fp64 reference `0.004099 Pa`, absolute-total32 `0.0 Pa`, perturbation32 `0.004099 Pa`.
- Recurrence L2 error vs fp64: pressure absolute-total32 `2.9125e-3 Pa`, perturbation32 `2.1396e-9 Pa`, ratio `1.361e6`.
- Recurrence geopotential L2 error ratio absolute-total32 over perturbation32: `1.841e6`.

The probes are mechanism evidence only. They are not WRF fixture parity, production forecast equivalence, profiler evidence, or a transfer audit.

## source audit notes

- `small_step_prep.py` still recovers `mub` as `state.mu_total - state.mu_perturbation` and `pb` as `state.p_total - state.p_perturbation`; `php` is likewise built from `state.ph_total - state.ph_perturbation`. Those are fp32-hostile if totals are demoted before explicit base plumbing.
- `small_step_finish.py` reconstructs `p_base`, `ph_base`, and `mu_base` from totals minus perturbations before rebuilding totals. This must not be the mixed-mode authority path.
- `calc_p_rho.py` is already mostly perturbation/work form, but its `mass_h`, `al`, EOS bracket, `p`, and `pm1` smdiv memory update should remain a first fp64 arithmetic island.
- `acoustic.py` pressure-gradient terms combine live `p/ph`, base pressure, `al/alt`, `php_stage`, `dpn`, and mass brackets. Store candidates may be fp32 later, but the accumulation should stay fp64 first.
- `advance_w.py` builds an implicit vertical solve around `c2a`, `alt`, mass coefficients, RHS, Thomas coefficients, `w`, and `ph_next`; this is a first fp64 island until isolated demotion proof exists.

## first fp64 islands

- Explicit base/reference fields: `pb/p_base`, `phb/ph_base`, `mub`, `php_stage`, and final total reconstruction.
- `calc_p_rho` local arithmetic: `mass_h`, `safe_mass`, hydrostatic `al`, EOS pressure bracket, and `pm1/smdiv` update.
- Horizontal pressure-gradient accumulation in `advance_uv_wrf`.
- Implicit-w coefficient build and Thomas solve: `a`, `alpha`, `gamma`, RHS, `t_2ave`, and `ph_next` local arithmetic.
- Terrain lower-boundary / terrain pressure-gradient terms.
- Boundary and nesting forcing leaves: `u_work_bdy`, `v_work_bdy`, `ph_bdy_target`, `ph_save_for_spec`, `rw_tend_pg_buoy`.
- Diagnostics, restart, history, and any interface that reconstructs absolute totals.

## plausible fp32 candidates

After R1 explicit base plumbing and R2 perturbation-authoritative state, plausible resident fp32 candidates are:

- `p`, `pm1` storage, while keeping `calc_p_rho` local arithmetic fp64 first.
- `ph`, `ph_work` storage, while keeping PGF and implicit-w local arithmetic fp64 first.
- `mu`, `muts`, `muave`, `mudf`, `mu_work`, and mass-flux carry storage.
- Coupled acoustic work arrays `u`, `v`, `w`, `ww`, `theta_coupled_work`, `theta_work`, `t_2ave`.
- Sumflux/carry arrays `ru_m`, `rv_m`, `ww_m` and same-shape save arrays where not used as fp64 boundary references.

## memory savings

Formula: demoting one resident array from fp64 to fp32 saves `4 * element_count` bytes.

For grid `nx=641`, `ny=321`, `nz=50`:

- mass 3D array: `nz * ny * nx = 10,288,050` elements, saving `39.25 MiB`.
- vertical-face 3D array: `(nz + 1) * ny * nx = 10,493,811` elements, saving `40.03 MiB`.
- x-staggered 3D array: `nz * ny * (nx + 1) = 10,304,100` elements, saving `39.31 MiB`.
- y-staggered 3D array: `nz * (ny + 1) * nx = 10,320,100` elements, saving `39.37 MiB`.

Concrete candidate-set arithmetic from the JSON:

- Core acoustic candidate set, 23 arrays: `754.07 MiB` saved if resident fp64 storage is demoted to fp32.
- Prep/carry candidate set, 13 arrays: `437.57 MiB` additional arithmetic savings.
- Combined core plus prep/carry candidate set: `1191.63 MiB` arithmetic savings.

This is not a measured VRAM claim. Real savings depend on JAX liveness, aliasing, donation, compile choices, and which fp64 islands are local temporaries versus resident buffers.

## v0.13 pull-in recommendation

Do not pull into v0.13.

The evidence supports the numerical mechanism for a v0.14 mixed perturbation-authoritative acoustic lane. It does not satisfy the sprint contract's minimum v0.13 proof standard: no source integration, no default-fp64 bit-inertness proof, no WRF fixture parity, no source dtype trace for a real mixed path, no transfer audit, and no production forecast validation. v0.13 should remain fp64.

## unresolved risks

- The probes isolate cancellation; they do not prove the current acoustic kernels are stable under mixed storage.
- Terrain PGF and surface-w coupling remain high-risk and should keep fp64 local arithmetic initially.
- Nested `ph'` and normal-momentum boundary forcing are high-risk and should not be demoted in the first mixed pass.
- `calc_p_rho` may allow fp32 resident `p/pm1`, but only with fp64 local arithmetic and explicit cast boundaries.
- Memory savings are arithmetic estimates, not profiler or HBM residency artifacts.

## next decision needed

Keep this as v0.14-only. If v0.13 fp64 release gates close cleanly, approve R0/R1/R3 work: precision-mode ADR, explicit base-state plumbing, perturbation-authoritative acoustic storage contract, and CPU/source audit gates before any GPU smoke.

GPT FP32 PROBES DONE
