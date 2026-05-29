# Sprint Contract — F7F: WRF-balanced idealized IC (ph rebalance) + remove synthetic p_buoy → idealized cases PASS (dycore close)

**Sprint ID**: `2026-05-29-f7f-balanced-ic-and-real-pbuoy`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7d-pressure-mass-fix` (CONTINUE on this branch — it carries the verified F7D mass-semantics fix; this sprint validates the whole chain via the idealized cases, then the manager merges it)
**GPU**: YES — `taskset -c 0-3`; confirm `cuda:0`; fp64.
**Builds on**: Sprints A–D (acoustic core + damping + advection + circulation + MUT/MUTS mass semantics, all on this branch).

## Binding spec
The authoritative fix is the GPT-5.5 WRF-verified fork resolution: **`.agent/sprints/2026-05-29-f7e-ic-vs-dycore-fork/gpt-fork-findings.md` §3** — read it in full. It settled a pivotal fork from WRF source. Two prior over-simplified hypotheses were BOTH refuted: there is **no mu' iteration** (mu'=0 is correct for dry cases) and **no `pg_buoy_w` formula change** (the JAX dry formula already matches WRF). Implement exactly the §3 spec.

## The verified root cause (two interacting bugs)
1. **IC bug**: the idealized IC applies the θ perturbation but leaves geopotential `ph` at the base value. WRF (`module_initialize_ideal.F:1103-1130`, `:1278-1313`) perturbs θ then **rebalances `ph_1/ph_2/ph0` hydrostatically at fixed column dry mass** (mu'=0). Without the `ph` rebalance there is no physical perturbation pressure, so the bubble is "dead" on the real pressure path.
2. **Operator bug (the Sprint-B hack)**: because the unbalanced IC gave a dead bubble, Sprint B fed `pg_buoy_w` a **synthetic absolute** θ-derived pressure `p_buoy` (`src/gpuwrf/runtime/operational_mode.py:664-681`, given priority in `src/gpuwrf/dynamics/core/acoustic.py:523-531`). That synthetic pressure is NOT what WRF passes to `pg_buoy_w` and over-forces w by 9.4×. WRF passes the actual perturbation-pressure diagnostic `grid%p`.

These compound: fix the IC `ph` rebalance AND remove the synthetic `p_buoy` together (removing the hack without rebalancing `ph` → dead bubble; rebalancing `ph` without removing the hack → still over-forced).

## Cardinal rule
WRF Fortran source is ground truth. Verify every change against `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_initialize_ideal.F` (the `quarter_ss`/`squall2d`/`grav2d_x` setups), `module_big_step_utilities_em.F` (`pg_buoy_w`, `calc_p_rho_phi`), `module_small_step_em.F` (`calc_p_rho`). If anything disagrees, WRF wins — note it.

## Scope — implement the GPT-5.5 §3 fix spec exactly

1. **`src/gpuwrf/ic_generators/idealized.py`** — WRF fixed-mass balance, explicit + testable:
   - Keep `mu_perturbation = 0` for dry warm bubble and density current.
   - After applying the θ perturbation, recompute `alt_full = EOS(theta_full, pb+p)`, `al = alt_full − alb`.
   - Integrate the perturbation geopotential from the lower boundary using WRF's hydrostatic recurrence (`module_initialize_ideal.F:1123-1129` / `:1305-1313`). Store `ph_perturbation = ph_full − phb`, `ph_total = phb + ph_perturbation`.
   - Keep initial dry `p_perturbation` as WRF's fixed-mass diagnostic (zero for the dry pure-sigma neutral-base cases at t=0).
   - Remove/rewrite the comments claiming "base ph + θ is the buoyancy source" — WRF's source is the balanced perturbation fields.
2. **`src/gpuwrf/runtime/operational_mode.py:664-681`** — REMOVE the synthetic `p_buoy_abs` construction. `pg_buoy_w_dry` must consume WRF's actual perturbation-pressure diagnostic (`state.p_perturbation` = WRF `grid%p`), not a second θ-derived pressure. Set `p_buoy=None` or `state.p_perturbation` (only if it is the WRF `grid%p` from the current RK stage). Keep `calc_p_rho_wrf`/`calc_p_rho_step` as the small-step pressure source.
3. **`src/gpuwrf/dynamics/core/rk_addtend_dry.py:93-133`** (`_absolute_diagnostics`) — stop deriving `p_abs` from absolute θ. Use `state.p_perturbation` for WRF `p`; compute `al` from `ph_perturbation` and `mu_perturbation`; keep `alt` from EOS and `php` from full `ph`. (Preserves WRF horizontal PGF inputs without inventing a vertical pressure source.)

## Acceptance gates (all required for `F7F_COMPLETE` = F7 dycore close) — GPT-5.5 §4 falsifiable checks

- **AC1 — frozen-buoyancy sanity** (`proofs/f7f/rwtend_after_fix.json`): warm bubble with the WRF-balanced fixed-mass IC has `max_abs(c1f·mu') = 0`, and the direct stage-constant `pg_buoy_w` source `max_abs_rw_phys < 0.01 m/s²` (NOT 0.615). I.e. the 9.4× frozen over-forcing is gone.
- **AC2 — negative control**: a deliberately-bad IC (θ perturbation but base `ph`, no rebalance) **reproduces** the large p′ artifact / over-forcing — proving the checker can fail (not a tautology).
- **AC3 — Skamarock warm bubble PASS**: finite through 500 s with **no** linear `max|w| ≈ 0.615·t` growth; the thermal **rises** (centroid ≥ 500 m — i.e. NOT dead), θ′ transported, max|w| in the physical range; symmetric; mass-conserving.
- **AC4 — Straka density current PASS**: finite through 900 s; front position ≈ 15 km (±~2 km) on the contracted grid; min θ′ ≈ −9..−10 K; max|w| O(10); mass drift ≤ 1e-8.
- **AC5 — no regression**: Sprint A/B/C/D gates (no-stub, flat-rest=0, analytic dipole, 300-step conservation, circulation, MUT/MUTS mass identities) re-run and hold; no test weakened/xfailed. Report the d02 operational-dt audit (F7D moved first_critical 8→5) — note whether the balanced-ph/real-p_buoy change improves it.

## Proof objects (into `proofs/f7f/`)
`rwtend_after_fix.json` (AC1), `negative_control_unbalanced_ic.json` (AC2), `straka_density_current.json`+verdict+plots, `skamarock_warm_bubble.json`+verdict+plots (AC3/AC4), `ic_balance_proof.md` (the WRF ph-rebalance derivation + before/after), `regression_recheck.json` (AC5), `worker-report.md` (AGENTS.md format) ending `F7F_COMPLETE` or `F7F_PARTIAL` + precise gaps.

## Hard rules
1. `taskset -c 0-3`; `cuda:0`; fp64.
2. WRF source is ground truth; cite `file:line` in every changed function docstring.
3. **No masking clamps/caps/sanitizers, no `pg_buoy_w` coefficient tuning.** The fix is field-consistency (balanced IC + real perturbation pressure), per GPT-5.5. If the cases still misbehave after the spec'd fix, report it honestly with traces — do NOT paper over with a clamp or a coefficient fudge.
4. **No performance work** (no fp32/fusion).
5. Commit incrementally on `worker/opus/f7d-pressure-mass-fix`; do not push; do not switch branches.
6. Files writable: `src/gpuwrf/ic_generators/idealized.py`, `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/dynamics/core/rk_addtend_dry.py`, other `src/gpuwrf/dynamics/**` only if the spec requires, `scripts/**` (instrumentation, never weaken invariants), `tests/**` (add/fix, never weaken), `proofs/f7f/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, comparator scripts under `scripts/m6b6_*`.
8. If the cases still don't pass after the verified fix, deliver the largest gated subset, mark `F7F_PARTIAL`, and report with substep pressure/ph traces exactly what residual remains — that would be a critical signal (escalation to agy/full-council).

## Forward pointer
- On AC3+AC4 PASS → manager runs the GPT-5.5 WRF-domain **pre-close code critique** of the whole dycore, then declares the **F7 dynamical-core milestone closed**, merges the `f7d` chain to `manager-2026-05-23`.
- Then **F7-perf** (XLA fusion + fp32 downcast + ≥10× speedup recert) and **M9** (instrumented WRF savepoints → per-operator parity = the rigorous near-identical-RMSE-vs-real-WRF gate).
