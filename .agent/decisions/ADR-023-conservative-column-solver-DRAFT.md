# ADR-023 — Conservative Column Solver (DRAFT — third option from scout RECOMMEND-THIRD-OPTION)

**Status**: DRAFT — replaces ADR-022 as the manager's working recommendation pending the second critic round
**Date**: 2026-05-23
**Author**: Manager (Claude Opus 4.7, 1M-context), based on `2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md` (codex scout, 2026-05-23)
**Scope**: M6.x dycore vertical-acoustic + vertical-theta-transport operator
**Triggered by**: scout return `RECOMMEND-THIRD-OPTION` + critic claimed `RATIFY-EITHER-WITH-CONDITIONS` — both agree neither ADR-021 nor ADR-022 cleanly maps to successful GPU-NWP precedent
**Supersedes (if ratified)**: ADR-022-DRAFT

## Decision

Keep ADR-022's JAX backend, small `AcousticScanCarry` (no expansion to WRF small-step scratch families), and ADR-020 architecture skeleton verbatim. **Replace ADR-022's simplified `ph_new = ph_old + dt * g * w_new` with a conservative tridiagonal column solver over the coupled `(w, mu, theta, phi/exner)` system**, patterned after the strongest external GPU-NWP precedents:

- **SCREAM / HOMME** `DirkFunctorImpl.hpp:344-356, 707-778` — DIRK with Newton outer + cyclic-reduction / Thomas tridiagonal Jacobian solve.
- **ICON4Py** `vertically_implicit_dycore_solver.py:139-206, 283-356`; `solve_tridiagonal_matrix_for_w_forward_sweep.py:18-68` — explicit tridiagonal forward sweep + back substitution for w.
- **MPAS-A** `mpas_atm_time_integration.F:1461-1656, 2141-2208` — Klemp et al. MWR 2007 forward-backward vertically implicit integration, up-sweep + down-sweep + back-substitution.

This is the scout's recommended "ADR-022b" path. The implementation is **smaller than ADR-021** (no carry expansion, no WRF Fortran harness, no `_1`/`_save` field families) and **safer than ADR-022** (uses a conservative column solve rather than a simplified geopotential update that drops `msfty/mu`, `(1+epssm)²` off-centering, and `c2a*alt*t_2ave` buoyancy weighting).

## Rationale

Three converging lines of external evidence (all from `m6x-dycore-alt-methods-scout/worker-report.md`):

1. **No successful GPU-NWP project carries WRF small-step scratch in its scan API.** Pace wraps the vertical solve as `NonhydrostaticVerticalSolver` (`dyn_core.py:472-479`); MPAS uses perturbation variables `ru_p`, `rw_p`, `rtheta_pp`, `rho_pp` inside its acoustic step (`mpas_atm_time_integration.F:1824-1833, 2141-2208`) but as local in-step variables, not as a permanent carry expansion. ADR-021's seven WRF-scratch carry families have no precedent in any operational or pre-operational GPU port the scout found.

2. **The SCREAM/HOMME HEVI-IMEX column solve is the strongest production GPU evidence**, and it is **not** a simplified `ph += dt*g*w`. It carries nonlinear EOS pressure/exner coupling, forms a tridiagonal Jacobian analytically, and emits explicit Newton-failure warnings (`DirkFunctorImpl.hpp:384-390`). ADR-022 as written drops half of this and asserts the omitted terms are "tiny"; the scout shows that omission is not supported by external evidence.

3. **ICON4Py + MPAS together cover the entire tridiagonal column solver design space.** ICON4Py's forward-sweep + back-substitution is the canonical JAX-friendly shape; MPAS's Klemp et al. 2007 forward-backward integration is the canonical hybrid-eta-coordinate shape. ADR-023 inherits both — ICON4Py's algorithmic structure under `lax.scan`/`vmap`, MPAS's variable-perturbation form on the eta grid.

## Specification

### 1. Column-solver structure

Per acoustic substep, for each (i, j) column, solve the coupled tridiagonal system:

```
F(w, phi/exner, theta', mu')_implicit + F(...)_explicit = 0
```

with the implicit (column-coupled) terms being:

- **w equation**: `w_new = w_old + dt * (-(c2a*alt)/mu_total * grad_z(p') - g*theta'/theta_base)` with `grad_z` solved implicitly via tridiagonal.
- **phi equation**: `phi_new = phi_old + dt * g * 0.5 * ((1-epssm)*w_old + (1+epssm)*w_new)` — off-centered Crank-Nicolson form following MPAS, NOT the simplified ADR-022 form.
- **theta' equation**: vertical advection of theta' by w_new, solved on faces using `fnm/fnp` interpolation coefficients.
- **mu' equation**: column-integrated continuity, solved by the existing `mu_continuity_tendency` (preserved verbatim from c2-A2).

The tridiagonal coefficient builder uses **per-entry hybrid-eta denominators** matching WRF lines 626/632/637-639/646 — same as ADR-022 §1 and ADR-021 — but the coefficients flow into a Newton/tridiagonal solve rather than a w-only solve.

### 2. Solver implementation in JAX

Algorithm:
1. Build (a, b, c, rhs) tridiagonal entries per column using `jnp.vectorize` / `jax.vmap`.
2. Forward sweep + back substitution (Thomas algorithm) implemented as a `lax.scan` over the vertical dimension.
3. For GPU: cyclic-reduction (CR) tridiagonal solve from the same XLA primitive wrapper already in use (`src/gpuwrf/numerics/tridiagonal_solver.py` from M5-S2).
4. No Newton outer loop in v0; the implicit acoustic-gravity system is linear-enough on flat hydrostatic perturbations that a single tridiagonal pass is sufficient. Newton is reserved for a future ADR if RMSE budget demands.

### 3. AcousticScanCarry — no expansion

Same 5-tuple `(state, previous_pressure, al, alt, cqu, cqv)` as ADR-022. This is the architectural payoff inherited from ADR-022.

### 4. Other inherited fixes

- **R3 hybrid-eta denominators**: built into the new tridiagonal coefficient builder.
- **R4 msf-factor in `uncouple_horizontal_pgf_tendency`**: same as ADR-022 §5.
- **R7 analytic oracle**: already in `tests/test_m6x_vertical_acoustic_oracle.py` (3 RED tests). The implementation must turn these green.
- **R8 layer thickness**: operator uses `rdnw`, not `dz_m`.
- **R9 `top_lid`**: honored in coefficient builder; `w(nz) = 0` enforced post-solve.
- **R10**: drop defensive `abs(...)` and clamp bandaids in production paths.

### 5. Off-centering parameter

`epssm` is an `AcousticConfig` field, default 0.1 (MPAS-canonical), wired through the coefficient builder. ADR-022 hard-coded epssm=0 and dropped Crank-Nicolson; ADR-023 restores both.

## Constraints

- No host/device transfer inside the timestep loop (transfer-audit gate remains binding).
- fp64 for pressure / mass / geopotential; fp32 acceptable for θ' per ADR-007.
- The horizontal PGF path from c2-A2 is preserved verbatim except for the R4 msf-factor multiplication.
- **Tier-1 WRF-savepoint parity is relaxed but not abandoned**: the column solver is conservative and Crank-Nicolson off-centered; once implemented, savepoint comparison against MPAS (which uses Klemp 2007 forward-backward, mathematically equivalent to our CN with epssm=0.1) should match within ~5% at the column level. ADR-022's "Tier-1 not binding" rule is softened.
- Tier-4 RMSE on U10/V10/T2 at 24h/72h vs Gen2 backfill is the binding acceptance gate.
- The R7 oracle tests in `tests/test_m6x_vertical_acoustic_oracle.py` must turn green before the sprint can close.

## Trade-offs vs ADR-021 and ADR-022

| Dimension | ADR-021 WRF port | ADR-022 hybrid simple | **ADR-023 conservative** |
|---|---|---|---|
| `AcousticScanCarry` expansion | 7 new field families | none | **none** |
| Tier-1 WRF parity | binding | not binding | **softly binding (~5% vs MPAS Klemp 2007)** |
| Tier-4 RMSE binding | yes | yes | **yes** |
| Worker-time to warm-bubble PASS | 5-9 days | 2-4 days | **3-5 days (oracle test already on disk)** |
| Risk of late "missing-term" discovery | medium-high | medium | **low (conservative column form is well-understood)** |
| Operational GPU-NWP precedent | MPAS (Fortran/OpenACC) | Dinosaur/NeuralGCM (global spectral) | **SCREAM HEVI + ICON4Py + MPAS Klemp 2007 (3 supports)** |
| Future portability beyond WRF baseline | low | high | **high** |
| ADR-007 4× target compat | conditional | conditional | **conditional, same gates** |

## Evidence

All from `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`:
- §1 comparative table (SCREAM, ICON4Py, MPAS, Pace, Dinosaur, NeuralGCM)
- §2 pattern 1 (split-explicit + tridiagonal) and pattern 2 (HEVI-IMEX) both ratify the conservative column form
- §3 "ADR-022 should not claim the omitted mass/geopotential weighting is tiny without either a 1-D column oracle or a WRF/MPAS-derived fixture"
- §5 explicit `RECOMMEND-THIRD-OPTION` text proposing this exact ADR

Plus:
- R7 analytic oracle now on disk (`tests/test_m6x_vertical_acoustic_oracle.py`, commit `f6965be`)
- ADR-022-DRAFT and ADR-021-DRAFT for trade-off comparison

## Open questions for the second critic round

1. **Is the linear single-tridiagonal-pass sufficient, or is a Newton outer required?** Scout cited SCREAM uses Newton; ADR-023 v0 omits it on the linearity argument. Verify against the R7 analytic oracle's amplitude-decay test — if a Newton outer would change the verdict, escalate.
2. **Off-centering parameter `epssm = 0.1` (MPAS default).** Is this defensible for the WRF baseline target, or should it be `epssm = 0.0` (centered, matches ADR-022's implicit choice) or `epssm = 0.3` (WRF Klemp default)?
3. **Tridiagonal solver: forward-sweep `lax.scan` vs cyclic-reduction XLA primitive.** Performance trade-off — `lax.scan` is portable but sequential along z; CR is parallel but more code. ADR-023 v0 specifies CR; reconsider if launch budget breaks.
4. **Does the ADR-023 path require the WRF Fortran harness?** v0 says no — the analytic oracle plus MPAS savepoint (if extractable from local MPAS source) suffices. Verify the local MPAS source `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/` is harness-extractable.

## Status

DRAFT — awaiting second critic round comparing ADR-021 vs ADR-022 vs ADR-023. Critic must return one of: `RATIFY-ADR-023`, `RATIFY-ADR-022`, `RATIFY-ADR-021`, or `RATIFY-NEITHER` (with proposed fourth option). Manager target: ratify within 24h of the critic return.

In parallel, an **implementation prototype worker** is dispatched on a single large sprint to build the ADR-023 column solver against the R7 oracle tests. This is the user-mandated "rewrite-with-different-method, testable after one large sprint" path. If the prototype turns the R7 tests green, the critic decision is informed by code-running evidence rather than paper analysis.
