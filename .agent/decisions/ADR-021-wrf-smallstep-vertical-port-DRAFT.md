# ADR-021 — WRF Small-Step Shape Vertical Port (DRAFT — alternative to ADR-022)

**Status**: DRAFT — opposing position for Codex critical-review
**Date**: 2026-05-23
**Author**: Manager (Claude Opus 4.7, 1M-context) — drafting both pivots so the critic has both fully-articulated alternatives to evaluate
**Scope**: M6.x dycore vertical-acoustic + vertical-theta-transport operator
**Triggered by**: same as ADR-022 — `2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` §4 NEEDS-HYBRID-PIVOT verdict

## Decision

Port `advance_w`, `advance_mu_t` (θ + ω terms), and `calc_coef_w` from WRF `dyn_em/module_small_step_em.F:1094-1597` **line-for-line into JAX**, expanding `AcousticScanCarry` with the seven WRF small-step scratch field families: `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `_1` family (`t_1`, `w_1`, `ph_1`, `u_1`, `v_1`), `_save` family (`mu_save`, `t_save`, `w_save`, `ph_save`, `u_save`, `v_save`). Hold Tier-1 WRF-savepoint parity as the binding gate alongside Tier-4 RMSE.

Keep the c2-A2 horizontal PGF and mu continuity **unchanged** under this ADR as well (same as ADR-022).

## Rationale

Three lines of evidence support this alternative:

1. **Gemini methodology step-back** (`2026-05-22-c2-methodology-stepback/worker-report.md §4`): "The WRF-port split-explicit direction is the fastest and only viable path to a 3 km operational forecast on a single RTX 5090. Since the regional boundaries are pre-determined by WRF Gen2 files, any deviation from WRF's grid nesting, hybrid-eta coordinate, or split-explicit acoustic formulation introduces boundary/coordinate interpolation mismatches that will cause immediate gravity-wave blowup at the boundaries."
2. **Tier-1 oracle integrity.** WRF small-step savepoints are extractable; once carried correctly they make every operator step independently testable against a numerical reference. This is the anti-tautology property the M5 cycle proved load-bearing — every "real X" worker label that survived was the one that linked to a real WRF callable and compared against real WRF outputs.
3. **The carry expansion is a one-time cost.** Once `t_2ave`, `ww`, `muave`, `ph_tend`, and the `_1`/`_save` families are in the carry, every subsequent dycore feature (Klemp-Skamarock divergence damping, smdiv memory, ADR-014 surface-coupled fields) inherits them. ADR-022's hybrid pivot pays the carry-simplicity dividend once but loses the WRF-parity anti-tautology dividend forever.

## Specification

### 1. `AcousticScanCarry` expansion

The 5-tuple becomes a structured pytree with named leaves:

```
AcousticScanCarry = {
  state, previous_pressure, al, alt, cqu, cqv,       # c2-A2 base
  t_2ave, ww, muave, muts, ph_tend,                  # WRF small-step intermediates
  t_1, w_1, ph_1, u_1, v_1,                          # large-step states for off-centering
  mu_save, t_save, w_save, ph_save, u_save, v_save,  # large-step originals for small_step_finish
}
```

Per ADR-020 §"Intermediate Fields Policy", these are **scan-carried intermediates, not `State` leaves.** They live for the duration of the acoustic scan and are reconstructed each RK stage.

### 2. `advance_w` line-for-line port

Vertical w/φ RHS structure mirrors `module_small_step_em.F:1340-1489, 1533-1584`:

```
rhs_phi(k)   = dts * (ph_tend(k) + 0.5*g*(1-epssm)*w(k)) - ww*∂φ/∂η  
buoyancy(k)  = dts*g*msft_inv*( rdn(k)*(c2a(k)*alt(k)*t_2ave(k) 
                              - c2a(k-1)*alt(k-1)*t_2ave(k-1)) 
                              - c1f(k)*muave )
ph_coupling  = (1-epssm)*(ph(k+1)-ph(k)) + (1+epssm)*(rhs(k+1)-rhs(k))   [implicit]
              * c2a*rdnw / ((c1h*MUT+c2h)*(c1f*MUT+c2f))
ph_update    = ph_new(k) = rhs(k) + msfty*0.5*dts*g*(1+epssm)*w(k)/(c1f*muts+c2f)
```

Every term cited above is read directly from the WRF line range; no JAX-side simplification.

### 3. `advance_mu_t` theta + omega path

Build `ww(k)` from continuity (`module_small_step_em.F:1109-1114`) using the divergence of coupled momentum across the column. Build `wdtn(k) = ww(k)*(fnm(k)*t_1(k) + fnp(k)*t_1(k-1))` for face θ transport on mass levels. Apply `t -= dts*msfty*(advect_terms + rdnw(k)*(wdtn(k+1)-wdtn(k)))` per WRF lines 1148-1173.

### 4. `calc_coef_w` per-entry hybrid denominators

Same as ADR-022 §1 — closes R3 either way.

### 5. Test oracle

A WRF savepoint exposing `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and the `_1`/`_save` family at multiple substep boundaries is the **mandatory** anti-tautology test. This means a one-time investment in a Fortran harness that runs WRF's small-step prep → first acoustic substep → small_step_finish and dumps the intermediates. The harness pattern already exists for M5-S1 / M5-S2 / M5-S3 (Thompson, MYNN, RRTMG); extending it to small_step_em is mechanically similar.

## Constraints

- Same transfer-audit, precision, ADR-001/002/003/007 constraints as ADR-022.
- **Tier-1 WRF-savepoint parity is binding** (the inverse of ADR-022's relaxation).
- WRF Fortran harness for `module_small_step_em.F` is a **prerequisite** for the implementation sprint — must land before the worker dispatches.

## Trade-offs vs ADR-022

(See ADR-022 §Trade-offs — same table read the other way.)

Plus:

- **WRF Fortran harness build cost**: 1-2 worker-days of NVIDIA HPC SDK Fortran integration work that ADR-022 avoids entirely.
- **Carry-tied recurrence**: every future dycore feature must either consume from or contribute to the expanded carry; this is the contract change the arch step-back warned about.
- **Maintenance cost long-term**: tracking WRF source as it evolves (currently pinned to v4.7.1) costs ongoing review effort that ADR-022 sidesteps.

## Evidence

- `2026-05-22-c2-methodology-stepback/worker-report.md §4` (Gemini argues this path)
- `2026-05-22-c2-architecture-stepback/worker-report.md §3 c2-continue row` (codex: 65% / 15-35× / 120-220 agent-hours)
- WRF source `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` lines 1094-1597 (advance_w, advance_mu_t, calc_coef_w)
- M5-S1 / M5-S2 / M5-S3 Fortran harness pattern as the prior art for the WRF callable wrapper

## Open questions for the manager and critic

1. Is the WRF small-step Fortran harness genuinely a 1-2 day build, or — given M5-S3 RRTMG took 5+ attempts to bind real driver subroutines vs reimplement — should the budget be 3-5 days with reviewer cycles?
2. The "_1"/"_save" families implicitly assume the RK3 cadence will be replaced by WRF's split-explicit cadence. Is that the manager's intent, or does the RK3 outer loop survive?
3. Does the Gemini methodology critique on "deviation from WRF causes boundary blowup" actually hold when our boundary forcing comes from a CPU WRF wrfbdy and the deviation is internal to the column step? The c2-A2 horizontal PGF *is* line-for-line WRF; only the vertical operator deviates under ADR-022.

## Status

DRAFT — opposing position. The Codex critic should evaluate both ADR-021 and ADR-022 and return a recommendation. The manager will ratify one to PROPOSED based on the critique.
