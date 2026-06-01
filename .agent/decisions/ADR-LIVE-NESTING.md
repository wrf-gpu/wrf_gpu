# ADR-031 — Live two-way nesting (d01 9 km → d02 3 km → d03 1 km) on a single GPU

Status: **APPROVED for implementation (manager, 2026-06-01)** — Phase 0 + Phase 1 (one-way d02→d03), built worktree-isolated, GPU gates serialized behind the running consolidation (ONE GPU job at a time).

**Manager approval notes / §10 decisions:**
1. **Reframe (important):** the +2.6 kPa drift this ADR was written to cure is **ALREADY FIXED** independently by the base-`alb`/`phb`-inversion dycore fix (`6d284ba`/`75b9b40`) — d03 reaches 0.72 K T2 *without* nesting. So live nesting's justification is now the **operational NESTED replacement (the principal's explicit v0.2.0 mandate)**, not the bias cure. Gate **V3 is reframed** from "collapse the +2.6 kPa" to "**live nesting MAINTAINS the now-correct pressure** (no re-introduced drift, no w-pump, T2 stays sub-1 K)". The in-loop ph/w machinery may need less aggressive forcing now that the interior is correctly balanced — verify empirically (the toggles may even be near-no-ops if the re-synced interior is already balanced).
2. Same-vertical nests (44 levels all domains) → **DIRECT parent-ph interpolation** at force time; hydrostatic recompute only as a consistency fallback (O2/T4).
3. Memory: schedule **(A) co-resident-serialized** default; **(B) staged-offload** ADR-approved fallback for the d01 extent.
4. Two-way feedback = **Phase-3-optional** behind a default-off flag; ship one-way if its gate (V7) fails (one-way delivers the operational product).
5. **O0 RESOLVED:** the GPT WRF-nesting-architecture review landed — `.agent/reviews/2026-06-01-gpt-wrf-nesting-architecture.md` (driver/subcycling `module_integrate.F`, force, and feedback `copy_fcn` area-averaging + the full feedback field set). Reconcile its file:lines into §2/§3.3/§3.4 during implementation.
Date: 2026-06-01
Author: Opus 4.8 MAX (worker/opus/final-verdict)
Supersedes (for the operational real-case path): the d02/d03 hourly side-history REPLAY
(`scripts/d03_replay.py`, `integration/d02_replay.py` `boundary_domain=` path, `BoundaryConfig.force_geopotential=False`).
Relates to: ADR-011 (M6 shared IO + boundary replay), ADR-023 (conservative column solver / operational dycore),
ADR-026 (operational-mode design), ADR-002 (state layout), the v0.2.0 plan (P0-1 promoted to CORE).

> **Scope discipline.** This ADR is a reviewable design. It does **not** touch
> `runtime/operational_mode.py` (active sibling sprint), `publish/paper/**` (principal's codex),
> or the v0.1.0 release. No production code is changed by this document.

---

## 1. Context & decision

### 1.1 Why this ADR exists now

The principal (2026-06-01) redefined v0.2.0 as **a fully-stable, live-NESTED (d01→d02→d03),
no-compromise operational replacement for the real Canary runs.** Live nesting (P0-1) was
promoted from optional/Wave-3 to a **CORE v0.2.0 requirement** because it is *both* the
explicit deliverable ("nesting for our runs") *and* the proper, WRF-faithful fix for the d03
1 km quality defect documented below.

### 1.2 The defect live nesting fixes — the d03 +2.6 kPa / Exner-T2 drift (root cause, proven)

Three 2026-06-01 RCAs converge on a single, *architectural* root cause that is **not** a
dycore bug, not a surface-flux bug, and not a similarity-function bug:

- **Bisection** (`.agent/reviews/2026-06-01-opus-t2bias-bisection.md`): the d03 1 km +1.5 K
  T2 warm bias is an **atmosphere-side diagnostic surface-pressure error of ~+2.6 kPa**,
  generated in-flight within the first forecast hour, near-uniform up the column, that
  inflates T2 by ~+2 K through the Exner conversion `T2 = θ2·(psfc/p0)^κ`. TSK bias = 0.000 K
  (corpus-refreshed), lowest-level θ bias ≈ 0, HFX is *low* not high. A machine-exact Exner
  knockout with the correct psfc collapses T2 RMSE 1.94→0.93 K at hour 1 across all 24 leads.
  At t=0 the GPU psfc equals corpus exactly; the offset is *grown by stepping*.
- **In-loop ph-fix attempt** (`...-opus-d03-phfix-INLOOP-findings.md` + `...-design.md`): the
  WRF-faithful in-acoustic-loop, mass-coupled ph′/w boundary tendency (`relax_bdy_dry` +
  `rk_addtend_dry` + `spec_bdyupdate_ph`) was implemented line-for-line, is **dynamically
  STABLE** (a real advance over the prior end-of-step overwrite that detonated at hour 1),
  and collapses the psfc error ~50% — yet **forcing the child ring toward the DECOUPLED
  hourly parent leaf injects spurious interior vertical motion** (interior max|W| ~13 m/s vs
  CPU-WRF 7.2 and free-drift baseline 6.2), adiabatically warming θ +5..+9 K and making T2
  RMSE *worse* (1.94→5..8 K). It was therefore left default-OFF.
- **GPT-5.5 WRF ph-boundary review** (`...-gpt-nest-ph-boundary-wrf-review.md`, the binding
  WRF source map): WRF carries `ph_2` in the nested boundary arrays/tendencies and forces it
  *inside* the acoustic `(w,ph)` solve; but — decisively — WRF **re-synchronises the child to
  the parent every parent step** via `med_nest_force` (`frame/module_integrate.F:409-430`),
  which recomputes the child base state + hydrostatic `ph_2` and refills `ph_b*`/`ph_bt*` via
  `bdy_interp`. The child interior therefore *never* drifts 2.6 kPa low; the ring residual
  stays small.

**The unifying conclusion (across all three RCAs):** the v0.1.0 d03 path is a *decoupled
hourly side-history replay* — u/v/w/θ/qv/mu are forced from hourly parent wrfout leaves but
**`ph′` free-drifts with no per-parent-step re-sync.** WRF's `med_nest_force` re-sync is the
missing piece. Any boundary-only band-aid (free-drift, hydrostatic ring, in-loop ph relax)
either drifts (+2.6 kPa) or pumps (worse T2), because the *interior* has equilibrated to a
wrong geopotential reference that no lateral-boundary treatment can repair. **Live nesting —
with the parent re-forcing the child every parent step — is the architecturally correct fix,
not a workaround.**

### 1.3 Decision

Adopt **live, device-resident, two-way nesting** for the operational real-case path:
d01 (9 km) → d02 (3 km) → d03 (1 km), 3:1 refinement, integrated on a single RTX 5090 with
**no host↔device transfer inside the timestep loop.** Live nesting **replaces** the hourly
side-history replay for real runs. The design **reuses** the already-built WRF-faithful nested
machinery (`coupling/boundary_apply.py` in-loop ph/w + the `AcousticCoreState.ph_bdy_target`
hooks; the host-side `_interp_parent_horizontal`/`_nested_axis_coords`/`bdy_interp`-equivalent
in `integration/d02_replay.py`), and adds the one thing the replay lacked: a per-parent-step
`med_nest_force`-equivalent re-sync of the child boundary arrays *and* a hydrostatic child
base-state recompute at force time.

---

## 2. WRF nesting cadence we are porting (source-grounded)

All file:line references below are pristine WRF (`/home/enric/src/wrf_pristine/WRF`), taken
from the GPT-5.5 ph-boundary review (binding) and direct source reads. **A separate GPT-5.5
WRF-nesting-architecture review (`.agent/reviews/2026-06-01-gpt-wrf-nesting-architecture.md`)
was requested and was NOT YET LANDED when this ADR was drafted (polled ~18 min); its findings
MUST be cross-checked into §2/§3/§4 at manager+GPT review — see §11 open question O0.**

### 2.1 The recursion (parent step drives N child steps)

`frame/module_integrate.F:409-430` — for each parent grid step, WRF:
1. calls `med_nest_force(parent, nest)` (the parent→child boundary *construction*),
2. recursively integrates the nest over the *parent time interval* — i.e. the child runs
   `parent_time_step_ratio` substeps (3 for our nests) per parent step,
3. at child completion, feedback (`med_nest_feedback` / `med_feedback_domain`) sends the child
   interior back up to the parent (two-way).

### 2.2 `med_nest_force` — what the parent→child force does (the re-sync we lack)

`share/mediation_force_domain.F:111-177` → `couple_or_uncouple_em.F:270-286` →
`external/RSL_LITE/force_domain_em_part2.F` → `inc/nest_forcedown_interp.inc:82-105`:
- **couple** parent prognostics by their mass factors (`ph_2 *= mutf`, `w_2`, `t_2`),
- interpolate parent → child (`interp_domain_em_part1` / `force_domain_em_part2`),
- **recompute the child base state + hydrostatic `ph_2`** from the interpolated mass/thermo
  (`force_domain_em_part2.F:165-275`: base pressure, `t_init`, `alb`, base `phb` integrated
  hydrostatically; then perturbation pressure/inverse-density down from the top; then `ph_2`
  hydrostatic for `hypsometric_opt==1|2`),
- **re-couple** and call `bdy_interp` for every prognostic incl. `grid%ph_2`
  (`nest_forcedown_interp.inc:82-105`), filling child `ph_b*` and `ph_bt*`,
- `bdy_interp1` (`share/interp_fcn.F:2578-2615`) sets `bdy_t* = rdt·(interp_current − nfld)`
  and `bdy_* = nfld` — i.e. the child boundary value AND a **per-parent-step time tendency**
  used to interpolate across child substeps.

The full prognostic set forced this way: **u, v, w, ph(=ph′), θ(t_2), μ(=mu_2), qv** (and other
moist species WRF carries). This is the `med_nest_force`-equivalent the replay omitted.

### 2.3 Inside the child step — the in-loop ph/w boundary path (already built)

Per acoustic substep, for `specified .OR. nested` (`solve_em.F` small-step loop):
- `relax_bdy_dry` (`module_bc_em.F:274-344`) builds a relaxation tendency from the
  **mass-weighted** full-level ph (and, for nests, w) toward the child `ph_b*` leaf; the
  `relax_bdytend` stencil is `fcx·fls0 − gcx·laplacian` (`module_bc.F:1293-1427`).
- `rk_addtend_dry` (`module_em.F`) folds `ph_tendf/msfty` into the carried `ph_tend`
  (and `rw_tendf/msfty` into `rw_tend`).
- `advance_w` consumes `ph_tend` in the φ-equation RHS and `rw_tend` in the buoyancy/PGF —
  so the forcing flows **through** the implicit `(w,ph)` solve.
- After `advance_w`, `spec_bdyupdate_ph` (`module_bc_em.F:17-157`) mass-coupled-updates the
  outer 1-row spec zone, then (nests) `spec_bdyupdate(w_2)`, then `calc_p_rho`.
- End-of-step `spec_bdy_final` (`solve_em.F:4687`) is a 1-cell anti-roundoff pin only.

**This entire path already exists in our codebase** (`boundary_apply.py`:
`nested_ph_relax_tendency`, `nested_w_relax_tendency`, `spec_bdyupdate_ph_inloop`,
`_relax_tendency_row`, `_full_ring_target_from_leaf`; `acoustic.py`:
`AcousticCoreState.ph_bdy_target`/`ph_save_for_spec`, applied in `acoustic_substep_core`).
It is default-OFF only because, under the *decoupled replay*, the target leaf was stale. Under
live nesting the target leaf is the just-forced, re-synced, hydrostatically-consistent parent
ring — so the same machinery becomes correct *and* stable. **This ADR reuses it verbatim.**

---

## 3. Design

### 3.1 Multi-domain state & device residency

A new `NestTower` (proposed, in a new module, e.g. `runtime/nesting.py` — **not** in
`operational_mode.py`) holds, per domain `d`:
- `state[d]: State` (the frozen ADR-002 SoA pytree, device-resident),
- `namelist[d]: OperationalNamelist` (per-domain dt, acoustic substeps, physics flags),
- `grid[d]: GridSpec` + `metrics[d]: DycoreMetrics`,
- static `nest_map[d]`: `parent_grid_ratio`, `i_parent_start`, `j_parent_start` (already in the
  child `BCMetadata`/case meta — see `_nested_axis_coords`), precomputed gather/scatter index
  arrays for parent↔child interpolation.

All states + metrics live on-device for the whole run. The host loop only schedules; **no
field crosses the PCIe bus inside any timestep.** (`contracts/halo.py` stays a single-GPU
no-op; see §6.)

### 3.2 Component 1 — nesting scheduler + subcycling

The scheduler is a **host-side recursion** mirroring `module_integrate.F`, calling
already-compiled, device-resident per-domain segment kernels (the existing
`run_forecast_operational_segmented` segment body, *unmodified* — we call it, we do not edit
it). Pseudocode (host orchestration only):

```
def integrate_domain(tower, d, n_parent_steps):           # host loop
    for _ in range(n_parent_steps):
        advance_one_parent_step(tower, d)                  # device segment (existing kernel)
        for child in tower.children[d]:
            force = med_nest_force(tower, parent=d, child=child)   # §3.3, device, once/parent-step
            tower.state[child], tower.bdy[child] = force
            integrate_domain(tower, child, ratio)          # ratio child steps over the parent dt
            tower.state[d] = med_nest_feedback(tower, parent=d, child=child)   # §3.4, device
```

- **N steps per parent step** = `parent_time_step_ratio` (3 for both 9→3 km and 3→1 km;
  matches the L3 namelist d01=18 s, d02=6 s, d03=2 s → in our pinned numerics d02 dt≈6 s,
  d03 dt≈3 s with acoustic_substeps≈10, per `d03_replay.py`'s CFL note). The ratio is a
  **static** scheduler constant → each domain's segment kernel compiles once and is reused for
  every parent step (the `start_step`-traced reuse already proven for the segmented driver).
- **Time-interpolation across child substeps** (`*_bt*` × dt cadence): `med_nest_force`
  produces, per parent step, both the child ring value `ph_b*` (and u/v/w/θ/μ/qv) **and** a
  per-parent-step tendency `ph_bt*` (WRF `bdy_interp1`: `bdy_t = rdt·(interp_current − nfld)`).
  Inside the child's N substeps the boundary leaf is advanced linearly:
  `value(t) = ph_b* + (t − t_parent)·ph_bt*`. This is **exactly the existing
  `interpolate_boundary_leaf`** (`boundary_apply.py:235`, `field_bdy + dtbc·field_bdy_tend`) —
  except the leaves are now refreshed every parent step on-device, not loaded hourly from disk.
  `OperationalNamelist.boundary_config.update_cadence_s` becomes the **parent dt** (e.g. 6 s),
  not 3600 s.
- **Composition with the existing scan/segment structure:** untouched. Each child step is one
  call into the existing `_physics_boundary_step`→`_rk_scan_step`→`_acoustic_scan`→
  `acoustic_substep_core` chain. The only new per-step input is a fresher `*_bdy` leaf (already
  a State field) and the (now non-stale) `ph_bdy_target`. **No change to the dycore scan.**

### 3.3 Component 2 — parent→child boundary forcing each parent step (THE +2.6 kPa fix)

This is the `med_nest_force`-equivalent the replay lacked, run **once per parent step**,
fully on-device:

1. **Couple** the parent prognostics by mass (`ph′·mutf`, `w`, `θ→t_2=μθ′`, etc.) —
   mirrors `couple_or_uncouple_em.F:270-286`.
2. **Interpolate parent → child** over the child window selected by `i/j_parent_start` +
   `parent_grid_ratio`. We reuse the *math* of `_interp_parent_horizontal` /
   `_nested_axis_coords` (`d02_replay.py:433-540`), re-expressed as a **static-index JAX
   gather** (precomputed bilinear weights, applied as one fused device op) so it runs in-loop
   with zero host transfer. (The replay does this on the host with NumPy at case-build time;
   live nesting must do it on-device every parent step.)
3. **Recompute the child base state + hydrostatic child `ph_2`** from the interpolated
   mass/θ/qv — `force_domain_em_part2.F:165-275`. We already have the verified inverse
   (`boundary_apply._hydrostatic_ph_perturbation`, the exact inverse of
   `diagnose_pressure_al_alt` / WRF `calc_p_rho_phi`, round-trips to 6e-11 Pa). **Crucially:
   this is the step that keeps the child interior from drifting 2.6 kPa low** — the child ph′
   is re-anchored to a parent-consistent, hydrostatically-balanced reference every parent step,
   so the interior never equilibrates to a wrong reference. (Same-vertical nests: WRF
   interpolates ph_2/phb directly per `nest_interpdown_interp.inc`; the hydrostatic recompute
   is the vertical-refinement / rebalance path. Our nests are same-vertical (44 levels all
   domains) → **interpolate ph_2 directly at force time, recompute hydrostatic ph only if a
   consistency check fails** — see O2.)
4. **Re-couple and fill the child ring** `*_b*`/`*_bt*` for the **full prognostic set**
   (u, v, w, ph′, θ, μ, qv) via the `bdy_interp`/`bdy_interp1`-equivalent
   (`_full_ring_target_from_leaf` already builds the ring; we add the `*_bt*` tendency =
   `rdt·(new − old)`). These become the child's `*_bdy` State leaves + `ph_bdy_target`.
5. Inside the child substeps, the **already-built in-loop mass-coupled ph/w path** (§2.3)
   consumes the ring: `nested_ph_relax_tendency` + `spec_bdyupdate_ph_inloop` +
   `nested_w_relax_tendency`, with `force_geopotential=False`, `nested_ph_relax=True`,
   `nested_ph_spec=True`, `nested_w_relax=True`. **No new dycore code** — we flip the toggles
   ON for the live path because the target is now correct.

**Why this fixes the +2.6 kPa and does NOT pump:** under the replay, the in-loop ph relax was
fighting a child interior that had drifted 2.6 kPa from a stale hourly target → large sustained
residual → w-pump. Under live nesting, step 3 re-anchors the child ph′ to the parent every
parent step, so the ring residual stays small (WRF's regime), and the in-loop relax is the
gentle correction it is designed to be. The pump documented in `...-phfix-INLOOP-findings.md`
was *the symptom of the missing re-sync*, not a flaw in the in-loop machinery.

### 3.4 Component 3 — child→parent feedback (two-way)

`med_nest_feedback` / `med_feedback_domain` after each child completes its N substeps,
on-device:
- **Fields:** WRF feeds back the coupled prognostics over the child interior (u, v, w, θ, μ,
  qv, ph) onto coincident parent cells, **excluding** a boundary buffer ring (`nfeedback`
  cells) so the parent's relaxation/specified zone is not overwritten by child boundary noise.
- **Smoothing:** WRF applies a smoother (`smooth_option`: 1-2-1 / smdsig) to the fed-back
  field before injection to avoid 2Δx parent noise. We port the WRF smoother as a small
  static stencil.
- **Aggregation:** for a 3:1 ratio, the parent cell value is the average (mass-weighted for
  coupled variables) of the `ratio²=9` child cells it covers — a static reduce/scatter.
- **Mass conservation:** feedback must conserve column dry mass on the parent — the
  mass-weighted average + the coupled-variable convention (feed back `μθ`, `μu`… not raw θ,u)
  preserves this to the WRF tolerance. A feedback mass-budget term is added to the
  conservation instrumentation (P0-7a).
- **Cadence:** once per parent step, after the child's N substeps complete (WRF cadence).
- **Phasing note:** two-way feedback is the **highest stability-risk** element (it couples
  fine-scale child noise back into the coarse parent). **Phase 1 runs one-way** (force only,
  no feedback) to validate the parent→child fix in isolation; two-way is enabled in Phase 3
  (§9) behind a `feedback: bool` flag, defaulting off until its own gate passes.

### 3.5 Component 4 — replacing the operational REPLAY

| Aspect | v0.1.0 REPLAY (current) | Live nesting (this ADR) |
|---|---|---|
| Boundary source | hourly parent wrfout from disk, host NumPy interp at case-build | live parent state, on-device interp every parent step |
| `ph′` handling | free-drift (`force_geopotential=False`) → +2.6 kPa drift | re-synced + in-loop forced (toggles ON) → no drift |
| Cadence | 3600 s leaves, linear-in-time | parent dt (≈6/3 s) leaves, linear-in-time |
| Land state | hourly corpus snap (`_refresh_hourly_land_state`) | unchanged for v0.2.0 (prescribed Noah-MP Opt A); prognostic = P0-3, separate |
| Re-sync | **none** (the defect) | `med_nest_force`-equivalent every parent step |
| Two-way feedback | none | optional Phase 3 |

**Migration / compatibility:**
- The replay path (`scripts/d03_replay.py`, `d02_replay.build_replay_case(boundary_domain=…)`)
  **stays** as a validation/oracle tool and a fallback; it is not deleted. Live nesting is a
  *new* driver (`runtime/nesting.py`) that the daily pipeline selects when a tower is
  configured. The single-domain d02 self-replay (`force_geopotential=True`, validated, byte-
  exact) is **unaffected** — it remains the v0.1.0 island proof and a no-regression anchor.
- `daily_pipeline.py` gains a tower-aware sequence path (parallel to `_run_forecast_sequence`)
  that writes wrfout per domain. Existing single-domain entry points unchanged.
- The initial condition for d02/d03 is still read from the corpus wrfout at t=0 (same as today)
  until native init (P0-2, explicitly *out* of v0.2.0 scope) lands; live nesting changes only
  *boundary* sourcing during integration, not the t=0 IC.

### 3.6 Component 5 — single-GPU memory plan (32 GB)

Per-domain dominant cost = the dycore scratch during a segment, not the persistent State.
Persistent State per domain ≈ O(20 prognostic+diag 3-D arrays × nz·ny·nx × 8 B fp64). For our
grids (44 levels): d02 ~66×159, d03 ~75×93, d01 (9 km, the new outer domain) larger in
footprint per cell but coarser — order tens of MB of *persistent* state each; trivial. **The
real consumer is the transient acoustic/advection scratch** inside a segment (the same scratch
that already bounds the single-domain run; the sibling currently shows 28.5/32 GB *in use* by a
single-domain GPU job — confirming scratch, not state, is the binding constraint).

Two viable schedules:
- **(A) Co-resident states, serialized compute (RECOMMENDED for Phase 1-2).** All three
  domains' *persistent* states co-reside on-device (cheap). Only **one domain integrates at a
  time** (the scheduler is serial by construction — the child cannot run until its parent step
  + force completes). So only **one domain's transient scratch is live at any instant**; we
  `block_until_ready` + free scratch between a parent step and its child recursion (the
  segmented driver already does this between segments). Peak ≈ max(single-domain scratch) +
  Σ(persistent states) ≈ current single-domain peak + small. **Fits 32 GB.**
- **(B) Memory-staged with state offload (fallback if A is tight at d01 9 km extent).** Keep
  only the actively-integrating domain + its immediate parent/child *ring* on-device; the
  idle domain's interior may be staged to host between its turns. **This re-introduces host
  transfer — but OUT of the timestep loop (only at domain switches, a handful of times per
  parent step), which the GPU-kernel rule permits with an ADR.** Documented here as the
  approved fallback; preferred only if (A) cannot fit.

`MEM_FRACTION` and the `block_until_ready` scratch-freeing discipline from the segmented driver
carry over unchanged. Relation to `halo.py`/sharding: single-GPU → `apply_halo` stays a no-op;
the `HaloSpec(edge_type="nest_boundary")` enum value already exists for the eventual multi-GPU
(S1) path, where a domain could be sharded — but **S1 is explicitly out of v0.2.0 scope**, so
this ADR commits to single-GPU residency only and keeps the halo call-shape stable.

---

## 4. Validation strategy (no-compromise)

Every gate is falsifiable, predeclared, and non-gameable. **All gates compare against
same-workstation CPU-WRF corpus truth, never JAX-vs-JAX self-compare, never persistence-only.**

| # | Gate | What it proves | Pass criterion (predeclared) |
|---|---|---|---|
| V0 | **Idealized still-PASS** | nesting code is a strict no-op for the doubly-periodic dycore | warm-bubble 6/6 + Straka 6/6, **bit-identical** to baseline (scheduler/force gated on `lead_seconds`/tower presence, exactly as the existing in-loop toggles are) |
| V1 | **Idealized nesting parity** | the scheduler + interp + feedback are correct in a *controlled* setting | a coarse parent + fine child of a known analytic/idealized flow (e.g. advecting bubble): child interior matches a single-domain fine run to predeclared tol; conservation V4 holds; **no skill claim** (this is mechanism, per GPT #3's Wave-3 split P0-1a) |
| V2 | **Interpolation conservation** | force + feedback conserve mass/water | parent→child force conserves column dry mass on the child window; child→parent feedback conserves parent column dry mass + total water to WRF tolerance (budget terms predeclared, instrumented via P0-7a) |
| V3 | **The +2.6 kPa fix (PRIMARY)** | live nesting collapses the d03 drift | live d01→d02→d03 vs corpus: **interior** psfc bias \|·\| < ~300 Pa (vs +2717 Pa free-drift / +1372..1906 in-loop-on-replay), interior max\|W\| within ~1 m/s of corpus 7.2 (vs 13 pumped), **T2 RMSE ≤ ~1.0 K** at hour 1 and ≤ corpus-WRF-comparable across 24 leads (vs 1.94 free-drift), **no w-pump** |
| V4 | **d03 skill vs CPU-WRF** (P0-1b) | the no-compromise operational bar | boundary-strip U/V/W/T/QV/P/PH/MU vs corpus; interior T2/Q2/U10/V10/PBLH beat persistence at every lead and are ≤ CPU-WRF-comparable; subcycling cadence audited; **only after P0-6 + P0-4 closed** (GPT #3) |
| V5 | **d02 no-regression** | live nesting does not harm the validated island | d02 (the v0.1.0 island proof) under the tower ≤ the standalone d02 self-replay result; the self-replay path stays byte-identical |
| V6 | **Transfer audit** | the device-residency invariant holds | profiler/trace audit: **zero** H2D/D2H bytes inside the timestep loop for schedule (A); for fallback (B), transfers only at domain switches, counted + bounded, ADR-approved |
| V7 | **Two-way feedback stability** (Phase 3) | feedback does not destabilize the parent | parent run finite + no 2Δx noise growth through 24 h; parent skill with feedback ≥ parent skill without (or document a neutral result); gated behind `feedback` flag |

Sequencing per the v0.2.0 plan: V0/V1/V2/V6 are Phase-1 (P0-1a) gates (mechanism, no skill);
V3/V4/V5 are Phase-3 (P0-1b) gates and **must wait on P0-6 (real-terrain/map-factor/boundary
dynamics) + P0-4 (KF cumulus on d01) closing first** — a correct dycore is a precondition for a
real skill claim.

---

## 5. How this resolves the d03 +2.6 kPa / replay drift (explicit)

1. The drift is caused by `ph′` free-drifting with **no per-parent-step re-sync** while
   u/v/θ/qv/μ are forced → child interior equilibrates to a wrong geopotential reference →
   uniform +2.6 kPa diagnostic pressure → +2 K Exner T2.
2. No lateral-boundary-only treatment fixes it: free-drift drifts (+2.6 kPa), in-loop ph relax
   toward the *stale* hourly leaf pumps (worse T2). Both are symptoms of the missing re-sync.
3. **Live nesting installs WRF's `med_nest_force` re-sync** (§3.3 step 3): every parent step the
   child base state + ph′ are recomputed hydrostatically from the *current* parent and the ring
   `*_b*`/`*_bt*` refilled. The child interior is continuously re-anchored to a parent-
   consistent reference, so it never drifts 2.6 kPa, and the **already-built in-loop
   mass-coupled ph/w forcing (§2.3) becomes the small, stable correction it was designed to
   be** (the toggles flip ON because the target is finally correct).
4. Net: the in-loop machinery that *was unstable under the replay* (because it fought a drifted
   interior) is *correct under live nesting* (because the re-sync keeps the residual small).
   This is precisely WRF's regime, and it is the no-compromise fix — not a diagnostic-psfc
   band-aid (which the bisection flagged as a stopgap that leaves the prognostic geopotential
   wrong).

---

## 6. Interfaces to freeze before parallel work (per Operating Rules)

- `NestTower` and the scheduler API (`runtime/nesting.py`) — **new file, no conflict.**
- `med_nest_force` / `med_nest_feedback` device functions — **new file** (e.g.
  `coupling/nesting_force.py`); reuse `boundary_apply` helpers, do not edit `boundary_apply`'s
  public functions (extend additively if needed, frozen-signature).
- `BoundaryConfig` live-nesting field defaults: a new `live_nested: bool = False` that, when
  True, implies `force_geopotential=False, nested_ph_relax=True, nested_ph_spec=True,
  nested_w_relax=True` — additive, backward-compatible (replay/idealized unaffected).
- The per-domain `nest_map` (parent_grid_ratio / i_parent_start / j_parent_start) is **already
  in `BCMetadata`/case meta** (`_nested_axis_coords`) — freeze its read interface.
- **DO NOT EDIT** `operational_mode.py` (active sibling). Live nesting *calls* its existing
  public segment entry points; any required hook must be requested via a manager-merged,
  frozen-signature change after the sibling sprint closes.

---

## 7. Risks: feasibility tail vs diligent execution

**Diligent execution (well-understood, just work):**
- Parent→child force + re-sync (§3.3): the interp math, the hydrostatic ph inverse, and the
  in-loop ph/w path are **all already built and unit-verified**; this is wiring them into a
  per-parent-step on-device force. Medium effort.
- Subcycling scheduler (§3.2): a host recursion over existing compiled segment kernels; static
  ratios → one compile per domain. Low-medium.
- One-way nesting (Phase 1-2): the +2.6 kPa fix lives here; high confidence it works because it
  is exactly WRF's regime and the failure mode is fully understood.

**Feasibility tail (genuine risk, needs its own gate / fallback):**
- **Two-way feedback stability (T1).** Coupling child noise back to the parent can seed 2Δx
  instability; smoother tuning + buffer-ring width are WRF-empirical. Mitigation: one-way
  first; feedback behind a default-off flag with its own gate (V7); if it cannot be made
  stable+beneficial in v0.2.0, **ship one-way down-nesting and document two-way as v0.2.x**
  (one-way already delivers the operational d03 quality fix; feedback improves the *parent*,
  which is secondary for the Canary product).
- **Single-GPU memory at d01 9 km extent (T2).** Schedule (A) should fit, but the d01 outer
  domain footprint is unproven on this box (sibling already at 28.5/32 GB single-domain).
  Mitigation: serialized compute (A) frees scratch between domains; fallback (B) stages idle
  interiors to host *outside* the loop (ADR-approved here). A pre-implementation memory probe
  (allocate all three states + run one segment of the largest domain) is the first task.
- **On-device interp performance (T3).** Moving `_interp_parent_horizontal` from host NumPy to
  an in-loop static-index JAX gather must not dominate the parent step. Mitigation: precompute
  bilinear weights once (static), apply as one fused op; profile in V6.
- **Hydrostatic re-sync vs same-vertical direct interp (T4 / O2).** Our nests are same-vertical
  (44 levels) so WRF would interpolate ph_2 directly; the hydrostatic recompute is the
  vertical-refinement path. Open question whether we *need* the recompute or whether direct
  ph_2 interpolation from the (consistent, live) parent already suffices once the re-sync
  cadence is per-parent-step. The offline check in `...-phfix-INLOOP-design.md` showed the
  *re-derived hydrostatic* target diverged aloft from the corpus-matching ph′ while the parent
  leaf matched — **suggesting direct parent-ph interpolation may be the right target under live
  nesting**, with the hydrostatic recompute reserved for a consistency fallback. Resolve in V1.

---

## 8. Phasing plan (one-way first; d02→d03 before d01)

**Phase 0 — interfaces + memory probe (no skill).** Freeze §6 interfaces. Allocate all three
domain states on-device, run one segment of the largest domain, confirm schedule (A) fits 32 GB
(else adopt (B)). Stand up `NestTower` + the static `nest_map` gather indices. Gate: V6
(transfer audit) on a no-op tower; V0 (idealized bit-identical).

**Phase 1 — one-way d02→d03 force + re-sync (the +2.6 kPa fix), TWO domains.** Implement
`med_nest_force` (couple → interp → re-sync ph → fill ring `*_b*`/`*_bt*`), drive d03 from a
**live d02** (not hourly disk leaves), flip the in-loop ph/w toggles ON. Two domains only (d02
parent, d03 child) to isolate the fix from d01/KF. Gates: V0, V1 (idealized parity), V2
(conservation), **V3 (the +2.6 kPa collapse — the headline)**, V6. *This is the milestone that
proves the architectural fix.* Start here because it directly attacks the proven defect with the
least new surface and the smallest memory footprint (two same-vertical domains).

**Phase 2 — add d01 (9 km) outer domain, one-way d01→d02→d03, THREE domains.** Add the d01
parent + its KF cumulus (P0-4) and the d01→d02 force. Requires P0-6 (real-terrain/map-factor/
boundary dynamics) closed first. Gates: V4 (d03 skill vs CPU-WRF — the no-compromise bar), V5
(d02 no-regression), re-run V2/V6. **This is P0-1b and closes only after P0-6 + P0-4.**

**Phase 3 — two-way feedback (optional within v0.2.0).** Enable `med_nest_feedback` behind the
default-off flag. Gate: V7 (feedback stability + parent skill). If it cannot be made
stable+beneficial, ship one-way and defer two-way to v0.2.x with an honest note (one-way already
delivers the d03 operational product).

Rationale for the order (answering the sprint's explicit phasing question):
- **One-way before two-way:** the d03 quality fix lives entirely in one-way down-nesting;
  two-way only improves the parent and carries the dominant stability risk. Ship the fix first.
- **d02→d03 before d01:** the proven defect is d03-specific; isolating it on two same-vertical
  domains gives the fastest falsifiable proof of the architectural fix with the smallest memory
  and compile footprint, before taking on the d01 outer domain + KF cumulus (which depend on
  P0-6/P0-4 anyway).

---

## 9. Consequences

- **Positive:** the no-compromise d03 fix; reuses already-built+verified machinery (low new
  risk on the hard parts); keeps the replay as an oracle/fallback; idealized + d02 island proofs
  preserved byte-identical; no in-loop host transfer (schedule A).
- **Negative / cost:** a new on-device per-parent-step force path (the interp gather + ph
  re-sync) and feedback; a memory schedule that must be validated on this box; two-way feedback
  is a feasibility tail that may slip to v0.2.x.
- **Out of scope (explicit, per v0.2.0 plan):** native init (P0-2), S1 multi-GPU sharding
  (v0.2.x), prognostic Noah-MP (P0-3, separate Wave-4 item — live nesting keeps prescribed land
  for now).

---

## 10. Acceptance for this ADR (review checklist for manager + GPT)

1. Is the `med_nest_force` re-sync (§3.3 step 3) the agreed root-cause fix for the +2.6 kPa
   drift? (vs the diagnostic-psfc stopgap — which this ADR rejects as non-no-compromise.)
2. Same-vertical nests: **direct parent-ph interpolation** vs **hydrostatic ph recompute** at
   force time (O2/T4)? — resolve before Phase 1 coding.
3. Memory schedule (A) co-resident-serialized vs (B) staged-offload — accept (A) as default,
   (B) as ADR-approved fallback for the d01 extent?
4. Two-way feedback as Phase-3-optional behind a default-off flag, shippable as one-way if its
   gate fails — acceptable for the v0.2.0 "no-compromise" definition? (one-way fixes the
   product; two-way improves the parent.)
5. Cross-check against the pending GPT WRF-nesting-architecture review (O0) before coding.

---

## 11. Open questions

- **O0 (BLOCKING for coding, not for this ADR):** the dedicated GPT-5.5 WRF-nesting-architecture
  review (`.agent/reviews/2026-06-01-gpt-wrf-nesting-architecture.md`) had **not landed** when
  this ADR was drafted (polled ~18 min). Its WRF file:line findings on `med_nest_force` /
  `med_nest_feedback` / feedback smoothing / buffer-ring width / coupling order **must be
  reconciled into §2/§3.3/§3.4** at review. This ADR is grounded in the *ph-boundary* GPT review
  (which already maps the nest force/feedback recursion) + direct source reads, so the
  architecture is sound; O0 is a confirmation pass, not a redesign.
- **O1:** exact `nfeedback` buffer-ring width + `smooth_option` for our nests (read from the L3
  namelist / `module_nesting`).
- **O2:** same-vertical direct ph_2 interpolation vs hydrostatic recompute as the force-time ph
  target (see T4) — the offline evidence leans toward direct parent-ph interpolation under live
  nesting; confirm in V1.
- **O3:** does Phase 1 need the d03 numerics (dt≈3 s, acoustic_substeps≈10) re-tuned for the
  tighter per-parent-step boundary cadence, or do the `d03_replay.py` values carry over?
- **O4:** feedback variable convention — feed back coupled (`μθ`,`μu`) vs decoupled, and the
  exact mass-weighted averaging for the 3:1 (9-cell) parent aggregation.
- **O5:** whether the d01 (9 km) outer domain IC/terrain/corpus exists in the manifest (the
  Wave −1 corpus is d02/d03-centric; d01 9 km coverage must be confirmed for Phase 2).
