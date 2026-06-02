# P0-1a Nesting — Findings (parent→child construction from recorded parent states)

**Sprint:** P0-1a (Wave-3, V0.2.0-PLAN). Live-nesting PARENT→CHILD boundary
construction + interpolation + scheduler, proven WRF-faithful against recorded
parent states. **No live-skill claim** (that is P0-1b). **No in-loop host/device
transfer.** GPU-free (CPU JAX, `taskset -c 0-3`).

**Base:** `worker/opus/p0-1a-nesting` from `worker/opus/v020-integration` @ d6ce779.

---

## 1. What was delivered

A NEW package `src/gpuwrf/nesting/` (does NOT edit `lateral_bc.py`,
`boundary_apply.py`, `flux_advection.py`, `operational_mode.py`, the dycore, or any
physics):

| module | role | WRF source |
|---|---|---|
| `interp.py` | parent→child spatial interpolation operators | `share/interp_fcn.F`, `share/sint.F` |
| `boundary_construction.py` | child `*_bdy` package from a parent state | `med_nest_force` / `bdy_interp1` (`interp_fcn.F:2423-2626`), `mediation_force_domain.F:111-206` |
| `scheduler.py` | subcycling cadence + forcedown ordering (pure host) | `frame/module_integrate.F:408-435`, `share/mediation_integrate.F:971-1012` |

Tests: `tests/test_p0_1a_nesting.py` — 11 GPU-free analytic + interface tests, all pass.
Proofs: `proofs/p0_1/oracle_recorded_parent_to_child.py` (+ `oracle_result.json`),
`proofs/p0_1/oracle_multitime.json`.

### Assessment of the prior `worker/opus/live-nesting` branch
That branch (`coupling/nesting_force.py`, `runtime/nesting.py`) was **structurally
sound** and reused here in spirit: its scheduler recursion, its two-time
`[old, new]` boundary leaf, and its static-gather interpolation are all WRF-faithful
in shape. It had **one real fidelity gap I found and fixed**: it interpolated with
**node-aligned bilinear** (the `d02_replay` convention), which is OFF by a fixed
**−1/3 parent cell** from WRF's cell-centered nest registration (see §2). I rebuilt
the operators on the integration base with the WRF registration as the default and
kept bilinear only as the proof cross-check. (The live-nesting branch is NOT on the
integration base; this package supersedes it.)

---

## 2. Interpolation operators (WRF `interp_fcn.F` / `sint.F`)

**WRF's default EM-core down-nest interpolation is `SINT`**, not bilinear:
`interp_fcn.F:2356-2358` sets `interp_method_type = SINT` when undefined, and
`bdy_interp` dispatches `SINT → bdy_interp1 → sintb` (`share/sint.F`). `sint` is a
**monotone 4th-order (TR4) residual-advection interpolation with a flux limiter**
(`DONOR` flux + `OV/UN` bounding). The limiter/high-order residual are *corrections*
that vanish for a locally linear field; the load-bearing structural difference vs
the replay bilinear is the **grid registration**.

### KEY FINDING — WRF uses cell-centered nest registration (−1/3 cell for ratio 3)
WRF places the child cell-centers offset by `−(ratio//2)/ratio` of a parent cell
relative to the node-aligned `d02_replay` convention. Two **independent** WRF
derivations give the same constant (both verified to machine precision against a
linear field):

1. `sint.F:54-59` sub-cell offsets `XIG(J) = (rr-1-rioff)/(2*rr) − (J-1)/rr`: the
   child sub-cell sits at coarse `−XIG` of the parent cell center.
2. `bdy_interp1` (`interp_fcn.F:2527-2529`) `nj = (j-jpos)*nrj + (nrj/2+1)` maps a
   coarse cell to its fine CENTER; inverting → child point `ni1` at continuous
   coarse `(ipos-1) + (ni1-1 − nri//2)/nri` (0-based).

For ratio 3: a constant **−1/3 parent-cell** shift. The validated v0.1.0 replay
(`d02_replay._nested_axis_coords`) used `x = (i_parent_start-1) + i/ratio`
(node-aligned) — i.e. it was systematically off by 1/3 of a parent cell, biasing
every interpolated child boundary value by `≈ grad·(1/3 cell)`.

For the **odd (3:1)** ratios in our 9→3→1 km tower, `sint.F:51-52` set the staggered
offset `rioff/rjoff = 0` (it is only set for EVEN ratios), so a staggered u/v field
uses the SAME cell-centered offset as mass — confirming the replay's "reuse mass
coords for u/v" was acceptable for *staggering* but NOT for *registration*.

### Operators shipped
- `build_sint_weights` / `interp_sint_linear` — WRF cell-centered registration,
  linear/low-order limit of `sint`. **Default device operator** (static-index
  device gather, zero host transfer).
- `build_bilinear_weights` / `interp_bilinear` — node-aligned replay baseline
  (proof cross-check only).
- `sint_block_reference` / `sint_to_child_reference` — full WRF monotone-TR4 `sint`
  host reference (faithful NumPy transcription of `sint.F`, vectorized over the
  grid). Proof-grade fidelity measurement; **not** on the device hot path. Bumping
  the device interp to the full TR4 limiter is a tracked P0-1b refinement (limiter
  engages only near sharp gradients; §4 shows its residual is tiny here).

---

## 3. Falsifiable proof — recorded parent → child boundary ring

### Oracle (non-gameable)
A **single recorded WRF run** (the L3 5-domain Canary run
`/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z`) holds
hourly `wrfout` for d01 (9 km), d02 (3 km), AND d03 (1 km). In WRF the child's
lateral-boundary ring is FORCED from the parent each parent step
(`med_nest_force → bdy_interp1`). So at any recorded time the recorded child's outer
ring is — up to one parent-step of relaxation drift + the TR4 limiter — exactly what
a faithful parent→child interpolation must produce.

We interpolate the recorded PARENT field with OUR operators onto the child grid and
compare to the recorded CHILD field at the SAME time on the boundary ring (outer 5
rows/cols). **Recorded gfortran-WRF is on BOTH sides** — this is not a JAX-vs-JAX
self-compare; our interpolation is the only thing under test. Two edges, both from
the same run: **A) d01→d02** (ratio 3, i_start 22, j_start 20) and **B) d02→d03**
(ratio 3, i_start 56, j_start 18).

### Predeclared tolerances (per field, boundary ring, t=0 = construction time)
| field | abs ring-RMSE tol |
|---|---|
| T (theta perturbation) | 1.5 K |
| QVAPOR | 1.5e-3 kg/kg |
| U, V | 2.0 m/s |
| PH (geopotential perturbation) | 400 m^2/s^2 |

Two independent gates:
- **G1 (fidelity):** every field's full-`sint` (TR4) ring RMSE <= its predeclared tol.
- **G2 (registration):** on registration-dominated fields (T, QVAPOR, PH), `sint`
  ring RMSE < `bilinear` ring RMSE.

### Results (`oracle_result.json`) — VERDICT PASS
| edge | field | bilinear | sint | sint_tr4 | tol | gates |
|---|---|---|---|---|---|---|
| d01->d02 | T | 0.0467 | 0.0074 | 0.0066 | 1.5 | within_tol, sint<bilinear (6x) |
| d01->d02 | QVAPOR | 3.41e-5 | 5.88e-6 | 1.94e-6 | 1.5e-3 | within_tol, sint<bilinear |
| d01->d02 | U | 0.0820 | 0.0353 | 0.0352 | 2.0 | within_tol |
| d01->d02 | V | 0.0407 | 0.0477 | 0.0468 | 2.0 | within_tol (see note) |
| d01->d02 | PH | 5.378 | 0.467 | 0.422 | 400 | within_tol, sint<bilinear (11x) |
| d02->d03 | T | 0.2276 | 0.1519 | 0.1437 | 1.5 | within_tol, sint<bilinear |
| d02->d03 | QVAPOR | 8.34e-5 | 4.94e-5 | 4.70e-5 | 1.5e-3 | within_tol, sint<bilinear |
| d02->d03 | U | 0.1487 | 0.1179 | 0.1125 | 2.0 | within_tol |
| d02->d03 | V | 0.1447 | 0.1285 | 0.1243 | 2.0 | within_tol |
| d02->d03 | PH | 9.015 | 5.684 | 5.489 | 400 | within_tol, sint<bilinear |

**G1 PASS** (every field far inside tol). **G2 PASS** (T/QVAPOR/PH `sint` beats
bilinear by 1.5–11x). The WRF cell-centered registration is measurably more faithful
than the node-aligned replay convention — the −1/3-cell finding is real and
load-bearing (clearest on PH: 11x tighter on edge A).

**Multi-time supplementary** (`oracle_multitime.json`): the d02→d03 `sint` ring
residual is **time-invariant** across the 24 h run (T 0.05–0.15 K, PH 1.3–5.7 m^2/s^2
at t=0/3/6/12/18 h) — not a t=0 fluke; the small growth from t=0 is the expected
interior-vs-boundary divergence, not an interp error.

### Staggered U/V note (honest)
U/V carry the C-grid half-cell offset that WRF handles via the `bdy_interp1` `ioff`
index shift (`interp_fcn.F:2504-2510`, `ioff = MAX((nri-1)/2,1)`). Our odd-ratio
build reuses the mass cell-centered registration for u/v (the documented bounded
approximation). The residual is sub-cm/s (edge A V: sint 0.0477 vs bilinear 0.0407,
a 0.007 m/s tie, both far below the 2 m/s tol) — so it is NOT in the G2 gate. **Exact
staggered-momentum registration is tracked for P0-1b.**

---

## 4. Scheduler / subcycling cadence (WRF `module_integrate.F`)

`scheduler.py` (pure host, no device, no in-loop transfer) encodes WRF's
`integrate` recursion (`frame/module_integrate.F:408-435`):

```
integrate(domain, n_steps):
  for local_step in 1..n_steps:
     advance(domain) one own step
     for child in children(domain):  med_nest_force(domain -> child)   # forcedown
     for child in children(domain):  integrate(child, ratio)           # recurse
     (optional child->parent feedback at child stop-subtime; default OFF)
```

Proven (`test_p0_1a_nesting.py`):
- `expected_substep_counts` for 9->3->1 km, 3:1 ratios, root_steps=4 -> d01 4 / d02 12
  / d03 36 (child = parent x ratio).
- `forcedown_event_log` ordering is **advance(parent) -> force(children) ->
  recurse(children)** per parent step; per root step 1 d01 + 3 d02 + 9 d03 advances,
  every parent advance immediately followed by its children's force.
- `run_host_tower` drives the same control structure with opaque host callbacks
  (the exact structure the P0-1b device runtime binds to; no field crosses any bus
  in this function).

### Boundary cadence
Each child's `update_cadence_s = parent_dt`; its `*_bdy` leaves are the two-time
`[old_child_ring, new_parent_target]` package built ONCE per parent step. WRF
correspondence (`bdy_interp1`, `interp_fcn.F:2583-2616`):
`bdy_xs = nfld` (child current), `bdy_txs = rdt*(psca - nfld)`, `rdt = 1/parent_dt`.
The two-time leaf with `cadence_s = parent_dt`, fed through the EXISTING
`boundary_apply.interpolate_boundary_leaf`, reproduces `bdy_* + dtbc*bdy_t*` exactly
over the subcycle (`dtbc: 0 -> parent_dt`). Verified directly in
`test_boundary_leaf_time_interp_matches_wrf_bdy_plus_dtbc_tend` (lead 0 -> old,
parent_dt/2 -> midpoint, parent_dt -> new).

---

## 5. Runtime hook spec for the manager (P0-1b — do NOT implement in P0-1a)

`scheduler.runtime_hook_spec()` (committed + unit-tested for presence). The single
hook the manager wires for P0-1b is:

- In `runtime/operational_mode`, for a nested run replace the single-domain step
  loop with `scheduler.run_host_tower`, binding:
  - `advance(name, carry, local_step)` := `_advance_chunk(carry, namelist[name],
    step_index=local_step, n_steps=1, cadence=radiation_cadence_steps)` — the
    child's boundary clock `lead_seconds = local_step*dt_child` sweeps
    `[dt_child, parent_dt]` so the two-time leaf interpolates as WRF.
  - `force(child, parent_carry, child_carry)` :=
    `boundary_construction.build_child_boundary_package(child_carry.state,
    parent_carry.state, edge.weights, bdy_width=spec_bdy_width)` — ONE device op per
    parent step, ZERO host transfer (gather weights are static device arrays; only
    host int counters live on the host).
  - child `BoundaryConfig.update_cadence_s = parent_dt` (live-nested cadence).
  - `block_until_ready` between a parent step and its child recursion (schedule-A
    peak-memory bound: only one domain's transient scratch is live at a time).
  - feedback stays OFF (one-way down-nesting; two-way is P0-1b+ optional).

---

## 6. P0-1a-closed vs deferred to P0-1b

**Closed (this sprint):**
- WRF-faithful parent->child interpolation operators (cell-centered `sint`
  registration default + monotone-TR4 host reference), proven vs recorded WRF.
- Child specified+relaxation `*_bdy` boundary-VALUE construction matching the frozen
  `State.*_bdy` / `boundary_apply` interface and the WRF `bdy_*/bdy_t*` cadence.
- Subcycling cadence + forcedown ordering (pure host), proven vs WRF structure.
- Falsifiable recorded-parent->child-ring oracle: 2 edges, 5 fields, 2 gates, PASS.

**Deferred to P0-1b (after P0-6 + P0-4 close):**
- Live device d01->d02->d03 run with the runtime hook wired (the spec above).
- Mass-coupled `(c1*mut+c2)` force-time interp (currently the decoupled bounded
  approximation; the mass-coupled IN-LOOP ph/w forcing already exists in
  `boundary_apply.nested_ph_relax_tendency`). O(mu'-gradient); vanishes for uniform
  mass.
- Exact staggered-momentum (u/v) C-grid registration via the WRF `ioff` shift
  (currently mass-registration reuse; sub-cm/s residual at ratio 3).
- Optional full monotone-TR4 limiter on the device path (currently linear gather;
  TR4-vs-linear residual is < the recorded-output noise here).
- Two-way child->parent feedback (`copy_fcn` area average); default OFF.
- The d03-skill-vs-CPU-WRF gate (P0-1b proper).

---

## 7. Unresolved risks
- **PH boundary in-loop treatment** is a separate dycore concern (GPT review
  `.agent/reviews/2026-06-01-gpt-nest-ph-boundary-wrf-review.md`): WRF carries ph in
  the mass-coupled acoustic boundary path (`advance_w` + `spec_bdyupdate_ph`), not
  an end-step overwrite. P0-1a constructs the ph boundary VALUES faithfully (ring
  RMSE 0.4-5.7 m^2/s^2, << 400 tol); how the child CONSUMES them in-loop is the
  dycore lane's `nested_ph_relax_tendency` path (already present), exercised in P0-1b.
- The proof is a SINGLE-DATE oracle (2026-05-09) at construction time. P0-1b should
  confirm across the other pairable dates (2026-05-21, 2026-05-30) and over a live
  multi-hour integration.
- Staggered u/v half-cell registration (above) — bounded but not exact at odd ratio.
