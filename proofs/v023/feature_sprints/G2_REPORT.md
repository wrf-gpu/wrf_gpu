# G2 — Moving Nests + Adaptive-Δt: Implementation + CPU Validation Report

Sprint: `proofs/v023/feature_sprints/G2_FABLE_BRIEF.md` (Fable-5, CPU-ONLY — the
GPU belongs to another lane; everything below ran `JAX_PLATFORMS=cpu`, pinned to
cores 4-31 per the 0:2-coordination directive).
Branch: `worker/fable/g2-moving-nests`.

**VERDICT: DONE (PASS on all four oracle prongs).** Operational gate PASS,
17/17 pytest gates, 69/69 nested-path regression, and the external real-WRF
`-DMOVE_NESTS` fixture cross-check PASS (`G2_WRF_FIXTURE_COMPARE.json`).
Honest carried limitations in the scope section (all fail-closed or
documented); none blocks the flat-terrain moving-nest + adaptive-Δt capability
shipped here.

## Objective

Turn the v0.22.0 fail-closed moving-nest scaffold into the REAL, WRF-faithful
implementation: a nest-motion driver (prescribed + vortex-following), the full
per-move choreography (state shift, exposed-region re-interpolation from the
parent, boundary re-derivation, two-way feedback re-coupling), and CFL-driven
adaptive-Δt — with CPU-provable oracles (fixture, conservation, analytic,
fail-closed) and the static path byte-identical.

## What was built (files changed)

| File | Content |
| --- | --- |
| `src/gpuwrf/nesting/moving_driver.py` | **The moving-nest driver.** `MovingNestConfig` (prescribed schedule / vortex-following, `corral_dist`, optional global wrap), `MovingNestDriver.make_move_fn()` for the live runtime. Per move: WRF `time_for_move2` decision (smoothed 500-hPa height minimum via log-p interpolation, boundary-ring exclusion, one-parent-cell clip) or preset schedule → corral fence → `med_nest_move` metadata reposition → **rebuild** of the parent→child SINT forcedown weights **and** child→parent feedback weights → `shift_domain_em`-style resident shift of the child `State` **and** the persistent `OperationalCarry` scratch (`ww`, `rthraten`, `*_save`, `t_2ave`, `mu*`, `ph_tend`) with **parent-interpolated fill of the newly exposed cells** (mass/u/v stagger-aware; nearest-neighbour for categorical masks `xland`/`lakemask`). |
| `src/gpuwrf/nesting/adaptive_driver.py` | **Adaptive-Δt on the live tree.** CFL diagnostics (`max_horiz_cfl`, `max_vert_cfl`) from the resident root state; the v0.22 `adapt_timestep` planner (WRF `calc_dt`: growth cap, precision rounding, min/max clamp) drives the root dt; write-through cascade pins every child to `child_dt = parent_dt / ratio` **and re-pins the child two-time boundary cadence to the new parent dt** (a stale `dtbc` cadence would mistime the relaxation). |
| `src/gpuwrf/runtime/domain_tree.py` | **Stale-weights bug fix** (pre-existing, latent): after a move, `resolve_edge` paired the moved spec with the STATIC tree's gather plans, so subsequent forcedown/feedback would interpolate/scatter through the ORIGINAL nest position. A moved-edge registry now returns the rebuilt edge; the dict is empty (inert) unless a move callback returns an edge — the static path is untouched. |
| `src/gpuwrf/validation/moving_nest_testbed.py` | Flat-terrain 3-D nested operational pair (the F7 idealized hydrostatic recipe generalized to `ny>1`); parent/child ICs describe one continuous atmosphere through the WRF cell-centered nest registration. |
| `src/gpuwrf/validation/moving_nest_operational_gate.py` | Proof-object gate → `G2_OPERATIONAL_GATE.json` (verdict PASS). |
| `tests/test_v023_moving_nest_operational.py` | 17 CPU gates (below). |
| `proofs/v023/feature_sprints/g2_wrf_fixture_compare.py` | The CPU-WRF moving-nest fixture comparison tool (below). |

## WRF correspondence

* Move decision: `share/mediation_nest_move.F::time_for_move2` (vortex = smoothed
  500-hPa height minimum, displacement from nest center, clipped to the search
  radius, ÷ ratio, clipped to ±1 parent cell) and the preset-moves namelist
  (`num_moves`/`move_interval`/`move_cd_x`/`move_cd_y`) → `PrescribedMove`.
* Reposition: `med_nest_move` updates `i/j_parent_start`; `corral_dist` fences
  the nest away from the parent boundary.
* State shift: `dyn_em/shift_domain_em.F` shifts EVERY registry array by
  `parent_grid_ratio` fine cells per parent-cell move → our shift covers the
  full prognostic `State` **plus** the persistent carry scratch.
* Exposed-region re-init: WRF re-runs the nest-initialization interpolation from
  the parent over exposed cells → our parent fill uses the SAME cell-centered
  SINT-linear registration as the forcedown (`share/interp_fcn.F`/`sint.F`).
* Boundary re-derivation: the runner calls `move` immediately before `force`
  (WRF `med_nest_move` → `med_nest_force`), so the full specified+relaxation
  ring is rebuilt from the parent at the NEW position through the REBUILT
  weights on the same parent step.
* Timing note (WRF-matched): at move time the parent has just advanced to
  `t+dt_parent` while the child is at `t`; WRF fills exposed cells from the
  already-advanced parent — ours does the identical thing.
* Adaptive-Δt: `dyn_em/adapt_timestep_em.F::adapt_timestep/calc_dt` via the
  v0.22 planner; children keep WRF's fixed step ratio (the runtime's subcycle
  count is structurally `parent_grid_ratio`).

## Proof objects (all CPU, `JAX_PLATFORMS=cpu`)

### 1. Operational gate — `G2_OPERATIONAL_GATE.json`, verdict **PASS**

Real dycore (full RK3 + acoustic, live nested forcedown), 30×27/24×21×8 pair,
ratio 3, uniform flow + Gaussian cold low:

* **Move semantics, bit-level**: overlap carried **bitwise** for
  `theta/u/v/w/qv/p_total/ph_perturbation/mu_total` AND carry scratch
  `ww/rthraten/u_save/t_2ave/muts`; newly exposed cells **bitwise-equal** to the
  direct parent SINT interpolation for mass/u/v staggering; force + feedback
  weights rebuilt (not the stale plans).
* **Conservation across the move**: overlap dry-mass delta = **0.0 Pa** (exact),
  overlap `∫mu·theta` energy-proxy delta = **0.0** (exact); total child dry mass
  vs uniform analytic: rel err **< 1e-14**; linear-plane exposed-fill max abs
  err **< 1e-9 K** (SINT-linear reproduces linear fields exactly —
  conservation-grade re-interpolation).
* **Zero-move bit-identity**: a run with the driver configured but no moves is
  **bit-identical** on every compared leaf to the `move=None` static path.
* **Full run with a mid-run move**: finite everywhere, physical theta range,
  move event at the prescribed step; two-way feedback variant finite.
* **Vortex tracker on a real hydrostatic state**: cold-column low placed at
  (24, 21) km found at ≤1 child cell error from `p_total/ph_total` via the z500
  log-p tracker.
* **Adaptive-Δt on the live tree**: step-1 keeps the starting dt (WRF
  `starting_time_step`), growth respects the 20 % cap, child namelist dt ==
  parent/3 and child boundary cadence == parent dt after every change; run
  finite.
* **Fail-closed named reasons** (error before compute): unknown mode; corral
  violation at initial placement; >1-parent-cell prescribed move; non-flat
  terrain (terrain/base-state re-blend not landed — see honest scope); active
  prognostic land carry; telescoped moving nest; missing geometry.

### 2. Pytest suite — 17/17 PASS (`G2_PYTEST_TRANSCRIPT.txt`)

Adds to the gate:

* **Exact analytic advection-through-moves oracle**: CFL=1 upwind translation is
  the exact solution of the advection PDE; a linear field advected through **6
  consecutive nest moves** matches the analytic solution on the moved window to
  **atol 1e-9** (machine exact) — shift bookkeeping, window trajectory, and
  inflow all correct through repeated moves.
* **Analytic vortex-following**: a Gaussian low advected at exactly 1 parent
  cell/parent step is tracked **move-for-move** (6/6 moves applied, window
  trajectory exact); moved-window field matches the analytic translated Gaussian
  (never-exposed interior at 1e-9; exposed strip parent-derived).
* **Stale-weights regression pin**: after a move, the feedback callback receives
  the REBUILT edge (would fail on pre-v0.23 `resolve_edge`).
* **Corral fence**: a move into the fence is blocked (logged `corral_blocked`),
  run continues unmoved — WRF behaviour, not an error.
* CFL diagnostics hand-check.

### 3. Regression: static/default path untouched

* 69/69 existing nested-path tests pass
  (`test_v0110_domain_tree`, `test_v022_moving_nest_adaptive`,
  `test_v014_noahmp_nested_pipeline`, `test_v0222_nested_wallclock`,
  `test_v020_nested_event_tail_guard`).
* Wider module-touching slice (`test_p0_1a_nesting`, `test_v017_edge_only_boundary`,
  `test_v0120_feedback_smoother`, `test_aot_executable`, `test_b2_compile_efficiency`,
  `test_parallel_compile`): 153 passed; the 7 fails + 5 errors in
  `test_aot_cheap_key.py` were verified **pre-existing on clean HEAD** via a
  stashed run (identical failure set).

### 4. CPU-WRF moving-nest fixture (real WRF v4.7.1, `-DMOVE_NESTS`)

Real-WRF oracle for the move operator (`g2_wrf_fixture_compare.py` →
`G2_WRF_FIXTURE_COMPARE.json`): an isolated clone of the pristine WRF tree
(`<USER_HOME>/src/wrf_pristine_movenest_g2` — the other lane's binaries
untouched) rebuilt with preset-moves nesting, running the nested em_quarter_ss
supercell (42²/43², Kessler, feedback=1) with per-parent-step wrfout frames and
two preset moves (+1,0 at min 3; +1,+1 at min 6). Observed WRF semantics (run
log): the preset move is applied at the END of the parent step ending at the
move time, after the nest completes that step and before history write — so the
post-move frame is `move(state)` and the pre-move frame is one step older:
`frame[k+1] = move(one_step(frame[k]))`. On the overlap the residual of OUR
shift applied to the pre-move frame is therefore the (shifted) one-step dycore
tendency; the oracle requires exactly that, against two nulls.

**Result: PASS — `G2_WRF_FIXTURE_COMPARE.json` (46 frames/domain, run
`SUCCESS COMPLETE WRF`).**

* Detected moves: exactly the 2 prescribed, at the prescribed times
  (frame pairs 02:48→03:00 and 05:48→06:00), displacements (+1,0)/(+1,+1),
  cross-checked against WRF's own `moving 2 <dx> <dy>` log record
  (`wrf_log_consistent_with_detected: true`); reconstructed trajectory
  (13,15)→(14,15)→(15,16) matching our `apply_move_to_edge` bookkeeping.
  Note: moves are detected FROM THE DATA (argmin over candidate parent-cell
  shifts) because this build freezes the wrfout `I/J_PARENT_START` global
  attribute at its initial value — metadata alone is blind to the move.
* Overlap shift, two-part criterion on T/U/V/QVAPOR/PH over both moves:
  (i) our shift strictly beats the no-shift null on all 10 field-move combos —
  `summary.max_shift_vs_null_ratio = 0.543` (9 of 10 are ≤ 0.25; the 0.543 is
  early-time PH, whose fast-wave one-step tendency is comparable to its
  still-developing displacement signal, i.e. a tendency-dominated null, not a
  shift error); (ii) the residual is one-step-tendency-sized — vs the natural
  per-step rms measured on the nearest non-move frame pairs,
  `summary.max_residual_vs_tendency_ratio = 1.09` (bound 1.5, most ≈ 0.9–1.1).
* Exposed strip: our SINT-linear parent fill beats the stale-value null on
  every checked field (`summary.max_exposed_vs_null_ratio = 0.132`).

Build provenance (documented, no physics change): upstream WRF serial builds
cannot nest ("nesting requires either an MPI build or use of the -DSTUBMPI
option") and the plain-serial registry stub never generates `SHIFT_HALO.inc`
(moving nests are dmpar/STUBMPI-only upstream). The fixture binary is the
officially supported **serial STUBMPI + basic nesting** configure with
`-DMOVE_NESTS` added to `ARCH_LOCAL` (`LANDREAD_STUB=1`, idealized flat
terrain), GNU gfortran, netCDF4. Two fixture-side fixes were needed, both
documented and WRF-behavior-neutral:

1. **Upstream latent link bug in the STUBMPI+MOVE_NESTS combo**: in
   `share/mediation_nest_move.F::med_nest_move` the
   `USE module_dm, ONLY: wrf_dm_move_nest` is guarded
   `#if defined(DM_PARALLEL) && !defined(STUBMPI)` while the bare CALL remains,
   so under STUBMPI it compiles as an EXTERNAL reference with no definition
   (the only definition is a module procedure — itself a `RETURN` no-op even in
   dmpar). Fix: an appended external-linkage no-op shim guarded to exactly
   this build combo (zero behavior change; the real state shift is
   `shift_domain_em`, which is single-process-complete).
2. **Namelist**: idealized cases must keep `input_from_file = .false.` even for
   d01 (`start_em` FATALs otherwise); d01 reads `wrfinput_d01` unconditionally
   and the ideal nest initializes by parent interpolation, as upstream intends.

(The prior session died while this WRF build was mid-compile; the resumed
incremental build had zero compile errors — the "build blocker" was the killed
session plus the two items above.)

## Honest scope + carried limitations

1. **Non-flat terrain moves FAIL CLOSED** (named reason). A real-orography move
   needs the WRF terrain re-blend + `start_domain` re-derivation of the child
   base state/metrics (and hi-res terrain input). The algorithmic core
   (shift/re-interp/re-derive/re-couple) is landed and validated; the orography
   choreography is the tracked follow-up. WRF's own idealized moving-nest cases
   are flat; vortex-following TC applications are ocean-dominated.
2. **Active prognostic land carries fail closed** (Noah-MP/Noah-classic/slab/
   Pleim-Xiu/cumulus/base_state sub-states) — a registry-wide shift for those
   pytrees is mechanical but not yet landed; silently not shifting them would be
   a wrong-position land state.
3. **Adaptive-Δt compile variants**: `dt_s` is a static key of the compiled
   advance, so each distinct quantized dt costs one cold compile per domain
   (WRF `precision` rounding bounds the count; the gate uses precision=10).
   Folding dt into a traced scalar is the GPU-lane follow-up.
4. **CFL diagnostic latency + host read**: CFL is computed from the end-of-step
   resident state (WRF records it inside the RK loop — same quantities, one step
   of latency), and reads u/v/w/ph back to host once per ROOT step. CPU-lane
   acceptable; a GPU deployment must fold the reduction on-device (ADR rule).
5. **Radiation cadence under adaptive dt** stays step-indexed (`radt` drift in
   wall time), as documented in the namelist contract.
6. The WRF fixture comparison bounds the move OPERATOR (shift + fill +
   trajectory) at one-step-tendency tolerance; it is not a full-physics
   supercell parity run (our port's quarter-ss physics parity is a separate
   matter from G2).

## Commands to reproduce

```bash
# gates + suite (CPU, pinned)
JAX_PLATFORMS=cpu taskset -c 4-31 python -m pytest tests/test_v023_moving_nest_operational.py -v
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 4-31 \
  python -m gpuwrf.validation.moving_nest_operational_gate proofs/v023/feature_sprints/G2_OPERATIONAL_GATE.json

# WRF fixture (isolated clone; binaries already built via resume_stubmpi.sh)
<USER_HOME>/src/wrf_pristine_movenest_g2/run_move_fixture/run_fixture.sh
JAX_PLATFORMS=cpu taskset -c 4-31 \
  python proofs/v023/feature_sprints/g2_wrf_fixture_compare.py \
  <USER_HOME>/src/wrf_pristine_movenest_g2/run_move_fixture \
  proofs/v023/feature_sprints/G2_WRF_FIXTURE_COMPARE.json
```

## Handoff

* **Objective**: real WRF-faithful moving nests + adaptive-Δt, CPU-proven — DONE
  within the honest scope above.
* **Files changed**: table above; commits on `worker/fable/g2-moving-nests`
  (implementation `37f009b0`, report draft + pytest transcript `3aa3a563`,
  + the final fixture-result/report commit).
* **Proof objects**: `G2_OPERATIONAL_GATE.json` (PASS),
  `G2_PYTEST_TRANSCRIPT.txt` (17/17), `G2_WRF_FIXTURE_COMPARE.json` (PASS),
  69/69 nested-path regression, pre-existing-failure adjudication for
  `test_aot_cheap_key.py`.
* **Unresolved risks**: items 1–5 above (all fail-closed or documented).
* **Next decision**: whether v0.23 wants the orography-move choreography and the
  land-carry shift registry in-scope (G2b) or carried.
