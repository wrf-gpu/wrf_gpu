# M6b RK1 Fix Sanity Check — Memo

Tester: opus (sprint `2026-05-25-m6b-rk1-sanity-check`)
Worktree: `/tmp/wrf_gpu2_rk1sanity`
Branch: `tester/opus/m6b-rk1-sanity-check`
Sources at HEAD `5204866` (no code edits made here).

Companion proofs:
- `proof_diff_inspection.txt` (Part 1)
- `proof_rk_stages_diff.txt` (Part 2)
- `proof_operational_vs_validation_rk1.txt` (Part 3)
- `proof_no_regression.txt` (Part 5)
- `raw_879ef56.diff` (verbatim 879ef56 patch to operational_mode.py)

## 1. Does the 879ef56 diff actually add an acoustic loop at RK1?

**Yes, for control flow.** The diff:

1. Adds a `substeps: int | None = None` kw-only override to
   `_acoustic_scan` and threads the override into `dt_sub` and
   `jax.lax.scan(length=...)`.
2. Replaces the boolean `use_acoustic` flag in `advance_stage` with an
   `acoustic_substeps: int | None` count, calling `_acoustic_scan` whenever
   the count is not `None`.
3. Dispatches the RK stages as
   `RK1 -> substeps=1`, `RK2/RK3 -> namelist.acoustic_substeps`,
   with an inline citation to WRF `solve_em.F:1472-1475`
   (`number_of_small_timesteps = 1` for RK3 stage 1).

There is no `pass`, no no-op, no `if rk_step == 1: skip` guard, and no
constant-zero substep count. The follow-up commit `268e38d` (D2H lift)
preserves this behaviour and removes the residual `is not None` guard in
favour of three unconditional `advance_stage(...)` calls, making the RK1
acoustic call structural and unconditional. **`_wrf_small_step_acoustic`
is entered exactly once for RK1 per call to `_rk_scan_step`.**

## 2. Is RK1 structurally identical to RK2/RK3 in operational_mode.py?

**Yes, modulo two scalar parameters.** RK1, RK2 and RK3 all execute the same
`advance_stage` body (halo, advection tendencies, scaled-tendency candidate,
save-family update, `_acoustic_scan`, post-acoustic halo). The only
per-stage differences are:

- `factor` ∈ {1/3, 1/2, 1} (RK3 weights).
- substep count: RK1 = 1, RK2 = RK3 = `namelist.acoustic_substeps`
  (default 10).

There is no operator-order divergence, no scratch-init divergence, and no
conditional that excludes RK1 from the acoustic path. Part 2 of the
proof bundle includes the line-aligned table.

## 3. Is operational RK1 structurally identical to validation `coupled_step` RK1?

**No.** Part 3 records five divergences (A–E). The dominant one (B) is the
smoking gun. Inside `_wrf_small_step_acoustic`:

```python
mu_new = state.mu_perturbation          # NOT advanced["mu"]
next_state = state.replace(w=w_solved)  # theta NOT updated
return _advance_promoted_scratch(
    carry, state, next_state,
    mu_new=mu_new,                      # carry scratch only
    ww_new=advanced["ww"],
    dt_sub=dt_sub,
    epssm=float(namelist.epssm),
)
```

With an explicit in-code comment (L307-309): "The promoted WRF scratch
recurrence is resident, but operational prognostic theta/mu remain on the
existing ADR-007 state path until a separate savepoint-aligned composition
sprint approves replacing them."

Validation `acoustic_substep_wrf` writes nine prognostic/scratch fields
(`mu`, `mudf`, `muts`, `muave`, `ww`, `theta`, `theta_ave`, `ph_tend`, `w`,
`t_2ave`) on every substep. Operational writes only `w` to the State and
mu/ww/scratch into the resident `OperationalCarry`, never into the
prognostic theta/mu_perturbation that drives RK2/RK3 advection.

Additional structural deltas: substep count and dt_sub (1×dt/3 vs 10×dt/10),
coefficient cadence (operational recomputes coefficients per substep;
validation computes once per RK stage in `acoustic_loop_wrf`), halo
placement (operational halos inside the substep, validation in the caller),
ph_tend formula (operational: `(ph_new-ph_old)/dt_sub`; validation:
`0.01 * theta_delta` placeholder), and precision (operational mixed,
validation fp64).

## 4. Probable defect class

`wrong-scratch-init / partial-state-promotion`.

The fix wires the *call* to the acoustic substep, but the *body* of the
substep deliberately does not promote `advanced["mu"]` and
`advanced["theta"]` into the prognostic State. The pre-acoustic candidate
theta/mu remains the value that RK2 advection re-reads, so the acoustic
loop has no effect on prognostic theta/mu evolution — exactly the failure
mode the synthetic-IC bisection picked up on the real Gen2 IC
(`proof_bisection_substep_level.json`:
`theta=5506.97`, `mu=49.85`, `ww=15.22`).

This is not "cosmetic" (control flow is real, scratch is updated, `w` is
updated, the scratch carry is consistent within itself), and it is not
"wrong-operator-order" (operator order is fine). It is precisely the gap
the worker called out in their own report:

> "Running RK1 acoustic scratch without promoting `advance_mu_t_wrf`
> theta/mu into operational prognostic state may not be enough to close
> the original theta divergence. A direct theta/mu promotion attempt was
> tested locally and backed out because it made the 70-second probe
> nonfinite; that needs a separate contracted design/fix."
> — `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/worker-report.md`

## Verdict

**`RK1-FIX-IS-PARTIAL` — named gap: prognostic theta/mu (and ph_tend
formula) are not promoted out of the operational acoustic substep.**

The control-flow plumbing in 879ef56 is real and correct. The semantic
fix is incomplete: `_wrf_small_step_acoustic` updates resident scratch
and `w` but, by design (per L307-313 comment + ADR-007), holds the
prognostic theta and mu_perturbation at their pre-acoustic candidate
value. Validation `acoustic_substep_wrf` writes those fields every
substep, which is why the bisection found a step-1 prognostic theta
divergence on the real Gen2 IC even after the 879ef56 fix landed.

Auxiliary gaps (lower priority than the prognostic promotion):
- RK1 dt_sub is dt_s/3 in operational and dt_s/10 in validation (operational
  matches WRF `solve_em.F:1472-1475`; validation does not).
- W solver coefficients are recomputed per substep in operational vs once
  per RK stage in validation.
- `ph_tend` accumulator formula differs (`(ph_new-ph_old)/dt_sub` vs
  `0.01 * theta_delta`).

## Part 5 — No regression

`pytest --collect-only` (cores 0-3) yields:

```
========================= 662 tests collected in 1.00s =========================
```

No collection error. See `proof_no_regression.txt`.

---

## AGENT REPORT

The 879ef56 RK1 fix is real but partial. Static inspection of the diff, the
operational `_rk_scan_step` at HEAD (post 268e38d), and side-by-side
comparison against validation `acoustic_loop_wrf` / `acoustic_substep_wrf`
all confirm that operational RK1 now enters `_wrf_small_step_acoustic`
exactly once with `substeps=1` at `dt_sub=dt_s/3`. RK1 is structurally
identical to RK2/RK3 in operational_mode.py modulo the substep count.
However, the operational acoustic substep deliberately drops the
prognostic theta and mu_perturbation promotion (only `w` is written to
the State; theta/mu/ww/muave/muts/ph_tend/t_2ave land in the resident
`OperationalCarry` scratch family, with an explicit L307-313 comment
deferring prognostic promotion to a separate ADR-007 sprint). This is
the smoking gun the parallel real-IC bisection should still see, and it
matches the worker's own admission that a direct theta/mu promotion
attempt destabilised the 70-second probe and was backed out. Verdict:
`RK1-FIX-IS-PARTIAL`, named gap is prognostic theta/mu promotion in
`_wrf_small_step_acoustic`; secondary gaps are coefficient cadence,
RK1 dt_sub vs validation, and ph_tend formula. No code edits were made.
`pytest --collect-only` clean at 662 tests. Proofs committed alongside.
