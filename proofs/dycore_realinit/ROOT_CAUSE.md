# Dycore real-init stability: root cause + fix

Branch: `worker/opus/dycore-realinit` (from `worker/opus/recomp` @ 2600b8d).
Real case: Gen2 d02 `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
(mass grid nz=44, ny=66, nx=159; dt=10 s, n_acoustic=10; fp64). All GPU/JAX, no
WRF launched. Env: taskset -c 0-3, OMP_NUM_THREADS=4, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3.

## Step 1 verdict (the Sprint-U-vs-recomp discrepancy)

The two prior proofs used DIFFERENT code paths and configs:

* Sprint U (`proofs/sprintU/guards_off_operational_proof.json`, w=16.2 STABLE):
  `_build_real_case` + operational `_physics_boundary_step`, dt=10/acoustic=10,
  validated namelist, **50 steps**.
* recomp ladder (`proofs/recomp/`, blew up): `build_replay_case` +
  `_dycore_step_adr023` directly, dt=1.0/2.0/0.25, `n_acoustic=4`, NO operational
  damping (only a `damping.py` skeleton smdiv/rayleigh, and it forced
  `top_lid=True` in `install_damping`), no `w_damping`/`damp_opt`/`diff_6th`.

Running the EXACT validated `_build_real_case` namelist, dycore-only
(run_physics=False), guards-OFF, for **360 steps** (`step1_longrun.py`):

**Verdict: UNSTABLE — but NOT an interior dycore defect.** The instability is
entirely in the BOUNDARY treatment. Two distinct modes, both at boundaries:

| step | \|w\| | \|w\| origin | \|u\| | \|u\| origin |
|---|---|---|---|---|
| 1   | 307 | k=44 (model TOP face) | 25.6 | interior |
| 30  | 15  | top decayed | 25.7 | — |
| 200 | 14.6| interior | 70  | x=159 (E edge) |
| 360 | 110 | k=44 top | 300 | x=42 upper, edge-fed |

`u` shape is `(44,66,160)` (x-staggered, x=159 = east domain edge); `w` shape is
`(45,66,159)` (k=44 = the open model-top face).

## Two boundary modes, isolated (`step2_bc_isolation.json`, 60 steps)

| top_lid | run_boundary | step1 \|w\| | step1 top-face | final \|w\| | final \|u\| |
|---|---|---|---|---|---|
| False (open) | False | 307.3 | 307.3 | 12.1 | 32.4 |
| **True (lid)** | False | **13.5** | **0.00** | 8.8 | 32.4 |
| False (open) | True | 307.3 | 307.3 | 135.8 | **180.2** |
| **True (lid)** | True | **13.5** | **0.00** | 14.4 | **33.1** |

**Mode A — top-face w spike = the OPEN top.** `top_lid=True` drops step-1 |w|
from 307 to 13.5 and the top face to exactly 0.00. The open-top w solve
(WRF-faithful, `module_small_step_em.F:1421-1429`, verified line-by-line in
`advance_w.py:345-367`) produces a ~300 m/s spurious w on the model-top face on
the first step from the real d02 upper-level state. The idealized validation
gates ALL used `top_lid=True`; the open top was never validated on real data.

**Mode B — slow u growth at BOTH lateral edges = periodic advection on a LAM.**
The flux advection is periodic-x/-y (`flux_advection.py`: `flux5_face_periodic`,
`couple_velocities_periodic`). The d02 domain is limited-area, NOT periodic.
With `run_boundary=False` the periodic wrap glues the (mismatched) east and west
edges; `step1b_spatial_probe.json` confirms the late-step u peaks at BOTH x=0
(68.6) and x=159 (70.2) symmetrically while interior is 54 — the periodic-wrap
signature. Real lateral boundaries (`run_boundary=True`) pin the edges and the
u-mode vanishes (edges 23.9/22.9 < interior 28.0).

**The two modes couple destructively:** open top + boundaries (the worst case)
keeps the top-face w pinned at ~120 m/s for the whole run (the relaxation zone
re-feeds the top corner faster than damp_opt=3 removes it) and drives u to ~200
by step 5 (`step5_opentop_bndy.json`, both epssm=0.1 and 0.5 UNSTABLE).

## Secondary config mismatch: epssm

The real Gen2 d02 `namelist.input` `&dynamics` sets **`epssm = 0.5`**, but
`_build_real_case` left epssm at the dataclass default **0.1**. epssm is the
vertically-implicit acoustic off-centering: `cof=(0.5*dts*g*(1+epssm))^2`
(`module_small_step_em.F:624,646`); `eps_m=1-epssm` weights the explicit
old-time pressure/w. 0.1 under-damps the vertical/top acoustic mode. epssm=0.5
alone does NOT fix the run (`step3_epssm.json`: still UNSTABLE), but it is the
correct WRF value and reduces the top transient (307 -> 249).

## Fix (WRF-faithful, no masking)

`src/gpuwrf/integration/daily_pipeline.py::_build_real_case`:
* `top_lid=True`  — rigid lid (`w(kde)=0`), the SAME top BC every idealized
  validation gate used; zeroes the spurious open-top w mode.
* `epssm=0.5`     — match the real d02 `namelist.input`.
* `run_boundary` stays True (operational default) — the real LAM lateral
  boundaries pin the edges (the missing piece the recomp ladder ran without).

`OperationalNamelist.from_grid` gained a `top_lid` kwarg (default False;
additive, no behavior change elsewhere).

NO clamp / no mask / no synthetic path. The guards stay OFF in the proof.

## Acceptance (`step4_fix_longrun.json`, `step6_accept.json`)

`_build_real_case` (the real pipeline path, fix wired), dycore-only
(run_physics=False), real boundaries, **guards OFF**, **360 steps** (~1 h):

* **STABLE.** first_unphysical = None, first_nonfinite = None.
* |w| = 14.4 (top face 0.012 ~ 0), |u| = 31.1 (steady, not growing),
  |v| = 23.6, theta in [289.8, 495.5] K.
* All prognostics fp64 throughout. Guards proven NOT load-bearing.

## Re-attribution of the recomp ladder

The recomp `STABILITY_ISOLATION.md` claim "rooted in the DRY DYNAMICAL CORE,
structurally unstable on this real init" is **incorrect**. The dycore is stable.
The ladder ran the dry core (a) at non-operational dt/acoustic, (b) without
`w_damping`/`damp_opt`/`diff_6th`, (c) with a `damping.py` skeleton instead of
the operational damping, and crucially (d) WITHOUT the lateral boundaries that a
limited-area domain requires, on a periodic-advection core. Its triggers 2 (B4
boundaries) and 3 (B3 radiation cadence) remain valid follow-ups for the COUPLED
run; this work clears the dycore as the blocker.

## Reproduce

```
PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 taskset -c 0-3 \
  python proofs/dycore_realinit/step1_longrun.py --steps 360       # UNSTABLE (pre-fix, open top + no bndy)
  python proofs/dycore_realinit/step2_bc_isolation.py --steps 60   # isolates the two modes
  python proofs/dycore_realinit/step3_epssm.py --steps 360         # epssm 0.1 vs 0.5 (both unstable alone)
  python proofs/dycore_realinit/step4_fix_longrun.py --steps 360   # rigid lid +/- bndy
  python proofs/dycore_realinit/step5_opentop_bndy.py --steps 360  # open top + bndy (UNSTABLE)
  python proofs/dycore_realinit/step6_accept.py --steps 360        # FIX via _build_real_case -> STABLE
```
