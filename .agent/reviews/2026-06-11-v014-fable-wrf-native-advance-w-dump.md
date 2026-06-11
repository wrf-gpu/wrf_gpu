# v0.14 Fable WRF-Native Advance-W Dump Review

Verdict: **FIXED** (two proven local WRF-faithful source defects fixed in
production; the jitted h36/call-21602 short gate improves `p` by 67% and `ph`
by 87% with mass invariants unchanged; plus the `top_lid` defect proven and
flipped behind a bisection env, gated below).

## Objective

Build the shortest rigorous WRF-native oracle for the Switzerland/Gotthard h36
RK1 acoustic substep `WRF call 21601 -> 21602`, inside `advance_w` +
immediate `calc_p_rho`; fix a proven local WRF-faithful defect or name the
first mismatching WRF-anchored term.

## What was proven (full numbers: `proofs/v014/wrf_native_advance_w_dump.{json,md}`)

Two disposable instrumented WRF re-runs of the bit-exact 36h30m 24-rank truth
(~25 min each) dumped every `advance_w` input/internal/output, the in-loop
`calc_p_rho`, and per-contributor `rk_tendency` `rw_tend`/`ph_tend` snapshots
at the exact substep. Gating sanity: dumped small-step `mu''` == independent
HPG (21602−21601) `mu` increment at fp32 roundoff.

Earliest-mismatch cascade (interior, depth 8):

1. **`pg_buoy_w` stage pressure input — ROOT CAUSE (FIXED).** The JAX stage
   entry *recomputed* `p` via `diagnose_pressure_al_alt`; WRF consumes the
   *carried* `grid%p` (end-of-previous-stage `calc_p_rho_phi`). The carried
   JAX leaf `state.p_perturbation` is **bit-exact** vs WRF `p` (diff 0.0); the
   recompute differs 0.30 Pa interior / 1.1 Pa near-surface, which `pg_buoy_w`
   amplifies into the dominant `rw_tend` error (511 of 1337 rms, the entire
   near-surface error peak). With WRF `p`, the JAX `pg_buoy_w` matches the
   native term to 4.6e-4 (operator, `mu'`, `mub`, moist `cqw` all exact).
2. **w cosine-Coriolis + curvature terms — MISSING (FIXED).** WRF adds both to
   `rw_tend` once per stage (`module_big_step_utilities_em.F:3836,4283`);
   measured native magnitudes 172 + 7.2 rms. The JAX omission was documented
   as deliberate in `rk_addtend_dry`.
3. **`top_lid=True` — PROVEN WRF-UNFAITHFUL (flipped, env-gated).** WRF truth
   runs open-top (`flags_toplid=F` in the dump). The rigid lid zeroes the
   top-face `rhs`/`w` of the implicit (w,phi) solve; the Thomas
   back-substitution propagates the error down the whole column.
4. **Excluded by the oracle:** `advect_w`+filters (4.1% rel), `w_damp`
   (identically 0 here), the `advance_w` implicit solve and `calc_p_rho`
   operators (all-WRF-input open-top isolation: `ph_out` 5.2e-7, `p` 5.3e-8 —
   bit-faithful), surface-w BC u/v feed (known deliberate trade-off; zero
   interior effect at this substep), dry-mass inputs (confirms the term-split
   sprint).
5. **Named remaining (next targets, none dominant):** `ph_tend` 13.9% rel,
   coupled work `t_2` 53.7% rel (small absolute), `ww` 51% rel of a 0.003-rms
   field, `mu''` 15.5% rel.

After the fixes the production `rw_tend` matches the WRF-native total at
**0.37% rel** (4.85 of 1318 rms); the stage `w_out/ph_out` error vs the native
dump drops 923→109.5 / 0.435→0.0554 (lid on) and to 35.7/0.0192 (open top) —
identical to the bound obtained by feeding WRF's exact `rw_tend`, i.e. the
`rw_tend` lane is closed.

## Production source changes (minimal, on this branch)

- `src/gpuwrf/runtime/operational_mode.py`
  - `pg_buoy_w` consumes the carried `state.p_perturbation` (WRF carry
    cadence) instead of a per-stage full-diagnostics recompute (also removes
    one full-grid diagnostics pass per RK stage — perf win, no new transfers);
  - added the WRF `rk_tendency` w cosine-Coriolis + curvature terms to the
    once-per-stage `rw_tend` assembly (`GPUWRF_W_CORIOLIS=0` opt-out);
- `src/gpuwrf/integration/daily_pipeline.py`
  - real-case `top_lid` now defaults open (False), WRF-faithful;
    `GPUWRF_TOP_LID=1` restores the lid for bisection.

No clamps, no tolerance changes, no host/device transfers inside timestep
loops; the two new stage terms are a handful of elementwise ops once per RK
stage.

## Gates run

- **Replica fidelity:** proof-local captured replica == production
  `advance_w_wrf` bitwise (0.0 on w/ph/t_2ave).
- **CPU stage oracle** (vs WRF-native dumped `w_out`/`ph_out`): table above
  (`fix_progression` in the JSON).
- **GPU jitted production stage gate** (`switzerland_acoustic_substep_blocker
  --stage-compare`, replica-vs-jit max diff ≤6e-8), step1 stage1 vs call
  21602, interior increment rmse:

  | config | mu | p | ph | al |
  |---|---:|---:|---:|---:|
  | pre-fix baseline (prior sprints) | 0.020896 | 1.1262 | 0.43526 | (larger) |
  | post-fix, `top_lid=True` (tag `v014_awd_fixes_lid`) | 0.020896 | 0.37086 | 0.055421 | 4.93e-5 |
  | post-fix, open top (tag `v014_awd_fixes_open`) | 0.020896 | 0.36941 (−67%) | **0.015090 (−96.5%)** | 6.75e-6 |

  The residual `p` (0.369) is no longer created in the (w,phi) solve: `ph` is
  down 96.5% and `al/alt` are at 1e-6; the remaining `p` error tracks the
  coupled work-theta lane (`t_2`, 53.7% rel) through the EOS — the named next
  target — plus the known secondary `mu''` (15.5%).

- **Short GPU forecast (stability + venting budget):** 2h open-top production
  forecast from the h36 re-init: `status=PASS`, max|W| 3.20 (h37) / 3.39 (h38)
  m/s — the 2026-05-30 open-top top-face blow-up does NOT recur with the
  current acoustic stack. Hourly venting budget (depth-8 interior, Pa/cell/h
  vs CPU −74.5): excess −26.6 at h37 (prior baselines −28.3…−28.8, ~7%
  better), −21.7/h averaged to h38. **Honest read: the 72h venting KI is NOT
  closed by these fixes** — the per-substep (w,phi) creator is closed (ph
  −96.5%), and the budget improvement is real but small; the surviving hourly
  venting tracks the named theta-lane (`t_2`) residual that now also carries
  the remaining stage `p` error.

## Files changed

- `proofs/v014/wrf_native_advance_w_dump.py` (new; assembler/comparator/replica)
- `proofs/v014/wrf_native_advance_w_dump.json` (new; manifest + all comparisons)
- `proofs/v014/wrf_native_advance_w_dump.md` (new; proof summary)
- `proofs/v014/wrf_native_advance_w_dump_wrf_patch.diff` (new; disposable WRF
  instrumentation, additive env-gated)
- `src/gpuwrf/runtime/operational_mode.py` (two fixes above)
- `src/gpuwrf/integration/daily_pipeline.py` (open-top default, env-gated)
- `.agent/notes/2026-06-11-efficiency-notes-advance-w-lane.md` (new;
  inefficiency inventory per principal directive — includes the fixed
  per-stage diagnostics recompute, the per-substep `w_damp` placement, the
  hot-path `jnp.where` floors, Thomas scan unroll candidates)
- `proofs/v014/switzerland_acoustic_substep_blocker.json` (gate tags appended)

## Commands run (key)

- two instrumented WRF re-runs via
  `/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/launch_wrf.sh`
  (rc=0, wall 1442/1486 s; `tcsh ./compile em_real` rebuilds in the disposable
  tree)
- `python -m py_compile proofs/v014/wrf_native_advance_w_dump.py` (rc 0)
- `python proofs/v014/wrf_native_advance_w_dump.py --manifest --compare` (rc 0)
- `python proofs/v014/wrf_native_advance_w_dump.py --compare-rw` (rc 0)
- `python -m json.tool proofs/v014/wrf_native_advance_w_dump.json` (rc 0)
- `python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare
  --tag v014_awd_fixes_lid --steps 1` (rc 0, GPU)
- `python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare
  --tag v014_awd_fixes_open --steps 1` (rc 0, GPU)
- `git diff --check` (clean)

## WRF dump schema / data locations

Schema v1 frozen in the sidecars and `wrf_native_advance_w_dump.md`; data at
`/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps/` (96 files,
130,257,560 B, per-file sha256 in JSON `manifest`; not in git).

## Unresolved risks

- The open-top default change is validated on the Switzerland h36 stage gate
  and the short forecast only; Canary d02/d03 short gates should be re-run
  before release (the 2026-05-30 open-top instability predates the acoustic
  fixes, but that is an inference for Canary until gated). `GPUWRF_TOP_LID=1`
  is the immediate rollback.
- The remaining stage residual (`ph` 0.019 open-top) is carried by the named
  smaller inputs (`ph_tend`, `t_2`, `ww`, `mu''`); the 72h venting verdict
  needs the manager-controlled long gate.
- The w-Coriolis/curvature terms use stage-entry `state.u/v` with `prep.muu/
  muv` (WRF `couple_momentum` cadence); spec-zone rows are later pinned by the
  boundary updates as in WRF.

## Next decision needed

Manager: (1) accept the open-top default + schedule Canary short gates before
merge to trunk, or keep `GPUWRF_TOP_LID=1` pinned for Canary lanes; (2)
schedule the 72h Switzerland venting gate with these fixes; (3) next
narrowing target order suggestion: `ph_tend` (13.9%) -> `mu''`/`t_2` -> `ww`.
