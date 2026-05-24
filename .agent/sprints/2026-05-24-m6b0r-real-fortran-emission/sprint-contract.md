# Sprint Contract — M6B0-R: Real Fortran Savepoint Emission (B-direct, savepoint-first)

## Objective

M6B0 (commit `fefd51d`) built the JAX-side scaffold (schema, I/O, comparator, perturbation test). The worker honestly flagged that **no relinked Fortran `wrf.exe` exists**; the "first coefficient parity" is JAX-vs-JAX self-consistency, not JAX-vs-WRF. M6B0-R closes that gap: build a relinked CPU-WRF binary that emits real per-operator savepoints via a Fortran wrapper module gated by `#ifdef WRF_SAVEPOINT`, run it on real Canary d02 input at the column / 16×16 patch / golden small-domain tiers, and demonstrate the **first real JAX-vs-WRF coefficient parity** for `calc_coef_w`.

This sprint inherits all scaffold from `fefd51d`. It applies the 5 critic amendments from `.agent/sprints/2026-05-24-m6b0-plan-critic/reviewer-report.md §7` and all environment recommendations from `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md`.

## Non-Goals

- NO RMSE tuning. No clamps. No tanh caps. No new stabilizers.
- NO modifications to operational `wrf.exe` at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/`. Pre/post sha256 enforced.
- NO modifications to `src/gpuwrf/dynamics/acoustic_wrf.py` body.
- NO multi-operator parity in this sprint (`calc_coef_w` only). Other operators are M6B1+.
- NO 1h / 24h forecast.
- NO `chmod a-w` of the operational binary without explicit principal approval.
- NO promotion of ADR-025 to ACCEPTED — that is the reviewer's call at sprint close.
- NO remote push.
- NO selecting the GPU operator path in the WRF namelist for the instrumented run (binary has both CPU+GPU paths; use CPU for first parity).

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b0r` on branch `worker/gpt/m6b0r-real-fortran-emission`.

Write-only:
- `external/wrf_savepoint_patch/` (NEW dir, may overwrite scaffold from M6B0)
  - `dyn_em/savepoint_wrapper.F90` — the Fortran wrapper module (new file in the instrumented build tree)
  - `solve_em.F.patch` — the call-site patch to be applied at build time
  - `configure.wrf.patch` — adds `WRF_SAVEPOINT` to `CPP_OPTS`
  - `build.sh` — does pre-sha-check, source `env_wrf_gpu.sh`, patch apply, `compile em_real`, post-sha-check
- `scripts/m6b0r_wrf_savepoint_extract.py` — orchestrator (replaces/extends M6B0's stub)
- `scripts/m6b0r_synthetic_dryrun.py` (NEW, Amendment #1) — synthetic-savepoint dry-run proving fail-closed before real bundles
- `scripts/m6b0r_golden_slice_extract.py` (NEW, Amendment #3) — golden small-domain extraction
- `src/gpuwrf/validation/savepoint_schema.py` (extend M6B0 version) — add machine-readable tolerance ladder per Amendment #4; add advance_w internal-checkpoint enum
- `src/gpuwrf/validation/savepoint_io.py` (extend) — HDF5 with chunking/compression, schema-version, tamper detection
- `tests/test_m6b0r_*` (NEW)
- `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md` — fill in: file format (HDF5), instrumentation strategy (Fortran wrapper module), CPU-path namelist switch documented, tolerance ladder schema, golden-slice mandatory ordering
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/sprint-contract.md` (this file)
2. `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md` (env recommendations — **READ EVERY R1-R7**)
3. `.agent/sprints/2026-05-24-m6b0-plan-critic/reviewer-report.md` (the 5 amendments — **READ §7**)
4. `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/worker-report.md` (what M6B0 already built)
5. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md`
6. `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`
7. **Canonical WRF source** (per env audit Part 2): `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/` (git head `115e5756...`)
8. **Env script** (mandatory before nvfortran): `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`
9. Operator line ranges in `dyn_em/module_small_step_em.F`: per env audit Part 2 table
10. `solve_em.F` call sites: per env audit Part 2 line table (2544-4448)
11. Gen2 d02 input: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-22_00:00:00`

## Acceptance Criteria

### Stage 0 — Pre-flight (MANDATORY)

`build.sh` MUST:
1. Verify operational `wrf.exe` sha256 = `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` BEFORE doing anything; FAIL if changed.
2. Verify canonical WRF source git head = `115e5756f98ee2370d62b6709baac6417d8f7338`; FAIL if drifted.
3. Source `env_wrf_gpu.sh`.

Capture proof: `proof_preflight.txt`.

### Stage 1 — Fortran wrapper module + relinked WRF build (MANDATORY)

- Create `dyn_em/savepoint_wrapper.F90` exposing one subroutine per M6B0-R operator boundary. Initial set (Amendment #2):
  - `sp_calc_coef_w_pre`, `sp_calc_coef_w_post` (PRIMARY parity target)
  - `sp_small_step_prep_post` (MU/MUTS/ww start-of-step state)
  - `sp_advance_mu_t_pre`, `sp_advance_mu_t_post` (MU/MUDF/MUTS/MUAVE/ww/θ update — Amendment #2)
  - `sp_advance_uv_post` (PGF + emdiv state — Amendment #2)
  - `sp_advance_w_rhs_ready`, `sp_advance_w_raw_w`, `sp_advance_w_tridiag_fwd`, `sp_advance_w_tridiag_back`, `sp_advance_w_rayleigh`, `sp_advance_w_ph_final` (6 internal checkpoints — Amendment #2)
  - `sp_calc_p_rho_post` (pressure/geopotential restore)
  - `sp_small_step_finish_post` (end-of-step)
  - `sp_acoustic_substep_boundary`, `sp_rk_stage_boundary` (composite boundaries)
- Each subroutine writes one HDF5 file per call with: variable name, dtype, shape, stagger, units, RK stage idx, acoustic substep idx, run-ID, WRF commit hash, namelist hash, dt, domain idx, map factors, vertical-grid params, operator name. Use `h5fortran` (bundled with HDF5 1.14.5) or hand-rolled HDF5 calls via `H5*` Fortran API.
- Wrapper integration: `solve_em.F.patch` adds `#ifdef WRF_SAVEPOINT` blocks around each existing `CALL` to inject `sp_*` calls. Patch must not change subroutine signatures or runtime behavior when `WRF_SAVEPOINT` is undefined.
- `configure.wrf.patch` appends `-DWRF_SAVEPOINT` to `CPP_OPTS`.
- `build.sh` runs `./compile em_real` into `external/wrf_savepoint_patch/build/`. Final binary path: `external/wrf_savepoint_patch/build/main/wrf.exe.instrumented`.
- Decision (Amendment for ADR-025): **document the namelist flag** that selects the CPU operator path (not the GPU one). Record in `external/wrf_savepoint_patch/namelist.savepoint`.

Capture proofs: `proof_build_log.txt`, `proof_instrumented_sha256.txt`, `proof_operational_unchanged.txt`.

### Stage 2 — Synthetic-savepoint dry-run (MANDATORY, Amendment #1)

`scripts/m6b0r_synthetic_dryrun.py`:
- Generate a tiny synthetic savepoint (4×4×4, all metadata populated) WITHOUT running WRF
- Round-trip write→read with the M6B0 schema
- Run the comparator against itself (clean parity expected)
- Inject a +1e-3 perturbation into one field and verify the comparator reports it correctly (delta exceeds the tolerance)
- Test schema-version mismatch detection (write v1, attempt read with strict v0 reader; expect clean failure)
- Test tamper detection (corrupt one byte; expect detection)

Capture proof: `proof_synthetic_dryrun.txt` + `proof_synthetic_dryrun.json`.

**This stage MUST pass before Stage 3.** If schema validation or fail-closed semantics break, Stage 3 is meaningless.

### Stage 3 — Real CPU-WRF savepoint extraction (MANDATORY)

Run `external/wrf_savepoint_patch/build/main/wrf.exe.instrumented` with namelist set to CPU operator path. Three tiers:
- **Tier 1 (column)**: 1 column extracted from Gen2 d02 IC; emit savepoints for `calc_coef_w_pre/post` across 1, 2, 5, 10 acoustic substeps.
- **Tier 2 (16×16 patch)**: 16×16 patch from Gen2 d02; same step coverage.
- **Tier 3 (golden small-domain slice)** (Amendment #3, MANDATORY): a pinned-run-ID Canary d02 sub-domain (recommend 64×40×44 over a flat-ocean region) with terrain, map-factor, and lateral-boundary metadata present. Run for 10 acoustic substeps. Pin the run ID in `proof_golden_slice_runid.txt`.
- Full d02 is **deferred** per env-audit storage budget (517 GB/hr infeasible).

Capture proofs: `proof_savepoint_column.txt`, `proof_savepoint_patch16.txt`, `proof_savepoint_golden.txt`, `proof_storage_actual.txt`.

### Stage 4 — Machine-readable tolerance ladder (MANDATORY, Amendment #4)

`src/gpuwrf/validation/savepoint_schema.py` MUST expose a `tolerance_ladder.yaml` (or JSON) with per-field entries:
- field name + units + dtype
- abs / rel / ULP threshold (any of the three may be used)
- accumulation exception rule (e.g., field-accumulated MUTS allowed 2× ULP per step)
- perturbation magnitude rule: deliberate test perturbation MUST be ≥ 10× the applicable pass tolerance

Generate the comparator's pass/fail decision from this ladder. Update the M6B0 perturbation test to use the new ladder.

Capture proof: `proof_tolerance_ladder.yaml` + `proof_tolerance_ladder_applied.txt`.

### Stage 5 — First REAL JAX-vs-WRF coefficient parity (MANDATORY)

For `calc_coef_w` only:
- For each of Tier 1, Tier 2, Tier 3, load the WRF savepoint pre-state, run the JAX `calc_coef_w` equivalent on the same inputs, compare against the WRF post-state.
- Sanitizer-OFF. No clamps. No tolerance relaxation outside the ladder.
- Report per-field max-abs delta vs ladder tolerance. Per-tier pass/fail.
- **If parity is NOT achieved** at any tier, the sprint outcome is `PARITY-DEFECT-LOCALIZED` (still a successful sprint — the harness exposed a real bug). Document the discrepancy with field, location, magnitude, and route it to M6B1. **Do NOT try to fix the JAX operator in this sprint.**

Capture proof: `proof_real_coefficient_parity.json` + `proof_real_coefficient_parity.txt`.

### Stage 6 — ADR-025 promotion to PROPOSED (MANDATORY)

Update `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-DRAFT.md` to resolve every open question:
- File format: **HDF5** (committed)
- Instrumentation: **Fortran wrapper module + `#ifdef WRF_SAVEPOINT`** (committed)
- Schema field list: frozen (committed)
- Tolerance ladder: committed per Stage 4
- Golden-domain ordering: **mandatory between 16×16 and full d02** (committed)
- CPU vs GPU operator path: **CPU for parity; GPU evaluation deferred to M6B6+** (committed)

Rename the file from `ADR-025-...-DRAFT.md` to `ADR-025-...-PROPOSED.md` if rebrand is straightforward; otherwise just update the Status header.

### Stage 7 — No regression (MANDATORY)

Full pytest set + new tests:
```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v
```
All PASS.

### Stage 8 — Worker report

`worker-report.md` covering all 7 stages, build wall-time, savepoint storage actual, ladder, parity result + dissent, ADR-025 changes, files changed, risks, handoff to M6B1.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b0r
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_build_log.txt
python scripts/m6b0r_synthetic_dryrun.py 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_synthetic_dryrun.txt
python scripts/m6b0r_wrf_savepoint_extract.py --tier column --steps 10  2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_savepoint_column.txt
python scripts/m6b0r_wrf_savepoint_extract.py --tier patch16 --steps 10  2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_savepoint_patch16.txt
python scripts/m6b0r_golden_slice_extract.py --steps 10 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_savepoint_golden.txt
python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_real_coefficient_parity.txt
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_no_regression.txt
```

## Performance Metrics

- N/A for correctness sprint. Performance gated by post-M6c.
- Build wall-time: report informational; nvfortran `compile em_real` typically 10–30 min on this machine.

## Proof Object

All proofs listed Stages 0–7, plus:
- `worker-report.md`
- ADR-025 updated to PROPOSED
- Branch `worker/gpt/m6b0r-real-fortran-emission` committed

Time budget: **16–32 hours** of focused work (WRF rebuild adds 1–3 hours; rest is comparator + extraction).

## Kill Gates (Amendment #5)

- **Cannot produce instrumented WRF binary + fail-closed comparator in ≤2 sprints** → escalate to manager; consider AceCAST commercial-discovery sprint (E-lane defer can be reversed) OR external WRF-expert review.
- Multi-suspect changes claimed as single fix → reject.
- Post-sanitize "passing" acceptance → reject.
- Operational `wrf.exe` sha256 modified at any point → STOP, revert, escalate.

## Risks

- nvfortran's HDF5 Fortran API may have surprise quirks under WRF's existing config.wrf — keep `gfortran fallback` as plan B (env audit notes none installed; codex may install via apt if needed for the wrapper module only — not for the main WRF binary).
- `solve_em.F` patch may collide with other branches in the future; keep patch as a `.patch` file, not as a permanent fork.
- Storage: Tier-3 golden slice keep ≤ 5 GB.
- CPU budget: cores 0-3 (taskset). 28 cores reserved for CPU WRF baseline.

## Handoff Requirements

When all proofs + ADR-025 PROPOSED + worker-report.md committed on branch `worker/gpt/m6b0r-real-fortran-emission`: `/exit`. Manager will read `worker-report.md` and dispatch M6B1 (extend to next operator: `advance_mu_t` parity).

## Failure modes the manager will reject

- Skipping Stage 2 (synthetic dry-run) and going straight to real extraction.
- Skipping Stage 3 Tier-3 (golden slice).
- "Parity demonstrated" with comparator self-consistency only (M6B0's mistake).
- Operational `wrf.exe` modified.
- Tolerance ladder is a hardcoded constant instead of a machine-readable file.
- Multi-operator parity claimed (other operators are M6B1+).
- Acceptance based on post-sanitize finiteness.
