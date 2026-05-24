# Sprint Contract — M6B Ladder Hygiene Cleanup (opus tester)

**Status:** Pre-drafted 2026-05-25. **Activates after M6B3 merges, before M6B4 dispatches** (per ladder-audit recommendation).

## Objective

Close the 6 hygiene defects surfaced by the M6B ladder cumulative audit (`.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/audit_memo.md`):

1. **`external/wrf_savepoint_patch/solve_em.F.patch` malformed** — cross-file pollution (some hunks target `module_small_step_em.F` but header says `solve_em.F`), bare `@@` markers without offsets, wrong line counts. `patch -p1 --dry-run` returns RC=2.
2. **10/16 wrapper hooks still empty-body** — M6B0-R/M6B1/M6B2 hooks defined but bodies are no-ops; emission still goes through Python extractor. (NOTE: full ABI fix is a separate `m6b0r-fortran-hook-abi-followup` sprint; this hygiene sprint only documents/inventories which hooks are stubs.)
3. **SCHEMA_VERSION not bumped** — `savepoint_schema.py` still `m6b0r-savepoint-v1` and `m6b0r-tolerance-ladder-v1` after 3 extensions.
4. **M6B1 worker-report omits Critic Amendment #1 classification** — backfill the operational-compatibility table for the M6B1 sprint deliverables.
5. **Comparator drift** — `_threshold` + `_field_compare` duplicated across `scripts/m6b{0r,1,2}_*_compare.py`; M6B1 hardcodes Gen2 path; M6B2 has `sys.path.insert` cross-script hack.
6. **Pre-M6B0-R legacy fields accreting** — `cofrz/cofwr/cofwz/coftz/cofwt/rdzw` only used by one diagnostic; remove or clearly mark as legacy.

This sprint also adds **future-proofing rules** to the wrapper system so M6B4/B5/B6 don't keep re-creating the same drift.

## Non-Goals

- NO changes to JAX runtime semantics (acoustic_wrf.py / mu_t_advance.py / tridiag_solve.py).
- NO changes to operational `wrf.exe`. Pre/post sha256 check.
- NO new operator parity work — that's M6B4's job.
- NO promotion of ADR-025 or any ADR.
- NO modification of comparator parity verdicts (don't recompute calc_coef_w deltas; trust the M6B0-R fix).
- NO Fortran hook ABI full fix (that's the queued `m6b0r-fortran-hook-abi-followup`).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_hygiene` on branch `tester/opus/m6b-ladder-hygiene-cleanup`.

Write-only:
- `external/wrf_savepoint_patch/solve_em.F.patch` — repair to valid unified diff applying to `dyn_em/solve_em.F` only
- `external/wrf_savepoint_patch/module_small_step_em.F.patch` (NEW) — extract M6B2's tridiag chunks into their own file targeting `dyn_em/module_small_step_em.F`
- `external/wrf_savepoint_patch/build_relinked.sh` — update to apply both patches
- `src/gpuwrf/validation/savepoint_schema.py` — bump `SCHEMA_VERSION` to v2 (or per-extension counter); add deprecation marker on legacy fields
- `src/gpuwrf/validation/tolerance_ladder.json` — bump version; mark legacy entries
- `src/gpuwrf/validation/comparator_common.py` (NEW) — extract `_threshold`, `_field_compare`, common comparator utilities; updates the 3 comparator scripts to import from this module
- `scripts/m6b1_advance_mu_t_compare.py` — parameterize hardcoded Gen2 path via CLI flag
- `scripts/m6b2_tridiag_solve_compare.py` — remove `sys.path.insert` hack
- `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/operational_compatibility_backfill.md` (NEW) — backfill M6B1's Amendment #1 classification table
- `external/wrf_savepoint_patch/HOOK_INVENTORY.md` (NEW) — table of all 16 hooks: which are empty-body stubs vs typed, which are wired in solve_em.F vs module_small_step_em.F
- `tests/test_m6b_hygiene_patch_apply.py` (NEW) — regression: assert `patch -p1 --dry-run` on both patches returns RC=0
- `.agent/sprints/2026-05-25-m6b-ladder-hygiene-cleanup/` — proofs + memo

Read-only:
- Don't touch M6B3 sprint dir or M6B4 worktree if it exists yet
- Don't touch JAX dynamics code (acoustic_wrf.py / mu_t_advance.py / tridiag_solve.py)
- Don't touch operational WRF / canonical WRF source

## Inputs (mandatory)

1. `.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/audit_memo.md` (the audit findings)
2. `.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/proof_patch_dryrun.txt` (the RC=2 evidence)
3. `external/wrf_savepoint_patch/solve_em.F.patch` (current malformed state)
4. `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` (the 16 hook surface)
5. All 3 comparator scripts in `scripts/m6b*_compare.py`
6. M6B0-R, M6B1, M6B2 worker-reports
7. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F` + `module_small_step_em.F`
8. `PROJECT_PLAN.md §14.5.1` (Amendment #1 classification requirement)

## Acceptance Criteria

### Stage 1 — Repair solve_em.F.patch + extract module_small_step_em.F.patch (MANDATORY)

- Inspect each existing hunk; determine target file (`solve_em.F` vs `module_small_step_em.F`)
- Build clean unified-diff `solve_em.F.patch` with proper `@@ -OLDLINE,COUNT +NEWLINE,COUNT @@` headers
- Build new `module_small_step_em.F.patch` with M6B2's tridiag chunks (advance_w internal)
- `patch -p1 --dry-run < solve_em.F.patch` returns RC=0 against canonical WRF source
- `patch -p1 --dry-run < module_small_step_em.F.patch` returns RC=0 against canonical WRF source
- Update `build_relinked.sh` to apply both
- Update `build_registry.json` schema in `build.sh` to list both patches

Capture proof: `proof_patch_dryrun_after.txt` (RC=0 for both).

### Stage 2 — Hook inventory + classification (MANDATORY)

`external/wrf_savepoint_patch/HOOK_INVENTORY.md`:

| Hook | Target file | Hook body | Wired-in-solve_em? | Active emission? | Defining sprint |
|---|---|---|---|---|---|
| `sp_calc_coef_w_pre` | `solve_em.F:2685,2720` | EMPTY | YES | NO | M6B0-R |
| `sp_calc_coef_w_post` | ... | EMPTY | YES | NO | M6B0-R |
| `sp_advance_mu_t_pre` | `solve_em.F:?` | TYPED ARGS | YES | NO (stub body) | M6B1 |
| `sp_advance_mu_t_post` | ... | TYPED ARGS | YES | NO (stub body) | M6B1 |
| `sp_advance_w_tridiag_fwd_pre` | `module_small_step_em.F:1533` | TYPED ARGS | YES | NO (stub body) | M6B2 |
| ... | ... | ... | ... | ... | ... |

All 16 hooks listed.

### Stage 3 — SCHEMA_VERSION bump (MANDATORY)

- `savepoint_schema.py`: `SCHEMA_VERSION = "m6b2-savepoint-v4"` (or whatever post-M6B3 should be — coordinate with M6B3 merge timing)
- `tolerance_ladder.json`: `version = "m6b2-tolerance-ladder-v4"`
- Document the bump rationale: per-sprint version monotonic
- Add reader test that rejects schema-version mismatch (already tested per M6B0-R, just verify)

### Stage 4 — Comparator deduplication (MANDATORY)

- Create `src/gpuwrf/validation/comparator_common.py` with extracted `_threshold`, `_field_compare`, common CLI arg parser
- Refactor M6B0-R, M6B1, M6B2 comparator scripts to import from `comparator_common`
- Verify all 3 still produce identical output (re-run each comparator; outputs unchanged)
- Parameterize M6B1 Gen2 path via `--gen2-runs-dir` flag
- Remove M6B2's `sys.path.insert` (proper package import)

Capture proof: before/after parity output verbatim.

### Stage 5 — M6B1 Amendment #1 backfill (MANDATORY)

`.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/operational_compatibility_backfill.md`:

| Item | Classification | Evidence |
|---|---|---|
| `sp_advance_mu_t_pre/post` hooks | Validation-only | savepoint emission |
| `mu_t_advance.advance_mu_t_wrf` callable | Validation-only | NOT wired into runtime; only invoked by comparator |
| New ladder entries for mu/mudf/muts/muave/ww/theta/ph_tend | Validation-only | tolerance values are validation tolerances |
| Default state-API impact | NONE | no operational carry change |

### Stage 6 — Legacy field cleanup (MANDATORY)

For `cofrz/cofwr/cofwz/coftz/cofwt/rdzw` (per audit Part 6):
- `grep -rn 'cofrz\|cofwr\|cofwz\|coftz\|cofwt\|rdzw' src/ scripts/ tests/`
- If used only by 1 diagnostic: mark with `# DEPRECATED M6B0-R; remove in M6B7`
- If unused: delete

Capture: `proof_legacy_field_audit.txt`.

### Stage 7 — Patch-apply regression test (MANDATORY)

`tests/test_m6b_hygiene_patch_apply.py`: asserts the two patches apply cleanly to a temp copy of canonical WRF source. Run on every CI invocation. Future sprints that extend the patches now have a guard against re-introducing malformedness.

### Stage 8 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py -v
```

All previously-passing tests still pass. New patch-apply test passes.

### Stage 9 — Worker report

`worker-report.md`: per-defect remediation status, before/after patch RC, schema version bumps, comparator dedup summary, legacy cleanup count, files changed, handoff to M6B4.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_hygiene
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/solve_em.F.patch
patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/module_small_step_em.F.patch
pytest tests/test_m6b_hygiene_patch_apply.py -v
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-ladder-hygiene-cleanup/proof_no_regression.txt
```

## Performance Metrics

N/A — hygiene sprint.

## Risks

- M6B3 may have extended `solve_em.F.patch` already by the time this sprint dispatches. Worker must rebase against post-M6B3 state and incorporate M6B3's extensions in the repair.
- The legacy field grep may surface unexpected dependencies — be conservative; mark DEPRECATED rather than delete if uncertain.
- Comparator output formatting changes may break downstream parsing. Verify before/after parity output is byte-identical.

## Handoff Requirements

When all proofs + worker-report committed on branch `tester/opus/m6b-ladder-hygiene-cleanup`: stop. Manager merges + dispatches M6B4 acoustic recurrence parity.

Time budget: **90-150 min**.
