# CPU test-hygiene: honest env-failure marking (worker/opus/test-hygiene)

Base: trunk `695b140`. Method: every `tests/**/test_*.py` (+`evals/agentos`) run in its OWN
process (`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""`), so a SIGSEGV in one file cannot mask the rest.
All runs CPU-only — zero GPU JAX context (the release GATE owns the GPU).

## Headline before/after (per-file rc, 246 files)

| state  | rc=0 (clean) | rc=1 (fail) | rc=-11 (SIGSEGV) |
|--------|-------------:|------------:|-----------------:|
| BEFORE |          185 |          60 |                1 |
| AFTER  |          222 |          24 |                0 |

37 files flipped rc!=0 -> rc=0. The 1 CPU-SIGSEGV is eliminated. The 24 remaining rc=1 are
**intentionally LEFT** real/ambiguous failures (NOT env) — listed below, NOT masked.
(Per-test transition was measured on the first 61-file marking pass: 250 passed->passed, 125
failed->skipped, 51 skipped->skipped, 33 failed->failed, 2 SEGV->skipped, 0 violations.)

## Per-TEST before->after transition on the 61 touched files (PROOF: no passer masked)

| count | transition          | meaning                                            |
|------:|---------------------|----------------------------------------------------|
|   250 | passed  -> passed   | every previously-passing test still passes         |
|   125 | failed  -> skipped  | env failures now honestly skipped                  |
|    51 | skipped -> skipped  | pre-existing skips unchanged                       |
|    33 | failed  -> failed   | intentionally-left real/ambiguous failures (unmasked)|
|     2 | (SEGV) -> skipped   | m6b6 heavy tests: crashed-unrecordable -> clean skip|

**VIOLATIONS (passed-before now NOT passed): 0.** This is the binding guardrail check.

## Mechanism

1. `tests/conftest.py` (NEW) — a `pytest_runtest_makereport` hookwrapper that converts ONLY the
   exact `RuntimeError("State.zeros requires a GPU device; no JAX GPU backend is visible")`
   (from `gpuwrf.contracts.state._gpu_device`, raised ONLY when no GPU backend is visible) into a
   SKIP. Surgical: a real `AssertionError`/other exception still FAILS; passing tests untouched;
   handles both setup- and call-phase. Replaces dozens of file-level skipifs that would have wrongly
   skipped the many CPU-runnable tests sharing those files. On a GPU run the guard never fires, so
   these tests run/assert normally.
2. Explicit `@pytest.mark.skipif` on specific tests for: purged Gen2 corpus dir, un-vendored
   external fixture payload, purged WPS-case artifacts, M2 GPU/CUDA-toolchain bakeoff, and the
   coupled-step CPU-SIGSEGV.

## Files/tests marked (honest env failures)

### A. GPU-required (conftest hook; `State.zeros requires a GPU device` on CPU)
Whole-file (all failing tests were GPU-required): test_m3_dummy_loop, test_m3_halo, test_m3_state,
test_m4_acoustic, test_m4_advection, test_m4_debug_hooks, test_m4_dycore_step, test_m4_rk3,
test_m4_tester_adversarial_attempt2, test_m4_tier2_invariants, test_m4_tier3_convergence,
test_m6_boundary_apply, test_m6_dummy_coupled, test_m6_dycore_cap_lift, test_m6_precision_matrix,
test_m6_state_extension, test_m6_tier2_coupled, test_m6_tier3_tsc, test_m6x_pressure_diagnose_wiring,
test_m6x_tier3_convergence_infra, test_m6x_warm_bubble_operator_sanity.
Partial (GPU tests skipped, real asserts remain FAILED — see section D): test_m3_edge_cases,
test_m4_tester_adversarial, test_m7_skill_fix_algorithmic, test_m7_skill_fix_iter2.

### B. Purged corpus / un-vendored data (explicit skipif on path existence)
- test_m6_gen2_accessor — module skipif on `DEFAULT_M6_GEN2_RUN_DIR.exists()` (corpus run dir not vendored)
- test_m6_validation_io — same module skipif
- test_m7_default_gen2_run_dir — corrected existing skipif from `/mnt/...mounted` to the actual `DEFAULT_M6_GEN2_RUN_DIR.exists()`
- test_canary_wrf_fixture — 3 tests skipif on external `full.npz` payload (only the small slice SAMPLE is committed)
- test_m7_s0a_schemas — 1 test skipif when the cited Gen2 WPS-case artifacts (namelist.wps/ungrib gribs) are purged

### C. M2 GPU/CUDA-toolchain bakeoff (explicit skipif `jax.default_backend()!="gpu"`, per failing test)
- test_m2_cuda_tile (2), test_m2_cupy (1), test_m2_jax (1), test_m2_kokkos (2), test_m2_triton (1)
- test_m2_cuda_tile_edge_cases (1), test_m2_kokkos_edge_cases (2), test_m2_cupy_edge_cases (8),
  test_m2_jax_edge_cases (9), test_m2_triton_edge_cases (8)
  (only the GPU/venv/CUDA-bound tests; the committed static-artifact + pure-python tests still run)

### D. CPU-SIGSEGV coupled-step parity (explicit skipif `jax.default_backend()=="cpu"`)
- test_m6b6_coupled_step_parity — 2 heavy tests (build a huge XLA graph that segfaults the CPU
  backend); 3 schema/constant tests in the file still run. Matches the existing test_dycore_100_steps pattern.

## D. Files LEFT UNMARKED + REPORTED (REAL or ambiguous — NOT environmental; NOT masked)

These still FAIL on CPU. They are NOT GPU-required / not purged-data / not missing-backend, so per the
guardrail they were left alone for the manager to triage:

| file | failing assertion | nature |
|------|-------------------|--------|
| test_m3_edge_cases | contract-allocator drift; `fp32 != fp64`; `DID NOT RAISE ValueError` | source/contract drift (3 tests; GPU ones skipped) |
| test_m4_tester_adversarial | `DID NOT RAISE ValueError` (halospec) | real assert (1 test; GPU ones skipped) |
| test_m7_skill_fix_algorithmic / _iter2 | `assert 450.0 == 700.0` (theta-ceiling) | real assert (GPU ones skipped) |
| test_m5_rrtmg_tier1 | `flux_down err 2.546 <= 1.0` | REAL RRTMG-SW clear-sky accuracy gap (known, see memory) |
| test_m5_rrtmg_gate | `tier1_sw_pass True is False` | gate expects SW FALLBACK; coupled to rrtmg_tier1 above |
| test_m5_rrtmg_harness / test_m5_thompson_fortran_harness / test_m5_mynn_harness | Fortran-harness `.o` absent/checksum | needs WRF Fortran harness build (toolchain) — arguably env but harness-regen logic, left for triage |
| test_m5_thompson_column_shapes | `fusion count 1 != 0` | CPU-vs-GPU XLA fusion-count difference — backend-dependent, left for triage |
| test_m6b4 / test_m6b5 parity | `AttributeError: 'NoneType'.shape` in mu_t_advance | deep dycore CPU path produces None theta — GPU-targeted but not the clean GPU-guard; needs GPU run to confirm |
| test_m6b_dycore_rk_acoustic_fix | `21.98 < 5.0` | numeric (v runaway) — left for triage |
| test_m6b_fix_advance_mu_t_commit / test_m6b_rk1_d2h_acceptance | source-substring asserts | stale source-provenance audit |
| test_m6b_d2h_warmed_zero / _v2 | "missing warmed Nsight trace" | needs GPU Nsight capture (arguably env; left — depends on a GPU profiling run) |
| test_m6x_adr023_production_grade | `rmse 0.666 < 0.15` | numeric MPAS-slice tolerance — left for triage |
| test_m6x_c2_acoustic | `1.29e16 < 1e-08` | numeric blow-up (CPU?) — left for triage |
| test_m6x_c2_metrics | `Array(False)` identity | numeric/identity — left for triage |
| test_m6x_s3narrow_stabilizer_audit | `72 < 28`, `3 == 0` | STALE source-provenance count snapshot (dynamics code evolved) |
| test_m6_boundary_replay | fixture `run_id` mismatch | vendored zarr fixture pinned to a superseded run_id (re-pin needed) |
| test_m7_gen2_corpus_scout_inventory | pinned-run-id snapshot drift | corpus present but contents differ from frozen pin |
| test_paper_control_edge_cases | `KeyError 'missing_citations'` | publication-audit return-schema drift |

## Files OWNED / changed (test files + conftest only; NO src/ changes)
- tests/conftest.py (new)
- tests/test_m6_gen2_accessor.py, tests/test_m6_validation_io.py
- tests/test_m6b6_coupled_step_parity.py
- tests/test_m7_default_gen2_run_dir.py, tests/test_m7_s0a_schemas.py, tests/test_canary_wrf_fixture.py
- tests/test_m2_{cuda_tile,cupy,jax,kokkos,triton}.py
- tests/test_m2_{cuda_tile,cupy,jax,kokkos,triton}_edge_cases.py
