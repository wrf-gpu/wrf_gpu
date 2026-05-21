# M5-S1.y Worker Report - Thompson HLO + Process Residuals

## Objective

Continue the M5-S1.x Thompson microphysics column work by wiring an HLO-safe rain-freezing table gather, tightening the four named process residuals, adding a non-tautological WRF-linked Tier-2 budget check, preserving honest launch/debug-transfer evidence, and amending ADR-006 for reviewer audit.

This report is written for the mandatory Claude Opus 4.7 reviewer pass. The requested prompt path `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/reviewer-report.md` did not exist; I read the actual M5-S1 reviewer artifact `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/reviewer-a5-report.md` plus the named closeout/ADR/example reports.

## Files Changed

- `src/gpuwrf/physics/thompson_tables.py`: packed default-IN rain-freezing table bundle, preserving `data/fixtures/thompson-tables-v1.npz` provenance.
- `src/gpuwrf/physics/thompson_constants.py`: added WRF `TNO`, `ATO`, and `CRE1`; graupel gamma coefficients remain computed with `math.gamma`.
- `src/gpuwrf/physics/thompson_column.py`: default-IN qrfz gather, rain evaporation guard/factor, deposition nucleation staging, graupel melt/sublimation limiter, and process-order flag plumbing.
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`: matched production process order for the zero-debug HLO identity proof.
- `src/gpuwrf/validation/tier2_thompson.py`: added WRF-linked one-step aggregate water/tracked-number budget comparison.
- `scripts/m5_run_thompson.py`: records raw HLO launch marker count, full HLO bytes, tracked HLO bytes, and regenerated proof artifacts.
- `scripts/m5_gate_thompson.py`: fails on raw/reported launch mismatch and emits `GRAY-ZONE` when full HLO exceeds the M5-S1.y size target.
- `tests/test_m5_thompson_constants.py`, `tests/test_m5_thompson_tier2.py`, `tests/test_m5_thompson_process_residuals.py`: qrfz shape, WRF-linked Tier-2, and focused process residual regression checks.
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`: M5-S1.y amendment with HLO pattern, process map, evidence, and remaining strict-parity debt.
- `artifacts/m5/*` and `fixtures/manifests/analytic-thompson-column-v1.yaml`: regenerated proof objects and Thompson harness SHA.

## AC Status

| AC | Status | Evidence |
| --- | --- | --- |
| AC1 HLO-safe rain-freezing gather | Partial / reviewer decision needed | Packed 2-D default-IN qrfz gather is implemented at `src/gpuwrf/physics/thompson_tables.py:104-113` and consumed by `_take_qrfz` at `src/gpuwrf/physics/thompson_column.py:263-269` and the rain-freezing path at `src/gpuwrf/physics/thompson_column.py:556-584`. `artifacts/m5/thompson_profile.json` reports `kernel_launches_per_step=10` and `raw_hlo_launch_marker_count=10`, exactly +5 over the M5-S1.x baseline of 5. Full HLO is `421083` bytes, above the 350 KB target, so `scripts/m5_gate_thompson.py:78-80` emits `GRAY-ZONE`. |
| AC2 process residual closure | Partial | Rain evaporation and deposition nucleation improved sharply, but strict field targets are not all met. Dominant remaining `qg/qv/T` residual is tied to WRF collision table paths not in the S1.x table asset. See per-process table below. |
| AC3 non-tautological Tier-2 | Partial / useful but not per-process | `src/gpuwrf/validation/tier2_thompson.py:53-76` compares one-step aggregate candidate deltas against the WRF-linked Fortran harness fixture, not the JAX path on both sides. The harness still does not expose per-process tendencies or internal `Ns/Ng/Qb`, so number budget is carry-forward diagnostic. |
| AC4 no fudge | Met | `scripts/m5_run_thompson.py:64-93` derives launch count from compiled HLO; `scripts/m5_run_thompson.py:108-122` records raw/reported launches and zero post-init transfers. `scripts/m5_gate_thompson.py:55-57` falls back if raw and reported launches diverge. Debug-vs-stripped diff SHA is the empty-file SHA with a zero-byte diff. |
| AC5 ADR amendment | Met | `.agent/decisions/ADR-006-thompson-jax-implementation.md:74-86` documents the chosen gather pattern, HLO outcome, process map, and strict-parity debt. |

## WRF Formula Map

- Rain-freezing indexes are transcribed from `module_mp_thompson.F.pre:2374-2400`. The table use is `module_mp_thompson.F.pre:2658-2669`; table construction/provenance is `module_mp_thompson.F.pre:4664-4855`. JAX packs the default non-aerosol IN slice because WRF sets `xni = 1.0 *1000.` at `module_mp_thompson.F.pre:2637-2641` and derives `idx_IN` at `module_mp_thompson.F.pre:2643-2656`.
- Rain evaporation maps `module_mp_thompson.F.pre:3561-3638`. The important residual fixes are the cloud-condensation skip guard at `module_mp_thompson.F.pre:3565-3566` and graupel-melt evaporation factor at `module_mp_thompson.F.pre:3617-3620`, implemented at `src/gpuwrf/physics/thompson_column.py:463-513`.
- Cloud-water/ice nucleation maps `module_mp_thompson.F.pre:2684-2695`. WRF constants `TNO=5.0` and `ATO=0.304` come from `module_mp_thompson.F.pre:188-189` and are in `src/gpuwrf/physics/thompson_constants.py:20-21`. JAX staging is `src/gpuwrf/physics/thompson_column.py:595-607`.
- Cloud-ice deposition/number handling maps `module_mp_thompson.F.pre:2709-2742`, especially the sublimation-only `pni_ide` update at `module_mp_thompson.F.pre:2719-2727`; JAX keeps positive deposition from creating new cloud-ice number at `src/gpuwrf/physics/thompson_column.py:650-703`.
- Graupel sublimation and melting map `module_mp_thompson.F.pre:2760-2770` and `module_mp_thompson.F.pre:2845-2889`, with the cross-deposition vapor limiter at `module_mp_thompson.F.pre:2922-2939`; JAX is `src/gpuwrf/physics/thompson_column.py:617-635` and `src/gpuwrf/physics/thompson_column.py:672-684`.
- Final number/mass balancing maps `module_mp_thompson.F.pre:4033-4142`; the current JAX `_finish` remains a subset and the harness does not expose WRF's internal `Ns/Ng/Qb` finalization state.

Coefficient verification was computational: the graupel `CGE11 = 0.5 * (0.640961647 + 5 + 2*0) = 2.8204808235`, `math.gamma(CGE11) = 1.7057543783678366`, and the resulting `0.28*Sc3*sqrt(av_g)*CGG11 = 4.904839488524536`. The constant test checks `CGG11 == math.gamma(CGE11)` at `tests/test_m5_thompson_constants.py:35-38`, avoiding the prior CGG11 literal-copy failure pattern.

## Per-Process Residual Table

| Process | WRF lines | JAX lines | Focused evidence | Remaining risk |
| --- | --- | --- | --- | --- |
| Rain evaporation | `module_mp_thompson.F.pre:3561-3638` | `src/gpuwrf/physics/thompson_column.py:463-513` | Focus cell `(2,2)` now matches WRF for `qv`, `qr`, `qg`, and `T` within `tests/test_m5_thompson_process_residuals.py:19-25`. Global `qr` max abs is `3.327517035206257e-08`, meeting the `<=1e-7` target. | `Nr` remains carry-forward because WRF's final rain-number redistribution includes paths outside this subset. |
| Graupel sublimation/melting | `module_mp_thompson.F.pre:2760-2770`, `2845-2889`, limiter `2922-2939` | `src/gpuwrf/physics/thompson_column.py:617-635`, `672-684` | Warm graupel melt plus rain evaporation cell `(2,2)` mass/temperature is now tight. | Global `qg=2.9509294563467847e-06`, `qv=2.9510734091959916e-06`, `T=0.008302000075161686 K` remain above target. Dominant residual cell leaves JAX `qg=0` while WRF retains about `2.95e-6`; likely missing rain/snow/graupel collision table offsets at `module_mp_thompson.F.pre:2547-2609`. |
| Cloud-water freezing/nucleation | `module_mp_thompson.F.pre:2684-2695`, `4005-4031` | `src/gpuwrf/physics/thompson_column.py:595-607`, `522-544` | Focus cell `(1,8)` matches WRF `Ni <= 1`, `qi/qc <= 1e-8`, `T <= 1e-4` in `tests/test_m5_thompson_process_residuals.py:28-34`. Global `Ni` improved from `126975` to `772.3940057673026`. | Global target `Ni <= 10` not met; exact WRF finalization and collision/ice-number side paths still missing. |
| Number-balance finalization | `module_mp_thompson.F.pre:4033-4142` | `_finish` plus `src/gpuwrf/validation/tier2_thompson.py:53-76` | Tracked number residual bounded by `Nr <= 1e5`, `Ni <= 1e3` in `tests/test_m5_thompson_process_residuals.py:37-42`; Tier-2 WRF budget records tracked-number pass under `1e5`. | Strict targets `Ni,Nr <= 10` are not met (`Ni=772.3940057673026`, `Nr=33653.28915198239`). This is explicitly recorded, not closed. |

## Tier-1 Residuals

`python scripts/m5_run_thompson.py` regenerated `artifacts/m5/tier1_thompson_parity.json` with these max absolute residuals:

| Field | Max abs | Target status |
| --- | ---: | --- |
| `qv` | `2.9510734091959916e-06` | Miss |
| `qc` | `2.375154443240955e-09` | Improved, still above `<=1e-9` |
| `qr` | `3.327517035206257e-08` | Met `<=1e-7` |
| `qi` | `4.855188355740517e-09` | Improved, still above `<=1e-9` |
| `qs` | `9.2685395199886e-11` | Met |
| `qg` | `2.9509294563467847e-06` | Miss |
| `Ni` | `772.3940057673026` | Miss `<=10` |
| `Nr` | `33653.28915198239` | Miss `<=10` |
| `T` | `0.008302000075161686` | Miss `<=1e-4` |

This is meaningful progress over M5-S1.x for `qr`, `qc`, `qi`, and `Ni`, but it is not strict closure.

## HLO and Transfer Evidence

`artifacts/m5/thompson_profile.json` records:

- `kernel_launches_per_step = 10`
- `raw_hlo_launch_marker_count = 10`
- `hlo_full_bytes = 421083`
- `hlo_tracked_bytes = 73243`
- `host_to_device_bytes_post_init = 0`
- `device_to_host_bytes_post_init = 0`
- `temporary_bytes_per_step = 0`

The launch count is honest and unfudged: `raw_hlo_launch_marker_count == kernel_launches_per_step`. The gate result is `GRAY-ZONE`, not `GO`, because full HLO size exceeds the target. This matches AC4's no-fudge requirement and leaves AC1's HLO-size risk visible for manager/reviewer decision.

## Commands Run

- `bash scripts/wrf_thompson_harness_build.sh` - passed.
- `python scripts/m5_generate_thompson_fixture.py` - passed; manifest harness SHA updated to `46d97536ff09c5bddac365f6a22d0d2cb506c17bc4b1ccdf47e510d8cd0ac011`, table SHA unchanged `a76b0f28e8b910df0a5dde529f02460b5e7a3ea92d9e543f673c43e2a5b02f9f`.
- `python scripts/m5_run_thompson.py` - passed; Tier-1/Tier-2/profile/HLO artifacts regenerated.
- `python scripts/m5_gate_thompson.py` - exited 0 with `gate_status=GRAY-ZONE`.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml` - passed.
- `python scripts/validate_agentos.py` - passed.
- `pytest -q tests/test_m5_thompson_process_residuals.py tests/test_m5_thompson_constants.py tests/test_m5_thompson_tier2.py` - `8 passed`.
- `pytest -q` - failed outside Thompson: missing external Canary WRF fixture files, M2 JAX/Triton environment gaps and no-space package install failure, missing M2 profiler artifacts, and pre-existing RRTMG gray-zone/fallback parity failures. No Thompson test failure was present in the full failure list.

## Proof Objects Produced

- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/hlo_dump/thompson_column_production.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_stripped.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`

## Unresolved Risks

1. AC1 launch delta meets the manager target exactly (+5), but full HLO size is still over target. Reviewer/manager should decide whether `GRAY-ZONE` is acceptable for M6 prologue or whether to require static-axis unroll/surrogate work now.
2. AC2 strict residuals are not closed for `qg`, `qv`, `T`, `Ni`, or `Nr`. The strongest current hypothesis is missing WRF rain/snow/graupel collision table paths at `module_mp_thompson.F.pre:2547-2609`, whose tables were not exported in M5-S1.x.
3. AC3 is WRF-linked but aggregate, not per-process. Per-process oracle dumps require extending the real WRF harness, not worker-authored Fortran substitutes.
4. The current table asset is preserved. If the next sprint wires collision tables or cloud-water freezing tables, `data/fixtures/thompson-tables-v1.npz` and the manifest SHA must be intentionally repinned.

## Next Decision Needed

Decide whether to accept this as an honest M6 prologue checkpoint with `GRAY-ZONE` HLO size and incomplete strict process closure, or amend AC1/AC2 to require a follow-up table-export sprint for collision/cloud-water-freezing tables and a second HLO-size reduction pass.
