# Worker Report - M5-S1.x Thompson lookup-table export

## Objective

Export the initialized WRF Thompson lookup tables into a reproducible repo asset, wire the safe hot-path tables into the JAX Thompson column, regenerate proof artifacts, and report honestly whether the sprint contract's strict ADR-005, HLO-size, and one-launch acceptance criteria are met. The extractor owns the WRF private-table dump path in `scripts/extract_thompson_tables.py`, including table names/shapes from `TABLE_SPECS` and the injected `m5_dump_thompson_tables` subroutine (`scripts/extract_thompson_tables.py:36`, `scripts/extract_thompson_tables.py:122`). The current gate result is not strict GO; it is `GO_CARRYFORWARD` with carry-forward tolerances (`artifacts/m5/thompson_gate_result.json:2`, `artifacts/m5/thompson_gate_result.json:9`).

## Files changed

- Added `scripts/extract_thompson_tables.py`, which finds the WRF source snapshot, compiles a scratch Thompson module, calls `thompson_init`, dumps private tables as Fortran stream data, reshapes the stream using Fortran order, and writes `data/fixtures/thompson-tables-v1.npz` (`scripts/extract_thompson_tables.py:77`, `scripts/extract_thompson_tables.py:285`, `scripts/extract_thompson_tables.py:360`, `scripts/extract_thompson_tables.py:379`).
- Added `src/gpuwrf/physics/thompson_tables.py`, which pins the table asset path, enumerates all exported asset table names, exposes a compact runtime bundle, and records source-line provenance for the mapped WRF tables (`src/gpuwrf/physics/thompson_tables.py:18`, `src/gpuwrf/physics/thompson_tables.py:21`, `src/gpuwrf/physics/thompson_tables.py:56`, `src/gpuwrf/physics/thompson_tables.py:115`).
- Updated `src/gpuwrf/physics/thompson_column.py` to use WRF-exported `snow_sa/snow_sb/cse`, `t_Efrw`, and packed `tps_iaus/tni_iaus/tpi_ide` tables in the active JAX column path (`src/gpuwrf/physics/thompson_column.py:255`, `src/gpuwrf/physics/thompson_column.py:289`, `src/gpuwrf/physics/thompson_column.py:397`, `src/gpuwrf/physics/thompson_column.py:586`).
- Left the large rain-freezing tables extracted and pinned but outside the jitted timestep body because the dynamic 4-D gather path caused an HLO/launch regression; that decision is documented inline and in the blocker (`src/gpuwrf/physics/thompson_column.py:518`, `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:27`).
- Updated the WRF fixture generator and manifest to include the Thompson table asset SHA and file entry (`scripts/m5_generate_thompson_fixture.py:249`, `scripts/m5_generate_thompson_fixture.py:280`, `fixtures/manifests/analytic-thompson-column-v1.yaml:3`, `fixtures/manifests/analytic-thompson-column-v1.yaml:268`).
- Updated `scripts/m5_run_thompson.py`, `scripts/m5_gate_thompson.py`, ADR-006, the M5 proof artifacts, constants, stripped debug sibling, and tests to reflect the table asset, current launch count, and carry-forward status (`scripts/m5_run_thompson.py:135`, `scripts/m5_gate_thompson.py:73`, `.agent/decisions/ADR-006-thompson-jax-implementation.md:60`, `tests/test_m5_thompson_constants.py:41`).

## Table extraction summary

The exported asset contains scalar-grid tables (`r_c/r_i/r_r/r_s/r_g`, `n0r_exp/n0g_exp`, `nt_i/nt_in`, `dr/dc/t_nc`), rain/cloud collection tables (`t_Efrw`, `t_Efsw`), ice autoconversion/deposition tables (`tps_iaus`, `tni_iaus`, `tpi_ide`), rain-freezing tables (`tpi_qrfz`, `tpg_qrfz`, `tni_qrfz`, `tnr_qrfz`), snow/graupel moment coefficients (`snow_sa`, `snow_sb`, `cse`, `csg`, `graupel_cge`, `graupel_cgg`), and graupel scalar arrays (`am_g`, `av_g`, `bv_g`, `rho_g`) (`scripts/extract_thompson_tables.py:36`, `scripts/extract_thompson_tables.py:68`). The reproducible asset is pinned as SHA `a76b0f28e8b910df0a5dde529f02460b5e7a3ea92d9e543f673c43e2a5b02f9f` and 29,429,841 bytes (`fixtures/manifests/analytic-thompson-column-v1.yaml:268`, `fixtures/manifests/analytic-thompson-column-v1.yaml:270`).

WRF provenance is recorded for the mapped tables: `t_Efrw` initialization/consumption is mapped to `module_mp_thompson.F.pre:4921-4977` and `2260-2268`; `tps_iaus/tni_iaus/tpi_ide` are mapped to `4870-4913` and `2719-2742`; rain-freezing tables are mapped to `4664-4855` and `2658-2669`; snow moments are mapped to `337-356,730-750` and `2093-2191`; graupel coefficients are mapped to `73-156,760-770` (`.agent/decisions/ADR-006-thompson-jax-implementation.md:66`, `.agent/decisions/ADR-006-thompson-jax-implementation.md:70`).

## Before/after residuals

Attempt-4 pre-table strict residuals were `qv=1.4304079020558032e-05`, `qc=1.517228938283358e-04`, `qr=4.760876436193939e-06`, `qi=1.3708094759935232e-04`, `qs=1.447943623500527e-04`, `qg=1.5218435328806104e-05`, `Ni=126975.12500000044`, `Nr=67300.453125`, and `T=0.040290844661740266 K` (`.agent/decisions/ADR-006-thompson-jax-implementation.md:76`). Post-table residuals are `qv=4.7608781466789915e-06`, `qc=1.2657008724556353e-07`, `qr=4.760876436193939e-06`, `qi=1.269506099959173e-07`, `qs=9.2685395199886e-11`, `qg=2.9509294563467847e-06`, `Ni=126975.12500000079`, `Nr=67300.453125`, and `T=0.011792500929288963 K` (`artifacts/m5/tier1_thompson_parity.json:15`, `artifacts/m5/tier1_thompson_parity.json:24`).

The table export clearly reduced the largest snow/cloud-water/cloud-ice proxy errors, but strict ADR-005 remains blocked because the manifest still uses carry-forward output tolerances and the gate reports the carry-forward regime (`fixtures/manifests/analytic-thompson-column-v1.yaml:160`, `fixtures/manifests/analytic-thompson-column-v1.yaml:256`, `artifacts/m5/thompson_gate_result.json:5`).

## HLO and performance status

The current profile proof reports `kernel_launches_per_step=5`, no post-init host-to-device transfer bytes, no post-init device-to-host transfer bytes, and zero temporary bytes per step (`artifacts/m5/thompson_profile.json:10`, `artifacts/m5/thompson_profile.json:17`, `artifacts/m5/thompson_profile.json:24`). The full HLO text is 343,407 bytes, while the committed HLO text is only the truncated audit artifact; this violates the sprint contract's strict <=200 KB full-HLO expectation (`.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:29`, `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:30`). Debug-vs-stripped HLO identity still holds with the zero-byte diff SHA recorded by the maintainability artifact (`artifacts/m5/maintainability.md:7`).

## Commands run

- `python scripts/extract_thompson_tables.py --output data/fixtures/thompson-tables-v1.npz` -> reproduced the table asset SHA and byte count pinned in the manifest (`fixtures/manifests/analytic-thompson-column-v1.yaml:268`, `fixtures/manifests/analytic-thompson-column-v1.yaml:270`).
- `python scripts/m5_generate_thompson_fixture.py` -> regenerated the WRF harness fixture manifest with harness SHA `043ff5b8bf41d635d02e2014b50fcf8553212b7372f5e2d2cd41545a48aa8c2c` and unchanged sample SHA `a357e2ef8f6e77ede0c6a79debc0dd0d8de1582585743fad3f6e533ba05d7102` (`fixtures/manifests/analytic-thompson-column-v1.yaml:3`, `fixtures/manifests/analytic-thompson-column-v1.yaml:260`, `fixtures/manifests/analytic-thompson-column-v1.yaml:265`).
- `python scripts/m5_run_thompson.py` -> regenerated Tier-1, Tier-2, HLO, profile, maintainability, and agent-success artifacts (`scripts/m5_run_thompson.py:157`, `artifacts/m5/tier1_thompson_parity.json:14`, `artifacts/m5/tier2_thompson_invariants.json:13`, `artifacts/m5/thompson_profile.json:16`).
- `python scripts/m5_gate_thompson.py` -> wrote `GO_CARRYFORWARD`, not strict GO (`artifacts/m5/thompson_gate_result.json:2`).
- `python scripts/validate_agentos.py` and `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml` -> passed; the manifest shape/checksum entries include the new table file (`fixtures/manifests/analytic-thompson-column-v1.yaml:259`).
- `python -m pytest tests/test_m5_thompson_constants.py tests/test_m5_thompson_tier1.py tests/test_m5_thompson_column_shapes.py -q` -> passed 11 focused tests; those tests cover table pinning and expected runtime table shapes (`tests/test_m5_thompson_constants.py:41`, `tests/test_m5_thompson_constants.py:49`).
- `python scripts/extract_canary_wrf_fixture.py` was run only to restore the local external Canary fixture required by the full suite; that fixture generator writes the external `full.npz` and checksum files under `data/fixtures` (`src/gpuwrf/fixtures/wrf_slice.py:165`, `src/gpuwrf/fixtures/wrf_slice.py:180`).
- `python -m pytest -q` -> passed after restoring that local external fixture: 400 passed, 1 skipped.

## Proof objects produced

- Table asset: `data/fixtures/thompson-tables-v1.npz`, SHA and bytes pinned in the Thompson manifest (`fixtures/manifests/analytic-thompson-column-v1.yaml:268`, `fixtures/manifests/analytic-thompson-column-v1.yaml:270`).
- Tier-1 proof: `artifacts/m5/tier1_thompson_parity.json` passes only because carry-forward tolerances remain in force (`artifacts/m5/tier1_thompson_parity.json:14`, `fixtures/manifests/analytic-thompson-column-v1.yaml:160`).
- Tier-2 proof: `artifacts/m5/tier2_thompson_invariants.json` passes positivity, NaN/Inf, water budget, and latent-heating checks (`artifacts/m5/tier2_thompson_invariants.json:9`, `artifacts/m5/tier2_thompson_invariants.json:18`).
- Performance proof: `artifacts/m5/thompson_profile.json` records 5 launches, zero post-init transfers, zero temp bytes, and null register/local-memory counters due to perfmon restrictions (`artifacts/m5/thompson_profile.json:16`, `artifacts/m5/thompson_profile.json:21`).
- Gate proof: `artifacts/m5/thompson_gate_result.json` records `GO_CARRYFORWARD` and says strict ADR-005 parity remains blocked after M5-S1.x table export (`artifacts/m5/thompson_gate_result.json:2`, `artifacts/m5/thompson_gate_result.json:5`).
- Blocker proof: `BLOCKER-m5-s1x-strict-tolerance.md` records strict residual misses, the HLO/launch regression, and required follow-up tracks (`.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:11`, `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:51`).

## Unresolved risks

- Strict ADR-005 parity is not closed; `qv`, `qr`, `qg`, `Ni`, `Nr`, and `T` still miss strict tolerances after the table export (`artifacts/m5/tier1_thompson_parity.json:16`, `artifacts/m5/tier1_thompson_parity.json:24`).
- The contract's HLO <=200 KB and one-launch target is not met; the current full HLO is 343,407 bytes and current launch count is 5 (`.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:29`, `artifacts/m5/thompson_profile.json:17`).
- Large rain-freezing lookup tables are exported and pinned but still not active in the hot path because direct and packed dynamic gathers produced 23 and 9 launches respectively (`.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:31`, `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:32`).
- Nsight register/local-memory counters are still unavailable under the workstation perfmon policy, so those profile fields remain null (`artifacts/m5/thompson_profile.json:18`, `artifacts/m5/thompson_profile.json:21`).

## Next decision needed

Manager should open a follow-up fix cycle rather than close this as strict GO. The two needed tracks are: table-gather/fusion design for `t_Efrw`, `iaus`, and `qrfz` to restore one launch and reduce full HLO size, and physics residual closure against WRF per-process tendencies for rain evaporation, graupel sublimation/melting, cloud-water freezing/nucleation, and number-balance finalization (`.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:53`, `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/BLOCKER-m5-s1x-strict-tolerance.md:56`).
