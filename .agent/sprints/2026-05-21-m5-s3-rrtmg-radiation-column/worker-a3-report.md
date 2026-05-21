# Worker A3 Report - M5-S3 RRTMG Bounded Rework

objective

Resolve the A2 reviewer rejection items without undoing A2's real WRF driver binding. Attempt 3 removed the clip-dominated table reductions, restored non-vacuous Tier-1 tolerances, replaced the JAX-side tautological Tier-2 records, and amended ADR-009. The strict correctness result is still not acceptable: Tier-1 fails against the real WRF RRTMG driver after tolerances are tightened.

files changed

- `scripts/extract_rrtmg_tables.py`: typed SW and LW unformatted record parsing, WRF reduced-g-point maps/weights, reference-pressure KAO/KBO extraction, WRF source cloud optics, and updated table metadata (`scripts/extract_rrtmg_tables.py:35`, `scripts/extract_rrtmg_tables.py:56`, `scripts/extract_rrtmg_tables.py:72`, `scripts/extract_rrtmg_tables.py:161`, `scripts/extract_rrtmg_tables.py:177`, `scripts/extract_rrtmg_tables.py:319`, `scripts/extract_rrtmg_tables.py:357`, `scripts/extract_rrtmg_tables.py:403`, `scripts/extract_rrtmg_tables.py:433`, `scripts/extract_rrtmg_tables.py:446`, `scripts/extract_rrtmg_tables.py:485`).
- `src/gpuwrf/physics/rrtmg_tables.py`, `src/gpuwrf/physics/rrtmg_sw.py`, `src/gpuwrf/physics/rrtmg_lw.py`: table bundle expanded to carry reduced-g-point masks/weights and pressure-resolved coefficients; SW/LW kernels select nearest WRF reference-pressure coefficients and use WRF harness pressure-layer mass reconstruction (`src/gpuwrf/physics/rrtmg_lw.py:128`, `src/gpuwrf/physics/rrtmg_lw.py:141`, `src/gpuwrf/physics/rrtmg_lw.py:160`).
- `scripts/m5_generate_rrtmg_fixture.py` plus `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml` and `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml`: output tolerances tightened to `abs=1.0 W m-2, rel=0.05` for flux/scalar flux-like outputs and `abs=1.0e-4 K s-1, rel=0.05` for heating rates (`fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:162`, `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:173`, `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml:161`, `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml:172`).
- `src/gpuwrf/validation/tier2_rrtmg.py`: replaced `shortwave_candidate_energy_conservation` and old `longwave_surface_emission` key with candidate heating/flux-divergence checks and a named Stefan-Boltzmann surface-emission check (`src/gpuwrf/validation/tier2_rrtmg.py:34`, `src/gpuwrf/validation/tier2_rrtmg.py:38`, `src/gpuwrf/validation/tier2_rrtmg.py:71`, `src/gpuwrf/validation/tier2_rrtmg.py:91`, `src/gpuwrf/validation/tier2_rrtmg.py:96`).
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`: amended to describe reduced-g-point consumption, strict tolerance failure, and the corrected LW citation.
- Regenerated `data/fixtures/rrtmg-tables-v1.npz`, `data/fixtures/rrtmg-tables-v1.json`, and `artifacts/m5/*rrtmg*`.

Fix 1 - R-2 spectral coefficients

The A2 table floor behavior was replaced. SW now parses each `RRTMG_SW_DATA` band record according to the WRF read-list and consumes KAO/KBO reference-pressure absorption profiles, Rayleigh terms, and source flux weights. LW now also parses each `RRTMG_LW_DATA` record into pressure-resolved KAO/KBO profiles rather than selecting generic positive payload words. Both paths apply WRF's reduced-g-point mapping and original RRTM g weights.

WRF citations: SW g-point reduction is `module_ra_rrtmg_sw.F:4763-4784`, SW groups/weights are `module_ra_rrtmg_sw.F:4927-5027`, SW data records and KAO/KBO comments are `module_ra_rrtmg_sw.F:11705-11710` and `module_ra_rrtmg_sw.F:11734-11760`. LW g-point reduction is `module_ra_rrtmg_lw.F:8073-8104`, LW groups/weights and `delwave` are `module_ra_rrtmg_lw.F:8144-8152` and `module_ra_rrtmg_lw.F:8244-8315`, and LW records are `module_ra_rrtmg_lw.F:13085-13090`.

Before: A2 reviewer reported 74/86 compact spectral values pinned to clip floors. After: active-value old-floor fraction is `0.0` over 14,670 active values for floors `0.0025`, `1e-5`, `0.25`, `0.16`, `0.003`, and `0.2`. Regenerated table asset: `data/fixtures/rrtmg-tables-v1.npz`, size `1,747,092` bytes, SHA-256 `cffd87d494e3f8c2da6bedac42d6626a993bdcd777dcd0bad53dee5e4f7f96c8`.

Fix 2 - R-3a strict tolerances

The SW manifest uses `output_heating_rate abs=0.0001, rel=0.05`; SW flux and scalar flux-like outputs use `abs=1.0, rel=0.05`. The LW manifest uses the same heating/flux policy. No `abs=1200`, `abs=500`, `rel=15.0`, or carry-forward output rationale remains in the RRTMG manifests.

This exposed the blocker: Tier-1 does not pass. Latest artifacts report SW max absolute errors `flux_down=863.4149601378938`, `flux_up=1578.875792806668`, `heating_rate=0.0006909078736834584`; LW max absolute errors `flux_down=228.98306589556717`, `flux_up=176.3891440048552`, `heating_rate=0.00015330015754548213`. This is above the restored tolerances and should be treated as correctness failure, not a tolerance debate.

Fix 3 - R-3b Tier-2 invariants

The JAX-side candidate check now compares each candidate heating-rate layer integral against candidate flux divergence using WRF fixture pressure-layer mass (`src/gpuwrf/validation/tier2_rrtmg.py:34-41`). This is not the old by-construction scalar-energy record. LW also retains an explicit Stefan-Boltzmann surface-emission check (`src/gpuwrf/validation/tier2_rrtmg.py:42-43`, `src/gpuwrf/validation/tier2_rrtmg.py:96-100`). Tier-2 passes in `artifacts/m5/tier2_rrtmg_invariants.json`.

Fix 4 - ADR-009 labeling and LW citation

ADR-009 now describes the attempt-3 implementation as real WRF reduced-g-point table consumption, not compact clipped effective reductions. It explicitly states that `module_ra_rrtmg_lw.F:12823-12829` is only the K/day to K/s plus Exner tendency conversion, while pressure-layer/heating context is from `module_ra_rrtmg_lw.F:8212-8233` and local harness pressure interfaces `scripts/wrf_rrtmg_harness.f90:294-321`.

commands run

- `python scripts/extract_rrtmg_tables.py --output data/fixtures/rrtmg-tables-v1.npz` - pass.
- `python scripts/m5_generate_rrtmg_fixture.py` - pass.
- `python scripts/m5_run_rrtmg.py` - fail as expected under strict Tier-1; wrote updated Tier-1/Tier-2/profile/HLO artifacts.
- `python scripts/m5_gate_rrtmg.py` - fail with `gate_status=FALLBACK`, `tolerance_regime=strict`, `tier1_sw_pass=false`, `tier1_lw_pass=false`, `tier2_pass=true`, `raw_hlo_launch_marker_count=28`, `kernel_launches_per_step=28`.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml` - pass.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml` - pass.
- `python scripts/validate_agentos.py` - pass.
- `pytest -q tests/test_m5_rrtmg_tables.py tests/test_m5_rrtmg_tier2.py` - pass, 2 passed.
- `pytest -q` - fail: 415 passed, 1 skipped, 4 failed. Three failures are the expected RRTMG strict Tier-1/gate assertions. One unrelated failure is `tests/test_m5_mynn_harness.py::test_mynn_fixture_generation_records_harness_binary`, caused by local `data/scratch/wrf_mynn_harness` checksum mismatch against the checked-in MYNN manifest.

proof objects produced

- `data/fixtures/rrtmg-tables-v1.npz` and `.json`.
- `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml` and `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml`.
- `artifacts/m5/tier1_rrtmg_sw_parity.json`, `artifacts/m5/tier1_rrtmg_lw_parity.json`, `artifacts/m5/tier2_rrtmg_invariants.json`.
- `artifacts/m5/rrtmg_profile.json`, `artifacts/m5/rrtmg_gate_result.json`, `artifacts/m5/hlo_dump/rrtmg_*`.
- `artifacts/m5/extract_rrtmg_tables_m5_s3.json`, `generate_rrtmg_fixture_m5_s3.json`, `run_rrtmg_m5_s3.json`, `gate_rrtmg_m5_s3.txt`, and manifest validation logs.

unresolved risks

- Blocker: strict Tier-1 parity fails. The reduced-g-point table extraction is real, but the compact SW/LW transfer kernels still lack full WRF/AER spectral interpolation and transfer physics.
- R-4 reporting rule is preserved in mechanism (`kernel_launches_per_step == raw_hlo_launch_marker_count`) but the value is now `28`, not A2's `22`, because the real pressure-resolved reduced-g-point arrays increased HLO work.
- The latest full `pytest -q` includes an unrelated MYNN scratch checksum failure from local generated state; I did not modify MYNN tracked files.

next decision needed

Do not merge as a completed M5-S3 RRTMG parity implementation. Manager should either dispatch an M5-S3.x full RRTMG interpolation/transfer sprint or explicitly decide that this compact-column JAX implementation is only a table-provenance foundation and remains blocked for physics parity.
