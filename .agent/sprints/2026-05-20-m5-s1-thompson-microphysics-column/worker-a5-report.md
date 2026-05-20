# Worker Report - M5-S1 Thompson Microphysics Column Attempt 5

## Objective

Apply the attempt-5 narrow fix: correct the ice `lami` clipping numerator, clean stale fixture-manifest sedimentation text, regenerate M5 artifacts, decide tolerance posture, and produce proof objects without touching lookup-table or sedimentation code paths.

## Files Changed

- `src/gpuwrf/physics/thompson_constants.py`
- `src/gpuwrf/physics/thompson_column.py`
- `scripts/m5_generate_thompson_fixture.py`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/hlo_dump/thompson_column_production.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_stripped.txt`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/M5-S1-NEEDS-S1X.md`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-a5-report.md`

## Fix Applied

Before A5, `_finish` clipped ice `lami` with `6.0 / 5.0e-6` and `6.0 / 300.0e-6` at `src/gpuwrf/physics/thompson_column.py:277-278` in the attempt-4 baseline. That used the gamma constant (`cig(2)=6`) where WRF uses `cie(2)`.

After A5, `src/gpuwrf/physics/thompson_constants.py:44` defines `CIE2 = BM_I + MU_I + 1.0`, `src/gpuwrf/physics/thompson_column.py:22` imports it, and `src/gpuwrf/physics/thompson_column.py:278-279` uses `CIE2 / 5.0e-6` and `CIE2 / 300.0e-6`.

WRF source truth: `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre:688` defines `cie(2) = bm_i + mu_i + 1.`, `:695` defines `cig(2) = WGAMMA(cie(2))`, and `:1928/:1931` use `cie(2)` in the clamp.

## Fixture Manifest Cleanup

The stale `1e30 m layer depths` narrative is gone from the generated manifest. The manifest now says sedimentation is bypassed by locally patched WRF terminal velocities at `fixtures/manifests/analytic-thompson-column-v1.yaml:6-8`, and the generator emits the same text at `scripts/m5_generate_thompson_fixture.py:253`.

## Parity Before/After

Attempt-4 before values came from the committed A4 report (`.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-report.md:64-65`). Attempt-5 after values are the regenerated artifact (`artifacts/m5/tier1_thompson_parity.json:15-35`).

| Field | A4 max abs | A5 max abs | A4 max rel | A5 max rel |
|---|---:|---:|---:|---:|
| `qv` | `1.4304079020558032e-05` | `1.4304079020558032e-05` | `0.004086360113325722` | `0.004086360113325722` |
| `qc` | `0.0001517228938283358` | `0.0001517228938283358` | `0.9999988999202443` | `0.9999988999202443` |
| `qr` | `4.760876436193939e-06` | `4.760876436193939e-06` | `45035996.27370496` | `45035996.27370496` |
| `qi` | `0.00013708094759935302` | `0.00013708094759935302` | `0.3049187629215758` | `0.3049187629215758` |
| `qs` | `0.0001447943623500526` | `0.0001447943623500526` | `1249.636404283526` | `1249.636404283526` |
| `qg` | `1.521843532880611e-05` | `1.521843532880611e-05` | `983428014.0158471` | `983428014.0158471` |
| `Ni` | `126975.12500000041` | `126975.12500000041` | `0.3883326751538069` | `0.3883326751538069` |
| `Nr` | `67300.453125` | `67300.453125` | `45425309332488.81` | `45425309332488.81` |
| `T` | `0.040290844661740266` | `0.040290844661740266` | `0.00015064605175968627` | `0.00015064605175968627` |

No field regressed; all deltas were zero. The coefficient fix is still required for source truth, but this fixture did not move because the current post-process positive-`qi` cells did not exercise the final ice-diameter clamp.

## Tolerance Posture

Strict ADR-005 parity did not close. ADR-005 sets `abs=1e-10, rel=1e-8` for hydrometeors and `abs=1e-3, rel=1e-6` for `Ni/Nr` at `.agent/decisions/ADR-005-first-physics-suite.md:27`, while current strict residuals remain at `artifacts/m5/tier1_thompson_parity.json:15-35`. I left the attempt carry-forward tolerances in the manifest at `fixtures/manifests/analytic-thompson-column-v1.yaml:157-254` and the generator at `scripts/m5_generate_thompson_fixture.py:217-220`, then wrote `M5-S1-NEEDS-S1X.md` for the remaining table/proxy residual fields.

Under those carry-forward tolerances, Tier-1 now reports `pass=true` and `tolerances_met=true` at `artifacts/m5/tier1_thompson_parity.json:14` and `:38`. This is not a strict ADR-005 closeout; it is an M5-S1.x handoff posture.

## Tier-2 / HLO / Profile

Tier-2 did not regress: `artifacts/m5/tier2_thompson_invariants.json:13` is `pass=true`, water residual is `2.670445271854754e-12` at `:18-21`, positivity violations are `0` at `:14-16`, and NaN/Inf violations are `0` at `:9-11`.

The HLO debug-vs-stripped diff remains 0 bytes, and its SHA-256 remains `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The profile did not regress on the hard metrics: `artifacts/m5/thompson_profile.json:17` reports `kernel_launches_per_step=1`, `:14` reports `host_to_device_bytes_post_init=0`, and `:24` reports `temporary_bytes_per_step=0`.

The gate artifact is `GO` under carry-forward tolerances: `artifacts/m5/thompson_gate_result.json:2-8` records `gate_status="GO"`, tier1/tier2 pass, and one launch.

## Commands Run + Output

`python scripts/m5_generate_thompson_fixture.py`
stdout included WRF table-generation messages and final JSON: `bytes=7026`, `path=fortran-harness`, sample SHA `a357e2ef8f6e77ede0c6a79debc0dd0d8de1582585743fad3f6e533ba05d7102`, harness SHA `7e24b30e50e88c251d7917ff84266fc636658991a5e6988b4d881406c92357da`, `wrf_source_exists=true`.

`python scripts/m5_run_thompson.py`
stdout: HLO diff SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; Tier-1 `pass=true`; Tier-2 `pass=true`; profile `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0`.

`python scripts/m5_gate_thompson.py`
stdout: `{"gate_status": "GO", "kernel_launches_per_step": 1, "local_memory_bytes_per_kernel": null, "rationale": "tier-1/tier-2 pass and HLO-derived launches are within the GO threshold; register/local-memory counters are null due to perfmon restriction", "registers_per_kernel": null, "tier1_pass": true, "tier2_pass": true}`.

`python -m json.tool artifacts/m5/tier1_thompson_parity.json`
stdout: valid JSON; `pass=true`; `tolerances_met=true`; `Ni_abs=126975.12500000041`; `qi_abs=0.00013708094759935302`; `qs_abs=0.0001447943623500526`.

`python -m json.tool artifacts/m5/tier2_thompson_invariants.json`
stdout: valid JSON; `pass=true`; `water_residual=2.670445271854754e-12`; positivity violations `0`; NaN/Inf violations `0`.

`python -m json.tool artifacts/m5/thompson_gate_result.json`
stdout: valid JSON; `gate_status=GO`; tier1 `true`; tier2 `true`; launches `1`.

`python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml`
stdout: `fixtures/manifests/analytic-thompson-column-v1.yaml: ok`.

`rg -n "1e30|1\\.0e30|dz=1e30|layer depths" fixtures/manifests/analytic-thompson-column-v1.yaml scripts/m5_generate_thompson_fixture.py || true`
stdout: empty.

`wc -c artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stdout: `0 artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`.

`sha256sum artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

`python scripts/validate_agentos.py`
stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.

`pytest -q`
stdout: `398 passed in 249.51s (0:04:09)`.

## Proof Objects Produced

- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/M5-S1-NEEDS-S1X.md`

## Handoff

Objective: attempt-5 lami coefficient fix, artifact regeneration, and tolerance-posture handoff.

Files changed: listed above.

Commands run: listed above; final `validate_agentos.py` and `pytest -q` both passed.

Proof objects produced: listed above.

Unresolved risks: strict ADR-005 parity still fails for `qc/qi/qs` and other coupled fields; exact WRF lookup-table/moment parity remains M5-S1.x work. Register/local-memory counters remain null due workstation perfmon restrictions.

Next decision needed: dispatch M5-S1.x for WRF Thompson table/moment export before treating this Thompson implementation as strict-parity complete.
