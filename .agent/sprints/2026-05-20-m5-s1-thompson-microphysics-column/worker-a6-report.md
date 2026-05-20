# Worker A6 Report - M5-S1 Final Required Fix Cycle

## Objective

Close Reviewer A5 findings R-1 through R-4 without expanding M5-S1 scope.

## Fixes Applied

- R-1a: `CGG11` is no longer a drift-prone literal. `src/gpuwrf/physics/thompson_constants.py:71-74` now derives `CGE11` from `BV_G_MP8`/`MU_G_MP8` and computes `CGG11 = math.gamma(CGE11)`, matching WRF `module_mp_thompson.F.pre:763,767`. The graupel prefactors still consume `CGG11` at `src/gpuwrf/physics/thompson_constants.py:93-96`.
- R-1b: one regression test now checks the derived-constant relationships for `CIE2`, `CGE11`, and `CGG11` at `tests/test_m5_thompson_constants.py:31-34`.
- R-2: I chose the script-level semantic fix. `scripts/m5_gate_thompson.py:23-32` detects the manifest tolerance regime, `scripts/m5_gate_thompson.py:73-89` emits `gate_status="GO_CARRYFORWARD"` plus `tolerance_regime="carry-forward"` when Tier-1 passes under carry-forward tolerances, and `scripts/m5_gate_thompson.py:101` treats that status as non-failing. The regenerated gate artifact records this at `artifacts/m5/thompson_gate_result.json:2-9`.
- R-3: `M5-S1-NEEDS-S1X.md:23` now states that `qr`, `qs`, `qg`, `Nr`, and intermittent `qc` max-relative errors are dominated by near-zero WRF references, so max-absolute error is the load-bearing S1.x metric; the same note cross-links the validation-philosophy memory.
- R-4: `artifacts/m5/maintainability.md:3-7` and its generator template at `scripts/m5_run_thompson.py:135-142` no longer describe the stale attempt-4 blocker posture and now include the final `CGG11 = math.gamma(CGE11)` source-truth correction. The generated `agent_success.json` note was also refreshed at `artifacts/m5/agent_success.json:2-7` from the template at `scripts/m5_run_thompson.py:145-153`.

## Parity Movement vs `2798b05`

`2798b05` values are the A5 post-graupel values recorded in `worker-a5-supplement.md:39-47`; A6 values are regenerated in `artifacts/m5/tier1_thompson_parity.json:15-35`.

| Field | `2798b05` max abs | A6 max abs | Abs delta | `2798b05` max rel | A6 max rel |
|---|---:|---:|---:|---:|---:|
| `qv` | `1.5090182928232613e-05` | `1.5091010112033235e-05` | `+8.271838006224108e-10` | `0.00431093267396623` | `0.004311168982146883` |
| `qc` | `0.0001519669795683178` | `0.0001519672247928526` | `+2.4522453479728507e-10` | `0.9999988999202443` | `0.9999988999202443` |
| `qr` | `4.760876436193939e-06` | `4.760876436193939e-06` | `0.0` | `45035996.27370496` | `45035996.27370496` |
| `qi` | `0.00013713997976800624` | `0.00013714003800346464` | `+5.8235458393497e-11` | `0.30505007231324216` | `0.30505020185045534` |
| `qs` | `0.0001447943623500526` | `0.0001447943623500526` | `0.0` | `1249.636404283526` | `1249.636404283526` |
| `qg` | `1.6510481869645087e-05` | `1.651181415760252e-05` | `+1.3322879574329026e-09` | `986567533.2761194` | `986570506.4244672` |
| `Ni` | `126975.12500000047` | `126975.12500000041` | `-5.820766091346741e-11` | `0.3883326751538071` | `0.3883326751538069` |
| `Nr` | `67300.453125` | `67300.453125` | `0.0` | `45425309332488.81` | `45425309332488.81` |
| `T` | `0.042506264947576256` | `0.04250859614040792` | `+2.3311928316616104e-06` | `0.00015892943032500158` | `0.00015893814657304574` |

Side finding: the reviewer expected a tiny residual improvement, but the source-truth-correct `CGG11` moved `qv`, `qg`, and `T` slightly in the opposite direction. The movement is tiny relative to carry-forward tolerances, and the blocker-class metrics stayed unchanged: Tier-2 pass and water residual are at `artifacts/m5/tier2_thompson_invariants.json:13-21`; launch count, H2D bytes, and temp bytes are at `artifacts/m5/thompson_profile.json:13-24`; the HLO diff SHA remains empty at `artifacts/m5/maintainability.md:7`.

## Commands Run

- `python scripts/m5_generate_thompson_fixture.py` - ok; sample SHA stayed `a357e2ef8f6e77ede0c6a79debc0dd0d8de1582585743fad3f6e533ba05d7102` at `fixtures/manifests/analytic-thompson-column-v1.yaml:259-261`; rebuilt external harness SHA is `1dbb0ce7675967ca2ef15d139af8f9304384ca7fb904eac0ebaab2035bf3762f` at `fixtures/manifests/analytic-thompson-column-v1.yaml:263-265`.
- `python scripts/m5_run_thompson.py` - ok; Tier-1 `pass=true` at `artifacts/m5/tier1_thompson_parity.json:13-14`; Tier-2 `pass=true` at `artifacts/m5/tier2_thompson_invariants.json:13`; HLO diff SHA recorded at `artifacts/m5/maintainability.md:7`; profile hard metrics at `artifacts/m5/thompson_profile.json:13-24`.
- `python scripts/m5_gate_thompson.py` - ok; generated `GO_CARRYFORWARD` at `artifacts/m5/thompson_gate_result.json:2-9`.
- First `pytest -q` run exposed the generated note missing the existing harness-test phrase; fixed by adding "Fortran harness" to the generated note at `scripts/m5_run_thompson.py:152` and `artifacts/m5/agent_success.json:3`.
- Second `pytest -q` - `399 passed in 295.67s`; the new derived-constant regression is `tests/test_m5_thompson_constants.py:31-34`.
- `python scripts/validate_agentos.py` - ok (`errors=[]`, `ok=true`).
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml` - ok.
- `rg -n "1e30|dz=1e30|layer depths|pre-attempt-4|BLOCKER-m5-s1-attempt4|Attempt 4 still fails" artifacts/m5 scripts/m5_run_thompson.py scripts/m5_generate_thompson_fixture.py fixtures/manifests/analytic-thompson-column-v1.yaml || true` - no matches.

## Proof Objects Produced

- Corrected constants and regression: `src/gpuwrf/physics/thompson_constants.py:71-96`, `tests/test_m5_thompson_constants.py:31-34`.
- Regenerated validation/profile/gate artifacts: `artifacts/m5/tier1_thompson_parity.json:1-39`, `artifacts/m5/tier2_thompson_invariants.json:1-23`, `artifacts/m5/thompson_profile.json:1-26`, `artifacts/m5/thompson_gate_result.json:1-10`.
- Refreshed side artifacts and generator template: `artifacts/m5/maintainability.md:1-7`, `artifacts/m5/agent_success.json:1-8`, `scripts/m5_run_thompson.py:132-153`.
- Updated S1.x caveat: `M5-S1-NEEDS-S1X.md:13-27`.

## Unresolved Risks

- Strict ADR-005 Tier-1 parity still remains M5-S1.x work; current strict residuals are enumerated at `M5-S1-NEEDS-S1X.md:13-21`.
- The fixture harness binary SHA changed after rebuild while the fixture sample SHA stayed stable; the manifest records both at `fixtures/manifests/analytic-thompson-column-v1.yaml:259-265`.

## Next Decision Needed

Merge this worker branch to main and open M5-S1.x for the table/moment parity work listed in `M5-S1-NEEDS-S1X.md:25-27`.
