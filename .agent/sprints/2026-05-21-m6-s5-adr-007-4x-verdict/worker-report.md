# M6-S5 Worker Report — ADR-007 4x Verdict + Dycore Cap Lift

## Objective

Produce the binding ADR-007 full-domain 4x verdict after lifting the M6-S2 `dycore_dt_s = min(dt_s, 1.0)` cap.

## Outcome

**FAIL.** The measured end-to-end GPU wall clears the 4x speed threshold, but the lifted-cap forecast fails the required stability and Tier-2 invariant gates.

- Path chosen: **Path B**, coupled `dt_s=10s`; dycore receives the full coupled `dt_s`.
- Binding GPU wall: `500.78218523820397s`.
- Binding CPU denominator: `4859.53s`, M6-S2a raw timing subtraction from `artifacts/m6/cpu_denominator.json`.
- Speedup: `9.703879537345157x`.
- Blocking failure: lifted-cap 24h sanitize firing exceeds legacy capped baseline, final state saturates finite-guard bounds, and 1h lifted-cap Tier-2 fails.

## Files Changed

- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/io/proof_schemas.py`
- `scripts/m6_full_domain_batching.py`
- `tests/test_m6_dycore_cap_lift.py`
- `tests/test_m6_4x_verdict.py`
- `.agent/decisions/ADR-007-precision-policy.md`
- `artifacts/m6/performance/full_domain_batching_verdict.json`
- `artifacts/m6/performance/full_domain_batching_verdict.outputs.json`
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`
- `artifacts/m6/performance/profile/m6_s5_nsys_audit.log`
- `artifacts/m6/performance/profile/m6_s5_nsys_audit.nsys-rep`

## Commands Run

- `python -m py_compile src/gpuwrf/coupling/driver.py src/gpuwrf/io/proof_schemas.py scripts/m6_full_domain_batching.py tests/test_m6_dycore_cap_lift.py tests/test_m6_4x_verdict.py`
- `pytest -q tests/test_m6_dycore_cap_lift.py tests/test_m6_4x_verdict.py`
- `python scripts/m6_full_domain_batching.py --hours 0.002777777777777778 --tier2-hours 0.002777777777777778 --output artifacts/m6/performance/full_domain_batching_smoke.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/full_domain_batching_smoke --skip-nsys --skip-legacy-baseline-sanitize-audit`
- `python scripts/m6_full_domain_batching.py --hours 24 --output artifacts/m6/performance/full_domain_batching_verdict.json`
- `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --output artifacts/m6/performance/profile/m6_s5_nsys_audit /home/enric/miniconda3/bin/python scripts/m6_full_domain_batching.py --profile-child --profile-steps 1 --dt-s 10.0 --n-acoustic 2 --radiation-cadence-steps 60 --run-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z --boundary data/fixtures/m6/d02_boundary_replay_v2.zarr --spec-bdy-width 5 --spec-zone 1 --relax-zone 4 --spec-exp 0.0`
- `cat artifacts/m6/performance/full_domain_batching_verdict.json | jq '.pass, .speedup_ratio, .dycore_cap_status'`
- `pytest -q tests/test_m6_dycore_cap_lift.py tests/test_m6_4x_verdict.py`

## Proof Objects Produced

- `artifacts/m6/performance/full_domain_batching_verdict.json`
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`
- `/home/enric/.cache/gpuwrf_tmp/trace_m6_s5_full_domain_batching/plugins/profile/2026_05_21_17_31_21/enric-battlestation.xplane.pb`
- `/home/enric/.cache/gpuwrf_tmp/trace_m6_s5_full_domain_batching/plugins/profile/2026_05_21_17_31_21/enric-battlestation.trace.json.gz`
- `artifacts/m6/performance/profile/m6_s5_nsys_audit.nsys-rep`

## Unresolved Risks

- The reduced M4 dycore is not stable enough under full 10s coupled integration for a physical 24h forecast. It requires either real dycore stabilization or a different architecture decision; the old 1s cap cannot be used for ADR-007 speed claims.
- The JAX transfer-audit parser reports `167904` host/device bytes on the warmed audit trace. The event is preserved in the proof object; it was not used to hide or improve the speed verdict.
- Tier-2 failure is severe: 1h lifted-cap audit reports `nan_inf=1014298726` and total-water residual `4.2711782132553853e-4`.

## Next Decision Needed

Project-scope decision: either invest in M4 dycore stabilization under realistic dt, redesign the coupled integration path, or accept that ADR-007 mixed-precision/full-domain batching clears throughput but does not yet produce a valid lifted-cap forecast.
