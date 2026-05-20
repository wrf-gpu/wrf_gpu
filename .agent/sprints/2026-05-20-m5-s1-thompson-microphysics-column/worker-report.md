# Worker Report - M5-S1 Thompson Microphysics Column Attempt 2

Summary: Replaced the attempt-1 compact Thompson source terms with WRF-source-mapped Path-B-strict formulas and regenerated the fixture/proof artifacts. The fixture oracle now uses an independent NumPy WRF-style tendency ledger (`qvten`, `qcten`, `tten`) instead of the JAX helper sequence. M5-owned tier-1, tier-2, HLO, profile, gate, and pytest checks pass. M3/M4 done checks still report pre-existing lifecycle-file gaps outside this worker's allowed scope.

## Objective

Fix attempt-1 reviewer blockers for the ADR-005 Thompson column sprint: remove self-consistency oracle behavior, replace compact relaxations with WRF-faithful formulas, rerun the M5 gate on the new kernel, and update ADR-006 and proof artifacts.

## Files Changed

- `src/gpuwrf/physics/thompson_constants.py`
- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/physics/thompson_column_debug_stripped.py` indirectly recompiled into HLO artifacts
- `src/gpuwrf/validation/tier1_thompson.py` unchanged logic, regenerated artifact
- `src/gpuwrf/validation/tier2_thompson.py` unchanged logic, regenerated artifact
- `scripts/m5_generate_thompson_fixture.py`
- `scripts/m5_run_thompson.py`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz`
- `artifacts/m5/*`
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`
- `tests/test_m5_thompson_constants.py`
- `tests/test_m5_thompson_tier1.py`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-report.md`

## Commands Run + Output

Path-A investigation:

`find ../wrf_gpu /mnt/data/wrf_gpu2 -name 'module_mp_thompson*' -o -name '*.o' -o -name '*.mod'`

Output showed the Thompson source snapshot but no reusable `module_mp_thompson` object/module. Existing WRF binaries were found under `../wrf_gpu/builds/.../wrf.exe`. Path A wrapper compile was not pursued because the wrapper would need WRF module dependencies (`module_wrf_error`, `module_mp_radar`, model constants, preprocessor flags, lookup-table init) outside this worker scope.

Validation:

`python scripts/validate_agentos.py`
stdout:
```json
{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}
```
stderr: empty

`python scripts/check_m1_done.py`
stdout:
```json
{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}
```
stderr: empty

`python scripts/check_m2_done.py`
stdout:
```json
{"candidates_satisfied": 6, "candidates_total": 6, "errors": [], "ok": true, "sprints_closed": 7}
```
stderr: empty

`python scripts/check_m3_done.py`
stdout:
```json
{
  "errors": [
    "sprint 2026-05-19-m3-state-grid-halo-skeleton not closed: {\n  \"errors\": [\n    \"missing reviewer-report.md\"\n  ],\n  \"ok\": false\n}"
  ],
  "ok": false,
  "sprints_closed": 0
}
```
stderr: empty

`python scripts/check_m4_done.py`
stdout:
```json
{
  "errors": [
    "check_m3_done.py regressed: ['sprint 2026-05-19-m3-state-grid-halo-skeleton not closed: {\\n  \"errors\": [\\n    \"missing reviewer-report.md\"\\n  ],\\n  \"ok\": false\\n}']",
    "sprint 2026-05-19-m4-dycore-rk3-advection-acoustic not closed: {\n  \"errors\": [\n    \"missing manager-closeout.md\",\n    \"missing memory-patch.md\"\n  ],\n  \"ok\": false\n}"
  ],
  "ok": false,
  "sprints_closed": 0
}
```
stderr: empty

`python scripts/m5_generate_thompson_fixture.py`
stdout:
```json
{
  "bytes": 7789,
  "manifest": "fixtures/manifests/analytic-thompson-column-v1.yaml",
  "path": "B-strict",
  "path_a_investigation": "no reusable module_mp_thompson object/module found under ../wrf_gpu or /mnt/data/wrf_gpu2; direct wrapper compile would require WRF module dependencies outside this worker scope",
  "sample": "fixtures/samples/analytic-thompson-column-v1.npz",
  "sha256": "3bdd721ac380620dfcf12fbef2693833205885674587ca5099390f31ef001ae1",
  "wrf_source_exists": true
}
```
stderr: empty

`python scripts/m5_run_thompson.py`
stdout summary:
```json
{
  "hlo_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "tier1": {"pass": true, "scenarios_tested": 3, "tolerances_met": true},
  "tier2": {"pass": true, "iterations": 10, "dt_s": 60.0},
  "profile": {
    "kernel_launches_per_step": 1,
    "temporary_bytes_per_step": 0,
    "host_to_device_bytes_post_init": 0,
    "device_to_host_bytes_post_init": 0,
    "registers_per_kernel": null,
    "local_memory_bytes_per_kernel": null,
    "wall_time_s": 0.00022413604892790318
  }
}
```
stderr: empty

`python scripts/m5_gate_thompson.py`
stdout:
```json
{
  "gate_status": "GO",
  "kernel_launches_per_step": 1,
  "local_memory_bytes_per_kernel": null,
  "rationale": "tier-1/tier-2 pass and HLO-derived launches are within the GO threshold; register/local-memory counters are null due to perfmon restriction",
  "registers_per_kernel": null,
  "tier1_pass": true,
  "tier2_pass": true
}
```
stderr: empty

`python -m json.tool artifacts/m5/tier1_thompson_parity.json`
stdout: valid JSON; `pass=true`; all output fields pass. Max abs errors include `Ni=6.984919309616089e-10`, `Nr=4.802132025361061e-10`, `T=5.684341886080802e-14`, and hydrometeor fields at or below `1.36e-19`.
stderr: empty

`python -m json.tool artifacts/m5/tier2_thompson_invariants.json`
stdout: valid JSON; `pass=true`; positivity violations `0`; NaN/Inf violations `0`; water residual `4.703652943237793e-13`; max latent heating delta `0.8504670669483403 K`.
stderr: empty

`python -m json.tool artifacts/m5/thompson_gate_result.json`
stdout: valid JSON; `gate_status="GO"`; launches `1`; register/local-memory counters `null`; tier-1/tier-2 pass.
stderr: empty

`ls -l artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stdout:
```text
-rw-rw-r-- 1 enric enric 0 May 20 07:57 artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff
```
stderr: empty

`pytest -q`
stdout:
```text
395 passed in 279.54s (0:04:39)
```
stderr: empty

Additional focused check:

`python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml`
stdout: `fixtures/manifests/analytic-thompson-column-v1.yaml: ok`
stderr: empty

## Proof Objects Produced

- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz` (7,789 bytes, sha256 `3bdd721ac380620dfcf12fbef2693833205885674587ca5099390f31ef001ae1`)
- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/hlo_dump/thompson_column_production.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_stripped.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` (0 bytes)
- `artifacts/m5/maintainability.md`
- `artifacts/m5/agent_success.json`
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`

## WRF Source Mapping

Included formulas cite `module_mp_thompson.F.pre`: thermodynamic scalars lines 2040-2064; Berry-Reinhardt autoconversion lines 2242-2258; rain-cloud collection shape lines 2260-2268; Srivastava-Coen rain evaporation lines 3561-3636; cloud-ice/snow/graupel deposition lines 2709-2770; rain freezing and snow/graupel melting lines 2658-2669 and 2845-2889; mass/number constraints lines 4033-4142. Generated WRF lookup tables remain approximated/proxied and are explicitly documented.

## Spacetime Budget

| Item | Value | Justification |
|---|---:|---|
| state bytes | 3,168 | 11 fp64 leaves x 3 scenarios x 12 levels x 8 bytes |
| tendency bytes | 0 | Thompson step applies fused source/sink updates directly to the state pytree |
| temporary bytes per step | 0 | no `jnp.array`, `jnp.zeros`, or `jnp.empty` in traced Thompson body; profile records 0 |
| host/device transfer bytes post-init | 0 | profile records 0; scalar `dt` and `debug` are static |
| kernel launches per step | 1 | HLO-derived launch count from production HLO |
| wall time per step | 224.1 us | median cached JAX call from `thompson_profile.json` |

## Allocation Audit

- `src/gpuwrf/physics/thompson_column.py`: no `jnp.array`, `jnp.zeros`, or `jnp.empty`. Hot-path `state.replace(...)` constructs pytree containers only; array values are fused expressions over existing leaves.
- `src/gpuwrf/physics/thompson_saturation.py`: no array constructors; pure elementwise formulas.
- `scripts/m5_generate_thompson_fixture.py`: NumPy allocations are fixture-generation only, outside the model hot path.
- `src/gpuwrf/validation/tier1_thompson.py`: `jnp.asarray` calls are validation/init-only when loading fixture arrays.
- Tests use `jnp.asarray`/`jnp.ones` only for setup.

HLO debug-vs-stripped diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Risks

- Path A direct WRF wrapper remains unsolved; this attempt uses Path B-strict source transcription. Exact WRF table parity requires either table export or a dedicated wrapper sprint.
- WRF-generated lookup tables (`t_Efrw`, rain/cloud freezing tables, some snow/graupel moment tables) are not carried as fixture inputs; this implementation uses documented proxies at those call sites.
- Register/local-memory counters remain `null` due to the known workstation perfmon restriction. Gate status is GO on correctness and HLO launch count.
- M3/M4 done oracles fail on lifecycle artifacts outside this worker scope: M3 missing `reviewer-report.md`; M4 missing `manager-closeout.md` and `memory-patch.md`.

## Handoff

Objective: M5-S1 attempt-2 worker fix for Thompson column source/sink pipeline.

Files changed: listed above; all intended changes are within the sprint contract's worker ownership or AC 6 HLO sibling/proof scope. The milestone checks rewrote M2 profile JSONs during validation; those out-of-scope generated changes were restored before this report.

Commands run: all sprint validation commands were run. M5-owned commands pass; M3/M4 prior-oracle lifecycle failures are recorded above.

Proof objects produced: all required M5 proof objects listed above.

Unresolved risks: exact WRF-wrapper/table parity; null profiler counters; prior lifecycle artifacts outside worker ownership.

Next decision needed: reviewer should decide whether Path-B-strict with documented lookup-table proxies is acceptable for M5-S1, or whether manager should open a dedicated WRF-wrapper/table-export fixture sprint.
