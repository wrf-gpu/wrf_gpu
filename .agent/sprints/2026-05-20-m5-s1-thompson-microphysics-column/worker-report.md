# Worker Report - M5-S1 Thompson Microphysics Column

Summary: Implemented the M5-S1 Thompson column deliverable as a JAX source/sink column kernel with sedimentation out of scope, a Path-B WRF-source-mapped analytic fixture, tier-1 parity, tier-2 invariants, HLO production-vs-stripped proof, profile proxy, GO gate dry-run, ADR-006 draft, and tests. The M5-owned proof objects pass. Required M3/M4 done-oracles were run and failed only on prior lifecycle files outside this worker's allowed scope.

## Objective

Implement `step_thompson_column(state, dt, *, debug=False) -> state` for the ADR-005 frozen Thompson source/sink subset; generate `analytic-thompson-column-v1`; emit tier-1/tier-2/profile/gate/HLO proof objects; draft ADR-006 implementation mapping.

## Files Changed

- `src/gpuwrf/physics/__init__.py`
- `src/gpuwrf/physics/thompson_constants.py`
- `src/gpuwrf/physics/thompson_saturation.py`
- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`
- `src/gpuwrf/validation/tier1_thompson.py`
- `src/gpuwrf/validation/tier2_thompson.py`
- `scripts/m5_generate_thompson_fixture.py`
- `scripts/m5_run_thompson.py`
- `scripts/m5_gate_thompson.py`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz`
- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/hlo_dump/thompson_column_production.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_stripped.txt`
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
- `artifacts/m5/maintainability.md`
- `artifacts/m5/agent_success.json`
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`
- `tests/test_m5_thompson_column_shapes.py`
- `tests/test_m5_thompson_constants.py`
- `tests/test_m5_thompson_saturation.py`
- `tests/test_m5_thompson_tier1.py`
- `tests/test_m5_thompson_tier2.py`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-report.md`

## Commands Run + Output

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
  "bytes": 7770,
  "manifest": "fixtures/manifests/analytic-thompson-column-v1.yaml",
  "path": "B",
  "sample": "fixtures/samples/analytic-thompson-column-v1.npz",
  "sha256": "81de264f443f5ad153a1b05256dabc2a01856a41e800181aea1b332f2f5011d7",
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
    "wall_time_s": 0.00012779596727341413
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

stdout: valid JSON, `pass=true`, all nine output fields pass, max abs errors: `Ni=1.862645149230957e-09`, `qc=4.0657581468206416e-20`, `qg=1.3552527156068805e-20`, all others `0.0`.
stderr: empty

`python -m json.tool artifacts/m5/tier2_thompson_invariants.json`

stdout: valid JSON, `pass=true`, positivity violations `0`, NaN/Inf violations `0`, water residual `1.6258738137704087e-16`, max latent-heating delta `0.8232820038460318 K`.
stderr: empty

`python -m json.tool artifacts/m5/thompson_gate_result.json`

stdout: valid JSON, `gate_status="GO"`, launches `1`, register/local memory `null`, tier-1/tier-2 pass.
stderr: empty

`ls -l artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`

stdout:
```text
-rw-rw-r-- 1 enric enric 0 May 20 04:57 artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff
```
stderr: empty

`pytest -q`

stdout:
```text
393 passed in 270.64s (0:04:30)
```
stderr: empty

Additional focused check run before full validation:

`python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml` -> `fixtures/manifests/analytic-thompson-column-v1.yaml: ok`

## Proof Objects Produced

- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz` (7,770 bytes, sha256 `81de264f443f5ad153a1b05256dabc2a01856a41e800181aea1b332f2f5011d7`)
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

## Spacetime Budget

| Item | Value | Justification |
|---|---:|---|
| state bytes | 3,168 | 11 fp64 leaves x 3 scenarios x 12 levels x 8 bytes |
| tendency bytes | 0 | Thompson step applies source/sink updates directly to the state pytree |
| temporary bytes per step | 0 | no `jnp.array`, `jnp.zeros`, or `jnp.empty` in traced Thompson body; profile JSON records 0 |
| host/device transfer bytes post-init | 0 | profile records no post-init transfers; no host materialization in kernel body |
| kernel launches per step | 1 | HLO-derived launch count from `thompson_column_production.txt` |
| wall time per step | 127.8 us | median cached JAX call on the analytic fixture |

## Allocation Audit

- `src/gpuwrf/physics/thompson_column.py`: no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls. Hot-path replacements construct `ThompsonColumnState` containers only; array values are fused expressions over existing leaves.
- `src/gpuwrf/physics/thompson_saturation.py`: no array constructors; pure elementwise formulas.
- `src/gpuwrf/validation/tier1_thompson.py`: `jnp.asarray` calls are validation/init-only when loading fixture arrays, outside the kernel hot path.
- `tests/test_m5_thompson_column_shapes.py`: `jnp.asarray`/`jnp.ones` calls are test setup only.
- `scripts/m5_generate_thompson_fixture.py`: NumPy allocations are fixture-generation only.
- `scripts/m5_run_thompson.py`: timing/profile calls materialize outputs only after compiled kernel completion.

HLO debug-vs-stripped diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Risks

- Path B was used. The fixture is source-mapped to WRF formulas, but not generated by compiling and calling WRF `module_mp_thompson.F.pre`; reviewer should treat this as a source-transcription validation object, not a compiled-WRF savepoint.
- This is a Thompson-shaped source/sink subset, not full WRF Thompson. Sedimentation, WRF lookup-table collection integrals, aerosol activation/scavenging, graupel volume/hail state, radar diagnostics, and effective-radius diagnostics remain out of scope.
- Register and local-memory counters are `null` due to the known workstation `ncu`/perfmon restriction. The gate GO is based on correctness and HLO-derived launch count.
- Required M3/M4 oracles fail on lifecycle artifacts outside worker ownership: M3 missing `reviewer-report.md`; M4 sprint folder missing `manager-closeout.md` and `memory-patch.md`.

## Handoff

Objective: M5-S1 worker implementation for Thompson column source/sink pipeline.

Files changed: listed above; all are within the sprint contract scope, except `src/gpuwrf/physics/thompson_column_debug_stripped.py`, which is implied by AC 6.1 for the HLO proof.

Commands run: all sprint validation commands were run; M5-owned commands pass; M3/M4 prior-oracle failures are recorded above.

Proof objects produced: all required M5 proof objects listed above.

Unresolved risks: Path-B fixture source-transcription review, missing prior lifecycle artifacts outside worker scope, null register/local-memory counters.

Next decision needed: reviewer should decide whether the Path-B source mapping is acceptable for M5-S1 or requires a follow-up fixture-generation sprint that compiles WRF Thompson directly.
