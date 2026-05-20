# Worker Report - M5-S1 Thompson Microphysics Column Attempt 3

Summary: Replaced the attempt-2 Python formula oracle with a compiled WRF Fortran harness oracle. The fixture generator now writes synthetic columns to `data/scratch/fortran_input_*.dat`, runs `data/scratch/wrf_thompson_harness`, reads `fortran_output_*.dat`, and packages the fixture NPZ/manifest. The JAX Thompson candidate remains the attempt-2 WRF-source transcription. M5 tier-1/tier-2/profile/gate/HLO proof objects pass; M3/M4 milestone checks still fail only on pre-existing lifecycle files outside worker ownership.

## Objective

Implement the attempt-3 contract amendment: build a standalone Fortran harness backed by existing WRF objects, use it as the independent Thompson Tier-1 fixture oracle, regenerate artifacts, update ADR-006, and preserve the M5 validation/gate proof chain.

## Files Changed

- `scripts/wrf_thompson_harness.f90`
- `scripts/wrf_thompson_harness_build.sh`
- `scripts/m5_generate_thompson_fixture.py`
- `scripts/m5_run_thompson.py`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz`
- `artifacts/m5/agent_success.json`
- `artifacts/m5/maintainability.md`
- `artifacts/m5/thompson_profile.json`
- `artifacts/m5/thompson_gate_result.json`
- `artifacts/m5/tier1_thompson_parity.json`
- `artifacts/m5/tier2_thompson_invariants.json`
- `artifacts/m5/hlo_dump/*`
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`
- `tests/test_m5_thompson_tier1.py`
- `tests/test_m5_thompson_fortran_harness.py`
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
{"errors": ["sprint 2026-05-19-m3-state-grid-halo-skeleton not closed: {\\n  \"errors\": [\\n    \"missing reviewer-report.md\"\\n  ],\\n  \"ok\": false\\n}"], "ok": false, "sprints_closed": 0}
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
stdout included WRF table-init messages plus:
```json
{
  "bytes": 7023,
  "harness": "data/scratch/wrf_thompson_harness",
  "harness_sha256": "68af97a244df4902995d9833918bd4569a7e6126bce944a6412f6042cd7b0098",
  "manifest": "fixtures/manifests/analytic-thompson-column-v1.yaml",
  "path": "fortran-harness",
  "sample": "fixtures/samples/analytic-thompson-column-v1.npz",
  "sha256": "6105d779ca86c3e354cb13cda91aedfdd72bca793a3ee424782dbb81a84a3a32",
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
  "profile": {"kernel_launches_per_step": 1, "temporary_bytes_per_step": 0, "host_to_device_bytes_post_init": 0, "device_to_host_bytes_post_init": 0, "registers_per_kernel": null, "local_memory_bytes_per_kernel": null, "wall_time_s": 0.0003023024764843285}
}
```
stderr: empty

`python scripts/m5_gate_thompson.py`
stdout:
```json
{"gate_status": "GO", "kernel_launches_per_step": 1, "local_memory_bytes_per_kernel": null, "registers_per_kernel": null, "tier1_pass": true, "tier2_pass": true}
```
stderr: empty

`python -m json.tool artifacts/m5/tier1_thompson_parity.json`
stdout: valid JSON; `pass=true`, `tolerances_met=true`, max abs errors: `qv=1.1249136010742282e-4`, `qc=1.3156242553626768e-5`, `qi=9.110253202400062e-5`, `Ni=1414680.648483333`.
stderr: empty

`python -m json.tool artifacts/m5/tier2_thompson_invariants.json`
stdout: valid JSON; `pass=true`, water residual `2.670445271854754e-12`, positivity violations `0`, NaN/Inf violations `0`, max latent heating `2.9791947680550948 K`.
stderr: empty

`python -m json.tool artifacts/m5/thompson_gate_result.json`
stdout: valid JSON; `gate_status="GO"`, launches `1`, register/local-memory counters `null`, tier-1/tier-2 pass.
stderr: empty

`ls -l artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stdout:
```text
-rw-rw-r-- 1 enric enric 0 May 20 09:55 artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff
```
stderr: empty

Focused M5 pytest:
```text
11 passed in 10.82s
```

`python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml`
stdout: `fixtures/manifests/analytic-thompson-column-v1.yaml: ok`
stderr: empty

`pytest -q`
stdout:
```text
398 passed in 262.93s (0:04:22)
```
stderr: empty

## Proof Objects Produced

- `scripts/wrf_thompson_harness.f90`
- `scripts/wrf_thompson_harness_build.sh`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz` (7,023 bytes, sha256 `6105d779ca86c3e354cb13cda91aedfdd72bca793a3ee424782dbb81a84a3a32`)
- `data/scratch/wrf_thompson_harness` (external, gitignored, sha256 `68af97a244df4902995d9833918bd4569a7e6126bce944a6412f6042cd7b0098`)
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
| tendency bytes | 0 | Thompson step applies fused source/sink updates directly to the state pytree |
| temporary bytes per step | 0 | no `jnp.array`, `jnp.zeros`, or `jnp.empty` in traced Thompson body; profile records 0 |
| host/device transfer bytes post-init | 0 | profile records 0; scalar `dt` and `debug` are static |
| kernel launches per step | 1 | HLO-derived launch count from production HLO |
| wall time per step | 302.3 us | cached JAX call from `thompson_profile.json` |

## Allocation Audit

- `src/gpuwrf/physics/thompson_column.py`: no `jnp.array`, `jnp.zeros`, or `jnp.empty`. Hot-path `state.replace(...)` constructs pytrees only; array values are fused expressions over existing leaves.
- `src/gpuwrf/physics/thompson_saturation.py`: no array constructors; pure elementwise formulas.
- `scripts/m5_generate_thompson_fixture.py`: NumPy allocations are fixture-generation only and contain no Thompson source/sink formulas.
- `scripts/wrf_thompson_harness.f90`: external oracle binary, not model hot path.
- `src/gpuwrf/validation/tier1_thompson.py`: `jnp.asarray` calls are validation/init-only when loading fixture arrays.
- Tests use `jnp.asarray`/`jnp.ones` only for setup.

HLO debug-vs-stripped diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Risks

- The fixture oracle is now structurally independent, but Tier-1 tolerances are broad because the JAX candidate still lacks exact WRF generated lookup-table parity. This is not final Thompson parity evidence.
- Sedimentation is suppressed in the harness by `dz=1.0e30`, not by a source-level bypass. A dedicated wrapper/table sprint should replace this if exact no-sedimentation WRF parity becomes blocking.
- `gfortran` is not installed; build uses NVHPC `nvfortran` because the WRF objects/modules are NVHPC-built.
- Register/local-memory counters remain `null` due to the known workstation perfmon restriction.
- M3/M4 done checks fail on previous lifecycle artifacts outside this worker's scope.

## Handoff

Objective: M5-S1 attempt-3 worker fix for Thompson column independent oracle.

Files changed: listed above; changes are within the attempt-3 pre-approved scope extension plus the original M5 worker-owned paths.

Commands run: every contract validation command was run. M5-owned commands pass; M3/M4 prior-oracle lifecycle failures are recorded above.

Proof objects produced: all required M5 artifacts plus the Fortran harness source/build script and external harness SHA entry.

Unresolved risks: exact WRF table parity, source-level sedimentation bypass, null profiler counters, and prior lifecycle files outside worker ownership.

Next decision needed: reviewer should decide whether the attempt-3 harness unlocks the sprint despite broad parity tolerances, or whether manager should open a dedicated exact-table/export or patched-WRF no-sedimentation fixture sprint.
