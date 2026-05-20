# Worker Report - M5-S1 Thompson Microphysics Column Attempt 4

Summary: Attempt 4 implemented the diagnosis-prescribed narrow fixes: WRF-order source/sink sequencing in the JAX kernel, cloud-ice number changes gated to sublimation only, a real no-sedimentation WRF harness patch instead of the `dz=1e30` workaround, and restored ADR-005 strict Tier-1 tolerances. The order fix worked on the main thermal error (`T` max error down to `0.040290844661740266 K`, below the `<0.1 K` backstop), Tier-2 and HLO identity still pass, but strict Tier-1 parity still fails. I wrote `BLOCKER-m5-s1-attempt4-tolerance.md` as required; remaining work is WRF lookup-table/moment parity.

## Objective

Resolve attempt-4 blockers without widening scope: process-order refactor, Ni deposition fix, proper sedimentation bypass, strict tolerance restoration, MORNING-REPORT preservation check.

## Files Changed

- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`
- `scripts/wrf_thompson_harness.f90`
- `scripts/wrf_thompson_harness_build.sh`
- `scripts/m5_generate_thompson_fixture.py`
- `scripts/m5_run_thompson.py`
- `fixtures/manifests/analytic-thompson-column-v1.yaml`
- `fixtures/samples/analytic-thompson-column-v1.npz`
- `artifacts/m5/*`
- `.agent/decisions/ADR-006-thompson-jax-implementation.md`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/BLOCKER-m5-s1-attempt4-tolerance.md`
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-report.md`

## Commands Run + Output

`python scripts/validate_agentos.py`
stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
stderr: empty

`python scripts/check_m1_done.py`
stdout: `ok=false`; failed because nested `pytest -q` sees `tests/test_m5_thompson_tier1.py::test_m5_thompson_tier1_parity_passes`; one run also exposed transient M4/M2 generated-test failures from nested full-suite reruns. Canonical standalone pytest below shows the persistent failure is the Thompson strict Tier-1 blocker.
stderr: empty

`python scripts/check_m2_done.py`
stdout: `ok=false`, `candidates_satisfied=6/6`; failed through nested pytest/M1 regression because strict Thompson Tier-1 now fails.
stderr: empty

`python scripts/check_m3_done.py`
stdout: `ok=false`; failed through nested pytest/M1/M2 regressions from strict Thompson Tier-1 and pre-existing `missing reviewer-report.md` for `2026-05-19-m3-state-grid-halo-skeleton`.
stderr: empty

`python scripts/check_m4_done.py`
stdout: `ok=false`; failed through nested pytest/M1/M2/M3 regressions from strict Thompson Tier-1 plus pre-existing M4 lifecycle gaps: `missing manager-closeout.md`, `missing memory-patch.md`, `missing artifacts directory`.
stderr: empty

`python scripts/m5_generate_thompson_fixture.py`
stdout included WRF table-generation messages and:
```json
{"bytes": 7026, "harness": "data/scratch/wrf_thompson_harness", "harness_sha256": "6a5b66e1e6d625b804bdff99b0c42c9df38737fce683eabdab24a5a5b49cc3d6", "manifest": "fixtures/manifests/analytic-thompson-column-v1.yaml", "path": "fortran-harness", "sample": "fixtures/samples/analytic-thompson-column-v1.npz", "sha256": "a357e2ef8f6e77ede0c6a79debc0dd0d8de1582585743fad3f6e533ba05d7102", "wrf_source_exists": true}
```
stderr: empty

`python scripts/m5_run_thompson.py`
stdout: exited `1` by design because strict Tier-1 failed. Key output: HLO diff sha `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; Tier-2 `pass=true`; profile `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, transfer bytes post-init `0`; Tier-1 `pass=false`.
stderr: empty

`python scripts/m5_gate_thompson.py`
stdout:
```json
{"gate_status": "FALLBACK", "kernel_launches_per_step": 1, "local_memory_bytes_per_kernel": null, "rationale": "correctness failed", "registers_per_kernel": null, "tier1_pass": false, "tier2_pass": true}
```
stderr: empty

`python -m json.tool artifacts/m5/tier1_thompson_parity.json`
stdout: valid JSON; `pass=false`, `tolerances_met=false`; max abs errors: `qv=1.4304079020558032e-05`, `qc=0.0001517228938283358`, `qr=4.760876436193939e-06`, `qi=0.00013708094759935302`, `qs=0.0001447943623500526`, `qg=1.521843532880611e-05`, `Ni=126975.12500000041`, `Nr=67300.453125`, `T=0.040290844661740266`.
stderr: empty

`python -m json.tool artifacts/m5/tier2_thompson_invariants.json`
stdout: valid JSON; `pass=true`, water residual `2.670445271854754e-12`, positivity violations `0`, NaN/Inf violations `0`, max latent heating `2.729736643819791 K`.
stderr: empty

`python -m json.tool artifacts/m5/thompson_gate_result.json`
stdout: valid JSON; `gate_status="FALLBACK"`, tier1 `false`, tier2 `true`.
stderr: empty

`ls -l artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stdout: `-rw-rw-r-- 1 enric enric 0 May 20 13:53 artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`
stderr: empty

`python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml`
stdout: `fixtures/manifests/analytic-thompson-column-v1.yaml: ok`
stderr: empty

`pytest -q`
stdout:
```text
1 failed, 397 passed in 275.26s (0:04:35)
FAILED tests/test_m5_thompson_tier1.py::test_m5_thompson_tier1_parity_passes
```
stderr: empty

`ls -l MORNING-REPORT.md`
stdout: `-rw-rw-r-- 1 enric enric 12708 May 20 13:34 MORNING-REPORT.md`
stderr: empty

`git diff --name-status main...HEAD | rg 'MORNING-REPORT|^D' || true`
stdout: empty, so integration diff does not delete `MORNING-REPORT.md`.
stderr: empty

## Proof Objects Produced

- `artifacts/m5/tier1_thompson_parity.json` (strict tolerance failure proof)
- `artifacts/m5/tier2_thompson_invariants.json` (pass)
- `artifacts/m5/thompson_profile.json` (1 launch, transfer/temp bytes zero, register/local null due perfmon)
- `artifacts/m5/thompson_gate_result.json` (correctness failure)
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` (0 bytes)
- `fixtures/manifests/analytic-thompson-column-v1.yaml` (strict tolerances restored)
- `fixtures/samples/analytic-thompson-column-v1.npz`
- `data/scratch/wrf_thompson_harness` (external, gitignored, sha256 in manifest)
- `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/BLOCKER-m5-s1-attempt4-tolerance.md`

## Spacetime Budget

| Item | Value | Justification |
|---|---:|---|
| state bytes | 3,168 | 11 fp64 leaves x 3 scenarios x 12 levels x 8 bytes |
| tendency bytes | 0 | Thompson step applies fused source/sink updates directly to state leaves |
| temporary bytes per step | 0 | no `jnp.array`, `jnp.zeros`, or `jnp.empty` in traced Thompson body; profile records 0 |
| host/device transfer bytes post-init | 0 | scalar `dt` and `debug` are static; profile records 0 |
| kernel launches per step | 1 | HLO-derived launch count |
| wall time per step | 281.99 us | cached JAX call from `thompson_profile.json` |

## Allocation Audit

- `src/gpuwrf/physics/thompson_column.py`: no `jnp.array`, `jnp.zeros`, or `jnp.empty`; hot-path `state.replace(...)` constructs pytrees around fused expressions.
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`: no array constructors; stripped sequencing only.
- `scripts/m5_generate_thompson_fixture.py`: NumPy allocations are fixture generation only.
- `src/gpuwrf/validation/tier1_thompson.py`: `jnp.asarray` calls are validation/init-only.
- HLO debug-vs-stripped diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Risks

- Strict Tier-1 parity remains blocked by WRF lookup-table and moment parity debt (`t_Efrw`, ice/snow/rain-freezing tables, snow/graupel moments, and related number/mass balance details).
- `m5_gate_thompson.py` reports `FALLBACK` because correctness fails; this is a physics-parity blocker, not evidence that JAX performance failed.
- Register/local-memory counters remain `null` due workstation perfmon restrictions.
- The no-sedimentation harness patch zeroes terminal velocity arrays before the WRF sedimentation flux loops; it is cleaner than `dz=1e30` but still not a factored WRF source-only subroutine.

## Handoff

Objective: complete attempt-4 narrow fixes and stop honestly if strict ADR-005 tolerance parity cannot be met.

Files changed: listed above; no tracked out-of-scope generated M2 artifact changes are left in the worktree.

Commands run: every contract validation command was run; strict Tier-1 parity and downstream full-suite checks fail as recorded.

Proof objects produced: listed above, especially the blocker file and strict Tier-1 failure artifact.

Unresolved risks: exact WRF table/moment parity and profiler register/local-memory counters.

Next decision needed: manager should dispatch M5-S1.x for table/moment export and exact parity, or amend ADR-005 if a narrower Thompson proof is acceptable.
