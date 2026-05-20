# Sprint Contract — M5-S1.x Thompson Lookup Table Export

**Sprint ID**: `2026-05-20-m5-s1x-thompson-lookup-table-export`
**Created**: 2026-05-20 evening by manager (Claude Opus 4.7 1M-context)
**Trigger**: M5-S1 attempt-5 reviewer A5 (Claude Opus 4.7) returned Accept-with-required-fixes; remaining residual debt enumerated in `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/M5-S1-NEEDS-S1X.md`. This sprint closes the strict-ADR-005-parity gap on Thompson microphysics.

## Objective

Export the WRF Thompson 2008 lookup tables (`t_Efrw`, `tps_iaus`, `tni_iaus`, rain-freezing tables, snow/graupel moments) from the compiled WRF binary into device-resident JAX arrays, replacing the current linear/bounded proxies in `src/gpuwrf/physics/thompson_column.py`. Tighten Tier-1 fixture parity to strict ADR-005 tolerances after table-substitution.

## Non-Goals

- Do NOT modify the saturation-adjustment algorithm, the process-order refactor (attempt-4 work), or the ice/graupel coefficient fixes (attempts 5-6 work). All landed and verified.
- Do NOT add sedimentation back — still out-of-scope per ADR-005.
- Do NOT pre-emptively downcast tables to FP32 — the ADR-007 sprint (dispatched in parallel) decides precision policy; this sprint stays FP64 until ADR-007 lands.
- Do NOT touch ADR-005, ADR-006 except minor amendment cross-referencing.

## File Ownership

Worker may modify:
- `src/gpuwrf/physics/thompson_column.py` (replace proxies with table lookups)
- `src/gpuwrf/physics/thompson_constants.py` (add table-array imports)
- `src/gpuwrf/physics/thompson_tables.py` (NEW — module hosting the extracted tables as device-resident `jnp.array` constants OR as a `pickle`-loaded asset)
- `scripts/extract_thompson_tables.py` (NEW — one-shot extractor that reads the compiled WRF binary state and dumps tables to a binary asset)
- `scripts/m5_generate_thompson_fixture.py` (regen with strict tolerances)
- `fixtures/manifests/analytic-thompson-column-v1.yaml` (tighten tolerances)
- `artifacts/m5/*` (regenerate)
- `.agent/decisions/ADR-006-thompson-jax-implementation.md` (amend with table-export details + WRF citations per table)
- `tests/test_m5_thompson_*` (extend coverage)
- Worker report.

Worker may NOT modify:
- ADRs other than ADR-006 (minor amendment only).
- `feedback_validation_philosophy.md` or any other memory file.
- Other sprint folders.

## Inputs

Required read order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/writing-gpu-kernels/SKILL.md`
4. `.agent/skills/validating-physics/SKILL.md`
5. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/M5-S1-NEEDS-S1X.md` (your scope)
6. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/reviewer-a5-report.md` (R-3 near-zero-rel-err caveat)
7. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/worker-a6-report.md` (attempt-6 final state)
8. `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md` (operational-RMSE binds, NOT per-cell)
9. `.agent/decisions/ADR-005-first-physics-suite.md` (per-field tolerance schema)
10. `.agent/decisions/ADR-006-thompson-jax-implementation.md` (where the proxies are documented)
11. WRF source: `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre` — esp. the table-initialization blocks (search `t_Efrw`, `tps_iaus`, `tni_iaus`, `tco_iagg`, `tcr_iagg`)
12. The compiled WRF object tree: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_mp_thompson.o` (the Fortran harness already links against this)

## Acceptance Criteria

1. **Tables extracted to a binary asset** stored under `data/fixtures/` (per fixture storage policy) and loaded at module-import time as `jnp.array` device-resident constants. Asset SHA pinned in the manifest. Extraction script reproducibly produces the same SHA.
2. **JAX kernel uses real WRF tables** (not proxies) for: `t_Efrw` (rain-cloud collection efficiency), `tps_iaus` (Berry-Reinhardt autoconv with ice), `tni_iaus` (Ni-autoconv), `tco_iagg` (cloud-ice aggregation), `tcr_iagg` (cloud-rain aggregation), plus snow + graupel moment tables. All replacements cite the WRF source line where the original table is initialized.
3. **Strict ADR-005 Tier-1 tolerances pass**: `abs=1e-10, rel=1e-8` for hydrometeors, `abs=1e-3, rel=1e-6` for `Ni/Nr`. Near-zero-reference rel-errors (per R-3 caveat) use abs-err as the binding metric where reference < 1e-10.
4. **`gate_status = GO`** (strict, not GO_CARRYFORWARD) — i.e. the gate semantics distinction added in attempt-6 lands in the green-strict regime.
5. **No new lookup-table OOM / HLO unroll** — HLO must remain ≤200 KB and 1 kernel launch (the Gemini side-audit's original concern). Tables consumed as device arrays via `jnp.take` or equivalent index-into-array operations, NOT as JIT-time constants.
6. **No regression on Tier-2 conservation, positivity, NaN/Inf**.
7. **No regression on profile metrics**: `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0`.
8. **Bug-fix-parallel-pair rule applied**: every confirmed issue during this sprint that surfaces (e.g. an extraction bug, a wrong table index, a missing offset) dispatches ≥2 AIs in parallel; one MUST be Gemini per project policy. Gemini quota resets ~20:45; sprint may need to delay parallel-pair work until then.
9. `python scripts/validate_agentos.py` passes.
10. `pytest -q` passes (no regression; new table-lookup tests added).

## Validation Commands

```bash
python scripts/validate_agentos.py
python scripts/extract_thompson_tables.py --output data/fixtures/thompson-tables-v1.bin
python scripts/m5_generate_thompson_fixture.py
python scripts/m5_run_thompson.py
python scripts/m5_gate_thompson.py
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-thompson-column-v1.yaml
pytest -q
```

## Performance Metrics

- `artifacts/m5/thompson_profile.json` — must still show 1 launch, 0 temp bytes, 0 H2D post-init.
- `artifacts/m5/hlo_dump/thompson_column_production.txt` — size must be ≤200 KB.
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` — must remain 0 bytes.
- Tier-1 strict residuals — all ≤ ADR-005 tolerances (with the R-3 near-zero caveat where applicable).

## Proof Object

- `data/fixtures/thompson-tables-v1.bin` (or .npz) with pinned SHA in manifest.
- `scripts/extract_thompson_tables.py` (reproducible extractor).
- `src/gpuwrf/physics/thompson_tables.py` (loader + device-resident array bindings).
- `src/gpuwrf/physics/thompson_column.py` (proxy-removed; table-lookup-applied).
- `artifacts/m5/tier1_thompson_parity.json` with strict tolerances met.
- `artifacts/m5/thompson_gate_result.json` with `gate_status=GO` (not GO_CARRYFORWARD).
- ADR-006 amendment documenting table-export approach + per-table WRF citations.
- Worker report.

## Risks

- **Extraction bug class**: reading lookup tables from a compiled Fortran binary is error-prone. Mitigations: (a) the Fortran harness already links against the WRF objects; extend the harness to dump initialized tables to disk as raw bytes; (b) cross-check first few entries by hand against `module_mp_thompson.F.pre` initialization formulas.
- **Gemini quota**: out until ~20:45; if sprint dispatches before then, parallel-pair work waits.
- **HLO unroll regression**: if tables get inlined as JIT-time constants instead of device-resident array references, HLO will balloon. Mitigation: use `jax.numpy.take` or `jax.numpy.asarray(jnp.array(table))` outside `@jit` so they become parameters not constants. Verify HLO size ≤200 KB before commit.
- **Strict tolerance failure**: if even with real tables, residuals stay above ADR-005 strict, this signals deeper process-order or numerical issue. Worker filing `BLOCKER-m5-s1x-strict-tolerance.md` is acceptable — but only after the table-export work itself is correctness-verified.

## Handoff Requirements

- Worker report ≥2500 bytes with: per-table extraction summary, WRF citations, before/after per-field strict-tolerance numbers, HLO size verification, gate verdict.
- Tester (Claude Opus 4.7 xhigh) verifies: extraction reproducibility (same SHA twice), table-lookup-correctness probe, no proxies remaining, strict-tolerance independent recomputation.
- Reviewer (codex critical-review) verifies governance + ADR-006 amendment + strict-gate transition (GO_CARRYFORWARD → GO).

## Dispatch Pattern

- Primary worker: codex gpt-5.5 xhigh.
- Parallel side-runner / bug-fix parallel-pair: Gemini 3.5 (when quota resets), focused on: (a) per-table extraction correctness, (b) any new transcription typos introduced.
- Tester: Claude Opus 4.7 xhigh.
- Reviewer (binding): Claude Opus 4.7 xhigh primary + Gemini parallel side-runner per large-review default-on policy.

## Expected wall-time

Worker phase: 8-16 hours. Table extraction is the long pole.
Tester phase: 1-2 hours.
Reviewer phase: 1-2 hours.
Total: 10-20 hours wall-clock.

## Sequencing

Dispatches **in parallel** with the ADR-007 precision-policy sprint (independent file ownership). Both must close before M6 dispatch. If ADR-007 ships a downcast permission for any Thompson field, this sprint's table loader must absorb that change (likely just changing the dtype on `jnp.array(table, dtype=...)`).
