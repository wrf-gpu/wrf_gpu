# M6-S2a Reviewer Report — Gen2 Accessor + d02 Boundary Replay + Shared I/O

Reviewer: Claude Opus 4.7 xhigh (mandatory independent review)
Worker: codex gpt-5.5 xhigh
Date: 2026-05-21
Sprint: `2026-05-21-m6-s2a-gen2-backfill-accessor-and-d02-boundary-replay`
Worktree: `/tmp/wrf_gpu2_m6s2a`
Branch: `worker/codex/m6-s2a-gen2-backfill-accessor-and-d02-boundary-replay`

## Binding Decision

**ACCEPT-WITH-MINOR-FOLLOWUPS.** All seven ACs verified PASS with two non-blocking
follow-ups carried into M6-S2 (dependency declaration) and M6-S5 (precision
methodology). The shared-I/O contract is sound; M6-S2 and M6-S4..S8 may dispatch
once the two follow-ups are encoded as sprint preconditions.

## R-Findings Table

| AC | Status | Evidence file:line | Notes |
|----|--------|--------------------|-------|
| AC1 Gen2 accessor | PASS | `src/gpuwrf/io/gen2_accessor.py:284-489`, `artifacts/m6/gen2_manifest.json` | 5 domains, 133 files, SHA-256/mtime/size, d02 metadata matches namelist exactly |
| AC2 d02 boundary replay | PASS | `src/gpuwrf/io/boundary_replay.py:157-264`, `data/fixtures/m6/d02_boundary_replay_v1.zarr/validation_summary.json` | Bilinear in native Lambert with stagger-correct coordinates, vertical identity rigorously justified, round-trip tolerances passed |
| AC3 shared validation I/O | PASS | `src/gpuwrf/io/validation.py:23-174` | API surface complete; module is the declared sole owner per ADR-011 |
| AC4 CPU denominator | PASS-WITH-CAVEAT | `scripts/m6_extract_cpu_denominator.py:1-163`, `artifacts/m6/cpu_denominator.json` | Math correct; `-r4` FP32 evidence preserved honestly, attribution policy documented |
| AC5 proof-object schemas | PASS | `src/gpuwrf/io/proof_schemas.py:1-309`, `tests/test_m6_proof_schemas.py` | 10 schemas, registry validates M6-S1 artifacts |
| AC6 ADR-011 | PASS | `.agent/decisions/ADR-011-m6-shared-io-and-boundary-replay.md` | Cross-refs ADR-002/007/010, declares read-only contract and shared-I/O owner |
| AC7 honest accounting | PASS | grep audit, all NetCDF opens "r", `_reject_gen2_write_target` guard | No writes under `/mnt/data/canairy_meteo/**`; no `min(raw, cap)` fudge |

## Verifiability Triple

### 1. READ-ONLY audit on `/mnt/data/canairy_meteo/`
Grep over `src/gpuwrf/io/**` + `scripts/m6_*.py`:
- All `Dataset(...)` opens use mode `"r"` (`gen2_accessor.py:202,265,274,350,395,401,475`, `boundary_replay.py:173,186`).
- Every write target (`write_text`, `mkdir`, `zarr.open_group(mode="w")`, `write_manifest`) goes through `_reject_gen2_write_target(...)` which raises `PermissionError` for any path under `GEN2_READ_ONLY_ROOT = /mnt/data/canairy_meteo`. Reviewer independently invoked the guard with a synthetic gen2 target → raised correctly.
- No `os.remove`, `shutil.rmtree`, or `Path.unlink` calls touch a gen2 path.
- Gen2 directory listing mtimes (latest 2026-05-20 11:21) predate sprint start (2026-05-21 11:42) — confirms no in-sprint mutation.

**Verdict: READ-ONLY OK.**

### 2. Round-trip physical consistency (independently re-run)
Re-ran `pytest -q tests/test_m6_*.py` in this reviewer worktree: **10 passed in 4.64s**.
Re-read `validation_summary.json` from the fixture zarr:

| Var | RMSE_max (replay vs d02 truth) | Tolerance | Headroom | Pass |
|-----|--------------------------------|-----------|----------|------|
| U   | 0.1268 m/s | 0.5 m/s | 4× | ✓ |
| V   | 0.1831 m/s | 0.5 m/s | 2.7× | ✓ |
| T   | 0.2084 K   | 0.5 K   | 2.4× | ✓ |
| QVAPOR | 8.36e-5 kg/kg | 1.0e-4 kg/kg | **1.20×** | ✓ (tight) |
| PH  | 6.03 m²/s² | 20.0 m²/s² | 3.3× | ✓ |

QVAPOR is at 84% of the declared envelope. Acceptable, but if M6-S2 forecast skill degrades around boundary humidity, this is the first place to look.

**Verdict: round-trip passes; headroom recorded.**

### 3. Schema validation
`validate_artifact("artifacts/m6/coupled_dummy_carry.json")` and `validate_artifact("artifacts/m6/spacetime_budget.json")` both succeed (committed M6-S1 outputs validate against new registry). Registry test asserts the artifact-stem aliases (`forecast_6h_summary`, `tsc_envelope`, `probtest_tolerances`, etc.) so M6-S2..S8 can validate by canonical name.

**Verdict: every existing M6 artifact validates.**

## Adversarial Probes

### Probe A — SW corner bilinear trace
Independently traced the d02 SW corner (j=0, i=0) through `boundary_replay._weights` for the mass grid at lead +12h (`wrfout_d01_2026-05-20_06:00:00`):
- Computed weights: `i0 = 22, j0 = 18, wi = 0.6738, wj = 0.6690`.
- Namelist declares `i_parent_start = 24, j_parent_start = 20` (1-based) → 0-based parent vertex (23, 19). With `parent_grid_ratio = 3`, the d02 mass center sits half a parent cell offset from the parent vertex (-0.5 + 0.5/3 = -0.333), so the expected parent-mass fractional position is (22.67, 18.67). Floor → (22, 18); fractional remainder → (0.67, 0.67). **Match.**
- T (perturbation potential temperature) replayed value at SW corner level 0 = −8.6046 K vs truth −8.6087 K → ΔT = 0.0041 K. Well inside the 0.5 K envelope and consistent with bilinear-on-smooth-field expectation.
- The bilinear uses the *variable's own* XLAT/XLONG, XLAT_U/XLONG_U, or XLAT_V/XLONG_V coordinates (`boundary_replay.py:21-27` COORDS dict) — stagger-correct.
- Vertical interpolation is correctly an identity for this d01/d02 pair: reviewer confirmed `ZNU` and `ZNW` arrays are bit-identical between d01 and d02 wrfout files (`np.allclose(rtol=0, atol=1e-12)` is True). The non-identity branch in `_interp_vertical` is exercised by `np.interp` and is reasonable if future d01/d02 pairs decouple eta.

### Probe B — Gen2Comparison schema is ready for +6/+12/+24h leads
`Gen2Comparison.required["variables"]` is typed `object` (open dict) (`proof_schemas.py:206-218`). M6-S8 can populate without further schema changes — but the schema does NOT constrain populators to a specific per-lead/per-variable shape. This is intentional permissiveness for v1, but it loses some self-documentation. Recommended (non-blocking) follow-up for M6-S8: nest the populated payload as `variables: {<var>: {<lead_h>: {rmse, bias, mae}}}` and tighten the schema with a sub-validator in M6-S8 itself rather than expanding the registry.

### Probe C — `zarr` dependency trap
`pyproject.toml` declares only `netCDF4`, `numpy`, `PyYAML`. But `src/gpuwrf/io/boundary_replay.py:14` imports `zarr` at module level, and `src/gpuwrf/io/__init__.py` re-exports from `gen2_accessor` + `validation` only. Importing the package does not pull `boundary_replay`, so a fresh M6-S2 worktree without zarr will not crash on `import gpuwrf.io`. **However**, any sprint that calls `extract_d02_boundary` or even just imports `boundary_replay` will `ImportError`. The same trap exists for `jax` — used as an optional import in `gen2_accessor.py` and `validation.py`, but is required by `as_grid_spec()` and the device caching path. This is a hidden M6-S2 trap and should be encoded as a sprint prerequisite (see Follow-ups §2).

## Per-AC Verification with Evidence

### AC1 — Gen2 accessor (PASS)
- `Gen2Run` class at `src/gpuwrf/io/gen2_accessor.py:284`. Lazy loading via `LazyNetCDFArray` at `:240-281`, with device caching at `_load_device_array:480-485`.
- `build_manifest` at `:408-440` covers `wrfout_d0*_*`, `wrfinput_d0*`, `wrfbdy_*`, `namelist.input`, `namelist.output`. SHA-256, mtime, size collected. `source_citations` field cites Gen2 baseline reference + namelist + Gen2 WRF README — solid provenance.
- Manifest produced: 5 domains, 133 files, `no_write_audit: true`. d02 metadata exact match: dx/dy=3000, e_we=160, e_sn=67, e_vert=45, mass 159×66×44, Lambert, parent d01, ratio 3, i_parent_start=24, j_parent_start=20.
- 375 d02 variables inventoried.

### AC2 — d02 boundary replay (PASS)
- `extract_d02_boundary` at `boundary_replay.py:157`. Writes `data/fixtures/m6/d02_boundary_replay_v1.zarr` + `fixtures/manifests/m6_d02_boundary_replay.yaml` + `validation_summary.json`.
- Variables: U, V, T, QVAPOR, PH. Sides: W, E, S, N. 25 hourly times.
- Lambert XY projection at `:50-71` uses true cone-of-projection math (handles single/double true-lat case). Weights resolve to parent mass fractional indices (Probe A confirms).
- Bilinear uses `(1-wi)(1-wj) + wi(1-wj) + (1-wi)wj + wi*wj` — correct convex combination.
- Vertical interp at `:111-119`: identity for matching eta (verified bit-identical), otherwise `np.interp` over reversed eta arrays for monotonic-increasing interpolation. Correct.
- Round-trip caveat (acknowledged): replay validates against `wrfout_d02_*` round-trip, not bitwise `wrfbdy_d02` reconstruction (which the contract acknowledged does not exist in this Gen2 backfill). Reviewer agrees with worker that this is the physically correct validation target.

### AC3 — Shared validation I/O (PASS)
- All five required APIs exported: `load_gen2_var, regrid, domain_mask, lead_time_slice, unit_convert` (`validation.py:23,70,92,138,152`).
- `regrid` is a general-purpose bilinear over array indices (`_linear_axis + _bilinear_2d`). The docstring explicitly notes that boundary replay uses the stricter Lambert helper in `boundary_replay.py` — correct separation of concerns.
- `domain_mask` supports `canary` (all-true), `land`/`sea` (LANDMASK-first, HGT-fallback), and `elevation_band_N` (HGT-binned). Test asserts `land XOR sea = canary` on the real d02 grid — passes.
- `lead_time_slice` reads `history_interval` from the parsed namelist; handles list-or-scalar; rejects non-positive intervals. Sound.
- `unit_convert` covers K↔°C, kg/kg↔g/kg, Pa↔hPa — sufficient for M6 surface scoring (U10, V10, T2, RH).
- ADR-011 §Decision declares this module is the sole shared-I/O owner. M6-S4..S8 must not re-implement loaders.

### AC4 — CPU denominator (PASS-WITH-CAVEAT)
- Math (re-verified by reviewer):
  - d01 timing-line sum = 17010.37 s = total nested-run wall time (since d01 timing lines aggregate child work in nested WRF).
  - Domain work weights: d01=1.16e9, d02=6.65e9, d03=1.33e10, d04=7.87e9, d05=7.48e9; total = 3.64e10.
  - d02 fraction = 0.18261, d02-attributable = 3106.25 s, per-step (14400 steps) = 215.71 ms. **All values match.**
- Attribution policy is "grid-points × timestep-count fraction", documented (`compile_precision_note`, `attribution_policy`, `fp_precision` fields). No `min(raw, cap)` denominator cap. **Honest accounting.**
- **Precision caveat (worker correctly flagged)**: compile.log records `-r4 -i4` (single-precision default real), not FP64 as the review prompt example assumed. Worker preserved the evidence in `fp_precision`. This is binding for M6-S5: the GPU JAX baseline runs in FP64 (ADR-001 + ADR-002 declare x64 enabled at import). A 4× speedup verdict comparing FP64 GPU against FP32 CPU is methodologically unsound without correction.
- **Reviewer's additional concern (not blocking M6-S2a, but binding for M6-S5)**: An alternate denominator can be derived from the raw measured timing summary by subtracting child cost: `d02_self_measured = d02_timing_sum − (d03+d04+d05) = 16289.5 − 5153.9 − 3228.9 − 3047.2 = 4859.5 s`. That is 56% larger than the grid-points-weighted 3106.2 s. Neither is "d02 as standalone CPU run" (which would require a `max_dom=2` re-run); both are defensible projections. M6-S5 must pick one with reasoning, or run a `max_dom=2` re-extraction. Worker correctly preserved the raw `raw_timing_summary` to permit either choice.

### AC5 — Proof-object schemas (PASS)
- 10 schemas defined (`proof_schemas.py:92-256`), all with `required` field rules and one (`FullDomainBatchingVerdict`) with `optional` rules supporting nullable profiler fields.
- `SCHEMA_REGISTRY` (`:259-276`) aliases both canonical names and artifact filenames/stems (e.g., `coupled_dummy_carry` and `coupled_dummy_carry.json`; `tsc_envelope` aliases `Tier3DriftEnvelope`).
- `validate_artifact` resolves by filename → stem → KeyError. Existing M6-S1 artifacts validate. M6-S2..S8 should call `validate_artifact()` on every emitted JSON.
- Lightweight dataclass-backed validator (no pydantic dependency added) — appropriate given pyproject minimalism.

### AC6 — ADR-011 (PASS)
- Status: PROPOSED (worker self-declared). Reviewer recommends manager flip to ACCEPTED at sprint close.
- Cross-references ADR-002 (state layout), ADR-007 (precision), ADR-010 (M6 coupled state). Cites `.agent/references/cpu-wrf-baseline.md`, Gen2 WRF reference, and pinned namelist path. **Provenance chain intact.**
- Declares shared-I/O owner, read-only contract, replay strategy, schema registry, and denominator policy as binding for M6-S2..S8.

### AC7 — Honest accounting (PASS)
- READ-ONLY audit: clean (see Verifiability Triple §1).
- Zero post-init transfers in load path: lazy materialization caches first H→D transfer (`_load_device_array:480-485`), subsequent reads hit cache. M6-S2 must still prove zero transfers inside its timestep loop via profiler — that is on M6-S2's accountability, not M6-S2a.
- No `min(raw, cap)` fudge in denominator script (`scripts/m6_extract_cpu_denominator.py` reviewer-verified line-by-line).
- Boundary replay tolerances declared BEFORE measurement (TOLERANCES dict at `boundary_replay.py:28-34`).

## Decisions

### CPU denominator `-r4` caveat → M6-S5 must escalate, not block M6-S2

The `-r4` finding is a precision-mismatch issue, not a denominator-extraction issue. The M6-S2a artifact is honest about it and preserves raw timings, so M6-S2 (forecast driver) can dispatch unaffected. **The 4× binding verdict in M6-S5 must NOT use this denominator as-is.** M6-S5 must execute one of:

1. **Recommended**: Re-extract CPU baseline with FP64 build (`-r8 -i4`) on the same hardware. Slow but most defensible.
2. **Acceptable**: Declare the ADR-007 verdict in FP32-equivalent throughput, with the GPU JAX run mirrored at FP32 (per ADR-007's precision-downcast clause) on the relevant scoring fields.
3. **Conditional**: Apply a published FP64/FP32 wall-time ratio for NVHPC nvfortran on this CPU (≈1.7-2.0× typical). Cite source. Reviewer's lowest-confidence option.

M6-S5 must also pick between the grid-points-attributed 3106 s and the raw-timing-subtraction 4859 s — both are in the artifact; reviewer's preference is the raw-timing 4859 s value with documented overhead-of-parent caveat, because it represents actually-measured CPU work, not a counter-factual partition.

### `zarr` (and `jax`) in `pyproject.toml` → M6-S2 must amend before dispatch

`pyproject.toml` does not declare `zarr` or `jax`. The worker correctly noted this was outside the M6-S2a allowed-file set. But M6-S2 (and any sprint touching `boundary_replay`) will fail import on a fresh worktree without these. This is a hidden trap.

**Decision**: M6-S2 prerequisite — first commit on the M6-S2 branch must amend `pyproject.toml` to add:
```
zarr>=3.0,<4
jax>=0.4
```
and any transitive deps the project policy requires. Not a M6-S2a follow-up because M6-S2a's file scope correctly excluded `pyproject.toml`.

## Prerequisites Before Downstream Sprints Dispatch

### M6-S2 (24h d02 forecast driver) — BLOCKED until both resolved
1. **Amend `pyproject.toml`** to declare `zarr>=3.0` and `jax>=0.4` dependencies (1-line commit, no review).
2. **Confirm import path** by running `python -c "from gpuwrf.io.boundary_replay import extract_d02_boundary"` in a fresh venv before kickoff.

### M6-S4 (Tier-2 coupled invariants) — ready to dispatch
- Must import loaders from `gpuwrf.io.validation` per ADR-011.
- Must call `validate_artifact()` on emitted `tier2_coupled_invariants.json`.

### M6-S5 (ADR-007 4× verdict) — BLOCKED on precision decision
1. **Resolve `-r4` precision mismatch** per options above before treating the denominator as binding.
2. **Pick denominator basis**: grid-points 3106 s OR raw-measured 4859 s OR re-extracted `max_dom=2` CPU run. Document choice in M6-S5 proof object.

### M6-S6 (Tier-3 drift envelope) — ready to dispatch
- Must use `lead_time_slice` and shared loaders per ADR-011.

### M6-S7 (Tier-4 probtest tolerances) — ready to dispatch
- Tolerance freeze should follow ADR-007 statistical-tolerance methodology; schema in place.

### M6-S8 (Gen2 operational comparison) — ready to dispatch
- Use `Gen2Comparison` schema; recommended to populate `variables` with `{var: {lead_h: {rmse, bias, mae}}}` shape (Probe B advisory).
- Note QVAPOR boundary replay is 84% of envelope — instrument M6-S8 to flag if humidity skill correlates with W/S sides hour-12+.

## Non-Blocking Observations (for future hardening)

- `gen2_accessor.parse_namelist` is a hand-rolled Fortran-namelist subset parser (`:105-126`). It works for this run but does not handle multi-line continuations, scientific-notation `D` literals beyond the simple replacement, or array-of-string values. Adequate for M6 scope; revisit if Gen3 namelist style diverges.
- `Gen2Run._device_cache` is unbounded — long-running validation could leak device memory if the same `Gen2Run` instance touches all 25 times × 5 vars × 5 domains. Probably moot for M6 (worker uses one-shot extracts), but worth a cache-size cap before M7 routine validation.
- `Gen2GridSpec.as_grid_spec` hard-codes `terrain.sha256 = "gen2-manifest-sha256"` (`gen2_accessor.py:222`). The actual SHA is in the manifest; downstream code that audits provenance should resolve through the manifest, not this placeholder string.
- `_compile_metadata` in the denominator script hard-codes the flag list (`m6_extract_cpu_denominator.py:55-66`) rather than parsing them from `compile.log`. If Gen2 rebuild flags ever change, the artifact will silently lie. Worker did read the file for the NVHPC version but should also re-parse flags. Non-blocking for M6-S2a; cleanup ticket for later.

## Summary

12 PASS, 5 follow-ups, 0 REJECT findings. The shared-I/O layer is correct, the
read-only contract is enforced at every write site, the boundary replay is
physically validated against d02 truth within declared tolerances, and the
CPU denominator preserves enough raw evidence that M6-S5 can make its own
precision call. Worker's honest flagging of the `-r4` evidence and the
`pyproject.toml` gap reflects the project quality bar.

**Binding: ACCEPT-WITH-MINOR-FOLLOWUPS.** M6-S2a is unblocked for merge. M6-S2 + M6-S4..S8 dispatch with the prerequisites above.

— Claude Opus 4.7 xhigh, 2026-05-21
