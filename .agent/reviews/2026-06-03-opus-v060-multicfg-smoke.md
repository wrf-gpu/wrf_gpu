# v0.6.0 Integrated Multi-Config Operational Smoke — Handoff

Date: 2026-06-03
Author: Opus 4.8 MAX (frontrunner)
Branch: `worker/opus/v060-multicfg-smoke` (base `a95c93c` = v0.6.0 trunk tip,
"[v060] Noah-classic operational land coupler → GPU-operational + scan-wire")

## Objective

Build the **v0.6.0 release prerequisite**: prove the consolidated 12-scheme physics
suite runs end-to-end **through the operational coupler** (`runtime.operational_mode`
+ `coupling.physics_dispatch` + `coupling.scan_adapters`) across **every supported
namelist config** — not just per-scheme in isolation. Each scheme already passed
savepoint parity individually; the OPEN gap this closes is INTEGRATION: do the
schemes run TOGETHER, in WRF physics-driver call order, across the supported config
combinations, staying finite + physical + actually-active (no silent no-op) +
JIT-traceable (== GPU-runnable). WRF-faithful discipline: NO masking/clamp/
self-compare/synthetic-happy-path.

## Files changed

- **NEW** `proofs/v060/multicfg_operational_smoke.py` — the runner.
- **NEW** `proofs/v060/multicfg_smoke_report.json` — the proof object (per-config table).
- **NEW** `.agent/reviews/2026-06-03-opus-v060-multicfg-smoke.md` — this handoff.

No frozen interfaces, schemes, or State were edited (file ownership respected).

## Commands run

```
git worktree add -b worker/opus/v060-multicfg-smoke .claude/worktrees/v060-multicfg-smoke a95c93c
taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 \
    PYTHONPATH=src python3 proofs/v060/multicfg_operational_smoke.py --steps 8
```

CPU only (cores 0-3), NO GPU — verified `jax.default_backend() == "cpu"`, x64 enabled.

## Proof object

`proofs/v060/multicfg_smoke_report.json`. **VERDICT: all_pass = True.**
RUN configs: **12/12 PASS**. FAIL-CLOSED configs: **4/4 OK** (coupler rejected as required).

## What the smoke actually runs

For each config it executes the **EXACT operational physics block** from
`operational_mode._physics_boundary_step` — dispatcher-selected **microphysics →
surface-layer → land (Noah-MP / Noah-classic / bulk) → MYNN PBL → KF cumulus**, in
WRF call order, threading the real KF `(w0avg, nca)` carry and the prognostic
Noah-MP / Noah-classic land carries — inside `jax.jit` + `jax.lax.scan` for 8 steps.
It asserts per config: **compiles** (jit-traceable == GPU-lowerable), **finite**,
**physical** (theta/qv/u/v/w/p/t_skin within WRF-physical envelopes), and
**schemes-active** (the selected suite actually moved its prognosed leaves — no
silent no-op). Each config is ALSO validated through the coupler's OWN fail-closed
gate `operational_mode._resolve_operational_suite` (the real authority the public
`run_forecast_operational` calls), so accept/reject is the coupler's verdict.

### Why the physics-dispatch path, not full `run_forecast_operational`

`run_forecast_operational` couples the WRF acoustic DYCORE to physics; the dycore
needs a dynamically-balanced IC. The corpus-replay state builder (`build_replay_case`
→ `State.zeros`/`Tendencies.zeros`) **hard-requires a JAX GPU device** — it cannot be
built on CPU (this sprint is NO-GPU). A synthetic b2 profile is not acoustically
balanced, so the full dycore NaNs on it (verified) — a DYNAMICS-on-synthetic-IC
artifact ORTHOGONAL to the v0.6.0 PHYSICS-INTEGRATION question. The smoke therefore
isolates the v0.6.0 integration surface (the physics-suite dispatch + adapter
coupling the lanes added) and proves it compiles+runs+stays-physical. The full
dycore+physics GPU forecast vs CPU-WRF is the MANAGER-scheduled GPU gate (see the
seam below).

## Real Canary case

- **Grid**: `GridSpec.canary_3km_template()` (the canonical Canary 3 km grid).
- **Noah-MP land** (sf_surface=4 configs): REAL d02 corpus **wrfinput**
  (`build_noahmp_land_state` over `/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_…`)
  — real Canary land mask + Noah-MP soil/veg warm-start (547 d02 land cells), subset
  to a land-rich 8×8 tile so the canopy-energy-balance smoke stays tiny.
- **Noah-classic land** (sf_surface=2 config): REAL pristine-WRF NOAHMP_SFLX savepoint
  column (`proofs/v060/savepoints_noahclassic.json`, `daytime_veg10`) → WRF-derived
  `NoahClassicStatic`/`NoahClassicLandState`.
- **Atmospheric profile**: validated b2 C-grid pattern (`scanwire_smoke._build_state`)
  on the Canary grid with the real land/sea mask overlaid.

## Config sweep + covering rationale (16 configs)

COVERING set (NOT the full 5×3×3×2×2 = 180-combo product): every supported scheme
appears in ≥1 RUN config plus the real operational Canary config. GF/Tiedtke cumulus
EXCLUDED = documented-TODO. To keep every smoke tiny, the prognostic Noah-MP land
(the heavy per-step cost) is exercised only in its dedicated land-coverage configs;
the MP/sfclay/cu coverage configs route through the fast bulk-surface land so each
config isolates the ONE new axis it covers.

| cfg_id | mp/bl/sf/cu/land | expect | verdict |
| --- | --- | --- | --- |
| real_canary_v020 | 8/5/5/0/Noah-MP | RUN | INTEGRATION_PASS |
| mp_thompson_kf | 8/5/5/1/bulk | RUN | INTEGRATION_PASS |
| mp_wsm6 | 6/5/1/0/bulk | RUN | INTEGRATION_PASS |
| mp_morrison | 10/5/7/0/bulk | RUN | INTEGRATION_PASS |
| mp_wdm6_kf | 16/5/5/1/bulk | RUN | INTEGRATION_PASS (Nc/Nn additive leaves) |
| mp_kessler | 1/5/1/0/bulk | RUN | INTEGRATION_PASS |
| sfclay_mynn | 8/5/5/0/bulk | RUN | INTEGRATION_PASS |
| cu_kf | 8/5/5/1/bulk | RUN | INTEGRATION_PASS |
| cu_none | 8/5/5/0/bulk | RUN | INTEGRATION_PASS |
| land_noahmp | 6/5/1/1/Noah-MP | RUN | INTEGRATION_PASS |
| land_noahclassic | 1/5/1/0/Noah-classic | RUN | INTEGRATION_PASS |
| land_bulk | 8/5/5/0/bulk | RUN | INTEGRATION_PASS |
| pbl_ysu_unwired | 8/**1**/1/0/4 | FAIL_CLOSED | FAIL_CLOSED_OK (coupler rejected) |
| pbl_acm2_unwired | 8/**7**/7/0/4 | FAIL_CLOSED | FAIL_CLOSED_OK (coupler rejected) |
| cu_grellfreitas_unwired | 8/5/5/**3**/4 | FAIL_CLOSED | FAIL_CLOSED_OK (coupler rejected) |
| cu_tiedtke_unwired | 8/5/5/**6**/4 | FAIL_CLOSED | FAIL_CLOSED_OK (coupler rejected) |

Scheme coverage (RUN configs): mp{8,6,10,16,1}, bl{5}, sfclay{1,5,7}, cu{0,1-KF},
land{Noah-MP, Noah-classic, bulk}, plus the WDM6 Nc/Nn additive leaves.

## Integration breakage / findings (HONEST)

**No integration breakage in the GPU-scan-wired suite** — all 12 RUN configs compile,
stay finite, stay physical, and the selected schemes are active. Representative
physical ranges after 8 steps (K / m·s⁻¹): real_canary_v020 theta∈[299.0, 311.1],
t_skin∈[290.8, 299.8] (Noah-MP advances it), w∈[-0.14, 0.18]; mp_wdm6_kf theta∈[299.0,
310.8], t_skin held [299.5, 304.0] on the bulk path (correct — bulk MYNN-sfclay does
NOT prognose skin temperature). KF did not trigger convection on these near-neutral
smoke columns (a stable, finite, evolving `(w0avg, nca)` carry validates the wiring;
KF tendency-firing is its lane savepoint parity, not this smoke's claim).

**The real integration finding is architectural, surfaced as fail-closed assertions
(not masked):** four supported schemes are NOT integrable into the GPU operational
scan as-is, and the coupler `_resolve_operational_suite` correctly REJECTS them
loudly (never silently no-ops):

- **YSU (bl=1) and ACM2 (bl=7) PBL** — single-column HOST-NumPy kernels, NOT
  `jax.lax.scan`-traceable on a device State. The operational scan threads **MYNN
  (bl=5) as the only PBL**. YSU/ACM2 passed per-scheme savepoint parity (lane reports)
  but need a jit/vmap rewrite before the GPU scan.
- **Grell-Freitas (cu=3) and Tiedtke (cu=6/16) cumulus** — faithful CPU-NumPy
  reference ports (`gpu_runnable=False`); GPU-batching TODO.

These are the v0.6.0 PBL/cumulus carry-overs for the post-release roadmap (jit/vmap
YSU+ACM2; GPU-batch GF+Tiedtke). They are documented in the report's
`scheme_coverage.fail_closed_schemes` and per-config `operational_coupler_reject_reason`.

## Decoupled from reference-scoring + the documented seam

This is an INTEGRATION smoke (compiles + finite + physical + schemes-active), NOT an
obs/CPU-WRF skill comparison. It deliberately does NOT duplicate/fork the v0.4.0
forecast-gate reference-resolution code (known namelist-path bug being fixed
separately). The clean seam where the FIXED v0.4.0 reference-scorer plugs in for the
full GPU gate is the documented function
**`proofs.v060.multicfg_operational_smoke.reference_scoring_seam(gpu_fields,
cpu_wrf_fields, *, fields=("T2","U10","V10"), diagnostic_fields=("Q2","PSFC","PBLH",
"SWDOWN","GLW","HFX","LH"))`** — a CONTRACT (raises `NotImplementedError`), not an
implementation. The report's `reference_scoring_seam` block records the manager GPU-run
plan: build the `OperationalNamelist` per RUN config with the real init bundle, run
`run_forecast_operational` on a corpus CPU-WRF d02 case (needs the GPU `State.zeros`
path this CPU smoke cannot use), emit GPU wrfout, then call the seam for per-lead
gridpoint-paired bias/RMSE (continuous_gate pattern).

## Unresolved risks

1. **CPU smoke ≠ full GPU forecast.** This proves the physics suite integrates +
   compiles + stays physical, NOT obs/CPU-WRF skill, and NOT the acoustic dycore on a
   balanced corpus IC (GPU-only). The full dycore+physics GPU gate vs CPU-WRF is
   MANAGER-scheduled (the seam).
2. **PBL coverage is MYNN-only end-to-end.** YSU/ACM2 are savepoint-parity-validated
   but not yet GPU-scan-integrable (carry-over).
3. **GF/Tiedtke cumulus** are CPU-reference ports, excluded by design (carry-over).
4. **Noah-MP CPU cost.** The canopy energy/water balance is the dominant per-step
   cost; the smoke subsets to an 8×8 land-rich tile to stay tiny — on the GPU gate it
   runs the full d02 grid.

## Next decision needed

None blocking. The v0.6.0 integrated multi-config operational smoke PASSES (12/12 RUN,
4/4 fail-closed). For the v0.6.0 release: (a) the manager-scheduled full GPU gate
plugs the FIXED v0.4.0 scorer into `reference_scoring_seam`; (b) the YSU/ACM2 jit/vmap
rewrite + GF/Tiedtke GPU-batch are explicit post-release carry-overs.
