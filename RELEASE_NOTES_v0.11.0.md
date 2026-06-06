# Release Notes — v0.11.0

- **Tag:** `v0.11.0`
- **Release commit:** resolve with `git rev-parse v0.11.0^{commit}` after the annotated tag is created. Trunk HEAD at docs-commit: see `git log --oneline -1`.
- **Branch:** `worker/opus/v0110-integration`
- **Binding numbers:** every performance or fidelity figure below traces to committed proof objects under [`proofs/v0110/`](proofs/v0110/) and the v0.9.0/v0.10.0 baselines. Nothing is rounded upward, invented, or relaxed to manufacture a pass.

## What v0.11.0 is

v0.11.0 is the **feature-complete release** of wrf_gpu. It adds live multi-domain nesting,
WRF restart continuity, closed conservation budgets, MYNN-EDMF, topographic/slope radiation,
terrain-slope diffusion, Kain-Fritsch cumulus, gravity-wave drag context, and optional
multi-GPU/DGX sharding to the validated standalone forecast system from v0.10.0. It also
ships a recompile fix (no per-chunk XLA recompile in production) and resolves two prior open
known issues (KI-1 d03 1 km instability and KI-2 long single-call qke edge).

**v0.10.1 hotfix note:** v0.10.1 (released on the same code lineage, between v0.10.0 and v0.11.0) was a targeted hotfix for the XLA recompile regression: `jit(_advance_chunk)` was recompiling on every chunk in long runs due to a non-JAX-contract-compliant `tree_unflatten` in `State` and `DycoreMetrics`. The fix is included in v0.11.0 and the proof is at `proofs/v0110/recompile_fix2_3chunks.json`.

## New features in v0.11.0

### Live multi-domain nesting

`src/gpuwrf/runtime/domain_tree.py` drives d01→d02→d03 one-way live nesting with:
- Per-domain subcycling at WRF-faithful step ratios (e.g. d01/d02/d03 = 18 s / 6 s / 2 s).
- Parent-produced live boundary packages passed to child domains at each boundary update cadence.
- Synchronized multi-domain output.
- Optional two-way feedback gate (`src/gpuwrf/coupling/boundary_feedback.py`), disabled by default; unit-proven including feedback conservation.

**Validation:** Full nested d01→d02→d03 24 h one-way forecast on case `20260521_18z` ran
finite and stable (72 records across 3 domains, all fields finite). Final-lead (h=24) T2
RMSE vs CPU-WRF: **d02 1.03 K / d03 1.10 K**. Mean T2 RMSE over 24 leads: d02 1.31 K /
d03 1.67 K. Single case, single season; no TOST equivalence is claimed.

### WRF restart (`wrfrst`)

`src/gpuwrf/io/wrfrst_netcdf.py` writes and reads WRF-compatible restart files covering
all 75 prognostic/carry fields (State, carry, optional NoahMP groups). Restart continuity
is **bit-identical**: a-path vs b1-path + b2-path (restart at midpoint) produce identical
final states on all 75 fields. Proof: `proofs/v0110/restart_continuity.json`.

### Conservation budgets closed

Dry-mass, total-water, and moist-static-energy relative budget residuals are **0.0** (fp64)
on the validated d02 case. The conserving physics coupling path applies dry tendencies through
`rk_addtend_dry` at each RK stage and non-dry physics deltas post-dycore. Conservation unit
tests (2/2 PASS). Proof: `proofs/v0110/conservation_budgets_closed.json`.

### MYNN-EDMF (PBL + surface)

The operational MYNN coupler now enables DMP mass flux (`bl_mynn_edmf=1`,
`bl_mynn_edmf_mom=1`), cloud-aware thermodynamics (`icloud_bl=1`), and scalar/momentum
plume fluxes. Mass-flux oracle parity: max relative error 0.00483 (tol 0.05).
Proof: `proofs/mynn_edmf/mf_oracle_compare.json`, `proofs/v0110/mynn_edmf_parity.json`.

### RRTMG topographic shading + slope-corrected surface radiation

`topo_shading=1` and `slope_rad=1` are now operational. Gross SWDOWN RMSE vs WRF SWDNB:
7.56 W/m² (tol 15 W/m²); SWNORM RMSE: 6.69 W/m² (tol 35 W/m²). Proof:
`proofs/v0110/rrtmg_slope_parity.json`.

### Terrain-slope + map-factor diffusion deformation terms

`src/gpuwrf/dynamics/explicit_diffusion.py` now includes WRF terrain-following coordinate
slope corrections and map-factor terms in all diffusion paths (constant-K and Smagorinsky).
Formula parity vs WRF `dyn_em/module_diffusion_em.F`: max residual 3.78e-15 (tol 2e-10).
Proof: `proofs/v0110/slopediff_parity.json`.

### Kain-Fritsch cumulus

`cu_physics=1` (Kain-Fritsch) is now scan-wired and dispatched in the operational loop.
Column/savepoint parity: max tendency abs error 7.99e-8, max relative error 7.11e-5.
Proof: `proofs/v0110/kf_parity.json`, `proofs/v0110/kf_status.md`.

### Gravity-wave drag context

`gwd_opt=1` is accepted by the namelist. On the Canary corpus cases, d02/d03 domains have
`GWD_OPT=0` (zero `CON/OA/OL` descriptor fields); d01 GWD effects are baked into the
CPU-WRF lateral-boundary artifacts. GWD on a live d01 parent is a v0.12.0 item.
Proof: `proofs/v0110/gwd_status.json`.

### Optional multi-GPU/DGX sharding (single-GPU default = zero overhead)

`src/gpuwrf/runtime/sharding.py` + `src/gpuwrf/dynamics/sharded_horizontal.py` implement
domain decomposition over a mesh of GPU devices. With `ShardingConfig.disabled()` (the
production default), all sharding code is behind unconditional early-return guards:
**56/56 State field SHA-256 hashes are bit-identical** between the reference (single-GPU) and
the DGX-d2 sharding-disabled path. Sharding tests: 10 passed, 6 skipped (multi-device tests
skipped on single GPU). Proof: `/tmp/v0110_overnight/dgx_bitident_result.json`.

## KI-1 RESOLVED — d03 1 km steep-terrain stability

The prior open KI-1 (gated-fp32 qke non-finite at h+1 over Tenerife steep terrain) is
**closed** by two fixes:
1. WRF-faithful qke cold-start seed (`mym_initialize` background TKE profile).
2. MYNN qke IEEE fmax/fmin fix (`jnp.fmin(jnp.fmax(value, QKEMIN), 150.0)` matching WRF
   Fortran MAX/MIN semantics at `module_bl_mynnedmf.F:3106-3107`).

The d03 Tenerife replay ran **24 h finite in gated-fp32** with T2/U10/V10 within operational
bars. **Requirement:** initial state must carry WRF-faithful qke cold-start seed (the
`build_real_init` path and WRF-restart path both satisfy this).

## KI-2 RESOLVED — long single-call qke edge

The MYNN IEEE fmax/fmin fix also closes the long single-call qke edge (KI-2). The KI-2
gate on the merged trunk confirms all 8 tracked prognostic fields finite including qke on
the case that previously triggered the edge. Proof: `proofs/v0110/qke_ki2_gate_merged_trunk.json`.

## d02 wind regression fix

The MYNN-EDMF/conservation integration introduced a dry-physics RK-cadence regression that
degraded d02 wind skill to U10 mean 5.54 m/s (was 4.41 m/s in v0.9.0), losing persistence
at all leads. The fix (commit `5e8aabe`) restores the correct cadence: U10 mean **4.43 m/s**,
V10 mean **3.59 m/s**, beats persistence **23/24 leads** — recovering to v0.9.0 parity.
Proof: `proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json`.

## Recompile fix

`jit(_advance_chunk)` now compiles once and is reused on every subsequent chunk. Chunks 2-3
run at ~65.7 ms/step (18s/chunk), no per-chunk XLA recompile. The root cause was a
non-JAX-contract-compliant `tree_unflatten` in `State` and `DycoreMetrics`. This fix was
also shipped as v0.10.1. Proof: `proofs/v0110/recompile_fix2_3chunks.json`.

## Known issues / carried forward

Full write-up: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

- **KI-3** (scope boundary): focused 64-variable wrfout vs CPU-WRF's 375.
- **KI-4** (residual): U10 final-lead RMSE 8.06 m/s (just above 7.5 m/s bar; 23/24 beats persistence).
- **KI-5** (scope boundary): powered n=15 TOST not scored; corpus prepared.
- **KI-6** (pre-existing): RRTMG SW intermediate taug in 4 UV bands; tier-1 flux outputs faithful.
- **KI-7** (new): free-running wide-domain (nx≈160+) without boundary relaxation can blow up after ~14 h. Operational path with forcing is stable.

## What v0.11.0 does NOT claim

No powered TOST PASS; no bitwise WRF parity (RMSE-equivalence is the operational bar);
no full WRF v4 physics catalog (unported schemes fail closed); no two-way nesting in a
long live forecast proof; no real multi-GPU throughput without a DGX/NVLink cluster.
The gap chain to a complete WRF replacement is inventoried in
[`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md).
