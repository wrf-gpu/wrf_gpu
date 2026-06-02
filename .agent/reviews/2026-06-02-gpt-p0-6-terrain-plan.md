# GPT P0-6 Terrain / Map-Factor / Boundary Dynamics Plan

Date: 2026-06-02
Branch: `worker/gpt/p0-6-terrain`
Scope: non-GPU source inventory, implementation sequence, and CPU oracle for the lowest-risk map-factor advection terms.

## Executive Headline

P0-6 is not one operator. The low-risk map-factor transcription in periodic flux-form advection is now implemented and proved by a CPU analytic WRF-transcription oracle. The biggest remaining forecast-quality risks are still high risk: the steep-terrain horizontal PGF / well-balanced correction, the coupled terrain surface-w boundary condition inside `advance_w`, WRF-faithful mass-coupled specified/relax/nested boundary forcing, boundary-order advection degradation near non-periodic edges, and physical-surface diffusion metric corrections.

No GPU forecast validation was run in this worktree by contract.

## Map-Factor Field Inventory

| Item | WRF anchor and formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Static map fields | `Registry.EM_COMMON:1391-1400`: `msft`, `msfu`, `msfv`, `msftx`, `msfty`, `msfux`, `msfuy`, `msfvx`, `msfvx_inv`, `msfvy`. | `src/gpuwrf/contracts/grid.py:66-71`, `:200-205`; loader `src/gpuwrf/dynamics/metrics.py:97-102`. | Present, except `msfvx_inv` is not a `DycoreMetrics` leaf; it is computed as `1/msfvx` or held in acoustic state where needed. |
| Idealized identity | WRF idealized self-test allocates map factors as 1 in `module_advect_em.F:10667-10670`. | `DycoreMetrics.flat()` sets all map factors to one at `grid.py:200-205`. | Present. This sprint added an explicit unit-map identity oracle for flux advection. |

## Operator Gap Inventory

### 1. Flux-Form Advection and `calc_ww_cp`

| Sub-operator | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Coupled horizontal mass fluxes | `module_big_step_utilities_em.F:747-748`: `divv=msftx*dnw*(rdx*((c1h*muu+c2h)*u/msfuy diff_i)+rdy*((c1h*muv+c2h)*v*msfvx_inv diff_j))`; `:767-770` onward integrates `ww(k)=ww(k-1)-dnw*c1h*dmdt-divv`. | `src/gpuwrf/dynamics/flux_advection.py:176-197`. | Implemented this sprint for periodic path. |
| Scalar flux advection | `module_advect_em.F:3387-3388`, `:4215-4355`: horizontal tendency uses `mrdx=msftx*rdx`, `mrdy=msftx*rdy`; vertical tendency has no extra horizontal map factor in this high-order branch. | `flux_advection.py:261-269`, `:271-306`. | Implemented this sprint. |
| U momentum advection | `module_advect_em.F:354`, `:479`, `:1395`: horizontal divergence uses `msfux*rdy` / `msfux*rdx`; vertical term is unscaled by horizontal map factors. | `flux_advection.py:356-370`. | Implemented this sprint. |
| V momentum advection | `module_advect_em.F:1784`, `:1897`, `:2875`: horizontal divergence uses `msfvy`; vertical term is `(msfvy/msfvx)*rdzw*(vflux(k+1)-vflux(k))`. | `flux_advection.py:386-401`. | Implemented this sprint. |
| W momentum advection | `module_advect_em.F:5220-5223`, `:5930-6028`: horizontal divergence uses `msftx`; vertical top/lid branch has no extra horizontal map factor. | `flux_advection.py:436-445`; pre-existing top-lid tests in `tests/dynamics/test_advect_w_topface.py`. | Implemented horizontal map factor this sprint; top-lid branch was already present. |
| Boundary-order degradation near edges | `module_advect_em.F:3394-3419` and analogous u/v/w blocks around `:11010`, `:11662`, `:12349`: high-order stencils degrade to lower order near specified/nested/non-periodic boundaries. | `flux_advection.py` still uses periodic `jnp.roll` stencils and documents periodic scope. | Missing. Must be a separate boundary-aware advection sprint. |

Proof object for implemented rows: `proofs/p0_6/map_factor_advection_oracle.json`.

### 2. Divergence / Continuity / Theta Coupling

| Sub-operator | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Acoustic `advance_mu_t` continuity | `module_small_step_em.F:1094-1099`: `dvdxi=msftx*msfty*(rdy*(v+mass*v_1*msfvx_inv diff_j)+rdx*(u+mass*u_1/msfuy diff_i))`; `DMDT=sum(dnw*dvdxi)`. | `src/gpuwrf/dynamics/mu_t_advance.py:124-149`. | Present. Needs real-grid savepoint beyond existing M6 parity. |
| Acoustic eta-dot / `ww` recurrence | `module_small_step_em.F:1109-1112`: `ww(kk)=ww(kk-1)-dnw*(c1h*dmdt+dvdxi+c1h*mu_tend)/msfty`; then subtract large-step `ww_1` at `:1115-1119`. | `mu_t_advance.py:161-172`. | Present. High terrain sensitivity because this feeds `advance_w`. |
| Theta update | `module_small_step_em.F:1141-1171`: add `msfty*dts*ft`; subtract `dts*msfty*(msftx*(0.5*rdy*vflux+0.5*rdx*uflux)+rdnw*vertical_flux)`. | `mu_t_advance.py:174-205`. | Present. Needs real-grid savepoint. |

### 3. Pressure Gradient Force and Coriolis / Curvature

| Sub-operator | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Large-step horizontal PGF map ratios | `module_big_step_utilities_em.F:2380-2392`: `dpx=(msfux/msfuy)*0.5*rdx*mass*(dphi+alt*dp+al*dpb) + (msfux/msfuy)*rdx*dphp*(rdnw*ddpn-0.5*c1*dmu)`; y uses `msfvy/msfvx`. | `src/gpuwrf/dynamics/core/rk_addtend_dry.py:194-238`. | Present by formula, but not accepted for P0-6 until steep-terrain WRF savepoint proves well-balanced behavior. |
| Acoustic horizontal PGF map ratios | WRF `module_small_step_em.F:800-944`: same `msfux/msfuy`, `msfvy/msfvx`; divergence damping uses `/msfuy` and `*msfvx_inv`. | `src/gpuwrf/dynamics/core/acoustic.py:402-446`. | Present. Needs real-grid acoustic savepoint. |
| Vertical PGF / buoyancy | `module_big_step_utilities_em.F:2468-2492`: `rw_tend += (1/msfty)*g*(rdn*dp - c1f*mu')`. | `src/gpuwrf/dynamics/core/advance_w.py:88-115`; staged in `runtime/operational_mode.py:859-865`; acoustic fallback at `acoustic.py:606-614`. | Present. Needs real-grid savepoint tied to `advance_w`. |
| Horizontal Coriolis / curvature | `module_big_step_utilities_em.F:3726-3729`, `:3800-3803`: u term `(msfux/msfuy)*f*rv - e*cosa*rw`; v term `-(msfvy/msfvx)*f*ru + (msfvy/msfvx)*e*sina*rw`. | `rk_addtend_dry.py:296-350`. | Present for horizontal tendencies. |
| Vertical Coriolis | `module_big_step_utilities_em.F:3839-3844`: `rw_tend += e*(cosa*ru - (msftx/msfty)*sina*rv)`. | `rk_addtend_dry.py:270-276` documents omission. | Missing / intentionally omitted. Medium risk for Canary real terrain; should be quantified before implementation. |

### 4. Terrain-Following Eta Metrics and Sloped Columns

| Sub-operator | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Vertical metric arrays | WRF `DN`, `DNW`, `RDN`, `RDNW`, `FNM`, `FNP`, `CF1/2/3` used throughout `module_small_step_em.F:1078-1171`, `:1295-1469`. | `grid.py:80-89`, signed flat construction at `grid.py:175-198`; loader at `metrics.py:111-119`. | Present. |
| Phi/omega vertical coupling | `module_small_step_em.F:1328-1355`: `ww*rdnw*dphi` staged onto full levels; `:1366-1369` applies `msfty*rhs/mass`. | `src/gpuwrf/dynamics/core/advance_w.py:247-270`. | Present by formula. Needs real-grid savepoint over sloped terrain. |
| Kinematic lower boundary for terrain-following `w` | `module_small_step_em.F:1372-1394`: `w_sfc=msfty*0.5*rdy*dht_y*v_cf + msftx*0.5*rdx*dht_x*u_cf`. | Formula exists at `advance_w.py:274-303`. Production call deliberately feeds decoupled `u_1/v_1` instead of WRF coupled work arrays at `acoustic.py:620-645`. | High-risk approximated. This is a prime P0-6 target, but it must not be changed without a WRF savepoint and stability proof. |
| Sloped-column eta-dot consistency | WRF couples `calc_ww_cp`, `advance_mu_t`, and `advance_w` through the same `ww`/eta metric recurrence. | Pieces are present in `flux_advection.py`, `mu_t_advance.py`, `advance_w.py`, but no end-to-end sloped-column savepoint exists. | Present in pieces, unproven as a coupled real-terrain operator. |

### 5. Sloped-Coordinate Horizontal PGF / Well-Balanced Correction

| Item | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Nonhydrostatic sloped-coordinate PGF correction | WRF horizontal PGF term 4 in `module_big_step_utilities_em.F:2389-2391`: `(msfux/msfuy)*rdx*(php_i-php_i-1)*(rdnw*(dpn(k+1)-dpn(k))-0.5*c1*(mu_i+mu_i-1))`, with y analog. This is the terrain-following correction that suppresses hydrostatic spurious PGF on steep topography. | `rk_addtend_dry.py:207-214` and `:228-235` implement the term using absolute diagnostics. | Present by code, high-risk/unaccepted until WRF savepoint. This is the leading suspect for residual terrain wind error and should get Opus-MAX review before any rewrite. |
| Hydrostatic-rest terrain oracle | WRF should give near-zero horizontal acceleration for a hydrostatic column over terrain, modulo discrete WRF truncation. | No committed hydrostatic-rest steep-terrain oracle. | Missing. Required before changing PGF. |

### 6. Horizontal Diffusion on Model Surfaces vs Physical Surfaces

| Item | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Terrain-metric deformation tensor | `module_diffusion_em.F:40-50`: Chen-Dudhia metric terms include `dpsi/dx`, `dpsi/dy` cross terms in `D11`, `D22`, `D12`, `D13`, `D23`. | `src/gpuwrf/dynamics/explicit_diffusion.py:18-20` states map factors are unity and scope is idealized/audit. | Missing for real terrain. |
| Physical-surface horizontal diffusion | `module_diffusion_em.F:2999-3018` passes `msftx/msfty/msfu/msfv/zx/zy/rho` into horizontal diffusion; `:3118-3155` starts u-diffusion with `msfux/msfuy`; stress calculation begins around `:5331`. | Current JAX diffusion is sixth-order filter plus flat constant-K approximations in `explicit_diffusion.py`. | Approximated. Medium/high risk after PGF and surface-w because it changes terrain wind shear and numerical damping. |

### 7. Specified / Relaxation / Nested Lateral Boundary Dynamics

| Item | WRF anchor and concrete formula | GPU anchor | Status |
| --- | --- | --- | --- |
| Specified ph update in acoustic loop | `module_bc_em.F:17-90`: `spec_bdyupdate_ph` updates outer spec-zone ph using old/new mass and `field_tend`. | In-loop support exists in `src/gpuwrf/dynamics/core/acoustic.py:693-714`; toggled by boundary configuration. | Mechanism present but high-risk/off by default for nested replay. |
| Dry specified and relaxation tendencies | `module_bc_em.F:161-235` (`relax_bdy_dry`), `:413-455` (`spec_bdy_dry`) operate on mass-coupled `ru/rv/ph/t/w/mu`; `:1297-1334` gives `fcx/gcx` taper. | `src/gpuwrf/coupling/boundary_apply.py:1-44` explicitly documents decoupled replay approximation; config and toggles at `:71-145`; application at `:148-232`. | Approximated for side-history replay; not WRF bit-faithful for mass-coupled nested boundaries. |
| Boundary normal momentum inside acoustic loop | WRF `advance_uv` loop bounds and `relax_bdy_dry` freeze/relax normal work arrays. | `acoustic.py:448-463` applies normal work boundary when staged. | Partial / targeted fix only. Needs WRF savepoint for full spec+relax cadence. |
| Nested parent forcing / child re-sync | WRF live nesting re-syncs child fields through parent interpolation and boundary tendencies, not hourly decoupled wrfout replay alone. | `boundary_apply.py:109-140` documents ph/w nested forcing toggles default off after d03 warming. | High-risk missing architecture piece. Do not implement blindly inside P0-6 without manager/Opus-MAX review. |

## Implementation Sequence

1. **Done in this sprint: periodic flux-advection map factors.** Implement `msfuy`, `msfvx_inv`, `msftx`, `msfux`, `msfvy`, and `msfvy/msfvx` in `couple_velocities_periodic` and `advect_*_flux`. Gate with CPU analytic WRF transcription and unit-map identity.
2. **Boundary-aware advection order degradation.** Add a boundary metadata path to `flux_advection.py` without touching nested forcing. Build analytic WRF stencils for scalar/u/v/w high-order-to-low-order degradation near specified/nested edges. This is lower risk than PGF because it is stencil selection, not force balance.
3. **Boundary mass-coupled savepoints.** Build WRF savepoints for `spec_bdy_dry`, `relax_bdy_dry`, `spec_bdyupdate_ph`, and `lbc_fcx_gcx`. Decide whether the operational side-history replay can ever be WRF-faithful without live parent re-sync. No broad nested forcing change until this answer is clear.
4. **Terrain eta-dot / surface-w savepoints.** Create a small steep-terrain WRF fixture around `advance_mu_t` and `advance_w`, including the surface chain-rule `w_sfc` formula. Prove whether the current decoupled `u_1/v_1` surface-w feed can be corrected with the WRF coupled work arrays without reintroducing the documented k0 mode.
5. **Sloped-coordinate horizontal PGF / well-balanced oracle.** Before rewriting PGF, savepoint WRF `horizontal_pressure_gradient` on a hydrostatic steep-terrain Canary cutout and on an analytic hydrostatic column. This is the highest-value and highest-risk wind lever; route to Opus-MAX after the oracle is in place.
6. **Physical-surface diffusion metric correction.** Port the WRF deformation/metric diffusion only after PGF/surface-w are stable, because diffusion can hide or amplify force-balance errors. Use WRF `module_diffusion_em` savepoints plus analytic linear/quadratic fields.
7. **Live nested-boundary architecture decision.** If d03 errors remain dominated by parent/child consistency, move from decoupled hourly side-history replay toward a WRF-like parent re-sync or a documented alternative ADR.

## Savepoint / Oracle Design By Operator

| Operator | Oracle design | Acceptance proof |
| --- | --- | --- |
| Map-factor flux advection | Deterministic CPU analytic fixture with non-unit map factors. Independent NumPy WRF formulas compare against JAX for `ru`, `rv`, `rom`, scalar/u/v/w tendencies; separate unit-map identity path. | Committed `proofs/p0_6/map_factor_advection_oracle.json`; max abs diff <= `1e-11` for every row. |
| Boundary-order degradation | Small CPU WRF savepoint or analytic extraction of `module_advect_em` stencil branch at a specified boundary. Include scalar/u/v/w and all four sides. | Per-face flux parity plus tendency parity at boundary rows; identity to current path when periodic. |
| `advance_mu_t` eta-dot | WRF savepoint around `advance_mu_t` with nonuniform `dn/dnw`, non-unit map factors, and sloped `ww_1`. | `mu`, `mudf`, `muts`, `muave`, `ww`, theta parity; column dry-mass conservation check. |
| Surface-w terrain BC | Analytic chain-rule field and WRF `advance_w` savepoint over a steep synthetic island ridge. Test both coupled-work and decoupled-stage wind feeds. | Surface `w(1)` parity, finite implicit solve, no k0-only growth in a short CPU repeated-step harness. |
| Sloped horizontal PGF | WRF savepoint around `horizontal_pressure_gradient` on hydrostatic rest over steep terrain and a real Canary cutout. | `ru_tend`, `rv_tend` parity; hydrostatic-rest spurious acceleration bounded by predeclared WRF truncation tolerance. |
| Vertical PGF / buoyancy | WRF `pg_buoy_w` savepoint with non-unit `msfty`, nonuniform eta, and nonzero `cqw` dry/moist variants if needed. | `rw_tend` parity at all w faces, including top lid/open top. |
| Coriolis / curvature | WRF `coriolis` savepoint with nonzero `f`, `e`, `sina`, `cosa`, non-unit map factors. | Horizontal tendency parity; decide whether vertical `rw_tend` term is material for Canary winds before implementing. |
| Physical-surface diffusion | WRF `module_diffusion_em` savepoints for `horizontal_diffusion_{u,v,w,s}` and stress tensors on flat and sloped analytic fields. | Tendencies and stress components parity; no regression of idealized diffusion gates. |
| Spec/relax/nested boundaries | WRF `module_bc_em` savepoints for dry spec, relax, ph update, weights. Include mass-coupled variables and decoupled wrfout replay comparison as a separate diagnostic. | Boundary-ring tendency parity; explicit decision on live nesting vs side-history approximation. |

## Risk Ranking

1. **Critical / highest:** sloped-coordinate horizontal PGF well-balanced correction. Biggest wind-quality upside and easiest place to create a terrain-force imbalance.
2. **Critical:** terrain surface-w lower boundary inside `advance_w`. Current production path is knowingly not WRF-coupled at `acoustic.py:620-645`; changing it without a savepoint can revive the documented surface k0 instability.
3. **High:** specified/relax/nested boundary forcing, especially ph/w in the acoustic loop and parent/child re-sync. Current replay is a documented decoupled approximation.
4. **Medium-high:** physical-surface diffusion metric correction. Important for terrain wind shear, but should follow PGF/surface-w so diffusion does not mask a force-balance bug.
5. **Medium:** boundary-order advection degradation. Mechanically clear but touches edge semantics and all staggered fields.
6. **Medium-low:** vertical Coriolis term in `rw_tend`. WRF has it; current JAX omits it. Quantify with savepoint before implementation.
7. **Low / completed:** periodic map-factor flux-form advection terms implemented here, with analytic CPU proof and unit-map identity.

## This Sprint's Proof

Command:

```bash
OMP_NUM_THREADS=4 JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu taskset -c 0-3 python proofs/p0_6/map_factor_advection_oracle.py
```

Result: PASS. The command emitted XLA CPU cache host-feature warnings but completed on CPU and wrote `proofs/p0_6/map_factor_advection_oracle.json`.
