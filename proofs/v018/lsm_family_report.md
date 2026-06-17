# v0.18 LSM Family — Ship-Gate Report

**Branch** `worker/opus/v018-lsm` (off `worker/opus/v018-trunk`) · **Frontrunner** Opus 4.8 (max) · 2026-06-16

## Objective
Bring **every** WRF v4 `sf_surface_physics` option to the 0.18 bar — EXACTLY ONE of:
(a) operational + REAL pristine-WRF oracle GREEN; (b) reference-only WITH a real
pristine-WRF oracle wired fail-closed; (c) documented architecture-boundary
fail-closed (real test) / proven-irrelevant. No silent gaps, no fake greens, no
reference-only-without-a-real-oracle, no tolerance widening.

## Result — full WRF v4 enumeration {0,1,2,3,4,5,6,7,8}, all at bar

| sf | scheme | class | oracle / proof | worst err | ship-gate |
|---:|--------|-------|----------------|-----------|:--------:|
| 0 | none | operational | — | — | ✅ |
| 1 | 5-layer thermal-diffusion SLAB | **operational (a)** | fp64 SLAB1D `proofs/v013/.../slab_case_*` | 2.33e-10 / 5e-9 | ✅ |
| 2 | Noah classic | **operational (a)** | real SFLX `tests/v060/test_noahclassic_parity` | within tol | ✅ |
| 3 | RUC multi-layer soil/snow | **reference-only + real oracle (b)** | fp64 LSMRUC `proofs/v017/.../ruclsm` (sha256 = pristine) | n/a | ✅ |
| 4 | Noah-MP | **operational (a)** | real NOAHMP_SFLX energy gate (11 Canary cols) | within tol | ✅ |
| 5 | CLM4 | **documented-boundary-fail-closed (c)** | `tests/test_v018_lsm_architecture_boundary.py` | n/a | ✅ |
| 6 | CTSM | **documented-boundary-fail-closed (c)** | same boundary test | n/a | ✅ |
| 7 | Pleim-Xiu 2-layer ISBA | **operational (a)** | fp64 SURFPX+QFLUX `proofs/v017/.../pxlsm` | 4.37e-11 / 1e-9 | ✅ |
| 8 | SSiB biophysical | **reference-only + real oracle (b)** | fp64 SSIB `proofs/v017/.../ssib` (sha256 = pristine) | n/a | ✅ |

**Family `full_ship_gate = true`.**

## What was done

### Verified already-at-bar (sf=1,2,3,4,7,8)
A verification pass confirmed: operational LSMs (1/2/4/7) are fp64/real-oracle
GREEN; reference-only LSMs (3 RUC, 8 SSiB) have a **real** fp64 pristine-WRF
single-column oracle whose recorded source checksum **sha256-matches the current
pristine** `module_sf_ruclsm.F` / `module_sf_ssib.F`, and fail-closed is enforced
at three layers (kernel raises, operational scan rejects with a named reason,
catalog `REFERENCE_ONLY`). RUC/SSiB faithful JAX ports stay intractable in-session
(~7.5k / ~6.6k LOC coupled solvers) → (b) is the correct bar, met.

### CLM4 (sf=5) + CTSM (sf=6) — documented v1.0 architecture boundary (manager-ruled)
These were **silent gaps** (absent from the catalog/accept-matrix). Now wired as a
distinct class **`documented-boundary-fail-closed`**: catalog-RECOGNIZED but
raising a clear, specific architecture-boundary error (never a happy-path stub).
- `scheme_catalog._PHYSICS_FAIL_CLOSED_REASON` + `classify_scheme` → CLM4/CTSM
  RECOGNIZED_FAIL_CLOSED with the cited reason (CLM4 ~61.5k LOC global `clmtype` +
  external surface-dataset; CTSM `-DWRF_USE_CTSM` external CESM/LILAC, empty state).
- `namelist_check` rejection names CLM4(5)/CTSM(6) + the v1.0 CAM/CLM/CTSM ADR.
- **Proof object:** `tests/test_v018_lsm_architecture_boundary.py` (9 tests) —
  catalog-recognized, namelist errors cleanly, dispatcher fail-closed, distinct
  from reference-only, catalog-consistent. Per the manager: the boundary is closed
  **only while this test is green**. No multi-session CLM oracle attempted.

### "LSM L2 statics not bounded" handoff (sf=1 slab + sf=7 Pleim-Xiu) — CLOSED
The generic coupled L2 sweep could not run slab/PX by family/option because the
real-case wrfinput lacks `THC/EMISS` (slab) and the PX ISBA constants. Built a
**faithful real-case extraction** `src/gpuwrf/io/lsm_static_extract.py`:
- **Slab**: `THC = THERIN[LU_INDEX,season]/100`, `EMISS = SFEM` (WRF
  `module_physics_init.F:1967`/`:1970` landuse_init); fixed WRF **5-layer**
  thermal-diffusion geometry.
- **PX**: the 11 Noilhan-Mahfouf ISBA constants via a port of WRF `SOILPROP`
  (`module_sf_pxlsm.F:1904-1915`) over the SOILCBOT 16-category fraction-weighted
  (clay,sand); `ds1/ds2 = 0.01/0.99 m`; RSTMIN/EMISS/ZNT by LU_INDEX.
- **Falsifiable:** the SOILPROP port **bit-matches the pxlsm fp64 oracle to machine
  epsilon** (worst rtol ~1.9e-16 across all 11 constants) — not a happy-path.
- Wired into `proofs/v016/coupled_coverage_gate.py --family lsm --option 1|7`.

Closing the handoff put the slab/PX operational coupling under a **real coupled
forecast for the first time** (the bundle was always absent → fail-closed before,
so the v0.17 "operational" claim rested on the fp64 single-column oracle + the
scan-wiring, NEVER a coupled run). That exposed **three real pre-existing latent
bugs** this worker root-caused and fixed:

1. **Held-radiation None-unpack crash** — `operational_mode.py` slab/PX passed
   `held_rad=None` to `_refresh_noahmp_rad` and unpacked a 3-tuple → crash on every
   non-radiation-cadence step (`jax.lax.cond` needs both branches to return the same
   structure). Fixed to pass a concrete `(gsw, glw, cosz)` 3-tuple from
   `carry.slab_rad`/`carry.px_rad`, mirroring the Noah-classic seam.
2. **Wrong soil geometry** — the extraction inherited the wrfinput **Noah 4-layer**
   ZS/DZS; the slab is a fixed **5-layer** thermal-diffusion model, PX a 2-layer
   ISBA (ds1/ds2 = 0.01/0.99 m). Corrected to the WRF scheme geometry, regression-
   tested.
3. **Slab FLHC/FLQC reconstruction blow-up (the deep one)** — with 1+2 fixed, the
   slab forecast ran a full hour but **blew up** (TSK 287 K→738 K→NaN in ~6 min →
   `w` 7.5e15, baseline stable → slab-specific). Root cause: the operational-only
   `slab_surface_hook._flhc_flqc_from_handles` reconstructs `FLHC = HFX/(THG−THX)`,
   which is **ill-posed near neutral** (ΔΘ→0 ⇒ FLHC→∞); WRF SLAB1D then re-evaluates
   `HFX = FLHC·(THG−THX)` as the skin temperature evolves (`module_sf_slab.F:443`),
   so a huge-but-finite FLHC becomes a runaway. The fp64 KERNEL is oracle-green
   because the oracle feeds FLHC **directly**, bypassing this reconstruction — which
   was therefore never validated. **WRF-faithful fix** (NOT an ad-hoc clamp): WRF's
   own surface layer prevents exactly this by flooring the heat resistance
   `PSIT = AMAX1(GZ1OZ0−PSIH, 2.0)` (`module_sf_sfclay.F:706`, in-source comment
   *"LOWER LIMIT ADDED TO PREVENT LARGE FLHC IN SOIL MODEL"*). The hook now caps the
   reconstructed **FLHC** at WRF's identical, EXACT ceiling
   `FLHC ≤ CPM·RHOX·UST·KARMAN/(2·PRT)` (that PSIT≥2 floor is applied to every
   land+water point), cited line-by-line. The **FLQC** moisture coefficient is
   capped at WRF's *water-branch* `PSIQ≥2` ceiling (`:731`) — labeled honestly as a
   **conservative guard**, since WRF does NOT floor *land* `PSIQ` (`:763`); it is
   the loosest moisture exchange WRF emits, so it bounds the equally-ill-posed
   `FLQC=QFX/(QSG−QX)` reconstruction without claiming to be WRF's exact land
   maximum. PX (sf=7) was verified to have **no** such defect (it reconstructs a
   clipped `RMOL`, not FLHC, and computes HFX via a resistance form `RAH = RA +
   5/UST` with no 1/ΔΘ division) — left untouched.

4. **Noah-MP (sf=4) generic-gate seeding gap (GPT-critic find)** — the production
   daily/nested pipelines seed the prognostic `noahmp_land` + held `noahmp_rad` into
   the carry via `carry.replace` AFTER `_initial_carry_for_run`, but the generic
   single-domain operational path (`run_forecast_operational` → the coverage gate)
   has no post-replace seam, so `carry.noahmp_rad=None` → `_NoahMPRadiation(*None)`
   crash. Fixed by an append-only `noahmp_land` namelist field (registered in the
   `OperationalNamelist` pytree flatten/unflatten) that `_initial_carry_for_run`
   seeds directly when supplied (left None, the pipelines' post-replace still owns
   it — non-breaking); the coverage gate builds the full Noah-MP bundle
   (`build_noahmp_land_state` + `build_noahmp_params`). Noah-MP was already
   operationally validated via the production path + the real NOAHMP_SFLX energy
   oracle; this closes the GENERIC single-domain coupled gate too.

> **Note (per manager):** the FLHC blow-up (bug 3) was the **2nd** pre-existing
> latent *coupling* bug a v0.18 family worker surfaced (after the RA `RTHRATEN`-NaN);
> the GPT critic's exhaustive coupled sweep then surfaced bug 4 — concrete evidence
> that exhaustive coupled testing (not just per-scheme oracles) is what catches the
> bugs that only appear when a scheme is actually wired into the forecast.

> **GPT-critic FIX-then-ACCEPT (`proofs/v018/lsm_critic_gpt.md`) resolved:** the
> critic confirmed the FLHC fix WRF-faithful and raised two bounded must-fixes —
> (1) the sf=4 gate crash (bug 4 above, now PASS) and (2) the FLQC cap overclaiming
> the same `PSIT` floor (corrected in bug 3 to the honest water-branch `PSIQ` guard).
> Both are now green for the critic's re-verify.

**Coupled-run result (authoritative, GPU lock):** the generic coverage L2 sweep
`--family lsm --option 1`, `--option 4`, and `--option 7` all **PASS** —
`all_finite=True`, zero bounds violations, no hard-gate fails, dynamics RMSE 16 % /
23 % / 18 % of the frozen v0.14 tolerance band respectively (the small perturbation
a faithful LSM swap should produce, not a damped hack). The slab fp64 oracle stays
bit-faithful (worst |jax−oracle| 2.328e-10).

## Tests / gates (fp64)
- `tests/test_v018_lsm_architecture_boundary.py` — 9 passed (CLM4/CTSM boundary proof).
- `tests/test_v018_lsm_static_extract.py` — 10 passed (SOILPROP-vs-oracle bit-match,
  real-case extraction, scan acceptance, slab-5-layer + PX-ds geometry contract).
- `tests/test_v013_t3_surface_lsm_wiring.py` — 10 passed; `tests/test_v017_lsm_pleim_xiu.py`
  — 2 passed (slab/PX fp64 oracle still GREEN: 2.328e-10 / 4.37e-11).
- `tests/test_scheme_catalog_fail_closed.py` — 21 passed; `test_v013_operational_smoke.py`
  — 48 passed (no regression).
- **Coupled L2 coverage gate (GPU lock):** `--family lsm --option 1` → **PASS**;
  `--option 4` → **PASS**; `--option 7` → **PASS**
  (`proofs/v016/coverage/lsm{1,4,7}_gate.json`).

## Files changed
- NEW `src/gpuwrf/io/lsm_static_extract.py` — faithful slab/PX real-case extraction.
- NEW `tests/test_v018_lsm_architecture_boundary.py`, `tests/test_v018_lsm_static_extract.py`.
- `src/gpuwrf/io/scheme_catalog.py` — `_PHYSICS_FAIL_CLOSED_REASON` + classify wiring (CLM4/CTSM).
- `src/gpuwrf/io/namelist_check.py` — sf=5/6 architecture-boundary message.
- `src/gpuwrf/runtime/operational_mode.py` — slab/PX held-radiation 3-tuple fix (bug 1) +
  `noahmp_land` namelist field + `_initial_carry_for_run` Noah-MP seeding + pytree
  registration (bug 4).
- `src/gpuwrf/coupling/slab_surface_hook.py` — WRF FLHC PSIT-floor ceiling (bug 3) +
  honest water-branch PSIQ guard for FLQC (critic must-fix 2).
- `proofs/v016/coupled_coverage_gate.py` — wire LSM static/Noah-MP bundle for sf=1/4/7.

## Unresolved risks / notes
- All sf=1/7 coupled-run gates now **PASS** under the GPU lock (above). Tier-A operational LSMs (1/2/4/7)
  GPU smoke passed on their source branches (re-run confirmatory).
- Noah-MP energy gate defaults `WRF_PRISTINE_ROOT` to a sibling path that does not
  exist in this worktree layout; it passes with the documented override. A one-line
  default-path robustness fix in `proofs/noahmp/*_savepoint_gate.py` is a candidate
  (not an LSM-kernel defect).
