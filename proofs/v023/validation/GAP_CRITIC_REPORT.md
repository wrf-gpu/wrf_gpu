# v0.23.0 Pre-Release Gap-Critic Report

**Reviewer:** adversarial pre-release gap-critic (CPU-only, no GPU touched)
**Branch/HEAD:** `worker/opus/v023-release` @ `b2ca242e`
**Base:** v0.22.2 `53bd20bf` (tag v0.22.2, wrfgpu `7fb48b7c`)
**Date:** 2026-07-04

## VERDICT: RELEASE-SAFE — no release-blocking findings

Default byte-identity, fail-closed reference-only gating, default-inertness of all
opt-in features, and claim honesty all hold up under adversarial scrutiny. Three
minor evidence-hygiene notes below (none block the tag).

---

## Adversarial checks — verified clean

### 1. Default byte-identity is SOUND (and better-proven than the cited A1 self-control)
The manager's headline evidence is the A1 *fresh-cache* self-control anchoring. That
is the **weakest** leg. The byte-identity conclusion is independently and more
decisively established by a chain I re-verified:

- **NET dynamics diff base→HEAD = 0 lines** (`git diff 53bd20bf..HEAD -- src/gpuwrf/dynamics/`).
  P1/P2/P4 (single-scan / safe floors / calc_coef_w vectorize) were **reverted** to
  preserve identity (confirmed by the `ts_proof_..._revert_p1`, `..._p2numeric_revert`,
  `..._p2p4_revert` proof dirs). The default acoustic/dycore is byte-for-byte v0.22.2.
- **GATE-2 same-cache control = BIT-IDENTICAL.** `proofs/v023/gate2_fix/FIXED2_REFCACHE...`
  and `proofs/v023/f1_loaderfix/DEFAULT_FINAL_REFCACHE_FIELD_COMPARE.json`:
  `exact=true, max_abs_global=0.0, 0/178 leaves`. When the AOT cache is held fixed
  (removing autotune nondeterminism), v0.23-default == v0.22.2 exactly. The
  fresh-cache `1.2e-8` residual is reproduced by **v0.22.2 vs its own second cold
  compile** (`FIXED3_VS_V0222_SAMECACHE...` = 0.0) → pure XLA autotune, not code.
- **P0 (only default-numerics HLO change #1):** `P0_DEFAULT_FULL_FIELD_COMPARE.json`
  — all 17 M9 writer fields `before_sha256 == after_sha256` (byte-identical to base).
  The `_m9_flux_slices_only=True` path is a diagnostic-only (side-channel) work cut
  feeding the identical `RRTMGRadiationDiagnostics` slices; it does not touch timestep
  physics/state evolution.
- **P3 (only default-numerics HLO change #2):** re-ran `p3_root_fusion_cpu_proof.py` on
  CPU myself → fused vs eager carry **sha256 identical** for d01/d02, identical events
  and own_steps. The net `domain_tree.py` diff shows P3 only fuses the *exact flat
  2-domain* case; the 3-dom canary (and all deeper trees) still take the eager path
  (`is_root and (len(order)!=2 or children!=1) → None`), i.e. unchanged.
- Adversarial "hidden large-diff field?" — no field is orders-of-magnitude above the
  self-control envelope: A1 worst field PBLH is the same ~13.8 m magnitude in both
  v0.23-vs-base and base-vs-base; `fields_orders_of_magnitude_above_selfcontrol = 0`.

### 2. Reference-only / fail-closed schemes cannot silently execute — SOUND
`_resolve_operational_suite()` (runtime/operational_mode.py:3939) is called at the top
of **every** forecast entry (operational_mode.py 5507/5593/5646/5735/5792/5826/5885,
nested_pipeline.py:1041, domain_tree.py:2095) and `raise UnsupportedSchemeSelection`
if any selected scheme is not in `_SCAN_WIRED_OPTIONS`. Verified each target scheme is
NOT wired and carries a named reason:
- `mp=18` (NSSL), `mp=40` (Morrison-aero): not in wired mp set → raise.
- `bl=9` (CAM-UW): not in wired bl set → raise. **Defense-in-depth:** the old RED
  CAM-UW adapter was gutted to `raise CamUwReferenceOnlyError` AND removed from
  `PBL_SCAN_ADAPTERS` (scan_adapters.py) — it cannot be dispatched even if the resolve
  guard were bypassed.
- `sf_urban=2/3`, `sf_lake=1`: explicit `!=0` guards append named reasons → raise.
- `cu=4/5/14/93/94/95/96/99` (SAS/Grell/KSAS), `sf_surface=3/8` (RUC/SSiB), ra tail
  3/5/7/99: all not wired → raise.
- `cu=16` (New-Tiedtke) correctly **graduated to IMPLEMENTED** (in wired set, correctly
  ABSENT from `_SCAN_UNWIRED_REASON`), and additionally fail-closes if
  `use_flux_advection`/`moist_adv_opt` are off (operational_mode.py:3967).

Tests confirm on the integrated tree: `test_v022_f2_missing_scheme_bundle`,
`test_v022_g3city_urban_lake`, `test_v022_camuw_pbl`, `test_v013_sfclay_pbl_pairing`,
`test_v060_physics_interfaces`, `test_v017_lsm_adv`, `test_scheme_catalog_fail_closed`,
`test_v018_mp_family_fail_closed` → **132 passed, 2 skipped**;
`test_v023_moving_nest_operational`, `test_namelist_check`, `test_ntiedtke_jax_parity`
→ **84 passed**. `assert_catalog_consistent()` passes (exercised by the G3 test).

### 3. Default-inertness of opt-in features — SOUND
- **F1 batch:** `_batch_ensemble_size_from_env()` returns 1 when unset (default). The
  run path branches `if batch_size > 1:` into the vmap loader, else a clean
  `_load_domains(...)` passthrough (nested_pipeline.py:2091/2103). All batch helpers
  guard `if int(batch_size) <= 1:`. The F1 re-enable (`8776e545`) preserved the default
  fingerprint (DEFAULT_FINAL_REFCACHE = 0.0 was captured after it).
- **Moving nests (G2):** `moving_driver`/`adaptive_driver` are NOT wired into the default
  `domain_tree`/`nested_pipeline` path; the only domain_tree change is a `dynamic_edges`
  registry that is empty (inert) unless a `move` callback returns an edge (opt-in).
- **P1 single-scan:** default-off (reverted; net dynamics diff 0). CHANGELOG labels it
  "default-inert".

### 4. Honesty of reference-only / claim strings — SOUND
- CAM-UW fix (`c8d648a3`) *strengthened* the disclaimer: the oracle string now explicitly
  says "not a pristine-WRF savepoint parity gate … savepoint parity is not claimed."
  The `_REDERIVING_PBLS` removal of pbl=9 is a correct status change (pbl=9 is now
  fail-closed, so the operational revised-MM5 pairing guard no longer applies) — not
  test-gaming; keeping it would falsely assert `gpu_gate_ready=True` for a fail-closed
  scheme.
- `mp=40` "machine-precision fp64 JAX kernel" claim is backed by
  `tests/savepoint/test_morrison_aero_parity.py`, which compares against gold savepoints
  from the **unmodified** `module_mp_morr_two_moment_aero.F` (aercu_opt=2) — a real
  oracle, NOT a JAX-vs-JAX self-compare (skips honestly if savepoints absent).
- `mp=18` honestly labeled "faithful traceable JAX kernel is not yet ported".
- CHANGELOG is honest: default-numerics statement carries the autotune-floor +
  self-control caveat; P-bundle perf is explicitly listed under "Deferred … GPU perf
  confirmations"; P1/P2/P4 labeled "default-inert" (accurate — they net to zero).

### 5. Integration merge seams — SOUND
`scheme_catalog._REFERENCE_ONLY` union carries mp18/40, urban2/3, lake1, cu SAS-family,
sf3/8, ra-tail, bl9/4/10/16/17; `_SCAN_UNWIRED_REASON` union matches; cu16 correctly
present in `_SCAN_WIRED_OPTIONS` and absent from unwired-reasons; `scan_adapters` diff is
purely additive for cu16 (`ntiedtke_adapter`) with no change to existing default adapters
(mynn/thompson/etc.). `assert_catalog_consistent()` + namelist_check tests pass.

---

## Minor / non-blocking notes (for the manager)

- **N1 (evidence pointer).** The A1 *ANCHORED verdict* JSON cites a base-vs-base
  self-control run whose underlying per-field compare artifact I could not locate on
  disk (only the hand-authored verdict). This is cosmetic: the byte-identity conclusion
  is more decisively carried by the GATE-2 **same-cache 0.0** proofs
  (`FIXED2_REFCACHE...`, `DEFAULT_FINAL_REFCACHE...`, `FIXED3_VS_V0222_SAMECACHE...`) +
  the net-zero dynamics diff + P0/P3 CPU proofs. Recommend the release notes cite those
  same-cache 0.0 proofs as the primary byte-identity evidence, with the fresh-cache
  self-control as corroboration only.
- **N2 (final-HEAD same-cache coverage).** The same-cache 0.0 proofs (Jul 2) predate the
  P0/P3/G2 integration commits (Jul 3–4). The gap is covered transitively (P0 CPU 17/17
  byte-identical, P3 CPU sha256 + 3-dom eager, G2 inert, and A1@HEAD fresh == autotune
  floor). A single same-cache GPU canary at HEAD would fully close it (needs GPU — out
  of scope for this CPU review).
- **N3 (env).** The P0 CPU proof could not be re-run in this ephemeral worktree: it needs
  the WRF Fortran source tree (`.../phys/module_ra_rrtmg_lw.F`) to parse RRTMG chi_mls
  tables, which is absent here (the #115 GPUWRF_WRF_ROOT issue). This is an environment
  dependency shared with v0.22.2, not a v0.23 regression; the committed P0 artifact
  stands. The `a4_feature_branch_inert` gate shows `BAD_FEATURE_REF` for the same reason
  (it expects separate pre-integration feature branches that no longer exist post-merge)
  — inertness is instead proven by the integrated A1 + fail-closed tests.
