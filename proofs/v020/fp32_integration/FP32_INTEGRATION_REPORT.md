# v0.20.0 fp32 INTEGRATION — merge fp32 onto the lowhang branch + harden

Worker: fp32 integration (the v0.20.0 headline milestone). Branch
`worker/integration/v020-fp32` in worktree `<USER_HOME>/src/wrf_gpu2_wt/v020-fp32-int`.

Objective: integrate the verified perturbation-authoritative mixed-fp32 mode
(`worker/codex/v020-s4` @ `f9d3fde7`) onto the low-hanging-fruit speed branch
(`c696010d`), ADDITIVELY / opt-in, keeping `fp64_default` BIT-IDENTICAL, plus the
two S4 path-forward hardening fixes. fp32 ADDITIVE/opt-in; `fp64_default` stays
the runnable default; `main` untouched.

---

## 1. MERGE RECORD

- Base (HEAD of branch): `c696010d` — FINAL lowhang tip (cuda_async + host-RAM
  guard + async-output levers, COMBINED_DONE, 12h VRAM-flat PASS).
- Merged: `worker/codex/v020-s4` @ `f9d3fde7` (mode
  `GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020` + aggressive ADR-007
  fp32-gated matrix; two-model-verified).
- Merge commit: `0c77c031`. Hardening commit on top: `b836a91b`.
- Lineage verified: both `c696010d` (lowhang) AND `f9d3fde7` (fp32) are ancestors
  of HEAD `b836a91b` (`git merge-base --is-ancestor` PASS for both).

### Conflict surface — ONE file, AUTO-RESOLVED cleanly

The pre-analysis predicted one overlapping src file: `nested_pipeline.py`. Computed
overlap (lowhang src changes ∩ fp32 src changes) = **exactly**
`src/gpuwrf/integration/nested_pipeline.py` — every other fp32 file
(operational_mode.py, operational_state.py, rrtmg_lw.py, `gpuwrf/_x64_config.py`,
the per-physics x64-config decoupling across ~80 files, tests/...) is disjoint from
the lowhang files (cli.py, domain_tree.py) → clean merges.

`nested_pipeline.py`: **git auto-merged with NO conflict markers** because the two
sides touched non-adjacent regions:
- fp32 side = +4 lines inside `_make_namelist`'s `OperationalNamelist(...)` call:
  `acoustic_precision_mode=os.environ.get("GPUWRF_ACOUSTIC_PRECISION_MODE", "fp64_default")`
  (now at merged line 234-237, right after `gwdo_statics=gwdo_statics,`).
- lowhang side = the forecast-loop restructure (async output + streaming events
  fold + `try/finally` + `async_writer.join()` drain) — a different region of the
  file.

Post-merge verification of `nested_pipeline.py`:
- fp32 hook present: `acoustic_precision_mode=os.environ.get(...)` at line 237.
- lowhang structure preserved: `async_writer` plumbing (lines 650/663/714/730/929/
  937), `async_writer.join()` (line 1023) inside the `try/finally` drain (line 1024).
- `os` already imported (line 36); no new import needed.
- `gpuwrf/_x64_config.py` kept (fp32/audit envs leave x64 OFF; fp64_default keeps
  x64 ON).

No manual conflict editing was required; the auto-merge result is exactly the
intended "keep lowhang structure + graft the fp32 +4-line hook" outcome, confirmed
by inspection.

---

## 2. HARDENING (S4_PATHFORWARD_ANALYSIS.md §"Where it still loses")

All three fixes are ZERO resident-VRAM (transient-only / dormant-in-fp64) and are
no-ops for `fp64_default`. Commit `b836a91b`.

### (a.1) `rhs_ph_wrf` — `ph_tend` accumulator forced fp64
`src/gpuwrf/dynamics/core/rhs_ph.py`. The `ph_tend` transient
(`jnp.zeros_like(ph)` + four staged `.add` contributions) is a large-cancellation
sum. In perturbation-authoritative fp32 it was summed in fp32 and only widened to
fp64 at the call site (`operational_mode.py:1896` `ph_tend=...astype(float64)`) —
AFTER the precision was already lost. Fix: at the operator boundary, push the
accumulation-driving inputs through `force_fp64_island(u, v, ww, ph, phb, w, mut,
muu, muv)` so the whole tendency sum runs in fp64. For fp64_default these inputs are
already fp64 → `force_fp64_island` returns them UNCHANGED (Python identity, no
`convert` HLO) → bit-identical. Added import `from gpuwrf.contracts.precision
import force_fp64_island`.

### (a.2) `small_step_prep` — coupled WORK primes forced fp64
`src/gpuwrf/dynamics/core/small_step_prep.py` (lines ~273-277). The five coupled
WORK primes (`u_work/v_work/theta_work/w_work/ph_work`) are mass-weighted
`reference - current` differences; reference≈current at a fresh RK stage → a
large-cancellation subtraction that loses fp32 precision before the prime reaches
the (fp64) acoustic substep scan. Fix: push the field operands (reference/state
u/v/w, theta_ref/cur, ph_perturbation) through `force_fp64_island` (mass factors
c1h/c2h/mu are already fp64) so each subtraction runs in fp64. fp64_default →
no-op (already-fp64 identity). Added the same precision import.

### (b) Per-field 1km finiteness gate for fp32 moisture/number species
`src/gpuwrf/runtime/operational_mode.py`, inside
`_apply_mixed_perturb_fp32_storage` (the per-step requantization called every
finished stage in mixed mode, line ~2037). Mirrors the qke promote intent
(precision.py:233-257) but as a LOCAL RUNTIME guard rather than a static FP64
promotion: each fp32-stored species (`qv qc qr qi qs qg qh`, `Ni Nr Ns Ng Nc Nn
Nh`, `qvolg qvolh nwfa nifa`) is gated `jnp.where(isfinite, value, 0.0)` so a single
non-finite fp32 cell is replaced by a finite zero floor instead of poisoning the
run. Optional species (None) are skipped; non-fp32 (statically promoted) species are
left untouched. Reached ONLY in the opt-in mixed_perturb_fp32 mode → fp64_default
byte-unchanged. Storage dtype preserved → no resident-VRAM change.

---

## 3. CPU RE-GATE — fp64_default BIT-IDENTICAL (hard gate) — PASS

Method (lesson #105): FULL-SWEEP per-test diff against a fresh lowhang-baseline
checkout (`/tmp/v020_lowhang_baseline` @ `c696010d`), NOT a partial run. CPU,
`taskset -c 0-3`, `JAX_ENABLE_X64=true JAX_PLATFORMS=cpu`.

### 3.1 fp64_default per-test result set — byte-identical to baseline
Gate union = 17 test files (lowhang levers + precision matrix + rhs_ph real-case +
state extension + fp32 suite + hardening-relevant): **97 tests** (a SUPERSET of the
78-test lowhang gate). Excluding the fp32-only file (not on the baseline), the
shared subset = **94 collected / 92 result lines**.
- Baseline (`c696010d`): **88 passed, 6 skipped, 0 failed**.
- HEAD (`b836a91b`, merge + all hardening): **88 passed, 6 skipped, 0 failed**.
- Per-test `-rA` PASS/SKIP result-set diff (sorted): **NO_DIFF — byte-for-byte
  identical result sets** (92/92 lines match, including
  `test_fused_cascade_is_scheduler_and_value_identical_to_eager_all7`, the async ==
  sync byte-identity tests, the streaming-fold count tests, and
  `test_m6_precision_matrix` / `test_v014_rhs_ph_real_case`).

### 3.2 fp32 suite — green
`tests/test_v020_s4_mixed_precision.py`: **3 passed** (downcasts only authorized
acoustic carry; stays mixed after one hot step; global-fp32-audit mode does not
force x64 on import).

### 3.3 Broader dynamics/operational regression (hardening blast-radius) — green
`tests/test_v014_rhs_ph_real_case.py` + `tests/dynamics/` +
`tests/test_m6_operational_mode_parity_envelope.py` +
`tests/test_m6_4x_verdict.py`: **72 passed, 0 failed** — confirms the
`force_fp64_island` insertion in rhs_ph + small_step_prep did not perturb any
fp64_default dynamics path.

### 3.4 Compiled-program proof the hardening is a no-op for fp64_default
`proofs/v020/s4/ultracode/dtype_hlo_audit.py` re-run on the hardened HEAD
(`proofs/v020/fp32_integration/dtype_hlo_audit_head_hardened.json`), compared to the
committed pre-hardening baseline `dtype_hlo_audit.json`:
- **fp64_default**: after-step carry dtypes IDENTICAL; StableHLO census IDENTICAL
  (`f32=12 f64=8743 converts=12 whiles=10` — same as baseline). → the hardening
  emits ZERO extra HLO for fp64_default; it is provably a compile-time no-op.
- **mixed_perturb_fp32_v020**: dtypes + HLO census IDENTICAL to baseline (`f32=128
  f64=8639 converts=151`); the 4 acoustic perts are float32 and SURVIVE a real step
  while totals stay fp64. `PROOF_fp32_takes_effect = True`.

---

## 4. GPU VERIFICATION

Method: `scripts/with_gpu_lock.sh`, host `taskset -c 0-3`, shared XLA cache
`<DATA_ROOT>/gpuwrf_jax_cache`.

### 4.1 fp64_default GPU byte-identity — all-7 9-domain canary nest
Case `<DATA_ROOT>/wrf_downscale/canary_all7/run` (max_dom=9), 1h, fp64_default (no
precision-mode env). Compared with `scripts/v020_wrfout_byte_compare.py` against a
fresh deterministic lowhang-baseline output (`c696010d` worktree).

**First all-7 byte-compare exposed an S4-merge fp64_default break (NOT my
hardening):** 879/963 vars matched, worst maxΔ=1.562e-2 Pa on PB (d08), with
P/PH/PB/PHB/MU/MUB/PSFC + U/V/W/QKE + radiation/surface (SWDOWN/GLW/LWDNB/HFX/LH/
UST) differing at fp64 round-off (~1e-9 to 1e-7 rel; all finite/stable).
Determinism confirmed (baseline-vs-baseline 963/963 maxΔ=0; head-vs-head 963/963
maxΔ=0), so the divergence is a genuine always-on op-order change from the merge,
not GPU noise.

**Root cause (bisected) + mode-gated fix (commit `d5a929e8`):**
- PRIMARY — `operational_mode._refresh_grid_p_from_finished` (runs every RK stage,
  was ungated): the S4 merge switched `phb` and the `p_total` base term from the
  historical FINISHED-state reconstruction (`next_state.{ph_total,p_total} −
  next_state.{ph,p}_perturbation`) to the pristine ENTRY-state `prep.pb/prep.phb`.
  Physically equal, but the FINISHED totals have evolved → entry-base ≠
  finished-base at fp64 round-off, fed to `diagnose_pressure_al_alt` and the
  written-back `p_total` every stage → cascade into P/PH/U/V/W/QKE and (output
  total−pert) PB/PHB/MU/MUB + downstream radiation/surface. (The earlier
  `small_step_finish` fix `2d875b59` was MASKED because this refresh runs AFTER it
  and overwrites `p/p_total`.) Fix: gate on the static `acoustic_precision_mode` —
  fp64_default uses the finished-state reconstruction (byte-identical pre-S4 HLO);
  mixed keeps `prep.pb/phb` (REQUIRED: total−fp32-pert re-introduces cancellation).
- SECONDARY — PSFC Kahan summation (`wrfout_writer.py` + `operational_mode.
  _psfc_from_state`): merge unconditionally replaced plain `.sum(axis=0)` with a
  Kahan compensated loop → different rounding broke fp64_default PSFC. Gate on the
  perturbation storage dtype: fp32→Kahan, fp64→plain sum (byte-identical pre-S4).

CPU re-gate after the fix: dynamics + rhs_ph + operational parity + fp32 suite +
precision matrix = 73 passed, 1 skipped.

GPU re-verify (one fixed-head all-7 fp64_default vs the saved deterministic
lowhang baseline, `scripts/v020_wrfout_byte_compare.py`):
**RESULT: 963/963 vars BIT-IDENTICAL across all 9 domain files, worst maxΔ=
0.000e+00, verdict=BYTE-IDENTICAL** (head rc=0, all domains finite). Artifact:
`proofs/v020/fp32_integration/byte_compare_fp64_FIXED_963of963.txt`. **fp64_default
GPU BIT-IDENTITY GATE: PASS.** (Determinism independently confirmed: baseline-vs-
baseline and head-vs-head both 963/963 maxΔ=0, so this is a true byte match.)

### 4.1b fp64_default GPU SPEED — warm-vs-warm all-7 (the 3.83× was a cold-compile artifact)
The first head run measured 4551s "integration-only" vs the v0.19 baseline 1189s
"forecast-only" — a 3.83× apparent slowdown. A warm-vs-warm A/B (each branch
warmed once, then a timed warm pass; `/tmp/v020_fp32_warm_verify`) showed it was a
ONE-TIME cold-megacompile artifact, NOT a per-step regression:
- baseline warm forecast-only: 1178.1 s (warmup 1133.5 s)
- HEAD (fixed) warm forecast-only: **1168.3 s** — EQUAL to baseline (0.8% faster,
  within the noisy-desktop run-to-run spread).
This matches the dtype/HLO census (§3.4: fp64_default HLO byte-identical → identical
per-step FLOPs). The one-time cold compile is the merged program's larger
executable (added fp32-gated branches/variants) + a carry-pytree treedef change
(`OperationalCarry.base_state`) that invalidates the v0.19 warm cache key; after
HEAD's own warm-up the default-path speed matches the lowhang baseline.
**SPEED GATE: PASS** (fp64_default stays the fast, default-runnable path).

### 4.2 fp32 mixed AND aggressive — tolerance-PASS (bigswiss d01 full physics, 1h) — PASS
Case `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init` (461×461×45, dx=3km,
dt=18s, full physics), 1h. Arms run with the hardened HEAD code: fp64 (truth),
mixed (`GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020`), aggressive
(`GPUWRF_FORCE_FP64=0`). Compared vs the fp64 arm with
`proofs/v020/s4/ultracode/compare_precision_wrfout.py` against the frozen v0.20 fp32
acceptance bands (`proofs/v020/fp32_proto/acceptance_bands.py`). All arms rc=0, no
OOM (fp64 peak 20436 MiB, mixed 20481, aggressive 17805-class). Artifacts:
`proofs/v020/fp32_integration/tol_{mixed,aggressive}.{txt,json}`.

**MIXED (`mixed_perturb_fp32_v020`): OVERALL tolerance pass = True, hard gates = True**
— all 19 checked fields green, finite, no blow-up:
- wind: U rmse 0.00151 m/s, V 0.00115, W 0.00044, U10 0.00085, V10 0.00113
- temperature: T 0.00118 K, T2 0.00047 K
- pressure/mass/geopotential: P rmse 0.096 Pa, PSFC 0.064 Pa, PH 0.042, MU 0.064
- moisture: QVAPOR rmse 1e-6 (clean); condensate QCLOUD/QICE/QSNOW/QGRAUP rmse 0.

**AGGRESSIVE (`GPUWRF_FORCE_FP64=0`, full ADR-007 fp32-gated matrix): OVERALL
tolerance pass = True, hard gates = True** — all 19 fields green, finite:
- wind: U rmse 0.00182 m/s, V 0.00167, W 0.00053, U10 0.00116, V10 0.00161
- temperature: T 0.00151 K, T2 0.00079 K
- pressure/mass/geopotential: P rmse 0.101 Pa, PSFC 0.069 Pa, PH 0.044, MU 0.069
- moisture: QVAPOR rmse 1e-6; condensate clean.

Both ~2-3 orders of magnitude inside the 24h skill bands at the 1h lead, non-zero
(fp32 genuinely active), no field explodes. The hardening (fp64 islands + species
finiteness gate) did not break the fp32 path. fp32 takes effect confirmed by the
dtype/HLO audit (§3.4): mixed carries the 4 acoustic perts float32 surviving a step.

---

## 5. SUMMARY / RISK — BOTH BINDING GATES PASS

- **Merge:** clean (one auto-resolved file, lowhang structure + fp32 hook both
  present); both `c696010d` (lowhang) and `f9d3fde7` (fp32) are ancestors.
- **Hardening (commit `b836a91b`):** 3 fixes (fp64 islands for ph_tend +
  coupled-work primes; per-field 1km species finiteness gate), all
  zero-resident-VRAM, all no-ops for fp64_default (proven at test, regression, AND
  compiled-HLO census level).
- **fp64_default BIT-IDENTITY (binding gate #1): PASS.** The first GPU all-7
  byte-compare exposed an S4-merge break (NOT my hardening), root-caused to TWO
  always-on changes (`_refresh_grid_p_from_finished` entry-vs-finished base + PSFC
  Kahan summation) and fixed mode-gated (commits `2d875b59`, `d5a929e8`):
  GPU all-7 9-domain re-verify vs the deterministic lowhang baseline =
  **963/963 vars maxΔ=0.000e+00, BYTE-IDENTICAL** (§4.1). CPU per-test result-set
  byte-identical + HLO census byte-identical too.
- **fp64_default SPEED (binding gate #2): PASS.** Warm-vs-warm all-7: HEAD 1168.3 s
  == baseline 1178.1 s forecast-only (§4.1b). The earlier "3.83×" was a one-time
  cold-megacompile cache-key-miss artifact, not a per-step regression
  (HLO census identical → identical per-step FLOPs).
- **fp32 (opt-in): takes effect + tolerance-PASS.** dtype/HLO audit confirms the 4
  acoustic perts are float32 and survive a step (§3.4); GPU bigswiss 1h MIXED and
  AGGRESSIVE both OVERALL tolerance-pass=True, hard-gates=True, all 19 fields green,
  finite, condensate clean (§4.2).
- **main untouched; fp32 strictly opt-in via env** (`GPUWRF_ACOUSTIC_PRECISION_MODE`
  / `GPUWRF_FORCE_FP64=0`); `fp64_default` is the byte-identical, default-runnable,
  fast default.

### Branch lineage (commits on `worker/integration/v020-fp32`)
`c696010d` (lowhang tip) → `0c77c031` (merge S4) → `b836a91b` (harden) →
`2d875b59` (small_step_finish gate) → `ecd98f6d` (proofs) → `d5a929e8`
(_refresh_grid_p_from_finished + PSFC gate — the PRIMARY bit-identity fix).

### Unresolved risks / notes
- The merged program's COLD compile of the all-7 nest is long (~50 min on the
  contended desktop) because the executable is larger (dormant fp32-gated branches/
  variants) and the carry-pytree treedef changed (`OperationalCarry.base_state`),
  invalidating the v0.19 warm cache key. This is a ONE-TIME per-HLO cost, not a
  per-step regression (warm speed matches baseline). The release recipe's one-time
  warm-megacompile in a full-resource window still applies.
- The fp32 acceptance bands are checked at the 1h lead (~2-3 orders inside the 24h
  bands); 24-120h skill is the v0.20 stability/skill gate's remit, not this
  integration sprint.
