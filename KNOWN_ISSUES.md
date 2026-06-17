# Known Issues — v0.18.0

Honest, code-grounded list of what is open or bounded. Each entry states the
symptom, the current understanding, the workaround, and the tracked follow-up.
No spin. The deeper per-issue history (KI-1…KI-11, including resolved items) is in
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md); the v0.15 release-gate detail is
retained verbatim below the v0.18 section.

## v0.18 carried items (this release)

- **RAINNC bounded accumulated-precip residual.** Accumulated grid-scale precip is
  an operationally-bounded diagnostic. The v0.18 RAINNC/QVAPOR work is accepted as
  class-c with a measured **5.22 mm RMSE** vs the 1.0 mm atlas bound — **no
  tolerance widening**, judged no forecast-skill impact by decision. A
  precipitation-placement sensitivity, not a dynamics blow-up. Proof:
  `proofs/v018/rainnc_qvapor_status.json`. (Continuation of the v0.15 RAINNC KI.)
- **CLM4 (`sf_surface_physics=5`) / CTSM (`6`) — documented architecture boundary,
  fail-closed.** Not registry-accepted; `oracle_path=None` (no oracle claimed); the
  registry rejects them and the catalog lists them in the recognized fail-closed
  table. A deliberate v1.0 scope boundary, not a silent gap. Proof:
  `proofs/v018/lsm_family_status.json`.
- **K2 multi-GPU specified-BC — EXPERIMENTAL, default-OFF.** The K2 domain-decomp
  lane is periodic-only-faithful; **specified-BC is NOT faithful** (boundary ring
  excluded from the pass gate; a prior tolerance-widening was reverted). Default-off
  is proven (no collectives emitted: `proofs/v018/k2_flag_off_graph.json`). Do not
  enable specified-BC multi-GPU for production. Proof: `proofs/v018/k2_multigpu_report.md`.
- **PBL11 Shin-Hong TKE-diagnostic follow-up.** Shin-Hong is operational despite a
  28.5 % TKE-diagnostic residual: the TKE/EL fields are source-traced as
  non-driving (dynamics tendencies never read them), `tke_diagnostic_exact_pass`
  is surfaced (not masked), and a TKE-oracle upgrade is the tracked follow-up. The
  operational promotion is justified on the driving fields. Detail:
  `proofs/v018/schemes_critic_opus.md` §1.

### Carried CPU-suite test-debt (pre-existing on v0.17; honest xfail, not silent RED)

The full CPU test suite on this workstation carries **38 xfailed tests** (36
newly converted by the v0.18 triage + 2 pre-existing), all
**pre-existing** (each fails identically on tag `v0.17.0`; none is a
v0.18-introduced regression — verified). They are converted to documented
**non-strict xfail** (registry: `tests/conftest.py::_PREEXISTING_CPU_XFAILS`;
full table + reasons: `proofs/v018/suite_triage.md`). Each test still RUNS — an
unexpected pass surfaces as XPASS — and **no assertion was deleted or loosened**.
The three categories:

- **ENV (host environment).** External single-column WRF Fortran harnesses
  (MYNN/RRTMG/Thompson), warmed `nsys` traces, the read-only Gen2 corpus, the
  paper publication-audit corpus, and the agentos skill-evals corpus are not all
  present on a plain CPU checkout; plus three byte-/bitwise-identity checks
  (`test_v013_{mrf,myj}_operational` default-suite, `test_v015_stream_a_bitwise`)
  that are non-deterministic only on the **XLA:CPU AOT** path (machine-feature
  mismatch, `+prefer-no-gather`/SIGILL warnings) and are bit-stable on the GPU
  operational backend.
- **GPU-NUM (GPU-native numeric oracle on the CPU backend).** The m6b4/m6b5
  acoustic/dycore savepoint-parity and m6b0r `calc_coef_w` top-row oracles, the
  m6x ADR-023 MPAS-slice RMSE and `c2_metrics` identity, and the step-46
  v-runaway bound run on the CPU backend against fixtures that target the GPU
  operational hydrostatic path (the idealized fixtures predate the F7
  hydrostatic-column requirement and hit NaN on the CPU vertical-solver path).
  The operational GPU forecast is unaffected; the GPU-lock suite is green.
- **STALE (assertion pins a superseded value/source/design).** HaloSpec
  width-rejection rule, init-allocator allowlist, the all-fp64 precision-registry
  expectation (fp32 fields now exist), refactored operational/acoustic source
  strings, the theta-guard per-level ceiling (700→450 K), the default MYNN
  condensation `niter` (50→16, documented shipped default), and drifted
  stabilizer-count / scout-snapshot / boundary-replay baselines. These should be
  updated to the current shipped values in a follow-up test-maintenance sprint.

Nine other previously-RED tests were **fixed to green** this triage (not
carried): the three operational `device_get` source guards (made loop-precise —
the v0.18 `_assert_nonzero_initial_mu_total` host pull is a verified OUT-OF-LOOP
one-time pre-flight check, not a hot-loop transfer), the two `m7_1km_memory_audit`
tests (audit now sizes all 67 State leaves), the two `wsm_sm_savepoint_parity`
path-form checks (expanduser), and the two `v013_compile_perf2` recompile-hygiene
tests (cleared global JAX caches to remove a test-order artifact).

**Proof-artifact hygiene.** 18 committed proof JSONs that the suite previously
re-wrote with fp/path/timing noise on every run are now write-gated
(`GPUWRF_WRITE_PROOFS=1` or a tmp `out=`/`--proof-dir`); the canonical proofs are
unchanged and the worktree ends clean after a full suite run. Detail:
`proofs/v018/suite_triage.md`.

### Resolved in v0.18 (was a regression risk, fixed in code — not carried)

- **Thompson warm-process graupel-melt fidelity (m5 WRF mass oracle).** Accepted
  commit `044bb65a` (cold-process fidelity) bundled a finer diagnostic
  graupel-number distribution without WRF's sparse-graupel `N0_melt` melt override
  (module_mp_thompson.F:2802-2806), under-melting warm-cell graupel (spurious
  `qg≈5e-7` where WRF melts to 0), and added rci/sci ice-collection without WRF's
  `if(temp<T_0)` cold gate (line 2554). Both were transcribed from pristine WRF;
  the warm-process oracle (`tests/test_m5_thompson_process_residuals.py`) is now
  bit-exact and the cold-process gain is preserved (cold-collection oracle GREEN).
  **Fixed, not a carried tolerance.** Detail: `proofs/v018/integration_report.md`
  → "Honesty-fix closeout".

---

# Known Issues — v0.15.0 (retained release-gate detail)

Honest, code-grounded list of what was open or bounded in the v0.15 release. Each
entry states the symptom, the current understanding, the workaround, and the
tracked follow-up. No spin. The deeper per-issue history (KI-1…KI-11, including
resolved items) is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

> **Release framing.** v0.15 is a **kernel-architecture + WRF-fidelity** release,
> **not** an end-to-end speedup release. It delivers the project's final fp64 GPU
> kernel (adversarially confirmed near-optimal, device-bound), the MYNN-EDMF
> condensation `niter` 50 → 16 + Thompson cold-collection WRF-fidelity fixes, the
> MUB/PB nest-base-state seam fix, and re-closes both 72 h GPU-vs-CPU-WRF
> field-parity gates (Switzerland d01 + Canary L2 d02) with the reproducible
> identity-proof plot system ([`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md)).
> Performance is honestly **~parity total-wall** (0.99×/1.04×), forecast-only
> ~1.05–1.20×; **no multi-× and no large-grid speedup is claimed.** The honest
> numbers are below.

## Final-gate verdicts (both gates re-closed on the final v0.15 code)

- **Switzerland d01 72 h field-parity gate:** stable to h72; **9/10 prognostic
  fields within frozen tolerance**, dynamics/thermo/mass all green; **1 atlas
  tolerance failure** (vs v0.14's 3). The single Grid-Delta Atlas hard-gate miss
  is **RAINNC rmse 5.08 mm vs the 1.0 mm bound** (bounded precip sensitivity;
  cold-collection moved it −15 % from v0.14's 5.99 mm). DZS/ZS now PASS paired.
  Run `v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z` vs CPU truth
  `v014_switzerland_72h_cpu_20260610T122909Z`; **total-wall GPU 2941 s vs CPU
  2906 s ≈ 0.99× (forecast-only ~1.20×)**, peak VRAM **22.9 GiB** (up from
  v0.14's 20.5 GiB). Proof: `proofs/v015/finalgates/switzerland_d01/`.
- **Canary L2 d02 72 h field-parity gate:** stable to h72; operational verdict
  **L2_D02_GREEN** (bounds PASS, rmse PASS, pipeline green); **9/10 prognostic
  fields within frozen tolerance**; **1 atlas tolerance failure** (vs v0.14's 3).
  The lone bounded Atlas miss is **QVAPOR rmse 1.44×10⁻³ vs 1.0×10⁻³ kg/kg**
  (+44%, **no regression** from v0.14's 1.45×10⁻³). The v0.14 **MUB/PB**
  nest-frame-seam statics are now **fixed** (250.7 → 0.0078 Pa). Run
  `v015_canary_d02_72h_gpu_finalgates_20260613T095113Z` vs CPU truth
  `20260501_18z_l2_72h_20260519T173026Z`; **total-wall GPU 8414 s vs CPU ~8713 s
  ≈ 1.04× (forecast-only ~1.05×)**, peak VRAM **29.8 GiB** (up from v0.14's
  21.1 GiB — a disclosed regression). Proof: `proofs/v015/finalgates/canary_l2_d02/`.

## Bounded acceptances (honest, with their numeric justification)

These fields are **bounded-not-exact**: operationally acceptable but not painted
as bitwise-exact channels. They are drawn red (never green) when they breach
their envelope in the identity-proof plots.

- **KI — RAINNC bounded precipitation sensitivity.** Accumulated grid-scale
  precipitation is an operationally-bounded diagnostic with a 1.0 mm RMSE
  envelope. On the final Switzerland d01 72 h v0.15 run it sat at **5.08 mm RMSE**
  (out of the 1.0 mm bound) — a precipitation-placement sensitivity, not a
  dynamics blow-up; the full dynamics/thermodynamics core stayed within envelope.
  v0.15 **Thompson cold-collection** moved it −15 % from v0.14's 5.99 mm; the
  RAINNC WRF-convention bug (snow + graupel + ice were dropped from the
  accumulation) was FIXED in v0.14. Per-cell/number refinement → v0.16.
- **KI — Canary QVAPOR bounded.** 3D water-vapour mixing ratio carries a tight
  1.0×10⁻³ kg/kg envelope. On the final Canary L2 d02 72 h v0.15 run it was
  marginal at **1.44×10⁻³ kg/kg (+44%)** — **no regression** from v0.14's
  1.45×10⁻³ (MYNN marine entrainment-depth class; surface flux exonerated).
  Carried → v0.16 stability lane.
- **KI — Canary MUB/PB nest-frame-seam base-state artifact (FIXED in v0.15).**
  v0.14's static base-state mass (`MUB`)/base pressure (`PB`) nest-frame-seam
  artifact (Atlas max_abs MUB 250.7 / PB 249.9) is **fixed in v0.15** — the nest
  now packs its own static base ring (WRF never forces nest base state). On the
  final Canary L2 d02 72 h v0.15 run MUB/PB max_abs is **0.0078 Pa**. Resolved.
- **KI — GRAUPELNC source-fidelity gap.** Accumulated graupel has a microphysics
  source-fidelity gap vs CPU-WRF (the same Thompson parity debts tracked under
  KI-4 in `docs/KNOWN_ISSUES.md`: snow fall-speed approximation, cloud-water
  sedimentation, invalid-column fallback). Bounded; tracked, not a dynamics
  issue.

## Scope boundaries (deliberate, not silent gaps)

- **KI-3 — focused wrfout writer field subset.** The operational writer emits a
  focused **104-variable** `wrfout` (core met/spatial/vertical/soil + radiation
  flux + Noah-MP snow-layer) vs WRF's 375. Missing fields are stochastic-seed
  arrays and less-common diagnostics. Full 375-variable coverage is deferred.
- **KI — tier3_coupled double-count.** The tier-3 coupled validation aggregation
  can double-count a contribution in one path; this is a reporting/aggregation
  caveat in the validation tooling, not a forecast-state error. The per-field
  gate numbers (the identity-proof scoreboard and the grid-delta atlas) are the
  authoritative parity evidence and are not affected.

## Performance

- **KI — ~parity total-wall; fp64 kernel near-optimal (the lever is fp32-operational).**
  v0.15 is **not** an end-to-end speedup release: total-wall is ~parity
  (Switzerland 0.99×, Canary 1.04×), forecast-only ~1.05–1.20×. The fp64 GPU
  kernel is **adversarially confirmed near-optimal without megakernels** — the
  step is **device-bound** (~4 % idle), every host-side lever (whole-step
  CUDA-graph capture, denominator hoist, scan-unrolls) is shipped or proven
  wall-neutral, and cuSPARSE PCR + column tiling are already live. The per-step
  MYNN-condensation `niter` 50 → 16 win (~1.4× isolated) dilutes to forecast-only
  ~1.05–1.20× in the full pipeline. **No multi-× / large-grid speedup is claimed**
  (the prior "6–10× per-cell at large grids" framing is **refuted** by the
  project's own `km_bench`: per-cell cost slightly worsens; honest large-grid
  asymptote ~1.6×). Evidence: `proofs/perf/v015/kernel_characterization.md`.
- **KI — opt-in fp32-BouLac (~1.67×) blocked by a compile pathology.** fp32-BouLac
  and an O(nz) BouLac memory win are proven correct in isolation (WRF oracles to
  machine epsilon) but hit an XLA mixed-precision "very-slow-compile" pathology in
  the full forecast jit; they ship as **opt-in flags** (`GPUWRF_MYNN_BOULAC_FP32`,
  `GPUWRF_MYNN_BOULAC_ONZ`), default off.
- **KI — higher peak VRAM (a v0.15 regression).** Peak VRAM rose vs v0.14
  (Switzerland 20.5 → 22.9 GiB, Canary 21.1 → 29.8 GiB) from the `niter`-16
  unrolled temporaries + larger nested arena. The intrinsic ~21.7 GiB fp64 dycore
  working set is the binding 1 km ceiling; the deferred fp32-operational
  restructuring ~halves the dycore arena.

## Precision

- **KI — fp32 operational-state restructuring is the named next major lever.** The
  standalone CLI path is fp64-only; gated-fp32 remains an experimental ADR-007
  preview. The **fp32-operational-state restructuring (ADR-007 / ADR-031)** is the
  **single** structural fix for both the genuine speedup (the opt-in fp32-BouLac
  ~1.67× lands once the mixed-precision compile pathology is gone) and 1 km
  scalability (it ~halves the VRAM arena). It is a pervasive precision change with
  a full identity re-gate — the defined next major lever, not a v0.15 add-on.

## Carried from v0.13.0 (see `docs/KNOWN_ISSUES.md` for full detail)

- **KI-9** — 24 h/72 h forecast-skill equivalence (T2/U10/V10) vs CPU-WRF is the
  credibility gate; v0.14 reports the field-parity gates honestly but does not
  claim closed forecast-skill equivalence. Hard dynamics-`ph'`/MYNN/`*_tendf`
  GPU work, no cheap knob.
- **KI-4** — d02 U10 episodic final-lead under-prediction (tied to KI-9).
- **KI-6** — RRTMG SW intermediate `taug` top-layer convention differs in 4 UV
  bands; integrated fluxes pass tier-1 (< 0.05% rel). Isolated, pre-existing.
- **KI-7** — free-running (`run_boundary=False`) on wide domains (nx≈160+) can go
  unstable beyond ~14 h. The validated operational path uses boundary forcing.
- **KI-10** — moisture-advection cadence refinements (opt-in, default-off; no
  shipped-behavior impact).
- **KI-11** — 2-way nesting equivalence vs CPU-WRF untested (only finite/stable
  proven).
- **KI-5** — powered n=15 TOST is underpowered (n≈27 for full power); scoring
  path is unblocked. No TOST PASS is claimed.
