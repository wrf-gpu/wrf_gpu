# Known Issues — v0.16.0

Honest, code-grounded list of what is open or bounded in the v0.16 release. Each
entry states the symptom, the current understanding, the workaround, and the
tracked follow-up. No spin. The deeper per-issue history (KI-1…KI-11, including
resolved items) is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

> **v0.16 release framing.** v0.16 is the **STABILITY** release: 24/25 L2
> schemes coupled-green on a real case (`ALL_GREEN_OR_CARRIED`), the aerosol-aware
> Thompson "+1" (`mp_physics=28`, WRF-module oracle PASS), and a **1 km-unlock**
> (chunked MYNN BouLac: a 1 km single domain now fits on one RTX 5090,
> bit-identical to dense). **Performance is still honestly ~parity** with CPU-WRF
> (GeForce fp64 1/64 hardware law — no end-to-end speedup).

## v0.16 open / bounded items (new or updated this release)

| ID | Summary | Severity | Workaround / follow-up |
|---|---|---|---|
| **Performance ~parity** | fp64 GPU ≈ 24–28-rank CPU-WRF wall — a GeForce fp64 1/64 hardware law (the fp64 dycore sits at its 0.944 FLOP/byte roofline ridge, so the fp64-ALU term binds). **No end-to-end speedup.** | Honest finding | The fp32 make-or-break is **CONCLUDED** (double-confirmed: Opus + independent GPT): the valid-numerics fp32 ceiling is **~1.1×** (full-ws 16k 1.107× / 65k 1.110×, VRAM ratio 1.000; GPT reproduced 1.105× / 1.111×); larger fp32 speedups are **precluded** by the conservation/cancellation fp64 pins (the ~4.3× "cost proxy" is numerically invalid — corrupts conservation; qke non-finite at 1 km). The genuine ~1.1× fp32 lane ships now; the larger levers are algorithmic / multi-GPU. Fusion probed NEGATIVE (~0%). Evidence: `proofs/v016/fp32_verdict/`. |
| **Noah-classic `sf_surface=2`** | The coupled coverage gate for Noah-classic land surface is the lone L2 carry (`SCOPED_CARRY` in the rollup) — it needs the WRF land/static-data bundle (soil/veg tables + static fields) wired into the real-case harness. | Scope-carry → v0.17 | Use Noah-MP (`sf_surface=4`, operational). |
| **mp28 coupled field-gate** | The mp8-vs-mp28 ±advection coupled short-grid field-gate was queued on the contended single GPU and did not run in time. The mp28 **L1 WRF-module oracle (5187-col) is GREEN**, and CPU threading/restart/precision/catalog gates are green. | Bounded carry → v0.17 | GPU time only, no further code. mp28 is operationally wired + oracle-validated. |
| **1 km working set** | A 1 km **single domain** now fits on one RTX 5090 via chunked BouLac (this release; **orthogonal to fp32**), **measured in a fresh process per grid**. Repeated multi-grid runs in one process can fragment allocator memory. Larger working sets (≥ ~196 k cols) still exceed 32 GiB in fp64. | Improved; bounded | Isolate grids per process or recycle the process between grids; **multi-GPU horizontal sharding** for the larger sets. (fp32 does **not** shrink the peak — v0.16 proves it is transient working memory, not persistent fp64 State; demoting −700 MiB of State moves the peak 0 GiB.) |
| **RRTMG SW/LW variants** | Additional RRTMG radiation options need a WRF oracle rebuild before coupling. | Carry → v0.17 | Use the wired RRTMG/Dudhia/GSFC paths. |

All v0.15 items below carry forward unchanged **except** the fp32 forward
expectation — see the v0.16 forward-correction in the framing note below.

---

# Known Issues — v0.15.0 (carried forward)

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
>
> **Forward correction (v0.16).** The v0.15 Performance/Precision items below name
> the **fp32-operational-state restructuring** (ADR-007/031) as the deferred lever
> for a genuine speedup and for ~halving VRAM. v0.16's make-or-break investigation
> **supersedes that expectation** (double-confirmed: Opus + independent GPT): the
> valid-numerics fp32 ceiling is **~1.1×** (not a large speedup) and fp32 does
> **not** reduce the transient-dominated VRAM peak (0 GiB moved by a −700 MiB
> persistent-State demotion; the base absolutes `p_total`/`ph_total` are
> conservation-pinned to fp64 and corrupt the geopotential/PGF gradient 27×/127×
> if stored fp32; qke goes non-finite in fp32 at 1 km). The remaining
> genuine-speedup / VRAM levers are **algorithmic** (the chunked/O(nz) MYNN BouLac
> fix) and **multi-GPU sharding**, not fp32. The v0.15 text is retained as
> historical record. Evidence: `proofs/v016/fp32_verdict/`.

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
