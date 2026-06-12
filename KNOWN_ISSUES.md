# Known Issues — v0.14.0

Honest, code-grounded list of what is open or bounded in the v0.14 release. Each
entry states the symptom, the current understanding, the workaround, and the
tracked follow-up. No spin. The deeper per-issue history (KI-1…KI-11, including
resolved items) is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

> **Release framing.** v0.14 is a **memory + WRF-identity** release, not a
> performance release. The headline evidence is the 72 h GPU-vs-CPU-WRF
> field-parity gates (Canary L2 d02 and Switzerland d01) plus the reproducible
> identity-proof plot system ([`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md)).
> Both gates closed on the final code; the honest numbers are below.

## Final-gate verdicts (both gates closed on the final code)

- **Switzerland d01 72 h field-parity gate:** stable to h72; **9/10 prognostic
  fields within frozen tolerance**, dynamics/thermo/mass all green. The single
  Grid-Delta Atlas hard-gate miss is **RAINNC rmse 5.19 mm vs the 1.0 mm bound**
  (bounded precip sensitivity, ≈0.78× the field's own std 6.6 mm; the RAINNC
  WRF-convention bug — snow+graupel+ice dropped — is FIXED). DZS/ZS now PASS
  (writer fix). Run `v014_switzerland_d01_72h_FINAL_20260612T062354Z` vs CPU truth
  `v014_switzerland_72h_cpu_20260610T122909Z`; GPU ~2762 s vs CPU 2906 s ≈ 1.05×,
  peak VRAM ~19.8 GiB.
- **Canary L2 d02 72 h field-parity gate:** stable to h72; operational verdict
  **L2_D02_GREEN** (bounds PASS, rmse PASS, pipeline green); **9/10 prognostic
  fields within frozen tolerance**. The v0.14 default-on changes (open-top, 2D
  Smagorinsky, physics-`tendf` fold, theta-ceiling 1000 K) do **not** regress
  Canary. Three bounded Atlas misses: **MUB max_abs 250.7 + PB max_abs 249.9**
  (known STATIC nest-frame-seam base-state artifact, localized) and **QVAPOR rmse
  1.45×10⁻³ vs 1.0×10⁻³ kg/kg** (bounded moisture, +45%). Run
  `v014_canary_d02_72h_FINAL_20260612T062354Z` vs CPU truth
  `20260501_18z_l2_72h_20260519T173026Z`; GPU ~8200 s vs CPU 8713 s ≈ 1.06×, peak
  VRAM ~20.3 GiB.

## Bounded acceptances (honest, with their numeric justification)

These fields are **bounded-not-exact**: operationally acceptable but not painted
as bitwise-exact channels. They are drawn red (never green) when they breach
their envelope in the identity-proof plots.

- **KI — RAINNC bounded precipitation sensitivity.** Accumulated grid-scale
  precipitation is an operationally-bounded diagnostic with a 1.0 mm RMSE
  envelope. On the final Switzerland d01 72 h run it sat at **5.19 mm RMSE** (out
  of the 1.0 mm bound, but ≈0.78× the field's own std of 6.6 mm) — a
  precipitation-placement sensitivity, not a dynamics blow-up; the full
  dynamics/thermodynamics core stayed within envelope. The RAINNC WRF-convention
  bug (snow + graupel + ice were dropped from the accumulation) is **FIXED**.
- **KI — Canary QVAPOR bounded.** 3D water-vapour mixing ratio carries a tight
  1.0×10⁻³ kg/kg envelope. On the final Canary L2 d02 72 h run it was marginal at
  **1.45×10⁻³ kg/kg (+45%)**.
- **KI — Canary MUB/PB nest-frame-seam base-state artifact.** On the final Canary
  L2 d02 72 h run the static base-state mass (`MUB`) and base pressure (`PB`)
  carry a localized nest-frame-seam artifact (Atlas max_abs **MUB 250.7**, **PB
  249.9**). This is a known **static** base-state seam, not an evolving dynamics
  error; bounded, tracked, carried to v0.15.
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

- **KI — warm throughput ~1.05× (→ v0.15).** v0.14 is **not** a performance
  release: warm per-forecast-hour throughput is roughly on par (~1.05×) with
  v0.13.0. The v0.13.0 perf-triage attributed the earlier 3×→1× change to a
  double-compile (fp32→fp64 graph) with no trivial identity-safe fix; performance
  is the dedicated focus of v0.15. **No performance headline is claimed for
  v0.14.**

## Precision

- **KI — fp64 operational-state ADR pending.** The standalone CLI path is
  fp64-only; gated-fp32 remains an experimental ADR-007 preview and is no faster
  on this memory-bound workload. Whether/how to operationalize a reduced-precision
  state is pending a dedicated ADR (tied to the v0.15 performance work).

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
