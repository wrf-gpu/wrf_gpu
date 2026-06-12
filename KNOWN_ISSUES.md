# Known Issues — v0.14.0

Honest, code-grounded list of what is open or bounded in the v0.14 release. Each
entry states the symptom, the current understanding, the workaround, and the
tracked follow-up. No spin. The deeper per-issue history (KI-1…KI-11, including
resolved items) is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

> **Release framing.** v0.14 is a **memory + WRF-identity** release, not a
> performance release. The headline evidence is the 72 h GPU-vs-CPU-WRF
> field-parity gates (Canary L2 d02 and Switzerland d01) plus the reproducible
> identity-proof plot system ([`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md)).
> The honest current numbers are below; final gate verdicts are filled by the
> manager once both 72 h gates close.

## Final-gate placeholders (manager fills at release)

- **Canary L2 d02 72 h field-parity gate:** `<manager: final 72h numbers — verdict + worst-field RMSE vs bound>`
- **Switzerland d01 72 h field-parity gate:** `<manager: final 72h numbers — verdict + worst-field RMSE vs bound>`

## Bounded acceptances (honest, with their numeric justification)

These fields are **bounded-not-exact**: operationally acceptable but not painted
as bitwise-exact channels. They are drawn red (never green) when they breach
their envelope in the identity-proof plots.

- **KI — RAINNC bounded precipitation sensitivity.** Accumulated grid-scale
  precipitation is an operationally-bounded diagnostic with a 1.0 mm RMSE
  envelope. On the development Switzerland d01 72 h run it sat at ~5.99 mm RMSE
  (out of the 1.0 mm bound) — a precipitation-placement sensitivity, not a
  dynamics blow-up; the full dynamics/thermodynamics core stayed within
  envelope. Final number: `<manager: final 72h RAINNC RMSE>`.
- **KI — Canary QVAPOR bounded.** 3D water-vapour mixing ratio carries a tight
  1.0×10⁻³ kg/kg envelope. On the development Canary L2 d02 72 h run it was
  marginal at ~1.45×10⁻³ kg/kg (+45%). Final number:
  `<manager: final 72h QVAPOR RMSE>`.
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
