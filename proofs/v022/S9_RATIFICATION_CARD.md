# §9 op-relaxed tier — 2-minute ratification card (for the user)

Condensed from `ADR-OPERATIONAL-RELAXED-TIER.md` §9 (rev B, critic-hardened). This is the ONE block that becomes
critical-path the moment the v0.22 openers finish — it gates WI-0 (the harness), K3 (PBL cadence) and K6 (non-lossless
frontier). **It does NOT block the running openers (R0 / compile-pathology / K2).** Nothing here flips a default;
Tier-O is opt-in only and can never mutate WRF-faithful behavior.

**Fastest path:** if the proposed values look right, reply "§9 approve all as-proposed" + your pick on the 2 forks below.

| # | Item | Proposed value | Why it's set this way |
|---|---|---|---|
| 1 | Per-variable skill bands | T2/T-upper @120h ≤1.0 K · PSFC ≤120 Pa @72/120h · QVAPOR @24h ≤3e-4 / nRMSE ≤0.20 · new PBLH/W/QCLOUD/QICE rows · Q2/2m-RH deferred | calibrated to the KI-9 24h numbers; rev-B tightened 6 bands |
| 2 | Wind-class conjunctive rule | **ALL of**: RMSE ≤ strict×1.15 **AND** Δ-vs-strict (U10/V10 ≤1.8 m/s, upper U/V ≤2.0 m/s @120h) **AND** BOUNDED divergence **AND** obs-floor | winds are the weakest field; the ×1.15 cap is the one that actually bites |
| 4 | Per-lead trajectory | whole per-lead divergence series must be BOUNDED, not just @24/72/120h | stops a field "passing" at checkpoints while blowing up between them |
| 5 | Per-term clamp/limiter anti-cheat | gate count + sum_magnitude + max_magnitude **per GuardLimiterTerm**; any new term key = auto-REJECT pending ack | prevents a lever from silently leaning on clamps to "pass" |
| 6 | FSS precip spec | tiers 1/5/10 mm · neighbourhood = cells-per-domain (9 @3km, 25 @1km) · vs **dense CPU-WRF** field · per-tier minima | makes FSS executable + station-independent |
| 7 | Other HARD guards | conservation no-new-drift margins +1e-6 / +1e-5 · finite-detector exactly 0 · restart bit-identity | the non-negotiable safety floor |
| 8 | Aggregation | worst-case-over-cases for the dyn/thermo core (T,W,PH,PSFC,T2); conjunctive rule for the surface-wind class | core must hold on the hardest case, not on average |
| 10 | Sign-off rule | per-lever AND per-config the user sign-off + the "Tier-O may never flip a default" invariant | keeps you in the loop on every relaxed lever |

## The 2 genuine forks (your pick changes behavior)

- **Fork A (item 3) — `BOUNDED_GROWTH` divergence regime:** proposed = **hard REJECT** (only `BOUNDED` exactly passes;
  `BOUNDED_GROWTH` fails). Alternative = allow `BOUNDED_GROWTH` as a **per-lever quarantine with your explicit ack**.
  → Recommend **hard REJECT** (strictest; safest for the north-star). Your call: REJECT / quarantine-with-ack.
- **Fork B (item 9) — frozen case set:** core 3 = CANARY-L2-D02 + SWITZERLAND-D01 + CONVECTIVE-DIVERSE. Question: is the
  **1 km Alpine surface-wind case IN or DEFERRED?** It's the most-divergent field (KI-4) — including it makes the tier
  much stricter on winds but risks no lever ever passing. → Recommend **DEFER to a follow-up** (don't let the hardest
  uncalibrated case block the whole tier on day 1). Your call: IN / DEFER.

Once you give "approve as-proposed" + Fork A + Fork B, WI-0 (the harness) is unblocked and K3/K6 can proceed.
