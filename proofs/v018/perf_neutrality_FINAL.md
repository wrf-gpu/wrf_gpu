# v0.18 Perf-Neutrality — FINAL (dual-model confirmed)

**Verdict: PERF-NEUTRAL. v0.18 does not regress the default-scheme standard case vs v0.17.**

Branch: `worker/gpt/v018-integration` (set-UNION of all 8 v0.18 family branches + radfix hygiene + scan-overhead fix `292a4431`).
Standard case: default operational config (Thompson mp=8 + RRTMG + MYNN + Noah), canonical `proofs/perf/warmed_timing.py`, public entry, warmed/steady-state GPU clocks, single GPU lock.

## Two independent same-session canonical measurements (both bracket zero)

| Measurer | v0.17 (s/fc-h) | v0.18 @292a4431 (s/fc-h) | Δ | Committed JSON |
|---|---|---|---|---|
| Opus-max (canonical, same-session, public entry) | 21.1516 | 21.2661 | **+0.54%** | **NOT captured — prose-only** (see note) |
| GPT (independent, clean worktrees, P1 clocks)     | 20.4689 | 20.2539 | **−1.05%** (v0.18 slightly faster) | `gpt_verify_v017/v018_warmed_timing.json` |

**Evidence-honesty note (resolves integration-critic F2):** the Opus
`21.1516 / 21.2661` post-fix canonical pair was measured in-session but its
warmed-timing JSON was **not committed** — no committed JSON in the repo
contains those numbers. The only committed Opus same-session JSONs
(`perf_neutrality_v017_rerun_warmed_timing.json` = 20.6253 /
`perf_neutrality_v018_warmed_timing.json` = 21.8359 → **+5.86%**) are the
**pre-fix `7b3bcc89` transient** that this document supersedes; they are kept only
as the record of the transient that `292a4431` fixed, NOT as the canonical figure.
The **committed** dual-confirm of perf-neutrality therefore rests on:
1. the committed **GPT independent series** (`gpt_verify_*`): v0.17 → v0.18 is
   **faster** (−1.05%), i.e. v0.18 ≤ v0.17 on its own; and
2. the **Opus structural root-cause** (`perf_rootcause_opus.md`): cold-process
   warm-cost = +0.07% (noise, ablation-measured) and the only real regressor (the
   81→74 carry narrowing) was reverted bit-identically by `292a4431`.

The Opus +0.54% prose figure is consistent with (1) and (2) but is not itself a
committed artifact; the perf-neutral verdict does not depend on it. Both deltas
are within v0.17's own intra-run repeat spread (±0.48–1%); the committed GPT
series alone already brackets/clears zero → **robustly perf-neutral**. (The
earlier "+5.8%" was cross-session GPU-clock drift after hibernate + the
now-fixed program-shape artifact, NOT a real regression.)

## Root cause of the transient regression (fixed)
- **NOT the new physics.** Ablating the RAINNC cold-process additions (rci/sci ice-collection, cloud-water freezing, graupel-number diag) changes warm time by **+0.07% (noise)** — XLA already fuses the +157 ops for free. The new cold-process fidelity is warm-free and stays **default-ON**. (jaxpr-equation count ≠ runtime.)
- **Real (small) regressor:** the mp=8 conditional-leaf carry NARROWED 81→74 leaves (None leaves → a slower GPU program shape). Fixed by `292a4431` (re-materialize the conditional leaves at the operational entry; **bit-identical**).

## Guardrails confirmed
- **#37 contract intact:** the public default-state mp=8 carry leaves remain `None` (re-materialization is only at the operational compute entry); the #37 mp=8 conditional gate tests stay green.
- **VRAM-neutral:** the 7 re-materialized leaves' real payload on the standard 66×159×44 grid is **10.65 MiB** (warmed peak +104 MiB ≈ 2%, dominated by other transients) — no material regression.

## Evidence
- `proofs/v018/perf_rootcause_opus.md` (Opus HLO/jaxpr-diff + ablation + authoritative canonical timing)
- `proofs/v018/perf_neutrality_v017_rerun_warmed_timing.json`, `perf_neutrality_v018_warmed_timing.json`
- `proofs/v018/gpt_verify_v017_warmed_timing.json`, `gpt_verify_v018_warmed_timing.json` (GPT independent double-check)
- `proofs/v018/perf_probe_*` (ablation probes: physics-off, rrtmg-direct, full-conditional-leaves)
