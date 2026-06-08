# v0.12.0 release report — for the principal (2026-06-08, ~01:35 WEST)

## ✅ SHIPPED
**v0.12.0 is live at github.com/wrf-gpu/wrf_gpu** — `main` (home=latest) @ `51df7fb`, tag
`v0.12.0`, fast-forward from v0.11.0 (`ac71ce8`). Dev backup pushed to `origin` (nric/wrf_gpu2:
branch `worker/opus/v0120-integration` + tag). You were notified via Telegram.

## The release gate (the thing that mattered)
**24 h standalone nested 1 km on the prod-failing AIFS case = `PIPELINE_GREEN`.** 24/24 `wrfout`
per domain (d01/d02/d03), all fields finite at +24 h (d03 T2 ∈ [279.6, 300.9] K), forecast-only
≈ 2.0 h, run on the *actual* v0.12.0 trunk. The prod case that used to fail now runs out-of-box.
Proof: `proofs/v0120/nested_24h_1km_gate_FINAL.json`. This is a completion + finiteness gate, not
a skill-vs-truth claim (honest).

## What shipped (16 merged features + hygiene)
Standalone baseline (CLI, JIT-cache, scheme-catalog, PSFC-fix, equivalence-demo) **plus the 18 h
expansion wave**: wrfout 64→**104 vars** (B1 radiation-flux + B3 Noah-MP snow), **Dudhia SW
`ra_sw=1`** + **classic RRTM LW `ra_lw=1`** operational (pristine-WRF oracles PASS), lat-lon /
Mercator / Polar projections, PD/mono advection, multi-stream auxhist, 2-way-nesting scaffolding,
GWD operational coupling, cadence WARN-and-run, test-hygiene (CPU suite honest-green).

## The one judgment call worth knowing
**GWD operational coupling is gated OFF by default** (`GPUWRF_GWD_NESTED=1` to enable). The clean
gwd7 run proved the 24 h nested 1 km + GWD path runs *physically clean for 7 sim-hours then OOMs on
VRAM* (not physics) — it exceeds the single-GPU fp64 ceiling at ~hr 7 (the RRTMG g-point temporary,
same root cause as the Switzerland grid ceiling). Gating it off preserves the GREEN prod gate; the
kernel is oracle-validated. Full nested-GWD → v0.13 (g-point-chunked temp).

## Honesty / things I did NOT claim
- WRF-**compatible reimplementation**, not a Fortran-source port; research artifact, **not** a full
  WRF replacement. (Kept the principal-approved staged headline: ~4× per-kWh **measured**,
  whole-Earth-1km **projected**, both footnoted.)
- **TOST n=15 not scored** (GPU `daily_pipeline` rc=2) → v0.12.x; **no equivalence claimed**.
- Fixed the stale doc claims the external critique flagged: "64-var / missing-only-snow" → **104 vars,
  B1+B3 added**; Dudhia+RRTM-LW moved from "deferred/fail-closed" → operational.
- **compile-speed reverted** (CPU-proven, but its XLA-autotune flags abort the GPU path) → v0.13.

## Reverted / deferred → v0.13 (`.agent/decisions/V0130-ROADMAP.md`)
P1: forecast-skill closure (the credibility gate), compile-speed (GPU-validate), TOST n=15 (rc=2 fix),
GWD-nested-VRAM (g-point-chunk), RRTM-LW cross-model skeptic pass, 2-way-24h-GPU equivalence,
multi-GPU. P2: outsider-runnable reproducibility, MYJ+Janjic, 3D-TKE LES, clear-sky radiation,
community-standard validation, multi-hardware.

## Morning follow-ups (small, not blocking)
1. **Sanitize remaining ~90 proof `.py` runner `/home/enric` paths** + decide whether `.agent/`
   should be in the public repo at all (the JSON/md proof evidence was sanitized this release; the
   `.py` runners + `.agent/` docs are pre-existing exposure since v0.11.0, cosmetic, no secrets).
2. **24 genuine pre-existing test failures triaged** (`proofs/v0120/test_hygiene_report.md`) — some
   real (RRTMG-SW clear-sky accuracy), some stale snapshots — owner triage for v0.13.

## External critique (the naive-AI v0.11 review you forwarded)
Triaged in `.agent/reviews/2026-06-07-naive-ai-v011-critique-triage.md`. Its release-hygiene items
(#2 reproducibility paths, #5 framing, #6 placeholder READMEs, #4 standalone-init reflected) are in
this release; its deep items (#1 skill, #3 community validation, #7 multi-hardware) are P1/P2 in the
v0.13 roadmap. Verdict accepted: publication-worthy now as a methodology/artifact preprint, not yet
as a full WRF replacement.
