# Post-0.1.0 Roadmap — Path to a True WRF v4 Port

Status: **ACTIVE plan, principal-directed 2026-05-31.** Supersedes the long-horizon
tail of `PROJECT-RESET-PLAN-FINAL.md` (M8–M23) with a release-cadenced, swarm-velocity
plan grounded in the honest gap analysis.

Single source of truth for the gap inventory: **[`publish/GPU_PORT_GAPS_TODO.md`](../../publish/GPU_PORT_GAPS_TODO.md)**
(GPT-5.5 code-grounded P0/P1/P2 audit, 2026-05-31). This doc adds the **sequencing,
release cadence, effort estimates, and work mode**.

## What v0.1.0 IS (the release we are finishing now)

A **validated single-domain GPU replay forecast** for the Canary Islands:
- d02 (3 km) + d03 (1 km Tenerife) driven by replayed CPU-WRF/Gen2 boundaries + land.
- WRF-faithful core: RK3 + split-explicit acoustic dycore (fp64), Coriolis, flux-form
  advection, Thompson microphysics, MYNN PBL, revised surface layer, RRTMG SW/LW.
- Idealized gates pass vs published benchmarks (Skamarock warm bubble, Straka density
  current) + WRF savepoints; real-case skill near CPU-WRF on T2 and (post-Coriolis) winds,
  multi-day stable, beats persistence.
- **NOT** a full standalone WRF replacement — see the gaps TODO. The honesty is the point.

0.1.0 close criteria: all tests green + 3 km/1 km real-case verdicts + full paper rewrite + tag.

## Effort table (calibrated: the validated core ≈ 1 week of swarm wall-clock)

Unit costs from the core week: one dycore-depth effort ≈ 3–4 d; one physics scheme ≈ ½–1 d;
pure I/O / orchestration ≈ a bit less. Wall-clock = at demonstrated multi-agent swarm pace.

| # | Item | Priority | Wall-clock | Risk / note |
|---|---|---|---|---|
| **P0-6** | Real-terrain / map-factor / boundary dynamics closure | **#1 (quality)** | 0.5–1 wk | HIGH — deep, subtle; finishes dycore Phase-B; tied to wind skill |
| **P0-1** | Live multi-domain nesting | #2 | 0.5–1 wk | Med — orchestration; interp already exists from replay |
| **P0-3** | Prognostic Noah-MP | P0 (standalone) | 0.5–1 wk | Med-High — large LSM; messy oracle (error bars skew high) |
| **P0-5** | Full `wrfout`/`wrfrst` + diagnostics | P0 | 2–4 d | Low — I/O engineering |
| **P0-4** | d01 Kain-Fritsch cumulus | P0 (live-d01) | 2–4 d | Low-Med — one scheme |
| **P0-7** | Conservation budgets + non-masking policy | P0 | 2–4 d | Low-Med — diagnostics + guard reporting |
| **P1-3** | RRTMG fidelity (topo-shade, slope-rad, real lat/lon) | P1 (quality) | 2–4 d | Med — island diurnal heating/cloud |
| **P1-4** | MYNN completeness (EDMF/cloud) | P1 (quality) | 2–4 d | Med — marine PBL / wind |
| **P1-5** | Thompson parity debts (adaptive-nstep, cloud-w sed) | P1 | 1–3 d | Low — partly in progress (#32) |
| **P1-8** | Precision-policy proof gates | P1 | 1–2 d | Low — declare mode + re-run gates |
| **P1-7** | Gravity-wave drag (`gwd_opt=1`) | P1 (d01) | 1–2 d | Low — verify load-bearing first |
| **P1-6** | Pos-def/monotonic advection + boundary order | P1 | 2–3 d | Low — baseline uses simple opt=1 |
| **P2-2** | Namelist compatibility checker | P2 | ~1 d | hygiene — fail loudly on unsupported |
| **P2-3** | Extra `wrfout` diagnostics | P2 | 1–2 d | as downstream needs |
| **P2-1** | Map-projection / grid generality | P2 | defer | restrict claim to Canary grids |
| **P1-1** | DA / FDDA / nudging | P2 (defer) | 0.5–1.5 wk if ever | not in Canary baseline namelist |
| **P1-2** | Physics scheme breadth | P2 (document) | doc ~1 d | only if changing namelist |
| **S1** | **Multi-GPU single-node domain decomposition** (sharded stencils + halo exchange) | **0.2.0 (scalability)** | ~1–1.5 wk | Med-High — halo interface pre-designed (`contracts/halo.py`); extension not rearchitecture |
| **P0-2** | **Native init (WPS/real.exe replacement)** | **LAST** | 1.5–3 wk *(or ~2–3 d if real.exe kept)* | **HIGHEST risk** — do AFTER 0.2.0 |

## Release cadence & sequencing (principal-directed 2026-05-31)

1. **v0.1.0** — finish now: all tests + 3 km/1 km verdicts + full paper rewrite + tag.
2. **Immediately after 0.1.0** — start **Opus 4.8 (max effort) agent(s)**:
   - **P0-6 FIRST** (real-terrain/map-factor/boundary dynamics — the top forecast-quality lever).
   - **P0-1 NEXT** (live multi-domain nesting), after P0-6, **or in parallel iff file
     ownership is disjoint** (P0-6 = core dycore operators; P0-1 = runtime orchestration —
     freeze interfaces first per the Operating Rules; P0-1 must build on a correct dycore,
     so prefer sequential if there is any overlap).
3. **0.1.x cadence** — cut a `0.1.x` release after each completed table item (continuous,
   honest, incremental).
4. **v0.2.0** — **all table items complete EXCEPT native init (P0-2).** I.e. P0-1, P0-3,
   P0-4, P0-5, P0-6, P0-7 + the P1 quality items + **S1 (multi-GPU domain decomposition)**
   closed; standalone forecast still permitted to consume `real.exe`-produced static/IC/
   boundary inputs.
5. **After 0.2.0** — **native init (P0-2) LAST**, deliberately, to avoid hiccups and long
   delays on the riskiest, never-before-done item. Pragmatic default: **keep `real.exe`** as
   cheap one-shot CPU preprocessing (it is run nightly already); full WPS/real.exe replacement
   is the maximalist "zero CPU-WRF dependency" goal, not a forecast-compute blocker.

## Work mode (per item)

Established escalation (`AGENTS.md` operating model):
1. **Opus 4.8 max-effort** in-process agent(s) implement; manager reviews diff + runs gates + merges.
2. **If the agent has issues** → bring in **GPT-5.5 xhigh** (codex) for a second opinion / debug.
3. **If still stuck** → **split the problem + empirical bisection bug-hunt** (disable components
   sequentially), and/or council step-back (GPT + agy). Never theory-spiral past two failed hunts.

Every item closes with a proof object (idealized gate, WRF savepoint, conservation budget, or
real-case skill + persistence baseline) and a 0.1.x/0.2.0 release note. No "done" without proof.

## Hardware portability & scaling (added 2026-05-31, principal-directed)

Grounded in a code audit (no `shard_map`/`pmap`/`jax.sharding`/`Mesh`/`jax.distributed` anywhere;
no `sm_*`/Blackwell/32 GB/`XLA_FLAGS` device-specifics; `contracts/halo.py::apply_halo` is a
designed-in no-op with an MPI-compatible call shape):

- **Single H100 / H200 — compatible AS-IS, zero source changes.** Pure JAX/XLA recompiles for
  Hopper (sm_90) like it does for Blackwell (sm_120); only needs a standard `jax[cuda12]` install.
  Expected to run **faster** (full-rate fp64 vs throttled consumer fp64; ~2–2.7× HBM bandwidth on
  the bandwidth-bound core) and handle **larger** single-GPU domains (80/141 GB vs 32 GB — just
  raise `XLA_PYTHON_CLIENT_MEM_FRACTION`). Speedup-vs-CPU is hardware-specific → re-measure on the
  actual Hopper box (we have none). **No roadmap item needed — adoption is unblocked today.**
- **Throughput scaling (ensembles / many independent forecasts) across a multi-GPU node** —
  trivial today: one single-GPU job per GPU, linear. No code change.
- **One forecast across multiple GPUs (domain decomposition)** — the only real change → **S1**,
  added to the v0.2.0 list. Needed only for the upper tail (domains too big for 141 GB:
  continental@1 km / global, or strong-scaling for latency); a single H200 already serves most
  high-power-GPU users. Pre-designed-for (the frozen halo interface), so it's an extension:
  `jax.sharding.Mesh`/`NamedSharding` + per-stencil halo exchange via `shard_map` + collective
  permutes (or `jax.distributed` multi-node); keep the host-loop scan collective-correct.

## Honest caveat on the estimates

The core week had unusually clean oracles (analytic idealized cases + WRF savepoints). The two
items with the messiest validation — **native init (P0-2)** and **prognostic Noah-MP (P0-3)** —
have error bars that skew high; they are the most likely to overrun the table.

**Totals at swarm pace:** quality-critical subset (P0-6 + P1-3 + P1-4 + finish P1-5) ≈ 2 wk;
full pre-native P0/P1 chain (→ v0.2.0) ≈ 3–5 wk; native init tail ≈ +1.5–3 wk; full true port
≈ 2–3 months — vs the 32–45 weeks in the human-team-calibrated reset plan.
