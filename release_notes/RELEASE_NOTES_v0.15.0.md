# Release notes — wrf_gpu v0.15.0

**v0.15 is the kernel-architecture + WRF-fidelity release — NOT an end-to-end
speedup release.** It delivers the **final fp64 GPU kernel** for the project
(adversarially confirmed near-optimal), WRF-fidelity fixes (MYNN condensation
`niter`, Thompson cold-collection), **cleaner GPU↔CPU identity than v0.14**, and
an honest, measured characterization of the kernel.

> **Honest performance framing — read this first.** The 72 h end-to-end
> benchmark is **~parity total-wall** (Switzerland **0.99×**, Canary **1.04×** vs
> 24-rank CPU-WRF), with **forecast-only stepping ~1.05–1.20×** and **warmed
> steady-state ~1.51× vs 24-rank (~1.29× vs 28-rank)** — the latter is the
> operationally-relevant figure for repeated/long runs, since the ~parity
> total-wall is dragged down by the one-time ~15-min fp64 compile the CPU never
> pays (viability sprint `proofs/perf/v015/viability/`). The per-step
> MYNN-condensation `niter` win is real (~1.4× on that isolated kernel) but is
> **diluted in the full pipeline** to ~1.05–1.20× and offset by a heavier
> one-time compile (~8–12 min) and **higher peak VRAM** (Switzerland 20.5→22.9
> GiB, Canary 21.1→29.8 GiB). **No large-grid / 1 km speedup is claimed.** The
> genuine speedup AND 1 km scalability both require one lever — the
> **fp32-operational-state restructuring** (ADR-007/031) — which halves the VRAM
> arena and removes the mixed-precision compile pathology; it is a separate
> major milestone, deferred. v0.15 ships the fp64 kernel at its honest ceiling
> and names that next lever precisely.

## What v0.15 delivers

### 1. WRF-fidelity fixes (default-on)
- **MYNN-EDMF condensation `niter` 50 → 16** (`GPUWRF_MYNN_COND_NITER`, default
  16) — a **fidelity fix**: WRF's own loop early-exits ("usually converges in
  < 8 iterations", `module_bl_mynnedmf.F`); the port's fixed 50 was the
  deviation. Isolated MYNN-condensation kernel ~1.4×; **full-pipeline
  forecast-only ~1.05–1.20×, total-wall ~parity** (the rest of the step + the
  one-time compile dominate). Identity-green at 0.06–0.15 % of frozen limits.
- **Thompson cold-collection** (`GPUWRF_THOMPSON_COLD_COLLECTION`, default on) —
  WRF-faithful rain-collecting-snow + rain-collecting-graupel sinks, bit-exact
  tables decoded from the pristine WRF `.dat` files; closes the column-integrated
  rain→graupel conversion to ~3 % and moves Switzerland RAINNC 5.99 → 5.08 mm
  (−15 %; still a carried bounded diagnostic — see §4).

### 2. The final fp64 kernel architecture (verdict: near-optimal)
An adversarial kernel-completeness review (re-checking every prior claim against
the project's own measured artifacts) confirms the kernel is **near-optimal
without megakernels** and **`FIXABLE_WITHIN_ARCH`** — the SoA pytree, C-grid, and
XLA codegen are not the bottleneck. The step is **device-bound** (~4 % idle), and
the host-launch-removal hypothesis was actively **falsified** (whole-step
CUDA-graph capture works but is wall-neutral). The cuSPARSE PCR tridiagonal solve
and column-tiling for radiation / MYNN-BouLac are already live. The review also
**corrected two over-optimistic numbers** from earlier probes (there is no
~16 ms device floor — the real cond16 floor is ~112–116 ms/step; and large grids
do **not** give a 6–10× per-cell win).

### 3. Honest kernel characterization (new — plots + tables)
A reproducible characterization for review and the manifest:
per-phase timing breakdown @128², throughput / per-cell-cost / peak-VRAM scaling
with grid size, compile-overhead amortization vs forecast length, and the
asymptotic large-grid speedup limit (honest ~1.6× full-step floor vs 28-rank CPU
— **not** 6–10×, with the fp64 hardware caveat: a GeForce 5090 in fp64 is ~0.77×
the CPU's ALU peak; its edges are bandwidth + parallelism).
- Tool: `proofs/perf/v015/build_kernel_characterization.py`
- Tables + narrative: `proofs/perf/v015/kernel_characterization.md`
- Plots: `docs/assets/v015/kernel/{phase_breakdown_128,grid_scaling,compile_amortization,asymptotic_largegrid}.png`

### 4. Identity + physics fixes (cleaner than v0.14)
- **MUB/PB nest-base-state seam fixed** — 250.7 → **0.0078 Pa** (the nest packs
  its own static base ring; WRF never forces nest base state).
- **72 h GPU-vs-CPU-WRF field-parity, both regions, stable to h72, all fields
  finite — 9/10 within frozen tolerance per region**, and v0.15 is **cleaner than
  v0.14 at the atlas level (1 tolerance failure per region vs v0.14's 3)**
  (DZS/ZS now emitted; MUB/PB seam fixed).
  - **Switzerland d01:** lone out-of-envelope field **RAINNC 5.08 mm vs 1.0 mm**
    (cold-collection moved it −15 % toward the bound; bounded precip diagnostic,
    drawn red honestly). Worst non-RAINNC field U10 at 0.76× limit.
  - **Canary L2 d02:** lone out-of-envelope field **QVAPOR 1.44e-3 vs 1.0e-3**
    (carried, **no regression** vs v0.14's 1.45e-3; MYNN marine-entrainment class
    → 0.16 lane). Worst non-QVAPOR field U10 at 0.77× limit.
- **No run-aways (long-horizon non-escalating-divergence check).** Both carried
  fields are **BOUNDED / non-escalating over 72 h**, NOT stability run-aways:
  Switzerland RAINNC saturates (late divergence slope = 4.6 % of early; max 1.13×
  the precip field's own spatial std), Canary QVAPOR is flat/negative-slope
  (max 0.47× the oracle's moisture spread). Every other field is non-escalating
  too. So the two carries are a **tight-per-cell-tolerance miss correctly carried
  to 0.16, not a stability failure.** This is a SECOND gate ADDED alongside (not
  replacing) the strict frozen tolerance, which still draws the carry red.
  Verdict: `proofs/v015/LONG_HORIZON_DIVERGENCE_VERDICT.md`.
- Identity-proof dashboards (5 plots/region + a dual-gate long-horizon-divergence
  panel showing "NO RUN-AWAY"; the carried field still drawn red under the strict
  gate, not hidden): `docs/assets/v015/identity_proof/{switzerland_d01,canary_l2_d02}/`.

### 5. Benchmark (honest — single cold 72 h run)
| region | GPU total-wall | CPU baseline | total-wall | forecast-only | peak VRAM |
|---|---|---|---|---|---|
| Switzerland d01 | 2941 s | 2906 s (24-rank) | **0.99×** | ~1.20× | 22.9 GiB |
| Canary L2 d02 | 8414 s | ~8713 s (28-rank) | **1.04×** | ~1.05× | 29.8 GiB |

The ~8–12 min one-time compile is included in total-wall; for warm/operational
repeated runs the forecast-only figure governs. Peak VRAM rose vs v0.14
(niter16 temps + nested arena) — a watch-item the fp32-operational milestone
addresses (it ~halves the dycore arena).

## Honest scope — what v0.15 does NOT do, and the single next lever

Everything v0.15 could not deliver traces to **one root: the fp64 operational
state.**
- **fp32 perf (fp32-BouLac ~1.67× + an O(nz)-BouLac memory win)** — both are
  proven correct in isolation (independent WRF oracles to machine epsilon) but
  hit an XLA mixed-precision "very-slow-compile" pathology in the full forecast
  jit; they ship as **opt-in flags** (`GPUWRF_MYNN_BOULAC_FP32`,
  `GPUWRF_MYNN_BOULAC_ONZ`), default off.
- **1 km large grids do not fit on a single 32 GiB RTX 5090.** The binding
  constraint is the **intrinsic ~21.7 GiB fp64 dycore working set** (~394
  coexisting full-grid fp64 temporaries; XLA's own rematerialization cannot go
  below 21.6 GiB) — not a tileable transient. The largest stable single-GPU grid
  is ~125,928 cols.

**The single structural fix for both is an fp32-operational-state restructuring
(ADR-007 / ADR-031)** — it halves the VRAM arena (1 km fits) and removes the
mixed-precision compile pathology (the 1.67× lands). It is a pervasive precision
change with a full identity re-gate — the defined **next major lever**, not a
v0.15 add-on. (Multi-GPU horizontal sharding is the alternative scalability
path.)

## Carried forward unchanged
All v0.14 capability (72 h cell-for-cell WRF identity in two regions, the
identity-proof system, the venting theta-clamp fix, advance_w WRF-faithfulness,
2-D Smagorinsky, RAINNC all-phase convention, DZS/ZS) and the v0.11–v0.13 stack
(live multi-domain nesting, restart continuity, conservation-closed budgets,
MYNN-EDMF, RRTMG, KF/BMJ/Grell cumulus, standalone CLI, fail-closed scheme
catalog) carries forward.

## Known issues / scope boundaries
| ID | Summary | Severity |
|---|---|---|
| v0.15 perf scope | fp32-BouLac (~1.67×) + O(nz)-BouLac are opt-in, blocked by the fp64 mixed-precision compile pathology → fp32-operational milestone. | Documented next-lever |
| v0.15 large-grid | 1 km grids do not fit on one 32 GiB GPU (intrinsic ~21.7 GiB fp64 dycore working set); ceiling ~125,928 cols. Needs fp32-operational or multi-GPU sharding. | Documented next-lever |
| QVAPOR (Canary) | MYNN marine entrainment-depth deficit (surface flux exonerated); carried, → 0.16 stability lane. | Bounded carry |
| RAINNC (Switzerland) | Cold-collection added (bulk closed); per-cell/number refinement → 0.16. Final 72 h value **5.08 mm** (vs 1.0 mm bound; −15 % from v0.14's 5.99 mm). | Bounded carry |
| KI-9 / KI-3 / KI-5 | Carried from v0.14 (forecast-skill equivalence; 104-var wrfout subset; powered TOST deferred). | Carry-overs |
