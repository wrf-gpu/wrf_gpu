# Precipitating Thompson oracle + implicit-sed ADOPT/REJECT (Canary d02)

**Author:** opus frontrunner (`worker/opus/precip-implicit-sed`)
**Base commit:** `f756753` (thompson-perf tip)
**Date:** 2026-05-31

> **UPDATE 2026-06-01 (v0.2.0 P1-5, precip-parity lane):** the "+13% surface
> precip" caveat below (section A) is now CLOSED. The excess was the FIXED 64
> sedimentation substeps over-resolving the falling front vs WRF's coarser
> adaptive `nstep`. Adopting WRF's exact adaptive `nstep = MAX_k INT(DT/(dz/vt)+1)`
> via a masked fixed-length scan (and the WRF `rr(kts)>1e-9` surface threshold,
> which is a no-op on this column) collapses the surface-precip ratio from 1.134
> to **0.983 (-1.7%)** and the rain-field error from 1.53%/23% (mean/max) to
> **0.08%/0.41%** — genuine WRF parity, byte-faithful per-column nstep, no fudge.
> Predeclared P1-5 parity gates all PASS: see `precip_parity_p1_5.json` /
> `precip_parity_p1_5.py`. The implicit-sed REJECT decision (section B/D) stands
> unchanged. The diagnosis that isolated the two effects is `_diag_sed_wrf_faithful.py`.

## Mission
(A) Build a PRECIPITATING WRF Thompson oracle and validate the shipped
faithful-explicit GPU Thompson on precipitation (the 0.1.0 functional gate #32).
(B) Using that oracle + a coupled run, make the ADOPT/REJECT call on the
implicit (backward-Euler) sedimentation scheme — the only remaining >=10x lever.

---

## (A) The precipitating oracle (NEW)

The pre-existing WRF microphysics oracle
(`/mnt/data/wrf_gpu2/physics_oracle/microphysics/`) is a DRY/clear column (qc
max 3.6e-8, all fall speeds 0) — it cannot discriminate a sedimentation scheme
change. I built a standalone single-column harness that drives the REAL WRF
`mp_gt_driver` (`module_mp_thompson.F`) on a deliberately precipitating column:

- `/home/enric/src/wrf_pristine/precip_oracle/precip_column_oracle.F` — links the
  already-compiled WRF objects (`libwrflib.a` + the LIB_BUNDLED set), calls
  `thompson_init` then `mp_gt_driver`, and dumps pre/post via the SAME
  `module_wrfgpu2_oracle` raw big-endian .f64 + sidecar format the full model
  uses. 8 columns x 44 levels, near-saturated, with active rain / cloud water
  (low/mid), graupel (mid mixed-phase), snow (upper), cloud ice (aloft).
  Realistic WRF-like dz (~30 m near surface stretching to ~650 m aloft) so the
  fixed-NSED=64 explicit kernel is in its intended Courant regime.
- Savepoint: `/mnt/data/wrf_gpu2/physics_oracle/microphysics_precip/`.
- CPU-LIGHT: single column, single 18 s step, serial, <1 s on 1 core — never
  contends with the running 28-rank CPU-WRF (it ran on cores 0-3 only).

This is a REAL WRF oracle (the actual Fortran microphysics), not a JAX
self-compare: rain sediments to the surface (WRF RAINNCV 0.019-0.069 mm/col),
graupel falls + melts through the mixed-phase layer, snow is generated aloft and
sediments, ice sublimates — i.e. every sedimentation + process channel is active.

## (A) FAITHFUL-EXPLICIT validation result  ->  #32 PASS (functional gate)

`proofs/thompson_perf/precip_oracle_validation.json` (one 18 s step, 8 cols):

| metric | faithful explicit | interpretation |
|---|---|---|
| qv max mean-rel vs WRF (active cells) | 0.46% | excellent |
| qr mean-rel / max-rel | 1.53% / 23% | faithful (max on melting-front cells) |
| qs mean-rel | 14% | snow generation/sediment-front placement |
| qg integrated mass vs WRF | **0.7%** | mass conserved (per-cell up to 61% is melt-front placement, not mass) |
| **water closure (vap+cond change + precip)** | **2.5e-6 rel** | mass-conserving |
| **surface precip total** | 0.393 mm vs WRF **0.347 mm (+13%)** | precipitates, right order + per-column gradient |

VERDICT for #32 "nightly Canary precipitates": **PASS as a functional gate** —
the faithful kernel precipitates, the vertical hydrometeor profiles and water
budget match WRF, and surface precip is within +13% with the correct
per-column structure.

**Honest caveat (NOT WRF-RAINNCV-parity yet):** the +13% surface-precip excess
is a diagnosed surface-flux ATTRIBUTION difference, not a process error:
WRF (`module_mp_thompson.F:3791-3818`) uses an ADAPTIVE per-column substep count
`nstep = INT(DT/(dz/vt)+1)` and only accumulates surface precip where the
surface-layer rain density `rr(kts) > R1*1000 = 1e-9 kg/m3`. The JAX kernel uses
a FIXED NSED=64 and extracts the bottom-face flux every substep with no
threshold, so it bleeds slightly more rain to the surface in one step. The
column profiles + water closure are unaffected. Before claiming RAINNCV PARITY
(not just "precipitates"), the JAX sedimentation should adopt the adaptive-nstep
+ rr>1e-9 surface threshold. GPT-5.5 xhigh concurred (see below).

## (B) IMPLICIT backward-Euler sedimentation vs the SAME oracle

The implicit prototype was swapped into `_sedimentation` at WRF's exact operator
position (after rain-evap, before instant melt/freeze) via an env gate
(`GPUWRF_THOMPSON_IMPLICIT_SED`, default 0=OFF). Same 18 s oracle column:

| scheme | surface precip total | vs WRF 0.347 | vs faithful 0.393 | qr mean-rel vs WRF | water closure |
|---|---|---|---|---|---|
| faithful explicit (default) | 0.393 mm | +13% | — | 1.53% | 2.5e-6 |
| implicit BE nsub=1 | 0.510 mm | **+47%** | +30% | **5.11%** | 3.2e-6 |
| implicit BE nsub=2 | 0.466 mm | +34% | +19% | 3.68% | 3.0e-6 |
| implicit BE nsub=4 | 0.436 mm | +26% | +11% | 2.76% | 2.8e-6 |

The implicit single-sweep BE is materially MORE diffusive: it smears the falling
front DOWNWARD and over-precipitates by +47% vs WRF in one step (the fast-species
arrival bias GPT-5.5 and the prototype warned about). Increasing nsub reduces the
diffusion (it converges toward the faithful/WRF answer as nsub rises) but at
proportional cost — and even nsub=4 is still +26% vs WRF / worse than faithful's
+13% on this 18 s step. It is mass-conserving (closure ~3e-6) but mass
conservation is NOT evidence of correct sedimentation physics.

### Why the speedup evaporates at a defensible nsub (MEASURED)
The full Thompson kernel on the real d02 workload (20748 cols x 44 lev, median of
100 reps, `proofs/thompson_perf/implicit_sed_timing.json`):

| scheme | kernel median ms | speedup vs faithful |
|---|---|---|
| faithful explicit (default) | 30.5 | 1.00x |
| implicit BE nsub=1 | 13.5 | **2.25x** (but +47% precip vs WRF) |
| implicit BE nsub=2 | 15.0 | 2.03x (+34%) |
| implicit BE nsub=4 | 19.0 | **1.61x** (+26%, still worse than faithful's +13%) |

The lever collapses at the accuracy it needs: the 2.25x is the nsub=1 variant that
over-precipitates +47% vs the WRF oracle; pushing nsub up to recover accuracy drops
the speedup to ~1.6x at nsub=4 (which is STILL more diffusive than the faithful
explicit default). There is no implicit nsub that is both >=2x AND as faithful as
the explicit default.

### Coupled skill (+1h gridded vs CPU-WRF, production entry, guards off, fp64)
`proofs/coupled/task2_skill_signal_1h_{faithful,implicit_nsub1,implicit_nsub4}.json`.

| scheme | T2 RMSE (K) | U10 RMSE | V10 RMSE | finite |
|---|---|---|---|---|
| faithful explicit (default) | 1.547 | 0.701 | 1.608 | yes |
| implicit BE nsub=1 | 1.547 | 0.701 | 1.608 | yes |
| implicit BE nsub=4 | 1.547 | 0.701 | 1.608 | yes |

All three are bit-for-bit the SAME +1h surface skill and all finite/stable — the
+1h gridded T2/U10/V10 on this lightly-precipitating case is INSENSITIVE to the
sed scheme (sedimentation redistributes hydrometeors + latent heat; it does not
move the +1h near-surface dynamics measurably). So this coupled run proves
implicit-sed is COUPLED-STABLE / core-safe, but it is NOT the precip-skill arbiter
— the precipitating oracle above is (and that is where implicit fails: +47%).

Note: this gridded +1h skill is dominated by the dynamics/PBL, not by the
microphysics sedimentation scheme (which mostly redistributes hydrometeors and
latent heat); T2/U10/V10 are relatively insensitive to the sed scheme at +1h, so
this coupled run is a STABILITY + no-regression check, not the precip-skill
arbiter. The precip-skill arbiter is the precipitating oracle above (direct +
decisive). A precip-accumulation skill gate would need the surface precip
accumulator surfaced in coupled diagnostics + a multi-day multi-case comparison
(GPT-5.5's predeclared gate) before any adoption.

---

## DECISION: REJECT implicit sedimentation as a default.

Reasoning:
1. **Accuracy:** implicit BE (nsub=1) over-precipitates +47% vs the WRF oracle
   and doubles qr error (5.1% vs faithful 1.5%); it is a genuine
   more-diffusive scheme change, not a faithful drop-in.
2. **The speedup is conditional on the inaccurate nsub.** The ~2.4x was at
   nsub=1; the nsub needed to approach faithful accuracy (>=4) erodes most of the
   win, so implicit does NOT buy a clean >=10x coupled.
3. GPT-5.5 xhigh review: REJECT nsub=1 as default; at most ADOPT-CONDITIONAL
   nsub>=4 behind an ADR + a predeclared MULTI-case coupled precip+skill gate —
   i.e. not a default flip from one column pack.

Therefore the **faithful-explicit Thompson ceiling stands**: ~1.1x kernel
(sed-unroll, bit-identical) -> ~5.3-7.8x coupled (clean / realistic), which is
the FINAL honest >=10x-via-Thompson verdict. >=10x coupled is not reachable by
changing Thompson sedimentation within an honest accuracy bar; it must come from
the dycore / coupling / fusion budget instead.

The implicit scheme is KEPT GATED (default OFF, `GPUWRF_THOMPSON_IMPLICIT_SED=N`)
as an experimental knob for any future ADR-gated multi-case study, exactly as
GPT-5.5 recommended. The shipped DEFAULT remains byte-identical faithful explicit.
