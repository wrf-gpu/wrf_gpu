# Validation — community-standard evidence an outside reviewer can reproduce

This project validates its JAX/XLA WRF-compatible reimplementation against the
**community-standard tests** an external meteorologist or NWP reviewer expects
from a dynamical-core + physics port: published idealized dycore benchmarks,
closed-domain conservation budgets, and bitwise restart. This document is the
companion to [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (which covers the physics
savepoint-parity collection); together they describe everything an outsider can
reproduce on a CPU-only machine, and exactly which claims need a GPU or data that
cannot be redistributed.

## TL;DR — one command

```bash
bash scripts/community_validation.sh
```

- Forces `JAX_PLATFORMS=cpu` (no GPU context is created — the GPU is owned by the
  live/nested forecast lanes).
- Requires **nothing beyond this repository**: no GPU, no WRF source tree, no corpus.
- Pins Python/JAX to CPU cores 0–3 (`taskset`); the nightly CPU-WRF runs own
  cores 4–31 and are never touched.
- Runs in ~3 minutes (the two idealized dycore integrations dominate; the
  conservation and restart gates are seconds).
- Exit `0` = every CPU community gate is green; non-zero = a gate failed.
- Emits the aggregator proof object `proofs/v013/community_validation.json` and
  prints a per-gate summary + the honest CPU-vs-GPU/data gap list.

The suite runs three community-standard gates:

| Gate | What | Needs |
|------|------|-------|
| 1 | Idealized dycore benchmarks — Straka 1993 density current + Skamarock/Bryan–Fritsch warm bubble vs the published spec | repo only |
| 2 | Closed-domain conservation budgets — dry-mass / total-water / moist-static-energy closure (fp64) | repo only |
| 3 | Bitwise restart — full state+carry+stochastic-seed NetCDF `wrfrst` write→read→compare bit-identity | repo only |

---

## Gate 1 — idealized dycore benchmarks (published WRF/community spec)

The suite re-runs, **on CPU**, the two canonical idealized dynamical-core
benchmarks every WRF-class model is expected to reproduce, via the existing
runner `gpuwrf.ic_generators.idealized` (no GPU required; `require_gpu=False`).
Each is checked against the published benchmark spec, not a self-compare.

### Straka et al. 1993 density current

A cold block (Δθ ≈ −15 K) collapses and propagates as a density current with a
sharp front and Kelvin–Helmholtz rotors — the standard test for cold-pool
propagation, sharp-gradient handling, and numerical diffusion control.

- Published spec: `dx = 100 m`, constant diffusion `ν = 75 m²/s`, integrate to
  `t = 900 s`; the front reaches ≈ 15 km and the flow develops the characteristic
  rotor train.
- References:
  - <https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html>
  - Straka et al. 1993 / Skamarock–Klemp verification, MWR
    <https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml>
- Checks (all must pass): all snapshots finite; `−25 ≤ min(θ′) ≤ −5 K`;
  `1 ≤ max|w| ≤ 50 m/s`; front position within ±2 km of 15 km at 900 s; rotor
  count 2–4; relative dry-column mass drift ≤ `1e-8`.
- Latest CPU result (recorded in the proof object): **PASS** — front 14150 m,
  `max|w| = 14.6 m/s`, `min(θ′) = −9.97 K`, rotor proxy = 4, relative mass drift
  = `2.25e-9`.

### Skamarock–Klemp / Bryan–Fritsch dry warm bubble

A compact +2 K dry thermal rises buoyantly through a neutral atmosphere — the
standard test for the buoyant response, acoustic-substep stability, left/right
symmetry, and mass conservation.

- Published spec: dry thermal, +2 K perturbation on a 300 K base, integrate to
  `t = 500 s`; WRF `em_quarter_ss` is the stock-ARW reference path.
- Reference: Skamarock, *Conservation/verification of the ARW solver*
  <https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf>
- Checks (all must pass): all snapshots finite; `0.5 ≤ max(θ′) ≤ 2.5 K`;
  `1 ≤ max|w| ≤ 30 m/s`; positive-θ′ centroid rises ≥ 500 m; horizontal centroid
  drift ≤ 250 m (symmetry); relative dry-column mass drift ≤ `1e-8`.
- Latest CPU result: **PASS** — `max(θ′) = 1.92 K`, `max|w| = 11.7 m/s`, thermal
  rise = 1924 m, horizontal drift = `1.3e-11 m` (essentially zero), relative mass
  drift = `0.0`.

> Note on precision: the idealized cases are run in **fp64**. The operational
> fp32-gated matrix loses the 2 K perturbation on a 300 K base and destabilizes
> the acoustic solve; fp64 is the correctness target for these dycore benchmarks
> (per ADR-007).

---

## Gate 2 — closed-domain conservation budgets

A closed domain (no lateral fluxes) must conserve dry mass exactly and total
water / moist-static-energy to machine precision in fp64. The suite runs the
existing CPU controlled budget gate (`tests/test_conservation_budget.py` over
`gpuwrf.diagnostics.conservation_budget`, with the recorded proof object
`proofs/p0_7/conservation_budget_cpu_controlled.json`).

Budget terms covered: dry-column mass, layer-summed dry mass, total water in air,
and moist static energy, on a deterministic non-unit map-factor grid.

Predeclared tolerances (the gate fails if any is exceeded):

| Quantity | Tolerance |
|----------|-----------|
| Closed-domain dry-mass relative residual | `1e-10` |
| Closed-domain total-water relative residual | `1e-8` |
| Controlled-CPU budget relative residual | `1e-12` |
| Controlled-CPU budget absolute residual | `1e-6 kg` |

Latest CPU result: **PASS** — closed-domain dry-mass relative residual `0.0`,
total-water relative residual `−2.45e-16`, moist-static-energy residual `0.0 J`.

> The open-domain 24 h lateral-boundary-corrected budgets (`1e-5` dry mass,
> `1e-4` water with precip/evap correction) are part of the same machinery but
> exercise a full forecast and are a GPU/corpus gate — see the gap list.

---

## Gate 3 — bitwise restart

A restart must be bit-identical: writing the full model state to a checkpoint and
reading it back must reproduce every field exactly. The suite runs the existing
structural round-trip `v0110_restart_proof._cpu_full_carry_roundtrip` plus the CPU
restart pytest suite (`test_p0_5_restart_full_carry.py`, `test_v0110_wrfrst_netcdf.py`,
`test_m7_restart_checkpoint_roundtrip.py`).

It writes a full synthetic carry — the complete prognostic **state** (56 `GPUWRF_STATE_*`
variables), the operational **carry** (14 `GPUWRF_CARRY_*` variables), the optional
Noah-MP / cumulus groups, and the stochastic-physics seed arrays
(`ISEEDARR_SPPT`, `ISEEDARR_SKEBS`, `ISEEDARRAY_SPP_{CONV,PBL,LSM}`) — to a
WRF-compatible NetCDF `wrfrst` file, reads it back, and asserts **byte-for-byte
identity** of every field. The on-disk schema also carries the standard WRF
restart variables (`U,V,W,T,P,PB,PH,PHB,MU,MUB,QVAPOR,…`, plus map factors,
`XLAT/XLONG`, `TSLB/SMOIS/SH2O/…`).

Latest CPU result: **PASS** — full-carry bit-identical `True`, stochastic-seed
bit-identical `True`, schema version `v0.11.0-wrfrst-netcdf-2`.

> Scope: this is the **structural** bit-identity gate (the checkpoint format is
> lossless). The multi-hour **forecast-continuity acceptance** gate — that a run
> split at hour N, checkpointed, and restarted produces the identical trajectory
> to an uninterrupted run — needs a GPU + the real corpus and is listed in the gap
> table below.

---

## CPU-vs-GPU/data gap — what an outsider runs on CPU vs what needs more

Honest accounting of what this suite proves on CPU from the repo alone, versus
what is out of scope because it needs an NVIDIA GPU or the (non-redistributable)
CPU-WRF corpus.

### CPU-reproducible from this repo alone
- Idealized dycore benchmarks (Straka density current, Skamarock/Bryan–Fritsch
  warm bubble) — Gate 1 above.
- Closed-domain dry-mass / total-water / moist-static-energy budget closure — Gate 2.
- Bitwise restart (full state+carry+stochastic-seed `wrfrst` round-trip) — Gate 3.
- CPU physics savepoint-parity proofs — `bash scripts/verify_reproducibility.sh`
  (Kessler, BouLac PBL, Dudhia SW, RRTM LW, WSM, Grell–Freitas, Tiedtke, coupled
  moist closure, vs vendored unmodified-WRF Fortran savepoints).

### Needs an NVIDIA GPU (out of scope here)
- Speedup / throughput / per-watt and multi-GPU (DGX) claims — require a GPU and
  profiler artifacts (`proofs/perf/*`, `proofs/multigpu_dgx/*`).
- 1 km nested live-forecast stability gates (d03) and GWD-nested gates.
- Multi-hour restart **forecast-continuity acceptance** (restart trajectory ==
  uninterrupted trajectory) — the structural bit-identity above is CPU; the
  trajectory match needs a GPU + corpus.
- Open-domain 24 h LBC-corrected conservation budgets (full forecast).

### Needs the purged CPU-WRF corpus (not redistributable; `/mnt/data`)
- TOST operational equivalence vs 28-rank CPU-WRF (`proofs/m20/*`) — needs real
  CPU-WRF `wrfout` + AIFS forcing.
- Multi-day operational skill-vs-obs gates and station scoring.

---

## Proof object

`scripts/community_validation.sh` writes `proofs/v013/community_validation.json`
(schema `v013_community_validation_v1`): overall pass/fail, per-gate verdicts and
values, the published benchmark spec + reference URLs for the idealized cases, the
predeclared conservation tolerances, the restart schema, and the machine-readable
CPU-vs-GPU/data gap list. It records `jax_backend: "cpu"` and `cpu_only: true`.
