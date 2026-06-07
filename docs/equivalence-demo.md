# GPU-vs-CPU-WRF equivalence demo

A self-serve script that lets a skeptic *run and check* that the JAX GPU port
reproduces a retained CPU-WRF forecast — converting "equivalence is asserted"
into "equivalence you can run and verify yourself."

- Script: [`scripts/equivalence_demo.py`](../scripts/equivalence_demo.py)
- Authoritative proof object (post-PSFC-fix re-run, warm cache):
  [`proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`](../proofs/v0120/equivalence_demo_20260509_d02_FINAL.json)
  (the script's `--out` default writes `equivalence_demo_20260509_d02.json`; the
  `_FINAL` object is the re-run after the WRF-faithful PSFC surface-extrapolation
  fix landed, and is the one this page reports.)

## What it does

1. Runs the GPU port forecast through the **validated replay path**
   (`python -m gpuwrf.cli run --input-dir <case> --domain d02 ...`). In replay
   mode the GPU's initial state and lateral boundaries are taken from the SAME
   CPU-WRF `wrfout` history that we then compare against.
2. Loads each generated GPU `wrfout` and the CPU-WRF `wrfout` of the same
   timestamp.
3. Compares GPU vs CPU at **all grid points and all output timesteps**, per
   field — surface `T2`/`U10`/`V10`/`PSFC`/`RAINNC` and 3D
   `U`/`V`/`W`/`T`/`QVAPOR` — reporting per-timestep and pooled RMSE, mean bias,
   and max-absolute-difference.
4. Emits an **EQUIVALENCE VERDICT** against predeclared per-field tolerances
   (below).
5. Reports the **GPU-vs-CPU wall-clock speedup** (the demo payoff), using the
   retained CPU-WRF d02 per-step timing from the run's RSL logs.

## How to run

This demo compares the GPU port against **your own** retained CPU-WRF run, so
`--case-dir` is **required** and must point at a directory you have on disk
(its `namelist.input` plus hourly `wrfout_<domain>_*` history — see
[Reference case](#reference-case) for the format). There is no internal
scheduler, GPU mutex, or canairy-only path involved: a fresh clone just runs
the command below.

```bash
cd <repo>
PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/equivalence_demo.py \
    --case-dir /path/to/your/cpu_wrf_run \
    --domain   d02 \
    --hours    24 \
    --out      proofs/v0120/equivalence_demo.json
```

Flags:

| flag | meaning |
|---|---|
| `--case-dir` | **required** — a retained CPU-WRF run dir (its `namelist.input` + hourly `wrfout_<domain>_*`). |
| `--domain` | domain id to run/compare (default `d02`). |
| `--hours` | forecast lead hours to run and compare (12–24 h is a sensible single forecast). |
| `--out` | path for the verdict + stats proof JSON. |
| `--gpu-output-dir` / `--scratch-dir` | optional disk-backed dirs (default: temp dirs, cleaned up). Keep them off `/tmp` tmpfs. |

Prerequisites are the same as a normal forecast (see
[quickstart.md](quickstart.md)): a CUDA GPU with ~26 GiB free VRAM for d02 fp64,
a JAX CUDA build, and a local NVMe scratch path. The first launch pays the
~5-minute cold JIT compile. The script exits 0 on an `EQUIVALENT` verdict and
non-zero otherwise.

> **Running on your own machine — no internal scheduler needed.** This demo
> shells out to the public `python -m gpuwrf.cli run` entrypoint directly. It
> does **not** require any GPU mutex, lock wrapper, or canairy-internal
> infrastructure; on a single-GPU machine you just run the command above. (On
> the project's own *shared* workstation the maintainers have an internal
> multi-job GPU serializer at `/tmp/wrf_gpu_run.sh`; if — and only if — that
> wrapper exists on your machine you may prefix it, e.g.
> `/tmp/wrf_gpu_run.sh taskset -c 0-3 env PYTHONPATH=src python scripts/equivalence_demo.py ...`.
> A normal tester does not have it and does not need it.)

## Predeclared tolerances

These are fixed in the script header *before* the comparison, so the verdict
cannot be moved to fit the data. They are **operational** limits (comparable to,
or tighter than, the run-to-run / IC-uncertainty spread of CPU-WRF itself for a
short boundary-forced regional forecast), **not** round-off limits. A field
PASSES iff its **pooled** (all hours, all grid points) RMSE is at or below its
declared tolerance; the overall verdict is `EQUIVALENT` iff every compared field
passes.

| Field | RMSE tolerance | Notes |
|---|---|---|
| `T2` | 1.5 K | 2 m temperature |
| `U10`, `V10` | 1.5 m s⁻¹ | 10 m wind components |
| `PSFC` | 120 Pa | surface pressure (~0.1% of ~100 kPa) |
| `RAINNC` | 1.0 mm | accumulated grid-scale precip |
| `T` | 1.5 K | 3D perturbation potential temperature (θ − 300) |
| `U`, `V` | 1.8 m s⁻¹ | 3D horizontal wind components |
| `W` | 0.30 m s⁻¹ | 3D vertical velocity (small magnitude, noisy) |
| `QVAPOR` | 1.0×10⁻³ kg kg⁻¹ | 3D water-vapour mixing ratio |

## Observed result on the default case (24 h, d02, 20260509_18z)

Run on 2026-06-07 after the WRF-faithful PSFC surface-extrapolation fix landed
(proof:
[`proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`](../proofs/v0120/equivalence_demo_20260509_d02_FINAL.json),
GPU `cuda:0`, replay path, warm JIT cache). Pooled RMSE over all 24 hourly
timesteps and all grid points:

| Field | pooled RMSE | tol | pooled bias | verdict |
|---|---|---|---|---|
| T2 | 0.484 K | 1.5 K | +0.195 K | **PASS** |
| U10 | 2.237 m s⁻¹ | 1.5 | +1.491 | EXCEEDS |
| V10 | 2.441 m s⁻¹ | 1.5 | −1.521 | EXCEEDS |
| PSFC | 415.3 Pa | 120 | −407.2 | EXCEEDS |
| RAINNC | 0.501 mm | 1.0 | −0.034 | **PASS** |
| T (θ′) | 2.040 K | 1.5 | +0.119 | EXCEEDS |
| U | 3.167 m s⁻¹ | 1.8 | +1.660 | EXCEEDS |
| V | 8.130 m s⁻¹ | 1.8 | −5.417 | EXCEEDS |
| W | 0.126 m s⁻¹ | 0.30 | −0.028 | **PASS** |
| QVAPOR | 5.67×10⁻⁴ kg kg⁻¹ | 1.0×10⁻³ | +4.5×10⁻⁵ | **PASS** |

**Overall verdict: NOT_EQUIVALENT** (6 of 10 fields exceed tolerance). This is
the honest current state, reported as-is — and it is the post-fix re-run, not
a pre-fix snapshot. Read it precisely:

1. **PSFC is improved but still out of bar, and the residual is now dynamical,
   not diagnostic.** The pooled PSFC RMSE dropped from **707.8 Pa → 415.3 Pa**
   once the WRF-faithful surface-pressure extrapolation
   ([`proofs/v0120/psfc_extrapolation_proof.json`](../proofs/v0120/psfc_extrapolation_proof.json),
   `PSFC = p8w(kts)` from the total-geopotential faces, per
   `module_surface_driver.F` / `module_big_step_utilities_em.F`) replaced the
   old `p0`-based surface value. That fix closed the **systematic ~29 Pa
   diagnostic offset** in the internal surface-pressure definition (the
   extrapolation proof shows the internal PSFC bias going 328 Pa → −29 Pa). The
   **residual PSFC excess still seen here is no longer a constant diagnostic
   offset**: the per-lead PSFC bias is −295 Pa at h1, swings to −485 Pa around
   h6, relaxes, and re-grows — it **tracks the developing wind/mass divergence
   over the run**, not a fixed reference difference. In short: the surface
   extrapolation is now WRF-faithful; the remaining PSFC gap is **driven by the
   dynamical divergence**, dominated by the wind field, and is **not** an
   independent diagnostic bug. PSFC is **not** equivalent at 24 h.
2. **Winds dominate the verdict and diverge with lead time.** U10/V10/T/U/V
   start **within (or near) tolerance at short lead and grow monotonically** —
   the 3D meridional wind V is essentially identical at h1 (RMSE 0.17 m s⁻¹) and
   grows to ~11 m s⁻¹ by h19, drifting roughly **3× faster than U**. This is
   genuine lead-time error growth between two independent integrators,
   concentrated in the wind field, strongest in V. **Winds are not equivalent at
   24 h.** T2, W, QVAPOR and RAINNC stay inside tolerance for the full 24 h.

Do not read this page as "PSFC fixed" or "winds equivalent." The honest summary
is: **short-lead fields track CPU-WRF within tolerance; by 24 h the run is
`NOT_EQUIVALENT`, driven by wind divergence, and PSFC remains out of bar because
its residual is dominated by that same dynamical divergence** (after the
diagnostic offset was removed).

**Speedup (this demo):** the warm-cache GPU run integrated the 24 h d02 forecast
in **561.3 s** (forecast-only) vs an estimated **2393.2 s** for the CPU-WRF d02
solver (from the retained RSL per-step timing, d02 model step 6 s × 14 400
steps) — **~4.26×**. The earlier **cold-compile** run of the same demo took
**1408.6 s → ~1.70×**; the difference is entirely the persistent JIT cache (a
cold ~5-minute XLA compile vs a ~10 s cache read) plus IO/case-build overhead,
not a numerics change. This is a same-card, fp64, single-forecast **real-user
wall-clock** comparison of the d02 main solver only; it is **not** the
warm/compute-only kernel speedup (~5×, band 5–8×, dt-parity floor ~3.2×) quoted
in [`PERFORMANCE.md`](PERFORMANCE.md) and
[`../proofs/perf/speedup_denominator.md`](../proofs/perf/speedup_denominator.md).
See the speedup-reconciliation paragraph in [`PERFORMANCE.md`](PERFORMANCE.md)
for how the cold-real-user, warm-real-user, and warm-kernel numbers relate.

A documented exceedance with its numbers is exactly the point of a self-serve
demo. The PSFC diagnostic offset is now closed; the remaining gap is the
lead-time wind divergence, which is the tracked follow-up (KI-9 in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md)).

## What it proves — and what it does not

**Designed to test:** whether, given the *same* initial and lateral-boundary
conditions, the independent JAX GPU integrator reproduces the retained CPU-WRF
(Fortran WRF v4) forecast field-by-field, grid-point-by-grid-point,
hour-by-hour, within the predeclared operational tolerance, while running faster
on the GPU. It is an honest cross-implementation check that emits an
`EQUIVALENT` / `NOT_EQUIVALENT` verdict from the data (see the observed result
above — the current default-case verdict is `NOT_EQUIVALENT`, with the
exceedances and likely causes documented).

**Does NOT prove / important caveats:**

- This is **numerical / operational equivalence within a predeclared
  tolerance, NOT bitwise identity against Fortran.** Two independent integrators
  of the same PDE differ at the round-off level and diverge slowly under chaotic
  dynamics; the test is whether they stay within an operationally meaningful
  bound, not whether they match bit-for-bit.
- It is **not a self-compare.** The reference is an independent Fortran CPU-WRF
  run; the GPU port only borrows the ICs/LBCs from it (exactly the operational
  replay use case). It is also not a comparison of the model against its own
  output.
- The GPU port is separately **bitwise self-deterministic** (same inputs →
  identical outputs run-to-run). That property is asserted elsewhere in the
  suite and is **not** what this script measures.
- Boundary-forced replay keeps the two solutions tied at the domain edges,
  which is the regime in which equivalence is claimed (short-range, LBC-driven
  regional NWP). It does **not** speak to free-running, unbounded integration
  (see KI-7 in [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
- Complete-pair only: a field/timestep is compared only when both the GPU and
  CPU file carry it; nothing is imputed.

## Reference case

`--case-dir` is **any** retained CPU-WRF run directory that holds, for the
chosen `--domain`:

- the WRF `namelist.input` used for that run,
- at least two hourly `wrfout_<domain>_*` history files (init + ≥1 lead), and
- (optional, for the speedup number) the run's `rsl.error.0000` / `rsl.out.0000`
  per-step timing logs.

Bring your own — produce it with any standard WRF v4 chain (WPS + `real.exe` +
`wrf.exe`) for the case you want to check. No fresh CPU-WRF run is needed at
demo time; the demo only *reads* the retained history. The numbers reported in
[Observed result](#observed-result-on-the-default-case-24-h-d02-20260509_18z)
were generated by the project maintainers from an internal retained
`20260509_18z` d02 (3 km) case (73 hourly frames); that corpus is **not** shipped
with the clone, so reproduce the table against your own retained run.

> A small, self-contained, redistributable reference case (a checksummed compact
> `wrfout` subset) is the cleaner future convenience — the Switzerland test
> (`docs/equivalence-switzerland.md`) already implements exactly that pattern via
> `scripts/make_compact_reference.py`, and is the recommended self-serve path for
> a fresh clone that has no local corpus.
