# GPU-vs-CPU-WRF equivalence demo

A self-serve script that lets a skeptic *run and check* that the JAX GPU port
reproduces a retained CPU-WRF forecast — converting "equivalence is asserted"
into "equivalence you can run and verify yourself."

- Script: [`scripts/equivalence_demo.py`](../scripts/equivalence_demo.py)
- Default proof object:
  [`proofs/v0120/equivalence_demo_20260509_d02.json`](../proofs/v0120/equivalence_demo_20260509_d02.json)

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

```bash
cd <repo>
PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/equivalence_demo.py \
    --case-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --domain   d02 \
    --hours    24 \
    --out      proofs/v0120/equivalence_demo_20260509_d02.json
```

Flags:

| flag | meaning |
|---|---|
| `--case-dir` | retained CPU-WRF run dir (its `namelist.input` + hourly `wrfout_<domain>_*`). Default = the retained `20260509_18z` d02 case. |
| `--domain` | domain id to run/compare (default `d02`). |
| `--hours` | forecast lead hours to run and compare (12–24 h is a sensible single forecast). |
| `--out` | path for the verdict + stats proof JSON. |
| `--gpu-output-dir` / `--scratch-dir` | optional disk-backed dirs (default: temp dirs, cleaned up). Keep them off `/tmp` tmpfs. |

Prerequisites are the same as a normal forecast (see
[quickstart.md](quickstart.md)): a CUDA GPU with ~26 GiB free VRAM for d02 fp64,
a JAX CUDA build, and a local NVMe scratch path. The first launch pays the
~5-minute cold JIT compile. The script exits 0 on an `EQUIVALENT` verdict and
non-zero otherwise.

> On a shared workstation, run it through the GPU lock wrapper so it waits its
> turn for the GPU:
> `/tmp/wrf_gpu_run.sh taskset -c 0-3 env PYTHONPATH=src ... python scripts/equivalence_demo.py ...`

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

Run on 2026-06-07 (proof:
[`proofs/v0120/equivalence_demo_20260509_d02.json`](../proofs/v0120/equivalence_demo_20260509_d02.json),
GPU `cuda:0`, replay path). Pooled RMSE over all 24 hourly timesteps and all
grid points:

| Field | pooled RMSE | tol | pooled bias | verdict |
|---|---|---|---|---|
| T2 | 0.484 K | 1.5 K | +0.195 K | **PASS** |
| U10 | 2.237 m s⁻¹ | 1.5 | +1.491 | EXCEEDS |
| V10 | 2.441 m s⁻¹ | 1.5 | −1.521 | EXCEEDS |
| PSFC | 707.8 Pa | 120 | −703.0 | EXCEEDS |
| RAINNC | 0.501 mm | 1.0 | −0.034 | **PASS** |
| T (θ′) | 2.040 K | 1.5 | +0.119 | EXCEEDS |
| U | 3.167 m s⁻¹ | 1.8 | +1.660 | EXCEEDS |
| V | 8.130 m s⁻¹ | 1.8 | −5.417 | EXCEEDS |
| W | 0.126 m s⁻¹ | 0.30 | −0.028 | **PASS** |
| QVAPOR | 5.67×10⁻⁴ kg kg⁻¹ | 1.0×10⁻³ | +4.5×10⁻⁵ | **PASS** |

**Overall verdict: NOT_EQUIVALENT** (6 of 10 fields exceed tolerance). This is
the honest current state, reported as-is. Two distinct things are happening:

1. **PSFC** carries a near-constant **~590 Pa offset already at forecast hour 1**
   (GPU mean 100534 Pa vs CPU 101125 Pa) that grows to ~700 Pa — present before
   any dynamical divergence (hour-1 U/V/T RMSE are 0.32 / 0.17 / 0.19, i.e.
   essentially identical). That signature is a **systematic surface-pressure
   reference/diagnostic difference**, not chaotic drift. It is the prime
   follow-up.
2. **U10/V10/T/U/V** are **within tolerance at short lead and grow with lead
   time** (e.g. V RMSE 0.17 m s⁻¹ at h1 → ~11 m s⁻¹ by h19, monotonic; the 3D
   meridional wind V drifts ~3× faster than U). That is genuine lead-time error
   growth between two independent integrators, concentrated in the wind field,
   strongest in V. T2, W, QVAPOR and RAINNC stay inside tolerance for the full
   24 h.

**Speedup:** the GPU integrated the 24 h d02 forecast in **1408.6 s** vs an
estimated **2393.2 s** for the CPU-WRF d02 solver (from the retained RSL
per-step timing) — **~1.70×**. (This is a same-card, fp64, single-forecast
wall-clock comparison of the d02 main solver only; it is not the warm/fused
kernel speedup quoted elsewhere.)

A documented exceedance with its numbers is exactly the point of a self-serve
demo. Re-running after a PSFC-reference fix and/or a wind-divergence
investigation is the natural next step.

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

The default reference is the retained CPU-WRF `20260509_18z` d02 (3 km) case at
`/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z`,
which holds 73 hourly `wrfout_d02_*` files (init + 72 h) plus the WRF
`namelist.input` and RSL timing logs. No fresh CPU-WRF run is required.

Publishing a small, self-contained reference case (e.g. via Google Drive) so the
demo can be run off a fresh clone without the local corpus is a future
convenience; today the demo expects the retained case path above (or any
equivalent retained CPU-WRF run dir passed via `--case-dir`).
