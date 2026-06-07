# Switzerland (Gotthard) GPU-vs-CPU-WRF equivalence test

A self-serve, **non-Canary** equivalence test: run the JAX GPU port on a Central
Alps domain it was never tuned on, and compare it field-by-field against a
Fortran CPU-WRF reference that started from the *same* initial and boundary
conditions. It converts "the port generalizes" from an assertion into something
**you can run and check**.

- One-command user test: [`scripts/equivalence_switzerland.sh`](../scripts/equivalence_switzerland.sh)
- Comparator: [`scripts/equivalence_switzerland_compare.py`](../scripts/equivalence_switzerland_compare.py)
  (reuses the validated comparison engine + tolerances from
  [`scripts/equivalence_demo.py`](../scripts/equivalence_demo.py))
- Case builder (offline, run once): [`scripts/build_switzerland_case.sh`](../scripts/build_switzerland_case.sh)
- CPU reference producer (maintainer): [`scripts/run_switzerland_cpu_reference.sh`](../scripts/run_switzerland_cpu_reference.sh)
- Proof object: `proofs/v0120/equivalence_switzerland.json`

## The case (robust "it works" default)

| Parameter        | Value                                                       |
|------------------|-------------------------------------------------------------|
| Center           | 46.65 N, 8.55 E (Gotthard / Central Switzerland)            |
| Projection       | Lambert conformal, truelat 30/60, stand_lon 8.55            |
| Grid             | 43 x 43 (≈126 km square), **dx = dy = 3 km**                |
| Levels           | 45 (`e_vert`), p_top 5000 Pa                                |
| Domains          | single domain (`d01`), GFS lateral-boundary forced          |
| Forecast         | **24 h** (init 2023-01-15 00 UTC), boundaries every 3 h     |
| Forcing          | GFS 0.5° from the GCP public archive (`global-forecast-system`) |
| Terrain/landuse  | WPS global geog (`topo_gmted2010_5m`, MODIS landuse, etc.)  |

3 km (not 1 km) is deliberate — it avoids the steep-terrain 1 km
stability/OOM edges and is the **robust default**. Robustness was prioritized
over maximum resolution. To run shorter, rebuild the case with `FCST_HOURS=12`.

This is a **NON-Canary** region: the port's physics/dynamics were validated on
the Canary Islands; this test exercises a completely different regime (winter
mid-latitude Alpine terrain, snow, a jet aloft) to probe generalization.

## How to run (the one command)

Assuming `python` + (`conda` or `venv`) with the project installed
(`pip install -e .`, JAX with CUDA) + an NVIDIA driver:

```bash
cd <repo>
PYTHONPATH=src bash scripts/equivalence_switzerland.sh
```

That single command:

1. Builds a **clean standalone input dir** (`wrfinput_d01` + `wrfbdy_d01` +
   `namelist.input`, no CPU `wrfout`) so the port takes the **native-init**
   path (IC from `wrfinput`, LBC from `wrfbdy`).
2. Runs the GPU forecast (`python -m gpuwrf.cli run --input-dir ... --domain d01
   --hours 24`) into `run_gpu/`, measuring wall-clock.
3. Locates the **CPU-WRF reference** (see *CPU reference* below).
4. Compares GPU vs CPU at **all grid points, all output hours**, per field, and
   prints the verdict + speedup; writes `proofs/v0120/equivalence_switzerland.json`.

Useful overrides (env vars): `HOURS`, `DOMAIN`, `CASE_ROOT`, `CPU_REF`,
`CPU_REF_URL`, `CPU_REF_SHA256`, `PYTHON`.

## Predeclared tolerances (pooled RMSE; not tuned to data)

Identical to the Canary demo (`equivalence_demo.py::FIELD_TOLERANCES`):

| Field      | RMSE tol        | what it is                                  |
|------------|-----------------|---------------------------------------------|
| T2         | 1.5 K           | 2 m temperature                             |
| U10 / V10  | 1.5 m s⁻¹       | 10 m wind components                        |
| PSFC       | 120 Pa          | surface pressure (~0.1 % of ~100 kPa)       |
| RAINNC     | 1.0 mm          | accumulated grid-scale precip               |
| T          | 1.5 K           | 3D perturbation potential temperature       |
| U / V      | 1.8 m s⁻¹       | 3D horizontal wind components               |
| W          | 0.30 m s⁻¹      | 3D vertical velocity                        |
| QVAPOR     | 1.0e-3 kg kg⁻¹  | 3D water-vapour mixing ratio                |

A field PASSES if its pooled (all hours × all grid points) RMSE is ≤ tol. The
overall verdict is EQUIVALENT iff every field passes; any exceedance is reported
with its numbers.

## What it proves — and what it does NOT

- **Honest, reproducible, cross-code.** GPU (JAX) and CPU-WRF (Fortran) are two
  *independent* integrators of the same equations, started from the *same*
  `real.exe` ICs/LBCs. This is not a JAX-vs-JAX self-compare and not a model
  compared to its own output.
- **Numerical/operational equivalence within a predeclared tolerance — NOT
  bitwise identity vs Fortran.** Two independent integrators differ at round-off
  and diverge slowly under chaotic dynamics; the question is whether they stay
  within operationally meaningful limits.
- **NOT_EQUIVALENT at late leads is an expected, honest outcome.** As on the
  Canary case, the winds (`U`/`V`/`U10`/`V10`) are the most likely to exceed tol
  by hour ~24 as the two solutions drift apart — especially over steep Alpine
  terrain with a strong jet. That is not a failure of the test: the value is a
  **real, reproducible comparison on a new region**, with every per-field number
  reported. The verdict is whatever the data says.

## CPU reference (how a user gets it cheaply)

The user does **not** build WPS or CPU-WRF. The CPU reference is resolved by
`equivalence_switzerland.sh` in this precedence order:

1. `CPU_REF=<dir>` — an explicit directory containing `wrfout_d01_*`.
2. `tests/fixtures/switzerland/cpu_reference_compact/` — a shipped compact set,
   if present.
3. `${CASE_ROOT}/cpu_reference_compact/` — a locally-distilled compact set.
4. `${CASE_ROOT}/run_cpu/` — a full local CPU-WRF run (the maintainer's run).
5. `CPU_REF_URL=<tarball>` (+ optional `CPU_REF_SHA256`) — downloads and unpacks
   a **published, checksummed compact reference tarball**.

**Distribution decision.** A full hourly 24 h `wrfout` set is ~2 GB — too large
to ship. The scored fields only (the 10 above + coordinates) distil to a
~1–2 MB-per-frame, zlib-compressed NetCDF set via
[`scripts/make_compact_reference.py`](../scripts/make_compact_reference.py)
(~10–20 MB for 25 frames). That compact set is the unit of distribution:
published as a checksummed tarball (point `CPU_REF_URL`/`CPU_REF_SHA256` at it),
or dropped into `tests/fixtures/switzerland/cpu_reference_compact/`. Because the
*full* reference is **reproducible** from the in-repo scripts (`build_*` +
`run_*_cpu_reference.sh`), shipping the compact set is sufficient and honest.

### How the CPU reference was produced (maintainer note)

```bash
# 1. Mint the case from GFS (WPS geogrid/ungrib/metgrid + real.exe).  Run once.
#    Produces /mnt/data/wrf_gpu_switzerland/run_cpu/{wrfinput_d01,wrfbdy_d01,namelist.input}.
taskset -c 0-3 bash scripts/build_switzerland_case.sh

# 2. Run the SERIAL pristine gfortran CPU-WRF on those same inputs.  Times wall.
taskset -c 0-3 bash scripts/run_switzerland_cpu_reference.sh

# 3. (optional) distil to a compact, shippable reference + tarball.
python scripts/make_compact_reference.py \
    --src /mnt/data/wrf_gpu_switzerland/run_cpu \
    --dst /mnt/data/wrf_gpu_switzerland/cpu_reference_compact
tar -czf cpu_reference_switzerland_compact.tar.gz \
    -C /mnt/data/wrf_gpu_switzerland/cpu_reference_compact .
sha256sum cpu_reference_switzerland_compact.tar.gz
```

**The CPU reference is a *serial* (single-core) gfortran WRF build**, not a
28-rank MPI run, so the reported speedup is GPU-vs-single-core. Both `*_cpu_*`
and the compact maker record/forward the measured wall clock
(`cpu_wall_seconds.txt`) so the speedup is the real measured ratio, not an
RSL-extrapolation.

## Files

| File                                              | Role                                  |
|---------------------------------------------------|---------------------------------------|
| `scripts/build_switzerland_case.sh`               | offline: GFS→WPS→real.exe → IC/BC      |
| `scripts/run_switzerland_cpu_reference.sh`        | maintainer: serial CPU-WRF reference  |
| `scripts/make_compact_reference.py`               | distil full wrfout → shippable subset |
| `scripts/equivalence_switzerland.sh`              | **user one-command** GPU run + compare |
| `scripts/equivalence_switzerland_compare.py`      | GPU-vs-CPU comparator + verdict + JSON |
| `proofs/v0120/equivalence_switzerland.json`       | verdict + per-field stats proof object |
