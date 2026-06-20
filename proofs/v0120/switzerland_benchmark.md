# Switzerland (Gotthard) BIG-GRID GPU-vs-28-rank-CPU benchmark — v0.12.0

The biggest **honest** speedup the project can claim on this workstation: a
LARGE single-domain Alpine forecast where the RTX 5090 is well-saturated,
compared against **28-rank dmpar MPI CPU-WRF** — the project's honest
denominator — on the **same grid, same `wrfinput`/`wrfbdy`, per forecast hour**.

This extends F's robust 43×43 Switzerland equivalence case
(`docs/equivalence-switzerland.md`) by scaling the grid UP. Small grids leave
the 5090 launch-bound (≈267 W of 575 W, low SM occupancy); a big grid drives GPU
per-cell time down while the 28-rank CPU per-cell time stays ~flat, so the
speedup factor GROWS up to the GPU memory wall. We pick the largest grid that
comfortably fits single-GPU fp64 (~32 GB).

## The case (BIG grid)

| Parameter        | Value                                                       |
|------------------|-------------------------------------------------------------|
| Center           | 46.65 N, 8.55 E (Gotthard / Central Switzerland)            |
| Projection       | Lambert conformal, truelat 30/60, stand_lon 8.55            |
| Grid             | **150 × 150 mass points** (`e_we=e_sn=151`), **dx = dy = 3 km** (≈450 km square) |
| Levels           | 45 (`e_vert`), `bottom_top=44` mass levels, p_top 5000 Pa   |
| Time step        | 18 s (6·dx_km rule; max map factor 0.97, CFL-safe)          |
| Domains          | single domain (`d01`), GFS lateral-boundary forced          |
| Forecast         | **24 h** (init 2023-01-15 00 UTC), boundaries every 3 h     |
| Forcing          | GFS 0.5° (GCP public archive) — reuses F's downloaded cycle |
| Physics          | identical to F's case: Thompson MP(8), RRTMG LW/SW(4), MYNN sfclay(5)+PBL(5), Noah-MP(4), no cumulus |

Same center / projection / levels / date / physics as the 43×43 robust default;
**only the grid is scaled up** (43×43 → 151×151), so the comparison is
apples-to-apples science, just bigger. Real Alpine terrain is resolved
(HGT 4.7 → 3210 m over the domain).

### Memory sizing (why ~150²)

The RRTMG g-point radiation temporary is ≈0.75 MB per horizontal column. At
150×150 = 22.5 k columns that is ≈17 GiB transient + ≈6 GiB working ≈ 23 GiB —
fits 32 GB fp64. (The GPU CLI's startup probe requests ≈23.5 GiB, consistent
with this.) Fallback grid is 129×129 (128² mass pts) if 151² trips real.exe or
the GPU OOMs; the build script auto-falls-back. The manager MAY push bigger on a
GPU probe if 150² leaves headroom.

## HONEST speedup definition

```
speedup = CPU_wall(28-rank dmpar MPI, big grid, per forecast hour)
          ────────────────────────────────────────────────────────
          GPU_wall(same big grid, warm-cached, fp64, per forecast hour)
```

- **Denominator = 28-rank dmpar MPI CPU-WRF** on cores 0–27 — NOT 1-core serial,
  NOT a JAX-vs-JAX self-compare. This is the project's honest baseline.
- **Numerator = GPU port, warm-cached (post-JIT), fp64**, standalone native-init
  from the same `wrfinput_d01`/`wrfbdy_d01`.
- **Same grid, same dt, same forecast length, per-forecast-hour normalized.**
- Two timing bases are captured for the CPU run; report the matching basis for
  the GPU:
  - `total_wall_s` — end-to-end `wrf.exe` wall (incl. MPI init + final I/O).
  - `mainloop_sum_s` — sum of rank-0 "Timing for main" over all steps (pure
    integration; the fairest per-forecast-hour basis). In `cpu_mainloop_seconds.txt`.
- Numbers must state **grid, dt, rank count, warm/cold, fp64** explicitly. The
  equivalence verdict (likely NOT_EQUIVALENT on late-lead winds at 24 h over the
  Alps) is reported separately and honestly — it does not affect the speedup.

## How it was built / run (maintainer + manager)

```bash
# 1. Mint the BIG case (geogrid/ungrib/metgrid + real.exe via dmpar MPI).
#    Reuses F's cached GFS; writes <DATA_ROOT>/wrf_gpu_switzerland_big/run_cpu/.
RUNROOT=<DATA_ROOT>/wrf_gpu_switzerland_big \
  taskset -c 0-3 bash scripts/build_switzerland_big_case.sh

# 2. 28-rank CPU-WRF reference (HONEST denominator). Launch DETACHED (~25 min
#    main-loop). Captures cpu_wall_seconds.txt, cpu_mainloop_seconds.txt, cpu_timing.json.
RUNROOT=<DATA_ROOT>/wrf_gpu_switzerland_big NRANKS=28 \
  setsid nohup taskset -c 0-27 bash scripts/run_switzerland_cpu_reference_mpi.sh \
  > <DATA_ROOT>/wrf_gpu_switzerland_big/run_cpu/cpu_reference.out 2>&1 &

# 3. GPU forecast + comparison (MANAGER runs when the GPU is free; warm-cached fp64).
#    CPU_RANKS=28 + mainloop basis -> honest per-forecast-hour ratio in the proof.
CASE_ROOT=<DATA_ROOT>/wrf_gpu_switzerland_big \
  CPU_WALL_BASIS=mainloop CPU_RANKS=28 HOURS=24 DOMAIN=d01 \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src bash scripts/equivalence_switzerland.sh
```

The GPU forecast inside step 3 runs:

```bash
PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m gpuwrf.cli run \
  --input-dir  <DATA_ROOT>/wrf_gpu_switzerland_big/run_gpu_input \
  --output-dir <DATA_ROOT>/wrf_gpu_switzerland_big/run_gpu \
  --scratch-dir <DATA_ROOT>/wrf_gpu_switzerland_big/scratch \
  --domain d01 --max-dom 1 --hours 24
```

(`run_gpu_input` is a clean dir with only `wrfinput_d01` + `wrfbdy_d01` +
`namelist.input` so the port takes the standalone native-init path, not CPU
replay. The orchestrator builds it.) For a **warm** GPU number, run once to
warm the JIT cache, then time a second run — or report the orchestrator's
end-to-end wall and label it cold/warm explicitly.

## Files

| File                                                | Role                                          |
|-----------------------------------------------------|-----------------------------------------------|
| `scripts/build_switzerland_big_case.sh`             | mint BIG (≈150²) IC/BC via dmpar real.exe; auto-fallback to 129² |
| `scripts/run_switzerland_cpu_reference_mpi.sh`      | **28-rank** dmpar MPI CPU-WRF reference + dual timings |
| `scripts/equivalence_switzerland.sh`                | GPU run + compare (forwards `CPU_RANKS`/basis/label) |
| `scripts/equivalence_switzerland_compare.py`        | GPU-vs-CPU comparator; honest 28-rank speedup block + per-fcst-hour |
| `proofs/v0120/switzerland_benchmark.md`             | this doc (big-grid config + honest-speedup definition) |
| `proofs/v0120/equivalence_switzerland.json`         | verdict + per-field stats + speedup proof object (written by step 3) |
| `<DATA_ROOT>/wrf_gpu_switzerland_big/run_cpu/cpu_timing.json` | 28-rank ranks/grid/total+mainloop wall + per-fcst-hour |
