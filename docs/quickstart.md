# Quickstart — clone, install, run a standalone forecast

This is the out-of-box path: a fresh clone, an install, and one GPU forecast
that produces a `wrfout` history file. The **primary path is standalone native
init** — you supply a real-data case (`wrfinput_<domain>`, `wrfbdy_d01`, and the
met_em-stage forcing produced by `real.exe`/metgrid) and `wrf_gpu` builds the
initial/boundary state and integrates entirely on the GPU. **No CPU-WRF
`wrfout` is required** for a standalone run.

> **CLI-flag reconciliation note (v0.12.0):** the standalone auto-detect entry
> is being finalized in parallel with this guide. The commands below use the
> intended standalone invocation; if a flag name differs in your build, run
> `python -m gpuwrf.cli run --help` to see the exact names. The
> auto-detect/standalone behaviour and the resource expectations are stable.

## 0. Prerequisites

- A CUDA-capable NVIDIA GPU with **≥ 26 GiB free VRAM** for the 3 km d02 case at
  fp64 (RTX 5090 / 32 GiB is the reference). See
  [resource-profile.md](resource-profile.md) for the full memory/compile/scratch
  profile.
- CUDA 13 and a JAX CUDA build that sees the GPU.
- A **local NVMe scratch directory** (not tmpfs) with a few GiB free.

## 1. Clone and install

```bash
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu

# Isolated environment (conda or venv both work):
python -m venv .venv && . .venv/bin/activate
# or:  conda create -n wrfgpu python=3.11 && conda activate wrfgpu

# Install the GPU build of JAX first (CUDA 13 wheels):
pip install --upgrade "jax[cuda13]"
# (If the stable wheel does not resolve for your CUDA/driver, the JAX nightly
#  CUDA wheel is the documented fallback.)

# Install wrf_gpu (pulls netCDF4, numpy, PyYAML, zarr):
pip install -e .

# Sanity-check that JAX sees the GPU (should list a cuda device):
python -c "import jax; print(jax.devices())"
```

## 2. Get a case

A standalone case directory contains the `real.exe`/metgrid outputs and a WRF
namelist:

```
my_case/
├── namelist.input        # standard WRF v4 ARW namelist (bring your own)
├── wrfinput_d01          # initial state (one per domain you run)
├── wrfinput_d02
├── wrfbdy_d01            # lateral boundaries for the outer domain
└── met_em.d0*.*.nc       # met_em-stage forcing (metgrid output)
```

You can produce these from any standard WRF preprocessing chain (WPS +
`real.exe`). **Bring your existing WRF `namelist.input`** — the supported
matrix runs as-is, and anything unsupported fails closed with a named reason
(see [namelist-compatibility.md](namelist-compatibility.md)). The one common
edit: WRF's real-data diffusion defaults (`diff_opt=1`, `km_opt=4`,
Smagorinsky) are not yet supported — switch to constant-K (`diff_opt=2`,
`km_opt=1`).

## 3. Run a standalone forecast

```bash
python -m gpuwrf.cli run \
    --input-dir   my_case \
    --output-dir  runs/my_forecast \
    --domain      d02 \
    --hours       24 \
    --scratch-dir /fast/nvme/gpuwrf_scratch
```

`run` auto-detects the input directory:

- If the case has CPU-WRF `wrfout` history files → **replay mode** (consumes the
  CPU history for boundary/skill comparison; the legacy path).
- Otherwise (only `real.exe` outputs, no CPU `wrfout`) → **standalone native-init
  mode** — `wrf_gpu` assembles `wrfinput`/`wrfbdy` and integrates on the GPU with
  no CPU-WRF dependency.

What to expect on the **first** run:

1. A fail-closed namelist check (instant, no GPU).
2. **A ~5-minute cold JIT compile with no output** — this is XLA compiling, not a
   hang. Subsequent runs read the cached executable and skip this.
3. Integration: ≈ 15 s of wall-clock per forecast-hour on the reference GPU; peak
   **VRAM ≈ 24.6 GiB**.
4. A `wrfout` history file (and a run payload JSON) under `--output-dir`.

See [resource-profile.md](resource-profile.md) for the compile-cache override
(`GPUWRF_JAX_CACHE_DIR`) and memory sizing.

## 4. Read the output

The output is a standard WRF-compatible `wrfout` NetCDF history file. Inspect it
with any WRF tooling:

```bash
ncdump -h runs/my_forecast/wrfout_d02_*    # dimensions + variables
python -c "import xarray as xr; print(xr.open_dataset('runs/my_forecast/wrfout_d02_...'))"
```

The operational writer emits a focused **64-variable** subset of WRF's 375
variables (all core meteorological/spatial/vertical/soil fields; the missing
ones are stochastic-seed and Noah-MP snow-layer diagnostics). See known issue
**KI-3** in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and the README scope table.

## 5. (Optional) replay / compare against CPU-WRF

If you have a CPU-WRF run for the same case and want a dimension comparison,
point the comparator at it:

```bash
python -m gpuwrf.cli run \
    --namelist        my_case/namelist.input \
    --input-dir       my_case \
    --output-dir      runs/cmp \
    --domain          d02 \
    --hours           1 \
    --compare-cpu-dir my_case
```

This writes `runs/cmp/proofs/dimension_compare.json` and exits 0 only on a
clean pipeline + dimension PASS. (This replay/compare flow is the validated
skill-comparison harness; coupled-skill validation vs CPU-WRF is run through it,
not from-scratch native init — see the README honesty note.)

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Run "hangs" for ~5 min on first launch, no output | Cold JIT compile. Normal. Warm the cache; subsequent runs are fast. |
| `RESOURCE_EXHAUSTED` / OOM | Not enough free VRAM (need ≥ ~26 GiB for d02 fp64). Use a bigger card or a smaller domain. |
| Namelist rejected with a named scheme error | Unsupported option, fail-closed. Switch to a supported scheme (see namelist-compatibility.md); for diffusion use `diff_opt=2`/`km_opt=1`. |
| Scratch fills up / RAM exhausted | `--scratch-dir` pointed at tmpfs. Use a local NVMe path. |
| `jax.devices()` shows only CPU | JAX CUDA wheel/driver mismatch. Reinstall `jax[cuda13]` (or the nightly CUDA wheel). |
