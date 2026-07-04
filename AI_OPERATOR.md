# AI Operator Runbook for wrf_gpu

This playbook is for an AI assistant helping an end user run a forecast with
`wrf_gpu`. It is not the internal development protocol for changing the port.
If the user says "here is my input, get it running", follow this runbook.

Communication contract:

- Narrate each material step before running it.
- Give time estimates before any long wait.
- Never run silently for minutes. Tell the user whether the process is
  installing, checking the GPU, compiling, integrating, or blocked.
- Be clear about what the run proves and what remains an open research gate.

## Step 0 - Understand the Request

Ask for or detect:

- The case directory.
- Whether it contains `namelist.input` and usable prepared inputs, such as
  `met_em` files or `wrfinput_<domain>` plus `wrfbdy_d01`.
- The intended forecast length in hours.
- The desired output directory.
- The domain to run for single-domain mode, such as `d01` or `d02`.
- Whether the user wants a live-nested run. A namelist `max_dom > 1` does not
  auto-run nested; nested mode requires `--max-dom N`.
- A real disk-backed scratch directory. Do not use `/tmp` when it is tmpfs.

Recognize the input mode:

- Standalone native init: `wrfinput_<domain>` plus `wrfbdy_d01`, with no CPU
  `wrfout` required. This does not need `real.exe` or CPU-WRF.
- Replay: two or more CPU `wrfout` files.

If the user has only raw forcing data and no usable case directory, stop and
explain what case inputs are missing before attempting a run.

## Step 1 - Environment

Create an isolated Python 3.11 environment and install the CUDA JAX build:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade "jax[cuda13]"
pip install -e .
python -c "import jax; print(jax.devices())"
```

For a package install that uses the repo GPU extra, use
`pip install -e ".[cuda]"` after installing the CUDA JAX wheel.

Expected result: `jax.devices()` lists a CUDA device. CPU import can work
without a GPU, but GPU execution needs both the CUDA `jaxlib` and a compatible
NVIDIA driver.

Optional environment variables:

- `GPUWRF_JAX_CACHE_DIR`: persistent JIT cache directory. The standard
  `JAX_COMPILATION_CACHE_DIR` is also honored as an alias.
- `GPUWRF_TMPDIR` or `GPUWRF_SCRATCH`: scratch location. Use disk-backed
  storage, never tmpfs.

## Step 2 - GPU and Driver Check

Run:

```bash
nvidia-smi
python -c "import jax; print(jax.devices())"
```

Confirm that `nvidia-smi` sees an NVIDIA GPU and that JAX lists a CUDA device.
If `nvidia-smi` fails, fix the driver or machine assignment first. If
`nvidia-smi` works but JAX lists only CPU, the CUDA JAX wheel or driver stack is
not ready for GPU execution.

## Step 3 - Smoke Test the Bundled Switzerland Case

Prove the local toolchain before touching the user's data. This bundled real GFS
case needs only `GPUWRF_WRF_ROOT`:

```bash
export GPUWRF_WRF_ROOT=/path/to/pristine/WRF-v4
python -m gpuwrf.cli run --input-dir examples/switzerland_d01 --output-dir runs/switzerland_d01 \
  --domain d01 --hours 1 --scratch-dir /fast/nvme/scratch
```

Then verify that a `wrfout` file exists and can be inspected:

```bash
ncdump -h runs/switzerland_d01/wrfout_d01_*
```

Tell the user what is happening:

- The command checks the namelist before expensive compute.
- The first run pays a one-time cold compile with no output before integration.
  For a single-domain case, expect about 1/2-2 minutes. It is compiling, not
  hung.
- Later same-geometry runs should hit the persistent warm cache and start in
  seconds via the AOT cheap-key warm-start.

## Step 4 - Prepare the User's Case

Set the required WRF table root:

```bash
export GPUWRF_WRF_ROOT=/path/to/pristine/WRF-v4
```

`GPUWRF_WRF_ROOT` must point to a pristine WRF v4 source/run tree. It is needed
for Noah-MP and RRTM/RRTMG lookup tables, plus Thompson
`CCN_ACTIVATE.BIN`. Without it, table-loading schemes fail closed with a named
error.

`GPUWRF_CANAIRY_ROOT` points to the validation-case corpus. It is only for the
bundled validation cases, not ordinary user cases.

Use the CLI's fail-closed namelist behavior. Unsupported or unsafe choices stop
with a named reason before the run. Do not silently substitute schemes or edit
the user's physics choices unless the user approves a specific change.

Choose the run shape:

- Single domain: pass `--domain d01` or another domain. Default is `d02`.
- Live nested: pass `--max-dom N`. Values greater than 1 run `d01..dN`.

Remember: a namelist `max_dom > 1` does not auto-run nested; the operator must
pass `--max-dom N`.

## Step 5 - Run and Narrate

Single-domain template:

```bash
python -m gpuwrf.cli run \
  --input-dir /path/to/case \
  --output-dir runs/my_forecast \
  --namelist /path/to/case/namelist.input \
  --domain d01 \
  --hours 24 \
  --scratch-dir /fast/nvme/scratch
```

Live-nested template:

```bash
python -m gpuwrf.cli run \
  --input-dir /path/to/case \
  --output-dir runs/my_nested_forecast \
  --namelist /path/to/case/namelist.input \
  --max-dom 3 \
  --hours 24 \
  --scratch-dir /fast/nvme/scratch
```

CLI flags and defaults to know:

| Flag | Default / requirement |
| --- | --- |
| `--input-dir` | required |
| `--output-dir` | required |
| `--namelist` | `<input-dir>/namelist.input` |
| `--domain` | omitted → `d01` (root domain) for single-domain; explicit wins |
| `--max-dom` | `1`; values greater than 1 run live nested `d01..dN` |
| `--domains-from-namelist` | run all domains the namelist declares (`max_dom`) |
| `--hours` | omitted → read from namelist `&time_control` (`run_days`/`run_hours`/…), else 1; explicit wins |
| `--dry-run` | validate + resolve + print the effective plan, then exit without compiling |
| `--scratch-dir` | `<output-dir>/.scratch`; never `/tmp` tmpfs |
| `--proof-dir` | optional |
| `--compare-cpu-dir` | optional CPU reference comparison input |
| `--score` | optional; requires `GPUWRF_AEMET_ROOT` |
| `--feedback` | optional two-way nesting; nested only |
| `--force-gpu-run` | optional override |

WRF-parity note (v0.23.1): when `--hours` or `--domain` are omitted they are resolved
from `namelist.input` (forecast length from `&time_control`; root domain `d01`) and the
resolved value is printed — an explicit flag always overrides. Use
`gpuwrf namelist-support` to print which schemes are operational / reference-only /
fail-closed without touching the GPU, and `--dry-run` to preview the effective plan
before any compile.

Narrate expected timing before the long wait:

- Cold compile is one-time and produces no output before integration. It is
  compiling, not hung.
- Single-domain cold compile: about 1/2-2 minutes.
- Nested cold compile: about 8-12 minutes.
- Big all-7 9-domain nest: tens of minutes.
- Later runs with the same geometry should hit the persistent warm cache and
  start in seconds.

Runtime ETA framing:

- Warm single-card nested throughput is about 674 seconds per forecast-hour at
  the compute floor, grid-dependent.
- Estimate total integration time as:
  `forecast hours * estimated seconds per forecast-hour`.
- Once progress is visible, refine the estimate from observed throughput.
- Many same-geometry cases can be batched with `GPUWRF_BATCH_ENSEMBLE=B` to fill
  a small-grid GPU.

Safety rails to explain:

- The default-on finite guard aborts on the first non-finite prognostic and
  reports `{domain, field, level, step, sim-time, index}`.
- For `--max-dom > 1`, a CPU-side VRAM preflight runs before the long compile and
  fails closed with exit 75 if there is not enough headroom.
- Terrain above about 6000 m can diverge and is expected to fail closed with 0
  bad frames.

## Step 6 - Deliver and Verify

Show the user where the output landed:

```bash
ls runs/my_forecast/wrfout_*
ncdump -h runs/my_forecast/wrfout_*
```

Explain the output:

- The files are WRF-compatible `wrfout` NetCDF files.
- The focused output set is about 104 variables.
- The opt-in `GPUWRF_TRAINING_OUTPUT_SUBSET` writes a smaller training subset of
  about 39 core variables plus coordinates; retained variables are bit-identical.

State what was verified:

- The smoke test verifies install, GPU visibility, table roots, compile,
  integration, and NetCDF output on the bundled case.
- The user-case run verifies that this case ran under the selected supported
  configuration.
- A successful run does not close the 24-72 h forecast-skill equivalence gate
  against CPU-WRF.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Noah-MP, RRTM/RRTMG, or Thompson table error | `GPUWRF_WRF_ROOT` is missing or wrong | Set `GPUWRF_WRF_ROOT` to a pristine WRF v4 source/run tree and rerun. |
| Namelist rejected before compile | Unsupported or unsafe option | Read the named reason, report it to the user, and ask before changing physics. |
| Nested run exits 75 before compile | CPU-side VRAM preflight failed | Free GPU memory, reduce the case, or use `--force-gpu-run` only as an explicit operator override. |
| Scratch fills memory | Scratch path is tmpfs | Use `--scratch-dir` or `GPUWRF_TMPDIR`/`GPUWRF_SCRATCH` on real disk. |
| First run looks hung | Cold JAX compile with no output | Tell the user the expected compile range and wait. Later runs should use the warm cache. |
| `jax.devices()` lists only CPU | CPU JAX install or CUDA driver/JAX mismatch | Install `jax[cuda13]` and confirm the driver with `nvidia-smi`. |
| Terrain above about 6000 m fails | Known steep-terrain stability ceiling | Treat it as a fail-closed envelope, not a result to patch around. |

## Honesty Rules to Relay

- Default fp64 is byte-identical across releases.
- The 24-72 h forecast-skill equivalence versus CPU-WRF is an open gate. This is
  a research artifact, not a validated operational WRF replacement.
- `wrf_gpu` is WRF-compatible. It is not WRF and is not affiliated with
  UCAR/NCAR.
- Opt-in features are opt-in. Do not present `--feedback`, `--score`,
  `GPUWRF_BATCH_ENSEMBLE`, `GPUWRF_TRAINING_OUTPUT_SUBSET`, or
  `--force-gpu-run` as defaults.
- No masking or clamps on results. If the finite guard aborts, report the named
  failure instead of hiding it.
- For deeper human-facing context, point users to `README.md` and the project
  docs.
