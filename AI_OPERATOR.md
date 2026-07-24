# AI Operator Runbook for wrf_gpu

This playbook is for an AI assistant helping an end user run a forecast with
`wrf_gpu`. It is not the internal development protocol for changing the port.
If the user says "here is my input, get it running", follow this runbook.

This runbook describes the **v0.23.4 candidate**. Do not present it
as the current published release or recommend deployment ahead of its actual
publication: two-domain configurations are measured unaffected, but its
production-faithful maxdom3 canary is 51.67% slower than v0.23.3. That
regression was investigated exhaustively (every quick-fix candidate falsified
or rejected with evidence) and hardware-measured as occupancy/latency-bound,
not bandwidth-bound — understood, but not fixed, and shipping as a known
limitation rather than an open question.
The candidate's accepted correctness envelope
includes one-way live nesting through nine domains on the sealed all-physics
fixture: 27/27 outputs landed, all 177,497,511 numeric values were finite, and
all 9 domains × 102 fields passed the frozen WRF comparison gates. This is a
fixture-backed capability statement, not universal WRF-scheme coverage or a new
speed claim. The current published release remains v0.23.3 until v0.23.4 is
actually tagged and published.

Communication contract:

- Narrate each material step before running it.
- Give time estimates before any long wait.
- Never run silently for minutes. Tell the user whether the process is
  installing, checking the GPU, compiling, integrating, or blocked.
- Be clear about what the run proves and what remains an open research gate.

## Step 0 - Understand the Request

Ask for or detect:

- The case directory.
- Whether it contains `namelist.input` plus directly usable prepared
  `wrfinput_<domain>` files and `wrfbdy_d01`. Raw `met_em` alone is not a run
  input for the current CLI; it must first pass through the upstream WPS /
  `real.exe` preprocessing chain outside `wrf_gpu`.
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

- Single domain: pass `--domain d01` or another domain. When omitted, the
  single-domain default is root domain `d01`.
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
- Ordinary nested cold compile is geometry-dependent and can take 8-12 minutes.
- A large nine-domain fused nest is a separate, much larger compile; allow tens
  of minutes to well over an hour. Do not promise the ordinary-nest range.
- Later runs with the same geometry should hit the persistent warm cache and
  start in seconds.

Runtime ETA framing:

- Do not use one generic nested throughput number across topologies. The v0.23.4
  accepted all-physics nine-domain stress case used 5,744.407 s of warm model
  wall for one forecast hour; that is a correctness-stress workload, not a
  performance claim. Use the release canary or the user's observed progress for
  an ETA on a different case.
- Estimate total integration time as:
  `forecast hours * estimated seconds per forecast-hour`.
- Once progress is visible, refine the estimate from observed throughput.
- Many same-geometry cases can be batched with `GPUWRF_BATCH_ENSEMBLE=B` to fill
  a small-grid GPU.
- **As of v0.23.4 (prepared, not yet tagged): disclose a domain-count-dependent
  performance caveat before running.** Single-domain and 2-domain (`d01`+`d02`)
  cases are fully validated with no measured performance regression versus
  v0.23.3. **3-domain (`--max-dom 3`) and 9-domain (`--max-dom 9`) cases that
  activate `d03` are a real, measured ~51.67% slower wall-clock than v0.23.3**
  (root-caused to the FCT flux-limiter kernel's own memory-traffic cost;
  performance/profiler investigation remains active and unresolved). Tell
  the user this before starting a 3+/9-domain run so their time estimate is
  honest — do not silently use the pre-v0.23.4 674 s/forecast-hour figure for a
  `d03`-activating case without this caveat. Correctness is unaffected either
  way (nine-nest replay is fully GREEN); this is a wall-clock-only caveat.

Hardware and optimization choices:

- The measured reference is an RTX 5090 with 32 GiB VRAM. The accepted v0.23.4
  nine-domain stress fixture peaks at 16,624 MiB VRAM and 35,773,432 kB host
  RSS. Do not transfer those numbers to a different grid/physics mix without a
  preflight.
- H100/B200/GB300 scale-out behavior is not release-benchmarked. Describe it as
  prospective and measure the user's system; never promise a speedup from the
  card name alone.
- Keep `cuda_async`, the default allocator, for an exclusive/locked production
  run. Use `platform` only as an explicit diagnostic fallback; it is slower and
  is not the release performance path.
- Keep the version-keyed AOT/JIT cache enabled. A warm executable is the normal
  path; deleting the cache trades disk space for a large cold-compile bill.
- For many small, identical-geometry forecasts, propose
  `GPUWRF_BATCH_ENSEMBLE=B` after checking aggregate VRAM. Do not batch different
  forecast dates; that case is not supported.
- Keep the shipped RRTMG column tiling unless a measured, separately validated
  experiment says otherwise. Do not alter `time_step`, acoustic substeps,
  precision, or physics merely to meet an ETA; those are scientific changes,
  not generic performance knobs.
- Before compute, run `gpuwrf namelist-support` and `--dry-run`. Refuse or
  clearly flag unsupported schemes, terrain beyond the validated ~6000 m
  envelope, a projected VRAM overrun, or deep nesting the user did not
  explicitly request.

Safety rails to explain:

- The default-on finite guard aborts on the first non-finite prognostic and
  reports `{domain, field, level, step, sim-time, index}`.
- For `--max-dom > 1`, a CPU-side VRAM preflight runs before the long compile and
  fails closed with exit 75 if there is not enough headroom.
- Terrain above about 6000 m can diverge and is expected to fail closed with 0
  bad frames.
- The nested pipeline currently uses a fixed 30-minute radiation target rather
  than arbitrary namelist `radt`. Tell the user before a nested run when their
  requested cadence differs; do not silently claim it was honored.
- Nested `topo_shading=1` and `slope_rad=1` are currently accepted but bind to
  disabled values in the nested runtime. Report that terrain-radiation gap; do
  not describe those effects as active.
- The accepted nested path honors `moist_adv_opt`/`scalar_adv_opt`, but the
  single-domain daily pipeline currently drops requested values and runs `0/0`.
  Report the effective behavior; do not claim the requested limiter was active.
- When `time_step_sound` is omitted, the current runtime selects 10 acoustic
  substeps where pristine WRF derives 4 on the tracked fixture. A four-substep
  discriminator did not close the terminal gate and residual dry-mass behavior
  remains open. Keep the shipped default and disclose the mismatch rather than
  silently changing the acoustic control.
- v0.23.4 fixes the demonstrated late-Ni ordering defect, but the Thompson
  `NSED_MAX=16` static sedimentation-substep cap and widespread exact-zero
  carried Ni remain separate fidelity debt. Flag this before long or strongly
  ice-sedimentation-sensitive claims; do not alter the cap, clamp Ni, or claim
  the broader class is WRF-exact.
- Two AOT cheap-key fragmentation cases can miss an otherwise reusable warm
  executable when fused-phase shape or terrain-staging provenance differs.
  Preserve the versioned cache and a stable staging namespace. A safe JIT/AOT
  rebuild is expected on a miss; report the compile delay rather than treating
  it as a numerical failure or promising a warm hit.

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
- v0.23.4 closes the specific one-hour nine-nest fixture and a specific 24-hour
  SP2 chain gate. A successful user run does not close broad seasonal or
  configuration-independent 24-72 h forecast-skill equivalence against CPU-WRF.

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
| Nested radiation cadence differs from the namelist | Fixed 30-minute nested cadence limitation | Disclose the mismatch; do not edit the namelist or claim arbitrary `radt` was honored. |
| Nested terrain/slope radiation requested | `topo_shading`/`slope_rad` bind disabled on the current nested runtime | Disclose the limitation; do not silently substitute another result. |
| Daily single-domain run requests moist/scalar advection | That pipeline currently drops `moist_adv_opt`/`scalar_adv_opt` and runs `0/0` | Disclose the effective behavior; do not claim the requested limiter ran. |
| Omitted `time_step_sound` must match WRF exactly | Runtime selects 10 acoustic substeps where WRF derives 4 on the tracked fixture; residual dry-mass behavior is open | Keep the shipped setting, disclose the mismatch, and do not tune it as an operational workaround. |
| Long/ice-heavy run needs exact Thompson sedimentation parity | `NSED_MAX=16` and exact-zero carried Ni remain open after the late-Ni fix | Disclose the fidelity boundary; do not change the cap or sanitize Ni as an operational workaround. |
| Expected nested AOT warm hit recompiles | Known cheap-key fragmentation from fused-phase shape or terrain-staging provenance | Keep the versioned cache/staging namespace stable, allow the safe rebuild, and report the extra compile wall. |

## Honesty Rules to Relay

- Default precision is fp64. v0.23.4 intentionally changes physics results
  relative to v0.23.3 through correctness fixes, so old ice-sensitive golden
  trajectories require revalidation/rebaselining rather than a byte-identity
  assumption.
- Broad seasonal/configuration-independent 24-72 h forecast-skill equivalence
  versus CPU-WRF remains open. This is a research artifact, not a universally
  validated operational WRF replacement.
- `wrf_gpu` is WRF-compatible. It is not WRF and is not affiliated with
  UCAR/NCAR.
- Opt-in features are opt-in. Do not present `--feedback`, `--score`,
  `GPUWRF_BATCH_ENSEMBLE`, `GPUWRF_TRAINING_OUTPUT_SUBSET`, or
  `--force-gpu-run` as defaults.
- No masking or clamps on results. If the finite guard aborts, report the named
  failure instead of hiding it.
- If running v0.23.4: 2-domain configurations are
  fully validated (correctness and performance, measured no regression);
  3-domain/9-domain configurations that use `d03` are correct but a measured
  ~51.67% slower than v0.23.3 — measured, root-caused (occupancy/latency-bound),
  and understood, but not fixed. Relay this before,
  not after, a 3+/9-domain run.
- For deeper human-facing context, point users to `README.md` and the project
  docs.
