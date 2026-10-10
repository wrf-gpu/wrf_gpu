# AI Operator Runbook for wrf_gpu

Help the user run an existing WRF case with the public CLI. This is the
operator runbook; development lanes follow their separate agent kernel.

**Version:** v0.3.3. Eight final comparisons are closed; accepted deviations and raw failures are disclosed in the release evidence.
The live published guide was v0.3.2 at 2026-10-07 10:15Z, confirmed by direct
HTTP and a cache-busted fetch. Use the installed release and its matching
README/guide/evidence. W2 and earlier plots retain their source labels;
their source labels remain historical. See [final identity plots](docs/release/evidence/v033/final_w3/index.html), [validation](docs/release/V0.3.3.md),
[all retained identity galleries](docs/release/evidence/v033/index.html),
[known issues](KNOWN_ISSUES.md) and [history cleanup](docs/release/HISTORY_CLEANUP.md).

Explain each material action and the expected wait. During compilation or
integration, report progress and name any blocker. Use original CPU-WRF or
pristine WRF as fidelity truth; older GPU output is regression evidence.

## Understand the case

Find `namelist.input`, prepared `wrfinput_d0*` and `wrfbdy_d01`. Raw GRIB or
`met_em` requires upstream WPS/real preprocessing; the CLI does not perform
that chain. Standalone native initialization needs no CPU history. A directory
with at least two CPU `wrfout` files can select the compatibility replay path;
keep that mode explicit in the run report.

Resolve the duration and domain count with the user. Multiple fixed nests
use `--domains-from-namelist`; `--max-dom N` is the explicit alternative,
and those flags are mutually exclusive. Single-domain mode defaults to d01.
An explicit `--hours` wins over the namelist duration. Use a fresh output
directory and disk-backed scratch; preserve existing outputs and physics
choices. Unsupported choices fail with a named reason; do not silently
substitute a scheme or bypass preflight.

DFI admission is fail-closed: `dfi_opt=0` or omission preserves an unfiltered
forecast; any nonzero requested DFI is refused by the CLI before operational
model initialization. Programmatic forward-filter helpers are not the full
WRF operational DFI workflow, which remains future work. Admission-only change
`68704c1eb` does not replace the numerical source of existing forecast receipts.

## Install and check the GPU

Python 3.11 is the reference interpreter; the package requires Python ≥3.10.
The measured platform is RTX 5090, sm_120, CUDA 13, on a Ryzen 9 9950X host.
Other recent NVIDIA hardware is untested in these release measurements.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install "jax[cuda13]==0.10.*"
pip install -e .
export GPUWRF_WRF_ROOT=/path/to/WRF
python -c "import jax; print(jax.devices())"
```

The WRF root supplies `run/` runtime tables for the selected physics,
including Noah-MP, RRTMG and Thompson. Check the source/table provenance.
JAX must report a CUDA device for a GPU run. CPU imports alone are not a GPU
execution check. The public recipe uses the CLI directly. On a development
host that has the shared lock infrastructure, its lock proof is required;
public installations do not need that development lock. Do not default to
`--force-gpu-run`.

## Run the bundled Alpine case

```bash
python -m gpuwrf.cli run --input-dir examples/switzerland_d01 \
  --output-dir runs/switzerland --domain d01 --hours 24 \
  --scratch-dir runs/switzerland_scratch
```

The bundled 42×42×44, 3 km case initializes at 2023-01-15 00Z and uses
Thompson/RRTMG/Noah-MP/MYNN with cumulus disabled. The
[example page](examples/switzerland_d01/README.md) identifies the measured
source, CPU reference and any pending rerun. A one-hour smoke test checks
the toolchain; it does not replace the 24-hour identity evidence.

The first geometry/source run compiles and may spend minutes before stepping.
Later runs reuse compatible executables. A changed source, geometry or target
fingerprint can require compilation; copied cache files do not override a
compatibility refusal. Say whether the process is compiling or integrating.
Use `python -m gpuwrf.cli run --help` for the current controls.

## Native controls and history

- Native single-root and live-nested runs bind per-domain `radt`, `cudt`,
  `topo_shading`, `slope_rad`, `moist_adv_opt` and `scalar_adv_opt`.
  Inspect the effective-control payload; source binding is not a blanket
  fidelity verdict for every option combination.
- `time_step_sound` is honored. When zero/omitted, native runs derive WRF's
  automatic count from timestep and grid spacing (4 on the tracked grid).
  The compatibility CPU-history replay driver has a separate historical
  default of 10; do not apply that number to native forecasts.
- Full WRF history is the default. `GPUWRF_FULL_WRFOUT_VARIABLES=0` selects
  the reduced stream; the canonical variable takes precedence over the
  `GPUWRF_FULL_WRFOUT` alias. `GPUWRF_TRAINING_OUTPUT_SUBSET=1` selects the
  training subset. Counts depend on the actual file/schema; neither 104 nor
  375 is a universal active-physics count.
- Native single-root history emits initialized lead zero automatically.
  Nested lead zero is opt-in with `--emit-initial-history`.
  Nested `--aot-prefetch` defaults ON; `--no-aot-prefetch` loads at first
  dispatch. Prefetch affects readiness, not a guarantee of a cache hit.

## SST forcing: sst_update and wrflowinp

Native runs support ice-free `sst_update=1` using per-domain `wrflowinp`,
`io_form_auxinput4=2` and a positive auxiliary interval. Records start at
the run start and increase on that interval. Nonzero SEAICE and SST skin/lake
coupling are refused. Preserve requested SST forcing; do not advise turning
it off merely to get through setup. `sst_update=0` reads no auxiliary SST file.

## Memory admission: C-auto before cold fallback

Free-VRAM threshold precedence is explicit `GPUWRF_MIN_FREE_VRAM_GIB`, then
the verified geometry/settings C-auto plan, then the conservative fallback
`max(24 GiB, 0.50 × total VRAM)` (fraction is configurable). An initialized
reserved pool requires outside-pool headroom; a not-yet-reserved pool also
needs its budget. Explicit allocator settings take precedence over auto
configuration. Inspect measured VRAM and host RSS; there is no universal
26 GiB GPU or 36 GB host requirement. Cold compile needs more host memory
than a warmed forecast. Explain a refusal rather than bypassing it.

For independent cases:

```bash
scripts/run_parallel_cases.sh --out-root runs/batch CASE_A CASE_B -- \
  --domains-from-namelist --hours 24
```

The launcher admits work against both GPU and host budgets, and queues the
rest. A throughput result is distinct from individual-case latency.

## Verified checkpoint and resume

Checkpoint/resume works on the native drivers: nested runs (`max_dom > 1`) and
single-domain d01 runs initialized from `wrfinput`/`wrfbdy`. CPU-WRF replay
and non-d01 single-domain runs refuse the flags. Cadence is in root-domain
steps, not seconds; a generation is also written at the forecast end.
Retention defaults to 8 GiB, at least two generations, with 10 GiB filesystem
reserve. Resume authenticates the generation and continues the same output
stream; preserve its inputs, namelist and provenance. A generic WRF-style
history file is not a checkpoint. To continue a run past its original end,
resume with a larger `--hours` plus `--extend-run`; a changed end without the
flag, or a shorter end, is refused. Any native run whose end lies past the
`wrfbdy_d01` coverage (records x interval_seconds, and WRF's last boundary
time) is refused before it starts.

```bash
python -m gpuwrf.cli run --input-dir CASE --output-dir runs/case \
  --domains-from-namelist --hours 24 --checkpoint-dir runs/checkpoints \
  --checkpoint-interval-steps 200
python -m gpuwrf.cli run --input-dir CASE --output-dir runs/case \
  --domains-from-namelist --hours 24 --checkpoint-dir runs/checkpoints \
  --checkpoint-interval-steps 200 --resume-checkpoint VERIFIED_GENERATION
# later end time from the final generation of a finished 24 h run
python -m gpuwrf.cli run --input-dir CASE --output-dir runs/case \
  --domains-from-namelist --hours 48 --checkpoint-dir runs/checkpoints \
  --checkpoint-interval-steps 200 --resume-checkpoint VERIFIED_GENERATION --extend-run
```

Check the actual installed release's restart evidence before making a
cross-version or original-CPU-WRF restart compatibility claim.

## Report the result

Retain resolved controls, source/input hashes, timestamps, exit status and
file inventory. Inspect NetCDF dimensions/variables and finite values.
Header compatibility alone does not prove numerical fidelity. All native
cells enter the declared D6 statistics; an RMSE limit does not bound each
individual cell. The 24-hour D6 window remains strict; a separately frozen
72-hour floor annex retains raw failures and classifies them explicitly.

Current-source comparisons use original CPU-WRF. A single IC pair is empirical
spread, not a confidence interval. Keep primary/replicate identities and
member-level station qualifications alongside aggregate classifications.
Report every residual and exception with its source and evidence; do not
call an unsupported scheme, opt-out configuration or old-source gallery
validated on a new release. Speed/energy results retain their measured source
and clocks; forecasts under a new numerical source need their affected gates.
