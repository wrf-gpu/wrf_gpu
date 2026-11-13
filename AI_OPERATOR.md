# AI Operator Runbook for wrf_gpu

Help the user run an existing WRF case with the public CLI. This is the
operator runbook; model development follows separate developer instructions.

**Version:** v0.4.0 released 2026-10-17. Defaults are frozen on 5c15881ede:
Swiss 3356e249 / Canary Islands 9 km domain 196aed9f, with the CPU-WRF field-limit (D6) and compiled-kernel resource checks closed.
At 00:26Z on 17 October, 66/66 registered FINAL composites pass their accepted scientific scope;
all 66 registered scientific scopes pass. Consult the [exact-source table](docs/release/FINAL_COMPOSITES_v0.4.0.md).
Use the installed release's matching guide; do not recommend an unreleased
checkout as a published release. Read the [support matrix](docs/release/SUPPORT_MATRIX_v0.4.0.md),
[upgrade note](docs/release/UPGRADE_v0.4.0.md) and [known issues](KNOWN_ISSUES.md).
Historical [v0.3.3 comparisons](docs/release/V0.3.3.md) retain their sources.

Explain each material action and the expected wait. During compilation or
integration, report progress and name any blocker. Use original CPU-WRF or
pristine WRF as fidelity truth; older GPU output is regression evidence.

Read the [v0.4.0 quickstart/command guide](docs/release/USER_GUIDE_v0.4.0.md) for verified workflow names (`quickstart`, `cycle`, `ens`, `optimize`).

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

DFI modes `dfi_opt=1` (DFL), `2` (DDFI) and `3` (TDFI) run in supported
single-domain native configurations, subject to filter/land/forcing/window
rules. DFL has natural shifted timestamps; do not pad or retime them. Nested
DFI and unsupported combinations are refused. Zero/omission keeps an unfiltered
forecast; resume never filters. Check [source-specific evidence](docs/release/SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows).

Smagorinsky `km_opt=3` is GPU-refused. CAM-MG `mp_physics=11` uses the
merged stock-Triton3.6.x route and requires `pip install "triton==3.6.*"`.
Its separate offline native-kernel stack failure (1240 B) is disclosed; the registered-source CAM-MG composite now passes the CPU-WRF field limits, the output inventory and the finiteness check; the native compiled-kernel resource check remains unknown. Historical failures remain failed.
See [CAM-MG scope](docs/CAMMGMP.md).

## Install and check the GPU

Minimum NVIDIA GPU: **Ampere, compute capability ≥8.0**, for the Pallas/Triton
release fast path. Pascal and Turing are unsupported; seeing a CUDA device
alone does not meet this requirement. The legacy GTX1080Ti debug shim is not
an operational fallback. [Hardware scope](proofs/r_docs040/GPU_REQUIREMENTS.json).

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
- RRTMG SW surface direct beam: the default (since v0.4.1) is WRF's unscaled
  `ztdbt_nodel` (SWDDIR/SWDDIF, and the diffuse fraction that `slope_rad` /
  `topo_shading` use). `GPUWRF_SWDDIR_V040_SPLIT=1` is an opt-out that restores
  the v0.4.0 delta-scaled split bitwise. It is set before the process starts, it
  compiles a different program (new cache keys), and it is refused with
  `aer_opt=1/3`. It is not WRF's product; use it only for continuity comparisons.
- Native single-root history emits initialized lead zero automatically.
  Nested lead zero is opt-in with `--emit-initial-history`.
  Nested `--aot-prefetch` defaults ON; `--no-aot-prefetch` loads at first
  dispatch. Prefetch affects readiness, not a guarantee of a cache hit.

## SST forcing: sst_update and wrflowinp

Native runs support ice-free `sst_update=1` using per-domain `wrflowinp`,
`io_form_auxinput4=2` and a positive auxiliary interval. Records start at
the run start and increase on that interval. Nonzero SEAICE and unsupported skin/lake/forcing combinations are refused;
check the selected pair. Preserve requested SST forcing; do not advise turning
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

On a machine with several GPUs (ensembles, many cases), add `--gpus all` (or
`--gpus 0,1`): every case runs in its own process bound to one GPU
(`CUDA_VISIBLE_DEVICES`, PCI bus order), sized and admitted per GPU; host RAM
is shared. One failed member never stops the others. `--checkpoint-interval-steps N`
gives each member its own checkpoint store, and `--retry-failed K` re-runs a
failed member from its newest checkpoint (else from scratch, keeping the
failed history under `failed-attempt-<k>/`). All GPUs must be the same model.
Experimental: CPU-tested with simulated devices; multi-GPU hardware runs are
not yet qualified.

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

## v0.4 tools around a forecast

Use `python -m gpuwrf.cli --help` and the selected command's `--help` for the
installed syntax. These workflows supplement the forecast:

| Need | Tool / definition |
| --- | --- |
| Diagnose inputs and record provenance | [Doctor, manifest and reproduce](docs/doctor.md) |
| Prepare a region with original WPS/real.exe | `gpuwrf quickstart`; [onboarding](docs/onboarding.md) |
| Forecast/application products | `gpuwrf post`; [products](docs/post.md), [applications](docs/post_apps.md) |
| Observation verification | [Station scores and reports](docs/station_verification.md) |
| Ensemble products/preparation | [Ensemble workflows](docs/ensemble.md) |
| CF-NetCDF/Zarr/GRIB2 | `gpuwrf export`; [formats/dependencies](docs/export.md) |
| Station and regional outputs | `gpuwrf tslist-export`, `gpuwrf stream-check`; [stream qualification](proofs/u2_streams/QUALIFICATION.json) |
| Regional optimisation / land cycling | [Optimisation](docs/regional_optimization.md), [cycling](docs/cycling.md) |

### v0.5.0 experiment options (all off by default)

| Need | Switch | Guide and limits |
| --- | --- | --- |
| Many surface variants from one spin-up | `gpuwrf run --surface-controls`, `scripts/branch_series.py` | [Branch series](docs/checkpoint-branching.md#branch-series-many-value-variants-from-one-spin-up) (one-way nests only) |
| Stochastic physics in nests | WRF `&stoch` per domain (full vectors) | [Stochastic](docs/stochastic.md) (`rand_perturb=1` has no CPU-WRF reference) |
| Reproducible θ/qv noise in inputs | `gpuwrf perturb --thermo-config` | [Seeded noise](docs/stochastic.md#seeded-boundary-layer-temperature-and-moisture-noise) |
| Land use, roughness, terrain edits | `gpuwrf patch land/z0/geo-em/check-real/verify` | [Patches](docs/static_patch.md) (no GPU forecast with a patched case yet) |
| Evaluation files during the run | `gpuwrf run --extract-spec v1 --extract-config REQ.json --extract-dir DIR --emit-initial-history` | [Extract](docs/extract.md) |
| AFWA 10 m wind maximum | `&afwa afwa_diag_opt=1, afwa_severe_opt=1` | [WSPD10MAX](docs/ONLINE_DIAGNOSTICS.md#afwa-severe-10-m-maximum-wind-wspd10max) (Thompson/MYNN/Noah-MP only) |
| Several members per GPU | `scripts/parallel_cases.py --max-parallel N` | [Measured N=1/2/3](docs/ensemble.md#several-members-on-one-gpu-measured-with-v050-development-code) |
| Shared compile store across builds | `GPUWRF_AOT_HLO_STORE`, `GPUWRF_AOT_HLO_SEEDS` | [Experimental store](docs/aot_content_store.md) (not byte-identical across builds) |

Read [known issues](KNOWN_ISSUES.md#v050-limits-of-the-new-options) before combining these options.

Product definitions distinguish instantaneous samples from interval totals
and extrema. A finite smoke, a matching header or an analysis tool does not
establish station skill. Selected large runtime tables must match the installed
release's SHA-pinned inventory; follow its asset-fetch instructions rather than
substituting tables. Final packaging details belong to the release maintainers.

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
