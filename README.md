# wrf_gpu

**WRF-compatible forecasts on one NVIDIA GPU.**

`wrf_gpu` is a GPU rewrite of WRF v4's ARW dynamics and physics, validated against the
original Fortran. It reads standard WRF inputs and writes WRF history for existing
preparation and analysis tools.

On an RTX 5090, v0.3 is **22.4× faster than CPU-WRF on the two-domain Canary
case**, and delivers **12.75× its throughput with 4 Tenerife
forecasts sharing one GPU**.

| Forecast | CPU-WRF, 12 cores | One RTX 5090 |
|---|---:|---:|
| Canary, 9/3 km, 24 h | 92.3 s/forecast hour | **4.12 s/forecast hour** |
| Tenerife, six-hour parallel benchmark | 123.4 s/case-hour, 3 × 4 cores | **9.68 s/case-hour**, 4 cases |
| Tenerife, 72 h production batch: six cases, two waves of three | 123.4 s/case-hour, 3 × 4 cores | **6.28 s/case-hour**, 19.6× |

Rates are measured [M], including startup and output; parallel rates include
compression and launcher completion. N=3 and N=4 are within 0.5%: the GPU is saturated.

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/parallel_dark.png"><img src="docs/release/img/parallel.png" width="100%" alt="Forecast throughput, GPU memory and energy as more Tenerife cases share one RTX 5090"></picture></p>

## Try the bundled example

Tested on RTX 5090 (sm_120, CUDA 13); other recent NVIDIA GPUs should work but are
untested. With Python 3.11+ and the matching NVIDIA driver:

```bash
pip install "jax[cuda13]==0.10.*"
pip install -e .
export GPUWRF_WRF_ROOT=/path/to/WRF   # WRF v4 radiation and land-model tables
python -m gpuwrf.cli run --input-dir examples/switzerland_d01 \
  --output-dir runs/switzerland --domain d01 --hours 24 \
  --scratch-dir runs/switzerland_scratch
```

The [Switzerland example](examples/switzerland_d01/README.md) includes the CPU-WRF
comparison and benchmark. The first run compiles; later runs reuse the executable.

For your own WPS/`real.exe` case, use `--domains-from-namelist`. For several cases:

```bash
scripts/run_parallel_cases.sh --out-root runs/batch CASE_A CASE_B CASE_C -- \
  --domains-from-namelist --hours 24
```

The launcher checks GPU and host memory. See the
[User's Guide](https://wrf-gpu.github.io/wrf_gpu/) for setup, or
[AI_OPERATOR.md](AI_OPERATOR.md) to have an AI assistant help run a forecast.

## What's new in v0.3

- Fused fp32 kernels default on; 1.4× the throughput of the late v0.23.4 development baseline. Previous public release: v0.23.4.
- WRF-order corrections to snow, water and albedo updates.
- Full WRF history is the default.

See [CHANGELOG.md](CHANGELOG.md) and [release notes](release_notes/README.md).

## How the rewrite works

**JAX/XLA and Pallas** compile expensive dynamics and physics into fused fp32 GPU
kernels. State stays on the GPU; WRF's update order and required double precision
are preserved. [Methods](docs/release/METHODS.md) cover architecture, timing,
energy and B200 extrapolation.

## Checked against CPU-WRF

The release gate covers **6 Tenerife cases × three domains × 24 h**.
375 output fields are compared with original CPU-WRF; 62 have
frozen per-variable numerical limits, called D6. There are **0 bound failures**;
all compared numeric values finite. Output integrity: a strict check flags a few trace-level fields (ice/snow traces, canopy dew onset, one known writer field); every exception is [listed](docs/release/VALIDATION.md#wn3-output-integrity).

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/release/img/identity_curves_wn3_heatmap_dark.png"><img src="docs/release/img/identity_curves_wn3_heatmap.png" width="100%" alt="GPU minus CPU-WRF RMSE as a share of the frozen limit across all six cases, three domains and the 24-hour release window"></picture></p>

Component tests compare with pristine WRF Fortran. Station verification is done
separately by the ALISIOS forecasting project.
Beyond 24 h, a complete six-case comparison on an earlier internal build of v0.3.0 (1fece48da; identical except the Noah-MP albedo/snow-aging fix) exceeds limits on the 1 km domain: 0227 U10/V10 from +41 h, V at +67 h, W at +69–70 h and RAINNC at +67–72 h; 0120 U10 at +58 h. The scored v0.3.0 0227 case agrees with that displaced wake/convection pattern. [Exact hours and signed biases](docs/release/VALIDATION.md) preserve every failure. The Canary 162 h comparison stays within its limits.

Tested physics: Thompson, RRTMG, MYNN, Noah-MP, outer-domain Kain–Fritsch and GWDO,
with one-way nesting. Execution is single-GPU; MPI/multi-GPU are unsupported.
Glacier cells use ordinary Noah-MP; the dedicated glacier runtime is unported.
[Validation](docs/release/VALIDATION.md), [fidelity fixes](docs/release/FIDELITY_FIXES.md)
and [known issues](KNOWN_ISSUES.md) disclose fill, trace-ice and canopy differences.

Measurements: v0.3.0 (internal build dead7d004), 2026-10-04. CPU energy uses **≈200 W on 12 cores
(maintainer measurement)**; GPU energy is board power, excluding host CPU.
[Receipts](docs/release/PROVENANCE.md) identify every figure's source.

## Get involved

Interested in the rewrite? Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[methods](docs/release/METHODS.md); questions and [issues](https://github.com/wrf-gpu/wrf_gpu/issues) are welcome.

See [LICENSE_NOTES.md](LICENSE_NOTES.md) for attribution and licensing information.
