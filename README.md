# wrf_gpu

**A JAX-native, GPU-resident, open-source re-implementation of the WRF v4 Advanced Research dynamical core and a minimum operational physics suite. Designed for single-GPU workstation-scale regional numerical weather prediction.**

> **Version**: `0.0.1` (pre-arXiv preprint companion). Versioning policy below.
> **Status**: research prototype. Validated on a single workstation only. Not for operational forecasting.
> **License**: [AGPL-3.0-or-later](LICENSE).

---

## What this is

`wrf_gpu` is a clean-slate Python/JAX rewrite of WRF's ARW dynamical core (third-order Runge-Kutta outer step with split-explicit acoustic substeps, Arakawa C-grid staggering, terrain-following dry hydrostatic pressure coordinate) plus a minimum operational physics path (Thompson microphysics, MYNN PBL, RRTMG radiation, Noah/Noah-MP-style surface). The whole forecast state is GPU-resident for the high-frequency time loop, compiled into a single `jax.lax.scan` XLA program per timestep, with **zero inter-kernel host/device transfers inside the forecast loop**.

It is not a port of the WRF Fortran source. It is an independent implementation that reproduces WRF's small-step operator structure at the savepoint level, written in approximately 200 Python source files using JAX/XLA, runnable on commodity consumer hardware.

## Why it exists

Before this codebase, the practical options for GPU-accelerated mesoscale NWP were:

- **Directive-based ports** of WRF Fortran (OpenACC, CUDA Fortran, commercial AceCAST): preserve a large existing model but inherit a ~5-7x speedup ceiling and leave unported components on the host.
- **Clean-slate C++/Kokkos rewrites** (SCREAM, HOMMEXX): require a national-laboratory supercomputer and a C++ template-heavy code base.
- **DSL-based ports** (Pace + GT4Py + DaCe, ICON-exclaim): require a stencil-DSL toolchain.
- **Machine-learning emulators** (GraphCast, Pangu-Weather, FourCastNet, GenCast, Aurora, NeuralGCM, AIFS): fast and skilful in many regimes, but bypass the physics.

What did **not** previously exist: a clean-slate, physics-faithful, single-language Python/JAX port preserving the WRF Fortran dynamical-core structure at the savepoint level, runnable on commodity hardware, and openly modifiable in a single environment.

This repository is that port.

## Current quantitative status

Measured on the development workstation (see "Hardware tested" below). All numbers are honestly reported; the project's validation discipline produced both these results and the limitations listed below them.

| Result | Value | Status |
|---|---:|---|
| 24 h regional 3 km forecast wall-clock (single GPU) | ~12 minutes | measured, reproducible (CV 0.42%) |
| Apples-to-apples speedup vs 28-rank CPU WRF on same machine, d02-only | **22.26×** | derived from `rsl.error.0000`/`rsl.out.0000` per-step CPU timing |
| Inter-kernel device-to-host transfers inside the forecast loop | **0 copies, 0 bytes** | Nsight Systems verified |
| Restart bitwise continuity | exact (max delta 0.0 on all 47 State fields) | unit-tested |
| Run-to-run repeatability | exact (bitwise across full pipeline) | unit-tested |
| 1 km full-domain memory probe | 7278 MiB / 32607 MiB (78% headroom) | measured |
| WRF small-step savepoint parity (B6, 10-step coupled) | 0.0 bitwise vs WRF Fortran | unit-tested |

## What works

- Whole-state-on-GPU regional forecast for a configurable 3 km domain
- The four-tier validation pyramid (savepoint parity, physical invariants, short-run trajectory, statistical consistency)
- Restart / repeatability / D2H invariants are unit-tested and reproducible
- WRF-compatible NetCDF wrfout output (41-variable minimum subset)
- AEMET station-observation verification scaffold (BIAS / MAE / RMSE / Fractions Skill Score)
- Multi-day batch driver and 1-step 1km memory probe

## What remains (release blockers for operational claims)

`wrf_gpu` is currently a research prototype, not an operational replacement for CPU WRF. Side-by-side station scoring on T2, U10 and V10 against CPU WRF showed the GPU forecast is **materially less skilful** even after two iterations of algorithmic fixes (theta-guard envelope widening, surface→PBL flux wiring, RRTMG cadence enabling, hourly land-surface refresh, lateral boundary width-5 strip).

The remaining defect has been narrowed by validation to:

- A surface-flux magnitude / sign-coupling issue that drives T2 overshoot once the theta envelope is widened to allow diurnal warming.
- Residual theta-guard saturation in the lower 30 levels.
- The land-surface refresh path is a data replay from a reference CPU run, not a prognostic Noah-MP scheme.

These are publication-blocking for an operational-replacement claim. They are not publication-blocking for documenting the artifact, the architecture, and the methodology — which this preprint does.

## Hardware tested

This codebase has been validated **on a single workstation only**. Behaviour on any other hardware combination is untested.

| Component | Tested |
|---|---|
| CPU | AMD Ryzen 9 (32 cores; AI workers pinned to cores 0-3, CPU WRF reference baseline on cores 4-31) |
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32 607 MiB VRAM) |
| OS | Linux 6.17 x86_64 |
| Driver | NVIDIA 595.71.05 |
| CUDA runtime | 13.1.115 |
| Python | 3.13.11 |
| JAX | 0.10.0 (jaxlib 0.10.0) |

If you run this on different hardware, please open a bug report (see below) with `nvidia-smi`, `python --version`, `jax.print_environment_info()` output, and the offending command. We have no current evidence that it works elsewhere.

## How it was built

`wrf_gpu` was authored collaboratively by an AI-agent process supervised by a human senior author. The methodology is documented in [AI_USE.md](AI_USE.md) and is the subject of the accompanying preprint at [paper/paper.pdf](paper/paper.pdf). Briefly:

- **Manager agent** (Claude Opus 4.7, long-context orchestrator) owned sprint definition, repository-state synthesis, and final closeout recommendations.
- **Worker agent** (GPT-5.5 Codex, OpenAI) executed scoped changes under per-sprint contracts.
- **Tester / reviewer agents** challenged the worker's results with independent re-runs and verdict tokens.
- **Per-sprint contracts** named acceptance criteria, file ownership, and required proof objects on disk before any "done" claim.

This proof-object-driven workflow produced the artifact, found and corrected its own publication-blocking overclaim (the original 156× speedup figure was reduced to the honest 22.26× during a self-initiated audit), and preserved the evidence trail. The complete methodology, the failures, and the lessons are documented in the preprint.

## Quick start

```bash
set -euo pipefail
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
JAX_PLATFORMS=cpu pytest -q tests/ -k 'not gpu'
```

See [INSTALL.md](INSTALL.md) for CPU, full-GPU, and reviewer-frozen install paths. For a real forecast you need: a CUDA 13.x-capable NVIDIA driver, JAX with CUDA support, an initial-condition NetCDF file in WRF wrfinput format, and an hourly boundary-forcing source. See [paper/paper.md](paper/paper.md) §4-§6 for the model interfaces.

## Reference data

Public unit tests run without private reference data. Scripts that replay retained WRF runs or score AEMET observations read external data from `WRF_GPU_REFERENCE_ROOT`. That directory should contain the same relative layout used in the proof objects, such as `runs/wrf_l3`, `runs/wrf_l2`, and `artifacts/datasets/aemet_stations`. Proof JSONs in this repository use `<reference-data-root>` wherever the original workstation path appeared.

## How to verify the paper numbers

Run the lightweight verifier:

```bash
bash scripts/verify_reproducibility.sh
```

The paper's canonical proof objects are under [proofs/](proofs/), with paths flattened as `<sprint-id>__<filename>.json`. Start with:

- [proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_speedup.json](proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_speedup.json)
- [proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json](proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json)
- [proofs/2026-05-27-m7-profiler-window-fix__d2h_audit_v2.json](proofs/2026-05-27-m7-profiler-window-fix__d2h_audit_v2.json)
- [proofs/2026-05-27-m7-restart-continuity__restart_continuity.json](proofs/2026-05-27-m7-restart-continuity__restart_continuity.json)

## Repository layout

```
wrf_gpu/
├── src/gpuwrf/         model implementation (dynamics, physics, runtime, validation, io)
├── tests/              pytest suite
├── scripts/            CLI orchestrators and audit tools
├── paper/              preprint (paper.md, paper.pdf, references.bib, honesty_audit.md)
├── tables/             comparison + skill-evolution + benchmark tables (markdown)
├── figures/            figure specifications (markdown; renderable to PNG/PDF later)
├── manifest/           release manifests (environment, proof-object pointers, git state)
├── proofs/             canonical proof JSONs from the validation sprints
└── .github/            issue templates + PR template + minimal CI workflow
```

## Citation

If you use `wrf_gpu` in academic work, please cite using [CITATION.cff](CITATION.cff). When the arXiv preprint becomes available, this README will be updated with the preprint DOI and a BibTeX entry.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must be under AGPL-3.0-or-later. Forks and modifications, including those hosted as a service, must release source under AGPL-3.0-or-later (see [LICENSE](LICENSE)).

## Bug reports and feedback

Open an issue at <https://github.com/wrf-gpu/wrf_gpu/issues>. Issue templates are available for bug reports and feature requests. For security concerns, see [SECURITY.md](SECURITY.md).

## Versioning policy

The project uses semantic versioning with the following pre-arXiv schedule:

- `0.0.x` — pre-arXiv-preprint releases. Public reproducibility snapshots; no operational-quality claims.
- `0.1.0` — arXiv preprint companion release. Locked source corresponding to the cited paper version.
- `0.x.y` (post-preprint) — improvements, bug-fixes, additional validation evidence.
- `1.0.0` — reserved for when the operational-skill blockers in "What remains" are closed and a peer-reviewed paper has accepted the claim.

Each release is git-tagged and (when configured) automatically archived with a per-version DOI via Zenodo.

## License and liability

Licensed under [AGPL-3.0-or-later](LICENSE). The author (Enric R.G.) is a single individual maintaining this as a research project. **No warranty of any kind is provided. No liability is accepted for any use of this software, including but not limited to forecast outputs used for downstream decisions, financial loss, safety-critical applications, or any other consequence.** See the LICENSE file for the full warranty disclaimer (sections 15 and 16 of AGPL-3.0).

This codebase is independent of and not endorsed by the National Center for Atmospheric Research (NCAR) or the WRF community. WRF and ARW are trademarks of their respective owners; see [NOTICE](NOTICE) for derivative-work clarification.

## Acknowledgements

- The WRF community at NCAR/UCAR for the open ARW reference implementation and technical notes that anchored the savepoint validation.
- The JAX and XLA teams at Google for the compiler and runtime.
- Anthropic (Claude) and OpenAI (Codex/GPT-5.5) for the AI co-authoring tools.
