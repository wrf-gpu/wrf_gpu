# Contributing to wrf_gpu

Thanks for your interest. This document covers how to contribute, what kinds of contributions are welcome, and the licensing expectations that follow from AGPL-3.0-or-later.

## tl;dr

- Open an issue first for anything non-trivial.
- Fork → branch → PR. Run `pytest -q tests/` before pushing.
- All contributions are accepted under AGPL-3.0-or-later. By submitting a PR you confirm that you have the right to license the contribution that way.
- If you fork this project and distribute the modified version (including hosting it as a service), you must release your modifications under AGPL-3.0-or-later. This is the licensing intent; the LICENSE file is the legal document.

## What contributions are welcome

- **Bug reports** with reproduction steps and the relevant environment info (`nvidia-smi`, `python --version`, `jax.print_environment_info()`).
- **Test cases** that exercise corners of the dycore, physics couplers, or boundary handling.
- **Documentation** improvements, especially install / quickstart on hardware other than the development workstation.
- **Performance improvements** that preserve the validation invariants in `tests/` (savepoint parity, restart bitwise, repeatability, D2H = 0).
- **New physics couplers** following the existing `src/gpuwrf/coupling/physics_couplers.py` interface (proposal first in an issue).
- **Multi-GPU / multi-node** scaling work using the `halo` placeholder interface (proposal first in an issue; this is non-trivial).
- **Validation evidence** on hardware other than the development workstation, especially the idealized test cases in `src/gpuwrf/fixtures/idealized_cases/`.

## What contributions are **not** welcome (yet)

- Large refactors with no behavioural justification.
- Changes that break or weaken the unit-tested invariants without a documented physics / numerics reason.
- Speed claims without a corresponding profiler artifact (Nsight Systems or equivalent) attached to the PR.
- Dependency upgrades that change the JAX / jaxlib / CUDA major version without a documented test pass.

## Development setup

The full install matrix is in [INSTALL.md](INSTALL.md). The standard CPU path is:

```bash
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q tests/ -k 'not gpu'
```

The full validation suite requires an NVIDIA GPU with CUDA 13.x driver and ~32 GB of VRAM. Most unit tests run on CPU.

## CPU pinning convention

The project pins Python orchestrators to CPU cores 0-3 (`taskset -c 0-3`) on the development workstation. This convention preserves the remaining cores for an external CPU WRF reference run that is used as the speedup denominator. It is not required for your environment, but is mentioned because some scripts assume it.

## Pull-request checklist

Before opening a PR:

- [ ] `pytest -q tests/` passes on your machine (or you explain why it doesn't).
- [ ] `bash scripts/verify_reproducibility.sh` passes if the change touches paper-facing evidence, proof objects, or install documentation.
- [ ] The change is described in the PR body: what, why, evidence.
- [ ] If the change touches `src/gpuwrf/dynamics`, `src/gpuwrf/coupling`, `src/gpuwrf/runtime`, `src/gpuwrf/contracts`, or `src/gpuwrf/io`, you have run the validation-invariant tests under `tests/test_m6*.py` and `tests/test_m7*.py` that are relevant.
- [ ] If the change makes a performance claim, a profiler artifact is attached or linked.
- [ ] If you added or modified a public function, you added or modified a docstring and a unit test.
- [ ] `git log` is clean (no merge commits, no debug commits, no committed secrets).

## Issue templates

Use the bug-report or feature-request template in `.github/ISSUE_TEMPLATE/`. They exist to make triage tractable for a small maintainer team.

## Maintainership

This project is maintained as a research-grade hobby project. There is no SLA, no support contract, and no commercial entity behind it. The current maintainer is Enric R.G. with the AI co-authors named in [CITATION.cff](CITATION.cff). The maintainership may be transferred to the GitHub organisation `wrf-gpu` admins at any time; the AGPL-3.0-or-later license keeps the project open regardless of maintainer changes.

## AI co-authorship disclosure

This codebase was authored substantially by AI systems (Claude Opus 4.7 and GPT-5.5 Codex) under human supervision. Contributions to the project may be reviewed and integrated using AI assistance. The human maintainer takes final responsibility for what lands in `main`. See [AI_USE.md](AI_USE.md) for the disclosure that accompanies the preprint.

## Security

For security-sensitive disclosures, do not open a public issue. See [SECURITY.md](SECURITY.md) for the responsible-disclosure pathway.

## Code of conduct

By participating in this project you agree to abide by the Contributor Covenant code of conduct ([CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).
