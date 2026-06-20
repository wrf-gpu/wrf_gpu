#!/usr/bin/env python3
"""v0.13 community-standard validation aggregator (CPU-only).

Assembles the project's community-standard validation evidence that an outside
reviewer would expect, reusing existing proof generators / tests WITHOUT
modifying them:

  1. Idealized dycore gates -- Straka 1993 density current + Skamarock / Bryan-
     Fritsch warm bubble, re-run on CPU via the existing
     ``gpuwrf.ic_generators.idealized`` runner, checked against the published
     WRF benchmark spec (front position, bounded theta', active w, mass drift).
  2. Closed-domain conservation budgets -- the existing dry-mass / total-water /
     moist-static-energy budget closure (relative residual ~0 in fp64), via the
     existing ``tests/test_conservation_budget.py`` CPU controlled gate.
  3. Bitwise restart -- the existing full-carry NetCDF wrfrst write->read->compare
     bit-identity round-trip (state + carry + stochastic seeds), via the existing
     ``v0110_restart_proof._cpu_full_carry_roundtrip`` structural gate and the CPU
     restart pytest suite.

This is a thin aggregator: it RUNS the existing gates and records their verdicts,
the published benchmark reference numbers, and an honest CPU-vs-GPU/data gap list.
It does NOT invent new heavy multi-day corpus runs (those need a GPU + the purged
CPU-WRF corpus and are listed as the documented gap).

CPU-only by construction: callers must export ``JAX_PLATFORMS=cpu``; this module
asserts no GPU device is selected and never forces a GPU context.

Exit 0 = every CPU-reproducible community gate green; non-zero = a gate failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ROOT on the path so the existing (non-package) ``scripts/`` proof generators
# are importable for reuse without modifying them.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "proofs" / "v013" / "community_validation.json"


# ---------------------------------------------------------------------------
# Published WRF / community benchmark reference spec (what an outside reviewer
# checks the idealized cases against). Numbers/URLs are the project's vendored
# reference values, mirrored from the idealized fixtures + runner so the proof
# is self-describing.
# ---------------------------------------------------------------------------
IDEALIZED_BENCHMARK_SPEC = {
    "density_current": {
        "name": "Straka et al. 1993 density-current benchmark",
        "published_targets": {
            "front_speed_m_s": 33.0,
            "front_position_m_at_900s": 15000.0,
            "integration_s": 900,
            "dx_m": 100.0,
            "diffusion_nu_m2_s": 75.0,
        },
        "purpose": "cold-pool propagation, sharp-gradient handling, rotor structure",
        "references": [
            "https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html",
            "https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml",
        ],
    },
    "warm_bubble": {
        "name": "Bryan and Fritsch 2002 / Skamarock-Klemp dry warm-bubble benchmark",
        "published_targets": {
            "theta_perturbation_k": 2.0,
            "integration_s": 500,
        },
        "purpose": "buoyant response, acoustic stability, left/right symmetry, mass conservation",
        "references": [
            "https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf",
        ],
    },
}


def _ensure_cpu_only() -> str:
    """Assert we are NOT on a GPU backend (the GPU is owned by another lane)."""
    import jax  # local import so JAX_PLATFORMS is honored before init

    backend = jax.default_backend()
    if backend == "gpu":
        raise SystemExit(
            "community_validation must run CPU-only (JAX_PLATFORMS=cpu); a GPU "
            "backend was selected. The GPU is owned by another lane."
        )
    return backend


# ---------------------------------------------------------------------------
# Gate 1: idealized dycore cases (CPU re-run of the existing runner).
# ---------------------------------------------------------------------------
def _run_idealized(workdir: Path) -> dict[str, Any]:
    from gpuwrf.ic_generators.idealized import (
        run_density_current_case,
        run_warm_bubble_case,
    )

    cases: dict[str, Any] = {}
    overall = True
    for case_key, runner in (
        ("density_current", run_density_current_case),
        ("warm_bubble", run_warm_bubble_case),
    ):
        t0 = time.perf_counter()
        result = runner(proof_dir=workdir, require_gpu=False)
        wall_s = time.perf_counter() - t0
        passed = result.verdict == "PASS"
        overall = overall and passed
        cases[case_key] = {
            "pass": passed,
            "verdict": result.verdict,
            "status": result.status,
            "wall_s": round(wall_s, 1),
            "benchmark_spec": IDEALIZED_BENCHMARK_SPEC[case_key],
            "checks": {
                name: {
                    "value": row.get("value"),
                    "threshold": row.get("threshold"),
                    "passed": bool(row.get("passed")),
                }
                for name, row in result.checks.items()
            },
        }
    return {"pass": overall, "cases": cases}


# ---------------------------------------------------------------------------
# Gate 2: closed-domain conservation budgets (CPU controlled gate, reused).
# ---------------------------------------------------------------------------
def _run_conservation() -> dict[str, Any]:
    from gpuwrf.diagnostics.conservation_budget import PREDECLARED_TOLERANCES

    proof_path = ROOT / "proofs" / "p0_7" / "conservation_budget_cpu_controlled.json"
    rc = _pytest(["tests/test_conservation_budget.py"])
    payload: dict[str, Any] = {
        "pass": rc == 0,
        "gate": "tests/test_conservation_budget.py",
        "predeclared_tolerances": dict(PREDECLARED_TOLERANCES),
    }
    if proof_path.is_file():
        controlled = json.loads(proof_path.read_text())
        payload["closed_domain_residuals"] = {
            "dry_mass_relative_residual": controlled["closed_control"]["dry_mass_relative_residual"],
            "total_water_relative_residual": controlled["closed_control"]["water_relative_residual"],
            "moist_static_energy_residual_j": controlled["open_lbc_corrected_control"]["moist_static_energy_residual_j"],
            "passes_predeclared_tolerance": controlled["closed_control"]["passes_predeclared_tolerance"],
        }
        payload["proof_object"] = str(proof_path.relative_to(ROOT))
        payload["platform"] = controlled.get("platform")
    return payload


# ---------------------------------------------------------------------------
# Gate 3: bitwise restart round-trip (CPU structural gate, reused).
# ---------------------------------------------------------------------------
def _run_restart(workdir: Path) -> dict[str, Any]:
    from scripts import v0110_restart_proof as restart_mod  # type: ignore

    structural = restart_mod._cpu_full_carry_roundtrip(workdir)
    rc = _pytest(
        [
            "tests/test_p0_5_restart_full_carry.py",
            "tests/test_v0110_wrfrst_netcdf.py",
            "tests/test_m7_restart_checkpoint_roundtrip.py",
        ]
    )
    comparison = structural.get("comparison", {})
    seeds = structural.get("stochastic_seed_roundtrip", {})
    return {
        "pass": bool(structural["pass"]) and rc == 0,
        "gate": "v0110_restart_proof._cpu_full_carry_roundtrip + CPU restart pytest suite",
        "bit_identical_full_carry": bool(comparison.get("pass")),
        "bit_identical_stochastic_seeds": bool(seeds.get("pass")),
        "schema": structural.get("schema"),
        "purpose": structural.get("purpose"),
        "note": (
            "Structural NetCDF wrfrst write->read->compare bit-identity over the "
            "full state+carry+stochastic-seed field set. The multi-hour "
            "forecast-continuity acceptance gate (restart trajectory == "
            "uninterrupted trajectory) needs a GPU + real corpus -- see gap list."
        ),
    }


# ---------------------------------------------------------------------------
def _pytest(targets: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-2000:])
    return proc.returncode


CPU_GPU_GAP = {
    "cpu_reproducible_from_repo_alone": [
        "Idealized dycore gates (Straka density current, Skamarock/Bryan-Fritsch warm bubble) -- this aggregator re-runs them on CPU.",
        "Closed-domain dry-mass / total-water / moist-static-energy budget closure (fp64).",
        "Bitwise restart: full state+carry+stochastic-seed NetCDF wrfrst write->read->compare bit-identity.",
        "CPU physics savepoint-parity proofs (see scripts/verify_reproducibility.sh).",
    ],
    "needs_gpu": [
        "Speedup / throughput / per-watt and multi-GPU (DGX) claims -- require an NVIDIA GPU + profiler artifacts.",
        "1km nested live-forecast stability gates (d03) and GWD-nested gates.",
        "Multi-hour restart forecast-CONTINUITY acceptance (restart trajectory == uninterrupted) -- the structural bit-identity above is CPU; the trajectory match needs GPU + corpus.",
    ],
    "needs_purged_corpus": [
        "TOST operational equivalence vs 28-rank CPU-WRF (proofs/m20/*) -- needs real CPU-WRF wrfout + AIFS forcing (not redistributable; <DATA_ROOT>).",
        "Multi-day operational skill-vs-obs gates and station scoring.",
    ],
}


def build(workdir: Path) -> dict[str, Any]:
    backend = _ensure_cpu_only()
    idealized = _run_idealized(workdir)
    conservation = _run_conservation()
    restart = _run_restart(workdir)
    overall = bool(idealized["pass"] and conservation["pass"] and restart["pass"])
    return {
        "schema": "v013_community_validation_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "jax_backend": backend,
        "cpu_only": True,
        "pass": overall,
        "gates": {
            "idealized_dycore": idealized,
            "closed_domain_conservation": conservation,
            "bitwise_restart": restart,
        },
        "cpu_gpu_data_gap": CPU_GPU_GAP,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="community_validation_") as tmp:
        proof = build(Path(tmp))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"pass": proof["pass"], "artifact": str(args.output)}, sort_keys=True))
    for gate_name, gate in proof["gates"].items():
        print(f"  [{'PASS' if gate['pass'] else 'FAIL'}] {gate_name}")
    return 0 if proof["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
