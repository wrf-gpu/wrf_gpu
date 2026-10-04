#!/usr/bin/env python3
"""Emit the M0 environment / case / control manifests (contract §5.1, §12).

``control_manifest.json.in`` is deliberately a *template*, not the final object:
contract §5.1 makes CONTROL-v025 final only once the GPU-side fields (prepared
runtime/AOT state, cold-compile and cached-load timings, clocks/power policy
under load) are measured, and §13 forbids touching the GPU before the CPU-first
checkpoint is complete. Every field that is not yet measured is emitted as
``null`` with a ``pending`` reason rather than guessed, so the gap is visible in
the artifact itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports

from fast_case import (  # noqa: E402
    MPIRUN,
    SOURCE_BUNDLE,
    WRF_EXE,
    case_descriptor,
    derive_fast_namelist,
    sha256_file,
    verify_source_inputs,
)

REPO = Path(__file__).resolve().parents[2]


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=60
        ).stdout.strip() or None
    except Exception:
        return None


def environment_manifest() -> dict:
    import jax
    import jaxlib

    return {
        "schema": "wrf_gpu2.v025.m0.environment_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "logical_cpus": len(list(Path("/proc/cpuinfo").read_text().split("processor"))) - 1,
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "python": sys.version.split()[0],
        "python_prefix": sys.prefix,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_backends_available": None,
        "jax_backend_probe_note": (
            "NOT probed: importing a GPU backend counts as a GPU touch under "
            "contract §13 and requires dual-manager coordination first"
        ),
        "nvidia_smi_query": None,
        "nvidia_smi_query_note": (
            "NOT queried by the CPU manifest builder. Device identity/telemetry is "
            "captured only by the separately coordinated manager-owned executor."
        ),
        "nvcc_version": _cmd("nvcc", "--version"),
        "nsys_path": _cmd("which", "nsys"),
        "ncu_path": _cmd("which", "ncu"),
        "mpirun": str(MPIRUN),
        "mpirun_version": (_cmd(str(MPIRUN), "--version") or "").splitlines()[:1],
        "wrf_exe": str(WRF_EXE),
        "wrf_exe_sha256": sha256_file(WRF_EXE.resolve()),
        "wrf_exe_realpath": str(WRF_EXE.resolve()),
        "git_commit": _cmd("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_branch": _cmd("git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"),
        "src_gpuwrf_tree": _cmd("git", "-C", str(REPO), "rev-parse", "HEAD:src/gpuwrf"),
        "pending": {
            "gpu_clocks_and_power_policy_under_load": "requires a coordinated GPU window",
            "package_lock": "emitted separately by --pip-freeze",
        },
    }


def case_manifest() -> dict:
    desc = case_descriptor()
    observed = verify_source_inputs()
    derived_text, edits = derive_fast_namelist(
        (SOURCE_BUNDLE / "namelist.input").read_text(), max_dom=1
    )
    import hashlib

    return {
        "schema": "wrf_gpu2.v025.m0.case_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": desc,
        "source_inputs": {
            "bundle": str(SOURCE_BUNDLE),
            "expected_sha256": desc["expected_source_hashes"],
            "observed_sha256": observed,
            "independently_verified": observed == desc["expected_source_hashes"],
        },
        "derived_namelist": {
            "sha256": hashlib.sha256(derived_text.encode()).hexdigest(),
            "note": (
                "FAST-v025 runs a DERIVED namelist: the source bundle describes the "
                "full 2-domain 162 h production run. The derivation below is the "
                "complete diff; regenerating it reproduces this hash byte-for-byte."
            ),
            "derivations": [vars(e) for e in edits],
        },
        "provenance_chain": {
            "identity_chain": "forcing manifest -> WPS receipt/met_em -> real.exe wrfinput/wrfbdy",
            "production_run_receipt": str(SOURCE_BUNDLE.parent / "receipt.json"),
            "launch_identity": str(SOURCE_BUNDLE.parent / "wrf_launch_identity.json"),
        },
        "cpu_executable": {
            "path": str(WRF_EXE),
            "realpath": str(WRF_EXE.resolve()),
            "sha256": sha256_file(WRF_EXE.resolve()),
            "version": "WRF V4.7.1 (dmpar)",
        },
    }


def control_manifest_template() -> dict:
    """Contract §5.1 binding list, with unmeasured fields explicitly null."""
    pending = lambda why: {"value": None, "status": "PENDING", "reason": why}  # noqa: E731
    gpu = "requires a coordinated GPU window (contract §13); not measured on CPU"
    return {
        "schema": "wrf_gpu2.v025.m0.control_manifest.v1.TEMPLATE",
        "status": "TEMPLATE_NOT_FINAL",
        "note": (
            "CONTROL-v025 becomes final only when every field below is measured and "
            "the independent critic accepts. The immutable tag is created by the "
            "MANAGER, not by this worker (contract §5.1)."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "candidate_commit": "a1fa459cbcf193c240fa376a5b0e7e4b0e97741a",
            "src_gpuwrf_tree": "796aa24528876ef06c0cec198c1a0d16fbe94d10",
            "tree_identical_to_worker_head": _cmd(
                "git", "-C", str(REPO), "rev-parse", "HEAD:src/gpuwrf"
            )
            == "796aa24528876ef06c0cec198c1a0d16fbe94d10",
        },
        "entrypoint": pending(gpu),
        "effective_model_config": pending(gpu),
        "toolchain": {
            "python": sys.version.split()[0],
            "jax": None,
            "jaxlib": None,
            "cuda": None,
            "driver": None,
            "profiler": None,
            "package_lock": pending("emit with --pip-freeze at control-freeze time"),
        },
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5090",
            "clocks_power_policy": pending(gpu),
            "cpu_affinity": "12-rank CPU arm pinned to a quiet 12-core set; recorded per arm",
            "prepared_runtime_aot_state": pending(gpu),
        },
        "input_and_config_hashes": case_manifest()["source_inputs"],
        "cpu_executable_hash": sha256_file(WRF_EXE.resolve()),
        "timing_definitions": {
            "cold_compile": pending(gpu),
            "cached_load": pending(gpu),
            "warm_step": pending(gpu),
            "cpu_step_sum": "sum of domain-1 'Timing for main' lines over the actual advance",
            "cpu_wallclock": "end-to-end launcher wallclock over identical boundaries",
        },
        "artifacts": pending("populated by the proof manifest at Phase C"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "proofs/v025/m0")
    parser.add_argument("--pip-freeze", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, obj in (
        ("environment_manifest.json", environment_manifest()),
        ("case_manifest.json", case_manifest()),
        ("control_manifest.json.in", control_manifest_template()),
    ):
        path = args.out_dir / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        written.append({"path": str(path), "sha256": sha256_file(path)})

    if args.pip_freeze:
        out = _cmd(sys.executable, "-m", "pip", "freeze") or ""
        args.pip_freeze.write_text(out + "\n")
        written.append(
            {"path": str(args.pip_freeze), "sha256": sha256_file(args.pip_freeze)}
        )

    print(json.dumps({"written": written}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
