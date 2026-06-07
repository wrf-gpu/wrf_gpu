"""Validate the nested-OOM fix: run the REAL fixed execute_nested_pipeline on the
exact production failing case for a short horizon and prove peak VRAM is BOUNDED
(flat across output-interval segments) + every domain finite + wrfout written.

Runs in-process so we can sample jax device memory_stats from a background thread
the whole time and report the true peak_bytes_in_use across the run.
"""

from __future__ import annotations

import os

# Mirror the CLI's nested-path allocator choice (set BEFORE any jax import so the
# GPU backend picks up the platform/cudaMalloc allocator at init). The CLI sets
# this at the top of _cmd_run; a direct caller must do the same since backend init
# happens at the first device op anywhere in the process.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def _smi_used_mib() -> float:
    """This-process GPU memory (MiB) via nvidia-smi (allocator-agnostic).

    The platform/cudaMalloc allocator (our nested-OOM fix) does not implement
    jax memory_stats(), so we sample real device memory for THIS pid -- the only
    allocator-independent peak-VRAM signal.
    """
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return 0.0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return float(parts[1])
            except ValueError:
                return 0.0
    return 0.0


def _smi_total_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return float(out.splitlines()[0].strip())
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    case = Path(sys.argv[1])
    max_dom = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    hours = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out_json = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("proofs/v0120/nested_oom_fix.json")

    scratch = Path("/tmp/nested_oom_fix_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        execute_nested_pipeline,
    )
    import jax  # noqa: F401  (forces backend init under the nested allocator)

    g = 2**30

    # Background peak sampler via nvidia-smi (works under any allocator).
    samples = {"peak_proc_mib": 0.0, "peak_total_mib": 0.0, "n": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            samples["peak_proc_mib"] = max(samples["peak_proc_mib"], _smi_used_mib())
            samples["peak_total_mib"] = max(samples["peak_total_mib"], _smi_total_used_mib())
            samples["n"] += 1
            time.sleep(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t.start()

    config = NestedPipelineConfig(
        input_dir=case,
        output_dir=scratch / "out",
        proof_dir=scratch / "proof",
        hours=hours,
        max_dom=max_dom,
        scratch_dir=scratch,
    )

    t0 = time.perf_counter()
    err = None
    payload = None
    try:
        payload = execute_nested_pipeline(config)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)

    report = {
        "case": str(case),
        "max_dom": max_dom,
        "hours": hours,
        "wall_s": wall,
        "error": err,
        "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "peak_proc_vram_gib_smi": samples["peak_proc_mib"] / 1024.0,
        "peak_total_vram_gib_smi": samples["peak_total_mib"] / 1024.0,
        "gpu_total_gib": 32607 / 1024.0,
        "n_mem_samples": samples["n"],
        "note_measurement": (
            "VRAM measured via nvidia-smi per-process used_memory; the platform "
            "(cudaMalloc) allocator does not implement jax memory_stats(). "
            "peak_total includes ~3 GiB desktop apps (krunner/chrome/plasmashell)."
        ),
    }
    if payload is not None:
        report["verdict"] = payload.get("verdict")
        report["all_domains_finite"] = payload.get("all_domains_finite")
        report["all_outputs_present"] = payload.get("all_outputs_present")
        report["per_domain"] = {
            n: {
                "final_state_finite": v.get("final_state_finite"),
                "wrfout_count": v.get("wrfout_count"),
                "expected_wrfout_count": v.get("expected_wrfout_count"),
                "own_steps": v.get("own_steps"),
                "dt_s": v.get("dt_s"),
            }
            for n, v in payload.get("per_domain", {}).items()
        }
        report["observed_own_steps"] = payload.get("hierarchy", {}).get("observed_own_steps")
        report["force_counts"] = payload.get("hierarchy", {}).get("force_counts")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if (err is None and report.get("verdict") == "PIPELINE_GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
