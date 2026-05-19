"""CLI for the M2 OpenAI Triton bakeoff candidate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from .column import KernelRun as ColumnRun
from .column import run_column
from .stencil import KernelRun as StencilRun
from .stencil import run_stencil


ROOT = Path(__file__).resolve().parents[4]
STENCIL_CASE = "analytic-stencil-3d-advdiff-v1"
COLUMN_CASE = "analytic-column-thermo-v1"


def _rel(path: Path) -> str:
    path = path.absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _triton_cache_dir() -> Path:
    env = os.environ.get("TRITON_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".triton" / "cache"


def _parse_resource_usage(text: str) -> tuple[int, int, int]:
    regs: list[int] = []
    locals_: list[int] = []
    max_threads: list[int] = []
    for line in text.splitlines():
        reg = re.search(r"\bREG:(\d+)", line)
        local = re.search(r"\bLOCAL:(\d+)", line)
        threads = re.search(r"\b(?:MAX_THREADS|MAXTHREADS):(\d+)", line)
        if reg:
            regs.append(int(reg.group(1)))
        if local:
            locals_.append(int(local.group(1)))
        if threads:
            max_threads.append(int(threads.group(1)))
    return (max(regs) if regs else 0, max(locals_) if locals_ else 0, max(max_threads) if max_threads else 0)


def _derive_occupancy(registers_per_thread: int, block_threads: int) -> float:
    if registers_per_thread <= 0:
        return 0.0
    max_threads_per_sm = 1536
    max_registers_per_sm = 65536
    max_blocks_per_sm = 32
    blocks_by_threads = max_threads_per_sm // max(block_threads, 1)
    blocks_by_registers = max_registers_per_sm // max(registers_per_thread * block_threads, 1)
    active_blocks = max(0, min(max_blocks_per_sm, blocks_by_threads, blocks_by_registers))
    return 100.0 * active_blocks * block_threads / max_threads_per_sm


def _recent_cubins(marker: float) -> list[Path]:
    cache = _triton_cache_dir()
    if not cache.exists():
        return []
    cubins = [path for path in cache.rglob("*.cubin") if path.stat().st_mtime >= marker - 0.25]
    if cubins:
        return sorted(cubins, key=lambda path: path.stat().st_mtime, reverse=True)
    return sorted(cache.rglob("*.cubin"), key=lambda path: path.stat().st_mtime, reverse=True)[:4]


def _resource_metrics_factory(profiler_dir: Path):
    def resource_metrics(problem: str, marker: float, block_threads: int) -> tuple[int, int, float, list[str], str]:
        profiler_dir.mkdir(parents=True, exist_ok=True)
        resource_path = profiler_dir / f"{problem}_cuobjdump_resource_usage.txt"
        artifact_paths: list[str] = []
        cubins = _recent_cubins(marker)
        if cubins and shutil.which("cuobjdump"):
            parts: list[str] = []
            copied: list[Path] = []
            for index, cubin in enumerate(cubins[:4]):
                copied_cubin = profiler_dir / f"{problem}_triton_{index}.cubin"
                shutil.copy2(cubin, copied_cubin)
                copied.append(copied_cubin)
                proc = subprocess.run(
                    ["cuobjdump", "--dump-resource-usage", str(copied_cubin)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                parts.append(f"### {_rel(copied_cubin)}\n{proc.stdout}")
            resource_path.write_text("\n".join(parts), encoding="utf-8")
            regs, local, parsed_threads = _parse_resource_usage(resource_path.read_text(errors="replace"))
            threads = parsed_threads or block_threads
            occupancy = _derive_occupancy(regs, threads)
            artifact_paths.extend(_rel(path) for path in copied)
            artifact_paths.append(_rel(resource_path))
            limitation = (
                "ncu is invoked by the runner but local performance-counter permission may be unavailable; "
                "registers/local memory come from cuobjdump over Triton cached cubins, occupancy is "
                f"fallback-derived from registers and block size {threads}, and bandwidth is fallback-derived."
            )
            return regs or 1, local, occupancy, artifact_paths, limitation

        resource_path.write_text(
            f"No Triton cubin files were found under {_triton_cache_dir()}, or cuobjdump is unavailable.\n",
            encoding="utf-8",
        )
        artifact_paths.append(_rel(resource_path))
        limitation = (
            "Triton did not leave cuobjdump-readable cubins for this run, or cuobjdump is unavailable; "
            "registers, local memory, and occupancy are conservative fallback placeholders. "
            "Bandwidth is fallback-derived."
        )
        return 1, 0, 0.0, artifact_paths, limitation

    return resource_metrics


def _ncu_artifacts(problem: str, profiler_dir: Path) -> list[str]:
    paths = [
        profiler_dir / f"{problem}_ncu_stdout.txt",
        profiler_dir / f"{problem}_ncu_stderr.txt",
        profiler_dir / f"{problem}_ncu_exit.txt",
    ]
    report = profiler_dir / f"{problem}.ncu-rep"
    if report.exists():
        paths.append(report)
    return [_rel(path) for path in paths if path.exists()]


@triton.jit
def _broken_kernel(x, block_size: tl.constexpr) -> None:
    offsets = tl.arange(0, block_size)
    values = tl.load(x + offsets)
    bad = values + tl.arange(0, block_size + 1)
    tl.store(x + offsets, bad)


def _capture_deliberate_bug(profiler_dir: Path) -> str:
    profiler_dir.mkdir(parents=True, exist_ok=True)
    out = profiler_dir / "deliberate_triton_bug.txt"
    try:
        scratch = torch.zeros((8,), dtype=torch.float64, device="cuda")
        _broken_kernel[(1,)](scratch, block_size=8)
        torch.cuda.synchronize()
    except Exception as exc:  # Triton usually raises a source-location compile error.
        out.write_text(str(exc)[:6000] + "\n", encoding="utf-8")
    else:
        out.write_text("unexpected: deliberate invalid Triton program ran successfully\n", encoding="utf-8")
    return _rel(out)


def _profile(problem: str, case: str, run: StencilRun | ColumnRun) -> dict[str, Any]:
    wall = float(run.wall_time_s)
    transfer_bytes = int(run.host_device_transfer_bytes)
    return {
        "achieved_bandwidth_gbps": (transfer_bytes / wall / 1.0e9) if wall > 0.0 else 0.0,
        "achieved_bandwidth_method": "fallback-derived",
        "artifact_paths": run.artifact_paths,
        "backend": "triton",
        "benchmark": f"m2_{problem}",
        "case": case,
        "hardware": "RTX 5090 32GB",
        "host_device_transfer_bytes": transfer_bytes,
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "profiler_limitation": run.profiler_limitation,
        "registers_per_thread": int(run.registers_per_thread),
        "torch_cuda": torch.version.cuda,
        "torch_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "torch_version": torch.__version__,
        "triton_cache_dir": str(_triton_cache_dir()),
        "triton_version": triton.__version__,
        "wall_time_s": wall,
        "warmup_pattern": "one compile/first-run launch, one unmeasured post-compile warmup launch, then median of 5 torch.cuda.synchronize() kernel-only timings",
    }


def _run_json(problem: str, run: StencilRun | ColumnRun) -> dict[str, Any]:
    return {
        "host_device_transfer_bytes": int(run.host_device_transfer_bytes),
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "problem": problem,
        "registers_per_thread": int(run.registers_per_thread),
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "wall_time_s": float(run.wall_time_s),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M2 Triton stencil/column candidate.")
    parser.add_argument("--problem", choices=["both", "stencil", "column"], default="both")
    parser.add_argument("--stencil-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-stencil-3d-advdiff-v1.npz")
    parser.add_argument("--column-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-column-thermo-v1.npz")
    parser.add_argument("--scratch", type=Path, default=ROOT / "data/scratch/m2-triton")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/m2/triton")
    parser.add_argument("--profiler-dir", type=Path, default=ROOT / "data/profiler_artifacts/triton")
    parser.add_argument("--skip-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.profiler_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("m2 triton bench: expected torch CUDA availability", file=sys.stderr)
        return 1

    bug_path = _capture_deliberate_bug(args.profiler_dir)
    resource_metrics = _resource_metrics_factory(args.profiler_dir)
    backend_payload = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_cache_dir": str(_triton_cache_dir()),
        "triton_version": triton.__version__,
    }
    _json_write(args.scratch / "triton_backend.json", backend_payload)

    try:
        if args.problem in {"both", "stencil"}:
            stencil_run = run_stencil(
                args.stencil_fixture, args.scratch / "stencil_out.npz", resource_metrics=resource_metrics
            )
            stencil_run.artifact_paths.extend(_ncu_artifacts("stencil", args.profiler_dir))
            stencil_run.artifact_paths.append(bug_path)
            _json_write(args.scratch / "stencil_run.json", _run_json("stencil", stencil_run))
            if not args.skip_artifacts:
                _json_write(args.artifact_dir / "stencil_profile.json", _profile("stencil", STENCIL_CASE, stencil_run))

        if args.problem in {"both", "column"}:
            column_run = run_column(args.column_fixture, args.scratch / "column_out.npz", resource_metrics=resource_metrics)
            column_run.artifact_paths.extend(_ncu_artifacts("column", args.profiler_dir))
            column_run.artifact_paths.append(bug_path)
            _json_write(args.scratch / "column_run.json", _run_json("column", column_run))
            if not args.skip_artifacts:
                _json_write(args.artifact_dir / "column_profile.json", _profile("column", COLUMN_CASE, column_run))
    except Exception as exc:
        print(f"m2 triton bench: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
