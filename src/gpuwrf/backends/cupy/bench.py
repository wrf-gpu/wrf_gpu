"""CLI for the M2 CuPy raw-CUDA bakeoff candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cupy as cp

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


def _artifact_paths(problem: str, scratch: Path, profiler: Path) -> list[str]:
    paths = [
        scratch / f"{problem}_run.json",
        profiler / f"{problem}_ncu_stdout.txt",
        profiler / f"{problem}_ncu_stderr.txt",
        profiler / f"{problem}_ncu_exit.txt",
        profiler / "deliberate_kernel_bug.txt",
    ]
    report = profiler / f"{problem}.ncu-rep"
    if report.exists():
        paths.append(report)
    return [_rel(path) for path in paths if path.exists()]


def _profiler_limitation(problem: str, profiler: Path) -> str:
    exit_path = profiler / f"{problem}_ncu_exit.txt"
    if not exit_path.exists():
        return (
            "ncu was not invoked for this bench run; fallback-derived wall time/transfers/bandwidth "
            "come from CuPy host timing and transfer accounting."
        )
    exit_text = exit_path.read_text(errors="replace").strip()
    if exit_text == "0" and (profiler / f"{problem}.ncu-rep").exists():
        return (
            "ncu report exists, but profile JSON keeps fallback-derived wall time, transfer bytes, "
            "and bandwidth for parity with the cuda_tile fallback schema."
        )
    stderr = (profiler / f"{problem}_ncu_stderr.txt").read_text(errors="replace") if (profiler / f"{problem}_ncu_stderr.txt").exists() else ""
    stdout = (profiler / f"{problem}_ncu_stdout.txt").read_text(errors="replace") if (profiler / f"{problem}_ncu_stdout.txt").exists() else ""
    text = "\n".join((stderr, stdout))
    if "ERR_NVGPUCTRPERM" in text:
        return (
            "ncu invoked, but local user lacks NVIDIA performance-counter permission "
            "(ERR_NVGPUCTRPERM); RawKernel attributes provide registers/local memory, "
            "occupancy uses CuPy/CUDA occupancy API fallback, and bandwidth is fallback-derived."
        )
    return f"ncu invoked but exited {exit_text}; see artifact_paths logs. Bandwidth is fallback-derived."


def _profile(problem: str, case: str, run: StencilRun | ColumnRun, scratch: Path, profiler: Path) -> dict[str, Any]:
    wall = float(run.wall_time_s)
    transfer_bytes = int(run.host_device_transfer_bytes)
    return {
        "achieved_bandwidth_gbps": (transfer_bytes / wall / 1.0e9) if wall > 0.0 else 0.0,
        "achieved_bandwidth_method": "fallback-derived",
        "artifact_paths": _artifact_paths(problem, scratch, profiler),
        "backend": "cupy",
        "benchmark": f"m2_{problem}",
        "case": case,
        "hardware": "RTX 5090 32GB",
        "host_device_transfer_bytes": transfer_bytes,
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "profiler_limitation": _profiler_limitation(problem, profiler),
        "registers_per_thread": int(run.registers_per_thread),
        "wall_time_s": wall,
    }


def _run_json(problem: str, run: StencilRun | ColumnRun) -> dict[str, Any]:
    return {
        "cupy_version": cp.__version__,
        "host_device_transfer_bytes": int(run.host_device_transfer_bytes),
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "problem": problem,
        "registers_per_thread": int(run.registers_per_thread),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "wall_time_s": float(run.wall_time_s),
    }


def _capture_deliberate_bug(profiler: Path) -> None:
    profiler.mkdir(parents=True, exist_ok=True)
    out = profiler / "deliberate_kernel_bug.txt"
    try:
        cp.RawKernel('extern "C" __global__ void broken(double* x) { x[0] = ; }', "broken").compile()
    except Exception as exc:  # CuPy/NVRTC raises CompileException with the compiler log.
        out.write_text(str(exc)[:6000] + "\n", encoding="utf-8")
    else:
        out.write_text("unexpected: deliberate invalid RawKernel compiled successfully\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M2 CuPy raw-CUDA stencil/column candidate.")
    parser.add_argument("--problem", choices=["both", "stencil", "column"], default="both")
    parser.add_argument("--stencil-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-stencil-3d-advdiff-v1.npz")
    parser.add_argument("--column-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-column-thermo-v1.npz")
    parser.add_argument("--scratch", type=Path, default=ROOT / "data/scratch/m2-cupy")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/m2/cupy_or_numba")
    parser.add_argument("--profiler-dir", type=Path, default=ROOT / "data/profiler_artifacts/cupy_or_numba")
    parser.add_argument("--skip-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.profiler_dir.mkdir(parents=True, exist_ok=True)
    _capture_deliberate_bug(args.profiler_dir)

    try:
        if args.problem in {"both", "stencil"}:
            stencil_run = run_stencil(args.stencil_fixture, args.scratch / "stencil_out.npz")
            _json_write(args.scratch / "stencil_run.json", _run_json("stencil", stencil_run))
            if not args.skip_artifacts:
                _json_write(
                    args.artifact_dir / "stencil_profile.json",
                    _profile("stencil", STENCIL_CASE, stencil_run, args.scratch, args.profiler_dir),
                )

        if args.problem in {"both", "column"}:
            column_run = run_column(args.column_fixture, args.scratch / "column_out.npz")
            _json_write(args.scratch / "column_run.json", _run_json("column", column_run))
            if not args.skip_artifacts:
                _json_write(
                    args.artifact_dir / "column_profile.json",
                    _profile("column", COLUMN_CASE, column_run, args.scratch, args.profiler_dir),
                )
    except Exception as exc:
        print(f"m2 cupy bench: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
