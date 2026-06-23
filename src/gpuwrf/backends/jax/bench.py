"""CLI for the M2 JAX bakeoff candidate."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from .column import column_thermo
from .stencil import stencil_advdiff


configure_jax_x64()

ROOT = Path(__file__).resolve().parents[4]
STENCIL_CASE = "analytic-stencil-3d-advdiff-v1"
COLUMN_CASE = "analytic-column-thermo-v1"


@dataclass(frozen=True)
class KernelRun:
    arrays: dict[str, np.ndarray]
    wall_time_s: float
    kernel_launches: int
    host_device_transfer_bytes: int
    occupancy_pct: float
    registers_per_thread: int
    local_memory_bytes: int
    hlo_ops: list[str]
    artifact_paths: list[str]
    profiler_limitation: str


def _rel(path: Path) -> str:
    path = path.absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _block_until_ready(value: Any) -> None:
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def _as_text(compiled: Any) -> str:
    for kwargs in ({"dialect": "hlo"}, {}):
        try:
            return str(compiled.as_text(**kwargs))
        except TypeError:
            continue
    return str(compiled)


def _entry_hlo(text: str) -> str:
    match = re.search(r"ENTRY[^{]*\{(?P<body>.*)\}\s*$", text, flags=re.DOTALL)
    return match.group("body") if match else text


def _hlo_kernel_ops(text: str) -> list[str]:
    body = _entry_hlo(text)
    ops: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("ROOT tuple", "tuple(")):
            continue
        if re.search(r"=\s*(fusion|custom-call|reduce|sort|while|convolution|all-reduce|collective-permute|copy)\(", stripped):
            ops.append(stripped[:220])
    if not ops:
        fusion_count = len(re.findall(r"\bfusion\(", body))
        custom_count = len(re.findall(r"\bcustom-call\(", body))
        reduce_count = len(re.findall(r"\breduce\(", body))
        synthetic = fusion_count + custom_count + reduce_count
        if synthetic:
            ops.append(f"fallback-count fusion/custom-call/reduce total={synthetic}")
    return ops


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_dump(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


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


def _ptx_reqntid(text: str) -> int:
    match = re.search(r"\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    if not match:
        return 0
    return int(match.group(1)) * int(match.group(2)) * int(match.group(3))


def _ptx_arch(text: str) -> str:
    match = re.search(r"^\s*\.target\s+(sm_\d+[a-z]?)", text, flags=re.MULTILINE)
    return match.group(1) if match else "sm_120a"


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


def _resource_metrics(problem: str, work_dump: Path, profiler_dir: Path) -> tuple[int, int, float, list[str], str]:
    preserved_dump = profiler_dir / f"{problem}_xla_dump"
    _copy_dump(work_dump, preserved_dump)
    resource_path = profiler_dir / f"{problem}_cuobjdump_resource_usage.txt"
    cubins = sorted(preserved_dump.rglob("*.cubin"))
    paths: list[str] = []
    if cubins and shutil.which("cuobjdump"):
        parts: list[str] = []
        for cubin in cubins:
            proc = subprocess.run(
                ["cuobjdump", "--dump-resource-usage", str(cubin)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            parts.append(f"### {_rel(cubin)}\n{proc.stdout}")
        resource_path.write_text("\n".join(parts), encoding="utf-8")
        paths.append(_rel(resource_path))
        regs, local, max_threads = _parse_resource_usage(resource_path.read_text(errors="replace"))
        block_threads = max_threads or 128
        occupancy = _derive_occupancy(regs, block_threads)
        limitation = (
            "ncu is invoked by the runner but local performance-counter permission is unavailable; "
            "registers/local memory come from cuobjdump over XLA dumped cubins, occupancy is fallback-derived from "
            f"registers and an assumed/parsed block size of {block_threads}, and bandwidth is fallback-derived."
        )
        return regs, local, occupancy, paths, limitation

    ptx_files = sorted(preserved_dump.rglob("*.ptx"))
    if ptx_files and shutil.which("ptxas") and shutil.which("cuobjdump"):
        parts = []
        block_threads = 0
        cubin_paths: list[Path] = []
        for ptx in ptx_files:
            text = ptx.read_text(errors="replace")
            block_threads = max(block_threads, _ptx_reqntid(text))
            cubin = ptx.with_suffix(".ptxas.cubin")
            proc = subprocess.run(
                ["ptxas", "-v", f"-arch={_ptx_arch(text)}", str(ptx), "-o", str(cubin)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            parts.append(f"### ptxas {_rel(ptx)}\n{proc.stdout}")
            if proc.returncode == 0 and cubin.exists():
                cubin_paths.append(cubin)
                dump = subprocess.run(
                    ["cuobjdump", "--dump-resource-usage", str(cubin)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                parts.append(f"### cuobjdump {_rel(cubin)}\n{dump.stdout}")
        resource_path.write_text("\n".join(parts), encoding="utf-8")
        paths.append(_rel(resource_path))
        paths.extend(_rel(path) for path in cubin_paths)
        regs, local, parsed_threads = _parse_resource_usage(resource_path.read_text(errors="replace"))
        block_threads = parsed_threads or block_threads or 128
        occupancy = _derive_occupancy(regs, block_threads)
        limitation = (
            "ncu is invoked by the runner but local performance-counter permission is unavailable; "
            "XLA dumped PTX rather than cubins, so ptxas compiled the dumped PTX and cuobjdump parsed registers/local "
            f"memory from that cubin. Occupancy is fallback-derived from registers and block size {block_threads}; "
            "bandwidth is fallback-derived."
        )
        return regs, local, occupancy, paths, limitation

    resource_path.write_text(
        "No XLA cubin/PTX files were found under the dump directory, or ptxas/cuobjdump is unavailable.\n",
        encoding="utf-8",
    )
    paths.append(_rel(resource_path))
    limitation = (
        "XLA did not leave ptxas/cuobjdump-readable files in the dump directory for this run; "
        "registers, local memory, and occupancy are conservative fallback placeholders. "
        "Bandwidth is fallback-derived."
    )
    return 1, 0, 0.0, paths, limitation


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


def _capture_deliberate_bug(profiler_dir: Path) -> str:
    out = profiler_dir / "deliberate_jax_bug.txt"
    try:
        @jax.jit
        def broken(x: jax.Array) -> jax.Array:
            return x + jnp.ones((3,), dtype=x.dtype)

        broken(jnp.ones((2,), dtype=jnp.float64)).block_until_ready()
    except Exception as exc:  # JAX raises a long shape/broadcast trace.
        out.write_text(str(exc)[:6000] + "\n", encoding="utf-8")
    else:
        out.write_text("unexpected: deliberate invalid JAX program ran successfully\n", encoding="utf-8")
    return _rel(out)


def _time_warm_runs(fn: Any, args: tuple[jax.Array, ...], warmups: int = 1, runs: int = 5) -> tuple[Any, float]:
    result = fn(*args)
    _block_until_ready(result)
    for _ in range(warmups):
        result = fn(*args)
        _block_until_ready(result)
    timings: list[float] = []
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        result = fn(*args)
        _block_until_ready(result)
        timings.append((time.perf_counter_ns() - start_ns) / 1.0e9)
    return result, statistics.median(timings)


def run_stencil(input_path: Path, output_path: Path, scratch: Path, profiler_dir: Path) -> KernelRun:
    with np.load(input_path, allow_pickle=False) as loaded:
        phi_initial = np.ascontiguousarray(loaded["phi_initial"], dtype=np.float64)
        u_face = np.ascontiguousarray(loaded["u_face"], dtype=np.float32)
        v_face = np.ascontiguousarray(loaded["v_face"], dtype=np.float32)
        w_face = np.ascontiguousarray(loaded["w_face"], dtype=np.float32)

    work_dump = profiler_dir / "xla_dump_work"
    _clean_dir(work_dump)
    d_args = (
        jnp.asarray(phi_initial),
        jnp.asarray(u_face),
        jnp.asarray(v_face),
        jnp.asarray(w_face),
    )
    compiled = stencil_advdiff.lower(*d_args).compile()
    hlo_text = _as_text(compiled)
    hlo_path = profiler_dir / "stencil_compiled_hlo.txt"
    hlo_path.write_text(hlo_text, encoding="utf-8")
    result, wall_time_s = _time_warm_runs(compiled, d_args)
    phi_next = np.asarray(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "phi_initial": phi_initial,
        "u_face": u_face,
        "v_face": v_face,
        "w_face": w_face,
        "phi_next": phi_next,
    }
    np.savez(output_path, **arrays)
    regs, local, occupancy, paths, limitation = _resource_metrics("stencil", work_dump, profiler_dir)
    paths.extend(_ncu_artifacts("stencil", profiler_dir))
    transfer_bytes = int(phi_initial.nbytes + u_face.nbytes + v_face.nbytes + w_face.nbytes + phi_next.nbytes)
    hlo_ops = _hlo_kernel_ops(hlo_text)
    run_json = scratch / "stencil_run.json"
    paths.extend([_rel(hlo_path), _rel(run_json)])
    return KernelRun(
        arrays=arrays,
        wall_time_s=wall_time_s,
        kernel_launches=max(1, len(hlo_ops)),
        host_device_transfer_bytes=transfer_bytes,
        occupancy_pct=float(occupancy),
        registers_per_thread=int(regs),
        local_memory_bytes=int(local),
        hlo_ops=hlo_ops,
        artifact_paths=paths,
        profiler_limitation=limitation,
    )


def run_column(input_path: Path, output_path: Path, scratch: Path, profiler_dir: Path) -> KernelRun:
    with np.load(input_path, allow_pickle=False) as loaded:
        temperature_initial = np.ascontiguousarray(loaded["temperature_initial"], dtype=np.float64)
        qv_initial = np.ascontiguousarray(loaded["qv_initial"], dtype=np.float64)
        pressure_initial = np.ascontiguousarray(loaded["pressure_initial"], dtype=np.float64)
        saturation_qv = np.ascontiguousarray(loaded["saturation_qv"], dtype=np.float64)

    work_dump = profiler_dir / "xla_dump_work"
    _clean_dir(work_dump)
    d_args = (
        jnp.asarray(temperature_initial),
        jnp.asarray(qv_initial),
        jnp.asarray(pressure_initial),
        jnp.asarray(saturation_qv),
    )
    compiled = column_thermo.lower(*d_args).compile()
    hlo_text = _as_text(compiled)
    hlo_path = profiler_dir / "column_compiled_hlo.txt"
    hlo_path.write_text(hlo_text, encoding="utf-8")
    result, wall_time_s = _time_warm_runs(compiled, d_args)
    temperature_next, qv_next, pressure_next, mse_delta = (np.asarray(item) for item in result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mse_delta": mse_delta,
        "pressure_initial": pressure_initial,
        "pressure_next": pressure_next,
        "qv_initial": qv_initial,
        "qv_next": qv_next,
        "saturation_qv": saturation_qv,
        "temperature_initial": temperature_initial,
        "temperature_next": temperature_next,
    }
    np.savez(output_path, **arrays)
    regs, local, occupancy, paths, limitation = _resource_metrics("column", work_dump, profiler_dir)
    paths.extend(_ncu_artifacts("column", profiler_dir))
    transfer_bytes = int(
        temperature_initial.nbytes
        + qv_initial.nbytes
        + pressure_initial.nbytes
        + saturation_qv.nbytes
        + temperature_next.nbytes
        + qv_next.nbytes
        + pressure_next.nbytes
        + mse_delta.nbytes
    )
    hlo_ops = _hlo_kernel_ops(hlo_text)
    run_json = scratch / "column_run.json"
    paths.extend([_rel(hlo_path), _rel(run_json)])
    return KernelRun(
        arrays=arrays,
        wall_time_s=wall_time_s,
        kernel_launches=max(1, len(hlo_ops)),
        host_device_transfer_bytes=transfer_bytes,
        occupancy_pct=float(occupancy),
        registers_per_thread=int(regs),
        local_memory_bytes=int(local),
        hlo_ops=hlo_ops,
        artifact_paths=paths,
        profiler_limitation=limitation,
    )


def _profile(problem: str, case: str, run: KernelRun) -> dict[str, Any]:
    wall = float(run.wall_time_s)
    transfer_bytes = int(run.host_device_transfer_bytes)
    return {
        "achieved_bandwidth_gbps": (transfer_bytes / wall / 1.0e9) if wall > 0.0 else 0.0,
        "achieved_bandwidth_method": "fallback-derived",
        "artifact_paths": run.artifact_paths,
        "backend": "jax",
        "benchmark": f"m2_{problem}",
        "case": case,
        "hardware": "RTX 5090 32GB",
        "hlo_kernel_ops": run.hlo_ops,
        "host_device_transfer_bytes": transfer_bytes,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "profiler_limitation": run.profiler_limitation,
        "registers_per_thread": int(run.registers_per_thread),
        "wall_time_s": wall,
        "warmup_pattern": "lower().compile() once, one unmeasured post-compile warmup call, then median of 5 block_until_ready() runs",
    }


def _run_json(problem: str, run: KernelRun) -> dict[str, Any]:
    return {
        "host_device_transfer_bytes": int(run.host_device_transfer_bytes),
        "kernel_launches": int(run.kernel_launches),
        "local_memory_bytes": int(run.local_memory_bytes),
        "occupancy_pct": float(run.occupancy_pct),
        "problem": problem,
        "registers_per_thread": int(run.registers_per_thread),
        "wall_time_s": float(run.wall_time_s),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M2 JAX stencil/column candidate.")
    parser.add_argument("--problem", choices=["both", "stencil", "column"], default="both")
    parser.add_argument("--stencil-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-stencil-3d-advdiff-v1.npz")
    parser.add_argument("--column-fixture", type=Path, default=ROOT / "fixtures/samples/analytic-column-thermo-v1.npz")
    parser.add_argument("--scratch", type=Path, default=ROOT / "data/scratch/m2-jax")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/m2/jax")
    parser.add_argument("--profiler-dir", type=Path, default=ROOT / "data/profiler_artifacts/jax")
    parser.add_argument("--skip-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.profiler_dir.mkdir(parents=True, exist_ok=True)
    bug_path = _capture_deliberate_bug(args.profiler_dir)

    if jax.default_backend() != "gpu":
        print(f"m2 jax bench: expected gpu backend, got {jax.default_backend()} with {jax.devices()}", file=sys.stderr)
        return 1

    backend_payload = {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    _json_write(args.scratch / "jax_backend.json", backend_payload)

    try:
        if args.problem in {"both", "stencil"}:
            stencil_run = run_stencil(args.stencil_fixture, args.scratch / "stencil_out.npz", args.scratch, args.profiler_dir)
            stencil_run.artifact_paths.append(bug_path)
            _json_write(args.scratch / "stencil_run.json", _run_json("stencil", stencil_run))
            if not args.skip_artifacts:
                _json_write(args.artifact_dir / "stencil_profile.json", _profile("stencil", STENCIL_CASE, stencil_run))

        if args.problem in {"both", "column"}:
            column_run = run_column(args.column_fixture, args.scratch / "column_out.npz", args.scratch, args.profiler_dir)
            column_run.artifact_paths.append(bug_path)
            _json_write(args.scratch / "column_run.json", _run_json("column", column_run))
            if not args.skip_artifacts:
                _json_write(args.artifact_dir / "column_profile.json", _profile("column", COLUMN_CASE, column_run))
    except Exception as exc:
        print(f"m2 jax bench: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
