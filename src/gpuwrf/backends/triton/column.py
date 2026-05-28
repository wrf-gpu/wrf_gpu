"""Triton implementation of the M2 analytic thermo-column fixture."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KernelRun:
    arrays: dict[str, np.ndarray]
    wall_time_s: float
    kernel_launches: int
    host_device_transfer_bytes: int
    occupancy_pct: float
    registers_per_thread: int
    local_memory_bytes: int
    artifact_paths: list[str]
    profiler_limitation: str


@triton.jit
def _column_thermo_kernel(
    temperature_initial,
    qv_initial,
    pressure_initial,
    saturation_qv,
    temperature_next,
    qv_next,
    pressure_next,
    mse_delta,
    levels: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    k = tl.arange(0, block_size)
    mask = k < levels
    t0 = tl.load(temperature_initial + k, mask=mask, other=0.0).to(tl.float64)
    q0 = tl.load(qv_initial + k, mask=mask, other=0.0).to(tl.float64)
    p0 = tl.load(pressure_initial + k, mask=mask, other=0.0).to(tl.float64)
    sat = tl.load(saturation_qv + k, mask=mask, other=0.0).to(tl.float64)
    zero = t0 * 0.0

    excess = tl.maximum(q0 - sat, zero)
    deficit = tl.maximum((zero + 0.72) * sat - q0, zero)
    condensation = (zero + 0.32) * excess
    evaporation = tl.minimum((zero + 0.04) * deficit, (zero + 0.18) * q0)
    q1 = tl.maximum(q0 - condensation + evaporation, zero + 1.0e-8)

    cp_d = zero + 1004.0
    lv = zero + 2.5e6
    latent_mass = condensation - evaporation
    t1 = t0 + (lv / cp_d) * latent_mass

    tl.store(temperature_next + k, t1, mask=mask)
    tl.store(qv_next + k, q1, mask=mask)
    tl.store(pressure_next + k, p0, mask=mask)
    tl.store(mse_delta + k, cp_d * (t1 - t0) + lv * (q1 - q0), mask=mask)


def _cuda_array(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device="cuda")


def _launch(
    d_temperature_initial: torch.Tensor,
    d_qv_initial: torch.Tensor,
    d_pressure_initial: torch.Tensor,
    d_saturation_qv: torch.Tensor,
    d_temperature_next: torch.Tensor,
    d_qv_next: torch.Tensor,
    d_pressure_next: torch.Tensor,
    d_mse_delta: torch.Tensor,
    levels: int,
    block_size: int,
) -> None:
    _column_thermo_kernel[(1,)](
        d_temperature_initial,
        d_qv_initial,
        d_pressure_initial,
        d_saturation_qv,
        d_temperature_next,
        d_qv_next,
        d_pressure_next,
        d_mse_delta,
        levels,
        block_size,
        num_warps=2,
    )


def run_column(
    input_path: Path,
    output_path: Path,
    resource_metrics,
    warmups: int = 1,
    runs: int = 5,
) -> KernelRun:
    """Run one fp64 branch-heavy thermodynamic column update and write an NPZ candidate."""

    with np.load(input_path, allow_pickle=False) as loaded:
        temperature_initial = np.ascontiguousarray(loaded["temperature_initial"], dtype=np.float64)
        qv_initial = np.ascontiguousarray(loaded["qv_initial"], dtype=np.float64)
        pressure_initial = np.ascontiguousarray(loaded["pressure_initial"], dtype=np.float64)
        saturation_qv = np.ascontiguousarray(loaded["saturation_qv"], dtype=np.float64)

    levels = int(temperature_initial.shape[0])
    block_size = 64
    marker = time.time()
    d_temperature_initial = _cuda_array(temperature_initial)
    d_qv_initial = _cuda_array(qv_initial)
    d_pressure_initial = _cuda_array(pressure_initial)
    d_saturation_qv = _cuda_array(saturation_qv)
    d_temperature_next = torch.empty_like(d_temperature_initial)
    d_qv_next = torch.empty_like(d_qv_initial)
    d_pressure_next = torch.empty_like(d_pressure_initial)
    d_mse_delta = torch.empty_like(d_temperature_initial)
    transfer_bytes = int(
        temperature_initial.nbytes + qv_initial.nbytes + pressure_initial.nbytes + saturation_qv.nbytes
    )

    _launch(
        d_temperature_initial,
        d_qv_initial,
        d_pressure_initial,
        d_saturation_qv,
        d_temperature_next,
        d_qv_next,
        d_pressure_next,
        d_mse_delta,
        levels,
        block_size,
    )
    torch.cuda.synchronize()
    for _ in range(warmups):
        _launch(
            d_temperature_initial,
            d_qv_initial,
            d_pressure_initial,
            d_saturation_qv,
            d_temperature_next,
            d_qv_next,
            d_pressure_next,
            d_mse_delta,
            levels,
            block_size,
        )
        torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        _launch(
            d_temperature_initial,
            d_qv_initial,
            d_pressure_initial,
            d_saturation_qv,
            d_temperature_next,
            d_qv_next,
            d_pressure_next,
            d_mse_delta,
            levels,
            block_size,
        )
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start_ns) / 1.0e9)

    temperature_next = d_temperature_next.cpu().numpy().copy()
    qv_next = d_qv_next.cpu().numpy().copy()
    pressure_next = d_pressure_next.cpu().numpy().copy()
    mse_delta = d_mse_delta.cpu().numpy().copy()
    transfer_bytes += int(temperature_next.nbytes + qv_next.nbytes + pressure_next.nbytes + mse_delta.nbytes)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    regs, local, occupancy, artifact_paths, limitation = resource_metrics("column", marker, block_size)
    return KernelRun(
        arrays=arrays,
        wall_time_s=float(statistics.median(timings)),
        kernel_launches=1,
        host_device_transfer_bytes=transfer_bytes,
        occupancy_pct=float(occupancy),
        registers_per_thread=int(regs),
        local_memory_bytes=int(local),
        artifact_paths=artifact_paths,
        profiler_limitation=limitation,
    )
