"""CuPy RawKernel implementation of the M2 column fixture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from .stencil import _attribute, _kernel_attributes, _max_threads_per_sm


KERNEL_CODE = r"""
extern "C" {

__global__ void column_thermo_kernel(
    const double* __restrict__ temperature_initial,
    const double* __restrict__ qv_initial,
    const double* __restrict__ pressure_initial,
    const double* __restrict__ saturation_qv,
    double* __restrict__ temperature_next,
    double* __restrict__ qv_next,
    double* __restrict__ pressure_next,
    double* __restrict__ mse_delta,
    int levels) {
    extern __shared__ double shared[];
    double* sat_shared = shared;

    const int k = threadIdx.x;
    if (k < levels) {
        sat_shared[k] = saturation_qv[k];
    }
    __syncthreads();

    if (k >= levels) {
        return;
    }

    const double t0 = temperature_initial[k];
    const double q0 = qv_initial[k];
    const double p0 = pressure_initial[k];
    const double sat = sat_shared[k];
    const double excess = fmax(q0 - sat, 0.0);
    const double deficit = fmax(0.72 * sat - q0, 0.0);
    const double condensation = 0.32 * excess;
    const double evaporation = fmin(0.04 * deficit, 0.18 * q0);
    const double q1 = fmax(q0 - condensation + evaporation, 1.0e-8);

    const double cp_d = 1004.0;
    const double lv = 2.5e6;
    const double latent_mass = condensation - evaporation;
    const double t1 = t0 + (lv / cp_d) * latent_mass;

    temperature_next[k] = t1;
    qv_next[k] = q1;
    pressure_next[k] = p0;
    mse_delta[k] = cp_d * (t1 - t0) + lv * (q1 - q0);
}

}
"""


@dataclass(frozen=True)
class KernelRun:
    arrays: dict[str, np.ndarray]
    wall_time_s: float
    kernel_launches: int
    host_device_transfer_bytes: int
    occupancy_pct: float
    registers_per_thread: int
    local_memory_bytes: int


def _compile_kernel() -> cp.RawKernel:
    kernel = cp.RawKernel(KERNEL_CODE, "column_thermo_kernel", options=("--std=c++11",))
    kernel.compile()
    return kernel


def _occupancy_pct(kernel: cp.RawKernel, block_threads: int, dynamic_smem_bytes: int) -> float:
    try:
        function = getattr(kernel, "kernel", kernel)
        function_handle = getattr(function, "ptr", function)
        blocks_per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
            function_handle, block_threads, dynamic_smem_bytes
        )
        return 100.0 * float(blocks_per_sm * block_threads) / float(_max_threads_per_sm())
    except Exception:
        attrs = _kernel_attributes(kernel)
        max_threads = _attribute(attrs, "max_threads_per_block", "maxThreadsPerBlock", default=block_threads)
        return 100.0 * float(min(block_threads, max_threads)) / float(_max_threads_per_sm())


def run_column(input_path: Path, output_path: Path) -> KernelRun:
    """Run one fp64 branch-heavy thermodynamic column update and write an NPZ candidate."""

    with np.load(input_path, allow_pickle=False) as loaded:
        temperature_initial = np.ascontiguousarray(loaded["temperature_initial"], dtype=np.float64)
        qv_initial = np.ascontiguousarray(loaded["qv_initial"], dtype=np.float64)
        pressure_initial = np.ascontiguousarray(loaded["pressure_initial"], dtype=np.float64)
        saturation_qv = np.ascontiguousarray(loaded["saturation_qv"], dtype=np.float64)

    levels = int(temperature_initial.shape[0])
    kernel = _compile_kernel()
    block = (64, 1, 1)
    grid = (1, 1, 1)
    shared_mem = levels * np.dtype(np.float64).itemsize

    d_temperature_initial = cp.asarray(temperature_initial)
    d_qv_initial = cp.asarray(qv_initial)
    d_pressure_initial = cp.asarray(pressure_initial)
    d_saturation_qv = cp.asarray(saturation_qv)
    d_temperature_next = cp.empty_like(d_temperature_initial)
    d_qv_next = cp.empty_like(d_qv_initial)
    d_pressure_next = cp.empty_like(d_pressure_initial)
    d_mse_delta = cp.empty_like(d_temperature_initial)
    transfer_bytes = int(
        temperature_initial.nbytes + qv_initial.nbytes + pressure_initial.nbytes + saturation_qv.nbytes
    )

    cp.cuda.Stream.null.synchronize()
    start_ns = time.perf_counter_ns()
    kernel(
        grid,
        block,
        (
            d_temperature_initial,
            d_qv_initial,
            d_pressure_initial,
            d_saturation_qv,
            d_temperature_next,
            d_qv_next,
            d_pressure_next,
            d_mse_delta,
            np.int32(levels),
        ),
        shared_mem=shared_mem,
    )
    cp.cuda.Stream.null.synchronize()
    wall_time_s = (time.perf_counter_ns() - start_ns) / 1.0e9

    temperature_next = cp.asnumpy(d_temperature_next)
    qv_next = cp.asnumpy(d_qv_next)
    pressure_next = cp.asnumpy(d_pressure_next)
    mse_delta = cp.asnumpy(d_mse_delta)
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

    attrs = _kernel_attributes(kernel)
    return KernelRun(
        arrays=arrays,
        wall_time_s=wall_time_s,
        kernel_launches=1,
        host_device_transfer_bytes=transfer_bytes,
        occupancy_pct=_occupancy_pct(kernel, block[0] * block[1] * block[2], shared_mem),
        registers_per_thread=_attribute(attrs, "num_regs", "numRegs", "numregs"),
        local_memory_bytes=_attribute(attrs, "local_size_bytes", "localSizeBytes", "local"),
    )
