"""CuPy RawKernel implementation of the M2 stencil fixture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


KERNEL_CODE = r"""
extern "C" {

__device__ __forceinline__ int wrap_index(int value, int size) {
    value %= size;
    return value < 0 ? value + size : value;
}

__device__ __forceinline__ double diffusivity_for_level(int k) {
    const double values[8] = {
        18.0,
        19.414213562373095,
        20.0,
        19.414213562373095,
        18.0,
        16.585786437626904,
        16.0,
        16.585786437626904
    };
    return values[k & 7];
}

__global__ void stencil_advdiff_kernel(
    const double* __restrict__ phi_initial,
    const float* __restrict__ u_face,
    const float* __restrict__ v_face,
    const float* __restrict__ w_face,
    double* __restrict__ phi_next,
    int nx,
    int ny,
    int nz,
    double dx,
    double dy,
    double dz,
    double dt) {
    extern __shared__ double tile[];

    const int kHalo = 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int k = blockIdx.z;
    const int tile_w = blockDim.x + 2 * kHalo;
    const int tile_h = blockDim.y + 2 * kHalo;

    for (int local_y = ty; local_y < tile_h; local_y += blockDim.y) {
        const int y = wrap_index(by + local_y - kHalo, ny);
        for (int local_x = tx; local_x < tile_w; local_x += blockDim.x) {
            const int x = wrap_index(bx + local_x - kHalo, nx);
            tile[local_y * tile_w + local_x] = phi_initial[(k * ny + y) * nx + x];
        }
    }
    __syncthreads();

    const int x = bx + tx;
    const int y = by + ty;
    if (x >= nx || y >= ny || k >= nz) {
        return;
    }

    const int lx = tx + kHalo;
    const int ly = ty + kHalo;
    const int idx = (k * ny + y) * nx + x;
    const int kp = wrap_index(k + 1, nz);
    const int km = wrap_index(k - 1, nz);
    const double center = tile[ly * tile_w + lx];
    const double phi_xp1 = tile[ly * tile_w + lx + 1];
    const double phi_xp2 = tile[ly * tile_w + lx + 2];
    const double phi_xm1 = tile[ly * tile_w + lx - 1];
    const double phi_xm2 = tile[ly * tile_w + lx - 2];
    const double phi_yp1 = tile[(ly + 1) * tile_w + lx];
    const double phi_yp2 = tile[(ly + 2) * tile_w + lx];
    const double phi_ym1 = tile[(ly - 1) * tile_w + lx];
    const double phi_ym2 = tile[(ly - 2) * tile_w + lx];
    const double phi_zp1 = phi_initial[(kp * ny + y) * nx + x];
    const double phi_zm1 = phi_initial[(km * ny + y) * nx + x];

    const double ddx4 = (-phi_xp2 + 8.0 * phi_xp1 - 8.0 * phi_xm1 + phi_xm2) / (12.0 * dx);
    const double ddy4 = (-phi_yp2 + 8.0 * phi_yp1 - 8.0 * phi_ym1 + phi_ym2) / (12.0 * dy);
    const double ddz2 = (phi_zp1 - phi_zm1) / (2.0 * dz);
    const double lapx4 = (-phi_xp2 + 16.0 * phi_xp1 - 30.0 * center + 16.0 * phi_xm1 - phi_xm2) / (12.0 * dx * dx);
    const double lapy4 = (-phi_yp2 + 16.0 * phi_yp1 - 30.0 * center + 16.0 * phi_ym1 - phi_ym2) / (12.0 * dy * dy);
    const double lapz2 = (phi_zp1 - 2.0 * center + phi_zm1) / (dz * dz);

    const double u_mass = 0.5 * ((double)u_face[(k * ny + y) * (nx + 1) + x] +
                                 (double)u_face[(k * ny + y) * (nx + 1) + x + 1]);
    const double v_mass = 0.5 * ((double)v_face[(k * (ny + 1) + y) * nx + x] +
                                 (double)v_face[(k * (ny + 1) + y + 1) * nx + x]);
    const double w_mass = 0.5 * ((double)w_face[(k * ny + y) * nx + x] +
                                 (double)w_face[((k + 1) * ny + y) * nx + x]);
    const double diffusivity = diffusivity_for_level(k);
    const double advection = u_mass * ddx4 + v_mass * ddy4 + w_mass * ddz2;
    const double diffusion = diffusivity * (lapx4 + lapy4 + lapz2);
    phi_next[idx] = center + dt * (-advection + diffusion);
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


def _kernel_attributes(kernel: cp.RawKernel) -> dict[str, Any]:
    attrs = getattr(kernel, "attributes", None)
    if attrs is None:
        return {}
    return dict(attrs)


def _attribute(attrs: dict[str, Any], *names: str, default: int = 0) -> int:
    lowered = {str(key).lower(): value for key, value in attrs.items()}
    for name in names:
        if name in attrs:
            return int(attrs[name])
        if name.lower() in lowered:
            return int(lowered[name.lower()])
    return default


def _max_threads_per_sm() -> int:
    try:
        attr = cp.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor
        return int(cp.cuda.runtime.deviceGetAttribute(attr, cp.cuda.runtime.getDevice()))
    except Exception:
        return int(cp.cuda.Device().attributes.get("MaxThreadsPerMultiprocessor", 1536))


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


def _compile_kernel() -> cp.RawKernel:
    kernel = cp.RawKernel(KERNEL_CODE, "stencil_advdiff_kernel", options=("--std=c++11",))
    kernel.compile()
    return kernel


def run_stencil(input_path: Path, output_path: Path) -> KernelRun:
    """Run one fp64 periodic advection-diffusion update and write an NPZ candidate."""

    with np.load(input_path, allow_pickle=False) as loaded:
        phi_initial = np.ascontiguousarray(loaded["phi_initial"], dtype=np.float64)
        u_face = np.ascontiguousarray(loaded["u_face"], dtype=np.float32)
        v_face = np.ascontiguousarray(loaded["v_face"], dtype=np.float32)
        w_face = np.ascontiguousarray(loaded["w_face"], dtype=np.float32)

    nz, ny, nx = phi_initial.shape
    kernel = _compile_kernel()
    block = (16, 8, 1)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1], nz)
    shared_mem = (block[0] + 4) * (block[1] + 4) * np.dtype(np.float64).itemsize

    d_phi_initial = cp.asarray(phi_initial)
    d_u_face = cp.asarray(u_face)
    d_v_face = cp.asarray(v_face)
    d_w_face = cp.asarray(w_face)
    d_phi_next = cp.empty_like(d_phi_initial)
    transfer_bytes = int(phi_initial.nbytes + u_face.nbytes + v_face.nbytes + w_face.nbytes)

    cp.cuda.Stream.null.synchronize()
    start_ns = time.perf_counter_ns()
    kernel(
        grid,
        block,
        (
            d_phi_initial,
            d_u_face,
            d_v_face,
            d_w_face,
            d_phi_next,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float64(900.0),
            np.float64(900.0),
            np.float64(120.0),
            np.float64(3.0),
        ),
        shared_mem=shared_mem,
    )
    cp.cuda.Stream.null.synchronize()
    wall_time_s = (time.perf_counter_ns() - start_ns) / 1.0e9

    phi_next = cp.asnumpy(d_phi_next)
    transfer_bytes += int(phi_next.nbytes)
    arrays = {
        "phi_initial": phi_initial,
        "phi_next": phi_next,
        "u_face": u_face,
        "v_face": v_face,
        "w_face": w_face,
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
