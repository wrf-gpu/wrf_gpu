"""Triton implementation of the M2 analytic stencil fixture."""

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
def _stencil_advdiff_kernel(
    phi_initial,
    u_face,
    v_face,
    w_face,
    phi_next,
    n_elements: tl.constexpr,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = offsets % nx
    y = (offsets // nx) % ny
    k = offsets // (nx * ny)

    xp1 = (x + 1) % nx
    xp2 = (x + 2) % nx
    xm1 = (x + nx - 1) % nx
    xm2 = (x + nx - 2) % nx
    yp1 = (y + 1) % ny
    yp2 = (y + 2) % ny
    ym1 = (y + ny - 1) % ny
    ym2 = (y + ny - 2) % ny
    kp1 = (k + 1) % nz
    km1 = (k + nz - 1) % nz

    idx = (k * ny + y) * nx + x
    center = tl.load(phi_initial + idx, mask=mask, other=0.0).to(tl.float64)
    phi_xp1 = tl.load(phi_initial + (k * ny + y) * nx + xp1, mask=mask, other=0.0).to(tl.float64)
    phi_xp2 = tl.load(phi_initial + (k * ny + y) * nx + xp2, mask=mask, other=0.0).to(tl.float64)
    phi_xm1 = tl.load(phi_initial + (k * ny + y) * nx + xm1, mask=mask, other=0.0).to(tl.float64)
    phi_xm2 = tl.load(phi_initial + (k * ny + y) * nx + xm2, mask=mask, other=0.0).to(tl.float64)
    phi_yp1 = tl.load(phi_initial + (k * ny + yp1) * nx + x, mask=mask, other=0.0).to(tl.float64)
    phi_yp2 = tl.load(phi_initial + (k * ny + yp2) * nx + x, mask=mask, other=0.0).to(tl.float64)
    phi_ym1 = tl.load(phi_initial + (k * ny + ym1) * nx + x, mask=mask, other=0.0).to(tl.float64)
    phi_ym2 = tl.load(phi_initial + (k * ny + ym2) * nx + x, mask=mask, other=0.0).to(tl.float64)
    phi_zp1 = tl.load(phi_initial + (kp1 * ny + y) * nx + x, mask=mask, other=0.0).to(tl.float64)
    phi_zm1 = tl.load(phi_initial + (km1 * ny + y) * nx + x, mask=mask, other=0.0).to(tl.float64)

    u0 = tl.load(u_face + (k * ny + y) * (nx + 1) + x, mask=mask, other=0.0).to(tl.float64)
    u1 = tl.load(u_face + (k * ny + y) * (nx + 1) + x + 1, mask=mask, other=0.0).to(tl.float64)
    v0 = tl.load(v_face + (k * (ny + 1) + y) * nx + x, mask=mask, other=0.0).to(tl.float64)
    v1 = tl.load(v_face + (k * (ny + 1) + y + 1) * nx + x, mask=mask, other=0.0).to(tl.float64)
    w0 = tl.load(w_face + (k * ny + y) * nx + x, mask=mask, other=0.0).to(tl.float64)
    w1 = tl.load(w_face + ((k + 1) * ny + y) * nx + x, mask=mask, other=0.0).to(tl.float64)

    ddx4 = (-phi_xp2 + 8.0 * phi_xp1 - 8.0 * phi_xm1 + phi_xm2) / (12.0 * 900.0)
    ddy4 = (-phi_yp2 + 8.0 * phi_yp1 - 8.0 * phi_ym1 + phi_ym2) / (12.0 * 900.0)
    ddz2 = (phi_zp1 - phi_zm1) / (2.0 * 120.0)
    lapx4 = (-phi_xp2 + 16.0 * phi_xp1 - 30.0 * center + 16.0 * phi_xm1 - phi_xm2) / (12.0 * 900.0 * 900.0)
    lapy4 = (-phi_yp2 + 16.0 * phi_yp1 - 30.0 * center + 16.0 * phi_ym1 - phi_ym2) / (12.0 * 900.0 * 900.0)
    lapz2 = (phi_zp1 - 2.0 * center + phi_zm1) / (120.0 * 120.0)

    u_mass = 0.5 * (u0 + u1)
    v_mass = 0.5 * (v0 + v1)
    w_mass = 0.5 * (w0 + w1)
    k64 = k.to(tl.float64)
    diffusivity = 18.0 + 2.0 * tl.sin(2.0 * 3.141592653589793 * k64 / nz)
    advection = u_mass * ddx4 + v_mass * ddy4 + w_mass * ddz2
    diffusion = diffusivity * (lapx4 + lapy4 + lapz2)
    tl.store(phi_next + idx, center + 3.0 * (-advection + diffusion), mask=mask)


def _cuda_array(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device="cuda")


def _launch(
    d_phi_initial: torch.Tensor,
    d_u_face: torch.Tensor,
    d_v_face: torch.Tensor,
    d_w_face: torch.Tensor,
    d_phi_next: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    block_size: int,
) -> None:
    total = int(nx * ny * nz)
    grid = (triton.cdiv(total, block_size),)
    _stencil_advdiff_kernel[grid](
        d_phi_initial,
        d_u_face,
        d_v_face,
        d_w_face,
        d_phi_next,
        total,
        nx,
        ny,
        nz,
        block_size,
        num_warps=8,
    )


def run_stencil(
    input_path: Path,
    output_path: Path,
    resource_metrics,
    warmups: int = 1,
    runs: int = 5,
) -> KernelRun:
    """Run one fp64 periodic advection-diffusion update and write an NPZ candidate."""

    with np.load(input_path, allow_pickle=False) as loaded:
        phi_initial = np.ascontiguousarray(loaded["phi_initial"], dtype=np.float64)
        u_face = np.ascontiguousarray(loaded["u_face"], dtype=np.float32)
        v_face = np.ascontiguousarray(loaded["v_face"], dtype=np.float32)
        w_face = np.ascontiguousarray(loaded["w_face"], dtype=np.float32)

    nz, ny, nx = phi_initial.shape
    block_size = 256
    marker = time.time()
    d_phi_initial = _cuda_array(phi_initial)
    d_u_face = _cuda_array(u_face)
    d_v_face = _cuda_array(v_face)
    d_w_face = _cuda_array(w_face)
    d_phi_next = torch.empty_like(d_phi_initial)
    transfer_bytes = int(phi_initial.nbytes + u_face.nbytes + v_face.nbytes + w_face.nbytes)

    _launch(d_phi_initial, d_u_face, d_v_face, d_w_face, d_phi_next, nx, ny, nz, block_size)
    torch.cuda.synchronize()
    for _ in range(warmups):
        _launch(d_phi_initial, d_u_face, d_v_face, d_w_face, d_phi_next, nx, ny, nz, block_size)
        torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        _launch(d_phi_initial, d_u_face, d_v_face, d_w_face, d_phi_next, nx, ny, nz, block_size)
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start_ns) / 1.0e9)

    phi_next = d_phi_next.cpu().numpy().copy()
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

    regs, local, occupancy, artifact_paths, limitation = resource_metrics("stencil", marker, block_size)
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
