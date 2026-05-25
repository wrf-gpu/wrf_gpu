#!/usr/bin/env python
"""cuSPARSE reference benchmark for the M6 perf solver bakeoff."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
from numba import cuda
import numpy as np

from gpuwrf.profiling.transfer_audit import visible_gpu_name


SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design-acceptance"
ARTIFACTS = SPRINT / "artifacts"
PREVIOUS = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design" / "artifacts" / "proof_solver_bakeoff.json"
PROOF = SPRINT / "proof_solver_bakeoff_v2.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(batch: int = 10500, n: int = 44, seed: int = 260526) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dl = -0.02 * rng.random((batch, n), dtype=np.float64)
    du = -0.02 * rng.random((batch, n), dtype=np.float64)
    dl[:, 0] = 0.0
    du[:, -1] = 0.0
    d = 1.0 + np.abs(dl) + np.abs(du) + 0.01 * rng.random((batch, n), dtype=np.float64)
    rhs = rng.normal(0.0, 1.0, (batch, n)).astype(np.float64)
    return dl, d, du, rhs


def _thomas_numpy(dl: np.ndarray, d: np.ndarray, du: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = dl.copy()
    b = d.copy()
    c = du.copy()
    x = rhs.copy()
    n = x.shape[1]
    for k in range(1, n):
        factor = a[:, k] / b[:, k - 1]
        b[:, k] = b[:, k] - factor * c[:, k - 1]
        x[:, k] = x[:, k] - factor * x[:, k - 1]
    x[:, -1] = x[:, -1] / b[:, -1]
    for k in range(n - 2, -1, -1):
        x[:, k] = (x[:, k] - c[:, k] * x[:, k + 1]) / b[:, k]
    return x


def _residual(dl: np.ndarray, d: np.ndarray, du: np.ndarray, x: np.ndarray, rhs: np.ndarray) -> float:
    y = d * x
    y[:, 1:] += dl[:, 1:] * x[:, :-1]
    y[:, :-1] += du[:, :-1] * x[:, 1:]
    return float(np.linalg.norm(y - rhs) / max(np.linalg.norm(rhs), 1.0e-30))


def _ptr(array: cuda.cudadrv.devicearray.DeviceNDArray) -> int:
    return int(array.device_ctypes_pointer.value)


def _check(status: int, where: str) -> None:
    if int(status) != 0:
        raise RuntimeError(f"{where} failed with cuSPARSE status {status}")


def _cusparse_gtsv2(dl: np.ndarray, d: np.ndarray, du: np.ndarray, rhs: np.ndarray, *, samples: int = 5) -> dict[str, Any]:
    libname = ctypes.util.find_library("cusparse")
    if not libname:
        return {"status": "UNAVAILABLE", "reason": "libcusparse not found"}
    lib = ctypes.CDLL(libname)
    handle = ctypes.c_void_p()
    _check(lib.cusparseCreate(ctypes.byref(handle)), "cusparseCreate")
    try:
        batch, n = rhs.shape
        stride = n
        dl_dev = cuda.to_device(np.ascontiguousarray(dl))
        d_dev = cuda.to_device(np.ascontiguousarray(d))
        du_dev = cuda.to_device(np.ascontiguousarray(du))
        x_dev = cuda.to_device(np.ascontiguousarray(rhs))
        buffer_size = ctypes.c_size_t()
        _check(
            lib.cusparseDgtsv2StridedBatch_bufferSizeExt(
                handle,
                ctypes.c_int(n),
                ctypes.c_void_p(_ptr(dl_dev)),
                ctypes.c_void_p(_ptr(d_dev)),
                ctypes.c_void_p(_ptr(du_dev)),
                ctypes.c_void_p(_ptr(x_dev)),
                ctypes.c_int(batch),
                ctypes.c_int(stride),
                ctypes.byref(buffer_size),
            ),
            "cusparseDgtsv2StridedBatch_bufferSizeExt",
        )
        buffer = cuda.device_array((int(buffer_size.value),), dtype=np.uint8)

        def solve_once() -> None:
            _check(
                lib.cusparseDgtsv2StridedBatch(
                    handle,
                    ctypes.c_int(n),
                    ctypes.c_void_p(_ptr(dl_dev)),
                    ctypes.c_void_p(_ptr(d_dev)),
                    ctypes.c_void_p(_ptr(du_dev)),
                    ctypes.c_void_p(_ptr(x_dev)),
                    ctypes.c_int(batch),
                    ctypes.c_int(stride),
                    ctypes.c_void_p(_ptr(buffer)),
                ),
                "cusparseDgtsv2StridedBatch",
            )

        solve_once()
        cuda.synchronize()
        timings = []
        for _ in range(samples):
            dl_dev = cuda.to_device(np.ascontiguousarray(dl))
            d_dev = cuda.to_device(np.ascontiguousarray(d))
            du_dev = cuda.to_device(np.ascontiguousarray(du))
            x_dev = cuda.to_device(np.ascontiguousarray(rhs))
            start = cuda.event(timing=True)
            end = cuda.event(timing=True)
            start.record()
            solve_once()
            end.record()
            end.synchronize()
            timings.append(float(cuda.event_elapsed_time(start, end)))
        solution = x_dev.copy_to_host()
        return {
            "status": "measured",
            "wall_ms_per_call_median": float(np.median(timings)),
            "wall_ms_samples": timings,
            "residual_l2_relative": _residual(dl, d, du, solution, rhs),
            "buffer_bytes": int(buffer_size.value),
            "memory_traffic_bytes_lower_bound": int(4 * dl.nbytes + rhs.nbytes + buffer_size.value),
        }
    finally:
        lib.cusparseDestroy(handle)


def run_bakeoff(samples: int = 5) -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dl, d, du, rhs = _fixture()
    start = time.perf_counter()
    thomas = _thomas_numpy(dl, d, du, rhs)
    thomas_wall = (time.perf_counter() - start) * 1000.0
    cusparse = _cusparse_gtsv2(dl, d, du, rhs, samples=samples)
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8")) if PREVIOUS.exists() else {}
    payload = {
        "artifact_type": "m6_perf_acceptance_solver_bakeoff_v2",
        "status": "PASS" if cusparse.get("status") == "measured" else "FAIL",
        "device": visible_gpu_name(),
        "shape": {"columns": int(rhs.shape[0]), "n": int(rhs.shape[1])},
        "references": {
            "numpy_thomas_reference": {
                "status": "measured",
                "wall_ms_per_call": float(thomas_wall),
                "residual_l2_relative": _residual(dl, d, du, thomas, rhs),
            },
            "cusparse_gtsv2_strided_batch": cusparse,
            "cusolverdx": {
                "status": "UNAVAILABLE",
                "reason": "cuSolverDx Python/device integration is not present in this repo; cuSPARSE gtsv2 is the external vendor reference for this sprint",
            },
        },
        "previous_m6_perf_design": {
            "path": str(PREVIOUS),
            "algorithms": previous.get("algorithms", {}),
            "winner": previous.get("winner"),
        },
        "decision": "Thomas remains operational default; PCR remains M7 candidate; cuSPARSE is reference-only, not a deployment dependency.",
    }
    _write_json(PROOF, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    del argv
    payload = run_bakeoff()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
