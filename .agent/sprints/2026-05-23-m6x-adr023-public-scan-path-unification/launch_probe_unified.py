from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

from gpuwrf.dynamics.acoustic_wrf import MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE, vertical_acoustic_update
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.mpas_oracles import mpas_column_slice
from tests.test_m6x_mpas_column_slice_oracle import (
    COLUMN_HEIGHT_M,
    DT_ACOUSTIC_S,
    EPS_SM,
    N_LEVELS,
    N_SUBSTEPS,
    _column_grid,
    _state_and_base_from_slice_initial,
)


def main() -> None:
    slice_result = mpas_column_slice("warm_bubble_2km", N_LEVELS, COLUMN_HEIGHT_M, DT_ACOUSTIC_S, N_SUBSTEPS, EPS_SM)
    grid = _column_grid()
    metrics = flat_metrics_for_grid(grid)
    state, base = _state_and_base_from_slice_initial(slice_result, grid)
    fn = jax.jit(
        lambda state_arg, base_arg: vertical_acoustic_update(
            state_arg,
            base_arg,
            metrics,
            dt=DT_ACOUSTIC_S,
            epssm=EPS_SM,
            top_lid=True,
            pressure_scale=-1.0,
            buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
        )
    )
    out = fn(state, base)
    jax.block_until_ready(out.w)
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    cudart.cudaProfilerStart()
    out = fn(state, base)
    jax.block_until_ready(out.w)
    cudart.cudaProfilerStop()
    print("unified_launch_probe_complete")


if __name__ == "__main__":
    main()
