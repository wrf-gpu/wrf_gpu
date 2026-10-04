"""BP47 component probe: one MYNN surface-layer call on a real PROD domain view.

Arm = process environment (GPUWRF_SFCLAY_NATIVE_REAL=0/1 is read at import).
Warm WRF inputs (B39 State leaves) come from a CPU-WRF history frame like the
oracle gate. Seven NVTX ``BP47_sfclay`` ranges each enclose one synchronized
call (reduce with nvtx_range_device.py). Writes receipt.json.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import pickle
import site
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--wrfout", type=Path, required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="gpu")
    args = ap.parse_args()
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    import jax
    import jax.numpy as jnp
    import netCDF4
    platform = jax.devices()[0].platform
    assert platform == args.backend, platform
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.fp32.surface_layer_real import native_real_enabled
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    state, grid, *_ = pickle.load(args.inputs.open("rb"))[args.domain]
    nc = netCDF4.Dataset(args.wrfout)
    frame = lambda name: jnp.asarray(np.asarray(nc.variables[name][0], dtype=np.float32))  # noqa: E731
    warm = dict(hfx=frame("HFX"), qfx=frame("QFX"), pblh=frame("PBLH"),
                mol=jnp.zeros_like(frame("HFX")), qsfc=jnp.zeros_like(frame("HFX")))
    for name, value in warm.items():
        object.__setattr__(state, name, value)
    object.__setattr__(state, "ustar", frame("UST").astype(state.ustar.dtype))
    view = jax.device_put(_build_column_view(state, grid))
    fn = jax.jit(lambda v: surface_layer_with_diagnostics(v, first_timestep=False))
    start = time.perf_counter()
    compiled = fn.lower(view).compile()
    compile_s = time.perf_counter() - start
    result = jax.block_until_ready(compiled(view))
    samples = []
    nvtx = cudart = None
    if args.backend == "gpu":
        nvtx = ctypes.CDLL(str(next(Path(b) / "nvidia/nvtx/lib/libnvToolsExt.so.1" for b in site.getsitepackages()
                                    if (Path(b) / "nvidia/nvtx/lib/libnvToolsExt.so.1").is_file())))
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        assert cudart.cudaProfilerStart() == 0
    for _ in range(7):
        t0 = time.perf_counter()
        if nvtx is not None:
            nvtx.nvtxRangePushA(b"BP47_sfclay")
        try:
            result = jax.block_until_ready(compiled(view))
        finally:
            if nvtx is not None:
                nvtx.nvtxRangePop()
        samples.append(time.perf_counter() - t0)
    if cudart is not None:
        assert cudart.cudaProfilerStop() == 0
    leaves = jax.tree.leaves(result)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(np.asarray(leaf).tobytes())
    receipt = dict(platform=platform, device=str(jax.devices()[0]), native_real=native_real_enabled(),
                   domain=args.domain, shape=list(view.xland.shape), compile_s=compile_s, wall_s=samples,
                   median_s=float(np.median(samples)), finite=all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves),
                   out_dtypes=sorted({str(x.dtype) for x in leaves}), output_sha256=digest.hexdigest(),
                   flags={k: v for k, v in os.environ.items() if k.startswith(("GPUWRF_", "JAX_", "XLA_"))},
                   optimized_f64_lines=sum(1 for line in compiled.as_text().splitlines() if "= f64[" in line))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "flags"}))


if __name__ == "__main__":
    main()
