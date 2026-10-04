"""GWDO GPU probe: oracle on device, adapter-level oracle, per-call timing, one nsys call.

Modes: ``oracle`` (kernel + legacy arms vs pristine columns; adapter vs the
pristine increment) ; ``time --arm legacy|native`` (median wall of a jitted
adapter call) ; ``capture --arm ...`` (one call between cudaProfilerStart/Stop).
The State stand-in is real CPU-WRF wrfout_d01 (THM+300, P+PB, PH+PHB, U, V,
QVAPOR) in the product carry dtype (fp64, force_fp64).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
CASE = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/'
            'alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
FRAMES = {'night': 'wrfout_d01_2026-07-26_00:00:00', 'day': 'wrfout_d01_2026-07-26_12:00:00'}


class FakeState:
    def __init__(self, **leaves):
        self.__dict__.update(leaves)

    def replace(self, **kw):
        return FakeState(**{**self.__dict__, **kw})


def load_frame(case):
    import netCDF4
    with netCDF4.Dataset(CASE / FRAMES[case]) as d:
        g = lambda n: np.asarray(d.variables[n][0], np.float32)
        arrays = dict(theta=g('THM') + np.float32(300.), qv=g('QVAPOR'), p=g('P') + g('PB'),
                      ph=g('PH') + g('PHB'), u=g('U'), v=g('V'))
        statics = {n: g(n) for n in ('VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4',
                                     'OL1', 'OL2', 'OL3', 'OL4', 'SINALPHA', 'COSALPHA')}
        metrics = dict(fnm=g('FNM'), fnp=g('FNP'))
        dx = float(d.DX)
    return arrays, statics, metrics, dx


def build(case, dtype):
    import jax.numpy as jnp
    from gpuwrf.coupling.physics_couplers import build_gwdo_statics_from_wrf_fields
    arrays, st, metrics, dx = load_frame(case)
    statics = build_gwdo_statics_from_wrf_fields(
        st['VAR'], st['CON'], st['OA1'], st['OA2'], st['OA3'], st['OA4'], st['OL1'],
        st['OL2'], st['OL3'], st['OL4'], dx_m=dx, sina=st['SINALPHA'], cosa=st['COSALPHA'])
    grid = SimpleNamespace(metrics=SimpleNamespace(**{k: jnp.asarray(v, jnp.float64)
                                                      for k, v in metrics.items()}))
    arrays = {k: jnp.asarray(v, dtype) for k, v in arrays.items()}
    return arrays, statics, grid


def adapter_fn(arm, statics, grid, dt=54.0):
    import jax
    from gpuwrf.coupling.physics_couplers import gwdo_adapter
    os.environ['GPUWRF_GWDO_NATIVE_REAL'] = '1' if arm == 'native' else '0'

    @jax.jit
    def run(a):
        out = gwdo_adapter(FakeState(**a), dt, statics, grid)
        return out.u, out.v
    return run


def expected_increment(case, arrays, dt=54.0):
    """u/v face increments implied by the pristine WRF tendencies (add_a2c_u/v)."""
    from gwdo_oracle_compare import load
    _, _, wrf, _ = load(Path(os.environ['GWDO_ORACLE']) / case / 'columns.npz')
    nz, ny, nx = arrays['theta'].shape
    du = (wrf['rublten'].astype(np.float64) * dt).reshape(nz, ny, nx)
    dv = (wrf['rvblten'].astype(np.float64) * dt).reshape(nz, ny, nx)
    eu = np.zeros((nz, ny, nx + 1))
    ev = np.zeros((nz, ny + 1, nx))
    eu[:, :, 1:-1] = 0.5 * (du[:, :, :-1] + du[:, :, 1:])
    ev[:, 1:-1, :] = 0.5 * (dv[:, :-1, :] + dv[:, 1:, :])
    return eu, ev


def mode_oracle(args):
    import jax
    import jax.numpy as jnp
    from gwdo_oracle_compare import load, run_legacy, run_native, score
    res = dict(platform=jax.devices()[0].platform, device=str(jax.devices()[0]), cases={})
    assert res['platform'] == 'gpu' or args.allow_cpu, res
    for case in ('night', 'day'):
        cols, stat, wrf, dt = load(Path(os.environ['GWDO_ORACLE']) / case / 'columns.npz')
        rec = dict(native=score(run_native(cols, stat, dt), wrf),
                   legacy32=score(run_legacy(cols, stat, dt, np.float32), wrf),
                   legacy64=score(run_legacy(cols, stat, dt, np.float64), wrf))
        arrays, statics, grid = build(case, jnp.float64)
        eu, ev = expected_increment(case, arrays)
        for arm in ('legacy', 'native'):
            u, v = adapter_fn(arm, statics, grid)(arrays)
            du = np.asarray(u, np.float64) - np.asarray(arrays['u'], np.float64)
            dv = np.asarray(v, np.float64) - np.asarray(arrays['v'], np.float64)
            rec[f'adapter_{arm}'] = dict(
                u_max_err=float(np.abs(du - eu).max()), u_scale=float(np.abs(eu).max()),
                v_max_err=float(np.abs(dv - ev).max()), v_scale=float(np.abs(ev).max()),
                u_rms_err=float(np.sqrt(((du - eu) ** 2).mean())),
                v_rms_err=float(np.sqrt(((dv - ev) ** 2).mean())),
                nonfinite=int((~np.isfinite(du)).sum() + (~np.isfinite(dv)).sum()))
        res['cases'][case] = rec
    return res


def mode_time(args):
    import jax
    import jax.numpy as jnp
    assert jax.devices()[0].platform == 'gpu'
    arrays, statics, grid = build('night', jnp.float64)
    fn = adapter_fn(args.arm, statics, grid)
    t0 = time.perf_counter()
    jax.block_until_ready(fn(arrays))
    compile_s = time.perf_counter() - t0
    walls = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arrays))
        walls.append(time.perf_counter() - t0)
    hlo = fn.lower(arrays).compile().as_text()
    return dict(platform='gpu', arm=args.arm, n=args.n, first_call_s=compile_s,
                median_ms=1e3 * statistics.median(walls), min_ms=1e3 * min(walls),
                f64_in_optimized_hlo=hlo.count('f64['), custom_calls=hlo.count('custom-call('))


def mode_capture(args):
    import jax
    import jax.numpy as jnp
    assert jax.devices()[0].platform == 'gpu'
    arrays, statics, grid = build('night', jnp.float64)
    fn = adapter_fn(args.arm, statics, grid)
    for _ in range(3):
        jax.block_until_ready(fn(arrays))
    rt = ctypes.CDLL('libcudart.so')
    rt.cudaProfilerStart()
    jax.block_until_ready(fn(arrays))
    rt.cudaProfilerStop()
    return dict(platform='gpu', arm=args.arm, captured_calls=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('oracle', 'time', 'capture'))
    p.add_argument('--arm', default='native')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    res = dict(oracle=mode_oracle, time=mode_time, capture=mode_capture)[args.mode](args)
    args.out.write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
