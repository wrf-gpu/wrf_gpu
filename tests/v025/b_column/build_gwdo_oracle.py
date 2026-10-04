"""Pristine WRF GWDO (bl_gwdo_run, REAL) oracle on real CPU-WRF PROD d01 columns.

Inputs are built from one CPU-WRF history frame with WRF's own phy_prep
formulas in float32 (module_big_step_utilities_em.F:4830-4935):
th_phy=(THM+t0)/(1+Rv/Rd*qv), p=P+PB, pi=(p/p1000mb)**rcp, t=th_phy*pi,
u/v mass averages, z_at_w=(PH+PHB)/g, z=0.5*(z_at_w(k)+z_at_w(k+1)),
p8w = fzm*p(k)+fzp*p(k-1) with the z-extrapolated bottom and log top.
Statics VAR/CON/OA1-4/OL1-4/SINALPHA/COSALPHA come from the same file,
dx2d = DX (module_physics_init.F:5683), deltim = dt (PROD bldt=0).
The pristine routine is compiled unmodified with WRF's own gfortran flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

F = np.float32
CASE = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/'
            'alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
STATIC = ('VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4', 'OL1', 'OL2', 'OL3', 'OL4')


def phy_prep_columns(path: Path):
    import netCDF4
    with netCDF4.Dataset(path) as d:
        g = lambda name: np.asarray(d.variables[name][0], dtype=F)
        thm, qv, p, pb = g('THM'), g('QVAPOR'), g('P'), g('PB')
        u, v, ph, phb = g('U'), g('V'), g('PH'), g('PHB')
        fnm, fnp = g('FNM'), g('FNP')
        statics = {name: g(name) for name in STATIC}
        sina, cosa = g('SINALPHA'), g('COSALPHA')
        dx = F(d.DX)
    grav, r_d, r_v, t0, p1000 = F(9.81), F(287.), F(461.6), F(300.), F(100000.)
    cp = F(7.) * r_d / F(2.)
    rcp = r_d / cp
    th = (thm + t0) / (F(1.) + r_v / r_d * qv)
    p_phy = p + pb
    pi_phy = np.power(p_phy / p1000, rcp)
    t_phy = th * pi_phy
    u_phy = F(.5) * (u[:, :, :-1] + u[:, :, 1:])
    v_phy = F(.5) * (v[:, :-1, :] + v[:, 1:, :])
    z_at_w = (phb + ph) / grav
    z = F(.5) * (z_at_w[:-1] + z_at_w[1:])
    nz = p_phy.shape[0]
    p8w = np.empty((nz + 1,) + p_phy.shape[1:], F)
    for k in range(1, nz):
        p8w[k] = fnm[k] * p_phy[k] + fnp[k] * p_phy[k - 1]
    w1 = (z_at_w[0] - z[1]) / (z[0] - z[1])
    p8w[0] = w1 * p_phy[0] + (F(1.) - w1) * p_phy[1]
    w1 = (z_at_w[nz] - z[nz - 2]) / (z[nz - 1] - z[nz - 2])
    p8w[nz] = np.exp(w1 * np.log(p_phy[nz - 1]) + (F(1.) - w1) * np.log(p_phy[nz - 2]))
    flat = lambda a: np.ascontiguousarray(a.reshape(a.shape[0], -1), F)
    flat2 = lambda a: np.ascontiguousarray(a.reshape(-1), F)
    ncol = p_phy.shape[1] * p_phy.shape[2]
    cols = dict(uproj=flat(u_phy), vproj=flat(v_phy), t1=flat(t_phy), q1=flat(qv),
                prsl=flat(p_phy), prslk=flat(pi_phy), zl=flat(z), prsi=flat(p8w))
    stat = {k.lower(): flat2(a) for k, a in statics.items()}
    stat.update(sina=flat2(sina), cosa=flat2(cosa), dxmeter=np.full(ncol, dx, F))
    return cols, stat


COL_ORDER = ('uproj', 'vproj', 't1', 'q1', 'prsl', 'prslk', 'zl', 'prsi')
STAT_ORDER = ('var', 'con', 'oa1', 'oa2', 'oa3', 'oa4', 'ol1', 'ol2', 'ol3', 'ol4',
              'sina', 'cosa', 'dxmeter')


def build_driver(repo: Path, wrf: Path, out: Path, compiler: Path):
    src = wrf / 'phys'
    env = dict(os.environ, LD_LIBRARY_PATH=str(compiler.parent.parent / 'lib'))
    flags = ['-O2', '-ftree-vectorize', '-funroll-loops', '-w', '-ffree-form',
             '-ffree-line-length-none', '-cpp']  # configure.wrf FCOPTIM + FCBASEOPTS
    sources = [src / 'ccpp_kind_types.F', src / 'physics_mmm' / 'bl_gwdo.F90',
               repo / 'tests/v025/b_column/gwdo_driver.F90']
    for s in sources:
        subprocess.run([str(compiler), *flags, '-c', str(s)], cwd=out, env=env,
                       timeout=120, check=True)
    objs = [s.with_suffix('.o').name for s in sources]
    subprocess.run([str(compiler), *objs, '-o', 'gwdo_driver.exe'], cwd=out, env=env,
                   timeout=120, check=True)
    return env, flags, sources


def run_case(exe_dir: Path, env, cols, stat, dt, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    nk, ncol = cols['t1'].shape
    with open(out / 'gwdo_in.bin', 'wb') as f:
        np.asarray([ncol, nk], np.int32).tofile(f)
        np.asarray([dt], F).tofile(f)
        for name in COL_ORDER:
            cols[name].tofile(f)
        for name in STAT_ORDER:
            stat[name].tofile(f)
    subprocess.run([str(exe_dir / 'gwdo_driver.exe')], cwd=out, env=env, timeout=120,
                   check=True)
    raw = np.fromfile(out / 'gwdo_out.bin', F)
    n2 = nk * ncol
    res = {}
    for i, name in enumerate(('rublten', 'rvblten', 'dtaux3d', 'dtauy3d')):
        res[name] = raw[i * n2:(i + 1) * n2].reshape(nk, ncol)
    res['dusfcg'] = raw[4 * n2:4 * n2 + ncol]
    res['dvsfcg'] = raw[4 * n2 + ncol:4 * n2 + 2 * ncol]
    assert raw.size == 4 * n2 + 2 * ncol
    np.savez(out / 'columns.npz', dt=np.asarray(dt, F),
             **{f'in_{k}': v for k, v in cols.items()},
             **{f'st_{k}': v for k, v in stat.items()},
             **{f'wrf_{k}': v for k, v in res.items()})
    (out / 'gwdo_in.bin').unlink()
    (out / 'gwdo_out.bin').unlink()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--wrf', type=Path, default=Path('<USER_HOME>/src/wrf_pristine/WRF'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compiler', type=Path,
                   default=Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'))
    p.add_argument('--dt', type=float, default=54.0)
    args = p.parse_args()
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    args.out.mkdir(parents=True, exist_ok=True)
    env, flags, sources = build_driver(args.repo, args.wrf, args.out, args.compiler)
    frames = {'night': CASE / 'wrfout_d01_2026-07-26_00:00:00',
              'day': CASE / 'wrfout_d01_2026-07-26_12:00:00'}
    summary = {}
    for label, path in frames.items():
        cols, stat = phy_prep_columns(path)
        res = run_case(args.out, env, cols, stat, args.dt, args.out / label)
        active = np.any(res['rublten'] != 0, axis=0) | np.any(res['rvblten'] != 0, axis=0)
        summary[label] = dict(
            wrfout=str(path), wrfout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ncol=int(cols['t1'].shape[1]), nk=int(cols['t1'].shape[0]),
            columns_var_gt0=int(np.sum(stat['var'] > 0)), columns_with_drag=int(active.sum()),
            max_abs_rublten=float(np.abs(res['rublten']).max()),
            max_abs_rvblten=float(np.abs(res['rvblten']).max()))
    provenance = dict(compiler=str(args.compiler), flags=flags, real_bits=32, dt=args.dt,
                      sources={str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                      cases=summary)
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
