"""Run the direct-call pristine NUCOND dev oracle on batches of columns."""
import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, '<USER_HOME>/src/wrf_gpu2_wt/o1-nssl/tests/v034/nssl2mom')
import oracle_io as o
B = '<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev/fortran/b_%s'

def fmt(a, mode):
    a = np.asarray(a, np.float32 if mode == 'fp32' else np.float64)
    return ' '.join(repr(float(x)) for x in a.ravel())

def run(mode, an, dn, t77, pn, w, dtp):
    """an (19, ncol, nz) Fortran-indexed; cols (ncol, nz). Returns an_out, t0, ssat."""
    ncol, nz = dn.shape
    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir='<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev', delete=False) as fh:
        fh.write(f'{ncol} {nz} {repr(float(dtp))}\n')
        # Fortran order: ix fastest, then kz, then il
        fh.write(fmt(np.transpose(an[1:], (0, 2, 1)), mode) + '\n')
        for a in (dn, t77, pn, w):
            fh.write(fmt(a.T, mode) + '\n')
        path = fh.name
    out = subprocess.run(['taskset', '-c', '25', B % mode + '/nucond_direct', path], cwd=B % mode,
                         capture_output=True, text=True, check=True).stdout.split('\n')
    os.unlink(path)
    vals = [np.array([float(x) for x in l.split()]) for l in out if l.strip()]
    an_o = np.zeros_like(an, dtype=np.float64)
    an_o[1:] = np.transpose(vals[0].reshape(18, nz, ncol), (0, 2, 1))
    return an_o, vals[1].reshape(nz, ncol).T, vals[2].reshape(nz, ncol).T

if __name__ == '__main__':
    # wiring check: direct NUCOND on oracle S3 must reproduce S4 exactly
    for mode in ('fp32', 'fp64'):
        R = np.float32 if mode == 'fp32' else np.float64
        bad = 0
        for c in o.CASES:
            sp = o.load_case(mode, c)
            an = o.stage_an(sp, 'S3')[:, None, :].astype(R)
            cols = [o.stage_col(sp, 'S0', k)[None, :].astype(R) for k in ('dn1', 't77', 'pn', 'wn')]
            a2, t0, ss = run(mode, an, *cols, sp['scalars']['DT'])
            ref = o.stage_an(sp, 'S4')
            d = np.max(np.abs(a2[:, 0] - ref)); d2 = np.max(np.abs(ss[0] - o.stage_col(sp, 'S4', 'ssat')))
            d3 = np.max(np.abs(t0[0] - o.stage_col(sp, 'S4', 't0')))
            if d or d2 or d3:
                bad += 1; print(mode, c, d, d2, d3)
        print(mode, 'wiring mismatches:', bad)
