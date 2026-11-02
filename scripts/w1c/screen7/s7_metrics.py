"""SCREEN7 decision metrics exported BEFORE any raw unlink (ALISIOS decision 2026-10-31T06:58Z: at least north LWP/CWP,
north cloud fraction at CWP >= 10 g/m2, T2/wind means). Definitions are ALISIOS's, not ours: the per-frame surfaces and the
area-weighted region statistics are taken verbatim from their model-only driver screen6_A_eval/analyse_compact.py
(36abc15f): LWP = cwp({QCLOUD}), CWP = wrf_columns(QCLOUD+QICE), IWP = CWP - LWP, cloud_column_fraction = (CWP >= 10),
wind_speed_ms = hypot(U10, V10), plus their surface fields; the operator functions are executed from their SHA-bound cloud.py
(close6/SCIENCE_OPERATOR_BINDING_PENDING.json) exactly as their driver loads them; masks/cell areas from their frozen
screen4_eval/MASKS_SEAL.json -> FIXED_MASK_ARRAYS.npz, geometry asserted per frame. Model output only, no truth read."""
import ast
import hashlib
import json
from pathlib import Path

R = Path('<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009/re-wave-optimizer-v1')
DRIVER = (R / 'screen6_A_eval/analyse_compact.py', '36abc15fbcde2d734ce1c7220007d21d07f9780a272f81047c4378fb24259224')
BINDING = (R / 'close6/SCIENCE_OPERATOR_BINDING_PENDING.json', 'bec74fe3d93f67b7adf055b5b5066b496c6470c26816f77a9b4b309dacf151ff')
MASKS = (R / 'screen4_eval/MASKS_SEAL.json', 'd1e99270ed234cbb53a3cc3aac3c342056c29a89ff1e8abb817baeb822700090')
FIELDS = ['SWDOWN', 'GLW', 'HFX', 'LH', 'QFX', 'GRDFLX', 'PBLH', 'T2', 'Q2', 'PSFC', 'U10', 'V10']   # driver `fields`
DERIVED = ['LWP', 'CWP', 'IWP', 'cloud_column_fraction', 'wind_speed_ms']
GEOMETRY = ['XLAT', 'XLONG', 'HGT', 'LANDMASK', 'MAPFAC_M']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _pinned(pair, what):
    p, want = pair
    if sha(p) != want:
        raise ValueError(f'{what} changed: {p}')
    return Path(p)


def operator():
    """cloud.py functions exactly as the ALISIOS driver loads them (AST subset, R_D_OVER_CP = 287/1004.5)."""
    import numpy as np
    b = json.loads(_pinned(BINDING, 'ALISIOS operator binding').read_text())
    if sha(b['file']) != b['file_sha256'] or sha(b['constant_dependency']['path']) != b['constant_dependency']['sha256']:
        raise ValueError('ALISIOS cloud.py / constant dependency changed')
    cloud = {'np': np, 'R_D_OVER_CP': 287 / 1004.5}
    tree = ast.parse(Path(b['file']).read_text())
    nodes = [z for z in tree.body if isinstance(z, ast.FunctionDef) and z.name in {'finite', 'positive', 'cwp', '_volume', 'wrf_columns'}]
    if len(nodes) != 5:
        raise ValueError('ALISIOS operator functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), b['file'], 'exec'), cloud)
    return cloud, {'binding': {'path': str(BINDING[0]), 'sha256': BINDING[1]}, 'cloud_py': {'path': b['file'], 'sha256': b['file_sha256']}}


def masks():
    import numpy as np
    seal = json.loads(_pinned(MASKS, 'ALISIOS mask seal').read_text())
    if sha(seal['arrays']['path']) != seal['arrays']['sha256']:
        raise ValueError('ALISIOS mask arrays changed')
    G = np.load(seal['arrays']['path'])
    return G, {k: G[k] for k in seal['mask_cells']}, G['cell_area'], {'seal': {'path': str(MASKS[0]), 'sha256': MASKS[1]}, 'arrays': seal['arrays']}


def _stats(v, mask, area):
    """Verbatim driver `stats`: area-weighted mean and p10/p90 by weighted interpolation, min/max; refuse non-finite."""
    import numpy as np
    a = area[mask]; z = v[mask]
    if not np.isfinite(z).all():
        raise ValueError('non-finite value inside a region')
    order = np.argsort(z)
    q = np.interp([.1, .9], (np.cumsum(a[order]) - .5 * a[order]) / a.sum(), z[order])
    return {'mean': float(np.average(z, weights=a)), 'p10': float(q[0]), 'p90': float(q[1]), 'min': float(z.min()), 'max': float(z.max())}


def compute(d03_frames, row):
    """d03_frames: the member's 18 hourly raw wrfout_d03 files, hour order. Returns the metrics document (no write)."""
    import numpy as np
    from netCDF4 import Dataset
    _pinned(DRIVER, 'ALISIOS model-only driver')
    cloud, op_ref = operator()
    G, regions, area, mask_ref = masks()
    frames = [Path(p) for p in d03_frames]
    if len(frames) != 18:
        raise ValueError(f'need 18 hourly d03 frames, got {len(frames)}')
    records = []
    sums = {k: {reg: 0.0 for reg in regions} for k in FIELDS + DERIVED}
    for hour, p in enumerate(frames, 1):
        digest = sha(p)
        with Dataset(p) as ds:
            ds.set_auto_mask(False); ds.set_auto_scale(False)
            for key in GEOMETRY:
                if not np.array_equal(np.asarray(ds[key][0], float), G[key]):
                    raise ValueError(f'geometry drift in {p.name}: {key}')
            stamp = b''.join(np.asarray(ds['Times'][0]).tolist()).decode().strip('\x00')
            cwp, rho, dz = cloud['wrf_columns'](ds, 0, return_geometry=True)
            qc = np.asarray(ds['QCLOUD'][0], float)
            lwp = cloud['cwp']({'QCLOUD': qc}, rho, dz)
            surfaces = {k: np.asarray(ds[k][0], float) for k in FIELDS}
            surfaces.update(LWP=lwp, CWP=cwp, IWP=cwp - lwp, cloud_column_fraction=(cwp >= 10).astype(float),
                            wind_speed_ms=np.hypot(surfaces['U10'], surfaces['V10']))
        rec = {'tau_h': hour, 'Times': stamp, 'source': {'path': str(p), 'sha256': digest}, 'regions': {}}
        for reg, mask in regions.items():
            rec['regions'][reg] = {k: _stats(a, mask, area) for k, a in surfaces.items()}
            for k in sums:
                sums[k][reg] += rec['regions'][reg][k]['mean'] / 18
        records.append(rec)
    out = {'schema': 'wrf_gpu2.screen7.metrics_before_unlink.v1', 'row_id': row['row_id'], 'our_id': row['our_id'],
           'arm_id': row['arm_id'], 'case_id': row['case_id'], 'domain': 'd03', 'hours': 18,
           'definitions': {'driver': {'path': str(DRIVER[0]), 'sha256': DRIVER[1]}, **op_ref, 'masks': mask_ref,
                           'LWP': 'cwp({QCLOUD}, rho_dry, dz) g/m2', 'CWP': 'wrf_columns QCLOUD+QICE g/m2', 'IWP': 'CWP-LWP',
                           'cloud_column_fraction': 'CWP >= 10 g/m2 (area-weighted fraction)', 'wind_speed_ms': 'hypot(U10,V10)',
                           'stats': 'area-weighted mean/p10/p90, min/max (driver stats)'},
           'decision_summary_18h_mean': {reg: {k: sums[k][reg] for k in ('LWP', 'CWP', 'cloud_column_fraction', 'T2', 'wind_speed_ms', 'SWDOWN')}
                                         for reg in regions},
           'records': records, 'truth_reads': 0}
    return out


def encode(doc):
    return (json.dumps(doc, indent=1, allow_nan=False) + '\n').encode()


def publish(dir_fd, name, data):
    """Durable no-overwrite publication inside an already identity-verified directory fd (critic F1/F2):
    exclusive private temp file (O_CREAT|O_EXCL|O_NOFOLLOW, must be a fresh single-link regular file), write + fsync,
    link() to `name` (refuses an existing name or symlink), unlink the temp, fsync the directory, then read back through
    the same dir fd (O_NOFOLLOW, regular, single link, ours, bytes identical). Returns the published sha256."""
    import os, stat, uuid
    tmp = f'.{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    try:
        try:
            st = os.fstat(fd)
            if not (stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size == 0 and st.st_uid == os.getuid()):
                raise ValueError('metrics temp file is not a fresh private regular file')
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    finally:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
    os.fsync(dir_fd)
    return readback(dir_fd, name, expect=data)[0]


def readback(dir_fd, name, expect=None):
    """(sha256, bytes) of `name` read through dir_fd with O_NOFOLLOW; must be our private single-link regular file;
    the file and its directory are fsynced (durable before any irreversible step)."""
    import os, stat
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
        if not (stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_uid == os.getuid() and (st.st_mode & 0o022) == 0):
            raise ValueError('metrics file is not our private single-link regular file')
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        data = b''.join(chunks)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(dir_fd)
    if expect is not None and data != expect:
        raise ValueError('metrics readback differs from the written bytes')
    return hashlib.sha256(data).hexdigest(), data


def export(d03_frames, row, out_path):
    """Convenience (lane checks): compute + publish into out_path's directory (opened O_DIRECTORY|O_NOFOLLOW)."""
    import os
    out_path = Path(out_path)
    dfd = os.open(out_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        publish(dfd, out_path.name, encode(compute(d03_frames, row)))
    finally:
        os.close(dfd)
    return out_path
