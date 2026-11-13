"""Read-only verbatim copies of WT1 teacher code (ALISIOS R32-13), used as independent test oracles.

Copied because the test suite guards <DATA_ROOT>/alisios reads. Never edit; ``verify_pins.py`` (run outside
pytest) re-reads the sources and checks the SHA-256 and every copied line.

teacher_extract.py    sha256 8f8120df84c7c4513f5bdd39f08216f9403a1e7aeee9544065415d9fb76464d1
teacher_extract_v2.py sha256 75f02a61b5830cce9a2f624a067a5214bdde8b3011a0411439ef2837a14c3a5d
"""
import datetime as dt
import re
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree

UTC = dt.timezone.utc
TEACHER_DIR = "<DATA_ROOT>/alisios/state/manager/runs/v32_sprints_20261002/R32-13/"
TEACHER_SHA = {"teacher_extract.py": "8f8120df84c7c4513f5bdd39f08216f9403a1e7aeee9544065415d9fb76464d1",
               "teacher_extract_v2.py": "75f02a61b5830cce9a2f624a067a5214bdde8b3011a0411439ef2837a14c3a5d"}
# (file, first line, last line) of every verbatim block below, in order.
BLOCKS = (("teacher_extract.py", 35, 37), ("teacher_extract.py", 52, 54), ("teacher_extract.py", 163, 170),
          ("teacher_extract.py", 173, 178), ("teacher_extract_v2.py", 31, 40), ("teacher_extract_v2.py", 306, 309),
          ("teacher_extract_v2.py", 321, 336), ("teacher_extract_v2.py", 394, 418))
# frame_fields: only the profile statements, verbatim single lines (teacher_extract_v2.py).
FRAME_FIELDS_LINES = (348, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 388, 389, 390)

# ---- teacher_extract.py 35-37
def need(condition, reason):
    if not condition:
        raise ValueError(reason)


# ---- teacher_extract.py 52-54
def parse_issue(issue):
    need(re.fullmatch(r"\d{8}_\d{2}z", issue), "invalid issue")
    return dt.datetime.strptime(issue, "%Y%m%d_%Hz").replace(tzinfo=UTC)


# ---- teacher_extract.py 163-170
def raw_array(ds, name):
    v = ds.variables[name]
    v.set_auto_maskandscale(False)
    arr = np.asarray(v[0], dtype=np.float32)
    fill = getattr(v, "_FillValue", None)
    need(fill is None or not np.any(arr == fill), "raw fill value: " + name)
    need(np.all(np.isfinite(arr)), "raw nonfinite: " + name)
    return arr


# ---- teacher_extract.py 173-178
def time_coordinates(actual, issue):
    lead_s = (actual - parse_issue(issue)).total_seconds()
    nominal = int(round(lead_s / 3600))
    offset = lead_s - nominal * 3600
    need(0 <= offset < 60, "actual time outside [nominal, nominal+60s)")
    return nominal, offset


base = SimpleNamespace(need=need, parse_issue=parse_issue, raw_array=raw_array, time_coordinates=time_coordinates)

# ---- teacher_extract_v2.py 31-40
PROFILE_SCHEMA = pa.schema([
    ('phys_id', pa.string()), ('role', pa.string()), ('issue', pa.string()),
    ('attempt', pa.int32()), ('valid_utc', pa.string()), ('actual_utc', pa.string()),
    ('lead_h', pa.int16()), ('tau_h', pa.int16()), ('time_offset_seconds', pa.float32()),
    ('level', pa.int16()), ('grid_y', pa.int32()), ('grid_x', pa.int32()),
    ('T_K', pa.float32()), ('theta_K', pa.float32()), ('pressure_Pa', pa.float32()),
    ('QV_kg_kg', pa.float32()), ('U_ms', pa.float32()), ('V_ms', pa.float32()),
    ('QCLOUD_kg_kg', pa.float32()), ('CLDFRA', pa.float32()),
    ('z_msl_m', pa.float32()), ('z_agl_m', pa.float32())])



# ---- teacher_extract_v2.py 306-309
def earth_xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat).ravel() * np.cos(lon).ravel(),
                            np.cos(lat).ravel() * np.sin(lon).ravel(), np.sin(lat).ravel()))


# ---- teacher_extract_v2.py 321-336
def mapped_points(cfg, static):
    doc, _ = base.bound_json(cfg['point_set_path'], cfg['point_set_sha256'])
    base.need(len(doc['points']) <= cfg['max_points'], 'point cap')
    base.need(len({p['phys_id'] for p in doc['points']}) == len(doc['points']), 'duplicate point identity')
    tree = cKDTree(earth_xyz(static['XLAT'], static['XLONG']))
    rows = []
    for point in doc['points']:
        distance, ix = tree.query(earth_xyz([point['lat']], [point['lon']])[0])
        y, x = np.unravel_index(ix, static['XLAT'].shape)
        dist = float(2 * 6371000 * np.arcsin(min(1, distance/2)))
        ocean_ok = point['role'] != 'sar_support_geometry_only' or static['LANDMASK'][y, x] == 0
        available = dist <= cfg['profile_max_distance_m'] and ocean_ok
        rows.append({**point, 'grid_y': int(y), 'grid_x': int(x), 'distance_m': dist,
                     'profile_status': 'AVAILABLE' if available else 'OUTSIDE_SUPPORT_OR_NOT_SEA',
                     'grid_lat': float(static['XLAT'][y,x]), 'grid_lon': float(static['XLONG'][y,x])})
    return rows


def frame_profiles(src):
    """teacher_extract_v2.frame_fields, profile statements only (verbatim lines [348, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 388, 389, 390]);
    the static dict is reduced to HGT (the only static used by these lines)."""
    static = {'HGT': base.raw_array(src, 'HGT')}
    cf = base.raw_array(src, 'CLDFRA')
    geop = base.raw_array(src, 'PH').astype('f8') + base.raw_array(src, 'PHB').astype('f8')
    zw = geop / 9.81
    zmsl = ((zw[:-1] + zw[1:]) / 2).astype('f4')
    zagl = zmsl - static['HGT']
    base.need(np.all(np.diff(zw, axis=0) > 0), 'vertical geometry invalid')
    pressure = base.raw_array(src, 'P').astype('f8') + base.raw_array(src, 'PB').astype('f8')
    theta = base.raw_array(src, 'T').astype('f8') + 300
    temp = theta * (pressure / 100000)**(2/7)
    qv = base.raw_array(src, 'QVAPOR')
    qs = {k: base.raw_array(src, k) for k in ['QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP']}
    profiles = {'T_K': temp.astype('f4'), 'theta_K': theta.astype('f4'), 'pressure_Pa': pressure.astype('f4'),
                'QV_kg_kg': qv, 'QCLOUD_kg_kg': qs['QCLOUD'], 'CLDFRA': cf,
                'z_msl_m': zmsl, 'z_agl_m': zagl}
    return profiles


# ---- teacher_extract_v2.py 394-418
def profile_table(src, prof, points, issue, attempt, actual, nominal):
    available = [p for p in points if p['profile_status'] == 'AVAILABLE']
    ny, nx = prof['T_K'].shape[1:]
    raw_u, raw_v = base.raw_array(src, 'U'), base.raw_array(src, 'V')
    base.need(raw_u.shape == (prof['T_K'].shape[0], ny, nx+1) and
              raw_v.shape == (prof['T_K'].shape[0], ny+1, nx), 'staggered wind shape')
    c, s = base.raw_array(src, 'COSALPHA'), base.raw_array(src, 'SINALPHA')
    levels = prof['T_K'].shape[0]
    data = {k: [] for k in PROFILE_SCHEMA.names}
    raster = base.parse_issue(issue) + dt.timedelta(hours=nominal)
    for p in available:
        y, x = p['grid_y'], p['grid_x']
        ug = (raw_u[:,y,x].astype('f8') + raw_u[:,y,x+1].astype('f8'))/2
        vg = (raw_v[:,y,x].astype('f8') + raw_v[:,y+1,x].astype('f8'))/2
        vals = {**{k:a[:,y,x] for k,a in prof.items()},
                'U_ms': (ug*c[y,x]-vg*s[y,x]).astype('f4'),
                'V_ms': (vg*c[y,x]+ug*s[y,x]).astype('f4')}
        common = {'phys_id':p['phys_id'], 'role':p['role'], 'issue':issue, 'attempt':attempt,
                  'valid_utc':raster.isoformat(), 'actual_utc':actual.isoformat(), 'lead_h':nominal,
                  'tau_h':nominal-6, 'time_offset_seconds':(actual-raster).total_seconds(),
                  'grid_y':y, 'grid_x':x}
        for k,v in common.items(): data[k].extend([v]*levels)
        data['level'].extend(range(levels))
        for k,v in vals.items(): data[k].extend(v.tolist())
    return pa.Table.from_pydict(data, schema=PROFILE_SCHEMA)
