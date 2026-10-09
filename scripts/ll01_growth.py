"""Read-only late-lead diagnostics against original CPU-WRF, frozen regions.

This reports differences; it neither changes the pinned twin scorer nor grants
a predictability waiver. All mutable outputs stay in the lane artifact dir.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
import numpy as np
from netCDF4 import Dataset, chartostring

SCORER = Path('<DATA_ROOT>/alisios/state/manager/runs/v32_sprints_20261002/R32-01/scoring_core_R32_01.py')
SCORER_SHA = '72d8a76b4f7ddceb9cf49f726db6c4f5a8e04fb8d45f6d36cc1e119df70c72d8'
NATIVE = ('T', 'QVAPOR', 'U', 'V', 'W', 'QKE', 'TKE_PBL')
SURFACE = ('PBLH', 'HFX', 'SST', 'TSK', 'T2', 'Q2', 'U10', 'V10', 'PSFC', 'SWNORM', 'SWDNB')


def read_mass(ds, name, ys, xs):
    var = ds[name]
    ystop = ys.stop + int('south_north_stag' in var.dimensions)
    xstop = xs.stop + int('west_east_stag' in var.dimensions)
    indices = [0]
    if len(var.dimensions) == 4:
        indices.append(slice(None))
    indices += [slice(ys.start, ystop), slice(xs.start, xstop)]
    a = np.asarray(var[tuple(indices)], np.float64)
    if 'west_east_stag' in var.dimensions:
        a = .5 * (a[..., :-1] + a[..., 1:])
    if 'south_north_stag' in var.dimensions:
        a = .5 * (a[..., :-1, :] + a[..., 1:, :])
    if 'bottom_top_stag' in var.dimensions:
        a = .5 * (a[:-1] + a[1:])
    assert np.isfinite(a).all(), (name, 'nonfinite')
    return a


def stats(a, b, mask, *, vector=False):
    if vector:
        delta = np.hypot(b[0] - a[0], b[1] - a[1])[mask]
        ref = np.hypot(a[0], a[1])[mask]
    else:
        delta = (b - a)[mask]
        ref = a[mask]
    if not delta.size:
        return None
    return {'n': int(delta.size), 'bias': None if vector else float(delta.mean()),
            'rms': float(np.sqrt(np.mean(delta * delta))),
            'mae': float(np.mean(np.abs(delta))), 'p95': float(np.quantile(np.abs(delta), .95)),
            'max_abs': float(np.abs(delta).max()), 'cpu_mean': float(ref.mean()),
            'cpu_rms': float(np.sqrt(np.mean(ref * ref)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--issue', required=True)
    p.add_argument('--version', default='v031')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--leads', default='all')
    p.add_argument('--native-fields', default=','.join(NATIVE))
    p.add_argument('--surface-fields', default=','.join(SURFACE))
    args = p.parse_args()
    assert args.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/mass-sol'))
    assert hashlib.sha256(SCORER.read_bytes()).hexdigest() == SCORER_SHA
    spec = importlib.util.spec_from_file_location('ll01_frozen_regions', SCORER)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    inventory = json.loads(args.inventory.read_text())
    leads = list(range(73)) if args.leads == 'all' else [int(x) for x in args.leads.split(',')]
    fields = tuple(dict.fromkeys(args.native_fields.split(',') + args.surface_fields.split(',') + ['U10', 'V10']))
    records, provenance, geometries = [], [], {}
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'growth.jsonl').exists():
        raise FileExistsError('Use a fresh output directory for each immutable arm')
    for domain in ('d01', 'd02', 'd03'):
        chosen = {}
        for label in ('cpu_wrf', args.version):
            hits = [r for r in inventory['sources'] if r['issue'] == args.issue
                    and r['source'] == label and r['domain'] == domain]
            assert len(hits) == 1, (domain, label, hits)
            chosen[label] = hits[0]
        file_axes = {label: sorted(Path(row['dir']).glob(f'wrfout_{domain}_*'))
                     for label, row in chosen.items()}
        assert all(len(files) == 73 for files in file_axes.values()), file_axes
        first = Path(chosen['cpu_wrf']['dir']) / chosen['cpu_wrf']['first']
        init = datetime.strptime(first.name[-19:], '%Y-%m-%d_%H:%M:%S')
        with Dataset(first, 'r') as ds:
            ds.set_auto_mask(False)
            lat, lon, hgt, land, cos, sin = [np.asarray(ds[k][0], np.float64)
                for k in ('XLAT', 'XLONG', 'HGT', 'LANDMASK', 'COSALPHA', 'SINALPHA')]
            dx, dy = float(ds.DX), float(ds.DY)
            masks, coast, slope = frozen.pixel_classes(lat, lon, hgt, land, dx, dy)
            yy, xx = np.nonzero(masks['TENERIFE_LAND'])
            assert len(yy), domain
            # EXACT crop used by the pinned teacher_readonly.geometry function.
            ys = slice(int(yy.min()), int(yy.max()) + 1)
            xs = slice(int(xx.min()), int(xx.max()) + 1)
        geo = {'lat': lat[ys, xs], 'lon': lon[ys, xs], 'hgt': hgt[ys, xs],
               'land': land[ys, xs], 'coast_km': coast[ys, xs], 'slope_deg': slope[ys, xs],
               'masks': {k: m[ys, xs] for k, m in masks.items()}, 'dx': dx, 'dy': dy}
        cs, sn = cos[ys, xs], sin[ys, xs]
        geometries[domain] = {'dx': dx, 'dy': dy, 'bbox': [ys.start, ys.stop, xs.start, xs.stop],
            'frozen_mask_counts': {k: int(m.sum()) for k, m in geo['masks'].items()}}
        for lead in leads:
            if Path('/tmp/wrf_gpu2_quiet').exists():
                print('QUIET: stop before next read; completed lead records retained', flush=True)
                raise SystemExit(3)
            from datetime import timedelta
            stamp = (init + timedelta(hours=lead)).strftime('%Y-%m-%d_%H:%M:%S')
            paths = {label: files[lead] for label, files in file_axes.items()}
            values, clocks, native_times = {}, {}, {}
            for label, path in paths.items():
                before = path.stat()
                with Dataset(path, 'r') as ds:
                    ds.set_auto_mask(False)
                    native_times[label] = str(chartostring(ds['Times'][:])[0])
                    assert path.name.endswith(native_times[label]), path
                    assert (float(ds.DX), float(ds.DY)) == (dx, dy), path
                    assert np.max(np.abs(np.asarray(ds['HGT'][0, ys, xs]) - geo['hgt'])) < .02, path
                    values[label] = {k: read_mass(ds, k, ys, xs) for k in fields if k in ds.variables}
                    clocks[label] = float(ds['XTIME'][0])
                after = path.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
                provenance.append({'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns})
                f = values[label]
                for u, v in (('U', 'V'), ('U10', 'V10')):
                    if u not in f or v not in f:
                        continue
                    old_u, old_v = f[u], f[v]
                    f[u], f[v] = old_u * cs - old_v * sn, old_u * sn + old_v * cs
            assert clocks['cpu_wrf'] == clocks[args.version], (paths, clocks)
            assert abs(clocks['cpu_wrf'] - 60 * lead) <= 1., (paths, clocks)
            c, g = values['cpu_wrf'], values[args.version]
            surface_c = {'wind_u': c['U10'], 'wind_v': c['V10'],
                         'wind_speed': np.hypot(c['U10'], c['V10'])}
            surface_g = {'wind_u': g['U10'], 'wind_v': g['V10'],
                         'wind_speed': np.hypot(g['U10'], g['V10'])}
            axis = frozen.reference_axis(surface_c, geo)
            regions = dict(geo['masks'])
            regions['SEA_DIAGNOSTIC_ONLY'] = geo['land'] == 0
            if axis is not None:
                coastal = geo['masks']['TENERIFE_LAND'] & (geo['coast_km'] <= 5) & (geo['hgt'] < 300)
                regions['CPU_AXIS_LEE'] = coastal & (axis >= 10)
                regions['CPU_AXIS_UPWIND'] = coastal & (axis <= -10)
            row = {'issue': args.issue, 'version': args.version, 'domain': domain,
                   'lead': lead, 'nominal_times': stamp, 'native_times': native_times,
                   'xtime_min': clocks['cpu_wrf'], 'regions': {},
                   'structure_cpu': frozen.structure_scalars(surface_c, geo, axis),
                   'structure_gpu': frozen.structure_scalars(surface_g, geo, axis)}
            for region, mask in regions.items():
                measurements = {}
                for field in c.keys() & g.keys():
                    a, b = c[field], g[field]
                    if a.ndim == 3:
                        for k in (0, 1, 3, 6, 10, 20, 30, 40):
                            if k < a.shape[0]:
                                measurements[f'{field}:k{k}'] = stats(a[k], b[k], mask)
                    else:
                        measurements[field] = stats(a, b, mask)
                measurements['wind_vector10'] = stats((c['U10'], c['V10']), (g['U10'], g['V10']), mask, vector=True)
                for k in (0, 1, 3, 6, 10, 20, 30, 40):
                    if 'U' in c and 'V' in c and k < c['U'].shape[0]:
                        measurements[f'wind_vector:k{k}'] = stats((c['U'][k], c['V'][k]), (g['U'][k], g['V'][k]), mask, vector=True)
                row['regions'][region] = measurements
            records.append(row)
            with (args.out / 'growth.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            print(domain, lead, 'north vector10 RMS', row['regions']['NORTH_SLOPE']['wind_vector10'], flush=True)
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_s': time.monotonic() - start,
               'issue': args.issue, 'version': args.version, 'n_records': len(records),
               'region_operator_sha': SCORER_SHA, 'geometry': geometries, 'sources': provenance,
               'scope': 'diagnostic differences vs ORIGINAL CPU-WRF; no threshold changes or chaos waiver',
               'cache': 'none', 'source_writes': False}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('GROWTH_DONE', receipt['elapsed_s'], len(records), flush=True)


if __name__ == '__main__':
    main()
