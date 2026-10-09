"""Quantify history-projection loss separately from original CPU-WRF error."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
import numpy as np
from netCDF4 import Dataset, chartostring
from ll01_growth import SCORER, SCORER_SHA, read_mass, stats

FIELDS = ('U', 'V', 'W', 'T', 'THM', 'QVAPOR', 'QCLOUD', 'QKE', 'U10', 'V10',
          'T2', 'Q2', 'TSK', 'PBLH', 'HFX', 'SST', 'MU', 'SWNORM', 'SWDNB')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--projected', required=True, type=Path)
    p.add_argument('--cpu', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--leads', default='37,38,39,40,41,42,43,44,45,46,47,48')
    args = p.parse_args()
    assert args.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/mass-sol'))
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    assert hashlib.sha256(SCORER.read_bytes()).hexdigest() == SCORER_SHA
    spec = importlib.util.spec_from_file_location('ll01_pinned_regions', SCORER)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    started = time.monotonic()
    rows, sources = [], []
    for domain in ('d01', 'd02', 'd03'):
        baseline = sorted(args.baseline.glob(f'wrfout_{domain}_*'))
        cpu = sorted(args.cpu.glob(f'wrfout_{domain}_*'))
        assert len(baseline) == len(cpu) == 73
        with Dataset(cpu[0], 'r') as ds:
            ds.set_auto_mask(False)
            lat, lon, hgt, land, cs, sn = [np.asarray(ds[k][0], np.float64)
                for k in ('XLAT', 'XLONG', 'HGT', 'LANDMASK', 'COSALPHA', 'SINALPHA')]
            masks, coast, slope = frozen.pixel_classes(lat, lon, hgt, land, float(ds.DX), float(ds.DY))
            dx, dy = float(ds.DX), float(ds.DY)
            y, x = np.nonzero(masks['TENERIFE_LAND'])
            ys, xs = slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1)
        geo = {'lat': lat[ys, xs], 'lon': lon[ys, xs], 'hgt': hgt[ys, xs],
               'land': land[ys, xs], 'coast_km': coast[ys, xs], 'slope_deg': slope[ys, xs],
               'masks': {n: a[ys, xs] for n, a in masks.items()}, 'dx': dx, 'dy': dy}
        cs, sn = cs[ys, xs], sn[ys, xs]
        for lead in map(int, args.leads.split(',')):
            if Path('/tmp/wrf_gpu2_quiet').exists():
                print('QUIET', flush=True)
                raise SystemExit(3)
            paths = {'cpu': cpu[lead], 'baseline': baseline[lead],
                     'projected': args.projected / baseline[lead].name}
            data, clocks, times = {}, {}, {}
            for label, path in paths.items():
                before = path.stat()
                with Dataset(path, 'r') as ds:
                    ds.set_auto_mask(False)
                    clocks[label] = float(ds['XTIME'][0])
                    times[label] = str(chartostring(ds['Times'][:])[0])
                    data[label] = {n: read_mass(ds, n, ys, xs) for n in FIELDS}
                after = path.stat()
                assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), path
                sources.append({'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns})
                for u, v in (('U', 'V'), ('U10', 'V10')):
                    a, b = data[label][u], data[label][v]
                    data[label][u], data[label][v] = a * cs - b * sn, a * sn + b * cs
            assert len(set(clocks.values())) == len(set(times.values())) == 1, (clocks, times)
            c, b, r = data['cpu'], data['baseline'], data['projected']
            surface_c = {'wind_u': c['U10'], 'wind_v': c['V10'],
                         'wind_speed': np.hypot(c['U10'], c['V10'])}
            axis = frozen.reference_axis(surface_c, geo)
            regions = dict(geo['masks'])
            regions['SEA_DIAGNOSTIC_ONLY'] = geo['land'] == 0
            row = {'domain': domain, 'lead': lead, 'xtime_min': clocks['cpu'], 'time': times['cpu'], 'regions': {}}
            for region, mask in regions.items():
                cells = {}
                for field in FIELDS:
                    levels = (0, 6, 20) if c[field].ndim == 3 else (None,)
                    for k in levels:
                        arrays = [a[field] if k is None else a[field][k] for a in (c, b, r)]
                        cc, bb, rr = arrays
                        cells[field if k is None else f'{field}:k{k}'] = {
                            'projection_regression': stats(bb, rr, mask),
                            'baseline_vs_cpu_wrf': stats(cc, bb, mask),
                            'projected_vs_cpu_wrf': stats(cc, rr, mask)}
                cells['wind_vector10'] = {
                    'projection_regression': stats((b['U10'], b['V10']), (r['U10'], r['V10']), mask, vector=True),
                    'baseline_vs_cpu_wrf': stats((c['U10'], c['V10']), (b['U10'], b['V10']), mask, vector=True),
                    'projected_vs_cpu_wrf': stats((c['U10'], c['V10']), (r['U10'], r['V10']), mask, vector=True)}
                row['regions'][region] = cells
            for label, values in (('cpu_wrf', c), ('baseline', b), ('projected', r)):
                surface = {'wind_u': values['U10'], 'wind_v': values['V10'],
                           'wind_speed': np.hypot(values['U10'], values['V10'])}
                row['structure_' + label] = frozen.structure_scalars(surface, geo, axis)
            rows.append(row)
            with (args.out / 'projection_differences.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            cell = row['regions']['MEDANO_5KM']['wind_vector10']
            print(domain, lead, 'Medano projection/baselineCPU/projectedCPU RMS',
                  {n: None if a is None else a['rms'] for n, a in cell.items()}, flush=True)
    receipt = {'elapsed_s': time.monotonic() - started, 'n_rows': len(rows),
               'source_files': sources, 'region_operator_sha': SCORER_SHA,
               'projection_comparison': 'GPU-vs-GPU regression ONLY',
               'fidelity_reference': 'ORIGINAL CPU-WRF', 'cache': 'none',
               'scope': 'Measure projection loss; no invented tolerance or chaos waiver'}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('PROJECTION_SCORE_DONE', receipt['elapsed_s'], len(rows), flush=True)


if __name__ == '__main__':
    main()
