"""Describe retained history observations, without inventing missing stage inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    rows = []
    for item in manifest:
        path = root / Path(item['output']).name
        assert digest(path) == item['sha256']
        with np.load(path, allow_pickle=False) as z:
            for field in ('QCLOUD', 'QRAIN', 'QC_RAD'):
                nz, ny, nx = item['fields'][field]['shape']

                def at(name, k, j, i, side):
                    value = z[name + '__' + side]
                    shape = item['fields'][name]['shape']
                    width = item['fields'][name]['width']
                    if side == 'E':
                        i -= shape[-1] - width
                    if side == 'N':
                        j -= shape[-2] - width
                    return value[k, j, i] if len(shape) == 3 else value[j, i]

                for index, q in zip(z[field + '__negative_kji'], z[field + '__negative_values']):
                    k, j, i = map(int, index)
                    ring = min(j, ny - 1 - j, i, nx - 1 - i)
                    assert ring == 0
                    # WRF's Y boundary owns corners. Face velocity here is the
                    # FINAL HISTORY value, not acoustic/stage transport.
                    side = 'S' if j == 0 else 'N' if j == ny - 1 else 'W' if i == 0 else 'E'
                    if side in ('S', 'N'):
                        velocity = at('V', k, 0 if side == 'S' else ny, i, side)
                        neighbor = at(field, k, 1 if side == 'S' else ny - 2, i, side)
                    else:
                        velocity = at('U', k, j, 0 if side == 'W' else nx, side)
                        neighbor = at(field, k, j, 1 if side == 'W' else nx - 2, side)
                    outflow = velocity < 0 if side in ('S', 'W') else velocity > 0
                    # This positive denominator can exclude a sign flip in the
                    # history-based inverse only. It is NOT authoritative mu_total.
                    mass = np.float32(at('MU', k, j, i, side) + at('MUB', k, j, i, side))
                    den = np.float32(z['C1H'][k] * mass + z['C2H'][k])
                    assert np.isfinite(den) and den > 0
                    rows.append(dict(frame=path.stem, field=field, kji=[k, j, i],
                                     side=side, q=float(q), history_denominator=float(den),
                                     same_frame_face_velocity=float(velocity),
                                     same_frame_outflow=bool(outflow),
                                     same_frame_inward_neighbor=float(neighbor)))
    groups = {}
    for row in rows:
        name = row['frame'].split('_')[1] + '/' + row['field']
        group = groups.setdefault(name, dict(count=0, final_outflow=0, neighbor_zero=0,
                                            neighbor_nonnegative=0, denominator_min=float('inf')))
        group['count'] += 1
        group['final_outflow'] += row['same_frame_outflow']
        group['neighbor_zero'] += row['same_frame_inward_neighbor'] == 0
        group['neighbor_nonnegative'] += row['same_frame_inward_neighbor'] >= 0
        group['denominator_min'] = min(group['denominator_min'], row['history_denominator'])
    return dict(schema='v45-bdyneg.capture-mechanism.v1', manifest_sha256=digest(root / 'MANIFEST.json'),
                observations=len(rows), groups=groups, rows=rows,
                measured_scope='Final histories only; QC_RAD is a held solver-input diagnostic',
                inference='Positive history denominator cannot explain the sign by division alone. '
                          'A final outflow face plus nonnegative final inward neighbor does not '
                          'reconstruct flow_dep_bdy: acoustic transport and pre-MP neighbor are absent.',
                scientific_gate='OPEN: no event-time coupled records or PD/pre-MP operands retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('captures', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = inspect(args.captures)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(observations=result['observations'], groups=result['groups'])))
