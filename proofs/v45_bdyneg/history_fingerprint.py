"""Expand BD07 to all retained samples; inverse numerators remain INFERRED."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def real4(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bits(value):
    return struct.unpack('<I', struct.pack('<f', abs(value)))[0]


def inspect(path):
    report = json.loads(path.read_text())
    groups = {}
    for row in report['rows']:
        name = row['frame'].split('_')[1] + '/' + row['field']
        group = groups.setdefault(name, dict(samples=0, inferred_power_of_two=0,
                                            within_one_ulp_of_power_of_two=0,
                                            inverse_division_exact=0))
        q, den = row['q'], row['history_denominator']
        numerator = real4(q * den)
        mantissa, exponent = math.frexp(abs(numerator))
        powers = (math.ldexp(1., exponent - 1), math.ldexp(1., exponent))
        group['samples'] += 1
        group['inferred_power_of_two'] += mantissa == .5
        group['within_one_ulp_of_power_of_two'] += min(abs(bits(numerator) - bits(p)) for p in powers) <= 1
        group['inverse_division_exact'] += real4(numerator / den) == q
    return dict(schema='v45-bdyneg.all-history-fingerprint.v1', groups=groups,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                interpretation='INFERRED inverse numerators, not retained boundary or PD operands. '
                               'Power-of-two residuals are consistent with cancellation but do not '
                               'identify an operator. QC_RAD is a held diagnostic.',
                scientific_gate='OPEN')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = inspect(args.input)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['groups']))
