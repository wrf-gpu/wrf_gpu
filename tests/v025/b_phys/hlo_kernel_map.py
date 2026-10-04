"""Extract kernel/source crosswalk from exact optimized XLA modules.

A fusion spanning several scheme families remains MIXED. No duration is assigned
from instruction counts. Only exact generated instruction-name matches are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

TOOLS = Path(__file__).resolve().parents[3] / '.agent/sprints/2026-09-19-v0250-planning-opus/tools/hlo_structure.py'
spec = importlib.util.spec_from_file_location('b_phys_hlo_structure', TOOLS)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def source_family(file):
    file = file.lower()
    if 'mynn' in file:
        return 'MYNN'
    if 'thompson' in file:
        return 'THOMPSON'
    if 'cumulus_kf' in file:
        return 'KF'
    if 'noahmp' in file or 'noah_mp' in file:
        return 'NOAHMP'
    if 'rrtmg' in file:
        return 'RRTMG'
    if '/dynamics/' in file:
        return 'DYCORE'
    return None


def metadata_family(info, tabs):
    """Search nested trace frames for a scheme, skipping generic math helpers."""
    files, funcs, locs, frames = tabs
    fid = info.get('frame', 0)
    seen = set()
    while fid in frames and fid not in seen:
        seen.add(fid)
        fr = frames[fid]
        loc = locs.get(int(fr['file_location_id']), {})
        file = files.get(int(loc.get('file_name_id', 0)), '')
        family = source_family(file)
        if family:
            return family
        fid = int(fr.get('parent_frame_id', 0))
    match = re.search(r'source_file="([^"]*)"', info['line'])
    return source_family(match.group(1)) if match else None


def extract(path):
    comps, entry, tabs = H.parse(path)
    instructions = {name: [i for ln in lines if (i := H.classify(ln))]
                    for name, lines in comps.items()}
    memo = {}

    def origins(name, seen=frozenset()):
        if name in memo:
            return memo[name]
        if name in seen:
            return set()
        result = set()
        for info in instructions.get(name, ()):
            family = metadata_family(info, tabs)
            if family:
                result.add(family)
            if info.get('calls'):
                result.update(origins(info['calls'], seen | {name}))
        memo[name] = result
        return result

    header = Path(path).read_text().splitlines()[0]
    domain = 'd01' if re.search(r'\[(?:44|45),70,12[01]\]', header) else (
        'd02' if re.search(r'\[(?:44|45),117,26[78]\]', header) else None)
    mapping = defaultdict(set)
    metadata = {}
    for items in instructions.values():
        for info in items:
            if info['op'] != 'fusion':
                continue
            # XLA emitter replaces HLO punctuation in the generated symbol.
            kernel = re.sub(r'[^a-zA-Z0-9_]', '_', info['name'])
            metadata[kernel] = dict(hlo_type=info['type'],
                                    source=H.frame_str(tabs, info.get('frame', 0)),
                                    op_name=info.get('op_name'))
            families = origins(info.get('calls', ''))
            family = metadata_family(info, tabs)
            if family:
                families.add(family)
            mapping[kernel].add(next(iter(families)) if len(families) == 1
                                else 'MIXED' if families else 'UNKNOWN')
    mapping = {k: next(iter(v)) if len(v) == 1 else 'AMBIGUOUS' for k, v in mapping.items()}
    return dict(file=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                entry=entry, domain=domain, kernels=mapping, metadata=metadata, family_counts=dict(Counter(mapping.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('modules', nargs='+', type=Path)
    args = parser.parse_args()
    result = dict(schema='wrf_gpu2.b_phys.hlo_kernel_map.v1', modules=[extract(p) for p in args.modules])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({m['file']: m['family_counts'] for m in result['modules']}, indent=2))


if __name__ == '__main__':
    main()
