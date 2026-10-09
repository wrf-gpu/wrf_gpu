#!/usr/bin/env python3
"""Source-bound, one-frame-memory extraction of a complete 72-hour case."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import resource
import subprocess
from types import SimpleNamespace

from hourly_deviation_bands import extract, sha, validate_metrics


def source_receipt(run, revision, repository, domains, expected_hours=73):
    receipt_path = run / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    expected = subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', revision + '^{commit}'], text=True).strip()
    if receipt.get('rc') != 0 or receipt['source']['dirty'] or receipt['source']['git_head'] != expected:
        raise ValueError('receipt is not a successful clean run of the expected revision')
    tree = subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', expected + ':src/gpuwrf'], text=True).strip()
    if receipt['source']['src_tree'] != tree:
        raise ValueError('receipt forecast source-tree identity mismatch')
    compressed_path = run / 'compress_receipt.jsonl'
    compressed = [json.loads(s) for s in compressed_path.read_text().splitlines() if s]
    if len(compressed) != expected_hours * len(domains) or not all(r['value_identical'] for r in compressed):
        raise ValueError('require all compressed frames and exact decompressed values')
    hashes = {r['file']: r['comp_sha256'] for r in compressed}
    if len(hashes) != len(compressed):
        raise ValueError('duplicate compression frame receipt')
    return expected, hashes, {'receipt_sha256': sha(receipt_path),
                             'compression_receipt_sha256': sha(compressed_path),
                             'src_gpuwrf_tree': tree}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='directory with receipt, compression receipt and wrfout/')
    p.add_argument('--cpu', type=Path, required=True, help='original CPU-WRF history, including any explicitly named IC member')
    p.add_argument('--revision', required=True, help='expected actually executed immutable commit')
    p.add_argument('--repository', type=Path, required=True, help='git repository retaining the expected source object')
    p.add_argument('--case', required=True)
    p.add_argument('--domains', nargs='+', default=['d01', 'd02', 'd03'])
    p.add_argument('--hours', type=int, choices=(24, 72), default=72)
    p.add_argument('--binding-report', type=Path, help='existing complete comparator directory; never rerun comparator')
    p.add_argument('--pair', type=Path)
    p.add_argument('--pair-label')
    p.add_argument('--pair-reference', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--scope', required=True, help='source-specific status; original CPU IC and proxy distinctions')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('analysis is CPU only; set JAX_PLATFORMS=cpu')
    if a.out.exists():
        raise ValueError('fresh extraction output directory required')
    if a.pair and not a.pair_label:
        raise ValueError('pair input requires a source/IC description')
    revision, hashes, provenance = source_receipt(a.run, a.revision, a.repository, a.domains, a.hours+1)
    args = SimpleNamespace(out=a.out, cpu=a.cpu, gpu=a.run / 'wrfout', pair=a.pair,
        case=a.case, gpu_tree=revision, domains=a.domains, manifest=a.manifest,
        scope=a.scope, pair_label=a.pair_label, pair_reference=a.pair_reference,
        binding_report=a.binding_report, expected_hours=a.hours+1)
    result = validate_metrics(extract(args))
    for entry in result['domains'].values():
        if len(entry['sources']) != a.hours+1:
            raise ValueError('incomplete source-frame inventory')
        for source in entry['sources']:
            if source['gpu_sha256'] != hashes[Path(source['gpu']).name]:
                raise ValueError('loaded forecast frame differs from compression receipt')
    # Ratio guides are context; source-specific A3/A9/A10 decisions remain separate.
    for filename in ['metrics.json', 'metrics.private.json']:
        path = a.out / filename
        payload = json.loads(path.read_text())
        payload['ratio_note'] = 'Ratios 1 and 2 are empirical context; A3 uses C > limit or C/X >= 0.75 at 72 h.'
        payload['ratio_scope'] = 'A single pair is not a confidence interval or a waiver. Strict 24 h and source-bound release gates remain separate.'
        payload['ratio_guide_label'] = '(empirical guides, not the A3 release rule)'
        validate_metrics(payload)
        path.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n')
    done = dict(provenance, status='EXTRACTION_DONE', revision=revision,
        case=a.case, domains=a.domains, domain_hours=(a.hours+1)*len(a.domains),
        numeric_rows=(11*(a.hours+1)-1)*len(a.domains), raw_operands_full_finite=True,
        mask='none', cpu_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        cache='NONE', rendered=False, forecast_release_claim=False)
    (a.out / 'extraction_done.json').write_text(json.dumps(done, indent=2) + '\n')
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
