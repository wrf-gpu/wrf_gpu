"""Extract per-call device operations and kernel resources from NVTX captures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    db = sqlite3.connect(f'file:{args.sqlite.resolve()}?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute('select id,value from StringIds'))
    ranges = []
    for row in db.execute('select * from NVTX_EVENTS where end is not null'):
        r = dict(row)
        name = r.get('text') or strings.get(r.get('textId'),'')
        if name.startswith('NOAHMP:'):
            ranges.append((name,r['start'],r['end']))
    assert ranges, 'missing measured NOAHMP ranges'
    grouped = defaultdict(list)
    for label,start,end in ranges:
        _,mode,case,index = label.split(':')
        record = dict(label=label,wall_ms=(end-start)/1.e6,kinds={},kernels=[])
        for kind in ['KERNEL','MEMCPY','MEMSET']:
            table = 'CUPTI_ACTIVITY_KIND_'+kind
            rows = [dict(r) for r in db.execute(
                f'select * from {table} where start>=? and end<=?',(start,end))] if table in tables else []
            record['kinds'][kind] = dict(count=len(rows),device_ms=sum(r['end']-r['start'] for r in rows)/1.e6)
            if kind=='KERNEL':
                for r in rows:
                    name = strings.get(r.get('demangledName',r.get('shortName')),str(r.get('shortName')))
                    resources = {k:r[k] for k in ['registersPerThread','localMemoryPerThread',
                                 'staticSharedMemory','dynamicSharedMemory','gridX','gridY','gridZ',
                                 'blockX','blockY','blockZ'] if k in r}
                    record['kernels'].append(dict(name=name,device_ms=(r['end']-r['start'])/1.e6,resources=resources))
            if kind=='MEMCPY':
                record['copies'] = [dict(copy_kind=r['copyKind'],bytes=r['bytes']) for r in rows]
        record['device_ops'] = sum(r['count'] for r in record['kinds'].values())
        record['device_ms'] = sum(r['device_ms'] for r in record['kinds'].values())
        grouped[f'{mode}:{case}'].append(record)
    report = {}
    for case,records in grouped.items():
        kernels = defaultdict(lambda:dict(count=0,device_ms=0.,resources={}))
        for record in records:
            for k in record['kernels']:
                aggregate = kernels[k['name']]
                aggregate['count'] += 1
                aggregate['device_ms'] += k['device_ms']
                aggregate['resources'] = k['resources']
        top = [dict(name=name,**value,device_ms_per_call=value['device_ms']/len(records))
               for name,value in sorted(kernels.items(),key=lambda item:item[1]['device_ms'],reverse=True)]
        report[case] = dict(calls=len(records),
            device_ops_per_call=sum(r['device_ops'] for r in records)/len(records),
            device_ms_per_call=sum(r['device_ms'] for r in records)/len(records),
            kernels=top,raw_calls=records)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['calls','device_ops_per_call','device_ms_per_call']}
                      for k,v in report.items()},indent=2))
    db.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
