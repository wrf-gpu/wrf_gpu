"""Extract sanitized KF GPU time, operation and transfer counts from Nsight."""
import argparse
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('database', type=Path)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
conn = sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True)
tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
record = dict(database=str(args.database), kernels=0, graph_kernels=0,
              kernel_device_ms=0.0, memcpy={}, memset=0, device_ops=0, families=[])
if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
    columns = {row[1] for row in conn.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    count, duration = conn.execute('select count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    record.update(kernels=count, kernel_device_ms=(duration or 0) / 1e6)
    if 'graphNodeId' in columns:
        record['graph_kernels'] = conn.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId!=0').fetchone()[0]
    symbol = 'shortName' if 'shortName' in columns else 'demangledName'
    registers = 'max(k.registersPerThread)' if 'registersPerThread' in columns else 'null'
    local = 'max(k.localMemoryPerThread)' if 'localMemoryPerThread' in columns else 'null'
    groups = conn.execute(f'''select s.value,count(*),sum(k.end-k.start)/1e6,
                                     {registers},{local}
                              from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.{symbol}=s.id
                              group by k.{symbol} order by sum(k.end-k.start) desc''')
    record['families'] = [dict(name=name, count=count, device_ms=duration,
                              registers_per_thread=registers, local_bytes_per_thread=local)
                          for name,count,duration,registers,local in groups]
for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET','memset')):
    if table not in tables:
        continue
    if kind == 'memcpy':
        record[kind] = {str(copy_kind): dict(count=count, bytes=total_bytes,
                                           device_ms=duration / 1e6)
                        for copy_kind,count,total_bytes,duration in
                        conn.execute(f'select copyKind,count(*),sum(bytes),sum(end-start) from {table} group by copyKind')}
    else:
        record[kind] = conn.execute(f'select count(*) from {table}').fetchone()[0]
record['device_ops'] = record['kernels'] + record['memset'] + sum(x['count'] for x in record['memcpy'].values())
args.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key,value in record.items() if key!='families'}))
