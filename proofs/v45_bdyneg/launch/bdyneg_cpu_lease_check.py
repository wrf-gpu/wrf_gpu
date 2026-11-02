"""Manager-owned sensitive-core lease must precede GPU locking; stdlib only."""
import datetime,os
from pathlib import Path
from datetime import timezone
now=datetime.datetime.now(timezone.utc)
gate=os.environ['GA_GATE_ID']
assert gate and '/' not in gate
base=Path(os.environ['CPU_LEASE_DIR'])
lease=base/'leases'/(gate+'-cpu.lease')
r=dict(line.split(': ',1) for line in lease.read_text().splitlines() if ': ' in line)
start=datetime.datetime.fromisoformat(r['start-utc'].replace('Z','+00:00'))
end=datetime.datetime.fromisoformat(r['expires-utc'].replace('Z','+00:00'))
window_end=datetime.datetime.fromisoformat(os.environ['GA_END_UTC'].replace('Z','+00:00'))
assert (now-start).total_seconds()>=30 and end>=window_end and now<end
assert set(r['cpus'].split(','))=={'3','19','13','29'}
assert any('EXCLUSIVE' in line and ('leases/'+gate+'-cpu.lease') in line for line in (base/'CORE_TABLE.md').read_text().splitlines())
print('CPU_LEASE_OK',gate,r['cpus'])
