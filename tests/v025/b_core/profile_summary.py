"""Read only CUDA activity columns from private nsys SQLite exports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def summarize(path,dispatches):
    conn=sqlite3.connect(f"file:{path}?mode=ro",uri=True)
    tables={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels=defaultdict(lambda:dict(count=0,total_ns=0,min_ns=None,max_ns=0))
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise RuntimeError("missing CUDA kernel activity")
    cols={r[1] for r in conn.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    namecol="demangledName" if "demangledName" in cols else "shortName"
    rows=conn.execute(f"SELECT start,end,{namecol} FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    ids={r[2] for r in rows}
    # Never read TARGET_INFO_* or the whole StringIds table: nsys stores the
    # process environment. Only kernel-name IDs are selected here.
    names={}
    for id in ids:
        value=conn.execute("SELECT value FROM StringIds WHERE id=?",(id,)).fetchone()
        names[id]=value[0] if value else str(id)
    for start,end,id in rows:
        item=kernels[names[id]];dt=end-start
        item["count"]+=1;item["total_ns"]+=dt
        item["min_ns"]=dt if item["min_ns"] is None else min(item["min_ns"],dt)
        item["max_ns"]=max(item["max_ns"],dt)
    copies={}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for kind,count,bytes,total_ns in conn.execute(
            "SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"):
            copies[str(kind)]=dict(count=count,bytes=bytes,total_ns=total_ns)
    result=dict(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dispatches=dispatches,kernels=dict(kernels),copies=copies,
                kernels_per_dispatch=len(rows)/dispatches,
                device_s_per_dispatch=sum(v["total_ns"] for v in kernels.values())/1e9/dispatches,
                copy_kind_enum="CUPTI_ACTIVITY_MEMCPY_KIND: 1 HtoD, 2 DtoH, 8 DtoD")
    conn.close()
    if not rows:raise RuntimeError("zero CUDA kernel events; missing capture")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("database",type=Path)
    p.add_argument("--dispatches",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.database,args.dispatches)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("kernels","source_sha256")}),flush=True)
