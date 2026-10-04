"""Capture actual initializer leaves for OFF byte equality across frozen trees."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--domain',default='d01')
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    sys.path.insert(0,str(args.repo/'src'))
    import gpuwrf
    import jax
    import numpy as np
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    assert jax.devices()[0].platform=='cpu'
    land,static,meta = build_noahmp_land_state(args.case,domain=args.domain)
    paths, _ = jax.tree.flatten_with_path((land,static))
    arrays, records = {}, []
    for i,(path,value) in enumerate(paths):
        arr = np.asarray(value)
        arrays[f'leaf_{i}'] = arr
        records.append(dict(path=str(path),shape=list(arr.shape),dtype=str(arr.dtype),
                            sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez(args.out/'leaves.npz',**arrays)
    (args.out/'receipt.json').write_text(json.dumps(dict(backend='cpu',records=records,metadata=meta),indent=2,default=str)+'\n')
    print('captured',len(records),'actual initializer leaves',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
