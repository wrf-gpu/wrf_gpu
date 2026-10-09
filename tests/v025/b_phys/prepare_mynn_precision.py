"""Real day/night MYNN inputs through the native precision coupling boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/v025'))
from real_state import load_real_snapshot
import jax
from gpuwrf.contracts import state as S
from gpuwrf.coupling import physics_couplers as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    assert jax.devices()[0].platform=='cpu'
    assert os.environ['GPUWRF_MYNN_FP32_COLUMNS']=='1'
    S._gpu_device=lambda:jax.devices('cpu')[0]
    for period in ('night','day'):
        for domain in ('d01','d02'):
            assert not Path('/tmp/wrf_gpu2_quiet').exists()
            original=json.loads((args.root/f'columns_p0_{period}/{domain}.json').read_text())
            history=Path(original['provenance']['wrfout'])
            snapshot=load_real_snapshot(history.parent,domain=domain,wrfout_name=history.name)
            grid=SimpleNamespace(metrics=snapshot.metrics)
            column=C._mynn_column_from_state(snapshot.state,grid)
            ny,nx,_nz=column.theta.shape
            column=C._flatten_columns_to_batch(column,ny,nx)
            flux=C._flatten_columns_to_batch(C._surface_fluxes_from_state(snapshot.state),ny,nx)
            arrays={f'state_{name}':np.asarray(getattr(column,name)) for name in column.__slots__ if getattr(column,name) is not None}
            arrays.update({f'flux_{name}':np.asarray(getattr(flux,name)) for name in flux._fields})
            assert {a.dtype for a in arrays.values()}=={np.dtype('float32')}
            assert all(np.isfinite(a).all() for a in arrays.values())
            output=args.root/f'columns_s2_{period}'
            output.mkdir(exist_ok=True)
            np.savez(output/f'{domain}_whole.npz',**arrays)
            manifest=dict(source_history=str(history),history_sha256=original['provenance']['wrfout_sha256'],
                          scope='real WRF history through native MYNN coupling, input fixture only',
                          source_sha256=hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
                          shape=[ny,nx,_nz],dtype='float32',surface_note='virtual flux reconciled by WRF oracle loader')
            (output/f'{domain}.json').write_text(json.dumps(manifest,indent=2)+chr(10))
            print(period,domain,manifest['shape'],flush=True)


if __name__=='__main__':
    main()
