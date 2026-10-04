"""Fresh CPU process: change the live env AFTER importing radiation."""
import json
import os
from pathlib import Path
import runpy
import sys

flag, imported, live, output = sys.argv[1:]
os.environ[flag] = imported
import gpuwrf
from gpuwrf.physics import rrtmg_sw, rrtmg_lw
from gpuwrf.integration.init_kernels import radiation_init_key, resolved_radiation_constants
from gpuwrf.runtime.aot_executable import hlo_sha256_from_lowered
import jax
assert jax.devices()[0].platform == 'cpu'
os.environ[flag] = live
helpers = runpy.run_path(str(Path(__file__).parents[1]/'test_init_kernel_cache.py'))
records = []
for fn,state in helpers['_native_columns']():
    records.append(dict(function=fn.__name__,key=radiation_init_key(fn,(state,),{}),
                        hlo=hlo_sha256_from_lowered(fn.lower(state))))
Path(output).write_text(json.dumps(dict(flag=flag,imported=imported,live=live,
    resolved=resolved_radiation_constants(),records=records),indent=2))
