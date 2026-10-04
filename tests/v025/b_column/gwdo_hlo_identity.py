"""Lower coupling.physics_couplers.gwdo_adapter on CPU and hash raw HLO text + module proto.

Run once per commit from the SAME checkout path (E58), fresh process, flags as
given by the environment. With GPUWRF_GWDO_NATIVE_REAL unset the candidate must
reproduce the base hashes exactly (text with metadata AND serialized proto,
which carries the stack-frame index). Also counts f64 ops of the native path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'quiet'
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

from gwdo_gpu_probe import FakeState, build  # noqa: E402

F64OP = re.compile(r'^\s*(?:ROOT )?%?[\w.\-]+ = (\S*f64\S*) ([\w\-]+)\(', re.M)


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    from gpuwrf.coupling.physics_couplers import gwdo_adapter
    rec = dict(flags={k: v for k, v in os.environ.items() if k.startswith('GPUWRF_')})
    for dtype in ('f64', 'f32'):
        arrays, statics, grid = build('night', jnp.float64 if dtype == 'f64' else jnp.float32)
        lowered = jax.jit(lambda a: gwdo_adapter(FakeState(**a), 54.0, statics, grid).u).lower(arrays)
        hlo = lowered.compiler_ir('hlo')
        text = hlo.as_hlo_text()
        proto = hlo.as_serialized_hlo_module_proto()
        ops = {}
        for m in F64OP.finditer(text):
            ops[m.group(2)] = ops.get(m.group(2), 0) + 1
        rec[dtype] = dict(text=hashlib.sha256(text.encode()).hexdigest(),
                          proto=hashlib.sha256(proto).hexdigest(), f64_ops=ops,
                          has_metadata='source_line' in text)
    out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
