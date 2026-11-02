"""v46-topo TS03: CPU SM120 compile of the native topographic-shadow Pallas kernel per real grid shape.

E137 recipe (proofs/t_ntucompile/cubin_probe.py): make_jaxpr of the production entry with the CUDA
facade forced, lower the pallas_call to TTIR, triton.compile for sm_120, scan the cubin
(scripts/e41/cubin_scan.py). No CUDA device. Component evidence, not an actual GPU/E41 result.
"""
import hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
src = Path(sys.argv[1]); out = Path(sys.argv[2])
assert os.environ['JAX_PLATFORMS'] == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0, str(src / 'src'))
import gpuwrf, jax, jax.numpy as jnp
assert Path(gpuwrf.__file__).resolve().is_relative_to(src.resolve())
from jax._src.pallas.triton import lowering
import gpuwrf.backends.pallas as facade
facade.cuda_native = lambda: True
from gpuwrf.kernels import topographic_shadow as ts
import triton
from triton.backends.compiler import GPUTarget
spec = importlib.util.spec_from_file_location('cscan', src / 'scripts/e41/cubin_scan.py')
cscan = importlib.util.module_from_spec(spec); spec.loader.exec_module(cscan)
SHAPES = [('mini24_d01', 24, 24, 9000., False), ('mini24_d02', 24, 24, 3000., True), ('mini24_d03', 24, 24, 1000., True),
          ('PROD_d01', 69, 119, 9000., False), ('PROD_d02', 116, 266, 3000., True), ('WN3_d03', 93, 111, 1000., True)]
out.mkdir(parents=True, exist_ok=False)
rep = dict(source=subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip(),
           scope='CPU isolated native kernel compile (sm_120); not actual GPU/E41', shapes={})
for name, ny, nx, dx, nested in SHAPES:
    z = jax.ShapeDtypeStruct((ny, nx), jnp.float32)
    fn = lambda t, o, c, a: ts.topographic_shadow_native(t, o, c, a, dx_m=dx, dy_m=dx, shadow_length_m=25000.,
                                                         nested=nested, interpret=False)
    eqn = next(e for e in jax.make_jaxpr(fn)(z, z, z, z).jaxpr.eqns if e.primitive.name == 'pallas_call')
    module = lowering.lower_jaxpr_to_triton_module(eqn.params['jaxpr'], eqn.params['grid_mapping'], 'cuda', 120)
    ttir = out / f'{name}.ttir'; ttir.write_text(str(module.module))
    t = time.monotonic()
    k = triton.compile(str(ttir), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'enable_fp_fusion': True})
    cubin = out / f'{name}.cubin'; cubin.write_bytes(k.asm['cubin'])
    res = cscan.scan(cubin, '/usr/local/cuda/bin/cuobjdump')
    rep['shapes'][name] = dict(ny=ny, nx=nx, dx=dx, nested=nested, steps=int(25000. / dx + 1),
        compile_s=round(time.monotonic() - t, 2), n_regs=getattr(k, 'n_regs', None), n_spills=getattr(k, 'n_spills', None),
        scf_while=str(module.module).count('scf.while'), atan_tokens=str(module.module).count('atan'),
        cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(), native=res)
    print(name, json.dumps(rep['shapes'][name], default=str)[:400], flush=True)
(out / 'receipt.json').write_text(json.dumps(rep, indent=2, default=str) + '\n')
os._exit(0)
