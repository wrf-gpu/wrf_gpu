"""Real PROD full own-step scan: native RK must preserve every carry aval.

CPU abstract tracing only; no executable compilation or numerical GPU claim.
The negative control removes only the seven-leaf adapter and must break the
actual full-root scan in both domains. Physics/kernel arithmetic stays intact.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ('t_2ave', 'ww', 'u_save', 'v_save', 'w_save', 't_save', 'ww_save')
FLAGS = ('GPUWRF_DYN_FP32', 'GPUWRF_DYN_RK_FP32', 'GPUWRF_DYN_DIFFUSION_FP32',
         'GPUWRF_DYN_ADVECTION_FP32', 'GPUWRF_NOAHMP_NATIVE_REAL', 'GPUWRF_NOAHMP_ITERATION_BARRIER',
         'GPUWRF_MCICA_JUMPAHEAD', 'GPUWRF_RRTMG_LW_FUSED_TRANSFER',
         'GPUWRF_RRTMG_SW_FUSED_QUADRATURE', 'GPUWRF_MYNN_FP32_COLUMNS',
         'GPUWRF_MYNN_CLOUDMIX', 'GPUWRF_EDMF_FUSED_PLUME', 'GPUWRF_MYNN_FP32_PLUME',
         'GPUWRF_KF_RESIDENT_TABLES', 'GPUWRF_KF_COLUMN_FP32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--without-mutant', action='store_true')
    parser.add_argument('--boundary-flag', choices=('0', '1'), default='0')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source/'src'))
    sys.path.insert(0, str(args.source/'tests/v025/b_core'))
    os.environ.update({name: '1' for name in FLAGS})
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', GPUWRF_BOUNDARY_FP32=args.boundary_flag,
                      GPUWRF_CENSUS='0', GPUWRF_JAX_CACHE='0', GPUWRF_MCICA_LEGACY_FP64='0',
                      JAX_PALLAS_USE_MOSAIC_GPU='false')
    import jax
    import jax.numpy as jnp
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.domain_tree import DomainTree
    from gpuwrf.physics.noahmp.precision import real_tree
    from prod_inputs import prod_domains
    assert jax.devices()[0].platform == 'cpu'
    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    original = op._physics_boundary_step
    source_path = Path(op.__file__)
    parsed = ast.parse(source_path.read_text())
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == '_physics_boundary_step')
    glue = [node for node in function.body if isinstance(node, ast.If)
            and 'GPUWRF_DYN_RK_FP32' in ast.unparse(node.test)]
    if not args.without_mutant:
        assert len(glue) == 1, 'test requires the reviewed native-RK carry adapter'
        assert all(name in ast.unparse(glue[0]) for name in SCRATCH)
        mutant = copy.deepcopy(function)
        mutant.body = [node for node in mutant.body if not (isinstance(node, ast.If)
                       and 'GPUWRF_DYN_RK_FP32' in ast.unparse(node.test))]
        module = ast.fix_missing_locations(ast.Module(body=[mutant], type_ignores=[]))
        namespace = dict(op.__dict__)
        exec(compile(module, str(source_path), 'exec'), namespace)
        deleted = namespace['_physics_boundary_step']
    rows = []
    for domain in ('d01', 'd02'):
        carry = carries[domain].replace(noahmp_land=real_tree(carries[domain].noahmp_land))
        nml = tree.domains[domain].namelist
        clock = op.build_clock_base(nml)
        shapes = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype,
                              weak_type=getattr(value, 'weak_type', False)), carry)
        # Same public full own-step fori scan used by the forecast root.
        fn = lambda value: op._advance_chunk_fori(value, nml, jnp.asarray(1, jnp.int32), clock,
                    n_steps=1, cadence=int(nml.radiation_cadence_steps))
        output = jax.eval_shape(fn, shapes)
        before = {jax.tree_util.keystr(p): v for p, v in jax.tree.flatten_with_path(shapes)[0]}
        after = {jax.tree_util.keystr(p): v for p, v in jax.tree.flatten_with_path(output)[0]}
        assert before.keys() == after.keys()
        differences = [dict(path=p, before=str(before[p]), after=str(v)) for p, v in after.items()
                       if (before[p].shape, before[p].dtype, before[p].weak_type) != (v.shape, v.dtype, v.weak_type)]
        assert not differences, differences
        row = dict(domain=domain, carry_leaves=len(before), all_avals_exact=True,
                   scratch={n: dict(shape=list(getattr(output, n).shape), dtype=str(getattr(output, n).dtype)) for n in SCRATCH})
        if not args.without_mutant:
            op._physics_boundary_step = deleted
            jax.clear_caches()  # this mutation changes a captured global caller
            try:
                jax.eval_shape(fn, shapes)
            except TypeError as error:
                message = str(error)
                assert 'while_loop body function carry' in message, message
                assert message.count('corresponding output carry component has type float32[') == len(SCRATCH), message
                row.update(deletion_rejected=True, deletion_error=message)
            else:
                raise AssertionError('deleting native RK carry glue must fail the full root scan')
            finally:
                op._physics_boundary_step = original
                jax.clear_caches()
        rows.append(row)
        print(domain, 'carry avals exact', len(before), 'deletion rejected', row.get('deletion_rejected'), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(platform='cpu', mode='full-root abstract trace, no compilation/execution',
        source=str(args.source), runtime_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        boundary_fp32=args.boundary_flag,
        resolved_flags={name: value for name, value in os.environ.items()
                        if name.startswith('GPUWRF_')}, rows=rows, passed=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
