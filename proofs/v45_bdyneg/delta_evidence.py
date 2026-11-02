"""Prove the forecast/CPU-smoke code carried by a host-only deployment delta.

This is source/argv evidence, not a new exact SEQ_SMOKE receipt or GPU result.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def tree(path):
    return ast.parse(path.read_text())


def function(path, name):
    return next(n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name == name)


class CpuBranch(ast.NodeTransformer):
    def visit_If(self, node):
        condition = ast.unparse(node.test)
        # Exactly the two admission/monitor GPU-only branches in control.main.
        if condition == 'not smoke':
            return self.visit(ast.Module(body=node.orelse, type_ignores=[])).body
        if condition.startswith('not smoke and '):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value in ('CPU 16G smoke requires MemAvailable >=40GiB',
                          'CPU 32G smoke requires MemAvailable >=40GiB'):
            return ast.copy_location(ast.Constant('<same failed-admission diagnostic>'), node)
        return node


def main():
    p = argparse.ArgumentParser()
    p.add_argument('old', type=Path)
    p.add_argument('new', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    old, new = args.old / 'tools', args.new / 'tools'
    identical = []
    for name in ('post.py', 'bdyneg_registry.py', 'bdyneg_supervisor.py',
                 'bdyneg_bundle_consumer.py', 'bdyneg_pristine.py',
                 'wave_cubin_gate.py', 'e41/cubin_scan.py', 'e41/hlo_scan.py'):
        assert (old / name).read_bytes() == (new / name).read_bytes(), name
        identical.append(name)
    checks = []
    for file, names in (
        ('model.py', ('main',)),
        ('control.py', ('model_command', 'verify_inputs', 'seed_layer', 'logged_phase')),
        ('bdyneg_common.py', ('seal_read', 'clean_environment', 'remove_own_run',
                              'admit_root_space', 'allocated', 'check_disk', 'sha', 'stamp', 'utc', 'write'))):
        for name in names:
            assert ast.dump(function(old / file, name)) == ast.dump(function(new / file, name)), name
            checks.append(file + ':' + name)
    assert ast.dump(CpuBranch().visit(function(old / 'control.py', 'main'))) == ast.dump(
        CpuBranch().visit(function(new / 'control.py', 'main')))
    commands = []
    for directory in (old, new):
        fn = function(directory / 'model.py', 'arguments')
        namespace = {'DATA_FLOOR_GIB': 430}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<committed-arguments>', 'exec'), namespace)
        commands.append(namespace['arguments'](dict(input_dir='/original/C'), Path('/own/run'), True))
    assert commands[0] == commands[1]
    # Seal values that can change scientific work must remain exact.
    a, b = (json.loads((d / 'seal.json').read_text()) for d in (args.old, args.new))
    keys = ('model_commit', 'model_src_tree', 'bundle', 'input_dir', 'input_sha256',
            'hours', 'checkpoint_interval_steps', 'model_seconds', 'total_seconds',
            'CPU_model_seconds', 'CPU_total_seconds', 'CPU_cache', 'cache_seed', 'precision')
    assert all(a[k] == b[k] for k in keys)
    report = dict(schema='v45-bdyneg.host-delta-carry.v1', verdict='PASS',
                  old_seal_sha256=hashlib.sha256((args.old / 'seal.json').read_bytes()).hexdigest(),
                  new_seal_sha256=hashlib.sha256((args.new / 'seal.json').read_bytes()).hexdigest(),
                  byte_identical_tools=identical, ast_identical_functions=checks,
                  CPU_control_body='AST_IDENTICAL after removing native-only branches and error-label change',
                  CPU_model_argv='EXACT', exact_model_seal_keys=list(keys),
                  limitation='New entry whitelist tested separately; old exact SEQ receipt is carried, '
                             'not relabelled as a new exact call. Native calendar/floor require critic delta.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verdict=report['verdict'], exact_functions=len(checks), byte_tools=len(identical))))


if __name__ == '__main__':
    main()
