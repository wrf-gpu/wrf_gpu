"""Run the live ALISIOS parser on sealed native registry call expressions.

Only the storage append is intercepted. No production atlas is opened/written.
The expressions come from the hash-verified controller, not copied metadata.
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.deploy / 'tools'))
    from bdyneg_common import ROOT, sha, stamp, seal_read
    import bdyneg_registry as registry_module
    seal_path = args.deploy / 'seal.json'
    cfg = seal_read(seal_path, sha(seal_path))
    tree = ast.parse((args.deploy / 'tools/control.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'start' for t in n.targets))
    end = next(n.args[0] for n in ast.walk(function) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
               and n.func.value.id == 'registry' and n.func.attr == 'end')
    reg = registry_module.Registry(atlas_dir=ROOT / 'registry_schema_private', smoke=True)
    scope = dict(run_id='v45-bdyneg:gpu:private-schema-probe', run=Path('/private/native-output'),
                 smoke=False, stamp=stamp, model_rc=0, post_rc=0, cleaned=True, extracts=[])

    def evaluate(expression):
        return eval(compile(ast.Expression(expression), '<sealed-controller-registry>', 'eval'),
                    {'__builtins__': {}, 'dict': dict, 'str': str}, scope)

    records = []
    with reg._environment(), patch.object(reg.reg.rl, '_append',
            lambda path, event, check: records.append(event)), redirect_stdout(io.StringIO()):
        argv = registry_module.argv_for('start', dict(evaluate(start), started_utc=stamp()))
        assert reg.reg.main(argv) == 0
        assert reg.reg.main(registry_module.argv_for('end', evaluate(end))) == 0
        scope.update(model_rc=1, post_rc=125, cleaned=False)
        assert reg.reg.main(registry_module.argv_for('end', evaluate(end))) == 0
        # Removing the allowed forcing enum must fail validation before append.
        invalid = dict(evaluate(start), forcing='ERA5 v043 E2W300I')
        count = len(records)
        assert reg.reg.main(registry_module.argv_for('start', invalid)) != 0
        assert len(records) == count
    assert len(records) == 3
    assert [r['event'] for r in records] == ['start', 'end', 'end']
    report = dict(schema='v45-bdyneg.native-registry-schema.v1', verdict='PASS',
                  seal_sha256=sha(seal_path), controller_sha256=cfg['tools']['control.py'],
                  register_sha256=sha(registry_module.REGISTER),
                  measured='Native start + success/failure terminal, exact sealed AST expressions',
                  storage_appends_intercepted=len(records), production_writes=0,
                  invalid_forcing_mutant='REFUSED_BEFORE_APPEND',
                  state_transitions='Covered separately by actual private-atlas exact SEQ_SMOKE')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
