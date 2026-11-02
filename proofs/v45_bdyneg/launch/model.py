"""Original v0.4.4 CLI, read-only input mount, same-process backend assertion."""
import argparse
import json
import os
from pathlib import Path
from bdyneg_common import DATA_FLOOR_GIB


def arguments(cfg, run, smoke=False):
    return ['run', '--input-dir', cfg['input_dir'], '--domains-from-namelist',
            '--hours', '3', '--output-dir', str(run / 'output'),
            '--scratch-dir', str(run / 'scratch'),
            '--checkpoint-dir', str(run / 'checkpoints'),
            '--checkpoint-interval-steps', '400',
            '--checkpoint-max-bytes', str(2 * 1024**3),
            '--checkpoint-max-generations', '2',
            '--checkpoint-reserve-bytes', str((20 if smoke else DATA_FLOOR_GIB) * 1024**3)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seal', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=('smoke', 'gpu'), required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.seal).read_text())
    from gpuwrf import cli
    assert Path(cli.__file__).resolve().is_relative_to(Path(cfg['bundle']) / 'src/src')
    args = cli.build_parser().parse_args(arguments(cfg, a.run, a.mode == 'smoke'))
    # Product allocator admission precedes any backend initialization (E250 class).
    cli._maybe_reexec_for_nested_allocator(args)
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.assimilation.letkf.alisios_guard import guarded_session
    import jax
    from bdyneg_common import write
    smoke = a.mode == 'smoke'
    execute = pipeline._execute_nested_pipeline

    def checked(*pos, **kw):
        device = jax.devices()[0]
        assert device.platform == ('cpu' if smoke else 'gpu')
        write(a.run / 'BACKEND.json', dict(platform=device.platform, pid=os.getpid(),
              source_file=cli.__file__, GPU_native_gates='NOT-APPLICABLE' if smoke else 'measured',
              scope='same process as actual CLI forecast'))
        return execute(*pos, **kw)

    pipeline._execute_nested_pipeline = checked
    if smoke:
        original = pipeline._steps_by_domain

        def one_root(hours, dts, names):
            full = original(hours, dts, names)
            bounded = {n: round(dts[names[0]] / dts[n]) for n in names}
            assert hours == 3 and all(full[n] >= bounded[n] > 0 for n in names)
            write(a.run / 'CPU_BOUND.json', dict(requested_hours=hours, full_steps=full,
                  executed_steps=bounded, science='CPU launch/checkpoint smoke only'))
            return bounded

        pipeline._steps_by_domain = one_root
    inputs = {Path(cfg['input_dir']) / n: ['20260111'] for n in cfg['input_sha256']}
    with guarded_session(inputs, ['20260110'], a.run / 'input_guard',
                         purpose='Authorized unchanged C diagnostic, read only; no truth comparator'):
        return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
