"""Host integration controls; never acquire the GPU or touch the production atlas."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bdyneg_common as common
import bdyneg_registry as registry_module
from bdyneg_supervisor import supervise


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load('control.py', 'bdyneg_test_control')
model = load('model.py', 'bdyneg_test_model')


def host_bundle(path):
    """Private host-only checker fixture; no meteorology or GPU success is inferred."""
    (path / 'tools').mkdir(parents=True)
    (path / 'sealed').mkdir()
    (path / 'jax').mkdir()
    (path / 'sealed/BUNDLE_MANIFEST.json').write_text('{}')
    (path / 'tools/compile_bundle.py').write_text(
        'import sys,json\n'
        'if "--out" in sys.argv:\n'
        ' json.dump(dict(new=[],changed=[],missing=[],aliases=[]),open(sys.argv[sys.argv.index("--out")+1],"w"))\n'
        'print("BUNDLE_OK missing=0 changed=0 new=0 aliases=0 moved=False")\n')
    return path


def host_config(bundle):
    return dict(python=sys.executable, bundle=str(bundle),
                bundle_consumer=str(HERE / 'bdyneg_bundle_consumer.py'))


def done():
    return dict(schema='v45-bdyneg.nightly-done.v1', Nightly_DONE=True,
                by='manager', completed_utc='2026-10-28T22:59:00Z')


def test_valid_window():
    common.admit_window('2026-10-28T23:00:00Z', '2026-10-29T05:30:00Z', done(),
                        now=common.utc('2026-10-28T23:00:01Z'))
    common.admit_window('2026-10-31T00:00:00Z', common.HARD_END, done(),
                        now=common.utc('2026-10-31T00:00:01Z'))


@pytest.mark.parametrize('free,smoke,expected', [
    (20 * common.GIB - 1, False, False),
    (20 * common.GIB - 1, True, False),
    (20 * common.GIB + 256 * 1024**2 - 1, False, False),
    (20 * common.GIB + 256 * 1024**2, False, True),
    (22 * common.GIB, False, True),  # real failure class: native has room, CPU smoke does not
    (22 * common.GIB, True, False),
    (23 * common.GIB, True, True),
])
def test_mode_specific_root_admission(free, smoke, expected):
    try:
        common.admit_root_space(free, smoke)
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is expected


@pytest.mark.parametrize('start,end,now,edit', [
    ('2026-10-28T23:01:00Z', '2026-10-29T05:30:00Z', '2026-10-28T23:00:00Z', {}),
    ('2026-10-28T23:00:00Z', '2026-10-28T23:19:59Z', '2026-10-28T23:00:00Z', {}),
    ('2026-10-31T00:00:00Z', '2026-10-31T18:45:01Z', '2026-10-31T00:00:00Z', {}),
    ('2026-10-31T00:00:00Z', '2026-10-31T18:45:00Z', '2026-10-31T18:45:01Z', {}),
    ('2026-10-28T23:00:00Z', '2026-10-29T05:30:00Z', '2026-10-28T23:00:00Z', {'Nightly_DONE': False}),
    ('2026-10-28T23:00:00Z', '2026-10-29T05:30:00Z', '2026-10-28T23:00:00Z', {'by': 'worker'}),
    ('2026-10-28T23:00:00Z', '2026-10-29T05:30:00Z', '2026-10-28T23:00:00Z',
     {'completed_utc': '2026-10-28T23:00:01Z'}),
])
def test_refuses_incomplete_authority(start, end, now, edit):
    with pytest.raises(ValueError):
        common.admit_window(start, end, dict(done(), **edit), now=common.utc(now))


def test_original_model_arguments_and_operational_smoke_delta(tmp_path):
    cfg = dict(input_dir='/original/C')
    gpu = model.arguments(cfg, tmp_path)
    cpu = model.arguments(cfg, tmp_path, True)
    assert gpu[:6] == ['run', '--input-dir', '/original/C', '--domains-from-namelist', '--hours', '3']
    assert gpu[gpu.index('--checkpoint-interval-steps') + 1] == '400'
    assert [i for i, (a, b) in enumerate(zip(gpu, cpu)) if a != b] == [len(gpu) - 1]
    assert gpu[-1] == str(430 * common.GIB) and cpu[-1] == str(20 * common.GIB)


def test_secret_and_model_flag_environment_is_not_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_SECRET', 'fixture')
    monkeypatch.setenv('GPUWRF_TEST_FLAG', '1')
    monkeypatch.setenv('XLA_FLAGS', '--fixture')
    env = common.clean_environment({'bundle': '/sealed/P'}, tmp_path, False)
    assert not {'DEMO_SECRET', 'GPUWRF_TEST_FLAG', 'XLA_FLAGS'} & env.keys()
    assert env['JAX_ENABLE_X64'] == 'true' and env['JAX_PLATFORMS'] == 'cuda'
    cpu = common.clean_environment({'bundle': '/sealed/P'}, tmp_path, True)
    assert cpu['JAX_PLATFORMS'] == 'cpu' and cpu['CUDA_VISIBLE_DEVICES'] == ''
    assert cpu['GPUWRF_JAX_CACHE_DIR'] == str(tmp_path / 'cpu_cache')


def test_entry_whitelist_precedes_any_python_process():
    prefix = (HERE / 'entry.sh.in').read_text().split('  systemd-run --user', 1)[0]
    # Exercise the actual emitted shell prefix; the private parent contains
    # only explicit fixtures. Never print or inherit the agent environment.
    program = ("import os; assert 'DEMO_SECRET' not in os.environ; "
               "assert os.environ['JAX_PLATFORMS']=='cpu'; "
               "assert os.environ['GA_GATE_ID']=='private-fixture'")
    import shlex
    result = subprocess.run(['/bin/bash', '-c', prefix + shlex.quote(sys.executable)
                             + ' -c ' + shlex.quote(program)],
                            env=dict(HOME='<USER_HOME>', PATH='/usr/bin:/bin', LANG='C.UTF-8',
                                     LC_ALL='C.UTF-8', JAX_PLATFORMS='cpu',
                                     DEMO_SECRET='private-fixture', GA_GATE_ID='private-fixture'),
                            capture_output=True, timeout=10)
    assert result.returncode == 0


def test_cleanup_refuses_wrong_ownership(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    common.write(run / 'OWNERSHIP.json', dict(lane='another-lane', seal_sha256='a'))
    with pytest.raises(ValueError):
        common.remove_own_run(run, run, 'a')
    assert run.is_dir()


def test_cleanup_restores_only_owned_overlay_mode000_and_keeps_marker_on_error(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    work = run / 'layer/work/work'
    work.mkdir(parents=True)
    (work / 'own-file').write_bytes(b'own')
    common.write(run / 'OWNERSHIP.json', dict(lane='v45-bdyneg', seal_sha256='fixture'))
    work.chmod(0)
    original = common.shutil.rmtree
    with monkeypatch.context() as m:
        m.setattr(common.shutil, 'rmtree', lambda _p: (_ for _ in ()).throw(OSError('fixture failure')))
        with pytest.raises(OSError):
            common.remove_own_run(run, run, 'fixture')
        assert (run / 'OWNERSHIP.json').is_file()
    assert work.stat().st_mode & 0o700 == 0o700
    assert common.shutil.rmtree is original
    common.remove_own_run(run, run, 'fixture')
    assert not run.exists()


def test_live_registry_schema_and_two_terminal_events_use_private_atlas(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, 'PRODUCTION_ATLAS', tmp_path / 'fake_production')
    atlas = tmp_path / 'private_atlas'
    registry = registry_module.Registry(atlas_dir=atlas, smoke=True)
    for suffix, rc in (('ok', 0), ('failed', 3)):
        run_id = 'v45-bdyneg:test:' + suffix
        registry.start(dict(run_id=run_id, by='v45-bdyneg', producer='wrf_gpu2',
            purpose='CPU_SEQ_SMOKE control', family='research', forcing='OTHER',
            engine='wrf_gpu', nests=3, lead_h=3, issue='20260110_18z',
            run_dir=str(tmp_path / suffix), retention='raw_delete_after_extract'))
        registry.end(dict(run_id=run_id, by='v45-bdyneg', rc=rc,
            status='DONE_CLEAN' if rc == 0 else 'FAILED', corpus='held', raw_deleted=True))
    rows = registry._events()
    assert len(rows) == 4
    assert [r['event'] for r in rows] == ['start', 'end', 'start', 'end']


@pytest.mark.parametrize('code,cap,expected', [('pass', 5., 0), ('fail', 5., 7), ('timeout', .15, 124)])
def test_supervised_lifecycle_reaps_success_failure_timeout(tmp_path, code, cap, expected):
    script = {'pass': 'pass', 'fail': 'raise SystemExit(7)', 'timeout': 'import time;time.sleep(30)'}[code]
    receipt = tmp_path / (code + '.json')
    rc = supervise([sys.executable, '-c', script], seconds=cap, receipt=receipt,
                   group_file=tmp_path / 'group.json', lock_file=tmp_path / 'private.lock',
                   term_grace=.2, kill_grace=1.)
    assert rc == expected
    result = json.loads(receipt.read_text())
    assert result['cleanup']['all_children_reaped']
    assert result['cleanup']['owned_lock_fds_after'] == []


def test_read_only_mount_prevents_actual_child_write(tmp_path, monkeypatch):
    """Semantic mount check, not a lexical argv assertion; all files are private fixtures."""
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    data = inputs / 'fixture'
    data.write_bytes(b'original')
    deploy = tmp_path / 'deploy'
    (deploy / 'tools').mkdir(parents=True)
    (deploy / 'tools/model.py').write_text(
        'from pathlib import Path\n'
        'try:\n'
        f' Path({str(data)!r}).write_bytes(b"changed")\n'
        'except OSError as exc:\n'
        ' assert exc.errno == 30, exc\n'
        ' raise SystemExit(0)\n'
        'raise SystemExit(42)\n')
    run = tmp_path / 'run'
    run.mkdir()
    bundle = host_bundle(tmp_path / 'bundle')
    monkeypatch.setattr(control, 'READ_ONLY_ROOT', inputs)
    command = control.model_command(host_config(bundle),
                                    deploy, run, tmp_path / 'seal.json', True)
    result = subprocess.run(command, env={'PATH': os.environ['PATH'], 'JAX_PLATFORMS': 'cpu'},
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
    assert data.read_bytes() == b'original'


def test_mount_chain_preserves_an_explicit_fd_and_refuses_a_closed_one(tmp_path, monkeypatch):
    """Host-only witness for the extra bwrap boundary used inside the official GPU wrapper."""
    inputs, bundle, run, deploy = (tmp_path / n for n in ('inputs', 'bundle', 'run', 'deploy'))
    for p in (inputs, run, deploy / 'tools'):
        p.mkdir(parents=True)
    host_bundle(bundle)
    lock = inputs / 'private-lock'
    lock.write_bytes(b'fd fixture')
    monkeypatch.setattr(control, 'READ_ONLY_ROOT', inputs)
    with lock.open('rb') as stream:
        expected = os.fstat(stream.fileno())
        (deploy / 'tools/model.py').write_text(
            'import os\n'
            f's = os.fstat({stream.fileno()})\n'
            f'assert (s.st_dev,s.st_ino) == ({expected.st_dev},{expected.st_ino})\n')
        command = control.model_command(host_config(bundle),
                                        deploy, run, tmp_path / 'seal.json', True)
        env = {'PATH': os.environ['PATH'], 'JAX_PLATFORMS': 'cpu'}
        positive = subprocess.run(command, env=env, pass_fds=(stream.fileno(),),
                                  capture_output=True, timeout=10)
        assert positive.returncode == 0, positive.stderr.decode()
        negative = subprocess.run(command, env=env, capture_output=True, timeout=10)
        assert negative.returncode != 0


def test_real_consumer_single_namespace_and_token_pass_through_rejects_old_nested_case(tmp_path):
    """Run the real new consumer and actual bwrap; reproduce the old GPU-only seam on CPU."""
    bundle = host_bundle(tmp_path / 'bundle')
    readonly = tmp_path / 'inputs'
    readonly.mkdir()
    layer = tmp_path / 'layer'
    consumer = [sys.executable, str(HERE / 'bdyneg_bundle_consumer.py'),
                '--bundle', str(bundle), '--layer', str(layer), '--grace', '.1',
                '--readonly-dir', str(readonly), '--cwd', str(tmp_path), '--']
    env = dict(PATH=os.environ['PATH'], JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
               GPUWRF_GPU_LOCK_TOKEN='host-fixture-not-a-live-token')
    payload = [sys.executable, '-c',
               'import os;assert os.environ["GPUWRF_GPU_LOCK_TOKEN"]=="host-fixture-not-a-live-token"']
    ok = subprocess.run(consumer + payload, env=env, capture_output=True, timeout=15)
    assert ok.returncode == 0, ok.stderr.decode()
    old_nested = subprocess.run(consumer + ['bwrap', '--dev-bind', '/', '/', '--', *payload],
                                env=env, capture_output=True, timeout=15)
    assert old_nested.returncode != 0
    assert b'No permissions to create a new namespace' in old_nested.stderr
