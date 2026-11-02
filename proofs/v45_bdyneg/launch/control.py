"""Sealed, one-shot native C replay -> lock release -> verified CPU extraction.

Model and post phases have separate supervised process groups. This controller
never imports JAX or owns a CUDA pool. Only the exact sequencer enables smoke.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import uuid

from bdyneg_common import (DATA_RUN, DATA_FLOOR_GIB, GIB, ROOT, admit_window, admit_root_space, allocated, check_disk,
                    clean_environment, remove_own_run, seal_read, sha, stamp, utc, write)
from bdyneg_supervisor import supervise
from bdyneg_registry import Registry

READ_ONLY_ROOT = Path('<DATA_ROOT>/alisios')


def model_command(cfg, deploy, run, seal, smoke):
    command = [cfg['python'], str(deploy / 'tools/model.py'), '--seal', str(seal),
               '--run', str(run), '--mode', 'smoke' if smoke else 'gpu']
    # The bundle consumer is the sole namespace owner in BOTH modes. Read-only
    # inputs and cwd are mounts/options of that same bwrap, never a nested bwrap.
    command = [cfg['python'], cfg['bundle_consumer'], '--bundle', cfg['bundle'],
               '--layer', str(run / 'layer'), '--grace', '10',
               '--readonly-dir', str(READ_ONLY_ROOT), '--cwd', str(run),
               *(['--readonly-dir', '<DATA_ROOT>', '--readonly-dir', cfg['bundle']] if smoke else []),
               '--', *command]
    if smoke:
        return ['flock', str(run / 'private.lock'), *command]
    return [cfg['gpu_lock'], '--label', 'v45-bdyneg', '--timeout', '15', '--', *command]


def verify_inputs(cfg, receipt):
    import sys
    sys.path.insert(0, cfg['bundle'] + '/src/src')
    from gpuwrf.assimilation.letkf.alisios_guard import guarded_session
    paths = {Path(cfg['input_dir']) / n: ['20260111'] for n in cfg['input_sha256']}
    with guarded_session(paths, ['20260110'], receipt,
                         purpose='C diagnostic input SHA admission, read only; no copies'):
        for path in paths:
            if path.is_symlink() or sha(path) != cfg['input_sha256'][path.name]:
                raise ValueError('C input changed: ' + path.name)


def seed_layer(cfg, run):
    """Copy only sealed, authorized paid-cache entries; private aliases may share inodes."""
    entries = cfg.get('cache_seed', [])
    if not entries:
        return
    from gpuwrf.assimilation.letkf.alisios_guard import guarded_session
    seen = {}
    with guarded_session({Path(i['source']): ['20260111'] for i in entries}, ['20260110'],
                         run / 'cache_read_guard', purpose='Authorized paid C executable seed; no source writes'):
        for item in entries:
            source = Path(item['source'])
            if sha(source) != item['sha256']:
                raise ValueError('C cache input changed')
            relative = Path(item['relative'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('jax', 'autotune', 'triton', 'cuda'):
                raise ValueError('cache seed path escapes private layer')
            target = run / 'layer/upper' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = seen.get(item['sha256'])
            if previous is None:
                shutil.copyfile(source, target)  # NEVER a hard link to ALISIOS or the sealed bundle
                seen[item['sha256']] = target
            else:
                os.link(previous, target)  # aliases link only our own copied file


def logged_phase(command, *, run, receipt, phase, seconds, lock_file, env):
    """Bound and reap a whole phase, including any child that creates a new session."""
    import contextlib
    previous_env = dict(os.environ)
    os.environ.clear(); os.environ.update(env)
    try:
        with (receipt / (phase + '.log')).open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                saved_out, saved_err = os.dup(1), os.dup(2)
                try:
                    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
                    return supervise(command, seconds=seconds,
                        receipt=receipt / (phase.upper() + '_LIFECYCLE.json'),
                        group_file=run / (phase.upper() + '_GROUP.json'), lock_file=lock_file,
                        term_grace=10, kill_grace=5)
                finally:
                    os.dup2(saved_out, 1); os.dup2(saved_err, 2)
                    os.close(saved_out); os.close(saved_err)
    finally:
        os.environ.clear(); os.environ.update(previous_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--seal-sha', required=True)
    args = parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1'
    cfg = seal_read(args.seal, args.seal_sha)
    deploy = args.seal.parent
    smoke = os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1'
    model_seconds = cfg['CPU_model_seconds'] if smoke else cfg['model_seconds']
    total_seconds = cfg['CPU_total_seconds'] if smoke else cfg['total_seconds']
    if smoke and os.environ.get('SEQ_SMOKE') != '1':
        raise ValueError('only exact night_seq SEQ_SMOKE may select CPU mode')
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 3
    if not smoke:
        done = json.loads((deploy / 'NIGHTLY_DONE.json').read_text())
        now = float(subprocess.check_output([cfg['real_utc']], text=True))
        admit_window(os.environ['GA_START_UTC'], os.environ['GA_END_UTC'], done, now=now)
        if os.environ.get('GA_GATE_ID') != 'v45-bdyneg-night':
            raise ValueError('manager gate identity required')
        lease = Path('/tmp/wrf_gpu2_run_approvals/v45-bdyneg.lease').read_text()
        expires = next(l.split(':', 1)[1].strip() for l in lease.splitlines() if l.startswith('expires-utc:'))
        if utc(expires) < now + 1200:
            raise ValueError('lane GPU lease must cover the entire entry')
        if sha(cfg['gpu_lock']) != subprocess.check_output(
                ['git', '-C', '<USER_HOME>/src/wrf_gpu2', 'show', 'main:scripts/with_gpu_lock.sha256'],
                text=True).split()[0]:
            raise ValueError('official wrapper differs from main pin')
        # Only minimal GPU-arm cores may have an exclusive manager CPU lease.
        if not os.environ.get('CPU_LEASE_DIR'):
            raise ValueError('manager CPU lease directory required')
        if Path(os.environ['CPU_LEASE_DIR']).resolve() != Path('<USER_HOME>/wrf_gpu2_lanes/manager'):
            raise ValueError('CPU lease must come from the manager directory')
        subprocess.run([cfg['python'], str(deploy / 'tools/bdyneg_cpu_lease_check.py')], check=True)
        if shutil.disk_usage('<DATA_ROOT>').free < (DATA_FLOOR_GIB + 5) * GIB:
            raise ValueError(f'need floor{DATA_FLOOR_GIB}GiB plus authorized5GiB before DATA mkdir')
        run = DATA_RUN
    else:
        available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:'))
        if available < 40 * GIB:
            raise ValueError('CPU 32G smoke requires MemAvailable >=40GiB')
        run = ROOT / 'smoke' / ('run_' + stamp().replace(':', '') + '_' + uuid.uuid4().hex[:6])
    if run.resolve() != run:
        raise ValueError('work path must not traverse symlinks')
    admit_root_space(shutil.disk_usage(ROOT).free, smoke)
    # Inputs are hash-checked before work directories or registry START are created.
    admission = ROOT / 'admissions' / (stamp().replace(':', '') + '_' + uuid.uuid4().hex[:6])
    verify_inputs(cfg, admission)
    if not smoke:
        # A sealed v1 GPU entry is one-shot; no implicit retries after a terminal attempt.
        with (deploy / 'GPU_ATTEMPT.json').open('x') as f:
            json.dump(dict(utc=stamp(), seal_sha256=args.seal_sha), f)
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'OWNERSHIP.json', dict(lane='v45-bdyneg', seal_sha256=args.seal_sha))
    (run / 'tmp').mkdir()
    receipt = ROOT / 'receipts' / run.name
    receipt.mkdir(parents=True, exist_ok=False)
    limit = 3 * GIB if smoke else cfg['max_bytes']
    cap = {'peak_sampled_named_bytes': 0, 'accounting': 'allocated named inodes, sampled; not unlinked-file attribution'}
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(SystemExit(143)))
    signal.signal(signal.SIGINT, lambda _s, _f: (_ for _ in ()).throw(SystemExit(130)))

    def monitor():
        while not stopping.wait(.5):
            try:
                used = allocated(run) + allocated(receipt)
                if smoke and Path(cfg['CPU_cache']).exists():
                    used += allocated(cfg['CPU_cache'])
                if used > limit:
                    raise RuntimeError('own diagnostic exceeded disk cap')
                cap['peak_sampled_named_bytes'] = max(cap['peak_sampled_named_bytes'], used)
                if not smoke and shutil.disk_usage('<DATA_ROOT>').free < DATA_FLOOR_GIB * GIB:
                    raise RuntimeError('actual global DATA floor crossed')
            except RuntimeError as e:
                cap['refusal'] = str(e)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            except OSError as e:
                cap['accounting_uncertainty'] = type(e).__name__  # diagnostic uncertainty is not an abort

    run_id = 'v45-bdyneg:' + ('smoke:' if smoke else 'gpu:') + run.name + ':' + uuid.uuid4().hex[:8]
    start = dict(run_id=run_id, by='v45-bdyneg', producer='wrf_gpu2',
                 purpose='CPU_SEQ_SMOKE first root' if smoke else 'C boundary same-operand checkpoint diagnostic',
                 family='research', tags=['v45-bdyneg', 'no-truth-scoring'], forcing='OTHER',
                 engine='wrf_gpu', nests=3, lead_h=3, issue='20260110_18z',
                 run_dir=str(run / 'output'), retention='raw_delete_after_extract')
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    model_rc, post_rc, cleaned, registered, safe_to_clean = 125, 125, False, False, True
    began = time.monotonic()
    try:
        registry = Registry(atlas_dir=ROOT / 'smoke/private_atlas' if smoke else None, smoke=smoke)
        registry.start(dict(start, started_utc=stamp()))
        registered = True
        try:
            try:
                if not smoke:
                    seed_layer(cfg, run)
                env = clean_environment(cfg, run, smoke)
                env['ALISIOS_WRF_RUN_ID'] = run_id
                command = model_command(cfg, deploy, run, args.seal, smoke)
                write(receipt / 'INVOCATION.json', dict(mode='smoke' if smoke else 'gpu', argv=command,
                      run_id=run_id, input_dir=cfg['input_dir'], physics='unchanged exact C namelist',
                      model_commit=cfg['model_commit'], total_cap_s=total_seconds, model_cap_s=model_seconds,
                      actual_checkpoint_hours=3 if not smoke else 'one root CPU smoke',
                      environment={k: env[k] for k in ('JAX_PLATFORMS', 'JAX_ENABLE_X64', 'OMP_NUM_THREADS')}))
                # Supervisor installs an owned phase session and reaps escaped descendants.
                safe_to_clean = False
                model_rc = logged_phase(command, run=run, receipt=receipt, phase='model', seconds=model_seconds,
                    lock_file=Path('/tmp/wrf_gpu2_gpu.lock') if not smoke else run / 'private.lock', env=env)
                lifecycle = json.loads((receipt / 'MODEL_LIFECYCLE.json').read_text())
                safe_to_clean = lifecycle.get('cleanup', {}).get('all_children_reaped', False)
                if not safe_to_clean:
                    raise RuntimeError('owned model processes remain; cleanup deferred')
                if model_rc == 0:
                    env['JAX_PLATFORMS'] = 'cpu'
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    post = [cfg['python'], str(deploy / 'tools/post.py'), '--seal', str(args.seal),
                            '--run', str(run), '--receipt', str(receipt), '--mode', 'smoke' if smoke else 'gpu']
                    # Model/CUDA process has ended and official lock was released before this CPU phase.
                    remaining = min(300, max(1, total_seconds - 30 - (time.monotonic() - began)))
                    safe_to_clean = False
                    post_rc = logged_phase(post, run=run, receipt=receipt, phase='post', seconds=remaining,
                                           lock_file=run / 'private_post.lock', env=env)
                    lifecycle = json.loads((receipt / 'POST_LIFECYCLE.json').read_text())
                    safe_to_clean = lifecycle.get('cleanup', {}).get('all_children_reaped', False)
                    if not safe_to_clean:
                        raise RuntimeError('owned CPU post processes remain; cleanup deferred')
            finally:
                stopping.set(); thread.join(timeout=2)
                for name in ('BACKEND.json', 'CPU_BOUND.json'):
                    if (run / name).is_file():
                        shutil.copyfile(run / name, receipt / name)
                cap['last_named_bytes'] = allocated(run)
                # A successfully written checkpoint is scientific evidence. If extraction
                # failed before its complete readback receipt, retain it for the next
                # CPU-only consumer; never delete the only copy on a post-tool failure.
                compact_verified = (receipt / 'POST.json').is_file()
                hold_checkpoint = model_rc == 0 and not compact_verified
                if hold_checkpoint:
                    cap['held_for_verified_extraction'] = str(run / 'checkpoints')
                if safe_to_clean and not hold_checkpoint:
                    remove_own_run(run, run, args.seal_sha)
                    cleaned = True
                if smoke and model_rc == post_rc == 0:
                    # This paid CPU cache has no dependent arm after successful exact smoke/readback.
                    if Path(cfg['CPU_cache']).resolve() != ROOT / 'smoke_cache_C_v044':
                        raise ValueError('unexpected CPU cache cleanup path')
                    shutil.rmtree(cfg['CPU_cache'], ignore_errors=False)
        finally:
            if registered:
                extracts = [str(p) + '::' + sha(p) for p in receipt.glob('*actual_boundary_operands.npz')]
                registry.end(dict(run_id=run_id, by='v45-bdyneg', rc=model_rc or post_rc or (0 if cleaned else 125),
                    status='DONE_CLEAN' if model_rc == post_rc == 0 and cleaned else 'FAILED',
                    corpus='held', raw_deleted=cleaned, ended_utc=stamp(), extract=extracts,
                    notes='diagnostic acquisition only; scientific gate remains open; actual cleanup=' + str(cleaned)))
    finally:
        stopping.set(); thread.join(timeout=2)
        hold_checkpoint = model_rc == 0 and not (receipt / 'POST.json').is_file()
        if run.exists() and not cleaned and safe_to_clean and not hold_checkpoint:
            remove_own_run(run, run, args.seal_sha)
            cleaned = True
        write(receipt / 'RESULT.json', dict(model_rc=model_rc, post_rc=post_rc, raw_deleted=cleaned,
              disk=cap, wall_s=time.monotonic() - began, mode='smoke' if smoke else 'gpu',
              GPU_native_gates='NOT-APPLICABLE' if smoke else 'see actual BACKEND and POST',
              verdict='ACQUISITION_OK' if model_rc == post_rc == 0 else 'ACQUISITION_FAILED',
              scientific_gate='OPEN until actual pristine comparisons and root PD disposition'))
    print(json.dumps(dict(receipt=str(receipt), model_rc=model_rc, post_rc=post_rc, cleaned=cleaned)))
    return model_rc or post_rc


if __name__ == '__main__':
    raise SystemExit(main())
