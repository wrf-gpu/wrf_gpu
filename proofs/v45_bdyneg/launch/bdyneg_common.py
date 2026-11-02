"""Host-only admission and environment for the sealed C-arm diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

GIB = 1024**3
DATA_RUN = Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-bdyneg/diagnostic_20261031_v6')
# Proposed next-window ceiling; actual manager GA and Nightly-DONE remain required.
HARD_END = '2026-10-31T18:45:00Z'
DATA_FLOOR_GIB = 430
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v45-bdyneg')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def utc(text):
    result = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timezone required')
    return result.timestamp()


def admit_root_space(free_bytes, smoke):
    """Preserve the20GiB floor; only CPU smoke stages its large cache on ROOT.

    Native raw/cache/checkpoint/E41 temporary files live on DATA. Its ROOT
    products are compact operand NPZs and receipts; reserve256MiB for this
    frozen case. Actual retained-array sizes are recorded in the regression receipt.
    """
    reserve = 3 * GIB if smoke else 256 * 1024**2
    required = 20 * GIB + reserve
    if free_bytes < required:
        raise ValueError(f'ROOT requires20GiB floor +{reserve}B mode-specific reserve; free={free_bytes}B')
    return required


def admit_window(start, end, done, *, now):
    start, end = utc(start), utc(end)
    if not start <= now < end or end - now < 1200 or end > utc(HARD_END):
        raise ValueError('window must start now, cover 20 min and end by ' + HARD_END)
    if (done.get('schema') != 'v45-bdyneg.nightly-done.v1'
            or done.get('Nightly_DONE') is not True or done.get('by') != 'manager'
            or utc(done['completed_utc']) > start):
        raise ValueError('manager Nightly DONE receipt required before GA_START')


def seal_read(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError('seal changed')
    cfg = json.loads(path.read_text())
    if (cfg['model_commit'] != 'fcf12bc229' or cfg['max_bytes'] != 5 * GIB
            or cfg['model_seconds'] != 840 or cfg['total_seconds'] != 1170
            or cfg['hours'] != 3 or cfg['checkpoint_interval_steps'] != 400):
        raise ValueError('fixed authorization limits changed')
    if cfg.get('CPU_model_seconds') != 3600 or cfg.get('CPU_total_seconds') != 3900:
        raise ValueError('manager CPU-only 60-minute model grant changed')
    for filename, expected in cfg['tools'].items():
        if sha(path.parent / 'tools' / filename) != expected:
            raise ValueError('sealed tool changed: ' + filename)
    for filename, expected in cfg['immutable_files'].items():
        if sha(filename) != expected:
            raise ValueError('immutable dependency changed: ' + filename)
    return cfg


def clean_environment(cfg, run, smoke):
    # Never serialize or forward arbitrary parent environment/credentials.
    allowed = ('PATH', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL', 'USER', 'HOME',
               'DBUS_SESSION_BUS_ADDRESS', 'XDG_RUNTIME_DIR', 'TMPDIR')
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    env.update(JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(run / 'tmp'), JAX_PLATFORMS='cpu' if smoke else 'cuda')
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '/usr/bin:/bin')
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64' + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    # These are the unchanged C bundle.env bindings (canonical paths).
    bundle = cfg['bundle']
    env.update(PYTHONPATH=bundle + '/src/src', GPUWRF_JAX_CACHE_DIR=bundle + '/jax',
               JAX_COMPILATION_CACHE_DIR=bundle + '/jax',
               GPUWRF_XLA_AUTOTUNE_CACHE_DIR=bundle + '/autotune',
               TRITON_CACHE_DIR=bundle + '/triton', CUDA_CACHE_PATH=bundle + '/cuda')
    if smoke:
        # Same program/configuration, CPU backend; never consult GPU cache blobs.
        env.update(CUDA_VISIBLE_DEVICES='', WRF_GPU2_SEQ_SMOKE='1', SEQ_SMOKE='1',
                   GPUWRF_JAX_CACHE_DIR=cfg.get('CPU_cache', str(run / 'cpu_cache')),
                   JAX_COMPILATION_CACHE_DIR=cfg.get('CPU_cache', str(run / 'cpu_cache')),
                   GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(run / 'autotune'),
                   TRITON_CACHE_DIR=str(run / 'triton'), CUDA_CACHE_PATH=str(run / 'cuda'))
    else:
        env['GPUWRF_GPU_ARM_CPUS'] = '3,19,13,29'
    return env


def allocated(root):
    seen, total = set(), 0
    for p in Path(root).rglob('*'):
        try:
            s = p.lstat()
        except FileNotFoundError:  # atomic output publication can rename between scan/stat
            continue
        key = (s.st_dev, s.st_ino)
        if key not in seen:
            total += s.st_blocks * 512
            seen.add(key)
    return total


def check_disk(run, cap):
    used = allocated(run)
    if used > cap:
        raise RuntimeError('own diagnostic exceeded disk cap')
    return used


def remove_own_run(run, expected, seal_sha):
    run = Path(run)
    if run.is_symlink() or run.resolve() != Path(expected):
        raise ValueError('cleanup path differs from owned run')
    marker = json.loads((run / 'OWNERSHIP.json').read_text())
    if marker != {'lane': 'v45-bdyneg', 'seal_sha256': seal_sha}:
        raise ValueError('cleanup ownership differs')
    # Overlayfs may leave work/work mode000, owned by our uid. Restore only
    # owner permissions inside our verified tree before traversal/removal.
    for current, directories, _files in os.walk(run, topdown=True, followlinks=False):
        for name in directories:
            p = Path(current) / name
            info = p.lstat()
            if stat.S_ISDIR(info.st_mode):
                if info.st_uid != os.geteuid():
                    raise ValueError('cleanup directory is not owned by this uid: ' + str(p))
                if info.st_mode & 0o700 != 0o700:
                    os.chmod(p, (info.st_mode & 0o777) | 0o700, follow_symlinks=False)
    # Keep the ownership marker until everything else is gone: a failed cleanup
    # can be retried without deleting its own authorization first.
    for child in run.iterdir():
        if child.name == 'OWNERSHIP.json':
            continue
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
    (run / 'OWNERSHIP.json').unlink()
    run.rmdir()
