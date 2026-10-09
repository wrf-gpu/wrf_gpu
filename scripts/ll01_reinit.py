"""LL01 history projection control, using the unmodified nested restart path.

This is explicitly NOT an exact WRF restart. Available history fields replace
the cold seed after its geometry initialization; missing warm fields retain
their declared cold seeds. The own-GPU-history control measures that loss first.
Model and experiment-driver commits are attested separately, without weakening
the executable or checkpoint identity checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from netCDF4 import Dataset, chartostring

DOMAINS = ('d01', 'd02', 'd03')
STATE_FIELDS = {
    'u': 'U', 'v': 'V', 'w': 'W', 'qv': 'QVAPOR', 'qc': 'QCLOUD',
    'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP',
    'Ni': 'QNICE', 'Nr': 'QNRAIN', 'qke': 'QKE', 'ustar': 'UST',
    't_skin': 'TSK', 'hfx': 'HFX', 'qfx': 'QFX', 'pblh': 'PBLH',
    'rain_acc': 'RAINNC', 'rainc_acc': 'RAINC', 'snow_acc': 'SNOWNC',
    'graupel_acc': 'GRAUPELNC',
}
LAND_FIELDS = {
    'tslb': 'TSLB', 'smois': 'SMOIS', 'sh2o': 'SH2O', 'smcwtd': 'SMCWTD',
    'isnow': 'ISNOW', 'tsno': 'TSNO', 'snice': 'SNICE', 'snliq': 'SNLIQ',
    'zsnso': 'ZSNSO', 'snowh': 'SNOWH', 'sneqv': 'SNOW', 'sneqvo': 'SNEQVO',
    'tauss': 'TAUSS', 'albold': 'ALBOLD', 'tv': 'TV', 'tg': 'TG',
    'tah': 'TAH', 'eah': 'EAH', 'canliq': 'CANLIQ', 'canice': 'CANICE',
    'fwet': 'FWET', 'lai': 'LAI', 'sai': 'XSAI', 'cm': 'CM', 'ch': 'CH',
    't_skin': 'TSK', 'emiss': 'EMISS', 'albedo': 'ALBEDO',
    'sfcrunoff': 'SFROFF', 'udrunoff': 'UDROFF',
}
RAD_FIELDS = {
    'surface_albedo': 'ALBEDO', 'surface_emissivity': 'EMISS', 'coszen': 'COSZEN',
    'swdown': 'SWDNB', 'swnorm': 'SWNORM', 'swup': 'SWUPB', 'glw': 'GLW',
    'glw_up': 'LWUPB', 'sw_toa_down': 'SWDNT', 'sw_toa_up': 'SWUPT',
    'lw_toa_down': 'LWDNT', 'lw_toa_up': 'LWUPT',
    'sw_clear_toa_down': 'SWDNTC', 'sw_clear_toa_up': 'SWUPTC',
    'sw_clear_sfc_down': 'SWDNBC', 'sw_clear_sfc_up': 'SWUPBC',
    'lw_clear_toa_down': 'LWDNTC', 'lw_clear_toa_up': 'LWUPTC',
    'lw_clear_sfc_down': 'LWDNBC', 'lw_clear_sfc_up': 'LWUPBC',
    'cloud_fraction': 'CLDFRA',
}
REQUIRED = {'U', 'V', 'W', 'T', 'THM', 'QVAPOR', 'P', 'PB', 'PH', 'PHB',
            'MU', 'MUB', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP',
            'QNICE', 'QNRAIN', 'QKE', 'TSLB', 'SMOIS', 'SH2O', 'TSK'}
UNOBSERVED = ('RTHRATEN', 'H_DIABATIC', 'W0AVG', 'NCA', 'QSQ', 'QC_BL',
              'QI_BL', 'CLDFRA_BL', 'MOL', 'QSFC')


def quiet_check():
    if Path('/tmp/wrf_gpu2_quiet').exists():
        print('QUIET', flush=True)
        raise SystemExit(3)


def git_revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def stable_hash(path):
    quiet_check()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns), path
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def read_history(path, lead):
    quiet_check()
    before = path.stat()
    with Dataset(path, 'r') as ds:
        ds.set_auto_mask(False)
        assert float(ds['XTIME'][0]) == lead * 60, (path, float(ds['XTIME'][0]))
        stamp = str(chartostring(ds['Times'][:])[0])
        assert path.name.endswith(stamp), path
        fields = set(STATE_FIELDS.values()) | set(LAND_FIELDS.values()) | set(RAD_FIELDS.values())
        fields |= REQUIRED | {'XLAT', 'XLONG', 'HGT', 'LANDMASK', 'U10', 'V10', 'T2', 'Q2', 'LH', 'PSFC'}
        # Includes every available 2-D land flux and accumulated diagnostic.
        fields |= {n for n, v in ds.variables.items() if v.dimensions == ('Time', 'south_north', 'west_east')}
        values = {n: np.asarray(ds[n][0]).copy() for n in fields if n in ds.variables}
        missing = REQUIRED - values.keys()
        assert not missing, (path, missing)
        schema = {n: {'shape': list(a.shape), 'dtype': str(a.dtype),
                      'units': getattr(ds[n], 'units', '')} for n, a in values.items()}
        for name, array in values.items():
            assert np.isfinite(array).all(), (path, name, 'nonfinite')
        receipt = {'path': str(path.resolve()), 'time': stamp, 'xtime_min': lead * 60,
                   'fields': schema, 'missing_warm': [n for n in UNOBSERVED if n not in ds.variables],
                   'hgt_sha256': hashlib.sha256(values['HGT'].tobytes()).hexdigest()}
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns), path
    return values, receipt


def audit(args):
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    started = time.monotonic()
    prefix, domains = {}, {}
    for name in DOMAINS:
        files = sorted(args.history.glob(f'wrfout_{name}_*'))
        assert len(files) >= args.lead + 13, (name, len(files))
        _, domains[name] = read_history(files[args.lead], args.lead)
        prefix[name] = [stable_hash(path) for path in files[1:args.lead + 1]]
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_s': time.monotonic() - started,
               'model_sha': git_revision(args.model), 'driver_sha': git_revision(Path(__file__).resolve().parent),
               'driver_file_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'history': str(args.history.resolve()), 'input_dir': str(args.input_dir.resolve()),
               'lead': args.lead, 'domains': domains, 'prefix_receipts': prefix,
               'history_role': args.history_role,
               'state_projection': True, 'cache': 'none', 'cpu_wrf_restart_hydrator': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print('AUDIT_DONE', receipt['elapsed_s'], flush=True)


def hydrate(carry, values, *, use_theta_m):
    import jax
    import jax.numpy as jnp

    copied, missing = [], []

    def put(old, value, label):
        assert old is not None, (label, 'unseeded leaf')
        assert tuple(old.shape) == value.shape, (label, old.shape, value.shape)
        result = jnp.asarray(value, dtype=old.dtype)
        copied.append(label)
        return result

    updates = {attr: put(getattr(carry.state, attr), values[name], f'state.{attr}')
               for attr, name in STATE_FIELDS.items() if name in values}
    theta = values['THM'].astype(np.float64) + 300.
    assert use_theta_m == 1, 'This registered control uses moist THM; option0 needs its own adapter'
    updates['theta'] = put(carry.state.theta, theta, 'state.theta<-THM+300')
    for attr, perturb, base in (('p', 'P', 'PB'), ('ph', 'PH', 'PHB'), ('mu', 'MU', 'MUB')):
        total = values[perturb].astype(np.float64) + values[base].astype(np.float64)
        for leaf, value in ((attr, total), (attr + '_total', total), (attr + '_perturbation', values[perturb])):
            updates[leaf] = put(getattr(carry.state, leaf), value, f'state.{leaf}')
    state = carry.state.replace(**updates)
    base_differences = {attr: float(np.max(np.abs(np.asarray(jax.device_get(getattr(carry.base_state, attr))) - values[name])))
                        for attr, name in (('pb', 'PB'), ('phb', 'PHB'), ('mub', 'MUB'))}
    # Preserve the baseline's static base-state/metrics. Own-history should agree;
    # a CPU projection with a different base needs a separately audited adapter.
    assert max(base_differences.values()) <= .02, ('static base differs', base_differences)
    land_updates = {}
    for attr, name in LAND_FIELDS.items():
        if name in values:
            value = values[name] * (.001 if attr in ('sfcrunoff', 'udrunoff') else 1)
            land_updates[attr] = put(getattr(carry.noahmp_land, attr), value, f'land.{attr}')
        else:
            missing.append(f'land.{attr}')
    land = carry.noahmp_land.replace(**land_updates)
    rad_updates = {attr: put(getattr(carry.radiation_diagnostics, attr), values[name], f'rad.{attr}')
                   for attr, name in RAD_FIELDS.items() if name in values
                   and getattr(carry.radiation_diagnostics, attr) is not None}
    rad = carry.radiation_diagnostics._replace(**rad_updates)
    held_rad = tuple(put(old, values[name], f'noahmp_rad.{name}') for old, name in
                     zip(carry.noahmp_rad, ('SWNORM', 'GLW', 'COSZEN')))
    extra = {}
    for packed_name in ('land_history', 'energy_accumulators'):
        packed = getattr(carry, packed_name)
        if packed is not None:
            extra[packed_name] = type(packed).pack({n: put(packed[n], values[n], f'{packed_name}.{n}')
                                                   if n in values else packed[n] for n in packed})
    history_diag = carry.history_diagnostics
    if history_diag is not None:
        extra['history_diagnostics'] = history_diag._replace(**{
            attr: put(getattr(history_diag, attr), values[attr.upper()], f'history.{attr}')
            for attr in history_diag._fields if getattr(history_diag, attr) is not None
            and attr.upper() in values})
    # Only transition scratch is recreated. Held physics not in history retains
    # the cold template, explicitly listed in the receipt and bounded by control.
    projected = carry.replace(state=state, noahmp_land=land, noahmp_rad=held_rad,
        radiation_diagnostics=rad, u_save=state.u, v_save=state.v, w_save=state.w,
        t_save=state.theta, ph_save=state.ph, mu_save=state.mu_perturbation,
        muts=state.mu_total, **extra)
    return projected, {'copied': copied, 'missing_land': missing,
                        'state_not_hydrated': [n for n in state.__slots__ if n not in updates],
                        'missing_warm_policy': 'keep original cold seed',
                        'base_policy': 'keep baseline static base, verify history within .02',
                        'base_max_abs': base_differences,
                        'warm_seed_loss': list(UNOBSERVED)}


def run(args):
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert os.environ.get('JAX_PLATFORMS') == 'cuda'
    assert git_revision(args.model) == args.model_sha
    assert not subprocess.check_output(['git', '-C', str(args.model), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'cuda_async')
    sys.path.insert(0, str(args.model / 'src'))
    # Package bootstrap sets x64 and XLA command-buffer/cache flags. It MUST
    # precede backend initialization, just as the shipped CLI does.
    from gpuwrf.integration import nested_pipeline as pipeline
    import jax
    assert jax.devices()[0].platform == 'gpu', jax.devices()
    from gpuwrf.integration.d02_replay import _wrf_use_theta_m
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.runtime.restart_store import RestartStore, nested_identity
    receipt = json.loads(args.receipt.read_text())
    assert receipt['model_sha'] == args.model_sha
    assert receipt['driver_file_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert receipt['history'] == str(args.history.resolve()) and receipt['lead'] == args.lead
    assert receipt['history_role'] == args.history_role
    assert receipt['input_dir'] == str(args.input_dir.resolve())
    assert args.lead == 36, 'Only the synchronous tau36 protocol is registered'
    assert not args.out.exists(), 'fresh arm directory required'
    args.out.mkdir(parents=True)
    config = pipeline.NestedPipelineConfig(input_dir=args.input_dir, output_dir=args.out / 'wrfout',
        proof_dir=args.out / 'proof', scratch_dir=args.out / 'scratch', hours=48, max_dom=3,
        feedback=False, emit_initial_history=False)
    # The restart journal binds every prior output to this output directory.
    # Link the authenticated immutable prefix; these
    # frames were produced by the baseline, NOT by this projected continuation.
    # No data copy, reference mutation, foreign-path exception, or guard bypass.
    config.output_dir.mkdir()
    bound_prefix, aliases = {}, []
    for name, entries in receipt['prefix_receipts'].items():
        bound_prefix[name] = []
        for entry in entries:
            source = Path(entry['path'])
            target = config.output_dir / source.name
            kind = 'hardlink'
            try:
                os.link(source, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # CPU-WRF history lives on the read-only NAS. Keep the bound
                # prefix name inside our stream; the normal restore verifies
                # the dereferenced bytes against the CPU audit SHA. No copies.
                target.symlink_to(source)
                kind = 'symlink_readonly_reference'
            bound_prefix[name].append({'path': str(target.absolute()), 'sha256': entry['sha256']})
            aliases.append({'baseline': str(source), 'bound_prefix': str(target), 'kind': kind})
    started = time.monotonic()
    loaded = pipeline._load_domains(config, DOMAINS)
    hierarchy, bundles, metadata, epoch, dt, cold = loaded
    carries, mappings = {}, {}
    input_run = Gen2Run(args.input_dir)
    for name in DOMAINS:
        values, source = read_history(Path(receipt['domains'][name]['path']), args.lead)
        assert source == receipt['domains'][name], ('history changed after CPU audit', name)
        carries[name], mappings[name] = hydrate(cold[name], values,
            use_theta_m=_wrf_use_theta_m(input_run, name))
    own_steps = {n: int(args.lead * 3600 / dt[n]) for n in DOMAINS}
    assert all(own_steps[n] * dt[n] == args.lead * 3600 for n in DOMAINS)
    jax.block_until_ready(carries)
    store = RestartStore(args.out / 'projection_checkpoint', 8 * 1024**3, 1, 10 * 1024**3)
    generation, storage = store.save(carries, own_steps, dt, nested_identity(config, bundles, epoch),
        driver_state={'outputs': bound_prefix, 'writer_census': None,
                      'event_counts': {}, 'force_counts': {}, 'cascade_counts': {}, 'events_tail': ()})
    manifest = {**receipt, 'device_platform': jax.devices()[0].platform,
        'resolved_boolean_flags': {k: v for k, v in sorted(os.environ.items())
                                   if k.startswith('GPUWRF_') and v.lower() in
                                   ('0', '1', 'true', 'false', 'on', 'off', 'yes', 'no')},
        'projection_setup_s': time.monotonic() - started, 'own_steps': own_steps, 'dt_s': dt,
        'mappings': mappings, 'checkpoint': str(generation), 'storage': storage,
        'baseline_prefix_aliases': aliases, 'new_forecast_frame_leads': [37, 48],
        'scope': ('GPU own-history projection control, NOT WRF fidelity' if args.history_role == 'gpu_control'
                  else 'GPU re-init from original CPU-WRF HISTORY; projection control bounds wind inference'),
        'exact_warm_restart': False}
    (args.out / 'projection_receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PROJECTION_READY', manifest['projection_setup_s'], flush=True)
    # Reuse the EXACT already initialized bundles; no second CUDA child or cold
    # init. The public pipeline still authenticates and restores the checkpoint.
    original_loader = pipeline._load_domains
    def reuse_loaded(new_config, names):
        assert tuple(names) == DOMAINS and new_config.input_dir == config.input_dir
        return hierarchy, bundles, metadata, epoch, dt, cold
    pipeline._load_domains = reuse_loaded
    from dataclasses import replace
    try:
        result = pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=generation))
    finally:
        pipeline._load_domains = original_loader
    (args.out / 'pipeline.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['all_domains_finite'] and result['all_outputs_present'], result['verdict']
    assert all(result['per_domain'][n]['own_steps'] == int(48 * 3600 / dt[n]) for n in DOMAINS)
    print('PROJECTION_DONE', time.monotonic() - started, result['verdict'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'run'))
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--model-sha', default='3e194296c22518ae649506174a3a6e360fcd2c3d')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--history', required=True, type=Path)
    parser.add_argument('--lead', type=int, default=36)
    parser.add_argument('--history-role', choices=('gpu_control', 'cpu_wrf'), default='gpu_control')
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    assert args.receipt.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/mass-sol'))
    if args.mode == 'audit':
        audit(args)
    else:
        assert args.out is not None and args.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/mass-sol'))
        run(args)


if __name__ == '__main__':
    main()
