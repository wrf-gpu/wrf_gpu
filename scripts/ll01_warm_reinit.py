"""CPU gate and final-RC runner for the original-WRF warm re-init control."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

DOMAINS = ('d01', 'd02', 'd03')


def quiet_check():
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit(3)


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def compare(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError((label, 'ABI differs', actual.shape, expected.shape, actual.dtype, expected.dtype))
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError((label, 'nonfinite before comparison'))
    error = actual.astype(np.float64) - expected.astype(np.float64)
    result = {'leaf': label, 'byte_equal': actual.tobytes() == expected.tobytes(),
              'max_error': float(np.max(np.abs(error))),
              'rms_error': float(np.sqrt(np.mean(error * error)))}
    if not result['byte_equal']:
        raise AssertionError(result)
    return result


def field_gate(carry, snapshot, jax):
    """Independent critical binding table: actual file fields, not adapter receipts."""
    fields = snapshot.values
    rows = []
    theta = np.asarray(jax.device_get(carry.state.theta), dtype=np.float64)
    reference = np.asarray(fields['THM_2'], dtype=np.float64)
    if not np.isfinite(theta).all() or not np.isfinite(reference).all():
        raise AssertionError('nonfinite theta before independent bridge gate')
    error = theta - 300 - reference
    theta_row = {'leaf': 'state.theta<-THM_2',
                 'max_error': float(np.max(np.abs(error))),
                 'rms_error': float(np.sqrt(np.mean(error * error))),
                 'byte_equal': False, 'gate': 'registered fp32 absolute-theta bridge'}
    if theta_row['rms_error'] > 4.1e-6 or theta_row['max_error'] > 1.5259e-5:
        raise AssertionError(theta_row)
    rows.append(theta_row)
    direct = {
        'u': 'U_2', 'v': 'V_2', 'w': 'W_2', 'qv': 'QVAPOR',
        'qc': 'QCLOUD', 'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP',
        'Ni': 'QNICE', 'Nr': 'QNRAIN', 'qke': 'QKE', 'qsq': 'QSQ',
        'qc_bl': 'QC_BL', 'qi_bl': 'QI_BL', 'cldfra_bl': 'CLDFRA_BL',
        'mol': 'MOL', 'hfx': 'HFX', 'qfx': 'QFX', 'qsfc': 'QSFC', 'pblh': 'PBLH',
        're_cloud': 'RE_CLOUD', 're_ice': 'RE_ICE', 're_snow': 'RE_SNOW',
        'p_perturbation': 'P', 'ph_perturbation': 'PH_2', 'mu_perturbation': 'MU_2',
    }
    for attr, name in direct.items():
        old = getattr(carry.state, attr)
        if old is not None:
            rows.append(compare(jax.device_get(old), np.asarray(fields[name], dtype=old.dtype), 'state.' + attr))
    for attr, name in (('rthraten', 'RTHRATEN'), ('h_diabatic', 'H_DIABATIC'), ('ww', 'WW')):
        old = getattr(carry, attr)
        rows.append(compare(jax.device_get(old), np.asarray(fields[name], dtype=old.dtype), 'carry.' + attr))
    old = carry.o3rad
    rows.append(compare(jax.device_get(old),
        np.asarray(fields['O3RAD'], dtype=old.dtype), 'carry.o3rad'))
    if snapshot.physics['CU_PHYSICS'] == 1:
        for actual, name in zip(carry.cumulus_carry, ('W0AVG', 'NCA'), strict=True):
            rows.append(compare(jax.device_get(actual), np.asarray(fields[name], dtype=actual.dtype), 'kf.' + name))
        for actual, name in zip(carry.cumulus_tendencies,
                ('RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQRCUTEN', 'RQICUTEN', 'RQSCUTEN', 'PRATEC'), strict=True):
            rows.append(compare(jax.device_get(actual), np.asarray(fields[name], dtype=actual.dtype), 'kf.' + name))
    # Every declared land prognostic must be sourced, including fields omitted
    # by the earlier history projection (QSFC/ZNT and previous SWE).
    land = {'tslb': 'TSLB', 'smois': 'SMOIS', 'sh2o': 'SH2O', 'smcwtd': 'SMCWTD',
        'isnow': 'ISNOW', 'tsno': 'TSNO', 'snice': 'SNICE', 'snliq': 'SNLIQ', 'zsnso': 'ZSNSO',
        'snowh': 'SNOWH', 'sneqv': 'SNOW', 'sneqvo': 'SNEQVO', 'tauss': 'TAUSS',
        'albold': 'ALBOLD', 'tv': 'TV', 'tg': 'TG', 'tah': 'TAH', 'eah': 'EAH',
        'canliq': 'CANLIQ', 'canice': 'CANICE', 'fwet': 'FWET', 'lai': 'LAI', 'sai': 'XSAI',
        'cm': 'CM', 'ch': 'CH', 't_skin': 'TSK', 'qsfc': 'QSFC', 'znt': 'ZNT',
        'emiss': 'EMISS', 'albedo': 'ALBEDO', 'sfcrunoff': 'SFROFF', 'udrunoff': 'UDROFF'}
    for attr, name in land.items():
        old = getattr(carry.noahmp_land, attr)
        raw = fields[name].astype(np.float64)
        if attr in ('sfcrunoff', 'udrunoff'):
            raw *= .001
        rows.append(compare(jax.device_get(old), np.asarray(raw, dtype=old.dtype), 'land.' + attr))
    return rows


def adverse_gate(template, snapshot, namelist, hydrate):
    """Poison real saved arrays, including their final cell, before reductions."""
    checked = ('THM_2', 'RHO', 'RTHRATEN', 'H_DIABATIC', 'QKE', 'QSQ',
               'QC_BL', 'QI_BL', 'CLDFRA_BL', 'RE_CLOUD', 'RE_ICE', 'RE_SNOW',
               'O3RAD', 'TSLB', 'EAH', 'W0AVG', 'NCA', 'RTHCUTEN')
    rows = []
    for name in checked:
        if name not in snapshot.values:
            continue
        for kind, bad in (('NaN', np.nan), ('positive_inf', np.inf), ('negative_inf', -np.inf)):
            value = np.array(snapshot.values[name], copy=True)
            value.flat[-1] = bad
            poisoned = replace(snapshot, values={**snapshot.values, name: value})
            try:
                hydrate(template, poisoned, namelist)
            except ValueError as error:
                if 'nonfinite' not in str(error):
                    raise AssertionError(('adverse control failed for another reason', name, kind, str(error)))
                rows.append({'field': name, 'fault': kind, 'status': 'REJECTED_BEFORE_CONVERSION'})
            else:
                raise AssertionError(('nonfinite savepoint accepted', name, kind))
    missing = ('THM_2', 'RTHRATEN', 'QC_BL', 'RE_CLOUD', 'O3RAD', 'NCA')
    for name in missing:
        values = dict(snapshot.values)
        del values[name]
        try:
            hydrate(template, replace(snapshot, values=values), namelist)
        except ValueError as error:
            if 'missing active restart field' not in str(error):
                raise AssertionError(('missing control failed for another reason', name, str(error)))
            rows.append({'field': name, 'fault': 'missing', 'status': 'REJECTED'})
        else:
            raise AssertionError(('missing held leaf accepted', name))
    return rows


def result_use_controls(carry, snapshot, jax):
    """The independent field gate must reject dropped or miswired results."""
    import jax.numpy as jnp
    cases = {
        'dropped_radiation': carry.replace(rthraten=jnp.zeros_like(carry.rthraten)),
        'dropped_ozone': carry.replace(o3rad=jnp.zeros_like(carry.o3rad)),
        'dropped_radii': carry.replace(state=carry.state.replace(re_cloud=jnp.zeros_like(carry.state.re_cloud))),
        'wrong_dry_theta': carry.replace(state=carry.state.replace(theta=carry.state.theta+1)),
        'wrong_mu_level': carry.replace(state=carry.state.replace(mu_perturbation=carry.state.mu_total)),
    }
    rows = []
    for name, changed in cases.items():
        try:
            field_gate(changed, snapshot, jax)
        except AssertionError:
            rows.append({'mutant': name, 'status': 'KILLED_BY_NUMERIC_FIELD_ASSERTION'})
        else:
            raise AssertionError(('field gate ignored modified hydrated result', name))
    return rows


def cpu_gate(args):
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    from gpuwrf.validation.wrf_restart_hydration import read_wrfrst, load_restart_templates, hydrate_wrfrst
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.contracts import state as state_module
    from gpuwrf.runtime.restart_store import RestartStore, nested_identity
    import jax
    assert jax.devices()[0].platform == 'cpu'
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    config = NestedPipelineConfig(input_dir=args.input_dir, output_dir=args.out/'unused_history',
        proof_dir=args.out/'proof', hours=72, max_dom=3, feedback=False, emit_initial_history=False)
    reports = []
    cold = bundles = epoch = dt = None
    for lead, stamp in ((24, '2026-03-01_00:00:00'), (36, '2026-03-01_12:00:00')):
        quiet_check()
        snapshots = {n: read_wrfrst(args.wrfrst / f'wrfrst_{n}_{stamp}') for n in DOMAINS}
        if cold is None:
            original = state_module._gpu_device
            state_module._gpu_device = lambda: jax.devices('cpu')[0]
            try:
                _, bundles, _, epoch, dt, cold = load_restart_templates(config, snapshots)
            finally:
                state_module._gpu_device = original
        carries, phases, maps, bindings = {}, {}, {}, {}
        for name in DOMAINS:
            snap = snapshots[name]
            if snap.epoch != epoch or snap.dt_s != dt[name]:
                raise AssertionError(('restart/case epoch or timestep differs', name,
                                      snap.epoch, epoch, snap.dt_s, dt[name]))
            carries[name], phases[name], maps[name] = hydrate_wrfrst(cold[name], snap, bundles[name].namelist)
            bindings[name] = field_gate(carries[name], snap, jax)
            print('FIELD_GATE', lead, name, len(bindings[name]), maps[name]['theta_bridge'], flush=True)
        steps = {n: snapshots[n].own_steps for n in DOMAINS}
        adverse = adverse_gate(cold['d01'], snapshots['d01'], bundles['d01'].namelist,
                               hydrate_wrfrst) if lead == 24 else []
        result_controls = result_use_controls(carries['d01'], snapshots['d01'], jax) if lead == 24 else []
        phase_bundles = {n: replace(bundles[n], namelist=phases[n]) for n in DOMAINS}
        store = RestartStore(args.out / f'roundtrip_tau{lead}', 4 * 1024**3, 1, 1024**3)
        generation, storage = store.save(carries, steps, dt, nested_identity(config, phase_bundles, epoch),
            driver_state={'original_epoch': epoch.isoformat(), 'cpu_restart_lead': lead, 'outputs': {n: [] for n in DOMAINS}})
        restored, manifest = store.read(generation, expected_identity=nested_identity(config, phase_bundles, epoch))
        roundtrip = []
        for name in DOMAINS:
            left, right = jax.tree.leaves(carries[name]), jax.tree.leaves(restored['carries'][name])
            assert jax.tree.structure(carries[name]) == jax.tree.structure(restored['carries'][name])
            roundtrip.extend(compare(jax.device_get(a), jax.device_get(b), f'{name}.leaf{index}')
                for index, (a, b) in enumerate(zip(left, right, strict=True)))
        report = {'lead_h': lead, 'mappings': maps, 'field_bindings': bindings,
            'roundtrip_leaves': len(roundtrip), 'roundtrip_all_bytes': True, 'own_steps': steps,
            'adverse_controls': adverse,
            'result_use_controls': result_controls,
            'storage': storage, 'manifest_sha256': hashlib.sha256((generation/'manifest.json').read_bytes()).hexdigest()}
        reports.append(report)
        # Delete only this arm's exact checkpoint after its verified readback.
        import shutil
        shutil.rmtree(args.out / f'roundtrip_tau{lead}')
    result = {'status': 'PASS', 'device_platform': 'cpu', 'model_sha': revision(args.model),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_s': time.monotonic()-started, 'reports': reports,
        'scope': 'original CPU savepoint mapping and store byte roundtrip; no GPU timestep/fidelity claim'}
    (args.out/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
    print('WARM_CPU_GATE_DONE', result['wall_s'], flush=True)


def run(args):
    """Final-RC GPU continuation through the existing authenticated restart path."""
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert os.environ.get('JAX_PLATFORMS') == 'cuda'
    assert args.model_sha and revision(args.model) == args.model_sha
    from gpuwrf.validation.wrf_restart_hydration import read_wrfrst, load_restart_templates, hydrate_wrfrst
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.restart_store import RestartStore, nested_identity
    import ll01_reinit as projection
    import jax
    assert jax.devices()[0].platform == 'gpu'
    started = time.monotonic()
    assert args.lead == 36, 'registered final-RC warm control starts at tau36'
    args.out.mkdir(parents=True, exist_ok=False)
    config = pipeline.NestedPipelineConfig(input_dir=args.input_dir, output_dir=args.out/'wrfout',
        proof_dir=args.out/'proof', scratch_dir=args.out/'scratch', hours=72, max_dom=3,
        feedback=False, emit_initial_history=False)
    config.output_dir.mkdir()
    snapshots = {n: read_wrfrst(args.wrfrst/f'wrfrst_{n}_2026-03-01_12:00:00') for n in DOMAINS}
    prefix = {}
    for name in DOMAINS:
        entries = []
        for lead in range(1, 37):
            from datetime import timedelta
            # WRF writes the first step at/after the alarm; root dt54 gives
            # h1=01:00:18 and h2=02:00:36, while child clocks are exact hours.
            seconds = int(np.ceil(lead*3600/snapshots[name].dt_s))*snapshots[name].dt_s
            stamp = (snapshots[name].epoch + timedelta(seconds=seconds)).strftime('%Y-%m-%d_%H:%M:%S')
            source = args.wrfrst / f'wrfout_{name}_{stamp}'
            receipt = projection.stable_hash(source)
            target = config.output_dir/source.name
            target.symlink_to(source.resolve())
            entries.append({'path': str(target.absolute()), 'sha256': receipt['sha256']})
        prefix[name] = entries
    hierarchy, bundles, metadata, epoch, dt, cold = load_restart_templates(config, snapshots)
    carries, mappings, phase_bundles = {}, {}, {}
    for name in DOMAINS:
        snapshot = snapshots[name]
        assert snapshot.epoch == epoch and snapshot.dt_s == dt[name]
        carries[name], phase_namelist, mappings[name] = hydrate_wrfrst(
            cold[name], snapshot, bundles[name].namelist)
        phase_bundles[name] = replace(bundles[name], namelist=phase_namelist)
    own_steps = {n: snapshots[n].own_steps for n in DOMAINS}
    store = RestartStore(args.out/'warm_checkpoint', 8*1024**3, 1, 1024**3)
    generation, storage = store.save(carries, own_steps, dt, nested_identity(config, phase_bundles, epoch),
        driver_state={'outputs': prefix, 'writer_census': None, 'event_counts': {},
                      'force_counts': {}, 'cascade_counts': {}, 'events_tail': ()})
    receipt = {'model_sha': args.model_sha, 'device_platform': jax.devices()[0].platform,
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'own_steps': own_steps, 'dt_s': dt, 'epoch': epoch.isoformat(), 'mappings': mappings,
        'checkpoint': str(generation), 'storage': storage, 'setup_s': time.monotonic()-started,
        'reference': 'P0227 own original CPU continuation tau37-72; R/context and CPU pair separate',
        'new_forecast_leads': [37, 72], 'scope': 'warm CPU-WRF savepoint projection, fp32 bridge disclosed; '
            'preexisting prefix belongs to CPU, not GPU; no bitwise warm-trajectory claim'}
    (args.out/'warm_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('WARM_REINIT_READY', receipt['setup_s'], flush=True)
    original = pipeline._load_domains
    def initialized(new_config, names):
        assert tuple(names) == DOMAINS and new_config.input_dir == config.input_dir
        return hierarchy, phase_bundles, metadata, epoch, dt, cold
    pipeline._load_domains = initialized
    try:
        result = pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=generation))
    finally:
        pipeline._load_domains = original
    (args.out/'pipeline.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['all_domains_finite'] and result['all_outputs_present'], result['verdict']
    assert all(result['per_domain'][n]['own_steps'] == int(72*3600/dt[n]) for n in DOMAINS)
    print('WARM_REINIT_DONE', time.monotonic()-started, result['verdict'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cpu_gate', 'run'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--wrfrst', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-sha')
    parser.add_argument('--lead', type=int, default=36)
    args = parser.parse_args()
    quiet_check()
    assert args.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/mass-sol'))
    assert not subprocess.check_output(['git', '-C', str(args.model), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    sys.path.insert(0, str(args.model/'src'))
    if args.mode == 'cpu_gate':
        cpu_gate(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
