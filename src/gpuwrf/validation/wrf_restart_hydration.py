"""Host-only projection of original WRF restart fields into an existing carry.

The original epoch, own-step counters and held physics survive projection.
This adapter does not run or replace model physics. It rejects missing active
fields and nonfinite values before conversion or comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


STATE_FIELDS = {
    'u': 'U_2', 'v': 'V_2', 'w': 'W_2', 'qv': 'QVAPOR',
    'qc': 'QCLOUD', 'qr': 'QRAIN', 'qi': 'QICE', 'qs': 'QSNOW', 'qg': 'QGRAUP',
    'Ni': 'QNICE', 'Nr': 'QNRAIN', 'qke': 'QKE', 'qsq': 'QSQ',
    'qc_bl': 'QC_BL', 'qi_bl': 'QI_BL', 'cldfra_bl': 'CLDFRA_BL',
    'ustar': 'UST', 't_skin': 'TSK', 'mol': 'MOL', 'hfx': 'HFX', 'qfx': 'QFX',
    'qsfc': 'QSFC', 'pblh': 'PBLH', 'roughness_m': 'ZNT',
    'xland': 'XLAND', 'lakemask': 'LAKEMASK', 'mavail': 'MAVAIL', 'lu_index': 'LU_INDEX',
    'rain_acc': 'RAINNC', 'rainc_acc': 'RAINC', 'snow_acc': 'SNOWNC',
    'graupel_acc': 'GRAUPELNC', 'dtaux3d': 'DTAUX3D', 'dtauy3d': 'DTAUY3D',
    'dusfcg': 'DUSFCG', 'dvsfcg': 'DVSFCG',
    're_cloud': 'RE_CLOUD', 're_ice': 'RE_ICE', 're_snow': 'RE_SNOW',
}
LAND_FIELDS = {
    'tslb': 'TSLB', 'smois': 'SMOIS', 'sh2o': 'SH2O', 'smcwtd': 'SMCWTD',
    'isnow': 'ISNOW', 'tsno': 'TSNO', 'snice': 'SNICE', 'snliq': 'SNLIQ',
    'zsnso': 'ZSNSO', 'snowh': 'SNOWH', 'sneqv': 'SNOW', 'sneqvo': 'SNEQVO',
    'tauss': 'TAUSS', 'albold': 'ALBOLD', 'tv': 'TV', 'tg': 'TG', 'tah': 'TAH',
    'eah': 'EAH', 'canliq': 'CANLIQ', 'canice': 'CANICE', 'fwet': 'FWET',
    'lai': 'LAI', 'sai': 'XSAI', 'cm': 'CM', 'ch': 'CH', 't_skin': 'TSK',
    'qsfc': 'QSFC', 'znt': 'ZNT', 'emiss': 'EMISS', 'albedo': 'ALBEDO',
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
    'lw_clear_sfc_down': 'LWDNBC', 'lw_clear_sfc_up': 'LWUPBC', 'cloud_fraction': 'CLDFRA',
}
KF_FIELDS = ('RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQRCUTEN', 'RQICUTEN', 'RQSCUTEN', 'PRATEC')
CLOCK_FIELDS = ('ITIMESTEP', 'STEPRA', 'STEPBL', 'STEPCU')
STATIC_STATE = {
    'u_bdy', 'v_bdy', 'w_bdy', 'theta_bdy', 'qv_bdy', 'ph_bdy', 'mu_bdy',
    'p_bdy', 'pb_bdy', 'phb_bdy', 'mub_bdy',
}
SURFACE_OVERWRITTEN = {'theta_flux', 'qv_flux', 'tau_u', 'tau_v', 'fltv', 'sfc_wspd'}
PBL_DIAGNOSTICS_OVERWRITTEN = {'maxmf', 'maxwidth', 'ztop_plume'}
INACTIVE_THOMPSON = {'Ns', 'Ng', 'Nc', 'Nn', 'ice_acc', 'qh', 'Nh', 'qvolg', 'qvolh', 'nwfa', 'nifa'}


def finite(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError(f'nonfinite or nonnumeric restart field: {label}')
    return array


@dataclass(frozen=True)
class WRFRestart:
    path: Path
    stamp: datetime
    epoch: datetime
    dt_s: float
    own_steps: int
    radiation_steps: int
    values: dict
    units: dict
    physics: dict
    sha256: str


def read_wrfrst(path: Path) -> WRFRestart:
    """Read a complete, stable original-WRF savepoint, never a history substitute."""
    path = Path(path)
    before = path.stat()
    from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS, ENERGY_ACCUMULATORS
    wanted = set(STATE_FIELDS.values()) | set(LAND_FIELDS.values()) | set(RAD_FIELDS.values())
    wanted |= set(KF_FIELDS) | set(CLOCK_FIELDS) | set(LAND_FLUX_FIELDS) | set(ENERGY_ACCUMULATORS)
    wanted |= {'THM_2', 'P', 'PB', 'PH_2', 'PHB', 'MU_2', 'MUB', 'RHO', 'WW',
               'RTHRATEN', 'H_DIABATIC', 'W0AVG', 'NCA', 'O3RAD', 'EL_PBL',
               'RUCUTEN', 'RVCUTEN', 'RAINCV', 'RAINSHV', 'RAINNCV', 'SNOWNCV',
               'GRAUPELNCV', 'HAILNCV', 'XTIME', 'T2', 'Q2', 'U10', 'V10', 'LH', 'PSFC'}
    with Dataset(path, 'r') as ds:
        ds.set_auto_mask(False)
        times = list(chartostring(ds['Times'][:]))
        if len(times) != 1 or not path.name.endswith(str(times[0])):
            raise ValueError('restart timestamp/file name mismatch')
        stamp = datetime.strptime(str(times[0]), '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
        epoch = datetime.strptime(str(ds.SIMULATION_START_DATE), '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
        dt = float(finite(ds.DT, 'DT'))
        if dt <= 0:
            raise ValueError('restart DT must be positive')
        values = {}
        units = {}
        for name in sorted(wanted & ds.variables.keys()):
            var = ds[name]
            value = np.asarray(var[0] if var.dimensions and var.dimensions[0] == 'Time' else var[:]).copy()
            finite(value, name)  # before casts, subsets, arithmetic or reductions
            value.flags.writeable = False
            values[name] = value
            units[name] = getattr(var, 'units', '')
        for name in CLOCK_FIELDS:
            if name not in values or values[name].size != 1:
                raise ValueError(f'missing/non-scalar restart counter: {name}')
            if float(values[name]) != int(values[name]):
                raise ValueError(f'non-integral restart counter: {name}')
        steps = int(values['ITIMESTEP'])
        elapsed = (stamp - epoch).total_seconds()
        if steps < 0 or steps * dt != elapsed or float(values['XTIME']) * 60 != elapsed:
            raise ValueError('restart own-step/DT/XTIME/epoch disagree')
        physics = {name: int(ds.getncattr(name)) for name in
                   ('USE_THETA_M', 'MP_PHYSICS', 'RA_LW_PHYSICS', 'RA_SW_PHYSICS',
                    'BL_PBL_PHYSICS', 'CU_PHYSICS', 'SF_SURFACE_PHYSICS')}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('CPU restart changed while reading')
    return WRFRestart(path, stamp, epoch, dt, steps, int(values['STEPRA']), values, units, physics, digest)


def load_restart_templates(config, restarts):
    """Use the normal geometry/land loader with genuine saved radiation seeds.

    No radiation solver is run on the discarded cold atmospheric state. The
    temporary hook is confined to eager initialization and restored on error.
    """
    import jax.numpy as jnp
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics

    shapes = {tuple(r.values['THM_2'].shape[-2:]): r for r in restarts.values()}
    if len(shapes) != len(restarts):
        raise ValueError('initial-radiation restart mapping requires distinct domain shapes')

    def saved_radiation(state, namelist=None, *, land_state=None,
                        _with_diagnostics=False, _kernel_call=None):
        del land_state, _kernel_call
        if namelist is None:
            raise ValueError('saved radiation requires a namelist')
        snapshot = shapes[tuple(state.theta.shape[-2:])]
        fields = {}
        for attr, name in RAD_FIELDS.items():
            if name not in snapshot.values:
                raise ValueError(f'missing held radiation seed: {name}')
            fields[attr] = jnp.asarray(finite(snapshot.values[name], name), dtype=state.t_skin.dtype)
        fields.update(swup_topographic=fields['swup'],
            topographic_correction_factor=jnp.ones_like(fields['swup']),
            shadow_mask=jnp.zeros_like(fields['swup'], dtype=jnp.int32))
        rad = RRTMGRadiationDiagnostics(**fields)
        held = (rad.swnorm, rad.glw, rad.coszen)
        return (held, rad) if _with_diagnostics else held

    original = pipeline.noahmp_initial_rad
    pipeline.noahmp_initial_rad = saved_radiation
    try:
        return pipeline._load_domains(config, tuple(restarts))
    finally:
        pipeline.noahmp_initial_rad = original


def hydrate_wrfrst(carry, restart: WRFRestart, namelist, *, require_ozone=True):
    """Return the hydrated carry, phase-preserving namelist and per-leaf audit."""
    import jax
    import jax.numpy as jnp

    values = restart.values
    for name, value in values.items():
        finite(value, name)
    for target, source in (('mp_physics', 'MP_PHYSICS'), ('cu_physics', 'CU_PHYSICS'),
                           ('ra_lw_physics', 'RA_LW_PHYSICS'), ('ra_sw_physics', 'RA_SW_PHYSICS')):
        if int(getattr(namelist, target)) != restart.physics[source]:
            raise ValueError(f'restart/target physics mismatch: {source}')
    if restart.physics['USE_THETA_M'] != 1 or float(namelist.dt_s) != restart.dt_s:
        raise ValueError('this adapter requires matching moist-theta timestep configuration')
    audit = []

    def source(name):
        if name not in values:
            raise ValueError(f'missing active restart field: {name}')
        return finite(values[name], name)

    def put(old, raw, label, origin):
        if old is None:
            raise ValueError(f'active target leaf unseeded: {label}')
        raw = finite(raw, origin)
        if raw.shape != tuple(old.shape):
            raise ValueError(f'restart shape differs for {label}: {raw.shape} vs {old.shape}')
        if np.dtype(old.dtype).kind in 'iu' and np.any(raw != np.rint(raw)):
            raise ValueError(f'non-integral restart value for {label}')
        expected = np.asarray(raw, dtype=old.dtype)
        finite(expected, label)
        result = jnp.asarray(expected)
        actual = finite(np.asarray(jax.device_get(result)), label)
        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            raise ValueError(f'device conversion changed declared target values: {label}')
        loss = actual.astype(np.float64) - raw.astype(np.float64)
        audit.append({'leaf': label, 'source': origin, 'shape': list(raw.shape),
                      'dtype': str(old.dtype), 'policy': 'copied_or_declared_unit_conversion',
                      'converted_target_byte_equal': True,
                      'source_max_error': float(np.max(np.abs(loss))),
                      'source_rms_error': float(np.sqrt(np.mean(loss * loss)))})
        return result

    updates = {}
    for attr, name in STATE_FIELDS.items():
        old = getattr(carry.state, attr, None)
        if old is not None:
            updates[attr] = put(old, source(name), 'state.' + attr, name)
    updates['theta'] = put(carry.state.theta, source('THM_2').astype(np.float64) + 300.,
                           'state.theta', 'THM_2 + 300; one rounding to target dtype')
    for stem, perturb, base in (('p', 'P', 'PB'), ('ph', 'PH_2', 'PHB'), ('mu', 'MU_2', 'MUB')):
        updates[stem + '_total'] = put(getattr(carry.state, stem + '_total'),
            source(perturb).astype(np.float64) + source(base).astype(np.float64),
            'state.' + stem + '_total', perturb + '+' + base)
        updates[stem + '_perturbation'] = put(getattr(carry.state, stem + '_perturbation'),
            source(perturb), 'state.' + stem + '_perturbation', perturb)
    updates['soil_moisture'] = put(carry.state.soil_moisture, source('SMOIS')[0], 'state.soil_moisture', 'SMOIS[0]')
    updates['rhosfc'] = put(carry.state.rhosfc, source('RHO')[0], 'state.rhosfc', 'RHO[0]')
    if getattr(carry.state, 'el_pbl', None) is not None:
        el = source('EL_PBL')
        if el.shape[0] != carry.state.el_pbl.shape[0] + 1 or np.any(el[-1] != 0):
            raise ValueError('EL_PBL must have its WRF unused zero top plane')
        updates['el_pbl'] = put(carry.state.el_pbl, el[:-1], 'state.el_pbl', 'EL_PBL[:-1]; WRF unused top=0')
    state = carry.state.replace(_cast=False, **updates)
    for attr in carry.state.__slots__:
        if attr in updates or getattr(carry.state, attr) is None:
            continue
        if attr in STATIC_STATE or attr.endswith('_bdy'):
            policy = 'retain original-case time-indexed forcing package'
        elif attr in SURFACE_OVERWRITTEN:
            policy = 'first production surface call overwrites before MYNN consumes it'
        elif attr in PBL_DIAGNOSTICS_OVERWRITTEN:
            policy = 'first MYNN call overwrites this output-only diagnostic'
        elif attr in INACTIVE_THOMPSON and restart.physics['MP_PHYSICS'] == 8:
            policy = 'inactive in Thompson8; retain structural seed'
        else:
            raise ValueError(f'unclassified live State leaf: {attr}')
        audit.append({'leaf': 'state.' + attr, 'policy': policy})
    if carry.base_state is None:
        raise ValueError('release hydrator requires explicit native base-state carry')
    base = carry.base_state.replace(**{attr: put(getattr(carry.base_state, attr), source(name),
                   'base.' + attr, name) for attr, name in (('pb', 'PB'), ('phb', 'PHB'), ('mub', 'MUB'))})
    if carry.noahmp_land is None:
        raise ValueError('Noah-MP carry must be initialized structurally before hydration')
    land_updates = {attr: put(getattr(carry.noahmp_land, attr),
        source(name).astype(np.float64) * (.001 if attr in ('sfcrunoff', 'udrunoff') else 1),
        'land.' + attr, name + (' * .001 m/mm' if attr in ('sfcrunoff', 'udrunoff') else ''))
        for attr, name in LAND_FIELDS.items()}
    land = carry.noahmp_land.replace(**land_updates)
    if carry.radiation_diagnostics is None:
        raise ValueError('held radiation diagnostics must be initialized structurally')
    rad_updates = {attr: put(getattr(carry.radiation_diagnostics, attr), source(name), 'rad.' + attr, name)
        for attr, name in RAD_FIELDS.items() if getattr(carry.radiation_diagnostics, attr) is not None}
    # These three diagnostics are not persisted by WRF and have no model-step
    # consumers: the next radiation call replaces them. SWNORM, COSZEN and all
    # actual held fluxes above remain the original saved CPU values, including
    # the case's slope correction. Do not recompute a new radiation call here.
    rad_updates.update(swup_topographic=rad_updates['swup'],
        topographic_correction_factor=jnp.ones_like(carry.radiation_diagnostics.topographic_correction_factor),
        shadow_mask=jnp.zeros_like(carry.radiation_diagnostics.shadow_mask, dtype=jnp.int32))
    rad = carry.radiation_diagnostics._replace(**rad_updates)
    for attr in ('swup_topographic', 'topographic_correction_factor', 'shadow_mask'):
        audit.append({'leaf': 'rad.' + attr,
                      'policy': 'unpersisted output-only diagnostic; next radiation call overwrites; no physics consumer'})
    extra = {'state': state, 'base_state': base, 'noahmp_land': land, 'radiation_diagnostics': rad,
             'rthraten': put(carry.rthraten, source('RTHRATEN'), 'carry.rthraten', 'RTHRATEN uncoupled'),
             'h_diabatic': put(carry.h_diabatic, source('H_DIABATIC'), 'carry.h_diabatic', 'H_DIABATIC'),
             'ww': put(carry.ww, source('WW'), 'carry.ww', 'WW'),
             'ww_save': put(carry.ww_save, source('WW'), 'carry.ww_save', 'WW'),
             'u_save': state.u, 'v_save': state.v, 'w_save': state.w, 't_save': state.theta,
             'ph_save': state.ph_total, 'mu_save': state.mu_perturbation, 'muts': state.mu_total}
    extra['noahmp_rad'] = tuple(put(old, source(name), 'noahmp_rad.' + name, name)
        for old, name in zip(carry.noahmp_rad, ('SWNORM', 'GLW', 'COSZEN'), strict=True))
    for attr in ('t_2ave', 'mudf', 'muave', 'ph_tend'):
        extra[attr] = jnp.zeros_like(getattr(carry, attr))
        audit.append({'leaf': 'carry.' + attr, 'policy': 'RK1 transition scratch reset, no held physics'})
    if restart.physics['CU_PHYSICS'] == 1:
        if carry.cumulus_carry is None or carry.cumulus_tendencies is None:
            raise ValueError('active KF target carry unseeded')
        extra['cumulus_carry'] = tuple(put(old, source(name), 'kf.' + name, name)
            for old, name in zip(carry.cumulus_carry, ('W0AVG', 'NCA'), strict=True))
        extra['cumulus_tendencies'] = tuple(put(old, source(name), 'kf.' + name, name)
            for old, name in zip(carry.cumulus_tendencies, KF_FIELDS, strict=True))
        if np.any(source('RUCUTEN') != 0) or np.any(source('RVCUTEN') != 0):
            raise ValueError('port KF carry has no held momentum slots for nonzero CPU RUCUTEN/RVCUTEN')
    elif carry.cumulus_carry is not None or carry.cumulus_tendencies is not None:
        raise ValueError('inactive KF source cannot seed an active target')
    if require_ozone:
        old = getattr(carry, 'o3rad', None)
        if old is None:
            raise ValueError('RC must provide a seeded held o3rad leaf')
        if restart.units.get('O3RAD') != 'ppmv':
            raise ValueError('CPU O3RAD Registry units label must be ppmv')
        # WRF's ozn_p_int produces o3vmr and RRTMG consumes O3RAD directly
        # (module_radiation_driver.F:5126; module_ra_rrtmg_lw.F:12014,12406).
        # The Registry's "ppmv" label is provenance, not the stored unit.
        extra['o3rad'] = put(old, source('O3RAD'), 'carry.o3rad',
                            'O3RAD volume mixing ratio (WRF Registry label: ppmv)')
    for name in ('land_history', 'energy_accumulators'):
        old = getattr(carry, name)
        if old is None:
            continue
        fields = {}
        for key in old:
            if key in values:
                fields[key] = put(old[key], source(key), name + '.' + key, key)
            elif name == 'land_history':
                fields[key] = jnp.zeros_like(old[key])
                audit.append({'leaf': name + '.' + key, 'policy': 'instantaneous output rewritten by first Noah step'})
            else:
                raise ValueError(f'missing persistent energy accumulator: {key}')
        extra[name] = type(old).pack(fields)
    if carry.history_diagnostics is not None:
        extra['history_diagnostics'] = carry.history_diagnostics._replace(**{
            attr: put(getattr(carry.history_diagnostics, attr), source(attr.upper()), 'history.' + attr, attr.upper())
            for attr in carry.history_diagnostics._fields if getattr(carry.history_diagnostics, attr) is not None})
    if carry.noahmp_precipitation is not None:
        precipitation = (source('RAINCV') + source('RAINSHV'), source('RAINNCV'),
                         source('SNOWNCV'), source('GRAUPELNCV'), source('HAILNCV'))
        extra['noahmp_precipitation'] = type(carry.noahmp_precipitation)(*(
            put(old, value / restart.dt_s, 'precip.' + name, name + ' previous amount / DT')
            for name, old, value in zip(carry.noahmp_precipitation._fields,
                                        carry.noahmp_precipitation, precipitation, strict=True)))
    hydrated = carry.replace(**extra)
    for leaf in jax.tree.leaves(hydrated):
        finite(np.asarray(jax.device_get(leaf)), 'hydrated carry')
    phase_namelist = replace(namelist, radiation_cadence_steps=restart.radiation_steps,
        cumulus_cadence_steps=int(values['STEPCU']))
    theta = np.asarray(jax.device_get(hydrated.state.theta), dtype=np.float64) - 300
    delta = theta - source('THM_2').astype(np.float64)
    bridge = {'rms_K': float(np.sqrt(np.mean(delta * delta))), 'max_K': float(np.max(np.abs(delta)))}
    if bridge['rms_K'] > 4.1e-6 or bridge['max_K'] > 1.5259e-5:
        raise ValueError(f'theta bridge exceeds registered bound: {bridge}')
    return hydrated, phase_namelist, {'leaves': audit, 'theta_bridge': bridge,
        'own_steps': restart.own_steps, 'dt_s': restart.dt_s, 'epoch': restart.epoch.isoformat(),
        'stamp': restart.stamp.isoformat(), 'radiation_steps': restart.radiation_steps,
        'source': str(restart.path), 'source_sha256': restart.sha256,
        'scope': 'original CPU-WRF field projection; no exact warm GPU trajectory claim'}
