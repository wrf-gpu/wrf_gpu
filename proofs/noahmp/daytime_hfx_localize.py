"""Daytime HFX term-localization: run GPU Noah-MP coupled forecast, compare
land HFX/GRDFLX/CH/SAV/SAG/TSK/T2 vs pristine-WRF hourly wrfout at daytime leads
(09z/12z/15z = leads 15/18/21) to localize the +5% midday land-HFX over-flux.
Also keeps overnight leads (21z/00z/03z/06z) to confirm no regression."""
import sys, json, dataclasses, os
WT = os.environ.get('WT_SRC', '/tmp/noahmp-daytime-hfx/src')
sys.path.insert(0, WT)
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

RDIR = '/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z'
DOM = 'd03'
DT = 3.0
ACOUSTIC = 4
RADCAD = 200

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import (
    _advance_chunk, _enforce_operational_precision, compute_m9_diagnostics, noahmp_initial_rad,
    _NoahMPRadiation)
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table

cfg = DailyPipelineConfig(run_id=Path(RDIR).name, run_root=Path(RDIR).parent,
                          domain=DOM, dt_s=DT, acoustic_substeps=ACOUSTIC,
                          radiation_cadence_steps=RADCAD)
case, rdir = _build_real_case(cfg)
time_utc = case.run_start
nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                         disable_guards=True, radiation_cadence_steps=RADCAD, time_utc=time_utc)
julian = float(time_utc.timetuple().tm_yday)
noahmp_land, static, init_meta = build_noahmp_land_state(RDIR, DOM)
ep, rp, nroot = build_noahmp_params(static)
nl = dataclasses.replace(nl, use_noahmp=True, noahmp_static=static,
        noahmp_energy_params=ep, noahmp_rad_params=rp, noahmp_nroot=nroot,
        noahmp_julian=julian, noahmp_yearlen=365.0)
state0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
noahmp_rad = noahmp_initial_rad(state0, nl, land_state=noahmp_land)
carry = initial_operational_carry(state0, noahmp_land=noahmp_land, noahmp_rad=noahmp_rad)

xl = np.squeeze(np.asarray(read_wrfout_file(RDIR+'/wrfinput_'+DOM, fields=('XLAND',))['fields']['XLAND']))
land = xl < 1.5

def wrf_at(hour_label, flds):
    f = read_wrfout_file(RDIR+'/wrfout_'+DOM+'_'+hour_label, fields=flds)['fields']
    return {k: np.squeeze(f[k]) for k in flds}

def diag_terms(carry):
    """Run one noah_mp_step on current land carry to read HFX/GRDFLX/CH/TG; also
    re-derive SAV/SAG land-mean from the radiation step for absorbed-rad localization.

    carry.noahmp_rad is the RAW (soldn,lwdn,cosz) 3-tuple held in the carry; the
    coupler's rad2d() reads .lwdn/.soldn/.cosz off the object, so it MUST be wrapped
    in _NoahMPRadiation (exactly as operational_mode.py:1825 does) -- otherwise
    LWDN=0 and the one-step ground re-solve runs away cold (the documented harness
    artifact behind the bogus GRDFLX ~-217)."""
    radiation = _NoahMPRadiation(*carry.noahmp_rad)
    view = _build_column_view(carry.state)
    forcing = assemble_noahmp_forcing(view, static, radiation, None, float(DT))
    ls_out, nm = noah_mp_step(carry.noahmp_land, forcing, static, float(DT),
                              energy_params=ep, rad_params=rp)
    # re-derive radiation absorbed shortwave (SAV/SAG) for absorbed-net localization
    phen = noahmp_phenology_table(carry.noahmp_land, forcing, static)
    rad, _ = radiation_twostream(carry.noahmp_land, forcing, static, phen, rp, float(DT))
    out = {
        'hfx': np.asarray(jax.device_get(nm.hfx)),
        'lh': np.asarray(jax.device_get(nm.lh)),
        'qfx': np.asarray(jax.device_get(nm.qfx)),
        'grdflx': np.asarray(jax.device_get(nm.grdflx)),
        'chs': np.asarray(jax.device_get(nm.chs)),
        'tsk': np.asarray(jax.device_get(nm.tsk)),
        'sav': np.asarray(jax.device_get(rad.sav)),
        'sag': np.asarray(jax.device_get(rad.sag)),
        'fveg': np.asarray(jax.device_get(phen.fveg)),
        # GPU's OWN downward shortwave (held RRTMG) + GPU Noah-MP SALB albedo, to
        # disambiguate excess-absorbed-SW into SWDOWN-too-high vs albedo-too-low.
        'soldn': np.asarray(jax.device_get(forcing.soldn)),
        'salb': np.asarray(jax.device_get(rad.albedo)),
    }
    return out

# daytime leads only (overnight already confirmed clean) -- disambiguate SWDOWN vs albedo
leads = [15, 18, 21]
lead_labels = {3:'2026-05-21_21:00:00', 6:'2026-05-22_00:00:00', 9:'2026-05-22_03:00:00',
               12:'2026-05-22_06:00:00', 15:'2026-05-22_09:00:00', 18:'2026-05-22_12:00:00',
               21:'2026-05-22_15:00:00'}
cadence = RADCAD; seg = cadence
lead_steps = {h:int(round(h*3600.0/DT)) for h in leads}
start = 1
results = []
for h in leads:
    target = lead_steps[h]
    while start <= target:
        n = min(seg, target-start+1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=cadence)
        jax.block_until_ready(carry.state.theta); start += n
    diags = compute_m9_diagnostics(carry.state, nl, float(h)*3600.0,
                                   noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad)
    t2 = np.asarray(jax.device_get(diags.t2))
    tsk_diag = np.asarray(jax.device_get(diags.tsk))
    d = diag_terms(carry)
    tg = np.asarray(jax.device_get(carry.noahmp_land.tg))
    tslb = np.asarray(jax.device_get(carry.noahmp_land.tslb))
    w = wrf_at(lead_labels[h], ('TSK', 'GRDFLX', 'HFX', 'LH', 'T2', 'TSLB', 'SWDOWN'))
    r = {'lead': h, 'label': lead_labels[h][-8:],
         'gpu_T2_land': float(t2[land].mean()), 'wrf_T2_land': float(w['T2'][land].mean()),
         'T2_bias': float((t2[land] - w['T2'][land]).mean()),
         'gpu_HFX_land': float(d['hfx'][land].mean()), 'wrf_HFX_land': float(w['HFX'][land].mean()),
         'HFX_bias': float((d['hfx'][land] - w['HFX'][land]).mean()),
         'gpu_LH_land': float(d['lh'][land].mean()), 'wrf_LH_land': float(w['LH'][land].mean()),
         'LH_bias': float((d['lh'][land] - w['LH'][land]).mean()),
         'gpu_QFX_land': float(d['qfx'][land].mean()),
         'gpu_TSK_land': float(tsk_diag[land].mean()), 'wrf_TSK_land': float(w['TSK'][land].mean()),
         'gpu_TG_land': float(tg[land].mean()),
         'gpu_GRDFLX_land': float(d['grdflx'][land].mean()), 'wrf_GRDFLX_land': float(w['GRDFLX'][land].mean()),
         'gpu_CHS_land': float(d['chs'][land].mean()),
         'gpu_SAV_land': float(d['sav'][land].mean()), 'gpu_SAG_land': float(d['sag'][land].mean()),
         'gpu_SAVpSAG_land': float((d['sav'][land] + d['sag'][land]).mean()),
         'wrf_SWDOWN_land': float(w['SWDOWN'][land].mean()),
         'gpu_SWDOWN_land': float(d['soldn'][land].mean()),
         'gpu_SALB_land': float(np.where(d['salb'][land] > 0, d['salb'][land], np.nan).mean()) if np.any(d['salb'][land] > 0) else 0.0,
         'gpu_FVEG_land': float(d['fveg'][land].mean()),
         'gpu_TSLB0_land': float(tslb[0][land].mean()), 'wrf_TSLB0_land': float(w['TSLB'][0][land].mean())}
    results.append(r)
    print(json.dumps(r))
    sys.stdout.flush()

with open('/tmp/daytime_localize_out.json', 'w') as f:
    json.dump(results, f, indent=2)
print("DONE_LOCALIZE")
