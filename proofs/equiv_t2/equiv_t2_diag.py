"""Equivalence-T2 GPU diagnostic, EDMF-ON variant (TOST-critical daytime-T2 close).

This is the equiv-T2 harness from agent-a7d391d00678e2e44, copied into the
v020-tost-daytimefix consolidation branch and EXTENDED with a mandatory FULL-RUN
STABILITY audit (the surface-w lesson: an oracle-passing fix blew W to ~300 m/s in
a real run, so a column oracle is NOT sufficient -- we must verify the coupled full
run stays physical to +21h with edmf=True wired live in physics_couplers.mynn_adapter).

The MYNN-EDMF mass-flux qv transport is ENABLED via the wired coupler (edmf=True).
Compare LH/QFX/T2/qair land-means against the edmf=False baseline
(proofs/equiv_t2/equiv_t2_baseline_edmf_off.json) and the WRF references.

ONE GPU run, use_noahmp ON, d03 1km, daytime leads 09z/12z/15z.

LOAD-BEARING radiation cadence: radt = DT * RADCAD / 60 MUST == 30 min (the
pristine-WRF namelist radt). DT=3.0 -> RADCAD=600 -> radt=1800s=30min. The earlier
remeasure used RADCAD=200 (radt=10min) which leaves the L1 -radt/2 offset the wrong
magnitude (~5.7% off). See proofs/rad_time/coszen_phase_proof.json HANDOFF.

Purpose 1 (VALIDATE L1 @ radt=30min): compare the GPU HELD land-mean SWDOWN
(M9Diagnostics.swdown, which on the noahmp path is the held noahmp_rad[0], i.e. the
L1-fixed coszen(t - radt/2) field) to the pristine-WRF wrfout SWDOWN at each daytime
lead. PASS if |residual| <= 2%.

Purpose 2 (MEASURE post-L1 daytime T2 bias): GPU land-mean T2 vs WRF T2 at leads.

Purpose 3 (PINPOINT L2 coupling divergence): dump the coupler FORCING fed to
Noah-MP -- sfctmp, qair, uu, vv, zlvl, soldn, lwdn -- plus the Noah-MP-internal
surface exchange coeff CHS/CH and the evolved carry sh2o[0]/smois/tg/tah/eah, and
compare each to WRF at the same leads. The L2 lane (proofs/noahmp_lh/L2_verdict.json)
proved the LAND MODEL faithful when fed WRF's OWN T2/Q2/U10/V10 hourly forcing; so the
divergent coupler-forcing field (vs WRF's near-surface state) scopes the coupling fix.

Two WRF references are reported per forcing field so the comparison is unambiguous:
 (a) WRF lowest-MODEL-LEVEL state (T/QVAPOR/U/V/PH+PHB) -- apples-to-apples with the
     GPU coupler forcing, which is the lowest model level (zlvl=0.5*dz, ~15-30m).
 (b) WRF 2m/10m diagnostics (T2/Q2/U10/V10) -- the exact fields the L2 offline driver
     fed Noah-MP and proved faithful against.
"""
import sys, json, dataclasses, os, time
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

RDIR = '/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z'
DOM = 'd03'
DT = 3.0
ACOUSTIC = 4
RADCAD = 600          # radt = DT*RADCAD/60 = 3*600/60 = 30 min  (LOAD-BEARING)
RADT_MIN = DT * RADCAD / 60.0
assert abs(RADT_MIN - 30.0) < 1e-9, f"radt must be 30min, got {RADT_MIN}"

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import (
    _advance_chunk, _enforce_operational_precision, compute_m9_diagnostics,
    noahmp_initial_rad, _NoahMPRadiation)
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step

G_ACCEL = 9.81
P0 = 1.0e5
KAPPA = 287.0 / 1004.0   # R_d / c_p

cfg = DailyPipelineConfig(run_id=Path(RDIR).name, run_root=Path(RDIR).parent,
                          domain=DOM, dt_s=DT, acoustic_substeps=ACOUSTIC,
                          radiation_cadence_steps=RADCAD)
case, rdir = _build_real_case(cfg)
time_utc = case.run_start
nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                         disable_guards=True, radiation_cadence_steps=RADCAD,
                         time_utc=time_utc)
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
G = lambda x: np.asarray(jax.device_get(x))
S = lambda a: np.squeeze(np.asarray(a, dtype=np.float64))


def wrf_at(hour_label, flds):
    f = read_wrfout_file(RDIR+'/wrfout_'+DOM+'_'+hour_label, fields=flds)['fields']
    return {k: S(f[k]) for k in flds}


def wrf_lowest_level(hour_label):
    """WRF lowest-model-level near-surface state, apples-to-apples with the GPU
    coupler forcing (which is the lowest model level). theta=T[0]+300; pres=P[0]+PB[0];
    sfctmp=theta*(pres/P0)^kappa; q=QVAPOR[0]; winds destaggered to mass; zlvl =
    half the first-layer geometric thickness (PH+PHB)/g."""
    w = read_wrfout_file(RDIR+'/wrfout_'+DOM+'_'+hour_label,
                         fields=('T', 'QVAPOR', 'U', 'V', 'P', 'PB', 'PH', 'PHB'))['fields']
    T = S(w['T']); QV = S(w['QVAPOR']); P = S(w['P']); PB = S(w['PB'])
    PH = S(w['PH']); PHB = S(w['PHB']); U = S(w['U']); V = S(w['V'])
    theta0 = T[0] + 300.0
    pres0 = P[0] + PB[0]
    sfctmp = theta0 * (pres0 / P0) ** KAPPA
    qair = QV[0]
    # destagger lowest-level winds to mass points
    u0 = 0.5 * (U[0, :, :-1] + U[0, :, 1:])
    v0 = 0.5 * (V[0, :-1, :] + V[0, 1:, :])
    wspd = np.sqrt(u0 ** 2 + v0 ** 2)
    # geometric height of full (w) levels above ground; zlvl = mid of first mass layer
    z_full = (PH + PHB) / G_ACCEL          # (nz+1, ny, nx) absolute geopotential height
    zlvl = 0.5 * (z_full[1] - z_full[0])   # half the first layer thickness (AGL)
    return dict(sfctmp=sfctmp, qair=qair, u=u0, v=v0, wspd=wspd, zlvl=zlvl)


def diag_forcing_and_land(carry):
    """Assemble the operational Noah-MP forcing from the live carry (exactly as
    operational_mode does), run ONE noah_mp_step to read the Noah-MP-internal CHS/CH
    and fluxes, and return forcing + carry-state land-means."""
    radiation = _NoahMPRadiation(*carry.noahmp_rad)
    view = _build_column_view(carry.state)
    forcing = assemble_noahmp_forcing(view, static, radiation, None, float(DT))
    ls_out, nm = noah_mp_step(carry.noahmp_land, forcing, static, float(DT),
                              energy_params=ep, rad_params=rp)
    uu = G(forcing.uu); vv = G(forcing.vv)
    zlvl = G(forcing.zlvl)
    if np.ndim(zlvl) == 0:
        zlvl = np.full_like(uu, float(zlvl))
    return dict(
        # coupler forcing (lowest model level)
        f_sfctmp=G(forcing.sfctmp), f_qair=G(forcing.qair),
        f_uu=uu, f_vv=vv, f_wspd=np.sqrt(uu**2 + vv**2),
        f_zlvl=zlvl,
        f_soldn=G(forcing.soldn), f_lwdn=G(forcing.lwdn),
        # Noah-MP internal surface exchange + fluxes
        chs=G(nm.chs), hfx=G(nm.hfx), lh=G(nm.lh), qfx=G(nm.qfx), tsk=G(nm.tsk),
        # evolved carry state
        ch=G(carry.noahmp_land.ch),
        sh2o0=G(carry.noahmp_land.sh2o)[0], smois0=G(carry.noahmp_land.smois)[0],
        tg=G(carry.noahmp_land.tg), tah=G(carry.noahmp_land.tah),
        eah=G(carry.noahmp_land.eah), tv=G(carry.noahmp_land.tv),
    )


def lm(a):
    return float(np.asarray(a)[land].mean())


# daytime leads: 09z=lead15, 12z=lead18, 15z=lead21
leads = [15, 18, 21]
lead_labels = {15: '2026-05-22_09:00:00', 18: '2026-05-22_12:00:00', 21: '2026-05-22_15:00:00'}
lead_steps = {h: int(round(h * 3600.0 / DT)) for h in leads}
cadence = RADCAD
seg = cadence
start = 1
results = []
# --- mandatory full-run stability audit (surface-w lesson) ---
stability = {'edmf': True, 'worst_maxabs_W_mps': 0.0, 'worst_maxabs_W_lead': None,
             'nan_or_inf': False, 'first_bad_field': None, 'first_bad_step': None,
             'per_lead_maxabs_W': {}, 'physical': True}
W_PHYSICAL_LIMIT = 50.0   # |w| over d03 terrain should stay well under this; the
                          # surface-w failure mode hit ~300 m/s. Report worst lead.


def _audit_stability(carry, label, step):
    """Track max|W| and NaN/Inf across the live carry; mutate `stability`."""
    st = carry.state
    wmax = float(np.nanmax(np.abs(G(st.w)))) if st.w is not None else 0.0
    stability['per_lead_maxabs_W'][label] = wmax
    if wmax > stability['worst_maxabs_W_mps']:
        stability['worst_maxabs_W_mps'] = wmax
        stability['worst_maxabs_W_lead'] = label
    if wmax > W_PHYSICAL_LIMIT:
        stability['physical'] = False
    for fname in ('w', 'theta', 'qv', 'ph', 'u', 'v'):
        arr = G(getattr(st, fname))
        if not np.all(np.isfinite(arr)):
            stability['nan_or_inf'] = True
            stability['physical'] = False
            if stability['first_bad_field'] is None:
                stability['first_bad_field'] = fname
                stability['first_bad_step'] = step
    return wmax


t0 = time.time()
print(f"=== equiv-t2-diag EDMF=ON: DT={DT} ACOUSTIC={ACOUSTIC} RADCAD={RADCAD} radt={RADT_MIN}min ===", flush=True)
for h in leads:
    target = lead_steps[h]
    while start <= target:
        n = min(seg, target - start + 1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                               n_steps=int(n), cadence=cadence)
        jax.block_until_ready(carry.state.theta)
        start += n
        # stability sample after each chunk (cheap; catches mid-lead blow-up)
        wmax_chunk = _audit_stability(carry, f"step{start-1}", start - 1)
        if not stability['physical']:
            print(f"!!! STABILITY BREACH at step {start-1}: max|W|={wmax_chunk:.3g} "
                  f"nan={stability['nan_or_inf']} field={stability['first_bad_field']}", flush=True)
    diags = compute_m9_diagnostics(carry.state, nl, float(h) * 3600.0,
                                   noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad)
    t2 = G(diags.t2)
    swdown_held = G(diags.swdown)   # L1-fixed held SWDOWN (noahmp path -> noahmp_rad[0])
    glw_held = G(diags.glw)
    d = diag_forcing_and_land(carry)
    wll = wrf_lowest_level(lead_labels[h])
    w = wrf_at(lead_labels[h], ('T2', 'Q2', 'U10', 'V10', 'SWDOWN', 'GLW', 'HFX',
                                'LH', 'QFX', 'TSK', 'TSLB', 'SMOIS'))
    wrf_wspd10 = np.sqrt(w['U10'] ** 2 + w['V10'] ** 2)

    # --- L1 SWDOWN residual at radt=30min ---
    gpu_sw = lm(swdown_held); wrf_sw = lm(w['SWDOWN'])
    sw_resid_pct = 100.0 * (gpu_sw - wrf_sw) / wrf_sw if wrf_sw > 1.0 else 0.0

    r = {
        'lead': h, 'label': lead_labels[h][-8:], 'radt_min': RADT_MIN,
        # --- L1 ---
        'gpu_SWDOWN_held_land': gpu_sw, 'wrf_SWDOWN_land': wrf_sw,
        'SWDOWN_resid_pct': sw_resid_pct,
        'gpu_GLW_held_land': lm(glw_held), 'wrf_GLW_land': lm(w['GLW']),
        # --- T2 bias (post-L1) ---
        'gpu_T2_land': lm(t2), 'wrf_T2_land': lm(w['T2']),
        'T2_bias': lm(t2) - lm(w['T2']),
        # --- coupler forcing: GPU vs WRF lowest-model-level (apples-to-apples) ---
        'gpu_sfctmp_land': lm(d['f_sfctmp']), 'wrf_lev0_T_land': lm(wll['sfctmp']),
        'sfctmp_bias_vs_lev0': lm(d['f_sfctmp']) - lm(wll['sfctmp']),
        'gpu_qair_land': lm(d['f_qair']), 'wrf_lev0_QV_land': lm(wll['qair']),
        'qair_bias_vs_lev0': lm(d['f_qair']) - lm(wll['qair']),
        'gpu_wspd_land': lm(d['f_wspd']), 'wrf_lev0_wspd_land': lm(wll['wspd']),
        'wspd_bias_vs_lev0': lm(d['f_wspd']) - lm(wll['wspd']),
        'gpu_zlvl_land': lm(d['f_zlvl']), 'wrf_lev0_zlvl_land': lm(wll['zlvl']),
        'zlvl_bias_vs_lev0': lm(d['f_zlvl']) - lm(wll['zlvl']),
        # --- coupler forcing: GPU vs WRF 2m/10m diagnostics (L2 driver reference) ---
        'wrf_T2_for_forcing': lm(w['T2']), 'sfctmp_bias_vs_T2': lm(d['f_sfctmp']) - lm(w['T2']),
        'wrf_Q2_for_forcing': lm(w['Q2']), 'qair_bias_vs_Q2': lm(d['f_qair']) - lm(w['Q2']),
        'wrf_wspd10_for_forcing': lm(wrf_wspd10), 'wspd_bias_vs_wspd10': lm(d['f_wspd']) - lm(wrf_wspd10),
        # --- radiation forcing fed to land ---
        'gpu_forcing_soldn_land': lm(d['f_soldn']), 'gpu_forcing_lwdn_land': lm(d['f_lwdn']),
        # --- Noah-MP internal surface exchange CHS/CH ---
        'gpu_CHS_land': lm(d['chs']), 'gpu_CH_carry_land': lm(d['ch']),
        # --- fluxes ---
        'gpu_HFX_land': lm(d['hfx']), 'wrf_HFX_land': lm(w['HFX']),
        'HFX_bias': lm(d['hfx']) - lm(w['HFX']),
        'gpu_LH_land': lm(d['lh']), 'wrf_LH_land': lm(w['LH']),
        'LH_bias': lm(d['lh']) - lm(w['LH']),
        'gpu_QFX_land': lm(d['qfx']), 'wrf_QFX_land': lm(w['QFX']),
        'gpu_TSK_land': lm(d['tsk']), 'wrf_TSK_land': lm(w['TSK']),
        # --- evolved carry state vs WRF ---
        'gpu_TG_land': lm(d['tg']), 'gpu_TAH_land': lm(d['tah']), 'gpu_EAH_land': lm(d['eah']),
        'gpu_TV_land': lm(d['tv']),
        'gpu_SH2O0_land': lm(d['sh2o0']), 'gpu_SMOIS0_land': lm(d['smois0']),
        'wrf_SMOIS0_land': lm(w['SMOIS'][0]),
        'gpu_TG_vs_wrf_TSLB0': lm(d['tg']) - lm(w['TSLB'][0]), 'wrf_TSLB0_land': lm(w['TSLB'][0]),
        # --- stability at this lead (full d03 max|W|; NaN/Inf flag) ---
        'maxabs_W_mps_at_lead': float(np.nanmax(np.abs(G(carry.state.w)))),
        'all_finite_at_lead': bool(np.all(np.isfinite(G(carry.state.w))) and
                                   np.all(np.isfinite(G(carry.state.theta))) and
                                   np.all(np.isfinite(G(carry.state.qv)))),
        't_elapsed_s': round(time.time() - t0, 1),
    }
    results.append(r)
    print(json.dumps(r), flush=True)
    # commit-the-moment-it-lands: write incremental results after every lead
    with open('proofs/equiv_t2/equiv_t2_run_edmf_on.json', 'w') as f:
        json.dump({'config': {'DT': DT, 'ACOUSTIC': ACOUSTIC, 'RADCAD': RADCAD,
                              'radt_min': RADT_MIN, 'case': Path(RDIR).name, 'domain': DOM,
                              'edmf': True},
                   'stability': stability, 'leads': results}, f, indent=2)

# --- final stability verdict ---
stability['verdict'] = ('STABLE' if stability['physical'] and not stability['nan_or_inf']
                        else 'UNSTABLE')
print(f"=== STABILITY: {stability['verdict']} | worst max|W|={stability['worst_maxabs_W_mps']:.4g} m/s "
      f"@ {stability['worst_maxabs_W_lead']} | nan_or_inf={stability['nan_or_inf']} "
      f"(limit={W_PHYSICAL_LIMIT} m/s) ===", flush=True)
with open('proofs/equiv_t2/equiv_t2_run_edmf_on.json', 'w') as f:
    json.dump({'config': {'DT': DT, 'ACOUSTIC': ACOUSTIC, 'RADCAD': RADCAD,
                          'radt_min': RADT_MIN, 'case': Path(RDIR).name, 'domain': DOM,
                          'edmf': True},
               'stability': stability, 'leads': results}, f, indent=2)

print("DONE_EQUIV_T2", flush=True)
