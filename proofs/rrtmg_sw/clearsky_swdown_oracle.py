"""Clear-sky RRTMG-SW COLUMN ORACLE (CODE-vs-STATE disambiguation).

Feeds the EXACT WRF profile (corpus wrfout T/QVAPOR/P/PB + WRF COSZEN + WRF
ALBEDO) into the GPU `solve_rrtmg_sw_column` CLEAR-SKY path (clouds zeroed) and
compares the GPU surface-down shortwave to WRF's RRTMG SWDOWN at three solar
zeniths (09z/12z/15z = lead 15/18/21 in the L3 d03 run).

If GPU != WRF on IDENTICAL inputs -> RRTMG CODE bug (localize the term).
If GPU == WRF on IDENTICAL inputs -> it is a STATE difference (do NOT fudge).

WRF SWDOWN already includes the (near-zero) cloud field here (CLDFRA mean
~0.002, QICE 0), so a clear-sky GPU comparison is apples-to-apples to within the
tiny cloud residual.  We restrict to land columns (XLAND<1.5) to match the
HFX/T2 localization, and break the bias out vs air mass (1/coszen)."""
import argparse
import sys, os, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
import netCDF4 as nc
import jax
import jax.numpy as jnp
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column

RDIR = '/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z'
DOM = 'd03'
GRAVITY = 9.81
RD = 287.0
P00 = 100000.0
CP = 1004.5
RCP = RD / CP
RRSW_SCON = 1368.22

LEADS = [
    (15, 'wrfout_d03_2026-05-22_09:00:00', '09z'),
    (18, 'wrfout_d03_2026-05-22_12:00:00', '12z'),
    (21, 'wrfout_d03_2026-05-22_15:00:00', '15z'),
]


def load_wrf_columns(path):
    ds = nc.Dataset(path)
    g = lambda k: np.squeeze(np.asarray(ds.variables[k][:]))
    Tpert = g('T')               # (nz,ny,nx) perturbation potential temp
    P = g('P'); PB = g('PB')     # perturbation + base pressure (Pa)
    QV = g('QVAPOR')             # (nz,ny,nx)
    QC = g('QCLOUD'); QI = g('QICE'); QS = g('QSNOW'); QG = g('QGRAUP')
    CLDFRA = g('CLDFRA')
    PH = g('PH'); PHB = g('PHB')  # geopotential on w-levels (nz+1,ny,nx)
    COSZEN = g('COSZEN')         # (ny,nx)
    ALBEDO = g('ALBEDO')         # (ny,nx) WRF surface albedo (used by RRTMG)
    SWDOWN = g('SWDOWN')
    SWDNB = g('SWDNB') if 'SWDNB' in ds.variables else SWDOWN
    SWDNBC = g('SWDNBC') if 'SWDNBC' in ds.variables else SWDOWN
    SWDNTC = g('SWDNTC') if 'SWDNTC' in ds.variables else None
    SWNORM = g('SWNORM')
    o3_gfs_du = g('O3_GFS_DU') if 'O3_GFS_DU' in ds.variables else None
    with nc.Dataset(RDIR + '/wrfinput_' + DOM) as inp:
        XLAND = np.squeeze(np.asarray(inp.variables['XLAND'][:]))
    ds.close()

    p_full = (P + PB)                                  # (nz,ny,nx) Pa
    theta = Tpert + 300.0                              # WRF base theta = 300
    T = theta * (p_full / P00) ** RCP                  # absolute temp K
    z_w = (PH + PHB) / GRAVITY                          # (nz+1,ny,nx) geopotential height
    dz = z_w[1:, :, :] - z_w[:-1, :, :]                # (nz,ny,nx)
    rho = p_full / (RD * T * (1.0 + 0.61 * QV))
    return dict(T=T, p=p_full, qv=QV, qc=QC, qi=QI, qs=QS, qg=QG, cldfra=CLDFRA, dz=dz, rho=rho,
                coszen=COSZEN, albedo=ALBEDO, swdown=SWDOWN, swdnb=SWDNB, swdnbc=SWDNBC,
                swdntc=SWDNTC, swnorm=SWNORM, o3_gfs_du=o3_gfs_du, xland=XLAND)


def to_columns(arr3d):
    """(nz,ny,nx) -> (ny*nx, nz) bottom-to-top mass-level columns (GPU layout)."""
    nz, ny, nx = arr3d.shape
    return np.moveaxis(arr3d, 0, -1).reshape(ny * nx, nz)


def select_columns(w, max_columns):
    """Returns flattened column indices using deterministic coszen-stratified sampling."""

    flat = np.arange(w['coszen'].size)
    land_day = (w['xland'].reshape(-1) < 1.5) & (w['coszen'].reshape(-1) > 0.05) & (w['swdown'].reshape(-1) > 1.0)
    candidates = flat[land_day]
    if max_columns is None or max_columns <= 0 or candidates.size <= max_columns:
        return candidates

    coszen = w['coszen'].reshape(-1)[candidates]
    order = np.argsort(coszen)
    sorted_candidates = candidates[order]
    bins = np.array_split(sorted_candidates, min(8, max_columns))
    per_bin = max(1, max_columns // len(bins))
    chosen = []
    for chunk in bins:
        if chunk.size == 0:
            continue
        take = min(per_bin, chunk.size)
        positions = np.linspace(0, chunk.size - 1, take, dtype=int)
        chosen.extend(chunk[positions].tolist())
    if len(chosen) < max_columns:
        remaining = [idx for idx in sorted_candidates.tolist() if idx not in set(chosen)]
        chosen.extend(remaining[: max_columns - len(chosen)])
    return np.asarray(chosen[:max_columns], dtype=np.int64)


def run_solver_batched(T, p, qv, qc, qi, qs, qg, cldfra, albedo, coszen, dz, rho, solar_source_scale, batch_size):
    outputs = []
    for start in range(0, T.shape[0], batch_size):
        end = min(start + batch_size, T.shape[0])
        st = RRTMGSWColumnState(
            jnp.asarray(T[start:end]), jnp.asarray(p[start:end]), jnp.asarray(qv[start:end]),
            jnp.asarray(qc[start:end]), jnp.asarray(qi[start:end]), jnp.asarray(qs[start:end]), jnp.asarray(qg[start:end]),
            jnp.asarray(cldfra[start:end]), jnp.asarray(albedo[start:end]), jnp.asarray(coszen[start:end]),
            jnp.asarray(dz[start:end]), jnp.asarray(rho[start:end]),
            solar_source_scale=jnp.asarray(solar_source_scale[start:end]),
        )
        sw = solve_rrtmg_sw_column(st, debug=False)
        outputs.append((
            np.asarray(jax.device_get(sw.surface_down)),
            np.asarray(jax.device_get(sw.surface_direct)),
            np.asarray(jax.device_get(sw.toa_down)),
        ))
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*outputs, strict=True))


def pair_metrics(gpu, wrf, mask):
    diff = gpu[mask] - wrf[mask]
    wrf_sel = wrf[mask]
    gpu_sel = gpu[mask]
    return {
        'wrf_mean': float(np.mean(wrf_sel)),
        'gpu_mean': float(np.mean(gpu_sel)),
        'bias_Wm2': float(np.mean(diff)),
        'bias_pct': float(100.0 * np.mean(diff) / max(np.mean(wrf_sel), 1e-6)),
        'rmse_Wm2': float(np.sqrt(np.mean(diff * diff))),
        'mae_Wm2': float(np.mean(np.abs(diff))),
        'max_abs_Wm2': float(np.max(np.abs(diff))),
    }


def run_oracle(clear_sky=True, max_columns=0, batch_size=512, source_scale_mode='wrf-toa'):
    results = []
    for lead, fname, label in LEADS:
        w = load_wrf_columns(RDIR + '/' + fname)
        ny, nx = w['coszen'].shape
        indices = select_columns(w, max_columns=max_columns)
        T = to_columns(w['T']).astype(np.float64)[indices]
        p = to_columns(w['p']).astype(np.float64)[indices]
        qv = to_columns(w['qv']).astype(np.float64)[indices]
        if clear_sky:
            qc = np.zeros_like(T); qi = np.zeros_like(T); qs = np.zeros_like(T); qg = np.zeros_like(T)
            cldfra = np.zeros((T.shape[0], T.shape[1]), dtype=np.float64)
        else:
            qc = to_columns(w['qc']).astype(np.float64)[indices]; qi = to_columns(w['qi']).astype(np.float64)[indices]
            qs = to_columns(w['qs']).astype(np.float64)[indices]; qg = to_columns(w['qg']).astype(np.float64)[indices]
            cldfra = to_columns(w['cldfra']).astype(np.float64)[indices]
        dz = to_columns(w['dz']).astype(np.float64)[indices]
        rho = to_columns(w['rho']).astype(np.float64)[indices]
        coszen = w['coszen'].reshape(-1).astype(np.float64)[indices]
        albedo = w['albedo'].reshape(-1).astype(np.float64)[indices]
        if source_scale_mode == 'unit':
            solar_source_scale = np.ones_like(coszen)
        elif source_scale_mode == 'wrf-toa':
            if w['swdntc'] is None:
                raise RuntimeError('WRF SWDNTC is required for --source-scale wrf-toa')
            wrf_swdntc_selected = w['swdntc'].reshape(-1).astype(np.float64)[indices]
            solar_source_scale = wrf_swdntc_selected / np.maximum(coszen * RRSW_SCON, 1e-6)
        else:
            raise ValueError(source_scale_mode)

        gpu_swdown, gpu_direct, gpu_toa = run_solver_batched(
            T, p, qv, qc, qi, qs, qg, cldfra, albedo, coszen, dz, rho,
            solar_source_scale=solar_source_scale, batch_size=batch_size
        )

        wrf_swdown = w['swdown'].reshape(-1)[indices]
        wrf_swdnb = w['swdnb'].reshape(-1)[indices]
        wrf_swdnbc = w['swdnbc'].reshape(-1)[indices]
        wrf_swdntc = w['swdntc'].reshape(-1)[indices] if w['swdntc'] is not None else np.full_like(wrf_swdown, np.nan)
        wrf_cldfra = to_columns(w['cldfra']).astype(np.float64)[indices]
        sel = wrf_swdown > 1.0
        cz = coszen

        def m(a):
            return float(np.mean(a[sel]))

        airmass = 1.0 / np.maximum(cz, 1e-3)
        r = {
            'lead': lead, 'label': label,
            'source_file': RDIR + '/' + fname,
            'selection': {
                'method': 'all land/daylit columns' if max_columns is None or max_columns <= 0 else 'deterministic coszen-stratified land/daylit subset',
                'selected_columns': int(indices.size),
                'selected_daylit_columns': int(sel.sum()),
                'domain_shape': [int(ny), int(nx)],
                'max_columns_requested': int(max_columns or 0),
                'batch_size': int(batch_size),
                'source_scale_mode': source_scale_mode,
            },
            'coszen_mean': m(cz), 'airmass_mean': m(airmass),
            'wrf_SWDOWN': m(wrf_swdown), 'wrf_SWDNB': m(wrf_swdnb), 'wrf_SWDNBC_clear': m(wrf_swdnbc),
            'wrf_SWDNTC_toa_clear': m(wrf_swdntc),
            'gpu_SWDOWN': m(gpu_swdown),
            'bias_Wm2': m(gpu_swdown - wrf_swdown),
            'bias_pct': 100.0 * m(gpu_swdown - wrf_swdown) / m(wrf_swdown),
            'vs_SWDOWN': pair_metrics(gpu_swdown, wrf_swdown, sel),
            'vs_SWDNB': pair_metrics(gpu_swdown, wrf_swdnb, sel),
            'vs_SWDNBC_clear': pair_metrics(gpu_swdown, wrf_swdnbc, sel),
            'vs_SWDNTC_toa_clear': pair_metrics(gpu_toa, wrf_swdntc, sel),
            'gpu_direct': m(gpu_direct), 'gpu_TOA_down': m(gpu_toa),
            'solar_source_scale_mean': m(solar_source_scale),
            'solar_source_scale_min': float(np.min(solar_source_scale[sel])),
            'solar_source_scale_max': float(np.max(solar_source_scale[sel])),
            'wrf_SWNORM': m(w['swnorm'].reshape(-1)[indices]),
            'albedo_mean': m(albedo),
            'wrf_cldfra_mean': float(np.mean(wrf_cldfra[sel])),
            'wrf_cldfra_max': float(np.max(wrf_cldfra[sel])),
            'o3_source': 'GPU WRF climatological ozone profile; corpus exposes only O3_GFS_DU total column',
        }
        cz_sel = cz[sel]; bias_sel = (gpu_swdown - wrf_swdnbc)[sel]
        wrf_sel = wrf_swdnbc[sel]
        q = np.quantile(cz_sel, [0.0, 0.25, 0.5, 0.75, 1.0])
        bins = []
        for i in range(4):
            bsel = (cz_sel >= q[i]) & (cz_sel <= q[i + 1])
            if bsel.sum() > 0:
                bins.append({
                    'coszen_lo': float(q[i]), 'coszen_hi': float(q[i + 1]),
                    'n': int(bsel.sum()),
                    'mean_bias_pct': float(100.0 * bias_sel[bsel].mean() / max(wrf_sel[bsel].mean(), 1e-6)),
                    'mean_bias_Wm2': float(bias_sel[bsel].mean()),
                })
        r['airmass_bins'] = bins
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'airmass_bins'}))
        for b in bins:
            print('  bin cz[%.3f,%.3f] n=%d bias=%.2f%% (%.1f W/m2)' % (
                b['coszen_lo'], b['coszen_hi'], b['n'], b['mean_bias_pct'], b['mean_bias_Wm2']))
        sys.stdout.flush()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='clear', choices=('clear', 'allsky'))
    parser.add_argument('--max-columns', type=int, default=0, help='0 means all land/daylit columns; positive values use a deterministic coszen-stratified subset.')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--source-scale', default='wrf-toa', choices=('wrf-toa', 'unit'),
                        help='wrf-toa applies WRF SWDNTC/(COSZEN*1368.22); unit reproduces the old fixed-table source.')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'clearsky_swdown_oracle.json'))
    args = parser.parse_args()
    out = run_oracle(clear_sky=(args.mode != 'allsky'), max_columns=args.max_columns,
                     batch_size=args.batch_size, source_scale_mode=args.source_scale)
    outpath = args.out
    with open(outpath, 'w') as f:
        json.dump({
            'mode': args.mode,
            'source_scale': args.source_scale,
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'command_hint': 'taskset -c 0-3 env JAX_PLATFORMS=<cpu|cuda> PYTHONPATH=src python proofs/rrtmg_sw/clearsky_swdown_oracle.py clear --source-scale wrf-toa --max-columns <N> --batch-size <N>',
            'wrf_source': '$WRF_PRISTINE_ROOT/phys/module_ra_rrtmg_sw.F',
            'results': out,
        }, f, indent=2)
    print('WROTE', outpath)
    print('DONE_ORACLE')
