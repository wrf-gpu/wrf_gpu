"""Whole-MYNN gates on real columns against pristine driver + post-run truth."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.validation.tier2_mynn import INDEPENDENT_TOLERANCE_ABS


def inputs(root, period, domain, *, selected=True, dtype=None, source_forcing=False, input_prefix='columns_p0'):
    meta=json.loads((root/'BP28/manifest.json').read_text())[period+'_'+domain]
    idx=np.array(meta['columns'])
    with np.load(root/f'{input_prefix}_{period}/{domain}_whole.npz') as f:
        def value(key):
            a=np.asarray(f[key])
            if selected and a.ndim:a=a[idx]
            return jnp.asarray(a,dtype=dtype)
        state=P.MynnPBLColumnState(**{
            n: None if n in ('exner', 'ni') and 'state_'+n not in f else value('state_'+n)
            for n in P.MynnPBLColumnState.__slots__})
        flux=SurfaceFluxes(**{n:value('flux_'+n) for n in SurfaceFluxes._fields if 'flux_'+n in f})
        if source_forcing:
            # Reproduce the oracle caller's REAL hfx/qfx round trip and the
            # pristine MYNN driver F:865-876. real_state's FLTV is a dry-flux
            # placeholder; port P608=0.608 also rounds WRF's gas-constant ratio.
            def real(key):
                a=np.asarray(f[key],dtype=np.float32)
                return a[idx] if selected and a.ndim else a
            one=np.float32(1.0)
            rd=np.float32(287.0)
            cp=np.float32(7.0)*rd/np.float32(2.0)
            rcp=rd/cp
            p608=np.float32(461.6)/rd-one
            rho=real('state_rho')[...,0]
            sqv=real('state_qv')[...,0]/(one+np.maximum(real('state_qv')[...,0],np.float32(0.0)))
            cpm=cp*(one+np.float32(0.84)*np.maximum(sqv,np.float32(1e-8)))
            hfx=rho*cpm*real('flux_theta_flux')
            qfx=rho*real('flux_qv_flux')
            flt=hfx/(rho*cpm)
            flqv=qfx/rho
            exner=(real('state_p')[...,0]/np.float32(100000.0))**rcp
            fltv=flt+flqv*p608*(real('flux_t_skin')/exner)
            flux=flux._replace(theta_flux=jnp.asarray(flt,dtype=flux.theta_flux.dtype),
                               qv_flux=jnp.asarray(flqv,dtype=flux.qv_flux.dtype),
                               fltv=jnp.asarray(fltv,dtype=flux.fltv.dtype))
    return state,flux,meta


def compare(root, period, domain, initial, result, *, full_shape=False, oracle_dir='BP29'):
    meta=json.loads((root/'BP28/manifest.json').read_text())[period+'_'+domain]
    idx=np.array(meta['columns'])
    lines=(root/f'{oracle_dir}/{period}_{domain}.output').read_text().splitlines()
    expected=np.array([[list(map(float,line.split())) for line in lines[2+45*p:46+45*p]] for p in range(len(idx))])
    errors={}
    for name,slot in [('u',0),('v',1),('theta',2),('qv',3),('qc',12),('qi',13)]:
        after=np.asarray(getattr(result[0],name),np.float64)
        before=np.asarray(getattr(initial,name),np.float64)
        if full_shape:after,before=after[idx],before[idx]
        delta=(after-before)/meta['dt']-expected[:,:,slot]
        errors[name]=dict(max_abs=float(np.abs(delta).max()),rms=float(np.sqrt(np.mean(delta**2))),
                          pass_frozen=bool(np.abs(delta).max()<=INDEPENDENT_TOLERANCE_ABS))
    diagnostics = {}
    for name, slot, factor in [('tke',4,.5),('kh',5,1),('km',6,1),('el',7,1),
                               ('qsq',8,1),('qc_bl',9,1),('qi_bl',10,1),('cldfra_bl',11,1)]:
        actual=np.asarray(getattr(result[0],name))
        if full_shape:actual=actual[idx]
        truth=expected[:,:,slot]*factor
        difference=np.abs(actual-truth)
        diagnostics[name]=dict(max_abs=float(difference.max()),normalized_max=float(difference.max()/max(1e-30,np.abs(truth).max())))
    pblh=np.asarray(result[1])
    if full_shape:pblh=pblh[idx]
    diagnostics['pblh']=dict(max_abs=float(np.max(np.abs(pblh-np.array([float(lines[1+45*p].split()[0]) for p in range(len(idx))])))))
    nonfinite=int(sum(np.sum(~np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(result)))
    return dict(mean_tendency_errors=errors,diagnostics_without_frozen_gate=diagnostics,mean_gate_pass=all(e['pass_frozen'] for e in errors.values()),
                tolerance=INDEPENDENT_TOLERANCE_ABS,nonfinite=nonfinite,
                comparison_columns=meta['columns'],full_shape=full_shape,
                scope='original pristine mynnedmf driver and verbatim post-run conversion; real P0 inputs')
