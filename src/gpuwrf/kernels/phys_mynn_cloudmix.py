"""Flagged WRF CLOUDMIX=1 scalar solves and conservative moisture repair.

Shared by retained and native MYNN paths. Default-off compatibility is kept
in mynn_pbl; source equations are module_bl_mynnedmf.F:4242-4457,5063-5154.
"""
import os
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def cloudmix_enabled():
    return os.environ.get('GPUWRF_MYNN_CLOUDMIX', '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def pressure_mass_weights(state):
    """WRF F3980-3987 caller mass weights, including the surface bound."""
    dp = state.rho * jnp.asarray(9.81, state.rho.dtype) * state.dz
    surface = jnp.maximum(dp[..., :1], .5 * dp[..., 1:2])
    return jnp.concatenate((surface, dp[..., 1:]), axis=-1)


def _repair(qv, qc, qi, thl, dp, exner, vo, co, io, to, *, nz):
    lane=pl.program_id(0)*128+jnp.arange(128,dtype=jnp.int32)
    col=jnp.minimum(lane,qv.shape[0]-1)
    c=lambda x:jnp.asarray(x,qv.dtype)
    zero=c(0); minimum=c(1e-20); lv=c(2.5e6/(3.5*287));ls=c(2.85e6/(3.5*287))
    def step(i, borrowed):
        k=nz-1-i
        dc=jnp.maximum(-qc[col,k],zero)
        di=jnp.maximum(-qi[col,k],zero)
        v=qv[col,k]-borrowed/dp[col,k]-dc-di
        deficit=jnp.maximum(minimum-v,zero)
        vo[lane,k]=jnp.maximum(v+deficit,minimum)
        co[lane,k]=jnp.maximum(qc[col,k]+dc,zero)
        io[lane,k]=jnp.maximum(qi[col,k]+di,zero)
        to[lane,k]=thl[col,k]+lv/exner[col,k]*dc+ls/exner[col,k]*di
        return deficit*dp[col,k]
    missing=lax.fori_loop(0,nz,step,jnp.zeros((128,),qv.dtype))
    def total(k,value):
        v=vo[lane,k]
        return value+jnp.where(v>c(2)*minimum,v*dp[col,k],zero)
    mass=lax.fori_loop(0,nz,total,jnp.zeros((128,),qv.dtype))
    aa=missing/jnp.maximum(minimum,mass)
    use=(missing/dp[col,0]>minimum)&(aa<c(.5))
    def repay(k,unused):
        v=vo[lane,k]
        vo[lane,k]=v-jnp.where(use & (v>c(2)*minimum),aa*v,zero)
        return unused
    lax.fori_loop(0,nz,repay,jnp.int32(0))


def moisture_check(qv,qc,qi,thl,dp,exner):
    shape=qv.shape;nz=shape[-1]
    values=[v.reshape(-1,nz) for v in (qv,qc,qi,thl,dp,exner)]
    n=values[0].shape[0];padded=((n+127)//128)*128
    out=jax.ShapeDtypeStruct((padded,nz),qv.dtype)
    result=pl.pallas_call(lambda *refs:_repair(*refs,nz=nz),grid=(padded//128,),
                          out_shape=(out,out,out,out),interpret=jax.default_backend()=='cpu')(*values)
    return tuple(v[:n].reshape(shape) for v in result)


def _scalar(x,state,dt,kdz,aw,awx,rhs_bottom):
    from gpuwrf.physics import mynn_pbl as P
    dz=state.dz;dtz=dt/dz;ri=1/jnp.maximum(state.rho,1e-4)
    lower0=-dtz[...,:1]*kdz[...,:1]*ri[...,:1]
    diag0=1+dtz[...,:1]*(kdz[...,:1]+kdz[...,1:2])*ri[...,:1]-.5*dtz[...,:1]*ri[...,:1]*aw[...,1:2]
    upper0=-dtz[...,:1]*ri[...,:1]*(kdz[...,1:2]+.5*aw[...,1:2])
    rhs0=x[...,:1]+rhs_bottom[...,None]-dtz[...,:1]*ri[...,:1]*awx[...,1:2]
    d=dtz[...,1:-1]*ri[...,1:-1]
    a=-d*kdz[...,1:-2]+.5*d*aw[...,1:-2]
    b=1+d*(kdz[...,1:-2]+kdz[...,2:-1])+.5*d*(aw[...,1:-2]-aw[...,2:-1])
    c=-d*kdz[...,2:-1]-.5*d*aw[...,2:-1]
    rhs=x[...,1:-1]+d*(awx[...,1:-2]-awx[...,2:-1])
    zero=jnp.zeros_like(x[...,:1])
    return P._solve_tridiagonal(jnp.concatenate((lower0,a,zero),-1),
        jnp.concatenate((diag0,b,jnp.ones_like(zero)),-1),
        jnp.concatenate((upper0,c,zero),-1),jnp.concatenate((rhs0,rhs,x[...,-1:]),-1))


def apply_mean_cloudmix(state,turb,dt,flux,wind,rhosfc,mf,diss_heat=None):
    from gpuwrf.physics import mynn_pbl as P
    u,v,old_theta,_unused=P._apply_mean_tendencies_legacy(state,turb,dt,flux,wind,rhosfc,mf,diss_heat=diss_heat)
    exner=P._column_exner(state)
    sqv,sqc,sqi,_=P._specific_moisture_components(state)
    thl=old_theta-P.XLVCP_MYNN/exner*sqc-P.XLSCP_MYNN/exner*sqi
    kdz=P._rho_interfaces(state,turb['dfh'])
    zero=jnp.zeros_like(wind)
    interfaces=jnp.zeros(sqv.shape[:-1]+(sqv.shape[-1]+1,),sqv.dtype)
    aw=interfaces if mf is None else mf['s_aw']
    if mf is not None:kdz=P._apply_s_aw_stability_floor(kdz,aw)
    # Cloud water includes updraft flux; ice has only local mixing (WRF:4390).
    cq=interfaces if mf is None else mf['s_awqc']
    sqc2=_scalar(sqc,state,dt,kdz,aw,cq,zero)
    sqi2=_scalar(sqi,state,dt,kdz,interfaces,interfaces,zero)
    ni2=None
    if state.ni is not None:
        from gpuwrf.kernels.phys_mynn_ni import mix_ice_number
        # WRF applies the number floor before the final moisture_check.
        ni2=mix_ice_number(state.ni,sqi,sqi2,state,turb,dt,mf)
    qflux=jnp.maximum(flux.qv_flux,jnp.minimum(.9*sqv[...,0]-1e-8,0)/(dt/state.dz[...,0]))
    bottom=dt/state.dz[...,0]*rhosfc*qflux/jnp.maximum(state.rho[...,0],1e-4)
    vq=interfaces if mf is None else mf['s_awqv']
    sqv2=_scalar(sqv,state,dt,kdz,aw,vq,bottom)
    # WRF pressure-mass weights; gravity cancels in ratios but keep its source value.
    dp=pressure_mass_weights(state)
    sqv2,sqc2,sqi2,thl=moisture_check(sqv2,sqc2,sqi2,thl,dp,exner)
    theta=thl+P.XLVCP_MYNN/exner*sqc2+P.XLSCP_MYNN/exner*sqi2
    qv=sqv2/(1-sqv2)
    # Verbatim MYNN driver post-run conversion, including its specific-qv factor.
    qc=sqc2*(1+sqv2);qi=sqi2*(1+sqv2)
    result=(u,v,theta,qv,qc,qi)
    if ni2 is not None:
        result=result+(ni2,)
    return result
