"""WRF CLOUDMIX cloud-ice number solve (module_bl_mynnedmf.F:4485-4542).

QNI is the WRF scalar number mixing ratio, kg^-1 dry air.  The caller must
supply the actual DMP updraft flux when mass-flux mixing is active.
"""
import os
import jax.numpy as jnp


def ni_mixing_enabled():
    return os.environ.get('GPUWRF_MYNN_QNI_MIXING','0').strip().lower() in {'1','true','yes','on'}


def mix_ice_number(ni,qi_before,qi_after,state,turb,dt,mf=None):
    from gpuwrf.kernels.phys_mynn_cloudmix import _scalar
    from gpuwrf.physics import mynn_pbl as P
    nmin=jnp.asarray(1e-6,ni.dtype)
    zero=jnp.asarray(0.,ni.dtype)
    before=jnp.maximum(ni,zero)
    before=jnp.where(qi_before>jnp.asarray(1e-12,ni.dtype),jnp.maximum(before,nmin),before)
    kdz=P._rho_interfaces(state,turb['dfh'])
    empty=jnp.zeros(ni.shape[:-1]+(ni.shape[-1]+1,),ni.dtype)
    if mf is None:
        aw,awni=empty,empty
    else:
        aw,awni=mf['s_aw'],mf['s_awqni']
        kdz=P._apply_s_aw_stability_floor(kdz,aw)
    out=_scalar(before,state,dt,kdz,aw,awni,jnp.zeros_like(ni[...,0]))
    return jnp.where(qi_after>jnp.asarray(1e-12,ni.dtype),jnp.maximum(out,nmin),out)
