"""WRF REAL adapters and thermodynamic expressions for RK/prep operators.

The retained vector expressions are pointwise/stencil JAX operations; XLA
fuses them without a domain-wide column recurrence. Conversions disappear
when the caller already carries WRF REAL state and metrics.
"""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.contracts.state import BaseState

_EOS_INTERPRET = False


def real_state(state):
    names=("u","v","w","theta","p_total","p_perturbation","ph_total",
           "ph_perturbation","mu_total","mu_perturbation","qv","qc","qr","qi","qs","qg")
    return state.replace(_cast=False,**{n:jnp.asarray(getattr(state,n),jnp.float32) for n in names})


def real_metrics(metrics):
    return metrics if metrics.precision=="fp32" else replace(metrics,precision="fp32")


def real_base(base):
    if base is None:
        return None
    return BaseState(**{n:jnp.asarray(getattr(base,n),jnp.float32) for n in BaseState.__slots__})


def inverse_density_fp32(theta,pressure):
    """WRF moist-theta EOS: qvf=1 (use_theta_m=1)."""
    rd,p0,cvpm=jnp.float32(287),jnp.float32(100000),-jnp.float32(717.5)/jnp.float32(1004.5)
    return (rd/p0)*theta*((pressure/p0)**cvpm)


def _diagnose_pressure_reference(state,base,metrics,*,hypsometric_opt=1):
    """WRF calc_p_rho_phi REAL expressions with explicit base state."""
    if base is None:
        alt=inverse_density_fp32(state.theta,state.p_total)
        return state.p_perturbation,jnp.zeros_like(alt),alt
    muts=base.mub+state.mu_perturbation
    c1,c2=metrics.c1h[:,None,None],metrics.c2h[:,None,None]
    dphb=base.phb[1:]-base.phb[:-1]
    if hypsometric_opt==2:
        def alpha(dph,column):
            ptop=jnp.reshape(metrics.p_top,())
            upper=metrics.c3f[1:,None,None]*column+metrics.c4f[1:,None,None]+ptop
            lower=metrics.c3f[:-1,None,None]*column+metrics.c4f[:-1,None,None]+ptop
            half=metrics.c3h[:,None,None]*column+metrics.c4h[:,None,None]+ptop
            return dph/half/jnp.log(lower/upper)
        alb=alpha(dphb,base.mub[None])
        ph=state.ph_perturbation
        # Literal calc_p_rho_phi association; adding full geopotentials first
        # loses a different set of binary32 bits over terrain.
        total=ph[1:]-ph[:-1]+base.phb[1:]-base.phb[:-1]
        al=alpha(total,muts[None])-alb
    else:
        alb=-dphb/(metrics.dnw[:,None,None]*(c1*base.mub[None]+c2))
        al=-(alb*c1*state.mu_perturbation[None]+metrics.rdnw[:,None,None]*
             (state.ph_perturbation[1:]-state.ph_perturbation[:-1]))/(c1*muts[None]+c2)
    alt=al+alb
    p0,rd,cpovcv=jnp.float32(100000),jnp.float32(287),jnp.float32(1004.5)/jnp.float32(717.5)
    pressure=p0*((rd*state.theta)/(p0*alt))**cpovcv-base.pb
    return pressure,al,alt


def _eos(theta,ph,phb,pb,mu,mub,c1,c2,dnw,rdnw,c3h,c4h,c3f,c4f,ptop,
         pressure_out,al_out,alt_out,*,nz,ny,nx,option,block,interpret):
    idx=pl.program_id(0)*block+jnp.arange(block,dtype=jnp.int32)
    x,y,k=idx%nx,idx//nx%ny,idx//(nx*ny)
    valid=idx<nz*ny*nx
    zero=jnp.float32(0)
    rn=partial(_rn,interpret=interpret)
    def vload(ref,shift=0):return pt.load(ref.at[k+shift],mask=valid,other=zero)
    def fload(ref,shift=0):return pt.load(ref.at[k+shift,y,x],mask=valid,other=zero)
    base=pt.load(mub.at[y,x],mask=valid,other=zero)
    perturb=pt.load(mu.at[y,x],mask=valid,other=zero)
    total=rn("add",base,perturb)
    dbase=rn("sub",fload(phb,1),fload(phb))
    if option==2:
        top=jnp.broadcast_to(pt.load(ptop.at[()]),k.shape)
        def alpha(dph,column):
            upper=rn("add",rn("add",rn("mul",vload(c3f,1),column),vload(c4f,1)),top)
            lower=rn("add",rn("add",rn("mul",vload(c3f),column),vload(c4f)),top)
            half=rn("add",rn("add",rn("mul",vload(c3h),column),vload(c4h)),top)
            return rn("div",rn("div",dph,half),jnp.log(rn("div",lower,upper)))
        alb=alpha(dbase,base)
        dph=rn("sub",fload(ph,1),fload(ph))
        dph=rn("sub",rn("add",dph,fload(phb,1)),fload(phb))
        al=rn("sub",alpha(dph,total),alb)
    else:
        mass_base=rn("add",rn("mul",vload(c1),base),vload(c2))
        alb=rn("div",-dbase,rn("mul",vload(dnw),mass_base))
        mass_total=rn("add",rn("mul",vload(c1),total),vload(c2))
        term=rn("mul",rn("mul",alb,vload(c1)),perturb)
        term=rn("add",term,rn("mul",vload(rdnw),rn("sub",fload(ph,1),fload(ph))))
        al=rn("mul",rn("div",-jnp.ones_like(term),mass_total),term)
    alt=rn("add",al,alb)
    rd,p0,cp=jnp.full_like(alt,287.),jnp.full_like(alt,100000.),jnp.full_like(alt,jnp.float32(1.4))
    argument=rn("div",rn("mul",rd,fload(theta)),rn("mul",p0,alt))
    pressure=rn("sub",rn("mul",p0,argument**cp),fload(pb))
    for ref,value in ((pressure_out,pressure),(al_out,al),(alt_out,alt)):
        pt.store(ref.at[k,y,x],value,mask=valid)


def diagnose_pressure_fp32(state,base,metrics,*,hypsometric_opt=1):
    """Literal WRF REAL geometry/EOS; accurate libdevice log/pow, no FMA.

    Native rounded pressure ratios matter when log(pfd/pfu) is small. The
    ordinary XLA fusion reassociates these ratios and pressure error grows.
    """
    if base is None:
        return _diagnose_pressure_reference(state,base,metrics,hypsometric_opt=hypsometric_opt)
    if hypsometric_opt not in (1,2):raise ValueError("hypsometric option must be1/2")
    nz,ny,nx=state.theta.shape
    arrays=(state.theta,state.ph_perturbation,base.phb,base.pb,state.mu_perturbation,base.mub,
            metrics.c1h,metrics.c2h,metrics.dnw,metrics.rdnw,metrics.c3h,metrics.c4h,
            metrics.c3f,metrics.c4f,jnp.reshape(metrics.p_top,()))
    if any(a.dtype!=jnp.float32 for a in arrays):raise TypeError("native EOS requires WRF REAL")
    kernel=partial(_eos,nz=nz,ny=ny,nx=nx,option=hypsometric_opt,block=256,interpret=_EOS_INTERPRET)
    shape=jax.ShapeDtypeStruct(state.theta.shape,jnp.float32)
    return pl.pallas_call(kernel,out_shape=(shape,shape,shape),
        grid=((nz*ny*nx+255)//256,),name="b_diff_eos_real",interpret=_EOS_INTERPRET,
        compiler_params=pt.CompilerParams(num_warps=4))(*arrays)


def _rn(op,a,b,*,interpret):
    if interpret:
        return {"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,
                "div":lambda:a/b}[op]()
    return pt.elementwise_inline_asm(op+".rn.f32 $0, $1, $2;",args=[a,b],
        constraints="=f,f,f",pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape,jnp.float32)])[0]


def _stage_masses(mub,cur,ref,mut,muts,muu,muv,muus,muvs,work,*,ny,nx,rk1,block,interpret):
    idx=pl.program_id(0)*block+jnp.arange(block,dtype=jnp.int32)
    x,y=idx%(nx+1),idx//(nx+1)
    valid=(x<=nx)&(y<=ny)
    zero,half=jnp.float32(0),jnp.float32(.5)
    rn=partial(_rn,interpret=interpret)
    def load(a,xx,yy):
        return pt.load(a.at[jnp.clip(yy,0,ny-1),jnp.clip(xx,0,nx-1)],mask=valid,other=zero)
    def current_mass(axis):
        xx,yy=(x-1,y) if axis==2 else (x,y-1)
        v=rn("add",load(cur,x,y),load(cur,xx,yy))
        v=rn("add",v,load(mub,x,y));v=rn("add",v,load(mub,xx,yy))
        return half*v
    def reference_mass(axis):
        if rk1:
            # WRF assigns MUUS=MUU and MUVS=MUV at RK1. Recomputing an
            # algebraically equal four-term sum changes binary32 rounding.
            return current_mass(axis)
        xx,yy=(x-1,y) if axis==2 else (x,y-1)
        source=ref
        v=rn("add",load(mub,x,y),load(source,x,y))
        v=rn("add",v,load(mub,xx,yy));v=rn("add",v,load(source,xx,yy))
        return half*v
    b,c,r=load(mub,x,y),load(cur,x,y),load(cur if rk1 else ref,x,y)
    pt.store(mut.at[y,x],rn("add",b,c),mask=valid&(x<nx)&(y<ny))
    pt.store(muts.at[y,x],rn("add",b,r),mask=valid&(x<nx)&(y<ny))
    pt.store(work.at[y,x],jnp.zeros_like(c) if rk1 else rn("sub",r,c),mask=valid&(x<nx)&(y<ny))
    pt.store(muu.at[y,x],current_mass(2),mask=valid&(y<ny))
    pt.store(muv.at[y,x],current_mass(1),mask=valid&(x<nx))
    pt.store(muus.at[y,x],reference_mass(2),mask=valid&(y<ny))
    pt.store(muvs.at[y,x],reference_mass(1),mask=valid&(x<nx))


def stage_masses_fp32(mub,cur,ref,*,rk_step,interpret=False,block=256):
    """Literal calc_mu_uv caller and small_step_prep four-term REAL order."""
    if any(a.dtype!=jnp.float32 for a in (mub,cur,ref)):
        raise TypeError("native stage masses require WRF REAL")
    if cur.shape!=mub.shape or ref.shape!=mub.shape:
        raise ValueError("base/current/reference dry mass shapes differ")
    ny,nx=mub.shape
    shapes=((ny,nx),(ny,nx),(ny,nx+1),(ny+1,nx),(ny,nx+1),(ny+1,nx),(ny,nx))
    kernel=partial(_stage_masses,ny=ny,nx=nx,rk1=rk_step==1,block=block,interpret=interpret)
    outputs=pl.pallas_call(kernel,out_shape=tuple(jax.ShapeDtypeStruct(s,jnp.float32) for s in shapes),
        grid=(((ny+1)*(nx+1)+block-1)//block,),name="b_diff_stage_masses",
        interpret=interpret,compiler_params=pt.CompilerParams(num_warps=4))(mub,cur,ref)
    return dict(zip(("mut","muts","muu","muv","muus","muvs","mu_work"),outputs))


def _coupled_work(ref,cur,mref,mcur,c1,c2,maps,out,*,nz,ny,nx,name,block,interpret):
    idx=pl.program_id(0)*block+jnp.arange(block,dtype=jnp.int32)
    x,y,k=idx%nx,idx//nx%ny,idx//(nx*ny)
    valid=idx<nz*ny*nx
    zero=jnp.float32(0)
    rn=partial(_rn,interpret=interpret)
    a=pt.load(c1.at[k],mask=valid,other=zero);b=pt.load(c2.at[k],mask=valid,other=zero)
    mr=pt.load(mref.at[y,x],mask=valid,other=zero);mc=pt.load(mcur.at[y,x],mask=valid,other=zero)
    mr=rn("add",rn("mul",a,mr),b);mc=rn("add",rn("mul",a,mc),b)
    r=pt.load(ref.at[k,y,x],mask=valid,other=zero);c=pt.load(cur.at[k,y,x],mask=valid,other=zero)
    value=rn("sub",rn("mul",mr,r),rn("mul",mc,c))
    if name!="theta":
        factor=pt.load(maps.at[y,x],mask=valid,other=jnp.float32(1))
        if name=="v":
            factor=rn("div",jnp.ones_like(factor),factor)
            value=rn("mul",value,factor)
        else:value=rn("div",value,factor)
    pt.store(out.at[k,y,x],value,mask=valid)


def coupled_work_fp32(ref,cur,mref,mcur,c1,c2,maps,*,name,interpret=False,block=256):
    """WRF binary32 products before subtracting nearly equal coupled fields."""
    arrays=(ref,cur,mref,mcur,c1,c2,maps)
    if any(a.dtype!=jnp.float32 for a in arrays):
        raise TypeError("native coupled work requires WRF REAL")
    nz,ny,nx=cur.shape
    if ref.shape!=cur.shape or mref.shape!=(ny,nx) or mcur.shape!=(ny,nx):
        raise ValueError("inconsistent work/reference mass staggering")
    kernel=partial(_coupled_work,nz=nz,ny=ny,nx=nx,name=name,block=block,interpret=interpret)
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(cur.shape,jnp.float32),
        grid=((nz*ny*nx+block-1)//block,),name="b_diff_coupled_work_"+name,
        interpret=interpret,compiler_params=pt.CompilerParams(num_warps=4))(*arrays)
