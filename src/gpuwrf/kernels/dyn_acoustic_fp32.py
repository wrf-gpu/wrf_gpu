"""Phase-1 native-fp32 acoustic prototype behind AcousticCoreState.

Four ordered phases: UV stencil, column mass/theta, implicit W/phi, and EOS.
The implicit formulas reuse the accepted fp64 Pallas kernels with masked,
unpacked references. Existing fp64 APIs are untouched. The optional boundary
adapter supports the separately tested operational flag work; the enabler gate
exercises native interior kernels. Multi-GPU halo exchange is outside its scope.
"""
from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState
from gpuwrf.kernels import fused_vertical_implicit as vi

# Stage invariants are held outside the recurrence. theta_ave is a diagnostic
# alias of theta, reconstructed by the adapter instead of returned twice.
EVOLVING_FIELDS=("u","v","w","ph","mu","mu_work","muts","muave","mudf","ww",
                 "theta","theta_coupled_work","t_2ave","p","al","pm1","ru_m","rv_m","ww_m")


def evolving_payload(state):
    return tuple(getattr(state,name) for name in EVOLVING_FIELDS)


def state_from_payload(template,payload):
    updates=dict(zip(EVOLVING_FIELDS,payload,strict=True))
    return template.replace(**updates,theta_ave=updates["theta"])


def _load(ref, idx, mask=True, other=0.):
    return pt.load(ref.at[idx], mask=mask, other=other)


def _store(ref, idx, value, mask=True):
    pt.store(ref.at[idx], value, mask=mask)


class _ColumnRef:
    """View a raw SoA ref as packed columns without a device packing copy."""
    def __init__(self, ref, ny, nx, *, x_offset=0, y_offset=0):
        self.ref, self.ny, self.nx = ref, ny, nx
        self.x_offset, self.y_offset = x_offset, y_offset
        self.dtype = ref.dtype
        self.shape = ((ref.shape[0], ny*nx) if len(ref.shape) == 3 else (ny*nx,))

    def _index(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        col = idx[-1]
        if isinstance(col, pl.Slice):
            col = col.start + jnp.arange(col.size, dtype=jnp.int32) * col.stride
        mask = (col >= 0) & (col < self.ny*self.nx)
        y, x = col // self.nx + self.y_offset, col % self.nx + self.x_offset
        indices = (idx[0], y, x) if len(self.ref.shape) == 3 else (y, x)
        return indices, mask

    def __getitem__(self, idx):
        indices, mask = self._index(idx)
        return _load(self.ref, indices, mask, other=1.)

    def __setitem__(self, idx, value):
        indices, mask = self._index(idx)
        _store(self.ref, indices, value, mask)


# Outputs each kernel reads (if at all) only at the index it writes: aliasing the
# input buffer is race-free and lets XLA write the loop slot in place (no copy).
_POINTWISE_ALIASES = {
    "b_core_mass_fp32": ("mu", "mu_work", "muts", "mudf", "muave", "ww", "ww_m",
                         "theta_coupled_work", "t_2ave"),
    "b_core_pressure_fp32": ("p", "pm1", "al", "theta"),
    "b_core_w_p1_fp32": ("t_2ave",),
}


def _call(kernel, inputs, outputs, grid, *, name, interpret=False, warps=4):
    keys = tuple(inputs)
    outkeys = tuple(outputs)
    def body(*refs):
        kernel(dict(zip(keys, refs[:len(keys)], strict=True)),
               dict(zip(outkeys, refs[len(keys):], strict=True)))
    aliases = {}
    if os.environ.get("GPUWRF_ACOUSTIC_ALIAS", "0") == "1":
        aliases = {keys.index(field): outkeys.index(field)
                   for field in _POINTWISE_ALIASES.get(name, ())
                   if field in keys and field in outkeys
                   and inputs[field].shape == outputs[field].shape
                   and inputs[field].dtype == outputs[field].dtype}
    values = pl.pallas_call(body, grid=grid, name=name, interpret=interpret,
                           compiler_params=pt.CompilerParams(num_warps=warps),
                           input_output_aliases=aliases,
                           out_shape=tuple(outputs.values()))(*inputs.values())
    return dict(zip(outkeys, values, strict=True))


def _shape(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def _check(state, cfg,*,apply_boundary_forcing=False):
    if os.environ.get("GPUWRF_ADVANCE_W_SAFE_FLOORS", "0") == "1":
        raise ValueError("debug advance_w floors are outside the prototype contract")
    for name, value in state.to_dict().items():
        if value is not None and jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != jnp.float32:
            raise TypeError(f"acoustic_fp32 requires fp32 {name}; got {value.dtype}")
    for name in ("u_work_bdy", "v_work_bdy", "ph_bdy_target", "w_spec_target",
                 "mu_spec_target", "u_spec_tan_target", "v_spec_tan_target"):
        if getattr(state, name) is not None and not apply_boundary_forcing:
            raise ValueError(f"boundary forcing {name} is outside the phase-1 prototype")
    if cfg.spec_w_zero_grad and not apply_boundary_forcing:
        raise ValueError("specified W boundary projection is outside the phase-1 prototype")
    required = ("c2a", "cqw", "alt", "phb", "ph_1", "ht", "c1f", "c2f",
                "rdn", "rw_tend_pg_buoy", "theta_coupled_work", "u_tend", "v_tend",
                "p_base", "ph_base", "al", "cqu", "cqv", "msfux", "msfvx",
                "msfvy", "cf1", "cf2", "cf3", "php_stage", "pm1", "ru_m", "rv_m", "ww_m", "mu_work")
    for name in required:
        if getattr(state, name) is None:
            raise ValueError(f"prototype requires staged {name}")
    if cfg.w_damping not in (0, 1):
        raise ValueError("w_damping must be 0 or 1")


def _uv_kernel(r, out, *, nz, ny, nx, block, cfg):
    uv_masked = os.environ.get("GPUWRF_ACOUSTIC_UV_MASKED", "0") == "1"
    flat = pl.program_id(0)*block + jnp.arange(block, dtype=jnp.int32)
    dt, dx, dy, emdiv = (r["s"][i] for i in range(4))
    for axis in ("u", "v"):
        fy, fx = (ny, nx+1) if axis == "u" else (ny+1, nx)
        k, cell = flat // (fy*fx), flat % (fy*fx)
        y, x = cell // fx, cell % fx
        valid = flat < nz*fy*fx
        active = valid
        if cfg.specified or cfg.nested:
            if axis == "u":
                active = active & (y >= cfg.spec_zone) & (y < ny-cfg.spec_zone)
                if not cfg.periodic_x:
                    active = active & (x >= cfg.spec_zone) & (x <= nx-cfg.spec_zone)
            else:
                active = active & (y >= cfg.spec_zone) & (y <= ny-cfg.spec_zone)
                if not cfg.periodic_x:
                    active = active & (x >= cfg.spec_zone) & (x < nx-cfg.spec_zone)
        if axis == "u":
            yl = yr = y
            xl, xr = jnp.maximum(x-1, 0), jnp.minimum(x, nx-1)
            ratio = _load(r["msfux"], (y, x), valid) / _load(r["msfuy"], (y, x), valid, 1.)
            mass = _load(r["muu"], (y, x), valid)
            dd = dx
        else:
            xl = xr = x
            yl, yr = jnp.maximum(y-1, 0), jnp.minimum(y, ny-1)
            ratio = _load(r["msfvy"], (y, x), valid) / _load(r["msfvx"], (y, x), valid, 1.)
            mass = _load(r["muv"], (y, x), valid)
            dd = dy
        def left(name, kk=k):
            return _load(r[name], (kk, yl, xl), valid)
        def right(name, kk=k):
            return _load(r[name], (kk, yr, xr), valid)
        def psum(kk):
            return left("p", kk) + right("p", kk)
        cf1, cf2, cf3 = (r[name][()] for name in ("cf1", "cf2", "cf3"))
        def dpn(kk):
            interior_k = jnp.minimum(jnp.maximum(kk, 1), nz-1)
            if uv_masked:
                # Each branch loads p only on the lanes that select it (same values there).
                def mpsum(kp, m):
                    return (_load(r["p"], (kp, yl, xl), valid & m)
                            + _load(r["p"], (kp, yr, xr), valid & m))
                mid, bot, tp = (kk > 0) & (kk < nz), kk == 0, kk == nz
                middle = .5 * (_load(r["fnm"], (interior_k,), valid) * mpsum(interior_k, mid)
                               + _load(r["fnp"], (interior_k,), valid) * mpsum(interior_k-1, mid))
                bottom = .5 * (cf1*mpsum(0, bot) + cf2*mpsum(1, bot) + cf3*mpsum(2, bot))
                top = (.5*(cf1*mpsum(nz-1, tp) + cf2*mpsum(nz-2, tp) + cf3*mpsum(nz-3, tp))
                       if cfg.top_lid else jnp.zeros_like(middle))
                return jnp.where(kk == 0, bottom, jnp.where(kk == nz, top, middle))
            middle = .5 * (_load(r["fnm"], (interior_k,), valid) * psum(interior_k)
                           + _load(r["fnp"], (interior_k,), valid) * psum(interior_k-1))
            bottom = .5 * (cf1*psum(0) + cf2*psum(1) + cf3*psum(2))
            top = (.5*(cf1*psum(nz-1) + cf2*psum(nz-2) + cf3*psum(nz-3))
                   if cfg.top_lid else jnp.zeros_like(middle))
            return jnp.where(kk == 0, bottom, jnp.where(kk == nz, top, middle))
        c1, c2 = _load(r["c1h"], (k,), valid), _load(r["c2h"], (k,), valid)
        ph_term = (right("ph", k+1)-left("ph", k+1)) + (right("ph")-left("ph"))
        p_term = (left("alt")+right("alt"))*(right("p")-left("p"))
        pb_term = (left("al")+right("al"))*(right("p_base")-left("p_base"))
        grad = ratio*.5/dd * (c1*mass+c2) * (ph_term+p_term+pb_term)
        mu_l = _load(r["mu_work"], (yl,xl),valid)
        mu_r = _load(r["mu_work"], (yr,xr),valid)
        bracket = _load(r["rdnw"],(k,),valid)*(dpn(k+1)-dpn(k)) - .5*(c1*mu_l+c1*mu_r)
        grad = grad + ratio/dd*(right("php_stage")-left("php_stage"))*bracket
        val = _load(r[axis],(k,y,x),valid) + dt*_load(r[axis+"_tend"],(k,y,x),valid)
        val = val - dt*_load(r["cq"+axis],(k,y,x),valid)*grad
        mudf_diff = _load(r["mudf"],(yr,xr),valid)-_load(r["mudf"],(yl,xl),valid)
        damping = (-emdiv*dd*mudf_diff / _load(r["msfuy"],(y,x),valid,1.) if axis == "u" else
                   -emdiv*dd*mudf_diff * _load(r["msfvx_inv"],(y,x),valid))
        val = val + c1*damping
        val = jnp.where(active,val,_load(r[axis],(k,y,x),valid))
        _store(out[axis],(k,y,x),val,valid)
        accum = "ru_m" if axis == "u" else "rv_m"
        _store(out[accum],(k,y,x),_load(r[accum],(k,y,x),valid)+val,valid)


_MASS_BLOCK = 4


def _level_blocks(n, load, step, carry, size=_MASS_BLOCK):
    """Levels 0..n-1 in order, ``size`` per loop trip: all loads of a trip are issued
    before its first store (Triton keeps loads behind possibly-aliasing stores)."""
    def trip(b, c):
        ks = [b*size + u for u in range(size)]
        operands = [load(k) for k in ks]
        for k, d in zip(ks, operands):
            c = step(k, d, c)
        return c
    carry = jax.lax.fori_loop(0, n//size, trip, carry)
    for k in range(n//size*size, n):
        carry = step(k, load(k), carry)
    return carry


def _mass_kernel(r, out, *, nz, ny, nx, block, cfg, interpret=False):
    col = pl.program_id(0)*block + jnp.arange(block,dtype=jnp.int32)
    y, x = col//nx, col%nx
    valid = col < ny*nx
    active = valid
    if cfg.specified or cfg.nested:
        active = active & (y > 0) & (y < ny-1)
        if not cfg.periodic_x:
            active = active & (x > 0) & (x < nx-1)
    def l2(name, yy=y, xx=x, other=0.):
        return _load(r[name], (yy,xx), valid, other)
    def l3(name,k,yy=y,xx=x):
        return _load(r[name],(k,yy,xx),valid)
    def v1(name,k):
        return r[name][k]
    dts, rdx, rdy, eps = (r["s"][i] for i in range(4))
    mx,my = l2("msftx"),l2("msfty",other=1.)
    muu_w,muu_e = l2("muu"),l2("muu",y,x+1)
    muv_s,muv_n = l2("muv"),l2("muv",y+1,x)
    msuy_w,msuy_e = l2("msfuy",other=1.),l2("msfuy",y,x+1,1.)
    msvi_s,msvi_n = l2("msfvx_inv"),l2("msfvx_inv",y+1,x)
    mul=lambda a,b:a*b
    add=lambda a,b:a+b
    divide=lambda a,b:a/b
    def div(k):
        c1,c2 = v1("c1h",k),v1("c2h",k)
        vn = add(l3("v",k,y+1,x),mul(mul(add(mul(c1,muv_n),c2),l3("v_1",k,y+1,x)),msvi_n))
        vs = add(l3("v",k),mul(mul(add(mul(c1,muv_s),c2),l3("v_1",k)),msvi_s))
        ue = add(l3("u",k,y,x+1),divide(mul(add(mul(c1,muu_e),c2),l3("u_1",k,y,x+1)),msuy_e))
        uw = add(l3("u",k),divide(mul(add(mul(c1,muu_w),c2),l3("u_1",k)),msuy_w))
        value=mul(mul(mx,my),add(mul(rdy,vn-vs),mul(rdx,ue-uw)))
        if "debug_dvdxi" in out:
            for name,v in dict(dvdxi=value,vn=vn,vs=vs,ue=ue,uw=uw).items():
                _store(out["debug_"+name],(k,y,x),v,valid)
        return value
    blocked = os.environ.get("GPUWRF_ACOUSTIC_MASS_BLOCK", "0") == "1" and "debug_dvdxi" not in out
    def reduce(k, total):
        return total + v1("dnw",k)*div(k)
    if blocked:
        # Same sequential sum, MASS_BLOCK levels per trip: the block's loads are
        # independent of the running total and issue together (latency-bound loop).
        dmdt = _level_blocks(nz, lambda k: None, lambda k, _, total: reduce(k, total),
                             jnp.zeros((block,),jnp.float32))
    else:
        dmdt = jax.lax.fori_loop(0,nz,reduce,jnp.zeros((block,),jnp.float32))
    mut, old_muts = l2("mut"),l2("muts")
    old_work = l2("mu_work")
    mu_tend = l2("mu_tend")
    tendency = dmdt+mu_tend
    scale = jnp.ones_like(tendency)
    if cfg.specified or cfg.nested:
        floor = jnp.maximum(1.,.5*mut)
        delta = dts*tendency
        allowed = jnp.maximum(mut+old_work-floor,0.)
        finite = jnp.isfinite(delta) & jnp.isfinite(mut+old_work)
        scale = jnp.where(finite & (-delta > allowed), allowed/jnp.maximum(-delta,1.),
                          jnp.where(finite,1.,0.))
        scale = jnp.clip(scale,0.,1.)
    _store(out["guard_events"],(y,x),((scale != 1.) & active).astype(jnp.int32),valid)
    tendency,dmdt,mu_tend = tendency*scale,dmdt*scale,mu_tend*scale
    new_work = old_work+dts*tendency
    values = dict(mu=(l2("mu")-old_work)+new_work, mu_work=new_work, muts=mut+new_work, mudf=tendency,
                  muave=.5*((1.+eps)*new_work+(1.-eps)*old_work))
    for name,value in values.items():
        _store(out[name],(y,x),jnp.where(active,value,l2(name)),valid)
    # Carry the column recurrence in registers; no level-wide scan launches.
    ww0 = l3("ww",0)
    def level(k, carry):
        ww_prev, wdtn_prev = carry
        if cfg.periodic_x:
            xp,xm = (x+1)%nx,(x-1)%nx
        else:
            xp,xm = jnp.minimum(x+1,nx-1),jnp.maximum(x-1,0)
        if not (cfg.specified or cfg.nested):
            yp,ym = (y+1)%ny,(y-1)%ny
        else:
            yp,ym = jnp.minimum(y+1,ny-1),jnp.maximum(y-1,0)
        ww_k = ww_prev - l3("ww_1",k)
        increment = v1("dnw",k)*(v1("c1h",k)*dmdt+div(k)*scale+v1("c1h",k)*mu_tend)/my
        ww_next_raw = ww_prev-increment
        kp = jnp.minimum(k+1,nz-1)
        ww_next = ww_next_raw-l3("ww_1",kp)
        wdtn_next = ww_next*(v1("fnm",kp)*l3("theta_1",kp)+v1("fnp",kp)*l3("theta_1",k))
        wdtn_next = jnp.where(k < nz-1,wdtn_next,0.)
        th = l3("theta_1",k)
        vf = l3("v",k,y+1,x)*(l3("theta_1",k,yp,x)+th)-l3("v",k)*(th+l3("theta_1",k,ym,x))
        uf = l3("u",k,y,x+1)*(l3("theta_1",k,y,xp)+th)-l3("u",k)*(th+l3("theta_1",k,y,xm))
        tend = mx*(.5*rdy*vf+.5*rdx*uf)+v1("rdnw",k)*(wdtn_next-wdtn_prev)
        theta = l3("theta_coupled_work",k)+my*dts*l3("theta_tend",k)-dts*my*tend
        # WRF advance_mu_t saves PRE-update t_2 into shared t_2save; advance_w
        # overwrites that same array with its time-centered normalized value.
        old_theta = l3("theta_coupled_work",k)
        _store(out["t_2ave"],(k,y,x),jnp.where(active,old_theta,l3("t_2ave",k)),valid)
        _store(out["theta_coupled_work"],(k,y,x),jnp.where(active,theta,l3("theta_coupled_work",k)),valid)
        ww_value = jnp.where(active,ww_k,l3("ww",k))
        _store(out["ww"],(k,y,x),ww_value,valid)
        _store(out["ww_m"],(k,y,x),l3("ww_m",k)+ww_value,valid)
        return ww_next_raw,wdtn_next
    if blocked:
        if cfg.periodic_x:
            xp,xm = (x+1)%nx,(x-1)%nx
        else:
            xp,xm = jnp.minimum(x+1,nx-1),jnp.maximum(x-1,0)
        if not (cfg.specified or cfg.nested):
            yp,ym = (y+1)%ny,(y-1)%ny
        else:
            yp,ym = jnp.minimum(y+1,ny-1),jnp.maximum(y-1,0)
        def level_loads(k):
            kp = jnp.minimum(k+1,nz-1)
            return dict(kp=kp, c1=v1("c1h",k), c2=v1("c2h",k), dnw=v1("dnw",k), rdnw=v1("rdnw",k),
                        fnm=v1("fnm",kp), fnp=v1("fnp",kp), ww_1=l3("ww_1",k), ww_1p=l3("ww_1",kp),
                        th=l3("theta_1",k), thp=l3("theta_1",kp), th_n=l3("theta_1",k,yp,x),
                        th_s=l3("theta_1",k,ym,x), th_e=l3("theta_1",k,y,xp), th_w=l3("theta_1",k,y,xm),
                        v_n=l3("v",k,y+1,x), v_s=l3("v",k), u_e=l3("u",k,y,x+1), u_w=l3("u",k),
                        v1_n=l3("v_1",k,y+1,x), v1_s=l3("v_1",k), u1_e=l3("u_1",k,y,x+1), u1_w=l3("u_1",k),
                        tcw=l3("theta_coupled_work",k), tt=l3("theta_tend",k), t2a=l3("t_2ave",k),
                        ww=l3("ww",k), wwm=l3("ww_m",k))
        def level_step(k, d, carry):
            # Expressions and order identical to level() above, on prefetched operands.
            ww_prev, wdtn_prev = carry
            c1,c2 = d["c1"],d["c2"]
            vn = add(d["v_n"],mul(mul(add(mul(c1,muv_n),c2),d["v1_n"]),msvi_n))
            vs = add(d["v_s"],mul(mul(add(mul(c1,muv_s),c2),d["v1_s"]),msvi_s))
            ue = add(d["u_e"],divide(mul(add(mul(c1,muu_e),c2),d["u1_e"]),msuy_e))
            uw = add(d["u_w"],divide(mul(add(mul(c1,muu_w),c2),d["u1_w"]),msuy_w))
            div_k = mul(mul(mx,my),add(mul(rdy,vn-vs),mul(rdx,ue-uw)))
            ww_k = ww_prev - d["ww_1"]
            increment = d["dnw"]*(c1*dmdt+div_k*scale+c1*mu_tend)/my
            ww_next_raw = ww_prev-increment
            ww_next = ww_next_raw-d["ww_1p"]
            wdtn_next = ww_next*(d["fnm"]*d["thp"]+d["fnp"]*d["th"])
            wdtn_next = jnp.where(k < nz-1,wdtn_next,0.)
            th = d["th"]
            vf = d["v_n"]*(d["th_n"]+th)-d["v_s"]*(th+d["th_s"])
            uf = d["u_e"]*(d["th_e"]+th)-d["u_w"]*(th+d["th_w"])
            tend = mx*(.5*rdy*vf+.5*rdx*uf)+d["rdnw"]*(wdtn_next-wdtn_prev)
            theta = d["tcw"]+my*dts*d["tt"]-dts*my*tend
            _store(out["t_2ave"],(k,y,x),jnp.where(active,d["tcw"],d["t2a"]),valid)
            _store(out["theta_coupled_work"],(k,y,x),jnp.where(active,theta,d["tcw"]),valid)
            ww_value = jnp.where(active,ww_k,d["ww"])
            _store(out["ww"],(k,y,x),ww_value,valid)
            _store(out["ww_m"],(k,y,x),d["wwm"]+ww_value,valid)
            return ww_next_raw,wdtn_next
        _level_blocks(nz, level_loads, level_step, (ww0,jnp.zeros_like(ww0)))
    else:
        jax.lax.fori_loop(0,nz,level,(ww0,jnp.zeros_like(ww0)))
    ww_top = l3("ww",nz)
    _store(out["ww"],(nz,y,x),ww_top,valid)
    _store(out["ww_m"],(nz,y,x),l3("ww_m",nz)+ww_top,valid)


def calc_coef_fp32(state, cfg, *, interpret=False,apply_boundary_forcing=False):
    """One column-local coefficient build per RK stage."""
    _check(state,cfg,apply_boundary_forcing=apply_boundary_forcing)
    nz,ny,nx = state.theta.shape
    args = {name:getattr(state,name) for name in
            ("mut","cqw","c2a","c1h","c2h","c1f","c2f","rdn","rdnw")}
    dt,eps,g = (jnp.asarray(x,jnp.float32) for x in (cfg.dt,cfg.epssm,9.81))
    cof = (.5*dt*g*(1.+eps))**2
    args["s"] = jnp.stack([cof,jnp.asarray(0. if cfg.top_lid else 1.,jnp.float32)] +
                          [jnp.asarray(0.,jnp.float32)]*(vi._N_SLOTS-2))
    shape = jax.ShapeDtypeStruct((nz+1,ny,nx),jnp.float32)
    def kernel(r,o):
        col = lambda name: _ColumnRef(r[name],ny,nx)
        vi._calc_coef_w_kernel(col("mut"),col("cqw"),col("c2a"),
              *(r[name] for name in ("c1h","c2h","c1f","c2f","rdn","rdnw","s")),
              *(_ColumnRef(o[name],ny,nx) for name in ("a","alpha","gamma")),nz=nz)
    return _call(kernel,args,dict(a=shape,alpha=shape,gamma=shape),
                 ((ny*nx+vi.TX-1)//vi.TX,),name="b_core_coef_fp32",interpret=interpret)


def _w_flat_index(nz, ny, nx, block):
    """Flat (face j = 0..nz) x column lanes of the pointwise w kernels."""
    flat = pl.program_id(0)*block + jnp.arange(block, dtype=jnp.int32)
    j, col = flat//(ny*nx), flat%(ny*nx)
    return j, col//nx, col%nx, flat < (nz+1)*ny*nx


def _w_ring_active(y, x, ny, nx, cfg):
    active = (y > 0) & (y < ny-1)
    if not cfg.periodic_x:
        active = active & (x > 0) & (x < nx-1)
    return active


def _w_p1_kernel(r, out, *, nz, ny, nx, block, cfg):
    """A1c pointwise part 1 (expressions of _advance_w_kernel_2pass): t_2ave(j) (final,
    ring cells keep their inputs) and rhs(j) for every face j (rhs(0) = 0, rhs(nz) has
    no advection term and is 0 under top_lid)."""
    j, y, x, valid = _w_flat_index(nz, ny, nx, block)
    s = r["s"]
    dts, g, t0 = s[vi._DTS], s[vi._G], s[vi._T0]
    eps_p, eps_m = 1.0 + s[vi._EPSSM], 1.0 - s[vi._EPSSM]
    def l2(name):
        return _load(r[name], (y, x), valid, 1.)
    def l3(name, k, m):
        return _load(r[name], (k, y, x), m, 1.)
    def l1(name, k, m):
        return _load(r[name], (k,), m, 1.)
    mut, muts, muave, msfty = l2("mut"), l2("muts"), l2("muave"), l2("msfty")
    # t_2ave(j), j < nz
    mh = valid & (j < nz)
    jh = jnp.minimum(j, nz-1)
    c1h_j = l1("c1h", jh, mh)
    massh = c1h_j * muts + l1("c2h", jh, mh)
    half = 0.5 * (eps_p * l3("theta_coupled_work", jh, mh) + eps_m * l3("t_2ave", jh, mh))
    theta_ref = t0 + l3("theta_1", jh, mh)
    t2a = (half + (c1h_j * muave) * t0) / (massh * theta_ref)
    if cfg.specified or cfg.nested:
        t2a = jnp.where(_w_ring_active(y, x, ny, nx, cfg), t2a, l3("t_2ave", jh, mh))
    _store(out["t_2ave"], (j, y, x), t2a, mh)
    # rhs(j)
    adv = valid & (j >= 1) & (j <= nz-1)
    def wdwn_at(k):  # face k+1 value, k in [0, nz-1]
        kc = jnp.clip(k, 0, nz-1)
        dphi = ((l3("ph_1", kc+1, adv) - l3("ph_1", kc, adv)) + l3("phb", kc+1, adv)) - l3("phb", kc, adv)
        ww_mid = 0.5 * (l3("ww", kc+1, adv) + l3("ww", kc, adv))
        return (ww_mid * l1("rdnw", kc, adv)) * dphi
    wd_j, wd_jp1 = wdwn_at(j-1), wdwn_at(j)
    jw = jnp.minimum(j, nz)
    jc = jnp.clip(j, 0, nz-1)
    rr = dts * (l3("ph_tend", jw, valid) + ((0.5 * g) * eps_m) * l3("w", jw, valid))
    rr = jnp.where(adv, rr - dts * ((l1("fnm", jc, adv) * wd_jp1) + (l1("fnp", jc, adv) * wd_j)), rr)
    massf = l1("c1f", jw, valid) * mut + l1("c2f", jw, valid)
    rhs = l3("ph", jw, valid) + (msfty * rr) / massf
    rhs = jnp.where(j == 0, jnp.zeros_like(rhs), rhs)
    if cfg.top_lid:
        rhs = jnp.where(j == nz, jnp.zeros_like(rhs), rhs)
    _store(out["rhs"], (j, y, x), rhs, valid)


def _w_p2_kernel(r, out, *, nz, ny, nx, block, cfg):
    """A1c pointwise part 2: wupd(j) = w_surface (j = 0), explicit w update (1..nz-1),
    w_top (nz) from the stored rhs / t_2ave of part 1 (expressions of the 2-pass kernel)."""
    j, y, x, valid = _w_flat_index(nz, ny, nx, block)
    s = r["s"]
    dts, g = s[vi._DTS], s[vi._G]
    eps_p, eps_m = 1.0 + s[vi._EPSSM], 1.0 - s[vi._EPSSM]
    def l2(name, yy=y, xx=x):
        return _load(r[name], (yy, xx), valid, 1.)
    def l3(name, k, m, yy=y, xx=x):
        return _load(r[name], (k, yy, xx), m, 1.)
    def l1(name, k, m):
        if isinstance(k, int):  # static level: scalar load (as in the 2-pass kernel)
            return r[name][k]
        return _load(r[name], (k,), m, 1.)
    def safe(mass):
        return jnp.where(jnp.abs(mass) > 1.0e-12, mass, jnp.asarray(1.0e-12, dtype=mass.dtype))
    mut, muave, msfty = l2("mut"), l2("muave"), l2("msfty")
    msfty_inv = 1.0 / msfty
    # interior faces 1..nz-1
    mi = valid & (j >= 1) & (j <= nz-1)
    ji = jnp.clip(j, 1, nz-1)
    rhs_jp1, rhs_j, rhs_jm1 = l3("rhs", ji+1, mi), l3("rhs", ji, mi), l3("rhs", ji-1, mi)
    t2a_j, t2a_jm1 = l3("t2a", ji, mi), l3("t2a", ji-1, mi)
    w_e = l3("w", ji, mi) + dts * l3("rw_tend_pg_buoy", ji, mi)
    massh_k = l1("c1h", ji, mi) * mut + l1("c2h", ji, mi)
    massh_l = l1("c1h", ji-1, mi) * mut + l1("c2h", ji-1, mi)
    c2a_j, c2a_jm1 = l3("c2a", ji, mi), l3("c2a", ji-1, mi)
    rdn_j = l1("rdn", ji, mi)
    coef_k = (c2a_j * l1("rdnw", ji, mi)) / safe(massh_k)
    coef_l = (c2a_jm1 * l1("rdnw", ji-1, mi)) / safe(massh_l)
    ph_jp1, ph_j, ph_jm1 = l3("ph", ji+1, mi), l3("ph", ji, mi), l3("ph", ji-1, mi)
    term_upper = coef_k * (eps_p * (rhs_jp1 - rhs_j) + eps_m * (ph_jp1 - ph_j))
    term_lower = coef_l * (eps_p * (rhs_j - rhs_jm1) + eps_m * (ph_j - ph_jm1))
    term_a = (msfty_inv * l3("cqw", ji, mi)) * ((((0.5 * dts) * g) * rdn_j) * (term_upper - term_lower))
    buoy_u = c2a_j * l3("alt", ji, mi) * t2a_j
    buoy_l = c2a_jm1 * l3("alt", ji-1, mi) * t2a_jm1
    term_b = ((dts * g) * msfty_inv) * (rdn_j * (buoy_u - buoy_l) - l1("c1f", ji, mi) * muave)
    w_int = (w_e + term_a) + term_b
    # top face nz (WRF :1492-1502)
    mt = valid & (j == nz)
    km1 = nz - 1
    massh_t = l1("c1h", km1, mt) * mut + l1("c2h", km1, mt)
    rhs_diff_top = eps_p * (l3("rhs", nz, mt) - l3("rhs", km1, mt)) + eps_m * (l3("ph", nz, mt) - l3("ph", km1, mt))
    rdnw_t, c2a_t = l1("rdnw", km1, mt), l3("c2a", km1, mt)
    term_a_top = (((((-0.5 * dts) * g) / safe(massh_t)) * (rdnw_t ** 2)) * 2.0) * c2a_t * rhs_diff_top
    term_b_top = ((-dts) * g) * (
        (((2.0 * rdnw_t) * c2a_t) * l3("alt", km1, mt)) * l3("t2a", km1, mt) + l1("c1f", nz, mt) * muave)
    w_top = l3("w", nz, mt) + dts * l3("rw_tend_pg_buoy", nz, mt) + msfty_inv * (term_a_top + term_b_top)
    if cfg.top_lid:
        w_top = jnp.zeros_like(w_top)
    # surface face 0 (WRF :1417-1429)
    ms = valid & (j == 0)
    ht = l2("ht")
    ht_dy_n = jnp.where(y < ny-1, l2("ht", jnp.minimum(y+1, ny-1), x) - ht, 0.0)
    ht_dy_s = jnp.where(y > 0, ht - l2("ht", jnp.maximum(y-1, 0), x), 0.0)
    ht_dx_e = jnp.where(x < nx-1, l2("ht", y, jnp.minimum(x+1, nx-1)) - ht, 0.0)
    ht_dx_w = jnp.where(x > 0, ht - l2("ht", y, jnp.maximum(x-1, 0)), 0.0)
    cf1, cf2, cf3 = s[vi._CF1], s[vi._CF2], s[vi._CF3]
    v_cf_n = (cf1 * l3("v", 0, ms, y+1) + cf2 * l3("v", 1, ms, y+1)) + cf3 * l3("v", 2, ms, y+1)
    v_cf_s = (cf1 * l3("v", 0, ms) + cf2 * l3("v", 1, ms)) + cf3 * l3("v", 2, ms)
    u_cf_e = (cf1 * l3("u", 0, ms, y, x+1) + cf2 * l3("u", 1, ms, y, x+1)) + cf3 * l3("u", 2, ms, y, x+1)
    u_cf_w = (cf1 * l3("u", 0, ms) + cf2 * l3("u", 1, ms)) + cf3 * l3("u", 2, ms)
    w_surface = (((msfty * 0.5) * s[vi._RDY]) * (ht_dy_n * v_cf_n + ht_dy_s * v_cf_s)
                 + ((l2("msftx") * 0.5) * s[vi._RDX]) * (ht_dx_e * u_cf_e + ht_dx_w * u_cf_w))
    wupd = jnp.where(j == 0, w_surface, jnp.where(j == nz, w_top, w_int))
    _store(out["wupd"], (j, y, x), wupd, valid)


def _w_phase(state, coefficients, cfg, *, interpret):
    nz,ny,nx = state.theta.shape
    names = ("w","rw_tend_pg_buoy","ww","ph","ph_1","phb","ph_tend", "cqw", "w_save",
             "theta_coupled_work","t_2ave","theta_1","c2a","alt","c1f","c2f","c1h","c2h",
             "rdnw","rdn","fnm","fnp","mut","muts","muave","ht","msftx","msfty","u","v")
    args = {name:getattr(state,name) for name in names}
    args.update(coefficients)
    values = dict(dts=cfg.dt,epssm=cfg.epssm,t0=300.,g=9.81,rdx=1./cfg.dx,rdy=1./cfg.dy,
                  cf1=state.cf1,cf2=state.cf2,cf3=state.cf3,w_alpha=cfg.w_alpha,
                  w_crit_cfl=cfg.w_crit_cfl,w_damp_on=1.,dampmag=cfg.dt*cfg.dampcoef,
                  hdepth=cfg.zdamp,half_pi=.5*math.pi)
    args["s"] = jnp.stack([jnp.asarray(values.get(name,0.),jnp.float32) for name in vi._SLOTS])
    w2pass = os.environ.get("GPUWRF_ACOUSTIC_W2PASS", "0")
    if w2pass == "3":
        return _w_phase_split(state, args, cfg, nz=nz, ny=ny, nx=nx, interpret=interpret)
    two_pass = w2pass == "1"
    scratch = ("rhs",) if two_pass else ("rhs","wdwn","rw_eff","wtop")
    outputs = {name:_shape(state.w) for name in ("w","ph")+scratch}
    outputs["t_2ave"] = _shape(state.t_2ave)
    def kernel(r,o):
        col = lambda name: _ColumnRef(r[name],ny,nx)
        face = ("w","rw_tend_pg_buoy","ww","ph","ph_1","phb","ph_tend","cqw","a","alpha","gamma","w_save")
        mass = ("theta_coupled_work","t_2ave","theta_1","c2a","alt")
        vertical = ("c1f","c2f","c1h","c2h","rdnw","rdn","fnm","fnp")
        columns = ("mut","muts","muave","ht","msftx","msfty")
        damp3 = cfg.damp_opt==3 and cfg.dampcoef>0 and state.w_save is not None
        common = ((*(col(name) for name in face+mass),*(r[name] for name in vertical),
                   *(col(name) for name in columns),
                   _ColumnRef(r["u"],ny,nx,x_offset=1),col("u"),col("v"),
                   _ColumnRef(r["v"],ny,nx,y_offset=1),r["s"]))
        ring = cfg.specified or cfg.nested
        def ring_mask():
            ci=pl.program_id(0)*vi.TX+jnp.arange(vi.TX,dtype=jnp.int32)
            yy,xx=ci//nx,ci%nx
            valid=ci<ny*nx
            active=(yy>0)&(yy<ny-1)
            if not cfg.periodic_x:active=active&(xx>0)&(xx<nx-1)
            return yy,xx,valid,active
        if two_pass:
            # Same expressions, levels walked twice instead of 14 times (latency-bound
            # column loops); w_damping is applied upstream (w_damp=False here). The
            # ring cells keep their inputs in the final stores (no preserve passes).
            vi._advance_w_kernel_2pass(*common,
                  *(_ColumnRef(o[name],ny,nx) for name in ("w","ph","t_2ave","rhs")),
                  nz=nz,nzf=nz+1,nx=nx,ny=ny,top_lid=cfg.top_lid,damp3=damp3,wrf_fp32=True,
                  keep=ring_mask()[3] if ring else None)
        else:
            vi._advance_w_kernel(*common,
                  *(_ColumnRef(o[name],ny,nx) for name in ("w","ph","t_2ave","rhs","wdwn","rw_eff","wtop")),
                  nz=nz,nzf=nz+1,nx=nx,ny=ny,top_lid=cfg.top_lid,w_damp=False,
                  damp3=damp3,wrf_fp32=True)
        # WRF advance_w excludes specified/nested ring cells. Preserve those
        # exact inputs within this same launch; boundary forcing owns them.
        if ring and not two_pass:
            yy,xx,valid,active=ring_mask()
            def preserve(k,_):
                for name in ("w","ph"):
                    old=_load(r[name],(k,yy,xx),valid)
                    new=_load(o[name],(k,yy,xx),valid)
                    _store(o[name],(k,yy,xx),jnp.where(active,new,old),valid)
                return _
            jax.lax.fori_loop(0,nz+1,preserve,0)
            def preserve_theta(k,_):
                old=_load(r["t_2ave"],(k,yy,xx),valid)
                new=_load(o["t_2ave"],(k,yy,xx),valid)
                _store(o["t_2ave"],(k,yy,xx),jnp.where(active,new,old),valid)
                return _
            jax.lax.fori_loop(0,nz,preserve_theta,0)
    result = _call(kernel,args,outputs,((ny*nx+vi.TX-1)//vi.TX,),
                   name="b_core_w_fp32",interpret=interpret)
    return state.replace(w=result["w"],ph=result["ph"],t_2ave=result["t_2ave"])


def _w_phase_split(state, args, cfg, *, nz, ny, nx, interpret):
    """A1c (GPUWRF_ACOUSTIC_W2PASS=3): the advance_w phase as two k-parallel pointwise
    kernels (full occupancy) + one column kernel that keeps only the recurrences."""
    block = 128
    grid = (((nz+1)*ny*nx + block - 1)//block,)
    p1 = _call(lambda r,o: _w_p1_kernel(r,o,nz=nz,ny=ny,nx=nx,block=block,cfg=cfg), args,
               {"t_2ave": _shape(state.t_2ave), "rhs": _shape(state.w)}, grid,
               name="b_core_w_p1_fp32", interpret=interpret)
    args2 = dict(args, rhs=p1["rhs"], t2a=p1["t_2ave"])
    # P2 reads the new average through t2a. Keeping its unused old argument
    # makes XLA protect that buffer with a full-field copy before P1's alias.
    args2.pop("t_2ave")
    p2 = _call(lambda r,o: _w_p2_kernel(r,o,nz=nz,ny=ny,nx=nx,block=block,cfg=cfg), args2,
               {"wupd": _shape(state.w)}, grid, name="b_core_w_p2_fp32", interpret=interpret)
    args3 = {name: args[name] for name in ("a","alpha","gamma","w_save","ph_1","phb","w","ph",
                                            "c1f","c2f","mut","muts","msfty","s")}
    args3.update(wupd=p2["wupd"], rhs=p1["rhs"])
    damp3 = cfg.damp_opt==3 and cfg.dampcoef>0 and state.w_save is not None
    ring = cfg.specified or cfg.nested
    def kernel(r,o):
        col = lambda name: _ColumnRef(r[name],ny,nx)
        keep = None
        if ring:
            ci = pl.program_id(0)*vi.TX + jnp.arange(vi.TX, dtype=jnp.int32)
            keep = _w_ring_active(ci//nx, ci%nx, ny, nx, cfg)
        vi._advance_w_kernel_recur(
            *(col(name) for name in ("wupd","a","alpha","gamma","rhs","w_save","ph_1","phb","w","ph")),
            r["c1f"], r["c2f"], col("mut"), col("muts"), col("msfty"), r["s"],
            *(_ColumnRef(o[name],ny,nx) for name in ("w","ph")), nz=nz, damp3=damp3, keep=keep)
    res = _call(kernel, args3, {"w": _shape(state.w), "ph": _shape(state.w)},
                ((ny*nx+vi.TX-1)//vi.TX,), name="b_core_w_fp32", interpret=interpret)
    return state.replace(w=res["w"], ph=res["ph"], t_2ave=p1["t_2ave"])


def _pressure_kernel(r,out,*,nz,ny,nx,block):
    flat=pl.program_id(0)*block+jnp.arange(block,dtype=jnp.int32)
    k,col = flat//(ny*nx),flat%(ny*nx)
    y,x = col//nx,col%nx
    valid=flat<nz*ny*nx
    def l3(name,kk=k):
        return _load(r[name],(kk,y,x),valid)
    mut=_load(r["mut"],(y,x),valid)
    muts=_load(r["muts"],(y,x),valid)
    c1,c2=_load(r["c1h"],(k,),valid),_load(r["c2h"],(k,),valid)
    mass=c1*muts+c2
    mass_safe=jnp.where(jnp.abs(mass)>1e-12,mass,1e-12)
    mu_term=c1*_load(r["mu_work"],(y,x),valid)
    al=(-1./mass_safe)*(l3("alt")*mu_term+_load(r["rdnw"],(k,),valid)*(l3("ph",k+1)-l3("ph")))
    thref=300.+l3("theta_1")
    thsafe=jnp.where(jnp.abs(thref)>1e-6,thref,1e-6)
    ptmp=l3("c2a")*(l3("alt")*(l3("theta_coupled_work")-mu_term*l3("theta_1"))/(mass_safe*thsafe)-al)
    p=ptmp+r["smdiv"][()]*(ptmp-l3("pm1"))
    theta=(l3("theta_coupled_work")+l3("theta_1")*(c1*mut+c2))/mass
    for name,value in dict(p=p,pm1=ptmp,al=al,theta=theta).items():
        _store(out[name],(k,y,x),value,valid)


def _uv_boundaries(before,after,cfg):
    """Retain the product boundary adapter around the native interior kernel."""
    from gpuwrf.dynamics.core.acoustic import _pin_spec_ring_wrf_owned
    from gpuwrf.coupling.boundary_apply import apply_normal_bdy_work,DEFAULT_BOUNDARY_CONFIG
    u,v=after.u,after.v
    if before.u_work_bdy is not None and before.v_work_bdy is not None:
        if cfg.nested_frozen_wrf_boundary_bundle:
            u=_pin_spec_ring_wrf_owned(u,before.u+cfg.dt*before.u_work_bdy,spec_zone=cfg.spec_zone)
            v=_pin_spec_ring_wrf_owned(v,before.v+cfg.dt*before.v_work_bdy,spec_zone=cfg.spec_zone)
        else:
            u,v=apply_normal_bdy_work(u,v,before.u_work_bdy,before.v_work_bdy,cfg.dt,
                         cfg.dt_full or cfg.dt,config=DEFAULT_BOUNDARY_CONFIG,
                         relax_strength=cfg.normal_bdy_relax_strength,
                         relax_rows=not cfg.nested_frozen_wrf_boundary_bundle)
    if (before.u_spec_tan_target is not None and before.v_spec_tan_target is not None
            and os.environ.get("GPUWRF_SPEC_RING_SELECT", "0") == "1" and cfg.spec_zone > 0):
        # Same writes as the loop below as two masked selects (bit-identical).
        import numpy as _np
        from gpuwrf.kernels.ring_select import scatter_mask
        sz=int(cfg.spec_zone)
        uy=_np.arange(u.shape[-2])[:,None]+_np.zeros((1,u.shape[-1]),int)
        u=scatter_mask(u,before.u_spec_tan_target,(uy<sz)|(uy>=u.shape[-2]-sz))
        vy=_np.arange(v.shape[-2])[:,None]; vx=_np.arange(v.shape[-1])[None,:]
        v=scatter_mask(v,before.v_spec_tan_target,((vx<sz)|(vx>=v.shape[-1]-sz))&(vy>=1)&(vy<=v.shape[-2]-2))
    elif before.u_spec_tan_target is not None and before.v_spec_tan_target is not None:
        for b in range(cfg.spec_zone):
            u=u.at[:,b,:].set(before.u_spec_tan_target[:,b,:])
            u=u.at[:,-1-b,:].set(before.u_spec_tan_target[:,-1-b,:])
            v=v.at[:,1:-1,b].set(before.v_spec_tan_target[:,1:-1,b])
            v=v.at[:,1:-1,-1-b].set(before.v_spec_tan_target[:,1:-1,-1-b])
    return after.replace(u=u,v=v,ru_m=before.ru_m+u,rv_m=before.rv_m+v)


def _mass_boundaries(before,after,cfg):
    from gpuwrf.dynamics.core.acoustic import _pin_spec_ring_wrf_owned
    if before.mu_spec_target is None:return after
    pin=lambda value,target:_pin_spec_ring_wrf_owned(value,target,spec_zone=cfg.spec_zone)
    if cfg.nested_frozen_wrf_boundary_bundle:
        values=dict(mu=before.mu+cfg.dt*before.mu_spec_target,
                    mu_work=before.mu_work+cfg.dt*before.mu_spec_target,
                    muts=before.muts+cfg.dt*before.muts_spec_target,
                    theta_coupled_work=before.theta_coupled_work+cfg.dt*before.theta_spec_target)
    else:
        if before.mu_work_spec_target is None:
            raise ValueError("fp32 absolute boundary forcing requires staged mu_work_spec_target")
        values=dict(mu=before.mu_spec_target,mu_work=before.mu_work_spec_target,
                    muts=before.muts_spec_target,muave=before.muave_spec_target,
                    theta_coupled_work=before.theta_spec_target)
    return after.replace(**{name:pin(getattr(after,name),target) for name,target in values.items()})


def _w_boundaries(before,after,cfg):
    from gpuwrf.dynamics.core.acoustic import _pin_spec_ring_wrf_owned,_specified_w_zero_grad_work
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG,spec_bdyupdate_ph_inloop,spec_bdyupdate_ph_tendency_inloop
    w,ph=after.w,after.ph
    if before.ph_bdy_target is not None and before.ph_save_for_spec is not None:
        if cfg.nested_frozen_wrf_boundary_bundle:
            ph=spec_bdyupdate_ph_tendency_inloop(ph,before.ph,before.ph_bdy_target,
                    before.ph_save_for_spec,before.mu_spec_target,after.muts,before.c1f,
                    before.c2f,cfg.dt,DEFAULT_BOUNDARY_CONFIG,spec_zone=cfg.spec_zone)
        else:
            ph=spec_bdyupdate_ph_inloop(ph,before.ph_bdy_target,before.ph_save_for_spec,
                    None,after.muts,before.c1f,before.c2f,cfg.dt,DEFAULT_BOUNDARY_CONFIG)
    if before.w_spec_target is not None:
        w=_pin_spec_ring_wrf_owned(w,before.w+cfg.dt*before.w_spec_target,spec_zone=cfg.spec_zone)
    if cfg.spec_w_zero_grad:
        w=_specified_w_zero_grad_work(w,w_save=before.w_save,mut=before.mut,muts=after.muts,
              c1f=before.c1f,c2f=before.c2f,msfty=before.msfty,spec_zone=cfg.spec_zone)
    return after.replace(w=w,ph=ph)


def acoustic_substep_fp32(state: AcousticCoreState, *, coefficients: dict,
                          cfg: AcousticCoreConfig, emdiv=.01, smdiv=.1,
                          interpret=False, return_guard_events=False,
                          apply_boundary_forcing=False):
    """Four kernels, no domain-wide vertical loops and no host transfers."""
    _check(state,cfg,apply_boundary_forcing=apply_boundary_forcing)
    before=state
    nz,ny,nx=state.theta.shape
    for name,value in coefficients.items():
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} coefficients must be fp32")
    args={name:value for name,value in state.to_dict().items() if value is not None}
    args["s"]=jnp.asarray([cfg.dt,cfg.dx,cfg.dy,emdiv],jnp.float32)
    uv=_call(lambda r,o:_uv_kernel(r,o,nz=nz,ny=ny,nx=nx,block=128,cfg=cfg),
             args,{name:_shape(getattr(state,name)) for name in ("u","v","ru_m","rv_m")},
             ((max(state.u.size,state.v.size)+127)//128,),name="b_core_uv_fp32",interpret=interpret)
    state=state.replace(**uv)
    if apply_boundary_forcing:state=_uv_boundaries(before,state,cfg)
    args={name:value for name,value in state.to_dict().items() if value is not None}
    args["s"]=jnp.asarray([cfg.dt,1./cfg.dx,1./cfg.dy,cfg.epssm],jnp.float32)
    outs={name:_shape(getattr(state,name)) for name in ("mu","mu_work","muts","mudf","muave","ww","ww_m","theta_coupled_work","t_2ave")}
    outs["guard_events"]=jax.ShapeDtypeStruct((ny,nx),jnp.int32)
    mass=_call(lambda r,o:_mass_kernel(r,o,nz=nz,ny=ny,nx=nx,block=128,cfg=cfg,interpret=interpret),args,outs,
               ((ny*nx+127)//128,),name="b_core_mass_fp32",interpret=interpret)
    guards=mass.pop("guard_events")
    state=state.replace(**mass)
    if apply_boundary_forcing:state=_mass_boundaries(before,state,cfg)
    before_w=state
    state=_w_phase(state,coefficients,cfg,interpret=interpret)
    if apply_boundary_forcing:state=_w_boundaries(before_w,state,cfg)
    args={name:value for name,value in state.to_dict().items() if value is not None}
    args["smdiv"]=jnp.asarray(smdiv,jnp.float32)
    eos=_call(lambda r,o:_pressure_kernel(r,o,nz=nz,ny=ny,nx=nx,block=128),args,
              {name:_shape(state.theta) for name in ("p","al","pm1","theta")},
              ((state.theta.size+127)//128,),name="b_core_pressure_fp32",interpret=interpret)
    state=state.replace(**eos,theta_ave=eos["theta"])
    return (state,guards) if return_guard_events else state
