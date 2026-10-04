"""Literal WRF REAL coupling/interpolation boundaries for lateral forcing.

Opaque point kernels preserve the REAL products before small residuals are
formed in relaxation. Normal CPU interpretation is a test-only RN emulation;
production always uses explicit PTX RN32, with no fp64 arithmetic.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt
from gpuwrf.kernels.dyn_rk_fp32 import _rn as _literal_rn

_INTERPRET = False
BLOCK = 256


def _rn(op, a, b, *, interpret):
    a, b = jnp.broadcast_arrays(a, b)
    return _literal_rn(op, a, b, interpret=interpret)


def _interpolate(leaf, lead, cadence, out, *, records, width, nz, side_len):
    i = pl.program_id(0)*BLOCK+jnp.arange(BLOCK, dtype=jnp.int32)
    tang, k = i%side_len, (i//side_len)%nz
    b, side = (i//(side_len*nz))%width, i//(side_len*nz*width)
    valid = side < 4
    # Triton's inline-asm verifier needs a tensor operand for a tensor result.
    # Pallas represents scalar asm results as tensor<1>; broadcast the clock
    # to the point lanes so every RN operation has ordinary tensor operands.
    lead_value = jnp.broadcast_to(pt.load(lead.at[()]).astype(jnp.float32), (BLOCK,))
    cadence_value = jnp.broadcast_to(pt.load(cadence.at[()]), (BLOCK,))
    fraction = _rn('div', lead_value, cadence_value, interpret=_INTERPRET)
    lower = jnp.clip(jnp.ceil(fraction).astype(jnp.int32)-1, 0, records-2)
    alpha = jnp.clip(_rn('sub', fraction, lower.astype(jnp.float32), interpret=_INTERPRET),
                     jnp.float32(0), jnp.float32(1))
    dtbc = _rn('mul', alpha, cadence_value, interpret=_INTERPRET)
    # Storage-dtype records are rounded to REAL at load (masked lanes unstored).
    old = pt.load(leaf.at[lower, side, b, k, tang], mask=valid).astype(jnp.float32)
    new = pt.load(leaf.at[lower+1, side, b, k, tang], mask=valid).astype(jnp.float32)
    rate = _rn('div', _rn('sub', new, old, interpret=_INTERPRET), cadence_value, interpret=_INTERPRET)
    value = _rn('add', old, _rn('mul', dtbc, rate, interpret=_INTERPRET), interpret=_INTERPRET)
    pt.store(out.at[side, b, k, tang], value, mask=valid)


def interpolate_leaf(leaf, lead, cadence):
    leaf = jnp.asarray(leaf)
    if leaf.shape[0] == 1:
        return leaf[0].astype(jnp.float32)
    records, _, width, nz, side_len = leaf.shape
    size = 4*width*nz*side_len
    return pl.pallas_call(partial(_interpolate, records=records, width=width, nz=nz, side_len=side_len),
        out_shape=jax.ShapeDtypeStruct(leaf.shape[1:], jnp.float32), grid=((size+BLOCK-1)//BLOCK,),
        interpret=_INTERPRET, name='boundary_leaf_interp_real')(
            leaf, jnp.asarray(lead), jnp.asarray(cadence, jnp.float32))


def _dry_relax(field, mu, c1, c2, msf, leaf, lead, cadence, fcx, gcx, out,
               *, nz, ny, nx, mu_ny, mu_nx, records, kind, coupled, spec, relax):
    """One boundary stencil; interior tiles only write the zero tendency."""
    block = 32
    i32 = jnp.int32
    x = pl.program_id(0)*block+jnp.arange(block, dtype=i32)
    # Explicit int32 vectors: under x64 a scalar where(c, 2, 3) lowers to an
    # i64/i1 select that the Triton verifier rejects.
    y = jnp.broadcast_to(pl.program_id(1).astype(i32), (block,))
    k = jnp.broadcast_to(pl.program_id(2).astype(i32), (block,))
    valid = x < nx
    dx, dy = jnp.minimum(x, nx-1-x), jnp.minimum(y, ny-1-y)
    y_side = (dy >= spec) & (dy < relax) & (x >= dy) & (x < nx-dy)
    x_side = (dx >= spec) & (dx < relax) & (y >= dx+1) & (y < ny-dx-1)
    active = valid & (y_side | x_side)
    side = jnp.where(y_side, jnp.where(y < ny//2, i32(2), i32(3)), jnp.where(x < nx//2, i32(0), i32(1)))
    b = jnp.where(y_side, dy, dx)
    pt.store(out.at[k, y, x], jnp.zeros_like(x, dtype=jnp.float32), mask=valid)

    # Operands arrive in their storage dtype; WRF REAL is formed at load (RN
    # convert, the rounding an XLA convert would apply) so no convert kernels
    # run outside. Masked lanes are never stored.
    def real(ref, idx, mask):
        return pt.load(ref.at[idx], mask=mask).astype(jnp.float32)

    @pl.when(jnp.max(active.astype(jnp.int32)) > 0)
    def boundary_tile():
        rn = partial(_rn, interpret=_INTERPRET)
        clock = jnp.broadcast_to(pt.load(lead.at[()]).astype(jnp.float32), (block,))
        interval = jnp.broadcast_to(pt.load(cadence.at[()]), (block,))
        if records > 1:
            fraction = rn('div', clock, interval)
            lower = jnp.clip(jnp.ceil(fraction).astype(jnp.int32)-1, 0, records-2)
            alpha = jnp.clip(rn('sub', fraction, lower.astype(jnp.float32)), jnp.float32(0), jnp.float32(1))
            dtbc = rn('mul', alpha, interval)
        else:
            lower = jnp.zeros_like(x)

        def residual(gy, gx, width):
            gy, gx = jnp.broadcast_arrays(gy, gx)
            load = lambda ref, idx: real(ref, idx, active)
            scale = load(msf, (gy, gx))
            if kind == 'u':
                col = rn('mul', rn('add', load(mu, (gy, jnp.maximum(gx-1, 0))),
                    load(mu, (gy, jnp.minimum(gx, mu_nx-1)))), jnp.float32(.5))
            elif kind == 'v':
                col = rn('mul', rn('add', load(mu, (jnp.maximum(gy-1, 0), gx)),
                    load(mu, (jnp.minimum(gy, mu_ny-1), gx))), jnp.float32(.5))
            else:
                col = load(mu, (gy, gx))
            mass = rn('add', rn('mul', load(c1, (k,)), col), load(c2, (k,)))
            value = load(field, (k, gy, gx))
            if kind != 'mu':
                if kind == 'theta':
                    # WRF relax_bdy_dry relaxes the perturbation t_2 = theta - T0.
                    value = rn('sub', value, jnp.float32(300))
                value = rn('mul', mass, value)
                if kind in ('u', 'v'):
                    value = rn('div', value, scale)
            tang = jnp.where(side < 2, gy, gx)
            old = load(leaf, (lower, side, width, k, tang))
            if not coupled and kind == 'theta':
                old = rn('sub', old, jnp.float32(300))
            if not coupled and kind != 'mu':
                old = rn('mul', mass, old)
                if kind in ('u', 'v'):
                    old = rn('div', old, scale)
            if records > 1:
                new = load(leaf, (lower+1, side, width, k, tang))
                if not coupled and kind == 'theta':
                    new = rn('sub', new, jnp.float32(300))
                if not coupled and kind != 'mu':
                    new = rn('mul', mass, new)
                    if kind in ('u', 'v'):
                        new = rn('div', new, scale)
                old = rn('add', old, rn('mul', dtbc, rn('div', rn('sub', new, old), interval)))
            return rn('sub', old, value)

        centre = residual(y, x, b)
        plus = residual(jnp.where(y_side, y, jnp.minimum(y+1, ny-1)),
                        jnp.where(y_side, jnp.minimum(x+1, nx-1), x), b)
        minus = residual(jnp.where(y_side, y, jnp.maximum(y-1, 0)),
                         jnp.where(y_side, jnp.maximum(x-1, 0), x), b)
        toward = jnp.where((side == 0) | (side == 2), i32(-1), i32(1))
        edge = residual(jnp.where(y_side, y+toward, y), jnp.where(y_side, x, x+toward), b-1)
        inner = residual(jnp.where(y_side, y-toward, y), jnp.where(y_side, x, x-toward), b+1)
        lap = rn('sub', rn('add', rn('add', rn('add', plus, minus), edge), inner), rn('mul', jnp.float32(4), centre))
        f = real(fcx, (b,), active)
        g = real(gcx, (b,), active)
        value = rn('sub', rn('mul', f, centre), rn('mul', g, lap))
        if kind in ('theta', 'ph', 'w'):
            value = rn('div', value, real(msf, (y, x), active))
        pt.store(out.at[k, y, x], value, mask=active)


def dry_relax(field, mu, c1, c2, msf, leaf, lead, cadence, fcx, gcx,
              *, kind, coupled, spec, relax):
    arrays = [jnp.asarray(v) for v in (field, mu, c1, c2, msf, leaf, lead)]
    arrays += [jnp.asarray(v, jnp.float32) for v in (cadence, fcx, gcx)]
    nz, ny, nx = field.shape
    return pl.pallas_call(partial(_dry_relax, nz=nz, ny=ny, nx=nx,
        mu_ny=mu.shape[0], mu_nx=mu.shape[1], records=leaf.shape[0],
        kind=kind, coupled=bool(coupled), spec=int(spec), relax=int(relax)),
        out_shape=jax.ShapeDtypeStruct(field.shape, jnp.float32),
        grid=((nx+31)//32, ny, nz), interpret=_INTERPRET,
        compiler_params=pt.CompilerParams(num_warps=1), name='boundary_dry_relax_real')(*arrays)


def _coupled_work(leaf, save, mass, msf, out, *, nz, ny, nx, axis, zone):
    i = pl.program_id(0)*BLOCK+jnp.arange(BLOCK, dtype=jnp.int32)
    k, y, x = i//(ny*nx), (i//nx)%ny, i%nx
    distance = jnp.minimum(x, nx-1-x) if axis == 'x' else jnp.minimum(y, ny-1-y)
    side = jnp.where(x < nx//2, 0, 1) if axis == 'x' else jnp.where(y < ny//2, 2, 3)
    tang = y if axis == 'x' else x
    valid, active = i < nz*ny*nx, (i < nz*ny*nx) & (distance < zone)
    # Storage-dtype operands rounded to REAL at load; inactive lanes store 0.
    load = lambda ref, idx: jnp.where(active, pt.load(ref.at[idx], mask=active).astype(jnp.float32), jnp.float32(0))
    value = load(leaf, (side, distance, k, tang))
    saved = load(save, (k, y, x))
    weight = load(mass, (k, y, x))
    scale = jnp.where(active, pt.load(msf.at[y, x], mask=active).astype(jnp.float32), jnp.float32(1))
    saved = _rn('div', _rn('mul', saved, weight, interpret=_INTERPRET), scale, interpret=_INTERPRET)
    result = _rn('sub', value, saved, interpret=_INTERPRET)
    pt.store(out.at[k, y, x], result, mask=valid)


def coupled_work_target(leaf, save, mass, msf, *, axis, zone):
    arrays = [jnp.asarray(v) for v in (leaf, save, mass, msf)]
    nz, ny, nx = save.shape
    return pl.pallas_call(partial(_coupled_work, nz=nz, ny=ny, nx=nx, axis=axis, zone=int(zone)),
        out_shape=jax.ShapeDtypeStruct(save.shape, jnp.float32),
        grid=((save.size+BLOCK-1)//BLOCK,), interpret=_INTERPRET,
        name='boundary_coupled_work_real')(*arrays)


def _ring_update(field, leaf, base, lead, cadence, wf, wg, out, *, nz, ny, nx, records, spec, relax,
                 do_relax, with_base):
    """End-of-step boundary VALUE update of one field in one launch.

    Spec rows (released order: S/N written after W/E, so they own corners) get
    the time-interpolated record (``+ base`` record for totals); with
    ``do_relax`` the relax rows get ``cur + wf*fls0 - wg*lap`` (WRF ownership);
    every other cell is copied. REAL arithmetic, storage dtype kept.
    """
    block = 32
    i32 = jnp.int32
    x = pl.program_id(0)*block+jnp.arange(block, dtype=i32)
    y = jnp.broadcast_to(pl.program_id(1).astype(i32), (block,))
    k = jnp.broadcast_to(pl.program_id(2).astype(i32), (block,))
    valid = x < nx
    dx, dy = jnp.minimum(x, nx-1-x), jnp.minimum(y, ny-1-y)
    spec_y = valid & (dy < spec)
    spec_x = valid & (dx < spec) & ~spec_y
    if do_relax:
        relax_y = valid & (dy >= spec) & (dy < relax) & (x >= dy) & (x < nx-dy)
        relax_x = valid & (dx >= spec) & (dx < relax) & (y >= dx+1) & (y < ny-dx-1)
    else:
        relax_y = relax_x = valid & False
    y_own, ring = spec_y | relax_y, spec_y | spec_x
    lane_relax = relax_y | relax_x
    side = jnp.where(y_own, jnp.where(y < ny//2, i32(2), i32(3)), jnp.where(x < nx//2, i32(0), i32(1)))
    b = jnp.where(y_own, dy, dx)
    copy = valid & ~ring & ~lane_relax
    pt.store(out.at[k, y, x], pt.load(field.at[k, y, x], mask=copy), mask=copy)

    def real(ref, idx, mask):
        return pt.load(ref.at[idx], mask=mask).astype(jnp.float32)

    @pl.when(jnp.max((ring | lane_relax).astype(jnp.int32)) > 0)
    def boundary_tile():
        rn = partial(_rn, interpret=_INTERPRET)
        clock = jnp.broadcast_to(pt.load(lead.at[()]).astype(jnp.float32), (block,))
        interval = jnp.broadcast_to(pt.load(cadence.at[()]), (block,))
        if records > 1:
            fraction = rn('div', clock, interval)
            lower = jnp.clip(jnp.ceil(fraction).astype(i32)-1, 0, records-2)
            alpha = jnp.clip(rn('sub', fraction, lower.astype(jnp.float32)), jnp.float32(0), jnp.float32(1))
            dtbc = rn('mul', alpha, interval)
        else:
            lower = jnp.zeros_like(x)

        def record(ref, width, tang, mask):
            old = real(ref, (lower, side, width, k, tang), mask)
            if records > 1:
                new = real(ref, (lower+1, side, width, k, tang), mask)
                old = rn('add', old, rn('mul', dtbc, rn('div', rn('sub', new, old), interval)))
            return old

        tang0 = jnp.where(side < 2, y, x)
        value = record(leaf, b, tang0, ring)
        if with_base:
            value = rn('add', record(base, b, tang0, ring), value)
        pt.store(out.at[k, y, x], value.astype(out.dtype), mask=ring)
        if do_relax:
            def residual(gy, gx, width):
                tang = jnp.where(side < 2, gy, gx)
                return rn('sub', record(leaf, width, tang, lane_relax), real(field, (k, gy, gx), lane_relax))

            centre = residual(y, x, b)
            plus = residual(jnp.where(y_own, y, jnp.minimum(y+1, ny-1)), jnp.where(y_own, jnp.minimum(x+1, nx-1), x), b)
            minus = residual(jnp.where(y_own, y, jnp.maximum(y-1, 0)), jnp.where(y_own, jnp.maximum(x-1, 0), x), b)
            toward = jnp.where((side == 0) | (side == 2), i32(-1), i32(1))
            edge = residual(jnp.where(y_own, y+toward, y), jnp.where(y_own, x, x+toward), b-1)
            inner = residual(jnp.where(y_own, y-toward, y), jnp.where(y_own, x, x-toward), b+1)
            lap = rn('sub', rn('add', rn('add', rn('add', minus, plus), edge), inner), rn('mul', jnp.float32(4), centre))
            cur = real(field, (k, y, x), lane_relax)
            new_value = rn('sub', rn('add', cur, rn('mul', real(wf, (b,), lane_relax), centre)),
                           rn('mul', real(wg, (b,), lane_relax), lap))
            pt.store(out.at[k, y, x], new_value.astype(out.dtype), mask=lane_relax)


def ring_update(field, leaf, lead, cadence, wf, wg, *, spec, relax, do_relax, base=None):
    """Fused spec(+relax) boundary value update; ``base`` adds a second record (totals)."""
    field = jnp.asarray(field)
    nz, ny, nx = field.shape
    arrays = [field, jnp.asarray(leaf), jnp.asarray(leaf if base is None else base), jnp.asarray(lead),
              jnp.asarray(cadence, jnp.float32), jnp.asarray(wf, jnp.float32), jnp.asarray(wg, jnp.float32)]
    return pl.pallas_call(partial(_ring_update, nz=nz, ny=ny, nx=nx, records=int(jnp.shape(leaf)[0]),
        spec=int(spec), relax=int(relax), do_relax=bool(do_relax), with_base=base is not None),
        out_shape=jax.ShapeDtypeStruct(field.shape, field.dtype), grid=((nx+31)//32, ny, nz),
        interpret=_INTERPRET, compiler_params=pt.CompilerParams(num_warps=1),
        name='boundary_ring_update_real')(*arrays)
