"""Dense ring-stencil relax(+spec) tendency for lateral boundary scalars (WRF REAL).

One Pallas launch per field replaces the full-ring target scatter, the
relax_bdytend row scatter and the spec_bdytend scatter (and, with ``mass``, the
full-field mass-weight pass). Production uses explicit PTX RN32 operations; the
CPU interpreter is a test-only path.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt
from gpuwrf.kernels.dyn_rk_fp32 import _rn as _literal_rn

_INTERPRET = False


def _rn(op, a, b, *, interpret):
    a, b = jnp.broadcast_arrays(a, b)
    if interpret:
        # Exact RN32 on the CPU interpreter: binary64 then one rounding is the
        # correctly rounded binary32 result for + - * / (53 >= 2*24 + 2). The
        # explicit reduce_precision keeps XLA's excess-precision rewrite from
        # dropping the f64->f32->f64 round trip between chained operations.
        a64, b64 = a.astype(jnp.float64), b.astype(jnp.float64)
        exact = {'add': a64+b64, 'sub': a64-b64, 'mul': a64*b64, 'div': a64/b64}[op]
        return jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23).astype(jnp.float32)
    return _literal_rn(op, a, b, interpret=interpret)


def _interpret():
    """CPU backends run the kernel interpreted (tests); GPUs use the RN32 PTX path."""
    return _INTERPRET or jax.default_backend() == 'cpu'


def _ring_relax(field, value, rate, fcx, gcx, mu, c1, c2, out, *, nz, ny, nx, spec, relax, with_spec, couple,
                interpret):
    """WRF relax_bdytend (+ spec_bdytend) on an already-coupled field: one launch.

    ``value``/``rate`` are REAL boundary leaves ``(side, width, z, side_len)``;
    ownership is the Fortran trim (Y sides own corners), interior writes zero.
    """
    block = 32
    i32 = jnp.int32
    x = pl.program_id(0)*block+jnp.arange(block, dtype=i32)
    y = jnp.broadcast_to(pl.program_id(1).astype(i32), (block,))
    k = jnp.broadcast_to(pl.program_id(2).astype(i32), (block,))
    valid = x < nx
    dx, dy = jnp.minimum(x, nx-1-x), jnp.minimum(y, ny-1-y)
    y_own = (dy < relax) & (x >= dy) & (x < nx-dy)
    x_own = (dx < relax) & (y >= dx+1) & (y < ny-dx-1)
    b = jnp.where(y_own, dy, dx)
    relax_lane = valid & (y_own | x_own) & (b >= spec)
    spec_lane = valid & (y_own | x_own) & (b < spec) if with_spec else valid & False
    side = jnp.where(y_own, jnp.where(y < ny//2, i32(2), i32(3)), jnp.where(x < nx//2, i32(0), i32(1)))
    pt.store(out.at[k, y, x], jnp.zeros_like(x, dtype=jnp.float32), mask=valid)

    def real(ref, idx, mask):
        return pt.load(ref.at[idx], mask=mask).astype(jnp.float32)

    @pl.when(jnp.max((relax_lane | spec_lane).astype(jnp.int32)) > 0)
    def boundary_tile():
        rn = partial(_rn, interpret=interpret)

        def residual(gy, gx, width):
            tang = jnp.where(side < 2, gy, gx)
            current = real(field, (k, gy, gx), relax_lane)
            if couple:
                # WRF mass_weight: field*(c1(k)*mu(i,j)+c2(k)), each op REAL.
                mass = rn('add', rn('mul', real(c1, (k,), relax_lane), real(mu, (gy, gx), relax_lane)),
                          real(c2, (k,), relax_lane))
                current = rn('mul', current, mass)
            return rn('sub', real(value, (side, width, k, tang), relax_lane), current)

        centre = residual(y, x, b)
        plus = residual(jnp.where(y_own, y, jnp.minimum(y+1, ny-1)), jnp.where(y_own, jnp.minimum(x+1, nx-1), x), b)
        minus = residual(jnp.where(y_own, y, jnp.maximum(y-1, 0)), jnp.where(y_own, jnp.maximum(x-1, 0), x), b)
        toward = jnp.where((side == 0) | (side == 2), i32(-1), i32(1))
        edge = residual(jnp.where(y_own, y+toward, y), jnp.where(y_own, x, x+toward), b-1)
        inner = residual(jnp.where(y_own, y-toward, y), jnp.where(y_own, x, x-toward), b+1)
        lap = rn('sub', rn('add', rn('add', rn('add', plus, minus), edge), inner), rn('mul', jnp.float32(4), centre))
        tend = rn('sub', rn('mul', real(fcx, (b,), relax_lane), centre), rn('mul', real(gcx, (b,), relax_lane), lap))
        pt.store(out.at[k, y, x], tend, mask=relax_lane)
        if with_spec:
            tang = jnp.where(side < 2, y, x)
            pt.store(out.at[k, y, x], real(rate, (side, b, k, tang), spec_lane), mask=spec_lane)


def ring_relax_tendency(field, value, rate, fcx, gcx, *, spec, relax, mass=None):
    """Relax(+spec) tendency of a field toward REAL coupled boundary leaves (one kernel).

    ``field`` is already coupled, or uncoupled with ``mass=(mu, c1, c2)`` (coupled
    in-kernel, only at boundary cells). ``rate=None`` gives the relax band only.
    Storage-dtype operands are rounded to REAL at load; the result is REAL.
    """
    nz, ny, nx = field.shape
    with_spec = rate is not None
    mu, c1, c2 = (field[0], field[:, 0, 0], field[:, 0, 0]) if mass is None else mass
    arrays = [jnp.asarray(field), jnp.asarray(value), jnp.asarray(value if rate is None else rate),
              jnp.asarray(fcx, jnp.float32), jnp.asarray(gcx, jnp.float32),
              jnp.asarray(mu), jnp.asarray(c1), jnp.asarray(c2)]
    return pl.pallas_call(partial(_ring_relax, nz=nz, ny=ny, nx=nx, spec=int(spec), relax=int(relax),
        with_spec=with_spec, couple=mass is not None, interpret=_interpret()),
        out_shape=jax.ShapeDtypeStruct(field.shape, jnp.float32),
        grid=((nx+31)//32, ny, nz), interpret=_interpret(),
        compiler_params=pt.CompilerParams(num_warps=1), name='boundary_ring_relax_real')(*arrays)


def _ring_spec_final(value, mu, c1, c2, out, *, nz, ny, nx, spec, interpret):
    """WRF spec_bdy_final for mass-point scalars ('t'): bfield/(c1h*muts+c2h) on the spec zone (BD95).

    ``value`` is the stacked REAL coupled boundary value ``bdy + dtbc*bdy_tend``
    ``(species, side, width, z, side_len)``; ownership is the Fortran trim (Y sides
    own the corners, share/module_bc.F:2066-2216), every other cell writes zero.
    """
    block = 32
    i32 = jnp.int32
    x = pl.program_id(0)*block+jnp.arange(block, dtype=i32)
    y = jnp.broadcast_to(pl.program_id(1).astype(i32), (block,))
    sk = pl.program_id(2).astype(i32)
    s, k = jnp.broadcast_to(sk//nz, (block,)), jnp.broadcast_to(sk % nz, (block,))
    valid = x < nx
    dx, dy = jnp.minimum(x, nx-1-x), jnp.minimum(y, ny-1-y)
    y_own = (dy < spec) & (x >= dy) & (x < nx-dy)
    x_own = (dx < spec) & (y >= dx+1) & (y < ny-dx-1)
    b = jnp.where(y_own, dy, dx)
    lane = valid & (y_own | x_own)
    side = jnp.where(y_own, jnp.where(y < ny//2, i32(2), i32(3)), jnp.where(x < nx//2, i32(0), i32(1)))
    pt.store(out.at[s, k, y, x], jnp.zeros_like(x, dtype=jnp.float32), mask=valid)

    @pl.when(jnp.max(lane.astype(jnp.int32)) > 0)
    def spec_tile():
        rn = partial(_rn, interpret=interpret)

        def real(ref, idx):
            return pt.load(ref.at[idx], mask=lane).astype(jnp.float32)

        # xmu = c1(k)*mu(i,j)+c2(k); field = xmsf*bfield/xmu with xmsf = 1 ('t'), each op REAL.
        mass = rn('add', rn('mul', real(c1, (k,)), real(mu, (y, x))), real(c2, (k,)))
        tang = jnp.where(side < 2, y, x)
        pt.store(out.at[s, k, y, x], rn('div', real(value, (s, side, b, k, tang)), mass), mask=lane)


def ring_spec_final(values, mu, c1, c2, *, spec):
    """Stacked WRF ``spec_bdy_final`` scalar values (REAL) on the spec zone, zero elsewhere (one kernel).

    ``values`` ``(species, side, width>=spec, z, side_len)`` are coupled REAL boundary values
    at the step's ``dtbc``; ``mu`` is the end-of-step dry mass ``muts``. Storage-dtype
    operands are rounded to REAL at load.
    """
    n_species, _, _, nz, _ = values.shape
    ny, nx = mu.shape
    arrays = [jnp.asarray(values, jnp.float32), jnp.asarray(mu), jnp.asarray(c1), jnp.asarray(c2)]
    return pl.pallas_call(partial(_ring_spec_final, nz=nz, ny=ny, nx=nx, spec=int(spec), interpret=_interpret()),
        out_shape=jax.ShapeDtypeStruct((n_species, nz, ny, nx), jnp.float32),
        grid=((nx+31)//32, ny, n_species*nz), interpret=_interpret(),
        compiler_params=pt.CompilerParams(num_warps=1), name='boundary_spec_final_real')(*arrays)
