"""Native fp32 WRF SW adding-method recurrences in one Pallas launch."""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

TX = 4


def _kernel(pref, prefd, ptra, ptrad, beam, constants, down, up, direct,
            upward, upward_diffuse, *, nlev, ng):
    rr = pl.program_id(0) * TX + jnp.arange(TX)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    zero, one, eps = constants[0], constants[1], constants[2]
    ones = jnp.full((TX, ng), one, jnp.float32)
    zeros = jnp.zeros((TX, ng), jnp.float32)
    direct[rows, 0, points] = ones
    def beam_step(k, value):
        value = value * beam[rows, k, points]
        direct[rows, k + 1, points] = value
        return value
    jax.lax.fori_loop(0, nlev, beam_step, ones)
    u = pref[rows, nlev, points]
    ud = prefd[rows, nlev, points]
    upward[rows, nlev, points] = u
    upward_diffuse[rows, nlev, points] = ud
    def reflect_step(i, carry):
        u, ud = carry
        k = nlev - 1 - i
        r, rd = pref[rows, k, points], prefd[rows, k, points]
        t, td = ptra[rows, k, points], ptrad[rows, k, points]
        b = beam[rows, k, points]
        refl = one / jnp.maximum(one - ud * rd, eps)
        new_u = r + td * ((t - b) * ud + b * u) * refl
        new_ud = rd + td * td * ud * refl
        upward[rows, k, points] = new_u
        upward_diffuse[rows, k, points] = new_ud
        return new_u, new_ud
    jax.lax.fori_loop(0, nlev, reflect_step, (u, ud))
    def emit(k, transmission, diffuse):
        b = direct[rows, k, points]
        u, ud = upward[rows, k, points], upward_diffuse[rows, k, points]
        refl = one / jnp.maximum(one - diffuse * ud, eps)
        up[rows, k, points] = (b * u + (transmission - b) * ud) * refl
        down[rows, k, points] = b + (transmission - b + b * u * diffuse) * refl
    def transmit_step(k, carry):
        transmission, diffuse = carry
        emit(k, transmission, diffuse)
        b = direct[rows, k, points]
        r, rd = pref[rows, k, points], prefd[rows, k, points]
        t, td = ptra[rows, k, points], ptrad[rows, k, points]
        refl = one / jnp.maximum(one - rd * diffuse, eps)
        new_t = b * t + td * ((transmission - b) + b * r * diffuse) * refl
        new_d = rd + td * td * diffuse * refl
        return new_t, new_d
    transmission, diffuse = jax.lax.fori_loop(0, nlev, transmit_step, (ones, zeros))
    emit(nlev, transmission, diffuse)


def vertical_quadrature(pref, prefd, ptra, ptrad, direct_trans, *, interpret=False):
    """Same shape/order as RRTMG SW's reference vertical quadrature."""
    leading, nf, nb, ng = pref.shape[:-3], pref.shape[-3], pref.shape[-2], pref.shape[-1]
    ncol = (math.prod(leading) if leading else 1) * nb
    npad = ((ncol + TX - 1) // TX) * TX
    ngpad = 1 << (ng - 1).bit_length()
    def pack(a):
        # Treat independent bands as additional columns; pad only finite lanes.
        a = jnp.moveaxis(a, -2, -3).reshape(ncol, a.shape[-3], ng).astype(jnp.float32)
        return jnp.pad(a, ((0, npad - ncol), (0, 0), (0, ngpad - ng)))
    args = [pack(a) for a in (pref, prefd, ptra, ptrad, direct_trans)]
    args += [jnp.asarray([0, 1, 1e-12, 0], jnp.float32)]
    out = jax.ShapeDtypeStruct((npad, nf, ngpad), jnp.float32)
    result = pl.pallas_call(lambda *r: _kernel(*r, nlev=nf - 1, ng=ngpad),
        grid=(npad // TX,), out_shape=[out] * 5, interpret=interpret)(*args)
    def unpack(a):
        a = a[:ncol, :, :ng].reshape(leading + (nb, nf, ng))
        return jnp.moveaxis(a, -3, -2).astype(pref.dtype)
    return tuple(unpack(a) for a in result[:3])
