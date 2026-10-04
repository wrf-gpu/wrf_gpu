"""WRF previous-step precipitation passed to the next Noah-MP surface call."""
from typing import NamedTuple
import jax.numpy as jnp

class NoahMPPrecipitation(NamedTuple):
    prcpconv: object
    prcpnonc: object
    prcpsnow: object
    prcpgrpl: object
    prcphail: object

def seed_precipitation(state):
    zero = jnp.zeros(jnp.shape(state.t_skin), dtype=jnp.float32)
    return NoahMPPrecipitation(zero, zero, zero, zero, zero)

def precipitation_from_step(precip, convective_rate, dt, *, spec_zone=0):
    """Use instantaneous MP amounts (mm), never differences of rounded totals.

    WRF RAINNCV includes all phases; SNOWNCV includes snow+ice. Both the
    precipitation producer and its specified/nested exclusion match the MP
    driver's ownership. PRCPCONV is the actual held KF rain rate (mm/s).
    """
    p = {name: jnp.asarray(value, jnp.float32) for name, value in precip.items()}
    total = p['rain'] + p['snow'] + p['graupel'] + p['ice']
    snow = p['snow'] + p['ice']
    if spec_zone:
        ny, nx = total.shape
        mask = ((jnp.arange(ny) >= spec_zone) & (jnp.arange(ny) < ny - spec_zone))[:, None] & (
                (jnp.arange(nx) >= spec_zone) & (jnp.arange(nx) < nx - spec_zone))[None, :]
        total = jnp.where(mask, total, 0)
        snow = jnp.where(mask, snow, 0)
        p['graupel'] = jnp.where(mask, p['graupel'], 0)
    seconds = jnp.asarray(dt, jnp.float32)
    return NoahMPPrecipitation(jnp.asarray(convective_rate, jnp.float32), total / seconds,
                              snow / seconds, p['graupel'] / seconds, jnp.zeros_like(total))
