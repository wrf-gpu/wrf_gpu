"""CPU-harness safety: replace jaxlib's LAPACK tridiagonal_solve (E105 deadlock, INBOX 13:24Z) by a pure-JAX
Thomas scan.  Applied identically in every compared arm; it is NOT the product solver."""
import jax
import jax.numpy as jnp


def _thomas(dl, d, du, b):
    # dl/d/du: (..., m); b: (..., m, k).  dl[...,0] and du[...,-1] are ignored (LAPACK gtsv convention).
    def fwd(carry, xs):
        cp, dp = carry
        a, bb, c, r = xs
        den = bb - a * cp
        cn = c / den
        dn = (r - a[..., None] * dp) / den[..., None]
        return (cn, dn), (cn, dn)

    mv = lambda x: jnp.moveaxis(x, -1, 0)  # noqa: E731
    xs = (mv(dl), mv(d), mv(du), jnp.moveaxis(b, -2, 0))
    init = (jnp.zeros_like(d[..., 0]), jnp.zeros_like(b[..., 0, :]))
    _, (cps, dps) = jax.lax.scan(fwd, init, xs)

    def bwd(x_next, xs):
        cp, dp = xs
        x = dp - cp[..., None] * x_next
        return x, x

    _, xsol = jax.lax.scan(bwd, jnp.zeros_like(b[..., 0, :]), (cps, dps), reverse=True)
    return jnp.moveaxis(xsol, 0, -2)


def install() -> None:
    jax.lax.linalg.tridiagonal_solve = _thomas
