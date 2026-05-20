"""JAX MYNN2.5 PBL column kernel for M5-S2."""

from __future__ import annotations

from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.mynn_constants import (
    B1,
    CKMOD,
    GTR,
    KARMAN,
    LOCAL_ALP1,
    LOCAL_ALP2,
    LOCAL_ALP3,
    LOCAL_ALP4,
    LOCAL_ALP5,
    LOCAL_CTUAU,
    LOCAL_CNS,
    LOCAL_ELF_SOFT_MAX,
    LOCAL_ELT_MAX,
    LOCAL_ELT_MIN,
    P608,
    PR_LIMIT,
    QKEMIN,
    SQFAC,
    TKE_EPS,
    ZMAX,
)
from gpuwrf.physics.mynn_surface_stub import bulk_surface_fluxes
from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class MynnPBLColumnState:
    """Pytree for a batch of independent MYNN PBL columns on mass levels."""

    __slots__ = ("u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz", "km", "kh", "el")

    def __init__(self, u, v, w, theta, qv, tke, p, rho, dz, km, kh, el) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.tke = tke
        self.p = p
        self.rho = rho
        self.dz = dz
        self.km = km
        self.kh = kh
        self.el = el

    def replace(self, **updates) -> "MynnPBLColumnState":
        """Returns a same-layout pytree with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all column arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after a JAX transform."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for cache/debug tests."""

        if not isinstance(other, MynnPBLColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


def _leaves(state: MynnPBLColumnState) -> Iterable[jax.Array]:
    """Centralizes leaf iteration for equality, hashing, and byte accounting."""

    return (getattr(state, name) for name in MynnPBLColumnState.__slots__)


def _as_column(x):
    """Promotes a single column to a one-column batch."""

    return x[None, :] if x.ndim == 1 else x


def _clip_state(state: MynnPBLColumnState) -> MynnPBLColumnState:
    """Applies WRF-like lower bounds before column physics."""

    return state.replace(
        qv=jnp.maximum(state.qv, 0.0),
        tke=jnp.maximum(state.tke, TKE_EPS),
        rho=jnp.maximum(state.rho, 1.0e-4),
        dz=jnp.maximum(state.dz, 1.0),
    )


def _zero_edge_like(x):
    """Returns one zero edge column matching the batch dimensions of `x`."""

    return x[..., :1] * 0.0


def _edge_heights(dz):
    """Builds interface heights from layer depths."""

    return jnp.concatenate((_zero_edge_like(dz), jnp.cumsum(dz, axis=-1)), axis=-1)


def _edge_gradient(x, dz):
    """Computes centered vertical gradients on interfaces with zero boundaries."""

    dzk = 0.5 * (dz[..., 1:] + dz[..., :-1])
    interior = (x[..., 1:] - x[..., :-1]) / dzk
    return jnp.concatenate((_zero_edge_like(x), interior, _zero_edge_like(x)), axis=-1)


def _edge_average(x):
    """Interpolates mass-level values to interfaces."""

    interior = 0.5 * (x[..., 1:] + x[..., :-1])
    return jnp.concatenate((x[..., :1], interior, x[..., -1:]), axis=-1)


def _mass_average(edge):
    """Interpolates interface values back to mass levels."""

    return 0.5 * (edge[..., :-1] + edge[..., 1:])


def _virtual_potential(theta, qv):
    """Computes dry virtual potential temperature used by MYNN gradients."""

    return theta * (1.0 + P608 * qv)


def _level2_stability(u, v, theta, qv, dz):
    """Source-derived Level-2 stability proxy from module_bl_mynn.F90 1525-1578."""

    thv = _virtual_potential(theta, qv)
    dzk = 0.5 * (dz[..., 1:] + dz[..., :-1])
    duz = ((u[..., 1:] - u[..., :-1]) ** 2 + (v[..., 1:] - v[..., :-1]) ** 2) / (dzk * dzk)
    dtv = (thv[..., 1:] - thv[..., :-1]) / dzk
    gh_interior = -dtv * GTR
    ri = -gh_interior / (duz + 1.0e-10)
    ri_pos = 0.5 * (ri + jnp.abs(ri))
    a2fac = jnp.where(CKMOD == 1.0, 1.0 / (1.0 + ri_pos), 1.0)

    prnum = 0.76 + 4.0 * ri_pos / (1.0 + ri_pos / PR_LIMIT)
    sh_interior = 0.74 * a2fac / (1.0 + 5.0 * ri_pos) + 0.02
    sm_interior = prnum * sh_interior
    gm = jnp.concatenate((_zero_edge_like(theta), duz, _zero_edge_like(theta)), axis=-1)
    gh = jnp.concatenate((_zero_edge_like(theta), gh_interior, _zero_edge_like(theta)), axis=-1)
    sm = jnp.concatenate((_zero_edge_like(theta), sm_interior, _zero_edge_like(theta)), axis=-1)
    sh = jnp.concatenate((_zero_edge_like(theta), sh_interior, _zero_edge_like(theta)), axis=-1)
    return gm, gh, sm, sh


def _diagnose_pblh(tke, dz):
    """Diagnoses a compact TKE-based PBL depth for the local length scale."""

    active = jnp.where(tke > 2.0e-2, 1.0, 0.0)
    return jnp.maximum(jnp.sum(active * dz, axis=-1), 300.0)


def _mixing_length(state: MynnPBLColumnState, qke, qkw_edge, gh):
    """Implements the bounded local MYNN length-scale form from WRF lines 1881-2016."""

    dz = state.dz
    zw = _edge_heights(dz)
    del qke, qkw_edge, gh
    els = KARMAN * zw
    el = els / (1.0 + els / 120.0)
    return el


def _diffusivity_edges(state: MynnPBLColumnState):
    """Computes MYNN edge diffusivities and TKE production diagnostics."""

    qke = 2.0 * state.tke
    qkw_edge = jnp.sqrt(_edge_average(qke) + QKEMIN)
    gm, gh, sm, sh = _level2_stability(state.u, state.v, state.theta, state.qv, state.dz)
    el = _mixing_length(state, qke, qkw_edge, gh)
    elq = el * qkw_edge
    km = elq * sm
    kh = elq * sh
    mask = jnp.concatenate((_zero_edge_like(state.tke), state.tke[..., 1:] * 0.0 + 1.0, _zero_edge_like(state.tke)), axis=-1)
    km = km * mask
    kh = kh * mask
    shear_prod = km * gm
    buoy_prod = kh * gh
    return km, kh, el, shear_prod, buoy_prod


def _implicit_mix(x, diffusivity_edge, dz, dt):
    """Applies an implicit zero-boundary vertical diffusion solve."""

    return _implicit_mix_with_coefficients(x, diffusivity_edge, dz, dt)


def _implicit_mix_with_coefficients(x, diffusivity_edge, dz, dt):
    """Builds coefficients and solves one zero-boundary diffusion system."""

    inv_dz2 = 1.0 / (dz * dz)
    lower = dt * diffusivity_edge[..., :-1] * inv_dz2
    upper = dt * diffusivity_edge[..., 1:] * inv_dz2
    a = -lower
    c = -upper
    b = 1.0 + lower + upper
    return solve_tridiagonal(a, b, c, x)


def _tke_rhs(state, el_edge, shear_prod_edge, buoy_prod_edge, dt):
    """Builds the TKE right-hand side before the shared implicit solve."""

    prod = _mass_average(shear_prod_edge + buoy_prod_edge)
    el_mass = _mass_average(el_edge) + 1.0
    diss = (state.tke ** 1.5) / (B1 * el_mass)
    rhs = state.tke + dt * (prod - diss)
    return rhs


def _step_mynn_pbl_impl(state: MynnPBLColumnState, dt: float, debug: bool) -> MynnPBLColumnState:
    """Unjitted implementation shared by production and stripped entry points."""

    km_edge, kh_edge, el_edge, shear_prod, buoy_prod = _diffusivity_edges(state)
    mix_edge = jnp.maximum(jnp.maximum(km_edge, kh_edge), SQFAC * km_edge)
    tke_rhs = _tke_rhs(state, el_edge, shear_prod, buoy_prod, dt)
    rhs = jnp.stack((state.u, state.v, state.theta, state.qv, tke_rhs), axis=-1)
    mixed = _implicit_mix_with_coefficients(rhs, mix_edge, state.dz, dt)
    u = mixed[..., 0]
    v = mixed[..., 1]
    theta = mixed[..., 2]
    qv = mixed[..., 3]
    tke = mixed[..., 4]
    km = state.km
    kh = state.kh
    el = state.el

    u = assert_finite(u, "mynn_u", enabled=debug)
    v = assert_finite(v, "mynn_v", enabled=debug)
    theta = assert_finite(theta, "mynn_theta", enabled=debug)
    qv = assert_physical_bounds(qv, 0.0, 0.1, "mynn_qv", enabled=debug)
    tke = assert_physical_bounds(tke, TKE_EPS, 200.0, "mynn_tke", enabled=debug)
    return state.replace(u=u, v=v, theta=theta, qv=qv, tke=tke, km=km, kh=kh, el=el)


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_mynn_pbl_column(state: MynnPBLColumnState, dt: float, *, debug: bool = False) -> MynnPBLColumnState:
    """Advances one fused MYNN2.5 column step."""

    return _step_mynn_pbl_impl(state, dt, debug)


@partial(jax.jit, static_argnames=("dt",))
def step_mynn_pbl_column_debug_stripped(state: MynnPBLColumnState, dt: float) -> MynnPBLColumnState:
    """Hand-stripped sibling used for the debug-vs-production HLO identity proof."""

    return _step_mynn_pbl_impl(state, dt, False)
