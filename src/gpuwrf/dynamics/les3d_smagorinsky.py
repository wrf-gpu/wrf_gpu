"""WRF ``diff_opt=2`` / ``km_opt=3`` 3-D Smagorinsky diffusion (literal port).

Literal JAX transcription of the pristine WRF v4.7.1 ``dyn_em/module_diffusion_em.F``
sequence that ``module_first_rk_step_part2.F`` runs for ``diff_opt=2``:

``compute_diff_metrics`` -> ``set_physical_bc3d`` (rdzw/rdz/z ``'w'``, zx ``'e'``,
zy ``'f'``) -> ``cal_deform_and_div`` -> ``calculate_km_kh`` (``calculate_N2`` moist
BN2 + ``smag_km``) -> ``phy_bc`` -> [``vertical_diffusion_2`` when
``bl_pbl_physics == 0``] -> ``horizontal_diffusion_2``.

Every routine keeps WRF's loop ranges for specified/nested and periodic lateral
conditions, its halo copies (``module_bc.set_physical_bc3d`` open/specified and
periodic branches), the terrain-following ``zx``/``zy`` metric terms, the
zero-initialised outer coefficient ring (``smag_km`` never writes it), and
Fortran's left-to-right operation order.  Arrays live in a halo-padded "WRF
memory" layout ``(k, j, i)`` with Fortran index ``i`` at position ``i - ims``
(``ims = 1 - H``), ``j`` at ``j - jms`` and ``k`` at ``k - 1``.

The returned increments are the pure diffusion contributions that WRF adds to
``ru_tendf``/``rv_tendf``/``rw_tendf``/``t_tendf``/``moist_tend`` (coupled, the
same units WRF's RK1-frozen ``*_tendf`` carry).  ``sfs_opt``/``m_opt`` > 0,
``damp_opt=1`` and polar grids are not ported (fail closed in the caller).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# module_model_constants (REAL parameters; folded expressions evaluated in REAL4).
_G = np.float32(9.81)
_R_D = np.float32(287.0)
_CP = np.float32(np.float32(7.0) * _R_D / np.float32(2.0))
_R_V = np.float32(461.6)
_XLV = 2.5e6
_SVP1 = 0.6112
_SVP2 = 17.67
_SVP3 = 29.65
_SVPT0 = 273.15
_EP_2 = np.float32(_R_D / _R_V)
_P1000MB = 100000.0
_RVOVRD = np.float32(_R_V / _R_D)
_RCP = np.float32(_R_D / _CP)
_PRANDTL = np.float32(np.float32(1.0) / np.float32(3.0))
_EPSILON = 1.0e-15
_QC_CR = 0.00001
# gfortran folds constant sub-expressions in REAL4 (left-to-right operands).
_ES_COEF = np.float32(np.float32(1000.0) * np.float32(_SVP1))  # 1000.0*SVP1 (calculate_N2)
_G_1P61 = np.float32(_G * np.float32(1.61))  # g*1.61 (vertical_diffusion_2 moist-theta flux)

HALO = 3


def _k(c, like):
    """Opaque scalar divisor: keep WRF's IEEE ``x / const``.

    XLA's algebraic simplifier rewrites ``x / constant`` into ``x * (1/constant)``
    (e.g. ``BN2/prandtl`` -> ``BN2*3.0``), which gfortran -O2 does not; the
    rewrite changes REAL4 results and is amplified by smag_km's
    ``sqrt(def2 - BN2/pr)`` near-cancellation.  A scalar barrier keeps the
    divisor a runtime operand (no data movement; elementwise fusion unchanged).
    """
    return jax.lax.optimization_barrier(jnp.asarray(c, dtype=jnp.result_type(like)))


@dataclasses.dataclass(frozen=True)
class Les3dConfig:
    """Static WRF configuration of the diff_opt=2/km_opt=3 operator.

    ``boundary`` is ``"specified"`` (also nested children: identical ranges) or
    ``"periodic"`` (periodic_x and periodic_y).  ``vertical`` mirrors WRF's
    ``bl_pbl_physics == 0`` gate on ``vertical_diffusion_2``.  ``iqv``/``iqc``/
    ``iqi`` are the positions of qv/qc/qi in ``Les3dInputs.moist`` (WRF order).
    """

    boundary: str = "specified"
    isotropic: int = 0
    c_s: float = 0.25
    mix_upper_bound: float = 0.1
    vertical: bool = False
    isfflx: int = 1
    tke_drag_coefficient: float = 0.0
    tke_heat_flux: float = 0.0
    use_theta_m: int = 1
    mix_full_fields: bool = False
    moist_mix2_off: bool = False
    iqv: int = 0
    iqc: int | None = 1
    iqi: int | None = None
    km_opt: int = 3
    c_k: float = 0.15
    tke_mix2_off: bool = False

    def __post_init__(self) -> None:
        if self.boundary not in ("specified", "periodic"):
            raise ValueError(f"unsupported les3d boundary {self.boundary!r}")
        if self.km_opt not in (2, 3):
            raise ValueError("les3d supports km_opt=3 (smag_km) and km_opt=2 (tke_km/tke_rhs)")
        if self.isfflx not in (0, 1, 2):
            raise ValueError("isfflx value invalid for diff_opt=2")


class Les3dInputs(NamedTuple):
    """Operator inputs in the port's interior layout ``(k, j, i)``.

    ``moist`` holds the WRF moist species in WRF index order (positions of
    qv/qc/qi are static ``Les3dConfig`` fields).  1-D eta
    arrays have length ``nz`` or ``nz + 1`` (WRF ``kms:kme``).
    """

    u: jax.Array
    v: jax.Array
    w: jax.Array
    thp: jax.Array
    th_phy: jax.Array
    t_phy: jax.Array
    p_phy: jax.Array
    p8w: jax.Array
    t8w: jax.Array
    ph: jax.Array
    phb: jax.Array
    rho: jax.Array
    moist: tuple[jax.Array, ...]
    msftx: jax.Array
    msfty: jax.Array
    msfux: jax.Array
    msfuy: jax.Array
    msfvx: jax.Array
    msfvy: jax.Array
    fnm: jax.Array
    fnp: jax.Array
    dn: jax.Array
    dnw: jax.Array
    cf1: float
    cf2: float
    cf3: float
    rdx: float
    rdy: float
    dx: float
    dy: float
    dt: float
    ust: jax.Array | None = None
    hfx: jax.Array | None = None
    qfx: jax.Array | None = None
    u_base: jax.Array | None = None
    v_base: jax.Array | None = None
    t_base: jax.Array | None = None
    qv_base: jax.Array | None = None
    tke: jax.Array | None = None   # km_opt=2: grid%tke_2 (mass levels)
    mut: jax.Array | None = None   # km_opt=2: grid%mut (ny, nx)
    c1h: jax.Array | None = None   # km_opt=2: grid%c1h
    c2h: jax.Array | None = None   # km_opt=2: grid%c2h


class Les3dResult(NamedTuple):
    """Memory-layout (halo-padded) operator outputs."""

    z: jax.Array
    rdz: jax.Array
    rdzw: jax.Array
    zx: jax.Array
    zy: jax.Array
    div: jax.Array
    defor11: jax.Array
    defor22: jax.Array
    defor33: jax.Array
    defor12: jax.Array
    defor13: jax.Array
    defor23: jax.Array
    bn2: jax.Array
    xkmh: jax.Array
    xkmv: jax.Array
    xkhh: jax.Array
    xkhv: jax.Array
    ru_tendf: jax.Array
    rv_tendf: jax.Array
    rw_tendf: jax.Array
    t_tendf: jax.Array
    moist_tendf: tuple[jax.Array, ...]
    tke_tendf: jax.Array | None = None


class Les3dTendencies(NamedTuple):
    """Interior diffusion increments (``u``/``v``/``w``/mass grids)."""

    ru_tendf: jax.Array
    rv_tendf: jax.Array
    rw_tendf: jax.Array
    t_tendf: jax.Array
    moist_tendf: tuple[jax.Array, ...]
    xkmh: jax.Array
    xkmv: jax.Array
    xkhh: jax.Array
    xkhv: jax.Array
    tke_tendf: jax.Array | None = None


# --------------------------------------------------------------------------- #
# WRF memory-index helpers                                                     #
# --------------------------------------------------------------------------- #
class _Mem:
    """Fortran index arithmetic on a halo-padded ``(k, j, i)`` array."""

    def __init__(self, nx: int, ny: int, nz: int, dtype) -> None:
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ids, self.ide = 1, nx + 1
        self.jds, self.jde = 1, ny + 1
        self.kds, self.kde = 1, nz + 1
        self.ims, self.ime = 1 - HALO, nx + 1 + HALO
        self.jms, self.jme = 1 - HALO, ny + 1 + HALO
        self.its, self.ite, self.jts, self.jte = self.ids, self.ide, self.jds, self.jde
        self.kts, self.kte = self.kds, self.kde
        self.ktf = min(self.kte, self.kde - 1)
        self.shape = (self.kde, self.jme - self.jms + 1, self.ime - self.ims + 1)
        self.dtype = dtype

    def zeros(self) -> jax.Array:
        return jnp.zeros(self.shape, dtype=self.dtype)

    def _ks(self, k, d=0):
        return slice(k[0] - 1 + d, k[1] + d)

    def _js(self, j, d=0):
        return slice(j[0] - self.jms + d, j[1] - self.jms + 1 + d)

    def _is(self, i, d=0):
        return slice(i[0] - self.ims + d, i[1] - self.ims + 1 + d)

    def r(self, a, k, j, i, dk=0, dj=0, di=0):
        """``a(i+di, k+dk, j+dj)`` over the inclusive Fortran ranges."""
        return a[self._ks(k, dk), self._js(j, dj), self._is(i, di)]

    def s(self, a, k, j, i, value):
        """``a(i, k, j) = value`` over the inclusive Fortran ranges."""
        if k[0] > k[1] or j[0] > j[1] or i[0] > i[1]:
            return a
        return a.at[self._ks(k), self._js(j), self._is(i)].set(value)

    def v1(self, f, k, dk=0):
        """1-D ``f(k+dk)`` broadcast against a 3-D region."""
        return f[self._ks(k, dk)][:, None, None]

    def r2(self, a, j, i, dj=0, di=0):
        """2-D memory array ``a(i+di, j+dj)`` broadcast against a 3-D region."""
        return a[None, self._js(j, dj), self._is(i, di)]

    def put(self, interior, *, istag: int = 0, jstag: int = 0, kfull: bool = False):
        """Place an interior ``(k, j, i)`` field at Fortran ``(1.., jds.., ids..)``."""
        out = self.zeros()
        nk = interior.shape[0]
        nj = self.ny + jstag
        ni = self.nx + istag
        return out.at[0:nk, self._js((self.jds, self.jds + nj - 1)), self._is((self.ids, self.ids + ni - 1))].set(
            jnp.asarray(interior, dtype=self.dtype)
        )

    def put2(self, interior, *, istag: int = 0, jstag: int = 0):
        out = jnp.zeros(self.shape[1:], dtype=self.dtype)
        nj = self.ny + jstag
        ni = self.nx + istag
        return out.at[self._js((self.jds, self.jds + nj - 1)), self._is((self.ids, self.ids + ni - 1))].set(
            jnp.asarray(interior, dtype=self.dtype)
        )

    def col(self, f):
        """1-D eta array padded to ``kms:kme`` (length ``kde``)."""
        if f is None:
            return jnp.zeros((self.kde,), dtype=self.dtype)
        f = jnp.asarray(f, dtype=self.dtype).reshape(-1)
        if f.shape[0] == self.kde:
            return f
        return jnp.concatenate([f, jnp.zeros((self.kde - f.shape[0],), dtype=self.dtype)])

    def interior(self, a, *, istag: int = 0, jstag: int = 0, nk: int | None = None):
        nk = self.nz if nk is None else nk
        return a[0:nk, self._js((self.jds, self.jde - 1 + jstag)), self._is((self.ids, self.ide - 1 + istag))]


def _bc3(m: _Mem, a, var: str, periodic: bool):
    """``module_bc.set_physical_bc3d`` for one tile == one patch == the domain."""

    istag = 0 if var in ("u", "x", "d", "e") else -1
    jstag = 0 if var in ("v", "y", "d", "f") else -1
    k_end = m.kde if var in ("w", "e", "f") else max(1, min(m.kde - 1, m.kte))
    kr = (m.kts, k_end)
    ids, ide, jds, jde = m.ids, m.ide, m.jds, m.jde
    allj = (m.jms, m.jme)
    alli = (m.ims, m.ime)
    if periodic:
        jr = (max(jds, m.jts - 1), min(m.jte + 1, jde + jstag))
        for ii in range(0, -4, -1):  # dat(ids+i-1) = dat(ide+i-1), i = 0..-(bdyzone-1)
            if ids + ii - 1 >= m.ims:
                a = m.s(a, kr, jr, (ids + ii - 1, ids + ii - 1), m.r(a, kr, jr, (ide + ii - 1, ide + ii - 1)))
        for ii in range(-istag, 5):  # dat(ide+i+istag) = dat(ids+i+istag)
            if ide + ii + istag <= m.ime:
                a = m.s(a, kr, jr, (ide + ii + istag, ide + ii + istag), m.r(a, kr, jr, (ids + ii + istag, ids + ii + istag)))
        for jj in range(0, -4, -1):
            if jds + jj - 1 >= m.jms:
                a = m.s(a, kr, (jds + jj - 1, jds + jj - 1), alli, m.r(a, kr, (jde + jj - 1, jde + jj - 1), alli))
        for jj in range(-jstag, 5):
            if jde + jj + jstag <= m.jme:
                a = m.s(a, kr, (jde + jj + jstag, jde + jj + jstag), alli, m.r(a, kr, (jds + jj + jstag, jds + jj + jstag), alli))
        return a
    src = m.r(a, kr, allj, (ids, ids))
    a = m.s(a, kr, allj, (ids - 3, ids - 1), jnp.broadcast_to(src, (src.shape[0], src.shape[1], 3)))
    if var not in ("u", "x"):
        src = m.r(a, kr, allj, (ide - 1, ide - 1))
        a = m.s(a, kr, allj, (ide, ide + 2), jnp.broadcast_to(src, (src.shape[0], src.shape[1], 3)))
    else:
        src = m.r(a, kr, allj, (ide, ide))
        a = m.s(a, kr, allj, (ide + 1, ide + 3), jnp.broadcast_to(src, (src.shape[0], src.shape[1], 3)))
    src = m.r(a, kr, (jds, jds), alli)
    a = m.s(a, kr, (jds - 3, jds - 1), alli, jnp.broadcast_to(src, (src.shape[0], 3, src.shape[2])))
    if var not in ("v", "y"):
        src = m.r(a, kr, (jde - 1, jde - 1), alli)
        a = m.s(a, kr, (jde, jde + 2), alli, jnp.broadcast_to(src, (src.shape[0], 3, src.shape[2])))
    else:
        src = m.r(a, kr, (jde, jde), alli)
        a = m.s(a, kr, (jde + 1, jde + 3), alli, jnp.broadcast_to(src, (src.shape[0], 3, src.shape[2])))
    return a


def _bc2(m: _Mem, a2, var: str, periodic: bool):
    """``module_bc.set_physical_bc2d`` via the 3-D copy on a one-level view."""

    a3 = jnp.broadcast_to(a2[None], (m.kde,) + a2.shape)
    return _bc3(m, a3, var if var != "w" else "t", periodic)[0]


# --------------------------------------------------------------------------- #
# WRF routines                                                                 #
# --------------------------------------------------------------------------- #
def _compute_diff_metrics(m: _Mem, ph, phb, rdx, rdy, periodic: bool):
    ktf, kte = m.ktf, m.kte
    ids, ide, jds, jde = m.ids, m.ide, m.jds, m.jde
    z, rdz, rdzw, zx, zy = m.zeros(), m.zeros(), m.zeros(), m.zeros(), m.zeros()
    # j = jts-1 .. jte;  (j_start >= jts) is false -> i = its .. min(ite, ide-1)
    jr = (m.jts - 1, m.jte)
    ir = (m.its, min(m.ite, ide - 1))
    z_at_w = (ph + phb) / _k(_G, ph)
    rdzw = m.s(rdzw, (1, ktf), jr, ir, 1.0 / (m.r(z_at_w, (1, ktf), jr, ir, dk=1) - m.r(z_at_w, (1, ktf), jr, ir)))
    rdz = m.s(rdz, (2, ktf), jr, ir, 2.0 / (m.r(z_at_w, (2, ktf), jr, ir, dk=1) - m.r(z_at_w, (2, ktf), jr, ir, dk=-1)))
    rdz = m.s(rdz, (1, 1), jr, ir, 2.0 / (m.r(z_at_w, (2, 2), jr, ir) - m.r(z_at_w, (1, 1), jr, ir)))
    j_s, j_e = m.jts, min(m.jte, jde - 1)
    i_s, i_e = m.its, min(m.ite, ide - 1)
    kr = (1, kte)
    ixr = (max(ids + 1, m.its), i_e)
    zx = m.s(zx, kr, (j_s, j_e), ixr, rdx * (m.r(phb, kr, (j_s, j_e), ixr) - m.r(phb, kr, (j_s, j_e), ixr, di=-1)) / _k(_G, ph))
    zx = m.s(zx, kr, (j_s, j_e), ixr, m.r(zx, kr, (j_s, j_e), ixr)
             + rdx * (m.r(ph, kr, (j_s, j_e), ixr) - m.r(ph, kr, (j_s, j_e), ixr, di=-1)) / _k(_G, ph))
    jyr = (max(jds + 1, m.jts), j_e)
    zy = m.s(zy, kr, jyr, (i_s, i_e), rdy * (m.r(phb, kr, jyr, (i_s, i_e)) - m.r(phb, kr, jyr, (i_s, i_e), dj=-1)) / _k(_G, ph))
    zy = m.s(zy, kr, jyr, (i_s, i_e), m.r(zy, kr, jyr, (i_s, i_e))
             + rdy * (m.r(ph, kr, jyr, (i_s, i_e)) - m.r(ph, kr, jyr, (i_s, i_e), dj=-1)) / _k(_G, ph))
    kf = (1, ktf)
    if not periodic:
        zx = m.s(zx, kf, (j_s, j_e), (ide, ide), 0.0)
        zx = m.s(zx, kf, (j_s, j_e), (ids, ids), 0.0)
        zy = m.s(zy, kf, (jde, jde), (i_s, i_e), 0.0)
        zy = m.s(zy, kf, (jds, jds), (i_s, i_e), 0.0)
    else:
        for ib in (ide, ids):
            zx = m.s(zx, kf, (j_s, j_e), (ib, ib), rdx * (m.r(phb, kf, (j_s, j_e), (ib, ib)) - m.r(phb, kf, (j_s, j_e), (ib, ib), di=-1)) / _k(_G, ph))
            zx = m.s(zx, kf, (j_s, j_e), (ib, ib), m.r(zx, kf, (j_s, j_e), (ib, ib))
                     + rdx * (m.r(ph, kf, (j_s, j_e), (ib, ib)) - m.r(ph, kf, (j_s, j_e), (ib, ib), di=-1)) / _k(_G, ph))
        for jb in (jde, jds):
            zy = m.s(zy, kf, (jb, jb), (i_s, i_e), rdy * (m.r(phb, kf, (jb, jb), (i_s, i_e)) - m.r(phb, kf, (jb, jb), (i_s, i_e), dj=-1)) / _k(_G, ph))
            zy = m.s(zy, kf, (jb, jb), (i_s, i_e), m.r(zy, kf, (jb, jb), (i_s, i_e))
                     + rdy * (m.r(ph, kf, (jb, jb), (i_s, i_e)) - m.r(ph, kf, (jb, jb), (i_s, i_e), dj=-1)) / _k(_G, ph))
    z = m.s(z, kf, (j_s, j_e), (i_s, i_e), 0.5 * (
        m.r(ph, kf, (j_s, j_e), (i_s, i_e)) + m.r(phb, kf, (j_s, j_e), (i_s, i_e))
        + m.r(ph, kf, (j_s, j_e), (i_s, i_e), dk=1) + m.r(phb, kf, (j_s, j_e), (i_s, i_e), dk=1)) / _k(_G, ph))
    return z, rdz, rdzw, zx, zy


def _cal_deform_and_div(m: _Mem, cfgp, u, v, w, rdz, rdzw, zx, zy, msfux, msfuy, msfvx, msfvy, msftx, msfty,
                        fnm, fnp, dn, dnw, cf1, cf2, cf3, rdx, rdy, u_base, v_base, mix_full_fields):
    periodic = cfgp
    spec = not periodic
    kts, kte, ktf = m.kts, m.kte, m.ktf
    ids, ide, jds, jde = m.ids, m.ide, m.jds, m.jde
    ktes1, ktes2 = kte - 1, kte - 2
    cft2 = -0.5 * dnw[ktes1 - 1] / dn[ktes1 - 1]
    cft1 = 1.0 - cft2
    div, d11, d22, d33, d12, d13, d23 = (m.zeros() for _ in range(7))
    kr = (kts, ktf)
    # ---------------- defor11 / div ----------------
    i_s, i_e = m.its, min(m.ite, ide - 1)
    j_s, j_e = m.jts, min(m.jte, jde - 1)
    jr, ir = (j_s, j_e), (i_s, i_e)
    mm = m.r2(msftx, jr, ir) * m.r2(msfty, jr, ir)
    hat = m.zeros()
    hat = m.s(hat, kr, jr, (i_s, i_e + 1), m.r(u, kr, jr, (i_s, i_e + 1)) / m.r2(msfuy, jr, (i_s, i_e + 1)))
    hatavg = m.zeros()
    ka = (kts + 1, ktf)
    hatavg = m.s(hatavg, ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(hat, ka, jr, ir) + m.r(hat, ka, jr, ir, di=1))
        + m.v1(fnp, ka) * (m.r(hat, ka, jr, ir, dk=-1) + m.r(hat, ka, jr, ir, dk=-1, di=1))))
    h = lambda kk, di=0, dj=0: m.r(hat, (kk, kk), jr, ir, di=di, dj=dj)  # noqa: E731
    hatavg = m.s(hatavg, (1, 1), jr, ir, 0.5 * (
        cf1 * h(1) + cf2 * h(2) + cf3 * h(3) + cf1 * h(1, di=1) + cf2 * h(2, di=1) + cf3 * h(3, di=1)))
    hatavg = m.s(hatavg, (kte, kte), jr, ir, 0.5 * (
        cft1 * (h(ktes1) + h(ktes1, di=1)) + cft2 * (h(ktes2) + h(ktes2, di=1))))
    tmpzx = 0.25 * (m.r(zx, kr, jr, ir) + m.r(zx, kr, jr, ir, di=1) + m.r(zx, kr, jr, ir, dk=1) + m.r(zx, kr, jr, ir, dk=1, di=1))
    tmp1 = (m.r(hatavg, kr, jr, ir, dk=1) - m.r(hatavg, kr, jr, ir)) * tmpzx * m.r(rdzw, kr, jr, ir)
    tmp1 = mm * (rdx * (m.r(hat, kr, jr, ir, di=1) - m.r(hat, kr, jr, ir)) - tmp1)
    d11 = m.s(d11, kr, jr, ir, 2.0 * tmp1)
    div = m.s(div, kr, jr, ir, tmp1)
    # ---------------- defor22 / div ----------------
    hat = m.zeros()
    hat = m.s(hat, kr, (j_s, j_e + 1), ir, m.r(v, kr, (j_s, j_e + 1), ir) / m.r2(msfvx, (j_s, j_e + 1), ir))
    hatavg = m.zeros()
    hatavg = m.s(hatavg, ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(hat, ka, jr, ir) + m.r(hat, ka, jr, ir, dj=1))
        + m.v1(fnp, ka) * (m.r(hat, ka, jr, ir, dk=-1) + m.r(hat, ka, jr, ir, dk=-1, dj=1))))
    hatavg = m.s(hatavg, (1, 1), jr, ir, 0.5 * (
        cf1 * h(1) + cf2 * h(2) + cf3 * h(3) + cf1 * h(1, dj=1) + cf2 * h(2, dj=1) + cf3 * h(3, dj=1)))
    hatavg = m.s(hatavg, (kte, kte), jr, ir, 0.5 * (
        cft1 * (h(ktes1) + h(ktes1, dj=1)) + cft2 * (h(ktes2) + h(ktes2, dj=1))))
    tmpzy = 0.25 * (m.r(zy, kr, jr, ir) + m.r(zy, kr, jr, ir, dj=1) + m.r(zy, kr, jr, ir, dk=1) + m.r(zy, kr, jr, ir, dk=1, dj=1))
    tmp1 = (m.r(hatavg, kr, jr, ir, dk=1) - m.r(hatavg, kr, jr, ir)) * tmpzy * m.r(rdzw, kr, jr, ir)
    tmp1 = mm * (rdy * (m.r(hat, kr, jr, ir, dj=1) - m.r(hat, kr, jr, ir)) - tmp1)
    d22 = m.s(d22, kr, jr, ir, 2.0 * tmp1)
    div = m.s(div, kr, jr, ir, m.r(div, kr, jr, ir) + tmp1)
    # ---------------- defor33 / div ----------------
    tmp1 = (m.r(w, kr, jr, ir, dk=1) - m.r(w, kr, jr, ir)) * m.r(rdzw, kr, jr, ir)
    d33 = m.s(d33, kr, jr, ir, 2.0 * tmp1)
    div = m.s(div, kr, jr, ir, m.r(div, kr, jr, ir) + tmp1)
    # ---------------- defor12 ----------------
    i_s, i_e, j_s, j_e = m.its, m.ite, m.jts, m.jte
    if spec:
        i_s, i_e, j_s, j_e = max(ids + 1, m.its), min(ide - 1, m.ite), max(jds + 1, m.jts), min(jde - 1, m.jte)
    else:
        i_s, i_e = m.its, m.ite
    jr, ir = (j_s, j_e), (i_s, i_e)
    mm = 0.25 * (m.r2(msfux, jr, ir, dj=-1) + m.r2(msfux, jr, ir)) * (m.r2(msfvy, jr, ir, di=-1) + m.r2(msfvy, jr, ir))
    hat = m.zeros()
    hat = m.s(hat, kr, (j_s - 1, j_e), ir, m.r(u, kr, (j_s - 1, j_e), ir) / m.r2(msfux, (j_s - 1, j_e), ir))
    hatavg = m.zeros()
    hatavg = m.s(hatavg, ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(hat, ka, jr, ir, dj=-1) + m.r(hat, ka, jr, ir))
        + m.v1(fnp, ka) * (m.r(hat, ka, jr, ir, dk=-1, dj=-1) + m.r(hat, ka, jr, ir, dk=-1))))
    h = lambda kk, di=0, dj=0: m.r(hat, (kk, kk), jr, ir, di=di, dj=dj)  # noqa: E731
    hatavg = m.s(hatavg, (1, 1), jr, ir, 0.5 * (
        cf1 * h(1, dj=-1) + cf2 * h(2, dj=-1) + cf3 * h(3, dj=-1) + cf1 * h(1) + cf2 * h(2) + cf3 * h(3)))
    hatavg = m.s(hatavg, (kte, kte), jr, ir, 0.5 * (
        cft1 * (h(ktes1, dj=-1) + h(ktes1)) + cft2 * (h(ktes2, dj=-1) + h(ktes2))))
    tmpzy = 0.25 * (m.r(zy, kr, jr, ir, di=-1) + m.r(zy, kr, jr, ir) + m.r(zy, kr, jr, ir, dk=1, di=-1) + m.r(zy, kr, jr, ir, dk=1))
    tmp1 = (m.r(hatavg, kr, jr, ir, dk=1) - m.r(hatavg, kr, jr, ir)) * 0.25 * tmpzy * (
        m.r(rdzw, kr, jr, ir) + m.r(rdzw, kr, jr, ir, di=-1) + m.r(rdzw, kr, jr, ir, di=-1, dj=-1) + m.r(rdzw, kr, jr, ir, dj=-1))
    d12 = m.s(d12, kr, jr, ir, mm * (rdy * (m.r(hat, kr, jr, ir) - m.r(hat, kr, jr, ir, dj=-1)) - tmp1))
    hat = m.zeros()
    hat = m.s(hat, kr, jr, (i_s - 1, i_e), m.r(v, kr, jr, (i_s - 1, i_e)) / m.r2(msfvy, jr, (i_s - 1, i_e)))
    hatavg = m.zeros()
    hatavg = m.s(hatavg, ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(hat, ka, jr, ir, di=-1) + m.r(hat, ka, jr, ir))
        + m.v1(fnp, ka) * (m.r(hat, ka, jr, ir, dk=-1, di=-1) + m.r(hat, ka, jr, ir, dk=-1))))
    hatavg = m.s(hatavg, (1, 1), jr, ir, 0.5 * (
        cf1 * h(1, di=-1) + cf2 * h(2, di=-1) + cf3 * h(3, di=-1) + cf1 * h(1) + cf2 * h(2) + cf3 * h(3)))
    hatavg = m.s(hatavg, (kte, kte), jr, ir, 0.5 * (
        cft1 * (h(ktes1) + h(ktes1, di=-1)) + cft2 * (h(ktes2) + h(ktes2, di=-1))))
    tmpzx = 0.25 * (m.r(zx, kr, jr, ir, dj=-1) + m.r(zx, kr, jr, ir) + m.r(zx, kr, jr, ir, dk=1, dj=-1) + m.r(zx, kr, jr, ir, dk=1))
    tmp1 = (m.r(hatavg, kr, jr, ir, dk=1) - m.r(hatavg, kr, jr, ir)) * 0.25 * tmpzx * (
        m.r(rdzw, kr, jr, ir) + m.r(rdzw, kr, jr, ir, dj=-1) + m.r(rdzw, kr, jr, ir, di=-1, dj=-1) + m.r(rdzw, kr, jr, ir, di=-1))
    d12 = m.s(d12, kr, jr, ir, m.r(d12, kr, jr, ir) + mm * (rdx * (m.r(hat, kr, jr, ir) - m.r(hat, kr, jr, ir, di=-1)) - tmp1))
    kall = (kts, kte)
    if spec and i_s == ids + 1:
        d12 = m.s(d12, kall, (m.jts, m.jte), (ids, ids), m.r(d12, kall, (m.jts, m.jte), (ids + 1, ids + 1)))
    if spec and j_s == jds + 1:
        d12 = m.s(d12, kall, (jds, jds), (m.its, m.ite), m.r(d12, kall, (jds + 1, jds + 1), (m.its, m.ite)))
    if spec and i_e == ide - 1:
        d12 = m.s(d12, kall, (m.jts, m.jte), (ide, ide), m.r(d12, kall, (m.jts, m.jte), (ide - 1, ide - 1)))
    if spec and j_e == jde - 1:
        d12 = m.s(d12, kall, (jde, jde), (m.its, m.ite), m.r(d12, kall, (jde - 1, jde - 1), (m.its, m.ite)))
    # ---------------- defor13 ----------------
    i_s, i_e, j_s, j_e = m.its, min(m.ite, ide - 1), m.jts, min(m.jte, jde - 1)
    if spec:
        i_s, j_s = max(ids + 1, m.its), max(jds + 1, m.jts)
    else:
        i_s, i_e, j_e = m.its, min(m.ite, ide), min(m.jte, jde)
    jr, ir = (j_s, j_e), (i_s, i_e)
    mm = m.r2(msfux, jr, ir) * m.r2(msfuy, jr, ir)
    kw = (kts, kte)
    hat = m.zeros()
    hat = m.s(hat, kw, jr, ir, m.r(w, kw, jr, ir) / m.r2(msfty, jr, ir))
    jcol = (j_s, min(m.jte, jde - 1))
    hat = m.s(hat, kw, jcol, (i_s - 1, i_s - 1), m.r(w, kw, jcol, (i_s - 1, i_s - 1)) / m.r2(msfty, jcol, (i_s - 1, i_s - 1)))
    irow = (i_s, min(m.ite, ide - 1))
    hat = m.s(hat, kw, (j_s - 1, j_s - 1), irow, m.r(w, kw, (j_s - 1, j_s - 1), irow) / m.r2(msfty, (j_s - 1, j_s - 1), irow))
    hatavg = m.zeros()
    hatavg = m.s(hatavg, kr, jr, ir, 0.25 * (
        m.r(hat, kr, jr, ir) + m.r(hat, kr, jr, ir, dk=1) + m.r(hat, kr, jr, ir, di=-1) + m.r(hat, kr, jr, ir, dk=1, di=-1)))
    tmp1 = (m.r(hatavg, ka, jr, ir) - m.r(hatavg, ka, jr, ir, dk=-1)) * m.r(zx, ka, jr, ir) * 0.5 * (
        m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, di=-1))
    d13 = m.s(d13, ka, jr, ir, mm * (rdx * (m.r(hat, ka, jr, ir) - m.r(hat, ka, jr, ir, di=-1)) - tmp1))
    d13 = m.s(d13, (kts, kts), jr, ir, 0.0)
    d13 = m.s(d13, (ktf + 1, ktf + 1), jr, ir, 0.0)
    if mix_full_fields:
        tmp1 = (m.r(u, ka, jr, ir) - m.r(u, ka, jr, ir, dk=-1)) * 0.5 * (m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, di=-1))
    else:
        tmp1 = (m.r(u, ka, jr, ir) - m.v1(u_base, ka) - m.r(u, ka, jr, ir, dk=-1) + m.v1(u_base, ka, dk=-1)) * 0.5 * (
            m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, di=-1))
    d13 = m.s(d13, ka, jr, ir, m.r(d13, ka, jr, ir) + tmp1)
    # ---------------- defor23 ----------------
    i_s, i_e, j_s, j_e = m.its, min(m.ite, ide - 1), m.jts, min(m.jte, jde - 1)
    if spec:
        i_s, j_s = max(ids + 1, m.its), max(jds + 1, m.jts)
    else:
        j_e = min(m.jte, jde)
        i_s, i_e = m.its, min(m.ite, ide - 1)
    jr, ir = (j_s, j_e), (i_s, i_e)
    mm = m.r2(msfvx, jr, ir) * m.r2(msfvy, jr, ir)
    hat = m.zeros()
    hat = m.s(hat, kw, jr, ir, m.r(w, kw, jr, ir) / m.r2(msftx, jr, ir))
    jcol = (j_s, min(m.jte, jde - 1))
    hat = m.s(hat, kw, jcol, (i_s - 1, i_s - 1), m.r(w, kw, jcol, (i_s - 1, i_s - 1)) / m.r2(msftx, jcol, (i_s - 1, i_s - 1)))
    irow = (i_s, min(m.ite, ide - 1))
    hat = m.s(hat, kw, (j_s - 1, j_s - 1), irow, m.r(w, kw, (j_s - 1, j_s - 1), irow) / m.r2(msftx, (j_s - 1, j_s - 1), irow))
    hatavg = m.zeros()
    hatavg = m.s(hatavg, kr, jr, ir, 0.25 * (
        m.r(hat, kr, jr, ir) + m.r(hat, kr, jr, ir, dk=1) + m.r(hat, kr, jr, ir, dj=-1) + m.r(hat, kr, jr, ir, dk=1, dj=-1)))
    tmp1 = (m.r(hatavg, ka, jr, ir) - m.r(hatavg, ka, jr, ir, dk=-1)) * m.r(zy, ka, jr, ir) * 0.5 * (
        m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, dj=-1))
    d23 = m.s(d23, ka, jr, ir, mm * (rdy * (m.r(hat, ka, jr, ir) - m.r(hat, ka, jr, ir, dj=-1)) - tmp1))
    d23 = m.s(d23, (kts, kts), jr, ir, 0.0)
    d23 = m.s(d23, (ktf + 1, ktf + 1), jr, ir, 0.0)
    if mix_full_fields:
        tmp1 = (m.r(v, ka, jr, ir) - m.r(v, ka, jr, ir, dk=-1)) * 0.5 * (m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, dj=-1))
    else:
        tmp1 = (m.r(v, ka, jr, ir) - m.v1(v_base, ka) - m.r(v, ka, jr, ir, dk=-1) + m.v1(v_base, ka, dk=-1)) * 0.5 * (
            m.r(rdz, ka, jr, ir) + m.r(rdz, ka, jr, ir, dj=-1))
    d23 = m.s(d23, ka, jr, ir, m.r(d23, ka, jr, ir) + tmp1)
    if spec and i_s == ids + 1:
        d13 = m.s(d13, kall, (m.jts, m.jte), (ids, ids), m.r(d13, kall, (m.jts, m.jte), (ids + 1, ids + 1)))
        d23 = m.s(d23, kall, (m.jts, m.jte), (ids, ids), m.r(d23, kall, (m.jts, m.jte), (ids + 1, ids + 1)))
    if spec and j_s == jds + 1:
        d13 = m.s(d13, kall, (jds, jds), (m.its, m.ite), m.r(d13, kall, (jds + 1, jds + 1), (m.its, m.ite)))
        d23 = m.s(d23, kall, (jds, jds), (m.its, m.ite), m.r(d23, kall, (jds + 1, jds + 1), (m.its, m.ite)))
    if spec and i_e == ide - 1:
        d13 = m.s(d13, kall, (m.jts, m.jte), (ide, ide), m.r(d13, kall, (m.jts, m.jte), (ide - 1, ide - 1)))
        d23 = m.s(d23, kall, (m.jts, m.jte), (ide, ide), m.r(d23, kall, (m.jts, m.jte), (ide - 1, ide - 1)))
    if spec and j_e == jde - 1:
        d13 = m.s(d13, kall, (jde, jde), (m.its, m.ite), m.r(d13, kall, (jde - 1, jde - 1), (m.its, m.ite)))
        d23 = m.s(d23, kall, (jde, jde), (m.its, m.ite), m.r(d23, kall, (jde - 1, jde - 1), (m.its, m.ite)))
    return div, d11, d22, d33, d12, d13, d23


def _mass_ranges(m: _Mem, periodic: bool, *, shrink_end: bool = True):
    """WRF mass-point loop range used by calculate_N2/smag_km/*_s/*_w routines."""
    i_s, i_e, j_s, j_e = m.its, min(m.ite, m.ide - 1), m.jts, min(m.jte, m.jde - 1)
    if not periodic:
        i_s, i_e = max(m.ids + 1, m.its), min(m.ide - 2, m.ite)
        j_s, j_e = max(m.jds + 1, m.jts), min(m.jde - 2, m.jte)
    return (j_s, j_e), (i_s, i_e)


def _calculate_n2(m: _Mem, periodic, moist, iqv, iqc, iqi, theta, t, p, p8w, t8w, dnw, dn, rdz, rdzw, cf1, cf2, cf3):
    kts, kte, ktf = m.kts, m.kte, m.ktf
    ktes1, ktes2 = kte - 1, kte - 2
    jr, ir = _mass_ranges(m, periodic)
    kr = (kts, ktf)
    bn2 = m.zeros()
    qv = moist[iqv]
    qctmp = m.zeros()
    if iqc is not None:
        qctmp = m.s(qctmp, kr, jr, ir, m.r(moist[iqc], kr, jr, ir))
    tmp1 = m.zeros()
    tmp1sfc = jnp.zeros(bn2.shape[1:], dtype=bn2.dtype)[None]
    tmp1top = jnp.zeros(bn2.shape[1:], dtype=bn2.dtype)[None]
    jsl, isl = m._js(jr), m._is(ir)
    for ispe, q in enumerate(moist):
        if ispe in (iqv, iqc, iqi):
            tmp1 = m.s(tmp1, kr, jr, ir, m.r(tmp1, kr, jr, ir) + m.r(q, kr, jr, ir))
            q1, q2, q3 = (m.r(q, (kk, kk), jr, ir) for kk in (1, 2, 3))
            tmp1sfc = tmp1sfc.at[:, jsl, isl].set(tmp1sfc[:, jsl, isl] + cf1 * q1 + cf2 * q2 + cf3 * q3)
            qa, qb = m.r(q, (ktes1, ktes1), jr, ir), m.r(q, (ktes2, ktes2), jr, ir)
            tmp1top = tmp1top.at[:, jsl, isl].set(
                tmp1top[:, jsl, isl] + qa + (qa - qb) * 0.5 * dnw[ktes1 - 1] / dn[ktes1 - 1])
    tc = m.r(t, kr, jr, ir) - _SVPT0
    es = _ES_COEF * jnp.exp(_SVP2 * tc / (m.r(t, kr, jr, ir) - _SVP3))
    qvs = m.zeros()
    qvs = m.s(qvs, kr, jr, ir, _EP_2 * es / (m.r(p, kr, jr, ir) - es))

    def coefa_of(kk):
        xlvqv = _XLV * m.r(qv, kk, jr, ir)
        tk = m.r(t, kk, jr, ir)
        return (1.0 + xlvqv / _k(_R_D, tk) / tk) / (1.0 + _XLV * xlvqv / _k(_CP, tk) / _k(_R_V, tk) / tk / tk) / m.r(theta, kk, jr, ir)

    ki = (kts + 1, ktf - 1)
    tmpdz = 1.0 / m.r(rdz, ki, jr, ir) + 1.0 / m.r(rdz, ki, jr, ir, dk=1)
    sat = (m.r(qv, ki, jr, ir) >= m.r(qvs, ki, jr, ir)) | (m.r(qctmp, ki, jr, ir) >= _QC_CR)
    thetaep1 = m.r(theta, ki, jr, ir, dk=1) * (1.0 + _XLV * m.r(qvs, ki, jr, ir, dk=1) / _k(_CP, t) / m.r(t, ki, jr, ir, dk=1))
    thetaem1 = m.r(theta, ki, jr, ir, dk=-1) * (1.0 + _XLV * m.r(qvs, ki, jr, ir, dk=-1) / _k(_CP, t) / m.r(t, ki, jr, ir, dk=-1))
    dtmp1 = m.r(tmp1, ki, jr, ir, dk=1) - m.r(tmp1, ki, jr, ir, dk=-1)
    bn2_sat = _G * (coefa_of(ki) * (thetaep1 - thetaem1) / tmpdz - dtmp1 / tmpdz)
    bn2_dry = _G * ((m.r(theta, ki, jr, ir, dk=1) - m.r(theta, ki, jr, ir, dk=-1)) / m.r(theta, ki, jr, ir) / tmpdz
                    + 1.61 * (m.r(qv, ki, jr, ir, dk=1) - m.r(qv, ki, jr, ir, dk=-1)) / tmpdz
                    - dtmp1 / tmpdz)
    bn2 = m.s(bn2, ki, jr, ir, jnp.where(sat, bn2_sat, bn2_dry))
    k1 = (kts, kts)
    tmpdz = 1.0 / m.r(rdz, k1, jr, ir, dk=1) + 0.5 / m.r(rdzw, k1, jr, ir)
    thetasfc = m.r(t8w, k1, jr, ir) / jnp.power(m.r(p8w, k1, jr, ir) / _k(_P1000MB, p8w), _RCP)
    sat = (m.r(qv, k1, jr, ir) >= m.r(qvs, k1, jr, ir)) | (m.r(qctmp, k1, jr, ir) >= _QC_CR)
    qvsfc_s = cf1 * m.r(qvs, (1, 1), jr, ir) + cf2 * m.r(qvs, (2, 2), jr, ir) + cf3 * m.r(qvs, (3, 3), jr, ir)
    thetaep1 = m.r(theta, k1, jr, ir, dk=1) * (1.0 + _XLV * m.r(qvs, k1, jr, ir, dk=1) / _k(_CP, t) / m.r(t, k1, jr, ir, dk=1))
    thetaesfc = thetasfc * (1.0 + _XLV * qvsfc_s / _k(_CP, t8w) / m.r(t8w, k1, jr, ir))
    sfc = tmp1sfc[:, m._js(jr), m._is(ir)]
    dts = m.r(tmp1, k1, jr, ir, dk=1) - sfc
    bn2_sat = _G * (coefa_of(k1) * (thetaep1 - thetaesfc) / tmpdz - dts / tmpdz)
    qvsfc_d = cf1 * m.r(qv, (1, 1), jr, ir) + cf2 * m.r(qv, (2, 2), jr, ir) + cf3 * m.r(qv, (3, 3), jr, ir)
    tmpdz_d = 1.0 / m.r(rdzw, k1, jr, ir)
    bn2_dry = _G * ((m.r(theta, k1, jr, ir, dk=1) - m.r(theta, k1, jr, ir)) / m.r(theta, k1, jr, ir) / tmpdz_d
                    + 1.61 * (m.r(qv, k1, jr, ir, dk=1) - qvsfc_d) / tmpdz_d
                    - dts / tmpdz_d)
    bn2 = m.s(bn2, k1, jr, ir, jnp.where(sat, bn2_sat, bn2_dry))
    bn2 = m.s(bn2, (ktf, ktf), jr, ir, m.r(bn2, (ktf - 1, ktf - 1), jr, ir))
    return bn2


def _smag_km(m: _Mem, periodic, bn2, d11, d22, d33, d12, d13, d23, rdzw, msftx, msfty, dx, dy, dt,
             c_s, mix_upper_bound, isotropic):
    jr, ir = _mass_ranges(m, periodic)
    kr = (m.kts, m.ktf)
    pr = _k(_PRANDTL, d11)
    dt = _k(dt, d11)
    def2 = 0.5 * (m.r(d11, kr, jr, ir) * m.r(d11, kr, jr, ir) + m.r(d22, kr, jr, ir) * m.r(d22, kr, jr, ir)
                  + m.r(d33, kr, jr, ir) * m.r(d33, kr, jr, ir))
    tmp = 0.25 * (m.r(d12, kr, jr, ir) + m.r(d12, kr, jr, ir, dj=1) + m.r(d12, kr, jr, ir, di=1) + m.r(d12, kr, jr, ir, di=1, dj=1))
    def2 = def2 + tmp * tmp
    tmp = 0.25 * (m.r(d13, kr, jr, ir, dk=1) + m.r(d13, kr, jr, ir) + m.r(d13, kr, jr, ir, dk=1, di=1) + m.r(d13, kr, jr, ir, di=1))
    def2 = def2 + tmp * tmp
    tmp = 0.25 * (m.r(d23, kr, jr, ir, dk=1) + m.r(d23, kr, jr, ir) + m.r(d23, kr, jr, ir, dk=1, dj=1) + m.r(d23, kr, jr, ir, dj=1))
    def2 = def2 + tmp * tmp
    mtx, mty = m.r2(msftx, jr, ir), m.r2(msfty, jr, ir)
    rz = m.r(rdzw, kr, jr, ir)
    tmp = jnp.sqrt(jnp.maximum(0.0, def2 - m.r(bn2, kr, jr, ir) / pr))
    if int(isotropic) == 0:
        mlen_h = jnp.sqrt(dx / mtx * dy / mty)
        mlen_v = 1.0 / rz
        xkmh = jnp.maximum(c_s * c_s * mlen_h * mlen_h * tmp, 1.0e-6 * mlen_h * mlen_h)
        xkmh = jnp.minimum(xkmh, mix_upper_bound * mlen_h * mlen_h / dt)
        xkmv = jnp.maximum(c_s * c_s * mlen_v * mlen_v * tmp, 1.0e-6 * mlen_v * mlen_v)
        xkmv = jnp.minimum(xkmv, mix_upper_bound * mlen_v * mlen_v / dt)
        xkhh = jnp.minimum(xkmh / pr, mix_upper_bound * mlen_h * mlen_h / dt)
        xkhv = jnp.minimum(xkmv / pr, mix_upper_bound * mlen_v * mlen_v / dt)
    else:
        deltas = jnp.power(dx / mtx * dy / mty / rz, np.float32(0.33333333))
        xkmh = jnp.maximum(c_s * c_s * deltas * deltas * tmp, 1.0e-6 * deltas * deltas)
        xkmh = jnp.minimum(xkmh, mix_upper_bound * dx / mtx * dy / mty / dt)
        xkmv = jnp.minimum(xkmh, mix_upper_bound / rz / rz / dt)
        xkhh = jnp.minimum(xkmh / pr, mix_upper_bound * dx / mtx * dy / mty / dt)
        xkhv = jnp.minimum(xkmv / pr, mix_upper_bound / rz / rz / dt)
    out = []
    for val in (xkmh, xkmv, xkhh, xkhv):
        out.append(m.s(m.zeros(), kr, jr, ir, val.astype(m.dtype)))
    return tuple(out)


def _ranges_spec(m, periodic, *, i_end_off, j_end_off, i_px_end_off, i_s_ext=0, i_e_ext=0, j_s_ext=0, j_e_ext=0,
                 i_base_end=None, j_base_end=None):
    """Generic WRF ``i_start..i_end, j_start..j_end`` with spec/periodic edits + ext."""
    i_s = m.its
    i_e = m.ite if i_base_end is None else i_base_end
    j_s = m.jts
    j_e = m.jte if j_base_end is None else j_base_end
    if not periodic:
        i_s, i_e = max(m.ids + 1, m.its), min(m.ide - i_end_off, m.ite)
        j_s, j_e = max(m.jds + 1, m.jts), min(m.jde - j_end_off, m.jte)
    else:
        i_s, i_e = m.its, min(m.ite, m.ide - i_px_end_off)
    return (j_s - j_s_ext, j_e + j_e_ext), (i_s - i_s_ext, i_e + i_e_ext)


def _titau_11_22_33(m, periodic, xkx, defor, rho, ext):
    jr, ir = _ranges_spec(m, periodic, i_end_off=1, j_end_off=1, i_px_end_off=0,
                          i_s_ext=ext[0], i_e_ext=ext[1], j_s_ext=ext[2], j_e_ext=ext[3])
    kr = (m.kts, m.ktf)
    titau = m.zeros()
    return m.s(titau, kr, jr, ir, -m.r(rho, kr, jr, ir) * m.r(xkx, kr, jr, ir) * m.r(defor, kr, jr, ir))


def _titau_12_21(m, periodic, xkx, defor, rho, ext):
    jr, ir = _ranges_spec(m, periodic, i_end_off=1, j_end_off=1, i_px_end_off=0,
                          i_s_ext=ext[0], i_e_ext=ext[1], j_s_ext=ext[2], j_e_ext=ext[3])
    kr = (m.kts, m.ktf)
    rhoavg = 0.25 * (m.r(rho, kr, jr, ir, di=-1) + m.r(rho, kr, jr, ir)
                     + m.r(rho, kr, jr, ir, di=-1, dj=-1) + m.r(rho, kr, jr, ir, dj=-1))
    xkxavg = rhoavg * 0.25 * (m.r(xkx, kr, jr, ir, di=-1) + m.r(xkx, kr, jr, ir)
                              + m.r(xkx, kr, jr, ir, di=-1, dj=-1) + m.r(xkx, kr, jr, ir, dj=-1))
    return m.s(m.zeros(), kr, jr, ir, -xkxavg * m.r(defor, kr, jr, ir))


def _titau_13_31(m, periodic, defor, xkx, fnm, fnp, rho, ext):
    jr, ir = _ranges_spec(m, periodic, i_end_off=1, j_end_off=2, i_px_end_off=0,
                          i_s_ext=ext[0], i_e_ext=ext[1], j_s_ext=ext[2], j_e_ext=ext[3],
                          j_base_end=min(m.jte, m.jde - 1))
    ka = (m.kts + 1, m.ktf)
    rhoavg = 0.5 * (m.v1(fnm, ka) * (m.r(rho, ka, jr, ir, di=-1) + m.r(rho, ka, jr, ir))
                    + m.v1(fnp, ka) * (m.r(rho, ka, jr, ir, dk=-1, di=-1) + m.r(rho, ka, jr, ir, dk=-1)))
    xkxavg = rhoavg * 0.5 * (m.v1(fnm, ka) * (m.r(xkx, ka, jr, ir) + m.r(xkx, ka, jr, ir, di=-1))
                             + m.v1(fnp, ka) * (m.r(xkx, ka, jr, ir, dk=-1) + m.r(xkx, ka, jr, ir, dk=-1, di=-1)))
    titau = m.s(m.zeros(), ka, jr, ir, -xkxavg * m.r(defor, ka, jr, ir))
    titau = m.s(titau, (m.kts, m.kts), jr, ir, 0.0)
    return m.s(titau, (m.ktf + 1, m.ktf + 1), jr, ir, 0.0)


def _titau_23_32(m, periodic, defor, xkx, fnm, fnp, rho, ext):
    jr, ir = _ranges_spec(m, periodic, i_end_off=2, j_end_off=1, i_px_end_off=1,
                          i_s_ext=ext[0], i_e_ext=ext[1], j_s_ext=ext[2], j_e_ext=ext[3],
                          i_base_end=min(m.ite, m.ide - 1))
    ka = (m.kts + 1, m.ktf)
    rhoavg = 0.5 * (m.v1(fnm, ka) * (m.r(rho, ka, jr, ir) + m.r(rho, ka, jr, ir, dj=-1))
                    + m.v1(fnp, ka) * (m.r(rho, ka, jr, ir, dk=-1) + m.r(rho, ka, jr, ir, dk=-1, dj=-1)))
    xkxavg = rhoavg * 0.5 * (m.v1(fnm, ka) * (m.r(xkx, ka, jr, ir) + m.r(xkx, ka, jr, ir, dj=-1))
                             + m.v1(fnp, ka) * (m.r(xkx, ka, jr, ir, dk=-1) + m.r(xkx, ka, jr, ir, dk=-1, dj=-1)))
    titau = m.s(m.zeros(), ka, jr, ir, -xkxavg * m.r(defor, ka, jr, ir))
    titau = m.s(titau, (m.kts, m.kts), jr, ir, 0.0)
    return m.s(titau, (m.ktf + 1, m.ktf + 1), jr, ir, 0.0)


def _horizontal_diffusion_u_2(m, periodic, tend, d11, d12, xkmh, rho, msfux, msfuy, rdx, rdy, fnm, fnp, dnw,
                              zx, zy, rdzw):
    jr, ir = _ranges_spec(m, periodic, i_end_off=1, j_end_off=2, i_px_end_off=0, j_base_end=min(m.jte, m.jde - 1))
    titau1 = _titau_11_22_33(m, periodic, xkmh, d11, rho, (1, 0, 0, 0))
    titau2 = _titau_12_21(m, periodic, xkmh, d12, rho, (0, 0, 0, 1))
    kts, ktf = m.kts, m.ktf
    ka, kr = (kts + 1, ktf), (kts, ktf)
    t1avg = m.zeros()
    t2avg = m.zeros()
    t1avg = m.s(t1avg, ka, jr, ir, 0.5 * (m.v1(fnm, ka) * (m.r(titau1, ka, jr, ir, di=-1) + m.r(titau1, ka, jr, ir))
                                          + m.v1(fnp, ka) * (m.r(titau1, ka, jr, ir, dk=-1, di=-1) + m.r(titau1, ka, jr, ir, dk=-1))))
    t2avg = m.s(t2avg, ka, jr, ir, 0.5 * (m.v1(fnm, ka) * (m.r(titau2, ka, jr, ir, dj=1) + m.r(titau2, ka, jr, ir))
                                          + m.v1(fnp, ka) * (m.r(titau2, ka, jr, ir, dk=-1, dj=1) + m.r(titau2, ka, jr, ir, dk=-1))))
    zx_at_u = 0.5 * (m.r(zx, kr, jr, ir) + m.r(zx, kr, jr, ir, dk=1))
    zy_at_u = 0.125 * (m.r(zy, kr, jr, ir, di=-1) + m.r(zy, kr, jr, ir) + m.r(zy, kr, jr, ir, di=-1, dj=1)
                       + m.r(zy, kr, jr, ir, dj=1) + m.r(zy, kr, jr, ir, dk=1, di=-1) + m.r(zy, kr, jr, ir, dk=1)
                       + m.r(zy, kr, jr, ir, dk=1, di=-1, dj=1) + m.r(zy, kr, jr, ir, dk=1, dj=1))
    mrdx = m.r2(msfux, jr, ir) * rdx
    mrdy = m.r2(msfuy, jr, ir) * rdy
    tmpdz = (1.0 / m.r(rdzw, kr, jr, ir) + 1.0 / m.r(rdzw, kr, jr, ir, di=-1)) / 2.0
    inc = _G * tmpdz / m.v1(dnw, kr) * (
        mrdx * (m.r(titau1, kr, jr, ir) - m.r(titau1, kr, jr, ir, di=-1))
        + mrdy * (m.r(titau2, kr, jr, ir, dj=1) - m.r(titau2, kr, jr, ir))
        - m.r2(msfux, jr, ir) * zx_at_u * (m.r(t1avg, kr, jr, ir, dk=1) - m.r(t1avg, kr, jr, ir)) / tmpdz
        - m.r2(msfuy, jr, ir) * zy_at_u * (m.r(t2avg, kr, jr, ir, dk=1) - m.r(t2avg, kr, jr, ir)) / tmpdz)
    return m.s(tend, kr, jr, ir, m.r(tend, kr, jr, ir) + inc)


def _horizontal_diffusion_v_2(m, periodic, tend, d12, d22, xkmh, rho, msfvx, msfvy, rdx, rdy, fnm, fnp, dnw,
                              zx, zy, rdzw):
    jr, ir = _ranges_spec(m, periodic, i_end_off=2, j_end_off=1, i_px_end_off=1, i_base_end=min(m.ite, m.ide - 1))
    titau1 = _titau_12_21(m, periodic, xkmh, d12, rho, (0, 1, 0, 0))
    titau2 = _titau_11_22_33(m, periodic, xkmh, d22, rho, (0, 0, 1, 0))
    kts, ktf = m.kts, m.ktf
    ka, kr = (kts + 1, ktf), (kts, ktf)
    t1avg = m.s(m.zeros(), ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(titau1, ka, jr, ir, di=1) + m.r(titau1, ka, jr, ir))
        + m.v1(fnp, ka) * (m.r(titau1, ka, jr, ir, dk=-1, di=1) + m.r(titau1, ka, jr, ir, dk=-1))))
    t2avg = m.s(m.zeros(), ka, jr, ir, 0.5 * (
        m.v1(fnm, ka) * (m.r(titau2, ka, jr, ir, dj=-1) + m.r(titau2, ka, jr, ir))
        + m.v1(fnp, ka) * (m.r(titau2, ka, jr, ir, dk=-1, dj=-1) + m.r(titau2, ka, jr, ir, dk=-1))))
    zx_at_v = 0.125 * (m.r(zx, kr, jr, ir) + m.r(zx, kr, jr, ir, di=1) + m.r(zx, kr, jr, ir, dj=-1)
                       + m.r(zx, kr, jr, ir, di=1, dj=-1) + m.r(zx, kr, jr, ir, dk=1) + m.r(zx, kr, jr, ir, dk=1, di=1)
                       + m.r(zx, kr, jr, ir, dk=1, dj=-1) + m.r(zx, kr, jr, ir, dk=1, di=1, dj=-1))
    zy_at_v = 0.5 * (m.r(zy, kr, jr, ir) + m.r(zy, kr, jr, ir, dk=1))
    mrdx = m.r2(msfvx, jr, ir) * rdx
    mrdy = m.r2(msfvy, jr, ir) * rdy
    tmpdz = (1.0 / m.r(rdzw, kr, jr, ir) + 1.0 / m.r(rdzw, kr, jr, ir, dj=-1)) / 2.0
    inc = _G * tmpdz / m.v1(dnw, kr) * (
        mrdy * (m.r(titau2, kr, jr, ir) - m.r(titau2, kr, jr, ir, dj=-1))
        + mrdx * (m.r(titau1, kr, jr, ir, di=1) - m.r(titau1, kr, jr, ir))
        - m.r2(msfvx, jr, ir) * zx_at_v * (m.r(t1avg, kr, jr, ir, dk=1) - m.r(t1avg, kr, jr, ir)) / tmpdz
        - m.r2(msfvy, jr, ir) * zy_at_v * (m.r(t2avg, kr, jr, ir, dk=1) - m.r(t2avg, kr, jr, ir)) / tmpdz)
    return m.s(tend, kr, jr, ir, m.r(tend, kr, jr, ir) + inc)


def _horizontal_diffusion_w_2(m, periodic, tend, d13, d23, xkmv, rho, msftx, msfty, rdx, rdy, fnm, fnp, dn,
                              zx, zy, rdz):
    jr, ir = _mass_ranges(m, periodic)
    titau1 = _titau_13_31(m, periodic, d13, xkmv, fnm, fnp, rho, (0, 1, 0, 0))
    titau2 = _titau_23_32(m, periodic, d23, xkmv, fnm, fnp, rho, (0, 0, 0, 1))
    kts, ktf = m.kts, m.ktf
    kr = (kts, ktf)
    t1avg = m.s(m.zeros(), kr, jr, ir, 0.25 * (
        m.r(titau1, kr, jr, ir, dk=1, di=1) + m.r(titau1, kr, jr, ir, dk=1)
        + m.r(titau1, kr, jr, ir, di=1) + m.r(titau1, kr, jr, ir)))
    t2avg = m.s(m.zeros(), kr, jr, ir, 0.25 * (
        m.r(titau2, kr, jr, ir, dk=1, dj=1) + m.r(titau2, kr, jr, ir, dk=1)
        + m.r(titau2, kr, jr, ir, dj=1) + m.r(titau2, kr, jr, ir)))
    ka = (kts + 1, ktf)
    zx_at_w = 0.5 * (m.r(zx, ka, jr, ir) + m.r(zx, ka, jr, ir, di=1))
    zy_at_w = 0.5 * (m.r(zy, ka, jr, ir) + m.r(zy, ka, jr, ir, dj=1))
    mrdx = m.r2(msftx, jr, ir) * rdx
    mrdy = m.r2(msfty, jr, ir) * rdy
    rz = m.r(rdz, ka, jr, ir)
    inc = _G / (m.v1(dn, ka) * rz) * (
        mrdx * (m.r(titau1, ka, jr, ir, di=1) - m.r(titau1, ka, jr, ir))
        + mrdy * (m.r(titau2, ka, jr, ir, dj=1) - m.r(titau2, ka, jr, ir))
        - m.r2(msfty, jr, ir) * rz * (zx_at_w * (m.r(t1avg, ka, jr, ir) - m.r(t1avg, ka, jr, ir, dk=-1))
                                      + zy_at_w * (m.r(t2avg, ka, jr, ir) - m.r(t2avg, ka, jr, ir, dk=-1))))
    return m.s(tend, ka, jr, ir, m.r(tend, ka, jr, ir) + inc)


def _horizontal_diffusion_s(m, periodic, tend, var, xkhh, rho, msftx, msfty, msfux, msfvy, rdx, rdy, fnm, fnp,
                            cf1, cf2, cf3, zx, zy, rdzw, dnw, dn, doing_tke=False):
    kts, kte, ktf = m.kts, m.kte, m.ktf
    ktes1, ktes2 = kte - 1, kte - 2
    jr, ir = _mass_ranges(m, periodic)
    (j_s, j_e), (i_s, i_e) = jr, ir
    kr, ka = (kts, ktf), (kts + 1, ktf)
    xr = (i_s, i_e + 1)
    yr = (j_s, j_e + 1)
    dnw1e, dn1e = dnw[ktes1 - 1], dn[ktes1 - 1]
    # x fluxes on u faces i = i_start .. i_end+1
    xkxavg = 0.5 * (m.r(xkhh, kr, jr, xr, di=-1) + m.r(xkhh, kr, jr, xr)) * 0.5 * (
        m.r(rho, kr, jr, xr, di=-1) + m.r(rho, kr, jr, xr))
    h1avg = m.zeros()
    h1avg = m.s(h1avg, ka, jr, xr, 0.5 * (m.v1(fnm, ka) * (m.r(var, ka, jr, xr, di=-1) + m.r(var, ka, jr, xr))
                                          + m.v1(fnp, ka) * (m.r(var, ka, jr, xr, dk=-1, di=-1) + m.r(var, ka, jr, xr, dk=-1))))
    vv = lambda kk, di=0, dj=0, jrr=jr, irr=xr: m.r(var, (kk, kk), jrr, irr, di=di, dj=dj)  # noqa: E731
    h1avg = m.s(h1avg, (kts, kts), jr, xr, 0.5 * (
        cf1 * vv(1) + cf2 * vv(2) + cf3 * vv(3) + cf1 * vv(1, di=-1) + cf2 * vv(2, di=-1) + cf3 * vv(3, di=-1)))
    h1avg = m.s(h1avg, (ktf + 1, ktf + 1), jr, xr, 0.5 * (
        vv(ktes1) + (vv(ktes1) - vv(ktes2)) * 0.5 * dnw1e / dn1e + vv(ktes1, di=-1)
        + (vv(ktes1, di=-1) - vv(ktes2, di=-1)) * 0.5 * dnw1e / dn1e))
    tmpzx = 0.5 * (m.r(zx, kr, jr, xr) + m.r(zx, kr, jr, xr, dk=1))
    rdzu = 2.0 / (1.0 / m.r(rdzw, kr, jr, xr) + 1.0 / m.r(rdzw, kr, jr, xr, di=-1))
    h1 = m.s(m.zeros(), kr, jr, xr, -m.r2(msfux, jr, xr) * xkxavg * (
        rdx * (m.r(var, kr, jr, xr) - m.r(var, kr, jr, xr, di=-1))
        - tmpzx * (m.r(h1avg, kr, jr, xr, dk=1) - m.r(h1avg, kr, jr, xr)) * rdzu))
    # y fluxes on v faces j = j_start .. j_end+1
    xkxavg = 0.5 * (m.r(xkhh, kr, yr, ir, dj=-1) + m.r(xkhh, kr, yr, ir)) * 0.5 * (
        m.r(rho, kr, yr, ir, dj=-1) + m.r(rho, kr, yr, ir))
    h2avg = m.zeros()
    h2avg = m.s(h2avg, ka, yr, ir, 0.5 * (m.v1(fnm, ka) * (m.r(var, ka, yr, ir, dj=-1) + m.r(var, ka, yr, ir))
                                          + m.v1(fnp, ka) * (m.r(var, ka, yr, ir, dk=-1, dj=-1) + m.r(var, ka, yr, ir, dk=-1))))
    vy = lambda kk, dj=0: m.r(var, (kk, kk), yr, ir, dj=dj)  # noqa: E731
    h2avg = m.s(h2avg, (kts, kts), yr, ir, 0.5 * (
        cf1 * vy(1) + cf2 * vy(2) + cf3 * vy(3) + cf1 * vy(1, dj=-1) + cf2 * vy(2, dj=-1) + cf3 * vy(3, dj=-1)))
    h2avg = m.s(h2avg, (ktf + 1, ktf + 1), yr, ir, 0.5 * (
        vy(ktes1) + (vy(ktes1) - vy(ktes2)) * 0.5 * dnw1e / dn1e + vy(ktes1, dj=-1)
        + (vy(ktes1, dj=-1) - vy(ktes2, dj=-1)) * 0.5 * dnw1e / dn1e))
    tmpzy = 0.5 * (m.r(zy, kr, yr, ir) + m.r(zy, kr, yr, ir, dk=1))
    rdzv = 2.0 / (1.0 / m.r(rdzw, kr, yr, ir) + 1.0 / m.r(rdzw, kr, yr, ir, dj=-1))
    h2 = m.s(m.zeros(), kr, yr, ir, -m.r2(msfvy, yr, ir) * xkxavg * (
        rdy * (m.r(var, kr, yr, ir) - m.r(var, kr, yr, ir, dj=-1))
        - tmpzy * (m.r(h2avg, kr, yr, ir, dk=1) - m.r(h2avg, kr, yr, ir)) * rdzv))
    # flux averages back to w levels on mass columns (H1avg/H2avg overwritten)
    h1avg = m.s(h1avg, ka, jr, ir, 0.5 * (m.v1(fnm, ka) * (m.r(h1, ka, jr, ir, di=1) + m.r(h1, ka, jr, ir))
                                          + m.v1(fnp, ka) * (m.r(h1, ka, jr, ir, dk=-1, di=1) + m.r(h1, ka, jr, ir, dk=-1))))
    h2avg = m.s(h2avg, ka, jr, ir, 0.5 * (m.v1(fnm, ka) * (m.r(h2, ka, jr, ir, dj=1) + m.r(h2, ka, jr, ir))
                                          + m.v1(fnp, ka) * (m.r(h2, ka, jr, ir, dk=-1, dj=1) + m.r(h2, ka, jr, ir, dk=-1))))
    for kk in (kts, ktf + 1):
        h1avg = m.s(h1avg, (kk, kk), jr, ir, 0.0)
        h2avg = m.s(h2avg, (kk, kk), jr, ir, 0.0)
    zx_at_m = 0.25 * (m.r(zx, kr, jr, ir) + m.r(zx, kr, jr, ir, di=1) + m.r(zx, kr, jr, ir, dk=1) + m.r(zx, kr, jr, ir, dk=1, di=1))
    zy_at_m = 0.25 * (m.r(zy, kr, jr, ir) + m.r(zy, kr, jr, ir, dj=1) + m.r(zy, kr, jr, ir, dk=1) + m.r(zy, kr, jr, ir, dk=1, dj=1))
    mrdx = m.r2(msftx, jr, ir) * rdx
    mrdy = m.r2(msfty, jr, ir) * rdy
    rz = m.r(rdzw, kr, jr, ir)
    inc = _G / (m.v1(dnw, kr) * rz) * (
        mrdx * (m.r(h1, kr, jr, ir, di=1) - m.r(h1, kr, jr, ir))
        + mrdy * (m.r(h2, kr, jr, ir, dj=1) - m.r(h2, kr, jr, ir))
        - m.r2(msftx, jr, ir) * zx_at_m * (m.r(h1avg, kr, jr, ir, dk=1) - m.r(h1avg, kr, jr, ir)) * rz
        - m.r2(msfty, jr, ir) * zy_at_m * (m.r(h2avg, kr, jr, ir, dk=1) - m.r(h2avg, kr, jr, ir)) * rz)
    return _add_doubled(m, tend, kr, jr, ir, inc, doing_tke)


def _add_doubled(m, tend, kr, jr, ir, inc, doing_tke):
    """``tendency += inc``; for TKE WRF then sets ``tmp + 2*(tendency - tmp)``."""
    old = m.r(tend, kr, jr, ir)
    new = old + inc
    if doing_tke:
        new = old + 2.0 * (new - old)
    return m.s(tend, kr, jr, ir, new)


def _vertical_diffusion_s(m, periodic, tend, var, xkhv, rho, fnm, fnp, rdz, dnw, doing_tke=False):
    jr, ir = _mass_ranges(m, periodic)
    kts, ktf = m.kts, m.ktf
    ka, kr = (kts + 1, ktf), (kts, ktf)
    xkxavg = m.v1(fnm, ka) * m.r(xkhv, ka, jr, ir) + m.v1(fnp, ka) * m.r(xkhv, ka, jr, ir, dk=-1)
    xkxavg = xkxavg * (m.v1(fnm, ka) * m.r(rho, ka, jr, ir) + m.v1(fnp, ka) * m.r(rho, ka, jr, ir, dk=-1))
    h3 = m.s(m.zeros(), ka, jr, ir, -xkxavg * (m.r(var, ka, jr, ir) - m.r(var, ka, jr, ir, dk=-1)) * m.r(rdz, ka, jr, ir))
    inc = _G * (m.r(h3, kr, jr, ir, dk=1) - m.r(h3, kr, jr, ir)) / m.v1(dnw, kr)
    return _add_doubled(m, tend, kr, jr, ir, inc, doing_tke)


def _vertical_diffusion_uv(m, periodic, ru, rv, d13, d23, xkmv, rho, fnm, fnp, dnw):
    kts, ktf = m.kts, m.ktf
    ka, k1 = (kts + 1, ktf), (kts, kts)
    # u
    jr, ir = _ranges_spec(m, periodic, i_end_off=1, j_end_off=2, i_px_end_off=0, j_base_end=min(m.jte, m.jde - 1))
    titau3 = _titau_13_31(m, periodic, d13, xkmv, fnm, fnp, rho, (0, 0, 0, 0))
    rdzu = -_G / m.v1(dnw, ka)
    ru = m.s(ru, ka, jr, ir, m.r(ru, ka, jr, ir) - rdzu * (m.r(titau3, ka, jr, ir, dk=1) - m.r(titau3, ka, jr, ir)))
    rdzu = -_G / m.v1(dnw, k1)
    ru = m.s(ru, k1, jr, ir, m.r(ru, k1, jr, ir) - rdzu * m.r(titau3, k1, jr, ir, dk=1))
    # v
    jr, ir = _ranges_spec(m, periodic, i_end_off=2, j_end_off=1, i_px_end_off=1, i_base_end=min(m.ite, m.ide - 1))
    titau3 = _titau_23_32(m, periodic, d23, xkmv, fnm, fnp, rho, (0, 0, 0, 0))
    rdzv = -_G / m.v1(dnw, ka)
    rv = m.s(rv, ka, jr, ir, m.r(rv, ka, jr, ir) - rdzv * (m.r(titau3, ka, jr, ir, dk=1) - m.r(titau3, ka, jr, ir)))
    rdzv = -_G / m.v1(dnw, k1)
    rv = m.s(rv, k1, jr, ir, m.r(rv, k1, jr, ir) - rdzv * m.r(titau3, k1, jr, ir, dk=1))
    return ru, rv


def _vertical_diffusion_w_2(m, periodic, rw, d33, xkmh, rho, dn):
    jr, ir = _mass_ranges(m, periodic)
    titau3 = _titau_11_22_33(m, periodic, xkmh, d33, rho, (0, 0, 0, 0))
    ka = (m.kts + 1, m.ktf)
    return m.s(rw, ka, jr, ir, m.r(rw, ka, jr, ir) + _G * (m.r(titau3, ka, jr, ir) - m.r(titau3, ka, jr, ir, dk=-1)) / m.v1(dn, ka))


# --------------------------------------------------------------------------- #
# km_opt=2: tke_km / calc_l_scale / tke_rhs (tke_shear, tke_buoyancy, tke_dissip) #
# --------------------------------------------------------------------------- #
_TKE_SEED_VALUE = 1.0e-06


def _tke_seed(cfg: "Les3dConfig") -> float:
    """``tke_km``'s tke_seed (diff_opt=2): non-zero only for isfflx=0 without surface forcing."""
    if int(cfg.isfflx) == 0:
        if cfg.vertical:  # diff_opt == 2 .and. bl_pbl_physics == 0
            if cfg.tke_drag_coefficient < 1.0e-15 and cfg.tke_heat_flux < 1.0e-15:
                return _TKE_SEED_VALUE
            return 0.0
        return _TKE_SEED_VALUE
    return 0.0


def _calc_l_scale(m, periodic, tke, bn2, rdzw, msftx, msfty, dx, dy):
    jr, ir = _mass_ranges(m, periodic)
    kr = (m.kts, m.ktf)
    deltas = jnp.power(dx / m.r2(msftx, jr, ir) * dy / m.r2(msfty, jr, ir) / m.r(rdzw, kr, jr, ir), np.float32(0.33333333))
    b = m.r(bn2, kr, jr, ir)
    tmp = jnp.sqrt(jnp.maximum(m.r(tke, kr, jr, ir), 1.0e-6))
    stable = jnp.maximum(jnp.minimum(0.76 * tmp / jnp.sqrt(jnp.where(b > 1.0e-6, b, 1.0)), deltas), 0.001 * deltas)
    return jnp.where(b > 1.0e-6, stable, deltas), deltas


def _tke_km(m, periodic, cfg, tke, theta, p8w, t8w, bn2, rdz, rdzw, msftx, msfty, dx, dy, dt, c_k, mub):
    kts, ktf, kde = m.kts, m.ktf, m.kde
    jr, ir = _mass_ranges(m, periodic)
    seed = np.float32(_tke_seed(cfg))
    dt = _k(dt, tke)
    ki = (kts + 1, ktf - 1)
    dthrdn = m.zeros()
    tmpdz = 1.0 / m.r(rdz, ki, jr, ir, dk=1) + 1.0 / m.r(rdz, ki, jr, ir)
    dthrdn = m.s(dthrdn, ki, jr, ir, (m.r(theta, ki, jr, ir, dk=1) - m.r(theta, ki, jr, ir, dk=-1)) / tmpdz)
    k1 = (kts, kts)
    tmpdz = 1.0 / m.r(rdzw, k1, jr, ir, dk=1) + 1.0 / m.r(rdzw, k1, jr, ir)
    thetasfc = m.r(t8w, k1, jr, ir) / jnp.power(m.r(p8w, k1, jr, ir) / _k(_P1000MB, p8w), _RCP)
    dthrdn = m.s(dthrdn, k1, jr, ir, (m.r(theta, k1, jr, ir, dk=1) - thetasfc) / tmpdz)
    kt = (ktf, ktf)
    tmpdz = 1.0 / m.r(rdz, kt, jr, ir) + 0.5 / m.r(rdzw, kt, jr, ir)
    thetatop = m.r(t8w, (kde, kde), jr, ir) / jnp.power(m.r(p8w, (kde, kde), jr, ir) / _k(_P1000MB, p8w), _RCP)
    dthrdn = m.s(dthrdn, kt, jr, ir, (thetatop - m.r(theta, kt, jr, ir, dk=-1)) / tmpdz)
    kr = (kts, ktf)
    mtx, mty, rz = m.r2(msftx, jr, ir), m.r2(msfty, jr, ir), m.r(rdzw, kr, jr, ir)
    tmp = jnp.sqrt(jnp.maximum(m.r(tke, kr, jr, ir), seed))
    if int(cfg.isotropic) == 0:
        mlen_h = jnp.sqrt(dx / mtx * dy / mty)
        deltas = 1.0 / rz
        dd = m.r(dthrdn, kr, jr, ir)
        th = m.r(theta, kr, jr, ir)
        mlen_s = 0.76 * tmp / jnp.sqrt(jnp.abs(_G / th * dd))
        mlen_v = jnp.where(dd > 0.0, jnp.minimum(deltas, mlen_s), deltas)
        xkmh = jnp.maximum(c_k * tmp * mlen_h, 1.0e-6 * mlen_h * mlen_h)
        xkmh = jnp.minimum(xkmh, mub * mlen_h * mlen_h / dt)
        xkmv = jnp.maximum(c_k * tmp * mlen_v, 1.0e-6 * deltas * deltas)
        xkmv = jnp.minimum(xkmv, mub * deltas * deltas / dt)
        pr_inv_h = np.float32(np.float32(1.0) / _PRANDTL)
        pr_inv_v = 1.0 + 2.0 * mlen_v / deltas
        xkhh = xkmh * pr_inv_h
        xkhv = xkmv * pr_inv_v
    else:
        l_scale, _ = _calc_l_scale(m, periodic, tke, bn2, rdzw, msftx, msfty, dx, dy)
        deltas = jnp.power(dx / mtx * dy / mty / rz, np.float32(0.33333333))
        xkmh = jnp.minimum(mub * dx / mtx * dy / mty / dt, c_k * tmp * l_scale)
        xkmv = jnp.minimum(mub / rz / rz / dt, c_k * tmp * l_scale)
        pr_inv = 1.0 + 2.0 * l_scale / deltas
        xkhh = jnp.minimum(mub * dx / mtx * dy / mty / dt, xkmh * pr_inv)
        xkhv = jnp.minimum(mub / rz / rz / dt, xkmv * pr_inv)
    return tuple(m.s(m.zeros(), kr, jr, ir, val.astype(m.dtype)) for val in (xkmh, xkmv, xkhh, xkhv))


def _tke_rhs(m, periodic, cfg, d11, d22, d33, d12, d13, d23, u, v, tke, mut, c1, c2, theta, bn2, msftx, msfty,
             xkmh, xkmv, xkhv, dx, dy, dt, rdzw, ust, hfx, qv, rho, c_k):
    kts, ktf = m.kts, m.ktf
    jr, ir = _mass_ranges(m, periodic)
    kr, ka, k1 = (kts, ktf), (kts + 1, ktf), (kts, kts)
    mu = m.r2(mut, jr, ir)
    mass = lambda kk: m.v1(c1, kk) * mu + m.v1(c2, kk)  # noqa: E731  (c1(k)*mu(i,j)+c2(k))
    tend = m.zeros()
    r = lambda a, kk=kr, **d: m.r(a, kk, jr, ir, **d)  # noqa: E731
    # ---- tke_shear ----
    acc = r(tend)
    acc = acc + 0.5 * mass(kr) * r(xkmh) * (r(d11) * r(d11))
    acc = acc + 0.5 * mass(kr) * r(xkmh) * (r(d22) * r(d22))
    acc = acc + 0.5 * mass(kr) * r(xkmv) * (r(d33) * r(d33))
    sq = lambda a: a * a  # noqa: E731
    avg = 0.25 * (sq(r(d12)) + sq(r(d12, dj=1)) + sq(r(d12, di=1)) + sq(r(d12, di=1, dj=1)))
    acc = acc + mass(kr) * r(xkmh) * avg
    xr = (ir[0], ir[1] + 1)
    tmp2 = m.s(m.zeros(), ka, jr, xr, m.r(d13, ka, jr, xr))
    avg = 0.25 * (sq(r(tmp2, dk=1)) + sq(r(tmp2)) + sq(r(tmp2, dk=1, di=1)) + sq(r(tmp2, di=1)))
    acc = acc + mass(kr) * r(xkmv) * avg
    tend = m.s(tend, kr, jr, ir, acc)
    uu = r(u, k1) + r(u, k1, di=1)
    vv = r(v, k1) + r(v, k1, dj=1)
    if int(cfg.isfflx) == 0:
        absu = 0.5 * jnp.sqrt(uu * uu + vv * vv)
        cd = np.float32(cfg.tke_drag_coefficient)
    else:
        absu = 0.5 * jnp.sqrt(uu * uu + vv * vv) + _EPSILON
        us = m.r2(ust, jr, ir)
        cd = (us * us) / (absu * absu)
    tend = m.s(tend, k1, jr, ir, r(tend, k1) + mass(k1) * (
        uu * 0.5 * cd * absu * (r(d13, k1, dk=1) + r(d13, k1, dk=1, di=1)) * 0.5))
    yr = (jr[0], jr[1] + 1)
    tmp2 = m.s(m.zeros(), ka, yr, ir, m.r(d23, ka, yr, ir))
    avg = 0.25 * (sq(r(tmp2, dk=1)) + sq(r(tmp2)) + sq(r(tmp2, dk=1, dj=1)) + sq(r(tmp2, dj=1)))
    tend = m.s(tend, kr, jr, ir, r(tend) + mass(kr) * r(xkmv) * avg)
    tend = m.s(tend, k1, jr, ir, r(tend, k1) + mass(k1) * (
        vv * 0.5 * cd * absu * (r(d23, k1, dk=1) + r(d23, k1, dk=1, dj=1)) * 0.5))
    # ---- tke_buoyancy ----
    tend = m.s(tend, ka, jr, ir, r(tend, ka) - mass(ka) * r(xkhv, ka) * r(bn2, ka))
    if int(cfg.isfflx) in (0, 2):
        heat_flux = np.float32(cfg.tke_heat_flux)
    else:
        cpm = _CP * (1.0 + 0.8 * r(qv, k1))
        heat_flux = (m.r2(hfx, jr, ir) / cpm) / r(rho, k1)
    tend = m.s(tend, k1, jr, ir, r(tend, k1) - mass(k1) * (
        (r(xkhv, k1) * r(bn2, k1)) - (_G / r(theta, k1)) * heat_flux) / 2.0)
    # ---- tke_dissip ----  (c_k is a namelist REAL: gfortran evaluates ce1/ce2 in REAL4 at runtime)
    ce1 = np.float32(np.float32(np.float32(cfg.c_k) / np.float32(0.10)) * np.float32(0.19))
    ce2 = np.float32(max(np.float32(0.0), np.float32(np.float32(0.93) - ce1)))
    l_scale, deltas = _calc_l_scale(m, periodic, tke, bn2, rdzw, msftx, msfty, dx, dy)
    tketmp = jnp.maximum(r(tke), 1.0e-6)
    kidx = jnp.arange(kr[0], kr[1] + 1)[:, None, None]
    coefc = jnp.where((kidx == kts) | (kidx == ktf), np.float32(3.9), ce1 + ce2 * l_scale / deltas)
    tend = m.s(tend, kr, jr, ir, r(tend) - mass(kr) * coefc * jnp.power(tketmp, np.float32(1.5)) / l_scale)
    # ---- lower limit ----
    tend = m.s(tend, kr, jr, ir, jnp.maximum(r(tend), -mass(kr) * jnp.maximum(0.0, r(tke)) / _k(dt, tke)))
    return tend


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def les3d_smagorinsky_memory(inp: Les3dInputs, cfg: Les3dConfig) -> Les3dResult:
    """Run the WRF diff_opt=2/km_opt=3 sequence; return halo-padded arrays."""

    dtype = jnp.asarray(inp.u).dtype
    nz, ny, nx1 = inp.u.shape
    nx = nx1 - 1
    m = _Mem(nx, ny, nz, dtype)
    periodic = cfg.boundary == "periodic"
    f32 = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
    cf1, cf2, cf3, rdx, rdy = (f32(x) for x in (inp.cf1, inp.cf2, inp.cf3, inp.rdx, inp.rdy))
    dx, dy, dt = (f32(x) for x in (inp.dx, inp.dy, inp.dt))
    c_s, mub = f32(cfg.c_s), f32(cfg.mix_upper_bound)
    fnm, fnp, dn, dnw = m.col(inp.fnm), m.col(inp.fnp), m.col(inp.dn), m.col(inp.dnw)
    u_base, v_base, t_base, qv_base = m.col(inp.u_base), m.col(inp.v_base), m.col(inp.t_base), m.col(inp.qv_base)

    bc = lambda a, var: _bc3(m, a, var, periodic)  # noqa: E731
    u = bc(m.put(inp.u, istag=1), "u")
    v = bc(m.put(inp.v, jstag=1), "v")
    w = bc(m.put(inp.w), "w")
    ph = bc(m.put(inp.ph), "w")
    phb = bc(m.put(inp.phb), "w")
    thp = bc(m.put(inp.thp), "t")
    th_phy = bc(m.put(inp.th_phy), "t")
    t_phy = bc(m.put(inp.t_phy), "t")
    p_phy = bc(m.put(inp.p_phy), "t")
    p8w = bc(m.put(inp.p8w), "w")
    t8w = bc(m.put(inp.t8w), "w")
    rho = bc(m.put(inp.rho), "t")
    moist = tuple(bc(m.put(q), "t") for q in inp.moist)
    bc2 = lambda a, var: _bc2(m, a, var, periodic)  # noqa: E731
    msftx, msfty = bc2(m.put2(inp.msftx), "t"), bc2(m.put2(inp.msfty), "t")
    msfux, msfuy = bc2(m.put2(inp.msfux, istag=1), "u"), bc2(m.put2(inp.msfuy, istag=1), "u")
    msfvx, msfvy = bc2(m.put2(inp.msfvx, jstag=1), "v"), bc2(m.put2(inp.msfvy, jstag=1), "v")

    z, rdz, rdzw, zx, zy = _compute_diff_metrics(m, ph, phb, rdx, rdy, periodic)
    rdzw, rdz, z = bc(rdzw, "w"), bc(rdz, "w"), bc(z, "w")
    zx, zy = bc(zx, "e"), bc(zy, "f")
    div, d11, d22, d33, d12, d13, d23 = _cal_deform_and_div(
        m, periodic, u, v, w, rdz, rdzw, zx, zy, msfux, msfuy, msfvx, msfvy, msftx, msfty,
        fnm, fnp, dn, dnw, cf1, cf2, cf3, rdx, rdy, u_base, v_base, cfg.mix_full_fields)
    bn2 = _calculate_n2(m, periodic, moist, cfg.iqv, cfg.iqc, cfg.iqi, th_phy, t_phy, p_phy, p8w, t8w,
                        dnw, dn, rdz, rdzw, cf1, cf2, cf3)
    tke = None
    if int(cfg.km_opt) == 2:
        tke = bc(m.put(inp.tke), "t")
        xkmh, xkmv, xkhh, xkhv = _tke_km(m, periodic, cfg, tke, th_phy, p8w, t8w, bn2, rdz, rdzw, msftx, msfty,
                                         dx, dy, dt, f32(cfg.c_k), mub)
    else:
        xkmh, xkmv, xkhh, xkhv = _smag_km(m, periodic, bn2, d11, d22, d33, d12, d13, d23, rdzw, msftx, msfty,
                                          dx, dy, dt, c_s, mub, cfg.isotropic)
    # phy_bc
    xkmh, xkhh, xkmv, xkhv = bc(xkmh, "t"), bc(xkhh, "t"), bc(xkmv, "t"), bc(xkhv, "t")
    div, d11, d22, d33 = bc(div, "t"), bc(d11, "t"), bc(d22, "t"), bc(d33, "t")
    d12, d13, d23 = bc(d12, "d"), bc(d13, "e"), bc(d23, "f")
    rho = bc(rho, "t")

    ru, rv, rw, rt = m.zeros(), m.zeros(), m.zeros(), m.zeros()
    mt = [m.zeros() for _ in moist]
    tke_t = None
    if tke is not None:  # first_rk_step_part2: tke_rhs before vertical/horizontal diffusion
        zeros2 = jnp.zeros(m.shape[1:], dtype=dtype)
        ust2 = bc2(m.put2(inp.ust), "t") if inp.ust is not None else zeros2
        hfx2 = m.put2(inp.hfx) if inp.hfx is not None else zeros2
        tke_t = _tke_rhs(m, periodic, cfg, d11, d22, d33, d12, d13, d23, u, v, tke, bc2(m.put2(inp.mut), "t"),
                         m.col(inp.c1h), m.col(inp.c2h), th_phy, bn2, msftx, msfty, xkmh, xkmv, xkhv, dx, dy, dt,
                         rdzw, ust2, hfx2, moist[cfg.iqv], rho, f32(cfg.c_k))
    if cfg.vertical:
        ru, rv = _vertical_diffusion_uv(m, periodic, ru, rv, d13, d23, xkmv, rho, fnm, fnp, dnw)
        rw = _vertical_diffusion_w_2(m, periodic, rw, d33, xkmh, rho, dn)
        ru, rv, rt, mt = _vertical_surface_and_scalars(m, cfg, periodic, ru, rv, rt, mt, u, v, thp, th_phy, moist,
                                                        cfg.iqv, xkhv, rho, fnm, fnp, rdz, dnw, t_base, qv_base,
                                                        inp, dtype)
        if tke_t is not None:
            tke_t = _vertical_diffusion_s(m, periodic, tke_t, tke, xkmv, rho, fnm, fnp, rdz, dnw, doing_tke=True)
    ru = _horizontal_diffusion_u_2(m, periodic, ru, d11, d12, xkmh, rho, msfux, msfuy, rdx, rdy, fnm, fnp, dnw, zx, zy, rdzw)
    rv = _horizontal_diffusion_v_2(m, periodic, rv, d12, d22, xkmh, rho, msfvx, msfvy, rdx, rdy, fnm, fnp, dnw, zx, zy, rdzw)
    rw = _horizontal_diffusion_w_2(m, periodic, rw, d13, d23, xkmv, rho, msftx, msfty, rdx, rdy, fnm, fnp, dn, zx, zy, rdz)
    rt = _horizontal_diffusion_s(m, periodic, rt, thp, xkhh, rho, msftx, msfty, msfux, msfvy, rdx, rdy, fnm, fnp,
                                 cf1, cf2, cf3, zx, zy, rdzw, dnw, dn)
    if tke_t is not None and not cfg.tke_mix2_off:
        tke_t = _horizontal_diffusion_s(m, periodic, tke_t, tke, xkmh, rho, msftx, msfty, msfux, msfvy, rdx, rdy,
                                        fnm, fnp, cf1, cf2, cf3, zx, zy, rdzw, dnw, dn, doing_tke=True)
    if not cfg.moist_mix2_off:
        mt = [_horizontal_diffusion_s(m, periodic, mt[n], q, xkhh, rho, msftx, msfty, msfux, msfvy, rdx, rdy, fnm,
                                      fnp, cf1, cf2, cf3, zx, zy, rdzw, dnw, dn) for n, q in enumerate(moist)]
    return Les3dResult(z, rdz, rdzw, zx, zy, div, d11, d22, d33, d12, d13, d23, bn2, xkmh, xkmv, xkhh, xkhv,
                       ru, rv, rw, rt, tuple(mt), tke_t)


def _vertical_surface_and_scalars(m, cfg, periodic, ru, rv, rt, mt, u, v, thp, th_phy, moist, iqv, xkhv, rho,
                                  fnm, fnp, rdz, dnw, t_base, qv_base, inp, dtype):
    """vertical_diffusion_2 surface stress, theta/moist vertical diffusion and fluxes."""

    kts, kte = m.kts, m.kte
    k1 = (kts, kts)
    j_s, j_e = m.jts, min(m.jte, m.jde - 1)
    i_s, i_e = m.its, min(m.ite, m.ide - 1)
    dnw1 = m.v1(dnw, k1)
    zeros2 = jnp.zeros(m.shape[1:], dtype=dtype)
    ust = _bc2(m, m.put2(inp.ust), "t", periodic) if inp.ust is not None else zeros2
    hfx = m.put2(inp.hfx) if inp.hfx is not None else zeros2
    qfx = m.put2(inp.qfx) if inp.qfx is not None else zeros2
    # u surface stress: j = j_start..j_end, i = i_start..ite
    jr, ir = (j_s, j_e), (i_s, m.ite)
    vavg = (m.r(v, k1, jr, ir) + m.r(v, k1, jr, ir, dj=1) + m.r(v, k1, jr, ir, di=-1) + m.r(v, k1, jr, ir, di=-1, dj=1)) / 4.0
    uk = m.r(u, k1, jr, ir)
    v0_u = jnp.sqrt(uk * uk + vavg * vavg) + _EPSILON
    if cfg.isfflx == 0:
        tao_xz = np.float32(cfg.tke_drag_coefficient) * v0_u * uk
    else:
        ustar = 0.5 * (m.r2(ust, jr, ir) + m.r2(ust, jr, ir, di=-1))
        tao_xz = ustar * ustar * uk / v0_u
    ru = m.s(ru, k1, jr, ir, m.r(ru, k1, jr, ir) + _G * tao_xz * 0.5 * (m.r(rho, k1, jr, ir) + m.r(rho, k1, jr, ir, di=-1)) / dnw1)
    # v surface stress: j = j_start..jte, i = i_start..i_end
    jr, ir = (j_s, m.jte), (i_s, i_e)
    uavg = (m.r(u, k1, jr, ir) + m.r(u, k1, jr, ir, dj=-1) + m.r(u, k1, jr, ir, di=1) + m.r(u, k1, jr, ir, di=1, dj=-1)) / 4.0
    vk = m.r(v, k1, jr, ir)
    v0_v = jnp.sqrt(vk * vk + uavg * uavg) + _EPSILON
    if cfg.isfflx == 0:
        tao_yz = np.float32(cfg.tke_drag_coefficient) * v0_v * vk
    else:
        ustar = 0.5 * (m.r2(ust, jr, ir) + m.r2(ust, jr, ir, dj=-1))
        tao_yz = ustar * ustar * vk / v0_v
    rv = m.s(rv, k1, jr, ir, m.r(rv, k1, jr, ir) + _G * tao_yz * 0.5 * (m.r(rho, k1, jr, ir) + m.r(rho, k1, jr, ir, dj=-1)) / dnw1)
    # theta
    jr, ir = (j_s, j_e), (i_s, i_e)
    kmix = (kts, kte - 1)
    var_mix = m.zeros()
    if cfg.mix_full_fields:
        var_mix = m.s(var_mix, kmix, jr, ir, m.r(thp, kmix, jr, ir))
    else:
        var_mix = m.s(var_mix, kmix, jr, ir, m.r(thp, kmix, jr, ir) - m.v1(t_base, kmix))
    rt = _vertical_diffusion_s(m, periodic, rt, var_mix, xkhv, rho, fnm, fnp, rdz, dnw)
    qv1 = m.r(moist[iqv], k1, jr, ir)
    rho1 = m.r(rho, k1, jr, ir)
    th1 = m.r(th_phy, k1, jr, ir)
    if cfg.isfflx in (0, 2):
        heat_flux = np.float32(cfg.tke_heat_flux)
        if int(cfg.use_theta_m) == 1:
            inc = (m.r(rt, k1, jr, ir) - _G * heat_flux * (1.0 + _RVOVRD * qv1) * rho1 / dnw1
                   - _G_1P61 * th1 * m.r2(qfx, jr, ir) / dnw1)
        else:
            inc = m.r(rt, k1, jr, ir) - _G * heat_flux * rho1 / dnw1
    else:
        cpm = _CP * (1.0 + 0.8 * qv1)
        heat_flux = m.r2(hfx, jr, ir) / cpm
        if int(cfg.use_theta_m) == 1:
            inc = (m.r(rt, k1, jr, ir) - _G * heat_flux * (1.0 + _RVOVRD * qv1) / dnw1
                   - _G_1P61 * th1 * m.r2(qfx, jr, ir) / dnw1)
        else:
            inc = m.r(rt, k1, jr, ir) - _G * heat_flux / dnw1
    rt = m.s(rt, k1, jr, ir, inc)
    out = []
    for n, q in enumerate(moist):
        var_mix = m.zeros()
        if (not cfg.mix_full_fields) and n == iqv:
            var_mix = m.s(var_mix, kmix, jr, ir, m.r(q, kmix, jr, ir) - m.v1(qv_base, kmix))
        else:
            var_mix = m.s(var_mix, kmix, jr, ir, m.r(q, kmix, jr, ir))
        tq = _vertical_diffusion_s(m, periodic, mt[n], var_mix, xkhv, rho, fnm, fnp, rdz, dnw)
        if cfg.isfflx in (1, 2) and n == iqv:
            tq = m.s(tq, k1, jr, ir, m.r(tq, k1, jr, ir) - _G * m.r2(qfx, jr, ir) / dnw1)
        out.append(tq)
    return ru, rv, rt, out


def les3d_smagorinsky_tendencies(inp: Les3dInputs, cfg: Les3dConfig) -> Les3dTendencies:
    """Interior ``ru/rv/rw/t/moist_tendf`` increments + mass-point K fields."""

    res = les3d_smagorinsky_memory(inp, cfg)
    nz, ny, nx1 = inp.u.shape
    m = _Mem(nx1 - 1, ny, nz, jnp.asarray(inp.u).dtype)
    return Les3dTendencies(
        ru_tendf=m.interior(res.ru_tendf, istag=1),
        rv_tendf=m.interior(res.rv_tendf, jstag=1),
        rw_tendf=m.interior(res.rw_tendf, nk=nz + 1),
        t_tendf=m.interior(res.t_tendf),
        moist_tendf=tuple(m.interior(q) for q in res.moist_tendf),
        xkmh=m.interior(res.xkmh),
        xkmv=m.interior(res.xkmv),
        xkhh=m.interior(res.xkhh),
        xkhv=m.interior(res.xkhv),
        tke_tendf=None if res.tke_tendf is None else m.interior(res.tke_tendf),
    )


__all__ = [
    "HALO",
    "Les3dConfig",
    "Les3dInputs",
    "Les3dResult",
    "Les3dTendencies",
    "les3d_smagorinsky_memory",
    "les3d_smagorinsky_tendencies",
]
