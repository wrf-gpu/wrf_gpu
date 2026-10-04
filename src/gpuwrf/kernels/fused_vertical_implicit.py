"""Fused Pallas kernels for the implicit vertical w/phi solve (M2 bake-off).

Sprint ``.agent/sprints/2026-09-17-v0250-m2-fused-bakeoff``: ONE fused module
for the #1 launch family of the operational timestep (M2 census
``<DATA_ROOT>/wrf_gpu2/v025/m2/census/m2_launch_census.json``):
``dycore.vertical_implicit``.  Per timestep (RK3, ``acoustic_substeps=10``)
the current XLA implementation launches

* ``calc_coef_w_wrf_coefficients`` 3x (once per RK stage), 133 kernels each;
* ``advance_w_wrf`` 16x (once per acoustic substep), 377 kernel launches each
  (29 outside the two ``lax.scan`` Thomas sweeps + 2 whiles x 44/43 trips
  x 3-4 launches per trip).

This module replaces the two operators with one Pallas kernel each — both are
column-local recurrences (their ``k``-dependencies never cross columns), so a
single launch per operator resolves the whole family: ~2 kernels/substep
instead of ~377.

Semantics contract (frozen here, gated in ``tests/v025/test_m2_fused_*.py``)
===========================================================================

* fp64 arithmetic with the REFERENCE OPERATION ORDER of
  ``gpuwrf.dynamics.acoustic_wrf.calc_coef_w_wrf_coefficients`` and
  ``gpuwrf.dynamics.core.advance_w_wrf`` (worktree ``4bae18f19``).  Every
  scalar is combined in the same association order as the Python source.
* **Scalar constants are RUNTIME arguments** (the packed ``S`` f64 vector),
  never Python-float closure constants.  Measured on this tree: with ``cof``
  closed as a Python constant, the kernel's XLA fusion context reassociates
  ``(cqw*cof)*rdn`` into ``cqw*(cof*rdn)`` — a 1-ulp divergence from the
  reference, which compiles the same source expression WITHOUT the
  reassociation (the v0.15 "XLA FMA/contraction is fusion-context-dependent"
  lesson, now reproduced for constant folding).  Runtime scalars cannot be
  constant-folded, so the written association survives compilation and the
  interpreter output is BITWISE equal to the reference.  Runtime f64 scalar
  arithmetic is IEEE-identical to the reference's Python-float arithmetic
  (same single roundings in the same order).
* Reorder-free claim: no reassociation, no algebraic simplification, no
  hoisting beyond value-identical reuse of a repeated subexpression (e.g.
  ``ph_1 + phb``, ``c1f*mut + c2f``), documented inline at each reuse.  The
  JAX reference interpreter (``interpret=True``, CPU) is BITWISE equal to the
  XLA reference on fp64 inputs — that is the CPU gate.  On the device the
  same op order goes through the Triton compiler, where no such guarantee
  exists; the pre-registered device tolerance is the value-scaled envelope
  defined in the tests (``1e-12 * max(1, max|ref|)`` per output, ~9 orders
  above ulp noise, ~9 below any real reorder bug), with the bitwise fraction
  reported alongside.
* Environment gate: production default ``GPUWRF_ADVANCE_W_SAFE_FLOORS``
  unset/0 (divide directly, WRF-faithful).  The fused kernels implement the
  default path ONLY and refuse when the debug guard is on.
* ``advance_w`` configuration coverage: ``w_damping in (0, 1)``,
  ``damp_opt`` any (values other than 3 skip the Rayleigh block exactly like
  the reference), ``top_lid`` bool, flat or general terrain, moist or dry
  ``cqw``/``c2a`` — i.e. the full production substep call
  (``acoustic.py:1258-1333``).  Unsupported configurations raise.
* No host transfers: pure device function, jit-compatible, no
  ``block_until_ready``/``device_get``/numpy materialisation anywhere.

Launch geometry (aligned windows, no masks, no races)
=====================================================
The packed column axis is PADDED with ones up to a multiple of ``TX`` and the
grid covers the padded width, so every program's window starts at the ALIGNED
``start = pid*TX``.  An earlier revision used a clamped last window
(``start = min(pid*TX, NCOL-TX)`` with duplicated overlap columns) on the
theory that byte-identical duplicated stores are benign — FALSIFIED on device
(bake-off 2026-09-18): the misaligned clamped window computed WRONG values for
exactly ``start % TX`` low window positions (48 of 8400 columns, 0.5% of
cells, up to 2.2 max-abs), and the duplicate stores then raced
nondeterministically.  With padding, windows never overlap and the kernel is
bitwise-deterministic.  Cost: one finite pad copy per packed input (pad
columns compute garbage that the wrapper slices away; ones keep every
in-kernel divisor finite).  ``NCOL >= TX`` is still required; windows are
always a power of two (the Triton lowering's ``_check_tensor_size``); the only
gathers remain the four clamped terrain-neighbour loads, whose wrapped values
are discarded by ``where`` before use.  DEVICE NOTE: this jax build defaults
``jax_pallas_use_mosaic_gpu=True`` with no mosaic backend installed — device
callers must export ``JAX_PALLAS_USE_MOSAIC_GPU=false`` so the Triton lowering
is selected (the bake-off harness sets it).
Integration seam (NOT applied here — manager applies at adoption)
=================================================================
``src/gpuwrf/runtime/operational_mode.py``, ``_acoustic_scan`` (~line 2902):
``a, alpha, gamma = calc_coef_w_wrf_coefficients(...)`` becomes
``a, alpha, gamma = calc_coef_w_pallas(...)`` (same three face-length
outputs; stage-constant).  ``src/gpuwrf/dynamics/core/acoustic.py:1258``:
``w_solved, ph_next, t_2ave_next = advance_w_wrf(...)`` becomes
``advance_w_pallas(...)`` (same three outputs, per substep).  A mechanical
diff is in the sprint ``WORKER_REPORT.md``.
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

__all__ = [
    "TX",
    "calc_coef_w_pallas",
    "advance_w_pallas",
    "pack_columns",
    "unpack_columns",
]

#: Column-window width (power of two; the Triton lowering requires it).
TX = 128

GRAVITY_M_S2 = 9.81  # matches gpuwrf.dynamics.core.advance_w

# --------------------------------------------------------------------------- #
# runtime scalar stack (see module docstring: constants must be runtime values) #
# --------------------------------------------------------------------------- #
_SLOTS = (
    "cof", "lid_flag", "dts", "epssm", "t0", "g", "rdx", "rdy",
    "cf1", "cf2", "cf3", "w_alpha", "w_crit_cfl", "w_damp_on",
    "dampmag", "hdepth", "half_pi",
)
_COF, _LID, _DTS, _EPSSM, _T0, _G, _RDX, _RDY = 0, 1, 2, 3, 4, 5, 6, 7
_CF1, _CF2, _CF3, _W_ALPHA, _W_CRIT, _W_DAMP_ON = 8, 9, 10, 11, 12, 13
_DAMPMAG, _HDEPTH, _HALF_PI = 14, 15, 16
_N_SLOTS = len(_SLOTS)


def _scalar_stack(**values) -> jax.Array:
    """Pack named scalars into one f64 vector, missing slots zero-filled."""

    data = [0.0] * _N_SLOTS
    for name, value in values.items():
        data[_SLOTS.index(name)] = float(value)
    return jnp.asarray(data, jnp.float64)


def _reject_safe_floors() -> None:
    if os.environ.get("GPUWRF_ADVANCE_W_SAFE_FLOORS", "0") == "1":
        raise RuntimeError(
            "fused_vertical_implicit implements the production default "
            "(GPUWRF_ADVANCE_W_SAFE_FLOORS unset/0: divide directly, "
            "WRF-faithful). The debug floor guard changes per-substep "
            "arithmetic and is not implemented here - call the XLA reference."
        )


# --------------------------------------------------------------------------- #
# column pack / unpack                                                         #
# --------------------------------------------------------------------------- #
def pack_columns(field: jax.Array) -> jax.Array:
    """``(nz, ny, nx) -> (nz, ny*nx)``: planes on axis 0, columns contiguous."""

    nz = int(field.shape[0])
    return jnp.asarray(field).reshape(nz, -1)


def unpack_columns(cols: jax.Array, shape: tuple[int, int, int]) -> jax.Array:
    """Inverse of :func:`pack_columns`."""

    nz, ny, nx = shape
    return cols.reshape(nz, ny, nx)


def _pad_width(ncol: int) -> int:
    """Column count padded up to a multiple of TX (0 when already aligned)."""

    return (-ncol) % TX


def _pad_cols(x: jax.Array, pad: int) -> jax.Array:
    """Right-pad a (planes, ncol) or (ncol,) array with ONES (finite divisors)."""

    if pad == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[-1] = pad
    return jnp.concatenate([x, jnp.ones(pad_shape, dtype=x.dtype)], axis=-1)


# --------------------------------------------------------------------------- #
# kernel A: calc_coef_w — the whole tridiagonal coefficient build, one launch   #
# --------------------------------------------------------------------------- #
def _calc_coef_w_kernel(
    mut_ref, cqw_ref, c2a_ref, c1h_ref, c2h_ref, c1f_ref, c2f_ref,
    rdn_ref, rdnw_ref, s_ref, a_out, alpha_out, gamma_out,
    *, nz: int,
):
    """WRF ``calc_coef_w`` (``module_small_step_em.F:624-649``), column windows.

    Reference op order (``acoustic_wrf.py:808-852``).  The row-0 ``b`` the
    reference computes is dropped identically here: ``gamma[0]=0`` erases it
    from every downstream value and the reference never stores it.
    ``mass_h``/``mass_f`` are recomputed per use from ``c1h*mut + c2h`` etc —
    value-identical to the reference's materialised planes (deterministic
    fp64 arithmetic on the same operands).
    """

    pid = pl.program_id(0)
    start = pid * TX  # NCOL is padded to a TX multiple: every window aligned
    cols = pl.ds(start, TX)
    zeros = jnp.zeros((TX,), dtype=mut_ref.dtype)
    ones = jnp.ones((TX,), dtype=mut_ref.dtype)

    cof = s_ref[_COF]
    lid_flag = s_ref[_LID]
    mut = mut_ref[cols]

    def muh(k):
        return c1h_ref[k] * mut + c2h_ref[k]

    def muf(k):
        return c1f_ref[k] * mut + c2f_ref[k]

    # --- a rows (reference lines 815-826, 830-833) ------------------------- #
    # a[0] = 0; a[1] = 0 (pinned, reference line 825); interior rows 2..nz-1
    # (line 833, cqw[kk] already inside the formula); top row nz (line 826,
    # set AFTER the interior rows, so for nz==1 it overwrites the pinned a[1]
    # exactly like the reference's sequential .at calls do).
    a_out[0, cols] = zeros
    if nz >= 2:
        a_out[1, cols] = zeros
    for kk in range(2, nz):
        # reference line 833: -cqw[kk]*cof*rdn[kk]*rdnw[kk-1]*c2a[kk-1]/denom
        a_out[kk, cols] = (
            (((((-cqw_ref[kk, cols]) * cof) * rdn_ref[kk]) * rdnw_ref[kk - 1])
             * c2a_ref[kk - 1, cols]) / (muh(kk - 1) * muf(kk - 1))
        )
    a_out[nz, cols] = (
        ((((-2.0 * cof) * (rdnw_ref[nz - 1] ** 2)) * c2a_ref[nz - 1, cols])
         * lid_flag) / (muh(nz - 1) * muf(nz - 1))
    )

    # --- Thomas coefficient sweep (reference lines 816-817, 827, 836-851) -- #
    alpha_out[0, cols] = ones
    gamma_out[0, cols] = zeros
    for k in range(1, nz):
        denom_upper = muh(k) * muf(k)
        denom_lower = muh(k - 1) * muf(k)
        denom_c = muh(k) * muf(k + 1)
        t1 = (rdnw_ref[k] * c2a_ref[k, cols]) / denom_upper
        t2 = (rdnw_ref[k - 1] * c2a_ref[k - 1, cols]) / denom_lower
        b = 1.0 + ((cqw_ref[k, cols] * cof) * rdn_ref[k]) * (t1 + t2)
        c = (((((-cqw_ref[k, cols]) * cof) * rdn_ref[k]) * rdnw_ref[k])
             * c2a_ref[k, cols]) / denom_c
        alpha_k = 1.0 / (b - (a_out[k, cols] * gamma_out[k - 1, cols]))
        alpha_out[k, cols] = alpha_k
        gamma_out[k, cols] = c * alpha_k

    # top row (reference lines 849-851)
    b_top = 1.0 + (((2.0 * cof) * (rdnw_ref[nz - 1] ** 2))
                   * c2a_ref[nz - 1, cols]) / (muh(nz - 1) * muf(nz))
    alpha_out[nz, cols] = 1.0 / (b_top - (a_out[nz, cols]
                                          * gamma_out[nz - 1, cols]))
    gamma_out[nz, cols] = zeros


def calc_coef_w_pallas(
    mut: jax.Array,
    metrics_c1h: jax.Array, metrics_c2h: jax.Array,
    metrics_c1f: jax.Array, metrics_c2f: jax.Array,
    metrics_rdn: jax.Array, metrics_rdnw: jax.Array,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = False,
    cqw: jax.Array | None = None,
    c2a: jax.Array | None = None,
    gravity: float = GRAVITY_M_S2,
    interpret: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """XLA replacement for ``calc_coef_w_wrf_coefficients`` (same outputs).

    Returns ``(a, alpha, gamma)`` each ``(nz+1, ny, nx)`` fp64, bitwise equal
    to the reference under the interpreter and within the value-scaled
    envelope on device.  ``interpret=True`` (default) runs the JAX reference
    interpreter — correct on any backend including CPU; the device bake-off
    runs ``interpret=False``.
    """

    _reject_safe_floors()
    nz = int(metrics_c1h.shape[0])
    ny, nx = int(mut.shape[0]), int(mut.shape[1])
    ncol = ny * nx
    if ncol < TX:
        raise ValueError(f"need at least {TX} columns, got {ncol}")
    mut = jnp.asarray(mut, jnp.float64)
    if cqw is None:
        cqw = jnp.ones((nz + 1, ny, nx), dtype=jnp.float64)
    if c2a is None:
        c2a = jnp.ones((nz, ny, nx), dtype=jnp.float64)

    # The reference's own Python-float expressions (acoustic_wrf.py:813-814);
    # passed as RUNTIME scalars so XLA cannot reassociate around them.
    cof = (0.5 * float(dt) * float(gravity) * (1.0 + float(epssm))) ** 2
    lid_flag = 0.0 if bool(top_lid) else 1.0
    s = _scalar_stack(cof=cof, lid_flag=lid_flag)

    pad = _pad_width(ncol)
    args = (
        _pad_cols(mut.reshape(-1), pad), _pad_cols(pack_columns(cqw), pad),
        _pad_cols(pack_columns(c2a), pad),
        jnp.asarray(metrics_c1h, jnp.float64), jnp.asarray(metrics_c2h, jnp.float64),
        jnp.asarray(metrics_c1f, jnp.float64), jnp.asarray(metrics_c2f, jnp.float64),
        jnp.asarray(metrics_rdn, jnp.float64), jnp.asarray(metrics_rdnw, jnp.float64),
        s,
    )
    ncol_pad = ncol + pad
    out_shape = jax.ShapeDtypeStruct((nz + 1, ncol_pad), jnp.float64)
    a_c, alpha_c, gamma_c = pl.pallas_call(
        lambda *refs: _calc_coef_w_kernel(*refs, nz=nz),
        grid=(ncol_pad // TX,),
        out_shape=[out_shape] * 3,
        interpret=interpret,
    )(*args)
    shape = (nz + 1, ny, nx)
    return (
        unpack_columns(a_c[:, :ncol], shape),
        unpack_columns(alpha_c[:, :ncol], shape),
        unpack_columns(gamma_c[:, :ncol], shape),
    )


# --------------------------------------------------------------------------- #
# kernel B: advance_w — the full implicit w/phi substep, one launch             #
# --------------------------------------------------------------------------- #
def _advance_w_kernel(
    # face planes (nzf, NCOL)
    w_ref, rw_ref, ww_ref, ph_ref, ph1_ref, phb_ref, pht_ref,
    cqw_ref, a_ref, alpha_ref, gamma_ref, wsave_ref,
    # mass planes (nz, NCOL)
    t2_ref, t2ave_ref, t1_ref, c2a_ref, alt_ref,
    # vertical 1D (nzf then nz)
    c1f_ref, c2f_ref, c1h_ref, c2h_ref, rdnw_ref, rdn_ref, fnm_ref, fnp_ref,
    # column 1D (NCOL,)
    mut_ref, muts_ref, muave_ref, ht_ref, msftx_ref, msfty_ref,
    # staggered wind slices, levels only (3, NCOL)
    ue_ref, uw_ref, vs_ref, vn_ref,
    # runtime scalar stack
    s_ref,
    # outputs: w_solved, ph_next, t_2ave_next + scratch (rhs, wdwn, rw_eff, wtop)
    w_out, ph_out, t2ave_out, rhs_out, wdwn_out, rw_out, wtop_out,
    *, nz: int, nzf: int, nx: int, ny: int,
    top_lid: bool, w_damp: bool, damp3: bool, wrf_fp32: bool = False,
):
    """WRF ``advance_w`` (``module_small_step_em.F:1178-1597``), column windows.

    Block order and per-element op order follow ``advance_w_wrf`` exactly:
    w_damp -> t_2ave -> phi rhs -> phi advection -> rhs finalize -> terrain
    surface w -> explicit w update (interior/top/surface) -> Thomas forward ->
    Thomas back -> damp_opt=3 Rayleigh -> geopotential finish.
    ``_floor_pos`` guards are identity (production default, enforced by the
    wrapper); the two explicit ``jnp.where`` guards that the reference applies
    unconditionally (``w_damp_vertical_cfl`` and ``coef_mass``) are kept.
    Input refs are never written: the damped ``rw_tend`` goes to the ``rw_out``
    scratch (a value-identical copy).  All numeric scalars come from ``s_ref``
    (runtime values — see the module docstring for why that is load-bearing).
    """

    pid = pl.program_id(0)
    start = pid * TX  # NCOL is padded to a TX multiple: every window aligned
    ncol = mut_ref.shape[0]  # padded width; used only for gather clamps
    cols = pl.ds(start, TX)
    ci = start + jnp.arange(TX, dtype=jnp.int32)
    i_idx = ci % nx
    j_idx = ci // nx
    zeros = jnp.zeros((TX,), dtype=w_ref.dtype)
    mut = mut_ref[cols]
    muts = muts_ref[cols]
    muave = muave_ref[cols]
    msfty = msfty_ref[cols]
    msfty_inv = 1.0 / msfty

    dts = s_ref[_DTS]
    g = s_ref[_G]
    eps_p = 1.0 + s_ref[_EPSSM]   # same single rounding as the reference's
    eps_m = 1.0 - s_ref[_EPSSM]   # python 1.0 -+ float(epssm)

    # scratch rw_eff = rw_tend (input refs are read-only in the pallas ABI)
    def rw_copy(k, _):
        rw_out[k, cols] = rw_ref[k, cols]
        return _

    jax.lax.fori_loop(0, nzf, rw_copy, 0)

    # --- 1. w_damping=1 vertical-CFL damping (advance_w.py:92-132) --------- #
    # unconditional explicit guard kept exactly as the reference:
    #   safe_mass_f = where(|c1f*mut+c2f| > 1e-12, mass, 1e-12)
    if w_damp:
        def damp_body(k, _):
            massf = c1f_ref[k] * mut + c2f_ref[k]
            safe = jnp.where(jnp.abs(massf) > 1.0e-12, massf,
                             jnp.asarray(1.0e-12, dtype=massf.dtype))
            vert_cfl = jnp.abs(((ww_ref[k, cols] / safe) * rdnw_ref[k]) * dts)
            activate = vert_cfl > s_ref[_W_DAMP_ON]
            damp = (((-jnp.sign(w_ref[k, cols])) * s_ref[_W_ALPHA])
                    * (vert_cfl - s_ref[_W_CRIT])) * massf
            rw_out[k, cols] = rw_out[k, cols] + jnp.where(
                activate, damp, jnp.zeros_like(damp))
            return _

        jax.lax.fori_loop(1, nz, damp_body, 0)

    # --- 2. t_2ave (WRF :1341-1344), mass levels --------------------------- #
    def t2ave_body(k, _):
        massh = c1h_ref[k] * muts + c2h_ref[k]
        half = 0.5 * (eps_p * t2_ref[k, cols] + eps_m * t2ave_ref[k, cols])
        theta_ref = s_ref[_T0] + t1_ref[k, cols]
        t2ave_out[k, cols] = (half + (c1h_ref[k] * muave) * s_ref[_T0]) / (
            massh * theta_ref)
        return _

    jax.lax.fori_loop(0, nz, t2ave_body, 0)

    # --- 3. phi rhs (WRF :1345, :1333) -------------------------------------- #
    def rhs_main_body(k, _):
        rhs_out[k, cols] = dts * (pht_ref[k, cols]
                                  + ((0.5 * g) * eps_m) * w_ref[k, cols])
        return _

    jax.lax.fori_loop(1, nzf, rhs_main_body, 0)
    rhs_out[0, cols] = zeros  # rhs(i,1) = 0 (WRF :1333)

    # --- 4. phi advection (WRF :1370-1382), faces 1..nz --------------------- #
    # ph_total_1 = ph_1 + phb recomputed per use (value-identical expression).
    def wdwn_body(k, _):
        if wrf_fp32:
            # Literal pristine WRF association in binary32: differencing the
            # perturbation first avoids rounding it into the large base sum.
            dphi = ((ph1_ref[k + 1, cols] - ph1_ref[k, cols])
                    + phb_ref[k + 1, cols]) - phb_ref[k, cols]
        else:
            dphi = (ph1_ref[k + 1, cols] + phb_ref[k + 1, cols]) \
                - (ph1_ref[k, cols] + phb_ref[k, cols])
        ww_mid = 0.5 * (ww_ref[k + 1, cols] + ww_ref[k, cols])
        wdwn_out[k + 1, cols] = (ww_mid * rdnw_ref[k]) * dphi
        return _

    jax.lax.fori_loop(0, nz, wdwn_body, 0)

    # --- 5. rhs -= dts*(fnm*wdwn(k+1) + fnp*wdwn(k)), faces 1..nz-1 --------- #
    if nz >= 2:
        def rhs_adv_body(k, _):
            adv = dts * ((fnm_ref[k] * wdwn_out[k + 1, cols])
                         + (fnp_ref[k] * wdwn_out[k, cols]))
            rhs_out[k, cols] = rhs_out[k, cols] - adv
            return _

        jax.lax.fori_loop(1, nz, rhs_adv_body, 0)

    # --- 6. rhs finalize (WRF :1393-1398), faces 1..nz ---------------------- #
    def rhs_fin_body(k, _):
        massf = c1f_ref[k] * mut + c2f_ref[k]
        rhs_out[k, cols] = ph_ref[k, cols] + (msfty * rhs_out[k, cols]) / massf
        return _

    jax.lax.fori_loop(1, nzf, rhs_fin_body, 0)
    if top_lid:
        rhs_out[nz, cols] = zeros

    # --- 7. terrain-following surface w (WRF :1417-1429), face 0 ------------ #
    ncol_i32 = jnp.int32(ncol - 1)
    c_e = jnp.minimum(ci + 1, ncol_i32)
    c_w = jnp.maximum(ci - 1, jnp.int32(0))
    c_n = jnp.minimum(ci + nx, ncol_i32)
    c_s = jnp.maximum(ci - nx, jnp.int32(0))
    ht = ht_ref[cols]
    ht_dy_n = jnp.where(j_idx < ny - 1, ht_ref[c_n] - ht, 0.0)
    ht_dy_s = jnp.where(j_idx > 0, ht - ht_ref[c_s], 0.0)
    ht_dx_e = jnp.where(i_idx < nx - 1, ht_ref[c_e] - ht, 0.0)
    ht_dx_w = jnp.where(i_idx > 0, ht - ht_ref[c_w], 0.0)
    v_cf_n = (s_ref[_CF1] * vn_ref[0, cols] + s_ref[_CF2] * vn_ref[1, cols]) \
        + s_ref[_CF3] * vn_ref[2, cols]
    v_cf_s = (s_ref[_CF1] * vs_ref[0, cols] + s_ref[_CF2] * vs_ref[1, cols]) \
        + s_ref[_CF3] * vs_ref[2, cols]
    u_cf_e = (s_ref[_CF1] * ue_ref[0, cols] + s_ref[_CF2] * ue_ref[1, cols]) \
        + s_ref[_CF3] * ue_ref[2, cols]
    u_cf_w = (s_ref[_CF1] * uw_ref[0, cols] + s_ref[_CF2] * uw_ref[1, cols]) \
        + s_ref[_CF3] * uw_ref[2, cols]
    w_surface = (((msfty * 0.5) * s_ref[_RDY])
                 * (ht_dy_n * v_cf_n + ht_dy_s * v_cf_s)
                 + ((msftx_ref[cols] * 0.5) * s_ref[_RDX])
                 * (ht_dx_e * u_cf_e + ht_dx_w * u_cf_w))

    # --- 8. explicit w update (WRF :1477) ----------------------------------- #
    def wnext_body(k, _):
        w_out[k, cols] = w_ref[k, cols] + dts * rw_out[k, cols]
        return _

    jax.lax.fori_loop(0, nzf, wnext_body, 0)

    # --- 9. interior faces 1..nz-1 (WRF :1477-1489) ------------------------- #
    if nz >= 2:
        def interior_body(k, _):
            massh_k = c1h_ref[k] * mut + c2h_ref[k]
            safe_k = jnp.where(jnp.abs(massh_k) > 1.0e-12, massh_k,
                               jnp.asarray(1.0e-12, dtype=massh_k.dtype))
            massh_l = c1h_ref[k - 1] * mut + c2h_ref[k - 1]
            safe_l = jnp.where(jnp.abs(massh_l) > 1.0e-12, massh_l,
                               jnp.asarray(1.0e-12, dtype=massh_l.dtype))
            coef_k = (c2a_ref[k, cols] * rdnw_ref[k]) / safe_k
            coef_l = (c2a_ref[k - 1, cols] * rdnw_ref[k - 1]) / safe_l
            term_upper = coef_k * (eps_p * (rhs_out[k + 1, cols] - rhs_out[k, cols])
                                   + eps_m * (ph_ref[k + 1, cols] - ph_ref[k, cols]))
            term_lower = coef_l * (eps_p * (rhs_out[k, cols] - rhs_out[k - 1, cols])
                                   + eps_m * (ph_ref[k, cols] - ph_ref[k - 1, cols]))
            if wrf_fp32:
                term_a = (msfty_inv * cqw_ref[k, cols]) * (
                    (((0.5 * dts) * g) * rdn_ref[k]) * (term_upper - term_lower))
            else:
                term_a = (msfty_inv * cqw_ref[k, cols]) \
                    * (((0.5 * dts) * g) * rdn_ref[k]) * (term_upper - term_lower)
            buoy_u = c2a_ref[k, cols] * alt_ref[k, cols] * t2ave_out[k, cols]
            buoy_l = c2a_ref[k - 1, cols] * alt_ref[k - 1, cols] * t2ave_out[k - 1, cols]
            term_b = ((dts * g) * msfty_inv) * (
                rdn_ref[k] * (buoy_u - buoy_l) - c1f_ref[k] * muave)
            if wrf_fp32:
                w_out[k, cols] = (w_out[k, cols] + term_a) + term_b
            else:
                w_out[k, cols] = w_out[k, cols] + (term_a + term_b)
            return _

        jax.lax.fori_loop(1, nz, interior_body, 0)

    # --- 10. top face nz (WRF :1492-1502) ----------------------------------- #
    km1 = nz - 1
    massh_t = c1h_ref[km1] * mut + c2h_ref[km1]
    safe_t = jnp.where(jnp.abs(massh_t) > 1.0e-12, massh_t,
                       jnp.asarray(1.0e-12, dtype=massh_t.dtype))
    rhs_diff_top = eps_p * (rhs_out[nz, cols] - rhs_out[km1, cols]) \
        + eps_m * (ph_ref[nz, cols] - ph_ref[km1, cols])
    term_a_top = (((((-0.5 * dts) * g) / safe_t)
                   * (rdnw_ref[km1] ** 2)) * 2.0) * c2a_ref[km1, cols] * rhs_diff_top
    term_b_top = ((-dts) * g) * (
        (((2.0 * rdnw_ref[km1]) * c2a_ref[km1, cols]) * alt_ref[km1, cols])
        * t2ave_out[km1, cols] + c1f_ref[nz] * muave)
    # The reference recomputes the top face from the ORIGINAL w/rw_tend
    # (advance_w.py:553), not from the already-updated w_next row — read
    # w_ref[nz] here, or the dts*rw term is applied twice.
    w_top = w_ref[nz, cols] + dts * rw_out[nz, cols] \
        + msfty_inv * (term_a_top + term_b_top)
    if top_lid:
        w_top = zeros
    wtop_out[nz, cols] = w_top  # debug hook: pre-sweep top face
    w_out[nz, cols] = w_top
    # surface w (WRF :1417-1429) at face 0
    w_out[0, cols] = w_surface

    # --- 11. Thomas forward sweep (WRF :1533-1537), in place ---------------- #
    def thomas_fwd(k, _):
        w_out[k, cols] = (w_out[k, cols]
                          - (a_ref[k, cols] * w_out[k - 1, cols])) * alpha_ref[k, cols]
        return _

    jax.lax.fori_loop(1, nzf, thomas_fwd, 0)

    # --- 12. Thomas back substitution (WRF :1546-1550), faces nz-1..1 ------- #
    if nz >= 2:
        def thomas_back(i, _):
            k = (nz - 1) - i
            w_out[k, cols] = w_out[k, cols] - (gamma_ref[k, cols] * w_out[k + 1, cols])
            return _

        # length nz-1: the reference sweeps gamma[1:nz], i.e. faces nz-1..1
        jax.lax.fori_loop(0, nz - 1, thomas_back, 0)

    # --- 13. damp_opt=3 implicit Rayleigh w-damping (WRF :1559-1572) -------- #
    if damp3:
        def rayleigh_body(k, _):
            hk = (ph1_ref[k, cols] + phb_ref[k, cols]) / g
            hbot = (ph1_ref[nz, cols] + phb_ref[nz, cols]) / g - s_ref[_HDEPTH]
            sine = jnp.sin((s_ref[_HALF_PI] * (hk - hbot)) / s_ref[_HDEPTH])
            if wrf_fp32:
                ramp = (s_ref[_DAMPMAG] * sine) * sine
                dampwt = jnp.where(hk >= hbot, ramp, jnp.zeros_like(ramp))
            else:
                ramp = sine ** 2
                dampwt = jnp.where(hk >= hbot, s_ref[_DAMPMAG] * ramp,
                                   jnp.zeros_like(ramp))
            massf = c1f_ref[k] * mut + c2f_ref[k]
            w_out[k, cols] = (w_out[k, cols] - ((dampwt * massf)
                                                * wsave_ref[k, cols])) / (1.0 + dampwt)
            return _

        jax.lax.fori_loop(1, nzf, rayleigh_body, 0)

    # --- 14. geopotential finish (WRF :1581-1586), faces 1..nz -------------- #
    def ph_finish_body(k, _):
        massf_muts = c1f_ref[k] * muts + c2f_ref[k]
        ph_out[k, cols] = rhs_out[k, cols] + ((((((msfty * 0.5) * dts) * g)
                                                * eps_p) * w_out[k, cols]) / massf_muts)
        return _

    jax.lax.fori_loop(1, nzf, ph_finish_body, 0)
    ph_out[0, cols] = ph_ref[0, cols]  # WRF keeps face 0 (input ph row 0)


def _advance_w_kernel_2pass(
    w_ref, rw_ref, ww_ref, ph_ref, ph1_ref, phb_ref, pht_ref,
    cqw_ref, a_ref, alpha_ref, gamma_ref, wsave_ref,
    t2_ref, t2ave_ref, t1_ref, c2a_ref, alt_ref,
    c1f_ref, c2f_ref, c1h_ref, c2h_ref, rdnw_ref, rdn_ref, fnm_ref, fnp_ref,
    mut_ref, muts_ref, muave_ref, ht_ref, msftx_ref, msfty_ref,
    ue_ref, uw_ref, vs_ref, vn_ref,
    s_ref,
    w_out, ph_out, t2ave_out, rhs_out,
    *, nz: int, nzf: int, nx: int, ny: int, top_lid: bool, damp3: bool, wrf_fp32: bool = True,
    keep=None,
):
    """``_advance_w_kernel`` in two level passes (w_damping=0 path).

    Same per-element expressions and order as the 14-pass kernel; the level
    loops are merged so each column walks the levels twice instead of 14
    times.  Ascending: t_2ave(k-1), wdwn(k+1) (wdwn(k) carried), final rhs(k),
    explicit w update of face k-1 (rhs/t_2ave/ph of k-2..k carried) and its
    Thomas forward step.  Descending: back substitution (undamped value
    carried), damp_opt=3 Rayleigh, geopotential finish.  ``rw_tend`` is read
    in place (no damping copy) and the wdwn/rw_eff/wtop scratch planes are gone.
    ``keep`` (TX bool, optional): columns where the final w/ph/t_2ave stores take
    the new value; elsewhere the input value is stored (specified/nested ring).
    """

    def kept(new, old_ref, k):
        return new if keep is None else jnp.where(keep, new, old_ref[k, cols])

    pid = pl.program_id(0)
    start = pid * TX
    ncol = mut_ref.shape[0]
    cols = pl.ds(start, TX)
    ci = start + jnp.arange(TX, dtype=jnp.int32)
    i_idx = ci % nx
    j_idx = ci // nx
    zeros = jnp.zeros((TX,), dtype=w_ref.dtype)
    mut = mut_ref[cols]
    muts = muts_ref[cols]
    muave = muave_ref[cols]
    msfty = msfty_ref[cols]
    msfty_inv = 1.0 / msfty
    dts = s_ref[_DTS]
    g = s_ref[_G]
    eps_p = 1.0 + s_ref[_EPSSM]
    eps_m = 1.0 - s_ref[_EPSSM]

    # surface w (WRF :1417-1429), identical expressions
    ncol_i32 = jnp.int32(ncol - 1)
    c_e = jnp.minimum(ci + 1, ncol_i32)
    c_w = jnp.maximum(ci - 1, jnp.int32(0))
    c_n = jnp.minimum(ci + nx, ncol_i32)
    c_s = jnp.maximum(ci - nx, jnp.int32(0))
    ht = ht_ref[cols]
    ht_dy_n = jnp.where(j_idx < ny - 1, ht_ref[c_n] - ht, 0.0)
    ht_dy_s = jnp.where(j_idx > 0, ht - ht_ref[c_s], 0.0)
    ht_dx_e = jnp.where(i_idx < nx - 1, ht_ref[c_e] - ht, 0.0)
    ht_dx_w = jnp.where(i_idx > 0, ht - ht_ref[c_w], 0.0)
    v_cf_n = (s_ref[_CF1] * vn_ref[0, cols] + s_ref[_CF2] * vn_ref[1, cols]) \
        + s_ref[_CF3] * vn_ref[2, cols]
    v_cf_s = (s_ref[_CF1] * vs_ref[0, cols] + s_ref[_CF2] * vs_ref[1, cols]) \
        + s_ref[_CF3] * vs_ref[2, cols]
    u_cf_e = (s_ref[_CF1] * ue_ref[0, cols] + s_ref[_CF2] * ue_ref[1, cols]) \
        + s_ref[_CF3] * ue_ref[2, cols]
    u_cf_w = (s_ref[_CF1] * uw_ref[0, cols] + s_ref[_CF2] * uw_ref[1, cols]) \
        + s_ref[_CF3] * uw_ref[2, cols]
    w_surface = (((msfty * 0.5) * s_ref[_RDY])
                 * (ht_dy_n * v_cf_n + ht_dy_s * v_cf_s)
                 + ((msftx_ref[cols] * 0.5) * s_ref[_RDX])
                 * (ht_dx_e * u_cf_e + ht_dx_w * u_cf_w))

    def safe(mass):
        return jnp.where(jnp.abs(mass) > 1.0e-12, mass, jnp.asarray(1.0e-12, dtype=mass.dtype))

    def t2ave_at(k):
        massh = c1h_ref[k] * muts + c2h_ref[k]
        half = 0.5 * (eps_p * t2_ref[k, cols] + eps_m * t2ave_ref[k, cols])
        theta_ref = s_ref[_T0] + t1_ref[k, cols]
        value = (half + (c1h_ref[k] * muave) * s_ref[_T0]) / (massh * theta_ref)
        t2ave_out[k, cols] = kept(value, t2ave_ref, k)
        return value

    def wdwn_at(k):  # face k+1 value (original wdwn_out[k+1])
        if wrf_fp32:
            dphi = ((ph1_ref[k + 1, cols] - ph1_ref[k, cols]) + phb_ref[k + 1, cols]) - phb_ref[k, cols]
        else:
            dphi = (ph1_ref[k + 1, cols] + phb_ref[k + 1, cols]) - (ph1_ref[k, cols] + phb_ref[k, cols])
        ww_mid = 0.5 * (ww_ref[k + 1, cols] + ww_ref[k, cols])
        return (ww_mid * rdnw_ref[k]) * dphi

    def rhs_at(k, wd_k, wd_kp1):
        r = dts * (pht_ref[k, cols] + ((0.5 * g) * eps_m) * w_ref[k, cols])
        if wd_kp1 is not None:
            r = r - dts * ((fnm_ref[k] * wd_kp1) + (fnp_ref[k] * wd_k))
        massf = c1f_ref[k] * mut + c2f_ref[k]
        r = ph_ref[k, cols] + (msfty * r) / massf
        return r

    def w_interior(j, rhs_jp1, rhs_j, rhs_jm1, t2a_j, t2a_jm1):
        w_e = w_ref[j, cols] + dts * rw_ref[j, cols]
        massh_k = c1h_ref[j] * mut + c2h_ref[j]
        massh_l = c1h_ref[j - 1] * mut + c2h_ref[j - 1]
        coef_k = (c2a_ref[j, cols] * rdnw_ref[j]) / safe(massh_k)
        coef_l = (c2a_ref[j - 1, cols] * rdnw_ref[j - 1]) / safe(massh_l)
        term_upper = coef_k * (eps_p * (rhs_jp1 - rhs_j) + eps_m * (ph_ref[j + 1, cols] - ph_ref[j, cols]))
        term_lower = coef_l * (eps_p * (rhs_j - rhs_jm1) + eps_m * (ph_ref[j, cols] - ph_ref[j - 1, cols]))
        if wrf_fp32:
            term_a = (msfty_inv * cqw_ref[j, cols]) * ((((0.5 * dts) * g) * rdn_ref[j]) * (term_upper - term_lower))
        else:
            term_a = (msfty_inv * cqw_ref[j, cols]) * (((0.5 * dts) * g) * rdn_ref[j]) * (term_upper - term_lower)
        buoy_u = c2a_ref[j, cols] * alt_ref[j, cols] * t2a_j
        buoy_l = c2a_ref[j - 1, cols] * alt_ref[j - 1, cols] * t2a_jm1
        term_b = ((dts * g) * msfty_inv) * (rdn_ref[j] * (buoy_u - buoy_l) - c1f_ref[j] * muave)
        if wrf_fp32:
            return (w_e + term_a) + term_b
        return w_e + (term_a + term_b)

    rhs_out[0, cols] = zeros
    wf0 = w_surface
    # ascending pass: k = 1..nz (faces); carries: wdwn(k), rhs(k-1), rhs(k-2), t2ave(k-2), wf(k-2)
    if nz < 2:
        raise ValueError("advance_w 2-pass needs nz >= 2")
    # face 1 seeds the carries
    wd1 = wdwn_at(0)
    t2a_0 = t2ave_at(0)
    wd2 = wdwn_at(1)
    rhs_1 = rhs_at(1, wd1, wd2)
    rhs_out[1, cols] = rhs_1
    if True:
        def asc(k, carry):  # k = 2..nz-1
            wd_k, rhs_km1, rhs_km2, t2a_km2, wf_km2 = carry
            t2a_km1 = t2ave_at(k - 1)
            wd_kp1 = wdwn_at(k)
            rhs_k = rhs_at(k, wd_k, wd_kp1)
            rhs_out[k, cols] = rhs_k
            j = k - 1
            w_upd = w_interior(j, rhs_k, rhs_km1, rhs_km2, t2a_km1, t2a_km2)
            wf_j = (w_upd - (a_ref[j, cols] * wf_km2)) * alpha_ref[j, cols]
            w_out[j, cols] = wf_j
            return (wd_kp1, rhs_k, rhs_km1, t2a_km1, wf_j)
        # rhs(0) is zero (WRF :1333); rhs(-1) never used (face 1 lower term uses rhs(0))
        carry = (wd2, rhs_1, zeros, t2a_0, wf0)
        carry = jax.lax.fori_loop(2, nz, asc, carry)
        wd_nz, rhs_nzm1, rhs_nzm2, t2a_nzm2, wf_nzm2 = carry
        # k = nz: no rhs advection term (range 1..nz-1), top_lid zeroes rhs(nz)
        t2a_nzm1 = t2ave_at(nz - 1)
        rhs_nz = rhs_at(nz, None, None)
        if top_lid:
            rhs_nz = zeros
        rhs_out[nz, cols] = rhs_nz
        j = nz - 1
        w_upd = w_interior(j, rhs_nz, rhs_nzm1, rhs_nzm2, t2a_nzm1, t2a_nzm2)
        wf_nzm1 = (w_upd - (a_ref[j, cols] * wf_nzm2)) * alpha_ref[j, cols]
        w_out[j, cols] = wf_nzm1
    # top face nz (WRF :1492-1502), recomputed from the ORIGINAL w/rw_tend
    km1 = nz - 1
    massh_t = c1h_ref[km1] * mut + c2h_ref[km1]
    rhs_diff_top = eps_p * (rhs_nz - rhs_nzm1) + eps_m * (ph_ref[nz, cols] - ph_ref[km1, cols])
    term_a_top = (((((-0.5 * dts) * g) / safe(massh_t))
                   * (rdnw_ref[km1] ** 2)) * 2.0) * c2a_ref[km1, cols] * rhs_diff_top
    term_b_top = ((-dts) * g) * (
        (((2.0 * rdnw_ref[km1]) * c2a_ref[km1, cols]) * alt_ref[km1, cols]) * t2a_nzm1 + c1f_ref[nz] * muave)
    w_top = w_ref[nz, cols] + dts * rw_ref[nz, cols] + msfty_inv * (term_a_top + term_b_top)
    if top_lid:
        w_top = zeros
    wf_nz = (w_top - (a_ref[nz, cols] * wf_nzm1)) * alpha_ref[nz, cols]

    # descending pass: back substitution (WRF :1546-1550), Rayleigh, geopotential finish
    def finish(k, w_back):
        w_fin = w_back
        if damp3:
            hk = (ph1_ref[k, cols] + phb_ref[k, cols]) / g
            hbot = (ph1_ref[nz, cols] + phb_ref[nz, cols]) / g - s_ref[_HDEPTH]
            sine = jnp.sin((s_ref[_HALF_PI] * (hk - hbot)) / s_ref[_HDEPTH])
            if wrf_fp32:
                ramp = (s_ref[_DAMPMAG] * sine) * sine
                dampwt = jnp.where(hk >= hbot, ramp, jnp.zeros_like(ramp))
            else:
                ramp = sine ** 2
                dampwt = jnp.where(hk >= hbot, s_ref[_DAMPMAG] * ramp, jnp.zeros_like(ramp))
            massf = c1f_ref[k] * mut + c2f_ref[k]
            w_fin = (w_back - ((dampwt * massf) * wsave_ref[k, cols])) / (1.0 + dampwt)
        w_out[k, cols] = kept(w_fin, w_ref, k)
        massf_muts = c1f_ref[k] * muts + c2f_ref[k]
        ph_out[k, cols] = kept(rhs_out[k, cols] + ((((((msfty * 0.5) * dts) * g) * eps_p) * w_fin) / massf_muts),
                               ph_ref, k)

    finish(nz, wf_nz)

    def desc(i, w_next_back):
        k = (nz - 1) - i
        w_back = w_out[k, cols] - (gamma_ref[k, cols] * w_next_back)
        finish(k, w_back)
        return w_back

    jax.lax.fori_loop(0, nz - 1, desc, wf_nz)
    w_out[0, cols] = kept(w_surface, w_ref, 0)
    ph_out[0, cols] = ph_ref[0, cols]


def _advance_w_kernel_recur(
    wupd_ref, a_ref, alpha_ref, gamma_ref, rhs_ref, wsave_ref, ph1_ref, phb_ref, w_ref, ph_ref,
    c1f_ref, c2f_ref, mut_ref, muts_ref, msfty_ref, s_ref,
    w_out, ph_out,
    *, nz: int, damp3: bool, keep=None,
):
    """Column recurrences of ``_advance_w_kernel_2pass`` (WRF-REAL path) on operands
    precomputed by the pointwise w kernels: ``wupd`` = w_surface (face 0), explicit
    w update (faces 1..nz-1), w_top (face nz); ``rhs`` = final rhs. Thomas forward,
    back substitution, damp_opt=3 Rayleigh and geopotential finish: same expressions."""

    def kept(new, old_ref, k):
        return new if keep is None else jnp.where(keep, new, old_ref[k, cols])

    pid = pl.program_id(0)
    cols = pl.ds(pid * TX, TX)
    mut = mut_ref[cols]
    muts = muts_ref[cols]
    msfty = msfty_ref[cols]
    dts = s_ref[_DTS]
    g = s_ref[_G]
    eps_p = 1.0 + s_ref[_EPSSM]

    def fwd(j, wf_prev):
        wf_j = (wupd_ref[j, cols] - (a_ref[j, cols] * wf_prev)) * alpha_ref[j, cols]
        w_out[j, cols] = wf_j
        return wf_j

    w_surface = wupd_ref[0, cols]
    wf_nzm1 = jax.lax.fori_loop(1, nz, fwd, w_surface)
    wf_nz = (wupd_ref[nz, cols] - (a_ref[nz, cols] * wf_nzm1)) * alpha_ref[nz, cols]

    def finish(k, w_back):
        w_fin = w_back
        if damp3:
            hk = (ph1_ref[k, cols] + phb_ref[k, cols]) / g
            hbot = (ph1_ref[nz, cols] + phb_ref[nz, cols]) / g - s_ref[_HDEPTH]
            sine = jnp.sin((s_ref[_HALF_PI] * (hk - hbot)) / s_ref[_HDEPTH])
            ramp = (s_ref[_DAMPMAG] * sine) * sine
            dampwt = jnp.where(hk >= hbot, ramp, jnp.zeros_like(ramp))
            massf = c1f_ref[k] * mut + c2f_ref[k]
            w_fin = (w_back - ((dampwt * massf) * wsave_ref[k, cols])) / (1.0 + dampwt)
        w_out[k, cols] = kept(w_fin, w_ref, k)
        massf_muts = c1f_ref[k] * muts + c2f_ref[k]
        ph_out[k, cols] = kept(rhs_ref[k, cols] + ((((((msfty * 0.5) * dts) * g) * eps_p) * w_fin) / massf_muts),
                               ph_ref, k)

    finish(nz, wf_nz)

    def desc(i, w_next_back):
        k = (nz - 1) - i
        w_back = w_out[k, cols] - (gamma_ref[k, cols] * w_next_back)
        finish(k, w_back)
        return w_back

    jax.lax.fori_loop(0, nz - 1, desc, wf_nz)
    w_out[0, cols] = kept(w_surface, w_ref, 0)
    ph_out[0, cols] = ph_ref[0, cols]


def advance_w_pallas(
    *,
    w: jax.Array,
    rw_tend: jax.Array,
    ww: jax.Array,
    u: jax.Array,
    v: jax.Array,
    mu_work: jax.Array | None = None,
    mut: jax.Array,
    muave: jax.Array,
    muts: jax.Array,
    t_2ave: jax.Array,
    t_2: jax.Array,
    t_1: jax.Array,
    ph: jax.Array,
    ph_1: jax.Array,
    phb: jax.Array,
    ph_tend: jax.Array,
    ht: jax.Array,
    c2a: jax.Array,
    cqw: jax.Array,
    alt: jax.Array,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
    c1h: jax.Array,
    c2h: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    rdnw: jax.Array,
    rdn: jax.Array,
    fnm: jax.Array,
    fnp: jax.Array,
    cf1: float,
    cf2: float,
    cf3: float,
    msftx: jax.Array,
    msfty: jax.Array,
    rdx: float,
    rdy: float,
    dts: float,
    epssm: float,
    t0: float = 300.0,
    top_lid: bool = False,
    gravity: float = GRAVITY_M_S2,
    w_save: jax.Array | None = None,
    damp_opt: int = 0,
    dampcoef: float = 0.0,
    zdamp: float = 5000.0,
    w_damping: int = 0,
    w_alpha: float = 0.3,
    w_crit_cfl: float = 1.0,
    w_damp_on: float = 1.0,
    interpret: bool = True,
    debug_intermediates: bool = False,
):
    """XLA replacement for ``advance_w_wrf`` (same outputs, per substep).

    Returns ``(w_solved, ph_next, t_2ave_next)``.  ``mu_work`` is accepted for
    signature parity with the reference and ignored: it appears only in the
    reference's fp64-island upcast list and is never read by its arithmetic
    (verified by grep over ``advance_w_wrf``).  ``interpret=True`` (default)
    runs the JAX reference interpreter; the device bake-off runs
    ``interpret=False``.
    """

    _reject_safe_floors()
    nzf = int(w.shape[0])
    nz = nzf - 1
    ny, nx = int(mut.shape[0]), int(mut.shape[1])
    ncol = ny * nx
    if ncol < TX:
        raise ValueError(f"need at least {TX} columns, got {ncol}")
    if int(w_damping) not in (0, 1):
        raise ValueError(f"unsupported w_damping={w_damping}; use the XLA reference")
    if w_damping == 1 and float(w_damp_on) != 1.0:
        # w_damp_on is a module constant (W_BETA=1.0); the production call
        # site never overrides it, so any other value is out of contract.
        raise ValueError("non-default w_damp_on is outside the M2 contract")

    f64 = jnp.float64
    w_damp = int(w_damping) == 1
    damp3 = int(damp_opt) == 3 and w_save is not None and float(dampcoef) > 0.0
    wsave_eff = w_save if w_save is not None else jnp.zeros((nzf, ny, nx), f64)
    # Reference Python-float expressions, kept as runtime scalars (docstring).
    half_pi = 0.5 * math.pi
    dampmag = float(dts) * float(dampcoef)
    s = _scalar_stack(
        dts=dts, epssm=epssm, t0=t0, g=gravity, rdx=rdx, rdy=rdy,
        cf1=cf1, cf2=cf2, cf3=cf3, w_alpha=w_alpha, w_crit_cfl=w_crit_cfl,
        w_damp_on=w_damp_on, dampmag=dampmag, hdepth=zdamp, half_pi=half_pi,
    )

    as_f64 = lambda x: jnp.asarray(x, f64)
    pad = _pad_width(ncol)
    pc = lambda x: _pad_cols(pack_columns(as_f64(x)), pad)
    p1 = lambda x: _pad_cols(as_f64(x).reshape(-1), pad)
    args = (
        pc(w), pc(rw_tend), pc(ww), pc(ph), pc(ph_1), pc(phb), pc(ph_tend),
        pc(cqw), pc(a), pc(alpha), pc(gamma), pc(wsave_eff),
        pc(t_2), pc(t_2ave), pc(t_1), pc(c2a), pc(alt),
        as_f64(c1f), as_f64(c2f), as_f64(c1h), as_f64(c2h),
        as_f64(rdnw), as_f64(rdn), as_f64(fnm), as_f64(fnp),
        p1(mut), p1(muts), p1(muave), p1(ht), p1(msftx), p1(msfty),
        # The exact slices _cf_combo_3 consumes (advance_w.py:274-303):
        # u levels 0..2 at x-window 1..nx (east) and 0..nx-1 (west);
        # v levels 0..2 (south rows) and 1..3 (north rows) at y-rows 0..ny.
        pc(as_f64(u)[0:3, :, 1:]),
        pc(as_f64(u)[0:3, :, :nx]),
        pc(as_f64(v)[0:3, :ny, :]),
        pc(as_f64(v)[0:3, 1:, :]),
        s,
    )
    ncol_pad = ncol + pad
    out_shapes = [
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # w_solved
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # ph_next
        jax.ShapeDtypeStruct((nz, ncol_pad), f64),    # t_2ave_next
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # scratch rhs
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # scratch wdwn
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # scratch rw_eff
        jax.ShapeDtypeStruct((nzf, ncol_pad), f64),   # scratch wtop (debug)
    ]

    def kernel(*refs):
        return _advance_w_kernel(
            *refs,
            nz=nz, nzf=nzf, nx=nx, ny=ny,
            top_lid=bool(top_lid), w_damp=w_damp, damp3=bool(damp3),
        )

    w_c, ph_c, t2ave_c, rhs_c, wdwn_c, rw_c, wtop_c = pl.pallas_call(
        kernel, grid=(ncol_pad // TX,), out_shape=out_shapes, interpret=interpret,
    )(*args)
    outputs = (
        unpack_columns(w_c[:, :ncol], (nzf, ny, nx)),
        unpack_columns(ph_c[:, :ncol], (nzf, ny, nx)),
        unpack_columns(t2ave_c[:, :ncol], (nz, ny, nx)),
    )
    if debug_intermediates:
        # Debuggability hook: the rhs / wdwn / rw_eff / wtop scratch planes that
        # feed the sweeps, in column orientation (unpack with the same shapes).
        return outputs + (
            unpack_columns(rhs_c[:, :ncol], (nzf, ny, nx)),
            unpack_columns(wdwn_c[:, :ncol], (nzf, ny, nx)),
            unpack_columns(rw_c[:, :ncol], (nzf, ny, nx)),
            unpack_columns(wtop_c[:, :ncol], (nzf, ny, nx)),
        )
    return outputs
