"""Opt-in AFWA severe-weather WSPD10MAX (pristine WRF 4.7.1 module_diag_afwa.F).

WRF has no ``AFWA_GUST`` field. Under ``afwa_diag_opt=1, afwa_severe_opt=1`` the
AFWA driver (called after all RK stages and microphysics, every own step)
keeps the maximum 10 m wind since the last history write and, where the
step precipitation rate exceeds 50 mm/h, raises it towards the 1 km AGL
wind (module_diag_afwa.F:338-393, "surface wind gust potential"). The field
is a 10 m maximum-wind potential, not a measured gust.

Operands follow WRF's time levels exactly:
- ``u10/v10``: this step's surface-layer winds (computed on the step-start state);
- ``u/v``: step-START staggered winds (phy_prep ``u_phy/v_phy`` at RK1);
- ``ph/phb``: grid%z as the driver sees it (``heights``): moist_physics_prep_em
  recomputes it from the post-RK, pre-microphysics ``ph_2`` inside the microphysics
  zone only; the spec_zone ring keeps the step-start value (solve_em.F:3718-3775);
- ``raincv + rainncv``: this step's cumulus and microphysics amounts (mm).

The 1 km interpolation divides by ``zagl(k)``, not by the layer depth; this is
WRF's literal expression and is reproduced, not corrected. Arithmetic is
explicitly rounded (``.rn``) so the GPU program evaluates WRF REAL, not XLA's
``div.full`` (E84). Only tiles holding a heavy-precipitation column read the
3-D operands; ordinary steps touch five 2-D planes per column.
"""
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental.pallas import triton as pt

from gpuwrf.backends import pallas as pl

BLOCK = 32
SCHEMA_VERSION = 1
# &afwa sub-options of WRF 4.7.1 Registry/registry.afwa; only severe WSPD10MAX is ported.
UNSUPPORTED_SUBOPTIONS = ("afwa_ptype_opt", "afwa_vil_opt", "afwa_radar_opt", "afwa_icing_opt",
                          "afwa_vis_opt", "afwa_cloud_opt", "afwa_therm_opt", "afwa_turb_opt",
                          "afwa_buoy_opt")
_G = np.float32(9.81)  # module_model_constants g


class AfwaDiagnostics(NamedTuple):
    # Registry.EM_COMMON:2081 WSPD10MAX ij misc rh02 "WIND SPD MAX 10 M" "m s-1".
    wspd10max: object


def seed_afwa(state):
    # WRF allocates package state with 0 and resets it at step 1 anyway.
    return AfwaDiagnostics(jnp.zeros(state.t_skin.shape, jnp.float32))


def history_fields(diagnostics):
    return {} if diagnostics is None else {"WSPD10MAX": diagnostics.wspd10max}


def history_interval(namelist):
    """The reset window (minutes) a namelist resolves; None when not recorded."""
    from collections.abc import Mapping
    value = namelist.get("diagnostic_history_interval") if isinstance(namelist, Mapping) \
        else getattr(namelist, "diagnostic_history_interval", None)
    return None if value is None else float(value)


def configuration(namelist):
    """Resolved per-domain selection: ``{"afwa_severe_opt": 0|1}``."""
    from collections.abc import Mapping
    if isinstance(namelist, Mapping):
        if "afwa" not in namelist:  # flat resolved mapping (restart attributes)
            return {"afwa_severe_opt": int(namelist.get("afwa_severe_opt", 0) or 0)}
        group = namelist["afwa"] or {}
        first = lambda v: v[0] if isinstance(v, (list, tuple)) else v
        severe, diag = first(group.get("afwa_severe_opt", 0)), first(group.get("afwa_diag_opt", 0))
        return {"afwa_severe_opt": int(int(severe or 0) == 1 and int(diag or 0) == 1)}
    return {"afwa_severe_opt": int(getattr(namelist, "afwa_severe_opt", 0))}


def validate_restart_afwa(diagnostics, enabled, version, shape=None, *, history=None, check_history=False,
                          saved_interval=None, requested_interval=None):
    """E78: an active maximum cannot silently resume from zero, nor a stale one appear.

    Active restarts also need the in-step U10/V10 history carry (the next step's operand) and,
    when both are known, the saved history cadence must equal the requested one: the maximum's
    window is the history interval (critic FIX 6)."""
    if diagnostics is not None and check_history:
        # The carrier itself, not a look-alike: the step reads attributes and calls .write (Delta 1 FIX 1).
        from gpuwrf.runtime.operational_state import HistoryDiagnostics
        if not isinstance(history, HistoryDiagnostics):
            raise ValueError("restart AFWA WSPD10MAX lacks its U10/V10 history operands: HistoryDiagnostics "
                             f"carrier required, got {type(history).__name__} (E78)")
        u10, v10 = history.u10, history.v10
        if u10 is None or v10 is None or (shape is not None and (tuple(getattr(u10, "shape", ())) != tuple(shape)
                                                                 or tuple(getattr(v10, "shape", ())) != tuple(shape))):
            raise ValueError("restart AFWA WSPD10MAX lacks its U10/V10 history operands (E78)")
    if diagnostics is not None and saved_interval is not None and requested_interval is not None \
            and float(saved_interval) != float(requested_interval):
        raise ValueError("restart AFWA WSPD10MAX history cadence differs from the requested one (E78)")
    if diagnostics is None and not enabled:
        return
    if diagnostics is None:
        raise ValueError("restart lacks the AFWA WSPD10MAX carry (E78)")
    if not enabled:
        raise ValueError("restart carries AFWA WSPD10MAX but afwa_severe_opt=0 (E78)")
    if version != SCHEMA_VERSION or not isinstance(diagnostics, AfwaDiagnostics):
        raise ValueError("restart AFWA diagnostics schema is missing (E78)")
    leaf = diagnostics.wspd10max
    if leaf is None or getattr(leaf, "dtype", None) != jnp.float32 or len(leaf.shape) != 2:
        raise ValueError("restart AFWA WSPD10MAX is not a REAL surface plane (E78)")
    if shape is not None and tuple(leaf.shape) != tuple(shape):
        raise ValueError("restart AFWA WSPD10MAX differs from the surface grid (E78)")


def _exact(value):
    """The history writer's exact decimal rational (nested_pipeline._positive_fraction)."""
    import math
    from fractions import Fraction
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"AFWA clock needs a finite positive value, got {value!r}")
    return Fraction(str(numeric))


def reset_due(itimestep, dt, history_minutes):
    """``is_after_history_dump`` (module_diag_afwa.F:139) for the one-based own step.

    During solve WRF's CurrTime is the step start; the history alarm's
    PrevRingTime advances by the nominal interval (ESMF_Clock.F90:1027) and is
    the start time before the first ring. The step that STARTS right after a
    history write therefore resets; so does step 1. History is written after the
    completed step ``ceil(i * interval / dt)`` in the shared writer clock
    (nested_pipeline._history_alarm_steps, exact decimal rationals); this test is
    the same integer statement: some ``i`` has ``ceil(i*r) == step - 1``, r = p/q.
    """
    p, q = clock_ratio(dt, history_minutes)
    last = jnp.asarray(itimestep).astype(jnp.int64) - 1  # the completed step before this one
    if last.dtype != jnp.int64:
        raise ValueError("AFWA history clock needs 64-bit integers (jax_enable_x64)")
    return (last * q) // p > ((last - 1) * q) // p


def clock_ratio(dt, history_minutes):
    """(p, q) with history interval / dt = p/q in the writer's exact decimal rationals.

    Steps stay below 2**31, so (step * q) is exact in int64 when q < 2**31. A dt whose
    decimal rational is wider (e.g. 54/81 s = 0.6666666666666666) is refused, not rounded.
    """
    ratio = _exact(history_minutes) * 60 / _exact(dt)
    p, q = ratio.numerator, ratio.denominator
    if p >= 2**31 or q >= 2**31:
        raise ValueError(f"AFWA WSPD10MAX: history_interval/dt = {p}/{q} is not a short exact rational; "
                         "use a decimal-exact time step")
    return p, q


def _opaque(x, token):
    # Identity through a runtime zero: XLA:CPU/LLVM cannot fuse across it.
    return lax.bitcast_convert_type(lax.bitcast_convert_type(x, jnp.int32) ^ token, jnp.float32)


def _rn(op, a, b, *, interpret, token):
    """WRF REAL op: explicit ``.rn`` PTX on CUDA (no div.full, no contraction).

    XLA:CPU contracts ``u*u + v*v`` into an FMA inside a fusion, even across
    optimization barriers and DOUBLE round trips (measured 53/432 1-ULP oracle
    misses), and algsimp turns divisions by constants into reciprocal products
    (E123; 8/432 misses). The interpreter therefore passes operands and results
    through an opaque identity (operands only) keyed on a runtime zero: bit-exact REAL.
    """
    if not interpret and pl.cuda_native():
        return pt.elementwise_inline_asm(op + ".rn.f32 $0, $1, $2;", args=[a, b], constraints="=f,f,f",
            pack=1, result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]
    if not interpret:
        return pl.portable_rn(op, a, b, interpret=False)
    # Opaque operands too: algsimp rewrites x/const as x*(1/const) (E123).
    # Operands only: a second XOR with the same token would cancel in LLVM.
    return pl.portable_rn(op, _opaque(a, token), _opaque(b, token), interpret=True)


def _sqrt(x, interpret, token):
    if interpret:
        return jnp.sqrt(_opaque(x, token))
    if not pl.cuda_native():
        return jnp.sqrt(x)
    return pt.elementwise_inline_asm("sqrt.rn.f32 $0, $1;", args=[x], constraints="=f,f", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, jnp.float32)])[0]


def _afwa_kernel(old, u10, v10, raincv, rainncv, u, v, ph, phb, ht, clock, out,
                 *, nz, ny, nx, dt, interpret):
    lane = pl.program_id(0) * BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
    y, x = lane // nx, lane % nx
    valid = lane < ny * nx
    r = jnp.float32
    # clock = (reset,) on CUDA; (reset, runtime zero) for the interpreter's opaque rounding.
    token = jnp.broadcast_to(pt.load(clock.at[1]), (BLOCK,)) if interpret else None
    def rn(op, a, b):
        a, b = jnp.broadcast_arrays(jnp.asarray(a, r), jnp.asarray(b, r))
        return _rn(op, a, b, interpret=interpret, token=token)
    add, sub, mul, div = (partial(rn, op) for op in ("add", "sub", "mul", "div"))
    surface = lambda a: pt.load(a.at[y, x], mask=valid, other=r(0))
    reset = pt.load(clock.at[0]) > jnp.int32(0)
    previous = jnp.where(reset, r(0), surface(old))
    us, vs = surface(u10), surface(v10)
    wind = _sqrt(add(mul(us, us), mul(vs, vs)), interpret, token)
    # afwa_precip = raincv + rainncv; prate_mm_per_hr = (afwa_precip / dt) * 3600.
    prate = mul(div(add(surface(raincv), surface(rainncv)), r(dt)), r(3600))
    heavy = valid & (prate > r(50))

    def blend(wind):
        terrain = surface(ht)
        def level(k):
            zw = lambda kk: div(add(pt.load(ph.at[kk, y, x], mask=heavy, other=r(0)),
                                    pt.load(phb.at[kk, y, x], mask=heavy, other=r(0))), r(_G))
            zagl = sub(mul(r(.5), add(zw(k), zw(k + 1))), terrain)
            uk = mul(r(.5), add(pt.load(u.at[k, y, x], mask=heavy, other=r(0)),
                                 pt.load(u.at[k, y, x + 1], mask=heavy, other=r(0))))
            vk = mul(r(.5), add(pt.load(v.at[k, y, x], mask=heavy, other=r(0)),
                                 pt.load(v.at[k, y + 1, x], mask=heavy, other=r(0))))
            return zagl, uk, vk

        def cond(s):
            k, found = s[0], s[1]
            return (k < nz) & (jnp.max((heavy & ~found).astype(jnp.int32)) > jnp.int32(0))

        def body(s):
            k, found, u1, v1, zp, up, vp = s
            zk, uk, vk = level(k)
            hit = heavy & ~found & (zk >= r(1000))
            dz = sub(r(1000), zp)
            u1 = jnp.where(hit, add(up, mul(dz, div(sub(uk, up), zk))), u1)
            v1 = jnp.where(hit, add(vp, mul(dz, div(sub(vk, vp), zk))), v1)
            return k + 1, found | hit, u1, v1, zk, uk, vk

        z0, u0, v0 = level(jnp.int32(0))
        zero = jnp.zeros((BLOCK,), r)
        # The source loop starts at kms but skips k == kms. A column with no level
        # >= 1 km leaves u1km/v1km undefined in WRF (unreachable for real tops);
        # here they stay 0.
        _, _, u1, v1, _, _, _ = lax.while_loop(
            cond, body, (jnp.int32(1), jnp.zeros((BLOCK,), jnp.bool_), zero, zero, z0, u0, v0))
        factor = jnp.maximum(div(sub(r(150), prate), r(100)), r(0))
        rest = sub(r(1), factor)
        ub = add(mul(us, factor), mul(u1, rest))
        vb = add(mul(vs, factor), mul(v1, rest))
        gusty = _sqrt(add(mul(ub, ub), mul(vb, vb)), interpret, token)
        return jnp.where(heavy & (gusty > wind), gusty, wind)

    wind = lax.cond(jnp.max(heavy.astype(jnp.int32)) > jnp.int32(0), blend, lambda w: w, wind)
    pt.store(out.at[y, x], jnp.where(wind > previous, wind, previous), mask=valid)


def update_afwa(diagnostics, *, u10, v10, raincv, rainncv, u, v, ph, phb, terrain,
                itimestep, dt, history_interval, interpret=None):
    """One own step of the AFWA WSPD10MAX maximum (fixed dt, mass-grid cells)."""
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    nz = u.shape[0]
    ny, nx = diagnostics.wspd10max.shape
    if u.shape != (nz, ny, nx + 1) or v.shape != (nz, ny + 1, nx):
        raise ValueError("AFWA WSPD10MAX needs staggered u (z,y,x+1) and v (z,y+1,x)")
    if ph.shape != (nz + 1, ny, nx) or phb.shape != ph.shape:
        raise ValueError("AFWA WSPD10MAX needs full z-face PH/PHB")
    if float(dt) <= 0 or float(history_interval) <= 0 or float(history_interval) * 60 < float(dt):
        raise ValueError("AFWA WSPD10MAX needs positive dt no larger than the history interval")
    real = lambda a: jnp.asarray(a, jnp.float32)
    planes = tuple(real(a).reshape(ny, nx) for a in (u10, v10, raincv, rainncv))
    reset = reset_due(itimestep, dt, history_interval).astype(jnp.int32).reshape(1)
    if interpret:
        # A zero XLA cannot fold even when this wrapper is jitted with the kernel:
        # max(-step, 0) == 0 for every one-based step (a literal 0 let the jitted
        # CPU program contract FMAs again: critic 29/2160 ULP misses).
        zero = jnp.maximum(-jnp.asarray(itimestep).astype(jnp.int32), jnp.int32(0)).reshape(1)
        clock = jnp.concatenate((reset, zero))
    else:
        clock = reset
    shape = jax.ShapeDtypeStruct((ny, nx), jnp.float32)
    value = pl.pallas_call(
        partial(_afwa_kernel, nz=nz, ny=ny, nx=nx, dt=float(np.float32(dt)), interpret=interpret),
        out_shape=shape, grid=((ny * nx + BLOCK - 1) // BLOCK,), interpret=interpret,
        name="afwa_wspd10max_column", compiler_params=pt.CompilerParams(num_warps=4))(
        real(diagnostics.wspd10max), *planes, real(u), real(v), real(ph), real(phb),
        real(terrain).reshape(ny, nx), clock)
    return AfwaDiagnostics(value)


def validate_options(namelist, *, wrf_order, kf_rk):
    """Refuse configurations whose WSPD10MAX operands or ownership are unqualified."""
    import math
    if int(namelist.afwa_severe_opt) not in (0, 1):
        raise ValueError("afwa_severe_opt supports only 0/1")
    if not namelist.run_physics or int(namelist.mp_physics) != 8 or int(namelist.cu_physics) not in (0, 1):
        raise ValueError("AFWA WSPD10MAX requires physics with Thompson mp_physics=8 and cu_physics=0/1 "
                         "(raw RAINNCV/RAINCV step amounts)")
    if not wrf_order or (int(namelist.cu_physics) == 1 and not kf_rk):
        raise ValueError("AFWA WSPD10MAX requires WRF-order post-RK microphysics and WRF RK KF rates")
    # U10/V10 must be this step's surface_driver values: the Noah-MP surface step writes them into the
    # history carry every step (the same values history U10/V10 publish). Other suites (e.g. sfclay=5
    # without an LSM) leave the carry at its seed (critic FIX 1): refused, not guessed.
    if not bool(namelist.use_noahmp) or getattr(namelist, "sf_surface_physics", 4) not in (None, 4) \
            or int(namelist.sf_sfclay_physics) != 5:
        raise ValueError("AFWA WSPD10MAX is qualified for MYNN surface layer sf_sfclay_physics=5 with Noah-MP "
                         "sf_surface_physics=4 only (the producer of the in-step U10/V10)")
    if int(getattr(namelist, "nwp_diagnostics", 0)):
        raise ValueError("afwa_severe_opt=1 and nwp_diagnostics=1 both own WSPD10MAX with different reset "
                         "rules; select one")
    minutes = float(namelist.diagnostic_history_interval)
    if not math.isfinite(minutes) or minutes <= 0 or float(namelist.dt_s) <= 0 or minutes * 60 < float(namelist.dt_s):
        raise ValueError("AFWA WSPD10MAX requires a positive fixed dt no larger than history_interval")
    clock_ratio(namelist.dt_s, minutes)


def heights(entry, pre_microphysics, base_state, spec_zone):
    """PH for grid%z as WRF's AFWA driver sees it (critic FIX 2).

    moist_physics_prep_em (solve_em.F:3718-3775) refreshes grid%z from the post-RK,
    pre-microphysics ph_2 only inside the microphysics zone (spec_zone ring excluded
    on specified/nested domains), before spec_bdy_final pins ph_2 (:4687). The ring keeps
    the step-entry grid%z of phy_prep. Returns (ph, phb) REAL.
    """
    from gpuwrf.diagnostics.online import ph_parts
    ph_new, phb = ph_parts(pre_microphysics, base_state)
    if not spec_zone:
        return ph_new, phb
    ph_old, _ = ph_parts(entry, base_state)
    ny, nx = ph_new.shape[-2:]
    y, x = jnp.arange(ny)[:, None], jnp.arange(nx)[None, :]
    inside = (y >= spec_zone) & (y < ny - spec_zone) & (x >= spec_zone) & (x < nx - spec_zone)
    return jnp.where(inside[None], ph_new, ph_old), phb


def after_step(carry, namelist, step_index, origin, amounts, ph_operands):
    """WRF ``afwa_diagnostics_driver`` WSPD10MAX at the end of one own step.

    origin: step-entry State (u_phy/v_phy of phy_prep at RK1); ph_operands: ``heights(...)``;
    amounts: (RAINCV, RAINNCV) of this step; U10/V10: this step's history carry.
    """
    history = carry.history_diagnostics
    if history is None or getattr(history, "u10", None) is None or getattr(history, "v10", None) is None:
        raise ValueError("AFWA WSPD10MAX needs the in-step U10/V10 history carry (E78)")
    raincv, rainncv = amounts
    ph, phb = ph_operands
    return carry.replace(afwa_diagnostics=update_afwa(
        carry.afwa_diagnostics, u10=history.u10, v10=history.v10, raincv=raincv, rainncv=rainncv,
        u=origin.u, v=origin.v, ph=ph, phb=phb, terrain=namelist.grid.terrain_height,
        itimestep=step_index, dt=namelist.dt_s, history_interval=namelist.diagnostic_history_interval))
