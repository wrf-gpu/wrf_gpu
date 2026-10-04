"""GPUWRF_DYN_REAL_ALL (S2-DYN, default off): WRF REAL DycoreMetrics plus REAL
arithmetic in the dycore consumers (rhs_ph, EOS diagnostics, advection prep,
advance_w moist terms, large-step augment, moisture/scalar updates, theta
limiter, diff_opt=1). WRF stores grid metrics and all dycore fields as REAL
(RWORDSIZE=4); the flag is read at trace/construction time (live env hash).
"""
import os

_REQUIRED = ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_DYN_CARRY_FP32")


def enabled():
    return os.environ.get("GPUWRF_DYN_REAL_ALL", "0") == "1"


def require_native():
    """The REAL package assumes the native acoustic/RK path and REAL carry."""
    missing = [name for name in _REQUIRED if os.environ.get(name, "0") != "1"]
    if missing:
        raise ValueError("GPUWRF_DYN_REAL_ALL requires " + ", ".join(missing))


def real(*arrays):
    """WRF REAL operands (None passes through); one value in, one value out."""
    import jax.numpy as jnp

    def _real(value):
        if not hasattr(value, "astype") or not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        return value if value.dtype == jnp.float32 else value.astype(jnp.float32)

    out = tuple(_real(a) for a in arrays)
    return out[0] if len(out) == 1 else out


def dyn_island():
    """``force_fp64_island``, or ``real`` under GPUWRF_DYN_REAL_ALL.

    The v0.20 fp64 islands (EOS, PGF brackets, implicit w/phi, rhs_ph sums) are
    REAL in WRF. Returns the function itself (call ``dyn_island()(...)``), so
    the flag-off trace keeps the original call stack and HLO.
    """
    if enabled():
        return real
    from gpuwrf.contracts.precision import force_fp64_island
    return force_fp64_island


def glue_parts():
    """GPUWRF_DYN_GLUE_FUSED parts (lever #3): "1" = {mom, uv}; else "_"-separated
    names from mom (momentum advection), uv / uv2 (large-step u/v; uv2 keeps the
    moisture-coupled cq as an XLA operand), omega (stage omega column kernel), rhsph
    (rhs_ph stencil), pin (row-major layout constraint on the fused kernels' operands)."""
    value = os.environ.get("GPUWRF_DYN_GLUE_FUSED", "0")
    if value in ("", "0"):
        return frozenset()
    if value == "1":
        return frozenset(("mom", "uv"))
    return frozenset(value.split("_"))


def pin_rows(arrays):
    """Part ``pin``: fix every >=2-D operand of a fused glue kernel to the row-major layout
    its Pallas custom call reads, so XLA cannot move the operand's producer (or the
    step-loop carry slot behind it) to a z-fastest layout and transpose per use (A62)."""
    if "pin" not in glue_parts():
        return tuple(arrays)
    from jax.experimental.layout import Layout, with_layout_constraint
    return tuple(with_layout_constraint(a, Layout(major_to_minor=tuple(range(a.ndim))))
                 if getattr(a, "ndim", 0) >= 2 else a for a in arrays)
