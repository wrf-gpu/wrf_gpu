"""Column physics kernels for M5 and later coupling work."""

from .thompson_column import ThompsonColumnState, step_thompson_column

__all__ = [
    "ThompsonColumnState",
    "step_thompson_column",
    "morrison_run",
    "morrison_tendency",
]


def __getattr__(name):
    # Lazy access to the Morrison 2-moment kernel (mp_physics=10) so importing
    # this package does not eagerly pull in the JAX kernel unless it is used.
    if name in ("morrison_run", "morrison_tendency"):
        from . import microphysics_morrison as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
