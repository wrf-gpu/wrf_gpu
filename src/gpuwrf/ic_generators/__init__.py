"""Initial-condition generators for operational validation cases."""

from __future__ import annotations

__all__ = [
    "IdealizedRunResult",
    "build_density_current_setup",
    "build_warm_bubble_setup",
    "run_density_current_case",
    "run_warm_bubble_case",
    "run_all_idealized_cases",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import idealized

    return getattr(idealized, name)
