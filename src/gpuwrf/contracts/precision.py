"""Precision registry for M3 state fields; centralizes fp64 defaults."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


FP64 = jnp.float64


@dataclass(frozen=True)
class DTypeRegistry:
    """Encapsulates per-field dtype lookup reused by state allocation and tests."""

    defaults: tuple[tuple[str, object], ...]

    @classmethod
    def fp64_defaults(cls) -> "DTypeRegistry":
        """Builds the M3 fp64 registry; single call-site documents the precision policy."""

        fields = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        return cls(tuple((field, FP64) for field in fields))

    def dtype_for(self, field: str):
        """Returns the dtype for one state field; guards against misspelled field names."""

        mapping = dict(self.defaults)
        if field not in mapping:
            raise KeyError(f"unknown state field {field!r}")
        return mapping[field]


DEFAULT_DTYPES = DTypeRegistry.fp64_defaults()
