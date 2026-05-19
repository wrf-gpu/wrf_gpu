"""Zero-production-cost debug hooks for M4 hot-path modules."""

from .asserts import assert_finite, assert_physical_bounds
from .snapshots import dump_snapshots, snapshot

__all__ = ["assert_finite", "assert_physical_bounds", "dump_snapshots", "snapshot"]
