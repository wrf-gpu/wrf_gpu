"""Optional multi-device sharding controls for DGX-scale runs.

The default path is deliberately host-level and inert: with
``ShardingConfig.enabled=False`` the selected runner is exactly
``run_forecast_operational``. That preserves the current single-GPU compiled graph
for users who do not opt in to sharding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal
import hashlib
import re

import jax


ShardAxis = Literal["x", "y"]
ShardBackend = Literal["pmap"]

COLLECTIVE_TOKENS: tuple[str, ...] = (
    "collective-permute",
    "all-gather",
    "all-reduce",
    "all-to-all",
    "partition-id",
    "replica-id",
)


@dataclass(frozen=True)
class ShardingConfig:
    """Static opt-in configuration for domain decomposition."""

    enabled: bool = False
    axis: ShardAxis = "x"
    num_partitions: int | None = None
    halo_width: int = 2
    backend: ShardBackend = "pmap"
    multi_node: bool = False
    coordinator_address: str | None = None
    process_id: int = 0
    process_count: int = 1

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"unsupported shard axis {self.axis!r}")
        if self.backend != "pmap":
            raise ValueError(f"unsupported sharding backend {self.backend!r}")
        if not 1 <= int(self.halo_width) <= 4:
            raise ValueError("halo_width must be in [1, 4]")
        if self.num_partitions is not None and int(self.num_partitions) < 1:
            raise ValueError("num_partitions must be positive when supplied")
        if int(self.process_count) < 1:
            raise ValueError("process_count must be positive")
        if int(self.process_id) < 0 or int(self.process_id) >= int(self.process_count):
            raise ValueError("process_id must be in [0, process_count)")
        if self.multi_node and not self.coordinator_address:
            raise ValueError("multi_node sharding requires coordinator_address")

    @classmethod
    def disabled(cls) -> "ShardingConfig":
        """Return the canonical default-off config."""

        return cls(enabled=False)

    def resolved_partitions(self, *, platform: str | None = None) -> int:
        """Return the local partition count this process should use."""

        if self.num_partitions is not None:
            return int(self.num_partitions)
        devices = jax.devices(platform) if platform is not None else jax.local_devices()
        return max(1, len(devices))


def select_forecast_runner(config: ShardingConfig | None = None) -> Callable[..., Any]:
    """Select the public forecast runner for an optional sharding config.

    Disabled sharding returns the existing default function object, not a jitted
    wrapper. This is the zero-overhead contract.
    """

    from gpuwrf.runtime.operational_mode import run_forecast_operational

    cfg = ShardingConfig.disabled() if config is None else config
    if not bool(cfg.enabled):
        return run_forecast_operational
    return run_forecast_operational_sharded


def run_forecast_operational_optional_sharding(
    state,
    namelist,
    hours: float,
    *,
    sharding: ShardingConfig | None = None,
):
    """Host-level optional entry point.

    With the default config this calls the exact existing compiled entry point.
    """

    return select_forecast_runner(sharding)(state, namelist, hours)


def run_forecast_operational_sharded(state, namelist, hours: float):
    """Opt-in sharded runtime entry point.

    S2/S3/S4 fill this in behind explicit sharding gates. It is intentionally not
    reachable from the default path.
    """

    del state, namelist, hours
    raise NotImplementedError("sharded runtime execution is opt-in and implemented after halo/operator gates")


def hlo_operation_count(hlo_text: str) -> int:
    """Count HLO operation definition lines in a backend-stable way."""

    count = 0
    for line in hlo_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if re.search(r"\b[-%.A-Za-z0-9_]+ = ", stripped):
            count += 1
    return count


def hlo_collective_counts(hlo_text: str) -> dict[str, int]:
    """Count collective/SPMD tokens that must be absent from flag-off HLO."""

    lower = hlo_text.lower()
    return {token: lower.count(token) for token in COLLECTIVE_TOKENS}


def hlo_sha256(hlo_text: str) -> str:
    """Stable hash for proof objects."""

    return hashlib.sha256(hlo_text.encode("utf-8")).hexdigest()


def hlo_graph_stats(hlo_text: str) -> dict[str, Any]:
    """Return the graph stats used by the disabled-sharding proof."""

    collectives = hlo_collective_counts(hlo_text)
    return {
        "op_count": int(hlo_operation_count(hlo_text)),
        "hlo_sha256": hlo_sha256(hlo_text),
        "hlo_size_bytes": len(hlo_text.encode("utf-8")),
        "collective_counts": collectives,
        "collectives_present": any(count > 0 for count in collectives.values()),
    }


def assert_flag_off_graph_unchanged(reference_hlo: str, candidate_hlo: str) -> None:
    """Raise if disabled sharding changes op count or introduces collectives."""

    reference = hlo_graph_stats(reference_hlo)
    candidate = hlo_graph_stats(candidate_hlo)
    if int(reference["op_count"]) != int(candidate["op_count"]):
        raise AssertionError(
            f"flag-off op count changed: {reference['op_count']} != {candidate['op_count']}"
        )
    if bool(candidate["collectives_present"]):
        raise AssertionError(f"flag-off HLO contains collectives: {candidate['collective_counts']}")


__all__ = [
    "COLLECTIVE_TOKENS",
    "ShardingConfig",
    "assert_flag_off_graph_unchanged",
    "hlo_collective_counts",
    "hlo_graph_stats",
    "hlo_operation_count",
    "hlo_sha256",
    "run_forecast_operational_optional_sharding",
    "run_forecast_operational_sharded",
    "select_forecast_runner",
]
