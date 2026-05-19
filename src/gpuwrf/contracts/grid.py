"""Grid contract for the M3 GPU-resident state skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


ProjectionKind = Literal["lambert", "mercator", "polar"]
BcSource = Literal["AIFS", "GFS", "ERA5", "ideal"]
Interpolation = Literal["linear", "cubic"]


@dataclass(frozen=True)
class Projection:
    """Groups projection scalars so GridSpec metadata stays hashable and readable."""

    kind: ProjectionKind
    lat_0: float
    lon_0: float
    dx_m: float
    dy_m: float
    nx: int
    ny: int


@dataclass(frozen=True)
class TerrainProvenance:
    """Captures terrain provenance without carrying the terrain array in metadata."""

    source_path: str
    sha256: str
    shape: tuple[int, int]
    units: str
    projection_transform: str
    max_elevation_m: float
    coastline_sanity_check_passed: bool


@dataclass(frozen=True)
class VerticalCoord:
    """Stores vertical-coordinate metadata needed by state shape invariants."""

    kind: Literal["hybrid_eta"]
    nz: int
    top_pressure_pa: float
    eta_levels: jax.Array


@dataclass(frozen=True)
class BCMetadata:
    """Stores boundary-condition provenance for future restart-compatible coupling."""

    source: BcSource
    fields: tuple[str, ...]
    update_cadence_h: int
    interpolation: Interpolation
    restart_compatible: bool


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GridSpec:
    """JAX pytree grid contract; array leaves are eta levels and terrain height."""

    projection: Projection
    terrain: TerrainProvenance
    vertical: VerticalCoord
    bc: BCMetadata
    eta_levels: jax.Array
    terrain_height: jax.Array
    halo_width: int = 2
    staggering: Literal["c-grid"] = "c-grid"

    def __post_init__(self) -> None:
        """Enforces M3 grid invariants once, before the object enters JIT call sites."""

        if self.vertical.eta_levels is not self.eta_levels:
            object.__setattr__(
                self,
                "vertical",
                VerticalCoord(
                    self.vertical.kind,
                    self.vertical.nz,
                    self.vertical.top_pressure_pa,
                    self.eta_levels,
                ),
            )
        if self.projection.kind not in ("lambert", "mercator", "polar"):
            raise ValueError(f"unsupported projection {self.projection.kind!r}")
        if self.vertical.kind != "hybrid_eta":
            raise ValueError(f"unsupported vertical coordinate {self.vertical.kind!r}")
        if not 1 <= int(self.halo_width) <= 4:
            raise ValueError("halo_width must be in [1, 4]")
        if self.staggering != "c-grid":
            raise ValueError("M3 only supports Arakawa C-grid staggering")
        if self.terrain.shape != (self.projection.ny, self.projection.nx):
            raise ValueError("terrain provenance shape must match projection dimensions")
        if tuple(self.terrain_height.shape) != self.terrain.shape:
            raise ValueError("terrain_height shape must match terrain provenance")
        if tuple(self.eta_levels.shape) != (self.vertical.nz + 1,):
            raise ValueError("eta_levels shape must be (nz + 1,)")
        if self.terrain_height.dtype != jnp.float64 or self.eta_levels.dtype != jnp.float64:
            raise TypeError("GridSpec arrays must be fp64")

    @property
    def nx(self) -> int:
        """Exposes nx for state allocation; avoids duplicating projection unpacking."""

        return self.projection.nx

    @property
    def ny(self) -> int:
        """Exposes ny for state allocation; avoids duplicating projection unpacking."""

        return self.projection.ny

    @property
    def nz(self) -> int:
        """Exposes nz for state allocation; avoids duplicating vertical unpacking."""

        return self.vertical.nz

    def tree_flatten(self):
        """Splits JAX array leaves from static hashable grid metadata."""

        children = (self.eta_levels, self.terrain_height)
        vertical_meta = (self.vertical.kind, self.vertical.nz, self.vertical.top_pressure_pa)
        aux = (
            self.projection,
            self.terrain,
            vertical_meta,
            self.bc,
            int(self.halo_width),
            self.staggering,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds GridSpec after JAX pytree transforms."""

        projection, terrain, vertical_meta, bc, halo_width, staggering = aux
        eta_levels, terrain_height = children
        vertical = VerticalCoord(*vertical_meta, eta_levels)
        return cls(
            projection=projection,
            terrain=terrain,
            vertical=vertical,
            bc=bc,
            eta_levels=eta_levels,
            terrain_height=terrain_height,
            halo_width=halo_width,
            staggering=staggering,
        )

    @staticmethod
    def _array_equal(left: jax.Array, right: jax.Array) -> bool:
        """Compares static grid arrays for cache-key equality outside timestep code."""

        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
        )

    @staticmethod
    def _array_hash(array: jax.Array) -> int:
        """Hashes small static grid arrays so equal GridSpecs share JIT cache keys."""

        host = np.asarray(array)
        return hash((tuple(host.shape), str(host.dtype), host.tobytes()))

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality required by JAX static-argument caching."""

        if not isinstance(other, GridSpec):
            return NotImplemented
        return (
            self.projection == other.projection
            and self.terrain == other.terrain
            and (self.vertical.kind, self.vertical.nz, self.vertical.top_pressure_pa)
            == (other.vertical.kind, other.vertical.nz, other.vertical.top_pressure_pa)
            and self.bc == other.bc
            and int(self.halo_width) == int(other.halo_width)
            and self.staggering == other.staggering
            and self._array_equal(self.eta_levels, other.eta_levels)
            and self._array_equal(self.terrain_height, other.terrain_height)
        )

    def __hash__(self) -> int:
        """Hashes static metadata and small grid arrays for static_argnames use."""

        return hash(
            (
                self.projection,
                self.terrain,
                (self.vertical.kind, self.vertical.nz, self.vertical.top_pressure_pa),
                self.bc,
                int(self.halo_width),
                self.staggering,
                self._array_hash(self.eta_levels),
                self._array_hash(self.terrain_height),
            )
        )

    @classmethod
    def canary_3km_template(cls) -> "GridSpec":
        """Constructs the canonical small Canary 3 km grid used by M3 tests and audits."""

        projection = Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, 8, 8)
        terrain = TerrainProvenance(
            source_path="data/static/canary_3km_terrain.nc",
            sha256="analytic-m3-template",
            shape=(projection.ny, projection.nx),
            units="m",
            projection_transform="native-lambert",
            max_elevation_m=3715.0,
            coastline_sanity_check_passed=True,
        )
        eta_levels = jnp.linspace(1.0, 0.0, 11, dtype=jnp.float64)
        vertical = VerticalCoord("hybrid_eta", 10, 5000.0, eta_levels)
        bc = BCMetadata(
            source="AIFS",
            fields=("u", "v", "T", "qv", "p_s"),
            update_cadence_h=6,
            interpolation="linear",
            restart_compatible=True,
        )
        terrain_height = jnp.zeros(terrain.shape, dtype=jnp.float64)
        return cls(projection, terrain, vertical, bc, eta_levels, terrain_height)
