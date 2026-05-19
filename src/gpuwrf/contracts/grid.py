"""Grid contract for the M3 GPU-resident state skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


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
        aux = (
            self.projection,
            self.terrain,
            self.vertical,
            self.bc,
            int(self.halo_width),
            self.staggering,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds GridSpec after JAX pytree transforms."""

        projection, terrain, vertical, bc, halo_width, staggering = aux
        eta_levels, terrain_height = children
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

    def __hash__(self) -> int:
        """Hashes static metadata and leaf shape/dtype, enough for static_argnames use."""

        return hash(
            (
                self.projection,
                self.terrain,
                self.vertical,
                self.bc,
                int(self.halo_width),
                self.staggering,
                tuple(self.eta_levels.shape),
                str(self.eta_levels.dtype),
                tuple(self.terrain_height.shape),
                str(self.terrain_height.dtype),
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
        vertical = VerticalCoord("hybrid_eta", 10, 5000.0)
        bc = BCMetadata(
            source="AIFS",
            fields=("u", "v", "T", "qv", "p_s"),
            update_cadence_h=6,
            interpolation="linear",
            restart_compatible=True,
        )
        eta_levels = jnp.linspace(1.0, 0.0, vertical.nz + 1, dtype=jnp.float64)
        terrain_height = jnp.zeros(terrain.shape, dtype=jnp.float64)
        return cls(projection, terrain, vertical, bc, eta_levels, terrain_height)
