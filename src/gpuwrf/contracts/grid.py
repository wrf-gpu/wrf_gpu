"""Grid contract for the GPU-resident dycore state skeleton."""

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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DycoreMetrics:
    """Static WRF dycore metric and vertical-coordinate arrays.

    Map factors, hybrid-eta coefficients, and terrain slopes live here, not in
    the timestep ``State`` carry. This follows the c2 architecture split:
    static grid data is device-resident, while prognostic fields remain
    separate SoA leaves. The ``cf*`` and ``fnm/fnp`` coefficients support
    WRF's non-hydrostatic PGF face-pressure construction.
    """

    msftx: jax.Array
    msfty: jax.Array
    msfux: jax.Array
    msfuy: jax.Array
    msfvx: jax.Array
    msfvy: jax.Array
    c1h: jax.Array
    c2h: jax.Array
    c3h: jax.Array
    c4h: jax.Array
    c1f: jax.Array
    c2f: jax.Array
    c3f: jax.Array
    c4f: jax.Array
    dn: jax.Array
    dnw: jax.Array
    rdn: jax.Array
    rdnw: jax.Array
    cf1: jax.Array
    cf2: jax.Array
    cf3: jax.Array
    fnm: jax.Array
    fnp: jax.Array
    dzdx: jax.Array
    dzdy: jax.Array
    dzdx_u: jax.Array
    dzdy_v: jax.Array
    p_top: jax.Array
    provenance: str = "analytic-flat"

    def __post_init__(self) -> None:
        """Normalizes scalar metadata and enforces fp64 metric storage."""

        for name in self._array_names():
            array = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            object.__setattr__(self, name, array)
            if array.dtype != jnp.float64:
                raise TypeError(f"DycoreMetrics.{name} must be fp64")
        if tuple(self.p_top.shape) not in ((), (1,)):
            raise ValueError("DycoreMetrics.p_top must be scalar or shape (1,)")

    @staticmethod
    def _array_names() -> tuple[str, ...]:
        """Returns metric array field names in pytree order."""

        return (
            "msftx",
            "msfty",
            "msfux",
            "msfuy",
            "msfvx",
            "msfvy",
            "c1h",
            "c2h",
            "c3h",
            "c4h",
            "c1f",
            "c2f",
            "c3f",
            "c4f",
            "dn",
            "dnw",
            "rdn",
            "rdnw",
            "cf1",
            "cf2",
            "cf3",
            "fnm",
            "fnp",
            "dzdx",
            "dzdy",
            "dzdx_u",
            "dzdy_v",
            "p_top",
        )

    @classmethod
    def flat(
        cls,
        *,
        ny: int,
        nx: int,
        nz: int,
        eta_levels: jax.Array,
        top_pressure_pa: float,
        provenance: str = "analytic-flat",
    ) -> "DycoreMetrics":
        """Builds a flat, unit-map-factor fixture for idealized tests."""

        if nz < 3:
            raise ValueError("DycoreMetrics.flat requires nz >= 3 for WRF cf coefficients")
        eta = jnp.asarray(eta_levels, dtype=jnp.float64)
        if tuple(eta.shape) != (nz + 1,):
            raise ValueError("eta_levels shape must be (nz + 1,)")
        eta_mass = 0.5 * (eta[:-1] + eta[1:])
        # WRF metric construction (module_initialize_ideal.F:711-727).  ``dnw`` is
        # the eta FACE spacing; ``dn`` is the MASS-LEVEL spacing dn(k)=0.5*(dnw(k)
        # +dnw(k-1)), distinct from ``dnw`` whenever eta is non-uniform.  The
        # earlier ``dn=dnw`` shortcut was only correct for uniform eta levels and
        # produced a singular calc_coef_w tridiagonal on hydrostatic eta grids.
        dnw = jnp.abs(eta[1:] - eta[:-1])  # (nz,) face spacing
        rdnw = 1.0 / dnw
        dn = jnp.ones((nz,), dtype=jnp.float64)
        dn = dn.at[1:].set(0.5 * (dnw[1:] + dnw[:-1]))  # dn[k]=0.5*(dnw[k]+dnw[k-1]), k=1..nz-1
        dn = dn.at[0].set(dnw[0])  # dn[0] is unused by WRF; set finite.
        rdn = 1.0 / dn
        fnm = jnp.zeros((nz,), dtype=jnp.float64)
        fnp = jnp.zeros((nz,), dtype=jnp.float64)
        fnp = fnp.at[1:].set(0.5 * dnw[1:] / dn[1:])
        fnm = fnm.at[1:].set(0.5 * dnw[:-1] / dn[1:])
        cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
        cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
        return cls(
            msftx=jnp.ones((ny, nx), dtype=jnp.float64),
            msfty=jnp.ones((ny, nx), dtype=jnp.float64),
            msfux=jnp.ones((ny, nx + 1), dtype=jnp.float64),
            msfuy=jnp.ones((ny, nx + 1), dtype=jnp.float64),
            msfvx=jnp.ones((ny + 1, nx), dtype=jnp.float64),
            msfvy=jnp.ones((ny + 1, nx), dtype=jnp.float64),
            c1h=eta_mass,
            c2h=jnp.zeros((nz,), dtype=jnp.float64),
            c3h=eta_mass,
            c4h=jnp.zeros((nz,), dtype=jnp.float64),
            c1f=eta,
            c2f=jnp.zeros((nz + 1,), dtype=jnp.float64),
            c3f=eta,
            c4f=jnp.zeros((nz + 1,), dtype=jnp.float64),
            dn=dn,
            dnw=dnw,
            rdn=rdn,
            rdnw=rdnw,
            cf1=fnp[1] + cof1,
            cf2=fnm[1] - cof1 - cof2,
            cf3=cof2,
            fnm=fnm,
            fnp=fnp,
            dzdx=jnp.zeros((ny, nx), dtype=jnp.float64),
            dzdy=jnp.zeros((ny, nx), dtype=jnp.float64),
            dzdx_u=jnp.zeros((ny, nx + 1), dtype=jnp.float64),
            dzdy_v=jnp.zeros((ny + 1, nx), dtype=jnp.float64),
            p_top=jnp.asarray(top_pressure_pa, dtype=jnp.float64),
            provenance=provenance,
        )

    def validate_shapes(self, *, ny: int, nx: int, nz: int) -> None:
        """Validates WRF staggering shapes against a GridSpec."""

        expected = {
            "msftx": (ny, nx),
            "msfty": (ny, nx),
            "msfux": (ny, nx + 1),
            "msfuy": (ny, nx + 1),
            "msfvx": (ny + 1, nx),
            "msfvy": (ny + 1, nx),
            "c1h": (nz,),
            "c2h": (nz,),
            "c3h": (nz,),
            "c4h": (nz,),
            "c1f": (nz + 1,),
            "c2f": (nz + 1,),
            "c3f": (nz + 1,),
            "c4f": (nz + 1,),
            "dn": (nz,),
            "dnw": (nz,),
            "rdn": (nz,),
            "rdnw": (nz,),
            "cf1": (),
            "cf2": (),
            "cf3": (),
            "fnm": (nz,),
            "fnp": (nz,),
            "dzdx": (ny, nx),
            "dzdy": (ny, nx),
            "dzdx_u": (ny, nx + 1),
            "dzdy_v": (ny + 1, nx),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"DycoreMetrics.{name} shape must be {shape}")

    def tree_flatten(self):
        """Splits metric arrays from static provenance metadata."""

        return tuple(getattr(self, name) for name in self._array_names()), self.provenance

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds DycoreMetrics after JAX pytree transforms."""

        values = dict(zip(cls._array_names(), children, strict=True))
        values["provenance"] = aux
        return cls(**values)

    @staticmethod
    def _array_equal(left: jax.Array, right: jax.Array) -> bool:
        """Compares metric arrays outside timestep code."""

        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
        )

    @staticmethod
    def _array_hash(array: jax.Array) -> int:
        """Hashes static metric arrays for existing static-grid call sites."""

        host = np.asarray(array)
        return hash((tuple(host.shape), str(host.dtype), host.tobytes()))

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality for tests and static cache keys."""

        if not isinstance(other, DycoreMetrics):
            return NotImplemented
        return self.provenance == other.provenance and all(
            self._array_equal(getattr(self, name), getattr(other, name))
            for name in self._array_names()
        )

    def __hash__(self) -> int:
        """Hashes metric provenance and arrays for current GridSpec static use."""

        return hash(
            (
                self.provenance,
                tuple(self._array_hash(getattr(self, name)) for name in self._array_names()),
            )
        )


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
    """JAX pytree grid contract with static WRF dycore metrics."""

    projection: Projection
    terrain: TerrainProvenance
    vertical: VerticalCoord
    bc: BCMetadata
    eta_levels: jax.Array
    terrain_height: jax.Array
    metrics: DycoreMetrics | None = None
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
        if self.metrics is None:
            object.__setattr__(
                self,
                "metrics",
                DycoreMetrics.flat(
                    ny=self.projection.ny,
                    nx=self.projection.nx,
                    nz=self.vertical.nz,
                    eta_levels=self.eta_levels,
                    top_pressure_pa=self.vertical.top_pressure_pa,
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
        assert self.metrics is not None
        self.metrics.validate_shapes(ny=self.projection.ny, nx=self.projection.nx, nz=self.vertical.nz)

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

        children = (self.eta_levels, self.terrain_height, self.metrics)
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
        eta_levels, terrain_height, metrics = children
        vertical = VerticalCoord(*vertical_meta, eta_levels)
        return cls(
            projection=projection,
            terrain=terrain,
            vertical=vertical,
            bc=bc,
            eta_levels=eta_levels,
            terrain_height=terrain_height,
            metrics=metrics,
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
            and self.metrics == other.metrics
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
                hash(self.metrics),
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
