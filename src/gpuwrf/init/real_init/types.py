"""FROZEN inter-lane handoff types for v0.4.0 native real-init.

This module is the interface contract the four parallel lanes (S1/S2/S3) and the
manager driver (S5) are all built against. It is FROZEN at the v0.4.0 S0 gate:
do NOT change a field, name, dtype, shape convention, or unit after the freeze
without a manager sign-off recorded in ``.agent/decisions/V0.4.0-S0-PLAN.md``.

Design notes
------------
* These are *plain dataclasses of numpy/jax arrays*, NOT the GPU-resident
  prognostic ``State``. real-init is an OFFLINE/setup-time computation; the
  driver (S5) assembles the final ``State``/``BaseState``/``DycoreMetrics``/
  ``BoundaryState`` from these handoff objects at the end. This keeps the heavy
  ``State`` contract (manager-owned shared core) out of the four lanes — they
  produce typed intermediate products, the driver does the single final pack.
* Shape conventions follow the WRF wrfinput layout the oracle uses:
  mass 3D = ``(nz, ny, nx)``; U = ``(nz, ny, nx+1)``; V = ``(nz, ny+1, nx)``;
  face/W = ``(nz+1, ny, nx)``; surface 2D = ``(ny, nx)``; soil = ``(nsoil, ny, nx)``.
  ``nz`` = number of mass (half) levels = ``kde-1``; full levels = ``nz+1``.
* Vertical index order is WRF model order: ``k=0`` is the LOWEST model level
  (surface), ``k=nz-1`` the top — i.e. eta DECREASES with k. (This is the
  opposite of the met_em ``num_metgrid_levels`` order; the vinterp lane handles
  the flip.)
* Units are SI / WRF-native (Pa, K, m, m s^-1, kg kg^-1, m2 s^-2). ``theta`` is
  WRF perturbation potential temperature ``T = theta_full - t0`` (t0=300 K), to
  match wrfout/wrfinput ``T``. Geopotential ``ph_*`` is m2 s^-2.

The frozen per-field tolerances vs the real.exe oracle live in
:data:`WRFINPUT_TOLS` and :data:`WRFBDY_TOLS`; the comparator (S4) reads them and
may NOT loosen them post-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


REAL_INIT_TYPES_VERSION = "0.4.0-S0-frozen-2026-06-02"

# WRF physical constants real.exe uses (module_model_constants), pinned here so
# every lane uses the identical values (a 0.01% constant drift poisons hour-0).
T0 = 300.0  # K, reference potential temperature (WRF t0)
G = 9.81  # m s^-2 (WRF g; NOTE WRF uses 9.81 exactly, not 9.80665)
R_D = 287.0  # J kg^-1 K^-1 (WRF r_d)
CP = 7.0 * R_D / 2.0  # 1004.5 J kg^-1 K^-1 (WRF cp)
CV = CP - R_D
RVOVRD = 461.6 / R_D  # Rv/Rd
P1000MB = 100000.0  # Pa
CVPM = -CV / CP
CPOVCV = CP / CV
RD_OVER_CP = R_D / CP

# Canary default base-state namelist constants (real.exe const_module_initialize
# pulls these from the namelist; confirmed defaults for this case in S0 recon).
# A lane reads these from RealInitConfig (below), NOT hard-coded, so a future
# namelist override is honored; these are the documented defaults.
DEFAULT_BASE_PRES = 100000.0  # p00 (Pa)
DEFAULT_BASE_TEMP = 290.0  # t00 (K)
DEFAULT_BASE_LAPSE = 50.0  # a / tlp (K per log-pressure)
DEFAULT_ISO_TEMP = 0.0  # tiso (K); 0 disables the iso cap unless set
DEFAULT_BASE_PRES_STRAT = 0.0  # p_strat (Pa)
DEFAULT_BASE_LAPSE_STRAT = 0.0  # a_strat


@dataclass(frozen=True)
class RealInitConfig:
    """Namelist-equivalent scalars real-init needs (frozen subset).

    Mirrors the ``time_control`` / ``domains`` / ``dynamics`` namelist fields
    that ``real.exe`` reads. The v0.4.0 supported subset only; unsupported
    options must be rejected loudly by the driver (P2-2 checker pattern), not
    silently defaulted.
    """

    # vertical coordinate
    nz: int  # number of mass levels (e_vert-1); Canary = 44
    p_top_pa: float  # grid%p_top (Pa); Canary = 5000.0
    hybrid_opt: int  # 0/1/2/3; Canary = 2 (Klemp polynomial); ONLY 0,1,2 supported v0.4.0
    etac: float  # hybrid transition eta (config_flags%etac); used when hybrid_opt>=2
    # eta generation: either explicit levels or auto. -1 sentinel => auto.
    eta_levels: tuple[float, ...] = ()  # explicit full-eta levels (len nz+1) or () for auto
    auto_levels_opt: int = 2
    max_dz: float = 1000.0
    dzbot: float = 50.0
    dzstretch_s: float = 1.3
    dzstretch_u: float = 1.1
    # base state
    base_pres: float = DEFAULT_BASE_PRES
    base_temp: float = DEFAULT_BASE_TEMP
    base_lapse: float = DEFAULT_BASE_LAPSE
    iso_temp: float = DEFAULT_ISO_TEMP
    base_pres_strat: float = DEFAULT_BASE_PRES_STRAT
    base_lapse_strat: float = DEFAULT_BASE_LAPSE_STRAT
    # soil / surface
    num_soil_layers: int = 4  # Noah-MP wrfinput soil_layers_stag; met_em provides 2 -> interp
    sf_surface_physics: int = 4  # 4 = Noah-MP (Canary); selects ZS/DZS layer depths
    interp_theta: bool = False  # real.exe config_flags%interp_theta (Canary default F)
    # LBC
    spec_bdy_width: int = 5  # boundary relaxation/spec zone width (Canary = 5)
    interval_seconds: int = 21600  # forcing interval; Canary AIFS = 6 h
    # provenance / identity
    grid_id: int = 1

    def __post_init__(self) -> None:
        if self.hybrid_opt not in (0, 1, 2):
            raise ValueError(
                f"v0.4.0 supports hybrid_opt in (0,1,2); got {self.hybrid_opt} "
                "(option 3 / sin^2 is tracked, not yet supported)"
            )
        if self.nz <= 1:
            raise ValueError("nz (mass levels) must be > 1")
        if self.p_top_pa <= 0:
            raise ValueError("p_top_pa must be positive")


@dataclass(frozen=True)
class VerticalCoord1D:
    """S1 output: the 1D vertical-coordinate + hybrid-coefficient arrays.

    These are the ``compute_eta`` + ``compute_vcoord_1d_coeffs`` products
    (module_initialize_real.F:1590 / nest_init_utils.F:1033). All in WRF order
    (znw[0]=1.0 at surface, znw[nz]=0.0 at top). Lengths: full-level arrays len
    ``nz+1``; half-level arrays len ``nz``.
    """

    znw: np.ndarray  # (nz+1,) full eta levels, 1.0 -> 0.0
    znu: np.ndarray  # (nz,)   half eta levels
    dnw: np.ndarray  # (nz,)   znw(k+1)-znw(k)
    rdnw: np.ndarray  # (nz,)  1/dnw
    dn: np.ndarray  # (nz,)    0.5*(dnw(k)+dnw(k-1))
    rdn: np.ndarray  # (nz,)   1/dn
    fnp: np.ndarray  # (nz,)   vertical interp weight +
    fnm: np.ndarray  # (nz,)   vertical interp weight -
    # hybrid coefficients (full f / half h); hybrid_opt=2 uses the Klemp poly
    c1f: np.ndarray  # (nz+1,)
    c2f: np.ndarray  # (nz+1,)
    c3f: np.ndarray  # (nz+1,)
    c4f: np.ndarray  # (nz+1,)
    c1h: np.ndarray  # (nz,)
    c2h: np.ndarray  # (nz,)
    c3h: np.ndarray  # (nz,)
    c4h: np.ndarray  # (nz,)
    cf1: float  # extrapolation coefs (real.exe cof/cf scalars)
    cf2: float
    cf3: float
    cfn: float
    cfn1: float
    p_top_pa: float


@dataclass(frozen=True)
class BaseStateColumns:
    """S1 output: terrain-aware dry base state (the ``setup_base_state`` block,
    module_initialize_real.F:3781-3835).

    pb/alb/t_init are 3D mass-level; mub is 2D; phb is 3D full-level. All in WRF
    model vertical order. These are exactly the wrfinput PB/MUB/PHB fields plus
    the t_init base potential temperature used to build alb.
    """

    pb: np.ndarray  # (nz, ny, nx)   base pressure (Pa)
    alb: np.ndarray  # (nz, ny, nx)  base inverse density (m3 kg-1)
    t_init: np.ndarray  # (nz, ny, nx) base perturbation theta (K, minus t0)
    mub: np.ndarray  # (ny, nx)      base dry-column mass (Pa)
    phb: np.ndarray  # (nz+1, ny, nx) base geopotential (m2 s-2)


@dataclass(frozen=True)
class DynamicsInit:
    """S1 output: the full prognostic dynamics initial state (wrfinput-equiv
    dynamics half), after vertical interp + hydrostatic balance
    (module_initialize_real.F:1450-2809 vinterp, :3876-4044 balance).

    All WRF model vertical order; perturbation quantities relative to
    BaseStateColumns. theta is WRF T (perturbation, minus t0).
    """

    u: np.ndarray  # (nz, ny, nx+1)   m s-1
    v: np.ndarray  # (nz, ny+1, nx)   m s-1
    w: np.ndarray  # (nz+1, ny, nx)   m s-1 (init 0)
    theta: np.ndarray  # (nz, ny, nx)  K perturbation (T)
    qv: np.ndarray  # (nz, ny, nx)     kg kg-1
    mu: np.ndarray  # (ny, nx)         perturbation dry-column mass (MU_2)
    mu0: np.ndarray  # (ny, nx)        full dry-column mass (MU0 = mub+mu)
    p: np.ndarray  # (nz, ny, nx)      perturbation pressure (Pa)
    ph: np.ndarray  # (nz+1, ny, nx)   perturbation geopotential (m2 s-2)
    al: np.ndarray  # (nz, ny, nx)     perturbation inverse density
    alt: np.ndarray  # (nz, ny, nx)    full inverse density
    p_hyd: np.ndarray  # (nz, ny, nx)  hydrostatic pressure (Pa)
    # hydrometeor / scalar moisture optionally interpolated (None if absent)
    qc: np.ndarray | None = None
    qr: np.ndarray | None = None
    qi: np.ndarray | None = None
    qs: np.ndarray | None = None
    qg: np.ndarray | None = None


@dataclass(frozen=True)
class SurfaceInit:
    """S2 output: surface/skin/SST/coordinate/map-metric initial fields
    (module_initialize_real.F surface block + carried met_em statics).

    These are the wrfinput 2D surface + map-factor + Coriolis fields. Many come
    straight from the met_em artifact (XLAT/XLONG/MAPFAC*/F/E/SINALPHA/COSALPHA/
    HGT/LANDMASK); SST/TSK/TMN get the real.exe water/skin protection logic
    (lines 2820-3340).
    """

    # coordinates / metric (mass + staggered) -- mostly passthrough from met_em
    xlat: np.ndarray  # (ny, nx)
    xlong: np.ndarray  # (ny, nx)
    xlat_u: np.ndarray  # (ny, nx+1)
    xlong_u: np.ndarray  # (ny, nx+1)
    xlat_v: np.ndarray  # (ny+1, nx)
    xlong_v: np.ndarray  # (ny+1, nx)
    mapfac_m: np.ndarray  # (ny, nx)
    mapfac_u: np.ndarray  # (ny, nx+1)
    mapfac_v: np.ndarray  # (ny+1, nx)
    mapfac_mx: np.ndarray  # (ny, nx)
    mapfac_my: np.ndarray  # (ny, nx)
    mapfac_ux: np.ndarray  # (ny, nx+1)
    mapfac_uy: np.ndarray  # (ny, nx+1)
    mapfac_vx: np.ndarray  # (ny+1, nx)
    mapfac_vy: np.ndarray  # (ny+1, nx)
    f: np.ndarray  # (ny, nx)   Coriolis f
    e: np.ndarray  # (ny, nx)   Coriolis e
    sinalpha: np.ndarray  # (ny, nx)
    cosalpha: np.ndarray  # (ny, nx)
    hgt: np.ndarray  # (ny, nx)  terrain height (m); the SAME ht S1 used for base state
    # surface state
    tsk: np.ndarray  # (ny, nx)  skin temp (K), water/SST-protected
    sst: np.ndarray  # (ny, nx)  sea-surface temp (K)
    tmn: np.ndarray  # (ny, nx)  deep soil / annual-mean temp (K)
    xland: np.ndarray  # (ny, nx) 1=land 2=water (WRF convention)
    landmask: np.ndarray  # (ny, nx) 1=land 0=water
    snowh: np.ndarray | None = None  # (ny, nx) snow depth (m)
    seaice: np.ndarray | None = None  # (ny, nx) fractional/0-1 sea ice


@dataclass(frozen=True)
class SoilInit:
    """S2 output: soil thermodynamic + categorical init (process_soil_real +
    process_percent_cat_new, module_initialize_real.F:3009-3530).

    Note: met_em provides 2 metgrid soil layers; Noah-MP needs ``num_soil_layers``
    (Canary = 4). The S2 soil lane reproduces real.exe's vertical soil
    interpolation (module_soil_pre) onto ZS/DZS. tslb/smois are WRF model soil
    order (layer 0 = top).
    """

    tslb: np.ndarray  # (nsoil, ny, nx)  soil temperature (K)
    smois: np.ndarray  # (nsoil, ny, nx) soil moisture (m3 m-3)
    sh2o: np.ndarray | None  # (nsoil, ny, nx) liquid soil moisture
    zs: np.ndarray  # (nsoil,)  soil layer node depths (m)
    dzs: np.ndarray  # (nsoil,) soil layer thicknesses (m)
    isltyp: np.ndarray  # (ny, nx) int dominant soil category
    ivgtyp: np.ndarray  # (ny, nx) int dominant veg/landuse category
    lu_index: np.ndarray  # (ny, nx) int land-use index
    vegfra: np.ndarray | None = None  # (ny, nx) green fraction (monthly-interp)
    canwat: np.ndarray | None = None  # (ny, nx) canopy water


@dataclass(frozen=True)
class LateralBC:
    """S3 output: wrfbdy-equivalent specified values + tendencies.

    Reproduces real_em.F::assemble_output (couple by (mu+mub)/msf, stuff_bdy,
    stuff_bdytend). One entry per coupled prognostic field. Arrays are stored in
    WRF wrfbdy layout per side: XS/XE indexed (j, k, bdy_width); YS/YE indexed
    (i, k, bdy_width). The first time level holds the specified VALUE at t0; the
    tendency arrays hold (coupled_{n+1}-coupled_n)/interval_seconds.

    Each field name maps to a dict of the 8 wrfbdy arrays: ``_bxs/_bxe/_bys/_bye``
    (the coupled value at the loop's start) and ``_btxs/_btxe/_btys/_btye`` (the
    tendency). Coupling stagger per field follows real.exe (U->msfuy, V->msfvx,
    T/QV/PH->msfty, MU 2D->msfty).
    """

    # values: name -> {"bxs","bxe","bys","bye"} each np.ndarray
    values: Mapping[str, Mapping[str, np.ndarray]]
    # tendencies: name -> {"btxs","btxe","btys","btye"} each np.ndarray
    tendencies: Mapping[str, Mapping[str, np.ndarray]]
    spec_bdy_width: int
    bdyfrq_seconds: float
    valid_times: tuple[str, ...]  # the forcing-interval valid times used
    coupled_field_names: tuple[str, ...] = (
        "u",
        "v",
        "t",  # theta
        "ph",  # geopotential (stuff_bdy 'W' stagger in real.exe)
        "qv",
        "mu",  # 2D
    )


@dataclass(frozen=True)
class RealInitProduct:
    """The full real-init result the driver (S5) packs into the runtime objects.

    This is the single object the driver consumes to build
    State/BaseState/DycoreMetrics/BoundaryState. It bundles every lane's typed
    output for one (domain, init_time). The comparator (S4) also consumes this
    to compare against the real.exe wrfinput/wrfbdy oracle.
    """

    domain: str
    init_time: str
    config: RealInitConfig
    vcoord: VerticalCoord1D
    base: BaseStateColumns
    dynamics: DynamicsInit
    surface: SurfaceInit
    soil: SoilInit
    lateral_bc: LateralBC | None  # None for the wrfinput-only (single-time) path
    provenance: dict[str, str] = field(default_factory=dict)


# --- PREDECLARED per-variable tolerances vs the real.exe oracle (S4-binding) --
# Absolute tolerance in the field's native unit on the RMSE-over-domain metric,
# plus a max-abs cap. These are the FROZEN wrfinput/wrfbdy parity gates. They are
# tighter for the dynamics fields that poison hour-0 (MU/PH/P/T/U/V) and looser
# for diagnostic/categorical fields. See V0.4.0-S0-PLAN.md section 5 for the
# rationale and the masking policy (land/water/below-ground). NO post-hoc
# loosening: changing any value requires a manager-signed note in the plan doc.
#   key -> (rmse_tol, maxabs_tol)  in field units
WRFINPUT_TOLS: dict[str, tuple[float, float]] = {
    # dynamics (hour-0-critical)
    "MU": (5.0, 50.0),  # Pa  (dry column mass perturbation)
    "MUB": (5.0, 50.0),  # Pa
    "PB": (5.0, 50.0),  # Pa
    "P": (20.0, 200.0),  # Pa  perturbation pressure
    "PH": (1.0, 20.0),  # m2 s-2  perturbation geopotential
    "PHB": (1.0, 20.0),  # m2 s-2
    "T": (0.10, 1.0),  # K   perturbation theta
    "U": (0.15, 1.5),  # m s-1
    "V": (0.15, 1.5),  # m s-1
    "W": (0.01, 0.1),  # m s-1 (init ~0)
    "QVAPOR": (1e-4, 1e-3),  # kg kg-1
    "AL": (1e-4, 1e-3),  # m3 kg-1  (al perturbation inverse density)
    "ALT": (1e-4, 1e-3),
    # vertical coordinate 1D (must be near-exact)
    "ZNW": (1e-6, 1e-5),
    "ZNU": (1e-6, 1e-5),
    "C1H": (1e-6, 1e-5),
    "C2H": (1.0, 10.0),  # Pa-scaled
    "C3H": (1e-6, 1e-5),
    "C4H": (1.0, 10.0),  # Pa-scaled
    "C1F": (1e-6, 1e-5),
    "C3F": (1e-6, 1e-5),
    "P_TOP": (1e-3, 1e-3),  # Pa
    # surface / metric (mostly passthrough; must match met_em-derived)
    "MAPFAC_M": (1e-5, 1e-4),
    "MAPFAC_U": (1e-5, 1e-4),
    "MAPFAC_V": (1e-5, 1e-4),
    "F": (1e-9, 1e-8),  # s-1
    "E": (1e-9, 1e-8),
    "SINALPHA": (1e-6, 1e-5),
    "COSALPHA": (1e-6, 1e-5),
    "XLAT": (1e-4, 1e-3),  # deg
    "XLONG": (1e-4, 1e-3),
    "HGT": (0.5, 5.0),  # m (terrain; matches met_em HGT_M tol)
    # surface state
    "TSK": (0.30, 3.0),  # K
    "SST": (0.30, 3.0),  # K
    "TMN": (0.50, 5.0),  # K
    "XLAND": (0.0, 0.0),  # categorical, exact
    # soil
    "TSLB": (0.50, 5.0),  # K  (looser: 2->4-layer interp adds spread)
    "SMOIS": (0.03, 0.30),  # m3 m-3
    "ZS": (1e-4, 1e-3),  # m
    "DZS": (1e-4, 1e-3),  # m
    "ISLTYP": (0.0, 0.0),  # categorical, exact
    "IVGTYP": (0.0, 0.0),  # categorical, exact
}

# wrfbdy tolerances: the bdy stores COUPLED ((mu+mub)*field/msf) quantities, so
# magnitudes are ~mu (~1e5 Pa) times the field; tolerances scale accordingly.
# The tendency arrays carry value/interval_seconds, so a 1e5-Pa coupling change
# over 21600 s gives O(1) tendency magnitudes; tendency tol is the value tol /
# interval_seconds, applied on the tendency RMSE.
WRFBDY_TOLS: dict[str, tuple[float, float]] = {
    # coupled VALUE tolerances (field_tol * ~mu_scale ~ 1e5)
    "U": (2.0e4, 2.0e5),  # coupled m s-1 * Pa
    "V": (2.0e4, 2.0e5),
    "T": (1.0e4, 1.0e5),  # coupled K * Pa
    "PH": (1.0e5, 1.0e6),  # coupled m2 s-2 * Pa
    "QV": (10.0, 100.0),  # coupled kg kg-1 * Pa
    "MU": (5.0, 50.0),  # the 2D mu boundary (Pa)
}
