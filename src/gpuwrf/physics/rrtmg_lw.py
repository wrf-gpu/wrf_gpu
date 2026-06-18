"""JAX longwave RRTMG-style radiation column kernel for M5-S3."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache, partial
from pathlib import Path
import re
from typing import NamedTuple

import jax
from jax import lax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.rrtmg_constants import (
    AVOGADRO,
    CH4_VMR,
    CO2_VMR,
    CP_AIR,
    DRY_AIR_MOLECULAR_WEIGHT,
    GRAVITY,
    LW_DIFFUSIVITY_A0,
    LW_DIFFUSIVITY_A1,
    LW_DIFFUSIVITY_A2,
    LW_BPADE,
    LW_EXP_EPS,
    LW_NTBL,
    LW_TBLINT,
    MAX_OPTICAL_DEPTH,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    N2O_VMR,
    O2_VMR,
    O3_BACKGROUND_VMR,
    STEFAN_BOLTZMANN,
    WATER_VAPOR_MOLECULAR_WEIGHT_RATIO,
)
from gpuwrf.physics.rrtmg_tables import RRTMGTableBundle, RRTMG_TABLES, TABLE_ASSET


config.update("jax_enable_x64", True)

O3_MMR_TO_VMR = 0.603461
_O3SUM = (
    5.297e-8,
    5.852e-8,
    6.579e-8,
    7.505e-8,
    8.577e-8,
    9.895e-8,
    1.175e-7,
    1.399e-7,
    1.677e-7,
    2.003e-7,
    2.571e-7,
    3.325e-7,
    4.438e-7,
    6.255e-7,
    8.168e-7,
    1.036e-6,
    1.366e-6,
    1.855e-6,
    2.514e-6,
    3.240e-6,
    4.033e-6,
    4.854e-6,
    5.517e-6,
    6.089e-6,
    6.689e-6,
    1.106e-5,
    1.462e-5,
    1.321e-5,
    9.856e-6,
    5.960e-6,
    5.960e-6,
)
_PPSUM = (
    955.890,
    850.532,
    754.599,
    667.742,
    589.841,
    519.421,
    455.480,
    398.085,
    347.171,
    301.735,
    261.310,
    225.360,
    193.419,
    165.490,
    141.032,
    120.125,
    102.689,
    87.829,
    75.123,
    64.306,
    55.086,
    47.209,
    40.535,
    34.795,
    29.865,
    19.122,
    9.277,
    4.660,
    2.421,
    1.294,
    0.647,
)
_O3WIN = (
    4.629e-8,
    4.686e-8,
    5.017e-8,
    5.613e-8,
    6.871e-8,
    8.751e-8,
    1.138e-7,
    1.516e-7,
    2.161e-7,
    3.264e-7,
    4.968e-7,
    7.338e-7,
    1.017e-6,
    1.308e-6,
    1.625e-6,
    2.011e-6,
    2.516e-6,
    3.130e-6,
    3.840e-6,
    4.703e-6,
    5.486e-6,
    6.289e-6,
    6.993e-6,
    7.494e-6,
    8.197e-6,
    9.632e-6,
    1.113e-5,
    1.146e-5,
    9.389e-6,
    6.135e-6,
    6.135e-6,
)
_PPWIN = (
    955.747,
    841.783,
    740.199,
    649.538,
    568.404,
    495.815,
    431.069,
    373.464,
    322.354,
    277.190,
    237.635,
    203.433,
    174.070,
    148.949,
    127.408,
    108.915,
    93.114,
    79.551,
    67.940,
    58.072,
    49.593,
    42.318,
    36.138,
    30.907,
    26.362,
    16.423,
    7.583,
    3.620,
    1.807,
    0.938,
    0.469,
)
_LW_BUFFER_PPROF = (
    1000.00,
    855.47,
    731.82,
    626.05,
    535.57,
    458.16,
    391.94,
    335.29,
    286.83,
    245.38,
    209.91,
    179.57,
    153.62,
    131.41,
    112.42,
    96.17,
    82.27,
    70.38,
    60.21,
    51.51,
    44.06,
    37.69,
    32.25,
    27.59,
    23.60,
    20.19,
    17.27,
    14.77,
    12.64,
    10.81,
    9.25,
    7.91,
    6.77,
    5.79,
    4.95,
    4.24,
    3.63,
    3.10,
    2.65,
    2.27,
    1.94,
    1.66,
    1.42,
    1.22,
    1.04,
    0.89,
    0.76,
    0.65,
    0.56,
    0.48,
    0.41,
    0.35,
    0.30,
    0.26,
    0.22,
    0.19,
    0.16,
    0.14,
    0.12,
    0.10,
)
_LW_BUFFER_TPROF = (
    286.96,
    281.07,
    275.16,
    268.11,
    260.56,
    253.02,
    245.62,
    238.41,
    231.57,
    225.91,
    221.72,
    217.79,
    215.06,
    212.74,
    210.25,
    210.16,
    210.69,
    212.14,
    213.74,
    215.37,
    216.82,
    217.94,
    219.03,
    220.18,
    221.37,
    222.64,
    224.16,
    225.88,
    227.63,
    229.51,
    231.50,
    233.73,
    236.18,
    238.78,
    241.60,
    244.44,
    247.35,
    250.33,
    253.32,
    256.30,
    259.22,
    262.12,
    264.80,
    266.50,
    267.59,
    268.44,
    268.69,
    267.76,
    266.13,
    263.96,
    261.54,
    258.93,
    256.15,
    253.23,
    249.89,
    246.67,
    243.48,
    240.25,
    236.66,
    233.86,
)


@jax.tree_util.register_pytree_node_class
class RRTMGLWColumnState:
    """Pytree for independent longwave radiation columns on mass levels."""

    __slots__ = ("T", "p", "qv", "qc", "qi", "qs", "qg", "cloud_fraction", "surface_temperature", "surface_emissivity", "dz", "rho")

    def __init__(self, T, p, qv, qc, qi, qs, qg, cloud_fraction, surface_temperature, surface_emissivity, dz, rho) -> None:
        self.T = T
        self.p = p
        self.qv = qv
        self.qc = qc
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.cloud_fraction = cloud_fraction
        self.surface_temperature = surface_temperature
        self.surface_emissivity = surface_emissivity
        self.dz = dz
        self.rho = rho

    def replace(self, **updates) -> "RRTMGLWColumnState":
        """Returns a same-layout state with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all state arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after JAX transforms."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for tests."""

        if not isinstance(other, RRTMGLWColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


class RRTMGLWColumnResult(NamedTuple):
    """Longwave column outputs with bottom-to-top interface fluxes."""

    heating_rate: jnp.ndarray
    flux_down: jnp.ndarray
    flux_up: jnp.ndarray
    toa_down: jnp.ndarray
    toa_up: jnp.ndarray
    surface_down: jnp.ndarray
    surface_up: jnp.ndarray
    column_net_heating: jnp.ndarray
    surface_emission: jnp.ndarray
    # Clear-sky (cloud-free) interface fluxes, WRF `totdclfl`/`totuclfl` from the
    # parallel clear-sky stream in `rtrnmc` (`module_ra_rrtmg_lw.F` :3417-3489).
    # Populated only when ``solve_rrtmg_lw_column(..., with_clear_sky=True)``;
    # otherwise ``None`` (the main all-sky flux outputs are byte-identical with or
    # without the clear-sky pass).  WRF `...C` vars: LWUPTC=clear_flux_up[...,-1],
    # LWDNTC=clear_flux_down[...,-1], LWUPBC=clear_flux_up[...,0],
    # LWDNBC=clear_flux_down[...,0].
    clear_flux_down: jnp.ndarray | None = None
    clear_flux_up: jnp.ndarray | None = None


class RRTMGLWIntermediateState(NamedTuple):
    """LW solver-entry state exposed for M5-S3.z WRF intermediate-oracle checks."""

    tau: jnp.ndarray
    fracs: jnp.ndarray
    secdiff: jnp.ndarray
    planklay: jnp.ndarray
    planklev: jnp.ndarray
    plankbnd: jnp.ndarray
    dplankup: jnp.ndarray
    dplankdn: jnp.ndarray
    cldprmc_cldfmc: jnp.ndarray
    cldprmc_taucmc: jnp.ndarray
    rtrnmc_pfracs: jnp.ndarray
    rtrnmc_plansum: jnp.ndarray
    rtrnmc_tfn_tbl_output: jnp.ndarray
    rtrnmc_zfd_per_gpoint: jnp.ndarray
    rtrnmc_zfu_per_gpoint: jnp.ndarray


class _LWSetCoefState(NamedTuple):
    """WRF `setcoef` state needed by LW `taumol` branches."""

    jp: jnp.ndarray
    jt: jnp.ndarray
    jt1: jnp.ndarray
    lower_mask: jnp.ndarray
    pavel: jnp.ndarray
    coldry: jnp.ndarray
    wx: jnp.ndarray
    colh2o: jnp.ndarray
    colco2: jnp.ndarray
    colo3: jnp.ndarray
    coln2o: jnp.ndarray
    colco: jnp.ndarray
    colch4: jnp.ndarray
    colo2: jnp.ndarray
    colbrd: jnp.ndarray
    fac00: jnp.ndarray
    fac01: jnp.ndarray
    fac10: jnp.ndarray
    fac11: jnp.ndarray
    rat_h2oco2: jnp.ndarray
    rat_h2oco2_1: jnp.ndarray
    rat_h2oo3: jnp.ndarray
    rat_h2oo3_1: jnp.ndarray
    rat_h2on2o: jnp.ndarray
    rat_h2on2o_1: jnp.ndarray
    rat_h2och4: jnp.ndarray
    rat_h2och4_1: jnp.ndarray
    rat_n2oco2: jnp.ndarray
    rat_n2oco2_1: jnp.ndarray
    rat_o3co2: jnp.ndarray
    rat_o3co2_1: jnp.ndarray
    selffac: jnp.ndarray
    selffrac: jnp.ndarray
    indself: jnp.ndarray
    forfac: jnp.ndarray
    forfrac: jnp.ndarray
    indfor: jnp.ndarray
    minorfrac: jnp.ndarray
    scaleminor: jnp.ndarray
    scaleminorn2: jnp.ndarray
    indminor: jnp.ndarray


class _LWNTableBundle(NamedTuple):
    """Native reduced LW k/continuum/fraction tables parsed from WRF records."""

    nspa: jnp.ndarray
    nspb: jnp.ndarray
    absa: jnp.ndarray
    absb: jnp.ndarray
    selfref: jnp.ndarray
    forref: jnp.ndarray
    fracrefa: jnp.ndarray
    fracrefb: jnp.ndarray
    chi_mls: jnp.ndarray
    ka_mn2: jnp.ndarray
    kb_mn2: jnp.ndarray
    ka_mn2_r: jnp.ndarray
    ka_mn2o: jnp.ndarray
    kb_mn2o: jnp.ndarray
    ka_mn2o_r: jnp.ndarray
    kb_mn2o_r: jnp.ndarray
    ka_mco2: jnp.ndarray
    kb_mco2: jnp.ndarray
    ka_mco2_r: jnp.ndarray
    ka_mo3: jnp.ndarray
    kb_mo3: jnp.ndarray
    ka_mo3_r: jnp.ndarray
    ka_mco_r: jnp.ndarray
    ka_mo2: jnp.ndarray
    kb_mo2: jnp.ndarray
    ccl4: jnp.ndarray
    cfc11adj: jnp.ndarray
    cfc12: jnp.ndarray
    cfc22adj: jnp.ndarray


class _LWCloudTableBundle(NamedTuple):
    """WRF LW cloud absorption tables used by `cldprmc`."""

    liquid: jnp.ndarray
    ice: jnp.ndarray
    snow: jnp.ndarray


_LW_GPOINT_COUNTS = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
_LW_NSPA = np.asarray([1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9], dtype=np.int32)
_LW_NSPB = np.asarray([1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.int32)
_CFC_VMR = np.asarray([0.093e-9, 0.251e-9, 0.538e-9, 0.169e-9], dtype=np.float64)
_ONEMINUS = 1.0 - 1.0e-6
_LW_TAUMOL_BRANCH_ACCEPTED = (True,) * 16

# Number of LW spectral bands.  The `rtrnmc` band loop builds a per-band
# (..., nlay+1, 16) flux buffer; stacking all 16 into a (ncol, nlay+1, 16, 16)
# array before `sum(axis=(-1, -2))` would force every band's buffer to stay
# live.  Because the surface flux is an associative sum over (band, g-point) and
# bands never couple in `rtrnmc`, the production path accumulates the band-summed
# flux INCREMENTALLY (in fp64) so the full stacked array is never materialised,
# and the oracle entry (`compute_rrtmg_lw_intermediates`) still requests the full
# per-g-point arrays for WRF intermediate-boundary parity.
#
# MEASURED CAVEAT (proofs/v013): unlike the SW two-stream, this is VRAM-NEUTRAL
# at the measured grids — the LW peak floor is the UPSTREAM `_lw_solver_base`
# arrays (taumol branch+fallback `tau`/`fracs` and the planck tables, each
# ~(ncol, nlay, 16, 16) fp64), NOT the flux stack.  XLA's liveness analysis
# already scheduled the per-band buffers without a stack penalty, so removing the
# explicit stack is numerically inert + HLO-cleaner but does not lower peak here.
# Reducing the LW peak further requires chunking `_lw_solver_base` (taumol), out
# of scope for this g-point-flux sprint.
_LW_NBANDS = 16
# Band-tile width for the production flux accumulation.  1 = one band per flush
# (smallest per-tile buffer); set to `_LW_NBANDS` for a single-stack flush.  The
# numerical result is independent of this value (fp64 sum of disjoint per-band
# contributions; see proofs/v013).
_LW_GPOINT_CHUNK_BANDS = 1
# Taumol/optics construction chunking.  When True (default) the OPERATIONAL
# flux path (`_lw_solver_fluxes`) builds each band's `taumol` `(..., nlay, 16)`
# `tau`/`frac` lazily inside a band-axis `lax.scan` (the gas chemistry differs
# per band, so a `lax.switch` over the traced band index selects the per-band
# `_lw_taumol_band`).  The scan carry forces XLA to free each band's taumol +
# rtrnmc working set before the next band, so the full `(..., nlay, 16, 16)`
# `tau`/`fracs` stack (the dominant remaining LW fp64 VRAM floor, plus its dead
# fallback duplicate) is NEVER materialised.  Bit-identical to the upfront-stack
# path in fp64 (disjoint per-band g-point sums accumulated in fp64; proofs/v013).
# Set False to fall back to the upfront-stack flux path (oracle path always uses
# the full stack regardless of this flag).
_LW_TAUMOL_CHUNK = True


def _env_int(name: str, default: int) -> int:
    """Reads integer tuning knobs without making import-time env errors fatal."""

    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Reads boolean tuning knobs without making import-time env errors fatal."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# Column tiling over the leading horizontal/batch axes.  v0.13 band/taumol
# chunking removed per-spectral-stack blowups, but the public LW solve still had
# an `ncol`-wide transient floor when invoked on a whole 1 km nest.  The
# production entry flattens arbitrary leading dimensions, scans over fixed-size
# column tiles, and reshapes outputs back.  The 2048 default keeps the measured
# AC1_FIT d03 LW transient near 1.2 GiB while preserving one-tile execution for
# small domains; env overrides remain authoritative.  Set
# `_LW_COLUMN_TILING=False` or `_LW_COLUMN_TILE_COLS=0` for the exact
# whole-column reference path.
_LW_COLUMN_TILING = _env_bool("GPUWRF_RRTMG_LW_COLUMN_TILING", True)
_LW_COLUMN_TILE_COLS = max(0, _env_int("GPUWRF_RRTMG_LW_COLUMN_TILE_COLS", 2048))


def _leaves(state: RRTMGLWColumnState):
    """Centralizes leaf iteration for equality and hashing."""

    return (getattr(state, name) for name in RRTMGLWColumnState.__slots__)


def _trunc_int(value):
    """Fortran-style positive-range integer truncation."""

    return value.astype(jnp.int32)


def _take_rows(table, idx):
    """Gathers flattened WRF coefficient rows for every layer."""

    clipped = jnp.clip(idx, 0, table.shape[0] - 1)
    return jnp.take(table, clipped, axis=0)


@lru_cache(maxsize=1)
def _extract_rrtmg_tables_module():
    """Loads the repository-local table extractor without relying on package path."""

    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "extract_rrtmg_tables.py"
    spec = importlib.util.spec_from_file_location("_gpuwrf_extract_rrtmg_tables", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RRTMG table extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_chi_mls(lw_source) -> np.ndarray:
    """Parses WRF `chi_mls(7,59)` reference mixing-ratio table."""

    text = lw_source.read_text(encoding="utf-8", errors="replace")
    chi = np.zeros((7, 59), dtype=np.float64)
    pattern = re.compile(r"chi_mls\((\d+),\s*(\d+):(\d+)\)\s*=\s*\(/\s*&(?P<body>.*?)\s*/\)", re.DOTALL)
    extractor = _extract_rrtmg_tables_module()

    for match in pattern.finditer(text):
        gas = int(match.group(1)) - 1
        start = int(match.group(2)) - 1
        end = int(match.group(3))
        values = extractor._parse_source_block_numbers(match.group("body"))
        if values.size != end - start:
            raise ValueError(f"bad chi_mls slice size for gas {gas + 1}: {values.size}")
        chi[gas, start:end] = values
    if not np.all(chi > 0.0):
        raise ValueError("failed to parse complete WRF chi_mls table")
    return chi


def _read_lw_records_from_asset() -> list[bytes]:
    """Reconstructs WRF LW DATA records from the existing table fixture."""

    with np.load(TABLE_ASSET, allow_pickle=False) as loaded:
        raw = np.asarray(loaded["lw_raw_payload_bytes"], dtype=np.uint8)
        offsets = np.asarray(loaded["lw_record_offsets"], dtype=np.uint64)
        lengths = np.asarray(loaded["lw_record_lengths"], dtype=np.uint32)
    records: list[bytes] = []
    for offset, length in zip(offsets, lengths, strict=True):
        start = int(offset)
        end = start + int(length)
        records.append(raw[start:end].tobytes())
    return records


@lru_cache(maxsize=1)
def _native_lw_tables() -> _LWNTableBundle:
    """Builds reduced WRF LW `taumol` tables from the pinned raw payload."""

    extractor = _extract_rrtmg_tables_module()
    LW_READ_SPECS = extractor.LW_READ_SPECS
    LW_REDUCED_GROUPS = extractor.LW_REDUCED_GROUPS
    LW_SOURCE = extractor.LW_SOURCE
    ORIGINAL_GPOINT_WEIGHTS = extractor.ORIGINAL_GPOINT_WEIGHTS
    parse_record = extractor._parse_record

    records = _read_lw_records_from_asset()
    max_g = max(_LW_GPOINT_COUNTS)
    max_absa = 9 * 5 * 13
    max_absb = 5 * 5 * 47
    absa = np.zeros((16, max_absa, max_g), dtype=np.float64)
    absb = np.zeros((16, max_absb, max_g), dtype=np.float64)
    selfref = np.zeros((16, 10, max_g), dtype=np.float64)
    forref = np.zeros((16, 4, max_g), dtype=np.float64)
    fracrefa = np.zeros((16, max_g, 9), dtype=np.float64)
    fracrefb = np.zeros((16, max_g, 5), dtype=np.float64)

    minor2 = {
        name: np.zeros((16, 19, max_g), dtype=np.float64)
        for name in ("ka_mn2", "kb_mn2", "ka_mn2o", "kb_mn2o", "ka_mco2", "kb_mco2", "ka_mo3", "kb_mo3", "ka_mo2", "kb_mo2")
    }
    minor9 = {name: np.zeros((16, 9, 19, max_g), dtype=np.float64) for name in ("ka_mn2", "ka_mn2o", "ka_mco2", "ka_mo3", "ka_mco")}
    minor5 = {name: np.zeros((16, 5, 19, max_g), dtype=np.float64) for name in ("kb_mn2o",)}
    vector = {name: np.zeros((16, max_g), dtype=np.float64) for name in ("ccl4", "cfc11adj", "cfc12", "cfc22adj")}

    def reduce_gpoints(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
        values32 = np.asarray(values, dtype=np.float32)
        reduced = []
        start = 0
        for group_size in groups:
            end = start + group_size
            segment = values32[..., start:end]
            total = np.zeros(segment.shape[:-1], dtype=np.float32)
            if weighted:
                weights = np.asarray(ORIGINAL_GPOINT_WEIGHTS[start:end], dtype=np.float32)
                wtsum = np.float32(0.0)
                for weight in weights:
                    wtsum = np.float32(wtsum + weight)
                for local_idx, weight in enumerate(weights):
                    total = np.float32(total + segment[..., local_idx] * np.float32(weight / wtsum))
            else:
                for local_idx in range(group_size):
                    total = np.float32(total + segment[..., local_idx])
            reduced.append(total.astype(np.float64))
            start = end
        if start != 16:
            raise ValueError(f"g-point grouping consumed {start} original points, expected 16")
        return np.stack(reduced, axis=-1)

    def reduce_gpoints_first(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
        moved = np.moveaxis(values, 0, -1)
        reduced = reduce_gpoints(moved, groups, weighted=weighted)
        return np.moveaxis(reduced, -1, 0)

    def store_minor(parsed, source_name: str, band_index: int, groups, target_name: str) -> None:
        if source_name not in parsed:
            return
        reduced = reduce_gpoints(parsed[source_name], groups, weighted=True)
        ng = len(groups)
        if reduced.ndim == 2:
            minor2[target_name][band_index, : reduced.shape[0], :ng] = reduced
        elif reduced.shape[0] == 9:
            minor9[target_name][band_index, : reduced.shape[0], : reduced.shape[1], :ng] = reduced
        elif reduced.shape[0] == 5:
            minor5[target_name][band_index, : reduced.shape[0], : reduced.shape[1], :ng] = reduced
        else:
            raise ValueError(f"unsupported LW minor table {source_name} shape {reduced.shape}")

    for band_index, (record, (_band, spec), groups) in enumerate(zip(records, LW_READ_SPECS, LW_REDUCED_GROUPS, strict=True)):
        parsed = parse_record(record, spec)
        ng = len(groups)
        if "kao" in parsed:
            kao = parsed["kao"]
            if kao.ndim == 3:
                kao = kao[None, :, :, :]
            kao_red = reduce_gpoints(kao, groups, weighted=True)
            flat = kao_red.reshape((kao_red.shape[0] * kao_red.shape[1] * kao_red.shape[2], ng), order="F")
            absa[band_index, : flat.shape[0], :ng] = flat
        if "kbo" in parsed:
            kbo = parsed["kbo"]
            if kbo.ndim == 3:
                kbo = kbo[None, :, :, :]
            kbo_red = reduce_gpoints(kbo, groups, weighted=True)
            flat = kbo_red.reshape((kbo_red.shape[0] * kbo_red.shape[1] * kbo_red.shape[2], ng), order="F")
            absb[band_index, : flat.shape[0], :ng] = flat
        if "selfrefo" in parsed:
            reduced = reduce_gpoints(parsed["selfrefo"], groups, weighted=True)
            selfref[band_index, : reduced.shape[0], :ng] = reduced
        if "forrefo" in parsed:
            reduced = reduce_gpoints(parsed["forrefo"], groups, weighted=True)
            forref[band_index, : reduced.shape[0], :ng] = reduced
        if "fracrefao" in parsed:
            raw = parsed["fracrefao"]
            reduced = reduce_gpoints(raw, groups, weighted=False) if raw.ndim == 1 else reduce_gpoints_first(raw, groups, weighted=False)
            if reduced.ndim == 1:
                fracrefa[band_index, :ng, 0] = reduced
            else:
                fracrefa[band_index, :ng, : reduced.shape[1]] = reduced
        if "fracrefbo" in parsed:
            raw = parsed["fracrefbo"]
            reduced = reduce_gpoints(raw, groups, weighted=False) if raw.ndim == 1 else reduce_gpoints_first(raw, groups, weighted=False)
            if reduced.ndim == 1:
                fracrefb[band_index, :ng, 0] = reduced
            else:
                fracrefb[band_index, :ng, : reduced.shape[1]] = reduced

        store_minor(parsed, "kao_mn2", band_index, groups, "ka_mn2")
        store_minor(parsed, "kbo_mn2", band_index, groups, "kb_mn2")
        store_minor(parsed, "kao_mn2o", band_index, groups, "ka_mn2o")
        store_minor(parsed, "kbo_mn2o", band_index, groups, "kb_mn2o")
        store_minor(parsed, "kao_mco2", band_index, groups, "ka_mco2")
        store_minor(parsed, "kbo_mco2", band_index, groups, "kb_mco2")
        store_minor(parsed, "kao_mo3", band_index, groups, "ka_mo3")
        store_minor(parsed, "kbo_mo3", band_index, groups, "kb_mo3")
        store_minor(parsed, "kao_mco", band_index, groups, "ka_mco")
        store_minor(parsed, "kao_mo2", band_index, groups, "ka_mo2")
        store_minor(parsed, "kbo_mo2", band_index, groups, "kb_mo2")

        for source_name, target_name in (("ccl4o", "ccl4"), ("cfc11adjo", "cfc11adj"), ("cfc12o", "cfc12"), ("cfc22adjo", "cfc22adj")):
            if source_name in parsed:
                vector[target_name][band_index, :ng] = reduce_gpoints(parsed[source_name], groups, weighted=True)

    return _LWNTableBundle(
        nspa=np.asarray(_LW_NSPA, dtype=np.int32),
        nspb=np.asarray(_LW_NSPB, dtype=np.int32),
        absa=np.asarray(absa, dtype=np.float64),
        absb=np.asarray(absb, dtype=np.float64),
        selfref=np.asarray(selfref, dtype=np.float64),
        forref=np.asarray(forref, dtype=np.float64),
        fracrefa=np.asarray(fracrefa, dtype=np.float64),
        fracrefb=np.asarray(fracrefb, dtype=np.float64),
        chi_mls=np.asarray(_parse_chi_mls(LW_SOURCE), dtype=np.float64),
        ka_mn2=np.asarray(minor2["ka_mn2"], dtype=np.float64),
        kb_mn2=np.asarray(minor2["kb_mn2"], dtype=np.float64),
        ka_mn2_r=np.asarray(minor9["ka_mn2"], dtype=np.float64),
        ka_mn2o=np.asarray(minor2["ka_mn2o"], dtype=np.float64),
        kb_mn2o=np.asarray(minor2["kb_mn2o"], dtype=np.float64),
        ka_mn2o_r=np.asarray(minor9["ka_mn2o"], dtype=np.float64),
        kb_mn2o_r=np.asarray(minor5["kb_mn2o"], dtype=np.float64),
        ka_mco2=np.asarray(minor2["ka_mco2"], dtype=np.float64),
        kb_mco2=np.asarray(minor2["kb_mco2"], dtype=np.float64),
        ka_mco2_r=np.asarray(minor9["ka_mco2"], dtype=np.float64),
        ka_mo3=np.asarray(minor2["ka_mo3"], dtype=np.float64),
        kb_mo3=np.asarray(minor2["kb_mo3"], dtype=np.float64),
        ka_mo3_r=np.asarray(minor9["ka_mo3"], dtype=np.float64),
        ka_mco_r=np.asarray(minor9["ka_mco"], dtype=np.float64),
        ka_mo2=np.asarray(minor2["ka_mo2"], dtype=np.float64),
        kb_mo2=np.asarray(minor2["kb_mo2"], dtype=np.float64),
        ccl4=np.asarray(vector["ccl4"], dtype=np.float64),
        cfc11adj=np.asarray(vector["cfc11adj"], dtype=np.float64),
        cfc12=np.asarray(vector["cfc12"], dtype=np.float64),
        cfc22adj=np.asarray(vector["cfc22adj"], dtype=np.float64),
    )


@lru_cache(maxsize=1)
def _native_lw_cloud_tables() -> _LWCloudTableBundle:
    """Builds WRF `cldprmc` absorption coefficients for the fixture radii.

    The harness uses WRF's default effective radii for this sprint fixture:
    liquid 10 um, ice 30 um, and snow 75 um.  The interpolation mirrors
    `cldprmc` at module_ra_rrtmg_lw.F:2972-3018.
    """

    extractor = _extract_rrtmg_tables_module()
    source = extractor.LW_SOURCE
    liquid_grid = np.arange(2.5, 60.5, dtype=np.float64)
    ice_grid = 2.0 + 3.0 * np.arange(1, 47, dtype=np.float64)
    liquid = []
    ice = []
    snow = []
    for band in range(1, 17):
        liquid.append(extractor._interp_table(extractor._parse_source_array(source, "absliq1", band), liquid_grid, 10.0))
        ice.append(extractor._interp_table(extractor._parse_source_array(source, "absice3", band), ice_grid, 30.0))
        snow.append(extractor._interp_table(extractor._parse_source_array(source, "absice3", band), ice_grid, 75.0))
    return _LWCloudTableBundle(
        liquid=np.asarray(liquid, dtype=np.float64),
        ice=np.asarray(ice, dtype=np.float64),
        snow=np.asarray(snow, dtype=np.float64),
    )


def _clip_state(state: RRTMGLWColumnState) -> RRTMGLWColumnState:
    """Applies radiation-safe physical bounds before optical calculations."""

    return state.replace(
        T=jnp.maximum(state.T, 120.0),
        p=jnp.maximum(state.p, 1.0),
        qv=jnp.maximum(state.qv, 0.0),
        qc=jnp.maximum(state.qc, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        cloud_fraction=jnp.clip(state.cloud_fraction, 0.0, 1.0),
        surface_temperature=jnp.maximum(state.surface_temperature, 120.0),
        surface_emissivity=jnp.clip(state.surface_emissivity, 0.0, 1.0),
        dz=jnp.maximum(state.dz, 1.0),
        rho=jnp.maximum(state.rho, MIN_LAYER_MASS),
    )


def _column_count(leading_shape: tuple[int, ...]) -> int:
    """Returns the static number of flattened columns for a leading shape."""

    return int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1


def _flatten_layer_field(arr, leading_shape: tuple[int, ...], ncol: int):
    """Flattens arbitrary leading axes of a column-layer field to `(ncol, nz)`."""

    arr = jnp.asarray(arr)
    return jnp.reshape(arr, (ncol,) + arr.shape[len(leading_shape):])


def _flatten_surface_field(arr, leading_shape: tuple[int, ...], ncol: int):
    """Flattens a surface/column field when it carries the state leading axes."""

    arr = jnp.asarray(arr)
    if arr.shape == ():
        return jnp.reshape(arr, (ncol,)) if not leading_shape else arr
    if leading_shape and arr.shape[: len(leading_shape)] == leading_shape:
        return jnp.reshape(arr, (ncol,) + arr.shape[len(leading_shape):])
    return arr


def _pad_leading_columns(arr, ncol: int, padded_ncol: int):
    """Pads a flattened leading column axis by repeating the last real column."""

    arr = jnp.asarray(arr)
    if arr.shape == () or arr.shape[0] != ncol or padded_ncol == ncol:
        return arr
    pad_cols = padded_ncol - ncol
    tail = jnp.broadcast_to(arr[-1:], (pad_cols,) + arr.shape[1:])
    return jnp.concatenate((arr, tail), axis=0)


def _slice_leading_columns(arr, start, tile_cols: int, padded_ncol: int):
    """Slices one fixed-size tile from arrays that carry the padded column axis."""

    arr = jnp.asarray(arr)
    if arr.shape == () or arr.shape[0] != padded_ncol:
        return arr
    zero = jnp.zeros((), dtype=start.dtype)
    starts = [zero] * arr.ndim
    starts[0] = start
    sizes = list(arr.shape)
    sizes[0] = tile_cols
    return lax.dynamic_slice(arr, starts, sizes)


def _flatten_lw_state(state: RRTMGLWColumnState, leading_shape: tuple[int, ...], ncol: int) -> RRTMGLWColumnState:
    """Flattens a LW state from `leading_shape + (nz,)` to `(ncol, nz)`."""

    return state.replace(
        T=_flatten_layer_field(state.T, leading_shape, ncol),
        p=_flatten_layer_field(state.p, leading_shape, ncol),
        qv=_flatten_layer_field(state.qv, leading_shape, ncol),
        qc=_flatten_layer_field(state.qc, leading_shape, ncol),
        qi=_flatten_layer_field(state.qi, leading_shape, ncol),
        qs=_flatten_layer_field(state.qs, leading_shape, ncol),
        qg=_flatten_layer_field(state.qg, leading_shape, ncol),
        cloud_fraction=_flatten_layer_field(state.cloud_fraction, leading_shape, ncol),
        dz=_flatten_layer_field(state.dz, leading_shape, ncol),
        rho=_flatten_layer_field(state.rho, leading_shape, ncol),
        surface_temperature=_flatten_surface_field(state.surface_temperature, leading_shape, ncol),
        surface_emissivity=_flatten_surface_field(state.surface_emissivity, leading_shape, ncol),
    )


def _pad_lw_state(state: RRTMGLWColumnState, ncol: int, padded_ncol: int) -> RRTMGLWColumnState:
    """Pads all flattened LW state leaves that carry the leading column axis."""

    return state.replace(
        T=_pad_leading_columns(state.T, ncol, padded_ncol),
        p=_pad_leading_columns(state.p, ncol, padded_ncol),
        qv=_pad_leading_columns(state.qv, ncol, padded_ncol),
        qc=_pad_leading_columns(state.qc, ncol, padded_ncol),
        qi=_pad_leading_columns(state.qi, ncol, padded_ncol),
        qs=_pad_leading_columns(state.qs, ncol, padded_ncol),
        qg=_pad_leading_columns(state.qg, ncol, padded_ncol),
        cloud_fraction=_pad_leading_columns(state.cloud_fraction, ncol, padded_ncol),
        surface_temperature=_pad_leading_columns(state.surface_temperature, ncol, padded_ncol),
        surface_emissivity=_pad_leading_columns(state.surface_emissivity, ncol, padded_ncol),
        dz=_pad_leading_columns(state.dz, ncol, padded_ncol),
        rho=_pad_leading_columns(state.rho, ncol, padded_ncol),
    )


def _slice_lw_state(state: RRTMGLWColumnState, start, tile_cols: int, padded_ncol: int) -> RRTMGLWColumnState:
    """Slices a fixed-size LW state tile from a padded flattened state."""

    return state.replace(
        T=_slice_leading_columns(state.T, start, tile_cols, padded_ncol),
        p=_slice_leading_columns(state.p, start, tile_cols, padded_ncol),
        qv=_slice_leading_columns(state.qv, start, tile_cols, padded_ncol),
        qc=_slice_leading_columns(state.qc, start, tile_cols, padded_ncol),
        qi=_slice_leading_columns(state.qi, start, tile_cols, padded_ncol),
        qs=_slice_leading_columns(state.qs, start, tile_cols, padded_ncol),
        qg=_slice_leading_columns(state.qg, start, tile_cols, padded_ncol),
        cloud_fraction=_slice_leading_columns(state.cloud_fraction, start, tile_cols, padded_ncol),
        surface_temperature=_slice_leading_columns(state.surface_temperature, start, tile_cols, padded_ncol),
        surface_emissivity=_slice_leading_columns(state.surface_emissivity, start, tile_cols, padded_ncol),
        dz=_slice_leading_columns(state.dz, start, tile_cols, padded_ncol),
        rho=_slice_leading_columns(state.rho, start, tile_cols, padded_ncol),
    )


def _zero_lw_column_result(total_cols: int, nlayers: int, dtype, with_clear_sky: bool) -> RRTMGLWColumnResult:
    """Allocates the fixed-shape scan carry for LW tiled column outputs."""

    layer = jnp.zeros((total_cols, nlayers), dtype=dtype)
    interface = jnp.zeros((total_cols, nlayers + 2), dtype=dtype)
    column = jnp.zeros((total_cols,), dtype=dtype)
    clear_down = jnp.zeros_like(interface) if with_clear_sky else None
    clear_up = jnp.zeros_like(interface) if with_clear_sky else None
    return RRTMGLWColumnResult(
        heating_rate=layer,
        flux_down=interface,
        flux_up=interface,
        toa_down=column,
        toa_up=column,
        surface_down=column,
        surface_up=column,
        column_net_heating=column,
        surface_emission=column,
        clear_flux_down=clear_down,
        clear_flux_up=clear_up,
    )


def _update_tile(carry, tile, start):
    """Scatters one fixed-size tile into a padded scan-carry field."""

    zero = jnp.zeros((), dtype=start.dtype)
    starts = [zero] * carry.ndim
    starts[0] = start
    return lax.dynamic_update_slice(carry, tile, starts)


def _scatter_lw_result(carry: RRTMGLWColumnResult, tile: RRTMGLWColumnResult, start) -> RRTMGLWColumnResult:
    """Scatters one LW tile result into the full padded result carry."""

    clear_down = None if carry.clear_flux_down is None else _update_tile(carry.clear_flux_down, tile.clear_flux_down, start)
    clear_up = None if carry.clear_flux_up is None else _update_tile(carry.clear_flux_up, tile.clear_flux_up, start)
    return RRTMGLWColumnResult(
        heating_rate=_update_tile(carry.heating_rate, tile.heating_rate, start),
        flux_down=_update_tile(carry.flux_down, tile.flux_down, start),
        flux_up=_update_tile(carry.flux_up, tile.flux_up, start),
        toa_down=_update_tile(carry.toa_down, tile.toa_down, start),
        toa_up=_update_tile(carry.toa_up, tile.toa_up, start),
        surface_down=_update_tile(carry.surface_down, tile.surface_down, start),
        surface_up=_update_tile(carry.surface_up, tile.surface_up, start),
        column_net_heating=_update_tile(carry.column_net_heating, tile.column_net_heating, start),
        surface_emission=_update_tile(carry.surface_emission, tile.surface_emission, start),
        clear_flux_down=clear_down,
        clear_flux_up=clear_up,
    )


def _unflatten_lw_result(result: RRTMGLWColumnResult, leading_shape: tuple[int, ...], ncol: int) -> RRTMGLWColumnResult:
    """Restores LW result fields from flat columns to the caller's leading shape."""

    def restore(arr):
        if arr is None:
            return None
        arr = arr[:ncol]
        return jnp.reshape(arr, leading_shape + arr.shape[1:])

    return RRTMGLWColumnResult(
        heating_rate=restore(result.heating_rate),
        flux_down=restore(result.flux_down),
        flux_up=restore(result.flux_up),
        toa_down=restore(result.toa_down),
        toa_up=restore(result.toa_up),
        surface_down=restore(result.surface_down),
        surface_up=restore(result.surface_up),
        column_net_heating=restore(result.column_net_heating),
        surface_emission=restore(result.surface_emission),
        clear_flux_down=restore(result.clear_flux_down),
        clear_flux_up=restore(result.clear_flux_up),
    )


def _pressure_interfaces(p):
    """Reconstructs WRF harness pressure interfaces from midpoint pressures."""

    nz = p.shape[-1]
    dp_bottom = jnp.maximum(10.0, p[..., 0] - p[..., 1])
    bottom = p[..., :1] + 0.5 * dp_bottom[..., None]
    middle = 0.5 * (p[..., :-1] + p[..., 1:])
    dp_top = jnp.maximum(10.0, p[..., -2] - p[..., -1])
    top = jnp.maximum(400.0, p[..., -1:] - 0.5 * dp_top[..., None])
    return jnp.concatenate((bottom, middle, top), axis=-1)


def _temperature_interfaces(T):
    """Reconstructs WRF harness interface temperatures from midpoint temperatures."""

    bottom = T[..., :1]
    middle = 0.5 * (T[..., :-1] + T[..., 1:])
    top = T[..., -1:]
    return jnp.concatenate((bottom, middle, top), axis=-1)


def _lw_buffer_temperatures(t_layer, t_level, pressure_interfaces_pa):
    """Ports WRF LW wrapper top-buffer temperature adjustment."""

    original_layers = t_layer.shape[-1] - 1
    dtype = t_layer.dtype
    pprof = jnp.asarray(_LW_BUFFER_PPROF, dtype=dtype)
    tprof = jnp.asarray(_LW_BUFFER_TPROF, dtype=dtype)
    plev = (pressure_interfaces_pa * 0.01).at[..., -1].set(0.0)
    below = pprof < plev[..., None]
    has_interval = jnp.any(below, axis=-1)
    first_below = jnp.argmax(below, axis=-1)
    k0 = jnp.where(has_interval, jnp.maximum(first_below - 1, 0), pprof.shape[0] - 1)
    k1 = jnp.minimum(k0 + 1, pprof.shape[0] - 1)
    p0 = jnp.take(pprof, k0, axis=0)
    p1 = jnp.take(pprof, k1, axis=0)
    t0 = jnp.take(tprof, k0, axis=0)
    t1 = jnp.take(tprof, k1, axis=0)
    weight = jnp.where(k0 == k1, 0.0, (plev - p0) / (p1 - p0))
    varint = weight * (t1 - t0) + t0

    offset = t_level[..., original_layers - 1] - varint[..., original_layers - 1]
    adjusted_top_levels = varint[..., original_layers:] + offset[..., None]
    t_level_lw = jnp.concatenate((t_level[..., :original_layers], adjusted_top_levels), axis=-1)
    adjusted_top_layers = 0.5 * (t_level_lw[..., original_layers - 1 : -1] + t_level_lw[..., original_layers:])
    t_layer_lw = jnp.concatenate((t_layer[..., : original_layers - 1], adjusted_top_layers), axis=-1)
    return t_layer_lw, t_level_lw


def _pressure_layer_mass(p):
    """Reconstructs WRF harness layer mass from midpoint pressure interfaces."""

    nz = p.shape[-1]
    interfaces = _pressure_interfaces(p)
    return jnp.maximum((interfaces[..., :nz] - interfaces[..., 1 : nz + 1]) / GRAVITY, MIN_LAYER_MASS)


def _nearest_pressure_coefficients(state_p, tables: RRTMGTableBundle):
    """Selects WRF reference-pressure absorption coefficients per layer."""

    ref_log = jnp.log(tables.lw_reference_pressure_pa)
    layer_log = jnp.log(jnp.maximum(state_p, 1.0))
    idx = jnp.argmin(jnp.abs(layer_log[..., None] - ref_log), axis=-1)
    gathered = jnp.take(tables.lw_absorption_coefficients, idx, axis=1)
    return jnp.moveaxis(gathered, 0, -2)


def _rrtmg_column_amounts(qv, pressure_interfaces):
    """Computes RRTMG-style scaled molecular columns and precipitable water."""

    h2ovmr = qv * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * DRY_AIR_MOLECULAR_WEIGHT + h2ovmr * 18.0160
    dp_mb = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) * 0.01, 1.0e-8)
    coldry = dp_mb * 1.0e3 * AVOGADRO / (1.0e2 * GRAVITY * amm * (1.0 + h2ovmr))
    colh2o = 1.0e-20 * coldry * h2ovmr
    colco2 = 1.0e-20 * coldry * CO2_VMR
    colo3 = 1.0e-20 * coldry * O3_BACKGROUND_VMR
    coln2o = 1.0e-20 * coldry * N2O_VMR
    colch4 = 1.0e-20 * coldry * CH4_VMR
    colo2 = 1.0e-20 * coldry * O2_VMR
    absorber = colh2o + 0.03 * colco2 + 0.05 * colo3 + 0.02 * coln2o + 0.02 * colch4 + 0.0001 * colo2
    dry_plus_water = coldry + coldry * h2ovmr
    pwvcm = jnp.sum(18.0160 * coldry * h2ovmr, axis=-1) / jnp.maximum(DRY_AIR_MOLECULAR_WEIGHT * jnp.sum(dry_plus_water, axis=-1), 1.0e-12)
    pwvcm = pwvcm * (1.0e3 * pressure_interfaces[..., 0] * 0.01) / (1.0e2 * GRAVITY)
    return absorber, pwvcm


def _lw_o3_profile_vmr(pressure_interfaces_pa):
    """Ports WRF `INIRAD/O3DATA` annual ozone profile integration for LW."""

    plev = pressure_interfaces_pa * 0.01
    dtype = plev.dtype
    o3sum = jnp.asarray(_O3SUM, dtype=dtype)
    ppsum = jnp.asarray(_PPSUM, dtype=dtype)
    o3win = jnp.asarray(_O3WIN, dtype=dtype)
    ppwin = jnp.asarray(_PPWIN, dtype=dtype)

    o3ann_tail = o3win[:-1] + (o3win[1:] - o3win[:-1]) / (ppwin[1:] - ppwin[:-1]) * (ppsum[1:] - ppwin[:-1])
    o3ann = jnp.concatenate((0.5 * (o3sum[:1] + o3win[:1]), 0.5 * (o3ann_tail + o3sum[1:])))
    ppwrkh = jnp.concatenate((jnp.asarray([1100.0], dtype=dtype), 0.5 * (ppsum[1:] + ppsum[:-1]), jnp.asarray([0.0], dtype=dtype)))

    bottom = plev[..., :-1][..., None]
    top = plev[..., 1:][..., None]
    pp_bottom = ppwrkh[:-1]
    pp_top = ppwrkh[1:]
    pb1 = jnp.maximum(bottom - pp_bottom, 0.0)
    pb2 = jnp.maximum(bottom - pp_top, 0.0)
    pt1 = jnp.maximum(top - pp_bottom, 0.0)
    pt2 = jnp.maximum(top - pp_top, 0.0)
    integrated = jnp.sum((pb2 - pb1 - pt2 + pt1) * o3ann, axis=-1)
    dp = jnp.maximum(plev[..., :-1] - plev[..., 1:], 1.0e-12)
    return (integrated / dp) * jnp.asarray(O3_MMR_TO_VMR, dtype=dtype)


def _lw_diffusivity(pwvcm):
    """Returns RRTMG LW band diffusivity secants."""

    a0 = jnp.asarray(LW_DIFFUSIVITY_A0, dtype=jnp.float64)
    a1 = jnp.asarray(LW_DIFFUSIVITY_A1, dtype=jnp.float64)
    a2 = jnp.asarray(LW_DIFFUSIVITY_A2, dtype=jnp.float64)
    secdiff = a0 + a1 * jnp.exp(a2 * pwvcm[..., None])
    variable = jnp.asarray([False, True, True, False, True, True, True, True, True, False, False, False, False, False, False, False])
    return jnp.where(variable, jnp.clip(secdiff, 1.50, 1.80), 1.66)


def _interp_lw_planck(values, tables: RRTMGTableBundle):
    """Interpolates WRF `totplnk` integrated Planck tables for all LW bands."""

    bounded = jnp.clip(values, 160.0, 340.0)
    ind = jnp.clip((bounded - 159.0).astype(jnp.int32), 1, 180)
    frac = bounded - 159.0 - ind.astype(jnp.float64)
    low = jnp.take(tables.lw_totplnk, ind - 1, axis=0)
    high = jnp.take(tables.lw_totplnk, ind, axis=0)
    return low + frac[..., None] * (high - low)


def _lw_planck_state(t_layer, t_level, t_surface, emissivity, tables: RRTMGTableBundle):
    """Ports WRF `setcoef` Planck interpolation for all-band LW calls."""

    planklay = _interp_lw_planck(t_layer, tables)
    planklev = _interp_lw_planck(t_level, tables)
    plankbnd = _interp_lw_planck(t_surface, tables) * emissivity[..., None]
    return planklay, planklev, plankbnd


def _lw_setcoef(qv, p_pa, t_k, pressure_interfaces_pa, tables: RRTMGTableBundle) -> _LWSetCoefState:
    """Ports WRF LW `setcoef` gas-column and interpolation state."""

    nt = _native_lw_tables()
    h2ovmr = jnp.maximum(qv, 1.0e-12) * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * DRY_AIR_MOLECULAR_WEIGHT + h2ovmr * 18.0160
    pavel = jnp.maximum(p_pa * 0.01, 1.0e-12)
    pz = pressure_interfaces_pa * 0.01
    dp_mb = jnp.maximum(pz[..., :-1] - pz[..., 1:], 1.0e-12)
    coldry = dp_mb * 1.0e3 * AVOGADRO / (1.0e2 * GRAVITY * amm * (1.0 + h2ovmr))
    o3_vmr = _lw_o3_profile_vmr(pressure_interfaces_pa)

    plog = jnp.log(pavel)
    jp = _trunc_int(36.0 - 5.0 * (plog + 0.04))
    jp = jnp.clip(jp, 1, 58)
    jp1 = jp + 1
    fp = 5.0 * (jnp.take(tables.lw_preflog, jp - 1, axis=0) - plog)
    tref0 = jnp.take(tables.lw_tref, jp - 1, axis=0)
    tref1 = jnp.take(tables.lw_tref, jp1 - 1, axis=0)
    jt = jnp.clip(_trunc_int(3.0 + (t_k - tref0) / 15.0), 1, 4)
    jt1 = jnp.clip(_trunc_int(3.0 + (t_k - tref1) / 15.0), 1, 4)
    ft = ((t_k - tref0) / 15.0) - (jt - 3).astype(jnp.float64)
    ft1 = ((t_k - tref1) / 15.0) - (jt1 - 3).astype(jnp.float64)

    wkl_h2o = coldry * h2ovmr
    wbroad = coldry * jnp.maximum(0.0, 1.0 - (CO2_VMR + o3_vmr + N2O_VMR + CH4_VMR + O2_VMR))
    water = wkl_h2o / jnp.maximum(coldry, 1.0e-300)
    scalefac = pavel * (296.0 / 1013.0) / t_k
    lower = plog > 4.56

    forfac = scalefac / (1.0 + water)
    lower_for_factor = (332.0 - t_k) / 36.0
    upper_for_factor = (t_k - 188.0) / 36.0
    indfor_lower = jnp.minimum(2, jnp.maximum(1, _trunc_int(lower_for_factor)))
    indfor = jnp.where(lower, indfor_lower, 3)
    forfrac = jnp.where(lower, lower_for_factor - indfor.astype(jnp.float64), upper_for_factor - 1.0)

    selffac = water * forfac
    self_factor = (t_k - 188.0) / 7.2
    indself = jnp.minimum(9, jnp.maximum(1, _trunc_int(self_factor) - 7))
    selffrac = self_factor - (indself + 7).astype(jnp.float64)

    scaleminor = pavel / t_k
    scaleminorn2 = scaleminor * (wbroad / jnp.maximum(coldry + wkl_h2o, 1.0e-300))
    minor_factor = (t_k - 180.8) / 7.2
    indminor = jnp.minimum(18, jnp.maximum(1, _trunc_int(minor_factor)))
    minorfrac = minor_factor - indminor.astype(jnp.float64)

    chi = nt.chi_mls
    j0 = jp - 1
    j1 = jp

    def ratio(a_1b, b_1b, idx):
        return jnp.take(chi[a_1b - 1], idx, axis=0) / jnp.take(chi[b_1b - 1], idx, axis=0)

    colh2o = 1.0e-20 * wkl_h2o
    colco2 = 1.0e-20 * coldry * CO2_VMR
    colo3 = 1.0e-20 * coldry * o3_vmr
    coln2o = 1.0e-20 * coldry * N2O_VMR
    colco = 1.0e-32 * coldry
    colch4 = 1.0e-20 * coldry * CH4_VMR
    colo2 = 1.0e-20 * coldry * O2_VMR
    colbrd = 1.0e-20 * wbroad
    wx = coldry[..., None] * jnp.asarray(_CFC_VMR, dtype=jnp.float64) * 1.0e-20

    compfp = 1.0 - fp
    fac10 = compfp * ft
    fac00 = compfp * (1.0 - ft)
    fac11 = fp * ft1
    fac01 = fp * (1.0 - ft1)

    return _LWSetCoefState(
        jp=jp,
        jt=jt,
        jt1=jt1,
        lower_mask=lower,
        pavel=pavel,
        coldry=coldry,
        wx=wx,
        colh2o=colh2o,
        colco2=colco2,
        colo3=colo3,
        coln2o=coln2o,
        colco=colco,
        colch4=colch4,
        colo2=colo2,
        colbrd=colbrd,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        rat_h2oco2=ratio(1, 2, j0),
        rat_h2oco2_1=ratio(1, 2, j1),
        rat_h2oo3=ratio(1, 3, j0),
        rat_h2oo3_1=ratio(1, 3, j1),
        rat_h2on2o=ratio(1, 4, j0),
        rat_h2on2o_1=ratio(1, 4, j1),
        rat_h2och4=ratio(1, 6, j0),
        rat_h2och4_1=ratio(1, 6, j1),
        rat_n2oco2=ratio(4, 2, j0),
        rat_n2oco2_1=ratio(4, 2, j1),
        rat_o3co2=ratio(3, 2, j0),
        rat_o3co2_1=ratio(3, 2, j1),
        selffac=colh2o * selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=colh2o * forfac,
        forfrac=forfrac,
        indfor=indfor,
        minorfrac=minorfrac,
        scaleminor=scaleminor,
        scaleminorn2=scaleminorn2,
        indminor=indminor,
    )


def _interp_four_rows_lw(table, idx0_1b, idx1_1b, stride_1b, coef: _LWSetCoefState):
    """WRF four-corner pressure/temperature interpolation for LW tables."""

    idx0 = idx0_1b - 1
    idx1 = idx1_1b - 1
    stride = jnp.maximum(stride_1b, 1)
    return (
        coef.fac00[..., None] * _take_rows(table, idx0)
        + coef.fac10[..., None] * _take_rows(table, idx0 + stride)
        + coef.fac01[..., None] * _take_rows(table, idx1)
        + coef.fac11[..., None] * _take_rows(table, idx1 + stride)
    )


def _continuum_lw(band: int, coef: _LWSetCoefState, nt: _LWNTableBundle):
    """LW H2O self/foreign continuum contribution below the tropopause."""

    inds = coef.indself - 1
    indf = coef.indfor - 1
    self_table = nt.selfref[band].astype(coef.selffac.dtype)
    for_table = nt.forref[band].astype(coef.forfac.dtype)
    self_base = _take_rows(self_table, inds)
    for_base = _take_rows(for_table, indf)
    tauself = coef.selffac[..., None] * (self_base + coef.selffrac[..., None] * (_take_rows(self_table, inds + 1) - self_base))
    taufor = coef.forfac[..., None] * (for_base + coef.forfrac[..., None] * (_take_rows(for_table, indf + 1) - for_base))
    return tauself + taufor


def _foreign_lw(band: int, coef: _LWSetCoefState, nt: _LWNTableBundle):
    """LW H2O foreign continuum contribution above the tropopause."""

    indf = coef.indfor - 1
    table = nt.forref[band].astype(coef.forfac.dtype)
    base = _take_rows(table, indf)
    return coef.forfac[..., None] * (base + coef.forfrac[..., None] * (_take_rows(table, indf + 1) - base))


def _binary_params(spec_a, spec_b, ratio, multiplier):
    """Builds WRF binary-species interpolation coordinates."""

    speccomb = spec_a + ratio * spec_b
    specparm = jnp.minimum(spec_a / jnp.maximum(speccomb, 1.0e-300), _ONEMINUS)
    specmult = multiplier * specparm
    js = 1 + _trunc_int(specmult)
    fs = jnp.mod(specmult, 1.0)
    return speccomb, specparm, js, fs


def _lw_coef_as_dtype(coef: _LWSetCoefState, dtype) -> _LWSetCoefState:
    """Casts floating LW coefficient fields while preserving indices and masks."""

    return _LWSetCoefState(
        *(
            value.astype(dtype)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
            else value
            for value in coef
        )
    )


def _binary_lower_component(table, idx_1b, specparm, fs, fac0, fac1):
    """One WRF lower-atmosphere binary corner with edge interpolation."""

    idx = idx_1b - 1
    p_low = fs - 1.0
    p4_low = p_low**4
    fk0_low = p4_low
    fk1_low = 1.0 - p_low - 2.0 * p4_low
    fk2_low = p_low + p4_low

    p_high = -fs
    p4_high = p_high**4
    fk0_high = p4_high
    fk1_high = 1.0 - p_high - 2.0 * p4_high
    fk2_high = p_high + p4_high

    low_edge = (
        (fk0_low * fac0)[..., None] * _take_rows(table, idx)
        + (fk1_low * fac0)[..., None] * _take_rows(table, idx + 1)
        + (fk2_low * fac0)[..., None] * _take_rows(table, idx + 2)
        + (fk0_low * fac1)[..., None] * _take_rows(table, idx + 9)
        + (fk1_low * fac1)[..., None] * _take_rows(table, idx + 10)
        + (fk2_low * fac1)[..., None] * _take_rows(table, idx + 11)
    )
    high_edge = (
        (fk2_high * fac0)[..., None] * _take_rows(table, idx - 1)
        + (fk1_high * fac0)[..., None] * _take_rows(table, idx)
        + (fk0_high * fac0)[..., None] * _take_rows(table, idx + 1)
        + (fk2_high * fac1)[..., None] * _take_rows(table, idx + 8)
        + (fk1_high * fac1)[..., None] * _take_rows(table, idx + 9)
        + (fk0_high * fac1)[..., None] * _take_rows(table, idx + 10)
    )
    middle = (
        ((1.0 - fs) * fac0)[..., None] * _take_rows(table, idx)
        + (fs * fac0)[..., None] * _take_rows(table, idx + 1)
        + ((1.0 - fs) * fac1)[..., None] * _take_rows(table, idx + 9)
        + (fs * fac1)[..., None] * _take_rows(table, idx + 10)
    )
    return jnp.where(specparm[..., None] < 0.125, low_edge, jnp.where(specparm[..., None] > 0.875, high_edge, middle))


def _major_binary_lower(table, idx0_1b, idx1_1b, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef: _LWSetCoefState):
    """WRF lower-atmosphere binary major-species interpolation."""

    tau0 = _binary_lower_component(table, idx0_1b, specparm, fs, coef.fac00, coef.fac10)
    tau1 = _binary_lower_component(table, idx1_1b, specparm1, fs1, coef.fac01, coef.fac11)
    return speccomb[..., None] * tau0 + speccomb1[..., None] * tau1


def _major_binary_upper(table, idx0_1b, idx1_1b, speccomb, fs, speccomb1, fs1, coef: _LWSetCoefState):
    """WRF upper-atmosphere binary major-species interpolation."""

    idx0 = idx0_1b - 1
    idx1 = idx1_1b - 1
    return (
        speccomb[..., None]
        * (
            ((1.0 - fs) * coef.fac00)[..., None] * _take_rows(table, idx0)
            + (fs * coef.fac00)[..., None] * _take_rows(table, idx0 + 1)
            + ((1.0 - fs) * coef.fac10)[..., None] * _take_rows(table, idx0 + 5)
            + (fs * coef.fac10)[..., None] * _take_rows(table, idx0 + 6)
        )
        + speccomb1[..., None]
        * (
            ((1.0 - fs1) * coef.fac01)[..., None] * _take_rows(table, idx1)
            + (fs1 * coef.fac01)[..., None] * _take_rows(table, idx1 + 1)
            + ((1.0 - fs1) * coef.fac11)[..., None] * _take_rows(table, idx1 + 5)
            + (fs1 * coef.fac11)[..., None] * _take_rows(table, idx1 + 6)
        )
    )


def _minor2(table, coef: _LWSetCoefState):
    """Interpolates a WRF minor table indexed only by minor temperature."""

    indm = coef.indminor - 1
    base = _take_rows(table, indm)
    return base + coef.minorfrac[..., None] * (_take_rows(table, indm + 1) - base)


def _take_species_minor(table, js, indm):
    """Gathers WRF minor tables with species-ratio and minor-temperature axes."""

    species = jnp.take(table, jnp.clip(js - 1, 0, table.shape[0] - 1), axis=0)
    idx = jnp.clip(indm - 1, 0, table.shape[1] - 1)[..., None, None]
    return jnp.take_along_axis(species, idx, axis=-2)[..., 0, :]


def _minor_ratio(table, js, fs, coef: _LWSetCoefState):
    """Interpolates WRF minor tables over species ratio and minor temperature."""

    indm = coef.indminor
    m11 = _take_species_minor(table, js, indm)
    m21 = _take_species_minor(table, js + 1, indm)
    m12 = _take_species_minor(table, js, indm + 1)
    m22 = _take_species_minor(table, js + 1, indm + 1)
    low = m11 + fs[..., None] * (m21 - m11)
    high = m12 + fs[..., None] * (m22 - m12)
    return low + coef.minorfrac[..., None] * (high - low)


def _adj_minor_column(column, coef: _LWSetCoefState, chi_ref, threshold, base, exponent):
    """WRF empirical column adjustment used for abundant nominal minor species."""

    ratio = 1.0e20 * (column / jnp.maximum(coef.coldry, 1.0e-300)) / chi_ref
    adjusted = (base + (ratio - base) ** exponent) * chi_ref * coef.coldry * 1.0e-20
    return jnp.where(ratio > threshold, adjusted, column)


def _frac_const(frac_table, coef: _LWSetCoefState):
    """Broadcasts a WRF constant Planck-fraction table."""

    return jnp.broadcast_to(frac_table[:, 0], coef.pavel.shape + frac_table[:, 0].shape)


def _frac_interp(frac_table, jpl, fpl):
    """Interpolates WRF Planck fractions over binary-species ratio."""

    table = jnp.swapaxes(frac_table, 0, 1)
    base = _take_rows(table, jpl - 1)
    return base + fpl[..., None] * (_take_rows(table, jpl) - base)


def _binary_band(
    band: int,
    coef: _LWSetCoefState,
    nt: _LWNTableBundle,
    lower_a,
    lower_b,
    lower_ratio,
    upper_a,
    upper_b,
    upper_ratio,
    frac_lower_ratio,
    frac_upper_ratio=None,
):
    """Shared WRF binary major-species skeleton for LW bands without minors."""

    lower_ratio0, lower_ratio1 = lower_ratio if isinstance(lower_ratio, tuple) else (lower_ratio, lower_ratio)
    upper_ratio0, upper_ratio1 = upper_ratio if isinstance(upper_ratio, tuple) else (upper_ratio, upper_ratio)
    nspa = nt.nspa[band]
    nspb = nt.nspb[band]
    lower_idx0 = ((coef.jp - 1) * 5 + (coef.jt - 1)) * nspa + 1
    lower_idx1 = (coef.jp * 5 + (coef.jt1 - 1)) * nspa + 1
    speccomb, specparm, js, fs = _binary_params(lower_a, lower_b, lower_ratio0, 8.0)
    speccomb1, specparm1, js1, fs1 = _binary_params(lower_a, lower_b, lower_ratio1, 8.0)
    low = _major_binary_lower(nt.absa[band], lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
    low = low + _continuum_lw(band, coef, nt)
    _, specparm_planck, jpl, fpl = _binary_params(lower_a, lower_b, frac_lower_ratio, 8.0)
    del specparm_planck
    frac_low = _frac_interp(nt.fracrefa[band], jpl, fpl)

    if frac_upper_ratio is None:
        high = jnp.zeros_like(low)
        frac_high = jnp.zeros_like(frac_low)
    else:
        upper_idx0 = ((coef.jp - 13) * 5 + (coef.jt - 1)) * nspb + 1
        upper_idx1 = ((coef.jp - 12) * 5 + (coef.jt1 - 1)) * nspb + 1
        speccomb_u, _, js_u, fs_u = _binary_params(upper_a, upper_b, upper_ratio0, 4.0)
        speccomb1_u, _, js1_u, fs1_u = _binary_params(upper_a, upper_b, upper_ratio1, 4.0)
        high = _major_binary_upper(nt.absb[band], upper_idx0 + js_u - 1, upper_idx1 + js1_u - 1, speccomb_u, fs_u, speccomb1_u, fs1_u, coef)
        _, _, jpl_u, fpl_u = _binary_params(upper_a, upper_b, frac_upper_ratio, 4.0)
        frac_high = _frac_interp(nt.fracrefb[band], jpl_u, fpl_u)
    return jnp.where(coef.lower_mask[..., None], low, high), jnp.where(coef.lower_mask[..., None], frac_low, frac_high)


def _lw_fallback_taumol(qv, p_pa, pressure_interfaces_pa, tables: RRTMGTableBundle):
    """Nearest-pressure LW gas fallback retained for rejected branches."""

    gas_column, _ = _rrtmg_column_amounts(qv, pressure_interfaces_pa)
    gas_coeff = _nearest_pressure_coefficients(p_pa, tables)
    mask = tables.lw_gpoint_mask
    taug = gas_column[..., None, None] * jnp.maximum(gas_coeff, 0.0) * mask
    band_g_count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    fracs = jnp.broadcast_to(mask / band_g_count[:, None], taug.shape)
    return taug, fracs


def _lw_taumol_band(band, coef: _LWSetCoefState, nt, tables: RRTMGTableBundle):
    """One-band LW `taumol`: gas optical depth + g-point fraction for `band`.

    Extracted from the band loop so the chunked flux path can build a single
    band's `(..., nlay, 16)` `tau`/`frac` lazily inside the rtrnmc band scan,
    rather than materialising the full `(..., nlay, 16, 16)` stack up front (the
    dominant LW fp64 VRAM consumer).  `band` is a STATIC Python int — the
    per-band gas chemistry differs structurally, so the chunked driver resolves
    it with `lax.switch` over a traced band index.  Returns the gpoint-masked
    `(tau, frac)`; byte-identical to the per-band slice of `_lw_taumol`.
    """

    chi = nt.chi_mls

    def chi_ratio(a_1b, b_1b, level_1b):
        return chi[a_1b - 1, level_1b - 1] / chi[b_1b - 1, level_1b - 1]

    def chi_layer(gas_1b):
        return jnp.take(chi[gas_1b - 1], coef.jp, axis=0)

    absa = nt.absa[band]
    absb = nt.absb[band]
    nspa = nt.nspa[band]
    nspb = nt.nspb[band]
    lower_idx0 = ((coef.jp - 1) * 5 + (coef.jt - 1)) * nspa + 1
    lower_idx1 = (coef.jp * 5 + (coef.jt1 - 1)) * nspa + 1
    upper_idx0 = ((coef.jp - 13) * 5 + (coef.jt - 1)) * nspb + 1
    upper_idx1 = ((coef.jp - 12) * 5 + (coef.jt1 - 1)) * nspb + 1

    if band == 0:
        scalen2 = coef.colbrd * coef.scaleminorn2
        corr_low = jnp.where(coef.pavel < 250.0, 1.0 - 0.15 * (250.0 - coef.pavel) / 154.4, 1.0)
        low = corr_low[..., None] * (
            coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef)
            + _continuum_lw(band, coef, nt)
            + scalen2[..., None] * _minor2(nt.ka_mn2[band], coef)
        )
        corr_high = 1.0 - 0.15 * (coef.pavel / 95.6)
        high = corr_high[..., None] * (
            coef.colh2o[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef)
            + _foreign_lw(band, coef, nt)
            + scalen2[..., None] * _minor2(nt.kb_mn2[band], coef)
        )
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 1:
        corr = 1.0 - 0.05 * (coef.pavel - 100.0) / 900.0
        low = corr[..., None] * (coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_lw(band, coef, nt))
        high = coef.colh2o[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef) + _foreign_lw(band, coef, nt)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 2:
        speccomb, specparm, js, fs = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2_1, 8.0)
        major = _major_binary_lower(absa, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
        _, _, jmn2o, fmn2o = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 3), 8.0)
        adjcoln2o = _adj_minor_column(coef.coln2o, coef, chi_layer(4), 1.5, 0.5, 0.65)
        low = major + _continuum_lw(band, coef, nt) + adjcoln2o[..., None] * _minor_ratio(nt.ka_mn2o_r[band], jmn2o, fmn2o, coef)
        _, _, jpl, fpl = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 9), 8.0)
        frac_low = _frac_interp(nt.fracrefa[band], jpl, fpl)

        speccomb_u, _, js_u, fs_u = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2, 4.0)
        speccomb1_u, _, js1_u, fs1_u = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2_1, 4.0)
        major_u = _major_binary_upper(absb, upper_idx0 + js_u - 1, upper_idx1 + js1_u - 1, speccomb_u, fs_u, speccomb1_u, fs1_u, coef)
        _, _, jmn2o_u, fmn2o_u = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 13), 4.0)
        high = major_u + _foreign_lw(band, coef, nt) + adjcoln2o[..., None] * _minor_ratio(nt.kb_mn2o_r[band], jmn2o_u, fmn2o_u, coef)
        _, _, jpl_u, fpl_u = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 13), 4.0)
        frac_high = _frac_interp(nt.fracrefb[band], jpl_u, fpl_u)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], frac_low, frac_high)
    elif band == 3:
        low, frac_low = _binary_band(
            band,
            coef,
            nt,
            coef.colh2o,
            coef.colco2,
            (coef.rat_h2oco2, coef.rat_h2oco2_1),
            coef.colo3,
            coef.colco2,
            (coef.rat_o3co2, coef.rat_o3co2_1),
            chi_ratio(1, 2, 11),
            chi_ratio(3, 2, 13),
        )
        high_factor = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.88, 1.07, 1.10, 0.99, 0.88, 0.943, 1.0, 1.0], dtype=jnp.float64)
        low_tau = jnp.where(coef.lower_mask[..., None], low, 0.0)
        high_tau = jnp.where(coef.lower_mask[..., None], 0.0, low * high_factor)
        tau = low_tau + high_tau
        frac = frac_low
    elif band == 4:
        speccomb, specparm, js, fs = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(coef.colh2o, coef.colco2, coef.rat_h2oco2_1, 8.0)
        major = _major_binary_lower(absa, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
        _, _, jmo3, fmo3 = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 7), 8.0)
        low = major + _continuum_lw(band, coef, nt) + coef.colo3[..., None] * _minor_ratio(nt.ka_mo3_r[band], jmo3, fmo3, coef) + coef.wx[..., 0, None] * nt.ccl4[band]
        _, _, jpl, fpl = _binary_params(coef.colh2o, coef.colco2, chi_ratio(1, 2, 5), 8.0)
        frac_low = _frac_interp(nt.fracrefa[band], jpl, fpl)

        speccomb_u, _, js_u, fs_u = _binary_params(coef.colo3, coef.colco2, coef.rat_o3co2, 4.0)
        speccomb1_u, _, js1_u, fs1_u = _binary_params(coef.colo3, coef.colco2, coef.rat_o3co2_1, 4.0)
        high = _major_binary_upper(absb, upper_idx0 + js_u - 1, upper_idx1 + js1_u - 1, speccomb_u, fs_u, speccomb1_u, fs1_u, coef) + coef.wx[..., 0, None] * nt.ccl4[band]
        _, _, jpl_u, fpl_u = _binary_params(coef.colo3, coef.colco2, chi_ratio(3, 2, 43), 4.0)
        frac_high = _frac_interp(nt.fracrefb[band], jpl_u, fpl_u)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], frac_low, frac_high)
    elif band == 5:
        adjcolco2 = _adj_minor_column(coef.colco2, coef, chi_layer(2), 3.0, 2.0, 0.77)
        low = (
            coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef)
            + _continuum_lw(band, coef, nt)
            + adjcolco2[..., None] * _minor2(nt.ka_mco2[band], coef)
            + coef.wx[..., 1, None] * nt.cfc11adj[band]
            + coef.wx[..., 2, None] * nt.cfc12[band]
        )
        high = coef.wx[..., 1, None] * nt.cfc11adj[band] + coef.wx[..., 2, None] * nt.cfc12[band]
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = _frac_const(nt.fracrefa[band], coef)
    elif band == 6:
        c = _lw_coef_as_dtype(coef, jnp.float32)
        absa_r4 = absa.astype(jnp.float32)
        absb_r4 = absb.astype(jnp.float32)
        speccomb, specparm, js, fs = _binary_params(c.colh2o, c.colo3, c.rat_h2oo3, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(c.colh2o, c.colo3, c.rat_h2oo3_1, 8.0)
        major = _major_binary_lower(absa_r4, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, c)
        _, _, jmco2, fmco2 = _binary_params(c.colh2o, c.colo3, chi_ratio(1, 3, 3).astype(jnp.float32), 8.0)
        adjcolco2_l = _adj_minor_column(c.colco2, c, chi_layer(2).astype(jnp.float32), 3.0, 3.0, 0.79)
        low = major + _continuum_lw(band, c, nt) + adjcolco2_l[..., None] * _minor_ratio(nt.ka_mco2_r[band].astype(jnp.float32), jmco2, fmco2, c)
        _, _, jpl, fpl = _binary_params(c.colh2o, c.colo3, chi_ratio(1, 3, 3).astype(jnp.float32), 8.0)
        frac_low = _frac_interp(nt.fracrefa[band].astype(jnp.float32), jpl, fpl)

        adjcolco2_u = _adj_minor_column(c.colco2, c, chi_layer(2).astype(jnp.float32), 3.0, 2.0, 0.79)
        high = c.colo3[..., None] * _interp_four_rows_lw(absb_r4, upper_idx0, upper_idx1, nspb, c) + adjcolco2_u[..., None] * _minor2(nt.kb_mco2[band].astype(jnp.float32), c)
        high_factor = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.88, 1.07, 1.10, 0.99, 0.855, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
        high = high * high_factor.astype(jnp.float32)
        tau = jnp.where(c.lower_mask[..., None], low, high).astype(jnp.float64)
        frac = jnp.where(c.lower_mask[..., None], frac_low, _frac_const(nt.fracrefb[band].astype(jnp.float32), c)).astype(jnp.float64)
    elif band == 7:
        adjcolco2 = _adj_minor_column(coef.colco2, coef, chi_layer(2), 3.0, 2.0, 0.65)
        low = (
            coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef)
            + _continuum_lw(band, coef, nt)
            + adjcolco2[..., None] * _minor2(nt.ka_mco2[band], coef)
            + coef.colo3[..., None] * _minor2(nt.ka_mo3[band], coef)
            + coef.coln2o[..., None] * _minor2(nt.ka_mn2o[band], coef)
            + coef.wx[..., 2, None] * nt.cfc12[band]
            + coef.wx[..., 3, None] * nt.cfc22adj[band]
        )
        high = (
            coef.colo3[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef)
            + adjcolco2[..., None] * _minor2(nt.kb_mco2[band], coef)
            + coef.coln2o[..., None] * _minor2(nt.kb_mn2o[band], coef)
            + coef.wx[..., 2, None] * nt.cfc12[band]
            + coef.wx[..., 3, None] * nt.cfc22adj[band]
        )
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 8:
        speccomb, specparm, js, fs = _binary_params(coef.colh2o, coef.colch4, coef.rat_h2och4, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(coef.colh2o, coef.colch4, coef.rat_h2och4_1, 8.0)
        major = _major_binary_lower(absa, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
        _, _, jmn2o, fmn2o = _binary_params(coef.colh2o, coef.colch4, chi_ratio(1, 6, 3), 8.0)
        adjcoln2o = _adj_minor_column(coef.coln2o, coef, chi_layer(4), 1.5, 0.5, 0.65)
        low = major + _continuum_lw(band, coef, nt) + adjcoln2o[..., None] * _minor_ratio(nt.ka_mn2o_r[band], jmn2o, fmn2o, coef)
        _, _, jpl, fpl = _binary_params(coef.colh2o, coef.colch4, chi_ratio(1, 6, 9), 8.0)
        frac_low = _frac_interp(nt.fracrefa[band], jpl, fpl)
        high = coef.colch4[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef) + adjcoln2o[..., None] * _minor2(nt.kb_mn2o[band], coef)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], frac_low, _frac_const(nt.fracrefb[band], coef))
    elif band == 9:
        low = coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_lw(band, coef, nt)
        high = coef.colh2o[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef) + _foreign_lw(band, coef, nt)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 10:
        scaleo2 = coef.colo2 * coef.scaleminor
        low = coef.colh2o[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_lw(band, coef, nt) + scaleo2[..., None] * _minor2(nt.ka_mo2[band], coef)
        high = coef.colh2o[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef) + _foreign_lw(band, coef, nt) + scaleo2[..., None] * _minor2(nt.kb_mo2[band], coef)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 11:
        tau, frac = _binary_band(
            band,
            coef,
            nt,
            coef.colh2o,
            coef.colco2,
            (coef.rat_h2oco2, coef.rat_h2oco2_1),
            coef.colh2o,
            coef.colco2,
            (coef.rat_h2oco2, coef.rat_h2oco2_1),
            chi_ratio(1, 2, 10),
            None,
        )
    elif band == 12:
        speccomb, specparm, js, fs = _binary_params(coef.colh2o, coef.coln2o, coef.rat_h2on2o, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(coef.colh2o, coef.coln2o, coef.rat_h2on2o_1, 8.0)
        major = _major_binary_lower(absa, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
        _, _, jmco2, fmco2 = _binary_params(coef.colh2o, coef.coln2o, chi_ratio(1, 4, 1), 8.0)
        ratco2 = 1.0e20 * (coef.colco2 / jnp.maximum(coef.coldry, 1.0e-300)) / 3.55e-4
        adjcolco2 = jnp.where(ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.68) * 3.55e-4 * coef.coldry * 1.0e-20, coef.colco2)
        _, _, jmco, fmco = _binary_params(coef.colh2o, coef.coln2o, chi_ratio(1, 4, 3), 8.0)
        low = (
            major
            + _continuum_lw(band, coef, nt)
            + adjcolco2[..., None] * _minor_ratio(nt.ka_mco2_r[band], jmco2, fmco2, coef)
            + coef.colco[..., None] * _minor_ratio(nt.ka_mco_r[band], jmco, fmco, coef)
        )
        _, _, jpl, fpl = _binary_params(coef.colh2o, coef.coln2o, chi_ratio(1, 4, 5), 8.0)
        frac_low = _frac_interp(nt.fracrefa[band], jpl, fpl)
        high = coef.colo3[..., None] * _minor2(nt.kb_mo3[band], coef)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], frac_low, _frac_const(nt.fracrefb[band], coef))
    elif band == 13:
        low = coef.colco2[..., None] * _interp_four_rows_lw(absa, lower_idx0, lower_idx1, nspa, coef) + _continuum_lw(band, coef, nt)
        high = coef.colco2[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, nspb, coef)
        tau = jnp.where(coef.lower_mask[..., None], low, high)
        frac = jnp.where(coef.lower_mask[..., None], _frac_const(nt.fracrefa[band], coef), _frac_const(nt.fracrefb[band], coef))
    elif band == 14:
        speccomb, specparm, js, fs = _binary_params(coef.coln2o, coef.colco2, coef.rat_n2oco2, 8.0)
        speccomb1, specparm1, js1, fs1 = _binary_params(coef.coln2o, coef.colco2, coef.rat_n2oco2_1, 8.0)
        major = _major_binary_lower(absa, lower_idx0 + js - 1, lower_idx1 + js1 - 1, speccomb, specparm, fs, speccomb1, specparm1, fs1, coef)
        _, _, jmn2, fmn2 = _binary_params(coef.coln2o, coef.colco2, chi_ratio(4, 2, 1), 8.0)
        scalen2 = coef.colbrd * coef.scaleminor
        low = major + _continuum_lw(band, coef, nt) + scalen2[..., None] * _minor_ratio(nt.ka_mn2_r[band], jmn2, fmn2, coef)
        _, _, jpl, fpl = _binary_params(coef.coln2o, coef.colco2, chi_ratio(4, 2, 1), 8.0)
        tau = jnp.where(coef.lower_mask[..., None], low, jnp.zeros_like(low))
        frac = jnp.where(coef.lower_mask[..., None], _frac_interp(nt.fracrefa[band], jpl, fpl), jnp.zeros_like(low))
    else:
        tau, frac = _binary_band(
            band,
            coef,
            nt,
            coef.colh2o,
            coef.colch4,
            (coef.rat_h2och4, coef.rat_h2och4_1),
            coef.colch4,
            coef.colh2o,
            0.0,
            chi_ratio(1, 6, 6),
            0.0,
        )
        high = coef.colch4[..., None] * _interp_four_rows_lw(absb, upper_idx0, upper_idx1, jnp.maximum(nspb, 1), coef)
        tau = jnp.where(coef.lower_mask[..., None], tau, high)
        frac = jnp.where(coef.lower_mask[..., None], frac, _frac_const(nt.fracrefb[band], coef))

    return tau * tables.lw_gpoint_mask[band], frac * tables.lw_gpoint_mask[band]


def _lw_taumol(coef: _LWSetCoefState, tables: RRTMGTableBundle):
    """Ports WRF LW `taumol` reduced-g-point gas optical depth and fractions.

    Thin wrapper over `_lw_taumol_band` that stacks all 16 bands into the full
    `(..., nlay, 16, 16)` `tau`/`frac` arrays (oracle / intermediate-parity
    entry).  The operational flux path uses the chunked per-band driver instead.
    """

    nt = _native_lw_tables()
    taug = []
    fracs = []
    for band in range(16):
        tau_b, frac_b = _lw_taumol_band(band, coef, nt, tables)
        taug.append(tau_b)
        fracs.append(frac_b)
    return jnp.stack(taug, axis=-2), jnp.stack(fracs, axis=-2)


def _lw_taumol_fused(coef: _LWSetCoefState, tables: RRTMGTableBundle):
    """Returns LW `taumol` through a band-axis `lax.scan` barrier."""

    taug, fracs = _lw_taumol(coef, tables)

    def keep_band(_, band_index):
        return None, (taug[..., band_index, :], fracs[..., band_index, :])

    _, (taug_scan, fracs_scan) = lax.scan(keep_band, None, jnp.arange(16, dtype=jnp.int32))
    return jnp.moveaxis(taug_scan, 0, -2), jnp.moveaxis(fracs_scan, 0, -2)


def _lw_tfn_factor(odepth):
    """WRF `tfn_tbl` source correction from `rrtmg_lw.F:3403-3409,8054-8070`."""

    tau = jnp.maximum(odepth, 0.0)
    tblind = tau / (LW_BPADE + tau)
    idx = jnp.clip((LW_TBLINT * tblind + 0.5).astype(jnp.int32), 0, LW_NTBL)
    tfn = idx.astype(jnp.float64) / float(LW_NTBL)
    tau_tbl = jnp.where(idx == LW_NTBL, 1.0e10, LW_BPADE * tfn / jnp.maximum(1.0 - tfn, 1.0e-300))
    exp_tbl = jnp.maximum(jnp.exp(-tau_tbl), LW_EXP_EPS)
    table_factor = jnp.where(
        tau_tbl < 0.06,
        tau_tbl / 6.0,
        1.0 - 2.0 * ((1.0 / jnp.maximum(tau_tbl, 1.0e-300)) - (exp_tbl / jnp.maximum(1.0 - exp_tbl, 1.0e-300))),
    )
    return jnp.where(tau <= 0.06, tau / 6.0, table_factor)


def _source_recurrence_down(trans, source):
    """Top-to-bottom LW source recurrence."""

    nlay = trans.shape[-3]
    rad = jnp.zeros_like(source[..., :1, :, :])
    levels = [rad]
    for idx in range(nlay - 1, -1, -1):
        rad = rad * trans[..., idx : idx + 1, :, :] + source[..., idx : idx + 1, :, :] * (1.0 - trans[..., idx : idx + 1, :, :])
        levels.append(rad)
    return jnp.concatenate(levels[::-1], axis=-3)


def _source_recurrence_up(trans, source, surface):
    """Bottom-to-top LW source recurrence with surface emission."""

    nlay = trans.shape[-3]
    rad = surface[..., None, :, :]
    levels = [rad]
    for idx in range(nlay):
        rad = rad * trans[..., idx : idx + 1, :, :] + source[..., idx : idx + 1, :, :] * (1.0 - trans[..., idx : idx + 1, :, :])
        levels.append(rad)
    return jnp.concatenate(levels, axis=-3)


def _lw_band_to_global(values: jnp.ndarray) -> jnp.ndarray:
    """Converts padded band/local-g LW arrays to WRF global g-point order."""

    pieces = []
    for band, count in enumerate(_LW_GPOINT_COUNTS):
        pieces.append(values[..., band, :count])
    return jnp.concatenate(pieces, axis=-1)


def _lw_global_to_band(values: jnp.ndarray) -> jnp.ndarray:
    """Converts WRF global g-point arrays to padded band/local-g layout."""

    pieces = []
    start = 0
    for count in _LW_GPOINT_COUNTS:
        band_values = values[..., start : start + count]
        pad = max(_LW_GPOINT_COUNTS) - count
        if pad:
            band_values = jnp.pad(band_values, [(0, 0)] * (band_values.ndim - 1) + [(0, pad)])
        pieces.append(band_values)
        start += count
    return jnp.stack(pieces, axis=-2)


def _kiss_uint_to_float(value):
    signed = lax.bitcast_convert_type(value, jnp.int32)
    return signed.astype(jnp.float64) * 2.328306e-10 + 0.5


def _kiss_step(seed1, seed2, seed3, seed4):
    """WRF MCICA KISS generator step; mirrors module_ra_rrtmg_lw.F:2688-2706."""

    mask16 = jnp.asarray(65535, dtype=jnp.uint32)

    def m(value, shift):
        if shift > 0:
            return jnp.bitwise_xor(value, jnp.left_shift(value, jnp.asarray(shift, dtype=jnp.uint32)))
        return jnp.bitwise_xor(value, jnp.right_shift(value, jnp.asarray(-shift, dtype=jnp.uint32)))

    seed1 = seed1 * jnp.asarray(69069, dtype=jnp.uint32) + jnp.asarray(1327217885, dtype=jnp.uint32)
    seed2 = m(m(m(seed2, 13), -17), 5)
    seed3 = jnp.asarray(18000, dtype=jnp.uint32) * jnp.bitwise_and(seed3, mask16) + jnp.right_shift(seed3, jnp.asarray(16, dtype=jnp.uint32))
    seed4 = jnp.asarray(30903, dtype=jnp.uint32) * jnp.bitwise_and(seed4, mask16) + jnp.right_shift(seed4, jnp.asarray(16, dtype=jnp.uint32))
    kiss = seed1 + seed2 + jnp.left_shift(seed3, jnp.asarray(16, dtype=jnp.uint32)) + seed4
    return seed1, seed2, seed3, seed4, _kiss_uint_to_float(kiss)


def _lw_mcica_random_cloud_mask(p_layer_pa, cloud_fraction):
    """Builds the WRF random-overlap McICA mask for the LW fixture path.

    The WRF harness uses `icld=1`, `irng=0`, and `permuteseed=150`, so this
    ports only the random-overlap KISS path from module_ra_rrtmg_lw.F:2402-2438.
    """

    p_seed = p_layer_pa.astype(jnp.float32) * jnp.float32(0.01)
    p_seed = p_seed * jnp.float32(100.0)
    frac = p_seed[..., :4] - jnp.floor(p_seed[..., :4])
    seeds = [(frac[..., idx] * jnp.float32(1.0e9)).astype(jnp.uint32) for idx in range(4)]
    carry = tuple(seeds)

    def scan_step(carry, _):
        seed1, seed2, seed3, seed4, random_value = _kiss_step(*carry)
        return (seed1, seed2, seed3, seed4), random_value

    carry, _ = lax.scan(scan_step, carry, xs=None, length=150)
    nlay = int(p_layer_pa.shape[-1])
    carry, random_flat = lax.scan(scan_step, carry, xs=None, length=140 * nlay)
    cdf = jnp.reshape(random_flat, (140, nlay) + p_layer_pa.shape[:-1])
    leading = len(p_layer_pa.shape) - 1
    cdf = jnp.transpose(cdf, tuple(range(2, 2 + leading)) + (1, 0))
    cldf = jnp.where(cloud_fraction < 1.0e-20, 0.0, cloud_fraction)
    return (cdf >= (1.0 - cldf[..., :, None])).astype(jnp.float64)


def _lw_cldprmc_state(state, p_ext, layer_mass_ext, tables: RRTMGTableBundle):
    """Ports the WRF LW `mcica_subcol_lw` random mask plus `cldprmc` optical depth."""

    del tables
    cloud_ext = jnp.concatenate((state.cloud_fraction, jnp.zeros_like(state.cloud_fraction[..., -1:])), axis=-1)
    qc_ext = jnp.concatenate((state.qc, jnp.zeros_like(state.qc[..., -1:])), axis=-1)
    qi_ext = jnp.concatenate((state.qi, jnp.zeros_like(state.qi[..., -1:])), axis=-1)
    qs_ext = jnp.concatenate((state.qs, jnp.zeros_like(state.qs[..., -1:])), axis=-1)
    cloud_safe = jnp.maximum(cloud_ext, 0.01)
    clw_path = qc_ext * layer_mass_ext * 1000.0 / cloud_safe
    ciw_path = qi_ext * layer_mass_ext * 1000.0 / cloud_safe
    csw_path = qs_ext * 0.99 * layer_mass_ext * 1000.0 / cloud_safe
    cldf_global = _lw_mcica_random_cloud_mask(p_ext, cloud_ext)
    clw_global = jnp.where(cldf_global > 0.5, clw_path[..., :, None], 0.0)
    ciw_global = jnp.where(cldf_global > 0.5, ciw_path[..., :, None], 0.0)
    csw_global = jnp.where(cldf_global > 0.5, csw_path[..., :, None], 0.0)

    cldf_band = _lw_global_to_band(cldf_global)
    clw_band = _lw_global_to_band(clw_global)
    ciw_band = _lw_global_to_band(ciw_global)
    csw_band = _lw_global_to_band(csw_global)
    cloud = _native_lw_cloud_tables()
    liquid = jnp.asarray(cloud.liquid, dtype=clw_band.dtype).reshape((1,) * (clw_band.ndim - 2) + (16, 1))
    ice = jnp.asarray(cloud.ice, dtype=ciw_band.dtype).reshape((1,) * (ciw_band.ndim - 2) + (16, 1))
    snow = jnp.asarray(cloud.snow, dtype=csw_band.dtype).reshape((1,) * (csw_band.ndim - 2) + (16, 1))
    taucmc = clw_band * liquid + ciw_band * ice + csw_band * snow
    return cldf_band, taucmc


def _lw_lookup_terms(tau):
    """Returns WRF lookup-table tau/exp/tfn values for `rtrnmc`."""

    tblind = tau / (LW_BPADE + tau)
    idx = jnp.clip((LW_TBLINT * tblind + 0.5).astype(jnp.int32), 0, LW_NTBL)
    tfn = idx.astype(jnp.float64) / float(LW_NTBL)
    tau_tbl = jnp.where(idx == LW_NTBL, 1.0e10, LW_BPADE * tfn / jnp.maximum(1.0 - tfn, 1.0e-300))
    tau_tbl = jnp.where(idx == 0, 0.0, tau_tbl)
    exp_tbl = jnp.where(idx == LW_NTBL, LW_EXP_EPS, jnp.maximum(jnp.exp(-tau_tbl), LW_EXP_EPS))
    tfn_tbl = jnp.where(
        idx == LW_NTBL,
        1.0,
        jnp.where(
            tau_tbl < 0.06,
            tau_tbl / 6.0,
            1.0 - 2.0 * ((1.0 / jnp.maximum(tau_tbl, 1.0e-300)) - (exp_tbl / jnp.maximum(1.0 - exp_tbl, 1.0e-300))),
        ),
    )
    return tau_tbl, exp_tbl, tfn_tbl


def _lw_rtrnmc_layer_terms(radld, odepth_raw, odcld, efclfrac, cldfmc, plfrac, blay, dplankdn, dplankup):
    """Evaluates one WRF `rtrnmc` downward layer update for all g-points."""

    odtot_raw = odepth_raw + odcld

    atrans_a = odepth_raw - 0.5 * odepth_raw * odepth_raw
    odepth_rec = odepth_raw / 6.0
    gassrc_a = plfrac * (blay + dplankdn * odepth_rec) * atrans_a
    atot_a = odtot_raw - 0.5 * odtot_raw * odtot_raw
    odtot_rec = odtot_raw / 6.0
    bbdtot_a = plfrac * (blay + dplankdn * odtot_rec)
    bbd_a = plfrac * (blay + dplankdn * odepth_rec)
    bbugas_a = plfrac * (blay + dplankup * odepth_rec)
    bbutot_a = plfrac * (blay + dplankup * odtot_rec)
    tfn_a = odepth_rec

    _, exp_tot_b, tfn_tot_b = _lw_lookup_terms(odtot_raw)
    atot_b = 1.0 - exp_tot_b
    bbdtot_b = plfrac * (blay + tfn_tot_b * dplankdn)
    bbutot_b = plfrac * (blay + tfn_tot_b * dplankup)

    tau_gas_c, exp_gas_c, tfn_gas_c = _lw_lookup_terms(odepth_raw)
    atrans_c = 1.0 - exp_gas_c
    gassrc_c = atrans_c * plfrac * (blay + tfn_gas_c * dplankdn)
    odtot_c = tau_gas_c + odcld
    _, exp_tot_c, tfn_tot_c = _lw_lookup_terms(odtot_c)
    atot_c = 1.0 - exp_tot_c
    bbdtot_c = plfrac * (blay + tfn_tot_c * dplankdn)
    bbd_c = plfrac * (blay + tfn_gas_c * dplankdn)
    bbugas_c = plfrac * (blay + tfn_gas_c * dplankup)
    bbutot_c = plfrac * (blay + tfn_tot_c * dplankup)

    case_a = odtot_raw < 0.06
    case_b = jnp.logical_and(jnp.logical_not(case_a), odepth_raw <= 0.06)
    atrans_cloud = jnp.where(case_a | case_b, atrans_a, atrans_c)
    gassrc_cloud = jnp.where(case_a | case_b, gassrc_a, gassrc_c)
    atot_cloud = jnp.where(case_a, atot_a, jnp.where(case_b, atot_b, atot_c))
    bbdtot_cloud = jnp.where(case_a, bbdtot_a, jnp.where(case_b, bbdtot_b, bbdtot_c))
    bbd_cloud = jnp.where(case_a | case_b, bbd_a, bbd_c)
    bbugas_cloud = jnp.where(case_a | case_b, bbugas_a, bbugas_c)
    bbutot_cloud = jnp.where(case_a, bbutot_a, jnp.where(case_b, bbutot_b, bbutot_c))
    tfn_cloud = jnp.where(case_a | case_b, tfn_a, tfn_gas_c)
    rad_cloud = (
        radld
        - radld * (atrans_cloud + efclfrac * (1.0 - atrans_cloud))
        + gassrc_cloud
        + cldfmc * (bbdtot_cloud * atot_cloud - gassrc_cloud)
    )

    clear_small = odepth_raw <= 0.06
    atrans_clear_small = odepth_raw - 0.5 * odepth_raw * odepth_raw
    odepth_clear_rec = odepth_raw / 6.0
    bbd_clear_small = plfrac * (blay + dplankdn * odepth_clear_rec)
    bbugas_clear_small = plfrac * (blay + dplankup * odepth_clear_rec)
    _, exp_clear, tfn_clear_lookup = _lw_lookup_terms(odepth_raw)
    atrans_clear_lookup = 1.0 - exp_clear
    bbd_clear_lookup = plfrac * (blay + tfn_clear_lookup * dplankdn)
    bbugas_clear_lookup = plfrac * (blay + tfn_clear_lookup * dplankup)
    atrans_clear = jnp.where(clear_small, atrans_clear_small, atrans_clear_lookup)
    bbd_clear = jnp.where(clear_small, bbd_clear_small, bbd_clear_lookup)
    bbugas_clear = jnp.where(clear_small, bbugas_clear_small, bbugas_clear_lookup)
    tfn_clear = jnp.where(clear_small, odepth_clear_rec, tfn_clear_lookup)
    rad_clear = radld + (bbd_clear - radld) * atrans_clear

    return {
        "cloud": {
            "rad": rad_cloud,
            "atrans": atrans_cloud,
            "atot": atot_cloud,
            "bbugas": bbugas_cloud,
            "bbutot": bbutot_cloud,
            "bbd": bbd_cloud,
            "tfn": tfn_cloud,
        },
        "clear": {
            "rad": rad_clear,
            "atrans": atrans_clear,
            "atot": atrans_clear,
            "bbugas": bbugas_clear,
            "bbutot": bbugas_clear,
            "bbd": bbd_clear,
            "tfn": tfn_clear,
        },
    }


def _lw_rtrnmc_band_fluxes(
    state,
    tau_b,
    frac_b,
    cldf_b,
    taucmc_b,
    sec_raw,
    scale,
    valid,
    plank_b,
    planklev_b,
    plankbnd_b,
    cloud_layer,
    with_clear_sky: bool,
):
    """Per-band `rtrnmc` radiance solve -> `(zfd, zfu, zcd, zcu)` g-point fluxes.

    Band-agnostic: every band-indexed quantity is supplied already sliced by the
    caller (a static Python ``band`` in the oracle loop, or a traced index +
    ``lax.switch`` taumol in the chunked flux path).  This is the EXACT body that
    used to live inline in the ``for band`` loop, so both callers are byte-for-byte
    the same numerics.  Returns the cloudy/all-sky down/up g-point flux buffers
    ``zfd``/``zfu`` (and, when ``with_clear_sky``, the clear-sky ``zcd``/``zcu``;
    otherwise both are ``None``).
    """

    zcd = None
    zcu = None
    tau_b = tau_b
    frac_b = frac_b
    cldf_b = cldf_b
    taucmc_b = taucmc_b
    sec = sec_raw[..., None]
    odcld = jnp.where(cldf_b == 1.0, sec[..., None, :] * taucmc_b, 0.0)
    efclfrac = (1.0 - jnp.exp(-odcld)) * cldf_b

    def layer0(value):
        return jnp.moveaxis(value, -2, 0)

    plank_b = plank_b
    dplankdn_b = planklev_b[..., :-1] - plank_b
    dplankup_b = planklev_b[..., 1:] - plank_b

    # The down-recurrence body broadcasts the per-layer optical depth
    # (`sec*tau`, batch from `tau`/`sec`), the cloudy overlap terms
    # (`odcld`/`efclfrac`, batch from the cloud grid) and the per-layer
    # Planck source (batch from `plank_b`, i.e. the temperature grid).  Any
    # of these can carry the full (ny,nx) surface batch independently, so
    # initialize the scan carry at the FULL broadcast batch — this keeps the
    # lax.scan carry in/out shapes equal for single-column fixtures AND
    # (ny,nx) operational grids regardless of which input is broadcast.
    carry_batch = jnp.broadcast_shapes(
        tau_b[..., 0, :].shape,
        sec.shape,
        plank_b[..., :1].shape,
        odcld[..., 0, :].shape,
        efclfrac[..., 0, :].shape,
    )
    radld = jnp.zeros(carry_batch, dtype=tau_b.dtype)
    radclrd = jnp.zeros_like(radld)
    iclddn = jnp.zeros(carry_batch[:-1], dtype=bool)
    down_xs = (
        layer0(jnp.flip(tau_b, axis=-2)),
        layer0(jnp.flip(frac_b, axis=-2)),
        layer0(jnp.flip(odcld, axis=-2)),
        layer0(jnp.flip(efclfrac, axis=-2)),
        layer0(jnp.flip(cldf_b, axis=-2)),
        jnp.moveaxis(jnp.flip(cloud_layer, axis=-1), -1, 0),
        jnp.moveaxis(jnp.flip(plank_b, axis=-1), -1, 0),
        jnp.moveaxis(jnp.flip(dplankdn_b, axis=-1), -1, 0),
        jnp.moveaxis(jnp.flip(dplankup_b, axis=-1), -1, 0),
    )

    def down_body(carry, xs):
        radld, radclrd, iclddn = carry
        tau_l, frac_l, odcld_l, efclfrac_l, cldf_l, cloud_l, plank_l, dplankdn_l, dplankup_l = xs
        odepth = jnp.maximum(sec * tau_l, 0.0)
        terms = _lw_rtrnmc_layer_terms(
            radld,
            odepth,
            odcld_l,
            efclfrac_l,
            cldf_l,
            frac_l,
            plank_l[..., None],
            dplankdn_l[..., None],
            dplankup_l[..., None],
        )
        is_cloud = cloud_l[..., None]
        rad_new = jnp.where(is_cloud, terms["cloud"]["rad"], terms["clear"]["rad"])
        atrans = jnp.where(is_cloud, terms["cloud"]["atrans"], terms["clear"]["atrans"])
        atot = jnp.where(is_cloud, terms["cloud"]["atot"], terms["clear"]["atot"])
        bbugas = jnp.where(is_cloud, terms["cloud"]["bbugas"], terms["clear"]["bbugas"])
        bbutot = jnp.where(is_cloud, terms["cloud"]["bbutot"], terms["clear"]["bbutot"])
        bbd = jnp.where(is_cloud, terms["cloud"]["bbd"], terms["clear"]["bbd"])
        tfn = jnp.where(is_cloud, terms["cloud"]["tfn"], terms["clear"]["tfn"])
        iclddn_new = jnp.logical_or(iclddn, cloud_l)
        radclrd_new = jnp.where(iclddn_new[..., None], radclrd + (bbd - radclrd) * atrans, rad_new)
        # WRF `rtrnmc` :3417-3423 — clear-sky down stream tracks the all-sky
        # stream until the first cloud above (iclddn), then propagates with the
        # clear-sky source/transmittance only.  Emit it (and the cumulative
        # iclddn / clear surface-reflected atrans) for the clear-sky up sweep.
        return (rad_new, radclrd_new, iclddn_new), (
            rad_new * scale * valid,
            atrans,
            atot,
            bbugas,
            bbutot,
            tfn * valid,
            radclrd_new * scale * valid,
            iclddn_new,
            terms["clear"]["atrans"],
            terms["clear"]["bbugas"],
        )

    (radld, radclrd_final, _), down_outputs = lax.scan(down_body, (radld, radclrd, iclddn), down_xs)
    (
        zfd_scan,
        atrans_scan,
        atot_scan,
        bbugas_scan,
        bbutot_scan,
        tfn_scan,
        zcd_scan,
        iclddn_scan,
        atrans_clear_scan,
        bbugas_clear_scan,
    ) = down_outputs
    zfd_layer0 = jnp.flip(jnp.concatenate((jnp.zeros_like(radld)[None, ...], zfd_scan), axis=0), axis=0)
    zfd = jnp.moveaxis(zfd_layer0, 0, -2)
    atrans_layers = jnp.moveaxis(jnp.flip(atrans_scan, axis=0), 0, -2)
    atot_layers = jnp.moveaxis(jnp.flip(atot_scan, axis=0), 0, -2)
    bbugas_layers = jnp.moveaxis(jnp.flip(bbugas_scan, axis=0), 0, -2)
    bbutot_layers = jnp.moveaxis(jnp.flip(bbutot_scan, axis=0), 0, -2)
    tfn_layers = jnp.moveaxis(jnp.flip(tfn_scan, axis=0), 0, -2)

    radlu = frac_b[..., 0, :] * plankbnd_b[..., None] + (1.0 - state.surface_emissivity[..., None]) * radld
    up_xs = (
        layer0(atrans_layers),
        layer0(atot_layers),
        layer0(bbugas_layers),
        layer0(bbutot_layers),
        layer0(efclfrac),
        layer0(cldf_b),
        jnp.moveaxis(cloud_layer, -1, 0),
    )

    def up_body(radlu, xs):
        atrans_l, atot_l, bbugas_l, bbutot_l, efclfrac_l, cldf_l, cloud_l = xs
        is_cloud = cloud_l[..., None]
        gassrc = bbugas_l * atrans_l
        rad_cloud = (
            radlu
            - radlu * (atrans_l + efclfrac_l * (1.0 - atrans_l))
            + gassrc
            + cldf_l * (bbutot_l * atot_l - gassrc)
        )
        rad_clear = radlu + (bbugas_l - radlu) * atrans_l
        radlu = jnp.where(is_cloud, rad_cloud, rad_clear)
        return radlu, radlu * scale * valid

    _, zfu_scan = lax.scan(up_body, radlu, up_xs)
    zfu = jnp.moveaxis(jnp.concatenate((radlu[None, ...] * scale * valid, zfu_scan), axis=0), 0, -2)

    if with_clear_sky:
        # Clear-sky DOWN flux: same interface layout as `zfd`, built from the
        # clear-sky down radiance `radclrd` (WRF `clrdrad`).  zcd_scan is the
        # per-layer clear radiance emitted top-to-bottom; prepend the TOA zero
        # and flip back to bottom-to-top, exactly mirroring the all-sky `zfd`.
        zcd_layer0 = jnp.flip(jnp.concatenate((jnp.zeros_like(radld)[None, ...], zcd_scan), axis=0), axis=0)
        zcd = jnp.moveaxis(zcd_layer0, 0, -2)
        # Clear-sky UP sweep (WRF `rtrnmc` :3436-3467).  Surface boundary uses
        # the CLEAR down radiance at the surface (`radclrd_final`); the stream
        # follows the all-sky `radlu` while no cloud is above (iclddn=0) and
        # diverges with the clear-sky gas source/transmittance once iclddn=1.
        radclru0 = frac_b[..., 0, :] * plankbnd_b[..., None] + (1.0 - state.surface_emissivity[..., None]) * radclrd_final
        # The clear stream follows the all-sky up radiance `radlu` while no
        # cloud is above (iclddn=0); the carry recomputes that all-sky `radlu`
        # recurrence (identical formula to `up_body`) so no external radiance
        # series is needed.
        up_clear_xs = (
            layer0(jnp.moveaxis(jnp.flip(atrans_clear_scan, axis=0), 0, -2)),
            layer0(jnp.moveaxis(jnp.flip(bbugas_clear_scan, axis=0), 0, -2)),
            layer0(jnp.moveaxis(jnp.flip(iclddn_scan, axis=0), 0, -2)),
            layer0(atrans_layers),
            layer0(atot_layers),
            layer0(bbugas_layers),
            layer0(bbutot_layers),
            layer0(efclfrac),
            layer0(cldf_b),
            jnp.moveaxis(cloud_layer, -1, 0),
        )

        def up_clear_body(carry, xs):
            radlu_l, radclru_l = carry
            (
                atrans_clr_l,
                bbugas_clr_l,
                iclddn_l,
                atrans_l,
                atot_l,
                bbugas_l,
                bbutot_l,
                efclfrac_l,
                cldf_l,
                cloud_l,
            ) = xs
            is_cloud = cloud_l[..., None]
            gassrc = bbugas_l * atrans_l
            rad_cloud = (
                radlu_l
                - radlu_l * (atrans_l + efclfrac_l * (1.0 - atrans_l))
                + gassrc
                + cldf_l * (bbutot_l * atot_l - gassrc)
            )
            rad_clear_allsky = radlu_l + (bbugas_l - radlu_l) * atrans_l
            radlu_new = jnp.where(is_cloud, rad_cloud, rad_clear_allsky)
            # WRF :3461-3467 — clear stream = all-sky up while no cloud above
            # (iclddn=0); once a cloud is above, propagate with clear-sky gas
            # source + clear-sky transmittance.
            radclru_diverged = radclru_l + (bbugas_clr_l - radclru_l) * atrans_clr_l
            radclru_new = jnp.where(iclddn_l[..., None], radclru_diverged, radlu_new)
            return (radlu_new, radclru_new), radclru_new * scale * valid

        _, zcu_scan = lax.scan(up_clear_body, (radlu, radclru0), up_clear_xs)
        zcu = jnp.moveaxis(jnp.concatenate((radclru0[None, ...] * scale * valid, zcu_scan), axis=0), 0, -2)

    return zfd, zfu, zcd, zcu, tfn_layers


def _lw_rtrnmc_outputs(state, intermediate_base, cldfmc, taucmc, transfer_tau, tables: RRTMGTableBundle, *, flux_only: bool = False, with_clear_sky: bool = False):
    """Ports WRF `rtrnmc` cloudy-source recurrence and per-g-point flux capture.

    When ``flux_only`` is True (the operational path) the per-band
    ``(..., nlay+1, 16)`` flux buffers are reduced over the g-point axis and
    accumulated into fp64 band-summed fluxes ``flux_down``/``flux_up`` rather than
    stacked into the full ``(..., nlay+1, 16, 16)`` per-g-point array — so the
    peak live g-point temporary is one band (or one tile), not all 16.  The
    accumulation is over disjoint per-band contributions in fp64, so the result
    is independent of how bands are tiled and reproduces ``sum(axis=(-1, -2))``
    of the full array to fp64 precision.  When False (the WRF intermediate-oracle
    entry) the full per-g-point arrays are built as before.
    """

    del transfer_tau
    tau = intermediate_base.tau
    fracs = intermediate_base.fracs
    secdiff = intermediate_base.secdiff
    planklay = intermediate_base.planklay
    planklev = intermediate_base.planklev
    plankbnd = intermediate_base.plankbnd
    nlay = int(tau.shape[-3])
    cloud_layer = jnp.any(cldfmc > 0.5, axis=(-1, -2))
    scale_band = tables.lw_delwave * (jnp.pi * 1.0e4)
    zfd_bands = []
    zfu_bands = []
    tfn_bands = []
    # fp64 band-summed flux accumulators (flux_only path).  Shape
    # `(..., nlay+1)`; lazily initialised on the first tile from the per-band
    # `(..., nlay+1, 16)` buffers reduced over the g-point axis.  A tile of
    # `_LW_GPOINT_CHUNK_BANDS` bands is flushed into the fp64 accumulator before
    # the next tile starts, so the peak live g-point temporary is one tile
    # (default 1 band) rather than the full 16-band stack.
    flux_only_tile = max(1, min(int(_LW_GPOINT_CHUNK_BANDS), _LW_NBANDS))
    flux_down_acc = None
    flux_up_acc = None
    zfd_tile = []
    zfu_tile = []
    # Clear-sky (cloud-free) band-summed flux accumulators, WRF `totdclfl`/
    # `totuclfl` from the parallel clear-sky stream in `rtrnmc`
    # (`module_ra_rrtmg_lw.F` :3417-3489).  Only built when ``with_clear_sky``.
    clear_down_acc = None
    clear_up_acc = None
    zcd_tile = []
    zcu_tile = []

    def _flush_tile(down_acc, up_acc, dtile, utile):
        if not dtile:
            return down_acc, up_acc
        # Reduce each band over g-points, sum the tile's bands, accumulate fp64.
        dsum = sum(jnp.sum(z, axis=-1) for z in dtile)
        usum = sum(jnp.sum(z, axis=-1) for z in utile)
        down_acc = dsum if down_acc is None else down_acc + dsum
        up_acc = usum if up_acc is None else up_acc + usum
        return down_acc, up_acc

    for band in range(16):
        valid = tables.lw_gpoint_mask[band]
        tau_b = tau[..., :, band, :]
        frac_b = fracs[..., :, band, :]
        cldf_b = cldfmc[..., :, band, :]
        taucmc_b = taucmc[..., :, band, :]
        sec_raw = secdiff[..., band]
        scale = scale_band[band]
        plank_b = planklay[..., :, band]
        planklev_b = planklev[..., band]
        plankbnd_b = plankbnd[..., band]
        zfd, zfu, zcd, zcu, tfn_layers = _lw_rtrnmc_band_fluxes(
            state, tau_b, frac_b, cldf_b, taucmc_b, sec_raw, scale, valid,
            plank_b, planklev_b, plankbnd_b, cloud_layer, with_clear_sky,
        )

        if flux_only:
            # Collect this band into the current tile; flush (g-point-reduce +
            # fp64 band-accumulate) once the tile is full.  Only one tile's
            # buffers are live; the full 16-band stack is never built.  Summing
            # disjoint per-band contributions in fp64 is order-independent to
            # fp64 precision and reproduces `sum(zfd_all, axis=(-1, -2))`.
            zfd_tile.append(zfd)
            zfu_tile.append(zfu)
            if with_clear_sky:
                zcd_tile.append(zcd)
                zcu_tile.append(zcu)
            if len(zfd_tile) >= flux_only_tile:
                flux_down_acc, flux_up_acc = _flush_tile(flux_down_acc, flux_up_acc, zfd_tile, zfu_tile)
                zfd_tile = []
                zfu_tile = []
                if with_clear_sky:
                    clear_down_acc, clear_up_acc = _flush_tile(clear_down_acc, clear_up_acc, zcd_tile, zcu_tile)
                    zcd_tile = []
                    zcu_tile = []
        else:
            zfd_bands.append(zfd)
            zfu_bands.append(zfu)
            tfn_bands.append(tfn_layers)

    plansum = jnp.sum(fracs * tables.lw_gpoint_mask.reshape((1,) * (fracs.ndim - 2) + (16, 16)), axis=-1) * planklay
    if flux_only:
        flux_down_acc, flux_up_acc = _flush_tile(flux_down_acc, flux_up_acc, zfd_tile, zfu_tile)
        # `tfn`/per-g-point outputs are oracle-only; the production path consumes
        # the band-summed fluxes directly, so return them in the per-g-point
        # slots' place via the dedicated flux tuple.
        if with_clear_sky:
            clear_down_acc, clear_up_acc = _flush_tile(clear_down_acc, clear_up_acc, zcd_tile, zcu_tile)
            return flux_down_acc, flux_up_acc, plansum, clear_down_acc, clear_up_acc
        return flux_down_acc, flux_up_acc, plansum
    zfd_all = jnp.stack(zfd_bands, axis=-2)
    zfu_all = jnp.stack(zfu_bands, axis=-2)
    tfn_all = jnp.stack(tfn_bands, axis=-2)
    return tfn_all, zfd_all, zfu_all, plansum


def _lw_solver_base(state: RRTMGLWColumnState, tables: RRTMGTableBundle, *, build_taumol: bool = True):
    """Builds LW gas/source state up to the `rtrnmc` transfer entry.

    Shared by the operational flux path (`_lw_solver_fluxes`) and the WRF
    intermediate-oracle path (`_lw_solver_state`) so the only divergence between
    them is whether `_lw_rtrnmc_outputs` accumulates band-summed fluxes
    (operational, low VRAM) or materialises the full per-g-point arrays
    (oracle).  WRF formulas: `rtrnmc` lines 3253-3409.
    """

    state = _clip_state(state)
    original_layers = state.p.shape[-1]
    original_interfaces = _pressure_interfaces(state.p)
    top_pressure = 0.5 * original_interfaces[..., -1:]
    pressure_interfaces = jnp.concatenate((original_interfaces, jnp.full_like(top_pressure, 1.0e-3)), axis=-1)
    p_ext = jnp.concatenate((state.p, top_pressure), axis=-1)
    t_ext = jnp.concatenate((state.T, state.T[..., -1:]), axis=-1)
    t_interface = _temperature_interfaces(state.T)
    t_interface_ext = jnp.concatenate((t_interface, t_interface[..., -1:]), axis=-1)
    t_ext, t_interface_ext = _lw_buffer_temperatures(t_ext, t_interface_ext, pressure_interfaces)
    qv_ext = jnp.concatenate((state.qv, state.qv[..., -1:]), axis=-1)
    layer_mass = _pressure_layer_mass(state.p)
    layer_mass_ext = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) / GRAVITY, MIN_LAYER_MASS)
    _, pwvcm = _rrtmg_column_amounts(qv_ext, pressure_interfaces)
    mask = tables.lw_gpoint_mask

    secdiff = _lw_diffusivity(pwvcm)
    coef = _lw_setcoef(qv_ext, p_ext, t_ext, pressure_interfaces, tables)
    cldfmc, taucmc = _lw_cldprmc_state(state, p_ext, layer_mass_ext, tables)
    planklay, planklev, plankbnd = _lw_planck_state(t_ext, t_interface_ext, state.surface_temperature, state.surface_emissivity, tables)
    if not build_taumol:
        # Chunked flux path: do NOT materialise the full `(..., nlay, 16, 16)`
        # `tau`/`fracs` stack here — the per-band driver builds one band's taumol
        # at a time inside its band-scan (the dominant LW fp64 VRAM consumer is
        # then a single band's `(..., nlay, 16)` rather than the 16-band stack).
        # `coef` is returned so the driver can call `_lw_taumol_band` lazily.
        return state, coef, secdiff, planklay, planklev, plankbnd, cldfmc, taucmc, original_layers, layer_mass
    branch_tau, branch_fracs = _lw_taumol_fused(coef, tables)
    # VRAM: `_LW_TAUMOL_BRANCH_ACCEPTED` is statically all-True (the branch path
    # is always accepted), so the nearest-pressure fallback was a dead
    # `(..., nlay, 16, 16)` fp64 DUPLICATE of `tau`/`fracs` (built, then thrown
    # away by `jnp.where(True, branch, fallback)`).  Skip building it entirely
    # when the acceptance mask is all-True — a knob-independent, byte-exact VRAM
    # win (`where(True, a, b) == a`).  The fallback merge is retained verbatim for
    # the defensive case where some band is ever marked rejected.
    if all(bool(x) for x in _LW_TAUMOL_BRANCH_ACCEPTED):
        tau = branch_tau
        fracs = branch_fracs
    else:
        fallback_tau, fallback_fracs = _lw_fallback_taumol(qv_ext, p_ext, pressure_interfaces, tables)
        accepted = jnp.asarray(_LW_TAUMOL_BRANCH_ACCEPTED, dtype=bool)
        accepted = accepted.reshape((1,) * (branch_tau.ndim - 2) + (16, 1))
        tau = jnp.where(accepted, branch_tau, fallback_tau)
        fracs = jnp.where(accepted, branch_fracs, fallback_fracs)
    transfer_tau = jnp.clip(tau, 0.0, MAX_OPTICAL_DEPTH) * mask
    dplankup = planklev[..., 1:, :] - planklay
    dplankdn = planklev[..., :-1, :] - planklay
    intermediate_base = RRTMGLWIntermediateState(
        tau=tau,
        fracs=fracs,
        secdiff=secdiff,
        planklay=planklay,
        planklev=planklev,
        plankbnd=plankbnd,
        dplankup=dplankup,
        dplankdn=dplankdn,
        cldprmc_cldfmc=cldfmc,
        cldprmc_taucmc=taucmc,
        rtrnmc_pfracs=fracs,
        rtrnmc_plansum=jnp.zeros_like(planklay),
        rtrnmc_tfn_tbl_output=jnp.zeros_like(fracs),
        rtrnmc_zfd_per_gpoint=jnp.zeros(fracs.shape[:-3] + (fracs.shape[-3] + 1, 16, 16), dtype=fracs.dtype),
        rtrnmc_zfu_per_gpoint=jnp.zeros(fracs.shape[:-3] + (fracs.shape[-3] + 1, 16, 16), dtype=fracs.dtype),
    )
    return state, intermediate_base, cldfmc, taucmc, transfer_tau, original_layers, layer_mass


def _lw_solver_state(
    state: RRTMGLWColumnState, tables: RRTMGTableBundle
) -> tuple[RRTMGLWColumnState, RRTMGLWIntermediateState, int, jnp.ndarray]:
    """Oracle path: full per-g-point `rtrnmc` outputs for WRF intermediate parity."""

    state, intermediate_base, cldfmc, taucmc, transfer_tau, original_layers, layer_mass = _lw_solver_base(state, tables)
    tfn_output, zfd_per_gpoint, zfu_per_gpoint, plansum = _lw_rtrnmc_outputs(
        state, intermediate_base, cldfmc, taucmc, transfer_tau, tables
    )
    intermediate = intermediate_base._replace(
        rtrnmc_plansum=plansum,
        rtrnmc_tfn_tbl_output=tfn_output,
        rtrnmc_zfd_per_gpoint=zfd_per_gpoint,
        rtrnmc_zfu_per_gpoint=zfu_per_gpoint,
    )
    return state, intermediate, original_layers, layer_mass


def _lw_solver_fluxes_chunked(
    state: RRTMGLWColumnState, tables: RRTMGTableBundle, with_clear_sky: bool = False
):
    """Band-chunked operational LW flux path: taumol built per-band in a scan.

    Identical numerics to `_lw_solver_fluxes` (the per-band rtrnmc solve is the
    shared `_lw_rtrnmc_band_fluxes`; the per-band taumol is the shared
    `_lw_taumol_band`), but the full `(..., nlay, 16, 16)` `tau`/`fracs` stack is
    never built: a `lax.scan` over the 16 bands carries only the fp64
    band-summed down/up (and optional clear-sky down/up) interface fluxes, and
    each band's `(..., nlay, 16)` taumol + rtrnmc temporaries are freed by the
    scan carry barrier before the next band.  Disjoint per-band g-point sums are
    accumulated in fp64, so the result is bit-identical to the upfront-stack path
    (proofs/v013).
    """

    state, coef, secdiff, planklay, planklev, plankbnd, cldfmc, taucmc, original_layers, layer_mass = _lw_solver_base(
        state, tables, build_taumol=False
    )
    nt = _native_lw_tables()
    cloud_layer = jnp.any(cldfmc > 0.5, axis=(-1, -2))
    scale_band = tables.lw_delwave * (jnp.pi * 1.0e4)
    nlay = int(planklay.shape[-2])

    # Per-band taumol resolved by `lax.switch` over the traced band index: the
    # gas chemistry differs structurally per band, so each branch is the static
    # `_lw_taumol_band(b, ...)` closure (uniform `(..., nlay, 16)` output).
    taumol_branches = [
        (lambda b=b: _lw_taumol_band(b, coef, nt, tables)) for b in range(16)
    ]

    flux_shape = planklay.shape[:-2] + (nlay + 1,)
    zero_flux = jnp.zeros(flux_shape, dtype=jnp.float64)
    n_acc = 4 if with_clear_sky else 2
    init = tuple(zero_flux for _ in range(n_acc))

    def body(carry, band):
        tau_b, frac_b = lax.switch(band, taumol_branches)
        cldf_b = jnp.take(cldfmc, band, axis=-2)
        taucmc_b = jnp.take(taucmc, band, axis=-2)
        sec_raw = jnp.take(secdiff, band, axis=-1)
        scale = scale_band[band]
        valid = tables.lw_gpoint_mask[band]
        plank_b = jnp.take(planklay, band, axis=-1)
        planklev_b = jnp.take(planklev, band, axis=-1)
        plankbnd_b = jnp.take(plankbnd, band, axis=-1)
        zfd, zfu, zcd, zcu, _ = _lw_rtrnmc_band_fluxes(
            state, tau_b, frac_b, cldf_b, taucmc_b, sec_raw, scale, valid,
            plank_b, planklev_b, plankbnd_b, cloud_layer, with_clear_sky,
        )
        # Reduce this band over g-points (fp64) and add to the running fp64
        # band-sum — same disjoint-band fp64 accumulation as `_flush_tile`.
        down_part = jnp.sum(zfd, axis=-1)
        up_part = jnp.sum(zfu, axis=-1)
        if with_clear_sky:
            clear_down_part = jnp.sum(zcd, axis=-1)
            clear_up_part = jnp.sum(zcu, axis=-1)
            parts = (down_part, up_part, clear_down_part, clear_up_part)
        else:
            parts = (down_part, up_part)
        return tuple(acc + part for acc, part in zip(carry, parts)), None

    accumulated, _ = lax.scan(body, init, jnp.arange(16, dtype=jnp.int32))
    if with_clear_sky:
        flux_down_model, flux_up_model, clear_down, clear_up = accumulated
        return state, flux_down_model, flux_up_model, original_layers, layer_mass, clear_down, clear_up
    flux_down_model, flux_up_model = accumulated
    return state, flux_down_model, flux_up_model, original_layers, layer_mass


def _lw_solver_fluxes(
    state: RRTMGLWColumnState, tables: RRTMGTableBundle, with_clear_sky: bool = False
) -> tuple[RRTMGLWColumnState, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    """Operational path: band-summed LW fluxes without the full per-g-point array.

    Accumulates the (band, g-point)-summed up/down fluxes one band at a time in
    fp64, so the peak g-point temporary is a single band's `(..., nlay+1, 16)`
    buffer rather than the full `(..., nlay+1, 16, 16)` stack — the dominant LW
    fp64 VRAM consumer.  Bit-comparable (fp64) to the oracle path's
    `sum(zf*_per_gpoint, axis=(-1, -2))`.  When ``with_clear_sky`` the parallel
    clear-sky (`totdclfl`/`totuclfl`) band-summed fluxes are also returned.
    """

    if _LW_TAUMOL_CHUNK:
        return _lw_solver_fluxes_chunked(state, tables, with_clear_sky=with_clear_sky)
    state, intermediate_base, cldfmc, taucmc, transfer_tau, original_layers, layer_mass = _lw_solver_base(state, tables)
    outputs = _lw_rtrnmc_outputs(
        state, intermediate_base, cldfmc, taucmc, transfer_tau, tables, flux_only=True, with_clear_sky=with_clear_sky
    )
    if with_clear_sky:
        flux_down_model, flux_up_model, _, clear_down, clear_up = outputs
        return state, flux_down_model, flux_up_model, original_layers, layer_mass, clear_down, clear_up
    flux_down_model, flux_up_model, _ = outputs
    return state, flux_down_model, flux_up_model, original_layers, layer_mass


def _longwave_impl(
    state: RRTMGLWColumnState, tables: RRTMGTableBundle, debug: bool, with_clear_sky: bool = False
) -> RRTMGLWColumnResult:
    """Unjitted LW implementation shared by production and stripped paths."""

    if with_clear_sky:
        state, flux_down_model, flux_up_model, original_layers, layer_mass, clear_flux_down, clear_flux_up = _lw_solver_fluxes(
            state, tables, with_clear_sky=True
        )
    else:
        state, flux_down_model, flux_up_model, original_layers, layer_mass = _lw_solver_fluxes(state, tables)
        clear_flux_down = None
        clear_flux_up = None
    net_down = flux_down_model - flux_up_model
    layer_net_heating = net_down[..., 1 : original_layers + 1] - net_down[..., :original_layers]
    heating_rate = layer_net_heating / (layer_mass * CP_AIR)
    surface_emission = STEFAN_BOLTZMANN * state.surface_emissivity * state.surface_temperature**4
    flux_down = flux_down_model
    flux_up = flux_up_model

    heating_rate = assert_finite(heating_rate, "rrtmg_lw.heating_rate", enabled=debug)
    flux_down = assert_physical_bounds(flux_down, 0.0, 2000.0, "rrtmg_lw.flux_down", enabled=debug)
    flux_up = assert_physical_bounds(flux_up, 0.0, 2000.0, "rrtmg_lw.flux_up", enabled=debug)
    return RRTMGLWColumnResult(
        heating_rate=heating_rate,
        flux_down=flux_down,
        flux_up=flux_up,
        toa_down=flux_down[..., -1],
        toa_up=flux_up[..., -1],
        surface_down=flux_down[..., 0],
        surface_up=flux_up[..., 0],
        column_net_heating=jnp.sum(layer_net_heating, axis=-1),
        surface_emission=surface_emission,
        clear_flux_down=clear_flux_down,
        clear_flux_up=clear_flux_up,
    )


def _longwave_column_tiled_impl(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle,
    debug: bool,
    with_clear_sky: bool = False,
) -> RRTMGLWColumnResult:
    """Runs the LW solve over fixed-size flattened column tiles."""

    if not _LW_COLUMN_TILING or _LW_COLUMN_TILE_COLS <= 0:
        return _longwave_impl(state, tables, debug, with_clear_sky)

    leading_shape = state.p.shape[:-1]
    ncol = _column_count(leading_shape)
    tile_cols = min(max(int(_LW_COLUMN_TILE_COLS), 1), ncol)
    n_tiles = (ncol + tile_cols - 1) // tile_cols
    padded_ncol = n_tiles * tile_cols
    nlayers = state.p.shape[-1]
    out_dtype = jnp.result_type(state.p.dtype, jnp.float64)

    flat_state = _flatten_lw_state(state, leading_shape, ncol)
    padded_state = _pad_lw_state(flat_state, ncol, padded_ncol)
    init = _zero_lw_column_result(padded_ncol, nlayers, out_dtype, with_clear_sky)

    def body(carry, tile_index):
        start = tile_index * tile_cols
        tile_state = _slice_lw_state(padded_state, start, tile_cols, padded_ncol)
        tile_result = _longwave_impl(tile_state, tables, debug, with_clear_sky)
        return _scatter_lw_result(carry, tile_result, start), None

    tiled, _ = lax.scan(body, init, jnp.arange(n_tiles, dtype=jnp.int32))
    return _unflatten_lw_result(tiled, leading_shape, ncol)


def compute_rrtmg_lw_intermediates(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGLWIntermediateState:
    """Returns the JAX LW state compared to WRF `rtrnmc` entry oracles."""

    _, intermediate, _, _ = _lw_solver_state(state, tables)
    return intermediate


@partial(jax.jit, static_argnames=("debug", "with_clear_sky"))
def solve_rrtmg_lw_column(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
    with_clear_sky: bool = False,
) -> RRTMGLWColumnResult:
    """Computes one fused longwave column radiation call.

    When ``with_clear_sky`` is True the result also carries the WRF clear-sky
    (cloud-free) interface fluxes ``clear_flux_down``/``clear_flux_up`` (the
    parallel clear-sky `rtrnmc` stream, WRF ``totdclfl``/``totuclfl``); the main
    all-sky flux outputs are byte-identical regardless of this flag.
    """

    return _longwave_column_tiled_impl(state, tables, debug, with_clear_sky)


@jax.jit
def solve_rrtmg_lw_column_debug_stripped(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGLWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _longwave_impl(state, tables, False)
