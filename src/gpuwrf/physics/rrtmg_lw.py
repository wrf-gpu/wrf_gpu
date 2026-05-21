"""JAX longwave RRTMG-style radiation column kernel for M5-S3."""

from __future__ import annotations

import importlib.util
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


_LW_GPOINT_COUNTS = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
_LW_NSPA = np.asarray([1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9], dtype=np.int32)
_LW_NSPB = np.asarray([1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.int32)
_CFC_VMR = np.asarray([0.093e-9, 0.251e-9, 0.538e-9, 0.169e-9], dtype=np.float64)
_ONEMINUS = 1.0 - 1.0e-6
_LW_TAUMOL_BRANCH_ACCEPTED = (True,) * 16


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
    wbroad = coldry * jnp.maximum(0.0, 1.0 - (CO2_VMR + O3_BACKGROUND_VMR + N2O_VMR + CH4_VMR + O2_VMR))
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
    colo3 = 1.0e-20 * coldry * O3_BACKGROUND_VMR
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


def _lw_taumol(coef: _LWSetCoefState, tables: RRTMGTableBundle):
    """Ports WRF LW `taumol` reduced-g-point gas optical depth and fractions."""

    nt = _native_lw_tables()
    chi = nt.chi_mls
    taug = []
    fracs = []

    def chi_ratio(a_1b, b_1b, level_1b):
        return chi[a_1b - 1, level_1b - 1] / chi[b_1b - 1, level_1b - 1]

    def chi_layer(gas_1b):
        return jnp.take(chi[gas_1b - 1], coef.jp, axis=0)

    for band in range(16):
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

        taug.append(tau * tables.lw_gpoint_mask[band])
        fracs.append(frac * tables.lw_gpoint_mask[band])

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


def _lw_solver_state(
    state: RRTMGLWColumnState, tables: RRTMGTableBundle
) -> tuple[RRTMGLWColumnState, RRTMGLWIntermediateState, jnp.ndarray, int, jnp.ndarray, jnp.ndarray]:
    """Builds LW gas/source state before transfer; WRF formulas: `rtrnmc` lines 3253-3409."""

    state = _clip_state(state)
    original_layers = state.p.shape[-1]
    original_interfaces = _pressure_interfaces(state.p)
    top_pressure = 0.5 * original_interfaces[..., -1:]
    pressure_interfaces = jnp.concatenate((original_interfaces, jnp.full_like(top_pressure, 1.0e-3)), axis=-1)
    p_ext = jnp.concatenate((state.p, top_pressure), axis=-1)
    t_ext = jnp.concatenate((state.T, state.T[..., -1:]), axis=-1)
    t_interface = _temperature_interfaces(state.T)
    t_interface_ext = jnp.concatenate((t_interface, t_interface[..., -1:]), axis=-1)
    qv_ext = jnp.concatenate((state.qv, state.qv[..., -1:]), axis=-1)
    qc_ext = jnp.concatenate((state.qc, jnp.zeros_like(state.qc[..., -1:])), axis=-1)
    qi_ext = jnp.concatenate((state.qi, jnp.zeros_like(state.qi[..., -1:])), axis=-1)
    qs_ext = jnp.concatenate((state.qs, jnp.zeros_like(state.qs[..., -1:])), axis=-1)
    qg_ext = jnp.concatenate((state.qg, jnp.zeros_like(state.qg[..., -1:])), axis=-1)
    cloud_ext = jnp.concatenate((state.cloud_fraction, jnp.zeros_like(state.cloud_fraction[..., -1:])), axis=-1)
    layer_mass = _pressure_layer_mass(state.p)
    layer_mass_ext = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) / GRAVITY, MIN_LAYER_MASS)
    _, pwvcm = _rrtmg_column_amounts(qv_ext, pressure_interfaces)
    cloud_path_g = (qc_ext + qi_ext + qs_ext + qg_ext) * layer_mass_ext * 1000.0 * cloud_ext

    cloud_coeff = tables.lw_cloud_absorption[:, None]
    mask = tables.lw_gpoint_mask

    secdiff = _lw_diffusivity(pwvcm)
    coef = _lw_setcoef(qv_ext, p_ext, t_ext, pressure_interfaces, tables)
    branch_tau, branch_fracs = _lw_taumol_fused(coef, tables)
    fallback_tau, fallback_fracs = _lw_fallback_taumol(qv_ext, p_ext, pressure_interfaces, tables)
    accepted = jnp.asarray(_LW_TAUMOL_BRANCH_ACCEPTED, dtype=bool)
    accepted = accepted.reshape((1,) * (branch_tau.ndim - 2) + (16, 1))
    tau = jnp.where(accepted, branch_tau, fallback_tau)
    fracs = jnp.where(accepted, branch_fracs, fallback_fracs)
    cloud_tau = cloud_path_g[..., None, None] * cloud_coeff
    transfer_tau = jnp.clip(tau + cloud_tau, 0.0, MAX_OPTICAL_DEPTH) * mask
    optical_path = transfer_tau * secdiff[..., None, :, None]
    trans = jnp.exp(-jnp.minimum(jnp.maximum(optical_path, MIN_OPTICAL_DEPTH), MAX_OPTICAL_DEPTH))
    trans = jnp.where(mask > 0.0, trans, 1.0)
    planklay, planklev, plankbnd = _lw_planck_state(t_ext, t_interface_ext, state.surface_temperature, state.surface_emissivity, tables)
    dplankup = planklev[..., 1:, :] - planklay
    dplankdn = planklev[..., :-1, :] - planklay
    intermediate = RRTMGLWIntermediateState(
        tau=tau,
        fracs=fracs,
        secdiff=secdiff,
        planklay=planklay,
        planklev=planklev,
        plankbnd=plankbnd,
        dplankup=dplankup,
        dplankdn=dplankdn,
    )
    return state, intermediate, trans, original_layers, layer_mass, transfer_tau


def _longwave_impl(state: RRTMGLWColumnState, tables: RRTMGTableBundle, debug: bool) -> RRTMGLWColumnResult:
    """Unjitted LW implementation shared by production and stripped paths."""

    state, intermediate, trans, original_layers, layer_mass, transfer_tau = _lw_solver_state(state, tables)
    mask = tables.lw_gpoint_mask
    band_frac = intermediate.fracs
    planck_scale = tables.lw_delwave * (jnp.pi * 1.0e4)
    tfac = _lw_tfn_factor(transfer_tau * intermediate.secdiff[..., None, :, None])
    down_source = (intermediate.planklay[..., :, None] + intermediate.dplankdn[..., :, None] * tfac) * planck_scale[..., None] * band_frac
    up_source = (intermediate.planklay[..., :, None] + intermediate.dplankup[..., :, None] * tfac) * planck_scale[..., None] * band_frac
    surface_emission_band = intermediate.plankbnd[..., :, None] * planck_scale[..., None] * mask / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)[:, None]

    down_band = _source_recurrence_down(trans, down_source)
    surface_reflectance = (1.0 - state.surface_emissivity[..., None, None]) * down_band[..., 0, :, :]
    up_band = _source_recurrence_up(trans, up_source, surface_emission_band + surface_reflectance)

    flux_up_model = jnp.sum(up_band, axis=(-1, -2))
    flux_down_model = jnp.sum(down_band, axis=(-1, -2))
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
    )


def compute_rrtmg_lw_intermediates(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGLWIntermediateState:
    """Returns the JAX LW state compared to WRF `rtrnmc` entry oracles."""

    _, intermediate, _, _, _, _ = _lw_solver_state(state, tables)
    return intermediate


@partial(jax.jit, static_argnames=("debug",))
def solve_rrtmg_lw_column(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
) -> RRTMGLWColumnResult:
    """Computes one fused longwave column radiation call."""

    return _longwave_impl(state, tables, debug)


@jax.jit
def solve_rrtmg_lw_column_debug_stripped(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGLWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _longwave_impl(state, tables, False)
