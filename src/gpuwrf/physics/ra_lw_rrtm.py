"""Classic WRF RRTM longwave column endpoint (ra_lw_physics=1).

This is a CPU-only, column-local port of the legacy WRF
``phys/module_ra_rrtm.F`` path used by ``RRTMLWRAD``.  It follows the classic
RRTM 16-band, 140-g-point AER k-distribution path:

* parse the unmodified WRF ``RRTM_DATA_DBL``/``RRTM_DATA`` lookup records;
* reduce the original 16 g-points per band with WRF's ``rrtminit`` grouping;
* prepare the MM5/RRTM bottom-up column, including WRF's buffer layers above
  ``p_top``;
* evaluate ``TAUGB1..16`` and the legacy ``RTRN`` one-angle transfer.

The public result returns WRF's held-rate temperature tendency ``TTEN`` in
model order.  The radiation driver converts that to potential-temperature
rate as ``RTHRATEN = TTEN / pi``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import struct
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.config.paths import wrf_root


ROOT = Path(__file__).resolve().parents[3]
# Pristine WRF tree holding the unmodified RRTM source + lookup data.
# Env-overridable via GPUWRF_WRF_ROOT (config.paths.wrf_root). These module-level
# names are convenience snapshots; _load_tables() re-reads wrf_root() at call time
# so a late os.environ set still wins.
WRF_ROOT = wrf_root()
RRTM_SOURCE = WRF_ROOT / "phys" / "module_ra_rrtm.F"
RRTM_DATA_DBL = WRF_ROOT / "run" / "RRTM_DATA_DBL"
RRTM_DATA = WRF_ROOT / "run" / "RRTM_DATA"

NGPT = 140
NBANDS = 16
MG = 16
DELTAP_MB = 4.0
DEFAULT_PTOP_PA = 5000.0
ONEMINUS = 1.0 - 1.0e-6
SECANG = 1.66
WTNUM = 0.5

NGC = np.asarray([8, 14, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2], dtype=np.int32)
NGS = np.asarray([8, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138, 140], dtype=np.int32)
NGB = np.repeat(np.arange(NBANDS, dtype=np.int32), NGC)
NGB_START = np.concatenate(([0], np.cumsum(NGC)[:-1]))
DELWAVE = np.asarray(
    [240.0, 250.0, 130.0, 70.0, 120.0, 160.0, 100.0, 100.0,
     210.0, 90.0, 320.0, 280.0, 170.0, 130.0, 220.0, 400.0],
    dtype=np.float64,
)
NSPA = np.asarray([1, 1, 10, 9, 9, 1, 9, 1, 11, 1, 1, 9, 9, 1, 9, 9], dtype=np.int32)
NSPB = np.asarray([1, 1, 5, 6, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=np.int32)
GROUPS = (
    (2, 2, 2, 2, 2, 2, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    (2, 2, 2, 2, 4, 4),
    (1, 1, 2, 2, 2, 2, 3, 3),
    (1, 1, 1, 1, 2, 2, 4, 4),
    (3, 3, 4, 6),
    (8, 8),
    (8, 8),
    (8, 8),
)
ORIGINAL_GPOINT_WEIGHTS = np.asarray(
    [
        0.1527534276,
        0.1491729617,
        0.1420961469,
        0.1316886544,
        0.1181945205,
        0.1019300893,
        0.0832767040,
        0.0626720116,
        0.0424925,
        0.0046269894,
        0.0038279891,
        0.0030260086,
        0.0022199750,
        0.0014140010,
        0.000533,
        0.000075,
    ],
    dtype=np.float64,
)


class RRTMLWColumnState(NamedTuple):
    """Single-column classic RRTM inputs in model order.

    Arrays are ``(ncol, nz)`` with ``k=0`` the lowest mass layer.  ``p`` and
    ``p8w`` are Pa, matching the WRF savepoints.  ``t8w``/``p8w`` have
    ``nz + 1`` interface entries.
    """

    T: jnp.ndarray
    t8w: jnp.ndarray
    p: jnp.ndarray
    p8w: jnp.ndarray
    qv: jnp.ndarray
    qc: jnp.ndarray
    qr: jnp.ndarray
    qi: jnp.ndarray
    qs: jnp.ndarray
    qg: jnp.ndarray
    cloud_fraction: jnp.ndarray
    dz: jnp.ndarray
    rho: jnp.ndarray
    emiss: float | jnp.ndarray
    tsk: float | jnp.ndarray
    # Grid model-top pressure (Pa), used to size the WRF above-model-top buffer
    # ``nbuf = nint(p_top_mb / deltap)`` exactly as ``module_ra_rrtm.F:6781``
    # (``NLAYERS = kme + nint(p_top*0.01/deltap) - 1``).  ``None`` falls back to
    # the legacy hardcoded ``DEFAULT_PTOP_PA`` so existing 5000-Pa callers are
    # bit-identical.  It is a STATIC float (it sets array shapes), never traced.
    top_pressure_pa: float | None = None


class RRTMLWColumnResult(NamedTuple):
    heating_rate: jnp.ndarray
    glw: jnp.ndarray
    olr: jnp.ndarray
    flux_down: jnp.ndarray
    flux_up: jnp.ndarray


class _RRTMTables(NamedTuple):
    absa: tuple[np.ndarray, ...]
    absb: tuple[np.ndarray | None, ...]
    selfref: tuple[np.ndarray | None, ...]
    forref: tuple[np.ndarray | None, ...]
    fracrefa: tuple[np.ndarray, ...]
    fracrefb: tuple[np.ndarray | None, ...]
    minor: dict[str, np.ndarray]
    pref: np.ndarray
    preflog: np.ndarray
    tref: np.ndarray
    delwave: np.ndarray
    totplnk: np.ndarray
    tau: np.ndarray
    tf: np.ndarray
    trans: np.ndarray
    corr1: np.ndarray
    corr2: np.ndarray
    bpade: float
    fluxfac: float
    heatfac: float
    local: dict[str, np.ndarray]


RECORD_SPECS = (
    (("abscoefL1", (5, 13, 16)), ("abscoefH1", (5, 47, 16)), ("SELFREF1", (10, 16))),
    (("abscoefL2", (5, 13, 16)), ("abscoefH2", (5, 47, 16)), ("SELFREF2", (10, 16))),
    (("abscoefL3", (10, 5, 13, 16)), ("abscoefH3", (5, 5, 47, 16)), ("SELFREF3", (10, 16))),
    (("abscoefL4", (9, 5, 13, 16)), ("abscoefH4", (6, 5, 47, 16)), ("SELFREF4", (10, 16))),
    (("abscoefL5", (9, 5, 13, 16)), ("abscoefH5", (5, 5, 47, 16)), ("SELFREF5", (10, 16))),
    (("abscoefL6", (5, 13, 16)), ("SELFREF6", (10, 16))),
    (("abscoefL7", (9, 5, 13, 16)), ("abscoefH7", (5, 47, 16)), ("SELFREF7", (10, 16))),
    (("abscoefL8", (5, 7, 16)), ("abscoefH8", (5, 53, 16)), ("SELFREF8", (10, 16))),
    (("abscoefL9", (11, 5, 13, 16)), ("abscoefH9", (5, 47, 16)), ("SELFREF9", (10, 16))),
    (("abscoefL10", (5, 13, 16)), ("abscoefH10", (5, 47, 16))),
    (("abscoefL11", (5, 13, 16)), ("abscoefH11", (5, 47, 16)), ("SELFREF11", (10, 16))),
    (("abscoefL12", (9, 5, 13, 16)), ("SELFREF12", (10, 16))),
    (("abscoefL13", (9, 5, 13, 16)), ("SELFREF13", (10, 16))),
    (("abscoefL14", (5, 13, 16)), ("abscoefH14", (5, 47, 16)), ("SELFREF14", (10, 16))),
    (("abscoefL15", (9, 5, 13, 16)), ("SELFREF15", (10, 16))),
    (("abscoefL16", (9, 5, 13, 16)), ("SELFREF16", (10, 16))),
)


def _read_unformatted_records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    records: list[bytes] = []
    offset = 0
    while offset < len(data):
        (nbytes,) = struct.unpack(">i", data[offset : offset + 4])
        payload = data[offset + 4 : offset + 4 + nbytes]
        (tail,) = struct.unpack(">i", data[offset + 4 + nbytes : offset + 8 + nbytes])
        if tail != nbytes:
            raise ValueError(f"{path} has mismatched record markers at byte {offset}")
        records.append(payload)
        offset += 8 + nbytes
    if len(records) != 16:
        raise ValueError(f"{path} has {len(records)} records, expected 16")
    return records


def _parse_records(path: Path) -> list[dict[str, np.ndarray]]:
    records = _read_unformatted_records(path)
    itemsize = len(records[0]) // sum(int(np.prod(shape)) for _, shape in RECORD_SPECS[0])
    dtype = ">f8" if itemsize == 8 else ">f4"
    parsed_records: list[dict[str, np.ndarray]] = []
    for payload, spec in zip(records, RECORD_SPECS, strict=True):
        offset = 0
        parsed: dict[str, np.ndarray] = {}
        for name, shape in spec:
            count = int(np.prod(shape))
            nbytes = count * itemsize
            parsed[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).astype(np.float64).reshape(shape, order="F")
            offset += nbytes
        if offset != len(payload):
            raise ValueError(f"record parse for {path} consumed {offset}, expected {len(payload)}")
        parsed_records.append(parsed)
    return parsed_records


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("!")[0] for line in text.splitlines())


def _parse_fortran_numbers(body: str) -> np.ndarray:
    body = body.replace("&", " ").replace("_rb", "").replace("D", "E").replace("d", "e")
    out: list[float] = []
    for raw in re.split(r"[\s,]+", body):
        token = raw.strip()
        if not token:
            continue
        repeated = re.fullmatch(r"(\d+)\*(.+)", token)
        if repeated:
            out.extend([float(repeated.group(2))] * int(repeated.group(1)))
        else:
            out.append(float(token))
    return np.asarray(out, dtype=np.float64)


def _data_values(text: str, name: str) -> np.ndarray:
    pattern = re.compile(rf"\bDATA\s+{re.escape(name)}\s*/(?P<body>.*?)/", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing DATA {name}")
    return _parse_fortran_numbers(match.group("body"))


def _subroutine_text(text: str, name: str) -> str:
    pattern = re.compile(
        rf"\bSUBROUTINE\s+{re.escape(name)}\b(?P<body>.*?)\bEND\s+SUBROUTINE\s+{re.escape(name)}\b",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing subroutine {name}")
    return match.group("body")


def _data_array(text: str, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = _data_values(text, name)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"DATA {name} has {values.size} values, expected {expected}")
    return values.reshape(shape, order="F")


def _parse_totplnk(text: str) -> np.ndarray:
    out = np.zeros((181, NBANDS), dtype=np.float64)
    pattern = re.compile(
        r"DATA\s+\(TOTPLNK\(IDATA,\s*(\d+)\),IDATA=(\d+),(\d+)\)\s*/(?P<body>.*?)/",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        band = int(match.group(1)) - 1
        start = int(match.group(2)) - 1
        end = int(match.group(3))
        values = _parse_fortran_numbers(match.group("body"))
        out[start:end, band] = values
    if not np.any(out):
        raise ValueError("failed to parse TOTPLNK table")
    return out


def _reduce_last(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
    pieces = []
    start = 0
    for group_size in groups:
        end = start + group_size
        seg = values[..., start:end]
        if weighted:
            weights = ORIGINAL_GPOINT_WEIGHTS[start:end]
            weights = weights / np.sum(weights)
            pieces.append(np.sum(seg * weights, axis=-1))
        else:
            pieces.append(np.sum(seg, axis=-1))
        start = end
    return np.stack(pieces, axis=-1)


def _reduce_first(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
    moved = np.moveaxis(values, 0, -1)
    reduced = _reduce_last(moved, groups, weighted=weighted)
    return np.moveaxis(reduced, -1, 0)


def _flatten_fortran(values: np.ndarray) -> np.ndarray:
    ng = values.shape[-1]
    return values.reshape((int(np.prod(values.shape[:-1])), ng), order="F")


def _frac(text: str, name: str, band: int, shape: tuple[int, ...]) -> np.ndarray:
    return _reduce_first(_data_array(text, name, shape), GROUPS[band], weighted=False)


def _minor_vec(text: str, name: str, band: int, shape: tuple[int, ...] = (16,)) -> np.ndarray:
    return _reduce_first(_data_array(text, name, shape), GROUPS[band], weighted=True)


@lru_cache(maxsize=1)
def _load_tables() -> _RRTMTables:
    # Re-resolve under wrf_root() so GPUWRF_WRF_ROOT set after import still wins.
    root = wrf_root()
    rrtm_source = root / "phys" / "module_ra_rrtm.F"
    rrtm_data_dbl = root / "run" / "RRTM_DATA_DBL"
    rrtm_data = root / "run" / "RRTM_DATA"
    source_text = _strip_comments(rrtm_source.read_text(encoding="utf-8", errors="replace"))
    data_path = rrtm_data_dbl if rrtm_data_dbl.exists() else rrtm_data
    parsed = _parse_records(data_path)

    absa: list[np.ndarray] = []
    absb: list[np.ndarray | None] = []
    selfref: list[np.ndarray | None] = []
    for band, rec in enumerate(parsed):
        lower = rec[f"abscoefL{band + 1}"]
        absa.append(_flatten_fortran(_reduce_last(lower, GROUPS[band], weighted=True)))
        hname = f"abscoefH{band + 1}"
        absb.append(_flatten_fortran(_reduce_last(rec[hname], GROUPS[band], weighted=True)) if hname in rec else None)
        sname = f"SELFREF{band + 1}"
        selfref.append(_reduce_last(rec[sname], GROUPS[band], weighted=True) if sname in rec else None)

    forref: list[np.ndarray | None] = [None] * NBANDS
    fracrefa: list[np.ndarray] = [np.empty((0,)) for _ in range(NBANDS)]
    fracrefb: list[np.ndarray | None] = [None] * NBANDS
    for band in range(NBANDS):
        b = band + 1
        if b in (1,):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16,))
            fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16,))
            forref[band] = _minor_vec(source_text, f"FORREF{b}", band)
        elif b in (2,):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16, 13))
            fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16,))
            forref[band] = _minor_vec(source_text, f"FORREF{b}", band)
        elif b in (3,):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16, 10))
            fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16, 5))
            forref[band] = _minor_vec(source_text, f"FORREF{b}", band)
        elif b in (4, 5):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16, 9))
            fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16, 6 if b == 4 else 5))
        elif b in (6, 8, 10, 11, 14):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16,))
            if b in (8, 10, 11, 14):
                fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16,))
        elif b in (7, 9, 12, 13, 15, 16):
            fracrefa[band] = _frac(source_text, f"FRACREFA{b}", band, (16, 9))
            if b in (7, 9):
                fracrefb[band] = _frac(source_text, f"FRACREFB{b}", band, (16,))

    minor = {
        "ABSN2OAC3": _minor_vec(source_text, "ABSN2OA3", 2),
        "ABSN2OBC3": _minor_vec(source_text, "ABSN2OB3", 2),
        "CCL4C5": _minor_vec(source_text, "CCL45", 4),
        "ABSCO2C6": _minor_vec(source_text, "ABSCO26", 5),
        "CFC11ADJC6": _minor_vec(source_text, "CFC11ADJ6", 5),
        "CFC12C6": _minor_vec(source_text, "CFC126", 5),
        "ABSCO2C7": _minor_vec(source_text, "ABSCO27", 6),
        "ABSCO2AC8": _minor_vec(source_text, "ABSCO2A8", 7),
        "ABSCO2BC8": _minor_vec(source_text, "ABSCO2B8", 7),
        "ABSN2OAC8": _minor_vec(source_text, "ABSN2OA8", 7),
        "ABSN2OBC8": _minor_vec(source_text, "ABSN2OB8", 7),
        "CFC12C8": _minor_vec(source_text, "CFC128", 7),
        "CFC22ADJC8": _minor_vec(source_text, "CFC22ADJ8", 7),
    }
    absn2o9 = _reduce_first(_data_array(source_text, "ABSN2O9", (16, 3)), GROUPS[8], weighted=True)
    minor["ABSN2OC9"] = np.concatenate([absn2o9[:, 0], absn2o9[:, 1], absn2o9[:, 2]])

    local: dict[str, np.ndarray] = {}
    for sub, names in {
        "TAUGB2": ("REFPARAM",),
        "TAUGB3": ("ETAREF", "H2OREF", "N2OREF", "CO2REF"),
        "TAUGB8": ("H2OREF", "N2OREF", "O3REF"),
        "TAUGB9": ("ETAREF", "H2OREF", "N2OREF", "CH4REF"),
    }.items():
        subtext = _subroutine_text(source_text, sub)
        for name in names:
            local[f"{sub}_{name}"] = _data_values(subtext, name)

    pref = _data_values(source_text, "PREF")
    preflog = _data_values(source_text, "PREFLOG")
    tref = _data_values(source_text, "TREF")
    heatfac = float(_data_values(source_text, "HEATFAC")[0])
    tau, tf, trans, bpade = _lookup_tables()
    corr1, corr2 = _corr_tables()
    return _RRTMTables(
        absa=tuple(absa),
        absb=tuple(absb),
        selfref=tuple(selfref),
        forref=tuple(forref),
        fracrefa=tuple(fracrefa),
        fracrefb=tuple(fracrefb),
        minor=minor,
        pref=pref,
        preflog=preflog,
        tref=tref,
        delwave=DELWAVE.copy(),
        totplnk=_parse_totplnk(source_text),
        tau=tau,
        tf=tf,
        trans=trans,
        corr1=corr1,
        corr2=corr2,
        bpade=bpade,
        fluxfac=float(np.pi * 2.0e4),
        heatfac=heatfac,
        local=local,
    )


def _lookup_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    bpade = 1.0 / 0.278
    tau = np.zeros(5001, dtype=np.float64)
    tf = np.zeros(5001, dtype=np.float64)
    trans = np.zeros(5001, dtype=np.float64)
    tau[5000] = 1.0e10
    tf[5000] = 1.0
    trans[0] = 1.0
    for itre in range(1, 5000):
        tfn = itre / 5.0e3
        tau[itre] = bpade * tfn / (1.0 - tfn)
        trans[itre] = np.exp(-tau[itre])
        if tau[itre] < 0.1:
            tf[itre] = tau[itre] / 6.0
        else:
            tf[itre] = 1.0 - 2.0 * ((1.0 / tau[itre]) - (trans[itre] / (1.0 - trans[itre])))
    return tau, tf, trans, bpade


def _corr_tables() -> tuple[np.ndarray, np.ndarray]:
    corr1 = np.ones(201, dtype=np.float64)
    corr2 = np.ones(201, dtype=np.float64)
    for i in range(1, 200):
        fp = 0.005 * float(i)
        rtfp = np.sqrt(fp)
        corr1[i] = rtfp / fp
        corr2[i] = (1.0 - rtfp) / (1.0 - fp)
    return corr1, corr2


def _fint(x: float) -> int:
    return int(np.trunc(x))


def _nint(x: float) -> int:
    return int(np.floor(x + 0.5))


def _row(table: np.ndarray, idx_1b: int) -> np.ndarray:
    return table[max(0, min(table.shape[0] - 1, idx_1b - 1))]


def _interp4(table: np.ndarray, ind0: int, ind1: int, fac00: float, fac10: float, fac01: float, fac11: float) -> np.ndarray:
    return fac00 * _row(table, ind0) + fac10 * _row(table, ind0 + 1) + fac01 * _row(table, ind1) + fac11 * _row(table, ind1 + 1)


def _self_term(selfref: np.ndarray | None, indself: int, selffrac: float) -> np.ndarray:
    if selfref is None:
        raise ValueError("missing selfref table")
    base = _row(selfref, indself)
    return base + selffrac * (_row(selfref, indself + 1) - base)


def _binary_lower(table: np.ndarray, ind0: int, ind1: int, nsp: int, fs: float, fac00: float, fac10: float, fac01: float, fac11: float) -> np.ndarray:
    omf = 1.0 - fs
    return (
        omf * fac00 * _row(table, ind0)
        + fs * fac00 * _row(table, ind0 + 1)
        + omf * fac10 * _row(table, ind0 + nsp)
        + fs * fac10 * _row(table, ind0 + nsp + 1)
        + omf * fac01 * _row(table, ind1)
        + fs * fac01 * _row(table, ind1 + 1)
        + omf * fac11 * _row(table, ind1 + nsp)
        + fs * fac11 * _row(table, ind1 + nsp + 1)
    )


def _frac_interp(frac: np.ndarray, js: int, fs: float) -> np.ndarray:
    if frac.ndim == 1:
        return frac
    j0 = max(0, min(frac.shape[1] - 1, js - 1))
    j1 = max(0, min(frac.shape[1] - 1, js))
    return frac[:, j0] + fs * (frac[:, j1] - frac[:, j0])


def _column_inputs(state: RRTMLWColumnState):
    arrays = {
        name: np.asarray(getattr(state, name), dtype=np.float64)
        for name in ("T", "t8w", "p", "p8w", "qv", "qc", "qr", "qi", "qs", "qg", "cloud_fraction", "dz", "rho")
    }
    emiss = np.asarray(state.emiss, dtype=np.float64)
    tsk = np.asarray(state.tsk, dtype=np.float64)
    ncol = arrays["T"].shape[0]
    emiss = np.broadcast_to(emiss.reshape(-1) if emiss.ndim else emiss.reshape(1), (ncol,))
    tsk = np.broadcast_to(tsk.reshape(-1) if tsk.ndim else tsk.reshape(1), (ncol,))
    return arrays, emiss, tsk


def _o3_average_from_interfaces(pz_bottom_up: np.ndarray) -> np.ndarray:
    o3sum = np.asarray(
        [
            5.297e-8, 5.852e-8, 6.579e-8, 7.505e-8, 8.577e-8, 9.895e-8, 1.175e-7, 1.399e-7,
            1.677e-7, 2.003e-7, 2.571e-7, 3.325e-7, 4.438e-7, 6.255e-7, 8.168e-7, 1.036e-6,
            1.366e-6, 1.855e-6, 2.514e-6, 3.240e-6, 4.033e-6, 4.854e-6, 5.517e-6, 6.089e-6,
            6.689e-6, 1.106e-5, 1.462e-5, 1.321e-5, 9.856e-6, 5.960e-6, 5.960e-6,
        ],
        dtype=np.float64,
    )
    ppsum = np.asarray(
        [
            955.890, 850.532, 754.599, 667.742, 589.841, 519.421, 455.480, 398.085, 347.171, 301.735,
            261.310, 225.360, 193.419, 165.490, 141.032, 120.125, 102.689, 87.829, 75.123, 64.306,
            55.086, 47.209, 40.535, 34.795, 29.865, 19.122, 9.277, 4.660, 2.421, 1.294, 0.647,
        ],
        dtype=np.float64,
    )
    o3win = np.asarray(
        [
            4.629e-8, 4.686e-8, 5.017e-8, 5.613e-8, 6.871e-8, 8.751e-8, 1.138e-7, 1.516e-7,
            2.161e-7, 3.264e-7, 4.968e-7, 7.338e-7, 1.017e-6, 1.308e-6, 1.625e-6, 2.011e-6,
            2.516e-6, 3.130e-6, 3.840e-6, 4.703e-6, 5.486e-6, 6.289e-6, 6.993e-6, 7.494e-6,
            8.197e-6, 9.632e-6, 1.113e-5, 1.146e-5, 9.389e-6, 6.135e-6, 6.135e-6,
        ],
        dtype=np.float64,
    )
    ppwin = np.asarray(
        [
            955.747, 841.783, 740.199, 649.538, 568.404, 495.815, 431.069, 373.464, 322.354, 277.190,
            237.635, 203.433, 174.070, 148.949, 127.408, 108.915, 93.114, 79.551, 67.940, 58.072,
            49.593, 42.318, 36.138, 30.907, 26.362, 16.423, 7.583, 3.620, 1.807, 0.938, 0.469,
        ],
        dtype=np.float64,
    )
    o3ann = np.zeros(31, dtype=np.float64)
    o3ann[0] = 0.5 * (o3sum[0] + o3win[0])
    for k in range(1, 31):
        interp = o3win[k - 1] + (o3win[k] - o3win[k - 1]) / (ppwin[k] - ppwin[k - 1]) * (ppsum[k] - ppwin[k - 1])
        o3ann[k] = 0.5 * (interp + o3sum[k])
    ppwrkh = np.zeros(32, dtype=np.float64)
    ppwrkh[0] = 1100.0
    ppwrkh[1:31] = 0.5 * (ppsum[1:] + ppsum[:-1])
    out = np.zeros(pz_bottom_up.size - 1, dtype=np.float64)
    for k in range(out.size):
        bottom = pz_bottom_up[k]
        top = pz_bottom_up[k + 1]
        acc = 0.0
        for jj in range(31):
            pb1 = 0.0 if -(bottom - ppwrkh[jj]) >= 0.0 else bottom - ppwrkh[jj]
            pb2 = 0.0 if -(bottom - ppwrkh[jj + 1]) >= 0.0 else bottom - ppwrkh[jj + 1]
            pt1 = 0.0 if -(top - ppwrkh[jj]) >= 0.0 else top - ppwrkh[jj]
            pt2 = 0.0 if -(top - ppwrkh[jj + 1]) >= 0.0 else top - ppwrkh[jj + 1]
            acc += (pb2 - pb1 - pt2 + pt1) * o3ann[jj]
        out[k] = acc / max(bottom - top, 1.0e-300)
    return out


def _nbuf_for_ptop(ptop_pa: float | None) -> int:
    """Above-model-top buffer-layer count, WRF ``module_ra_rrtm.F:6781``
    (``nint(p_top*0.01/deltap)``).  ``None`` -> legacy ``DEFAULT_PTOP_PA``."""

    ptop = DEFAULT_PTOP_PA if ptop_pa is None else float(ptop_pa)
    return _nint(ptop * 0.01 / DELTAP_MB)


def _prepare_atmosphere(T, t8w, p, p8w, qv, qc, qr, qi, qs, qg, cldfra, dz, emiss, tsk,
                        ptop_pa: float | None = None):
    nz = T.size
    nbuf = _nbuf_for_ptop(ptop_pa)
    nlayers = nz + nbuf

    pz = np.zeros(nlayers + 1, dtype=np.float64)
    tz = np.zeros(nlayers + 1, dtype=np.float64)
    pavel = np.zeros(nlayers, dtype=np.float64)
    tavel = np.zeros(nlayers, dtype=np.float64)
    pz[: nz + 1] = p8w * 0.01
    tz[: nz + 1] = t8w
    pavel[:nz] = p * 0.01
    tavel[:nz] = T
    for l in range(nz + 1, nlayers):
        pz[l] = pz[l - 1] - DELTAP_MB
        pavel[l - 1] = 0.5 * (pz[l] + pz[l - 1])
    pz[nlayers] = 0.0
    pavel[nlayers - 1] = 0.5 * (pz[nlayers] + pz[nlayers - 1])

    pprof = np.asarray(
        [
            1000.00, 855.47, 731.82, 626.05, 535.57, 458.16, 391.94, 335.29, 286.83, 245.38,
            209.91, 179.57, 153.62, 131.41, 112.42, 96.17, 82.27, 70.38, 60.21, 51.51,
            44.06, 37.69, 32.25, 27.59, 23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
            9.25, 7.91, 6.77, 5.79, 4.95, 4.24, 3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
            1.42, 1.22, 1.04, 0.89, 0.76, 0.65, 0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
            0.22, 0.19, 0.16, 0.14, 0.12, 0.10,
        ],
        dtype=np.float64,
    )
    tprof = np.asarray(
        [
            279.94, 276.16, 270.73, 264.14, 256.71, 249.28, 241.97, 234.91, 228.78, 224.02,
            220.52, 217.31, 215.21, 213.48, 211.63, 211.45, 211.73, 212.71, 213.81, 214.95,
            215.96, 216.73, 217.42, 218.11, 218.89, 219.92, 221.31, 222.84, 224.39, 226.04,
            227.78, 229.73, 231.88, 234.22, 236.82, 239.50, 242.30, 245.21, 248.13, 251.08,
            254.04, 257.02, 259.84, 261.88, 263.38, 264.67, 265.42, 265.34, 264.45, 262.76,
            260.85, 258.78, 256.49, 254.02, 251.07, 248.23, 245.46, 242.77, 239.87, 237.53,
        ],
        dtype=np.float64,
    )
    varint = np.zeros(nlayers + 1, dtype=np.float64)
    for l in range(1, nlayers + 1):
        if pprof[-1] < pz[l]:
            klev = len(pprof) - 1
            for ll in range(1, len(pprof)):
                if pprof[ll] < pz[l]:
                    klev = ll - 1
                    break
        else:
            klev = len(pprof) - 1
        if klev != len(pprof) - 1:
            wght = (pz[l] - pprof[klev]) / (pprof[klev + 1] - pprof[klev])
            varint[l] = wght * (tprof[klev + 1] - tprof[klev]) + tprof[klev]
        else:
            varint[l] = tprof[klev]
    for l in range(nz + 1, nlayers + 1):
        tz[l] = varint[l] + (tz[nz] - varint[nz])
        tavel[l - 1] = 0.5 * (tz[l] + tz[l - 1])

    co2vmr = (280.0 + 90.0 * np.exp(0.02 * (2009 - 2000))) * 1.0e-6
    n2ovmr = 319.0e-9
    ch4vmr = 1774.0e-9
    amd, amw, avgdro = 28.9644, 18.0154, 6.022e23
    amdw, amdo = 1.607758, 0.603461
    gravit = 9.81 * 100.0

    wkl = np.zeros((35, nlayers), dtype=np.float64)
    wx = np.zeros((4, nlayers), dtype=np.float64)
    coldry = np.zeros(nlayers, dtype=np.float64)
    for l in range(nz):
        h2ovmr = max(qv[l], 1.0e-12) * amdw
        wkl[0, l] = h2ovmr
        wkl[1, l] = co2vmr
        wkl[3, l] = n2ovmr
        wkl[5, l] = ch4vmr
        amm = (1.0 - h2ovmr) * amd + h2ovmr * amw
        coldry[l] = (pz[l] - pz[l + 1]) * 1.0e3 * avgdro / (gravit * amm * (1.0 + h2ovmr))

    o3prof2 = np.zeros(nlayers, dtype=np.float64)
    shifted = _o3_average_from_interfaces(pz[1:])
    o3prof2[: shifted.size] = shifted
    o3prof2[-1] = 6.135e-6
    for l in range(nlayers):
        wkl[2, l] = o3prof2[l] * amdo
        if l >= nz:
            h2ovmr = 5.0e-6
            wkl[0, l] = h2ovmr
            wkl[1, l] = co2vmr
            wkl[3, l] = wkl[3, nz - 1]
            wkl[5, l] = wkl[5, nz - 1]
            amm = (1.0 - h2ovmr) * amd + h2ovmr * amw
            coldry[l] = (pz[l] - pz[l + 1]) * 1.0e3 * avgdro / (gravit * amm * (1.0 + h2ovmr))
    wkl[:6, :] *= coldry[None, :]

    cloudfrac = np.zeros(nlayers, dtype=np.float64)
    taucloud = np.zeros(nlayers, dtype=np.float64)
    for l in range(nz):
        ro = p[l] / (287.0 * T[l])
        clwp = ro * qc[l] * dz[l] * 1000.0
        ciwp = ro * qi[l] * dz[l] * 1000.0
        plwp = (ro * qr[l]) ** 0.75 * dz[l] * 1000.0
        piwp = (ro * qs[l]) ** 0.75 * dz[l] * 1000.0
        tau = 0.144 * clwp + 0.0735 * ciwp + 0.330e-3 * plwp + 2.34e-3 * piwp
        cf = float(cldfra[l])
        if tau > 0.01:
            cf = 1.0
        taucloud[l] = tau
        cloudfrac[l] = cf

    tbound = min(float(tsk), 339.99)
    semiss = np.full(NBANDS, float(emiss), dtype=np.float64)
    return pavel, tavel, pz, tz, cloudfrac, taucloud, coldry, wkl, wx, tbound, semiss, nz, nlayers


class _Coef(NamedTuple):
    colh2o: np.ndarray
    colco2: np.ndarray
    colo3: np.ndarray
    coln2o: np.ndarray
    colch4: np.ndarray
    colo2: np.ndarray
    water: np.ndarray
    co2mult: np.ndarray
    fac00: np.ndarray
    fac01: np.ndarray
    fac10: np.ndarray
    fac11: np.ndarray
    forfac: np.ndarray
    selffac: np.ndarray
    selffrac: np.ndarray
    jp: np.ndarray
    jt: np.ndarray
    jt1: np.ndarray
    indself: np.ndarray
    laytrop: int
    layswtch: int
    laylow: int


def _setcoef(pavel, tavel, coldry, wkl, tables: _RRTMTables) -> _Coef:
    n = pavel.size
    jp = np.zeros(n, dtype=np.int32)
    jt = np.zeros(n, dtype=np.int32)
    jt1 = np.zeros(n, dtype=np.int32)
    fac00 = np.zeros(n, dtype=np.float64)
    fac01 = np.zeros(n, dtype=np.float64)
    fac10 = np.zeros(n, dtype=np.float64)
    fac11 = np.zeros(n, dtype=np.float64)
    forfac = np.zeros(n, dtype=np.float64)
    selffac = np.zeros(n, dtype=np.float64)
    selffrac = np.zeros(n, dtype=np.float64)
    indself = np.zeros(n, dtype=np.int32)
    colh2o = np.zeros(n, dtype=np.float64)
    colco2 = np.zeros(n, dtype=np.float64)
    colo3 = np.zeros(n, dtype=np.float64)
    coln2o = np.zeros(n, dtype=np.float64)
    colch4 = np.zeros(n, dtype=np.float64)
    colo2 = np.zeros(n, dtype=np.float64)
    water_arr = np.zeros(n, dtype=np.float64)
    co2mult = np.zeros(n, dtype=np.float64)
    laytrop = layswtch = laylow = 0
    for l in range(n):
        plog = np.log(max(pavel[l], 1.0e-300))
        jp[l] = min(58, max(1, _fint(36.0 - 5.0 * (plog + 0.04))))
        jp1 = jp[l] + 1
        fp = 5.0 * (tables.preflog[jp[l] - 1] - plog)
        jt[l] = min(4, max(1, _fint(3.0 + (tavel[l] - tables.tref[jp[l] - 1]) / 15.0)))
        ft = ((tavel[l] - tables.tref[jp[l] - 1]) / 15.0) - float(jt[l] - 3)
        jt1[l] = min(4, max(1, _fint(3.0 + (tavel[l] - tables.tref[jp1 - 1]) / 15.0)))
        ft1 = ((tavel[l] - tables.tref[jp1 - 1]) / 15.0) - float(jt1[l] - 3)
        water = wkl[0, l] / max(coldry[l], 1.0e-300)
        water_arr[l] = water
        scalefac = pavel[l] * (296.0 / 1013.0) / tavel[l]
        if plog > 4.56:
            laytrop += 1
            if plog > 5.76:
                layswtch += 1
            if plog >= 6.62:
                laylow += 1
            factor = (tavel[l] - 188.0) / 7.2
            indself[l] = min(9, max(1, _fint(factor) - 7))
            selffrac[l] = factor - float(indself[l] + 7)
        else:
            indself[l] = 1
        forfac[l] = scalefac / (1.0 + water)
        selffac[l] = water * forfac[l]
        colh2o[l] = 1.0e-20 * wkl[0, l]
        colco2[l] = 1.0e-20 * wkl[1, l] or 1.0e-32 * coldry[l]
        colo3[l] = 1.0e-20 * wkl[2, l]
        coln2o[l] = 1.0e-20 * wkl[3, l] or 1.0e-32 * coldry[l]
        colch4[l] = 1.0e-20 * wkl[5, l] or 1.0e-32 * coldry[l]
        colo2[l] = 1.0e-20 * wkl[6, l] if wkl.shape[0] > 6 else 0.0
        co2reg = 3.55e-24 * coldry[l]
        co2mult[l] = (colco2[l] - co2reg) * 272.63 * np.exp(-1919.4 / tavel[l]) / (8.7604e-4 * tavel[l])
        compfp = 1.0 - fp
        fac10[l] = compfp * ft
        fac00[l] = compfp * (1.0 - ft)
        fac11[l] = fp * ft1
        fac01[l] = fp * (1.0 - ft1)
    if laylow == 0:
        laylow = 1
    if layswtch < n and jp[layswtch] <= 6:
        layswtch += 1
    return _Coef(
        colh2o, colco2, colo3, coln2o, colch4, colo2, water_arr, co2mult,
        fac00, fac01, fac10, fac11, forfac, selffac, selffrac,
        jp, jt, jt1, indself, laytrop, layswtch, laylow,
    )


def _pure_band(taug, pfrac, tables, coef, band: int, gas: np.ndarray, with_self: bool = False, with_for: bool = False):
    start = int(NGB_START[band])
    ng = int(NGC[band])
    absa = tables.absa[band]
    absb = tables.absb[band]
    nspa = int(NSPA[band])
    nspb = int(NSPB[band])
    for l in range(coef.colh2o.size):
        low = l < coef.laytrop or absb is None
        if low:
            ind0 = ((coef.jp[l] - 1) * 5 + (coef.jt[l] - 1)) * nspa + 1
            ind1 = (coef.jp[l] * 5 + (coef.jt1[l] - 1)) * nspa + 1
            tau = gas[l] * _interp4(absa, ind0, ind1, coef.fac00[l], coef.fac10[l], coef.fac01[l], coef.fac11[l])
            if with_self:
                tau += gas[l] * coef.selffac[l] * _self_term(tables.selfref[band], int(coef.indself[l]), coef.selffrac[l])
            if with_for and tables.forref[band] is not None:
                tau += gas[l] * coef.forfac[l] * tables.forref[band]
            frac = tables.fracrefa[band]
        else:
            ind0 = ((coef.jp[l] - 13) * 5 + (coef.jt[l] - 1)) * nspb + 1
            ind1 = ((coef.jp[l] - 12) * 5 + (coef.jt1[l] - 1)) * nspb + 1
            tau = gas[l] * _interp4(absb, ind0, ind1, coef.fac00[l], coef.fac10[l], coef.fac01[l], coef.fac11[l])
            if with_for and tables.forref[band] is not None:
                tau += gas[l] * coef.forfac[l] * tables.forref[band]
            frac = tables.fracrefb[band] if tables.fracrefb[band] is not None else tables.fracrefa[band]
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _apply_taugb(tables: _RRTMTables, coef: _Coef, wx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = coef.colh2o.size
    taug = np.zeros((n, NGPT), dtype=np.float64)
    pfrac = np.zeros((n, NGPT), dtype=np.float64)

    _pure_band(taug, pfrac, tables, coef, 0, coef.colh2o, with_self=True, with_for=True)
    _taugb2(taug, pfrac, tables, coef)
    _taugb3(taug, pfrac, tables, coef)
    _taugb4(taug, pfrac, tables, coef)
    _taugb5(taug, pfrac, tables, coef, wx)
    _taugb6(taug, pfrac, tables, coef, wx)
    _taugb7(taug, pfrac, tables, coef)
    _taugb8(taug, pfrac, tables, coef, wx)
    _taugb9(taug, pfrac, tables, coef)
    _pure_band(taug, pfrac, tables, coef, 9, coef.colh2o)
    _pure_band(taug, pfrac, tables, coef, 10, coef.colh2o, with_self=True)
    _taugb12(taug, pfrac, tables, coef)
    _taugb13(taug, pfrac, tables, coef)
    _pure_band(taug, pfrac, tables, coef, 13, coef.colco2, with_self=True)
    _taugb15(taug, pfrac, tables, coef)
    _taugb16(taug, pfrac, tables, coef)
    return taug, pfrac


def _taugb2(taug, pfrac, tables, c):
    start, ng, band = int(NGB_START[1]), int(NGC[1]), 1
    refparam = tables.local["TAUGB2_REFPARAM"]
    for l in range(c.colh2o.size):
        fp = c.fac11[l] + c.fac01[l]
        ifp = max(0, _fint(200.0 * fp + 0.5))
        fc00, fc10 = c.fac00[l] * tables.corr2[ifp], c.fac10[l] * tables.corr2[ifp]
        fc01, fc11 = c.fac01[l] * tables.corr1[ifp], c.fac11[l] * tables.corr1[ifp]
        if l < c.laytrop:
            h2oparam = c.water[l] / (c.water[l] + 0.002)
            ifrac = 13
            for idx in range(2, 13):
                if h2oparam >= refparam[idx - 1]:
                    ifrac = idx
                    break
            fracint = (h2oparam - refparam[ifrac - 1]) / (refparam[ifrac - 2] - refparam[ifrac - 1])
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + 1
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + 1
            tau = c.colh2o[l] * (
                fc00 * _row(tables.absa[band], ind0)
                + fc10 * _row(tables.absa[band], ind0 + 1)
                + fc01 * _row(tables.absa[band], ind1)
                + fc11 * _row(tables.absa[band], ind1 + 1)
                + c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l])
                + c.forfac[l] * tables.forref[band]
            )
            frac = tables.fracrefa[band][:, ifrac - 1] + fracint * (tables.fracrefa[band][:, ifrac - 2] - tables.fracrefa[band][:, ifrac - 1])
        else:
            ind0 = ((c.jp[l] - 13) * 5 + (c.jt[l] - 1)) * int(NSPB[band]) + 1
            ind1 = ((c.jp[l] - 12) * 5 + (c.jt1[l] - 1)) * int(NSPB[band]) + 1
            tau = c.colh2o[l] * (
                fc00 * _row(tables.absb[band], ind0)
                + fc10 * _row(tables.absb[band], ind0 + 1)
                + fc01 * _row(tables.absb[band], ind1)
                + fc11 * _row(tables.absb[band], ind1 + 1)
                + c.forfac[l] * tables.forref[band]
            )
            frac = tables.fracrefb[band]
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _binary_params(a, b, strrat, mult):
    speccomb = a + strrat * b
    specparm = min(a / max(speccomb, 1.0e-300), ONEMINUS)
    specmult = mult * specparm
    js = 1 + _fint(specmult)
    fs = specmult % 1.0
    return speccomb, js, fs


def _taugb3(taug, pfrac, tables, c):
    start, ng, band = int(NGB_START[2]), int(NGC[2]), 2
    h2oref = tables.local["TAUGB3_H2OREF"]
    n2oref = tables.local["TAUGB3_N2OREF"]
    co2ref = tables.local["TAUGB3_CO2REF"]
    etaref = tables.local["TAUGB3_ETAREF"]
    strrat = 1.19268
    for l in range(c.colh2o.size):
        if l < c.laytrop:
            speccomb, js, fs = _binary_params(c.colh2o[l], c.colco2[l], strrat, 8.0)
            if js == 8:
                if fs >= 0.9:
                    js, fs = 9, 10.0 * (fs - 0.9)
                else:
                    fs /= 0.9
            ns = js + _fint(fs + 0.5)
            fp = c.fac01[l] + c.fac11[l]
            wcomb1 = h2oref[c.jp[l] - 1] if ns == 10 else strrat * co2ref[c.jp[l] - 1] / (1.0 - etaref[ns - 1])
            wcomb2 = h2oref[c.jp[l]] if ns == 10 else strrat * co2ref[c.jp[l]] / (1.0 - etaref[ns - 1])
            ratio = (n2oref[c.jp[l] - 1] / wcomb1) + fp * ((n2oref[c.jp[l]] / wcomb2) - (n2oref[c.jp[l] - 1] / wcomb1))
            n2omult = c.coln2o[l] - speccomb * ratio
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + js
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + js
            tau = speccomb * _binary_lower(tables.absa[band], ind0, ind1, int(NSPA[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += c.colh2o[l] * (c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l]) + c.forfac[l] * tables.forref[band])
            tau += n2omult * tables.minor["ABSN2OAC3"]
            frac = _frac_interp(tables.fracrefa[band], js, fs)
        else:
            speccomb, js, fs = _binary_params(c.colh2o[l], c.colco2[l], strrat, 4.0)
            ns = js + _fint(fs + 0.5)
            fp = c.fac01[l] + c.fac11[l]
            wcomb1 = h2oref[c.jp[l] - 1] if ns == 5 else strrat * co2ref[c.jp[l] - 1] / (1.0 - etaref[ns - 1])
            wcomb2 = h2oref[c.jp[l]] if ns == 5 else strrat * co2ref[c.jp[l]] / (1.0 - etaref[ns - 1])
            ratio = (n2oref[c.jp[l] - 1] / wcomb1) + fp * ((n2oref[c.jp[l]] / wcomb2) - (n2oref[c.jp[l] - 1] / wcomb1))
            n2omult = c.coln2o[l] - speccomb * ratio
            ind0 = ((c.jp[l] - 13) * 5 + (c.jt[l] - 1)) * int(NSPB[band]) + js
            ind1 = ((c.jp[l] - 12) * 5 + (c.jt1[l] - 1)) * int(NSPB[band]) + js
            tau = speccomb * _binary_lower(tables.absb[band], ind0, ind1, int(NSPB[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += c.colh2o[l] * c.forfac[l] * tables.forref[band] + n2omult * tables.minor["ABSN2OBC3"]
            frac = _frac_interp(tables.fracrefb[band], js, fs)
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _simple_binary_band(taug, pfrac, tables, c, band, low_a, low_b, strrat_low, high_a=None, high_b=None, strrat_high=None, high_adjust=None):
    start, ng = int(NGB_START[band]), int(NGC[band])
    for l in range(c.colh2o.size):
        if l < c.laytrop or high_a is None:
            speccomb, js, fs = _binary_params(low_a[l], low_b[l], strrat_low, 8.0)
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + js
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + js
            tau = speccomb * _binary_lower(tables.absa[band], ind0, ind1, int(NSPA[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += c.colh2o[l] * c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l])
            frac = _frac_interp(tables.fracrefa[band], js, fs)
        else:
            speccomb, js, fs = _binary_params(high_a[l], high_b[l], strrat_high, 4.0)
            if high_adjust == "band4":
                if js > 1:
                    js += 1
                elif fs >= 0.0024:
                    js, fs = 2, (fs - 0.0024) / 0.9976
                else:
                    js, fs = 1, fs / 0.0024
            ind0 = ((c.jp[l] - 13) * 5 + (c.jt[l] - 1)) * int(NSPB[band]) + js
            ind1 = ((c.jp[l] - 12) * 5 + (c.jt1[l] - 1)) * int(NSPB[band]) + js
            tau = speccomb * _binary_lower(tables.absb[band], ind0, ind1, int(NSPB[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            frac = _frac_interp(tables.fracrefb[band], js, fs)
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _taugb4(taug, pfrac, tables, c):
    _simple_binary_band(taug, pfrac, tables, c, 3, c.colh2o, c.colco2, 850.577, c.colo3, c.colco2, 35.7416, "band4")


def _taugb5(taug, pfrac, tables, c, wx):
    _simple_binary_band(taug, pfrac, tables, c, 4, c.colh2o, c.colco2, 90.4894, c.colo3, c.colco2, 0.900502)
    start, ng = int(NGB_START[4]), int(NGC[4])
    for l in range(c.colh2o.size):
        taug[l, start : start + ng] += wx[0, l] * tables.minor["CCL4C5"]


def _taugb6(taug, pfrac, tables, c, wx):
    start, ng, band = int(NGB_START[5]), int(NGC[5]), 5
    for l in range(c.colh2o.size):
        if l < c.laytrop:
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) + 1
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) + 1
            tau = c.colh2o[l] * (_interp4(tables.absa[band], ind0, ind1, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l]) + c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l]))
            tau += wx[1, l] * tables.minor["CFC11ADJC6"] + wx[2, l] * tables.minor["CFC12C6"] + c.co2mult[l] * tables.minor["ABSCO2C6"]
        else:
            tau = wx[1, l] * tables.minor["CFC11ADJC6"] + wx[2, l] * tables.minor["CFC12C6"]
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = tables.fracrefa[band]


def _taugb7(taug, pfrac, tables, c):
    start, ng, band = int(NGB_START[6]), int(NGC[6]), 6
    for l in range(c.colh2o.size):
        if l < c.laytrop:
            speccomb, js, fs = _binary_params(c.colh2o[l], c.colo3[l], 8.21104e4, 8.0)
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + js
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + js
            tau = speccomb * _binary_lower(
                tables.absa[band],
                ind0,
                ind1,
                int(NSPA[band]),
                fs,
                c.fac00[l],
                c.fac10[l],
                c.fac01[l],
                c.fac11[l],
            )
            tau += c.colh2o[l] * c.selffac[l] * _self_term(
                tables.selfref[band], int(c.indself[l]), c.selffrac[l]
            )
            frac = _frac_interp(tables.fracrefa[band], js, fs)
        else:
            ind0 = ((c.jp[l] - 13) * 5 + (c.jt[l] - 1)) + 1
            ind1 = ((c.jp[l] - 12) * 5 + (c.jt1[l] - 1)) + 1
            tau = c.colo3[l] * _interp4(
                tables.absb[band],
                ind0,
                ind1,
                c.fac00[l],
                c.fac10[l],
                c.fac01[l],
                c.fac11[l],
            )
            frac = tables.fracrefb[band]
        taug[l, start : start + ng] = tau + c.co2mult[l] * tables.minor["ABSCO2C7"]
        pfrac[l, start : start + ng] = frac


def _taugb8(taug, pfrac, tables, c, wx):
    start, ng, band = int(NGB_START[7]), int(NGC[7]), 7
    h2oref = tables.local["TAUGB8_H2OREF"]
    n2oref = tables.local["TAUGB8_N2OREF"]
    o3ref = tables.local["TAUGB8_O3REF"]
    for l in range(c.colh2o.size):
        fp = c.fac01[l] + c.fac11[l]
        if l < c.layswtch:
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) + 1
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) + 1
            ratio = (n2oref[c.jp[l] - 1] / h2oref[c.jp[l] - 1]) + fp * ((n2oref[c.jp[l]] / h2oref[c.jp[l]]) - (n2oref[c.jp[l] - 1] / h2oref[c.jp[l] - 1]))
            n2omult = c.coln2o[l] - c.colh2o[l] * ratio
            tau = c.colh2o[l] * (_interp4(tables.absa[band], ind0, ind1, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l]) + c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l]))
            tau += wx[2, l] * tables.minor["CFC12C8"] + wx[3, l] * tables.minor["CFC22ADJC8"] + c.co2mult[l] * tables.minor["ABSCO2AC8"] + n2omult * tables.minor["ABSN2OAC8"]
            frac = tables.fracrefa[band]
        else:
            ind0 = ((c.jp[l] - 7) * 5 + (c.jt[l] - 1)) + 1
            ind1 = ((c.jp[l] - 6) * 5 + (c.jt1[l] - 1)) + 1
            ratio = (n2oref[c.jp[l] - 1] / o3ref[c.jp[l] - 1]) + fp * ((n2oref[c.jp[l]] / o3ref[c.jp[l]]) - (n2oref[c.jp[l] - 1] / o3ref[c.jp[l] - 1]))
            n2omult = c.coln2o[l] - c.colo3[l] * ratio
            tau = c.colo3[l] * _interp4(tables.absb[band], ind0, ind1, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += wx[2, l] * tables.minor["CFC12C8"] + wx[3, l] * tables.minor["CFC22ADJC8"] + c.co2mult[l] * tables.minor["ABSCO2BC8"] + n2omult * tables.minor["ABSN2OBC8"]
            frac = tables.fracrefb[band]
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _taugb9(taug, pfrac, tables, c):
    start, ng, band = int(NGB_START[8]), int(NGC[8]), 8
    h2oref = tables.local["TAUGB9_H2OREF"]
    n2oref = tables.local["TAUGB9_N2OREF"]
    ch4ref = tables.local["TAUGB9_CH4REF"]
    etaref = tables.local["TAUGB9_ETAREF"]
    ioff = 0
    for l in range(c.colh2o.size):
        if l < c.laytrop:
            speccomb, js, fs = _binary_params(c.colh2o[l], c.colch4[l], 21.6282, 8.0)
            jfrac, ffrac = js, fs
            if js == 8:
                if fs <= 0.68:
                    fs = fs / 0.68
                elif fs <= 0.92:
                    js, fs = js + 1, (fs - 0.68) / 0.24
                else:
                    js, fs = js + 2, (fs - 0.92) / 0.08
            elif js == 9:
                js, fs, jfrac, ffrac = 10, 1.0, 8, 1.0
            ns = js + _fint(fs + 0.5)
            if l + 1 == c.laylow:
                ioff = ng
            if l + 1 == c.layswtch:
                ioff = 2 * ng
            fp = c.fac01[l] + c.fac11[l]
            wcomb1 = h2oref[c.jp[l] - 1] if ns == 11 else 21.6282 * ch4ref[c.jp[l] - 1] / (1.0 - etaref[ns - 1])
            wcomb2 = h2oref[c.jp[l]] if ns == 11 else 21.6282 * ch4ref[c.jp[l]] / (1.0 - etaref[ns - 1])
            ratio = (n2oref[c.jp[l] - 1] / wcomb1) + fp * ((n2oref[c.jp[l]] / wcomb2) - (n2oref[c.jp[l] - 1] / wcomb1))
            n2omult = c.coln2o[l] - speccomb * ratio
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + js
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + js
            tau = speccomb * _binary_lower(tables.absa[band], ind0, ind1, int(NSPA[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += c.colh2o[l] * c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l])
            tau += n2omult * tables.minor["ABSN2OC9"][ioff : ioff + ng]
            frac = _frac_interp(tables.fracrefa[band], jfrac, ffrac)
        else:
            ind0 = ((c.jp[l] - 13) * 5 + (c.jt[l] - 1)) + 1
            ind1 = ((c.jp[l] - 12) * 5 + (c.jt1[l] - 1)) + 1
            tau = c.colch4[l] * _interp4(tables.absb[band], ind0, ind1, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            frac = tables.fracrefb[band]
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _lower_only_binary(taug, pfrac, tables, c, band, a, b, strrat):
    start, ng = int(NGB_START[band]), int(NGC[band])
    for l in range(c.colh2o.size):
        if l < c.laytrop:
            speccomb, js, fs = _binary_params(a[l], b[l], strrat, 8.0)
            ind0 = ((c.jp[l] - 1) * 5 + (c.jt[l] - 1)) * int(NSPA[band]) + js
            ind1 = (c.jp[l] * 5 + (c.jt1[l] - 1)) * int(NSPA[band]) + js
            tau = speccomb * _binary_lower(tables.absa[band], ind0, ind1, int(NSPA[band]), fs, c.fac00[l], c.fac10[l], c.fac01[l], c.fac11[l])
            tau += c.colh2o[l] * c.selffac[l] * _self_term(tables.selfref[band], int(c.indself[l]), c.selffrac[l])
            frac = _frac_interp(tables.fracrefa[band], js, fs)
        else:
            tau = np.zeros(ng, dtype=np.float64)
            frac = np.zeros(ng, dtype=np.float64)
        taug[l, start : start + ng] = tau
        pfrac[l, start : start + ng] = frac


def _taugb12(taug, pfrac, tables, c):
    _lower_only_binary(taug, pfrac, tables, c, 11, c.colh2o, c.colco2, 0.009736757)


def _taugb13(taug, pfrac, tables, c):
    _lower_only_binary(taug, pfrac, tables, c, 12, c.colh2o, c.coln2o, 16658.87)


def _taugb15(taug, pfrac, tables, c):
    _lower_only_binary(taug, pfrac, tables, c, 14, c.coln2o, c.colco2, 0.2883201)


def _taugb16(taug, pfrac, tables, c):
    _lower_only_binary(taug, pfrac, tables, c, 15, c.colh2o, c.colch4, 830.411)


def _gasabs(tables: _RRTMTables, coef: _Coef, wx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    taug, pfrac = _apply_taugb(tables, coef, wx)
    itr = np.zeros_like(taug, dtype=np.int32)
    for l in range(taug.shape[0]):
        for ipr in range(NGPT):
            odepth = SECANG * taug[l, ipr]
            tff = 0.0 if odepth <= 0.0 else odepth / (tables.bpade + odepth)
            itr[l, ipr] = min(5000, max(0, _fint(5.0e3 * tff + 0.5)))
    return taug, pfrac, itr


def _planck_value(table: np.ndarray, temp: float, band: int) -> float:
    idx = max(1, min(180, _fint(temp - 159.0)))
    frac = temp - _fint(temp)
    return table[idx - 1, band] + frac * (table[idx, band] - table[idx - 1, band])


def _rtrn(tables: _RRTMTables, tavel, pz, tz, cldfrac, taucloud, itr, pfrac, tbound, semiss, model_layers: int):
    n = tavel.size
    totuflux = np.zeros(n + 1, dtype=np.float64)
    totdflux = np.zeros(n + 1, dtype=np.float64)
    totuclfl = np.zeros(n + 1, dtype=np.float64)
    totdclfl = np.zeros(n + 1, dtype=np.float64)
    fnet = np.zeros(n + 1, dtype=np.float64)
    fnetc = np.zeros(n + 1, dtype=np.float64)

    play = np.zeros((NBANDS, n), dtype=np.float64)
    plvl = np.zeros((NBANDS, n + 1), dtype=np.float64)
    plankbnd = np.zeros(NBANDS, dtype=np.float64)
    plnkemit = np.zeros(NBANDS, dtype=np.float64)
    for band in range(NBANDS):
        plankbnd[band] = tables.delwave[band] * _planck_value(tables.totplnk, tbound, band)
        plnkemit[band] = semiss[band] * plankbnd[band]
        for lev in range(n + 1):
            plvl[band, lev] = tables.delwave[band] * _planck_value(tables.totplnk, tz[lev], band)
        for lev in range(n):
            play[band, lev] = tables.delwave[band] * _planck_value(tables.totplnk, tavel[lev], band)

    icldlyr = (cldfrac > 0.0).astype(np.int32)
    odcld = SECANG * taucloud
    abscld = 1.0 - np.exp(-odcld)
    efclfrac = abscld * cldfrac

    bbu = np.zeros((n, NGPT), dtype=np.float64)
    bbutot = np.zeros((n, NGPT), dtype=np.float64)
    abss = np.zeros((n, NGPT), dtype=np.float64)
    atot = np.zeros((n, NGPT), dtype=np.float64)
    radld = np.zeros(NGPT, dtype=np.float64)
    radclrd = np.zeros(NGPT, dtype=np.float64)
    radlu = np.zeros(NGPT, dtype=np.float64)
    radclru = np.zeros(NGPT, dtype=np.float64)
    bglev = np.zeros(NGPT, dtype=np.float64)
    semis = np.zeros(NGPT, dtype=np.float64)
    raduemit = np.zeros(NGPT, dtype=np.float64)
    for ipr in range(NGPT):
        band = int(NGB[ipr])
        semis[ipr] = semiss[band]
        raduemit[ipr] = pfrac[0, ipr] * plnkemit[band]
        bglev[ipr] = pfrac[n - 1, ipr] * plvl[band, n]

    iclddn = 0
    for lev in range(n, 0, -1):
        idx = lev - 1
        drad = 0.0
        clrdrad = 0.0
        cloudy = icldlyr[idx] == 1
        if cloudy:
            iclddn = 1
        for ipr in range(NGPT):
            band = int(NGB[ipr])
            ind = int(itr[idx, ipr])
            bglay = pfrac[idx, ipr] * play[band, idx]
            delbgup = bglev[ipr] - bglay
            tauf = tables.tf[ind]
            bbu[idx, ipr] = bglay + tauf * delbgup
            if cloudy:
                odsm = tables.tau[ind] + odcld[idx]
                factot = odsm / (tables.bpade + odsm)
                bbutot[idx, ipr] = bglay + factot * delbgup
            bglev[ipr] = pfrac[idx, ipr] * plvl[band, idx]
            delbgdn = bglev[ipr] - bglay
            bbd = bglay + tauf * delbgdn
            abss[idx, ipr] = 1.0 - tables.trans[ind]
            if cloudy:
                bbdlevd = bglay + (tables.tau[ind] + odcld[idx]) / (tables.bpade + tables.tau[ind] + odcld[idx]) * delbgdn
                atot[idx, ipr] = abss[idx, ipr] + abscld[idx] - abss[idx, ipr] * abscld[idx]
                gassrc = bbd * abss[idx, ipr]
                radld[ipr] = radld[ipr] - radld[ipr] * (abss[idx, ipr] + efclfrac[idx] * (1.0 - abss[idx, ipr])) + gassrc + cldfrac[idx] * (bbdlevd * atot[idx, ipr] - gassrc)
                radclrd[ipr] = radclrd[ipr] + (bbd - radclrd[ipr]) * abss[idx, ipr]
            else:
                radld[ipr] = radld[ipr] + (bbd - radld[ipr]) * abss[idx, ipr]
                if iclddn == 1:
                    radclrd[ipr] = radclrd[ipr] + (bbd - radclrd[ipr]) * abss[idx, ipr]
                else:
                    radclrd[ipr] = radld[ipr]
            drad += radld[ipr]
            if cloudy or iclddn == 1:
                clrdrad += radclrd[ipr]
            else:
                clrdrad = drad
        totdflux[lev - 1] = drad * WTNUM
        totdclfl[lev - 1] = clrdrad * WTNUM

    urad = 0.0
    clrurad = 0.0
    for ipr in range(NGPT):
        radlu[ipr] = raduemit[ipr] + (1.0 - semis[ipr]) * radld[ipr]
        radclru[ipr] = raduemit[ipr] + (1.0 - semis[ipr]) * radclrd[ipr]
        urad += radlu[ipr]
        clrurad += radclru[ipr]
    totuflux[0] = urad * WTNUM
    totuclfl[0] = clrurad * WTNUM

    for lev in range(1, n + 1):
        idx = lev - 1
        urad = 0.0
        clrurad = 0.0
        for ipr in range(NGPT):
            if icldlyr[idx] == 1:
                gassrc = bbu[idx, ipr] * abss[idx, ipr]
                radlu[ipr] = radlu[ipr] - radlu[ipr] * (abss[idx, ipr] + efclfrac[idx] * (1.0 - abss[idx, ipr])) + gassrc + cldfrac[idx] * (bbutot[idx, ipr] * atot[idx, ipr] - gassrc)
                radclru[ipr] = radclru[ipr] + (bbu[idx, ipr] - radclru[ipr]) * abss[idx, ipr]
            else:
                radlu[ipr] = radlu[ipr] + (bbu[idx, ipr] - radlu[ipr]) * abss[idx, ipr]
                radclru[ipr] = radclru[ipr] + (bbu[idx, ipr] - radclru[ipr]) * abss[idx, ipr]
            urad += radlu[ipr]
            clrurad += radclru[ipr]
        totuflux[lev] = urad * WTNUM
        totuclfl[lev] = clrurad * WTNUM

    htr = np.zeros(n + 1, dtype=np.float64)
    htrc = np.zeros(n + 1, dtype=np.float64)
    for lev in range(n + 1):
        totuflux[lev] *= tables.fluxfac
        totdflux[lev] *= tables.fluxfac
        totuclfl[lev] *= tables.fluxfac
        totdclfl[lev] *= tables.fluxfac
        fnet[lev] = totuflux[lev] - totdflux[lev]
        fnetc[lev] = totuclfl[lev] - totdclfl[lev]
        if lev >= 1:
            l = lev - 1
            htr[l] = tables.heatfac * (fnet[l] - fnet[lev]) / (pz[l] - pz[lev])
            htrc[l] = tables.heatfac * (fnetc[l] - fnetc[lev]) / (pz[l] - pz[lev])
    return htr[:model_layers] / 86400.0, totdflux[0], totuflux[model_layers], totdflux, totuflux


def _solve_one(T, t8w, p, p8w, qv, qc, qr, qi, qs, qg, cldfra, dz, emiss, tsk,
               ptop_pa: float | None = None):
    tables = _load_tables()
    atm = _prepare_atmosphere(T, t8w, p, p8w, qv, qc, qr, qi, qs, qg, cldfra, dz, emiss, tsk,
                              ptop_pa=ptop_pa)
    pavel, tavel, pz, tz, cloudfrac, taucloud, coldry, wkl, wx, tbound, semiss, model_layers, _ = atm
    coef = _setcoef(pavel, tavel, coldry, wkl, tables)
    _, pfrac, itr = _gasabs(tables, coef, wx)
    return _rtrn(tables, tavel, pz, tz, cloudfrac, taucloud, itr, pfrac, tbound, semiss, model_layers)


def solve_rrtm_lw_column(state: RRTMLWColumnState) -> RRTMLWColumnResult:
    arrays, emiss, tsk = _column_inputs(state)
    ptop_pa = state.top_pressure_pa
    heating = []
    glw = []
    olr = []
    fdown = []
    fup = []
    for col in range(arrays["T"].shape[0]):
        out = _solve_one(
            arrays["T"][col],
            arrays["t8w"][col],
            arrays["p"][col],
            arrays["p8w"][col],
            arrays["qv"][col],
            arrays["qc"][col],
            arrays["qr"][col],
            arrays["qi"][col],
            arrays["qs"][col],
            arrays["qg"][col],
            arrays["cloud_fraction"][col],
            arrays["dz"][col],
            emiss[col],
            tsk[col],
            ptop_pa=ptop_pa,
        )
        h, g, o, fd, fu = out
        heating.append(h)
        glw.append(g)
        olr.append(o)
        fdown.append(fd)
        fup.append(fu)
    return RRTMLWColumnResult(
        heating_rate=jnp.asarray(np.stack(heating), dtype=jnp.float64),
        glw=jnp.asarray(np.asarray(glw), dtype=jnp.float64),
        olr=jnp.asarray(np.asarray(olr), dtype=jnp.float64),
        flux_down=jnp.asarray(np.stack(fdown), dtype=jnp.float64),
        flux_up=jnp.asarray(np.stack(fup), dtype=jnp.float64),
    )


__all__ = ["RRTMLWColumnState", "RRTMLWColumnResult", "solve_rrtm_lw_column"]
