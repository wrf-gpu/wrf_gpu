"""S4 — real.exe wrfinput/wrfbdy parity comparator harness (v0.4.0).

FROZEN ENTRY SIGNATURES. S4 OWNS this comparator (the harness + the field
extraction + the gate logic + the campaign runner + the proof emitter) — it does
NOT own any production lane module. The harness:

  1. Reads a real.exe oracle pair (``wrfinput_d0N`` + ``wrfbdy_d01``) from the
     corpus (``<DATA_ROOT>/canairy_meteo/runs/{wrf_l2,wrf_l3}/<case>/``).
  2. Builds the native :class:`RealInitProduct` via
     :func:`gpuwrf.init.real_init.driver.build_real_init` from the matching
     met_em (``wps_cases``) for the SAME case/time/domain. (Until the S1/S2/S3
     lanes land, the campaign runs against a CANDIDATE FACTORY — see
     :func:`run_campaign`'s ``product_factory`` argument — so the harness is
     self-testable today: an oracle-derived stub candidate exercises the full
     PASS/FAIL mechanics, and a real.exe-vs-itself sanity candidate proves the
     comparator is not gameable, ~0 error.)
  3. Compares field-by-field against the FROZEN tolerances
     :data:`gpuwrf.init.real_init.types.WRFINPUT_TOLS` /
     :data:`gpuwrf.init.real_init.types.WRFBDY_TOLS` with the recorded masking
     policy (land/water/below-ground per V0.4.0-S0-PLAN.md section 5). NO
     post-hoc loosening — the comparator reads the frozen tables and may not
     mutate them.
  4. Emits a per-case, per-field structured report (rmse, maxabs, pass/fail) and
     a campaign roll-up across the >=10 cases (d01/d02/d03).

It ALSO scaffolds the 24h native-init FORECAST gate (:func:`run_forecast_gate`):
feed the native wrfinput/wrfbdy into the existing GPU forecast pipeline (NO
replay), run 24h, and compare the full-field forecast to the CPU-WRF wrfout for
the same case (reusing the existing per-case continuous-gate metric set
T2/U10/V10/PBLH/precip per lead). Conservation + restart gates must still pass.
The forecast gate is the only GPU-bound S4 step; it serializes onto the single
GPU (one job at a time) and the MANAGER invokes it in S5/integration. This module
defines the harness + the comparison metric but DOES NOT run a GPU forecast.

FILE OWNERSHIP: this file + the tests under ``tests/init/real_init/`` are S4's
exclusive files. ``types.py`` / ``driver.py`` are FROZEN (manager-owned); this
module only reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gpuwrf.init.real_init.types import (
    RealInitProduct,
    WRFBDY_TOLS,
    WRFINPUT_TOLS,
)

try:  # netCDF4 is a hard dep of the corpus oracle path; import lazily-friendly.
    from netCDF4 import Dataset
except Exception as _exc:  # pragma: no cover - exercised only without netCDF4
    Dataset = None  # type: ignore[assignment]
    _NETCDF_IMPORT_ERROR = _exc


COMPARATOR_SCHEMA_VERSION = "v0.4.0-S4-comparator-2026-06-02"

# Corpus root holding the real.exe oracle pairs.
ORACLE_ROOT_DEFAULT = Path("<DATA_ROOT>/canairy_meteo/runs")

# The four wrfbdy sides and their _Bxx / _BTxx suffixes (WRF convention).
_SIDES = ("W", "E", "S", "N")
_BDY_VALUE_SUFFIX = {"W": "BXS", "E": "BXE", "S": "BYS", "N": "BYE"}
_BDY_TEND_SUFFIX = {"W": "BTXS", "E": "BTXE", "S": "BTYS", "N": "BTYE"}
# LateralBC stores values under bxs/bxe/bys/bye and tendencies under
# btxs/btxe/btys/btye keyed by the lowercase side suffix.
_BDY_VALUE_KEY = {"W": "bxs", "E": "bxe", "S": "bys", "N": "bye"}
_BDY_TEND_KEY = {"W": "btxs", "E": "btxe", "S": "btys", "N": "btye"}

# Native LateralBC field name (lowercase) -> wrfbdy NetCDF variable prefix.
_BDY_NAME_TO_NC = {
    "u": "U",
    "v": "V",
    "t": "T",
    "ph": "PH",
    "qv": "QVAPOR",
    "mu": "MU",
}
# The wrfbdy tolerance-table key for each native LateralBC field name.
_BDY_NAME_TO_TOL = {
    "u": "U",
    "v": "V",
    "t": "T",
    "ph": "PH",
    "qv": "QV",
    "mu": "MU",
}


# ---------------------------------------------------------------------------
# Result records (the structured proof object).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldParityResult:
    """Per-field comparison record for the proof object."""

    name: str
    rmse: float
    maxabs: float
    rmse_tol: float
    maxabs_tol: float
    n_points: int
    passed: bool
    status: str = "OK"  # OK | SHAPE_MISMATCH | MISSING_NATIVE | MISSING_ORACLE | NO_POINTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rmse": self.rmse,
            "maxabs": self.maxabs,
            "rmse_tol": self.rmse_tol,
            "maxabs_tol": self.maxabs_tol,
            "n_points": self.n_points,
            "passed": self.passed,
            "status": self.status,
        }


@dataclass(frozen=True)
class CaseParityResult:
    """Per-(case, domain) roll-up of field parity vs the real.exe oracle."""

    case: str
    domain: str
    kind: str  # "wrfinput" | "wrfbdy"
    fields: tuple[FieldParityResult, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "domain": self.domain,
            "kind": self.kind,
            "passed": self.passed,
            "n_fields": len(self.fields),
            "n_failed": sum(1 for f in self.fields if not f.passed),
            "fields": [f.to_dict() for f in self.fields],
        }


# ---------------------------------------------------------------------------
# Oracle NetCDF reading.
# ---------------------------------------------------------------------------
def _require_netcdf() -> None:
    if Dataset is None:  # pragma: no cover
        raise RuntimeError(
            f"netCDF4 is required to read the real.exe oracle: {_NETCDF_IMPORT_ERROR}"
        )


def _squeeze_time(arr: np.ndarray) -> np.ndarray:
    """Drop a leading singleton WRF Time axis if present."""
    a = np.asarray(arr)
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]
    return a


def _read_oracle_var(ds: "Dataset", name: str) -> np.ndarray | None:
    """Read one wrfinput variable as a Time-squeezed float64 array (or None)."""
    if name not in ds.variables:
        return None
    raw = ds.variables[name][:]
    data = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
    return _squeeze_time(data)


def _read_oracle_scalar(ds: "Dataset", name: str) -> float | None:
    arr = _read_oracle_var(ds, name)
    if arr is None:
        return None
    return float(np.asarray(arr).ravel()[0])


# ---------------------------------------------------------------------------
# Metric.
# ---------------------------------------------------------------------------
def _masked_rmse_maxabs(
    native: np.ndarray,
    oracle: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float, float, int]:
    """RMSE and max-abs of (native - oracle) over the valid mask.

    ``mask`` (broadcastable to the field) selects the points that count. Points
    that are non-finite on EITHER side are always excluded (below-ground /
    fill-value protection). Returns (rmse, maxabs, n_points).
    """
    n = np.asarray(native, dtype=np.float64)
    o = np.asarray(oracle, dtype=np.float64)
    diff = n - o
    valid = np.isfinite(n) & np.isfinite(o)
    if mask is not None:
        valid = valid & np.broadcast_to(np.asarray(mask, dtype=bool), diff.shape)
    npts = int(valid.sum())
    if npts == 0:
        return float("nan"), float("nan"), 0
    d = diff[valid]
    rmse = float(np.sqrt(np.mean(d * d)))
    maxabs = float(np.max(np.abs(d)))
    return rmse, maxabs, npts


def _score_field(
    name: str,
    native: np.ndarray | None,
    oracle: np.ndarray | None,
    tols: Mapping[str, tuple[float, float]],
    *,
    mask: np.ndarray | None = None,
    categorical: bool = False,
) -> FieldParityResult:
    """Compute the parity record for one field against its frozen tolerance."""
    rmse_tol, maxabs_tol = tols[name]
    if native is None:
        return FieldParityResult(name, float("nan"), float("nan"), rmse_tol,
                                 maxabs_tol, 0, False, status="MISSING_NATIVE")
    if oracle is None:
        return FieldParityResult(name, float("nan"), float("nan"), rmse_tol,
                                 maxabs_tol, 0, False, status="MISSING_ORACLE")
    nat = np.asarray(native, dtype=np.float64)
    ora = np.asarray(oracle, dtype=np.float64)
    if nat.shape != ora.shape:
        return FieldParityResult(name, float("nan"), float("nan"), rmse_tol,
                                 maxabs_tol, 0, False, status="SHAPE_MISMATCH")
    rmse, maxabs, npts = _masked_rmse_maxabs(nat, ora, mask)
    if npts == 0:
        return FieldParityResult(name, float("nan"), float("nan"), rmse_tol,
                                 maxabs_tol, 0, False, status="NO_POINTS")
    if categorical:
        # Exact-match requirement: any mismatched point fails.
        passed = bool(maxabs <= maxabs_tol and rmse <= rmse_tol)
    else:
        passed = bool(rmse <= rmse_tol and maxabs <= maxabs_tol)
    return FieldParityResult(name, rmse, maxabs, rmse_tol, maxabs_tol, npts, passed)


# ---------------------------------------------------------------------------
# Native-product -> wrfinput-field extraction.
# ---------------------------------------------------------------------------
def _native_wrfinput_fields(product: RealInitProduct) -> dict[str, np.ndarray | None]:
    """Map a :class:`RealInitProduct` onto the wrfinput variable names.

    The keys are the wrfinput NetCDF variable names (the keys of WRFINPUT_TOLS).
    Vertical order is WRF model order on both sides (the lanes already produce
    model order; the oracle file is in model order), so no flip is applied here.
    Missing optional fields map to None (scored MISSING_NATIVE).
    """
    dyn = product.dynamics
    base = product.base
    surf = product.surface
    soil = product.soil
    vc = product.vcoord

    out: dict[str, np.ndarray | None] = {
        # --- dynamics (hour-0-critical) ---
        "MU": dyn.mu,
        "MUB": base.mub,
        "PB": base.pb,
        "P": dyn.p,
        "PH": dyn.ph,
        "PHB": base.phb,
        "T": dyn.theta,
        "U": dyn.u,
        "V": dyn.v,
        "W": dyn.w,
        "QVAPOR": dyn.qv,
        "AL": dyn.al,
        "ALT": dyn.alt,
        # --- 1D vertical coordinate ---
        "ZNW": vc.znw,
        "ZNU": vc.znu,
        "C1H": vc.c1h,
        "C2H": vc.c2h,
        "C3H": vc.c3h,
        "C4H": vc.c4h,
        "C1F": vc.c1f,
        "C3F": vc.c3f,
        "P_TOP": np.asarray(vc.p_top_pa, dtype=np.float64),
        # --- surface / metric ---
        "MAPFAC_M": surf.mapfac_m,
        "MAPFAC_U": surf.mapfac_u,
        "MAPFAC_V": surf.mapfac_v,
        "F": surf.f,
        "E": surf.e,
        "SINALPHA": surf.sinalpha,
        "COSALPHA": surf.cosalpha,
        "XLAT": surf.xlat,
        "XLONG": surf.xlong,
        "HGT": surf.hgt,
        "TSK": surf.tsk,
        "SST": surf.sst,
        "TMN": surf.tmn,
        "XLAND": surf.xland,
        # --- soil ---
        "TSLB": soil.tslb,
        "SMOIS": soil.smois,
        "ZS": soil.zs,
        "DZS": soil.dzs,
        "ISLTYP": np.asarray(soil.isltyp, dtype=np.float64),
        "IVGTYP": np.asarray(soil.ivgtyp, dtype=np.float64),
    }
    return out


# Fields whose comparison is restricted by a physical mask (per S0 plan sec 5).
_LAND_MASKED_FIELDS = ("TSLB", "SMOIS", "TMN")  # valid over land cells
_WATER_MASKED_FIELDS = ("SST",)  # SST valid over water
_CATEGORICAL_FIELDS = ("XLAND", "ISLTYP", "IVGTYP")

# WRFINPUT_TOLS keys that are NOT wrfinput-output variables (real.exe does not
# write them to wrfinput — they are runtime-diagnostic fields). The comparator
# records these informationally as NOT_IN_WRFINPUT (non-blocking) instead of a
# hard MISSING_ORACLE failure: there is no oracle to compare against. ALT (full
# inverse density alt = al + alb) is recomputed at runtime from AL+ALB, which ARE
# checked. TRACKED in the handoff for the manager (types.py is frozen; this is a
# harness-side classification, not a tolerance change).
_NOT_IN_WRFINPUT = ("ALT",)


def _build_masks(ds: "Dataset") -> dict[str, np.ndarray]:
    """Build land/water 2D masks from the oracle XLAND (1=land, 2=water)."""
    xland = _read_oracle_var(ds, "XLAND")
    if xland is None:
        landmask = _read_oracle_var(ds, "LANDMASK")
        land = (landmask >= 0.5) if landmask is not None else None
    else:
        land = xland < 1.5  # XLAND==1 -> land
    water = (~land) if land is not None else None
    return {"land": land, "water": water}


def compare_wrfinput(
    product: RealInitProduct,
    oracle_wrfinput_path: str | Path,
) -> CaseParityResult:
    """Compares a native init product to a real.exe wrfinput oracle file.

    Uses the FROZEN :data:`WRFINPUT_TOLS` and the S0 masking policy (soil/TMN
    over land, SST over water, categoricals exact-on-all-points, below-ground /
    fill-value points excluded). No post-hoc tolerance loosening.
    """
    _require_netcdf()
    path = Path(oracle_wrfinput_path)
    native = _native_wrfinput_fields(product)
    fields: list[FieldParityResult] = []
    with Dataset(path, "r") as ds:
        masks = _build_masks(ds)
        for name in WRFINPUT_TOLS:
            if name in _NOT_IN_WRFINPUT:
                # Not a wrfinput-output variable; record informationally, do not
                # block the case on an oracle that does not exist.
                rt, mt = WRFINPUT_TOLS[name]
                fields.append(FieldParityResult(
                    name, float("nan"), float("nan"), rt, mt, 0, True,
                    status="NOT_IN_WRFINPUT"))
                continue
            if name == "P_TOP":
                ora = _read_oracle_scalar(ds, "P_TOP")
                ora_arr = None if ora is None else np.asarray(ora, dtype=np.float64)
                fields.append(_score_field(name, native.get(name), ora_arr, WRFINPUT_TOLS))
                continue
            ora = _read_oracle_var(ds, name)
            mask = None
            if name in _LAND_MASKED_FIELDS:
                mask = masks.get("land")
            elif name in _WATER_MASKED_FIELDS:
                mask = masks.get("water")
            fields.append(
                _score_field(
                    name,
                    native.get(name),
                    ora,
                    WRFINPUT_TOLS,
                    mask=mask,
                    categorical=name in _CATEGORICAL_FIELDS,
                )
            )
    passed = all(f.passed for f in fields)
    return CaseParityResult(
        case=product.init_time,
        domain=product.domain,
        kind="wrfinput",
        fields=tuple(fields),
        passed=passed,
    )


# ---------------------------------------------------------------------------
# wrfbdy comparison.
# ---------------------------------------------------------------------------
def _stack_native_bdy(side_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the per-side arrays from a LateralBC value/tendency entry.

    LateralBC stores each coupled field as {bxs,bxe,bys,bye} (values) or
    {btxs,...} (tendencies). We score per-side and aggregate to a single
    worst-side RMSE/maxabs so one field == one parity record per case.
    """
    return dict(side_map)


def compare_wrfbdy(
    product: RealInitProduct,
    oracle_wrfbdy_path: str | Path,
) -> CaseParityResult:
    """Compares the native LBC to a real.exe wrfbdy oracle file.

    Scores both the coupled VALUE (the first wrfbdy time level, _BXS/_BXE/_BYS/
    _BYE) and the TENDENCY (_BTXS/...) for each coupled field U/V/T/PH/QV/MU,
    against the FROZEN :data:`WRFBDY_TOLS`. The native LateralBC stores the
    coupled value/tendency per side (WRF wrfbdy layout). One parity record per
    (field, value|tendency), aggregated over all four sides (worst side).
    """
    _require_netcdf()
    path = Path(oracle_wrfbdy_path)
    lbc = product.lateral_bc
    fields: list[FieldParityResult] = []
    if lbc is None:
        # No LBC in the product: every wrfbdy field is MISSING_NATIVE.
        for nm, tol_key in _BDY_NAME_TO_TOL.items():
            rt, mt = WRFBDY_TOLS[tol_key]
            fields.append(FieldParityResult(f"{tol_key}_value", float("nan"),
                                            float("nan"), rt, mt, 0, False,
                                            status="MISSING_NATIVE"))
            fields.append(FieldParityResult(f"{tol_key}_tend", float("nan"),
                                            float("nan"), rt, mt, 0, False,
                                            status="MISSING_NATIVE"))
        return CaseParityResult(product.init_time, product.domain, "wrfbdy",
                                tuple(fields), False)

    with Dataset(path, "r") as ds:
        for nm, nc_prefix in _BDY_NAME_TO_NC.items():
            tol_key = _BDY_NAME_TO_TOL[nm]
            rmse_tol, maxabs_tol = WRFBDY_TOLS[tol_key]
            native_vals = lbc.values.get(nm)
            native_tends = lbc.tendencies.get(nm)
            # ---- coupled VALUE (worst over sides) ----
            v_rmse, v_maxabs, v_n, v_status = _score_bdy_quantity(
                ds, nc_prefix, native_vals, _BDY_VALUE_SUFFIX, _BDY_VALUE_KEY)
            fields.append(_finalize_bdy_field(
                f"{tol_key}_value", v_rmse, v_maxabs, v_n, v_status,
                rmse_tol, maxabs_tol))
            # ---- TENDENCY (worst over sides) ----
            t_rmse, t_maxabs, t_n, t_status = _score_bdy_quantity(
                ds, nc_prefix, native_tends, _BDY_TEND_SUFFIX, _BDY_TEND_KEY)
            # tendency tolerance = value tolerance / interval_seconds (S0 sec 5)
            dt = float(lbc.bdyfrq_seconds) if lbc.bdyfrq_seconds else 1.0
            fields.append(_finalize_bdy_field(
                f"{tol_key}_tend", t_rmse, t_maxabs, t_n, t_status,
                rmse_tol / dt, maxabs_tol / dt))
    passed = all(f.passed for f in fields)
    return CaseParityResult(product.init_time, product.domain, "wrfbdy",
                            tuple(fields), passed)


def _score_bdy_quantity(
    ds: "Dataset",
    nc_prefix: str,
    native_side_map: Mapping[str, np.ndarray] | None,
    nc_suffix: Mapping[str, str],
    native_key: Mapping[str, str],
) -> tuple[float, float, int, str]:
    """Worst-over-sides RMSE/maxabs for one coupled quantity (value OR tend)."""
    if native_side_map is None:
        return float("nan"), float("nan"), 0, "MISSING_NATIVE"
    worst_rmse = 0.0
    worst_maxabs = 0.0
    total_n = 0
    any_scored = False
    for side in _SIDES:
        nc_name = f"{nc_prefix}_{nc_suffix[side]}"
        if nc_name not in ds.variables:
            continue
        var = ds.variables[nc_name]
        # wrfbdy arrays are (Time, bdy_width, [z,] side). Take the FIRST frame
        # explicitly (the t0 specified value / first-interval tendency, matching
        # the LateralBC convention) when a Time axis is present — regardless of
        # how many frames the file holds (Canary wrfbdy has 4).
        raw = var[0] if (var.dimensions and var.dimensions[0] == "Time") else var[:]
        ora = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
        nat = native_side_map.get(native_key[side])
        if nat is None:
            continue
        nat = np.asarray(nat, dtype=np.float64)
        ora = np.asarray(ora, dtype=np.float64)
        # The native LateralBC stacks per-side arrays over forcing INTERVALS:
        # value/tendency side arrays are (n_intervals, bdy_width, [z,] side),
        # while the oracle frame was already sliced to its FIRST time level above
        # (dropping the leading Time axis). Score the matching FIRST native
        # interval (the t0 value / first-interval tendency) so the two align —
        # one extra leading axis on the native side means "stacked intervals".
        if nat.ndim == ora.ndim + 1 and nat.shape[0] >= 1:
            nat = nat[0]
        if nat.shape != ora.shape:
            return float("nan"), float("nan"), 0, "SHAPE_MISMATCH"
        rmse, maxabs, npts = _masked_rmse_maxabs(nat, ora, None)
        if npts == 0:
            continue
        any_scored = True
        worst_rmse = max(worst_rmse, rmse)
        worst_maxabs = max(worst_maxabs, maxabs)
        total_n += npts
    if not any_scored:
        return float("nan"), float("nan"), 0, "MISSING_ORACLE"
    return worst_rmse, worst_maxabs, total_n, "OK"


def _finalize_bdy_field(
    name: str,
    rmse: float,
    maxabs: float,
    n: int,
    status: str,
    rmse_tol: float,
    maxabs_tol: float,
) -> FieldParityResult:
    if status != "OK":
        return FieldParityResult(name, rmse, maxabs, rmse_tol, maxabs_tol, n,
                                 False, status=status)
    passed = bool(rmse <= rmse_tol and maxabs <= maxabs_tol)
    return FieldParityResult(name, rmse, maxabs, rmse_tol, maxabs_tol, n, passed)


# ---------------------------------------------------------------------------
# Oracle discovery + case selection.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OracleCase:
    """One discovered oracle case (a corpus run dir with wrfinput/wrfbdy)."""

    case_id: str
    run_dir: Path
    level: str  # "l2" | "l3"
    wrfinput: Mapping[str, Path]  # domain -> path  ("d01".."d05")
    wrfbdy_d01: Path | None

    def domains(self) -> tuple[str, ...]:
        return tuple(sorted(self.wrfinput))


def discover_oracle_cases(
    root: str | Path = ORACLE_ROOT_DEFAULT,
    *,
    levels: Sequence[str] = ("wrf_l3", "wrf_l2"),
    require_domains: Sequence[str] = ("d01", "d02", "d03"),
    require_wrfbdy: bool = True,
    limit: int | None = None,
) -> list[OracleCase]:
    """Discover corpus oracle cases that carry the required wrfinput/wrfbdy set.

    Selects run dirs that have a ``wrfinput_d0N`` for every ``require_domains``
    entry and (optionally) a ``wrfbdy_d01``. NON-GPU: filesystem stat only.
    """
    root = Path(root)
    found: list[OracleCase] = []
    for level_dir in levels:
        base = root / level_dir
        if not base.is_dir():
            continue
        level_tag = "l2" if "l2" in level_dir else "l3"
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            inputs: dict[str, Path] = {}
            for dom in (f"d0{i}" for i in range(1, 6)):
                p = run_dir / f"wrfinput_{dom}"
                if p.is_file():
                    inputs[dom] = p
            if not all(d in inputs for d in require_domains):
                continue
            bdy = run_dir / "wrfbdy_d01"
            bdy_path = bdy if bdy.is_file() else None
            if require_wrfbdy and bdy_path is None:
                continue
            found.append(
                OracleCase(run_dir.name, run_dir, level_tag, inputs, bdy_path)
            )
            if limit is not None and len(found) >= limit:
                return found
    return found


# ---------------------------------------------------------------------------
# Candidate factory protocol + the campaign runner.
# ---------------------------------------------------------------------------
# A candidate factory builds the native RealInitProduct the comparator scores.
# Signature: factory(case, domain) -> RealInitProduct.
# At integration the manager passes a factory that calls
# driver.build_real_init(config, metem, forcing_sequence=..., domain=domain).
# For the S4 self-test we use oracle-derived factories (sanity + stub).
CandidateFactory = Callable[[OracleCase, str], RealInitProduct]


def run_campaign(
    product_factory: CandidateFactory,
    *,
    root: str | Path = ORACLE_ROOT_DEFAULT,
    domains: Sequence[str] = ("d01", "d02", "d03"),
    min_cases: int = 10,
    max_cases: int | None = None,
    include_wrfbdy: bool = True,
    cases: Sequence[OracleCase] | None = None,
) -> dict[str, Any]:
    """Run the full field-parity campaign across >=``min_cases`` oracle cases.

    For each case and each requested domain it builds the candidate via
    ``product_factory`` and scores ``compare_wrfinput`` (always) +
    ``compare_wrfbdy`` (d01 only, when ``include_wrfbdy``). Returns the campaign
    roll-up dict (the proof object body). NON-GPU offline scoring (cores 0-3).
    """
    if cases is None:
        cases = discover_oracle_cases(
            root, require_domains=domains, require_wrfbdy=include_wrfbdy,
            limit=max_cases)
    selected = list(cases)
    case_reports: list[dict[str, Any]] = []
    all_input_results: list[CaseParityResult] = []
    all_bdy_results: list[CaseParityResult] = []

    for oc in selected:
        per_domain: list[dict[str, Any]] = []
        for dom in domains:
            if dom not in oc.wrfinput:
                continue
            product = product_factory(oc, dom)
            inp = compare_wrfinput(product, oc.wrfinput[dom])
            all_input_results.append(inp)
            entry: dict[str, Any] = {"domain": dom, "wrfinput": inp.to_dict()}
            if include_wrfbdy and dom == "d01" and oc.wrfbdy_d01 is not None:
                bdy = compare_wrfbdy(product, oc.wrfbdy_d01)
                all_bdy_results.append(bdy)
                entry["wrfbdy"] = bdy.to_dict()
            per_domain.append(entry)
        case_reports.append({
            "case_id": oc.case_id,
            "level": oc.level,
            "run_dir": str(oc.run_dir),
            "domains": per_domain,
        })

    n_cases = len(selected)
    input_pass = all(r.passed for r in all_input_results) if all_input_results else False
    bdy_pass = all(r.passed for r in all_bdy_results) if all_bdy_results else (not include_wrfbdy)
    enough_cases = n_cases >= min_cases
    campaign_pass = bool(input_pass and bdy_pass and enough_cases)

    return {
        "schema": COMPARATOR_SCHEMA_VERSION,
        "n_cases": n_cases,
        "min_cases_required": min_cases,
        "enough_cases": enough_cases,
        "domains_scored": list(domains),
        "wrfinput_all_pass": input_pass,
        "wrfbdy_all_pass": bdy_pass,
        "campaign_pass": campaign_pass,
        "field_failure_summary": _field_failure_summary(
            all_input_results + all_bdy_results),
        "cases": case_reports,
    }


def _field_failure_summary(
    results: Sequence[CaseParityResult],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-field worst-rmse / fail-count across all (case,domain)."""
    summary: dict[str, dict[str, Any]] = {}
    for r in results:
        for f in r.fields:
            s = summary.setdefault(
                f.name,
                {"worst_rmse": 0.0, "worst_maxabs": 0.0, "rmse_tol": f.rmse_tol,
                 "maxabs_tol": f.maxabs_tol, "n_fail": 0, "n_scored": 0,
                 "statuses": {}},
            )
            s["statuses"][f.status] = s["statuses"].get(f.status, 0) + 1
            if f.status == "OK":
                s["n_scored"] += 1
                if np.isfinite(f.rmse):
                    s["worst_rmse"] = max(s["worst_rmse"], f.rmse)
                if np.isfinite(f.maxabs):
                    s["worst_maxabs"] = max(s["worst_maxabs"], f.maxabs)
            if not f.passed:
                s["n_fail"] += 1
    return summary


# ---------------------------------------------------------------------------
# 24h native-init FORECAST gate — SPEC + scaffold (NO GPU run here).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastGateSpec:
    """Predeclared spec for the 24h native-init forecast gate (S5/manager runs).

    The honest end-to-end standalone proof: native real-init -> native GPU
    forecast (NO replay of CPU-WRF boundaries beyond the native wrfbdy) -> 24h ->
    full-field comparison to the CPU-WRF wrfout for the same case. Reuses the
    existing per-lead continuous-gate metric set so the number tracks the same
    regression envelope as v0.2.0.
    """

    # Metric fields (mirror proofs/m20/continuous_gate.py CORE_FIELDS + the
    # PBLH/precip diagnostics the S0 plan section 5 calls out).
    core_fields: tuple[str, ...] = ("T2", "U10", "V10")
    diag_fields: tuple[str, ...] = ("PBLH", "RAINNC", "RAINC", "Q2", "PSFC")
    forecast_hours: int = 24
    # Per-lead gridpoint-paired metric: bias = mean(native - CPU),
    # rmse = sqrt(mean((native-CPU)^2)). PASS = worst-over-leads |bias| and rmse
    # within the frozen continuous-gate REGRESSION_MARGINS for the core fields;
    # diagnostics are descriptive (reported, not blocking) per the S0 plan.
    metric: str = "per-lead gridpoint-paired bias+rmse vs CPU-WRF wrfout"
    no_replay: bool = True  # native wrfbdy drives the LBC, NOT a CPU-WRF replay
    require_conservation_pass: bool = True
    require_restart_pass: bool = True
    # CPU-WRF reference roots (the backfill output + retained corpus wrfout).
    cpu_wrfout_roots: tuple[str, ...] = (
        "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output",
    )


def run_forecast_gate(
    product_factory: CandidateFactory | None = None,
    *,
    spec: ForecastGateSpec | None = None,
    cases: Sequence[OracleCase] | None = None,
    out_path: str | Path | None = None,
    execute: bool = False,
    forecast_hours: int | None = None,
    max_cases: int | None = None,
    dt_s: float = 60.0,
    acoustic_substeps: int = 4,
    radiation_cadence_steps: int = 30,
    output_root: str | Path = "/tmp/v040_forecast_gate",
) -> dict[str, Any]:
    """24h native-init forecast gate — the GPU-bound S5/manager entry point.

    With ``execute=False`` (the default) it returns the fully-specified PLAN dict
    (no GPU touched): which cases, which metric fields, the forecast length, the
    no-replay invariant, and the conservation/restart pre-conditions.

    With ``execute=True`` (the S5 GPU body, wired 2026-06-02) it, per case:

      1. Builds the native RealInitProduct via ``product_factory`` (the S5
         integrated ``driver.build_real_init`` factory).
      2. Packs the IC ``State`` + ``BaseState`` + the lateral-boundary leaves
         purely from the native product (the NATIVE wrfbdy DECOUPLED to the
         operational leaf layout is the ONLY LBC source — NO CPU-WRF replay).
      3. Drives the SAME validated operational forecast entry the d02/d03 replay
         path uses (``run_forecast_operational_segmented``) for ``forecast_hours``
         — ONE GPU job at a time — writing a per-lead wrfout.
      4. Scores the emitted wrfout per-lead vs the CPU-WRF wrfout reference using
         the frozen continuous-gate metric set (T2/U10/V10 blocking;
         PSFC/PBLH/Q2 descriptive) and records the per-lead stability
         (finite + gross-physical-range) check.

    The GPU body lives in ``proofs/v040/s5_forecast_gate_exec.py`` (the single-GPU
    serialization point per V0.4.0-S0-PLAN.md section 6); this entry imports it
    lazily so the comparator stays a light non-GPU import.
    """
    spec = spec or ForecastGateSpec()
    if cases is None:
        cases = discover_oracle_cases(
            require_domains=("d01",), require_wrfbdy=True, limit=None)
    cases = list(cases)
    hours = int(forecast_hours if forecast_hours is not None else spec.forecast_hours)
    plan = {
        "schema": "v0.4.0-S4-forecast-gate-spec-2026-06-02",
        "executed": False,
        "gpu_bound": True,
        "owner": "S5/manager (single-GPU serialization point)",
        "forecast_hours": hours,
        "no_replay": spec.no_replay,
        "metric": spec.metric,
        "core_fields_blocking": list(spec.core_fields),
        "diag_fields_descriptive": list(spec.diag_fields),
        "require_conservation_pass": spec.require_conservation_pass,
        "require_restart_pass": spec.require_restart_pass,
        "cpu_wrfout_roots": list(spec.cpu_wrfout_roots),
        "candidate_cases": [oc.case_id for oc in cases],
        "n_candidate_cases": len(cases),
        "continuous_gate_module": "proofs/m20/continuous_gate.py",
        "note": (
            "SCAFFOLD/PLAN with execute=False; execute=True runs the GPU body in "
            "proofs/v040/s5_forecast_gate_exec.py (native-init -> forecast, no replay)."
        ),
    }
    if not execute:
        if out_path is not None:
            import json
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        return plan

    if product_factory is None:
        raise ValueError(
            "run_forecast_gate(execute=True) requires the integrated product_factory "
            "(proofs.v040.s5_native_init_parity.make_factory)")

    # --- GPU body (lazy import; single-GPU serialization point) ---------------
    import json
    import re
    import sys
    from datetime import datetime, timezone

    sys_path0 = str(Path(__file__).resolve().parents[3] / "proofs" / "v040")
    if sys_path0 not in sys.path:
        sys.path.insert(0, sys_path0)
    proj_root = Path(__file__).resolve().parents[3]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from s5_forecast_gate_exec import run_one_case_forecast_gate  # type: ignore

    def _init_vt_label(case_id: str) -> str:
        ymd = case_id[:8]
        return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}_18:00:00"

    def _run_start(case_id: str) -> datetime:
        ymd = case_id[:8]
        return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]), 18, 0, 0,
                        tzinfo=timezone.utc)

    out_root = Path(output_root)
    selected = cases if max_cases is None else cases[: int(max_cases)]
    case_records: list[dict[str, Any]] = []
    for oc in selected:
        init_label = _init_vt_label(oc.case_id)
        ref_dir = oc.run_dir
        if not (ref_dir / f"wrfout_d01_{init_label}").is_file():
            case_records.append({"case_id": oc.case_id, "status": "NO_REFERENCE_T0_WRFOUT"})
            continue
        product = product_factory(oc, "d01")
        rec = run_one_case_forecast_gate(
            product,
            case_id=oc.case_id,
            reference_run_dir=ref_dir,
            run_start=_run_start(oc.case_id),
            init_vt_label=init_label,
            forecast_hours=hours,
            dt_s=float(dt_s),
            acoustic_substeps=int(acoustic_substeps),
            radiation_cadence_steps=int(radiation_cadence_steps),
            output_dir=out_root / f"{oc.case_id}_d01",
            core_fields=spec.core_fields,
            diag_fields=spec.diag_fields,
        )
        case_records.append(rec)

    scored = [r for r in case_records if r.get("status") != "NO_REFERENCE_T0_WRFOUT"]
    all_stable = bool(scored) and all(
        r.get("stability", {}).get("stable_finite") for r in scored)
    all_physical = bool(scored) and all(
        r.get("stability", {}).get("physical_range_ok") for r in scored)
    all_core_pass = bool(scored) and all(r.get("core_within_margin") for r in scored)

    if all_stable and all_physical and all_core_pass:
        verdict = "FOUNDATION_CONFIRMED"
    elif all_stable and all_physical:
        verdict = "STABLE_BUT_CORE_FIELD_MISMATCH"
    elif all_stable:
        verdict = "FINITE_BUT_UNPHYSICAL"
    else:
        verdict = "BLOWUP"

    result = dict(plan)
    result.update({
        "executed": True,
        "verdict": verdict,
        "foundation_confirmed": verdict == "FOUNDATION_CONFIRMED",
        "n_cases_scored": len(scored),
        "all_cases_stable_finite": all_stable,
        "all_cases_physical": all_physical,
        "all_cases_core_within_margin": all_core_pass,
        "forecast_dt_s": float(dt_s),
        "forecast_acoustic_substeps": int(acoustic_substeps),
        "cases": case_records,
    })
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2, default=str) + "\n",
                                  encoding="utf-8")
    return result


__all__ = [
    "COMPARATOR_SCHEMA_VERSION",
    "ORACLE_ROOT_DEFAULT",
    "FieldParityResult",
    "CaseParityResult",
    "OracleCase",
    "ForecastGateSpec",
    "compare_wrfinput",
    "compare_wrfbdy",
    "discover_oracle_cases",
    "run_campaign",
    "run_forecast_gate",
]
