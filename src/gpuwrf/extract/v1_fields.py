"""EXTRACT_SPEC V1 mass-grid fields of one history frame (pure numpy, host only).

Inputs are the exact host arrays the wrfout writer writes for the frame
(WRF names, float32, WRF shapes without the Time axis), so every value here is
reproducible offline from the full wrfout of the same run.

Arithmetic order (fixed, recorded in the manifest):
* rotation / speed / direction: float32 as ALISIOS ``ops/nightly_9nest/extract9.py``
  (``u10_east=U10*COSALPHA-V10*SINALPHA`` ...; ``np.degrees`` for 180/pi);
  calm (wind_speed == 0 exactly) -> direction NaN.
* cloud_max / cloud_low_max / LWP: verbatim ``cloud9.frame_cloud`` (sha 26424dc4...),
  z_w = (f8(PH)+f8(PHB))/9.81, z_msl = f4(mean of bounding z_w), z_agl = z_msl - HGT (f4).
* IWP: teacher_extract_v2 ``liquid_ice_mass`` (f8 sum, one f4 rounding).
* cloud base: lowest mass level with f4(QCLOUD+QICE) >= f4(1e-5); z_msl/z_agl of that level.
* inversion V1 (TM-S6 ``reretro.inversion_base`` order): theta_K = f4(f8(T)+300),
  gradient in f8 from the f4 theta_K and z_msl values.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

G = 9.81
CLOUD_BASE_THRESHOLD = np.float32(1.0e-5)
LOW_CLOUD_AGL_M = 2000.0
INV_ZMIN, INV_ZMAX, INV_MIN_GRAD = 300.0, 3500.0, 0.002

STATIC = ("XLAT", "XLONG", "HGT", "LANDMASK", "COSALPHA", "SINALPHA")
# Spec V1 surface list + Addendum 1 (SST, TSK). Mandatory: missing => FAIL.
SURFACE = ("U10", "V10", "T2", "Q2", "PSFC", "PBLH", "HFX", "LH", "UST",
           "RAINC", "RAINNC", "SWDOWN", "GLW", "SST", "TSK")
OPTIONAL_SURFACE = ("WSPD10MAX",)  # only under nwp_diagnostics=1
VOLUME = ("CLDFRA", "PH", "PHB", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "MU", "MUB", "T")
COLUMN = ("DNW", "C1H", "C2H")
ICE_SPECIES = ("QICE", "QSNOW", "QGRAUP")
# Raw mixing ratios below this are a QC failure; smaller negatives are round-off and only diagnosed
# (pristine CPU-WRF history carries ~-3.5e-21 kg/kg).
NEG_Q_FAIL = -1.0e-6
# NetCDF/WRF fill values (9.96921e36) are never data.
FILL_THRESHOLD = 9.9e36
# Surface names that exist only with a producer in the run (configuration, not a fault): absent -> NOT_AVAILABLE.
PRODUCER_OPTIONAL = ("SST",)
# Recorded SST provenance (field_status.source) = WHAT the value is, never HOW this extractor reached it, so the
# in-run extract and the offline re-extract of the same run are byte-identical (GPU arm 11.11 gate5 found only this
# attribute differing). WRF Registry.EM SST "i01245rh": grid%sst is read from wrfinput and replaced by each
# auxinput4 (wrflowinp) record under sst_update=1, written to history as held; water TSK = SST
# (module_surface_driver.F:1727). gpuwrf holds the same record (CPU-WRF oracle 225/225 exact, bac2a6457a).
SST_SOURCE = ("WRF state SST as written to history: the lower-boundary record held by the model (wrfinput at "
              "lead 0, then each wrflowinp record from its auxinput4 time under sst_update=1)")
# Ice species each WRF microphysics scheme carries (Registry moist arrays); unknown schemes -> None.
_ICE_BY_MP = {0: (), 1: (), 3: (), 4: ("QICE", "QSNOW"), 14: ("QICE", "QSNOW"),
              **{mp: ICE_SPECIES for mp in (2, 6, 7, 8, 10, 16, 24, 26, 28, 38)}}


def expected_ice_species(mp_physics):
    """Ice species the scheme must provide, () for warm-rain schemes, None when the scheme is not tabulated."""
    return None if mp_physics is None else _ICE_BY_MP.get(int(mp_physics))
# Names the extract needs from the prepared history payload (device fetch set).
RAW_NAMES = tuple(dict.fromkeys(STATIC + SURFACE + OPTIONAL_SURFACE + VOLUME + COLUMN))

# Per-cell status codes (int8). NaN in a float field is explained by its status.
CLOUD_BASE_STATUS = {0: "OK", 1: "CLEAR", 2: "MISSING_INPUT"}
INV_STATUS = {0: "OK", 1: "NO_INVERSION", 2: "MISSING_INPUT", 3: "NO_LAYER_IN_RANGE"}

GRID_FIELDS = (
    "U10", "V10", "u10_east", "v10_north", "wind_speed", "wind_direction_from_deg",
    "T2", "Q2", "PSFC", "PBLH", "HFX", "LH", "UST", "ZNT", "RAINC", "RAINNC",
    "SWDOWN", "GLW", "SST", "TSK", "WSPD10MAX",
    "cloud_max", "cloud_low_max", "LWP", "IWP",
    "cloud_base_msl", "cloud_base_agl", "cloud_base_status",
    "inv_base_msl", "inv_top_msl", "inv_strength_K", "inv_gradient_K_m", "inv_status",
)
STATUS_FIELDS = ("cloud_base_status", "inv_status")
# Time sampling of each frame value (manifest output_cadence; every field not listed is instantaneous).
SAMPLING = {
    "WSPD10MAX": "maximum over the producer's reset window ending at this frame -- NOT always since the previous "
                 "output: AFWA severe resets on the step after each history write (window = previous history "
                 "step + 1 .. this step); nwp_diagnostics resets every idump = trunc(interval*60/dt) steps "
                 "(window = 1 + idump*floor((step-1)/idump) .. step, shorter than the interval when interval/dt "
                 "is not an integer). Exact window per frame: field_status.WSPD10MAX.window_own_steps and "
                 "manifest domains.dNN.output_cadence.wspd10max",
    "RAINC": "accumulated since WRF start", "RAINNC": "accumulated since WRF start",
}

UNITS = {
    "XLAT": "degree_north", "XLONG": "degree_east", "HGT": "m", "LANDMASK": "1",
    "COSALPHA": "1", "SINALPHA": "1",
    "U10": "m s-1 (grid relative)", "V10": "m s-1 (grid relative)",
    "u10_east": "m s-1 (earth relative)", "v10_north": "m s-1 (earth relative)",
    "wind_speed": "m s-1", "wind_direction_from_deg": "degree (from, earth relative)",
    "T2": "K", "Q2": "kg kg-1", "PSFC": "Pa", "PBLH": "m AGL (native WRF diagnosis)",
    "HFX": "W m-2 (WRF sign, upward positive)", "LH": "W m-2 (WRF sign, upward positive)",
    "UST": "m s-1", "ZNT": "m", "RAINC": "mm (accumulated since WRF start)",
    "RAINNC": "mm (accumulated since WRF start)", "SWDOWN": "W m-2", "GLW": "W m-2",
    "SST": "K", "TSK": "K", "WSPD10MAX": "m s-1",
    "cloud_max": "1", "cloud_low_max": "1", "LWP": "kg m-2", "IWP": "kg m-2",
    "cloud_base_msl": "m MSL", "cloud_base_agl": "m AGL", "cloud_base_status": "code",
    "inv_base_msl": "m MSL", "inv_top_msl": "m MSL", "inv_strength_K": "K",
    "inv_gradient_K_m": "K m-1", "inv_status": "code",
}

DEFINITIONS = {
    "u10_east": "f4: U10*COSALPHA - V10*SINALPHA (extract9.py order)",
    "v10_north": "f4: V10*COSALPHA + U10*SINALPHA",
    "wind_speed": "f4: hypot(u10_east, v10_north)",
    "wind_direction_from_deg": "f4: (degrees(atan2(-u10_east,-v10_north))+360)%360; NaN where wind_speed == 0 (calm)",
    "ZNT": "carried State.roughness_m of the frame (current ZNT; same source as lane/z0bare --history-znt)",
    "WSPD10MAX": "WRF WSPD10MAX as written: max 10 m wind speed over the producer's reset window (see sampling; "
                 "AFWA severe also raises it towards the 1 km AGL wind where the step precipitation exceeds "
                 "50 mm/h); NOT_AVAILABLE without a producer; a model wind maximum, not an observed gust",
    "cloud_max": "max_k CLDFRA (cloud9.py verbatim)",
    "cloud_low_max": "max of CLDFRA over mass levels with z_agl < 2000 m; NaN if none (cloud9.py verbatim)",
    "LWP": "sum_k (QCLOUD+QRAIN)*m_k, m_k=-(C1H*(MU+MUB)+C2H)*DNW/9.81, f8 sum -> f4 (cloud9.py verbatim, incl. rain like WT1)",
    "IWP": "sum_k (QICE+QSNOW+QGRAUP)*m_k, f8 sum -> f4 (teacher_extract_v2.liquid_ice_mass); NEW contract incl. precipitating ice",
    "cloud_base_msl": "z_msl of the lowest mass level with f4(QCLOUD+QICE) >= 1e-5 kg/kg; no interpolation; NaN if CLEAR/MISSING_INPUT; not an observed ceiling",
    "cloud_base_agl": "z_agl of the same level",
    "inv_base_msl": "TM-S6 compatible: theta_K=f4(f8(T)+300); gk=(theta[k+1]-theta[k])/(z[k+1]-z[k]) in f8; layers with z[k]>=300 and z[k+1]<=3500 m MSL; largest gk (lowest k on ties); OK if gk>=0.002 K/m: z[k]",
    "inv_top_msl": "z[k+1] of the selected layer",
    "inv_strength_K": "f4(theta[k+1]-theta[k]) in f8",
    "inv_gradient_K_m": "f4(gk)",
}


class MissingInput(KeyError):
    pass


def vertical_geometry(raw: Mapping[str, np.ndarray]):
    """z_w (f8), z_msl (f4), z_agl (f4) exactly as cloud9.frame_cloud."""
    geop = np.asarray(raw["PH"], np.float32).astype("f8") + np.asarray(raw["PHB"], np.float32).astype("f8")
    zw = geop / G
    zmsl = ((zw[:-1] + zw[1:]) / 2).astype("f4")
    zagl = zmsl - np.asarray(raw["HGT"], np.float32)
    return zw, zmsl, zagl


def column_mass(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    mu = np.asarray(raw["MU"], np.float32).astype("f8") + np.asarray(raw["MUB"], np.float32).astype("f8")
    dnw = np.asarray(raw["DNW"], np.float32).astype("f8")
    c1 = np.asarray(raw["C1H"], np.float32).astype("f8")
    c2 = np.asarray(raw["C2H"], np.float32).astype("f8")
    return -(c1[:, None, None] * mu + c2[:, None, None]) * dnw[:, None, None] / G


def rotate_wind(u10, v10, cosalpha, sinalpha):
    u10, v10 = np.asarray(u10, np.float32), np.asarray(v10, np.float32)
    c, s = np.asarray(cosalpha, np.float32), np.asarray(sinalpha, np.float32)
    ue = u10 * c - v10 * s
    vn = v10 * c + u10 * s
    spd = np.hypot(ue, vn)
    wdir = (np.degrees(np.arctan2(-ue, -vn)) + 360.0) % 360.0
    wdir = np.where(spd == 0, np.float32(np.nan), wdir).astype(np.float32)
    return ue, vn, spd, wdir


def cloud_fields(cf, zagl):
    """cloud_max / cloud_low_max verbatim cloud9.frame_cloud."""
    out = {"cloud_max": cf.max(axis=0)}
    mask = (zagl >= -np.inf) & (zagl < LOW_CLOUD_AGL_M)
    val = np.max(np.where(mask, cf, -np.inf), axis=0)
    val[~mask.any(axis=0)] = np.nan
    out["cloud_low_max"] = val
    return {k: np.asarray(v, dtype="f4") for k, v in out.items()}


def water_path(raw, species, mass):
    return sum(np.asarray(raw[k], np.float32).astype("f8") * mass for k in species).sum(axis=0).astype("f4")


def cloud_base(qcloud, qice, zmsl, zagl):
    q = np.asarray(qcloud, np.float32) + np.asarray(qice, np.float32)
    hit = q >= CLOUD_BASE_THRESHOLD
    anyhit = hit.any(axis=0)
    k = np.argmax(hit, axis=0)
    base_msl = np.take_along_axis(zmsl, k[None], axis=0)[0]
    base_agl = np.take_along_axis(zagl, k[None], axis=0)[0]
    status = np.where(anyhit, 0, 1).astype(np.int8)
    nan = np.float32(np.nan)
    return (np.where(anyhit, base_msl, nan).astype(np.float32),
            np.where(anyhit, base_agl, nan).astype(np.float32), status)


def inversion_v1(t_pert, zmsl):
    """Vectorised TM-S6 inversion_base + layer top/strength/gradient."""
    theta = (np.asarray(t_pert, np.float32).astype("f8") + 300).astype("f4")
    th = theta.astype("f8")
    zz = np.asarray(zmsl, np.float32).astype("f8")
    nz = th.shape[0]
    finite = np.isfinite(th).all(axis=0) & np.isfinite(zz).all(axis=0)
    ascending = (np.diff(zz, axis=0) > 0).all(axis=0) if nz > 1 else np.zeros(th.shape[1:], bool)
    complete = finite & ascending & (nz >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.diff(th, axis=0) / np.diff(zz, axis=0)
    sel = (zz[:-1] >= INV_ZMIN) & (zz[1:] <= INV_ZMAX)
    has = sel.any(axis=0)
    gm = np.where(sel, g, -np.inf)
    k = np.argmax(gm, axis=0)
    gk = np.take_along_axis(g, k[None], axis=0)[0]
    ok = complete & has & (gk >= INV_MIN_GRAD)
    status = np.where(~complete, 2, np.where(~has, 3, np.where(ok, 0, 1))).astype(np.int8)
    take = lambda a, kk: np.take_along_axis(a, kk[None], axis=0)[0]
    nan = np.float32(np.nan)
    zb = take(np.asarray(zmsl, np.float32), k)
    zt = take(np.asarray(zmsl, np.float32), np.minimum(k + 1, nz - 1))
    strength = (take(th, np.minimum(k + 1, nz - 1)) - take(th, k)).astype("f4")
    return {
        "inv_base_msl": np.where(ok, zb, nan).astype(np.float32),
        "inv_top_msl": np.where(ok, zt, nan).astype(np.float32),
        "inv_strength_K": np.where(ok, strength, nan).astype(np.float32),
        "inv_gradient_K_m": np.where(ok, gk.astype("f4"), nan).astype(np.float32),
        "inv_status": status,
    }


def _range_check(qc, name, arr, lo=None, hi=None, lo_open=False, hi_open=False):
    a = np.asarray(arr)
    fin = np.isfinite(a)
    bad = 0
    if lo is not None:
        bad += int(np.count_nonzero(fin & ((a <= lo) if lo_open else (a < lo))))
    if hi is not None:
        bad += int(np.count_nonzero(fin & ((a >= hi) if hi_open else (a > hi))))
    if bad:
        qc.setdefault("range_failures", {})[name] = bad


def frame_fields(raw: Mapping[str, np.ndarray], znt: np.ndarray | None, *,
                 mp_physics: int | None = None, cu_physics: int | None = None, unobserved=(),
                 provenance: Mapping[str, str] | None = None):
    """All V1 mass-grid fields of one frame.

    Returns ``(fields, status, qc)``: fields name->(ny,nx) array, per-field status
    (OK / MISSING / UNAVAILABLE / NOT_AVAILABLE / PARTIAL_SPECIES / MISSING_INPUT
    with reason), and the frame QC record (counts, range failures; never clips).

    ``unobserved``: surface names whose payload value is a history-writer compatibility fallback
    (no producer in the run). They are never reported as observed and never substituted: SST ->
    NOT_AVAILABLE (gpuwrf has no SST carry; only the lower-boundary wrflowinp records under sst_update
    produce it); RAINC with cu_physics == 0 -> kept only if all zero (WRF definition); any other -> MISSING.
    ``provenance``: name -> where its producer was found (recorded as ``source`` for SST).
    ``mp_physics`` fixes the ice species the scheme must provide (missing one -> MISSING_INPUT).
    WSPD10MAX (max 10 m wind speed over the producer reset window, see SAMPLING; WRF nwp_diagnostics=1 or AFWA
    afwa_severe_opt=1): extracted whenever the frame carries it from a producer, else NOT_AVAILABLE (NaN);
    the history writer never fabricates it, and an unproduced payload value is refused (``unobserved``).
    """
    fields: dict[str, np.ndarray] = {}
    status: dict[str, dict] = {}
    qc: dict = {}
    unobserved = set(unobserved)
    nonfinite_in, fill_in = {}, {}
    for n in raw:  # every input incl. the profile-only ones (P, PB, QVAPOR, U, V, QNCLOUD)
        a = np.asarray(raw[n])
        if a.dtype.kind == "f":
            bad = int(np.count_nonzero(~np.isfinite(a)))
            if bad:
                nonfinite_in[n] = bad
            fill = int(np.count_nonzero(np.abs(a) >= FILL_THRESHOLD))
            if fill:
                fill_in[n] = fill
    if fill_in:
        qc["fill_value_inputs"] = fill_in
    if znt is not None and np.asarray(znt).dtype.kind == "f":
        bad = int(np.count_nonzero(~np.isfinite(np.asarray(znt))))
        if bad:
            nonfinite_in["ZNT"] = bad
    if nonfinite_in:
        qc["nonfinite_inputs"] = nonfinite_in
    missing = sorted(n for n in STATIC + tuple(x for x in SURFACE if x not in PRODUCER_OPTIONAL) + ("CLDFRA", "PH", "PHB", "QCLOUD", "QRAIN", "MU", "MUB", "T") + COLUMN
                     if n not in raw)
    if missing:
        qc["missing_mandatory_inputs"] = missing
    if "HGT" not in raw:
        raise MissingInput("HGT")
    hgt = np.asarray(raw["HGT"], np.float32)
    ny, nx = hgt.shape
    nan2d = np.full((ny, nx), np.nan, np.float32)

    def put(name, value, st="OK", reason=None):
        fields[name] = np.asarray(value)
        status[name] = {"status": st, **({"reason": reason} if reason else {})}

    for name in SURFACE:
        if name in PRODUCER_OPTIONAL and (name in unobserved or name not in raw):
            put(name, nan2d, "NOT_AVAILABLE",
                "no SST producer in this run (gpuwrf carries no SST; only the lower-boundary wrflowinp records "
                "under sst_update=1 provide it); never substituted by TSK")
        elif name not in raw:
            put(name, nan2d, "MISSING", "not present in the frame's history payload for this configuration")
        elif name not in unobserved:
            put(name, np.asarray(raw[name], np.float32))
            if name in PRODUCER_OPTIONAL and provenance and provenance.get(name):
                status[name]["source"] = provenance[name]
        elif name == "RAINC" and cu_physics == 0 and not np.asarray(raw[name]).any():
            put(name, np.asarray(raw[name], np.float32))  # no cumulus scheme: zero by WRF definition
        else:
            put(name, nan2d, "MISSING", "no producer in this run (history value is a writer fallback)")
    if unobserved:
        qc["unobserved_payload_fields"] = sorted(unobserved)
    if "WSPD10MAX" in raw and "WSPD10MAX" not in unobserved:
        put("WSPD10MAX", np.asarray(raw["WSPD10MAX"], np.float32))
        status["WSPD10MAX"]["sampling"] = SAMPLING["WSPD10MAX"]
    else:
        put("WSPD10MAX", nan2d, "NOT_AVAILABLE",
            "no WSPD10MAX producer in this run (WRF nwp_diagnostics=1 or AFWA afwa_severe_opt=1); "
            "never a fallback, not a gust")
    if znt is not None:
        z = np.asarray(znt, np.float32).reshape(ny, nx)
        put("ZNT", z)
    else:
        put("ZNT", nan2d, "MISSING", "carried roughness (State.roughness_m) unavailable")

    if all(n in raw for n in ("U10", "V10", "COSALPHA", "SINALPHA")):
        ue, vn, spd, wdir = rotate_wind(raw["U10"], raw["V10"], raw["COSALPHA"], raw["SINALPHA"])
        put("u10_east", ue); put("v10_north", vn); put("wind_speed", spd); put("wind_direction_from_deg", wdir)
        qc["calm_cells"] = int(np.count_nonzero(spd == 0))
    else:
        for n in ("u10_east", "v10_north", "wind_speed", "wind_direction_from_deg"):
            put(n, nan2d, "MISSING", "U10/V10/COSALPHA/SINALPHA missing")

    vol_ok = all(n in raw for n in ("CLDFRA", "PH", "PHB", "HGT"))
    if vol_ok:
        zw, zmsl, zagl = vertical_geometry(raw)
        qc["z_w_monotone"] = bool(np.all(np.diff(zw, axis=0) > 0))
        cf = np.asarray(raw["CLDFRA"], np.float32)
        _range_check(qc, "CLDFRA", cf, 0.0, 1.0)
        cl = cloud_fields(cf, zagl)
        put("cloud_max", cl["cloud_max"]); put("cloud_low_max", cl["cloud_low_max"])
        qc["cloud_low_max_nan"] = int(np.isnan(cl["cloud_low_max"]).sum())
    else:
        zmsl = zagl = None
        for n in ("cloud_max", "cloud_low_max"):
            put(n, nan2d, "MISSING_INPUT", "CLDFRA/PH/PHB/HGT missing")

    mass_ok = all(n in raw for n in ("MU", "MUB", "DNW", "C1H", "C2H"))
    mass = column_mass(raw) if mass_ok else None
    if mass is not None:
        qc["mass_positive"] = bool(np.all(mass > 0))
    if mass is not None and all(n in raw for n in ("QCLOUD", "QRAIN")):
        put("LWP", water_path(raw, ("QCLOUD", "QRAIN"), mass))
    else:
        put("LWP", nan2d, "MISSING_INPUT", "QCLOUD/QRAIN or column-mass inputs missing")
    present = tuple(s for s in ICE_SPECIES if s in raw)
    expected_ice = expected_ice_species(mp_physics)
    lacking = [s for s in (expected_ice or ()) if s not in raw]
    if lacking:
        qc["missing_expected_species"] = lacking
        put("IWP", nan2d, "MISSING_INPUT", f"mp_physics={mp_physics} species missing from the payload: {lacking}")
    elif mass is not None and present:
        absent = [s for s in ICE_SPECIES if s not in raw and (expected_ice is None or s in expected_ice)]
        put("IWP", water_path(raw, present, mass), "OK" if not absent else "PARTIAL_SPECIES",
            None if not absent else f"species NOT_AVAILABLE in this scheme: {absent}; summed {list(present)}")
        status["IWP"]["species_used"] = list(present)
    else:
        put("IWP", nan2d, "NOT_AVAILABLE", "no ice species or column-mass inputs in this configuration")
    # Raw mixing ratios: negative values are DIAGNOSED (count + min), not FAIL.
    # Pristine CPU-WRF history itself carries round-off negatives (~-1e-21 kg/kg);
    # derived LWP/IWP < 0 stay FAIL below. Never clipped.
    for q in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNCLOUD"):
        if q in raw:
            a = np.asarray(raw[q])
            n = int(np.count_nonzero(a < 0))
            if n:
                qc.setdefault("negative_mixing_ratio", {})[q] = {"count": n, "min": float(np.nanmin(a))}
            big = int(np.count_nonzero(a < NEG_Q_FAIL))
            if big:
                qc.setdefault("range_failures", {})[f"{q}<{NEG_Q_FAIL:g}"] = big

    if zmsl is not None and "QCLOUD" in raw and "QICE" in raw:
        bm, ba, bs = cloud_base(raw["QCLOUD"], raw["QICE"], zmsl, zagl)
        put("cloud_base_msl", bm); put("cloud_base_agl", ba); put("cloud_base_status", bs)
    else:
        put("cloud_base_msl", nan2d, "MISSING_INPUT", "QCLOUD/QICE or vertical geometry missing")
        put("cloud_base_agl", nan2d, "MISSING_INPUT", "QCLOUD/QICE or vertical geometry missing")
        put("cloud_base_status", np.full((ny, nx), 2, np.int8), "MISSING_INPUT", "QCLOUD/QICE missing")
    if zmsl is not None and "T" in raw:
        for name, value in inversion_v1(raw["T"], zmsl).items():
            put(name, value)
    else:
        for name in ("inv_base_msl", "inv_top_msl", "inv_strength_K", "inv_gradient_K_m"):
            put(name, nan2d, "MISSING_INPUT", "T or vertical geometry missing")
        put("inv_status", np.full((ny, nx), 2, np.int8), "MISSING_INPUT", "T or vertical geometry missing")

    qc["cloud_base_status_counts"] = {CLOUD_BASE_STATUS[c]: int(np.count_nonzero(fields["cloud_base_status"] == c))
                                      for c in CLOUD_BASE_STATUS}
    qc["inv_status_counts"] = {INV_STATUS[c]: int(np.count_nonzero(fields["inv_status"] == c)) for c in INV_STATUS}
    # Spec QC ranges (diagnose, never clip).
    _range_check(qc, "LANDMASK", raw.get("LANDMASK", 0), 0.0, 1.0)
    for name, lo, hi, lo_open, hi_open in (
            ("PBLH", 0.0, None, False, False), ("ZNT", 0.0, None, True, False), ("UST", 0.0, None, False, False),
            ("LWP", 0.0, None, False, False), ("IWP", 0.0, None, False, False),
            ("T2", 180.0, 340.0, False, False), ("PSFC", 30000.0, 110000.0, False, False),
            ("wind_speed", None, 75.0, False, True)):
        if status[name]["status"] in ("OK", "PARTIAL_SPECIES"):
            _range_check(qc, name, fields[name], lo, hi, lo_open, hi_open)
    # Non-finite outputs: inf never; NaN only where its definition explains it.
    explained = {"wind_direction_from_deg": fields["wind_speed"] == 0,
                 "cloud_base_msl": fields["cloud_base_status"] != 0, "cloud_base_agl": fields["cloud_base_status"] != 0,
                 **{k: fields["inv_status"] != 0 for k in ("inv_base_msl", "inv_top_msl", "inv_strength_K",
                                                           "inv_gradient_K_m")}}
    if zagl is not None:
        explained["cloud_low_max"] = ~(np.asarray(zagl) < LOW_CLOUD_AGL_M).any(axis=0)
    unexplained = {}
    for name, st in status.items():
        if name in STATUS_FIELDS or st["status"] not in ("OK", "PARTIAL_SPECIES"):
            continue
        a = np.asarray(fields[name])
        bad = ~np.isfinite(a)
        if name in explained:
            bad &= ~(np.isnan(a) & explained[name])
        n = int(np.count_nonzero(bad))
        if n:
            unexplained[name] = n
    if unexplained:
        qc["unexplained_nonfinite"] = unexplained
    return fields, status, qc
