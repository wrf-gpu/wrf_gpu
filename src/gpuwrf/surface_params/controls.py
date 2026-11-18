"""Traced Noah-MP surface-parameter controls (Track P, K1; opt-in, default off).

A control replaces ONE per-cell field that the land model reads from a gathered table (or from the
wrfinput FVGMAX field) with a caller-supplied value field of the same shape and dtype:

    name     replaces (where Noah-MP reads it)                      WRF source
    z0mvt    EnergyParams.z0mvt  (ENERGY Z0M/Z0MG/Z0WRF -> ZNT)      MPTABLE Z0MVT(VEGTYP)
    eg       EnergyParams.eg     (ENERGY EMG -> EMISSI/TRAD/FIRA)    MPTABLE EG(1) (soil)
    vcmx25   EnergyParams.vcmx25 (STOMATA, opt_crs=1)                MPTABLE VCMX25(VEGTYP)
    mp       EnergyParams.mp     (STOMATA Ball-Berry slope)          MPTABLE MP(VEGTYP)
    albsat   TwoStreamParams.albsat (GROUNDALB, vis/nir)             MPTABLE ALBSAT(SOILCOLOR)
    albdry   TwoStreamParams.albdry (GROUNDALB, vis/nir)             MPTABLE ALBDRY(SOILCOLOR)
    rhol     TwoStreamParams.rhol (TWOSTREAM leaf reflectance)       MPTABLE RHOL(VEGTYP)
    rhos     TwoStreamParams.rhos (TWOSTREAM stem reflectance)       MPTABLE RHOS(VEGTYP)
    laim     PHENOLOGY monthly LAI (dveg 1/3/4)                      MPTABLE LAIM(VEGTYP,month)
    saim     PHENOLOGY monthly SAI (dveg 1/3/4)                      MPTABLE SAIM(VEGTYP,month)
    shdmax   NoahMPStatic.shdmax (FVEG=SHDMAX, dveg=4)               wrfinput VEGMAX/100

The released program gathers exactly these values; ``prior_controls`` copies them byte for byte, so a
run with prior controls reads the same numbers through a traced input instead of a compile constant.
``expand_controls`` builds perturbed fields the way an edited table would be gathered: the original
table value per cell times a float64 multiplier, rounded once to the consumer dtype (per-class
multipliers are therefore bitwise an edit of the table row for that class).

The fields ride ``_ClockBase.surface_controls`` (a traced argument of every step program: B=1 eager/AOT,
fused nest cascades and B>1 vmap lanes). The namelist keeps them only in a content-blind aux holder
(``SurfaceControlsAux``: names/shapes/dtypes), so the compile keys never see the values and the step never
bakes them. The explicit W2 registry below adds soil hydraulics, canopy heights, Jarvis RSMIN and
snow energy parameters; those names remain separate from the legacy Track P prior set.

HOST controls (``HOST_CONTROLS``) replace the value of a namelist child that is ALREADY a traced pytree leaf
(no clock-base field, no new program input), at bind time on the host:

    ctopo    TopoWindStatic.ctopo (YSU/Shin-Hong surface drag)       start_em CTOPO (topo_wind 1/2)

A file with only host controls therefore runs exactly the default program of that case. The ctopo control is
an EXCESS scale: CTOPO' = 1 + f (CTOPO - 1), rounded once to REAL, so cells without a correction keep CTOPO = 1
and f = 1 reproduces the start_em field byte for byte. Not with two-way feedback onto a topo_wind domain (the
initial feedback recomputes CTOPO from the fed-back terrain after the bind) and not in B>1 lanes (v0.4.1).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

SCHEMA = "gpuwrf-surface-controls-v1"
SOILCOLOR = 4  # build_energy_params: WRF/offline-driver default soil colour
NMONTH = 12


class ControlSpec(NamedTuple):
    bundle: str      # 'energy' | 'rad' | 'static'
    axis: str        # 'col' (ny,nx) | 'band' (2,ny,nx) | 'month' (12,ny,nx)
    lower: float     # physical sanity bounds (not priors)
    upper: float
    units: str
    source: str
    consumer: str


CONTROLS: dict[str, ControlSpec] = {
    "z0mvt": ControlSpec("energy", "col", 0.0, 5.0, "m", "MPTABLE Z0MVT(VEGTYP)",
                         "ENERGY Z0M/Z0MG -> VEGE_FLUX/SFCDIF1, Z0WRF -> ZNT -> surface layer"),
    "eg": ControlSpec("energy", "col", 0.5, 1.0, "1", "MPTABLE EG(1) ground emissivity",
                      "ENERGY EMG -> FIRA/TRAD/EMISSI"),
    "vcmx25": ControlSpec("energy", "col", 0.0, 300.0, "umol CO2 m-2 s-1", "MPTABLE VCMX25(VEGTYP)",
                          "STOMATA (opt_crs=1) -> RSSUN/RSSHA -> transpiration"),
    "mp": ControlSpec("energy", "col", 0.0, 20.0, "1", "MPTABLE MP(VEGTYP) Ball-Berry slope",
                      "STOMATA (opt_crs=1) -> RSSUN/RSSHA -> transpiration"),
    "albsat": ControlSpec("rad", "band", 0.0, 1.0, "1", "MPTABLE ALBSAT(SOILCOLOR) vis/nir",
                          "GROUNDALB -> TWOSTREAM -> SAV/SAG, ALBEDO"),
    "albdry": ControlSpec("rad", "band", 0.0, 1.0, "1", "MPTABLE ALBDRY(SOILCOLOR) vis/nir",
                          "GROUNDALB -> TWOSTREAM -> SAV/SAG, ALBEDO"),
    "rhol": ControlSpec("rad", "band", 0.0, 1.0, "1", "MPTABLE RHOL(VEGTYP) vis/nir",
                        "TWOSTREAM leaf scattering -> SAV/SAG, ALBEDO"),
    "rhos": ControlSpec("rad", "band", 0.0, 1.0, "1", "MPTABLE RHOS(VEGTYP) vis/nir",
                        "TWOSTREAM stem scattering -> SAV/SAG, ALBEDO"),
    "laim": ControlSpec("energy", "month", 0.0, 20.0, "m2 m-2", "MPTABLE LAIM(VEGTYP,month)",
                        "PHENOLOGY LAI -> ELAI -> TWOSTREAM/VEGE_FLUX/EMV"),
    "saim": ControlSpec("energy", "month", 0.0, 20.0, "m2 m-2", "MPTABLE SAIM(VEGTYP,month)",
                        "PHENOLOGY SAI -> ESAI -> TWOSTREAM/VEGE_FLUX/EMV"),
    "z0bare": ControlSpec("energy", "col", 1.0e-4, 0.5, "m", "EXPERIMENTAL ENERGY Z0 = 0.002 m (source constant)",
                          "ENERGY Z0MG = Z0*(1-FSNO)+Z0SNO*FSNO; spatial extension, not WRF-faithful"),
    "shdmax": ControlSpec("static", "col", 0.0, 1.0, "1", "wrfinput VEGMAX/100 (FVGMAX)",
                          "PHENOLOGY FVEG=SHDMAX (dveg=4) -> tile blend"),
}
# Host controls: the value of an already traced namelist child, replaced at bind time (same program).
HOST_CONTROLS: dict[str, ControlSpec] = {
    "ctopo": ControlSpec("topo_wind", "col", 0.0, 50.0, "1", "start_em CTOPO (topo_wind 1/2)",
                         "YSU/Shin-Hong momentum surface drag (module_pbl_driver.F:1232/1317)"),
}
ALL_CONTROLS: dict[str, ControlSpec] = {**CONTROLS, **HOST_CONTROLS}
_LEADING = {"col": (), "band": (2,), "month": (NMONTH,)}
_TRANSMITTANCE = {"rhol": "taul", "rhos": "taus"}


def excess_scale(base, factor):
    """Host-control expansion: 1 + factor (base - 1) in float64 (base = the start_em field, exact in f64)."""
    base = np.asarray(base, dtype=np.float64)
    return 1.0 + np.asarray(factor, dtype=np.float64) * (base - 1.0)


# --------------------------------------------------------------------------- holder
@dataclasses.dataclass(frozen=True)
class SurfaceControlsAux:
    """Content-blind namelist aux marker; the control VALUES ride the traced clock base.

    Equality/hash and ``aot_cheap_key._walk`` (dataclass fields) see only the signature
    (names, shapes, dtypes), so two parameter sets of one grid share every compile key.
    The values are kept in a non-field attribute for ``build_clock_base``.
    """

    signature: tuple


def controls_aux(controls: Mapping[str, Any]) -> SurfaceControlsAux:
    refuse_unqualified_controls(controls)
    aux = SurfaceControlsAux(signature(controls))
    object.__setattr__(aux, "value", dict(controls))
    return aux


def signature(controls: Mapping[str, Any]) -> tuple:
    """(name, shape, dtype) per control: everything that determines the traced program."""
    return tuple((name, tuple(int(n) for n in np.shape(controls[name])), str(np.dtype(controls[name].dtype)))
                 for name in sorted(controls))


# --------------------------------------------------------------------------- traced use
def apply_surface_controls(static, energy_params, rad_params, controls):
    """Swap the controlled fields into the Noah-MP bundles (traced; no arithmetic)."""
    refuse_unqualified_controls(controls)
    if energy_params is None or rad_params is None:
        raise ValueError("surface controls need the pre-built Noah-MP energy/radiation parameter bundles")
    from gpuwrf.surface_params.lsmparams import COEFFICIENT_KEYS
    coefficients = {COEFFICIENT_KEYS[n]: v for n, v in controls.items() if n in COEFFICIENT_KEYS}
    if coefficients:
        if (set(coefficients) & {"inv_smcmax", "thkdry"} and "smcmax" not in controls or
                "inv_d_rsurf" in coefficients and not set(controls) & {"smcmax", "bexp"}):
            raise ValueError("private W2 clock coefficients require their public soil controls")
        controls = {n: v for n, v in controls.items() if n not in COEFFICIENT_KEYS}
    unknown = sorted(set(controls) - set(CONTROLS) - set(LSM_CONTROLS))
    if unknown:
        raise ValueError(f"unknown surface controls {unknown}")
    if set(controls) & set(LSM_CONTROLS):
        from gpuwrf.surface_params.lsmparams import apply
        static, energy_params = apply(static, energy_params, {n: v for n, v in controls.items() if n in LSM_CONTROLS}, coefficients=coefficients)
    by_bundle: dict[str, dict[str, Any]] = {"energy": {}, "rad": {}, "static": {}}
    for name, value in controls.items():
        if name in LSM_CONTROLS:
            continue
        by_bundle[CONTROLS[name].bundle][name] = value
    if by_bundle["energy"]:
        energy_params = energy_params._replace(**by_bundle["energy"])
    if by_bundle["rad"]:
        rad_params = rad_params._replace(**by_bundle["rad"])
    if by_bundle["static"]:
        static = static.replace(**by_bundle["static"])
    return static, energy_params, rad_params


def traced_controls(namelist, clock_base):
    """The traced control set of this step (clock base), refusing to bake namelist values."""
    controls = None if clock_base is None else getattr(clock_base, "surface_controls", None)
    refuse_unqualified_controls(controls)
    refuse_unqualified_controls(getattr(namelist, "surface_controls", None))
    if controls is None and getattr(namelist, "surface_controls", None) is not None:
        raise ValueError("surface controls must ride the traced clock base (build_clock_base); "
                         "refusing to bake them into the program")
    return controls


# --------------------------------------------------------------------------- host side
def _options(namelist):
    from gpuwrf.io.noahmp_options import DEFAULTS
    opts = getattr(namelist, "noahmp_options", None)
    get = (lambda key: DEFAULTS.get(key)) if opts is None else (lambda key: getattr(opts, key, DEFAULTS.get(key)))
    return get


def _require_noahmp(namelist):
    if not bool(getattr(namelist, "use_noahmp", False)):
        raise ValueError("surface controls act on Noah-MP only (sf_surface_physics=4); this domain does not run it")
    for key in ("noahmp_static", "noahmp_energy_params", "noahmp_rad_params"):
        if getattr(namelist, key, None) is None:
            raise ValueError(f"surface controls need the bound {key}")


def _topo_reason(namelist) -> str | None:
    from gpuwrf.io.topo_wind_static import TOPO_WIND_PBL
    static = getattr(namelist, "topo_wind_static", None)
    if static is None:
        return "ctopo: no start_em CTOPO here (topo_wind=0, or a PBL other than YSU/Shin-Hong reads none)"
    if int(getattr(namelist, "bl_pbl_physics", 0)) not in TOPO_WIND_PBL:
        return f"ctopo: bl_pbl_physics={getattr(namelist, 'bl_pbl_physics', None)} does not read CTOPO"
    if static.ctopo_first is not None:
        return "ctopo: two-way feedback onto this domain recomputes CTOPO after the bind (not supported)"
    return None


def _inactive_reason(name: str, namelist) -> str | None:
    if name in LSM_CONTROLS:
        from gpuwrf.surface_params.lsmparams import inactive_reason
        return inactive_reason(name, namelist)
    if name in HOST_CONTROLS:
        return _topo_reason(namelist)
    opt = _options(namelist)
    if name in ("laim", "saim") and int(opt("dveg")) not in (1, 3, 4):
        return f"{name}: dveg={opt('dveg')} computes LAI/SAI from carbon pools/input, not the LAIM/SAIM table"
    if name == "shdmax" and int(opt("dveg")) != 4:
        return f"shdmax: FVEG=SHDMAX only for dveg=4 (dveg={opt('dveg')})"
    if name in ("vcmx25", "mp") and int(opt("opt_crs")) != 1:
        return f"{name}: consumed only by Ball-Berry STOMATA (opt_crs=1), not opt_crs={opt('opt_crs')}"
    return None


def grid_shape(namelist) -> tuple[int, int]:
    if getattr(namelist, "noahmp_energy_params", None) is None:  # a host-control-only (non-Noah-MP) domain
        return tuple(int(n) for n in np.shape(namelist.topo_wind_static.ctopo)[-2:])
    return tuple(int(n) for n in np.shape(namelist.noahmp_energy_params.z0mvt)[-2:])


def expected_shape(name: str, namelist) -> tuple[int, ...]:
    return _LEADING[ALL_CONTROLS[name].axis] + grid_shape(namelist)


def _target(name: str, namelist):
    """The array the released program reads for ``name`` (exact dtype/values)."""
    if name in LSM_CONTROLS:
        from gpuwrf.surface_params.lsmparams import target
        return target(name, namelist)
    if name in HOST_CONTROLS:
        return namelist.topo_wind_static.ctopo
    spec = CONTROLS[name]
    if spec.bundle == "energy" and spec.axis != "month":
        return getattr(namelist.noahmp_energy_params, name)
    if spec.bundle == "rad":
        return getattr(namelist.noahmp_rad_params, name)
    if spec.bundle == "static":
        return getattr(namelist.noahmp_static, name)
    return None  # month tables: gathered inside PHENOLOGY at trace time


def target_dtype(name: str, namelist) -> np.dtype:
    target = _target(name, namelist)
    if target is None:  # PHENOLOGY gathers in the Noah-MP REAL dtype, as the energy bundle
        target = namelist.noahmp_energy_params.z0mvt
    return np.dtype(target.dtype)


def table_base(name: str, namelist) -> np.ndarray:
    """Float64 field of the ORIGINAL table values per cell (before the consumer's rounding)."""
    if name in LSM_CONTROLS:
        from gpuwrf.surface_params.lsmparams import table_base as lsm_base
        return lsm_base(name, namelist)
    if name in HOST_CONTROLS:
        return np.asarray(namelist.topo_wind_static.ctopo, dtype=np.float64)
    if name == "z0bare":
        from gpuwrf.physics.noahmp.energy import Z0_BARE
        return np.full(grid_shape(namelist), Z0_BARE, dtype=np.float64)
    static = namelist.noahmp_static
    p = static.parameters
    shape = grid_shape(namelist)
    vt = np.asarray(static.ivgtyp, dtype=np.int64)

    def per_veg(table):
        arr = np.asarray(table, dtype=np.float64)
        return arr[np.clip(vt, 0, arr.shape[0] - 1)]

    if name in ("z0mvt", "vcmx25", "mp"):
        return per_veg(getattr(p, name))
    if name == "eg":
        return np.full(shape, float(np.asarray(p.eg, dtype=np.float64).reshape(-1)[0]))
    if name in ("albsat", "albdry"):
        bands = np.asarray(getattr(p, name), dtype=np.float64)[SOILCOLOR]
        return np.broadcast_to(bands[:, None, None], (2,) + shape).copy()
    if name in ("rhol", "rhos"):
        return np.moveaxis(per_veg(getattr(p, name)), -1, 0).copy()
    if name in ("laim", "saim"):
        return np.moveaxis(per_veg(getattr(p, name)), -1, 0).copy()
    if name == "shdmax":
        source = static.shdmax if static.shdmax is not None else static.shdfac
        if source is None:
            raise ValueError("shdmax: the bound Noah-MP static has neither SHDMAX nor SHDFAC")
        return np.asarray(source, dtype=np.float64)
    raise KeyError(name)


def prior_controls(namelist, names=None, *, device: bool = True, land_state=None) -> dict[str, Any]:
    """Byte copies of the values the released program reads (identity controls)."""
    return expand_controls(namelist, {name: 1.0 for name in (names or CONTROLS)}, device=device, land_state=land_state)


def _multiplier_field(name: str, namelist, value) -> np.ndarray:
    shape = expected_shape(name, namelist)
    if isinstance(value, Mapping):  # per vegetation category {VEGTYP: multiplier}
        vt = np.asarray(namelist.noahmp_static.isltyp if name in LSM_CONTROLS and LSM_CONTROLS[name].axis == "soil"
                        else namelist.noahmp_static.ivgtyp, dtype=np.int64)
        m2 = np.ones(vt.shape, dtype=np.float64)
        for category, factor in value.items():
            if name in LSM_CONTROLS and LSM_CONTROLS[name].axis == "soil":
                if str(category) != str(int(category)) or not 1 <= int(category) < len(namelist.noahmp_static.parameters.bexp):
                    raise ValueError(f"{name}: soil class must be a 1-based SOILPARM category (slot 0 is unused)")
            m2[vt == int(category)] = float(factor)
        return np.broadcast_to(m2, shape)
    arr = np.asarray(value, dtype=np.float64)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name}: multiplier shape {arr.shape} does not broadcast to {shape}") from exc


def expand_controls(namelist, multipliers: Mapping[str, Any] | None = None, *,
                    values: Mapping[str, Any] | None = None, device: bool = True, land_state=None) -> dict[str, Any]:
    """Control fields = round_once(table value x float64 multiplier), or explicit ``values``.

    ``multipliers[name]`` is a scalar, a ``{VEGTYP: factor}`` mapping, or a field broadcasting to the
    control shape. Every expansion first proves that multiplier 1 reproduces the bound bundle value
    exactly (the field really is the table value the consumer reads), then validates the result.
    Host controls (``ctopo``) take a scalar or field factor of the excess form (``excess_scale``).
    """
    refuse_unqualified_controls({*(multipliers or {}), *(values or {})})
    if land_state is not None and "smcmax" in {*(multipliers or {}), *(values or {})}:
        from gpuwrf.surface_params.lsmparams import validation_namelist
        namelist = validation_namelist(namelist, land_state)
    if any(name not in HOST_CONTROLS for name in [*(multipliers or {}), *(values or {})]):
        _require_noahmp(namelist)
    out: dict[str, np.ndarray] = {}
    for name, mult in (multipliers or {}).items():
        if name not in ALL_CONTROLS:
            raise ValueError(f"unknown surface control {name!r}; known: {sorted(ALL_CONTROLS)}")
        reason = _inactive_reason(name, namelist) if name in HOST_CONTROLS else None
        if reason:
            raise ValueError(f"surface control has no active consumer: {reason}")
        dtype = target_dtype(name, namelist)
        base = table_base(name, namelist)
        prior = base.astype(dtype)
        target = _target(name, namelist)
        if target is not None and not np.array_equal(np.asarray(target), prior):
            raise ValueError(f"{name}: bound bundle differs from the table gather (not a plain table value here)")
        if name in HOST_CONTROLS:
            if isinstance(mult, Mapping):
                raise ValueError(f"{name}: per-vegetation-class factors do not apply (scalar or field only)")
            out[name] = excess_scale(base, np.broadcast_to(np.asarray(mult, np.float64), base.shape)).astype(dtype)
            continue
        out[name] = (base * _multiplier_field(name, namelist, mult)).astype(dtype)
        if name in LSM_CONTROLS:
            from gpuwrf.surface_params.lsmparams import scaled_value
            out[name] = scaled_value(name, namelist, out[name])
    for name, value in (values or {}).items():
        if name in out:
            raise ValueError(f"{name}: given both as multiplier and value")
        if name not in ALL_CONTROLS:
            raise ValueError(f"unknown surface control {name!r}")
        out[name] = np.asarray(value).astype(target_dtype(name, namelist))
    validate_controls(out, namelist)
    if device:
        import jax.numpy as jnp
        return {name: jnp.asarray(out[name]) for name in sorted(out)}
    return {name: out[name] for name in sorted(out)}


def validate_controls(controls: Mapping[str, Any], namelist, *, land_state=None) -> None:
    """Fail closed on unknown/inactive controls, shape/dtype mismatch, non-finite or out-of-bound values."""
    refuse_unqualified_controls(controls)
    if land_state is not None and "smcmax" in controls:
        from gpuwrf.surface_params.lsmparams import validation_namelist
        namelist = validation_namelist(namelist, land_state)
    if not controls:
        raise ValueError("empty surface control set (omit the option instead)")
    if any(name not in HOST_CONTROLS for name in controls):
        _require_noahmp(namelist)
    for name in sorted(controls):
        if name not in ALL_CONTROLS:
            raise ValueError(f"unknown surface control {name!r}; known: {sorted(ALL_CONTROLS)}")
        reason = _inactive_reason(name, namelist)
        if reason:
            raise ValueError(f"surface control has no active consumer: {reason}")
        spec = ALL_CONTROLS[name]
        value = np.asarray(controls[name])
        if value.shape != expected_shape(name, namelist):
            raise ValueError(f"{name}: shape {value.shape} != {expected_shape(name, namelist)}")
        if value.dtype != target_dtype(name, namelist):
            raise ValueError(f"{name}: dtype {value.dtype} != consumer dtype {target_dtype(name, namelist)}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}: non-finite values")
        checked = value
        if name in LSM_CONTROLS:
            from gpuwrf.surface_params.lsmparams import range_values
            checked = range_values(name, namelist, value)
        if checked.size and (checked.min() < spec.lower or checked.max() > spec.upper):
            raise ValueError(f"{name}: values [{value.min()}, {value.max()}] outside [{spec.lower}, {spec.upper}] {spec.units}")
        if name in _TRANSMITTANCE:
            tau = np.asarray(getattr(namelist.noahmp_rad_params, _TRANSMITTANCE[name]), dtype=np.float64)
            if np.any(value.astype(np.float64) + tau > 1.0):
                raise ValueError(f"{name}: reflectance + {_TRANSMITTANCE[name]} > 1 (negative absorption)")
    if set(controls) & set(LSM_CONTROLS):
        from gpuwrf.surface_params.lsmparams import validate
        validate(controls, namelist)


def controls_digest(controls: Mapping[str, Any] | None) -> str | None:
    """Canonical content digest (names, shapes, dtypes, bytes); None for no controls."""
    if controls is None:
        return None
    h = hashlib.sha256(SCHEMA.encode())
    for name in sorted(controls):
        arr = np.ascontiguousarray(np.asarray(controls[name]))
        h.update(json.dumps([name, list(arr.shape), str(arr.dtype)]).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def restart_controls_digest(namelist) -> str | None:
    """Authenticate the actual surface operands, including host-bound CTOPO.

    CTOPO already rides a traced namelist child rather than ``surface_controls``.
    Hash that consumer array even when it came from start_em: a resume must also
    detect adding or removing a host override. Non-topographic runs retain their
    original digest (including None for a run without controls).
    """
    get = namelist.get if isinstance(namelist, Mapping) else (lambda key, default=None: getattr(namelist, key, default))
    controls = get("surface_controls", None)
    topo = get("topo_wind_static", None)
    if topo is not None:
        controls = {**(controls or {}), "ctopo": topo.ctopo}
    return controls_digest(controls)


def validate_controls_restart(recorded, namelist) -> None:
    """Refuse a resume whose requested controls differ from the checkpoint's (E78 class)."""
    if namelist is None:
        return
    if recorded != restart_controls_digest(namelist):
        raise ValueError("surface-control values differ between the checkpoint and the requested run (Track P, E78)")


# --------------------------------------------------------------------------- files
def save_controls(path, controls_by_domain: Mapping[str, Mapping[str, Any]], *, provenance=None) -> dict:
    """Write ``{domain: {name: field}}`` + canonical manifest to one ``.npz`` (atomic)."""
    refuse_unqualified_controls({name for controls in controls_by_domain.values() for name in controls})
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    manifest = {"schema": SCHEMA, "domains": {}, "provenance": provenance or {}}
    for domain in sorted(controls_by_domain):
        controls = controls_by_domain[domain]
        if not controls:
            continue
        host = {name: np.asarray(controls[name]) for name in sorted(controls)}
        for name, arr in host.items():
            if name not in ALL_CONTROLS:
                raise ValueError(f"unknown surface control {name!r}")
            arrays[f"{domain}/{name}"] = arr
        manifest["domains"][domain] = {"names": sorted(host), "digest": controls_digest(host),
                                       "shapes": {n: list(a.shape) for n, a in host.items()},
                                       "dtypes": {n: str(a.dtype) for n, a in host.items()}}
    if not manifest["domains"]:
        raise ValueError("no surface controls to save")
    arrays["manifest"] = np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)
    return manifest


def load_controls(path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Read and authenticate a control file (digest per domain)."""
    with np.load(Path(path), allow_pickle=False) as data:
        if "manifest" not in data.files:
            raise ValueError(f"{path}: not a surface-control file (no manifest)")
        manifest = json.loads(bytes(data["manifest"]).decode())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {manifest.get('schema')!r} != {SCHEMA!r}")
        refuse_unqualified_controls({name for entry in manifest["domains"].values() for name in entry["names"]})
        out: dict[str, dict[str, np.ndarray]] = {}
        for domain, entry in manifest["domains"].items():
            out[domain] = {name: np.array(data[f"{domain}/{name}"]) for name in entry["names"]}
            if controls_digest(out[domain]) != entry["digest"]:
                raise ValueError(f"{path}: {domain} control digest mismatch (file altered)")
        extra = set(data.files) - {"manifest"} - {f"{d}/{n}" for d, e in manifest["domains"].items() for n in e["names"]}
        if extra:
            raise ValueError(f"{path}: unlisted arrays {sorted(extra)}")
    return out, manifest


def has_host_controls(source) -> bool:
    """True if a control file (or ``{domain: {name: field}}``) carries any host control (``ctopo``)."""
    by_domain = load_controls(source)[0] if not isinstance(source, Mapping) else source
    refuse_unqualified_controls({name for controls in by_domain.values() for name in controls})
    return any(name in HOST_CONTROLS for controls in by_domain.values() for name in controls)


def bind_domain_controls(namelist, source, domain: str, *, allow_host: bool = True, land_state=None, cold_start=True,
                         grid_admission=None):
    """Namelist with the validated controls of ``domain`` (unchanged if the file has none for it).

    Traced controls go to ``namelist.surface_controls`` (clock base); host controls replace the value of the
    already traced namelist child (``topo_wind_static.ctopo``), so a host-only set keeps the default program.
    A grid-locked file (``provenance["grid_lock"]``) binds only with the ``grid_admission`` receipt that
    ``grid_lock.admit_control_source`` returned for the case dir actually loaded; otherwise it is refused."""
    by_domain, manifest = load_controls(source) if not isinstance(source, Mapping) else (source, None)
    lock = ((manifest or {}).get("provenance") or {}).get("grid_lock")
    if lock is not None and not (isinstance(grid_admission, Mapping)
                                 and grid_admission.get("status") == "GRID_LOCK_PASS"
                                 and grid_admission.get("digest") == (lock or {}).get("digest")):
        raise ValueError("surface controls: grid-locked file needs grid admission against the actual case "
                         "input dir before binding (grid_lock.admit_control_source); no silent OFF")
    if lock is not None:
        missing = sorted(set((lock or {}).get("domains", {})) - set(by_domain))
        if missing:  # a receipt for another file sharing this lock cannot admit a file lacking locked domains
            raise ValueError(f"surface controls: locked domains without controls in the file: {missing}; no silent OFF")
    refuse_unqualified_controls({name for controls in by_domain.values() for name in controls})
    import jax.numpy as jnp
    controls = by_domain.get(domain)
    if not controls:
        return namelist
    validate_controls(controls, namelist, land_state=land_state)
    if cold_start and set(controls) & set(LSM_CONTROLS):
        from gpuwrf.surface_params.lsmparams import validate_cold_start
        validate_cold_start(controls, namelist, land_state)
    host = {n: controls[n] for n in controls if n in HOST_CONTROLS}
    traced = {n: controls[n] for n in controls if n not in HOST_CONTROLS}
    if host:
        if not allow_host:
            raise ValueError(f"{domain}: host surface controls {sorted(host)} are not supported here (B>1 lanes)")
        static = namelist.topo_wind_static
        namelist = dataclasses.replace(namelist, topo_wind_static=static._replace(
            ctopo=jnp.asarray(host["ctopo"], dtype=jnp.asarray(static.ctopo).dtype)))
    if traced:
        namelist = dataclasses.replace(namelist, surface_controls={n: jnp.asarray(traced[n]) for n in sorted(traced)})
    return namelist


# W2 is explicitly named: legacy prior_controls() and Track P keep their original
# set. Values use the same clock operands, signature and restart authentication.
from gpuwrf.surface_params.lsmparams import SPECS as _LSM_SPECS
LSM_CONTROLS = {name: ControlSpec(*spec) for name, spec in _LSM_SPECS.items()}
ALL_CONTROLS.update(LSM_CONTROLS)
_LEADING["soil"] = (4,)

# RC-only v0.4.4 admission policy. The older Track-P and host controls remain
# available. No environment/CLI/API bypass: retained research tests explicitly
# scope their own policy fixture, like the runtime-parameter refusal tests.
UNAVAILABLE_SURFACE_CONTROLS = frozenset(LSM_CONTROLS)


def refuse_unqualified_controls(controls):
    """Reject the complete request before any allocation, binding or file write."""
    if controls is None:
        return
    if isinstance(controls, SurfaceControlsAux):
        controls = controls.value
    blocked = sorted(set(controls) & UNAVAILABLE_SURFACE_CONTROLS)
    if blocked:
        raise ValueError(f"surface controls {', '.join(blocked)} not available in v0.4.4: "
                         "failed the GPU qualification; see KNOWN_ISSUES")


def refuse_unqualified_sources(sources):
    """Preflight every batch member before binding any lane; inspect names only."""
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Mapping):
            names = {name for controls in source.values() for name in controls}
        else:
            with np.load(Path(source), allow_pickle=False) as data:
                if "manifest" not in data.files:
                    raise ValueError(f"{source}: not a surface-control file (no manifest)")
                manifest = json.loads(bytes(data["manifest"]).decode())
                if manifest.get("schema") != SCHEMA:
                    raise ValueError(f"{source}: schema {manifest.get('schema')!r} != {SCHEMA!r}")
                names = {name for entry in manifest["domains"].values() for name in entry["names"]}
        refuse_unqualified_controls(names)
