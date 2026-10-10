"""Bind ``namelist.input`` physics/dynamics options into the operational namelist.

v0.3.4 P0 fix (lane o1-nlbind).  Up to v0.3.3 both CLI pipelines (the native
single-root/live-nested ``nested_pipeline._load_domains`` and the CPU-history
replay ``daily_pipeline._build_real_case``) built the run's
:class:`OperationalNamelist` from ``from_grid`` defaults and never read
``mp_physics``, ``bl_pbl_physics``, ``sf_sfclay_physics``, ``ra_lw_physics`` or
``ra_sw_physics`` (nor several ``&dynamics`` knobs).  ``validate_operational_namelist``
accepted e.g. ``bl_pbl_physics=1``; the forecast then silently ran MYNN.

This module is the single place where those options are bound, plus the CLI's
"is every explicit namelist value honoured?" check:

* :func:`bind_namelist_options` copies every scheme key and the bound dynamics
  knobs of one domain onto an ``OperationalNamelist``.  An ABSENT key keeps the
  value the pipeline already uses, so a namelist that omits a key (or sets the
  value the pipeline hard-wires, as every release case does) produces the
  byte-identical program.
* :func:`namelist_honour_report` / :func:`require_namelist_honoured` list every
  ``&physics``/``&dynamics``/``&noah_mp`` value the selected CLI pipeline does
  NOT honour (explicit value != the WRF Registry default for a key the port does
  not bind) and raise :class:`NamelistNotHonouredError` -- no silent substitution.

Pure Python (no JAX import): the CLI runs the check before the heavy pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from gpuwrf.io.wrf_registry_defaults import WRF_REGISTRY_DEFAULTS

# --------------------------------------------------------------------------- #
# Keys bound onto the OperationalNamelist field of the same name.             #
# --------------------------------------------------------------------------- #
#: Physics scheme selectors (per domain, ``max_domains`` in the Registry).  The
#: values themselves are validated by ``validate_operational_namelist`` (CLI) and
#: fail closed in ``operational_mode._resolve_operational_suite``.
SCHEME_KEYS: tuple[str, ...] = (
    "mp_physics",
    "bl_pbl_physics",
    "sf_sfclay_physics",
    "ra_lw_physics",
    "ra_sw_physics",
    "cu_physics",
    "sf_urban_physics",
    "sf_lake_physics",
)

#: &dynamics knobs the operational step consumes.  The pipelines hard-wired the
#: Gen2/WN3 values (epssm 0.5, damp_opt 3, zdamp 5000, dampcoef 0.2,
#: w_damping 1, diff_6th_opt 2 / 0.12); now an explicit value wins and an omitted
#: knob runs the WRF Registry default.
DYNAMICS_KEYS: dict[str, type] = {
    "epssm": float,
    "damp_opt": int,
    "zdamp": float,
    "dampcoef": float,
    "w_damping": int,
    "diff_6th_opt": int,
    "diff_6th_factor": float,
    "khdif": float,
    "kvdif": float,
    "c_s": float,
    "c_k": float,
    "mix_isotropic": int,
    "mix_upper_bound": float,
    "tke_upper_bound": float,
}

#: &physics knobs bound like the &dynamics ones (omitted -> WRF Registry default):
#: cam_abs_freq_s = CAM LW absorptivity/emissivity refresh interval (ra_lw_physics=3,
#: module_radiation_driver.F; o1-camrad OperationalNamelist.cam_abs_freq_s).
PHYSICS_KNOBS: dict[str, type] = {
    "cam_abs_freq_s": float,
}

#: Namelist keys bound onto a differently named OperationalNamelist field.
RENAMED_KEYS: dict[str, tuple[str, type]] = {
    "shadlen": ("topo_shadow_length_m", float),
}

#: Keys bound elsewhere in the pipelines (before this module) or consumed by the
#: input/init readers.  ``validate_operational_namelist`` restricts their values.
_PIPELINE_BOUND_KEYS: frozenset[str] = frozenset({
    # nested_pipeline._load_domains / daily_pipeline._build_real_case
    "radt", "cudt", "use_mp_re", "gwd_opt", "topo_shading", "slope_rad",
    "diff_opt", "km_opt", "h_sca_adv_order", "moist_adv_opt", "scalar_adv_opt",
    "time_step_sound", "top_lid", "sf_surface_physics",
    # io.lower_boundary (auxinput4) and the init readers
    "sst_update", "use_theta_m",
    # catalog-approximated cadence (warned by collect_namelist_warnings)
    "bldt",
    # CLI apply_cldovrlp (McICA overlap flag)
    "cldovrlp",
    # nested_pipeline full-history guard (refuses positive values with Noah-MP)
    "bucket_mm", "bucket_j", "noahmp_acc_dt",
})

#: Keys whose effect reaches the forecast only through the prepared inputs
#: (real.exe / wrfinput dimensions and base state) or that only configure
#: arrays of a scheme the CLI cannot select; any value is consistent.
_INPUT_ONLY_KEYS: frozenset[str] = frozenset({
    "num_soil_layers", "num_land_cat", "num_soil_cat", "surface_input_source",
    "hybrid_opt", "etac", "base_temp", "base_pres", "base_lapse", "iso_temp",
    "use_baseparam_fr_nml", "maxpatch",
    # CAM radiation input-array sizes (fixed by the CAM data files the port reads) and
    # WRF-internal dims
    "levsiz", "paerlev", "cam_abs_dim1", "cam_abs_dim2",
    "alevsiz", "no_src_types",
    # ifsnow only enters the slab/Noah-classic snow path (CLI refuses those LSMs)
    "ifsnow",
    # Noah-MP history output switch (does not change the forecast)
    "noahmp_output",
})

#: Output-only diagnostics the history writer does not produce: the forecast is
#: unchanged, the requested fields are missing -> warning, not refusal.
_OUTPUT_ONLY_KEYS: dict[str, str] = {
    "do_radar_ref": "REFL_10CM radar reflectivity is not written",
    "prec_acc_dt": "PREC_ACC_C/PREC_ACC_NC bucket accumulations are not written",
}

#: Options that are exact only on ice-free domains (the port has no sea-ice
#: surface tile; with XICE=0 everywhere WRF's fractional path equals the plain
#: water/land path) -> warning.
_ICE_FREE_ONLY_KEYS: dict[str, str] = {
    "fractional_seaice": "the port has no sea-ice surface tile; exact only when the domain is "
                         "ice-free (XICE=0 everywhere)",
}

#: WRF setup_physics_suite (share/module_check_a_mundo.F): a suite fills every
#: scheme key that is -1 (the Registry default, i.e. omitted) per domain.
PHYSICS_SUITES: dict[str, dict[str, int]] = {
    "conus": {"cu_physics": 6, "mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
              "bl_pbl_physics": 2, "sf_sfclay_physics": 2, "sf_surface_physics": 2},
    "tropical": {"cu_physics": 16, "mp_physics": 6, "ra_lw_physics": 4, "ra_sw_physics": 4,
                 "bl_pbl_physics": 1, "sf_sfclay_physics": 91, "sf_surface_physics": 2},
}

#: WRF share/module_check_a_mundo.F per-domain rules (fatal there): these must be
#: equal on every domain of the run, and mp_physics is overwritten on every domain
#: by the innermost domain's value (frame/module_configure.F:200).  The port binds
#: per domain, so an unequal selection is refused instead of diverging from WRF.
EQUAL_ON_ALL_DOMAINS: tuple[str, ...] = (
    "mp_physics", "sf_surface_physics", "sf_sfclay_physics", "ra_lw_physics", "ra_sw_physics",
)
EQUAL_OR_ZERO_ON_ALL_DOMAINS: tuple[str, ...] = ("bl_pbl_physics", "cu_physics", "gwd_opt")

#: PBL schemes the port runs only on the revised-MM5 surface layer (sf_sfclay=1):
#: they re-derive their surface forcing through it.  Mirror of
#: coupling.physics_dispatch._PBL_REQUIRES_REVISED_MM5_SFCLAY (that module pulls in
#: the coupling package; tests/test_namelist_binding.py pins the two equal), so the
#: CLI refuses the pairing before load instead of after it.
PBL_REQUIRES_REVISED_MM5_SFCLAY: frozenset[int] = frozenset({1, 7, 8, 11, 12, 99})

#: WRF share/module_check_a_mundo.F:671-684 makes cu_physics=4 FATAL for ARW ("should not be
#: used for ARW; cu_physics = 95 is suggested"), so no CPU-WRF reference can exist.  The CLI
#: refuses it exactly like WRF; the faithful kernel stays reachable from the Python API.
CU4_ARW_REFUSAL = (
    "WRF ARW rejects cu_physics=4 (check_a_mundo); scale-aware SAS kernel is available via the "
    "Python API only, no CPU-WRF reference exists"
)

#: (diff_opt, km_opt) pairs the CLI runs (operational_mode explicit-diffusion
#: seam): diff_opt=0 (any km_opt but 2/5), 2-D Smagorinsky (1, 4) and 3-D
#: Smagorinsky (2, 3); the 3-D TKE/SMS closures (2, 2/5) are refused below.  Constant-K (km_opt=1) runs only through the
#: programmatic const_nu_m2_s path, which the CLI does not bind to khdif/kvdif;
#: every other pair would silently run without explicit diffusion (v0.3.4 probe:
#: diff_opt=2/km_opt=1/khdif=100 traced the diff_opt=0 program).
DIFFUSION_PAIRS: frozenset[tuple[int, int]] = frozenset({(1, 4), (2, 3)})

#: km_opt codes refused for v0.3.4 (manager decision on rel034 18:08Z): unqualified
#: v022 scaffolds that produce NaN under the release REAL carry.
KM_OPT_REFUSED: dict[int, str] = {
    2: "prognostic 3-D TKE",
    5: "SMS-3DTKE",
}

#: Port spellings of WRF Registry keys.  WRF's namelist name is ``rk_ord``
#: (Registry.EM_COMMON: ``rconfig integer rk_ord ... "rk_order"`` -- the quoted
#: word is the description); the port's catalog/docs historically say ``rk_order``.
#: Both are accepted with identical semantics (rv-nlbind F1).
KEY_ALIASES: dict[str, str] = {"rk_order": "rk_ord"}

#: Registry keys the operational step fixes to one value (explicit other values refused).
FIXED_KEYS: dict[str, tuple[Any, str]] = {
    "rk_ord": (3, "the port runs WRF's RK3 time integration only (rk_ord=3)"),
}

#: &domains keys the operational step consumes and the port fixes.
_FIXED_DOMAINS_KEYS: dict[str, Any] = {"hypsometric_opt": 2}

#: Land options the native/nested CLI pipeline can build (Noah-MP or none);
#: nested_pipeline._domain_sf_surface_physics refuses the rest at load time.
NATIVE_LAND_OPTIONS: tuple[int, ...] = (0, 4)

#: Replay-mode (CPU-history compatibility harness) substitutions, reported as
#: loud warnings: the replay driver integrates at its fixed 10 s step with 10
#: sound steps and replays the CPU-WRF land state hourly on the bulk surface.
#: key -> (value the harness runs, explanation)
_REPLAY_HARNESS_KEYS: dict[str, tuple[Any, str]] = {
    "time_step_sound": (10, "the compatibility driver runs its fixed 10 acoustic substeps"),
    "sf_surface_physics": (0, "the compatibility driver runs the prescribed bulk surface with "
                              "hourly CPU-WRF land replay (no prognostic land model)"),
    "sst_update": (0, "the compatibility driver does not read auxinput4 (wrflowinp) SST"),
}
_REPLAY_LABEL = ("the compatibility single-domain driver (CPU-history replay, or a single "
                 "non-d01 domain)")


class NamelistNotHonouredError(ValueError):
    """An explicit ``namelist.input`` value that the selected CLI pipeline does not run."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(
            "namelist.input selects option values this GPU run would NOT honour "
            "(v0.3.4 fail-closed; <=v0.3.3 silently ran the defaults):\n  - "
            + "\n  - ".join(problems)
            + "\nRemove the key (to run the WRF Registry default) or set the honoured value."
        )


def _group_of(key: str) -> str:
    entry = WRF_REGISTRY_DEFAULTS.get(key)
    return entry[0] if entry else "physics"


def _domain_index(key: str, domain: str) -> int:
    entry = WRF_REGISTRY_DEFAULTS.get(key)
    if entry is not None and entry[1] == "1":
        return 0  # single-entry WRF key: one value for every domain
    return max(int(str(domain)[1:]) - 1, 0)


def _raw_value(wrf_namelist: Mapping[str, Any], group: str, key: str, index: int) -> Any:
    """Per-domain value of ``key`` (None when absent or the slot is omitted)."""

    raw = (wrf_namelist.get(group) or {}).get(key)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return raw[index] if index < len(raw) else raw[-1]
    return raw


def namelist_value(wrf_namelist: Mapping[str, Any], key: str, domain: str) -> Any:
    """The explicit per-domain value of a Registry key, or None when absent.

    Scheme keys follow WRF ``setup_physics_suite``: under ``physics_suite`` =
    'CONUS'/'tropical' an omitted (or -1) scheme key takes the suite's scheme.
    """

    value = _raw_value(wrf_namelist, _group_of(key), key, _domain_index(key, domain))
    suite = _raw_value(wrf_namelist, "physics", "physics_suite", 0)
    if suite is not None and (value is None or value == -1):
        value = PHYSICS_SUITES.get(str(suite).strip().strip("'\"").lower(), {}).get(key, value)
    return value


def bound_options(wrf_namelist: Mapping[str, Any], domain: str) -> dict[str, Any]:
    """Explicit scheme + dynamics values of ``domain`` to bind (absent keys omitted)."""

    out: dict[str, Any] = {}
    for key in SCHEME_KEYS:
        value = namelist_value(wrf_namelist, key, domain)
        if value is not None:
            out[key] = int(value)
    for key, cast in DYNAMICS_KEYS.items():
        value = namelist_value(wrf_namelist, key, domain)
        # An omitted &dynamics knob runs the WRF Registry default, like WRF (the
        # release namelists set every one explicitly; only epssm 0.1, w_damping 0
        # and diff_6th_opt 0 differ from the values the pipelines used to hard-wire).
        out[key] = cast(WRF_REGISTRY_DEFAULTS[key][3] if value is None else value)
    for key, cast in PHYSICS_KNOBS.items():
        value = namelist_value(wrf_namelist, key, domain)
        out[key] = cast(WRF_REGISTRY_DEFAULTS[key][3] if value is None else value)
    for key, (field, cast) in RENAMED_KEYS.items():
        value = namelist_value(wrf_namelist, key, domain)
        if value is not None:
            out[field] = cast(value)
    return out


def bind_namelist_options(namelist, wrf_namelist: Mapping[str, Any], domain: str, *, skip=()):
    """Return ``namelist`` with every explicit scheme/dynamics option of ``domain`` bound.

    Absent keys keep the pipeline's value, so an unchanged release namelist yields an
    equal ``OperationalNamelist`` (same static aux -> same program).  ``skip`` names
    keys a pipeline binds itself.
    """

    updates = {k: v for k, v in bound_options(wrf_namelist, domain).items() if k not in skip}
    changed = {k: v for k, v in updates.items() if getattr(namelist, k) != v}
    return replace(namelist, **changed) if changed else namelist


def _same(value: Any, default: Any) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return bool(value) == bool(default)
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(default)
    return str(value).strip().strip("'\"").lower() == str(default).strip().lower()


def namelist_honour_report(
    wrf_namelist: Mapping[str, Any], domains: tuple[str, ...], *, replay: bool = False
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for the explicit options of ``domains``.

    An error is an explicit ``&physics``/``&dynamics``/``&noah_mp`` value of a key
    the CLI pipeline does not bind, different from the WRF Registry default the
    port runs.  ``replay=True`` selects the CPU-history replay driver, whose fixed
    harness numerics are reported as warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    honoured = (set(SCHEME_KEYS) | set(DYNAMICS_KEYS) | set(PHYSICS_KNOBS) | set(RENAMED_KEYS)
                | _PIPELINE_BOUND_KEYS | _INPUT_ONLY_KEYS)
    for group in ("physics", "dynamics", "noah_mp"):
        for key in sorted((wrf_namelist.get(group) or {})):
            spelled = str(key).lower()
            key_l = KEY_ALIASES.get(spelled, spelled)
            values = [_raw_value(wrf_namelist, group, spelled, _domain_index(key_l, d)) for d in domains]
            if key_l in FIXED_KEYS:
                runs, why = FIXED_KEYS[key_l]
                bad = sorted({v for v in values if v is not None and not _same(v, runs)}, key=str)
                if bad:
                    errors.append(f"&{group} {spelled}={bad[0]!r}: {why}")
                continue
            values = [namelist_value(wrf_namelist, key_l, d) for d in domains]
            if replay and key_l in _REPLAY_HARNESS_KEYS:
                runs, why = _REPLAY_HARNESS_KEYS[key_l]
                bad = sorted({v for v in values if v is not None and not _same(v, runs)}, key=str)
                if bad:
                    warnings.append(f"&{group} {key_l}={bad}: NOT honoured by "
                                    f"{_REPLAY_LABEL} -- {why}")
                continue
            if not replay and key_l == "sf_surface_physics":
                bad = sorted({int(v) for v in values if v is not None
                              and int(v) not in NATIVE_LAND_OPTIONS})
                if bad:
                    errors.append(f"&physics sf_surface_physics={bad}: the native/nested CLI "
                                  "pipeline builds only 0 (no LSM) or 4 (Noah-MP); slab (1), "
                                  "Noah-classic (2) and Pleim-Xiu (7) need explicit static/land "
                                  "bundles that only the programmatic API can supply")
                continue
            if key_l == "physics_suite":
                suite = str(values[0] if values else "none").strip().strip("'\"").lower()
                if suite != "none" and suite not in PHYSICS_SUITES:
                    errors.append(f"&physics physics_suite={values[0]!r}: unknown WRF suite "
                                  "(WRF knows 'CONUS' and 'tropical')")
                continue
            if key_l in _ICE_FREE_ONLY_KEYS:
                entry = WRF_REGISTRY_DEFAULTS.get(key_l)
                if any(v is not None and entry and not _same(v, entry[3]) for v in values):
                    warnings.append(f"&{group} {key_l}={values[0]!r}: {_ICE_FREE_ONLY_KEYS[key_l]}")
                continue
            if key_l in _OUTPUT_ONLY_KEYS:
                entry = WRF_REGISTRY_DEFAULTS.get(key_l)
                if any(v is not None and entry and not _same(v, entry[3]) for v in values):
                    warnings.append(f"&{group} {key_l}: output-only option not honoured -- "
                                    f"{_OUTPUT_ONLY_KEYS[key_l]} (forecast unchanged)")
                continue
            if key_l in honoured:
                continue
            entry = WRF_REGISTRY_DEFAULTS.get(key_l)
            if entry is None or entry[0] != group:
                errors.append(f"&{group} {key_l}: not a WRF v4 Registry key of &{group}; "
                              "the port does not read it")
                continue
            for domain in domains:
                value = namelist_value(wrf_namelist, key_l, domain)
                if value is None or _same(value, entry[3]):
                    continue
                where = "" if entry[1] == "1" else f" ({domain})"
                errors.append(f"&{group} {key_l}={value!r}{where}: not bound by the GPU port, "
                              f"which runs the WRF Registry default {entry[3]!r}")
                if entry[1] == "1":
                    break
    omitted = [k for k in ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "ra_lw_physics",
                           "ra_sw_physics") if namelist_value(wrf_namelist, k, domains[0]) is None]
    if omitted:
        warnings.append(f"&physics {', '.join(omitted)} omitted (WRF keeps -1 without a physics_suite); "
                        "the port runs its release suite (Thompson 8 / MYNN 5 / MYNN-SL 5 / RRTMG 4/4) "
                        "-- set them explicitly")
    if len(domains) > 1:
        for key in EQUAL_ON_ALL_DOMAINS + EQUAL_OR_ZERO_ON_ALL_DOMAINS:
            vals = [namelist_value(wrf_namelist, key, d) for d in domains]
            if key == "gwd_opt" and all(v is None for v in vals):
                vals = [_raw_value(wrf_namelist, "physics", key, _domain_index(key, d)) for d in domains]
            first = vals[0]
            bad = [v for v in vals[1:] if v is not None and first is not None and int(v) != int(first)
                   and not (key in EQUAL_OR_ZERO_ON_ALL_DOMAINS and int(v) == 0)]
            if bad:
                rule = ("WRF overwrites every domain with the innermost value" if key == "mp_physics"
                        else "WRF requires the same value on every domain"
                        + (" (or 0)" if key in EQUAL_OR_ZERO_ON_ALL_DOMAINS else ""))
                errors.append(f"{key}={vals}: {rule} (share/module_check_a_mundo.F); "
                              "set one value for all domains")
    cu4 = [d for d in domains if namelist_value(wrf_namelist, "cu_physics", d) is not None
           and int(namelist_value(wrf_namelist, "cu_physics", d)) == 4]
    if cu4:
        errors.append(f"cu_physics=4 ({', '.join(cu4)}): {CU4_ARW_REFUSAL} "
                      "(share/module_check_a_mundo.F:671)")
    for domain in domains:
        pbl = namelist_value(wrf_namelist, "bl_pbl_physics", domain)
        sfc = namelist_value(wrf_namelist, "sf_sfclay_physics", domain)
        pbl = 5 if pbl is None else int(pbl)  # omitted keys keep the pipeline's MYNN / MYNN-SL
        sfc = 5 if sfc is None else int(sfc)
        if pbl in PBL_REQUIRES_REVISED_MM5_SFCLAY and sfc != 1:
            errors.append(f"bl_pbl_physics={pbl} with sf_sfclay_physics={sfc} ({domain}): the port "
                          "runs this PBL only on the revised-MM5 surface layer (sf_sfclay_physics=1); "
                          "it would substitute revised-MM5 surface forcing")
            break
    for domain in domains:
        diff = namelist_value(wrf_namelist, "diff_opt", domain)
        km = namelist_value(wrf_namelist, "km_opt", domain)
        if diff is None or km is None:  # WRF: fatal "Both km_opt and diff_opt need to be set"
            warnings.append("&dynamics diff_opt/km_opt omitted (WRF refuses this); the port runs "
                            "0/0 = no explicit diffusion -- set both")
            break
        diff, km = int(diff), int(km)
        if km in KM_OPT_REFUSED:
            errors.append(f"km_opt={km} ({KM_OPT_REFUSED[km]}, {domain}): refused for v0.3.4 -- an "
                          "unqualified v022 scaffold that gives NaN under the release REAL carry; use "
                          "diff_opt=1/km_opt=4 (2-D Smagorinsky) or diff_opt=2/km_opt=3 (3-D Smagorinsky)")
            break
        if diff != 0 and (diff, km) not in DIFFUSION_PAIRS:
            errors.append(f"diff_opt={diff}, km_opt={km} ({domain}): the operational step has no such "
                          "diffusion path and would run WITHOUT explicit diffusion; supported: diff_opt=0, "
                          "1/4 (2-D Smagorinsky), 2/3 (3-D Smagorinsky); constant-K km_opt=1 is not CLI-bound")
            break
    if not replay and "sf_surface_physics" not in (wrf_namelist.get("physics") or {}):
        land = sorted({int(v) for v in (namelist_value(wrf_namelist, "sf_surface_physics", d)
                                         for d in domains) if v is not None} - set(NATIVE_LAND_OPTIONS))
        if land:
            errors.append(f"&physics physics_suite implies sf_surface_physics={land}: the native/"
                          "nested CLI pipeline builds only 0 (no LSM) or 4 (Noah-MP)")
    for key, fixed in _FIXED_DOMAINS_KEYS.items():
        raw = (wrf_namelist.get("domains") or {}).get(key)
        values = raw if isinstance(raw, (list, tuple)) else ([] if raw is None else [raw])
        if any(not _same(v, fixed) for v in values[:1]):
            errors.append(f"&domains {key}={values[0]!r}: the GPU port runs {key}={fixed} only")
    if replay:
        tc = (wrf_namelist.get("domains") or {}).get("time_step")
        if tc is not None and not _same(tc, 10):
            warnings.append(f"&domains time_step={tc}: NOT honoured by {_REPLAY_LABEL} -- "
                            "it integrates at its fixed 10 s step")
    return errors, warnings


def require_namelist_honoured(
    wrf_namelist: Mapping[str, Any], domains: tuple[str, ...], *, replay: bool = False
) -> list[str]:
    """Raise :class:`NamelistNotHonouredError` for unhonoured values; return warnings."""

    errors, warnings = namelist_honour_report(wrf_namelist, domains, replay=replay)
    if errors:
        raise NamelistNotHonouredError(errors)
    return warnings
