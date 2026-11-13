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
    # o2-moistadv: <= v0.3.4 both pipelines bound an OMITTED moist/scalar_adv_opt
    # as 0 (and 0 then skipped moisture transport); the Registry default is 1.
    "moist_adv_opt": int,
    "scalar_adv_opt": int,
    "tracer_opt": int,
    "tracer_adv_opt": int,
    "tracer_mix2_off": bool,
    "tracer_mix6_off": bool,
}

#: &physics knobs bound like the &dynamics ones (omitted -> WRF Registry default):
#: cam_abs_freq_s = CAM LW absorptivity/emissivity refresh interval (ra_lw_physics=3,
#: module_radiation_driver.F; o1-camrad OperationalNamelist.cam_abs_freq_s).
PHYSICS_KNOBS: dict[str, type] = {
    "tke_budget": int,
    "bl_mynn_output": int,
    "bl_mynn_edmf": int,
    "bl_mynn_edmf_mom": int,
    "mfshconv": int,
    'cu_rad_feedback': bool,
    'icloud': int,
    'icloud_bl': int,
    'bl_mynn_cloudpdf': int,
    'cu_diag': int,
    'kfeta_trigger': int,
    'shcu_aerosols_opt': int,
    'shallowcu_forced_ra': bool,
    "cam_abs_freq_s": float,
    "ccn_conc": float,
    "ccnty": int,
    "accum_mode": float, "aitken_mode": float, "coarse_mode": float,
    "shcu_physics": int,
    "shinhong_tke_diag": int,
    # YSU top-down cloud mixing (module_pbl_driver.F:1234; Registry default 1).
    "ysu_topdown_pblmix": int,
    "lakedepth_default": float,
    "lake_min_elev": float,
    "use_lakedepth": int,
    "prec_acc_dt": float,
    "do_radar_ref": int,
    "swint_opt": int,
    "windfarm_opt": int,
    "windfarm_ij": int,
    "windfarm_tke_factor": float,
    "tracer_pblmix": int,
    "sf_ocean_physics": int,
    "oml_hml0": float,
    "oml_gamma": float,
    "oml_relaxation_time": float,
    "isftcflx": int,
    "sas_pgcon": float,
    "nsas_dx_factor": int,
    "sst_skin": int,
    # WRF aer_opt (io/aerosol_options.py admits 0/1/3; 1 = Tegen climatology, 3 = Thompson mp28 aerosol, RRTMG SW).
    "aer_opt": int,
}

#: Namelist keys bound onto a differently named OperationalNamelist field.
RENAMED_KEYS: dict[str, tuple[str, type]] = {
    "rad_diag": ("rad_diag", int),
    'numbins': ('cup_num_bins',int),
    'thbinsize': ('cup_th_bin_size',float),
    'rbinsize': ('cup_r_bin_size',float),
    'mindeepfreq': ('cup_min_deep_freq',float),
    'minshallowfreq': ('cup_min_shallow_freq',float),
    "shadlen": ("topo_shadow_length_m", float),
}

#: Keys bound elsewhere in the pipelines (before this module) or consumed by the
#: input/init readers.  ``validate_operational_namelist`` restricts their values.
_PIPELINE_BOUND_KEYS: frozenset[str] = frozenset({
    "gpuwrf_tracer_file",
    # nested_pipeline._load_domains / daily_pipeline._build_real_case
    "radt", "cudt", "use_mp_re", "gwd_opt", "topo_shading", "slope_rad", "topo_wind",
    "diff_opt", "km_opt", "h_sca_adv_order",
    "time_step_sound", "top_lid", "sf_surface_physics",
    # io.lower_boundary (auxinput4) and the init readers
    "sst_update", "use_theta_m",
    # catalog-approximated cadence (warned by collect_namelist_warnings)
    "bldt",
    # CLI apply_cldovrlp (McICA overlap flag)
    "cldovrlp",
    # nested_pipeline full-history guard (refuses positive values with Noah-MP)
    "bucket_mm", "bucket_j", "noahmp_acc_dt", "opt_gla",
    "use_wudapt_lcz", "distributed_ahe_opt", "urban_ahe_file",
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

#: &dynamics keys of the digital filter initialization (gpuwrf.dfi, lane o2-dfi):
#: ``moist_adv_dfi_opt`` is bound by the DDFI/TDFI backward stage (values the port does not
#: reproduce are refused by gpuwrf.dfi.control.dfi_refusals); ``diff_opt_dfi``/``km_opt_dfi``
#: are overwritten by WRF's DDFI/TDFI backward init. DFL skips that init and consumes them
#: directly (Registry defaults 0/1); driver12 binds them and dfi_refusals checks their scope.
_DFI_KEYS: frozenset[str] = frozenset({"moist_adv_dfi_opt", "diff_opt_dfi", "km_opt_dfi"})

#: Requested diagnostics without a qualified online producer must be refused.
#: A warning would let an explicit request succeed with missing history fields.
_OUTPUT_ONLY_KEYS: dict[str, str] = {
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
PBL_REQUIRES_REVISED_MM5_SFCLAY: frozenset[int] = frozenset({1, 7, 8, 11, 12, 16, 17, 99})

#: WRF share/module_check_a_mundo.F:671-684 makes cu_physics=4 FATAL for ARW ("should not be
#: used for ARW; cu_physics = 95 is suggested"), so no CPU-WRF reference can exist.  The CLI
#: refuses it exactly like WRF; the faithful kernel stays reachable from the Python API.
CU4_ARW_REFUSAL = (
    "WRF ARW rejects cu_physics=4 (check_a_mundo); scale-aware SAS kernel is available via the "
    "Python API only, no CPU-WRF reference exists"
)

#: (diff_opt, km_opt) pairs the CLI runs (operational_mode explicit-diffusion
#: seam): diff_opt=0 (any km_opt but 2/5), 2-D Smagorinsky (1, 4) and 3-D
#: Smagorinsky (2, 3), LES TKE (2, 2), and SMS-3DTKE (2, 5).
#: Constant-K (km_opt=1) runs only through the
#: programmatic const_nu_m2_s path, which the CLI does not bind to khdif/kvdif;
#: every other pair would silently run without explicit diffusion (v0.3.4 probe:
#: diff_opt=2/km_opt=1/khdif=100 traced the diff_opt=0 program).
DIFFUSION_PAIRS: frozenset[tuple[int, int]] = frozenset({(1, 4), (2, 2), (2, 3), (2, 5)})

#: km_opt codes refused for v0.3.4 (manager decision on rel034 18:08Z): unqualified
#: v022 scaffolds that produce NaN under the release REAL carry.
KM_OPT_REFUSED: dict[int, str] = {}

# LES adapter currently carries six bulk moist species plus Thompson QNI/QNR.
# These schemes need additional number/aerosol/hail scalar diffusion channels.
KM2_UNWIRED_MP = frozenset({9, 10, 14, 16, 17, 18, 19, 22, 24, 26, 28, 40})

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

#: Land options the native/nested CLI pipeline can build (bulk, RUC, Noah-MP);
#: nested_pipeline._domain_sf_surface_physics refuses the rest at load time.
NATIVE_LAND_OPTIONS: tuple[int, ...] = (0, 3, 4, 8)

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
    if key == "rad_diag":
        return "gpuwrf"
    if key == "cycling":
        return "time_control"
    if key == "gpuwrf_tracer_file":
        return "dynamics"
    entry = WRF_REGISTRY_DEFAULTS.get(key)
    return entry[0] if entry else "physics"


def _domain_index(key: str, domain: str) -> int:
    if key == "cycling":
        return 0
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

    raw_pdf = (wrf_namelist.get("physics") or {}).get("bl_mynn_cloudpdf")
    if isinstance(raw_pdf, (list, tuple)) and len(raw_pdf) != 1:
        raise NamelistNotHonouredError(["bl_mynn_cloudpdf is one scalar shared by all domains"])
    pbl = namelist_value(wrf_namelist,'bl_pbl_physics',domain)
    lsm = namelist_value(wrf_namelist,'sf_surface_physics',domain)
    if pbl is not None and int(pbl)==4 and lsm is not None and int(lsm) in (1,2,7):
        raise NamelistNotHonouredError([f'QNSE land coupling with sf_surface_physics={lsm} ({domain}) is not qualified; select no LSM (0) or Noah-MP (4)'])
    ccnty = namelist_value(wrf_namelist, "ccnty", domain)
    if ccnty is not None and int(ccnty) not in (1, 2, 3, 4):
        raise NamelistNotHonouredError(["NTU ccnty must be 1, 2, 3 or 4"])
    out: dict[str, Any] = {}
    from gpuwrf.io.noahmp_options import selected_options
    out["noahmp_options"] = selected_options(wrf_namelist)
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
        if key == 'windfarm_opt':
            from gpuwrf.io.windfarm import windfarm_option
            value = windfarm_option(wrf_namelist,domain)
        else:
            value = namelist_value(wrf_namelist, key, domain)
        out[key] = cast(WRF_REGISTRY_DEFAULTS[key][3] if value is None else value)
    for key, (field, cast) in RENAMED_KEYS.items():
        value = namelist_value(wrf_namelist, key, domain)
        if value is not None:
            out[field] = cast(value)
    nwp=_raw_value(wrf_namelist,"time_control","nwp_diagnostics",0)
    cycling = _raw_value(wrf_namelist, "time_control", "cycling", 0)
    if cycling is not None:
        raw_cycling = (wrf_namelist.get("time_control") or {}).get("cycling")
        if isinstance(raw_cycling, (list, tuple)) and len(raw_cycling) != 1:
            raise NamelistNotHonouredError(["&time_control cycling is one global logical"])
        if not isinstance(cycling, (bool, int)) or cycling not in (0, 1):
            raise NamelistNotHonouredError(["&time_control cycling supports logical 0/1 only"])
        out["cycling"] = bool(cycling)
        if cycling:
            dfi = _raw_value(wrf_namelist, "dfi_control", "dfi_opt", 0)
            if dfi not in (None, 0):
                raise NamelistNotHonouredError(["cycling currently requires dfi_opt=0; cycling DFI rebalance timing is unqualified"])
            if out.get("mp_physics", 8) != 8 or out.get("bl_pbl_physics", 5) != 5:
                raise NamelistNotHonouredError(["cycling is qualified for Thompson mp=8 and MYNN bl=5 only"])
    history=_raw_value(wrf_namelist,"time_control","history_interval",_domain_index("history_interval",domain))
    out["nwp_diagnostics"]=0 if nwp is None else int(nwp)
    # WRF check_a_mundo forces the global radar selector for NWP diagnostics.
    if out["nwp_diagnostics"]==1:
        out["do_radar_ref"]=1
    out["diagnostic_history_interval"]=60. if history is None else float(history)
    from gpuwrf.io.afwa_options import afwa_refusals, severe_selected
    # Native entry: the same &afwa refusals as the CLI (critic FIX 5), before anything is bound.
    afwa_errors = afwa_refusals(wrf_namelist, (domain,))
    if afwa_errors:
        raise NamelistNotHonouredError(afwa_errors)
    if severe_selected(wrf_namelist, domain):
        # Only the AFWA severe WSPD10MAX is bound; afwa_refusals rejects the rest.
        out["afwa_severe_opt"] = 1
    # WRF check_a_mundo: this switch is reset outside QNSE, even if supplied.
    if out.get('bl_pbl_physics',5) != 4:
        out['mfshconv'] = 0
    return out


def bind_namelist_options(namelist, wrf_namelist: Mapping[str, Any], domain: str, *, skip=()):
    """Return ``namelist`` with every explicit scheme/dynamics option of ``domain`` bound.

    Absent keys keep the pipeline's value, so an unchanged release namelist yields an
    equal ``OperationalNamelist`` (same static aux -> same program).  ``skip`` names
    keys a pipeline binds itself.
    """

    from gpuwrf.io.flg_gfdl_config import validate_selection
    validate_selection(wrf_namelist)
    from gpuwrf.io.cammg_config import validate_selection as validate_cammg
    validate_cammg(wrf_namelist)
    updates = {k: v for k, v in bound_options(wrf_namelist, domain).items() if k not in skip}
    if int(updates.get("cu_physics", namelist.cu_physics)) == 95:
        updates["osas_domain_id"] = int(domain.removeprefix("d"))
    changed = {k: v for k, v in updates.items() if getattr(namelist, k) != v}
    if wrf_namelist.get("stoch"):
        from gpuwrf.io.stochastic_config import bind_stochastic
        stochastic = bind_stochastic(wrf_namelist,domain)
        if stochastic is not None:
            changed["stochastic_config"] = stochastic
    return replace(namelist, **changed) if changed else namelist


def _same(value: Any, default: Any) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return bool(value) == bool(default)
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(default)
    return str(value).strip().strip("'\"").lower() == str(default).strip().lower()


def icloud_refusals(icloud, *, mp_physics, cu_physics, ra_lw_physics, ra_sw_physics) -> list[str]:
    """Bound radiation cloud-fraction options (WRF 4.7.1 radiation_driver.F:1446-1506).

    icloud=1 is the default path; icloud=2 (cal_cldfra2) and icloud=3 (cal_cldfra3) replace the
    whole icloud=1 block incl. the icloud_bl merge; icloud=0 runs none of them (CLDFRA = 0 and the
    RRTMG adapters skip every condensate/radius input: cloud-free radiation, physics unchanged).
    ``None`` arguments mean "omitted" (the port's bound scheme defaults 8/0/4/4).
    """
    pick = lambda value, default: default if value is None else int(value)  # noqa: E731
    icloud = int(icloud)
    mp, cu = pick(mp_physics, 8), pick(cu_physics, 0)
    ra = (pick(ra_lw_physics, 4), pick(ra_sw_physics, 4))
    if icloud == 1 or cu == 11:  # check_a_mundo.F:1392-1396 forces icloud=1 for cu=11
        return []
    if icloud not in (0, 2, 3):
        return [f'icloud={icloud} is unported; bound options are 0, 1 (default), 2 and 3']
    errors = []
    if icloud == 3 and mp in (0, 1, 3):
        # check_a_mundo.F:1900-1915 (Kessler/WSM3); radiation_driver.F:1505 (no QC/QI field)
        errors.append(f'icloud=3 needs QC and QI fields (WRF refuses mp_physics={mp})')
    if mp in (5, 11, 50, 51, 52, 53, 55, 85, 95):
        errors.append(f'icloud={icloud} with mp_physics={mp} (scheme-specific radiation cloud view) is unqualified')
    if ra != (4, 4):
        errors.append(f'icloud={icloud} is qualified with RRTMG ra_lw/ra_sw 4/4 only')
    if cu in (7, 10):
        errors.append(f'icloud={icloud} with cu_physics={cu} cloud mediation is unqualified')
    return errors


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
    from gpuwrf.io.mynn_budget_options import validate_options as validate_mynn_options
    for index,domain in enumerate(domains):
        try:validate_mynn_options(wrf_namelist,index)
        except ValueError as error:errors.append(f'{domain}: {error}')
    for domain in domains:
        ccnty=namelist_value(wrf_namelist,'ccnty',domain)
        if ccnty is not None and int(ccnty) not in (1,2,3,4):
            errors.append(f'{domain}: NTU ccnty must be 1, 2, 3 or 4')
        cu=namelist_value(wrf_namelist,'cu_physics',domain)
        cloud=namelist_value(wrf_namelist,'icloud',domain)
        trigger=namelist_value(wrf_namelist,'kfeta_trigger',domain)
        aerosol=namelist_value(wrf_namelist,'shcu_aerosols_opt',domain)
        feedback=namelist_value(wrf_namelist,'cu_rad_feedback',domain)
        if cloud not in (None,1) and cu!=11:
            errors.extend(f'{domain}: {reason}' for reason in icloud_refusals(
                cloud, mp_physics=namelist_value(wrf_namelist,'mp_physics',domain),
                cu_physics=cu, ra_lw_physics=namelist_value(wrf_namelist,'ra_lw_physics',domain),
                ra_sw_physics=namelist_value(wrf_namelist,'ra_sw_physics',domain)))
        if trigger not in (None,1) and cu in (1,11):
            errors.append(f'{domain}: kfeta_trigger={trigger} horizontal trigger 2 is unported')
        if aerosol not in (None,0):
            errors.append(f'{domain}: shcu_aerosols_opt>0 needs unported aerosol/chemistry inputs')
        if feedback and cu!=11:
            errors.append(f'{domain}: cu_rad_feedback=True additional Grell cloud diagnostics are unqualified')
    from gpuwrf.io.noahmp_mmf import selected as mmf_selected, refusals as mmf_refusals
    mmf = mmf_selected(wrf_namelist)
    errors.extend(mmf_refusals(wrf_namelist, domains))
    if wrf_namelist.get("nssl_mp_params") and any(
        namelist_value(wrf_namelist, "mp_physics", d) in (17, 18, 19, 21, 22) for d in domains
    ):
        # Pristine nssl_2mom_init reads this separate group before building
        # its tables. The qualified port freezes the default table parameters.
        errors.append("&nssl_mp_params: NSSL mp_physics=17/18/19/21/22 supports the WRF-default "
                      "internal parameters only; custom internal namelist values are not bound")
    from gpuwrf.io.noahmp_options import option_refusals, SUPPORTED
    errors.extend(option_refusals(wrf_namelist, replay=replay))
    from gpuwrf.io.aerosol_options import aerosol_refusals
    errors.extend(aerosol_refusals(wrf_namelist, domains))
    solar = namelist_value(wrf_namelist, "swint_opt", domains[0]) if domains else None
    if solar not in (None, 0, 1):
        errors.append("&physics swint_opt: supports only 0/1")
    if solar == 1:
        from gpuwrf.io.namelist_check import _solar_admission_failures
        errors.extend(f"&physics swint_opt: {failure.implemented}"
                      for failure in _solar_admission_failures(wrf_namelist))
        if replay:
            errors.append("Solar interpolation requires native full-carry mode, not CPU-history replay")
    nwp = _raw_value(wrf_namelist, "time_control", "nwp_diagnostics", 0)
    if nwp is not None and int(nwp) not in (0,1):
        errors.append("&time_control nwp_diagnostics: supports only 0/1")
    for domain in domains:
        from gpuwrf.io.windfarm import windfarm_errors
        errors.extend(windfarm_errors(wrf_namelist,domain))
        radar=namelist_value(wrf_namelist,"do_radar_ref",domain)
        mp=namelist_value(wrf_namelist,"mp_physics",domain)
        if replay and mp==9:
            errors.append("Milbrandt mp_physics=9 requires native mode for mandatory radar; CPU-history replay is unsupported")
        if radar is not None and int(radar) not in (0,1):
            errors.append("&physics do_radar_ref: supports only 0/1")
        if (nwp and int(nwp)==1) or (radar and int(radar)==1):
            if replay:
                errors.append("online NWP/radar diagnostics require native mode, not CPU-history replay")
            if mp not in (None,8,9,10,17,18,19,21,22,40) or (mp in (9,10,17,18,19,21,22,40) and nwp and int(nwp)==1):
                errors.append("online NWP/radar diagnostics require mp_physics=8 (radar-only also admits "
                              "mp_physics=9/10/17/18/19/21/22/40)")
            if _raw_value(wrf_namelist,"domains","use_adaptive_time_step",_domain_index("use_adaptive_time_step",domain)):
                errors.append("online NWP/radar diagnostics require fixed time_step")
            if nwp and (wrf_namelist.get("bdy_control") or {}).get("periodic_x",False):
                errors.append("nwp_diagnostics helicity requires a specified/nested nonperiodic domain")
    from gpuwrf.io.afwa_options import afwa_refusals
    errors.extend(afwa_refusals(wrf_namelist, domains, replay=replay))
    if wrf_namelist.get("stoch"):
        from gpuwrf.io.stochastic_config import stochastic_refusals
        errors.extend(stochastic_refusals(wrf_namelist,domains))
    from gpuwrf.io.nesting_options import resolve_nesting_options
    try:
        nesting = resolve_nesting_options(wrf_namelist)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if wrf_namelist.get("stoch"):
            from gpuwrf.io.stochastic_config import stochastic_nesting_refusals
            errors.extend(stochastic_nesting_refusals(wrf_namelist, domains, feedback=nesting["feedback"], batch_size=1))
    for domain in domains:
        shcu=namelist_value(wrf_namelist,"shcu_physics",domain)
        if shcu not in (None,0):
            errors.append(f"{domain}: shcu_physics={shcu} has no operational shallow-convection producer; CAM-MG itself does not require it")
    honoured = (set(SCHEME_KEYS) | set(DYNAMICS_KEYS) | set(PHYSICS_KNOBS) | set(RENAMED_KEYS)
                | _PIPELINE_BOUND_KEYS | _INPUT_ONLY_KEYS | _DFI_KEYS | set(SUPPORTED))
    if _raw_value(wrf_namelist, "time_control", "cycling", 0):
        for domain in domains:
            try:
                bound_options(wrf_namelist, domain)
            except NamelistNotHonouredError as exc:
                errors.extend(exc.problems)
        if replay:
            errors.append("cycling requires native wrfinput initialization, not history replay")
        honoured.add("use_input_w")
    import math
    for domain in domains:
        interval = namelist_value(wrf_namelist, "prec_acc_dt", domain)
        if interval is not None:
            interval = float(interval)
            if not math.isfinite(interval) or interval < 0.0:
                errors.append(f"&physics prec_acc_dt={interval}: must be finite and nonnegative")
            elif interval > 0.0:
                if replay:
                    errors.append("prec_acc_dt is not supported by the CPU-history replay clock; use native mode")
                mp = namelist_value(wrf_namelist, "mp_physics", domain)
                cu = namelist_value(wrf_namelist, "cu_physics", domain)
                if mp not in (None, 8) or cu not in (None, 0, 1):
                    errors.append("prec_acc_dt requires mp_physics=8 and cu_physics=0/1 (raw step amounts)")
    for domain in domains:
        tracer_opt = namelist_value(wrf_namelist, "tracer_opt", domain)
        tracer_file = namelist_value(wrf_namelist, "gpuwrf_tracer_file", domain)
        if tracer_opt not in (None, 0, 2):
            errors.append("tracer_opt: non-Chem WRF supports only 0/2; 1/3/4 require WRF-Chem")
        for key, choices in (("tracer_adv_opt", (0, 1, 2)), ("tracer_pblmix", (0, 1))):
            value = namelist_value(wrf_namelist, key, domain)
            if value is not None and value not in choices:
                errors.append(f"{key}: supports only {choices}")
        if tracer_opt == 2 or tracer_file:
            bldt = namelist_value(wrf_namelist, "bldt", domain)
            if bldt not in (None, 0, 0.0):
                errors.append("passive tracers currently require bldt=0 (PBL every step)")
    from gpuwrf.io.noahmp_glacier_options import glacier_refusals
    errors.extend(glacier_refusals(wrf_namelist))
    for domain in domains:
        value = namelist_value(wrf_namelist, "rad_diag", domain)
        if value is not None:
            if value not in (0, 1):
                errors.append(f"&gpuwrf rad_diag ({domain}) supports only 0/1")
            elif value and any(namelist_value(wrf_namelist, key, domain) not in (None, 4)
                               for key in ("ra_sw_physics", "ra_lw_physics")):
                errors.append(f"&gpuwrf rad_diag=1 ({domain}) requires RRTMG4/4")
            elif value and replay:
                errors.append("&gpuwrf rad_diag=1 requires the native held-radiation pipeline")
    for group in ("physics", "dynamics", "noah_mp", "gpuwrf"):
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
            if mmf and ((group == "noah_mp" and key_l in ("opt_run", "opt_inf"))
                        or (group == "physics" and key_l == "wtddt")):
                continue  # Both loaders initialize/seed the complete MMF group.
            if replay and key_l in _REPLAY_HARNESS_KEYS:
                if key_l == "sf_surface_physics" and all(v in (3,8) for v in values):
                    continue  # RUC uses its full-carry stepper on the daily path.
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
                                  "pipeline builds only 0 (no LSM), 3 (RUC), 4 (Noah-MP) or 8 (SSiB); slab (1), "
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
                    errors.append(f"&{group} {key_l}: requested diagnostic not honoured -- "
                                  f"{_OUTPUT_ONLY_KEYS[key_l]}")
                continue
            if key_l in honoured:
                continue
            # WRF check_a_mundo.F:3395-3440 overrides legacy flags except CCN
            # for19. Original19 keeps explicit CCN0; this qualified variant
            # carries default CCN1, so refuse that unbound configuration.
            if key_l in {"nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on", "nssl_density_on"} and all(
                namelist_value(wrf_namelist, "mp_physics", d) in (17, 19, 21, 22) for d in domains
            ):
                if key_l == "nssl_ccn_on" and any(
                    namelist_value(wrf_namelist, "mp_physics", d) == 19
                    and namelist_value(wrf_namelist, key_l, d) not in (-1, 1)
                    for d in domains
                ):
                    errors.append("&physics nssl_ccn_on: mp_physics=19 currently supports "
                                  "the WRF default CCN1 only; explicit CCN0 is not bound")
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
    radiation_on = any(namelist_value(wrf_namelist, k, d) not in (0, "0")
                       for k in ("ra_lw_physics", "ra_sw_physics") for d in domains)
    no_radt = [d for d in domains if namelist_value(wrf_namelist, "radt", d) is None]
    if radiation_on and no_radt:
        # o2-moistadv omitted-key audit: <= v0.3.4 an omitted radt silently ran a 30 min cadence;
        # WRF's Registry default radt=0 means STEPRA = max(nint(0),1) = radiation EVERY step
        # (phys/module_physics_init.F:2150), which the port does not run (explicit radt=0 is refused)
        # -> loud warning, like omitted scheme keys (refusing would reject minimal namelists).
        warnings.append(f"&physics radt omitted ({', '.join(no_radt)}): WRF's default radt=0 calls radiation "
                        "EVERY step; the port does not run that and uses its driver cadence instead "
                        "(native/nested 30 min) -- set radt explicitly (minutes, e.g. the dx-in-km rule)")
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
    for domain in domains:
        urb = namelist_value(wrf_namelist, "sf_urban_physics", domain)
        if urb is not None and int(urb) in (2,3):
            lsm = namelist_value(wrf_namelist, "sf_surface_physics", domain)
            pbl = namelist_value(wrf_namelist, "bl_pbl_physics", domain)
            lcz = namelist_value(wrf_namelist, "use_wudapt_lcz", domain)
            if lcz is not None and int(lcz)!=0:
                errors.append(f'BEP/BEM ({domain}): use_wudapt_lcz (LCZ) requires grouped per-type geometry/carries/PBL sources; use default UTYPE=2')
            if replay:
                errors.append(f'sf_urban_physics={urb} ({domain}): {_REPLAY_LABEL} cannot evolve BEP/BEM urban carry; use native/nested')
            if lsm is None or int(lsm)!=4:
                errors.append(f'sf_urban_physics={urb} ({domain}) requires sf_surface_physics=4 Noah-MP in this port')
            if pbl is not None and int(pbl)==2:
                errors.append(f'BEP/BEM+MYJ ({domain}) requires the absent module_bl_myjurb (MYJURB)')
            elif pbl is None or int(pbl) not in (1,8):
                errors.append(f'BEP/BEM ({domain}): WRF allows YSU1/MYJ2/BouLac8; the port admits YSU1 or BouLac8')
            for key in ('ra_sw_physics','ra_lw_physics'):
                value=namelist_value(wrf_namelist,key,domain)
                if value is None or int(value)!=4:
                    errors.append(f'BEP/BEM ({domain}) requires RRTMG4/4 held direct/diffuse radiation in this port')
        for key, allowed in (("use_wudapt_lcz",(0,1)),("distributed_ahe_opt",(0,1,2))):
            value = namelist_value(wrf_namelist,key,domains[0])
            if value is not None and (value not in allowed or (value and (replay or urb != 1))):
                errors.append(f"{key}={value} ({domain}): requires native sf_urban_physics=1; allowed {allowed}")
        ahe_file = _raw_value(wrf_namelist,"physics","urban_ahe_file",_domain_index("sf_urban_physics",domain))
        if ahe_file and (replay or urb != 1 or namelist_value(wrf_namelist,"distributed_ahe_opt",domains[0]) not in (1,2)):
            errors.append(f"urban_ahe_file ({domain}): requires native urban and distributed_ahe_opt=1/2")
        if namelist_value(wrf_namelist,"distributed_ahe_opt",domains[0]) == 1:
            pbl = namelist_value(wrf_namelist,"bl_pbl_physics",domain)
            if pbl not in (None,5):
                errors.append(f"distributed_ahe_opt=1 ({domain}): requires WRF source-tendency MYNN PBL (bl_pbl_physics=5)")
        if urb is None or int(urb) != 1:
            continue
        lsm = namelist_value(wrf_namelist, "sf_surface_physics", domain)
        if replay:
            errors.append(f"sf_urban_physics=1 ({domain}): {_REPLAY_LABEL} replays CPU land fields and "
                          "does not run the single-layer urban canopy model; use the native/nested pipeline")
        elif lsm is None or int(lsm) != 4:
            errors.append(f"sf_urban_physics=1 with sf_surface_physics={lsm} ({domain}): the port couples "
                          "the single-layer urban canopy model only to Noah-MP (sf_surface_physics=4)")
        if replay:
            break
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
        if (pbl == 10) != (sfc == 10):
            errors.append(f'TEMF requires bl_pbl_physics=10 and sf_sfclay_physics=10 together ({domain}; WRF check_a_mundo)')
        if pbl == 10:
            lsm = namelist_value(wrf_namelist,'sf_surface_physics',domain)
            if lsm is not None and int(lsm) not in (0,4):
                errors.append(f'TEMF requires qualified no-LSM/Noah-MP coupling (sf_surface_physics=0/4; {domain})')
        if pbl == 4:
            lsm = namelist_value(wrf_namelist,'sf_surface_physics',domain)
            if lsm is not None and int(lsm) in (1,2,7):
                errors.append(f'QNSE land coupling with sf_surface_physics={lsm} ({domain}) is not qualified; select no LSM (0) or Noah-MP (4)')
        if pbl in (16,17) and namelist_value(wrf_namelist,'bldt',domain) not in (None,0,0.0):
            errors.append(f'bl_pbl_physics={pbl} ({domain}) currently requires bldt=0; held off-cadence epsilon tendencies are not qualified')
        if pbl in (16,17):
            lsm=namelist_value(wrf_namelist,'sf_surface_physics',domain)
            if lsm is not None and int(lsm) not in (0,4):
                errors.append(f'bl_pbl_physics={pbl} requires qualified no-LSM/Noah-MP coupling (sf_surface_physics=0/4; {domain})')
        if pbl == 3 and sfc != 3:
            errors.append(f"bl_pbl_physics=3 with sf_sfclay_physics={sfc} ({domain}): WRF BL_GFS requires the "
                          "GFS surface layer (sf_sfclay_physics=3; module_physics_init.F:3743 is fatal otherwise)")
        if pbl in PBL_REQUIRES_REVISED_MM5_SFCLAY and sfc != 1:
            errors.append(f"bl_pbl_physics={pbl} with sf_sfclay_physics={sfc} ({domain}): the port "
                          "runs this PBL only on the revised-MM5 surface layer (sf_sfclay_physics=1); "
                          "it would substitute revised-MM5 surface forcing")
            break
    for domain in domains:
        from gpuwrf.io.ocean_options import validate_options as validate_ocean_options
        cfg={key:namelist_value(wrf_namelist,key,domain) for key in ('sf_ocean_physics','oml_hml0','oml_gamma','oml_relaxation_time','isftcflx')}
        try:
            validate_ocean_options(cfg,sfclay=namelist_value(wrf_namelist,'sf_sfclay_physics',domain),
                sst_update=bool(namelist_value(wrf_namelist,'sst_update',domain)),
                noahmp=namelist_value(wrf_namelist,'sf_surface_physics',domain) in (None,4),
                lake=namelist_value(wrf_namelist,'sf_lake_physics',domain) or 0,
                lsm=namelist_value(wrf_namelist,'sf_surface_physics',domain),
                urban=namelist_value(wrf_namelist,'sf_urban_physics',domain) or 0,
                pbl=namelist_value(wrf_namelist,'bl_pbl_physics',domain))
        except ValueError as exc:
            errors.append(str(exc))
        if namelist_value(wrf_namelist,'sf_lake_physics',domain) == 1:
            for key,default in (('sf_surface_physics',4),('mp_physics',8),('bl_pbl_physics',5),('sf_sfclay_physics',5)):
                value=namelist_value(wrf_namelist,key,domain)
                if value is not None and int(value) != default:
                    errors.append(f'sf_lake_physics=1 ({domain}) currently requires {key}={default}; got {value}')
            depthflag=namelist_value(wrf_namelist,'use_lakedepth',domain)
            if depthflag is not None and int(depthflag) not in (0,1):
                errors.append(f'use_lakedepth ({domain}) must be0 or1')
            if namelist_value(wrf_namelist,'sst_update',domain) not in (None,0):
                errors.append(f'sf_lake_physics=1 ({domain}) with sst_update is not wired; set sst_update=0')
            if int(wrf_namelist.get('dfi_control',{}).get('dfi_opt',0)):
                errors.append(f'sf_lake_physics=1 ({domain}) with DFI is not qualified; set dfi_opt=0')
    for domain in domains:
        if namelist_value(wrf_namelist,"sf_surface_physics",domain)!=8: continue
        for key,want,default in (("mp_physics",8,8),("sf_sfclay_physics",1,5),("bl_pbl_physics",1,5),
            ("ra_sw_physics",1,4),("ra_lw_physics",4,4),("num_soil_layers",3,4),
            ("sf_lake_physics",0,0),("sf_urban_physics",0,0),("fractional_seaice",1,0),("usemonalb",0,0)):
            raw=namelist_value(wrf_namelist,key,domain); actual=default if raw is None else int(raw)
            if actual!=want: errors.append(f"SSiB ({domain}) requires {key}={want}; got {actual}; spectral SW=3/4 and sea-ice tiles are not coupled")
    for domain in domains:
        if namelist_value(wrf_namelist, "sf_surface_physics", domain) != 3:
            continue
        for key, allowed, default in (("mp_physics", (8,), 8), ("sf_sfclay_physics", (5,), 5),
                                       ("sf_lake_physics", (0,), 0), ("num_soil_layers", (6, 9), 4)):
            value = namelist_value(wrf_namelist, key, domain)
            value = default if value is None else int(value)
            if value not in allowed:
                errors.append(f"sf_surface_physics=3 ({domain}): RUC requires {key} in {allowed}; got {value}")
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
        if km in (2, 5) and diff != 2:
            errors.append(f"km_opt={km} ({domain}) requires diff_opt=2 for the prognostic LES TKE path")
            break
        if km == 5:
            pbl = int(namelist_value(wrf_namelist, "bl_pbl_physics", domain) or 0)
            sf = int(namelist_value(wrf_namelist, "sf_sfclay_physics", domain) or 0)
            land = int(namelist_value(wrf_namelist, "sf_surface_physics", domain) or 0)
            cu = int(namelist_value(wrf_namelist, "cu_physics", domain) or 0)
            if pbl != 0:
                errors.append(f"km_opt=5 ({domain}) requires bl_pbl_physics=0 (WRF SMS-3DTKE pairing)")
            if sf not in (0, 1, 5, 91):
                errors.append(f"km_opt=5 ({domain}) requires sf_sfclay_physics in 0/1/5/91 (WRF)")
            elif sf in (1, 91):
                errors.append(f"km_opt=5 with sf_sfclay_physics={sf} ({domain}): SMS BR/RMOL/WSPD/USTM "
                              "caller is qualified for 0 or 5 only; MM5 surface caller qualification pending")
            if cu == 11:
                errors.append(f"km_opt=5 ({domain}) cannot work with cu_physics=11 (WRF)")
            if land not in (0,4) or (sf==0 and land!=0):
                errors.append(f"km_opt=5 ({domain}): SMS land caller is qualified for sf_surface_physics=0/4; "
                              "surface layer 0 requires land 0 (other land callers not qualified)")
        if km in (2, 5):
            mp = namelist_value(wrf_namelist, "mp_physics", domain)
            if mp in KM2_UNWIRED_MP:
                errors.append(f"km_opt={km} with mp_physics={mp} ({domain}): extra scalar diffusion "
                              "is not wired; the LES adapter covers bulk moisture and Thompson QNI/QNR")
                break
        if diff != 0 and (diff, km) not in DIFFUSION_PAIRS:
            errors.append(f"diff_opt={diff}, km_opt={km} ({domain}): the operational step has no such "
                          "diffusion path and would run WITHOUT explicit diffusion; supported: diff_opt=0, "
                          "1/4 (2-D Smagorinsky), 2/2 (3-D TKE), 2/3 (3-D Smagorinsky), 2/5 (SMS-3DTKE); constant-K km_opt=1 is not CLI-bound")
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
    # grid_fdda=1 (lane o2-fdda): WRF-fatal / unported &fdda selections, bound in both pipelines.
    from gpuwrf.io.fdda_namelist import fdda_namelist_errors, fdda_namelist_warnings
    errors.extend(fdda_namelist_errors(wrf_namelist, domains))
    warnings.extend(fdda_namelist_warnings(wrf_namelist, domains))
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
