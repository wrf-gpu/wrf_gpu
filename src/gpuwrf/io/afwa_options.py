"""&afwa namelist admission (no JAX import: refusals run before backend initialization).

Only the AFWA severe-weather WSPD10MAX (module_diag_afwa.F:338-393) is ported.
WRF 4.7.1 Registry/registry.afwa keys; every other product is refused by name.
"""
from collections.abc import Mapping
from typing import Any

PER_DOMAIN = ("afwa_diag_opt", "afwa_ptype_opt", "afwa_vil_opt", "afwa_radar_opt", "afwa_severe_opt",
              "afwa_icing_opt", "afwa_vis_opt", "afwa_cloud_opt", "afwa_therm_opt", "afwa_turb_opt",
              "afwa_buoy_opt")
UNSUPPORTED = tuple(key for key in PER_DOMAIN if key not in ("afwa_diag_opt", "afwa_severe_opt"))
# Only consulted by the (refused) precipitation-type product.
INERT = ("afwa_ptype_ccn_tmp", "afwa_ptype_tot_melt")
# WRF history-alarm controls besides history_interval (minutes). They move the alarm that resets
# WSPD10MAX (is_after_history_dump); the port's writer reads history_interval only (critic FIX 3).
ALARM_CONTROLS = ("history_interval_d", "history_interval_h", "history_interval_m", "history_interval_s",
                  "history_begin", "history_begin_y", "history_begin_mo", "history_begin_d",
                  "history_begin_h", "history_begin_m", "history_begin_s",
                  "history_end", "history_end_y", "history_end_mo", "history_end_d",
                  "history_end_h", "history_end_m", "history_end_s")


def _value(group: Mapping[str, Any], key: str, index: int) -> Any:
    """Per-domain &afwa value; an omitted slot is WRF's Registry default 0 (None), never the last
    supplied value (critic FIX 4: gfortran namelist read leaves omitted nests OFF)."""
    raw = group.get(key)
    if isinstance(raw, (list, tuple)):
        return raw[index] if index < len(raw) else None
    return raw if index == 0 else None


def _repeat(group: Mapping[str, Any], key: str, index: int) -> Any:
    """Non-&afwa keys as the port binds them (namelist_binding._raw_value: last value repeats)."""
    raw = group.get(key)
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return raw[index] if index < len(raw) else raw[-1]
    return raw


def _int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    return int(number) if number == int(number) else -1


def _index(domain: str) -> int:
    return max(int(str(domain)[1:]) - 1, 0)


def severe_selected(wrf_namelist: Mapping[str, Any], domain: str) -> bool:
    """WRF calls the AFWA driver iff afwa_diag_opt==1; WSPD10MAX iff afwa_severe_opt==1."""
    group = {str(k).lower(): v for k, v in (wrf_namelist.get("afwa") or {}).items()}
    i = _index(domain)
    return _int(_value(group, "afwa_diag_opt", i)) == 1 and _int(_value(group, "afwa_severe_opt", i)) == 1


def afwa_refusals(wrf_namelist: Mapping[str, Any], domains, *, replay: bool = False) -> list[str]:
    group = {str(k).lower(): v for k, v in (wrf_namelist.get("afwa") or {}).items()}
    if not group:
        return []
    errors: list[str] = []
    for key in sorted(group):
        if key not in PER_DOMAIN and key not in INERT and key != "afwa_bad_data_check":
            errors.append(f"&afwa {key}: not a WRF 4.7.1 &afwa key the port reads")
    bad = _int(_value(group, "afwa_bad_data_check", 0))
    if bad not in (None, 0):
        errors.append("&afwa afwa_bad_data_check: only 0 is supported (the port has no fatal-stop data check)")
    physics = wrf_namelist.get("physics") or {}
    time_control = wrf_namelist.get("time_control") or {}
    domains_group = wrf_namelist.get("domains") or {}
    for domain in domains:
        i = _index(domain)
        values = {key: _int(_value(group, key, i)) for key in PER_DOMAIN}
        for key in UNSUPPORTED:
            if values[key] not in (None, 0):
                errors.append(f"&afwa {key} ({domain}): AFWA product not implemented; only afwa_severe_opt "
                              "WSPD10MAX is ported (WRF has no AFWA_GUST field)")
        diag, severe = values["afwa_diag_opt"], values["afwa_severe_opt"]
        if diag not in (None, 0, 1) or severe not in (None, 0, 1):
            errors.append(f"&afwa afwa_diag_opt/afwa_severe_opt ({domain}): support only 0/1")
            continue
        if severe == 1 and diag != 1:
            errors.append(f"&afwa afwa_severe_opt=1 ({domain}) requires afwa_diag_opt=1: WRF only calls the "
                          "AFWA driver under afwa_diag_opt=1 (module_diagnostics_driver.F:949)")
        if diag == 1 and severe != 1:
            errors.append(f"&afwa afwa_diag_opt=1 ({domain}) without afwa_severe_opt=1 selects AFWA_MSLP/PWAT/"
                          "TOTPRECIP, which are not implemented; the port provides severe WSPD10MAX only")
        if diag != 1 or severe != 1:
            continue
        if replay:
            errors.append("&afwa WSPD10MAX requires native mode, not CPU-history replay")
        mp, cu = _int(_repeat(physics, "mp_physics", i)), _int(_repeat(physics, "cu_physics", i))
        if mp != 8 or cu not in (None, 0, 1):
            errors.append(f"&afwa WSPD10MAX ({domain}) requires mp_physics=8 and cu_physics=0/1 "
                          "(raw RAINNCV/RAINCV step amounts)")
        if _int(_repeat(time_control, "nwp_diagnostics", 0)) == 1:
            errors.append("&afwa afwa_severe_opt=1 and nwp_diagnostics=1 both own WSPD10MAX with different "
                          "reset rules; select one")
        if _repeat(domains_group, "use_adaptive_time_step", i):
            errors.append("&afwa WSPD10MAX requires a fixed time_step")
        # In-step U10/V10 producer: MYNN surface layer + Noah-MP every step (critic FIX 1). bldt>0 holds
        # WRF's surface driver (and U10/V10) off cadence; the port runs it every step.
        sfclay, lsm = _int(_repeat(physics, "sf_sfclay_physics", i)), _int(_repeat(physics, "sf_surface_physics", i))
        if sfclay not in (None, 5) or lsm not in (None, 4):
            errors.append(f"&afwa WSPD10MAX ({domain}) is qualified for sf_sfclay_physics=5 with Noah-MP "
                          "sf_surface_physics=4 only (the producer of the in-step U10/V10)")
        if _int(_repeat(physics, "bldt", i)) not in (None, 0):
            errors.append(f"&afwa WSPD10MAX ({domain}) requires bldt=0 (U10/V10 every step)")
        # The reset window is this domain's history alarm: an explicit history_interval slot, because
        # an omitted nest slot is WRF's default while the port's writer repeats the last value.
        history = _value(time_control, "history_interval", i)
        if history is None:
            errors.append(f"&afwa WSPD10MAX ({domain}) requires an explicit history_interval for this domain")
        else:
            try:
                minutes = float(history)
            except (TypeError, ValueError):
                minutes = -1.0
            if not minutes > 0:
                errors.append("&afwa WSPD10MAX requires history_interval > 0 (WRF stops otherwise)")
    if any(severe_selected(wrf_namelist, d) for d in domains):
        time_control = {str(k).lower(): v for k, v in time_control.items()}
        for key in ALARM_CONTROLS:
            if time_control.get(key) not in (None, [], ()) and any(
                    _int(v) not in (None, 0) for v in (time_control[key] if isinstance(time_control[key], (list, tuple))
                                                        else [time_control[key]])):
                errors.append(f"&time_control {key}: moves the history alarm that resets AFWA WSPD10MAX; "
                              "only history_interval (minutes) is supported")
    return errors
