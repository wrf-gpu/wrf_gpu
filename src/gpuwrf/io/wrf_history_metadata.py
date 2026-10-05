"""WRF history metadata from input headers and resolved namelist controls."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

# Frozen WRF v4 Registry defaults; per-domain values use the model's namelist resolution.
# Registry.EM_COMMON:2225-2884; registry.stoch:251. MPI task counts describe this GPU process.
_HISTORY_CONTROLS = {
    'AER_ANGEXP_OPT': ('physics', 'integer', True, 1),
    'AER_ANGEXP_VAL': ('physics', 'real', True, 1.3),
    'AER_AOD550_OPT': ('physics', 'integer', True, 1),
    'AER_AOD550_VAL': ('physics', 'real', True, 0.12),
    'AER_ASY_OPT': ('physics', 'integer', True, 1),
    'AER_ASY_VAL': ('physics', 'real', True, 0.9),
    'AER_OPT': ('physics', 'integer', False, 0),
    'AER_SSA_OPT': ('physics', 'integer', True, 1),
    'AER_SSA_VAL': ('physics', 'real', True, 0.85),
    'AER_TYPE': ('physics', 'integer', True, 1),
    'BLDT': ('physics', 'real', True, 0.0),
    'BUCKET_J': ('physics', 'real', False, -1.0),
    'BUCKET_MM': ('physics', 'real', False, -1.0),
    'CLDOVRLP': ('physics', 'integer', False, 2),
    'CUDT': ('physics', 'real', True, 0.0),
    'DFI_OPT': ('dfi_control', 'integer', False, 0),
    'DIFF_6TH_FACTOR': ('dynamics', 'real', True, 0.12),
    'DIFF_6TH_OPT': ('dynamics', 'integer', True, 0),
    'DVEG': ('noah_mp', 'integer', False, 4),
    'FEEDBACK': ('domains', 'integer', False, 1),
    'GRAV_SETTLING': ('physics', 'integer', True, 0),
    'ICLOUD': ('physics', 'integer', False, 1),
    'ICLOUD_CU': ('derived', 'integer', True, 0),
    'IDCOR': ('physics', 'integer', False, 0),
    'ISFFLX': ('physics', 'integer', False, 1),
    'ISFTCFLX': ('physics', 'integer', False, 0),
    'ISHALLOW': ('physics', 'integer', False, 0),
    'MFSHCONV': ('physics', 'integer', True, 1),
    'MOIST_ADV_OPT': ('dynamics', 'integer', True, 1),
    'OBS_NUDGE_OPT': ('fdda', 'integer', True, 0),
    'OPT_ALB': ('noah_mp', 'integer', False, 2),
    'OPT_BTR': ('noah_mp', 'integer', False, 1),
    'OPT_CROP': ('noah_mp', 'integer', False, 0),
    'OPT_CRS': ('noah_mp', 'integer', False, 1),
    'OPT_FRZ': ('noah_mp', 'integer', False, 1),
    'OPT_GLA': ('noah_mp', 'integer', False, 1),
    'OPT_INF': ('noah_mp', 'integer', False, 1),
    'OPT_IRR': ('noah_mp', 'integer', False, 0),
    'OPT_IRRM': ('noah_mp', 'integer', False, 0),
    'OPT_PEDO': ('noah_mp', 'integer', False, 1),
    'OPT_RAD': ('noah_mp', 'integer', False, 3),
    'OPT_RSF': ('noah_mp', 'integer', False, 1),
    'OPT_RUN': ('noah_mp', 'integer', False, 3),
    'OPT_SFC': ('noah_mp', 'integer', False, 1),
    'OPT_SNF': ('noah_mp', 'integer', False, 1),
    'OPT_SOIL': ('noah_mp', 'integer', False, 1),
    'OPT_STC': ('noah_mp', 'integer', False, 1),
    'OPT_TBOT': ('noah_mp', 'integer', False, 2),
    'PREC_ACC_DT': ('physics', 'real', True, 0.0),
    'RADT': ('physics', 'real', True, 0.0),
    'SCALAR_ADV_OPT': ('dynamics', 'integer', True, 1),
    'SCALAR_PBLMIX': ('physics', 'integer', True, 0),
    'SHCU_PHYSICS': ('physics', 'integer', True, 0),
    'SKEBS_ON': ('derived', 'integer', False, 0),
    'SMOOTH_OPTION': ('domains', 'integer', False, 2),
    'SWINT_OPT': ('physics', 'integer', False, 0),
    'SWRAD_SCAT': ('physics', 'real', False, 1.0),
    'TKE_ADV_OPT': ('dynamics', 'integer', True, 1),
    'TRACER_PBLMIX': ('physics', 'integer', True, 1),
    'USE_Q_DIABATIC': ('dynamics', 'integer', False, 0),
    'W_DAMPING': ('dynamics', 'integer', False, 0),
    'YSU_TOPDOWN_PBLMIX': ('physics', 'integer', False, 1),
}


def wrf_history_global_attributes(
    input_path: str | Path,
    namelist: Mapping[str, Mapping[str, Any]],
    domain: str,
    dt_s: float,
) -> dict[str, Any]:
    """Copy static WRF input metadata; derive history-only controls without CPU outputs."""
    with Dataset(input_path, "r") as initial:
        attrs = {name: initial.getncattr(name) for name in initial.ncattrs()
                 if name not in {"USE_TROP_LEVEL", "USE_MAXW_LEVEL"}}
    index = int(domain[1:]) - 1
    for name, (section, kind, per_domain, default) in _HISTORY_CONTROLS.items():
        value = namelist.get(section, {}).get(name.lower(), default)
        if isinstance(value, (tuple, list)):
            value = value[min(index, len(value)-1) if per_domain else 0] if value else default
        if value is None:
            value = default
        attrs[name] = np.float32(value) if kind == "real" else np.int32(value)
    # WRF share/module_check_a_mundo.F:1730-1743 resets this QNSE-only control.
    if int(attrs.get("BL_PBL_PHYSICS", 0)) != 4:
        attrs["MFSHCONV"] = np.int32(0)
    attrs.update(NTASKS_X=np.int32(1), NTASKS_Y=np.int32(1), NTASKS_TOTAL=np.int32(1))
    attrs["DT"] = np.float32(dt_s)
    return attrs
