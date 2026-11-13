"""Pure-Python binding of the run-wide WRF feedback controls.

An omitted feedback remains OFF for the released one-way program. An explicit
namelist feedback=1 or CLI --feedback opts into the experimental two-way path.
WRF's omitted smooth_option is 2 (Registry.EM_COMMON), not the old CLI's 1.
"""
from __future__ import annotations

from typing import Any, Mapping


def resolve_nesting_options(namelist: Mapping[str, Any], *, feedback: bool | None = None):
    def scalar(group, key, default, allowed=None):
        raw = (namelist.get(group) or {}).get(key, default)
        if isinstance(raw, (tuple, list)):
            if len(raw) != 1:
                raise ValueError(f"&{group} {key} is one run-wide integer")
            raw = raw[0]
        try:
            value = int(raw)
            if float(raw) != value:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"&{group} {key} must be an integer") from None
        if allowed is not None and value not in allowed:
            raise ValueError(f"&{group} {key}={value}: allowed values are {allowed}")
        return value

    selected = scalar("domains", "feedback", 0, (0, 1))
    smooth = scalar("domains", "smooth_option", 2, (0, 1, 2))
    zone = scalar("bdy_control", "spec_zone", 1)
    if zone < 0:
        raise ValueError("&bdy_control spec_zone must be nonnegative")
    return {"feedback": bool(selected if feedback is None else feedback),
            "smooth_option": smooth, "spec_zone": zone}


def admit_feedback(namelist,names,*,feedback=None):
    """Refuse unqualified combinations before loading/JAX initialization."""
    if not resolve_nesting_options(namelist,feedback=feedback)['feedback'] or len(names)==1:
        return
    if tuple(names)!=('d01','d02'):
        raise ValueError('two-way feedback is qualified for d01+d02 only; static-field lifetime for deeper trees is unqualified')
    ratios=(namelist.get('domains') or {}).get('parent_grid_ratio',[1,3])
    ratio=ratios[min(1,len(ratios)-1)] if isinstance(ratios,(tuple,list)) else ratios
    if float(ratio)!=int(ratio) or int(ratio)<=1 or int(ratio)%2==0:
        raise ValueError('two-way feedback requires an odd integer parent_grid_ratio greater than 1')
    allowed={('physics','mp_physics'):(8,),('physics','bl_pbl_physics'):(5,),
        ('physics','sf_sfclay_physics'):(5,),('physics','sf_surface_physics'):(4,),
        ('physics','ra_lw_physics'):(4,),('physics','ra_sw_physics'):(4,),
        ('physics','cu_physics'):(0,1),('physics','sf_urban_physics'):(0,),
        ('physics','sf_lake_physics'):(0,),('physics','sst_update'):(0,),
        ('physics','swint_opt'):(0,),('physics','prec_acc_dt'):(0,),
        ('physics','do_radar_ref'):(0,),('time_control','nwp_diagnostics'):(0,),('afwa','afwa_severe_opt'):(0,),
        ('physics','bl_mynn_tkeadvect'):(0,),('dynamics','hypsometric_opt'):(2,),
        ('fdda','grid_fdda'):(0,),('fdda','obs_nudge_opt'):(0,)}
    defaults={'mp_physics':8,'bl_pbl_physics':5,'sf_sfclay_physics':5,'sf_surface_physics':4,
              'ra_lw_physics':4,'ra_sw_physics':4,'cu_physics':0,'hypsometric_opt':2}
    for (group,key),choices in allowed.items():
        raw=(namelist.get(group) or {}).get(key,defaults.get(key,0))
        values=raw if isinstance(raw,(list,tuple)) else [raw]
        if any(float(value)!=int(value) or int(value) not in choices for value in values[:len(names)]):
            raise ValueError(f'two-way feedback has no coupled qualification for &{group} {key}={raw}; allowed {choices}')


def require_feedback_restart(carries):
    """Validate complete active groups against each domain's State metadata."""
    from gpuwrf.io.feedback_restart import configuration
    for domain,carry in carries.items():
        if getattr(carry,'state',None) is None:
            raise ValueError(f'two-way restart {domain} lacks State shape binding (E78)')
        configuration(getattr(carry,'base_state',None),getattr(carry,'feedback_history',None),
                      carry.state.qv.shape,carry.state.t_skin.shape,required=True)
