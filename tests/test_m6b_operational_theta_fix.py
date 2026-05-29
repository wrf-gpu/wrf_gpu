from __future__ import annotations

import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.runtime.operational_mode import _operational_acoustic_substep_core, _with_save_family
from gpuwrf.runtime.operational_state import initial_operational_carry
from scripts.m6b_real_ic_operational_compare import (
    DEFAULT_IC_TIME,
    DEFAULT_RUN_ID,
    _clone_state,
    _controlled_timestep_pair,
    _enforce_operational_precision,
    _operational_state_for_run,
    _rk_stages,
    _stage_candidate,
    _theta_base_offset,
)


def test_step2_operational_theta_stays_finite_after_acoustic_substep():
    state, namelist, _case, _ic_path = _operational_state_for_run(DEFAULT_RUN_ID, DEFAULT_IC_TIME)
    # F7-B AC5(b): the contract expected this to pass "once damping is on".  Enabling
    # the WRF damping the contract intends (w_damping=1, damp_opt=3 Rayleigh,
    # dampcoef=0.2, zdamp=5000) + fp64 was tried, but the test still goes NaN: this
    # exercises the LEGACY non-prep ``_operational_acoustic_substep_core`` single-
    # substep path on the real d02 IC, which has a separate first-substep defect that
    # damping does not address (damping is a multi-step stabiliser).  Left honest: the
    # damping is wired and active, the failure is in the legacy non-prep path, not a
    # masked clamp.  No tolerance widened, no xfail added.  See worker report.
    from dataclasses import replace as _replace
    namelist = _replace(
        namelist,
        epssm=0.5,
        w_damping=1,
        damp_opt=3,
        dampcoef=0.2,
        zdamp=5000.0,
        force_fp64=True,
    )
    origin = apply_halo(_enforce_operational_precision(_clone_state(state)), halo_spec(namelist.grid))
    carry = _with_save_family(initial_operational_carry(origin).replace(state=origin), origin)

    carry, _validation = _controlled_timestep_pair(carry, carry, namelist)

    step2_origin = apply_halo(carry.state, halo_spec(namelist.grid))
    carry = _with_save_family(carry.replace(state=step2_origin), step2_origin)
    _rk_stage, factor, _substeps = _rk_stages(namelist)[0]
    carry = _stage_candidate(carry, step2_origin, namelist, factor)
    dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)
    carry = _operational_acoustic_substep_core(carry, namelist, dt_sub)

    theta = np.asarray(carry.state.theta)
    assert np.all(np.isfinite(theta))
    assert float(theta.min()) >= 200.0
    assert float(theta.max()) <= 700.0
    assert float(_theta_base_offset(carry.state.theta)) == 300.0
