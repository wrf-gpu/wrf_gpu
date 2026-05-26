from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _operational_acoustic_substep_core,
    _with_save_family,
)
from gpuwrf.runtime.operational_state import OperationalCarry


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "m6-rk-save-family-fix" / "step46_input_state.npz"
RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
BAD_CELL = (11, 31, 67)


def _carry_from_fixture(payload: np.lib.npyio.NpzFile) -> OperationalCarry:
    state = State(**{name: jnp.asarray(payload[f"state_{name}"]) for name in State.__slots__})
    return OperationalCarry(
        state=state,
        **{
            name: jnp.asarray(payload[f"carry_{name}"])
            for name in OperationalCarry.__dataclass_fields__
            if name != "state"
        },
    )


def test_step46_save_family_keeps_theta_denominator_physical():
    assert FIXTURE.exists(), f"missing sprint fixture: {FIXTURE}"
    payload = np.load(FIXTURE)
    case = build_replay_case(RUN_DIR)
    carry = _carry_from_fixture(payload)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
        disable_guards=True,
    )

    prepped = _with_save_family(carry.replace(state=carry.state), carry.state)
    next_carry = _operational_acoustic_substep_core(prepped, namelist, 1.0)
    jax.block_until_ready(next_carry.state.theta)

    k, j, i = BAD_CELL
    denominator = float(jax.device_get(namelist.metrics.c1h[k] * next_carry.muts[j, i] + namelist.metrics.c2h[k]))
    assert denominator > 50000.0
