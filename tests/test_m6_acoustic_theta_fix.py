from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import OperationalNamelist, _operational_acoustic_substep_core
from gpuwrf.runtime.operational_state import OperationalCarry


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "m6-acoustic-theta-fix" / "step17_input_state.npz"
RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
BAD_CELL = (12, 30, 62)


def _carry_from_fixture(payload: np.lib.npyio.NpzFile) -> OperationalCarry:
    state = State(**{name: jnp.asarray(payload[f"state_{name}"]) for name in State.__slots__})
    return OperationalCarry(
        state=state,
        t_2ave=jnp.asarray(payload["carry_t_2ave"]),
        ww=jnp.asarray(payload["carry_ww"]),
        mudf=jnp.asarray(payload["carry_mudf"]),
        muave=jnp.asarray(payload["carry_muave"]),
        muts=jnp.asarray(payload["carry_muts"]),
        ph_tend=jnp.asarray(payload["carry_ph_tend"]),
        u_save=jnp.asarray(payload["carry_u_save"]),
        v_save=jnp.asarray(payload["carry_v_save"]),
        w_save=jnp.asarray(payload["carry_w_save"]),
        t_save=jnp.asarray(payload["carry_t_save"]),
        ph_save=jnp.asarray(payload["carry_ph_save"]),
        mu_save=jnp.asarray(payload["carry_mu_save"]),
        ww_save=jnp.asarray(payload["carry_ww_save"]),
    )


def test_step17_bad_cell_acoustic_theta_increment_is_bounded():
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
    next_carry = _operational_acoustic_substep_core(carry, namelist, 1.0)
    jax.block_until_ready(next_carry.state.theta)
    before = float(payload["state_theta"][BAD_CELL])
    after = float(jax.device_get(next_carry.state.theta[BAD_CELL]))
    assert abs(after - before) < 50.0
