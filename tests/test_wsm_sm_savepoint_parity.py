"""WSM3/WSM5 JAX ports vs pristine-WRF savepoint parity gates.

References are generated from UNMODIFIED pristine WRF
``/home/enric/src/wrf_pristine/WRF`` by
``proofs/v060/oracle/build_wsm_sm_oracles.sh``. This is not a JAX self-compare.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.physics_interfaces import PhysicsTendency  # noqa: E402
from gpuwrf.physics.microphysics_wsm3 import wsm3_physics_tendency  # noqa: E402
from gpuwrf.physics.microphysics_wsm5 import wsm5_physics_tendency  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "proofs" / "v060"
SAVE_FP64 = PROOFS / "savepoints_fp64"
RUNNER_PATH = PROOFS / "run_wsm_sm_parity.py"
CASES = (1, 2, 3, 4, 5, 6)


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_wsm_sm_parity", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _have_savepoints() -> bool:
    return all((SAVE_FP64 / f"{scheme}_case_{cid}.json").exists() for scheme in ("wsm3", "wsm5") for cid in CASES)


pytestmark = pytest.mark.skipif(
    not _have_savepoints(),
    reason="WSM3/WSM5 fp64 pristine-WRF savepoints missing; run proofs/v060/oracle/build_wsm_sm_oracles.sh fp64",
)


def _col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


@pytest.mark.parametrize("scheme_name", ("wsm3", "wsm5"))
@pytest.mark.parametrize("cid", CASES)
def test_wsm_sm_final_state_and_tendencies_vs_fp64_oracle(scheme_name: str, cid: int) -> None:
    ok, payload = runner.run_case(runner.SCHEMES[scheme_name], cid)
    assert ok, json.dumps(payload, indent=2)

    # Make the task-critical tendency coverage explicit in the test name and body.
    scheme = runner.SCHEMES[scheme_name]
    assert payload["fields"]["t_tendency"]["pass"] is True
    for leaf, _in_name, _out_name in scheme.q_fields:
        assert payload["fields"][f"{leaf}_tendency"]["pass"] is True


@pytest.mark.parametrize("scheme_name", ("wsm3", "wsm5"))
def test_wsm_sm_generated_report_declares_pristine_oracle_pass(scheme_name: str) -> None:
    report_path = PROOFS / f"{scheme_name}_savepoint_parity_report.json"
    if not report_path.exists():
        pytest.skip(f"{report_path} not generated yet")
    report = json.loads(report_path.read_text())
    assert report["schema"] == f"gpuwrf.v060.{scheme_name}_savepoint_parity.v1"
    assert report["overall_pass"] is True
    assert report["oracle"]["no_self_compare"] is True
    assert report["oracle"]["wrf_source"] == "/home/enric/src/wrf_pristine/WRF"
    assert report["predeclared_tolerances"] == runner.PREDECLARED_TOL


def test_wsm3_physics_tendency_adapter_contract() -> None:
    d = json.loads((SAVE_FP64 / "wsm3_case_2.json").read_text())
    s = d["scalars"]
    pii = _col(d, "PII")[None, :]
    theta = _col(d, "T_IN")[None, :] / pii
    tend = wsm3_physics_tendency(
        jnp.asarray(theta),
        jnp.asarray(_col(d, "QV_IN")[None, :]),
        jnp.asarray(_col(d, "QC_IN")[None, :]),
        jnp.asarray(_col(d, "QR_IN")[None, :]),
        jnp.asarray(_col(d, "W_IN")[None, :]),
        jnp.asarray(pii),
        jnp.asarray(_col(d, "DEN")[None, :]),
        jnp.asarray(_col(d, "P")[None, :]),
        jnp.asarray(_col(d, "DELZ")[None, :]),
        s["DT"],
    )
    assert isinstance(tend, PhysicsTendency)
    tend.validate_keys()
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr"}
    assert tend.state_tendencies == {} or len(tend.state_tendencies) == 0
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc"}
    assert {"re_cloud", "re_ice", "re_snow", "sr"} <= set(tend.diagnostics)

    t_new = np.asarray(tend.state_replacements["theta"], dtype=np.float64)[0] * pii[0]
    assert np.max(np.abs(t_new - _col(d, "T_OUT"))) <= runner.PREDECLARED_TOL["t_abs"]


def test_wsm5_physics_tendency_adapter_contract() -> None:
    d = json.loads((SAVE_FP64 / "wsm5_case_2.json").read_text())
    s = d["scalars"]
    pii = _col(d, "PII")[None, :]
    theta = _col(d, "T_IN")[None, :] / pii
    tend = wsm5_physics_tendency(
        jnp.asarray(theta),
        jnp.asarray(_col(d, "QV_IN")[None, :]),
        jnp.asarray(_col(d, "QC_IN")[None, :]),
        jnp.asarray(_col(d, "QR_IN")[None, :]),
        jnp.asarray(_col(d, "QI_IN")[None, :]),
        jnp.asarray(_col(d, "QS_IN")[None, :]),
        jnp.asarray(pii),
        jnp.asarray(_col(d, "DEN")[None, :]),
        jnp.asarray(_col(d, "P")[None, :]),
        jnp.asarray(_col(d, "DELZ")[None, :]),
        s["DT"],
    )
    assert isinstance(tend, PhysicsTendency)
    tend.validate_keys()
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr", "qi", "qs"}
    assert tend.state_tendencies == {} or len(tend.state_tendencies) == 0
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc"}
    assert {"re_cloud", "re_ice", "re_snow", "sr"} <= set(tend.diagnostics)

    t_new = np.asarray(tend.state_replacements["theta"], dtype=np.float64)[0] * pii[0]
    assert np.max(np.abs(t_new - _col(d, "T_OUT"))) <= runner.PREDECLARED_TOL["t_abs"]
