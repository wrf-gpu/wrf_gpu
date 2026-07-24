from pathlib import Path

import numpy as np

from scripts import v0234_thompson_first_ice_audit as audit


ROOT = Path(__file__).resolve().parents[1]


def test_literal_deposition_predicate_and_units() -> None:
    result = audit.deposition_nucleation_oracle(
        qv=np.array([[2.0e-3, 1.0e-5]]),
        qi=np.zeros((1, 2)),
        ni=np.zeros((1, 2)),
        p=np.array([[40000.0, 40000.0]]),
        temperature=np.array([[240.0, 270.0]]),
        dt=6.0,
        np=np,
    )
    assert result["predicate"].tolist() == [[True, False]]
    assert result["qi_increment_kg_kg"][0, 0] > 0.0
    assert np.isclose(
        result["qi_increment_kg_kg"][0, 0]
        / result["ni_increment_kg_inv"][0, 0],
        audit.XM0I,
        rtol=0.0,
        atol=1e-27,
    )
    assert result["qi_increment_kg_kg"][0, 1] == 0.0


def test_saturation_transcription_is_finite_and_ordered() -> None:
    p = np.array([30000.0, 50000.0, 90000.0])
    t = np.array([230.0, 250.0, 270.0])
    ice = audit.wrf_rsif(p, t, np)
    liquid = audit.wrf_rslf(p, t, np)
    assert np.all(np.isfinite(ice))
    assert np.all(np.isfinite(liquid))
    assert np.all(ice < liquid)
    assert np.all(ice > 0.0)


def test_literal_condensation_hgfr_path_and_number_ownership() -> None:
    result = audit.condensation_instant_freeze_oracle(
        qv=np.array([[2.0e-3, 1.0e-5]]),
        qc=np.zeros((1, 2)),
        p=np.array([[40000.0, 40000.0]]),
        temperature=np.array([[225.0, 270.0]]),
        dt=6.0,
        np=np,
    )
    assert result["condensation_predicate"].tolist() == [[True, False]]
    assert result[
        "direct_hgfr_freeze_predicate_before_cloud_sedimentation"
    ].tolist() == [[True, False]]
    assert result["clap_kg_kg"][0, 0] > 0.0
    assert result["port_instant_freeze_Ni_source_kg_inv"][0, 0] > 0.0
    assert result["wrf_nc1d_baseline_kg_inv"][0, 0] > 0.0


def test_contract_and_read_only_guards_are_bound() -> None:
    contract = (
        ROOT
        / ".agent/sprints/2026-07-17-v0234-thompson-ice-ownership-audit/CONTRACT.md"
    ).read_text()
    source = (ROOT / "scripts/v0234_thompson_first_ice_audit.py").read_text()
    assert audit.CONTRACT_COMMIT == "b86dcf701732cce525120baf29ecf53ce6d95b43"
    assert "No `src/gpuwrf` edits" in contract
    assert '"CUDA_VISIBLE_DEVICES": ""' in source
    assert '"JAX_PLATFORMS": "cpu"' in source
    assert "fresh_history_steps\": 0" in source
    assert "getattr(column, name)[yindex, xindex]" in source
    assert "y * int(namelist.grid.nx)" not in source
    assert "THOMPSON_FIRST_ICE_EXPECTED_REDIRECT" in source


def test_pristine_branch_literals_are_not_tunable() -> None:
    assert audit.TNO == 5.0
    assert audit.ATO == 0.304
    assert audit.XM0I == 1.0e-12
    assert audit.EPS == 1.0e-15
    source = (ROOT / "scripts/v0234_thompson_first_ice_audit.py").read_text()
    assert "ssati >= 0.25" in source
    assert "temperature < 253.15" in source
    assert "np.minimum(250.0e3, TNO * np.exp" in source


def test_real_savepoint_scope_respects_manifest_endianness() -> None:
    scope = audit._savepoint_scope(np)
    assert scope["byte_order"] == "big-endian"
    assert scope["dtype"] == "float64"
    assert scope["qi_input_max"] == 3.192096934001576e-11
    assert scope["qi_output_max"] == 0.0
    assert scope["ni_output_max"] == 0.0
