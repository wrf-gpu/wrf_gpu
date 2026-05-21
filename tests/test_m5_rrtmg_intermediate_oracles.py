from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwrf.validation.rrtmg_intermediate_oracles import (
    ORACLE,
    run_intermediate_validation,
    validate_lw_fracs_per_band,
    validate_lw_planck_corrections,
    validate_lw_planck_state,
    validate_lw_taug_per_band,
    validate_sw_setcoef_state,
    validate_sw_taug_per_band,
    validate_sw_taur,
)


def test_rrtmg_intermediate_oracle_fixture_shapes_and_budget():
    assert ORACLE.exists()
    assert ORACLE.stat().st_size <= 30_000_000
    with np.load(ORACLE, allow_pickle=False) as loaded:
        assert loaded["sw_taug"].shape == (3, 17, 12, 14)
        assert loaded["sw_taur"].shape == (3, 17, 12, 14)
        assert loaded["lw_taug"].shape == (3, 17, 16, 16)
        assert loaded["lw_fracs"].shape == (3, 17, 16, 16)
        assert loaded["sw_per_band_flux"].shape == (3, 4, 14)
        assert loaded["lw_per_band_flux"].shape == (3, 4, 16)
        assert not any(name.endswith("_clip_count") for name in loaded.files)


def test_rrtmg_intermediate_validation_helpers_report_pass_and_fail():
    zeros = np.zeros((2, 3))
    tiny = np.full((2, 3), 1.0e-12)
    assert validate_sw_taug_per_band(tiny, zeros, 1)["pass"] is True
    assert validate_lw_taug_per_band(np.ones((2, 3)), zeros, 1)["pass"] is False
    assert validate_lw_fracs_per_band(tiny, zeros, 1)["pass"] is True
    assert validate_sw_taur(tiny, zeros)["pass"] is True

    setcoef = {"jp": zeros, "jt": zeros, "jt1": zeros, "fac00": zeros, "fac01": zeros, "fac10": zeros, "fac11": zeros, "indself": zeros, "indfor": zeros, "selffac": zeros, "forfac": zeros, "colmol": zeros}
    setcoef_result = validate_sw_setcoef_state(setcoef, setcoef)
    assert setcoef_result["pass"] is True
    assert setcoef_result["fields"]["fac00"]["abs_tol"] == 1.0e-4
    assert setcoef_result["fields"]["fac00"]["rel_tol"] == 1.0e-3

    planck = {"planklay": zeros, "planklev": np.zeros((2, 4)), "plankbnd": np.zeros(3)}
    assert validate_lw_planck_state(planck, planck)["pass"] is True
    assert validate_lw_planck_corrections(zeros, zeros, zeros, zeros)["pass"] is True


def test_rrtmg_intermediate_artifacts_are_written_honestly():
    record = run_intermediate_validation()
    assert Path("artifacts/m5/rrtmg_intermediate_validation.json").exists()
    assert Path("artifacts/m5/rrtmg_per_band_status.json").exists()
    assert record["sw"]["taur"]["pass"] is True
    assert all(item["pass"] for item in record["sw"]["taug_per_band"])
    assert all(item["pass"] for item in record["lw"]["per_band"])
    assert record["pass"] is True
