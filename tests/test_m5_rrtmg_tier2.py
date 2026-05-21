from __future__ import annotations

from gpuwrf.validation.tier2_rrtmg import run_tier2


def test_rrtmg_tier2_invariants_pass():
    record = run_tier2()
    assert record["pass"] is True
    assert record["shortwave_candidate_heating_flux_closure"]["pass"] is True
    assert record["shortwave_real_driver_energy_conservation"]["pass"] is True
    assert record["shortwave_real_driver_heating_flux_closure"]["pass"] is True
    assert record["longwave_real_driver_heating_flux_closure"]["pass"] is True
    assert record["longwave_candidate_heating_flux_closure"]["pass"] is True
    assert record["longwave_candidate_surface_emission_stefan_boltzmann"]["pass"] is True
    assert record["nan_inf"]["violations"] == 0
