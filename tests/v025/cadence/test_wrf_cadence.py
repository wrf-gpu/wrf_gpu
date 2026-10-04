"""Cadence controls compared with expressions extracted from pristine WRF."""
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from gpuwrf.integration.nested_pipeline import (
    _domain_float, _radiation_cadence_steps, _wrf_sound_steps,
)


@pytest.fixture(scope="module")
def wrf_cadence_oracle(tmp_path_factory, wrf_fortran_compiler, frozen_wrf_oracles):
    # The unchanged WRF expressions are checked in beside the other frozen
    # oracles; runtime table/asset roots need not contain a complete WRF tree.
    source = Path(__file__).with_name("oracles") / "cadence_oracle.f90"
    out = tmp_path_factory.mktemp("wrf-cadence")
    subprocess.run([wrf_fortran_compiler, "-ffree-line-length-none", str(source),
                    "-o", str(out / "oracle")], check=True, capture_output=True)
    return out / "oracle"


def test_prod_and_nint_half_ties_match_wrf_source(wrf_cadence_oracle):
    cases = [(54., 9000., 30., 5.), (18., 3000., 30., 5.),
             (60., 9000., .5, .5), (120., 9000., 5., 5.),
             (90., 9000., 30., 0.), (40., 2000., 10., 5.)]
    output = subprocess.run([str(wrf_cadence_oracle)],
        input="".join(" ".join(map(str, row)) + "\n" for row in cases),
        text=True, capture_output=True, check=True).stdout.splitlines()
    for (dt, spacing, radt, cudt), line in zip(cases, output, strict=True):
        expected_sound, expected_rad, expected_cu = map(int, line.split())
        grid = SimpleNamespace(projection=SimpleNamespace(dx_m=spacing, dy_m=spacing))
        assert _wrf_sound_steps(grid, dt) == expected_sound
        assert _radiation_cadence_steps(dt, radt) == expected_rad
        assert _radiation_cadence_steps(dt, cudt) == expected_cu
    assert list(map(int, output[0].split())) == [4, 33, 6]
    assert list(map(int, output[1].split())) == [4, 100, 17]


def test_explicit_sound_steps_and_real_per_domain_controls():
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=9000., dy_m=9000.))
    assert _wrf_sound_steps(grid, 54., configured=8) == 8
    run = SimpleNamespace(namelist={"physics": {"radt": [30., 7.5], "cudt": 5.}})
    assert _domain_float(run, "physics", "radt", "d02", 30.) == 7.5
    assert _domain_float(run, "physics", "cudt", "d02", 0.) == 5.
