import json
import importlib.util
from pathlib import Path

import numpy as np


def test_grell_freitas_scale_factor_damps_fine_grid():
    from gpuwrf.physics.cumulus_grell_freitas import grell_freitas_scale_factor

    coarse = float(grell_freitas_scale_factor(15000.0))
    parent = float(grell_freitas_scale_factor(9000.0))
    fine = float(grell_freitas_scale_factor(3000.0))
    assert 0.0 < fine < parent < coarse <= 1.0


def test_grell_freitas_step_interface_keys():
    from gpuwrf.physics.cumulus_grell_freitas import CARRY_KEYS, grell_freitas_step

    k = 8
    p = np.linspace(100000.0, 60000.0, k)
    pi = (p / 100000.0) ** (287.0 / 1004.0)
    t = np.linspace(300.0, 260.0, k)
    result = grell_freitas_step(
        {
            "t": t,
            "qv": np.linspace(0.014, 0.002, k),
            "p": p,
            "pi": pi,
            "dz": np.full(k, 350.0),
            "rho": np.full(k, 1.0),
            "w": np.linspace(0.0, 1.0, k),
        },
        dt=54.0,
        dx=9000.0,
        kpbl=3,
        hfx=350.0,
        qfx=2.0e-4,
    )
    result.tendency.validate_keys()
    assert set(result.tendency.state_tendencies) == {"theta", "qv", "qc", "qr", "qi", "qs"}
    assert set(result.tendency.accumulator_increments) == {"rainc_acc"}
    assert set(CARRY_KEYS).issubset(result.carry.cumulus)


def test_grell_freitas_parity_report_schema_when_savepoints_exist():
    save = Path("proofs/v060/savepoints/gf_case_1.json")
    if not save.exists():
        return
    script = Path("proofs/v060/run_grellfreitas_parity.py")
    spec = importlib.util.spec_from_file_location("run_grellfreitas_parity", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = module.build_report()
    module.REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["schema"] == "gpuwrf.v060.grellfreitas_savepoint_parity.v1"
    assert report["oracle"]["full_wrf_exe_run"] is False
    assert {case["regime"] for case in report["cases"]} == {
        "deep_convective",
        "shallow_convective",
        "stable_nontriggering",
        "scale_aware_coarse_15km",
        "scale_aware_fine_3km",
    }
    assert report["verdict"] in {"PASS", "FAIL"}
