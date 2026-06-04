import json
import importlib.util
import os
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
    # The committed lane report is AUTHORITATIVE. By default this test ASSERTS the
    # schema/verdict without overwriting it (running pytest must not silently
    # regenerate a committed proof). Set GPUWRF_WRITE_PARITY_REPORT=1 to explicitly
    # regenerate the report (the intended, deliberate proof-refresh action).
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
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
    # The faithful cup_gf / cup_gf_sh port must reproduce the WRF-module
    # savepoints to within the predeclared tolerances across all 5 regimes.
    assert report["verdict"] == "PASS", report["failures"]


def test_grell_freitas_faithful_column_matches_oracle():
    """Direct per-field check: faithful column vs WRF-module savepoints."""
    from gpuwrf.physics.cumulus_grell_freitas import grell_freitas_column

    for case_id in (1, 2, 3, 4, 5):
        save = Path(f"proofs/v060/savepoints/gf_case_{case_id}.json")
        if not save.exists():
            return
        data = json.loads(save.read_text())
        s = data["scalars"]
        c = data["columns"]
        out = grell_freitas_column(
            np.asarray(c["T"]), np.asarray(c["QV"]), np.asarray(c["P"]),
            np.asarray(c["DZ"]), np.asarray(c["RHO"]), np.asarray(c["W"]),
            dt=float(s["DT"]), dx=float(s["DX"]),
            pi_exner=np.asarray(c["PI"]), u=np.asarray(c["U"]),
            v=np.asarray(c["V"]), rthblten=np.asarray(c["RTHBLTEN"]),
            rqvblten=np.asarray(c["RQVBLTEN"]), kpbl=int(s["KPBL"]),
            hfx=float(s["HFX"]), qfx=float(s["QFX"]), xland=float(s["XLAND"]),
        )
        # RAINCV within 5% (abs floor 1e-4)
        rc_o = float(s["RAINCV"])
        rc_j = float(out["RAINCV"])
        assert abs(rc_j - rc_o) <= max(1.0e-4, 0.05 * abs(rc_o)), (case_id, rc_j, rc_o)
        # tendency fields within 5% relative (abs floor 1e-8)
        for fld in ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN"):
            oracle = np.asarray(c[fld], dtype=np.float64)
            jax = np.asarray(out[fld], dtype=np.float64)
            max_abs = float(np.max(np.abs(jax - oracle)))
            scale = max(float(np.max(np.abs(oracle))), 1.0e-8)
            assert (max_abs / scale <= 0.05) or (max_abs <= 1.0e-8), (
                case_id, fld, max_abs, scale)
        # categorical trigger match
        deep_o = int(s["KTOP_DEEP"]) > 0 and float(s["RAINCV"]) > 0.0
        assert bool(out["TRIGGER_DEEP"]) == deep_o, (case_id, "deep")
        shallow_o = float(s["XMB_SHALLOW"]) > 0.0 or int(s["KTOP_SHALLOW"]) > 0
        assert bool(out["TRIGGER_SHALLOW"]) == shallow_o, (case_id, "shallow")


def test_grell_freitas_gpubatch_matches_oracle():
    """GPU-batched (jit/vmap) GF kernel reproduces the WRF-module savepoints
    within the predeclared 2e-2 tolerance across all 5 regimes."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    save = Path("proofs/v060/savepoints/gf_case_1.json")
    if not save.exists():
        return
    from gpuwrf.physics import _gf_jax as J

    for case_id in (1, 2, 3, 4, 5):
        data = json.loads(Path(f"proofs/v060/savepoints/gf_case_{case_id}.json").read_text())
        s = data["scalars"]; c = data["columns"]
        out = J.grell_freitas_column_gpu(
            np.asarray(c["T"]), np.asarray(c["QV"]), np.asarray(c["P"]),
            np.asarray(c["DZ"]), np.asarray(c["RHO"]), np.asarray(c["W"]),
            dt=float(s["DT"]), dx=float(s["DX"]), pi_exner=np.asarray(c["PI"]),
            u=np.asarray(c["U"]), v=np.asarray(c["V"]),
            rthblten=np.asarray(c["RTHBLTEN"]), rqvblten=np.asarray(c["RQVBLTEN"]),
            kpbl=int(s["KPBL"]), hfx=float(s["HFX"]), qfx=float(s["QFX"]),
            xland=float(s["XLAND"]))
        rc_o = float(s["RAINCV"]); rc_j = float(out["RAINCV"])
        assert abs(rc_j - rc_o) <= max(1.0e-4, 0.05 * abs(rc_o)), (case_id, rc_j, rc_o)
        for fld in ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN"):
            oracle = np.asarray(c[fld], dtype=np.float64)
            jax_v = np.asarray(out[fld], dtype=np.float64)
            max_abs = float(np.max(np.abs(jax_v - oracle)))
            scale = max(float(np.max(np.abs(oracle))), 1.0e-8)
            assert (max_abs / scale <= 0.02) or (max_abs <= 1.0e-8), (case_id, fld, max_abs / scale)
        deep_o = int(s["KTOP_DEEP"]) > 0 and float(s["RAINCV"]) > 0.0
        assert bool(out["TRIGGER_DEEP"]) == deep_o, (case_id, "deep")
        shallow_o = float(s["XMB_SHALLOW"]) > 0.0 or int(s["KTOP_SHALLOW"]) > 0
        assert bool(out["TRIGGER_SHALLOW"]) == shallow_o, (case_id, "shallow")


def test_grell_freitas_gpubatch_report_pass():
    if not Path("proofs/v060/savepoints/gf_case_1.json").exists():
        return
    import importlib.util
    script = Path("proofs/v060/run_grellfreitas_gpubatch_parity.py")
    spec = importlib.util.spec_from_file_location("run_gf_gpubatch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report()
    assert report["gpu_batched"] is True
    assert report["jit_vmap_native_kernel"] is True
    assert report["no_host_transfer_in_column_loop"] is True
    assert report["verdict"] == "PASS", report["failures"]
