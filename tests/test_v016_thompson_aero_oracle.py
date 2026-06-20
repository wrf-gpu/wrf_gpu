"""v0.16 aerosol-aware Thompson (mp=28) WRF savepoint-oracle regression.

Re-runs the per-scheme oracle parity (JAX vs unmodified pristine WRF
mp_physics=28 grid savepoint) and asserts the predeclared binding gate.
Skips when the savepoint store is not mounted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "proofs" / "v016" / "thompson_aero_savepoint_parity.py"
ORACLE_DIR = Path("<DATA_ROOT>/wrf_gpu2/physics_oracle_v090/microphysics_thompson_aero")


def _load_parity_module():
    spec = importlib.util.spec_from_file_location("thompson_aero_savepoint_parity", PARITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qrfz_in_axis_consistent_with_mp8_default_slice():
    """The 4-D rain-freezing gather at the default IN index must reproduce the
    validated mp=8 kernel's (IN-sliced) table exactly — same asset, same slice."""

    import numpy as np

    from gpuwrf.physics.thompson_tables import THOMPSON_TABLES
    from gpuwrf.physics.thompson_aero_tables import THOMPSON_AERO_TABLES, N_IN_TABLE

    qrfz4 = np.asarray(THOMPSON_AERO_TABLES.qrfz4).reshape(37, 37, 45, N_IN_TABLE, 4)
    mp8 = np.asarray(THOMPSON_TABLES.qrfz).reshape(37, 37, 45, 4)
    assert np.array_equal(qrfz4[:, :, :, 27, :], mp8)


def test_activ_ncloud_matches_fortran_semantics():
    """Spot-check the CCN activation bilinear lookup against a direct
    transcription of WRF activ_ncloud (module_mp_thompson.F:5178-5253)."""

    import math

    import numpy as np

    from gpuwrf.physics.thompson_aero_column import _activ_ncloud
    from gpuwrf.physics.thompson_aero_tables import THOMPSON_AERO_TABLES as A

    ta_na = np.asarray(A.ta_na)
    ta_ww = np.asarray(A.ta_ww)
    table = np.asarray(A.ccn_act)

    def reference(tt, ww, nccn):
        n_local = nccn * 1.0e-6
        if n_local >= ta_na[-1]:
            n_local = ta_na[-1] - 1.0
        elif n_local <= ta_na[0]:
            n_local = ta_na[0] + 1.0
        w_local = ww
        if w_local >= ta_ww[-1]:
            w_local = ta_ww[-1] - 1.0
        elif w_local <= ta_ww[0]:
            w_local = ta_ww[0] + 0.001
        i = next(n for n in range(1, len(ta_na)) if ta_na[n - 1] <= n_local < ta_na[n])
        j = next(n for n in range(1, len(ta_ww)) if ta_ww[n - 1] <= w_local < ta_ww[n])
        k = max(0, min(int(math.floor((tt - 243.15) * 0.1 + 0.5)), table.shape[2] - 1))
        x1, x2 = math.log(ta_na[i - 1]), math.log(ta_na[i])
        y1, y2 = math.log(ta_ww[j - 1]), math.log(ta_ww[j])
        a, b = table[i - 1, j - 1, k], table[i, j - 1, k]
        c, d = table[i, j, k], table[i - 1, j, k]
        t = (math.log(n_local) - x1) / (x2 - x1)
        u = (math.log(w_local) - y1) / (y2 - y1)
        frac = (1 - t) * (1 - u) * a + t * (1 - u) * b + t * u * c + (1 - t) * u * d
        return nccn * frac

    rng = np.random.default_rng(7)
    for _ in range(200):
        tt = float(rng.uniform(235.0, 310.0))
        ww = float(rng.uniform(-1.0, 12.0))
        nccn = float(10.0 ** rng.uniform(6.5, 9.9))
        got = float(np.asarray(_activ_ncloud(tt, ww, nccn, A)))
        want = reference(tt, ww, nccn)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-6), (tt, ww, nccn)


@pytest.mark.skipif(not (ORACLE_DIR / "manifest.json").exists(), reason="mp=28 WRF oracle savepoint not mounted")
def test_thompson_aero_savepoint_parity(tmp_path):
    parity = _load_parity_module()
    record = parity.run(tmp_path / "parity.json")
    assert record["status"] == "ORACLE-VALIDATED"
    per_field = record["per_field"]
    for field in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "th", "rainncv"):
        assert per_field[field]["carry_pass"], f"{field} failed the binding carry band: {per_field[field]}"
    assert record["water_closure_pass"], record["water_closure_max_rel_residual"]
    assert record["pass"]
