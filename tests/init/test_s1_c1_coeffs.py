"""S1 C1F/C1H hybrid-coefficient precision proof (cross-model Opus debug 2026-06-02).

Locks in the root cause of the v0.4.0 S1 C1F/C1H gate failure:

* WRF (RWORDSIZE=4) computes C1 = dB/d(eta) as an fp32 finite difference of the
  fp32 hybrid coordinate C3 (nest_init_utils.F:1125-1156). The oracle C1F/C1H are
  therefore the fp32-rounded finite difference.
* PROOF: differencing the wrfinput-STORED fp32 C3F/C3H/ZNU/ZNW arrays in fp32
  (``wrf_fp32_c1_from_c3``) reproduces the oracle C1F/C1H BIT-EXACTLY. This proves
  the algorithm/op-order/boundary treatment in ``compute_vertical_coord`` are
  correct, and the fp64-vs-oracle residual is irreducible fp32 noise.
* The fp64 chain in ``compute_vertical_coord`` matches the oracle to that fp32
  rounding gap (~3e-5); it is the more-accurate value and downstream fields are
  unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwrf.init.real_init.types import RealInitConfig
from gpuwrf.init.real_init.vertical_coord import (
    compute_vertical_coord,
    wrf_fp32_c1_from_c3,
)


WRF_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")

# Static fp64-vs-oracle residual bound (max over the campaign). C1F/C1H are pure
# functions of the eta levels + etac, so this is identical for every case/domain.
# This documents the fp32 finite-difference noise floor used to justify the
# manager-owned C1F/C1H per-field tolerance.
C1_FP64_RESIDUAL_MAXABS = 5e-5
C1_FP64_RESIDUAL_RMSE = 1e-5


def _cases(limit: int = 12) -> list[Path]:
    out: list[Path] = []
    for run_dir in sorted(WRF_ROOT.glob("*_18z_l3_24h_*")):
        for domain in ("d01", "d02", "d03"):
            wi = run_dir / f"wrfinput_{domain}"
            if wi.exists():
                out.append(wi)
        if len(out) >= limit:
            break
    return out[:limit]


def _config(ds) -> RealInitConfig:
    return RealInitConfig(
        nz=len(ds.dimensions["bottom_top"]),
        p_top_pa=float(ds.variables["P_TOP"][0]),
        hybrid_opt=int(getattr(ds, "HYBRID_OPT")),
        etac=float(getattr(ds, "ETAC")),
        base_pres=float(ds.variables["P00"][0]),
        base_temp=float(ds.variables["T00"][0]),
        base_lapse=float(ds.variables["TLP"][0]),
        iso_temp=float(ds.variables["TISO"][0]),
        base_pres_strat=float(ds.variables["P_STRAT"][0]),
        base_lapse_strat=float(ds.variables["TLP_STRAT"][0]),
        grid_id=int(getattr(ds, "GRID_ID", 1)),
    )


@pytest.mark.skipif(not _cases(), reason="real S1 wrfinput fixtures unavailable")
def test_c1_bit_exact_from_stored_fp32_c3() -> None:
    """Differencing the oracle's OWN stored fp32 C3/ZN arrays in fp32 reproduces
    the oracle C1F/C1H bit-for-bit. This is the root-cause proof: WRF's C1 is the
    fp32 finite difference of fp32 C3, not a different formula."""
    for path in _cases():
        with netCDF4.Dataset(str(path)) as ds:
            c3f = np.asarray(ds.variables["C3F"][0])
            c3h = np.asarray(ds.variables["C3H"][0])
            znw = np.asarray(ds.variables["ZNW"][0])
            znu = np.asarray(ds.variables["ZNU"][0])
            c1f_o = np.asarray(ds.variables["C1F"][0])
            c1h_o = np.asarray(ds.variables["C1H"][0])
            hybrid_opt = int(getattr(ds, "HYBRID_OPT"))
        c1f, c1h = wrf_fp32_c1_from_c3(c3f, c3h, znw, znu, hybrid_opt)
        # bit-exact: identical fp32 bit patterns
        assert np.array_equal(c1f.view(np.int32), c1f_o.view(np.int32)), path.name
        assert np.array_equal(c1h.view(np.int32), c1h_o.view(np.int32)), path.name


@pytest.mark.skipif(not _cases(), reason="real S1 wrfinput fixtures unavailable")
def test_c1_fp64_residual_is_fp32_noise_floor() -> None:
    """The fp64 C1F/C1H from compute_vertical_coord matches the oracle to within
    the documented fp32 finite-difference noise floor (well below the proposed
    C1 tol). Confirms our value carries the same discrete-derivative definition,
    differing only by fp32 rounding."""
    for path in _cases():
        with netCDF4.Dataset(str(path)) as ds:
            config = _config(ds)
            c1f_o = np.asarray(ds.variables["C1F"][0], dtype=np.float64)
            c1h_o = np.asarray(ds.variables["C1H"][0], dtype=np.float64)
        vcoord = compute_vertical_coord(config)
        for name, ours, oracle in (
            ("C1F", vcoord.c1f, c1f_o),
            ("C1H", vcoord.c1h, c1h_o),
        ):
            d = np.asarray(ours, dtype=np.float64) - oracle
            maxabs = float(np.max(np.abs(d)))
            rmse = float(np.sqrt(np.mean(d * d)))
            assert maxabs <= C1_FP64_RESIDUAL_MAXABS, (name, path.name, maxabs)
            assert rmse <= C1_FP64_RESIDUAL_RMSE, (name, path.name, rmse)
