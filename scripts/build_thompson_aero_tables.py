#!/usr/bin/env python3
"""Build the aerosol-aware Thompson (mp_physics=28) lookup-table asset.

Produces ``data/fixtures/thompson-aero-tables-v1.npz`` with the four table
families the aerosol-aware path needs BEYOND the existing mp=8 asset
(``thompson-tables-v1.npz``):

1. ``tnccn_act`` — the CCN activation fraction table (Eidhammer parcel-model
   results), read VERBATIM from the WRF run-directory binary
   ``CCN_ACTIVATE.BIN`` (big-endian Fortran sequential, one REAL*4 record of
   shape (ntb_arc=7, ntb_arw=9, ntb_art=7, ntb_arr=5, ntb_ark=4), F-order;
   ``module_mp_thompson.F:5110-5166``).
2. ``tpc_wev``/``tnc_wev`` — droplet-evaporation partial-moment tables
   (``table_dropEvap``, module_mp_thompson.F:5011-5099), recomputed here in
   fp64 from the WRF closed forms.
3. ``tpi_qcfz``/``tni_qcfz`` — heterogeneous cloud-water freezing tables over
   (r_c, t_Nc, -T, N_IN) (``freezeH2O``, module_mp_thompson.F:4697-4760),
   recomputed here in fp64 from the WRF closed forms (Bigg 1953 with the
   DeMott IN-count temperature adjustment).
4. Cloud gamma-distribution constant families ``cce``/``ccg``/``ocg1``/
   ``ocg2`` for nu_c = 1..15 (``thompson_init``, module_mp_thompson.F:672-685)
   plus the bin axes (Dc, dtc, t_Nc, r_c, Nt_IN) and index scalars.

FALSIFICATION built in: every axis this script *recomputes* from the WRF
closed forms (t_Nc, Dc, r_c, Nt_IN) is cross-checked against the values the
Fortran-extracted mp=8 asset carries (``thompson-tables-v1.npz``); any
mismatch beyond fp32 rounding aborts the build.  The CCN table is not
recomputed at all — it is the byte-identical WRF input data.

CPU-only; run pinned: taskset -c 0-3 python3 scripts/build_thompson_aero_tables.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MP8_ASSET = ROOT / "data" / "fixtures" / "thompson-tables-v1.npz"
DEFAULT_OUTPUT = ROOT / "data" / "fixtures" / "thompson-aero-tables-v1.npz"


def _ccn_bin_path() -> Path:
    """Pristine WRF ``run/CCN_ACTIVATE.BIN``, read VERBATIM as the CCN table input.

    Env-overridable via ``GPUWRF_WRF_ROOT`` (``config.paths.wrf_run_dir``) so a
    clean checkout never depends on a hardcoded ``/home/<name>`` path. Run this
    build script with ``PYTHONPATH=src`` so the ``gpuwrf`` package is importable.
    """

    from gpuwrf.config.paths import wrf_run_dir  # noqa: PLC0415 (deferred on purpose)

    return wrf_run_dir() / "CCN_ACTIVATE.BIN"


CCN_BIN = _ccn_bin_path()

# --- WRF module_mp_thompson.F constants (header block) -----------------------
PI = 3.1415926536
RHO_W = 1000.0
AM_R = PI * RHO_W / 6.0
BM_R = 3.0
BV_C = 2.0
D0C = 1.0e-6
NBC = 100  # nbins
NTB_C = 37
NTB_ARC, NTB_ARW, NTB_ART, NTB_ARR, NTB_ARK = 7, 9, 7, 5, 4
NTB_IN = 55
N_TC = 45  # 1..45 degC below freezing

# activ_ncloud axis vectors (module_mp_thompson.F:5187-5191).
TA_NA = np.array([10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0])
TA_WW = np.array([0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0])
TA_TK = np.array([243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15])


def _nint(x: np.ndarray | float) -> np.ndarray:
    """Fortran NINT: round half away from zero (inputs here are positive)."""

    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def _cloud_gammas() -> dict[str, np.ndarray]:
    """cce/ccg/ocg1/ocg2 for n = 1..15 (module_mp_thompson.F:672-685)."""

    cce = np.zeros((5, 15))
    ccg = np.zeros((5, 15))
    for n in range(1, 16):
        cce[0, n - 1] = n + 1.0
        cce[1, n - 1] = BM_R + n + 1.0
        cce[2, n - 1] = BM_R + n + 4.0
        cce[3, n - 1] = n + BV_C + 1.0
        cce[4, n - 1] = BM_R + n + BV_C + 1.0
        for row in range(5):
            ccg[row, n - 1] = math.gamma(cce[row, n - 1])
    return {
        "cce": cce,
        "ccg": ccg,
        "ocg1": 1.0 / ccg[0],
        "ocg2": 1.0 / ccg[1],
    }


def _bins() -> dict[str, np.ndarray]:
    """Cloud bins Dc/dtc, droplet-number bins t_Nc, and nic1/nic2 scalars."""

    dc = D0C + 1.0e-6 * np.arange(NBC, dtype=np.float64)  # Dc(1)=1e-6, +1e-6
    dtc = np.full(NBC, 1.0e-6, dtype=np.float64)  # dtc(1)=D0c=1e-6, rest 1e-6
    xdx = np.exp(np.arange(NBC + 1, dtype=np.float64) / NBC * np.log(3000.0 / 1.0) + np.log(1.0))
    t_nc = np.sqrt(xdx[:-1] * xdx[1:]) * 1.0e6
    nic1 = float(np.log(t_nc[-1] / t_nc[0]))
    r_c = np.concatenate([np.arange(1.0, 10.0) * 10.0**p for p in (-6, -5, -4, -3)] + [[1.0e-2]])
    assert r_c.shape == (NTB_C,)
    nic2 = int(_nint(np.log10(r_c[0])))
    nt_in = np.concatenate([np.arange(1.0, 10.0) * 10.0**p for p in (0, 1, 2, 3, 4, 5)] + [[1.0e6]])
    assert nt_in.shape == (NTB_IN,)
    return {"Dc": dc, "dtc": dtc, "t_Nc": t_nc, "r_c": r_c, "Nt_IN": nt_in,
            "nic1": np.float64(nic1), "nic2": np.int64(nic2)}


def _read_ccn_bin(path: Path) -> np.ndarray:
    """Read CCN_ACTIVATE.BIN (big-endian Fortran sequential, single record)."""

    raw = path.read_bytes()
    count = NTB_ARC * NTB_ARW * NTB_ART * NTB_ARR * NTB_ARK
    marker = int(np.frombuffer(raw[:4], dtype=">i4")[0])
    if marker != count * 4:
        raise RuntimeError(f"unexpected CCN_ACTIVATE.BIN record marker {marker} != {count * 4}")
    data = np.frombuffer(raw, dtype=">f4", count=count, offset=4)
    table = data.reshape((NTB_ARC, NTB_ARW, NTB_ART, NTB_ARR, NTB_ARK), order="F")
    if not (np.all(table >= 0.0) and np.all(table <= 1.0)):
        raise RuntimeError("CCN activation fractions outside [0, 1]")
    return np.ascontiguousarray(table.astype(np.float32))


def _drop_distribution(gam: dict[str, np.ndarray], bins: dict[str, np.ndarray]):
    """Per-(j: t_Nc, i: r_c) cloud gamma-distribution bin numbers N_c(n).

    Shared by table_dropEvap and the freezeH2O qcfz section; both build
    ``lamc``/``N0_c`` from (t_Nc(j), r_c(i)) with nu_c(j) and evaluate
    ``N_c(n) = N0_c * Dc(n)**nu_c * exp(-lamc*Dc(n)) * dtc(n)``.

    Returns ``n_c`` with shape (nbc_j, ntb_c_i, nbc_n) plus nu_c (j,).
    """

    t_nc = bins["t_Nc"]  # (100,)
    r_c = bins["r_c"]  # (37,)
    dc = bins["Dc"]  # (100,)
    dtc = bins["dtc"]  # (100,)
    nu_c = np.minimum(15, _nint(1000.0e6 / t_nc) + 2).astype(np.int64)  # (100,)
    ccg2 = gam["ccg"][1, nu_c - 1]  # (100,)
    ocg1 = gam["ocg1"][nu_c - 1]
    cce1 = gam["cce"][0, nu_c - 1]
    obmr = 1.0 / BM_R
    lamc = (t_nc[:, None] * AM_R * ccg2[:, None] * ocg1[:, None] / r_c[None, :]) ** obmr  # (j,i)
    n0_c = t_nc[:, None] * ocg1[:, None] * lamc ** cce1[:, None]  # (j,i)
    n_c = (
        n0_c[:, :, None]
        * dc[None, None, :] ** nu_c[:, None, None].astype(np.float64)
        * np.exp(-lamc[:, :, None] * dc[None, None, :])
        * dtc[None, None, :]
    )  # (j, i, n)
    return n_c, nu_c


def _table_drop_evap(gam, bins) -> tuple[np.ndarray, np.ndarray]:
    """tpc_wev/tnc_wev (nbc, ntb_c, nbc) per module_mp_thompson.F:5011-5047.

    WRF layout is (i: D-bin, j: r_c, k: t_Nc); partial sums run over bins
    1..i (ascending diameter).
    """

    n_c, _nu = _drop_distribution(gam, bins)  # (k=t_Nc, j=r_c, n)
    massc = AM_R * bins["Dc"] ** BM_R  # (n,)
    tnc = np.cumsum(n_c, axis=-1)  # partial number sums over n<=i
    tpc = np.cumsum(n_c * massc[None, None, :], axis=-1)
    # reorder (k, j, i) -> WRF (i, j, k)
    return (np.ascontiguousarray(np.transpose(tpc, (2, 1, 0))),
            np.ascontiguousarray(np.transpose(tnc, (2, 1, 0))))


def _freeze_h2o_qcfz(gam, bins) -> tuple[np.ndarray, np.ndarray]:
    """tpi_qcfz/tni_qcfz (ntb_c, nbc, 45, ntb_IN), module_mp_thompson.F:4697-4760.

    Bigg (1953) immersion freezing of the cloud gamma distribution, summed
    from the LARGEST bin down with WRF's early EXIT once the frozen mass
    reaches the available cloud mass r_c(i): the crossing bin is included,
    all smaller bins are excluded.
    """

    n_c, _nu = _drop_distribution(gam, bins)  # (j=t_Nc 100, i=r_c 37, n=100)
    dc = bins["Dc"]
    t_nc = bins["t_Nc"]
    r_c = bins["r_c"]
    nt_in = bins["Nt_IN"]
    massc = AM_R * dc**BM_R  # (n,)
    vol = massc / RHO_W

    t_adjust = np.maximum(-3.0, np.minimum(3.0 - np.log10(nt_in), 3.0))  # (m,)
    k_idx = np.arange(1, N_TC + 1, dtype=np.float64)  # (k,) == -T degC
    texp = np.exp(k_idx[None, :] - t_adjust[:, None]) - 1.0  # (m, k)

    tpi = np.zeros((NTB_C, NBC, N_TC, NTB_IN))
    tni = np.zeros((NTB_C, NBC, N_TC, NTB_IN))
    # Descending-bin views (n = nbc..1).
    n_c_desc = n_c[:, :, ::-1]  # (j, i, n_desc)
    mass_desc = massc[::-1]
    vol_desc = vol[::-1]
    for m in range(NTB_IN):
        prob = np.maximum(0.0, 1.0 - np.exp(-120.0 * vol_desc[None, :] * 5.2e-4 * texp[m][:, None]))  # (k, n_desc)
        num = prob[None, None, :, :] * n_c_desc[:, :, None, :]  # (j, i, k, n_desc)
        mass = num * mass_desc[None, None, None, :]
        cs_mass = np.cumsum(mass, axis=-1)
        cs_prev = cs_mass - mass  # cumulative BEFORE this bin
        include = cs_prev < r_c[None, :, None, None]  # crossing bin included
        sum_mass = np.sum(np.where(include, mass, 0.0), axis=-1)  # (j, i, k)
        sum_num = np.sum(np.where(include, num, 0.0), axis=-1)
        sum_num = np.minimum(sum_num, t_nc[:, None, None])
        tpi[:, :, :, m] = np.transpose(sum_mass, (1, 0, 2))
        tni[:, :, :, m] = np.transpose(sum_num, (1, 0, 2))
    return tpi, tni


def _cross_check(bins: dict[str, np.ndarray]) -> dict[str, float]:
    """Falsify the recomputed axes against the Fortran-extracted mp=8 asset."""

    with np.load(MP8_ASSET, allow_pickle=False) as mp8:
        checks = {
            "t_Nc": ("t_nc", bins["t_Nc"]),
            "Dc": ("dc", bins["Dc"]),
            "r_c": ("r_c", bins["r_c"]),
            "Nt_IN": ("nt_in", bins["Nt_IN"]),
        }
        report: dict[str, float] = {}
        for name, (asset_key, mine) in checks.items():
            ref = np.asarray(mp8[asset_key], dtype=np.float64)
            rel = float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-300)))
            report[name] = rel
            # The Fortran arrays pass through REAL*4 in places; allow fp32 dust.
            if rel > 1e-6:
                raise RuntimeError(f"axis cross-check FAILED for {name}: max rel diff {rel:.3e}")
    return report


def build(output: Path) -> dict[str, object]:
    gam = _cloud_gammas()
    bins = _bins()
    axis_report = _cross_check(bins)
    tnccn_act = _read_ccn_bin(CCN_BIN)
    tpc_wev, tnc_wev = _table_drop_evap(gam, bins)
    tpi_qcfz, tni_qcfz = _freeze_h2o_qcfz(gam, bins)

    payload = {
        "tnccn_act": tnccn_act,
        "ta_Na": TA_NA,
        "ta_Ww": TA_WW,
        "ta_Tk": TA_TK,
        "tpc_wev": tpc_wev,
        "tnc_wev": tnc_wev,
        "tpi_qcfz": tpi_qcfz,
        "tni_qcfz": tni_qcfz,
        "cce": gam["cce"],
        "ccg": gam["ccg"],
        "ocg1": gam["ocg1"],
        "ocg2": gam["ocg2"],
        "Dc": bins["Dc"],
        "dtc": bins["dtc"],
        "t_Nc": bins["t_Nc"],
        "r_c": bins["r_c"],
        "Nt_IN": bins["Nt_IN"],
        "nic1": np.asarray(bins["nic1"]),
        "nic2": np.asarray(bins["nic2"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    ccn_sha = hashlib.sha256(CCN_BIN.read_bytes()).hexdigest()
    return {
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "ccn_activate_bin": str(CCN_BIN),
        "ccn_activate_bin_sha256": ccn_sha,
        "mp8_asset": str(MP8_ASSET),
        "axis_cross_check_max_rel": axis_report,
        "tables": {k: list(np.asarray(v).shape) for k, v in payload.items()},
        "stats": {
            "tnccn_act": [float(tnccn_act.min()), float(tnccn_act.max())],
            "tnc_wev": [float(tnc_wev.min()), float(tnc_wev.max())],
            "tpi_qcfz": [float(tpi_qcfz.min()), float(tpi_qcfz.max())],
            "tni_qcfz": [float(tni_qcfz.min()), float(tni_qcfz.max())],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    record = build(args.output.resolve())
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
