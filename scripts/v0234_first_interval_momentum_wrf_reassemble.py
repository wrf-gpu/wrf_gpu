"""Reassemble WRFGPU2_MOMSP per-rank dumps into full-domain d03 arrays.

The instrumented WRF run (isolated tree, output-neutrality proven by
byte-identical wrfout) writes per-rank full-memory f64 big-endian stream
dumps plus one meta.txt per rank. This module reassembles global d03
arrays for the four momentum operator savepoints and validates the
reassembly against the retained wrfout truth frames.

Layout rules (WRF v4.7.1 d03, ids=1..ide=112, jds=1..jde=94, kds=1..kde=45):
  mass fields: global[k=1..44, j=1..93, i=1..111]; rank owns [ips:min(ipe,111)]x[jps:min(jpe,93)]
  u fields:    global[k=1..44, j=1..93, i=1..112]; rank owns [ips:ipe]x[jps:min(jpe,93)]
  v fields:    global[k=1..44, j=1..94, i=1..111]; rank owns [ips:min(ipe,111)]x[jps:jpe]
Python arrays are stored (k, j, i) C-order to match gpuwrf's (nz, ny, nx) layout.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

DUMP_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/momsp_dumps")
GLOBAL_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/momsp_global")

KDS, KDE = 1, 45
IDS, IDE = 1, 112
JDS, JDE = 1, 94
NK = KDE - 1  # 44 mass levels

FIELD_STAGGER = {
    "u": "u", "v": "v",
    "sp1_entry__u": "u", "sp1_entry__v": "v",
    "sp4_exit__u": "u", "sp4_exit__v": "v",
    "sp3_tendf__ru_tendf": "u", "sp3_tendf__rv_tendf": "v",
    "sp2_pbl__rublten": "mass", "sp2_pbl__rvblten": "mass",
    "sp2_pbl__mut": "mass", "sp5_sfc__u10": "mass", "sp5_sfc__v10": "mass",
    "l1_rk1_tend__ru_tend": "u", "l1_rk1_tend__rv_tend": "v",
    "l1_rk1_tend__u_save": "u", "l1_rk1_tend__v_save": "v",
    "l2_rk1_fin__u": "u", "l2_rk1_fin__v": "v",
    "l3_rk2_fin__u": "u", "l3_rk2_fin__v": "v",
    "l4_rk3_fin__u": "u", "l4_rk3_fin__v": "v",
    "l5_prebdry__u": "u", "l5_prebdry__v": "v",
}


def read_meta(rank_dir: Path) -> dict[str, list[int]]:
    meta: dict[str, list[int]] = {}
    for line in (rank_dir / "meta.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, rest = line.partition(" ")
        if key == "rank":
            meta["rank"] = [int(rest.strip())]
        else:
            meta[key] = [int(tok) for tok in rest.split()]
    return meta


def load_ranks(root: Path = DUMP_ROOT) -> list[dict[str, list[int]]]:
    ranks = []
    for rank_dir in sorted(root.glob("rank*")):
        meta = read_meta(rank_dir)
        meta["dir"] = rank_dir
        ranks.append(meta)
    assert ranks, f"no rank dirs under {root}"
    return ranks


def _read_dump(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, dtype=">f8")
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {data.size}")
    return data.reshape(shape).astype(np.float64)


def reassemble3d(tag_field: str, step: int, ranks: list[dict] | None = None) -> np.ndarray:
    """Reassemble one 3D savepoint into the global (k, j, i) array."""
    stagger = FIELD_STAGGER[tag_field]
    ranks = ranks if ranks is not None else load_ranks()
    if stagger == "u":
        out = np.full((NK, JDE - 1, IDE), np.nan)
    elif stagger == "v":
        out = np.full((NK, JDE, IDE - 1), np.nan)
    else:
        out = np.full((NK, JDE - 1, IDE - 1), np.nan)
    covered = np.zeros(out.shape[1:], dtype=bool)
    for meta in ranks:
        ims, ime, jms, jme, kms, kme = meta["ims_ime_jms_jme_kms_kme"]
        ips, ipe, jps, jpe, kps, kpe = meta["ips_ipe_jps_jpe_kps_kpe"]
        path = meta["dir"] / f"step{step:06d}_{tag_field}.f64"
        if not path.is_file():
            raise FileNotFoundError(path)
        mem = _read_dump(path, (jme - jms + 1, kme - kms + 1, ime - ims + 1))
        if stagger == "u":
            gi0, gi1 = ips, ipe          # patch already reaches ide=112 on east rank
            gj0, gj1 = jps, min(jpe, JDE - 1)
        elif stagger == "v":
            gi0, gi1 = ips, min(ipe, IDE - 1)
            gj0, gj1 = jps, jpe
        else:
            gi0, gi1 = ips, min(ipe, IDE - 1)
            gj0, gj1 = jps, min(jpe, JDE - 1)
        mi0, mi1 = gi0 - ims, gi1 - ims + 1
        mj0, mj1 = gj0 - jms, gj1 - jms + 1
        mk0, mk1 = KDS - kms, (KDE - 1) - kms + 1
        block = mem[mj0:mj1, mk0:mk1, mi0:mi1].transpose(1, 0, 2)  # (k, j, i)
        if covered[gj0 - 1 : gj1, gi0 - 1 : gi1].any():
            raise ValueError(f"double-covered cells by {meta['dir']} for {tag_field}")
        out[:, gj0 - 1 : gj1, gi0 - 1 : gi1] = block
        covered[gj0 - 1 : gj1, gi0 - 1 : gi1] = True
    if not covered.all():
        raise ValueError(f"uncovered cells for {tag_field} step {step}")
    if np.isnan(out).any():
        raise ValueError(f"NaN holes after reassembly of {tag_field} step {step}")
    return out


def reassemble2d(tag_field: str, step: int, ranks: list[dict] | None = None) -> np.ndarray:
    """Reassemble one 2D savepoint (mass points) into the global (j, i) array."""
    ranks = ranks if ranks is not None else load_ranks()
    out = np.full((JDE - 1, IDE - 1), np.nan)
    covered = np.zeros(out.shape, dtype=bool)
    for meta in ranks:
        ims, ime, jms, jme = meta["ims_ime_jms_jme_kms_kme"][:4]
        ips, ipe, jps, jpe = meta["ips_ipe_jps_jpe_kps_kpe"][:4]
        path = meta["dir"] / f"step{step:06d}_{tag_field}.f64"
        if not path.is_file():
            raise FileNotFoundError(path)
        mem = _read_dump(path, (jme - jms + 1, ime - ims + 1))
        gi0, gi1 = ips, min(ipe, IDE - 1)
        gj0, gj1 = jps, min(jpe, JDE - 1)
        block = mem[gj0 - jms : gj1 - jms + 1, gi0 - ims : gi1 - ims + 1]
        if covered[gj0 - 1 : gj1, gi0 - 1 : gi1].any():
            raise ValueError(f"double-covered cells by {meta['dir']} for {tag_field}")
        out[gj0 - 1 : gj1, gi0 - 1 : gi1] = block
        covered[gj0 - 1 : gj1, gi0 - 1 : gi1] = True
    if not covered.all() or np.isnan(out).any():
        raise ValueError(f"coverage/NaN failure for {tag_field} step {step}")
    return out


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    import netCDF4  # noqa: F401  (validated below by direct import use)

    ranks = load_ranks()
    print(f"ranks: {len(ranks)}")
    case = Path(
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf"
    )
    from netCDF4 import Dataset

    # --- Validation 1: SP1(itimestep=1) vs retained wrfout 00:00 U/V ---
    truth = Dataset(case / "wrfout_d03_2025-03-01_00:00:00")
    u_wrf = np.asarray(truth.variables["U"][0])   # (k, j, i_stag=112)
    v_wrf = np.asarray(truth.variables["V"][0])   # (k, j_stag=94, i)
    u_sp1 = reassemble3d("sp1_entry__u", 1, ranks)
    v_sp1 = reassemble3d("sp1_entry__v", 1, ranks)
    du = np.abs(u_sp1 - u_wrf)
    dv = np.abs(v_sp1 - v_wrf)
    print(f"SP1(step1) u vs wrfout00 U: maxabs={du.max():.3e} meanabs={du.mean():.3e}")
    print(f"SP1(step1) v vs wrfout00 V: maxabs={dv.max():.3e} meanabs={dv.mean():.3e}")

    # --- Validation 2: SP4(N) vs SP1(N+1) interior consistency ---
    for step in (1, 50, 100, 199):
        u4 = reassemble3d("sp4_exit__u", step, ranks)
        u1 = reassemble3d("sp1_entry__u", step + 1, ranks)
        d = np.abs(u4 - u1)
        interior = d[:, 5:-5, 5:-5]
        band = np.concatenate([d[:, :5, :].ravel(), d[:, -5:, :].ravel(),
                               d[:, :, :5].ravel(), d[:, :, -5:].ravel()])
        print(
            f"SP4({step}) vs SP1({step+1}) u: interior maxabs={interior.max():.3e} "
            f"boundary-band maxabs={band.max():.3e}"
        )
    print("reassemble validation complete")


if __name__ == "__main__":
    main()
