"""Tables for GPUWRF_THOMPSON_MIXED_PHASE_WRF: tcg_racg (warm rain-graupel branch, module_mp_thompson.F:2537) and
the cloud-droplet freezing planes tpi_qcfz/tni_qcfz at WRF's idx_n.

CPU-WRF reads qr_acr_qg_V4.dat ('ThompMP: read qr_acr_qg_V4.dat instead of computing', rsl.out.0000); the file is
byte-identical (sha256 441bd836...) in the WN3 0227/0502 and PROD D5 run directories and in a serial pristine table
run. mp8 allocates these tables with dimNRHG = NRHG1 = 1 (:465) but reads them at idx_bg(k) = idx_bg1 = 5 (:79,
:1896): out of bounds by S = (idx_bg1-1)*ntb_g1*ntb_g elements in Fortran memory order. This script proves that the
committed thompson-cold-collection-v1.npz holds exactly that effective read for tmr_racg/tcr_gacr/tnr_racg/tnr_gacr
(v1[:-S] == dat[S:], past-the-end tail = 0), then writes thompson-mixed-phase-v1.npz = tcg_racg as WRF writes it; the
loader applies the same offset. usage: extract_mixed_phase_tables.py <qr_acr_qg_V4.dat> [out.npz]
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "data" / "fixtures" / "thompson-cold-collection-v1.npz"
OUT = ROOT / "data" / "fixtures" / "thompson-mixed-phase-v1.npz"
NAMES = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")   # WRF WRITE order (qr_acr_qg)
N = 37
S = (5 - 1) * N * N                                                   # (idx_bg1 - 1) * ntb_g1 * ntb_g


def read_records(path: Path) -> list[np.ndarray]:
    """gfortran unformatted sequential, big-endian, 4-byte markers (WRF -fconvert=big-endian)."""
    data, out, pos = path.read_bytes(), [], 0
    while pos < len(data):
        n = struct.unpack(">i", data[pos:pos + 4])[0]
        if struct.unpack(">i", data[pos + 4 + n:pos + 8 + n])[0] != n:
            raise RuntimeError(f"record marker mismatch at {pos}")
        out.append(np.frombuffer(data[pos + 4:pos + 4 + n], ">f8").astype(np.float64))
        pos += 8 + n
    return out


def main(argv: list[str]) -> int:
    dat, frz = Path(argv[1]), Path(argv[2])
    out = Path(argv[3]) if len(argv) > 3 else OUT
    recs = read_records(dat)
    assert len(recs) == 5 and all(r.size == N ** 4 for r in recs), [r.size for r in recs]   # dimNRHG = 1 (mp8)
    ref = dict(zip(NAMES, recs))                                     # Fortran order (g1, g, 1, r1, r) flattened
    with np.load(V1) as v1:
        for name in NAMES[1:]:
            old = v1[name].reshape(-1, order="F")                    # stored C-contiguous (g1, g, r1, r)
            assert np.array_equal(old[:-S], ref[name][S:]) and not old[-S:].any(), name
    nbc = 100
    xdx = [1.0] + [math.exp((n - 1) / nbc * math.log(3000.0)) for n in range(2, nbc + 1)] + [3000.0]
    t_nc = [math.sqrt(xdx[n] * xdx[n + 1]) * 1.0e6 for n in range(nbc)]
    nic1_real = math.log(t_nc[-1] / t_nc[0])
    idx_n = {name: int(math.floor(1.0 + nbc * math.log(100.0e6 / t_nc[0]) / nic1 + 0.5))
             for name, nic1 in (("wrf_integer_nic1", float(int(nic1_real))), ("real_nic1", nic1_real))}
    frecs = read_records(frz)
    assert [r.size for r in frecs[4:]] == [N * 100 * 45 * 55] * 2
    qcfz = {name: r.reshape((N, 100, 45, 55), order="F") for name, r in zip(("tpi_qcfz", "tni_qcfz"), frecs[4:])}
    with np.load(V1) as v1:
        for name in qcfz:
            assert np.array_equal(v1[name], qcfz[name][:, idx_n["real_nic1"] - 1, :, 27]), name
    planes = {name: np.ascontiguousarray(t[:, idx_n["wrf_integer_nic1"] - 1, :, 27]) for name, t in qcfz.items()}
    np.savez_compressed(out, tcg_racg=ref["tcg_racg"].reshape((N, N, N, N), order="F"), **planes)
    receipt = {"dat": str(dat), "dat_sha256": hashlib.sha256(dat.read_bytes()).hexdigest(),
               "asset": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
               "asset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "idx_bg1_offset_elements": S,
               "v1_equals_wrf_idx_bg1_read": list(NAMES[1:]), "freeze_dat": str(frz),
               "freeze_dat_sha256": hashlib.sha256(frz.read_bytes()).hexdigest(), "qcfz_idx_n_1based": idx_n,
               "table_sha256": {n: hashlib.sha256(ref[n].tobytes()).hexdigest() for n in NAMES}}
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
