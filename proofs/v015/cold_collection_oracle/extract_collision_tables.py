"""Extract the WRF Thompson cold-collection lookup tables to a JAX fixture.

Reads the bit-exact Fortran-computed .dat tables (gfortran unformatted
sequential, big-endian, 4-byte record markers) produced by the pristine WRF
mp_gt_driver init (qr_acr_qs / qr_acr_qg / freezeH2O) and writes them as a
compressed .npz the JAX cold-collection lane loads.  No values are recomputed
-- these are the exact Fortran table contents.

qr_acr_qs records (each Fortran-order (ntb_s, ntb_t, ntb_r1, ntb_r)):
  tcs_racs1 tmr_racs1 tcs_racs2 tmr_racs2 tcr_sacr1 tms_sacr1
  tcr_sacr2 tms_sacr2 tnr_racs1 tnr_racs2 tnr_sacr1 tnr_sacr2
qr_acr_qg records (each Fortran-order (ntb_g1, ntb_g, dimNRHG=1, ntb_r1, ntb_r)):
  tcg_racg tmr_racg tcr_gacr tnr_racg tnr_gacr
freezeH2O records (each Fortran-order (ntb_r, ntb_r1, ntb_tc=45, ntb_IN=55)):
  tpi_qrfz tni_qrfz tpg_qrfz tnr_qrfz tpi_qcfz tni_qcfz
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

NTB_R = 37
NTB_R1 = 37
NTB_S = 37
NTB_T = 9
NTB_G = 37
NTB_G1 = 37
NTB_C = 37
NBC = 100  # cloud-droplet bins for qcfz (module_mp_thompson.F:630 tpi_qcfz(ntb_c,nbc,45,ntb_IN))
NTB_TC = 45
NTB_IN = 55
IDX_BG1 = 5  # WRF mp8 fixed graupel-density index, even with dimNRHG=1.

DAT_DIR = Path("<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run")
OUT = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "thompson-cold-collection-v1.npz"


def read_records(path: Path) -> list[np.ndarray]:
    """Reads gfortran big-endian, 4-byte-marker unformatted sequential records."""
    data = path.read_bytes()
    recs, pos = [], 0
    while pos < len(data):
        hdr = struct.unpack(">i", data[pos:pos + 4])[0]
        body = np.frombuffer(data[pos + 4:pos + 4 + hdr], dtype=">f8")
        ftr = struct.unpack(">i", data[pos + 4 + hdr:pos + 8 + hdr])[0]
        if hdr != ftr:
            raise RuntimeError(f"record marker mismatch {hdr}!={ftr} in {path}")
        recs.append(np.asarray(body, dtype=np.float64))
        pos += 8 + hdr
    return recs


def extract_qg_mp8_plane(rec: np.ndarray) -> np.ndarray:
    """Return the effective mp8 qr_acr_qg table read by WRF.

    In non-hail Thompson mp8, WRF allocates ``qr_acr_qg`` with ``dimNRHG=1`` but
    later indexes it with ``idx_bg1=5`` (module_mp_thompson.F:2527-2532).  With
    bounds checking off this is a deterministic column-major offset into the
    following rain-intercept planes.  Store that effective 4-D view so the JAX
    runtime remains a normal ``(g1,g,r1,r)`` gather while matching pristine WRF.
    """

    eff = np.zeros(NTB_G1 * NTB_G * NTB_R1 * NTB_R, dtype=np.float64)
    for m in range(NTB_R):
        for k in range(NTB_R1):
            for j in range(NTB_G):
                for i in range(NTB_G1):
                    out_off = i + NTB_G1 * (j + NTB_G * (k + NTB_R1 * m))
                    src_off = i + NTB_G1 * (
                        j + NTB_G * ((IDX_BG1 - 1) + k + NTB_R1 * m)
                    )
                    if src_off < rec.size:
                        eff[out_off] = rec[src_off]
    return eff.reshape((NTB_G1, NTB_G, NTB_R1, NTB_R), order="F")


def main():
    qs = read_records(DAT_DIR / "qr_acr_qsV2.dat")
    qg = read_records(DAT_DIR / "qr_acr_qg_V4.dat")

    assert len(qs) == 12, len(qs)
    assert len(qg) == 5, len(qg)

    qs_names = ["tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
                "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
                "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"]
    qg_names = ["tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"]

    # The cold (twet<T_0) rcs/rcg lane reads all 12 qr_acr_qs records and 4 of
    # the 5 qr_acr_qg records.  ``tcg_racg`` is only used by the WARM rcg branch
    # (rain melting graupel below the melting line), while the full Bigg
    # rain-freezing qrfz tables are already in thompson-tables-v1.npz and
    # consumed by the existing freeze block.  The default-IN/default-cloud-number
    # qcfz cloud-water-freezing planes are tiny, WRF-active in v0.18, and live
    # in the same freezeH2O.dat source, so keep those reduced planes here with
    # the cold lane.
    KEEP = set(qs_names) | {"tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"}

    out = {}

    # qr_acr_qs: Fortran (ntb_s, ntb_t, ntb_r1, ntb_r). Store C-order
    # (ntb_s, ntb_t, ntb_r1, ntb_r) so JAX gather indexing matches WRF.
    for name, rec in zip(qs_names, qs):
        if name not in KEEP:
            continue
        arr = rec.reshape((NTB_S, NTB_T, NTB_R1, NTB_R), order="F")
        out[name] = np.ascontiguousarray(arr)

    # qr_acr_qg: Fortran allocation is (ntb_g1, ntb_g, dimNRHG=1, ntb_r1, ntb_r),
    # but mp8 reads it with idx_bg1=5.  Extract the effective out-of-bounds plane
    # exactly as pristine WRF reads it, then store C-order (g1,g,r1,r).
    for name, rec in zip(qg_names, qg):
        if name not in KEEP:
            continue
        arr = extract_qg_mp8_plane(rec)
        out[name] = np.ascontiguousarray(arr)  # (ntb_g1, ntb_g, ntb_r1, ntb_r)

    fz = read_records(DAT_DIR / "freezeH2O.dat")
    assert len(fz) == 6, len(fz)
    default_in_index = 27  # WRF default 1 L^-1 IN -> idx_IN=28, stored 0-based.
    default_nc_index = 58  # WRF non-aerosol Nt_c=100e6 m^-3 -> idx_n=59, 0-based.
    for name, rec in zip(("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz"), fz):
        if name not in {"tpi_qcfz", "tni_qcfz"}:
            continue
        arr = rec.reshape((NTB_C, NBC, NTB_TC, NTB_IN), order="F")
        out[name] = np.ascontiguousarray(arr[:, default_nc_index, :, default_in_index])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} min={v.min():.3e} max={v.max():.3e}")


if __name__ == "__main__":
    main()
