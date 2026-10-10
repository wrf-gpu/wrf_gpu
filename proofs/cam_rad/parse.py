"""CAM01: parse cam01_output.bin (cam01_driver.F90 record layout) + cam01_input.bin into one npz fixture.

Arrays are per column (leading axis ncol).  WRF-order profiles (k = 1 bottom) come from camrad (REAL); CAM-order profiles
(k = 1 model top) come from the direct r8 radctl replay (prefix r8l_ / r8s_ / r8h_).
"""
import json
import sys

import numpy as np


class Reader:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0

    def take(self, dtype, n):
        dt = np.dtype(dtype)
        a = np.frombuffer(self.buf, dt, n, self.pos).copy()
        self.pos += dt.itemsize * n
        return a


def read_input(path):
    r = Reader(path)
    ncol, nz = r.take("<i4", 2)
    out = {}
    out["p_top"], out["gmt"], out["radt"], out["dt"] = r.take("<f4", 4)
    out["yr"] = r.take("<i4", 1)[0]
    out["znu"] = r.take("<f4", nz)
    out["julian"] = r.take("<f4", ncol)
    out["julday"] = r.take("<i4", ncol)
    for n in ("tsk", "emiss", "albedo", "xland", "xlat", "xlong", "snow", "xice", "tau"):
        out[n] = r.take("<f4", ncol)
    for n in ("t", "p", "pi", "rho", "z", "qv", "qc", "qr", "qi", "qs", "qg", "cldfra"):
        out[n] = r.take("<f4", ncol * nz).reshape(nz, ncol).T.copy()
    for n in ("p8w", "dz8w"):
        out[n] = r.take("<f4", ncol * (nz + 1)).reshape(nz + 1, ncol).T.copy()
    assert r.pos == len(r.buf)
    return int(ncol), int(nz), out


def read_lw_intermediates(path, ncol, nz):
    """cam01_lwint.bin: radclwmx's internal call chain on arm-L operands (CAM order, k = 1 model top)."""
    r = Reader(path)
    nzp = nz + 1
    acc = {}
    put = lambda n, v: acc.setdefault(f"lwi_{n}", []).append(v)
    for _ in range(ncol):
        put("pbr", r.take("<f8", nz)); put("pnm", r.take("<f8", nzp)); put("eccf", r.take("<f8", 1)[0])
        put("o3mmr", r.take("<f8", nz))
        for n in ("plco2", "plh2o", "tplnka", "s2c", "tcg", "w"):
            put(n, r.take("<f8", nzp))
        put("tplnke", r.take("<f8", 1)[0])
        for n in ("tint", "tint4", "tlayr", "tlayr4"):
            put(n, r.take("<f8", nzp))
        put("plh2ob", r.take("<f8", 2 * nzp).reshape(nzp, 2).T.copy())  # [band, k]
        put("wb", r.take("<f8", 2 * nzp).reshape(nzp, 2).T.copy())
        put("plol", r.take("<f8", nzp)); put("plos", r.take("<f8", nzp))
        for n in ("ucfc11", "ucfc12", "un2o0", "un2o1", "uch4", "uco211", "uco212", "uco213", "uco221", "uco222", "uco223",
                  "bn2o0", "bn2o1", "bch4", "uptype"):
            put(n, r.take("<f8", nzp))
        put("co2em", r.take("<f8", nzp)); put("co2eml", r.take("<f8", nz)); put("co2t", r.take("<f8", nzp))
        put("h2otr", r.take("<f8", nzp))
        put("abplnk1", r.take("<f8", 14 * nzp).reshape(nzp, 14).T.copy())  # [14, k]
        put("abplnk2", r.take("<f8", 14 * nzp).reshape(nzp, 14).T.copy())
        put("emstot", r.take("<f8", nzp))
        put("abstot", r.take("<f8", nzp * nzp).reshape(nzp, nzp).T.copy())  # [k1, k2]
        put("absnxt", r.take("<f8", nz * 4).reshape(4, nz).T.copy())  # [k, 4]
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {k: np.asarray(v) for k, v in acc.items()}


def read_sw_intermediates(path, ncol, nz):
    """cam01_swint.bin: aqsat esat/qsat, radctl rh expression, get_int_scales, get_aerosol AEROSOLt [k, species]."""
    r = Reader(path)
    acc = {}
    put = lambda n, v: acc.setdefault(f"swi_{n}", []).append(v)
    for _ in range(ncol):
        put("esat", r.take("<f8", nz)); put("qsat", r.take("<f8", nz)); put("rh", r.take("<f8", nz))
        put("scales", r.take("<f8", 13))
        put("aerosol", r.take("<f8", nz * 13).reshape(13, nz).T.copy())
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {k: np.asarray(v) for k, v in acc.items()}


def main(inp, outp, fixture, summary, lwint=None, swint=None):
    ncol, nz, inputs = read_input(inp)
    r = Reader(outp)
    hdr = r.take("<i4", 3)
    assert hdr[0] == ncol and hdr[1] == nz
    fix = {f"in_{k}": np.asarray(v) for k, v in inputs.items()}
    fix["mxaerl"] = np.asarray(hdr[2])
    fix["pin"] = r.take("<f4", 59)
    fix["m_hybi"] = r.take("<f4", 29)
    acc = {}

    def put(name, value):
        acc.setdefault(name, []).append(value)

    lw_scal = ("glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc", "lwdnb", "lwdnbc")
    sw_scal = ("gsw", "swcf", "coszr", "swddir", "swddni", "swddif", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc",
               "swdnb", "swdnbc")

    def r8_block(pfx, lw, sw):
        for n, v in zip(("co2vmr", "n2ovmr", "ch4vmr", "f11vmr", "f12vmr", "co2mmr", "lwups", "coszrs"), r.take("<f8", 8)):
            put(f"{pfx}{n}", v)
        for n in ("q1", "qliq", "qice", "cld", "pmid"):
            put(f"{pfx}{n}", r.take("<f8", nz))
        put(f"{pfx}pint", r.take("<f8", nz + 1))
        put(f"{pfx}t", r.take("<f8", nz))
        for n in ("cicewp", "cliqwp", "emis", "rel", "rei"):
            put(f"{pfx}{n}", r.take("<f8", nz))
        put(f"{pfx}pmxrgn", r.take("<f8", nz + 1))
        put(f"{pfx}nmxrgn", r.take("<i4", 1)[0])
        for n in ("o3vmr", "n2o", "ch4", "cfc11", "cfc12"):
            put(f"{pfx}{n}", r.take("<f8", nz))
        if lw:
            put(f"{pfx}qrl", r.take("<f8", nz)); put(f"{pfx}qrlcs", r.take("<f8", nz))
            for n in ("flup", "flupc", "fldn", "fldnc"):
                put(f"{pfx}{n}", r.take("<f8", nz + 1))
            for n, v in zip(("flwds", "flns", "flnt", "lwcftoa", "olrtoa"), r.take("<f8", 5)):
                put(f"{pfx}{n}", v)
            put(f"{pfx}abstot", r.take("<f8", (nz + 1) ** 2).reshape(nz + 1, nz + 1).T.copy())  # [k1, k2]
            put(f"{pfx}absnxt", r.take("<f8", nz * 4).reshape(4, nz).T.copy())  # [k, 4]
            put(f"{pfx}emstot", r.take("<f8", nz + 1))
        if sw:
            put(f"{pfx}qrs", r.take("<f8", nz)); put(f"{pfx}qrscs", r.take("<f8", nz))
            for n in ("fsup", "fsupc", "fsdn", "fsdnc", "fsdndir", "fsdndif"):
                put(f"{pfx}{n}", r.take("<f8", nz + 1))
            for n, v in zip(("fsns", "fsds", "fsdsdir", "fsdsdif", "swcftoa", "sols", "soll", "solsd", "solld"),
                            r.take("<f8", 9)):
                put(f"{pfx}{n}", v)
            put(f"{pfx}tauxcl", r.take("<f8", nz)); put(f"{pfx}tauxci", r.take("<f8", nz))

    for _ in range(ncol):
        put("l_rthratenlw", r.take("<f4", nz)); put("l_rthratenlwc", r.take("<f4", nz))
        for n, v in zip(lw_scal, r.take("<f4", 11)):
            put(f"l_{n}", v)
        put("l_abstot3", r.take("<f4", (nz + 1) ** 2).reshape(nz + 1, nz + 1).T.copy())
        put("l_absnxt3", r.take("<f4", nz * 4).reshape(4, nz).T.copy())
        put("l_emstot3", r.take("<f4", nz + 1))
        for n in ("cemiss", "taucldc", "taucldi"):
            put(f"l_{n}", r.take("<f4", nz))
        r8_block("r8l_", True, False)
        dec, sol, cosz = r.take("<f4", 3)
        put("declin", dec); put("solcon", sol); put("coszen", cosz)
        put("s_rthratensw", r.take("<f4", nz)); put("s_rthratenswc", r.take("<f4", nz))
        for n, v in zip(sw_scal, r.take("<f4", 14)):
            put(f"s_{n}", v)
        r8_block("r8s_", False, True)
        put("h_t", r.take("<f4", nz)); put("h_qv", r.take("<f4", nz)); put("h_tsk", r.take("<f4", 1)[0])
        put("h_rthratenlw", r.take("<f4", nz)); put("h_rthratenlwc", r.take("<f4", nz))
        for n, v in zip(lw_scal, r.take("<f4", 11)):
            put(f"h_{n}", v)
        r8_block("r8h_", True, False)
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    for k, v in acc.items():
        fix[k] = np.asarray(v)
    if lwint:
        fix.update(read_lw_intermediates(lwint, ncol, nz))
    if swint:
        fix.update(read_sw_intermediates(swint, ncol, nz))
    np.savez_compressed(fixture, **fix)
    lwz = int((np.abs(fix["l_rthratenlw"]).max(1) > 0).sum())
    day = int((fix["coszen"] > 0).sum())
    cloudy = int((fix["in_cldfra"].max(1) > 0).sum())
    multi = int((fix["r8l_nmxrgn"] > 1).sum())
    s = {"ncol": ncol, "nz": nz, "lw_nonzero_columns": lwz, "day_columns": day, "cloudy_columns": cloudy,
         "multi_region_columns": multi, "max_nmxrgn": int(fix["r8l_nmxrgn"].max()),
         "glw_range": [float(fix["l_glw"].min()), float(fix["l_glw"].max())],
         "olr_range": [float(fix["l_olr"].min()), float(fix["l_olr"].max())],
         "gsw_range": [float(fix["s_gsw"].min()), float(fix["s_gsw"].max())],
         "co2vmr": float(fix["r8l_co2vmr"][0]), "co2mmr": float(fix["r8l_co2mmr"][0]),
         "sha256_fixture": None}
    import hashlib
    s["sha256_fixture"] = hashlib.sha256(open(fixture, "rb").read()).hexdigest()
    open(summary, "w").write(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:7])
