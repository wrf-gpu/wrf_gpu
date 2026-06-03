"""Convert column_d03_12z.json into the flat key=value file the Fortran oracle reads."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "column_d03_12z.json"
DST = HERE / "fortran_oracle" / "column_d03_12z.flat"

with open(SRC) as f:
    c = json.load(f)

prof = c["profiles"]
surf = c["surface"]
cfg = c["config"]
nz = c["meta"]["nz"]


def arr(name, a):
    return f"{name}=" + ",".join(f"{x:.9e}" for x in a)


lines = [
    f"nz={nz}",
    f"delt={cfg['delt']}",
    f"dx={surf['dx']}",
    f"ust={surf['ust']:.9e}",
    f"pblh={surf['pblh']:.9e}",
    f"psfc={surf['psfc']:.9e}",
    f"wspd={surf['wspd']:.9e}",
    f"xland={surf['xland']:.9e}",
    f"ts={surf['tsk']:.9e}",
    f"ps={surf['psfc']:.9e}",
    f"flt={surf['flt']:.9e}",
    f"flq={surf['flq']:.9e}",
    f"flqv={surf['flqv']:.9e}",
    f"fltv={surf['fltv']:.9e}",
    f"th_sfc={surf['th_sfc']:.9e}",
    f"e_edmf={cfg['bl_mynn_edmf']}",
    f"e_mom={cfg['bl_mynn_edmf_mom']}",
    f"e_tke={cfg['bl_mynn_edmf_tke']}",
    f"e_mixs={cfg['bl_mynn_mixscalars']}",
    f"e_cmix={cfg['bl_mynn_cloudmix']}",
    f"e_mixqt={cfg['bl_mynn_mixqt']}",
    arr("u", prof["u"]),
    arr("v", prof["v"]),
    arr("w", prof["w"]),
    arr("th", prof["th"]),
    arr("tk", prof["tk"]),
    arr("p", prof["p"]),
    arr("exner", prof["exner"]),
    arr("rho", prof["rho"]),
    arr("dz", prof["dz"]),
    arr("qv", prof["qv"]),
    arr("qc", prof["qc"]),
    arr("qi", prof["qi"]),
    arr("qke", prof["qke"]),
]

with open(DST, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {DST} ({nz} levels)")
