"""o1-smag3d parity: pristine WRF REAL4 oracle vs JAX les3d_smagorinsky (fp32 and fp64).

Usage (CPU only):  JAX_PLATFORMS=cpu python proofs/o1_smag3d/parity.py [out.json]

For every case/config and every output (metrics, deformation, BN2, K, tendencies)
reports max|jax32 - wrf32| and the REAL4 floor max|jax64 - wrf32| (the same operator
evaluated in fp64 on the identical REAL4 inputs): parity means jax32 sits at WRF's
own REAL4 rounding level.  Also records a branch census (saturated BN2, floor/cap K).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io  # noqa: E402

from gpuwrf.dynamics.les3d_smagorinsky import HALO, Les3dConfig, Les3dInputs, les3d_smagorinsky_memory  # noqa: E402

WORK = Path("<USER_HOME>/wrf_gpu2_lanes/o1-smag3d/parity")
WRFOUT = oracle_io.SWISS_DIR / "wrfout_d01_2023-01-15_12:00:00"
CASES = {
    "A_real_cloudy": dict(i0=2, j0=16, ncx=22, ncy=18),
    "B_wind20_unstable": dict(i0=2, j0=16, ncx=22, ncy=18, wind_scale=20.0, theta_pert=0.05),
    "C_valley": dict(i0=12, j0=4, ncx=19, ncy=23, wind_scale=4.0),
}
CONFIGS = {
    "spec_h": dict(),
    "spec_vert_isfflx1": dict(vertical=True, isfflx=1),
    "spec_iso_vert_isfflx2": dict(vertical=True, isfflx=2, isotropic=1, tke_heat_flux=0.05,
                                  tke_drag_coefficient=0.0013),
    "spec_vert_isfflx0_thetad": dict(vertical=True, isfflx=0, use_theta_m=0, tke_drag_coefficient=0.002,
                                     tke_heat_flux=0.02, mix_full_fields=True),
    "periodic_vert": dict(periodic=True, vertical=True, isfflx=1),
}


def jax_inputs(case: dict, dtype) -> Les3dInputs:
    a = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
    return Les3dInputs(
        u=a(case["u"]), v=a(case["v"]), w=a(case["w"]), thp=a(case["thp"]), th_phy=a(case["th_phy"]),
        t_phy=a(case["t_phy"]), p_phy=a(case["p_phy"]), p8w=a(case["p8w"]), t8w=a(case["t8w"]),
        ph=a(case["ph"]), phb=a(case["phb"]), rho=a(case["rho"]), moist=tuple(a(q) for q in case["moist"]),
        msftx=a(case["msftx"]), msfty=a(case["msfty"]), msfux=a(case["msfux"]), msfuy=a(case["msfuy"]),
        msfvx=a(case["msfvx"]), msfvy=a(case["msfvy"]), fnm=a(case["fnm"]), fnp=a(case["fnp"]),
        dn=a(case["dn"]), dnw=a(case["dnw"]),
        cf1=np.float32(case["cf1"]), cf2=np.float32(case["cf2"]), cf3=np.float32(case["cf3"]),
        rdx=np.float32(case["rdx"]), rdy=np.float32(case["rdy"]), dx=np.float32(case["dx"]),
        dy=np.float32(case["dy"]), dt=np.float32(case["dt"]),
        ust=a(case["ust"]), hfx=a(case["hfx"]), qfx=a(case["qfx"]),
        u_base=a(case["u_base"]), v_base=a(case["v_base"]), t_base=a(case["t_base"]), qv_base=a(case["qv_base"]),
        tke=a(case["tke"]), mut=a(case["mut"]), c1h=a(case["c1h"]), c2h=a(case["c2h"]),
    )


def jax_config(cfg: dict) -> Les3dConfig:
    return Les3dConfig(
        boundary="periodic" if cfg.get("periodic") else "specified", isotropic=int(cfg.get("isotropic", 0)),
        c_s=0.25, mix_upper_bound=0.1, vertical=bool(cfg.get("vertical", False)), isfflx=int(cfg.get("isfflx", 1)),
        tke_drag_coefficient=float(cfg.get("tke_drag_coefficient", 0.0)),
        tke_heat_flux=float(cfg.get("tke_heat_flux", 0.0)), use_theta_m=int(cfg.get("use_theta_m", 1)),
        mix_full_fields=bool(cfg.get("mix_full_fields", False)), iqv=0, iqc=1, iqi=3,
        km_opt=int(cfg.get("km_opt", 3)),
    )


def compare(ora: dict, j32, j64) -> dict:
    rows = {}
    names = list(oracle_io.OUT_FIELDS) + [f"moist_tendf[{n}]" for n in range(len(ora["moist_tendf"]))]
    for name in names:
        if name.startswith("moist_tendf"):
            n = int(name[12:-1])
            o, a32, a64 = ora["moist_tendf"][n], j32.moist_tendf[n], j64.moist_tendf[n]
        else:
            o, a32, a64 = ora[name], getattr(j32, name), getattr(j64, name)
        o = oracle_io.crop_to_halo(o, HALO).astype(np.float64)
        a32 = np.asarray(a32, np.float64)
        a64 = np.asarray(a64, np.float64)
        scale = float(np.max(np.abs(o)))
        e32 = np.abs(a32 - o)
        e64 = np.abs(a64 - o)
        nz_o = int(np.count_nonzero(o))
        support = int(np.count_nonzero((o != 0) != (a32 != 0)))  # cells written by one side only
        rows[name] = dict(scale=scale, max_err32=float(e32.max()), max_err64=float(e64.max()),
                          rel32=float(e32.max() / scale) if scale else 0.0,
                          rel64=float(e64.max() / scale) if scale else 0.0,
                          bitwise=int(np.count_nonzero(e32 == 0)), n=int(o.size), nonzero=nz_o,
                          support_mismatch=support)
    return rows


def census(case: dict, ora: dict) -> dict:
    xkmh = oracle_io.crop_to_halo(ora["xkmh"], HALO)
    bn2 = oracle_io.crop_to_halo(ora["bn2"], HALO)
    return dict(
        bn2_negative=int(np.count_nonzero(bn2 < 0)), bn2_positive=int(np.count_nonzero(bn2 > 0)),
        qc_ge_qc_cr=int(np.count_nonzero(case["moist"][1] >= 1e-5)),
        qi_nonzero=int(np.count_nonzero(case["moist"][3] > 0)),
        xkmh_positive=int(np.count_nonzero(xkmh > 0)),
    )


def main(out_path: str | None = None) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    report = {}
    for cname, cspec in CASES.items():
        case = oracle_io.build_case(WRFOUT, **cspec)
        for gname, cfg in CONFIGS.items():
            ora = oracle_io.run_oracle(case, cfg, WORK, f"{cname}_{gname}")
            jc = jax_config(cfg)
            f32 = jax.jit(lambda inp: les3d_smagorinsky_memory(inp, jc))
            j32 = f32(jax_inputs(case, jnp.float32))
            j64 = f32(jax_inputs(case, jnp.float64))
            rows = compare(ora, j32, j64)
            report[f"{cname}/{gname}"] = dict(fields=rows, census=census(case, ora))
            worst = max(rows.items(), key=lambda kv: kv[1]["rel32"])
            print(f"{cname}/{gname}: worst rel32 {worst[0]} {worst[1]['rel32']:.3e} "
                  f"(rel64 floor {worst[1]['rel64']:.3e}) support_mismatch "
                  f"{sum(r['support_mismatch'] for r in rows.values())}", flush=True)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
