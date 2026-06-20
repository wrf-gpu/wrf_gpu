"""V0.14 PSFC moist pressure-state closure: CPU-only budget + offline ablation.

Sprint: .agent/sprints/2026-06-10-v014-fable-psfc-moist-pressure-closure/

WRF source anchor (pristine WRFv4, <USER_HOME>/src/wrf_pristine/WRF):
  - phys/module_surface_driver.F:1988          PSFC(I,J) = p8w(I,kts,J)
  - dyn_em/module_first_rk_step_part1.F:1400   surface driver gets P8W=grid%p_hyd_w
  - dyn_em/module_big_step_utilities_em.F:4946-4958 (phy_prep):
        p_hyd_w(i,kte,j) = p_top
        qtot = sum over ALL moist species of moist(i,k,j,n)
        p_hyd_w(i,k,j) = p_hyd_w(i,k+1,j)
                         - (1.+qtot)*(c1(k)*MUT(i,j)+c2(k))*dnw(k)
  i.e. the runtime PSFC that CPU WRF writes is the MOIST HYDROSTATIC surface
  pressure integrated over the full column in the hybrid dry-mass coordinate,
  NOT a height extrapolation of the nonhydrostatic total pressure P+PB.

This proof, for each paired lead (h1/h4/h10/latest):
  1. reproduces the GPT h1 budget numbers directly from NetCDF,
  2. proves the WRF formula on the CPU side (CPU PSFC vs CPU p_hyd_w(kts)
     recomputed from CPU MU/MUB/C1H/C2H/DNW/moist fields),
  3. ablation: applies the same WRF p_hyd_w(kts) formula to the GPU fields and
     scores it against CPU PSFC = the expected post-fix PSFC RMSE/bias,
  4. characterizes the 3D pressure state: is GPU P+PB at the lowest mass level
     consistent with its own DRY or MOIST hydrostatic column?

CPU-only; reads wrfout files only; writes psfc_moist_pressure_state_closure.json.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python proofs/v014/psfc_moist_pressure_state_closure.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RUN_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z"
)
GPU_DIR = RUN_ROOT / "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z"
CPU_DIR = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260501_18z_l2_72h_20260519T173026Z"
)
DOMAIN = "d02"
INIT_HOUR = 18  # 2026-05-01 18z
MOIST_ALL = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
OUT_JSON = Path(__file__).with_suffix(".json")


def _lead_path(base: Path, lead_h: int) -> Path:
    from datetime import datetime, timedelta

    t = datetime(2026, 5, 1, INIT_HOUR) + timedelta(hours=lead_h)
    return base / f"wrfout_{DOMAIN}_{t:%Y-%m-%d_%H:%M:%S}"


def _stats(diff: np.ndarray) -> dict:
    d = np.asarray(diff, dtype=np.float64).ravel()
    return {
        "mean": float(d.mean()),
        "rmse": float(np.sqrt((d * d).mean())),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        "max_abs": float(np.abs(d).max()),
    }


def _load(path: Path) -> dict:
    with Dataset(path) as nc:
        out = {
            "PSFC": np.array(nc["PSFC"][0], dtype=np.float64),
            "MU": np.array(nc["MU"][0], dtype=np.float64),
            "MUB": np.array(nc["MUB"][0], dtype=np.float64),
            "P": np.array(nc["P"][0], dtype=np.float64),
            "PB": np.array(nc["PB"][0], dtype=np.float64),
            "PH": np.array(nc["PH"][0], dtype=np.float64),
            "PHB": np.array(nc["PHB"][0], dtype=np.float64),
            "P_TOP": float(np.asarray(nc["P_TOP"][:]).ravel()[0]),
            "C1H": np.array(nc["C1H"][0], dtype=np.float64),
            "C2H": np.array(nc["C2H"][0], dtype=np.float64),
            "DNW": np.array(nc["DNW"][0], dtype=np.float64),
        }
        for q in MOIST_ALL:
            out[q] = (
                np.array(nc[q][0], dtype=np.float64)
                if q in nc.variables
                else np.zeros_like(out["P"])
            )
    return out


def _dp_dry(d: dict) -> np.ndarray:
    """Per-layer dry-air mass (Pa, positive): (c1h*MUT + c2h) * (-dnw)."""
    mut = d["MU"] + d["MUB"]
    return (
        d["C1H"][:, None, None] * mut[None, :, :] + d["C2H"][:, None, None]
    ) * (-d["DNW"][:, None, None])


def _p_hyd_w_sfc(d: dict, species: tuple[str, ...]) -> np.ndarray:
    """WRF phy_prep p_hyd_w at the bottom full level (= runtime PSFC)."""
    qtot = np.zeros_like(d["P"])
    for q in species:
        qtot = qtot + d[q]
    return d["P_TOP"] + ((1.0 + qtot) * _dp_dry(d)).sum(axis=0)


def _p_hyd_half(d: dict, species: tuple[str, ...], k: int) -> np.ndarray:
    """WRF phy_prep p_hyd at half level k = 0.5*(p_hyd_w(k)+p_hyd_w(k+1))."""
    qtot = np.zeros_like(d["P"])
    for q in species:
        qtot = qtot + d[q]
    layer = (1.0 + qtot) * _dp_dry(d)
    # p_hyd_w from the top: p_hyd_w[nz] = p_top (w-level index nz = model top)
    nz = d["P"].shape[0]
    p_w = np.empty((nz + 1,) + d["P"].shape[1:], dtype=np.float64)
    p_w[nz] = d["P_TOP"]
    for kk in range(nz - 1, -1, -1):
        p_w[kk] = p_w[kk + 1] + layer[kk]
    return 0.5 * (p_w[k] + p_w[k + 1])


def _p_extrap_sfc(d: dict) -> np.ndarray:
    """GPT/GPU-writer style: total P extrapolated in height to the ground."""
    p_tot = d["P"] + d["PB"]
    phi = d["PH"] + d["PHB"]
    phi0 = phi[0]
    phi1 = 0.5 * (phi[0] + phi[1])
    phi2 = 0.5 * (phi[1] + phi[2])
    w1 = (phi0 - phi2) / (phi1 - phi2)
    return w1 * p_tot[0] + (1.0 - w1) * p_tot[1]


def analyse_lead(lead_h: int) -> dict:
    cpu = _load(_lead_path(CPU_DIR, lead_h))
    gpu = _load(_lead_path(GPU_DIR, lead_h))

    res: dict = {"lead_h": lead_h}

    # static coordinate identity between the two runs
    res["coord_identity"] = {
        "c1h_max_abs_diff": float(np.abs(cpu["C1H"] - gpu["C1H"]).max()),
        "c2h_max_abs_diff": float(np.abs(cpu["C2H"] - gpu["C2H"]).max()),
        "dnw_max_abs_diff": float(np.abs(cpu["DNW"] - gpu["DNW"]).max()),
        "p_top_cpu": cpu["P_TOP"],
        "p_top_gpu": gpu["P_TOP"],
        "dry_integral_identity_cpu": _stats(
            _dp_dry(cpu).sum(axis=0) - (cpu["MU"] + cpu["MUB"] - cpu["P_TOP"])
        ),
    }

    # 1. budget (reproduces GPT numbers)
    for name, fn in (
        ("PSFC", lambda d: d["PSFC"]),
        ("MU", lambda d: d["MU"]),
        ("MUB", lambda d: d["MUB"]),
        ("dry_col", lambda d: d["P_TOP"] + d["MU"] + d["MUB"]),
        ("vapor_proxy", lambda d: d["PSFC"] - (d["P_TOP"] + d["MU"] + d["MUB"])),
        ("qv_load", lambda d: (d["QVAPOR"] * _dp_dry(d)).sum(axis=0)),
        ("qtot_load", lambda d: _p_hyd_w_sfc(d, MOIST_ALL)
         - (d["P_TOP"] + d["MU"] + d["MUB"])),
        ("p_extrap_sfc", _p_extrap_sfc),
    ):
        c, g = fn(cpu), fn(gpu)
        res[name] = {
            "cpu_mean": float(np.mean(c)),
            "gpu_mean": float(np.mean(g)),
            "diff": _stats(g - c),
        }

    # 2. WRF formula proof on CPU truth: PSFC == p_hyd_w(kts)?
    res["cpu_formula_proof"] = {
        "psfc_minus_p_hyd_w_all_species": _stats(
            cpu["PSFC"] - _p_hyd_w_sfc(cpu, MOIST_ALL)
        ),
        "psfc_minus_p_hyd_w_qv_only": _stats(
            cpu["PSFC"] - _p_hyd_w_sfc(cpu, ("QVAPOR",))
        ),
        "psfc_minus_p_extrap": _stats(cpu["PSFC"] - _p_extrap_sfc(cpu)),
    }

    # 3. ablation: WRF p_hyd_w(kts) on GPU fields vs CPU PSFC (= post-fix score)
    gpu_fix_all = _p_hyd_w_sfc(gpu, MOIST_ALL)
    gpu_fix_qv = _p_hyd_w_sfc(gpu, ("QVAPOR",))
    res["ablation_post_fix"] = {
        "gpu_p_hyd_w_all_vs_cpu_psfc": _stats(gpu_fix_all - cpu["PSFC"]),
        "gpu_p_hyd_w_qv_vs_cpu_psfc": _stats(gpu_fix_qv - cpu["PSFC"]),
        "current_gpu_psfc_vs_cpu_psfc": _stats(gpu["PSFC"] - cpu["PSFC"]),
        "gpu_psfc_equals_own_extrap": _stats(gpu["PSFC"] - _p_extrap_sfc(gpu)),
    }

    # 4. 3D pressure-state characterization at lowest mass level k=0
    k = 0
    for tag, d in (("cpu", cpu), ("gpu", gpu)):
        p_tot_k0 = d["P"][k] + d["PB"][k]
        res[f"{tag}_p_state_k0"] = {
            "p_total_minus_p_hyd_moist": _stats(
                p_tot_k0 - _p_hyd_half(d, MOIST_ALL, k)
            ),
            "p_total_minus_p_hyd_dry": _stats(p_tot_k0 - _p_hyd_half(d, (), k)),
        }
    res["p_total_k0_gpu_minus_cpu"] = _stats(
        (gpu["P"][k] + gpu["PB"][k]) - (cpu["P"][k] + cpu["PB"][k])
    )
    return res


def main() -> None:
    leads = [1, 4, 10]
    # latest common lead beyond 10
    for h in range(24, 10, -1):
        if _lead_path(GPU_DIR, h).exists() and _lead_path(CPU_DIR, h).exists():
            if h not in leads:
                leads.append(h)
            break

    out = {
        "proof": "psfc_moist_pressure_state_closure",
        "date": "2026-06-10",
        "run_root": str(RUN_ROOT),
        "cpu_truth": str(CPU_DIR),
        "domain": DOMAIN,
        "wrf_anchor": {
            "psfc": "phys/module_surface_driver.F:1988 PSFC=p8w(kts)",
            "p8w_binding": "dyn_em/module_first_rk_step_part1.F:1400 P8W=grid%p_hyd_w",
            "p_hyd_w": "dyn_em/module_big_step_utilities_em.F:4946-4958 phy_prep",
            "source_tree": "<USER_HOME>/src/wrf_pristine/WRF",
            "note": "contract path <USER_HOME>/src/canairy_meteo/Gen2/artifacts/"
                    "wrf_gpu_src/WRF does not exist on this box",
        },
        "leads": [analyse_lead(h) for h in leads],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_JSON}")
    for lead in out["leads"]:
        cur = lead["ablation_post_fix"]["current_gpu_psfc_vs_cpu_psfc"]
        fix = lead["ablation_post_fix"]["gpu_p_hyd_w_all_vs_cpu_psfc"]
        prf = lead["cpu_formula_proof"]["psfc_minus_p_hyd_w_all_species"]
        print(
            f"h{lead['lead_h']:>2}: GPU PSFC now bias={cur['mean']:+9.3f} "
            f"rmse={cur['rmse']:8.3f} | post-fix bias={fix['mean']:+9.3f} "
            f"rmse={fix['rmse']:8.3f} | CPU formula residual "
            f"mean={prf['mean']:+8.4f} rmse={prf['rmse']:8.4f} Pa"
        )


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
