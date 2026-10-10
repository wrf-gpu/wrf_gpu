"""Stage the pristine-WRF km_opt=3/km_opt=2 oracle fixture for tests (data/fixtures/les3d-smag3d-oracle-v1.npz).

Usage: JAX_PLATFORMS=cpu python proofs/o1_smag3d/make_fixture.py <out.npz>
Inputs: Swiss CPU-WRF wrfout 2023-01-15_12 (Alps), two 12x10x44 crops (real winds; winds x20 + theta noise).
Outputs: the pristine driver's arrays cropped to the JAX memory halo, per config, plus a branch census.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io  # noqa: E402

HALO = 3
WRFOUT = oracle_io.SWISS_DIR / "wrfout_d01_2023-01-15_12:00:00"
CROPS = {
    "real": dict(i0=4, j0=20, ncx=12, ncy=10),
    "wind20": dict(i0=4, j0=20, ncx=12, ncy=10, wind_scale=20.0, theta_pert=0.05),
}
CONFIGS = {
    "real/spec_vert_isfflx1": ("real", dict(vertical=True, isfflx=1)),
    "real/periodic_vert": ("real", dict(periodic=True, vertical=True, isfflx=1)),
    "wind20/spec_h": ("wind20", dict()),
    "wind20/spec_iso_vert_isfflx2": ("wind20", dict(vertical=True, isfflx=2, isotropic=1, tke_heat_flux=0.05,
                                                     tke_drag_coefficient=0.0013)),
    "wind20/spec_vert_isfflx0_thetad": ("wind20", dict(vertical=True, isfflx=0, use_theta_m=0,
                                                        tke_drag_coefficient=0.002, tke_heat_flux=0.02,
                                                        mix_full_fields=True)),
    # km_opt=2 (prognostic 3-D TKE: tke_km + tke_rhs + doubled TKE diffusion); TKE = 0.5*QKE of the CPU run.
    "real/km2_spec_vert_isfflx1": ("real", dict(km_opt=2, vertical=True, isfflx=1)),
    "wind20/km2_spec_h": ("wind20", dict(km_opt=2)),
    "real/km2_iso_periodic_vert_isfflx0": ("real", dict(km_opt=2, isotropic=1, periodic=True, vertical=True,
                                                         isfflx=0)),
}
KEEP = ("bn2", "xkmh", "xkmv", "xkhh", "xkhv", "defor11", "defor12", "defor13", "defor23", "zx", "zy",
        "ru_tendf", "rv_tendf", "rw_tendf", "t_tendf")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out: str) -> int:
    arrays: dict[str, np.ndarray] = {}
    meta = {"wrfout": str(WRFOUT), "crops": CROPS, "configs": {k: v[1] for k, v in CONFIGS.items()},
            "driver_sha256": _sha(Path(__file__).with_name("smag3d_driver.F90")),
            "libwrflib_sha256": _sha(Path("<USER_HOME>/src/wrf_pristine/WRF/main/libwrflib.a")),
            "module_diffusion_em_sha256": _sha(Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F")),
            "halo": HALO, "census": {}}
    cases = {}
    for cname, spec in CROPS.items():
        case = oracle_io.build_case(WRFOUT, **spec)
        cases[cname] = case
        for k, v in case.items():
            if isinstance(v, np.ndarray):
                arrays[f"in/{cname}/{k}"] = v
            elif k == "moist":
                for n, q in enumerate(v):
                    arrays[f"in/{cname}/moist{n}"] = q
            else:
                arrays[f"in/{cname}/{k}"] = np.asarray(v, np.float32)
    work = Path("<USER_HOME>/wrf_gpu2_lanes/o1-smag3d/fixture")
    for key, (cname, cfg) in CONFIGS.items():
        ora = oracle_io.run_oracle(cases[cname], cfg, work, key.replace("/", "_"))
        for f in KEEP:
            arrays[f"out/{key}/{f}"] = oracle_io.crop_to_halo(ora[f], HALO)
        for n, q in enumerate(ora["moist_tendf"]):
            arrays[f"out/{key}/moist{n}"] = oracle_io.crop_to_halo(q, HALO)
        if int(cfg.get("km_opt", 3)) == 2:
            arrays[f"out/{key}/tke_tendf"] = oracle_io.crop_to_halo(ora["tke_tendf"], HALO)
        bn2 = oracle_io.crop_to_halo(ora["bn2"], HALO)
        xkmh = oracle_io.crop_to_halo(ora["xkmh"], HALO)
        d = cases[cname]
        cap = 0.1 * (d["dx"] / d["msftx"]) * (d["dy"] / d["msfty"]) / d["dt"]
        inner = xkmh[:44, HALO + 1:HALO + d["msftx"].shape[0] - 1, HALO + 1:HALO + d["msftx"].shape[1] - 1]
        meta["census"][key] = dict(
            bn2_neg=int((bn2 < 0).sum()), bn2_pos=int((bn2 > 0).sum()),
            qc_ge_crit=int((d["moist"][1] >= 1e-5).sum()), qi_pos=int((d["moist"][3] > 0).sum()),
            xkmh_at_cap=int(np.isclose(inner, cap[1:-1, 1:-1][None], rtol=1e-6).sum()),
            xkmh_at_floor=int(np.isclose(inner, 1e-6 * (d["dx"] / d["msftx"][1:-1, 1:-1] * d["dy"]
                                                        / d["msfty"][1:-1, 1:-1])[None], rtol=1e-5).sum()),
            xkmh_inner=int(inner.size),
        )
        print(key, meta["census"][key], flush=True)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    np.savez_compressed(out, **arrays)
    print(out, Path(out).stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
