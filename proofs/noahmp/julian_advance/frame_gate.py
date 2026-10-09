"""Noah-MP phenology clock vs CPU-WRF, every history frame (CPU only; GPUWRF_NOAHMP_JULIAN_ADVANCE).

For each CPU-WRF frame at valid time t the written LAI/XSAI come from the LSM call of the step that started at
t - dt (grid%julian at step start, clock advanced after solve). The port phenology (noahmp_phenology_table,
dveg=4) is evaluated with the port clock helper _noahmp_clock at that lead and compared on all land cells.
usage: frame_gate.py <case_run_dir> <out.json> [--clock advance|frozen|step_end] [--doms d01,d02,d03]
"""
import argparse, glob, json, os, sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from gpuwrf.config.paths import wrf_run_dir  # noqa: E402
from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.integration.nested_pipeline import _wrf_julian_yearlen  # noqa: E402
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table  # noqa: E402
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_gas_clock  # noqa: E402
from gpuwrf.runtime.operational_mode import _noahmp_clock  # noqa: E402


def inputs(ivgtyp, lat, tv, snowh):
    shape = ivgtyp.shape
    z = jnp.zeros(shape, jnp.float64)
    f = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
    land = NoahMPLandState(
        tslb=jnp.zeros((NSOIL, *shape)), smois=jnp.zeros((NSOIL, *shape)), sh2o=jnp.zeros((NSOIL, *shape)), smcwtd=z,
        isnow=jnp.zeros(shape, jnp.int32), tsno=jnp.zeros((NSNOW, *shape)), snice=jnp.zeros((NSNOW, *shape)),
        snliq=jnp.zeros((NSNOW, *shape)), zsnso=jnp.zeros((NSNOW + NSOIL, *shape)), snowh=f(snowh), sneqv=z,
        sneqvo=z, tauss=z, albold=z, tv=f(tv), tg=z, tah=z, eah=z, canliq=z, canice=z, fwet=z, lai=z, sai=z, cm=z,
        ch=z, t_skin=z, qsfc=z, znt=z, emiss=z, albedo=z, sfcrunoff=z, udrunoff=z)
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(ivgtyp, jnp.int32), isltyp=jnp.ones(shape, jnp.int32), xland=z + 1.0, landmask=z + 1.0,
        lakemask=z, lu_index=jnp.asarray(ivgtyp, jnp.int32), tbot=z, dzs=jnp.zeros(4), zsoil=jnp.zeros(4),
        lat=f(lat), dx_m=1000.0, parameters=load_noahmp_parameters(wrf_run_dir()), shdmax=z + 0.5, shdfac=z + 0.5)
    return land, static, z


def phenology(land, static, z, clock):
    forcing = NoahMPForcing(sfctmp=z, sfcprs=z, psfc=z, uu=z, vv=z, qair=z, qc=z, soldn=z, lwdn=z, prcpconv=z,
                            prcpnonc=z, prcpsnow=z, prcpgrpl=z, prcphail=z, cosz=z, zlvl=z,
                            julian=jnp.asarray(clock.julian), yearlen=jnp.asarray(clock.yearlen))
    ph = noahmp_phenology_table(land, forcing, static)
    return np.asarray(ph.lai, np.float64), np.asarray(ph.sai, np.float64)


def frame_error(port, ref, mask):
    """max |port - ref| over the masked cells; inf if any masked port or reference value is non-finite.

    E200: Python max(worst, nan) keeps worst, so a NaN candidate/reference would otherwise pass (review-b RB138).
    """
    p = np.asarray(port, np.float64)[mask]
    r = np.asarray(ref, np.float64)[mask]
    if not (np.isfinite(p).all() and np.isfinite(r).all()):
        return float("inf")
    return float(np.abs(p - r).max()) if p.size else 0.0


def clock_for(start, lead, mode):
    julian0, yearlen0 = _wrf_julian_yearlen(start)
    base = SimpleNamespace(noahmp_julian=jnp.asarray(julian0, jnp.float64), noahmp_yearlen=jnp.asarray(yearlen0, jnp.float64),
                           ghg_clock=jax.tree_util.tree_map(jnp.asarray, clwrf_gas_clock(start)))
    if mode == "frozen":
        os.environ.pop("GPUWRF_NOAHMP_JULIAN_ADVANCE", None)
    else:
        os.environ["GPUWRF_NOAHMP_JULIAN_ADVANCE"] = "1"
    return _noahmp_clock(None, base, lead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--clock", default="advance", choices=("advance", "frozen", "step_end"))
    ap.add_argument("--doms", default="d01,d02,d03")
    a = ap.parse_args()
    report = {"run": a.run, "clock": a.clock, "domains": {}}
    for dom in a.doms.split(","):
        files = sorted(glob.glob(f"{a.run}/wrfout_{dom}_*"))
        d0 = Dataset(files[0]); dt = float(d0.DT)
        start = datetime.strptime(d0.START_DATE, "%Y-%m-%d_%H:%M:%S")
        ivg, lat, land_mask = (np.asarray(d0[v][0]) for v in ("IVGTYP", "XLAT", "LANDMASK"))
        land_mask = land_mask > 0.5
        # dveg=4 LAI/SAI depend only on the clock, VEGTYP and the hemisphere (TV/SNOWH feed IGS/ELAI/ESAI only).
        land, static, z = inputs(ivg, lat, np.asarray(d0["TV"][0]), np.asarray(d0["SNOWH"][0]))
        worst = {"LAI": 0.0, "XSAI": 0.0}; frames = 0
        for path in files[1:]:
            d = Dataset(path)
            t = datetime.strptime(path.split(f"wrfout_{dom}_")[1], "%Y-%m-%d_%H:%M:%S")
            lead = (t - start).total_seconds() - (0.0 if a.clock == "step_end" else dt)
            lai, sai = phenology(land, static, z, clock_for(start, lead, a.clock))
            for name, port in (("LAI", lai), ("XSAI", sai)):
                worst[name] = max(worst[name], frame_error(port, d[name][0], land_mask))
            frames += 1
        report["domains"][dom] = dict(frames=frames, dt=dt, start=str(start), land_cells=int(land_mask.sum()), max_abs=worst)
        print(dom, report["domains"][dom], flush=True)
    report["pass_1e-6"] = all(max(v["max_abs"].values()) <= 1e-6 for v in report["domains"].values())
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print("PASS" if report["pass_1e-6"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
