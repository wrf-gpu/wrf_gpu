"""CPU JAX smoke for the operational Noah-classic land coupler.

This is not a WRF parity test. It verifies that the operational physics-slot
ordering can thread explicit Noah-classic land/static inputs through a JAX scan
for a few steps with sf_surface_physics=2 selected: finite State,
finite/evolving land carry, land flux writeback, and no water-tile land-state
mutation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gpuwrf.contracts.grid import GridSpec  # noqa: E402
from gpuwrf.contracts.state import Tendencies  # noqa: E402
from gpuwrf.coupling.noahclassic_surface_hook import (  # noqa: E402
    NoahClassicLandState,
    NoahClassicRadiation,
    NoahClassicStatic,
)
from gpuwrf.physics.lsm_noah_classic import NoahClassicParams  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _initial_carry_for_run,
    _resolve_operational_suite,
)
from gpuwrf.coupling.physics_couplers import mynn_adapter  # noqa: E402
from gpuwrf.coupling.noahclassic_surface_hook import noahclassic_surface_step  # noqa: E402
from gpuwrf.coupling.scan_adapters import sfclay_revised_mm5_adapter  # noqa: E402
from gpuwrf.coupling.physics_dispatch import resolve_physics_suite  # noqa: E402

from proofs.v060.scanwire_smoke import _build_state  # noqa: E402

SAVEPOINTS = ROOT / "proofs" / "v060" / "savepoints_noahclassic.json"


def _tile(value, ny: int, nx: int, *, dtype=jnp.float64):
    return jnp.full((ny, nx), value, dtype=dtype)


def _tile4(values, ny: int, nx: int):
    arr = jnp.asarray(values, dtype=jnp.float64)
    return jnp.broadcast_to(arr, (ny, nx, arr.shape[0]))


def _load_column(name: str = "daytime_veg10") -> dict:
    data = json.loads(SAVEPOINTS.read_text())
    for col in data["columns"]:
        if col["name"] == name:
            return col
    raise KeyError(name)


def _bundle_from_column(col: dict, ny: int, nx: int) -> tuple[NoahClassicStatic, NoahClassicLandState]:
    rp = col["wrf"]["redprm"]
    snow = col["wrf"]["snow_in"]
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    z4 = jnp.zeros((ny, nx, 4), dtype=jnp.float64)
    params = NoahClassicParams(
        bexp=_tile(rp["bexp"], ny, nx),
        dksat=_tile(rp["dksat"], ny, nx),
        dwsat=_tile(rp["dwsat"], ny, nx),
        psisat=_tile(rp["psisat"], ny, nx),
        quartz=_tile(rp["quartz"], ny, nx),
        f1=_tile(rp["f1"], ny, nx),
        smcmax=_tile(rp["smcmax"], ny, nx),
        smcwlt=_tile(rp["smcwlt"], ny, nx),
        smcref=_tile(rp["smcref"], ny, nx),
        smcdry=_tile(rp["smcdry"], ny, nx),
        kdt=_tile(rp["kdt"], ny, nx),
        frzx=_tile(rp["frzx"], ny, nx),
        slope=_tile(rp["slope"], ny, nx),
        snup=_tile(rp["snup"], ny, nx),
        salp=_tile(rp["salp"], ny, nx),
        czil=_tile(rp["czil"], ny, nx),
        sbeta=_tile(rp["sbeta"], ny, nx),
        csoil=_tile(rp["csoil"], ny, nx),
        fxexp=_tile(rp["fxexp"], ny, nx),
        zbot=_tile(rp["zbot"], ny, nx),
        cfactr=_tile(rp["cfactr"], ny, nx),
        cmcmax=_tile(rp["cmcmax"], ny, nx),
        rsmax=_tile(rp["rsmax"], ny, nx),
        topt=_tile(rp["topt"], ny, nx),
        rgl=_tile(rp["rgl"], ny, nx),
        hs=_tile(rp["hs"], ny, nx),
        rsmin=_tile(rp["rsmin"], ny, nx),
        lvcoef=_tile(rp["lvcoef"], ny, nx),
        nroot=_tile(int(rp["nroot"]), ny, nx, dtype=jnp.int32),
        rtdis=_tile4(rp["rtdis"], ny, nx),
        alb=_tile(rp["alb"], ny, nx),
        embrd=_tile(rp["embrd"], ny, nx),
        xlai=_tile(rp["xlai"], ny, nx),
        z0brd=_tile(rp["z0brd"], ny, nx),
        shdfac=_tile(rp["shdfac"], ny, nx),
        is_urban=jnp.full((ny, nx), bool(col["vegtyp"] == col["isurban"])),
    )
    smav = (
        (_tile4(col["wrf"]["smc_in"], ny, nx) - params.smcwlt[..., None])
        / (params.smcmax - params.smcwlt)[..., None]
    )
    static = NoahClassicStatic(
        params=params,
        zsoil=_tile4(col["zsoil"], ny, nx),
        sldpth=_tile4(col["sldpth"], ny, nx),
        snoalb=_tile(col["state_in"]["snoalb"], ny, nx),
        tbot=_tile(col["tbot"], ny, nx),
        solnet_albedo=_tile(col["state_in"]["albbck"], ny, nx),
        lwdn_emissivity=_tile(col["state_in"]["emiss"], ny, nx),
    )
    land = NoahClassicLandState(
        t1=_tile(col["wrf"]["t1_in"], ny, nx),
        stc=_tile4(col["wrf"]["stc_in"], ny, nx),
        smc=_tile4(col["wrf"]["smc_in"], ny, nx),
        sh2o=_tile4(col["wrf"]["sh2o_in"], ny, nx),
        cmc=_tile(snow["cmc"], ny, nx),
        sneqv=_tile(snow["sneqv"], ny, nx),
        snowh=_tile(snow["snowh"], ny, nx),
        sncovr=_tile(snow["sncovr"], ny, nx),
        snotime1=_tile(col["state_in"]["snotime1"], ny, nx),
        ribb=_tile(col["wrf"]["chcm_in"]["ribb"], ny, nx),
        flx4=zero,
        fvb=zero,
        fbur=zero,
        fgsn=zero,
        smcrel=smav,
        xlaidyn=params.xlai,
        hfx=zero,
        qfx=zero,
        lh=zero,
        grdflx=zero,
    )
    return static, land


def _state_with_noah_seed(grid: GridSpec, static: NoahClassicStatic, land: NoahClassicLandState):
    state, _ = _build_state(nz=grid.nz, ny=grid.ny, nx=grid.nx, seed=11)
    is_land = jnp.asarray(state.xland) < 1.5
    return state.replace(
        t_skin=jnp.where(is_land, land.t1, state.t_skin),
        soil_moisture=jnp.where(is_land, land.smc[..., 0], state.soil_moisture),
        mavail=jnp.where(is_land, land.smcrel[..., 0], state.mavail),
        roughness_m=jnp.where(is_land, static.params.z0brd, state.roughness_m),
        theta_flux=jnp.full((grid.ny, grid.nx), 0.02, dtype=jnp.float64),
        qv_flux=jnp.full((grid.ny, grid.nx), 1.0e-4, dtype=jnp.float64),
    )


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    return Tendencies(
        u=jnp.zeros((nz, ny, nx + 1), dtype=jnp.float64),
        v=jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64),
        w=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        theta=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        qv=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        p=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu=jnp.zeros((ny, nx), dtype=jnp.float64),
    )


def run(*, steps: int = 3, column_name: str = "daytime_veg10") -> dict:
    grid = GridSpec.canary_3km_template()
    col = _load_column(column_name)
    static, land0 = _bundle_from_column(col, grid.ny, grid.nx)
    state0 = _state_with_noah_seed(grid, static, land0)
    nl0 = OperationalNamelist.from_grid(
        grid,
        tendencies=_cpu_tendencies(grid),
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=1,
        radiation_cadence_steps=999999,
        disable_guards=True,
        force_fp64=True,
    )
    namelist = nl0.__class__(
        **{
            **{name: getattr(nl0, name) for name in nl0.__dataclass_fields__},
            "mp_physics": 0,
            "bl_pbl_physics": 5,
            "sf_sfclay_physics": 1,
            "sf_surface_physics": 2,
            "cu_physics": 0,
            "use_noahmp": False,
            "run_boundary": False,
            "noahclassic_static": static,
            "noahclassic_land": land0,
            "noahclassic_rad": NoahClassicRadiation(
                soldn=_tile(col["wrf"]["forcing"]["soldn"], grid.ny, grid.nx),
                lwdn=_tile(col["wrf"]["forcing"]["glw"], grid.ny, grid.nx),
                cosz=jnp.ones((grid.ny, grid.nx), dtype=jnp.float64),
            ),
        }
    )
    suite = resolve_physics_suite(namelist)
    _resolve_operational_suite(namelist)
    initial = _initial_carry_for_run(state0, namelist)

    def body(carry, _step):
        state = sfclay_revised_mm5_adapter(carry.state, float(namelist.dt_s), namelist.grid)
        state, land = noahclassic_surface_step(
            state,
            carry.noahclassic_land,
            namelist.noahclassic_static,
            float(namelist.dt_s),
            radiation=carry.noahclassic_rad,
        )
        state = mynn_adapter(state, float(namelist.dt_s), namelist.grid)
        return carry.replace(state=state, noahclassic_land=land), None

    final, _ = jax.lax.scan(body, initial, jnp.arange(int(steps), dtype=jnp.int32))
    jax.block_until_ready(final.state.theta)
    land1 = final.noahclassic_land
    is_land = np.asarray(state0.xland) < 1.5
    is_water = ~is_land

    def max_abs(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    finite_state = all(
        bool(jnp.all(jnp.isfinite(getattr(final.state, leaf))))
        for leaf in ("theta", "qv", "t_skin", "soil_moisture", "theta_flux", "qv_flux", "fltv")
    )
    finite_land = all(
        bool(jnp.all(jnp.isfinite(getattr(land1, leaf))))
        for leaf in ("t1", "stc", "smc", "sh2o", "smcrel", "hfx", "qfx", "lh", "grdflx")
    )
    land_t1_delta = max_abs(np.asarray(land1.t1)[is_land], np.asarray(land0.t1)[is_land])
    land_stc_delta = max_abs(np.asarray(land1.stc)[is_land], np.asarray(land0.stc)[is_land])
    land_smc_delta = max_abs(np.asarray(land1.smc)[is_land], np.asarray(land0.smc)[is_land])
    water_stc_delta = max_abs(np.asarray(land1.stc)[is_water], np.asarray(land0.stc)[is_water])
    water_smc_delta = max_abs(np.asarray(land1.smc)[is_water], np.asarray(land0.smc)[is_water])
    soil_water0 = float(jnp.sum(jnp.asarray(land0.smc)[..., :] * jnp.asarray(static.sldpth)))
    soil_water1 = float(jnp.sum(jnp.asarray(land1.smc)[..., :] * jnp.asarray(static.sldpth)))
    soil_rel = abs(soil_water1 - soil_water0) / max(abs(soil_water0), 1.0e-12)
    flux_nonzero = float(jnp.max(jnp.abs(land1.hfx))) > 0.0 and float(jnp.max(jnp.abs(land1.grdflx))) > 0.0
    state_flux_written = float(jnp.max(jnp.abs(final.state.theta_flux - state0.theta_flux))) > 0.0
    pass_flag = bool(
        finite_state
        and finite_land
        and land_t1_delta > 0.0
        and (land_stc_delta > 0.0 or land_smc_delta > 0.0)
        and water_stc_delta == 0.0
        and water_smc_delta == 0.0
        and soil_rel < 0.05
        and flux_nonzero
        and state_flux_written
        and suite.gpu_gate_ready
    )
    return {
        "proof": "v060-noahclassic-operational-coupler-smoke",
        "kind": "CPU JAX operational physics-slot scan smoke; explicit WRF savepoint-derived NoahClassicStatic/LandState",
        "jax_platform": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "steps": int(steps),
        "selected_physics": {
            "mp_physics": 0,
            "bl_pbl_physics": 5,
            "sf_sfclay_physics": 1,
            "sf_surface_physics": 2,
            "cu_physics": 0,
        },
        "dispatcher_gpu_gate_ready": bool(suite.gpu_gate_ready),
        "operational_resolver_accepts": True,
        "finite_state": bool(finite_state),
        "finite_land": bool(finite_land),
        "land_updates": {
            "t1_max_abs_delta_K": land_t1_delta,
            "stc_max_abs_delta_K": land_stc_delta,
            "smc_max_abs_delta_m3_m3": land_smc_delta,
        },
        "water_tile_land_carry_unchanged": {
            "stc_max_abs_delta_K": water_stc_delta,
            "smc_max_abs_delta_m3_m3": water_smc_delta,
            "pass": bool(water_stc_delta == 0.0 and water_smc_delta == 0.0),
        },
        "flux_feedback": {
            "hfx_minmax_w_m2": [float(jnp.min(land1.hfx)), float(jnp.max(land1.hfx))],
            "qfx_minmax_kg_m2_s": [float(jnp.min(land1.qfx)), float(jnp.max(land1.qfx))],
            "grdflx_minmax_w_m2": [float(jnp.min(land1.grdflx)), float(jnp.max(land1.grdflx))],
            "state_theta_flux_written": bool(state_flux_written),
            "nonzero_land_fluxes": bool(flux_nonzero),
        },
        "soil_water": {
            "total_before": soil_water0,
            "total_after": soil_water1,
            "relative_change": soil_rel,
            "bounded": bool(soil_rel < 0.05),
        },
        "pass": pass_flag,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "proofs" / "v060" / "noah_coupler_smoke.json")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    report = run(steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)
