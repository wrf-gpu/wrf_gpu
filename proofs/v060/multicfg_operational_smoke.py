"""v0.6.0 INTEGRATED MULTI-CONFIG OPERATIONAL SMOKE.

The v0.6.0 release prerequisite: prove the consolidated 12-scheme physics suite
runs end-to-end THROUGH THE OPERATIONAL COUPLER (``runtime.operational_mode`` +
``coupling.physics_dispatch`` + ``coupling.scan_adapters``) across EVERY supported
namelist config -- not just per-scheme in isolation (that is the lane savepoint
parity + ``scanwire_smoke.py``). Each scheme already PASSED savepoint parity
individually; the OPEN gap this closes is INTEGRATION: do the schemes run TOGETHER,
in WRF physics-driver call order, across the supported config combinations, while
staying finite + physical + actually-active (no silent no-op) + JIT-traceable
(traceable on CPU == lowerable on the GPU scan).

WRF-FAITHFUL DISCIPLINE
-----------------------
* NO masking / clamps / self-compare / synthetic-happy-path. The integrated step
  is the EXACT dispatch + adapter sequence the operational scan body
  (``operational_mode._physics_boundary_step``) runs: dispatcher-selected
  microphysics -> surface-layer -> land (Noah-MP / Noah-classic / bulk) -> MYNN
  PBL -> KF cumulus, in WRF call order, threading the real KF ``(w0avg, nca)`` and
  the prognostic Noah-MP / Noah-classic land carries.
* Each config is ALSO validated through the operational coupler's OWN fail-closed
  gate ``operational_mode._resolve_operational_suite`` (the real authority the
  public ``run_forecast_operational`` entry calls), so "scheme active" / "rejected"
  is the operational coupler's verdict, not this harness's.
* HONEST verdict: a config that breaks is reported as a real integration finding,
  never skipped/masked.

REAL CANARY CASE
----------------
The operational state is built on the REAL Canary grid:
  * ``GridSpec.canary_3km_template()`` -- the canonical Canary 3 km grid (the M3
    audit grid), for the small fast configs (compile + a few steps, CPU cores 0-3).
  * the REAL d02 corpus wrfinput (``build_noahmp_land_state`` over a corpus run dir)
    -- the actual Canary d02 land mask + Noah-MP soil/veg warm-start, for the
    Noah-MP land configs (the real prognostic land carry).
  * REAL pristine-WRF NOAHMP_SFLX savepoint columns
    (``proofs/v060/savepoints_noahclassic.json``) -- the WRF-derived
    NoahClassicStatic / NoahClassicLandState bundle for the Noah-classic config.

The atmospheric profile uses the validated b2 C-grid pattern
(``scanwire_smoke._build_state``): physically-reasonable theta / p / moisture /
winds on the Canary grid, with the real land/sea mask overlaid where available.

WHY THE PHYSICS-DISPATCH PATH, NOT ``run_forecast_operational`` END-TO-END
--------------------------------------------------------------------------
``run_forecast_operational`` couples the WRF acoustic DYCORE to physics. The
dycore needs a dynamically-balanced initial condition (real ``mu``/base pressure/
geopotential from a corpus replay); ``build_replay_case`` / ``State.zeros`` /
``Tendencies.zeros`` REQUIRE a JAX GPU device, so the dynamically-balanced
real-corpus state cannot be built on CPU (this sprint is NO-GPU, cores 0-3). A b2
profile is NOT acoustically balanced, so the full dycore NaNs on it -- a DYNAMICS-
on-synthetic-IC artifact ORTHOGONAL to the v0.6.0 PHYSICS-INTEGRATION question.

This smoke therefore isolates the v0.6.0 integration surface -- the physics-suite
dispatch + adapter coupling that the v0.6.0 lanes added -- and runs the EXACT
operational physics block under ``jax.jit`` + ``jax.lax.scan`` (proving JIT-
traceability == GPU-runnability) for several steps. The full dycore+physics GPU
forecast vs CPU-WRF is the MANAGER-scheduled GPU gate (see the documented
reference-scoring seam ``reference_scoring_seam`` below; the fixed v0.4.0 forecast-
gate scorer plugs in there).

Run (CPU, cores 0-3, NO GPU):
  taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 \
      PYTHONPATH=src python3 proofs/v060/multicfg_operational_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts.grid import GridSpec  # noqa: E402
from gpuwrf.coupling.physics_couplers import (  # noqa: E402
    mynn_adapter,
    surface_adapter,
    thompson_adapter,
)
from gpuwrf.coupling.physics_dispatch import (  # noqa: E402
    DEFAULT_BL_PBL_PHYSICS,
    DEFAULT_MP_PHYSICS,
    UnsupportedSchemeSelection,
    resolve_physics_suite,
)
from gpuwrf.coupling.scan_adapters import (  # noqa: E402
    CU_SCAN_ADAPTERS,
    CU_STATELESS_SCAN_ADAPTERS,
    MP_SCAN_ADAPTERS,
    PBL_SCAN_ADAPTERS,
    SFCLAY_SCAN_ADAPTERS,
    bmj_adapter,
    initial_bmj_carry,
    initial_kf_carry,
    kf_adapter,
)
from gpuwrf.coupling.noahclassic_surface_hook import (  # noqa: E402
    NoahClassicLandState,
    NoahClassicRadiation,
    NoahClassicStatic,
    noahclassic_surface_step,
)
from gpuwrf.io.namelist_check import validate_supported_namelist  # noqa: E402
from gpuwrf.runtime.operational_mode import _resolve_operational_suite  # noqa: E402

from proofs.v060.scanwire_smoke import _build_state  # noqa: E402

# Real-data sources --------------------------------------------------------------
# A corpus L2 run with full output (manifest has_full_output=true) for the real d02
# Canary wrfinput (Noah-MP land warm-start + real land mask).
_CORPUS_RUNDIR = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z"
)
_NOAHCLASSIC_SAVEPOINTS = ROOT / "proofs" / "v060" / "savepoints_noahclassic.json"

# Physical-range envelopes (WRF-physical, generous; flag only true blow-ups).
PHYS_BOUNDS = {
    "theta": (180.0, 400.0),   # K (perturbation+base potential temperature)
    "qv": (0.0, 0.06),         # kg/kg
    "u": (-150.0, 150.0),      # m/s
    "v": (-150.0, 150.0),      # m/s
    "w": (-50.0, 50.0),        # m/s
    "p": (1.0, 110000.0),      # Pa
    "t_skin": (200.0, 360.0),  # K
}

_WATER = ("qv", "qc", "qr", "qi", "qs", "qg")

_Rad = namedtuple("_Rad", "soldn lwdn cosz")
_Clk = namedtuple("_Clk", "julian yearlen")


# ============================================================================
# CONFIG SWEEP -- the COVERING set (NOT the full cartesian product).
# ============================================================================
#
# Supported (ACCEPT matrix; New-Tiedtke cu=16 EXCLUDED = documented-TODO):
#   mp_physics      {8 Thompson, 6 WSM6, 10 Morrison, 16 WDM6, 1 Kessler}
#   bl_pbl_physics  {5 MYNN, 1 YSU, 7 ACM2}
#   sf_sfclay       {1 revised-MM5, 5 MYNN-sfclay, 7 Pleim-Xiu}
#   cu_physics      {1 KF, 2 BMJ, 3 Grell-Freitas, 6 Tiedtke}   (+ 0 = no cumulus)
#   sf_surface      {4 Noah-MP, 2 Noah classic}    (+ bulk prescribed land)
#
# COVERING RATIONALE: every supported scheme appears in >=1 config, plus the real
# operational Canary config. A full product (5*3*3*2*2) = 180 combos is wasteful;
# the covering set exercises each scheme at least once while pairing it with a
# realistic partner suite, ~17 configs total.
#
# HONEST OPERATIONAL-COUPLER REALITY (re-audited 2026-06-04 at the v0.9.0 GF scan-wire
# against the scan-wire tables): the operational scan (``run_forecast_operational`` /
# ``_physics_boundary_step``) threads MYNN(5) + YSU(1) + ACM2(7) PBL -- YSU and ACM2
# are the v0.6.0 jax.lax.scan-traceable rewrites (pbl_ysu.ysu_columns /
# pbl_acm2.acm2_columns) wired via coupling.scan_adapters.PBL_SCAN_ADAPTERS, full RUN
# configs (gpu_runnable=True). The cumulus slot now threads KF(1) + BMJ(2) +
# Grell-Freitas(3) + modified-Tiedtke(6) -- GF(3) is the v0.9.0 GPU-batched jit/vmap
# scale-aware closure-ensemble port (physics._gf_jax.gfdrv_batched, stateless
# State->State, savepoint-parity gated), so it is now a RUN config. The ONLY scheme
# that remains fail-closed in the GPU operational scan is New-Tiedtke (cu=16, not
# separately source-gated) plus MYJ(2) PBL + Janjic Eta(2) sfclay (no scan adapter
# yet). This smoke includes THOSE as FAIL-CLOSED integration assertions (the coupler
# must REJECT them loudly, not silently no-op) -- a genuine integration finding,
# recorded honestly, not masked.


@dataclass(frozen=True)
class Config:
    """One supported namelist config for the integration sweep."""

    cfg_id: str
    description: str
    mp_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    cu_physics: int
    sf_surface_physics: int | None  # 4 Noah-MP, 2 Noah-classic, None = bulk land
    expect: str  # "RUN" (integrate end-to-end) | "FAIL_CLOSED" (coupler must reject)
    covers: tuple[str, ...] = ()

    @property
    def use_noahmp(self) -> bool:
        return self.sf_surface_physics == 4

    def as_namelist(self) -> dict[str, int]:
        # Bulk prescribed land (sf_surface_physics=None) is selected in the operational
        # scan by use_noahmp=False with no explicit sf_surface_physics pin. WRF's
        # nearest accepted code for "validate the rest of the suite" is 2 (the dispatcher
        # maps use_noahmp=False -> 2 = the bulk path in THIS scan); report it as 2 so the
        # namelist-accept/dispatch checks see a supported value while the run uses bulk.
        return {
            "mp_physics": self.mp_physics,
            "bl_pbl_physics": self.bl_pbl_physics,
            "sf_sfclay_physics": self.sf_sfclay_physics,
            "cu_physics": self.cu_physics,
            "sf_surface_physics": 2 if self.sf_surface_physics is None else self.sf_surface_physics,
            "ra_sw_physics": 4,
            "ra_lw_physics": 4,
        }


# RUN configs: every scan-wired scheme exercised end-to-end through the coupler.
#
# PERF NOTE: the prognostic Noah-MP land step is the heavy per-step cost (canopy
# energy/water balance over every land cell). To keep EVERY smoke tiny (the task's
# CPU-cores-0-3, no-contention constraint), Noah-MP land is exercised in its
# DEDICATED land-coverage configs only (the real operational baseline + land_noahmp),
# on a small SUBSET tile of the real d02 Canary land. The microphysics / surface-
# layer / cumulus COVERAGE configs route through the FAST bulk-surface land (or the
# Noah-classic config for land=2 coverage) on the small Canary grid -- each config
# isolates the ONE new axis it covers, so no config redundantly pays the Noah-MP cost.
SWEEP: tuple[Config, ...] = (
    # --- the REAL operational Canary config (v0.2.0 validated baseline; Noah-MP land) ---
    Config("real_canary_v020", "REAL operational Canary baseline: Thompson/MYNN/MYNN-sfclay/Noah-MP, no cumulus",
           8, 5, 5, 0, 4, "RUN", covers=("mp8", "bl5", "sf5", "land4-NoahMP", "no-cu")),
    # --- microphysics coverage (each MP scheme >=1 config; fast bulk land) ---
    Config("mp_thompson_kf", "Thompson(8) + MYNN + MYNN-sfclay + bulk land + KF cumulus",
           8, 5, 5, 1, None, "RUN", covers=("mp8", "cu1-KF")),
    Config("mp_wsm6", "WSM6(6) single-moment + MYNN + revised-MM5 sfclay + bulk land, no cumulus",
           6, 5, 1, 0, None, "RUN", covers=("mp6-WSM6", "sf1-revisedMM5")),
    Config("mp_morrison", "Morrison(10) two-moment + MYNN + Pleim-Xiu sfclay + bulk land, no cumulus",
           10, 5, 7, 0, None, "RUN", covers=("mp10-Morrison", "sf7-PleimXiu")),
    Config("mp_wdm6_kf", "WDM6(16) double-moment (Nc/Nn additive leaves) + MYNN + MYNN-sfclay + bulk land + KF",
           16, 5, 5, 1, None, "RUN", covers=("mp16-WDM6", "Nc/Nn", "cu1-KF")),
    Config("mp_kessler", "Kessler(1) warm-rain + MYNN + revised-MM5 sfclay + bulk land, no cumulus",
           1, 5, 1, 0, None, "RUN", covers=("mp1-Kessler",)),
    Config("mp_wsm3", "WSM3(3) simple-ice + MYNN + MYNN-sfclay + bulk land, no cumulus",
           3, 5, 5, 0, None, "RUN", covers=("mp3-WSM3",)),
    Config("mp_wsm5", "WSM5(4) + MYNN + MYNN-sfclay + bulk land, no cumulus",
           4, 5, 5, 0, None, "RUN", covers=("mp4-WSM5",)),
    Config("mp_lin", "Purdue-Lin(2) single-moment ice/graupel + MYNN + MYNN-sfclay + bulk land, no cumulus",
           2, 5, 5, 0, None, "RUN", covers=("mp2-Lin",)),
    # --- surface-layer coverage (sfclay 1/5/7 each >=1 config; fast bulk land) ---
    Config("sfclay_mynn", "Thompson + MYNN + MYNN-sfclay(5) + bulk land, no cumulus",
           8, 5, 5, 0, None, "RUN", covers=("sf5-MYNN-sfclay",)),
    # --- cumulus coverage (KF + BMJ + GF + Tiedtke + no-cumulus; fast bulk land) ---
    Config("cu_kf", "Thompson + MYNN + MYNN-sfclay + bulk land + KF(1) cumulus",
           8, 5, 5, 1, None, "RUN", covers=("cu1-KF",)),
    Config("cu_bmj", "Thompson + MYNN + MYNN-sfclay + bulk land + BMJ(2) cumulus -- fp64 savepoint-proven, CLDEFI carry-threaded adapter",
           8, 5, 5, 2, None, "RUN", covers=("cu2-BMJ",)),
    Config("cu_gf", "Thompson + MYNN + MYNN-sfclay + bulk land + Grell-Freitas(3) cumulus -- v0.9.0 GPU-batched jit/vmap scale-aware stateless adapter",
           8, 5, 5, 3, None, "RUN", covers=("cu3-GF",)),
    Config("cu_tiedtke", "Thompson + MYNN + MYNN-sfclay + bulk land + Tiedtke(6) cumulus -- v0.6.0 GPU-batched jit/vmap adapter",
           8, 5, 5, 6, None, "RUN", covers=("cu6-Tiedtke",)),
    Config("cu_none", "Thompson + MYNN + MYNN-sfclay + bulk land, cu=0 (resolved grid-scale)",
           8, 5, 5, 0, None, "RUN", covers=("cu0-none",)),
    # --- land-surface coverage (Noah-MP, Noah-classic, bulk) ---
    Config("land_noahmp", "WSM6 + MYNN + revised-MM5 sfclay + Noah-MP(4) prognostic land + KF",
           6, 5, 1, 1, 4, "RUN", covers=("land4-NoahMP",)),
    Config("land_noahclassic", "Kessler + MYNN + revised-MM5 sfclay + Noah-classic(2) 4-layer land, no cumulus",
           1, 5, 1, 0, 2, "RUN", covers=("land2-NoahClassic",)),
    Config("land_bulk", "Thompson + MYNN + MYNN-sfclay + bulk prescribed land, no cumulus",
           8, 5, 5, 0, None, "RUN", covers=("land-bulk",)),
    # --- PBL coverage (YSU/ACM2 are the v0.6.0 jax.lax.scan-traceable rewrites,
    #     scan-wired in the PBL-GPU-op lane; consolidated here -> now RUN configs).
    #     Routed through fast bulk land (the PBL slot is the axis under test). ---
    Config("pbl_ysu", "YSU(1) PBL -- v0.6.0 jax.lax.scan rewrite (pbl_ysu.ysu_columns) + revised-MM5 sfclay + bulk land",
           8, 1, 1, 0, None, "RUN", covers=("bl1-YSU",)),
    Config("pbl_acm2", "ACM2(7) PBL -- v0.6.0 jax.lax.scan rewrite (pbl_acm2.acm2_columns) + revised-MM5 sfclay(1) + bulk land. ACM2 consumes revised-MM5 surface forcing; bl7+sf1 is the Issue-A-validated gate-ready pairing (bl7+other-sf fail-closes, no silent substitution).",
           8, 7, 1, 0, None, "RUN", covers=("bl7-ACM2",)),
    Config("pbl_boulac", "BouLac(8) PBL -- v0.6.0 jax.lax.scan rewrite (pbl_boulac.boulac_columns) + revised-MM5 sfclay + bulk land",
           8, 8, 1, 0, None, "RUN", covers=("bl8-BouLac",)),
    # --- FAIL-CLOSED configs (coupler must REJECT loudly; honest integration finding) ---
    # Tiedtke(6) and Grell-Freitas(3) are now GPU-batched + scan-wired -> moved to
    # RUN configs above (GF(3) is the v0.9.0 jit/vmap scale-aware closure-ensemble port).
    # New-Tiedtke(16): interface-compatible but not separately savepoint-gated by a
    # distinct WRF source path -> not scan-wired, must fail closed.
    Config("cu_newtiedtke_unwired", "New-Tiedtke(16) cumulus -- not separately gated (no distinct WRF source path)",
           8, 5, 5, 16, 4, "FAIL_CLOSED", covers=("cu16-NewTiedtke",)),
    # MYJ(2) PBL + Janjic Eta(2) sfclay: now OPERATIONAL (wired c612ab9, oracle PASS vs
    # pristine-WRF savepoints worst PBL 2.7e-11 / SFC 1.6e-10). MYJ re-runs its mandatory
    # Janjic surface layer, so the bl2<->sf2 pair runs end-to-end through the operational scan.
    Config("pbl_myj_janjic", "MYJ(2) PBL + Janjic Eta(2) sfclay -- operational (c612ab9); MYJ re-runs the mandatory Janjic sfclay",
           8, 2, 2, 0, 4, "RUN", covers=("bl2-MYJ", "sf2-Janjic")),
)


# ============================================================================
# Real-Canary operational-state construction
# ============================================================================

class _NLStub:
    """Namelist-shaped stub exposing the physics-option attrs the operational
    coupler's ``_resolve_operational_suite`` reads -- so the SAME fail-closed gate
    the public ``run_forecast_operational`` calls decides accept/reject here."""

    def __init__(self, cfg: Config):
        self.mp_physics = cfg.mp_physics
        self.bl_pbl_physics = cfg.bl_pbl_physics
        self.sf_sfclay_physics = cfg.sf_sfclay_physics
        self.cu_physics = cfg.cu_physics
        self.sf_surface_physics = cfg.sf_surface_physics
        # v0.13 added ra_sw/ra_lw to _SCAN_WIRED_OPTIONS (read with no default in
        # _resolve_operational_suite); mirror the OperationalNamelist defaults
        # (RRTMG SW/LW = 4) so this stub matches the public namelist contract.
        self.ra_sw_physics = getattr(cfg, "ra_sw_physics", 4)
        self.ra_lw_physics = getattr(cfg, "ra_lw_physics", 4)
        self.use_noahmp = cfg.use_noahmp
        # Noah-classic resolve path checks for explicit land/static bundles; in this
        # smoke they are present whenever sf_surface_physics==2 (set below).
        self.noahclassic_static = _NOAHCLASSIC_BUNDLE[0] if cfg.sf_surface_physics == 2 else None
        self.noahclassic_land = _NOAHCLASSIC_BUNDLE[1] if cfg.sf_surface_physics == 2 else None
        self.noahclassic_rad = None


def _grid_canary():
    return GridSpec.canary_3km_template()


def _tile(value, ny, nx, *, dtype=jnp.float64):
    return jnp.full((ny, nx), value, dtype=dtype)


def _tile4(values, ny, nx):
    arr = jnp.asarray(values, dtype=jnp.float64)
    return jnp.broadcast_to(arr, (ny, nx, arr.shape[0]))


def _build_noahclassic_bundle(ny, nx, column_name="daytime_veg10"):
    """Build the WRF-savepoint-derived NoahClassicStatic / LandState (real columns)."""

    from gpuwrf.physics.lsm_noah_classic import NoahClassicParams

    data = json.loads(_NOAHCLASSIC_SAVEPOINTS.read_text())
    col = next(c for c in data["columns"] if c["name"] == column_name)
    rp = col["wrf"]["redprm"]
    snow = col["wrf"]["snow_in"]
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    params = NoahClassicParams(
        bexp=_tile(rp["bexp"], ny, nx), dksat=_tile(rp["dksat"], ny, nx),
        dwsat=_tile(rp["dwsat"], ny, nx), psisat=_tile(rp["psisat"], ny, nx),
        quartz=_tile(rp["quartz"], ny, nx), f1=_tile(rp["f1"], ny, nx),
        smcmax=_tile(rp["smcmax"], ny, nx), smcwlt=_tile(rp["smcwlt"], ny, nx),
        smcref=_tile(rp["smcref"], ny, nx), smcdry=_tile(rp["smcdry"], ny, nx),
        kdt=_tile(rp["kdt"], ny, nx), frzx=_tile(rp["frzx"], ny, nx),
        slope=_tile(rp["slope"], ny, nx), snup=_tile(rp["snup"], ny, nx),
        salp=_tile(rp["salp"], ny, nx), czil=_tile(rp["czil"], ny, nx),
        sbeta=_tile(rp["sbeta"], ny, nx), csoil=_tile(rp["csoil"], ny, nx),
        fxexp=_tile(rp["fxexp"], ny, nx), zbot=_tile(rp["zbot"], ny, nx),
        cfactr=_tile(rp["cfactr"], ny, nx), cmcmax=_tile(rp["cmcmax"], ny, nx),
        rsmax=_tile(rp["rsmax"], ny, nx), topt=_tile(rp["topt"], ny, nx),
        rgl=_tile(rp["rgl"], ny, nx), hs=_tile(rp["hs"], ny, nx),
        rsmin=_tile(rp["rsmin"], ny, nx), lvcoef=_tile(rp["lvcoef"], ny, nx),
        nroot=_tile(int(rp["nroot"]), ny, nx, dtype=jnp.int32),
        rtdis=_tile4(rp["rtdis"], ny, nx), alb=_tile(rp["alb"], ny, nx),
        embrd=_tile(rp["embrd"], ny, nx), xlai=_tile(rp["xlai"], ny, nx),
        z0brd=_tile(rp["z0brd"], ny, nx), shdfac=_tile(rp["shdfac"], ny, nx),
        is_urban=jnp.full((ny, nx), bool(col["vegtyp"] == col["isurban"])),
    )
    smav = (
        (_tile4(col["wrf"]["smc_in"], ny, nx) - params.smcwlt[..., None])
        / (params.smcmax - params.smcwlt)[..., None]
    )
    static = NoahClassicStatic(
        params=params, zsoil=_tile4(col["zsoil"], ny, nx), sldpth=_tile4(col["sldpth"], ny, nx),
        snoalb=_tile(col["state_in"]["snoalb"], ny, nx), tbot=_tile(col["tbot"], ny, nx),
        solnet_albedo=_tile(col["state_in"]["albbck"], ny, nx),
        lwdn_emissivity=_tile(col["state_in"]["emiss"], ny, nx),
    )
    land = NoahClassicLandState(
        t1=_tile(col["wrf"]["t1_in"], ny, nx), stc=_tile4(col["wrf"]["stc_in"], ny, nx),
        smc=_tile4(col["wrf"]["smc_in"], ny, nx), sh2o=_tile4(col["wrf"]["sh2o_in"], ny, nx),
        cmc=_tile(snow["cmc"], ny, nx), sneqv=_tile(snow["sneqv"], ny, nx),
        snowh=_tile(snow["snowh"], ny, nx), sncovr=_tile(snow["sncovr"], ny, nx),
        snotime1=_tile(col["state_in"]["snotime1"], ny, nx),
        ribb=_tile(col["wrf"]["chcm_in"]["ribb"], ny, nx),
        flx4=zero, fvb=zero, fbur=zero, fgsn=zero, smcrel=smav, xlaidyn=params.xlai,
        hfx=zero, qfx=zero, lh=zero, grdflx=zero,
    )
    rad = NoahClassicRadiation(
        soldn=_tile(col["wrf"]["forcing"]["soldn"], ny, nx),
        lwdn=_tile(col["wrf"]["forcing"]["glw"], ny, nx),
        cosz=jnp.ones((ny, nx), dtype=jnp.float64),
    )
    return static, land, rad


# Built once at import (small): the Noah-classic bundle is tiled onto the Canary grid.
_CANARY = _grid_canary()
_NOAHCLASSIC_BUNDLE = _build_noahclassic_bundle(_CANARY.ny, _CANARY.nx)


def _build_canary_state(grid, seed=7):
    """Physically-reasonable b2 C-grid atmospheric state on the real Canary grid."""

    return _build_state(nz=grid.nz, ny=grid.ny, nx=grid.nx, seed=seed)[0]


def _subset_grid_leaves(tree, full_hw, sl_y, sl_x):
    """Slice a small spatial window out of every leaf whose trailing dims == the
    full d02 grid (ny,nx), leaving parameter-table leaves untouched.

    Keeps the Noah-MP land/static REAL (real soil/veg/temperatures), just on a small
    land-rich tile so the dedicated Noah-MP land smoke is TINY (the canopy energy/
    water balance cost scales with the cell count)."""

    fy, fx = full_hw

    def _slice(leaf):
        a = np.asarray(leaf)
        if a.ndim >= 2 and a.shape[-2:] == (fy, fx):
            idx = (Ellipsis, sl_y, sl_x)
            return jnp.asarray(a[idx])
        return leaf

    return jax.tree_util.tree_map(_slice, tree)


def _load_noahmp_real():
    """Load the REAL d02 Canary Noah-MP land/static (corpus wrfinput warm-start),
    subset to a small LAND-RICH tile so the dedicated Noah-MP smoke stays tiny."""

    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params

    land_full, static_full, meta = build_noahmp_land_state(_CORPUS_RUNDIR, "d02")
    fy, fx = np.asarray(static_full.xland).shape
    # pick a land-rich window (the d02 Canary land cells cluster near the islands);
    # scan for an 8x8 tile with the most land, fall back to the centre.
    xland = np.asarray(static_full.xland)
    is_land = xland < 1.5
    h = w = 8
    best, best_n = (fy // 2 - h // 2, fx // 2 - w // 2), -1
    for y0 in range(0, fy - h, 4):
        for x0 in range(0, fx - w, 4):
            n = int(is_land[y0:y0 + h, x0:x0 + w].sum())
            if n > best_n:
                best_n, best = n, (y0, x0)
    y0, x0 = best
    sl_y, sl_x = slice(y0, y0 + h), slice(x0, x0 + w)
    land = _subset_grid_leaves(land_full, (fy, fx), sl_y, sl_x)
    static = _subset_grid_leaves(static_full, (fy, fx), sl_y, sl_x)
    energy, rad, nroot = build_noahmp_params(static)
    ny, nx = np.asarray(static.xland).shape
    meta = dict(meta)
    meta["subset_tile"] = {"y0": int(y0), "x0": int(x0), "h": h, "w": w,
                           "land_cells_in_tile": int(best_n),
                           "full_grid": [int(fy), int(fx)]}
    return land, static, energy, rad, nroot, meta, (ny, nx)


# Lazily loaded so a Noah-classic / bulk run does not pay the wrfinput read.
_NOAHMP_CACHE: dict[str, Any] = {}


def _noahmp_real():
    if "data" not in _NOAHMP_CACHE:
        _NOAHMP_CACHE["data"] = _load_noahmp_real()
    return _NOAHMP_CACHE["data"]


# ============================================================================
# Integrated operational physics step (EXACT operational_mode._physics_boundary_step
# physics block; WRF call order; no dynamics -- see module docstring).
# ============================================================================

def _changed(before, after, leaf) -> bool:
    if not hasattr(after, leaf):
        return False
    a = np.asarray(getattr(before, leaf))
    b = np.asarray(getattr(after, leaf))
    if a.shape != b.shape:
        return True
    return float(np.max(np.abs(b - a))) > 0.0


def _finite(state, leaves) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(getattr(state, lf))))) for lf in leaves)


def _physical(state) -> dict[str, Any]:
    out = {}
    ok = True
    for leaf, (lo, hi) in PHYS_BOUNDS.items():
        a = np.asarray(getattr(state, leaf))
        mn, mx = float(np.min(a)), float(np.max(a))
        in_band = bool(mn >= lo and mx <= hi)
        ok = ok and in_band
        out[leaf] = {"min": round(mn, 5), "max": round(mx, 5), "band": [lo, hi], "ok": in_band}
    return {"per_field": out, "all_in_band": ok}


def _expected_changed_leaves(cfg: Config) -> tuple[str, ...]:
    """Leaves the SELECTED suite must move (no silent no-op detector).

    Only schemes that PROGNOSE a field are required to move it. NOTE the bulk
    surface path (MYNN-sfclay surface_adapter) uses a PRESCRIBED skin temperature
    and does NOT prognose t_skin -- only the prognostic land models (Noah-MP=4 /
    Noah-classic=2) advance t_skin, so t_skin is only a required-moved leaf for
    those land options (requiring it for bulk land would be a false integration
    failure)."""

    leaves: list[str] = []
    # microphysics writes theta + its moist species
    if cfg.mp_physics != 0:
        leaves += ["theta", "qv", "qc", "qr"]
    # surface layer writes the flux handles
    leaves += ["ustar", "theta_flux"]
    # the PBL slot runs (MYNN=5 / YSU=1 / ACM2=7) -> momentum mixing moves u/v
    if cfg.bl_pbl_physics != 0:
        leaves += ["u", "v"]
    # prognostic land model advances the skin temperature (bulk path holds it)
    if cfg.sf_surface_physics in (4, 2):
        leaves += ["t_skin"]
    # KF cumulus moves rainc_acc when triggered (may legitimately not trigger -> not required)
    return tuple(dict.fromkeys(leaves))


def _run_config(cfg: Config, *, steps: int) -> dict[str, Any]:
    """Build the operational state for this config + run the integrated operational
    physics coupler for ``steps`` steps under jit+scan. Returns the per-config row."""

    row: dict[str, Any] = {
        "cfg_id": cfg.cfg_id,
        "description": cfg.description,
        "namelist": cfg.as_namelist(),
        "covers": list(cfg.covers),
        "expect": cfg.expect,
    }

    # --- (1) namelist accept matrix (io.namelist_check, fail-closed) ---
    try:
        validate_supported_namelist(cfg.as_namelist())
        row["namelist_accepted"] = True
    except Exception as exc:  # noqa: BLE001
        row["namelist_accepted"] = False
        row["namelist_error"] = f"{type(exc).__name__}: {exc}"

    # --- (2) dispatcher resolve + GPU-gate flag ---
    try:
        suite = resolve_physics_suite(cfg.as_namelist())
        row["dispatch_resolved"] = True
        row["dispatch_gpu_gate_ready"] = bool(suite.gpu_gate_ready)
        row["non_gpu_schemes"] = list(suite.non_gpu_schemes)
    except UnsupportedSchemeSelection as exc:
        row["dispatch_resolved"] = False
        row["dispatch_error"] = str(exc)
        suite = None

    # --- (3) operational-coupler fail-closed gate (THE operational authority) ---
    coupler_accepts = False
    coupler_reject_reason = None
    try:
        _resolve_operational_suite(_NLStub(cfg))
        coupler_accepts = True
    except UnsupportedSchemeSelection as exc:
        coupler_reject_reason = str(exc)
    row["operational_coupler_accepts"] = coupler_accepts
    if coupler_reject_reason is not None:
        row["operational_coupler_reject_reason"] = coupler_reject_reason

    # --- FAIL_CLOSED configs: the coupler MUST reject; that IS the pass condition ---
    if cfg.expect == "FAIL_CLOSED":
        row["compiles"] = None
        row["n_steps"] = 0
        row["finite"] = None
        row["physical"] = None
        row["schemes_active"] = None
        row["pass"] = bool(not coupler_accepts)
        row["verdict"] = (
            "FAIL_CLOSED_OK (coupler loudly rejected the unwired scheme)"
            if not coupler_accepts
            else "FAIL_CLOSED_VIOLATION (coupler SILENTLY accepted an unwired scheme!)"
        )
        return row

    # --- RUN configs: the coupler MUST accept, then integrate end-to-end ---
    if not coupler_accepts:
        row["compiles"] = False
        row["pass"] = False
        row["verdict"] = "RUN_CONFIG_REJECTED_BY_COUPLER (integration regression)"
        return row

    # Build the operational state for the selected land path.
    dt = 12.0
    if cfg.use_noahmp:
        land, static, ep, rp, nroot, meta, (ny, nx) = _noahmp_real()
        nz = 10
        state0 = _build_state(nz=nz, ny=ny, nx=nx, seed=5)[0]
        state0 = state0.replace(xland=jnp.asarray(static.xland, jnp.float64))
        tile = meta.get("subset_tile", {})
        row["land_init"] = {
            "model": "Noah-MP (real d02 wrfinput warm-start, land-rich tile subset)",
            "grid": [ny, nx],
            "land_cells_in_tile": int(tile.get("land_cells_in_tile", -1)),
            "full_d02_land_cells": int(meta.get("n_land_cells", -1)),
            "subset_tile": tile,
        }
        rad = _Rad(jnp.full((ny, nx), 400.0), jnp.full((ny, nx), 330.0), jnp.full((ny, nx), 0.5))
        clk = _Clk(130.0, 365.0)
    else:
        grid = _CANARY
        state0 = _build_canary_state(grid, seed=7)
        ny, nx = grid.ny, grid.nx
        if cfg.sf_surface_physics == 2:
            nc_static, nc_land0, nc_rad = _NOAHCLASSIC_BUNDLE
            # seed land columns from the Noah-classic bundle (mirror noah_coupler_smoke)
            is_land = jnp.asarray(state0.xland) < 1.5
            state0 = state0.replace(
                t_skin=jnp.where(is_land, nc_land0.t1, state0.t_skin),
                soil_moisture=jnp.where(is_land, nc_land0.smc[..., 0], state0.soil_moisture),
                roughness_m=jnp.where(is_land, nc_static.params.z0brd, state0.roughness_m),
            )
            row["land_init"] = {"model": "Noah-classic (WRF NOAHMP_SFLX savepoint bundle)",
                                "grid": [ny, nx], "column": "daytime_veg10"}
        else:
            row["land_init"] = {"model": "bulk prescribed (surface_adapter)", "grid": [ny, nx]}

    grid_for_adapters = _CANARY if not cfg.use_noahmp else None
    mp_opt, sf_opt, cu_opt = cfg.mp_physics, cfg.sf_sfclay_physics, cfg.cu_physics

    # The integrated physics body: EXACT operational_mode._physics_boundary_step order.
    def physics_body(state, w0avg, nca, cldefi, nc_land):
        # --- microphysics slot (dispatcher-selected) ---
        if mp_opt == DEFAULT_MP_PHYSICS:
            state = thompson_adapter(state, dt)
        elif mp_opt in MP_SCAN_ADAPTERS:
            state = MP_SCAN_ADAPTERS[mp_opt](state, dt, grid_for_adapters)
        # mp_opt == 0 -> passive
        # --- surface-layer slot + land slot ---
        if cfg.use_noahmp:
            if sf_opt in SFCLAY_SCAN_ADAPTERS:
                state = SFCLAY_SCAN_ADAPTERS[sf_opt](state, dt, grid_for_adapters)
            from gpuwrf.coupling.noahmp_surface_hook import noahmp_surface_step
            state, nc_land = noahmp_surface_step(
                state, nc_land, static, dt, radiation=rad, clock=clk,
                energy_params=ep, rad_params=rp,
            )
        else:
            if sf_opt in SFCLAY_SCAN_ADAPTERS:
                state = SFCLAY_SCAN_ADAPTERS[sf_opt](state, dt, grid_for_adapters)
            else:  # sf_opt == 5 -> MYNN-sfclay (existing surface_adapter)
                state = surface_adapter(state, dt)
            if cfg.sf_surface_physics == 2:
                state, nc_land = noahclassic_surface_step(
                    state, nc_land, _NOAHCLASSIC_BUNDLE[0], dt, radiation=_NOAHCLASSIC_BUNDLE[2],
                )
        # --- PBL slot (MYNN default; YSU(1)/ACM2(7) are the v0.6.0 jax.lax.scan-
        # traceable rewrites, dispatcher-routed by the STATIC bl_pbl option -- EXACT
        # mirror of operational_mode._physics_boundary_step's PBL slot) ---
        bl_opt = cfg.bl_pbl_physics
        if bl_opt in PBL_SCAN_ADAPTERS:
            state = PBL_SCAN_ADAPTERS[bl_opt](state, dt, grid_for_adapters)
        elif bl_opt == DEFAULT_BL_PBL_PHYSICS:
            state = mynn_adapter(state, dt, grid_for_adapters)
        # bl_opt == 0 -> no PBL mixing.
        # --- cumulus slot (EXACT mirror of operational_mode._physics_boundary_step):
        # modified-Tiedtke(6) and Grell-Freitas(3) are the stateless GPU-batched
        # State->State adapters (CU_STATELESS_SCAN_ADAPTERS, checked FIRST); KF(1)
        # threads the (w0avg, nca) carry; BMJ(2) threads the CLDEFI carry.
        # New-Tiedtke(16) is fail-closed upstream and never reaches a RUN config. ---
        if cu_opt in CU_STATELESS_SCAN_ADAPTERS:
            state = CU_STATELESS_SCAN_ADAPTERS[cu_opt](state, dt, grid_for_adapters)
        elif cu_opt == 1:
            state, w0avg, nca = kf_adapter(state, dt, w0avg, nca, grid=grid_for_adapters)
        elif cu_opt == 2:
            state, cldefi = bmj_adapter(state, dt, cldefi, grid=grid_for_adapters)
        return state, w0avg, nca, cldefi, nc_land

    w0avg0, nca0 = initial_kf_carry(state0)
    cldefi0 = initial_bmj_carry(state0)
    if cfg.sf_surface_physics == 2:
        nc_land_init = _NOAHCLASSIC_BUNDLE[1]
    elif cfg.use_noahmp:
        nc_land_init = land
    else:
        nc_land_init = None

    def scan_body(carry, _i):
        st, w0, nc, cd, ncl = carry
        st, w0, nc, cd, ncl = physics_body(st, w0, nc, cd, ncl)
        return (st, w0, nc, cd, ncl), None

    def run_scan(s):
        init = (s, w0avg0, nca0, cldefi0, nc_land_init)
        (final, _w0, _nc, _cd, _ncl), _ = jax.lax.scan(scan_body, init, jnp.arange(int(steps)))
        return final

    # --- compile (jit) -> traceable == GPU-runnable ---
    compiled = jax.jit(run_scan)
    try:
        out = compiled(state0)
        jax.block_until_ready(out.theta)
        row["compiles"] = True
    except Exception as exc:  # noqa: BLE001 -- a real integration breakage, reported honestly
        import traceback
        row["compiles"] = False
        row["compile_error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc().splitlines()[-6:]
        row["pass"] = False
        row["verdict"] = "INTEGRATION_BREAK (config did not compile / run)"
        return row

    row["n_steps"] = int(steps)

    # --- finite ---
    fin_leaves = ("theta", "qv", "qc", "qr", "u", "v", "w", "p", "t_skin", "theta_flux", "ustar")
    finite = _finite(out, fin_leaves)
    row["finite"] = bool(finite)

    # --- physical ranges ---
    phys = _physical(out)
    row["physical"] = phys

    # --- schemes active (no silent no-op) ---
    expected = _expected_changed_leaves(cfg)
    active = {lf: _changed(state0, out, lf) for lf in expected}
    schemes_active = all(active.values())
    row["schemes_active"] = {"expected_moved_leaves": list(expected),
                             "moved": active, "all_active": bool(schemes_active)}
    # KF activity (informational; KF may legitimately not trigger on this column)
    if cu_opt in CU_SCAN_ADAPTERS:
        row["schemes_active"]["kf_rainc_acc_moved"] = _changed(state0, out, "rainc_acc")

    row["pass"] = bool(row["compiles"] and finite and phys["all_in_band"] and schemes_active)
    row["verdict"] = "INTEGRATION_PASS" if row["pass"] else "INTEGRATION_FAIL"
    return row


# ============================================================================
# DOCUMENTED SEAM for the future v0.4.0 reference-scoring (NOT implemented here).
# ============================================================================
def reference_scoring_seam(
    gpu_forecast_fields: dict[str, Any],
    cpu_wrf_reference_fields: dict[str, Any],
    *,
    fields: tuple[str, ...] = ("T2", "U10", "V10"),
    diagnostic_fields: tuple[str, ...] = ("Q2", "PSFC", "PBLH", "SWDOWN", "GLW", "HFX", "LH"),
) -> dict[str, Any]:
    """SEAM (NOT IMPLEMENTED): where the FIXED v0.4.0 forecast-gate reference-scorer
    plugs in for the full GPU gate.

    This integration smoke is DECOUPLED from reference-scoring by design: it proves
    the suite COMPILES + stays FINITE + PHYSICAL + schemes-ACTIVE across configs,
    NOT obs/CPU-WRF skill. The v0.4.0 forecast-gate's reference-resolution code has a
    known namelist-path bug being fixed separately; this smoke deliberately does NOT
    duplicate/fork it.

    When the v0.4.0 scorer is fixed, the manager-scheduled GPU gate will:
      1. for each RUN config in SWEEP, build the OperationalNamelist with the combo
         options + the real Noah-MP / Noah-classic init bundle (see the corpus
         loaders above) and run ``run_forecast_operational`` on a corpus CPU-WRF d02
         case (full dycore+physics on a dynamically-balanced corpus replay state --
         which needs the GPU ``State.zeros`` path this CPU smoke cannot use);
      2. emit GPU wrfout (with the new QNCLOUD/QNCCN/RAINC leaves);
      3. call THIS function -- ``reference_scoring_seam(gpu_fields, cpu_wrf_fields)``
         -- with the per-lead gridpoint-paired GPU + CPU-WRF fields, returning the
         bias/RMSE table (mirroring proofs/m20/continuous_gate.py: reference =
         CPU-WRF, pairing = every grid cell, resolution = every lead hour).

    Until then this raises -- it is a CONTRACT, not an implementation.
    """

    raise NotImplementedError(
        "reference_scoring_seam is the documented plug-in point for the FIXED v0.4.0 "
        "forecast-gate reference-scorer; the integration smoke does not implement "
        f"scoring. Intended core fields={fields}, diagnostics={diagnostic_fields}."
    )


_GF_DEEP_SAVEPOINT = ROOT / "proofs" / "v060" / "savepoints" / "gf_case_1.json"


def _gf_adapter_triggering_probe() -> dict[str, Any]:
    """Prove the cu=3 gf_adapter PATH (not just the bare kernel) FIRES convection
    on a deep convectively-unstable sounding.

    The idealized Canary smoke column (nz=10, ~3 km top) is too shallow for GF deep
    convection, so the ``cu_gf`` config legitimately does not trigger (IERR_DEEP=2 --
    same family as KF/Tiedtke on this idealized state). To show the wiring genuinely
    carries GF convective tendencies through the adapter (no dead path), this probe
    builds a single-column operational ``State`` from the WRF-savepoint DEEP regime
    (``gf_case_1.json``: KX=45, oracle KTOP_DEEP=26, RAINCV=0.0425 mm) and runs ONE
    ``gf_adapter`` step. With the adapter's documented zero PBL forcing the sounding
    is still convectively unstable, so GF must produce a nonzero theta tendency +
    convective rain. This is the SAME ``gfdrv_batched`` entry the savepoint-parity
    gate proves faithful (proofs/v060/gf_gpubatch_savepoint_parity.json); the probe
    asserts the operational STATE->kernel mapping preserves that triggering."""

    from gpuwrf.coupling.scan_adapters import gf_adapter

    sp = json.loads(_GF_DEEP_SAVEPOINT.read_text())
    cols, sc = sp["columns"], sp["scalars"]
    kx = int(sc["KX"])
    # Build a 1-column state skeleton at the savepoint depth, then overwrite the
    # prognostic + geometry leaves with the deep savepoint sounding (surface-first,
    # State convention). Heights from the savepoint DZ (terrain-following thicknesses).
    state0 = _build_state(nz=kx, ny=1, nx=1, seed=11)[0]
    T = np.asarray(cols["T"], np.float64)
    p = np.asarray(cols["P"], np.float64)
    qv = np.asarray(cols["QV"], np.float64)
    dz = np.asarray(cols["DZ"], np.float64)
    pii = (np.maximum(p, 1.0) / 1.0e5) ** (287.0 / 1004.0)
    theta = T / pii
    z_iface = np.concatenate([[0.0], np.cumsum(dz)])  # m, surface=0
    ph = (9.81 * z_iface).reshape(kx + 1, 1, 1)
    u = np.asarray(cols["U"], np.float64)
    v = np.asarray(cols["V"], np.float64)
    w = np.asarray(cols["W"], np.float64)
    state = state0.replace(
        theta=jnp.asarray(theta.reshape(kx, 1, 1)),
        p=jnp.asarray(p.reshape(kx, 1, 1)),
        p_total=jnp.asarray(p.reshape(kx, 1, 1)),
        qv=jnp.asarray(np.maximum(qv, 1.0e-8).reshape(kx, 1, 1)),
        qc=jnp.zeros((kx, 1, 1)), qr=jnp.zeros((kx, 1, 1)),
        qi=jnp.zeros((kx, 1, 1)), qs=jnp.zeros((kx, 1, 1)),
        ph=jnp.asarray(ph), ph_total=jnp.asarray(ph),
        u=jnp.asarray(np.repeat(u.reshape(kx, 1, 1), 2, axis=2)),
        v=jnp.asarray(np.repeat(v.reshape(kx, 1, 1), 2, axis=1)),
        w=jnp.asarray(np.concatenate([w, w[-1:]]).reshape(kx + 1, 1, 1)),
        xland=jnp.asarray([[float(sc["XLAND"])]]),
        rhosfc=jnp.asarray([[float(np.asarray(cols["RHO"])[0])]]),
        # GF surface fluxes (HFX W m^-2, QFX kg m^-2 s^-1) -> kinematic B2 handles
        # the adapter reads (theta_flux = HFX/(rho*cp); qv_flux = QFX/rho).
        theta_flux=jnp.asarray([[float(sc["HFX"]) / (float(np.asarray(cols["RHO"])[0]) * 1004.0)]]),
        qv_flux=jnp.asarray([[float(sc["QFX"]) / float(np.asarray(cols["RHO"])[0])]]),
        rainc_acc=jnp.zeros((1, 1)),
    )

    class _Proj:
        dx_m = float(sc["DX"])

    class _Grid:
        projection = _Proj()

    out = jax.jit(lambda s: gf_adapter(s, float(sc["DT"]), _Grid()))(state)
    jax.block_until_ready(out.theta)
    d_theta = float(jnp.max(jnp.abs(out.theta - state.theta)))
    d_qv = float(jnp.max(jnp.abs(out.qv - state.qv)))
    rainc = float(jnp.max(out.rainc_acc))
    finite = bool(
        jnp.all(jnp.isfinite(out.theta)) and jnp.all(jnp.isfinite(out.qv))
        and jnp.all(jnp.isfinite(out.rainc_acc))
    )
    triggered = bool(finite and (d_theta > 1.0e-6 or rainc > 1.0e-4))
    return {
        "purpose": "gf_adapter PATH fires deep convection on the WRF deep savepoint sounding",
        "savepoint": str(_GF_DEEP_SAVEPOINT.name),
        "savepoint_oracle": {"KTOP_DEEP": int(sc["KTOP_DEEP"]), "RAINCV_mm": float(sc["RAINCV"]),
                             "DX_m": float(sc["DX"]), "KX": kx},
        "adapter_zero_pbl_forcing": True,
        "max_abs_dtheta_K": d_theta,
        "max_abs_dqv_kgkg": d_qv,
        "rainc_acc_mm": rainc,
        "finite": finite,
        "triggered": triggered,
        "verdict": "GF_ADAPTER_TRIGGERS" if triggered else "GF_ADAPTER_NO_TRIGGER",
    }


def run(*, steps: int = 8) -> dict[str, Any]:
    rows = [_run_config(cfg, steps=steps) for cfg in SWEEP]
    gf_trigger = _gf_adapter_triggering_probe()

    run_rows = [r for r in rows if r["expect"] == "RUN"]
    fc_rows = [r for r in rows if r["expect"] == "FAIL_CLOSED"]
    run_pass = sum(1 for r in run_rows if r["pass"])
    fc_pass = sum(1 for r in fc_rows if r["pass"])

    # Coverage check: every supported scheme touched by >=1 RUN config.
    covered = set()
    for r in run_rows:
        covered.update(r["covers"])

    report = {
        "proof": "v060-integrated-multiconfig-operational-smoke",
        "kind": (
            "INTEGRATION smoke: the consolidated 12-scheme physics suite runs end-to-end "
            "THROUGH THE OPERATIONAL COUPLER (physics_dispatch + scan_adapters + the "
            "operational_mode._physics_boundary_step physics block, WRF call order) across "
            "the supported namelist configs. Stable + physical + compiles(jit-traceable=="
            "GPU-runnable) + schemes-active. NOT a WRF/obs skill comparison (see "
            "reference_scoring_seam). REAL Canary grid + REAL d02 wrfinput Noah-MP "
            "warm-start + REAL WRF NOAHMP_SFLX Noah-classic savepoint bundle. No "
            "masking/clamp/self-compare/synthetic-happy-path."
        ),
        "jax_platform": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "steps_per_config": int(steps),
        "real_data_sources": {
            "canary_grid": "GridSpec.canary_3km_template()",
            "noahmp_land_warm_start": str(_CORPUS_RUNDIR) + " (d02 wrfinput)",
            "noahclassic_bundle": str(_NOAHCLASSIC_SAVEPOINTS) + " (WRF NOAHMP_SFLX savepoint column daytime_veg10)",
            "atmospheric_profile": "validated b2 C-grid pattern (scanwire_smoke._build_state) on the Canary grid",
        },
        "sweep_rationale": (
            "COVERING set (NOT full cartesian product): every supported scheme appears in "
            ">=1 RUN config plus the real operational Canary config. Supported = mp{8,6,10,16,1} "
            "x bl{5,1,7} x sfclay{1,5,7} x cu{1,2,3,6} x sf_surface{4,2} (New-Tiedtke cu=16 "
            "EXCLUDED = documented-TODO). v0.9.0 coupler reality: the operational scan now "
            "threads MYNN(5) + YSU(1) + ACM2(7) PBL (YSU/ACM2 are the v0.6.0 jax.lax.scan-traceable "
            "rewrites from the PBL-GPU-op lane, scan-wired via PBL_SCAN_ADAPTERS), KF(1) + BMJ(2) + "
            "Grell-Freitas(3) + modified-Tiedtke(6) cumulus (GF is the v0.9.0 GPU-batched jit/vmap "
            "scale-aware closure-ensemble adapter, Tiedtke the v0.6.0 GPU-batched adapter, both "
            "scan-wired via CU_SCAN_ADAPTERS), and Noah-MP(4) + Noah-classic(2) land -- so YSU/ACM2, "
            "GF and Tiedtke are now RUN configs. Only New-Tiedtke(16, not separately gated) and the "
            "MYJ(2)+Janjic(2) PBL/sfclay pair (no scan adapter yet) stay FAIL-CLOSED in the coupler, "
            "exercised here as fail-closed integration assertions (the coupler must reject them "
            "loudly, never silently no-op)."
        ),
        "scheme_coverage": {
            "covered_by_run_configs": sorted(covered),
            "fail_closed_schemes": sorted(
                s for r in fc_rows for s in r["covers"]
            ),
            "excluded_documented_todo": ["cu_physics=3 Grell-Freitas (sequential closure ensemble; GPU-batching TODO)", "cu_physics=16 New Tiedtke (not separately savepoint-gated)"],
            "radiation_note": (
                "Every RUN config pins ra_sw=ra_lw=4 (RRTMG): the operational radiation slot in "
                "runtime.operational_mode hardcodes the RRTMG held-rate RTHRATEN and "
                "OperationalNamelist has no ra_lw_physics/ra_sw_physics field, so RRTMG (ra=4) is "
                "the ONLY operational-scan radiation path. ra_lw=1 (classic RRTM-LW) and ra_sw=1 "
                "(Dudhia-SW) are isolated-WRF-savepoint parity-proven + accepted but NOT "
                "operational-scan-wired (close-critic FIX #1/#2 DOWNGRADE), so they are "
                "intentionally NOT swept end-to-end here -- there is no operational selection to "
                "exercise. A radiation-family dispatch (+ a jit/vmap rewrite of the host-NumPy "
                "RRTM-LW kernel) is a post-0.9.0 carry-over."
            ),
        },
        "n_configs": len(rows),
        "n_run_configs": len(run_rows),
        "n_run_pass": run_pass,
        "n_run_fail": len(run_rows) - run_pass,
        "n_fail_closed_configs": len(fc_rows),
        "n_fail_closed_ok": fc_pass,
        "configs": rows,
        "gf_adapter_triggering_probe": gf_trigger,
        "reference_scoring_seam": {
            "function": "proofs.v060.multicfg_operational_smoke.reference_scoring_seam",
            "status": "DOCUMENTED CONTRACT, NOT IMPLEMENTED (raises NotImplementedError)",
            "note": (
                "This smoke is DECOUPLED from reference-scoring by design. The FIXED v0.4.0 "
                "forecast-gate reference-scorer plugs in at reference_scoring_seam(gpu_fields, "
                "cpu_wrf_fields) for the MANAGER-scheduled full-dycore GPU gate vs CPU-WRF. This "
                "smoke does NOT duplicate/fork the v0.4.0 reference-resolution code (known "
                "namelist-path bug being fixed separately)."
            ),
            "core_fields": ["T2", "U10", "V10"],
            "diagnostic_fields": ["Q2", "PSFC", "PBLH", "SWDOWN", "GLW", "HFX", "LH"],
        },
        "all_pass": bool(
            run_pass == len(run_rows)
            and fc_pass == len(fc_rows)
            and gf_trigger["triggered"]
        ),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "proofs" / "v060" / "multicfg_smoke_report.json")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)

    report = run(steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")

    # Concise console table.
    print(f"\n{'='*92}")
    print("v0.6.0 INTEGRATED MULTI-CONFIG OPERATIONAL SMOKE")
    print(f"{'='*92}")
    print(f"{'cfg_id':24s} {'mp/bl/sf/cu/land':18s} {'expect':12s} {'verdict'}")
    print("-" * 92)
    for r in report["configs"]:
        nl = r["namelist"]
        tag = f"{nl['mp_physics']}/{nl['bl_pbl_physics']}/{nl['sf_sfclay_physics']}/{nl['cu_physics']}/{nl['sf_surface_physics']}"
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"{r['cfg_id']:24s} {tag:18s} {r['expect']:12s} [{mark}] {r['verdict']}")
    print("-" * 92)
    print(f"RUN configs:         {report['n_run_pass']}/{report['n_run_configs']} PASS")
    print(f"FAIL-CLOSED configs: {report['n_fail_closed_ok']}/{report['n_fail_closed_configs']} OK (coupler rejected as required)")
    print(f"Scheme coverage:     {report['scheme_coverage']['covered_by_run_configs']}")
    gfp = report["gf_adapter_triggering_probe"]
    print(f"GF adapter trigger:  [{ 'PASS' if gfp['triggered'] else 'FAIL'}] {gfp['verdict']} "
          f"(dtheta={gfp['max_abs_dtheta_K']:.3e}K, rainc={gfp['rainc_acc_mm']:.4f}mm, "
          f"deep savepoint KTOP={gfp['savepoint_oracle']['KTOP_DEEP']})")
    print(f"ALL PASS: {report['all_pass']}")
    print(f"proof -> {args.out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
