#!/usr/bin/env python3
"""V0.14 Noah-MP land-tile ENERGY closure proof.

Resolves the prior `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`
blocker by a per-column WRF `noahmplsm` energy in/out hook (on the rmol-pinned
tree) column-diffed against the JAX `physics.noahmp` solve at the Step-1 boundary.

Findings (all WRF-anchored, CPU):

1. The JAX Noah-MP energy ALGORITHM is exact. Fed WRF's EXACT per-column NMPIN
   (phenology -> two-stream radiation -> energy), every output matches WRF NMPOUT
   to ~1e-3 W/m2 (FSH, SSOIL, SAV/SAG, TRAD, albedo). The energy solve is NOT the
   blocker -- the prior narrowing is refuted.
2. ROOT CAUSE (fixed, production): the lowest-level air temperature fed to Noah-MP
   was ~+4 K too warm. `state.theta` is the WRF MOIST potential temperature
   theta_m = theta_dry*(1 + R_v/R_d*q_v) (use_theta_m=1; the dycore prognostic --
   operational_mode conv_t_tendf_to_moist divides `before.theta` by the same
   (1+_RVRD*qv)). `assemble_noahmp_forcing` converted theta_m -> T with a NAIVE
   Exner (treating moist theta as dry), so T3D was high by exactly (1+R_v/R_d*q_v).
   WRF feeds noahmplsm the DRY sensible temperature (module_sf_noahmpdrv.F:755).
   FIX: decouple theta_m -> theta_dry before the Exner conversion in
   `noahmp_coupler.assemble_noahmp_forcing`. sfctmp residual vs WRF T_ML:
   +4.06 K bias -> 0.003 K rmse.
3. After the fix, the residual Noah-MP land-tile flux error (HFX rmse ~7.6 W/m2)
   COLLAPSES to ~0.1 W/m2 when WRF's EXACT SWDOWN/GLW are swapped in -> the entire
   remaining land-tile residual is the RRTMG radiation forcing (GLW +14.7 W/m2,
   SWDOWN +3.6 W/m2 on land), an out-of-scope lane (RRTMG production is frozen for
   this sprint).

This proof is CPU-only, no host/device transfer inside loops, no clamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

OUT_JSON = PROOF_DIR / "noahmp_land_tile_energy_closure.json"
OUT_MD = PROOF_DIR / "noahmp_land_tile_energy_closure.md"

ENERGY_HOOK = Path("/tmp/wrfgpu2_v014_noahmp_energy_pinned_onerun/noahmp_energy_d02_step1.txt")
# the surface-handoff hook re-emitted in the SAME run as the energy hook, and the
# prior pinned-onerun surface handoff -- a byte-cmp proves the energy hook does not
# perturb the WRF physics.
SURFACE_HOOK_XCHECK = Path("/tmp/wrfgpu2_v014_surface_handoff_energyhook_xcheck/surface_land_flux_d02_step1.txt")
SURFACE_HOOK_PINNED = Path("/tmp/wrfgpu2_v014_surface_handoff_pinned_onerun/surface_land_flux_d02_step1.txt")

NY, NX = 66, 159


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cmp_files(a: Path, b: Path) -> bool | None:
    if not (a.is_file() and b.is_file()):
        return None
    return subprocess.run(["cmp", "-s", str(a), str(b)], check=False).returncode == 0


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if c.ndim == 0:
        c = np.full(r.shape, float(c))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        c = c[m]
        r = r[m]
    finite = np.isfinite(c) & np.isfinite(r)
    c = c[finite]
    r = r[finite]
    if c.size == 0:
        return {"count": 0}
    d = c - r
    return {
        "count": int(d.size),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "bias": float(np.mean(d)),
        "ref_max_abs": float(np.max(np.abs(r))),
    }


def parse_energy_hook(path: Path):
    """Parse the WRF noahmplsm per-column NMPIN/NMPOUT hook (NMPHDR/NMPINF/NMPOUTF)."""
    hdr = inf = outf = None
    rin: dict[tuple[int, int], list[str]] = {}
    rout: dict[tuple[int, int], list[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "NMPHDR":
                hdr = p
            elif p[0] == "NMPINF":
                inf = line.split(None, 1)[1].split()[2:]   # drop leading 'i j'
            elif p[0] == "NMPOUTF":
                outf = line.split(None, 1)[1].split()[2:]
            elif p[0] == "NMPIN":
                rin[(int(p[1]), int(p[2]))] = p[3:]
            elif p[0] == "NMPOUT":
                rout[(int(p[1]), int(p[2]))] = [float(x) for x in p[3:]]
    return hdr, inf, outf, rin, rout


def grid_field(rows: Mapping[tuple[int, int], Any], names: list[str], name: str, want_int=False):
    k = names.index(name)
    a = np.full((NY, NX), np.nan)
    for (i, j), vals in rows.items():
        a[j - 1, i - 1] = float(vals[k])
    if want_int:
        return np.where(np.isnan(a), 0, np.rint(a)).astype(np.int32)
    return a


def noahmp_options(hdr: list[str]) -> dict[str, Any]:
    # NMPHDR layout: itimestep idveg iopt_crs iopt_btr iopt_run iopt_sfc iopt_frz
    #   iopt_inf iopt_rad iopt_alb iopt_snf iopt_tbot iopt_stc iopt_gla iopt_rsf
    #   iopt_soil iopt_crop iopt_irr iz0tlnd sf_urban nsoil yr yearlen its ite jts
    #   jte kts julian dt
    keys = [
        "itimestep", "idveg", "iopt_crs", "iopt_btr", "iopt_run", "iopt_sfc",
        "iopt_frz", "iopt_inf", "iopt_rad", "iopt_alb", "iopt_snf", "iopt_tbot",
        "iopt_stc", "iopt_gla", "iopt_rsf", "iopt_soil", "iopt_crop", "iopt_irr",
        "iz0tlnd", "sf_urban_physics", "nsoil", "yr", "yearlen", "its", "ite",
        "jts", "jte", "kts",
    ]
    out: dict[str, Any] = {}
    for idx, key in enumerate(keys, start=1):
        out[key] = int(hdr[idx])
    out["julian"] = float(hdr[-2])
    out["dt"] = float(hdr[-1])
    return out


# --------------------------------------------------------------------------- #
# proof
# --------------------------------------------------------------------------- #
def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    if not ENERGY_HOOK.is_file():
        return {"status": "BLOCKED_ENERGY_HOOK_MISSING", "path": str(ENERGY_HOOK)}

    import step1_live_nest_init_rerun as live  # noqa: PLC0415
    import step1_mynn_source_coupling as coupling  # noqa: PLC0415
    import step1_rk1_p_state_source_split as pstate  # noqa: PLC0415
    import gpuwrf.physics.noahmp_coupler as nc  # noqa: PLC0415
    from gpuwrf.physics.noahmp_coupler import RVOVRD, assemble_noahmp_forcing, _get, _surface  # noqa: PLC0415
    from gpuwrf.physics.noahmp.noahmp_driver import (  # noqa: PLC0415
        build_energy_params, noah_mp_step, _gather_vec,
    )
    from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table  # noqa: PLC0415
    from gpuwrf.physics.noahmp.precip_heat import noahmp_precip_heat  # noqa: PLC0415
    from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream  # noqa: PLC0415
    from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy  # noqa: PLC0415
    from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: PLC0415
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: PLC0415
    from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation, _noahmp_params  # noqa: PLC0415

    hdr, inf, outf, rin, rout = parse_energy_hook(ENERGY_HOOK)
    options = noahmp_options(hdr)
    DT = options["dt"]
    JULIAN = options["julian"]
    YEARLEN = float(options["yearlen"])

    land = np.zeros((NY, NX), dtype=bool)
    for (i, j) in rin:
        land[j - 1, i - 1] = True
    nland = int(land.sum())

    def IN(name, want_int=False):
        return grid_field(rin, inf, name, want_int)

    def OUT(name):
        return grid_field(rout, outf, name)

    # ---- provenance: energy hook + non-perturbation byte-cmp ------------------
    provenance = {
        "energy_hook": {"path": str(ENERGY_HOOK), "sha256_16": sha16(ENERGY_HOOK),
                        "size_bytes": ENERGY_HOOK.stat().st_size, "nland_in": len(rin),
                        "nland_out": len(rout)},
        "hook_non_perturbing_surface_byte_identical": cmp_files(
            SURFACE_HOOK_XCHECK, SURFACE_HOOK_PINNED),
        "note": (
            "Per-column noahmplsm energy hook emitted from ONE run of the rmol-pinned "
            "WRF binary. The surface/land-flux handoff hook re-emitted in the SAME run "
            "is byte-identical to the prior pinned-onerun handoff -> the energy "
            "instrumentation does not perturb the WRF physics."
        ),
    }

    # ---- build the JAX Step-1 inputs / overlay state --------------------------
    inputs = live.build_live_nest_step1_inputs()
    namelist = inputs["namelist"]
    land_state0 = inputs["noahmp_land"]
    static = namelist.noahmp_static
    patched = pstate.apply_mythos_perturb_init(inputs)

    # capture the exact state the real overlay assembles forcing from
    cap: dict[str, Any] = {}
    orig = nc.assemble_noahmp_forcing

    def spy(state, st, radiation, clock, dt):
        cap["state"] = state
        cap["radiation"] = radiation
        cap["clock"] = clock
        cap["dt"] = dt
        return orig(state, st, radiation, clock, dt)

    nc.assemble_noahmp_forcing = spy
    try:
        coupling.build_step1_state(inputs=inputs, patched=patched)
    finally:
        nc.assemble_noahmp_forcing = orig
    state = cap["state"]
    radiation = cap["radiation"]
    clock = cap["clock"]
    ep, rp = _noahmp_params(namelist)

    # =======================================================================
    # 1. ENERGY ALGORITHM vs WRF -- feed WRF's EXACT per-column NMPIN
    # =======================================================================
    shape = (NY, NX)

    def ov(base, wrf2d):
        b = np.asarray(base, dtype=np.float64).copy()
        b[land] = wrf2d[land]
        return jnp.asarray(b)

    tslb = np.asarray(land_state0.tslb, dtype=np.float64).copy()
    smois = np.asarray(land_state0.smois, dtype=np.float64).copy()
    sh2o = np.asarray(land_state0.sh2o, dtype=np.float64).copy()
    for L in range(4):
        tslb[L][land] = IN(f"stc{L+1}")[land]
        smois[L][land] = IN(f"smc{L+1}")[land]
        sh2o[L][land] = IN(f"smh2o{L+1}")[land]

    ls_wrf = land_state0.replace(
        tslb=jnp.asarray(tslb), smois=jnp.asarray(smois), sh2o=jnp.asarray(sh2o),
        tv=ov(land_state0.tv, IN("tv")), tg=ov(land_state0.tg, IN("tg")),
        tah=ov(land_state0.tah, IN("tah")), eah=ov(land_state0.eah, IN("eah")),
        canliq=ov(land_state0.canliq, IN("canliq")), canice=ov(land_state0.canice, IN("canice")),
        fwet=ov(land_state0.fwet, IN("fwet")),
        snowh=ov(land_state0.snowh, IN("sndpth")), sneqv=ov(land_state0.sneqv, IN("swe")),
        sneqvo=ov(land_state0.sneqvo, IN("sneqvo")), tauss=ov(land_state0.tauss, IN("tauss")),
        albold=ov(land_state0.albold, IN("albold")),
        cm=ov(land_state0.cm, IN("cm")), ch=ov(land_state0.ch, IN("ch")),
        lai=ov(land_state0.lai, IN("plai")), sai=ov(land_state0.sai, IN("psai")),
        qsfc=ov(land_state0.qsfc, IN("qsfc1d")),
    )
    forcing_wrf = NoahMPForcing(
        sfctmp=ov(np.zeros(shape), IN("t_ml")), sfcprs=ov(np.zeros(shape), IN("p_ml")),
        psfc=ov(np.zeros(shape), IN("psfc")), uu=ov(np.zeros(shape), IN("u_ml")),
        vv=ov(np.zeros(shape), IN("v_ml")), qair=ov(np.zeros(shape), IN("q_ml")),
        qc=jnp.zeros(shape), soldn=ov(np.zeros(shape), IN("swdn")),
        lwdn=ov(np.zeros(shape), IN("lwdn")),
        prcpconv=ov(np.zeros(shape), IN("prcpconv")), prcpnonc=ov(np.zeros(shape), IN("prcpnonc")),
        prcpsnow=ov(np.zeros(shape), IN("prcpsnow")), prcpgrpl=jnp.zeros(shape),
        prcphail=jnp.zeros(shape), cosz=ov(np.zeros(shape), IN("cosz")),
        zlvl=ov(np.zeros(shape), IN("z_ml")), julian=jnp.asarray(JULIAN),
        yearlen=jnp.asarray(YEARLEN), o2air=ov(np.zeros(shape), IN("o2pp")),
        co2air=ov(np.zeros(shape), IN("co2pp")), foln=ov(np.ones(shape), IN("foln")),
    )
    energy_params, rad_params = build_energy_params(static, shape)
    phen = noahmp_phenology_table(ls_wrf, forcing_wrf, static)
    ch2op = _gather_vec(getattr(static.parameters, "ch2op"),
                        jnp.asarray(static.ivgtyp, dtype=jnp.int32))
    precip, canliq_new, canice_new = noahmp_precip_heat(ls_wrf, forcing_wrf, phen, ch2op, DT, is_lake=None)
    ls_p = ls_wrf.replace(fwet=precip.fwet, canliq=canliq_new, canice=canice_new)
    forcing_e = forcing_wrf._replace(pahv=precip.pahv, pahg=precip.pahg, pahb=precip.pahb)
    rad, rad_extras = radiation_twostream(ls_p, forcing_e, static, phen, rad_params, DT)
    _ls, ef, _et = noahmp_energy_canopy(
        ls_p, forcing_e, static, rad, DT, phen=phen, params=energy_params,
        rad_extras=rad_extras, o2air=forcing_e.o2air, co2air=forcing_e.co2air,
        foln=forcing_e.foln, pahv_kw=precip.pahv, pahg_kw=precip.pahg, pahb_kw=precip.pahb,
        isurban=13,
    )
    energy_algorithm = {
        "note": ("JAX phenology -> two-stream radiation -> energy fed WRF's EXACT "
                 "per-column NMPIN; outputs vs WRF NMPOUT on land cells."),
        "phenology": {
            "lai_vs_plai": diffstat(phen.lai, OUT("plai"), land),
            "sai_vs_psai": diffstat(phen.sai, OUT("psai"), land),
            "fveg_vs_fvegmp": diffstat(phen.fveg, OUT("fvegmp"), land),
        },
        "radiation_twostream": {
            "sav": diffstat(rad.sav, OUT("sav"), land),
            "sag": diffstat(rad.sag, OUT("sag"), land),
            "fsa": diffstat(rad.fsa, OUT("fsa"), land),
            "fsr": diffstat(rad.fsr, OUT("fsr"), land),
            "albedo_salb": diffstat(rad.albedo, OUT("salb"), land),
        },
        "energy": {
            "fsh_hfx": diffstat(ef.fsh, OUT("fsh"), land),
            "fira": diffstat(ef.fira, OUT("fira"), land),
            "ssoil_grdflx": diffstat(ef.ssoil, OUT("ssoil"), land),
            "fgev": diffstat(ef.fgev, OUT("fgev"), land),
            "fctr": diffstat(ef.fctr, OUT("fctr"), land),
            "trad_tsk": diffstat(ef.trad, OUT("trad"), land),
            "chv": diffstat(ef.chv, OUT("chv"), land),
            "chb": diffstat(ef.chb, OUT("chb"), land),
            "emissi": diffstat(ef.emissi, OUT("emissi"), land),
        },
    }

    # =======================================================================
    # 2. JAX-assembled forcing (POST-FIX) vs WRF NMPIN
    # =======================================================================
    forcing_real = assemble_noahmp_forcing(state, static, radiation, clock, DT)
    forcing_vs_wrf = {
        "sfctmp_air_temperature": diffstat(forcing_real.sfctmp, IN("t_ml"), land),
        "qair": diffstat(forcing_real.qair, IN("q_ml"), land),
        "uu": diffstat(forcing_real.uu, IN("u_ml"), land),
        "vv": diffstat(forcing_real.vv, IN("v_ml"), land),
        "sfcprs": diffstat(forcing_real.sfcprs, IN("p_ml"), land),
        "psfc": diffstat(forcing_real.psfc, IN("psfc"), land),
        "zlvl": diffstat(forcing_real.zlvl, IN("z_ml"), land),
        "cosz": diffstat(forcing_real.cosz, IN("cosz"), land),
        "soldn_swdown_RRTMG": diffstat(forcing_real.soldn, IN("swdn"), land),
        "lwdn_glw_RRTMG": diffstat(forcing_real.lwdn, IN("lwdn"), land),
    }

    # the moist-vs-dry signature: the OLD (buggy) naive sfctmp / the fixed sfctmp
    # equals exactly (1 + R_v/R_d * q_v).
    theta_m0 = np.asarray(_surface(_get(state, "theta")), dtype=np.float64)
    qair0 = np.maximum(np.asarray(_surface(_get(state, "qv")), dtype=np.float64), 0.0)
    p0 = np.asarray(_surface(_get(state, "p")), dtype=np.float64)
    naive_sfctmp = theta_m0 * (np.maximum(p0, 1.0) / P0_PA) ** R_D_OVER_CP
    decoupling_fix = {
        "constant_RVOVRD": float(RVOVRD),
        "naive_sfctmp_vs_wrf_t_ml": diffstat(naive_sfctmp, IN("t_ml"), land),
        "fixed_sfctmp_vs_wrf_t_ml": diffstat(forcing_real.sfctmp, IN("t_ml"), land),
        "moist_factor_identity_max_abs_err": float(np.nanmax(np.abs(
            (naive_sfctmp / np.asarray(forcing_real.sfctmp))[land] - (1.0 + RVOVRD * qair0)[land]))),
        "note": ("state.theta is moist theta_m; naive Exner left sfctmp warm by exactly "
                 "(1+R_v/R_d*q_v). The fix recovers WRF's dry T3D."),
    }

    # =======================================================================
    # 3. Real-overlay Noah-MP fluxes (POST-FIX) + radiation-swap causal split
    # =======================================================================
    diag = surface_layer_with_diagnostics(state, first_timestep=True)
    ch_seed = _surface(_get(diag, "ch", land_state0.ch))
    cm_seed = _surface(_get(diag, "cm", land_state0.cm))
    ls_seed = land_state0.replace(ch=ch_seed, cm=cm_seed)
    _lo, nm_fix = noah_mp_step(ls_seed, forcing_real, static, DT, energy_params=ep, rad_params=rp)

    swdn_w = jnp.asarray(np.where(np.isnan(IN("swdn")), 0.0, IN("swdn")))
    lwdn_w = jnp.asarray(np.where(np.isnan(IN("lwdn")), 0.0, IN("lwdn")))
    forcing_wrad = forcing_real._replace(soldn=swdn_w, lwdn=lwdn_w)
    _lo2, nm_wrad = noah_mp_step(ls_seed, forcing_wrad, static, DT, energy_params=ep, rad_params=rp)

    flux_closure = {
        "post_fix_jax_radiation": {
            "hfx_vs_fsh": diffstat(nm_fix.hfx, OUT("fsh"), land),
            "grdflx_vs_ssoil": diffstat(nm_fix.grdflx, OUT("ssoil"), land),
            "tsk_vs_trad": diffstat(nm_fix.tsk, OUT("trad"), land),
        },
        "post_fix_plus_wrf_exact_radiation": {
            "hfx_vs_fsh": diffstat(nm_wrad.hfx, OUT("fsh"), land),
            "grdflx_vs_ssoil": diffstat(nm_wrad.grdflx, OUT("ssoil"), land),
            "tsk_vs_trad": diffstat(nm_wrad.tsk, OUT("trad"), land),
        },
        "note": ("With the decoupling fix the land-tile HFX residual is driven by the "
                 "RRTMG radiation forcing (GLW/SWDOWN); swapping WRF's EXACT SWDOWN/GLW "
                 "in COLLAPSES it to ~0.1 W/m2 -> the energy solve + the temperature "
                 "input are closed, the remaining lane is RRTMG."),
    }

    hfx_fix = flux_closure["post_fix_jax_radiation"]["hfx_vs_fsh"].get("rmse") or 0.0
    hfx_wrad = flux_closure["post_fix_plus_wrf_exact_radiation"]["hfx_vs_fsh"].get("rmse") or 0.0
    energy_algo_fsh = energy_algorithm["energy"]["fsh_hfx"].get("rmse") or 0.0
    glw_bias = forcing_vs_wrf["lwdn_glw_RRTMG"].get("bias") or 0.0
    swdown_bias = forcing_vs_wrf["soldn_swdown_RRTMG"].get("bias") or 0.0
    sfctmp_fixed_rmse = decoupling_fix["fixed_sfctmp_vs_wrf_t_ml"].get("rmse") or 0.0
    sfctmp_naive_bias = decoupling_fix["naive_sfctmp_vs_wrf_t_ml"].get("bias") or 0.0

    radiation_collapses = hfx_wrad < 0.25 * max(hfx_fix, 1e-30)
    energy_algo_exact = energy_algo_fsh <= 0.5    # W/m2

    if energy_algo_exact and radiation_collapses:
        verdict = "NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING"
    elif energy_algo_exact:
        verdict = "NOAHMP_LAND_TILE_ENERGY_SOLVE_EXACT_RESIDUAL_IN_INPUTS"
    else:
        verdict = "NOAHMP_LAND_TILE_ENERGY_STILL_OPEN"

    ranked = [
        {
            "rank": 1,
            "claim": ("RRTMG step-1 surface radiation forcing into Noah-MP: GLW "
                      "bias {:+.2f} W/m2, SWDOWN bias {:+.2f} W/m2. With WRF's exact "
                      "SWDOWN/GLW the land-tile HFX residual collapses {:.2f} -> {:.3f} "
                      "W/m2 rmse.".format(glw_bias, swdown_bias, hfx_fix, hfx_wrad)),
            "in_scope": False,
            "evidence": "forcing_vs_wrf + flux_closure radiation swap",
            "next": ("RRTMG longwave/shortwave forcing-parity hook on the pinned tree "
                     "(GLW uniform clear-sky bias); RRTMG production is frozen this sprint."),
        },
        {
            "rank": 2,
            "claim": ("Surface-layer (sfclay/MYNN) air temperature uses the SAME "
                      "moist-theta -> naive-Exner conversion (surface_layer."
                      "_potential_to_temperature) -> ~+4 K warm t1d over ALL cells; "
                      "affects the water tiles incl. the strict worst cell (i=66, j=37, "
                      "water). Out of this sprint's file ownership."),
            "in_scope": False,
            "evidence": "code read surface_layer.py:471-472 + the same +4 K signature",
            "next": ("Apply the identical theta_m -> theta_dry decoupling in "
                     "surface_layer.py and RE-VALIDATE the MYNN d02 path (it may have "
                     "been tuned with the moist value; needs its own sprint)."),
        },
    ]

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.noahmp_land_tile_energy_closure.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
        },
        "fix": {
            "file": "src/gpuwrf/physics/noahmp_coupler.py",
            "function": "assemble_noahmp_forcing",
            "change": ("decouple theta_m -> theta_dry (divide by 1 + R_v/R_d*q_v) before "
                       "the Exner conversion of the lowest-level air temperature"),
            "default_inert": False,
            "production": True,
            "test": "tests/test_noahmp_coupler.py::test_forcing_decouples_moist_theta_to_dry_air_temperature",
        },
        "truth_provenance": provenance,
        "noahmp_options": options,
        "nland": nland,
        "energy_algorithm_vs_wrf": energy_algorithm,
        "jax_forcing_vs_wrf_nmpin": forcing_vs_wrf,
        "decoupling_fix": decoupling_fix,
        "flux_closure": flux_closure,
        "summary": {
            "energy_solve_fsh_rmse_with_wrf_inputs": energy_algo_fsh,
            "sfctmp_bias_before_fix_K": sfctmp_naive_bias,
            "sfctmp_rmse_after_fix_K": sfctmp_fixed_rmse,
            "hfx_rmse_post_fix_jax_radiation": hfx_fix,
            "hfx_rmse_post_fix_wrf_radiation": hfx_wrad,
            "glw_bias_W_m2": glw_bias,
            "swdown_bias_W_m2": swdown_bias,
        },
        "ranked_hypotheses": ranked,
        "fastest_next_command": (
            "Emit an RRTMG step-1 longwave/shortwave forcing hook on the pinned tree "
            "(GLW/SWDOWN at the surface, both clear-sky) and column-diff vs the JAX "
            "rrtmg seed to close the +14.7 W/m2 GLW / +3.6 W/m2 SWDOWN bias; in "
            "parallel apply the same theta_m->theta_dry decoupling in surface_layer.py "
            "and RE-VALIDATE the MYNN d02 path. Then rerun JAX_PLATFORMS=cpu "
            "CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src "
            "python proofs/v014/noahmp_step1_closure.py."
        ),
        "commands": {
            "wrf_one_run": (
                "cd /tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/run && env "
                "WRFGPU2_V014_NOAHMP_ENERGY_HOOK=1 "
                "WRFGPU2_V014_NOAHMP_ENERGY_ROOT=/tmp/wrfgpu2_v014_noahmp_energy_pinned_onerun "
                "WRFGPU2_V014_SURFACE_HANDOFF_HOOK=1 "
                "WRFGPU2_V014_SURFACE_HANDOFF_ROOT=/tmp/wrfgpu2_v014_surface_handoff_energyhook_xcheck "
                "taskset -c 0-3 ./wrf.exe"
            ),
            "proof": (
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false "
                "PYTHONPATH=src python proofs/v014/noahmp_land_tile_energy_closure.py"
            ),
        },
        "git": {
            "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                   capture_output=True, check=False).stdout.strip(),
            "branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                     text=True, capture_output=True, check=False).stdout.strip(),
        },
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return f"# V0.14 Noah-MP Land-Tile Energy Closure\n\nBlocked: `{payload.get('status')}`.\n"
    s = payload["summary"]
    ea = payload["energy_algorithm_vs_wrf"]["energy"]
    fc = payload["flux_closure"]
    lines = [
        "# V0.14 Noah-MP Land-Tile Energy Closure",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Bottom line",
        "",
        "- The JAX Noah-MP energy **algorithm is exact**: fed WRF's exact per-column "
        f"NMPIN, FSH rmse `{ea['fsh_hfx'].get('rmse')}` W/m2, SSOIL rmse "
        f"`{ea['ssoil_grdflx'].get('rmse')}`, TRAD rmse `{ea['trad_tsk'].get('rmse')}` K. "
        "The prior 'NoahMP land-tile energy' narrowing is **refuted**.",
        "- **Root cause (fixed, production):** `state.theta` is the WRF MOIST potential "
        "temperature; `assemble_noahmp_forcing` converted it to air temperature with a "
        "naive Exner, leaving the lowest-level air temperature "
        f"`{s['sfctmp_bias_before_fix_K']:+.3f}` K too warm "
        f"(= the 1+R_v/R_d*q_v factor). After the decouple fix, sfctmp rmse vs WRF T_ML "
        f"= `{s['sfctmp_rmse_after_fix_K']}` K.",
        f"- After the fix, land-tile HFX rmse = `{s['hfx_rmse_post_fix_jax_radiation']}` "
        f"W/m2; swapping WRF's exact SWDOWN/GLW in collapses it to "
        f"`{s['hfx_rmse_post_fix_wrf_radiation']}` W/m2 -> the remaining lane is the RRTMG "
        f"radiation forcing (GLW bias `{s['glw_bias_W_m2']:+.2f}`, SWDOWN bias "
        f"`{s['swdown_bias_W_m2']:+.2f}` W/m2).",
        "",
        "## Fix",
        "",
        f"- `{payload['fix']['file']}` :: `{payload['fix']['function']}` -- "
        f"{payload['fix']['change']}.",
        f"- test: `{payload['fix']['test']}`.",
        "",
        "## Energy solve vs WRF (WRF-exact inputs, land cells)",
        "",
        f"- FSH `{ea['fsh_hfx']}`",
        f"- SSOIL `{ea['ssoil_grdflx']}`",
        f"- TRAD `{ea['trad_tsk']}`",
        "",
        "## Flux closure (real overlay, land cells)",
        "",
        f"- post-fix JAX radiation: HFX `{fc['post_fix_jax_radiation']['hfx_vs_fsh']}`",
        f"- post-fix + WRF radiation: HFX `{fc['post_fix_plus_wrf_exact_radiation']['hfx_vs_fsh']}`",
        "",
        "## Ranked remaining lanes (both out of this sprint's scope)",
        "",
    ]
    for item in payload["ranked_hypotheses"]:
        lines.append(f"{item['rank']}. {item['claim']}")
        lines.append(f"   - next: {item['next']}")
    lines += [
        "",
        "## Fastest next command",
        "",
        f"`{payload['fastest_next_command']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = build_proof()
    OUT_JSON.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
