"""V0.14 surface-layer water-path moist-theta -> dry-T decoupling proof.

WRF-anchored isolation for the strict Step-1 worst cell (water, Fortran
i=66 j=37 k=3). The operational surface slot under ``use_noahmp=True`` runs the
revised surface layer (sfclay) over ALL columns inside
``coupling.noahmp_surface_hook.noahmp_surface_step``; over WATER Noah-MP does not
run, so the water HFX/theta_flux that MYNN consumes is the sfclay bulk flux.

Before this sprint, ``noahmp_surface_hook._build_column_view`` handed the surface
layer the operational ``State.theta`` -- which is the WRF MOIST potential
temperature ``theta_m = theta_dry*(1 + R_v/R_d*qv)`` (use_theta_m=1) -- with no
``t_air``.  The revised surface layer then derived the lowest-level air
temperature with a naive Exner (``theta_m*(p/p0)^kappa``), leaving it ~+4 K too
warm: the SAME defect Fable fixed for the Noah-MP land forcing
(``assemble_noahmp_forcing``) and that ``physics_couplers._surface_column_view``
already avoids on the grid-backed path.

This proof feeds the real Step-1 column view to ``surface_layer_with_diagnostics``
two ways -- the legacy moist fallback (``t_air=None``) and the fixed dry ``t_air``
-- and compares the lowest-level air temperature and the resulting HFX/theta_flux
to WRF's PRE_NOAHMP surface hook (which equals the WRF SFCLAY1D output, so it is
the sfclay truth on EVERY column, water and land).

CPU-only; no production edits here (the fix lives in
``coupling.noahmp_surface_hook``).
"""

from __future__ import annotations

import hashlib
import json
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

import mynn_driver_source_output_fix as mynn_prior  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_surface_land_flux_handoff as handoff  # noqa: E402

OUT_JSON = PROOF_DIR / "surface_layer_theta_decoupling.json"
OUT_MD = PROOF_DIR / "surface_layer_theta_decoupling.md"

PINNED_SURFACE = Path(
    "/tmp/wrfgpu2_v014_surface_handoff_pinned_onerun/surface_land_flux_d02_step1.txt"
)

# strict Step-1 worst cell (water), Fortran 1-based (i=west-east/nx, j=south-north/ny)
WORST_FORTRAN = {"i": 66, "j": 37, "k": 3}


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
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


def git_metadata() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=False
            ).stdout.strip()
        except Exception:
            return None

    return {"head": run(["rev-parse", "HEAD"]), "branch": run(["rev-parse", "--abbrev-ref", "HEAD"])}


def jax_environment() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    return {
        "backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(),
    }


def _column_views(state, grid):
    """Return (buggy, fixed_tair, fixed_full) trailing-z Noah-MP column views.

    * ``buggy``       = legacy moist-theta fallback (``t_air=None``, no psfc/rho);
    * ``fixed_tair``  = dry ``t_air`` only (grid-less ``_build_column_view``);
    * ``fixed_full``  = production grid path (dry ``t_air`` + true ``psfc`` +
      phy_prep ``rho`` + hydrostatic ``p``), mirroring
      ``physics_couplers._surface_column_view``.
    """

    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view  # noqa: PLC0415
    from gpuwrf.coupling.physics_couplers import _to_columns  # noqa: PLC0415

    fixed_tair = _build_column_view(state)
    fixed_full = _build_column_view(state, grid)
    # legacy/buggy view: MOIST theta_m handed straight to the surface layer with no
    # t_air/psfc/rho -- exactly the pre-fix `_build_column_view` behavior.
    buggy_view = fixed_tair._replace(theta=_to_columns(state.theta), t_air=None, psfc=None)
    return buggy_view, fixed_tair, fixed_full


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    if not PINNED_SURFACE.is_file():
        return {"status": "BLOCKED_PINNED_SURFACE_MISSING", "path": str(PINNED_SURFACE)}

    from gpuwrf.coupling.physics_dispatch import DEFAULT_MP_PHYSICS  # noqa: PLC0415
    from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP  # noqa: PLC0415
    from gpuwrf.physics.surface_layer import (  # noqa: PLC0415
        _potential_to_temperature,
        surface_layer_with_diagnostics,
    )
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    inputs = live.build_live_nest_step1_inputs()
    nl = inputs["namelist"]
    state = inputs["carry"].state
    # operational order: microphysics runs before the surface slot.
    if int(nl.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, float(nl.dt_s))

    buggy_view, fixed_view, fixed_full_view = _column_views(state, nl.grid)

    # lowest-level air temperature each path hands the surface layer. The buggy path
    # re-derives t1d from MOIST theta_m via a naive Exner; the fixed path uses dry t_air.
    theta_m_low = np.asarray(buggy_view.theta, dtype=np.float64)[..., 0]
    p_low = np.asarray(buggy_view.p, dtype=np.float64)[..., 0]
    t1d_buggy = np.asarray(_potential_to_temperature(jnp.asarray(theta_m_low), jnp.asarray(p_low)))
    t1d_fixed = np.asarray(fixed_view.t_air, dtype=np.float64)[..., 0]

    # WRF truth: PRE_NOAHMP == SFCLAY1D output (the sfclay flux on EVERY column).
    surface = handoff.parse_surface_hook(PINNED_SURFACE)
    if surface["status"] != "READY":
        return {"status": "BLOCKED_SURFACE_HOOK", "detail": surface.get("status")}
    pre = surface["arrays"]["PRE_NOAHMP"]
    xland = handoff.field(pre, "xland")
    wrf_hfx = handoff.field(pre, "hfx")
    wrf_qfx = handoff.field(pre, "qfx")
    wrf_ust = handoff.field(pre, "ust")
    wrf_tsk = handoff.field(pre, "tsk")
    is_water = np.asarray(xland, dtype=np.float64).reshape(-1) >= 1.5
    is_land = ~is_water

    # WRF kinematic theta flux (the quantity MYNN actually consumes)
    hooks = mynn_prior.parse_hook_set(
        mynn_prior.SCRATCH / "wrf_truth_mynn_pinned_onerun"
        if (mynn_prior.SCRATCH / "wrf_truth_mynn_pinned_onerun").is_dir()
        else mynn_prior.HOOK_ROOT
    )
    wrf_flt = None
    if hooks is not None:
        wrf_flt = np.asarray(mynn_prior.wrf_kinematic_fluxes(hooks["pre_c"], hooks["pre_s"])["flt"])

    def run_sfclay(view):
        diag = surface_layer_with_diagnostics(view, first_timestep=True)
        return {
            "hfx": np.asarray(diag.hfx, dtype=np.float64),
            "theta_flux": np.asarray(diag.fluxes.theta_flux, dtype=np.float64),
            "ust": np.asarray(diag.fluxes.ustar, dtype=np.float64),
            "t2": np.asarray(diag.t2, dtype=np.float64),
        }

    buggy = run_sfclay(buggy_view)
    fixed = run_sfclay(fixed_view)
    fixed_full = run_sfclay(fixed_full_view)

    def lane(jax_field, wrf_field, mask):
        return diffstat(jax_field, wrf_field, mask)

    # worst-cell focus (surface 2-D index j-1, i-1)
    shp = np.asarray(xland).shape
    jj, ii = WORST_FORTRAN["j"] - 1, WORST_FORTRAN["i"] - 1
    cell: dict[str, Any] = {"fortran": WORST_FORTRAN, "surface_shape": list(shp)}
    if len(shp) == 2 and jj < shp[0] and ii < shp[1]:
        flat = jj * shp[1] + ii
        cell.update(
            {
                "is_water": bool(np.asarray(xland).reshape(-1)[flat] >= 1.5),
                "xland": float(np.asarray(xland).reshape(-1)[flat]),
                "t1d_buggy_K": float(t1d_buggy.reshape(-1)[flat]),
                "t1d_fixed_K": float(t1d_fixed.reshape(-1)[flat]),
                "t1d_bias_K": float((t1d_buggy - t1d_fixed).reshape(-1)[flat]),
                "tsk_wrf_K": float(np.asarray(wrf_tsk).reshape(-1)[flat]),
                "hfx_buggy": float(buggy["hfx"].reshape(-1)[flat]),
                "hfx_fixed_tair_only": float(fixed["hfx"].reshape(-1)[flat]),
                "hfx_fixed_full": float(fixed_full["hfx"].reshape(-1)[flat]),
                "hfx_wrf": float(np.asarray(wrf_hfx).reshape(-1)[flat]),
                "ust_buggy": float(buggy["ust"].reshape(-1)[flat]),
                "ust_fixed_full": float(fixed_full["ust"].reshape(-1)[flat]),
                "ust_wrf": float(np.asarray(wrf_ust).reshape(-1)[flat]),
            }
        )

    metrics = {
        "t1d_air_temperature_bias_vs_dry_K": {
            "water": diffstat(t1d_buggy, t1d_fixed, is_water),
            "land": diffstat(t1d_buggy, t1d_fixed, is_land),
            "note": "buggy(moist theta_m Exner) - fixed(dry t_air); = the +Rv/Rd*qv warm bias.",
        },
        "hfx_vs_wrf_pre_noahmp": {
            "water_buggy_moist": lane(buggy["hfx"], wrf_hfx, is_water),
            "water_fixed_tair_only": lane(fixed["hfx"], wrf_hfx, is_water),
            "water_fixed_full_phy_prep": lane(fixed_full["hfx"], wrf_hfx, is_water),
            "land_buggy_moist": lane(buggy["hfx"], wrf_hfx, is_land),
            "land_fixed_tair_only": lane(fixed["hfx"], wrf_hfx, is_land),
            "land_fixed_full_phy_prep": lane(fixed_full["hfx"], wrf_hfx, is_land),
            "note": (
                "buggy=moist theta_m + air-pressure/ideal-gas fallback; tair_only=dry "
                "t_air only (grid-less); full_phy_prep=production grid path (dry t_air + "
                "true psfc + phy_prep rho + hydrostatic p)."
            ),
        },
        "ust_vs_wrf_pre_noahmp": {
            "water_buggy_moist": lane(buggy["ust"], wrf_ust, is_water),
            "water_fixed_tair_only": lane(fixed["ust"], wrf_ust, is_water),
            "water_fixed_full_phy_prep": lane(fixed_full["ust"], wrf_ust, is_water),
        },
    }
    if wrf_flt is not None:
        metrics["theta_flux_vs_wrf_kinematic_flt"] = {
            "water_buggy_moist": lane(buggy["theta_flux"], wrf_flt, is_water),
            "water_fixed_full_phy_prep": lane(fixed_full["theta_flux"], wrf_flt, is_water),
        }

    water_buggy = metrics["hfx_vs_wrf_pre_noahmp"]["water_buggy_moist"]
    water_fixed = metrics["hfx_vs_wrf_pre_noahmp"]["water_fixed_full_phy_prep"]
    closes = (
        water_buggy.get("count", 0) > 0
        and water_fixed.get("rmse", 1e9) < 0.1 * water_buggy.get("rmse", 0.0)
    )
    verdict = (
        "WATER_PATH_MOIST_THETA_BUG_CONFIRMED_DRY_TAIR_DECOUPLING_CLOSES_SFCLAY_FLUX"
        if closes
        else "WATER_PATH_DECOUPLING_INCONCLUSIVE"
    )

    return {
        "status": "READY",
        "schema": "wrfgpu2.v014.surface_layer_theta_decoupling.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_metadata(),
        "environment": jax_environment(),
        "verdict": verdict,
        "config": {
            "use_noahmp": bool(nl.use_noahmp),
            "sf_sfclay_physics": int(nl.sf_sfclay_physics),
            "bl_pbl_physics": int(nl.bl_pbl_physics),
            "mp_physics": int(nl.mp_physics),
            "WRF_RV_OVER_RD": 461.6 / 287.0,
        },
        "worst_cell": cell,
        "metrics": metrics,
        "fix": {
            "file": "src/gpuwrf/coupling/noahmp_surface_hook.py",
            "change": (
                "_build_column_view supplies dry t_air = theta_dry*(p/p0)^kappa with "
                "theta_dry = state.theta/(1 + Rv/Rd*qv); surface layer + noahmp forcing "
                "consume it instead of re-deriving from moist theta_m."
            ),
            "mirrors": [
                "coupling.physics_couplers._surface_column_view (grid-backed path)",
                "physics.noahmp_coupler.assemble_noahmp_forcing (land forcing)",
            ],
        },
        "truth": {
            "pinned_surface": str(PINNED_SURFACE),
            "pinned_surface_sha16": sha16(PINNED_SURFACE),
            "note": "WRF PRE_NOAHMP == WRF SFCLAY1D output (sfclay truth on every column).",
        },
    }


def render_markdown(p: Mapping[str, Any]) -> str:
    if p.get("status") != "READY":
        return f"# V0.14 surface-layer theta decoupling\n\nBLOCKED: `{p.get('status')}`\n"
    m = p["metrics"]
    c = p["worst_cell"]
    tb = m["t1d_air_temperature_bias_vs_dry_K"]
    hf = m["hfx_vs_wrf_pre_noahmp"]
    lines = [
        "# V0.14 Surface-Layer Water-Path Moist-Theta Decoupling",
        "",
        f"Verdict: `{p['verdict']}`.",
        "",
        "## Air-temperature bias (moist theta_m Exner - dry t_air)",
        f"- water: bias `{tb['water'].get('bias')}` K, max_abs `{tb['water'].get('max_abs')}` K "
        f"(n=`{tb['water'].get('count')}`).",
        f"- land:  bias `{tb['land'].get('bias')}` K, max_abs `{tb['land'].get('max_abs')}` K.",
        "",
        "## sfclay HFX vs WRF PRE_NOAHMP (= WRF SFCLAY1D)",
        f"- WATER buggy(moist):       rmse `{hf['water_buggy_moist'].get('rmse')}`, bias "
        f"`{hf['water_buggy_moist'].get('bias')}`, max_abs `{hf['water_buggy_moist'].get('max_abs')}` W/m2.",
        f"- WATER fixed(t_air only):  rmse `{hf['water_fixed_tair_only'].get('rmse')}` W/m2.",
        f"- WATER fixed(full phy_prep): rmse `{hf['water_fixed_full_phy_prep'].get('rmse')}`, "
        f"max_abs `{hf['water_fixed_full_phy_prep'].get('max_abs')}` W/m2.",
        f"- LAND  buggy(moist):       rmse `{hf['land_buggy_moist'].get('rmse')}` W/m2.",
        f"- LAND  fixed(full phy_prep): rmse `{hf['land_fixed_full_phy_prep'].get('rmse')}` W/m2.",
        "",
        "## Strict worst cell (water, Fortran i=66 j=37 k=3)",
        f"- is_water `{c.get('is_water')}` (xland `{c.get('xland')}`); "
        f"t1d buggy `{c.get('t1d_buggy_K')}` K -> fixed `{c.get('t1d_fixed_K')}` K "
        f"(bias `{c.get('t1d_bias_K')}` K); tsk `{c.get('tsk_wrf_K')}` K.",
        f"- HFX buggy `{c.get('hfx_buggy')}` -> tair_only `{c.get('hfx_fixed_tair_only')}` -> "
        f"full `{c.get('hfx_fixed_full')}` vs WRF `{c.get('hfx_wrf')}` W/m2.",
        f"- UST buggy `{c.get('ust_buggy')}` -> full `{c.get('ust_fixed_full')}` vs WRF `{c.get('ust_wrf')}`.",
        "",
        "## Fix",
        f"- `{p['fix']['file']}`: {p['fix']['change']}",
    ]
    if "theta_flux_vs_wrf_kinematic_flt" in m:
        tf = m["theta_flux_vs_wrf_kinematic_flt"]
        lines[lines.index("## Fix"):lines.index("## Fix")] = [
            "## kinematic theta_flux vs WRF (MYNN bottom BC)",
            f"- WATER buggy rmse `{tf['water_buggy_moist'].get('rmse')}` -> full phy_prep "
            f"`{tf['water_fixed_full_phy_prep'].get('rmse')}` K m/s.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build_proof()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_markdown(payload))
    print(f"status={payload.get('status')} verdict={payload.get('verdict')}")
    if payload.get("status") == "READY":
        c = payload["worst_cell"]
        hf = payload["metrics"]["hfx_vs_wrf_pre_noahmp"]
        print(
            f"worst-cell t1d bias={c.get('t1d_bias_K')} K | water HFX rmse "
            f"buggy={hf['water_buggy_moist'].get('rmse')} -> tair_only={hf['water_fixed_tair_only'].get('rmse')} "
            f"-> full={hf['water_fixed_full_phy_prep'].get('rmse')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
