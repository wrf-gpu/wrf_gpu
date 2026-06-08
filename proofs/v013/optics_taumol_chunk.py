"""v0.13 keystone follow-up proof: RRTMG taumol/optics CONSTRUCTION chunking.

The g-point-chunk sprint tiled the SW two-stream FLUX reduction over bands
(SW peak -45..57%) but flagged the upstream taumol/optics CONSTRUCTION as the
remaining dominant fp64 VRAM floor:

  * LW: ``_lw_solver_base`` built the full ``(ncol, nlay, 16, 16)`` ``tau``/
    ``fracs`` stack PLUS a dead nearest-pressure fallback DUPLICATE (the
    ``_LW_TAUMOL_BRANCH_ACCEPTED`` mask is all-True, so the fallback was thrown
    away by ``jnp.where(True, branch, fallback)``).  g-point-chunk found LW
    VRAM-NEUTRAL precisely because this taumol floor remained.
  * SW: the gas/Rayleigh taumol + the 6 delta-scaled clear/cloud optics
    ``(ncol, nlay, 14, 16)`` arrays were built for all 14 bands up front and
    only THEN sliced per flux tile.

This sprint:

  1. **Drops the LW fallback duplicate** when the branch mask is statically
     all-True (a byte-exact ``where(True, a, b) == a`` VRAM win).
  2. **Builds the LW taumol per-band inside the rtrnmc band scan** (a
     ``lax.scan`` over the 16 bands; per-band gas chemistry resolved by
     ``lax.switch`` over ``_lw_taumol_band``) so the full 16-band ``tau``/
     ``fracs`` stack is never materialised.
  3. **Builds the SW taumol + delta-scaled optics per band-tile inside the
     two-stream scan** (``_sw_band_scan_optics_fluxes``) so the full 14-band
     taumol + 6 optics arrays are never materialised; the McICA cloud sample is
     built once and sliced per tile.

Proofs:

  * **Numerical inertness** (CPU, fp64): the chunked-construction operational
    path is bit-identical (max_rel = 0.0) to the upfront-construction path, for
    SW and LW, all-sky and with-clear-sky, across chunk widths.  The chunk knobs
    are pure VRAM controls.
  * **Peak-VRAM before -> after** (GPU): on a deep-column grid the chunked
    construction lowers ``peak_bytes_in_use`` materially vs the upfront build.

Run modes::

    # bit-inertness (CPU, default)
    PYTHONPATH=src JAX_PLATFORMS=cpu python proofs/v013/optics_taumol_chunk.py --mode inertness

    # peak VRAM (GPU; one fresh process per config)
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
      python proofs/v013/optics_taumol_chunk.py --mode vram --kind lw --construct chunked --ncol 16384 --nrep 2

The driver (``--mode all``) runs the CPU inertness sweep in-process and spawns
fresh GPU subprocesses per VRAM config, then writes
``proofs/v013/optics_taumol_chunk.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v013" / "optics_taumol_chunk.json"

SW_FIELDS = (
    "heating_rate", "flux_down", "flux_up", "toa_down", "toa_up",
    "surface_down", "surface_up", "column_absorbed", "surface_absorbed",
    "surface_direct", "surface_diffuse", "surface_diffuse_fraction",
    "topographic_correction_factor", "surface_down_topographic",
    "surface_up_topographic", "surface_absorbed_topographic",
)
LW_FIELDS = (
    "heating_rate", "flux_down", "flux_up", "toa_down", "toa_up",
    "surface_down", "surface_up", "column_net_heating", "surface_emission",
)
SW_CLEAR_FIELDS = ("clear_flux_down", "clear_flux_up")
LW_CLEAR_FIELDS = ("clear_flux_down", "clear_flux_up")


def _enable_x64():
    from jax import config

    config.update("jax_enable_x64", True)


def _sw_state(cls, arr, ncol_tile: int = 1, nrep: int = 1):
    import jax.numpy as jnp

    def col(name):
        a = np.asarray(arr[name], dtype=np.float64)
        if nrep > 1 and a.ndim >= 2:
            a = np.repeat(a, nrep, axis=-1)
        if ncol_tile > 1:
            a = np.tile(a, (ncol_tile,) + (1,) * (a.ndim - 1))
        return jnp.asarray(a)

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_albedo=col("input_surface_albedo"), coszen=col("input_coszen"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _lw_state(cls, arr, ncol_tile: int = 1, nrep: int = 1):
    import jax.numpy as jnp

    def col(name):
        a = np.asarray(arr[name], dtype=np.float64)
        if nrep > 1 and a.ndim >= 2:
            a = np.repeat(a, nrep, axis=-1)
        if ncol_tile > 1:
            a = np.tile(a, (ncol_tile,) + (1,) * (a.ndim - 1))
        return jnp.asarray(a)

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_temperature=col("input_surface_temperature"),
        surface_emissivity=col("input_surface_emissivity"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _worst(cand, ref, fields):
    worst_abs = 0.0
    worst_rel = 0.0
    bit_equal = True
    for f in fields:
        c = np.asarray(getattr(cand, f))
        e = np.asarray(getattr(ref, f))
        d = np.abs(c - e)
        r = d / np.maximum(np.abs(e), 1e-30)
        worst_abs = max(worst_abs, float(d.max()))
        worst_rel = max(worst_rel, float(r.max()))
        if not np.array_equal(c, e):
            bit_equal = False
    return worst_abs, worst_rel, bit_equal


def run_inertness(ncol_tile: int = 1) -> dict:
    """Chunked-construction-vs-upfront-construction bit-comparison.

    Reference = the upfront-construction operational path (chunk knobs set so the
    full taumol/optics arrays are built and only then sliced).  Each chunked
    candidate (taumol/optics built per-band-tile inside the scan) is bit-compared
    against it, all-sky and with-clear-sky, across chunk widths.
    """

    _enable_x64()
    from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE, LW_SAMPLE
    import gpuwrf.physics.rrtmg_sw as swmod
    import gpuwrf.physics.rrtmg_lw as lwmod
    from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column, RRTMGSWColumnState
    from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column, RRTMGLWColumnState

    sw_arr = _arrays(SW_SAMPLE)
    lw_arr = _arrays(LW_SAMPLE)
    result = {"ncol_tile": ncol_tile, "sw": {}, "lw": {}}

    # ---- SW ----
    sw_state_all = _sw_state(RRTMGSWColumnState, sw_arr, ncol_tile)
    sw_rows = []
    for wcs in (False, True):
        # Reference: upfront construction (chunk OFF).
        swmod._SW_TAUMOL_CHUNK = False
        sw_ref = solve_rrtmg_sw_column(sw_state_all, debug=False, with_clear_sky=wcs)
        # Candidates: per-tile construction at several chunk widths.
        swmod._SW_TAUMOL_CHUNK = True
        for chunk in (1, 2, 7, 14):
            swmod._SW_GPOINT_CHUNK_BANDS = chunk
            cand = solve_rrtmg_sw_column(sw_state_all, debug=False, with_clear_sky=wcs)
            fields = SW_FIELDS + (SW_CLEAR_FIELDS if wcs else ())
            wa, wr, bit = _worst(cand, sw_ref, fields)
            sw_rows.append({"with_clear_sky": wcs, "chunk_bands": chunk, "max_abs": wa, "max_rel": wr, "bit_equal": bit})
    swmod._SW_GPOINT_CHUNK_BANDS = 1
    swmod._SW_TAUMOL_CHUNK = True
    result["sw"] = {
        "construction_reference": "upfront (_SW_TAUMOL_CHUNK=False)",
        "rows": sw_rows,
        "max_rel_over_all": max(r["max_rel"] for r in sw_rows),
        "all_bit_identical": all(r["bit_equal"] for r in sw_rows),
    }

    # ---- LW ----
    lw_state_all = _lw_state(RRTMGLWColumnState, lw_arr, ncol_tile)
    lw_rows = []
    for wcs in (False, True):
        lwmod._LW_TAUMOL_CHUNK = False
        lw_ref = solve_rrtmg_lw_column(lw_state_all, debug=False, with_clear_sky=wcs)
        lwmod._LW_TAUMOL_CHUNK = True
        cand = solve_rrtmg_lw_column(lw_state_all, debug=False, with_clear_sky=wcs)
        fields = LW_FIELDS + (LW_CLEAR_FIELDS if wcs else ())
        wa, wr, bit = _worst(cand, lw_ref, fields)
        lw_rows.append({"with_clear_sky": wcs, "max_abs": wa, "max_rel": wr, "bit_equal": bit})
    lwmod._LW_TAUMOL_CHUNK = True
    result["lw"] = {
        "construction_reference": "upfront (_LW_TAUMOL_CHUNK=False)",
        "rows": lw_rows,
        "max_rel_over_all": max(r["max_rel"] for r in lw_rows),
        "all_bit_identical": all(r["bit_equal"] for r in lw_rows),
        "fallback_duplicate_dropped": "yes (branch mask all-True -> tau=branch_tau, no _lw_fallback_taumol alloc)",
    }
    return result


def run_vram(kind: str, construct: str, ncol: int, nrep: int = 1) -> dict:
    """Measure peak device bytes for one (kind, construct) config on GPU.

    ``construct`` is ``chunked`` (taumol/optics built per-band-tile in the scan,
    the new default) or ``upfront`` (full taumol/optics stack built then sliced,
    the pre-sprint behaviour).  The kernel is wrapped in ``jax.jit`` (the
    operational path runs the column kernel inside a jitted ``lax.scan``, so only
    under jit does XLA do whole-program memory planning).  One fresh process per
    config for a clean BFC peak counter.
    """

    _enable_x64()
    import jax
    from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE, LW_SAMPLE

    dev = jax.devices()[0]
    chunked = construct == "chunked"

    if kind == "sw":
        import gpuwrf.physics.rrtmg_sw as swmod
        from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column, RRTMGSWColumnState

        swmod._SW_TAUMOL_CHUNK = chunked
        swmod._SW_GPOINT_CHUNK_BANDS = 1 if chunked else swmod._SW_NBANDS
        arr = _arrays(SW_SAMPLE)
        state = _sw_state(RRTMGSWColumnState, arr, ncol, nrep)
        fields = SW_FIELDS
        fn = jax.jit(lambda s: solve_rrtmg_sw_column(s, debug=False))
    else:
        import gpuwrf.physics.rrtmg_lw as lwmod
        from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column, RRTMGLWColumnState

        lwmod._LW_TAUMOL_CHUNK = chunked
        arr = _arrays(LW_SAMPLE)
        state = _lw_state(RRTMGLWColumnState, arr, ncol, nrep)
        fields = LW_FIELDS
        fn = jax.jit(lambda s: solve_rrtmg_lw_column(s, debug=False))

    try:
        dev.memory_stats()
    except Exception:
        pass
    out = fn(state)
    for f in fields:
        jax.block_until_ready(getattr(out, f))
    ms = dev.memory_stats() or {}
    return {
        "kind": kind,
        "construct": construct,
        "ncol": int(np.asarray(state.p).shape[0]),
        "nlev": int(np.asarray(state.p).shape[-1]),
        "peak_bytes_in_use": int(ms.get("peak_bytes_in_use", 0)),
        "bytes_in_use_after": int(ms.get("bytes_in_use", 0)),
        "peak_mib": round(int(ms.get("peak_bytes_in_use", 0)) / (1024 * 1024), 2),
    }


def drive_all(ncol_vram: int, nrep: int) -> dict:
    record: dict = {
        "proof": "v0.13 RRTMG taumol/optics CONSTRUCTION chunking — bit-inertness + peak-VRAM",
        "branch": "worker/opus/v013-optics-taumol-chunk",
        "files_changed": [
            "src/gpuwrf/physics/rrtmg_lw.py",
            "src/gpuwrf/physics/rrtmg_sw.py",
        ],
    }
    record["inertness"] = run_inertness(ncol_tile=1)

    gpu_env = dict(os.environ)
    gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpu_env.setdefault("PYTHONPATH", "src")
    gpu_env.setdefault("GPUWRF_XLA_AUTOTUNE_CACHE", "0")
    vram_rows = []
    # Apples-to-apples before->after at a grid where the UPFRONT build still fits
    # (nlev = 16*nrep), plus a DEEP-column probe (nrep=4 -> nlev=64) where the
    # upfront build OOMs and the chunked build FITS — the GWD-nested-1km step-0
    # OOM family.
    deep_ncol = ncol_vram * 2
    deep_nrep = 4
    configs = [
        ("lw", "upfront", ncol_vram, nrep, "compare"),
        ("lw", "chunked", ncol_vram, nrep, "compare"),
        ("sw", "upfront", ncol_vram, nrep, "compare"),
        ("sw", "chunked", ncol_vram, nrep, "compare"),
        ("lw", "upfront", deep_ncol, deep_nrep, "deep_oom"),
        ("lw", "chunked", deep_ncol, deep_nrep, "deep_oom"),
        ("sw", "upfront", deep_ncol, deep_nrep, "deep_oom"),
        ("sw", "chunked", deep_ncol, deep_nrep, "deep_oom"),
    ]
    for kind, construct, ncol, nr, role in configs:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--mode", "vram", "--kind", kind, "--construct", construct,
             "--ncol", str(ncol), "--nrep", str(nr), "--emit-json"],
            capture_output=True, text=True, env=gpu_env, cwd=str(ROOT),
        )
        line = ""
        for ln in proc.stdout.splitlines():
            if ln.startswith("VRAM_JSON:"):
                line = ln[len("VRAM_JSON:"):]
        if line:
            row = json.loads(line)
            row["role"] = role
            vram_rows.append(row)
        else:
            oom = any("RESOURCE_EXHAUSTED" in s or "Out of memory" in s
                      for s in (proc.stderr or "").splitlines())
            vram_rows.append({
                "kind": kind, "construct": construct, "ncol_arg": ncol, "nrep": nr,
                "nlev": 16 * nr, "role": role,
                "result": "OOM" if oom else "error",
                "error_tail": (proc.stderr or "").strip().splitlines()[-3:],
                "rc": proc.returncode,
            })
    record["vram"] = {"compare_ncol_tile": ncol_vram, "compare_nrep": nrep,
                      "deep_ncol_tile": deep_ncol, "deep_nrep": deep_nrep, "rows": vram_rows}

    def peak_for(kind, construct, role):
        for r in vram_rows:
            if (r.get("kind") == kind and r.get("construct") == construct
                    and r.get("role") == role and "peak_bytes_in_use" in r):
                return r["peak_bytes_in_use"]
        return None

    def oom_for(kind, construct, role):
        for r in vram_rows:
            if (r.get("kind") == kind and r.get("construct") == construct
                    and r.get("role") == role):
                return r.get("result") == "OOM"
        return False

    summary = {}
    for kind in ("lw", "sw"):
        up = peak_for(kind, "upfront", "compare")
        ch = peak_for(kind, "chunked", "compare")
        if up and ch:
            summary[f"{kind}_peak_mib_upfront"] = round(up / 1024 / 1024, 2)
            summary[f"{kind}_peak_mib_chunked"] = round(ch / 1024 / 1024, 2)
            summary[f"{kind}_peak_reduction_pct"] = round(100.0 * (1.0 - ch / up), 2)
        deep_up_oom = oom_for(kind, "upfront", "deep_oom")
        deep_ch = peak_for(kind, "chunked", "deep_oom")
        summary[f"{kind}_deep_upfront_oom"] = bool(deep_up_oom)
        summary[f"{kind}_deep_chunked_peak_mib"] = round(deep_ch / 1024 / 1024, 2) if deep_ch else None
        summary[f"{kind}_deep_oom_then_fits"] = bool(deep_up_oom and deep_ch)

    record["summary"] = summary

    inert = record["inertness"]
    record["verdict"] = {
        "sw_construction_inert_bit_identical": inert["sw"]["all_bit_identical"],
        "lw_construction_inert_bit_identical": inert["lw"]["all_bit_identical"],
        "sw_inert_max_rel": inert["sw"]["max_rel_over_all"],
        "lw_inert_max_rel": inert["lw"]["max_rel_over_all"],
        "lw_vram_reduced": bool(peak_for("lw", "upfront", "compare") and peak_for("lw", "chunked", "compare")
                                and peak_for("lw", "chunked", "compare") < peak_for("lw", "upfront", "compare")),
        "sw_vram_reduced": bool(peak_for("sw", "upfront", "compare") and peak_for("sw", "chunked", "compare")
                                and peak_for("sw", "chunked", "compare") < peak_for("sw", "upfront", "compare")),
        "deep_column_oom_then_fits": bool(
            (oom_for("lw", "upfront", "deep_oom") and peak_for("lw", "chunked", "deep_oom"))
            or (oom_for("sw", "upfront", "deep_oom") and peak_for("sw", "chunked", "deep_oom"))
        ),
    }
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("inertness", "vram", "all"), default="all")
    ap.add_argument("--kind", choices=("sw", "lw"), default="lw")
    ap.add_argument("--construct", choices=("chunked", "upfront"), default="chunked")
    ap.add_argument("--ncol", type=int, default=8192, help="column-tile multiplier (fixture has 3 cols)")
    ap.add_argument("--nrep", type=int, default=2, help="vertical-level multiplier (fixture has 16 layers)")
    ap.add_argument("--emit-json", action="store_true")
    args = ap.parse_args()

    if args.mode == "inertness":
        print(json.dumps(run_inertness(), indent=2))
    elif args.mode == "vram":
        rec = run_vram(args.kind, args.construct, args.ncol, args.nrep)
        if args.emit_json:
            print("VRAM_JSON:" + json.dumps(rec))
        else:
            print(json.dumps(rec, indent=2))
    else:
        rec = drive_all(args.ncol, args.nrep)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=2))
        print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
