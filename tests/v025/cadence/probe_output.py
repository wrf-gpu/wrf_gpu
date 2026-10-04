#!/usr/bin/env python3
"""Measure whole-output compile events on frozen PROD inputs.

Adapted from the existing S1 critic_m9_probe: retains two compile instruments,
sentinel liveness, native case loader and production writer diagnostics.
This is a component probe, not an end-to-end S1 speed claim. GPU invocations
must use scripts/with_gpu_lock.sh --label cadence-out; CPU invocations must
set JAX_PLATFORMS=cpu and hide CUDA. Declare evidence and cache paths per arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
ap.add_argument("--label", required=True)
ap.add_argument("--case", required=True)
ap.add_argument("--domain", default="d01")
ap.add_argument("--leads", default="3618,7236,10854,43416")
ap.add_argument("--out", required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--scratch", required=True)
args = ap.parse_args()

SRC = Path(args.src).resolve()


sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
jax.config.update("jax_log_compiles", True)
assert jax.default_backend() == args.backend, jax.default_backend()

import gpuwrf  # noqa: E402

assert str(Path(gpuwrf.__file__).resolve()).startswith(str(SRC)), gpuwrf.__file__

T0 = time.perf_counter()


def now() -> float:
    return time.perf_counter() - T0


# ---- instrument A: jax_log_compiles records + gpuwrf call site -------------
COMPILES: list[dict] = []
_RX = re.compile(r"^Compiling (\S+)")


class _CompileLog(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        m = _RX.match(msg)
        if not m:
            return
        frames = [
            f"{Path(fr.filename).name}:{fr.lineno}:{fr.name}"
            for fr in traceback.extract_stack()
            if "/gpuwrf/" in fr.filename
        ]
        COMPILES.append({"t": now(), "name": m.group(1), "site": frames[-3:]})


_h = _CompileLog(level=logging.WARNING)
logging.getLogger("jax").addHandler(_h)

# ---- instrument B: jax.monitoring backend-compile durations ---------------
BACKEND: list[tuple[float, float]] = []


def _dur(event: str, duration: float, **_kw) -> None:
    if event.endswith("backend_compile_duration"):
        BACKEND.append((now(), float(duration)))


jax.monitoring.register_event_duration_secs_listener(_dur)


def window():
    return len(COMPILES), len(BACKEND)


def since(mark):
    c = COMPILES[mark[0]:]
    b = BACKEND[mark[1]:]
    by_name: dict[str, int] = {}
    for row in c:
        by_name[row["name"]] = by_name.get(row["name"], 0) + 1
    return {
        "log_compiles": len(c),
        "log_by_name": by_name,
        "log_sites": [[row["name"], row["site"]] for row in c],
        "backend_compiles": len(b),
        "backend_compile_s": round(sum(d for _, d in b), 3),
    }


def sentinel(tag: str) -> dict:
    """A brand-new jit must register exactly one compile on BOTH instruments."""
    mark = window()
    fn = jax.jit(lambda x: x * 3.0 + float(len(tag)))
    fn(np.arange(3.0)).block_until_ready()
    got = since(mark)
    got["alive"] = got["log_compiles"] >= 1 and got["backend_compiles"] >= 1
    return got


# ---- CPU shims (placement only) --------------------------------------------
import gpuwrf.contracts.state as state_mod  # noqa: E402
from gpuwrf.runtime import operational_mode as op  # noqa: E402

if args.backend == "cpu":
    _cpu = jax.devices("cpu")[0]
    state_mod._gpu_device = lambda: _cpu
    op._operational_device = lambda: _cpu
from gpuwrf.coupling import physics_couplers as pc  # noqa: E402
from gpuwrf.integration import nested_pipeline as npl  # noqa: E402
from gpuwrf.integration.daily_pipeline import _surface_diagnostics_for_output  # noqa: E402
from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_m9_flux_slices as LW  # noqa: E402
from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_m9_flux_slices as SW  # noqa: E402

# Spy on the host GHG interpolation actually used by _rrtmg_column_inputs.
GHG_CALLS: list[dict] = []
_real_ghg = pc.clwrf_ssp245_gases_for_time


def _ghg_spy(when, *a, **k):
    out = _real_ghg(when, *a, **k)
    GHG_CALLS.append({"t": now(), "valid_time": str(when), "gases": [
        float(out.co2_vmr), float(out.n2o_vmr), float(out.ch4_vmr),
        float(out.cfc11_vmr), float(out.cfc12_vmr)]})
    return out


pc.clwrf_ssp245_gases_for_time = _ghg_spy


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    case = Path(args.case)
    scratch = Path(args.scratch)
    rec: dict = {
        "schema": "wrf_gpu2.v025.cadence_out.output_probe.v1",
        "label": args.label,
        "src": str(SRC),
        "src_rrtmg_sha256": {
            n: sha((SRC / "gpuwrf/physics" / n).read_bytes())
            for n in ("rrtmg_sw.py", "rrtmg_lw.py")
        },
        "jax": jax.__version__,
        "case": str(case),
        "case_sha256": {
            n: sha((case / n).read_bytes()) for n in ("namelist.input", f"wrfinput_{args.domain}")
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.nice(0),
        "wrf_root": os.environ.get("GPUWRF_WRF_ROOT"),
    }
    cfg = npl.NestedPipelineConfig(
        input_dir=case, output_dir=scratch / "out", proof_dir=scratch / "proof",
        hours=3, max_dom=1,
    )
    t = now()
    _h2, bundles, _meta, run_start, dts, carries = npl._load_domains(cfg, (args.domain,))
    rec["load_s"] = round(now() - t, 3)
    rec["run_start"] = str(run_start)
    bundle, carry = bundles[args.domain], carries[args.domain]
    namelist, state = bundle.namelist, carry.state
    noahmp_land = getattr(carry, "noahmp_land", None)
    rec["grid"] = [int(namelist.grid.nz), int(namelist.grid.ny), int(namelist.grid.nx)]
    rec["use_noahmp"] = bool(getattr(namelist, "use_noahmp", False))
    rec["writer_branch"] = (
        "_noahmp_surface_diagnostics_for_output"
        if rec["use_noahmp"] and noahmp_land is not None
        else "_surface_diagnostics_for_output"
    )
    rec["nested_m9_from_carry_env"] = npl._nested_m9_radiation_from_carry_from_env()
    rec["training_subset"] = npl._resolve_training_output_subset()
    rec["topo_flags"] = [int(namelist.topo_shading), int(namelist.slope_rad)]
    rec["radiation_static_present"] = namelist.radiation_static is not None

    def m9(lead: float):
        # byte-for-byte the _PerDomainWrfoutWriter M9-diag branch
        if rec["use_noahmp"] and noahmp_land is not None:
            return npl._noahmp_surface_diagnostics_for_output(
                state, namelist, run_start, lead_seconds=float(lead),
                noahmp_land=noahmp_land, noahmp_rad=getattr(carry, "noahmp_rad", None),
                variable_subset=None,
            )
        return _surface_diagnostics_for_output(
            state, namelist, run_start, lead_seconds=float(lead), variable_subset=None
        )

    arrays: dict[str, np.ndarray] = {}
    rec["sentinel_start"] = sentinel("start")
    calls = []
    for lead in [float(x) for x in args.leads.split(",")]:
        g0 = len(GHG_CALLS)
        sw0, lw0 = int(SW._cache_size()), int(LW._cache_size())
        mark = window()
        t = now()
        out = m9(lead)
        wall = now() - t
        row = {"lead_s": lead, "wall_s": round(wall, 3), **since(mark),
               "sw_cache": [sw0, int(SW._cache_size())],
               "lw_cache": [lw0, int(LW._cache_size())],
               "ghg_calls": GHG_CALLS[g0:]}
        for name, val in (out or {}).items():
            host = np.asarray(val)
            arrays[f"{lead:g}|{name}"] = host
        row["fields"] = sorted((out or {}).keys())
        if out and "COSZEN" in out:
            row["coszen_max"] = float(np.max(out["COSZEN"]))
        if out and "SWDOWN" in out:
            row["swdown_max"] = float(np.max(out["SWDOWN"]))
        row["nonfinite_fields"] = [k for k, v in (out or {}).items()
                                   if not np.all(np.isfinite(np.asarray(v)))]
        calls.append(row)
        print(f"[{args.label}] lead={lead:g} wall={wall:.1f}s compiles(log)={row['log_compiles']} "
              f"backend={row['backend_compiles']}/{row['backend_compile_s']}s "
              f"sw={row['sw_cache']} lw={row['lw_cache']} by={row['log_by_name']}", flush=True)
    rec["calls"] = calls
    rec["sentinel_after_warm"] = sentinel("after_warm")

    rec["sentinel_end"] = sentinel("end")
    rec["status"] = "ok"
    np.savez(args.npz, **arrays)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({"status": rec["status"], "out": args.out}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        Path(args.out).write_text(json.dumps({"status":"error", "error":traceback.format_exc()},indent=2))
        raise
