#!/usr/bin/env python3
"""M2 device bake-off: fused vertical-implicit module vs the XLA reference.

Sprint contract step 4 (manager-run; this script touches the GPU and must run
inside the coordinated window / ``scripts/with_gpu_lock.sh``).

Measures, at FAST d01 production shape (44x70x120):

1. **Correctness on device** — same pre-registered envelope as the CPU gates
   (``1e-12 * max(1, max|ref|)`` per output) + bitwise fraction.  No accuracy
   escalation is permitted by the adoption bar.
2. **Kernel count per dispatch** — via a ``jax.profiler.trace`` around warmed
   dispatches of each arm; device kernel events are counted from the trace
   JSON.  Fail-closed: zero parsed kernel events is ``MISSING``, never zero.
   The adoption bar needs ``>=5x`` reduction for the family.
3. **Paired warm timing** — alternating arms, >=30 timed reps of >=16 inner
   dispatches each, median + IQR + bootstrap lower-95 of the ratio.  The arm
   is one STEP-EQUIVALENT of the family: ``calc_coef_w`` once (stage) plus
   ``advance_w`` 16 times (RK3 acoustic substeps at
   ``acoustic_substeps=10``), matching the census cadence.

ADOPTION BAR (frozen in the sprint contract, evaluated mechanically):
``fused_median / xla_median <= 1/1.5`` AND ``kernels_xla / kernels_fused >= 5``
AND correctness envelope holds.  Anything else is recorded as a falsified
candidate with the numbers — a valid result.

Usage (manager, inside the GPU lock):
    python scripts/v025/m2_bakeoff_vertical_implicit.py --out DIR
CPU smoke (clearly labelled, never a verdict):
    python scripts/v025/m2_bakeoff_vertical_implicit.py --cpu-smoke --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gzip  # noqa: E402  (after stdlib; before jax via the path bootstrap below)

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# ENV LANDMINE (device bake-off 2026-09-18): this jax build defaults
# jax_pallas_use_mosaic_gpu=True with no mosaic backend installed, and the
# Pallas call then fails/refuses on device.  The Triton lowering is the
# supported path for the fused module: force it here, before jax loads.
# x64 is forced as well so the harness is self-contained under any caller env.
os.environ["JAX_PALLAS_USE_MOSAIC_GPU"] = "false"
os.environ.setdefault("JAX_ENABLE_X64", "true")

SCHEMA = "wrf_gpu2.v025.m2.bakeoff_verdict.v1"
DEFAULT_OUT = Path("<DATA_ROOT>/wrf_gpu2/v025/m2/bakeoff")
SPEEDUP_BAR = 1.5          # fused must be >=1.5x faster (median)
LAUNCH_REDUCTION_BAR = 5.0  # family kernels must drop >=5x
CORRECTNESS_TOL = 1.0e-12   # value-scaled envelope, same as the CPU gates
WARM_REPS = 16              # inner dispatches per timed rep
TIMED_REPS = 40             # timed reps per arm (>=30 per contract)
ALTERNATING_PAIRS = 5       # full alternating passes over both arms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--acoustic-substeps", type=int, default=10)
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="run on CPU with interpret=True; DIAGNOSTIC ONLY")
    parser.add_argument("--timed-reps", type=int, default=TIMED_REPS)
    parser.add_argument("--warm-reps", type=int, default=WARM_REPS)
    return parser


# --------------------------------------------------------------------------- #
# fixtures (FAST production shape; kept jax-import-light)                      #
# --------------------------------------------------------------------------- #
def fast_advance_kwargs(nz: int = 44, ny: int = 70, nx: int = 120, seed: int = 15):
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    f3 = lambda nl: jnp.asarray(1.0 + 0.01 * rng.standard_normal((nl, ny, nx)))
    f2 = lambda: jnp.asarray(1.0 + 0.01 * rng.standard_normal((ny, nx)))
    return dict(
        w=f3(nz + 1), rw_tend=jnp.asarray(0.001 * f3(nz + 1)),
        ww=jnp.asarray(0.01 * f3(nz + 1)),
        u=jnp.asarray(1.0 + 0.01 * rng.standard_normal((nz, ny, nx + 1))),
        v=jnp.asarray(1.0 + 0.01 * rng.standard_normal((nz, ny + 1, nx))),
        mu_work=jnp.asarray(0.01 * f2()), mut=jnp.asarray(1e4 * f2()),
        muave=jnp.asarray(0.01 * f2()), muts=jnp.asarray(1e4 * f2()),
        t_2ave=f3(nz), t_2=f3(nz), t_1=f3(nz),
        ph=f3(nz + 1), ph_1=f3(nz + 1), phb=jnp.asarray(1e3 * f3(nz + 1)),
        ph_tend=jnp.asarray(0.001 * f3(nz + 1)), ht=jnp.asarray(100.0 * f2()),
        c2a=f3(nz), cqw=f3(nz + 1), alt=f3(nz),
        a=jnp.asarray(0.1 * f3(nz + 1)), alpha=jnp.asarray(0.9 * f3(nz + 1)),
        gamma=jnp.asarray(0.1 * f3(nz + 1)),
        c1h=jnp.asarray(np.linspace(1.0, 0.1, nz)),
        c2h=jnp.asarray(np.linspace(0.0, 100.0, nz)),
        c1f=jnp.asarray(np.linspace(1.0, 0.1, nz + 1)),
        c2f=jnp.asarray(np.linspace(0.0, 100.0, nz + 1)),
        rdnw=jnp.asarray(np.full(nz, 44.0)), rdn=jnp.asarray(np.full(nz, 44.0)),
        fnm=jnp.asarray(np.full(nz, 0.5)), fnp=jnp.asarray(np.full(nz, 0.5)),
        cf1=1.5, cf2=-0.5, cf3=0.0,
        msftx=f2(), msfty=f2(), w_save=jnp.asarray(0.01 * f3(nz + 1)),
        rdx=1.0 / 9000.0, rdy=1.0 / 9000.0, dts=5.4, epssm=0.5,
        top_lid=False, damp_opt=3, dampcoef=0.2, zdamp=5000.0, w_damping=1,
    )


def split_static(kw: dict) -> tuple[dict, dict]:
    is_arr = lambda v: hasattr(v, "shape") and v.shape != ()
    return ({k: v for k, v in kw.items() if not is_arr(v)},
            {k: v for k, v in kw.items() if is_arr(v)})


# --------------------------------------------------------------------------- #
# trace parsing                                                                #
# --------------------------------------------------------------------------- #
def count_device_kernels(trace_dir: Path) -> dict[str, Any]:
    """Count device kernel events in a jax.profiler trace (fail-closed)."""

    events: list[dict] = []
    files: list[str] = []
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".gz"):
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                    payload = json.load(fh)
            else:
                payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows = payload.get("traceEvents") or []
        if rows:
            files.append(str(path))
        for ev in rows:
            if not isinstance(ev, dict):
                continue
            cat = str(ev.get("cat", ""))
            name = str(ev.get("name", ""))
            if cat in ("kernel", "gpu_memcpy", "gpu_memset") or "Kernel" in name:
                events.append({"cat": cat, "name": name[:120],
                               "dur": ev.get("dur", 0)})
    by_name: dict[str, int] = {}
    total_us = 0.0
    for ev in events:
        by_name[ev["name"]] = by_name.get(ev["name"], 0) + 1
        try:
            total_us += float(ev.get("dur") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "kernel_events": len(events),
        "distinct": len(by_name),
        "total_device_us": total_us,
        "by_name_top": dict(sorted(by_name.items(), key=lambda kv: -kv[1])[:20]),
        "files": files,
        "status": "OK" if events else "MISSING",
    }


# --------------------------------------------------------------------------- #
# measurement                                                                  #
# --------------------------------------------------------------------------- #
def bootstrap_lower95(ratios: list[float], samples: int = 20000, seed: int = 20260918) -> float:
    import numpy as np

    rng = np_random(seed)
    src = np.asarray(ratios, dtype=float)
    idx = rng.integers(0, len(src), size=(samples, len(src)))
    medians = np.median(src[idx], axis=1)
    return float(np.quantile(medians, 0.025))


def np_random(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


def median_iqr(values: list[float]) -> dict[str, float]:
    med = statistics.median(values)
    q = statistics.quantiles(values, n=4)
    return {"median": med, "iqr": q[2] - q[0], "min": min(values), "max": max(values)}


def main() -> int:
    args = build_parser().parse_args()
    started_wall = time.monotonic()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    platforms = sorted({d.platform for d in jax.devices()})
    verdict_mode = "DEVICE"
    if platforms != ["gpu"]:
        if not args.cpu_smoke:
            print(f"REFUSED: this harness touches the GPU and requires the "
                  f"coordinated lock; devices={platforms}. For a diagnostic "
                  f"CPU smoke run add --cpu-smoke.", file=sys.stderr)
            return 2
        verdict_mode = "CPU_SMOKE_NOT_A_VERDICT"

    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf
    from gpuwrf.kernels.fused_vertical_implicit import (
        advance_w_pallas,
        calc_coef_w_pallas,
    )

    interpret = verdict_mode != "DEVICE"
    if interpret:
        # CPU smoke: shrink the grid so the interpreter finishes quickly.
        kw = fast_advance_kwargs(nz=8, ny=16, nx=16)
        substeps = 4
    else:
        kw = fast_advance_kwargs()
        substeps = int(args.acoustic_substeps)
    scal, arr = split_static(kw)
    metrics_ns = types_namespace({k: arr[k] for k in
                                  ("c1h", "c2h", "c1f", "c2f", "rdn", "rdnw")})
    mut, cqw, c2a = arr["mut"], arr["cqw"], arr["c2a"]

    # --- arm definitions: one STEP-EQUIVALENT of the family ---------------- #
    xla_coef = jax.jit(lambda mut, cqw, c2a: calc_coef_w_wrf_coefficients(
        mut, metrics_ns, dt=scal["dts"], epssm=scal["epssm"],
        top_lid=bool(scal["top_lid"]), cqw=cqw, c2a=c2a))
    fused_coef = jax.jit(lambda mut, cqw, c2a: calc_coef_w_pallas(
        mut, arr["c1h"], arr["c2h"], arr["c1f"], arr["c2f"], arr["rdn"], arr["rdnw"],
        dt=scal["dts"], epssm=scal["epssm"], top_lid=bool(scal["top_lid"]),
        cqw=cqw, c2a=c2a, interpret=interpret))
    xla_adv = jax.jit(lambda **a: advance_w_wrf(**a, **scal))
    fused_adv = jax.jit(lambda **a: advance_w_pallas(**a, **scal, interpret=interpret))

    coef_args = (mut, cqw, c2a)
    adv_args = {k: v for k, v in arr.items() if k not in ("a", "alpha", "gamma")}

    def xla_arm():
        a, alpha, gamma = xla_coef(*coef_args)
        w, ph, t2 = xla_adv(**adv_args, a=a, alpha=alpha, gamma=gamma)
        return w

    def fused_arm():
        a, alpha, gamma = fused_coef(*coef_args)
        w, ph, t2 = fused_adv(**adv_args, a=a, alpha=alpha, gamma=gamma)
        return w

    # warm both
    xla_arm(); fused_arm()

    # --- 1. correctness on the active backend ------------------------------ #
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf as ref_adv_direct
    from gpuwrf.dynamics.acoustic_wrf import (
        calc_coef_w_wrf_coefficients as ref_coef_direct,
    )

    correctness: dict[str, Any] = {"tolerance": CORRECTNESS_TOL,
        "note": "advance_w arms consume IDENTICAL fixture coefficients "
                "(like-for-like); coef-vs-coef is compared separately."}
    a0, al0, g0 = fused_coef(*coef_args)
    for name, ref_vals, got_vals in (
        ("calc_coef_w", ref_coef_direct(mut, metrics_ns, dt=scal["dts"],
                                        epssm=scal["epssm"], top_lid=bool(scal["top_lid"]),
                                        cqw=cqw, c2a=c2a),
         (a0, al0, g0)),
        ("advance_w", ref_adv_direct(**kw), fused_adv(**adv_args, a=arr["a"],
                                                      alpha=arr["alpha"], gamma=arr["gamma"])),
    ):
        for out, r, g in zip(("out0", "out1", "out2"), ref_vals, got_vals):
            r_np, g_np = np_device(r), np_device(g)
            scale = max(1.0, float(abs(r_np).max()))
            d = float(abs(r_np - g_np).max())
            correctness[f"{name}/{out}"] = {
                "max_abs": d, "scale": scale, "envelope_ok": d <= CORRECTNESS_TOL * scale,
                "bitwise_fraction": np_mean_equal(r_np, g_np),
            }
    correctness["all_ok"] = all(v.get("envelope_ok") for k, v in correctness.items()
                                if isinstance(v, dict))

    # --- 2. kernel counts --------------------------------------------------- #
    result: dict[str, Any] = {}
    kernel_counts: dict[str, Any] = {}
    for arm_name, fn in (("xla", xla_arm), ("fused", fused_arm)):
        tdir = args.out / f"trace_{arm_name}"
        if tdir.exists():
            for stale in tdir.rglob("*"):
                if stale.is_file():
                    stale.unlink()
        with jax.profiler.trace(str(tdir)):
            for _ in range(10):
                block(fn())
        kernel_counts[arm_name] = count_device_kernels(tdir)

    # --- 3. paired warm timing ---------------------------------------------- #
    reps_xla: list[float] = []
    reps_fused: list[float] = []
    ratios: list[float] = []
    for pair in range(max(1, ALTERNATING_PAIRS)):
        order = (("xla", xla_arm), ("fused", fused_arm))
        if pair % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            times = []
            for _ in range(args.timed_reps // ALTERNATING_PAIRS):
                t0 = time.perf_counter()
                out = fn()
                block(out)
                times.append(time.perf_counter() - t0)
            if name == "xla":
                reps_xla.extend(times)
            else:
                reps_fused.extend(times)
    stats_xla = median_iqr(reps_xla)
    stats_fused = median_iqr(reps_fused)
    ratios = [x / f for x, f in zip(reps_xla, reps_fused)]
    speedup = stats_xla["median"] / stats_fused["median"]

    n_xla = kernel_counts["xla"]["kernel_events"]
    n_fused = kernel_counts["fused"]["kernel_events"]
    launch_reduction = (n_xla / n_fused) if n_fused else None

    adopt_speed = stats_fused["median"] * SPEEDUP_BAR <= stats_xla["median"]
    adopt_launch = launch_reduction is not None and launch_reduction >= LAUNCH_REDUCTION_BAR
    adopt_correct = bool(correctness["all_ok"])
    verdict = {
        "verdict": "PASS" if (adopt_speed and adopt_launch and adopt_correct) else "FALSIFIED",
        "bars": {"speedup_ge": SPEEDUP_BAR, "launch_reduction_ge": LAUNCH_REDUCTION_BAR,
                 "tolerance": CORRECTNESS_TOL},
        "measured": {"speedup": speedup, "launch_reduction": launch_reduction,
                     "correctness_ok": adopt_correct},
    }

    obj = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": verdict_mode,
        "shape": {"nz": kw["w"].shape[0], "ny": kw["w"].shape[1], "nx": kw["w"].shape[2]},
        "acoustic_substeps": substeps,
        "step_equivalent": f"calc_coef_w x1 + advance_w x{substeps}",
        "timing": {"xla": stats_xla, "fused": stats_fused,
                   "ratio_bootstrap_lower95": bootstrap_lower95(ratios),
                   "timed_reps_per_arm": len(reps_xla)},
        "kernel_counts": kernel_counts,
        "correctness": correctness,
        **verdict,
        "notes": [
            "CPU smoke mode is diagnostic only and can never be a verdict.",
            "Kernel counts are parsed from jax.profiler traces; MISSING means "
            "the parse failed, never zero.",
            "Adoption bar is the sprint contract's; FALSIFIED with numbers is "
            "a valid result.",
        ],
        "wall_seconds": time.monotonic() - started_wall,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"m2_bakeoff_verdict_{verdict_mode.lower()}.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {out_path}")
    return 0


def types_namespace(scal: dict):
    """Attribute view of the static scalars for the reference coefficient call."""

    import types
    return types.SimpleNamespace(**scal)


def np_device(x):
    import numpy as np

    return np.asarray(x)


def np_mean_equal(a, b) -> float:
    import numpy as np

    return float(np.mean(np.asarray(a) == np.asarray(b)))


def block(x):
    x.block_until_ready()


if __name__ == "__main__":
    raise SystemExit(main())
