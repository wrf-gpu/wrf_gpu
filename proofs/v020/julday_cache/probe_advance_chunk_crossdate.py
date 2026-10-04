#!/usr/bin/env python3
"""END-TO-END cross-date cache test on the REAL operational `_advance_chunk`.

Builds the v0.17 bigswiss single-domain operational state + namelist ONCE from the
real wrfinput, then lowers+compiles the production `_advance_chunk` (full RRTMG
SW+LW + Noah-MP physics, fp64_default) for:

    DATE_A cold  -> writes a cache entry
    DATE_A again -> WARM HIT (harness check)
    DATE_B       -> HIT (fixed) / MISS (bug)

against an ISOLATED temp cache. It also dumps the lowered HLO text for DATE_A vs
DATE_B and asserts byte-equality (the compile cache keys on the HLO, so identical
HLO => guaranteed cross-date hit). A tiny n_steps keeps the compile fast and well
clear of the 9-nest megacompile (single domain, no fused cascade).

Minimize-megacompile rule honoured: single bigswiss d01, small n_steps.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

_CACHE_DIR = tempfile.mkdtemp(prefix="advchunk_crossdate_")
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
os.environ.pop("GPUWRF_JAX_CACHE", None)
# fp64_default production path; honour the bigswiss MYNN BouLac knob the AB uses.
os.environ.setdefault("GPUWRF_MYNN_BOULAC_ONZ", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import gpuwrf  # noqa: E402
from gpuwrf.runtime.compile_cache import cache_entry_count  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.io.radiation_static import load_radiation_static  # noqa: E402
from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: E402
import dataclasses  # noqa: E402

BIG = os.environ.get("V020_BIG_INPUT", "<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init")
N_STEPS = int(os.environ.get("PROBE_N_STEPS", "2"))

DATE_A = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)
DATE_B = datetime(2023, 7, 15, 0, tzinfo=timezone.utc)
DATE_C = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)


def _build_namelist_and_carry():
    """Build a real bigswiss operational namelist + committed carry (fp64_default)."""
    replay = build_replay_case(BIG, domain="d01", standalone=True)
    radiation_static, _ = load_radiation_static(
        replay.run, "d01", grid=replay.grid, metrics=replay.metrics
    )
    nl = OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=18.0,
        acoustic_substeps=4,
        radiation_cadence_steps=int(os.environ.get("PROBE_RADT_STEPS", "100")),
        force_fp64=True,
        radiation_static=radiation_static,
    )  # run_physics / run_boundary default True (full operational path)
    carry = om._committed_initial_carry_for_run(replay.state, nl)
    return nl, carry


def _with_date(nl, dt):
    return dataclasses.replace(nl, time_utc=dt)


def lower_text(nl, carry, start=1):
    """Lower (not compile) the real _advance_chunk and return the StableHLO text +
    a sha256, for a date-independence byte-compare."""
    clock_base = om.build_clock_base(nl)
    lowered = jax.jit(
        om._advance_chunk_fori, static_argnames=("n_steps", "cadence")
    ).lower(
        carry, nl, jnp.asarray(start, dtype=jnp.int32), clock_base,
        n_steps=N_STEPS, cadence=int(nl.radiation_cadence_steps),
    )
    txt = lowered.as_text()
    return txt, hashlib.sha256(txt.encode()).hexdigest()


def compile_chunk(nl, carry, start=1):
    clock_base = om.build_clock_base(nl)
    jax.jit(
        om._advance_chunk_fori, static_argnames=("n_steps", "cadence")
    ).lower(
        carry, nl, jnp.asarray(start, dtype=jnp.int32), clock_base,
        n_steps=N_STEPS, cadence=int(nl.radiation_cadence_steps),
    ).compile()


def count():
    return cache_entry_count(_CACHE_DIR)


def main():
    print(f"# advance_chunk cross-date probe  isolated_cache={_CACHE_DIR}  n_steps={N_STEPS}")
    print(f"# jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
    nl0, carry = _build_namelist_and_carry()
    print(f"# built bigswiss carry: theta shape {carry.state.theta.shape}  dtype {carry.state.theta.dtype}")

    nlA = _with_date(nl0, DATE_A)
    nlB = _with_date(nl0, DATE_B)
    nlC = _with_date(nl0, DATE_C)

    # --- HLO byte-equality across dates (decisive: cache keys on HLO) ---
    txtA, hA = lower_text(nlA, carry)
    txtB, hB = lower_text(nlB, carry)
    txtC, hC = lower_text(nlC, carry)
    hlo_identical_AB = (hA == hB)
    hlo_identical_AC = (hA == hC)
    print(f"# HLO sha256  A={hA[:16]}  B={hB[:16]}  C={hC[:16]}")
    print(f"# HLO A==B (different day-of-year): {hlo_identical_AB}")
    print(f"# HLO A==C (leap year):            {hlo_identical_AC}")

    # --- cache entry before/after across dates (compile) ---
    results = []
    for label, nl in [("A_cold", nlA), ("A_warm", nlA), ("B_other_date", nlB), ("C_leap", nlC)]:
        before = count()
        compile_chunk(nl, carry)
        after = count()
        rec = {"label": label, "date": nl.time_utc.isoformat(),
               "before": before, "after": after, "new": after - before,
               "warm_hit": (after == before and before > 0)}
        print(f"[{label:14s}] before={before} after={after} new={rec['new']} warm_hit={rec['warm_hit']}")
        results.append(rec)

    harness_ok = results[0]["new"] >= 1 and results[1]["warm_hit"]
    cross_date_hit = results[2]["warm_hit"] and results[3]["warm_hit"]

    verdict = {
        "case": "v017_bigswiss_d01_real_operational_advance_chunk",
        "physics": "RRTMG SW+LW (ra_sw=4/ra_lw=4) + fp64_default",
        "n_steps": N_STEPS,
        "jax_version": jax.__version__,
        "hlo_sha256_A": hA, "hlo_sha256_B": hB, "hlo_sha256_C": hC,
        "hlo_identical_across_dates_AB": hlo_identical_AB,
        "hlo_identical_across_dates_AC_leap": hlo_identical_AC,
        "harness_ok": harness_ok,
        "cross_date_cache_hit": cross_date_hit,
        "entry_results": results,
        "verdict": "FIXED" if (hlo_identical_AB and hlo_identical_AC and cross_date_hit) else "NOT_FIXED",
    }
    print("\n# VERDICT")
    print(json.dumps(verdict, indent=2))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advance_chunk_crossdate_result.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\n# wrote {out}")
    return 0 if verdict["verdict"] == "FIXED" else 1


if __name__ == "__main__":
    sys.exit(main())
