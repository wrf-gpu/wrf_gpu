#!/usr/bin/env python3
"""Empirically test whether the JAX persistent compile cache HITS across dates.

#91 julday cache probe. Minimize-megacompile rule: this does NOT run the 9-nest.
It compiles the *exact same code path* that bakes per-date constants into the HLO
(`_compute_solar_geometry`, the solar/declination/EOT geometry consumed by the
radiation coupler, where `julian`/`utc_minute` are extracted as Python scalars at
trace time) on a tiny lat/lon grid, into an ISOLATED temp cache dir, and counts
cache entries before/after for:

    DATE_A cold  -> must WRITE a new entry
    DATE_A again -> must be a WARM HIT (no new entry)  [proves the harness works]
    DATE_B       -> HIT  => cache already works across dates (#91 no-op)
                    MISS => date-dependent-HLO bug confirmed

We use an isolated temp cache (not the shared 62k-entry /mnt cache) so the
before/after counts are unambiguous and we never pollute the shared cache.

Usage:
    python probe_julday_cache.py [post]   # 'post' tag => after-fix run
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

# Point the persistent cache at a fresh isolated dir BEFORE importing gpuwrf/jax.
_CACHE_DIR = tempfile.mkdtemp(prefix="julday_cache_probe_")
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
os.environ.pop("GPUWRF_JAX_CACHE", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import gpuwrf  # noqa: E402  (import hook enables x64 + configures the cache)
from gpuwrf.runtime.compile_cache import cache_entry_count  # noqa: E402
from gpuwrf.coupling import physics_couplers as pc  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "pre"

# Tiny grid -> compiles in milliseconds.
LAT = jnp.linspace(40.0, 47.0, 8).reshape(8, 1) * jnp.ones((8, 8))
LON = jnp.linspace(-5.0, 5.0, 8).reshape(1, 8) * jnp.ones((8, 8))

DATE_A = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)   # bigswiss case date
DATE_B = datetime(2023, 7, 15, 0, tzinfo=timezone.utc)   # very different day-of-year
DATE_C = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)   # leap year, same yday-ish


FIXED = os.environ.get("PROBE_FIXED", "0") == "1"


def compile_solar_geom(time_utc):
    """Compile the solar geometry kernel.

    PRE-FIX (FIXED=0): time_utc is a host datetime baked at trace time (the bug).
    POST-FIX (FIXED=1): the (julian, utc_minute) pair is built on the host via
    pc.time_utc_clock_base(time_utc) and passed as a TRACED jit argument
    (clock_base) -- exactly the operational-path fix -- so the HLO is identical
    for every date."""

    if FIXED:
        def fn(lat, lon, lead_seconds, clock_base):
            return pc._compute_solar_geometry(
                lat, lon, None, lead_seconds, clock_base=clock_base
            )

        lead = jnp.asarray(0.0, dtype=jnp.float64)
        clock_base = pc.time_utc_clock_base(time_utc)  # traced (julian, utc_minute)
        jax.jit(fn).lower(LAT, LON, lead, clock_base).compile()
    else:
        def fn(lat, lon, lead_seconds):
            return pc._compute_solar_geometry(lat, lon, time_utc, lead_seconds)

        lead = jnp.asarray(0.0, dtype=jnp.float64)
        # lower+compile forces a cache lookup/write.
        jax.jit(fn).lower(LAT, LON, lead).compile()


def count():
    return cache_entry_count(_CACHE_DIR)


def step(label, time_utc):
    before = count()
    compile_solar_geom(time_utc)
    after = count()
    wrote = after - before
    rec = {
        "label": label,
        "date": time_utc.isoformat(),
        "entries_before": before,
        "entries_after": after,
        "new_entries": wrote,
        "warm_hit": wrote == 0 and before > 0,
    }
    print(f"[{label:20s}] date={time_utc.date()}  before={before}  after={after}  "
          f"new={wrote}  warm_hit={rec['warm_hit']}")
    return rec


def main():
    print(f"# julday cache probe  TAG={TAG}  isolated_cache={_CACHE_DIR}")
    print(f"# jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
    results = []
    results.append(step("A_cold", DATE_A))
    results.append(step("A_warm_again", DATE_A))
    results.append(step("B_other_date", DATE_B))
    results.append(step("C_leap_year", DATE_C))

    a_cold = results[0]
    a_warm = results[1]
    b = results[2]

    harness_ok = (a_cold["new_entries"] == 1 and a_warm["warm_hit"] is True)
    cross_date_hits = (b["warm_hit"] is True)

    verdict = {
        "tag": TAG,
        "jax_version": jax.__version__,
        "harness_ok": harness_ok,
        "A_cold_wrote_one": a_cold["new_entries"] == 1,
        "A_warm_was_hit": a_warm["warm_hit"],
        "B_other_date_warm_hit": cross_date_hits,
        "C_leap_year_warm_hit": results[3]["warm_hit"],
        "cross_date_cache_hits": cross_date_hits,
        "results": results,
    }
    print("\n# VERDICT")
    print(json.dumps(verdict, indent=2))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"probe_result_{TAG}.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\n# wrote {out}")

    if not harness_ok:
        print("\n!! HARNESS BROKEN: A_cold must write exactly 1 entry and A_warm "
              "must be a warm hit. Investigate before trusting the cross-date result.")
        return 2
    if cross_date_hits:
        print("\n=> CACHE ALREADY HITS ACROSS DATES (DATE_B warm hit). #91 candidate NO-OP.")
        return 0
    print("\n=> CROSS-DATE CACHE MISS CONFIRMED (DATE_B wrote a new entry). "
          "Date-dependent HLO bug is real.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
