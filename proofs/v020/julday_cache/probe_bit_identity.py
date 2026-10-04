#!/usr/bin/env python3
"""Numerically-inert proof: clock_base (traced) == baked-literal, byte-for-byte.

Runs the REAL bigswiss operational `_advance_chunk` (fp64_default, full RRTMG
SW+LW physics) for the SAME forecast date TWICE:

    arm "baked"  : clock_base=None  -> the legacy path; julian/utc_minute are
                   Python-float literals baked into the HLO (pre-#91 behaviour).
    arm "traced" : clock_base=build_clock_base(nl) -> the #91 fix; the same date
                   scalars passed as TRACED runtime jit arguments.

The two output carries are byte-compared on every State leaf. Byte-equality proves
the #91 change is numerically inert (only the binding time of the date scalars
changed, not their values) -- so the fp64_default 963/963 bit-identity is preserved.

Single bigswiss d01, small n_steps (minimize-megacompile rule).
"""
from __future__ import annotations

import json
import os
import sys
import dataclasses
from datetime import datetime, timezone

os.environ.setdefault("GPUWRF_MYNN_BOULAC_ONZ", "1")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import gpuwrf  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.io.radiation_static import load_radiation_static  # noqa: E402
from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: E402

BIG = os.environ.get("V020_BIG_INPUT", "<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init")
N_STEPS = int(os.environ.get("PROBE_N_STEPS", "3"))
# radiation cadence 2 -> radiation FIRES on step 2 (exercises the date/solar path)
# without the pathological radiation-EVERY-step compile/run cost (cadence=1).
RADT_STEPS = int(os.environ.get("PROBE_RADT_STEPS", "2"))
DATE = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)


def build():
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
        radiation_cadence_steps=RADT_STEPS,
        force_fp64=True,
        radiation_static=radiation_static,
    )
    nl = dataclasses.replace(nl, time_utc=DATE)
    carry = om._committed_initial_carry_for_run(replay.state, nl)
    return nl, carry


def advance(nl, carry, clock_base):
    out = om._advance_chunk_fori(
        carry, nl, jnp.asarray(1, dtype=jnp.int32), clock_base,
        n_steps=N_STEPS, cadence=int(nl.radiation_cadence_steps),
    )
    jax.block_until_ready(out.state.theta)
    return out


def main():
    print(f"# bit-identity probe  n_steps={N_STEPS}  date={DATE.date()}  radt_steps={RADT_STEPS}")
    print(f"# jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
    nl, carry = build()
    print(f"# carry theta {carry.state.theta.shape} {carry.state.theta.dtype}")

    baked = advance(nl, carry, None)                       # legacy baked-literal path
    traced = advance(nl, carry, om.build_clock_base(nl))   # #91 traced path

    # State is a registered JAX pytree (not a dataclass): flatten with key paths and
    # byte-compare leaves positionally (identical treedef => positional match).
    a_leaves, a_tree = jax.tree_util.tree_flatten_with_path(baked.state)
    b_leaves, b_tree = jax.tree_util.tree_flatten_with_path(traced.state)
    assert a_tree == b_tree, "treedef mismatch between baked and traced state"
    report = {"date": DATE.isoformat(), "n_steps": N_STEPS, "radt_steps": RADT_STEPS,
              "fields": {}, "all_bit_identical": True}
    for (path, a), (_p, b) in zip(a_leaves, b_leaves):
        name = jax.tree_util.keystr(path)
        if not hasattr(a, "shape"):
            continue
        an = np.asarray(a); bn = np.asarray(b)
        if an.shape != bn.shape:
            report["fields"][name] = {"shape_mismatch": [list(an.shape), list(bn.shape)]}
            report["all_bit_identical"] = False
            continue
        equal = bool(np.array_equal(an, bn, equal_nan=True))
        if not equal:
            diff = float(np.nanmax(np.abs(an.astype(np.float64) - bn.astype(np.float64))))
            report["fields"][name] = {"bit_identical": False, "max_abs_diff": diff}
            report["all_bit_identical"] = False
        else:
            report["fields"][name] = {"bit_identical": True}

    n_id = sum(1 for v in report["fields"].values() if v.get("bit_identical"))
    n_tot = len(report["fields"])
    report["n_identical"] = n_id
    report["n_fields"] = n_tot
    print(f"\n# {n_id}/{n_tot} State leaves byte-identical (baked vs traced clock)")
    for name, v in report["fields"].items():
        if not v.get("bit_identical"):
            print(f"  !! {name}: {v}")
    report["verdict"] = "BIT_IDENTICAL" if report["all_bit_identical"] else "DIFFERS"
    print(f"# VERDICT: {report['verdict']}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bit_identity_result.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"# wrote {out}")
    return 0 if report["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
