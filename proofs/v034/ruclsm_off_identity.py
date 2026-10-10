#!/usr/bin/env python3
"""OFF-path proof (E151): the RUC change leaves existing configurations' step programs identical.

Traces the operational physics+boundary step (``_physics_boundary_step``) of the real
Swiss d01 Noah-MP/Thompson/MYNN/RRTMG configuration (ordinary and radiation step, plus a
KF cu_physics=1 variant) with ``jax.make_jaxpr`` in the source tree on ``sys.path`` and
writes the location-stripped jaxpr text + digest.  Run once per tree (base export /
candidate) with identical environment, then diff the digests.

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<tree>/src taskset -c 9 \\
        python proofs/v034/ruclsm_off_identity.py --out DIR [--interpret]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--interpret", action="store_true", help="Pallas interpret mode (fast defaults on CPU)")
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_cpu_enable_async_dispatch", False)
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
if args.interpret:
    from jax.experimental import pallas as pl

    _orig = pl.pallas_call
    pl.pallas_call = lambda *a, **k: _orig(*a, **{**k, "interpret": True})

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

_LOC = re.compile(r"(/[^\s\"',)]*\.py)(:\d+(:\d+)?)?")
_NAME_SRC = re.compile(r"name_and_src_info=[^\n]*?(?=,\s*\w+=|\])")


def normalise(text: str) -> str:
    text = _NAME_SRC.sub("name_and_src_info=<src>", text)
    return _LOC.sub("<loc>", text)


args.out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
nml = bundles["d01"].namelist
carry = carries["d01"]
variants = {"noahmp_thompson_mynn_rrtmg": (nml, carry)}
nml_kf = replace(nml, cu_physics=1)
carry_kf = om._initial_carry_for_run(bundles["d01"].state, nml_kf).replace(
    noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad,
    radiation_diagnostics=carry.radiation_diagnostics)
variants["noahmp_thompson_mynn_rrtmg_kf"] = (nml_kf, carry_kf)
record = {"fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "interpret": args.interpret,
          "gpuwrf_file": gpuwrf.__file__, "programs": {}}
for vname, (namelist, c) in variants.items():
    for rad in (False, True):
        jaxpr = jax.make_jaxpr(
            lambda cc, si, n=namelist, r=rad: om._physics_boundary_step(cc, n, si, run_radiation=r)
        )(c, jax.numpy.asarray(1, jax.numpy.int32))
        text = normalise(str(jaxpr))
        key = f"{vname}/{'radiation' if rad else 'ordinary'}"
        (args.out / f"{key.replace('/', '__')}.jaxpr.txt").write_text(text)
        record["programs"][key] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                   "n_eqns_top": len(jaxpr.jaxpr.eqns), "chars": len(text)}
        print(key, record["programs"][key], flush=True)
(args.out / "identity.json").write_text(json.dumps(record, indent=1) + "\n")
