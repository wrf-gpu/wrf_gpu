#!/usr/bin/env python3
"""OFF-path proof (E151): the CAM radiation change leaves existing configurations' step programs identical.

Traces the operational physics+boundary step (``_physics_boundary_step``) of the real Swiss d01 case for the release
radiation (ra_lw=ra_sw=4 RRTMG) and the legacy held-rate radiation family (ra_lw=1 RRTM + ra_sw=1 Dudhia), ordinary
and radiation step, with ``jax.make_jaxpr``; writes location-stripped jaxpr text + digests.  Run once per tree
(base export / candidate) with identical environment and diff the digests.  (Pattern of lane o1-ruc RUC05.)

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<tree>/src taskset -c 28 \\
        python proofs/cam_rad/cam_off_identity.py --input examples/switzerland_d01 --out DIR
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
ap.add_argument("--interpret", action="store_true", help="Pallas interpret mode (release fast defaults on CPU)")
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
variants = {"release_rrtmg44": (nml, carry)}
nml_legacy = replace(nml, ra_lw_physics=1, ra_sw_physics=1)
carry_legacy = om._initial_carry_for_run(bundles["d01"].state, nml_legacy).replace(
    noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad, radiation_diagnostics=None)
variants["legacy_rrtm1_dudhia1"] = (nml_legacy, carry_legacy)
record = {"fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "interpret": args.interpret, "gpuwrf_file": gpuwrf.__file__,
          "ra": {k: (int(n.ra_lw_physics), int(n.ra_sw_physics)) for k, (n, _c) in variants.items()}, "programs": {}}
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
