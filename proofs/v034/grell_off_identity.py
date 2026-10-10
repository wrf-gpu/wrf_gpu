#!/usr/bin/env python3
"""OFF-path proof (E151) for o1-grell: wiring cu=5/93 leaves existing configurations' step
programs identical.

Traces the operational physics+boundary step (``_physics_boundary_step``) of the real Swiss
d01 case (release suite Noah-MP/Thompson/MYNN/RRTMG, cu=0) plus the KF (cu=1) and
Grell-Freitas (cu=3) cumulus variants, ordinary and radiation step, with ``jax.make_jaxpr``;
writes location- and tree-root-stripped jaxpr text + digests.  Run once per tree (main export
/ candidate) with identical environment and diff the digests.  (Pattern of o1-ruc RUC05.)

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<tree>/src taskset -c 13 \\
        python proofs/v034/grell_off_identity.py --input examples/switzerland_d01 --out DIR
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
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_cpu_enable_async_dispatch", False)
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
# Pallas kernels execute during case loading: interpret mode on CPU, same in both arms.
from jax.experimental import pallas as pl  # noqa: E402

_orig_pallas_call = pl.pallas_call
pl.pallas_call = lambda *a, **k: _orig_pallas_call(*a, **{**k, "interpret": True})

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

ROOT = str(Path(gpuwrf.__file__).resolve().parents[2])
_LOC = re.compile(r"(/[^\s\"',)]*\.py)(:\d+(:\d+)?)?")
_NAME_SRC = re.compile(r"name_and_src_info=[^\n]*?(?=,\s*\w+=|\])")


def normalise(text: str) -> str:
    text = text.replace(ROOT, "<ROOT>")
    text = _NAME_SRC.sub("name_and_src_info=<src>", text)
    return _LOC.sub("<loc>", text)


args.out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
nml = bundles["d01"].namelist
carry = carries["d01"]
variants = {"release_cu0": (nml, carry)}
for cu in (1, 3):
    n_cu = replace(nml, cu_physics=cu)
    c_cu = om._initial_carry_for_run(bundles["d01"].state, n_cu).replace(
        noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad,
        radiation_diagnostics=carry.radiation_diagnostics)
    variants[f"release_cu{cu}"] = (n_cu, c_cu)
record = {"fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "gpuwrf_file": gpuwrf.__file__,
          "programs": {}}
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
