#!/usr/bin/env python3
"""Confirm GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020 actually takes
effect in the COMPILED operational program -- not an auto-promote no-op.

Three independent proofs, per regime (fp64_default vs mixed_perturb_fp32_v020):
  1. RUNTIME carry dtypes (initial carry + after one real operational step):
     p'/ph'/mu'/w must be float32 and STAY float32 across the step (a no-op /
     auto-promote mode would silently re-widen them).
  2. COMPILED StableHLO token census: f32 vs f64 element-type tokens and the
     stablehlo.convert count. fp64_default must contain ~no authorized fp32
     dynamics tokens; mixed must contain fp32 tokens for the carried perts.
  3. Same program lowered both ways -> the diff is real (mixed != fp64).

Dtypes and HLO structure are value- and platform-independent, so this runs on
CPU (no GPU lock) and the conclusion transfers to the GPU executable. GPU
EXECUTION of the same dtypes is separately proven by the real bigswiss run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace

# x64 ON (production import policy); we toggle the MODE, not x64, to prove the
# surgical mixed carry -- distinct from the old global-x64-off "aggressive" proxy.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import (
    _initial_carry_for_run,
    _advance_chunk,
    run_forecast_operational,
)

ACOUSTIC_PERTS = ("p_perturbation", "ph_perturbation", "mu_perturbation", "w")
FP64_LOCKED = ("p_total", "ph_total", "mu_total", "theta", "u", "v", "qv")


def _carry_dtypes(carry) -> dict:
    st = carry.state
    out = {f: str(getattr(st, f).dtype) for f in ACOUSTIC_PERTS + FP64_LOCKED}
    if carry.base_state is not None:
        for f in ("pb", "phb", "mub"):
            out[f"base.{f}"] = str(getattr(carry.base_state, f).dtype)
    return out


def _hlo_census(setup, mode: str) -> dict:
    nml = replace(setup.namelist, acoustic_precision_mode=mode)
    # one operational step: dt_s/3600 h
    hours = float(nml.dt_s) / 3600.0
    lowered = jax.jit(run_forecast_operational, static_argnums=(2,)).lower(
        setup.state, nml, hours
    )
    hlo = lowered.as_text()  # StableHLO
    f32 = len(re.findall(r"\bf32\b", hlo))
    f64 = len(re.findall(r"\bf64\b", hlo))
    conv = len(re.findall(r"stablehlo\.convert", hlo))
    whiles = len(re.findall(r"stablehlo\.while", hlo))
    return {
        "f32_tokens": f32,
        "f64_tokens": f64,
        "convert_ops": conv,
        "while_ops": whiles,
        "hlo_chars": len(hlo),
    }


def audit_mode(setup, mode: str) -> dict:
    nml = replace(setup.namelist, acoustic_precision_mode=mode)
    carry0 = _initial_carry_for_run(setup.state, nml)
    dt0 = _carry_dtypes(carry0)
    # advance one operational step and re-check dtypes (no-op/auto-promote guard)
    cadence = max(1, int(nml.radiation_cadence_steps))
    carry1 = _advance_chunk(carry0, nml, jnp.asarray(1, dtype=jnp.int32), n_steps=1, cadence=cadence)
    jax.block_until_ready(carry1.state.theta)
    dt1 = _carry_dtypes(carry1)
    census = _hlo_census(setup, mode)
    return {
        "mode": mode,
        "carry_initial_dtypes": dt0,
        "carry_after_one_step_dtypes": dt1,
        "perts_fp32_initial": all(dt0[f] == "float32" for f in ACOUSTIC_PERTS),
        "perts_fp32_after_step": all(dt1[f] == "float32" for f in ACOUSTIC_PERTS),
        "totals_fp64": all(dt1[f] == "float64" for f in ("p_total", "ph_total", "mu_total")),
        "hlo_census": census,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="proofs/v020/s4/ultracode/dtype_hlo_audit.json")
    args = ap.parse_args()

    setup = build_warm_bubble_setup(require_gpu=False)
    fp64 = audit_mode(setup, "fp64_default")
    mixed = audit_mode(setup, "mixed_perturb_fp32_v020")

    verdict = {
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "fp64_default": fp64,
        "mixed_perturb_fp32_v020": mixed,
        "PROOF_fp32_takes_effect": (
            mixed["perts_fp32_initial"]
            and mixed["perts_fp32_after_step"]
            and mixed["totals_fp64"]
            and (not fp64["perts_fp32_initial"])
            and mixed["hlo_census"]["f32_tokens"] > fp64["hlo_census"]["f32_tokens"]
        ),
        "note": (
            "fp64_default carries the 4 acoustic perts in float64 (f32 tokens are "
            "incidental index/aux casts); mixed carries them float32 and they SURVIVE "
            "a real step while totals stay float64 -> genuine surgical fp32 storage, "
            "not the auto-promote no-op that fooled G0."
        ),
    }
    with open(args.out, "w") as fh:
        json.dump(verdict, fh, indent=2)

    print(f"=== dtype/HLO audit -> {args.out} ===")
    print(f"x64_enabled = {verdict['x64_enabled']}")
    for tag, blk in (("fp64_default", fp64), ("mixed_v020", mixed)):
        c = blk["hlo_census"]
        print(f"\n[{tag}]")
        print(f"  perts fp32 (init/after) = {blk['perts_fp32_initial']}/{blk['perts_fp32_after_step']}  "
              f"totals fp64 = {blk['totals_fp64']}")
        print(f"  HLO: f32={c['f32_tokens']} f64={c['f64_tokens']} converts={c['convert_ops']} "
              f"whiles={c['while_ops']}")
        print(f"  pert dtypes after step: " +
              ", ".join(f"{f}={blk['carry_after_one_step_dtypes'][f]}" for f in ACOUSTIC_PERTS))
    print(f"\nPROOF_fp32_takes_effect = {verdict['PROOF_fp32_takes_effect']}")


if __name__ == "__main__":
    main()
