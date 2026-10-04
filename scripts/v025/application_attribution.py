#!/usr/bin/env python3
"""H2 and H3: application-family attribution (CPU-only parsing).

Manager review on main `6a497c0e`: *"XLA phase-family attribution is not
radiation-vs-dycore application-family attribution."* Correct, and I had dropped
H2/H3 by treating the two as the same thing. They are not:

* `nvtx_exclusive.py` answers **which XLA phase** the compiler spent time in
  (codegen, hlo_pass, autotune). That is a property of the compiler.
* H2/H3 ask **which part of WRF** that time belongs to (radiation, dycore
  advection, PBL). That is a property of the model, and no XLA phase name
  carries it.

The pre-registered statements, restored verbatim in substance:

| | hypothesis | falsified when |
|---|---|---|
| **H2** | one module dominates and its cost is concentrated in a few HLO passes | **no pass exceeds 15%** of that module's compile |
| **H3** | the radiation subgraph dominates that module | **radiation < 25%** of module compile |

Two different mechanisms, because the two questions have different evidence:

* **H2 is measured.** Per-pass exclusive seconds from NVTX, scoped to the
  dominant module. A real measurement of real seconds.
* **H3 is apportioned.** No profiler reports "seconds of compile spent on
  radiation". The attribution runs over dumped HLO, mapping instruction metadata
  to the frozen A6 families, and apportions module compile time by a structural
  weight. That is a **proxy**, and it is labelled one everywhere it appears --
  reporting it as measured seconds would be the same category error the manager
  just corrected.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
A6_CENSUS = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"

# Pre-registered falsification bars. Frozen numbers, not judgement calls.
H2_MAX_PASS_SHARE = 0.15
H3_MIN_RADIATION_SHARE = 0.25

RADIATION_FAMILY = "physics.radiation"

# `metadata={op_name="jit(f)/jit(main)/rrtmg_lw/..." source_file=...}`
_OP_NAME = re.compile(r'op_name="([^"]*)"')
_FUSED_CALL = re.compile(r"calls=%([\w.\-]+)")
_COMPUTATION = re.compile(r"^\s*%?([\w.\-]+)\s*\(|^ENTRY\s+%?([\w.\-]+)")


def load_family_map(census_path: Path = A6_CENSUS) -> dict[str, str]:
    """operator name -> A6 application family, from the frozen census."""
    census = json.loads(census_path.read_text())
    return {op["name"]: op["family"] for op in census["operators"]}


def family_of(op_name: str, family_map: dict[str, str]) -> str | None:
    """Longest-match an HLO `op_name` against the frozen operator names.

    Longest match matters: `calc_p_rho` is a prefix of `calc_p_rho_step`, and a
    first-match rule would silently fold the second family into the first.
    """
    best: tuple[int, str] | None = None
    for operator, family in family_map.items():
        if operator in op_name and (best is None or len(operator) > best[0]):
            best = (len(operator), family)
    return best[1] if best else None


def attribute_hlo(hlo_text: str, family_map: dict[str, str]) -> dict[str, Any]:
    """Structural attribution of HLO instructions to A6 families.

    The weight is the instruction count. It is a PROXY for compile cost, not a
    measurement of it.
    """
    per_family: dict[str, int] = defaultdict(int)
    unmatched = 0
    total = 0
    for match in _OP_NAME.finditer(hlo_text):
        total += 1
        family = family_of(match.group(1), family_map)
        if family is None:
            unmatched += 1
        else:
            per_family[family] += 1

    attributed = total - unmatched
    return {
        "instructions_with_metadata": total,
        "attributed": attributed,
        "unattributed": unmatched,
        "unattributed_share": (unmatched / total) if total else 1.0,
        "per_family_instructions": dict(sorted(per_family.items(), key=lambda kv: -kv[1])),
        "per_family_share": {
            family: count / attributed
            for family, count in sorted(per_family.items(), key=lambda kv: -kv[1])
        } if attributed else {},
        "weight": "instruction count",
        "IS_A_PROXY": (
            "no profiler reports compile seconds per WRF family. This apportions a module's "
            "compile time by a structural weight over dumped HLO. It is not a measurement of "
            "per-family seconds and must not be reported as one."
        ),
    }


def h2_verdict(pass_exclusive_seconds: dict[str, float],
               module_compile_seconds: float) -> dict[str, Any]:
    """H2: is the module's compile concentrated in a few passes?

    Falsified when NO pass exceeds 15% of module compile.
    """
    if module_compile_seconds <= 0:
        return {"hypothesis": "H2", "verdict": "BLOCKED",
                "reason": "module compile time is not positive"}
    if not pass_exclusive_seconds:
        return {"hypothesis": "H2", "verdict": "BLOCKED",
                "reason": "no pass timings for the module"}

    shares = {name: seconds / module_compile_seconds
              for name, seconds in pass_exclusive_seconds.items()}
    ranked = sorted(shares.items(), key=lambda kv: -kv[1])
    top_name, top_share = ranked[0]
    over_bar = [{"pass": n, "share": s} for n, s in ranked if s > H2_MAX_PASS_SHARE]

    return {
        "hypothesis": "H2",
        "verdict": "SUPPORTED" if over_bar else "FALSIFIED",
        "reason": (
            f"{len(over_bar)} pass(es) exceed the {H2_MAX_PASS_SHARE:.0%} bar; the largest is "
            f"{top_name!r} at {top_share:.1%}"
            if over_bar else
            f"no pass exceeds {H2_MAX_PASS_SHARE:.0%} of module compile (largest {top_name!r} "
            f"at {top_share:.1%}), so the cost is diffuse rather than concentrated"
        ),
        "bar": H2_MAX_PASS_SHARE,
        "largest_pass": {"pass": top_name, "share": top_share},
        "passes_over_bar": over_bar,
        "top_passes": [{"pass": n, "share": s} for n, s in ranked[:10]],
        "basis": "measured exclusive NVTX seconds, scoped to the dominant module",
    }


def h3_verdict(family_share: dict[str, float], *, unattributed_share: float,
               max_unattributed: float = 0.05) -> dict[str, Any]:
    """H3: does radiation dominate the module? Falsified when radiation < 25%."""
    if unattributed_share > max_unattributed:
        return {
            "hypothesis": "H3", "verdict": "BLOCKED",
            "reason": (f"{unattributed_share:.1%} of HLO instructions could not be attributed "
                       f"to an A6 family, above the {max_unattributed:.0%} budget"),
        }
    if not family_share:
        return {"hypothesis": "H3", "verdict": "BLOCKED", "reason": "no family attribution"}

    radiation = family_share.get(RADIATION_FAMILY, 0.0)
    ranked = sorted(family_share.items(), key=lambda kv: -kv[1])
    return {
        "hypothesis": "H3",
        "verdict": "SUPPORTED" if radiation >= H3_MIN_RADIATION_SHARE else "FALSIFIED",
        "reason": (
            f"radiation is {radiation:.1%} of the attributed module structure, "
            f"{'at or above' if radiation >= H3_MIN_RADIATION_SHARE else 'below'} the "
            f"{H3_MIN_RADIATION_SHARE:.0%} bar"
        ),
        "bar": H3_MIN_RADIATION_SHARE,
        "radiation_share": radiation,
        "largest_family": {"family": ranked[0][0], "share": ranked[0][1]},
        "ranked_families": [{"family": f, "share": s} for f, s in ranked],
        "basis": "STRUCTURAL PROXY over dumped HLO, not measured per-family seconds",
    }


def cpu_side_family_compile_seconds(census_path: Path = A6_CENSUS) -> dict[str, Any]:
    """A real per-family compile measurement that already exists, CPU-side.

    A6 compiled each of the 49 operators SEPARATELY and timed each one. Summing
    by family gives genuinely measured per-family compile seconds -- for
    standalone operators on CPU. That is emphatically NOT the same as their share
    inside the fused GPU module, which is what H3 asks about; a fused module
    shares work between families that standalone compiles pay for separately.
    It is recorded because it is real, measured, already paid for, and it puts a
    prior on H3 before any GPU window.
    """
    census = json.loads(census_path.read_text())
    per_family: dict[str, float] = defaultdict(float)
    fields_used: set[str] = set()
    for operator in census["operators"]:
        if operator.get("compile_seconds") is not None:
            seconds, field = operator["compile_seconds"], "compile_seconds"
        else:
            seconds, field = operator.get("seconds", 0.0), "seconds"
        fields_used.add(field)
        per_family[operator["family"]] += float(seconds or 0.0)

    total = sum(per_family.values())
    ranked = sorted(per_family.items(), key=lambda kv: -kv[1])
    return {
        "total_seconds": total,
        "per_family_seconds": dict(ranked),
        "per_family_share": {f: (s / total if total else None) for f, s in ranked},
        "radiation_share": (per_family.get(RADIATION_FAMILY, 0.0) / total) if total else None,
        "timing_field_used": sorted(fields_used),
        "timing_field_note": (
            "`seconds` is lower+compile combined. The committed A6 census predates the "
            "lower/compile split, so this prior measures TOTAL per-operator time, not "
            "compilation alone. Censuses produced by the shape sweep carry the split."
            if "seconds" in fields_used else
            "per-operator compile time, separated from lowering"
        ),
        "SCOPE": (
            "standalone per-operator CPU runs from A6, summed by family. NOT the fused GPU "
            "module H3 is about -- fusion shares work that standalone compiles pay for "
            "separately. A prior, not an answer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlo", type=Path, default=None,
                        help="dumped HLO text for the dominant module")
    parser.add_argument("--pass-seconds-json", type=Path, default=None,
                        help="{pass_name: exclusive_seconds} scoped to that module")
    parser.add_argument("--module-compile-seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-prior-only", action="store_true",
                        help="emit only the CPU-side family prior; needs no GPU artifacts")
    args = parser.parse_args()

    family_map = load_family_map()
    obj: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.application_attribution.v1",
        "families": sorted(set(family_map.values())),
        "operators": len(family_map),
        "cpu_side_family_prior": cpu_side_family_compile_seconds(),
        "bars": {"H2_max_pass_share": H2_MAX_PASS_SHARE,
                 "H3_min_radiation_share": H3_MIN_RADIATION_SHARE},
    }

    if not args.cpu_prior_only:
        if args.hlo and args.hlo.is_file():
            attribution = attribute_hlo(args.hlo.read_text(errors="replace"), family_map)
            obj["hlo_attribution"] = attribution
            obj["H3"] = h3_verdict(attribution["per_family_share"],
                                   unattributed_share=attribution["unattributed_share"])
        else:
            obj["H3"] = {"hypothesis": "H3", "verdict": "BLOCKED", "reason": "no HLO dump"}

        if args.pass_seconds_json and args.module_compile_seconds:
            passes = json.loads(args.pass_seconds_json.read_text())
            obj["H2"] = h2_verdict(passes, args.module_compile_seconds)
        else:
            obj["H2"] = {"hypothesis": "H2", "verdict": "BLOCKED",
                         "reason": "no module-scoped pass timings"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
