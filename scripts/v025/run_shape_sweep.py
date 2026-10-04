#!/usr/bin/env python3
"""Step 0: the compile shape sweep, as ONE executable command (CPU-only).

Manager review of `dd5ed7b8`: "Manual prose is not a proof harness." This is the
harness. It launches every repeat as a fresh process in the registered rotation,
verifies the invariants that make the comparison meaningful, and emits
`compile_shape_sweep.json` with the three-band verdict already applied.

What it verifies before believing any number
--------------------------------------------
* **Exact state pairs.** Every shape names both wrfout files explicitly with a
  full SHA-256, checked byte-for-byte. Implicit first/last selection is refused
  by `real_state.select_state_pair` -- on d02 it would silently span 6.75 days.
* **Identical operator inventory at every shape.** A digest over the 49 operator
  names must match across all shapes. If d06 quietly ran fewer operators than
  d01, the comparison would measure inventory, not shape.
* **Cumulus and GWD lower successfully at every shape.** The smaller domains run
  `cu=0, gwd=0` in production, so the diagnostic inventory FORCES those
  operators in. Inclusion in a list is not evidence they worked; their
  `LOWERED` status is checked per shape.
* **Snapshot config recorded separately from the forced inventory**, so a reader
  can see that d06's namelist says `cu=0` while the sweep ran cumulus anyway.

This is a DIAGNOSTIC instrument. It qualifies nothing and nominates no case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import wrf_source_authority as wsa  # noqa: E402

# The registered shapes. Both state files are named EXACTLY and hash-verified.
# No hash is written here. A literal in this table is a hash nobody verified, and
# a truncated one reads like a real digest while matching nothing. `resolve_hashes`
# computes all eight over the actual bytes at run time, and refuses to proceed if
# a file is missing.
SHAPES: dict[str, dict[str, Any]] = {
    "d06": {
        "run_dir": "<DATA_ROOT>/alisios/runs/20260704_00z/real1km/real1km_9nest_case",
        "previous": "wrfout_d06_2026-07-04_06:00:00",
        "snapshot": "wrfout_d06_2026-07-04_06:20:00",
    },
    "d03": {
        "run_dir": "<DATA_ROOT>/alisios/runs/20260704_00z/real1km/real1km_9nest_case",
        "previous": "wrfout_d03_2026-07-04_06:00:00",
        "snapshot": "wrfout_d03_2026-07-04_06:20:00",
    },
    "d01": {
        "run_dir": "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1",
        "previous": "wrfout_d01_2026-07-26_00:00:00",
        "snapshot": "wrfout_d01_2026-07-26_01:00:18",
    },
    "d02": {
        "run_dir": ("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
                    "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case"),
        "previous": "wrfout_d02_2026-07-26_00:00:00",
        "snapshot": "wrfout_d02_2026-07-26_01:00:00",
    },
}

ROTATIONS = [
    ["d01", "d02", "d03", "d06"],
    ["d02", "d03", "d06", "d01"],
    ["d03", "d06", "d01", "d02"],
]

FORCED_OPERATORS = ("kain_fritsch_cumulus", "gwdo_gravity_wave_drag")

# The FROZEN A6 inventory, as adjudicated by the manager on main `6a497c0e`.
#
# Equality across the four shapes is NOT sufficient. Four shapes could agree on a
# wrong inventory -- an operator silently dropped from the harness would be dropped
# identically everywhere, every cross-shape check would pass, and the sweep would
# compare a consistent but unfrozen subset. The digest is therefore required
# against this constant, not merely against the other shapes.
FROZEN_OPERATOR_COUNT = 49
FROZEN_OPERATOR_DIGEST = (
    "abd39cd2a8cce2876cf4d824658ce08840fa4d93ab4d20846250306df980fd9e"
)

SWEEP_ENV = {
    "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "GPUWRF_JAX_CACHE": "0", "GPUWRF_JAX_CACHE_LOCK": "0",
    "PYTHONPATH": "src",
}
SWEEP_ENV.update(
    {variable: str(wsa.CANONICAL_ROOT) for variable in wsa.ROOT_ENV_VARS}
)
AFFINITY = "0-3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_hashes() -> dict[str, dict[str, Any]]:
    """Fill in the full SHA-256 of both state files for every shape."""
    resolved: dict[str, dict[str, Any]] = {}
    for shape, spec in SHAPES.items():
        root = Path(spec["run_dir"])
        prev, snap = root / spec["previous"], root / spec["snapshot"]
        entry = dict(spec)
        entry["run_dir"] = str(root)
        entry["exists"] = prev.is_file() and snap.is_file()
        entry["sha256"] = (
            [sha256_file(prev), sha256_file(snap)] if entry["exists"] else [None, None]
        )
        entry["bytes"] = (
            [prev.stat().st_size, snap.stat().st_size] if entry["exists"] else [None, None]
        )
        resolved[shape] = entry
    return resolved


def census_command(shape: str, spec: dict[str, Any], repeat: int, out: Path) -> list[str]:
    digests = spec.get("sha256") or []
    if len(digests) != 2 or not all(isinstance(d, str) and len(d) == 64 for d in digests):
        raise SystemExit(
            f"{shape}: refusing to build a command without two full 64-char digests "
            f"(got {digests!r}). A missing or truncated hash disables the fail-closed "
            f"byte verification that the whole comparison rests on."
        )
    return [
        "taskset", "-c", AFFINITY, sys.executable,
        "scripts/v025/build_hlo_census.py",
        "--domain", shape,
        "--run-dir", spec["run_dir"],
        "--previous", spec["previous"],
        "--snapshot", spec["snapshot"],
        "--expect-sha256", spec["sha256"][0],
        "--expect-sha256", spec["sha256"][1],
        "--repeat-tag", f"r{repeat}",
        "--out", str(out),
    ]


def _default_launch(command: Sequence[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(list(command), cwd=cwd, env=env, capture_output=True,
                          text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def operator_digest(census: dict[str, Any]) -> str:
    names = sorted(op["name"] for op in census.get("operators", []))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def bootstrap_ci(values: list[float], *, resamples: int = 20000, seed: int = 20260727):
    if len(values) < 2:
        return {"low": None, "high": None, "n": len(values)}
    rng = random.Random(seed)
    medians = sorted(
        st.median(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    return {"low": medians[int(0.025 * resamples)],
            "high": medians[int(0.975 * resamples)], "n": len(values)}


def summarise(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean = st.mean(values)
    return {
        "n": len(values), "median": st.median(values), "mean": mean,
        "cv_percent": (100.0 * st.pstdev(values) / mean) if mean else None,
        "bootstrap_ci95": bootstrap_ci(values),
        "values": values,
    }


def verdict(medians: dict[str, float]) -> dict[str, Any]:
    """The three-band rule, fixed in advance, applied mechanically."""
    if len(medians) < 2:
        return {"band": "INSUFFICIENT", "reason": "fewer than two shapes produced a median"}
    low, high = min(medians.values()), max(medians.values())
    spread = (high / low) if low > 0 else float("inf")
    if spread <= 1.20:
        band, action = "GRID_INDEPENDENT", (
            "hypothesis upheld. The fully representative d01 case is the right optimisation "
            "target; proceed to Step 1 attribution. No case substitution implied or requested."
        )
    elif spread <= 2.0:
        band, action = "PARTIAL_SENSITIVITY", (
            "neither upheld nor falsified. Report the measured slope; infer NO cheaper case. "
            "Step 1 proceeds unchanged with the slope carried as a caveat."
        )
    else:
        band, action = "FALSIFIED", (
            "grid size is a real lever. Still no case substitution: the gates are unchanged and "
            "d01 stays the target. The finding goes to the manager as evidence."
        )
    return {"band": band, "spread_ratio": spread, "next_action": action,
            "min_median": low, "max_median": high}


def run_sweep(
    *,
    out: Path,
    raw_dir: Path,
    repeats: int = 3,
    launch: Callable[..., tuple[int, str]] = _default_launch,
    read_census: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    read_census = read_census or (lambda p: json.loads(p.read_text()))
    shapes = resolve_hashes()
    missing = [s for s, e in shapes.items() if not e.get("exists")]
    if missing:
        raise SystemExit(f"state files missing for {missing}; refusing to sweep")

    env = dict(os.environ)
    env.update(SWEEP_ENV)
    raw_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        order = ROTATIONS[(repeat - 1) % len(ROTATIONS)]
        for shape in order:
            target = raw_dir / f"census_{shape}_r{repeat}.json"
            command = census_command(shape, shapes[shape], repeat, target)
            code, log = launch(command, cwd=REPO, env=env)
            record: dict[str, Any] = {
                "repeat": repeat, "shape": shape, "order": order,
                "command": command, "returncode": code, "out": str(target),
            }
            if code != 0 or not target.is_file():
                record["status"] = "FAILED"
                record["log_tail"] = log
            else:
                census = read_census(target)
                ops = census.get("operators", [])
                lowered = {o["name"] for o in ops if o.get("status") == "LOWERED"}
                record.update({
                    "status": "OK",
                    "operator_digest": operator_digest(census),
                    "operators_total": len(ops),
                    "operators_lowered": len(lowered),
                    "forced_operators_lowered": {
                        name: (name in lowered) for name in FORCED_OPERATORS
                    },
                    "snapshot_config": {
                        k: census.get("case_config", {}).get(k)
                        for k in ("nx", "ny", "nz", "cu_physics", "gwd_opt",
                                  "mp_physics", "ra_lw_physics", "bl_pbl_physics")
                    },
                    "lower_seconds_total": sum(o.get("lower_seconds", 0.0) for o in ops),
                    "compile_seconds_total": sum(o.get("compile_seconds", 0.0) for o in ops),
                })
            runs.append(record)

    ok = [r for r in runs if r["status"] == "OK"]
    digests = {r["operator_digest"] for r in ok}
    forced_failures = [
        {"shape": r["shape"], "repeat": r["repeat"], "forced": r["forced_operators_lowered"]}
        for r in ok if not all(r["forced_operators_lowered"].values())
    ]
    # Every run must match the FROZEN A6 inventory, not merely each other.
    frozen_mismatches = [
        {"shape": r["shape"], "repeat": r["repeat"],
         "operators_total": r["operators_total"], "operator_digest": r["operator_digest"]}
        for r in ok
        if r["operator_digest"] != FROZEN_OPERATOR_DIGEST
        or r["operators_total"] != FROZEN_OPERATOR_COUNT
    ]

    per_shape: dict[str, Any] = {}
    for shape in SHAPES:
        rows = [r for r in ok if r["shape"] == shape]
        per_shape[shape] = {
            "runs": len(rows),
            "lower_seconds": summarise([r["lower_seconds_total"] for r in rows]),
            "compile_seconds": summarise([r["compile_seconds_total"] for r in rows]),
            "snapshot_config": rows[0]["snapshot_config"] if rows else None,
        }

    invariants_ok = (len(digests) == 1 and not forced_failures
                     and not frozen_mismatches and len(ok) == len(runs))
    compile_medians = {s: v["compile_seconds"]["median"] for s, v in per_shape.items()
                       if v["compile_seconds"].get("median") is not None}
    lower_medians = {s: v["lower_seconds"]["median"] for s, v in per_shape.items()
                     if v["lower_seconds"].get("median") is not None}

    obj = {
        "schema": "wrf_gpu2.v025.m0.compile_shape_sweep.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": ("DIAGNOSTIC shape sweep. Qualifies nothing, nominates no case. Tests whether "
                    "compile time is grid-sensitive at a constant operator inventory."),
        "environment": {"affinity": AFFINITY, **SWEEP_ENV},
        "shapes": shapes,
        "rotation": ROTATIONS[:repeats],
        "runs": runs,
        "invariants": {
            "identical_operator_digest": len(digests) == 1,
            "operator_digests_seen": sorted(digests),
            "frozen_a6_inventory": {
                "required_count": FROZEN_OPERATOR_COUNT,
                "required_digest": FROZEN_OPERATOR_DIGEST,
                "source": "manager adjudication, main 6a497c0e",
                "matches_frozen": not frozen_mismatches,
                "mismatches": frozen_mismatches,
                "why_not_just_cross_shape_equality": (
                    "four shapes can agree on a WRONG inventory. An operator dropped from the "
                    "harness would be dropped identically at every shape, every cross-shape "
                    "check would pass, and the sweep would compare a consistent but unfrozen "
                    "subset. The digest is required against the frozen constant."
                ),
            },
            "forced_operators": list(FORCED_OPERATORS),
            "forced_operator_failures": forced_failures,
            "all_runs_succeeded": len(ok) == len(runs),
            "all_invariants_hold": invariants_ok,
            "note": ("the smaller domains run cu=0/gwd=0 in production; the diagnostic inventory "
                     "FORCES cumulus and GWD in at every shape, and their LOWERED status is "
                     "checked rather than assumed from list membership"),
        },
        "per_shape": per_shape,
        "verdict_compile": verdict(compile_medians) if invariants_ok else {
            "band": "BLOCKED", "reason": "invariants did not hold; no verdict is issued"},
        "verdict_lower": verdict(lower_medians) if invariants_ok else {
            "band": "BLOCKED", "reason": "invariants did not hold; no verdict is issued"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "proofs/v025/m0/compile_shape_sweep.json")
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/shape_sweep"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--print-plan", action="store_true",
                        help="print the 12 exact commands and the verified hashes, run nothing")
    args = parser.parse_args()

    if args.print_plan:
        shapes = resolve_hashes()
        plan = {"shapes": shapes, "rotation": ROTATIONS[:args.repeats], "commands": []}
        for repeat in range(1, args.repeats + 1):
            for shape in ROTATIONS[(repeat - 1) % len(ROTATIONS)]:
                plan["commands"].append(" ".join(census_command(
                    shape, shapes[shape], repeat,
                    args.raw_dir / f"census_{shape}_r{repeat}.json")))
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    obj = run_sweep(out=args.out, raw_dir=args.raw_dir, repeats=args.repeats)
    print(f"wrote {args.out}")
    print(f"  invariants hold: {obj['invariants']['all_invariants_hold']}")
    print(f"  compile verdict: {obj['verdict_compile']['band']}")
    return 0 if obj["invariants"]["all_invariants_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
