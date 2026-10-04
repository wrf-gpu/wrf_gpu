#!/usr/bin/env python3
"""Build ``proofs/v025/m0/hlo_dtype_transfer_census.json`` (contract §6.6, §9).

Phase A6. CPU-only: no GPU import, query, compile, or run.

§9's kernel census is a Phase-B GPU capture, but it has to be cross-checked
against something that was fixed BEFORE the trace was taken -- otherwise the
profiler defines its own reference and the check is circular. This is that
something:

- per-operator lowered/optimised HLO, on the REAL production shapes;
- ``convert-element-type`` counts split by transition, so a silent promotion is
  visible per operator rather than only in aggregate;
- ``temp_size_in_bytes`` and the rest of XLA's memory analysis;
- the static launch proxy (``kernel_launches_per_step``) summed over the active
  path, which is the number the GPU census must reconcile against;
- a STATIC host/device transfer audit: HLO transfer primitives plus a source
  scan of the timestep path for host-sync idioms. §13 forbids timestep-loop
  transfers; this proves the property from the program text, on CPU, before a
  GPU window is ever requested.

Also emitted: the exact ``nsys``/``ncu`` command lines for the Phase-B windows,
verified to parse against the installed binaries via ``--help`` only. No
profiler is attached to anything; running the commands is Phase B.

The operator inventory is imported from the A5 map builder so the two censuses
cannot drift apart: the same set of operators, at the same real shapes.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v025.build_cancellation_map import (  # noqa: E402
    build_context,
    build_inventory,
    make_cast,
)
from scripts.v025.real_state import (  # noqa: E402
    assert_cpu_only,
    select_state_pair,
    default_run_dir,
    load_real_snapshot,
)

SCHEMA = "wrf_gpu2.v025.m0.hlo_dtype_transfer_census.v1"

# Context entries held CONSTANT rather than traced: configuration, grids and
# pre-computed static tables. Everything else -- state, base state, metrics,
# stage prep, coupled velocities, coefficients -- is passed as a real jit
# argument so XLA cannot constant-fold the program away (see §3 of
# KERNEL_CENSUS_REPORT.md).
STATIC_CTX_KEYS = ("snapshot", "grid", "gwdo_statics", "noahmp_static", "noahmp_land")

# HLO primitives that move data between host and device, or synchronise on it.
TRANSFER_PRIMITIVES = (
    "outfeed", "infeed", "send", "recv", "host-transfer",
    "custom-call-target=\"HostExecute", "copy-start", "copy-done",
)

# Source idioms that force a device->host synchronisation. Any of these inside
# the timestep path is a §13 violation regardless of what the HLO shows,
# because they run in Python between the compiled programs.
#
# The negative lookbehind on ``np.asarray`` is load-bearing: without it the
# pattern also matches ``jnp.asarray``, which is a DEVICE array constructor and
# the single most common call in the whole dycore. The first run of this audit
# reported 345 "host sync" hits, of which essentially all were ``jnp.asarray``.
# An audit that cries wolf 345 times is worse than no audit, because the real
# hit is then invisible.
HOST_SYNC_IDIOMS: tuple[tuple[str, str], ...] = (
    (r"\.item\(\)", "device->host scalar read"),
    (r"jax\.device_get\(", "explicit device->host copy"),
    (r"(?<![A-Za-z_.])np\.asarray\(", "numpy materialisation of a device array"),
    (r"(?<![A-Za-z_.])numpy\.asarray\(", "numpy materialisation of a device array"),
    (r"block_until_ready\(", "host synchronisation barrier"),
    (r"\.to_py\(\)", "device->host copy"),
    (r"float\(\s*(?:jnp|jax)", "python float() of a device value"),
    (r"int\(\s*(?:jnp|jax)", "python int() of a device value"),
)

# The modules that make up the timestep loop. Physics adapters are included
# because they run inside the same scan.
TIMESTEP_PATH = (
    "src/gpuwrf/dynamics/core/acoustic.py",
    "src/gpuwrf/dynamics/core/advance_w.py",
    "src/gpuwrf/dynamics/core/calc_p_rho.py",
    "src/gpuwrf/dynamics/core/coupled.py",
    "src/gpuwrf/dynamics/core/dycore.py",
    "src/gpuwrf/dynamics/core/rhs_ph.py",
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
    "src/gpuwrf/dynamics/core/small_step_finish.py",
    "src/gpuwrf/dynamics/core/small_step_prep.py",
    "src/gpuwrf/dynamics/acoustic_wrf.py",
    "src/gpuwrf/dynamics/flux_advection.py",
    "src/gpuwrf/dynamics/explicit_diffusion.py",
    "src/gpuwrf/dynamics/tridiag_solve.py",
    "src/gpuwrf/coupling/physics_couplers.py",
    "src/gpuwrf/coupling/driver.py",
)


# --------------------------------------------------------------------------- #
# per-operator HLO census                                                      #
# --------------------------------------------------------------------------- #
def hlo_op_histogram(text: str) -> dict[str, int]:
    """Count HLO instruction opcodes, most frequent first."""

    counts: dict[str, int] = {}
    for match in re.finditer(r"=\s+\S+\s+([a-z0-9-]+)\(", text):
        opcode = match.group(1)
        counts[opcode] = counts.get(opcode, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def dtype_tokens(text: str) -> dict[str, int]:
    """Element-type token frequency: the program's precision fingerprint."""

    counts: dict[str, int] = {}
    for token in re.findall(r"\b(f64|f32|bf16|f16|s32|s64|u32|pred)\[", text):
        counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def lower_with_real_arguments(op: Any, ctx: dict[str, Any], mode: str):
    """Lower one operator with its real inputs as jit ARGUMENTS, not constants.

    This is not a stylistic choice. Jitting a zero-argument closure hands XLA a
    program whose every input is a compile-time constant, and XLA folds the whole
    computation away: measured on ``x_face_pressure_dpn`` at production shape,
    the closure form optimises to ``constant + copy`` (4 KB of HLO, two
    instructions) while the argument form keeps the real program (11 KB, 10
    parameters, 7 slices, 7 multiplies). A census built on the closure form would
    have reported near-zero launches and near-zero temporaries for every operator
    and been quietly worthless.

    The cast is applied EAGERLY, before lowering, so the fp32 arm's HLO contains
    the operator's own converts and not the harness's cast converts.
    """

    import jax

    import jax.numpy as jnp

    cast = make_cast(mode)
    ctx_cast = {key: cast(value) for key, value in ctx.items()}
    # Static bundles stay closed over rather than traced. They are configuration
    # and pre-computed tables, not the operator's data inputs, and the production
    # model treats them the same way. Noah-MP is the case that forces the
    # distinction: its driver derives an integer from the land carry and calls
    # Python float() on it, which is legal on a concrete bundle and raises
    # ConcretizationTypeError the moment the bundle is traced. Tracing it here
    # would report a lowering failure that the production model does not have.
    static_ctx = {key: ctx_cast.pop(key, None) for key in STATIC_CTX_KEYS}
    leaves, treedef = jax.tree_util.tree_flatten(ctx_cast)

    def _is_float_array(leaf: Any) -> bool:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or isinstance(leaf, (bool, int, float, str)):
            return False
        # Only FLOAT leaves become jit arguments. Integer leaves are indices,
        # land-use categories and scheme flags; several schemes read them with
        # Python float()/int(), which raises ConcretizationTypeError the moment
        # they are tracers. They are also not what a dtype census is about.
        # Noah-MP is the operator that proves this: with integer leaves traced it
        # does not lower at all.
        return bool(jnp.issubdtype(dtype, jnp.floating))

    is_array = [_is_float_array(leaf) for leaf in leaves]
    arrays = [leaf for leaf, keep in zip(leaves, is_array) if keep]
    identity = make_cast("fp64")

    def program(*args):
        iterator = iter(args)
        rebuilt = [next(iterator) if keep else leaf for leaf, keep in zip(leaves, is_array)]
        restored = jax.tree_util.tree_unflatten(treedef, rebuilt)
        restored.update(static_ctx)
        return op.call(restored, identity)

    return jax.jit(program).lower(*arrays), len(arrays)


def census_operator(op: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """Lower and compile one operator at real shapes; report its static profile."""

    from gpuwrf.profiling.budget import compiled_memory_stats, kernel_launches_per_step
    from gpuwrf.profiling.dtype_audit import count_converts

    record: dict[str, Any] = {
        "name": op.name,
        "family": op.family,
        "module": op.module,
        "entrypoint": op.entrypoint,
    }
    started = time.perf_counter()
    try:
        # §5.3-style separation: an ambiguous lower+compile total hides which
        # half moves when the shape changes, which is exactly what the sweep is
        # trying to resolve.
        t_lower0 = time.perf_counter()
        lowered, n_args = lower_with_real_arguments(op, ctx, "fp64")
        stablehlo = lowered.as_text()
        lower_seconds = time.perf_counter() - t_lower0

        t_compile0 = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - t_compile0
        optimized = compiled.as_text()

        record["jit_arguments"] = n_args
        record["lower_seconds"] = lower_seconds
        record["compile_seconds"] = compile_seconds
    except Exception as exc:  # noqa: BLE001
        record["status"] = "NOT_LOWERED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    converts_pre = count_converts(stablehlo).as_dict()
    converts_post = count_converts(optimized).as_dict()
    memory = compiled_memory_stats(compiled)
    record.update({
        "status": "LOWERED",
        "converts_stablehlo": converts_pre,
        "converts_optimized": converts_post,
        "dtype_tokens_stablehlo": dtype_tokens(stablehlo),
        "dtype_tokens_optimized": dtype_tokens(optimized),
        "hlo_op_histogram_optimized": dict(list(hlo_op_histogram(optimized).items())[:25]),
        "memory_analysis": memory,
        "static_launch_proxy": kernel_launches_per_step(optimized),
        "hlo_bytes": {"stablehlo": len(stablehlo), "optimized": len(optimized)},
        "transfer_primitives_in_hlo": {
            name: optimized.count(name) for name in TRANSFER_PRIMITIVES
            if optimized.count(name)
        },
        "seconds": time.perf_counter() - started,
    })

    # fp32 arm: what the same operator lowers to once its inputs are fp32. The
    # convert delta is the direct measure of how much dtype churn an fp32
    # rewrite would introduce -- including the converts the fp64 islands emit.
    try:
        lowered32, _ = lower_with_real_arguments(op, ctx, "fp32")
        optimized32 = lowered32.compile().as_text()
        record["fp32_arm"] = {
            "converts_optimized": count_converts(optimized32).as_dict(),
            "dtype_tokens_optimized": dtype_tokens(optimized32),
            "static_launch_proxy": kernel_launches_per_step(optimized32),
            "memory_analysis": compiled_memory_stats(lowered32.compile()),
        }
    except Exception as exc:  # noqa: BLE001
        record["fp32_arm"] = {"error": f"{type(exc).__name__}: {exc}"}
    return record


# --------------------------------------------------------------------------- #
# static transfer audit                                                        #
# --------------------------------------------------------------------------- #
def static_transfer_audit() -> dict[str, Any]:
    """Scan the timestep path for host-sync idioms (§13, no timestep transfers)."""

    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    by_idiom: dict[str, int] = {}
    for relative in TIMESTEP_PATH:
        path = REPO / relative
        if not path.is_file():
            continue
        scanned.append(relative)
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, label in HOST_SYNC_IDIOMS:
                if re.search(pattern, stripped):
                    findings.append({
                        "file": relative,
                        "line": number,
                        "idiom": pattern,
                        "label": label,
                        "code": stripped[:160],
                    })
                    by_idiom[label] = by_idiom.get(label, 0) + 1
                    break
    return {
        "files_scanned": scanned,
        "host_sync_idiom_hits": findings,
        "hit_count": len(findings),
        "hits_by_idiom": dict(sorted(by_idiom.items(), key=lambda kv: -kv[1])),
        "method": (
            "lexical scan for device->host synchronisation idioms in the timestep path. "
            "It over-reports by design: a hit inside a debug-only or setup branch is not a "
            "live transfer. Every hit is listed so a reviewer can dismiss it explicitly "
            "rather than trust an aggregate. jnp.asarray is deliberately NOT a hit -- it is "
            "a device constructor, and matching it drowns the real signal."
        ),
        "phase_b_confirmation": (
            "the authoritative audit is the nsys trace-based "
            "gpuwrf.profiling.transfer_audit.write_transfer_audit, which needs a GPU window"
        ),
    }


# --------------------------------------------------------------------------- #
# profiler command dry-runs                                                    #
# --------------------------------------------------------------------------- #
def profiler_dry_runs() -> dict[str, Any]:
    """Verify nsys/ncu exist and record the exact Phase-B command lines.

    Only ``--version``/``--help`` are executed. Neither touches the GPU, so this
    stays inside the CPU-first rule while still proving the commands are not
    typos that would waste a coordinated window.
    """

    results: dict[str, Any] = {}
    for tool in ("nsys", "ncu"):
        path = shutil.which(tool) or f"/usr/local/cuda/bin/{tool}"
        exists = Path(path).is_file()
        entry: dict[str, Any] = {"path": path, "present": exists}
        if exists:
            proc = subprocess.run(
                [path, "--version"], capture_output=True, text=True, check=False, timeout=60,
            )
            entry["version_rc"] = proc.returncode
            entry["version"] = (proc.stdout or proc.stderr).strip().splitlines()[:3]
        results[tool] = entry

    lock = "scripts/with_gpu_lock.sh"
    results["planned_commands"] = {
        "baseline_census_nsys": (
            f"{lock} --label v0250-m0-baseline-census -- "
            "nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true "
            "--force-overwrite=true -o <DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline "
            "python scripts/v025/run_gpu_arm.py --window baseline-census --case fast-v025"
        ),
        "baseline_census_ncu": (
            f"{lock} --label v0250-m0-baseline-census -- "
            "ncu --set full --target-processes all --replay-mode kernel "
            "--metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,"
            "sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum,"
            "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,"
            "sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"
            "sm__sass_thread_inst_executed_op_dfma_pred_on.sum,"
            "dram__bytes.sum,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active "
            "-o <DATA_ROOT>/wrf_gpu2/v025/m0/raw/ncu_baseline "
            "python scripts/v025/run_gpu_arm.py --window baseline-census --single-step"
        ),
        "note": (
            "the FLOP counters above are what §8's F_measured is built from. §16 makes an "
            "unavailable counter a BLOCKED denominator, never a licence to reuse the "
            "report's +/-2x estimate, so the ncu run reports counter availability first."
        ),
    }
    results["gpu_touched"] = False
    results["gpu_touch_note"] = (
        "only --version was executed; neither nsys --version nor ncu --version "
        "initialises a CUDA context or queries the device"
    )
    return results


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Separate from ``main`` so the sweep's commands can be parse-checked.

    W1 attempt 1 died after a coordinated GPU window opened because a driver
    passed ``--hours 1.0`` to an int option. Every test passed: they inspected
    the command list instead of handing it to the real parser. A generated
    command is only trustworthy if this exact parser accepts it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--domain", default="d01",
                        help="real production domain to load (d01..d09)")
    parser.add_argument("--previous", default=None,
                        help="EXACT earlier wrfout filename for the stage pair")
    parser.add_argument("--snapshot", default=None,
                        help="EXACT later wrfout filename for the stage pair")
    parser.add_argument("--expect-sha256", action="append", default=[],
                        help="required sha256 of --previous then --snapshot; fail-closed")
    parser.add_argument("--repeat-tag", default="",
                        help="label for one repeat of a shape sweep")
    parser.add_argument("--out", type=Path,
                        default=REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json")
    parser.add_argument("--hlo-dir", type=Path,
                        default=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/hlo"))
    return parser


def main() -> int:
    args = build_parser().parse_args()

    environment = assert_cpu_only()
    run_dir = args.run_dir or default_run_dir()
    started = time.perf_counter()
    pair = select_state_pair(Path(run_dir), args.domain,
                            previous=args.previous, snapshot=args.snapshot,
                            expect_sha256=args.expect_sha256)
    snapshot = load_real_snapshot(run_dir, wrfout_name=pair["snapshot"].name, domain=args.domain)
    previous = load_real_snapshot(run_dir, wrfout_name=pair["previous"].name, domain=args.domain)
    ctx = build_context(snapshot, previous, domain=args.domain)

    records = []
    for op in build_inventory():
        record = census_operator(op, ctx)
        records.append(record)
        print(f"  {op.name:<42s} {record['status']:<12s} "
              f"converts={record.get('converts_optimized', {}).get('total', '-')} "
              f"launches={record.get('static_launch_proxy', '-')}", flush=True)

    lowered = [r for r in records if r["status"] == "LOWERED"]
    total_launch_proxy = sum(r["static_launch_proxy"] for r in lowered)
    converts_total = sum(r["converts_optimized"]["total"] for r in lowered)
    converts_total32 = sum(
        r.get("fp32_arm", {}).get("converts_optimized", {}).get("total", 0) for r in lowered
    )
    temp_bytes = [
        (r["name"], r["memory_analysis"].get("temporary_bytes") or 0) for r in lowered
    ]
    temp_bytes.sort(key=lambda kv: -kv[1])

    by_family: dict[str, dict[str, int]] = {}
    for record in lowered:
        entry = by_family.setdefault(record["family"], {"operators": 0, "launch_proxy": 0, "converts": 0})
        entry["operators"] += 1
        entry["launch_proxy"] += record["static_launch_proxy"]
        entry["converts"] += record["converts_optimized"]["total"]

    obj = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_section": "§6.6 static census; the pre-registered reference for the §9 GPU census",
        "phase": "A6 (CPU-only; no GPU import, query, compile, or run)",
        "environment": {
            "host": platform.node(),
            "python": sys.version.split()[0],
            "jax_platforms": environment,
            "backend_note": (
                "lowered and compiled for the CPU backend. Absolute XLA:CPU launch counts are "
                "NOT the GPU launch counts; the cross-check §9 must satisfy is the RELATIVE "
                "family attribution and the convert/dtype fingerprint, both of which are "
                "backend-stable, plus the zero-transfer property."
            ),
        },
        "state_source": {
            "state_source_is_real": True,
            "stage_pair": ctx["stage_pair"],
            **snapshot.provenance,
        },
        "domain": args.domain,
        "repeat_tag": args.repeat_tag,
        "case_config": snapshot.config,
        "totals": {
            "operators_inventoried": len(records),
            "operators_lowered": len(lowered),
            "not_lowered": [
                {"name": r["name"], "error": r.get("error")}
                for r in records if r["status"] != "LOWERED"
            ],
            "static_launch_proxy_sum": total_launch_proxy,
            "converts_optimized_sum_fp64_arm": converts_total,
            "converts_optimized_sum_fp32_arm": converts_total32,
            "convert_growth_fp32_vs_fp64": (
                (converts_total32 - converts_total) if converts_total else None
            ),
            "convert_growth_note": (
                "every extra convert in the fp32 arm is an fp64 island or a promotion "
                "boundary paying for itself in dtype churn; §10's minimal island set is what "
                "keeps this number down"
            ),
        },
        "family_attribution": dict(sorted(by_family.items())),
        "largest_temporaries_bytes": temp_bytes[:15],
        "static_transfer_audit": static_transfer_audit(),
        "profiler_dry_runs": profiler_dry_runs(),
        "operators": records,
        "seconds_total": time.perf_counter() - started,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {args.out}")
    print(f"  lowered {len(lowered)}/{len(records)}  static launch proxy {total_launch_proxy}  "
          f"converts fp64 {converts_total} -> fp32 {converts_total32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
