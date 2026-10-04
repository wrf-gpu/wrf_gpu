#!/usr/bin/env python3
"""CPU stub standing in for the Step 1 GPU stage, so the dry run is a real run.

The manager's requirement is that the dry path *prove stage execution and all
artifact plumbing*, not print commands. That only works if the stub produces
artifacts in the **real formats**: a `nvtx_pushpop_trace` CSV with the columns
nsys actually emits, and an HLO dump directory with the file-per-pass layout
`--xla_dump_to` actually writes. Then the dry run exercises the same parsers,
the same module scoping, the same one-file HLO selection, and the same
fail-closed paths that a real window would.

Everything it writes is stamped `STUB` and lands under a `dryrun` directory, and
the driver records `dry_run: true` in the object. These numbers are synthetic and
must never be read as evidence about the compiler -- the dry run proves the
*path*, not the physics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE = "jit__run_forecast_operational_jit"
SMALL_MODULE = "jit_convert_element_type"

# nsys's real column set, in nsys's real order.
HEADER = ("Start (ns),End (ns),Duration (ns),DurChild (ns),DurNonChild (ns),Name,PID,TID,"
          "Lvl,NumChild,RangeId,ParentId,RangeStack,NameTree")

MS = 1_000_000

# Process startup before the first compile range: interpreter start, JAX import,
# backend init, input load. NON-ZERO on purpose. With a zero offset the
# launch-relative T_off and the old first-range-to-last-range span would be
# numerically identical, and the test for the clock fix would prove nothing.
STARTUP_MS = 12_000


def _row(start_ms, dur_ms, child_ms, name, rid, parent, lvl, nchild):
    start, dur, child = (start_ms + STARTUP_MS) * MS, dur_ms * MS, child_ms * MS
    return (f'{start},{start + dur},{dur},{child},{dur - child},"{name}",4242,4242,'
            f'{lvl},{nchild},{rid},{parent},:{rid},"{name}"')


def write_trace(path: Path) -> Path:
    """A nested trace: XlaCompile > XlaPassPipeline > XlaPass, plus a second module."""
    rows = [
        # dominant module: 100 s compile frame, children below
        _row(0, 100_000, 96_000, f"TSL:XlaCompile:#module={MODULE},program_id=1#", 1, "", 0, 2),
        _row(1_000, 60_000, 58_000,
             f"TSL:XlaPassPipeline:#name=optimization,module={MODULE}#", 2, 1, 1, 2),
        _row(2_000, 40_000, 0, f"TSL:XlaPass:#name=fusion,module={MODULE}#", 3, 2, 2, 0),
        _row(43_000, 18_000, 0, f"TSL:XlaPass:#name=call-inliner,module={MODULE}#", 4, 2, 2, 0),
        _row(62_000, 36_000, 0, f"TSL:XlaCompileGpuAsm:#module={MODULE}#", 5, 1, 1, 0),
        # a smaller module, so "dominant" is a real selection rather than the only option
        _row(120_000, 8_000, 6_000,
             f"TSL:XlaCompile:#module={SMALL_MODULE},program_id=2#", 6, "", 0, 1),
        _row(121_000, 6_000, 0, f"TSL:XlaPass:#name=fusion,module={SMALL_MODULE}#", 7, 6, 1, 0),
        # execution, which must stay OUT of the compile denominator
        _row(200_000, 50_000, 0, "TSL:Thunk:#name=fusion_1#", 8, "", 0, 0),
        # timestep ranges: without these the §7 loop-scoped transfer gate is
        # MISSING, which is the fail-closed branch a mutation test removes them
        # to exercise.
        _row(300_000, 1_000, 0, "timestep_0", 9, "", 0, 0),
        _row(301_000, 1_200, 0, "timestep_1", 10, "", 0, 0),
        _row(302_200, 1_100, 0, "timestep_2", 11, "", 0, 0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Generating NVTX Push/Pop Range Trace [STUB]...\n\n"
                    + HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def write_hlo_dump(dump_dir: Path) -> Path:
    """The real `--xla_dump_to` layout: many pass snapshots, one final module.

    The decoy files matter. `before_optimizations` describes a different program,
    and the per-pass snapshots repeat the same computations -- selecting any of
    them, or globbing them all, is exactly the double-counting the driver has to
    avoid. A stub with only the right file would prove nothing.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)

    def hlo(ops: list[tuple[str, str]]) -> str:
        lines = ["HloModule STUB", "", "ENTRY %main {"]
        for index, (operator, kind) in enumerate(ops):
            lines.append(f'  %o{index} = f32[8]{{0}} {kind}(), '
                         f'metadata={{op_name="jit(main)/{operator}/{kind}" source_line={index}}}')
        lines.append("}")
        return "\n".join(lines) + "\n"

    radiation_heavy = ([("rrtmg_lw", "add")] * 6 + [("rrtmg_sw", "multiply")] * 6
                       + [("advect_u_flux", "add")] * 5
                       + [("thompson_microphysics", "subtract")] * 3
                       + [("mynn_pbl", "divide")] * 3)

    (dump_dir / f"module_0001.{MODULE}.before_optimizations.txt").write_text(
        hlo([("advect_u_flux", "add")] * 20))
    for index, pass_name in enumerate(("fusion", "call-inliner", "layout-assignment")):
        (dump_dir / f"module_0001.{MODULE}.after_{pass_name}.txt").write_text(hlo(radiation_heavy))
    (dump_dir / f"module_0002.{SMALL_MODULE}.after_optimizations.txt").write_text(
        hlo([("advect_u_flux", "add")]))

    final = dump_dir / f"module_0001.{MODULE}.after_optimizations.txt"
    final.write_text(hlo(radiation_heavy))
    return final


PROJ_CSV = """Time (%),Total Time (ns),Instances,Range
54.5,600000000,120,TSL:Thunk:#name=rrtmg_lw#
18.2,200000000,120,TSL:Thunk:#name=rrtmg_sw#
13.6,150000000,240,TSL:Thunk:#name=thompson_microphysics#
 9.1,100000000,240,TSL:Thunk:#name=advect_u_flux#
 4.6,50000000,240,TSL:Thunk:#name=mynn_pbl#
"""

MEM_CSV = """Time (%),Total Time (ns),Count,Operation
92.0,46000000,900,[CUDA memcpy Device-to-Device]
 8.0,4000000,12,[CUDA memset]
"""

MEM_USAGE_CSV = """Bytes,Count,Operation
3221225472,1,[CUDA memory allocation peak]
1073741824,1,[CUDA memory allocation]
"""


KERN_CSV = """Time (%),Total Time (ns),Instances,Avg (ns),Name
54.5,600000000,120,5000000.0,rrtmg_lw_fused_kernel
18.2,200000000,120,1666666.0,rrtmg_sw_fused_kernel
13.6,150000000,240,625000.0,thompson_microphysics_kernel
 9.1,100000000,240,416666.0,advect_u_flux_kernel
 4.6,50000000,240,208333.0,mynn_pbl_kernel
"""

MEM_TIME_CSV = """Time (%),Total Time (ns),Count,Operation
92.0,46000000,900,[CUDA memcpy Device-to-Device]
 8.0,4000000,12,[CUDA memset]
"""

CUDA_GPU_TRACE_CSV = """Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
5000000000,1000000,1,,,,,,,,,,1.000,1000.0,Host,Device,STUB GPU,1,,7,[CUDA memcpy HtoD]
314250000000,1000000,2,,,,,,,,,,2.000,2000.0,Device,Device,STUB GPU,1,,7,[CUDA memcpy Device-to-Device]
"""

ACTIVE_SCHEMES = [
    "mp_physics=8", "ra_lw_physics=4", "ra_sw_physics=4",
    "sf_sfclay_physics=5", "sf_surface_physics=4", "bl_pbl_physics=5",
    "cu_physics=1", "gwd_opt=1",
]
CADENCE_EVENTS = ["ordinary", "radiation", "cumulus", "pbl", "history"]


def write_canned_exports(export_dir: Path) -> Path:
    """What the stub `nsys stats` runner returns, per report name.

    These stand in for the nsys PROCESS only. The exporter still builds the
    commands, writes the files, hashes them and applies the empty-report gate.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "cuda_gpu_kern_sum.csv").write_text(KERN_CSV)
    (export_dir / "cuda_gpu_trace.csv").write_text(CUDA_GPU_TRACE_CSV)
    (export_dir / "cuda_gpu_mem_time_sum.csv").write_text(MEM_TIME_CSV)
    (export_dir / "nvtx_gpu_proj_sum.csv").write_text(PROJ_CSV)
    # The pushpop trace is exported through the same path, so it is canned here
    # too. Generated directly rather than copied, so `write_canned_exports` does
    # not depend on `write_trace` having run first.
    write_trace(export_dir / "nvtx_pushpop_trace.csv")
    return export_dir


def write_census_inputs(out_root: Path) -> None:
    """The baseline-census CSVs, so the dry run exercises that path too.

    Without these the census reports BLOCKED for want of input, which proves the
    fail-closed branch but never the working one. Both need covering.
    """
    (out_root / "nvtx_gpu_proj_sum.csv").write_text(PROJ_CSV)
    (out_root / "cuda_gpu_mem_time_sum.csv").write_text(MEM_CSV)
    (out_root / "cuda_gpu_mem_usage.csv").write_text(MEM_USAGE_CSV)


def write_scope_and_representativeness(out_root: Path, run_id: str) -> None:
    """Dry representativeness fixtures, explicitly marked non-scientific.

    The production analyser derives integration scope from the current
    capture's NVTX rows. No hand-authored interval fixture is accepted.
    """
    family_shares = {
        "physics.radiation": 0.7272727272727273,
        "physics.microphysics": 0.13636363636363635,
        "dycore.advection": 0.09090909090909091,
        "physics.pbl": 0.045454545454545456,
    }
    (out_root / "production_reference.json").write_text(json.dumps({
        "schema": "wrf_gpu2.v025.m0.production_reference.v1",
        "run_id": "dry-reference-two-domain",
        "case_role": "matched_short_two_domain_production",
        "source_rep": {"path": "DRY-STUB", "sha256": "a" * 64},
        "workload_identity_sha256": "c" * 64,
        "non_nesting_families": family_shares,
        "rank_families": list(family_shares),
        "active_d01_schemes": ACTIVE_SCHEMES,
        "cadence_events": CADENCE_EVENTS,
        "dry_run_stub": True,
    }, indent=2) + "\n")
    (out_root / "candidate_coverage.json").write_text(json.dumps({
        "schema": "wrf_gpu2.v025.m0.candidate_coverage.v1",
        "run_id": run_id,
        "workload_identity_sha256": "c" * 64,
        "executed_schemes": ACTIVE_SCHEMES,
        "executed_cadence_events": CADENCE_EVENTS,
        "dry_run_stub": True,
    }, indent=2) + "\n")


def stub_stage_command(out_root: Path, *, run_id: str) -> list[str]:
    """The command the driver runs INSTEAD of nsys+gpuwrf during a dry run."""
    return [sys.executable, str(Path(__file__).resolve()), "--out-root", str(out_root),
            "--run-id", run_id]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    write_trace(args.out_root / "step1_pushpop.csv")
    write_hlo_dump(args.out_root / "dump")
    write_census_inputs(args.out_root)
    write_canned_exports(args.out_root / "canned_exports")
    write_scope_and_representativeness(args.out_root, args.run_id)
    # A stand-in for the .nsys-rep, so the "was a report produced" check is real.
    (args.out_root / "step1_autotune_off.nsys-rep").write_bytes(b"STUB nsys report\n")

    # Record the environment THIS CHILD actually received. The previous driver
    # only printed its environment, so "the flags are applied" was never checked
    # against anything. Writing it from inside the child makes the assertion an
    # end-to-end observation rather than a mock of the launcher.
    import os
    (args.out_root / "stub_env.json").write_text(json.dumps({
        key: os.environ.get(key)
        for key in (
            "XLA_FLAGS",
            "GPUWRF_JAX_CACHE",
            "GPUWRF_JAX_CACHE_DIR",
            "GPUWRF_M0_EVIDENCE",
            "GPUWRF_M0_EVIDENCE_PATH",
            "GPUWRF_M0_RUN_ID",
            "GPUWRF_M0_SOURCE_SHA256",
            "GPUWRF_M0_CONFIG_SHA256",
            "GPUWRF_M0_INPUT_MANIFEST_SHA256",
            "GPUWRF_M0_DEVICE_UUID",
        )
    }, indent=2) + "\n")

    print(f"STUB wrote trace + HLO dump under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
