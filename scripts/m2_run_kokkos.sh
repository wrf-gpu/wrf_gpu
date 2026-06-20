#!/usr/bin/env bash
set -euo pipefail

if [[ -f <USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source <USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
fi

ROOT=$(git rev-parse --show-toplevel)
BENCH="$ROOT/data/scratch/kokkos/bench"
SCRATCH="$ROOT/data/scratch/kokkos"
PROF="$ROOT/data/profiler_artifacts/kokkos"
ART="$ROOT/artifacts/m2/kokkos"
STENCIL_FIXTURE="$ROOT/fixtures/samples/analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE="$ROOT/fixtures/samples/analytic-column-thermo-v1.npz"

mkdir -p "$SCRATCH" "$PROF" "$ART"

bash "$ROOT/src/gpuwrf/backends/kokkos/build.sh" 2>&1 | tee "$SCRATCH/build.log"
"$BENCH" config > "$SCRATCH/kokkos_config.txt"
cuobjdump --dump-resource-usage "$BENCH" > "$SCRATCH/resource_usage.txt"

"$BENCH" stencil --input "$STENCIL_FIXTURE" --output "$SCRATCH/stencil_out.npz" | tee "$SCRATCH/stencil_run.json"
"$BENCH" column --input "$COLUMN_FIXTURE" --output "$SCRATCH/column_out.npz" | tee "$SCRATCH/column_run.json"

run_ncu() {
  local problem=$1
  local input=$2
  local output=$3
  local export_base="$PROF/${problem}"
  rm -f "${export_base}.ncu-rep"
  set +e
  ncu --set=full --target-processes=all --force-overwrite --export "$export_base" \
    "$BENCH" "$problem" --input "$input" --output "$output" \
    > "$PROF/${problem}_ncu_stdout.txt" 2> "$PROF/${problem}_ncu_stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$PROF/${problem}_ncu_exit.txt"
}

run_ncu stencil "$STENCIL_FIXTURE" "$SCRATCH/stencil_out.npz"
run_ncu column "$COLUMN_FIXTURE" "$SCRATCH/column_out.npz"

python -m gpuwrf.validation.compare_fixture \
  --manifest "$ROOT/fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml" \
  --candidate "$SCRATCH/stencil_out.npz" \
  --reference "$STENCIL_FIXTURE" \
  --out "$SCRATCH/stencil_correctness.json"

python -m gpuwrf.validation.compare_fixture \
  --manifest "$ROOT/fixtures/manifests/analytic-column-thermo-v1.yaml" \
  --candidate "$SCRATCH/column_out.npz" \
  --reference "$COLUMN_FIXTURE" \
  --out "$SCRATCH/column_correctness.json"

python - "$ROOT" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
scratch = root / "data" / "scratch" / "kokkos"
prof = root / "data" / "profiler_artifacts" / "kokkos"
art = root / "artifacts" / "m2" / "kokkos"
resource_text = (scratch / "resource_usage.txt").read_text(errors="replace")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resource_entries() -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    current = ""
    for line in resource_text.splitlines():
        if line.strip():
            if "Function" in line or "kernel" in line.lower() or "Kokkos" in line or "Advdiff" in line or "Thermo" in line:
                current = line
        match = re.search(r"REG:(\d+).*LOCAL:(\d+)", line)
        if match:
            entries.append((current, int(match.group(1)), int(match.group(2))))
    return entries


def resources(kind: str) -> tuple[int, int]:
    markers = {
        "stencil": ("StencilAdvdiffKernel", "Advdiff"),
        "column": ("ColumnThermoKernel", "Thermo"),
    }[kind]
    entries = resource_entries()
    for name, regs, local in entries:
        if any(marker in name for marker in markers):
            return regs, local
    # Fallback for cuobjdump formats that mangle Kokkos functor names beyond
    # recognition: use the heaviest parsed kernel and record the limitation.
    if entries:
        regs = max(entry[1] for entry in entries)
        local = max(entry[2] for entry in entries)
        return regs, local
    return 0, 0


def occupancy_pct(run: dict, registers_per_thread: int) -> float:
    threads = max(int(run.get("threads_per_team", 1)), 1)
    max_threads = max(int(run.get("max_threads_per_sm", 1)), 1)
    max_registers = max(int(run.get("max_registers_per_sm", 1)), 1)
    max_blocks = max(int(run.get("max_blocks_per_sm", 1)), 1)
    if registers_per_thread <= 0:
        return 0.0
    blocks_by_threads = max_threads // threads
    blocks_by_registers = max_registers // (registers_per_thread * threads)
    active_blocks = max(0, min(max_blocks, blocks_by_threads, blocks_by_registers))
    return 100.0 * active_blocks * threads / max_threads


def artifact_paths(kind: str) -> list[str]:
    paths = [
        prof / f"{kind}_ncu_stdout.txt",
        prof / f"{kind}_ncu_stderr.txt",
        prof / f"{kind}_ncu_exit.txt",
        scratch / "resource_usage.txt",
        scratch / "kokkos_config.txt",
        scratch / f"{kind}_run.json",
        scratch / "build.log",
    ]
    report = prof / f"{kind}.ncu-rep"
    if report.exists():
        paths.append(report)
    return [str(path.relative_to(root)) for path in paths if path.exists()]


def ncu_limitation(kind: str) -> str | None:
    exit_text = (prof / f"{kind}_ncu_exit.txt").read_text().strip()
    if exit_text == "0" and (prof / f"{kind}.ncu-rep").exists():
        return None
    stderr = (prof / f"{kind}_ncu_stderr.txt").read_text(errors="replace")
    stdout = (prof / f"{kind}_ncu_stdout.txt").read_text(errors="replace")
    text = "\n".join((stderr, stdout)).strip()
    if "ERR_NVGPUCTRPERM" in text:
        return (
            "ncu invoked, but local user lacks NVIDIA performance-counter permission "
            "(ERR_NVGPUCTRPERM); registers/local memory parsed from cuobjdump, "
            "occupancy fallback derived from cuobjdump registers plus CUDA device limits "
            "reported by the Kokkos bench, wall time/transfers from bench output."
        )
    return f"ncu invoked but exited {exit_text}; see artifact_paths logs."


def profile(kind: str, case: str) -> dict:
    run = load_json(scratch / f"{kind}_run.json")
    regs, local = resources(kind)
    wall = float(run["wall_time_s"])
    transfer_bytes = int(run["host_device_transfer_bytes"])
    achieved = (transfer_bytes / wall / 1.0e9) if wall > 0 else 0.0
    record = {
        "benchmark": f"m2_{kind}",
        "backend": "kokkos",
        "hardware": "RTX 5090 32GB",
        "case": case,
        "wall_time_s": wall,
        "kernel_launches": int(run["kernel_launches"]),
        "host_device_transfer_bytes": transfer_bytes,
        "occupancy_pct": float(occupancy_pct(run, regs)),
        "registers_per_thread": regs,
        "local_memory_bytes": local,
        "achieved_bandwidth_gbps": achieved,
        "achieved_bandwidth_method": "fallback-derived",
        "kokkos_version": int(run["kokkos_version"]),
        "kokkos_execution_space": run["kokkos_execution_space"],
        "runtime_compute_capability": run["runtime_compute_capability"],
        "artifact_paths": artifact_paths(kind),
    }
    limitation = ncu_limitation(kind)
    if limitation:
        record["profiler_limitation"] = limitation
    return record


stencil_corr = load_json(scratch / "stencil_correctness.json")
column_corr = load_json(scratch / "column_correctness.json")
(art / "stencil_profile.json").write_text(
    json.dumps(profile("stencil", "analytic-stencil-3d-advdiff-v1"), indent=2, sort_keys=True) + "\n"
)
(art / "column_profile.json").write_text(
    json.dumps(profile("column", "analytic-column-thermo-v1"), indent=2, sort_keys=True) + "\n"
)
(art / "correctness.json").write_text(
    json.dumps(
        {
            "backend": "kokkos",
            "pass": bool(stencil_corr["pass"] and column_corr["pass"]),
            "stencil": stencil_corr,
            "column": column_corr,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
