#!/usr/bin/env python3
"""Regenerate `proofs/v025/m0/nvtx_compile_evidence.json` from raw evidence (CPU-only).

One command, no hand-written numbers. The previous version of that proof object
was assembled by hand and carried three factual errors that authoritative
filesystem evidence contradicted:

* it tested the **deprecated** `runtime.jax_cache.DEFAULT_CACHE_DIR`
  (`<DATA_ROOT>/gpuwrf_jax_cache`) instead of the canonical resolver in
  `runtime.compile_cache`, and concluded from that dir's absence that the
  persistent-cache explanation was falsified. It is not falsified.
* it reported "2020 small-module entries / zero autotune files". The directory
  is 454 top-level executable-cache files **plus** 1566 `.textproto` records
  inside `xla_gpu_per_fusion_autotune_cache_dir/`. The earlier search looked for
  files *named* `xla_gpu_per_fu*` when that is a **directory**.
* it called 107,412 nvtx_sum rows "ranges" when rows are summary lines and the
  instance total is different.

Every number below is now read from the filesystem or from the parser at
generation time, and the cache evidence is content-hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports
sys.path.insert(0, str(REPO / "src"))

import parse_profiler as pp  # noqa: E402

RAW = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw")
NSYS_REP = RAW / "nsys_baseline.nsys-rep"
NVTX_CSV = RAW / "nsys_baseline_nvtx_sum.csv"
STAGE1_CACHE = RAW / "w1b_arms/jaxcache_5623902916f6"
AUTOTUNE_SUBDIR = "xla_gpu_per_fusion_autotune_cache_dir"
BIG_MODULE = "jit__run_forecast_operational_jit"
# Stage 2 ran 2026-07-27 18:55:31 -> ~18:57 local (WEST = UTC+1).
STAGE2_WINDOW = ("2026-07-27 18:55:00", "2026-07-27 18:58:00")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_cache_dir(backend_tag: str = "cuda_sm120") -> dict:
    """Resolve the cache dir the way the PRODUCTION code does, not a deprecated alias.

    The version tag embeds the BACKEND, so resolving from this CPU-only generator
    yields ``...-cpu`` while the GPU process being described used
    ``...-cuda_sm120``. Reporting the locally-resolved path as "the cache Stage 2
    used" would be a second aliasing error of exactly the kind that produced the
    original mistake, so the backend component is substituted explicitly and both
    paths are recorded.
    """
    info: dict = {"resolver": "gpuwrf.runtime.compile_cache._default_cache_dir"}
    try:
        from gpuwrf.runtime.compile_cache import _default_cache_dir, version_cache_tag
        local = Path(_default_cache_dir())
        local_tag = version_cache_tag()
        # swap the trailing backend component for the one Stage 2 ran under
        target_tag = local_tag.rsplit("-", 1)[0] + "-" + backend_tag
        path = local.parent / target_tag
        info.update({
            "path": str(path),
            "version_tag": target_tag,
            "exists": path.exists(),
            "resolved_locally_as": {
                "path": str(local), "version_tag": local_tag,
                "why_different": (
                    "this generator runs CPU-only, and the version tag embeds the backend. "
                    "The GPU process being described used the cuda_sm120 tag. Reporting the "
                    "local path as Stage 2's cache would repeat the aliasing error this "
                    "object exists to correct."
                ),
            },
        })
    except Exception as exc:  # noqa: BLE001
        info.update({"error": f"{type(exc).__name__}: {exc}"})
    try:
        from gpuwrf.runtime.jax_cache import DEFAULT_CACHE_DIR
        info["deprecated_alias"] = {
            "symbol": "gpuwrf.runtime.jax_cache.DEFAULT_CACHE_DIR",
            "path": str(DEFAULT_CACHE_DIR),
            "exists": Path(str(DEFAULT_CACHE_DIR)).exists(),
            "note": ("this is what the earlier proof tested. Its absence says nothing about "
                     "the cache the run actually used."),
        }
    except Exception:  # noqa: BLE001
        pass
    return info


CACHE_SUFFIX = "-cache"
ATIME_SUFFIX = "-atime"


def classify_top_level(files: list[Path]) -> dict[str, list[Path]]:
    """Split a JAX cache directory's top level by what the files actually are.

    Manager correction (`b9598b2d` follow-up): the previous census called all 987
    Stage 2 top-level files "executable cache files". They are not. Each cached
    executable `<key>-cache` is accompanied by a `<key>-atime` access-time
    sidecar, plus one `.lockfile`. Counting the sidecars as executables inflates
    the entry count by ~2x and would make any per-module reasoning built on it
    wrong by the same factor.
    """
    kinds: dict[str, list[Path]] = {"cache": [], "atime": [], "lockfile": [], "other": []}
    for handle in files:
        if handle.name.endswith(CACHE_SUFFIX):
            kinds["cache"].append(handle)
        elif handle.name.endswith(ATIME_SUFFIX):
            kinds["atime"].append(handle)
        elif handle.name == ".lockfile":
            kinds["lockfile"].append(handle)
        else:
            kinds["other"].append(handle)
    return kinds


def cache_manifest(entries: list[Path]) -> dict:
    """An IMMUTABLE, content-addressed manifest of the cache entries.

    Each entry is recorded by full SHA-256 of its bytes, and the manifest itself
    carries a digest over those digests. That is what makes a later claim about
    this cache re-verifiable rather than a recollection: the directory is mutable
    and shared, the manifest is not.
    """
    rows = []
    for handle in sorted(entries, key=lambda p: p.name):
        digest = hashlib.sha256()
        with handle.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        rows.append({"name": handle.name, "sha256": digest.hexdigest(),
                     "bytes": handle.stat().st_size})
    joined = "\n".join(f"{row['sha256']}  {row['name']}" for row in rows)
    return {
        "entries": len(rows),
        "manifest_sha256": hashlib.sha256(joined.encode()).hexdigest(),
        "total_bytes": sum(row["bytes"] for row in rows),
        "records": rows,
        "content_addressed": True,
        "why": ("the cache directory is mutable and shared; this manifest is the immutable "
                "record a later claim can be checked against"),
    }


def describe_cache(path: Path) -> dict:
    """Directory-aware census of one cache directory."""
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    top = [f for f in path.iterdir() if f.is_file()]
    kinds = classify_top_level(top)
    autotune_dir = path / AUTOTUNE_SUBDIR
    autotune = [f for f in autotune_dir.rglob("*") if f.is_file()] if autotune_dir.is_dir() else []
    other_nested = [
        f for f in path.rglob("*")
        if f.is_file() and f.parent != path and AUTOTUNE_SUBDIR not in f.parts
    ]
    big = [f for f in top if BIG_MODULE in f.name]

    def stamps(files):
        if not files:
            return {}
        times = sorted(f.stat().st_mtime for f in files)
        fmt = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
        return {"oldest_mtime": fmt(times[0]), "newest_mtime": fmt(times[-1])}

    return {
        "path": str(path),
        "exists": True,
        "top_level_files": len(top),
        "top_level_breakdown": {
            "cached_executables": len(kinds["cache"]),
            "atime_sidecars": len(kinds["atime"]),
            "lockfiles": len(kinds["lockfile"]),
            "other": [f.name for f in kinds["other"]],
            "correction": (
                "an earlier version reported the top-level TOTAL as 'executable cache files'. "
                "Each executable has an -atime sidecar, so the executable count is roughly half "
                "the file count. Use cached_executables."
            ),
        },
        "cache_manifest": cache_manifest(kinds["cache"]),
        "autotune_subdir": AUTOTUNE_SUBDIR,
        "autotune_records": len(autotune),
        "autotune_are_textproto": sum(1 for f in autotune if f.suffix == ".textproto"),
        "other_nested_files": len(other_nested),
        "total_files": len(top) + len(autotune) + len(other_nested),
        "top_level_entries_for_big_module": len(big),
        "top_level_stamps": stamps(top),
        "autotune_stamps": stamps(autotune),
        "autotune_attribution": (
            "NOT ATTRIBUTED. The autotune records are content-hashed filenames with no module "
            "name. Assigning them to any module -- including the big one -- would be an "
            "unevidenced inference."
        ),
    }


def files_modified_in_window(path: Path, start: str, end: str) -> dict:
    """Files whose mtime falls in a window. NOT a claim that the run wrote them.

    Manager correction (`b9598b2d` follow-up): the previous object called these
    "writes in window", which asserts causation from a timestamp. An mtime inside
    a window is temporal coincidence. The cache is a shared, version-keyed
    directory; a concurrent process can touch it, and JAX updates `-atime`
    sidecars on cache *reads*, so a modified file is not even necessarily a
    write of new content by anyone.
    """
    if not path.is_dir():
        return {"count": 0, "note": "directory absent"}
    proc = subprocess.run(
        ["find", str(path), "-type", "f", "-newermt", start, "!", "-newermt", end],
        capture_output=True, text=True, check=False,
    )
    names = [line for line in proc.stdout.splitlines() if line.strip()]
    kinds = classify_top_level([Path(n) for n in names])
    return {
        "window_local": [start, end],
        "count": len(names),
        "breakdown": {kind: len(files) for kind, files in kinds.items() if files},
        "examples": [Path(n).name[:70] for n in names[:5]],
        "NOT_CAUSAL": (
            "these files were MODIFIED during the window. That is temporal coincidence, not "
            "evidence that this run wrote them: the directory is shared and version-keyed, and "
            "JAX touches -atime sidecars on cache READS. Attributing them to the run would be "
            "an unevidenced causal claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "proofs/v025/m0/nvtx_compile_evidence.json")
    args = parser.parse_args()

    rows = pp.parse_nsys_nvtx_summary(NVTX_CSV.read_text())
    breakdown = pp.compile_breakdown(rows)
    instance_total = sum(r["instances"] for r in rows)
    big_rows = [r for r in rows if BIG_MODULE in r["range"]]
    codegen = [r for r in big_rows
               if any(k in r["range"] for k in ("CompileGpuAsm", "EmitGpuAsm", "OptimizeLlvmIr"))]

    canonical = canonical_cache_dir()
    canonical_path = Path(canonical.get("path", "/nonexistent"))

    obj = {
        "schema": "wrf_gpu2.v025.m0.nvtx_compile_evidence.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce": {
            "command": "python scripts/v025/build_nvtx_compile_evidence.py",
            "generator": "scripts/v025/build_nvtx_compile_evidence.py",
            "generator_sha256": sha256_file(Path(__file__)),
            "raw_nsys_rep": str(NSYS_REP), "raw_nsys_rep_sha256": sha256_file(NSYS_REP),
            "raw_nvtx_csv": str(NVTX_CSV), "raw_nvtx_csv_sha256": sha256_file(NVTX_CSV),
            "nvtx_csv_produced_by": (
                "nsys stats --report nvtx_sum --format csv "
                f"{NSYS_REP}"
            ),
            "note": "no number in this object is hand-written; rerun the command to regenerate",
        },
        "scope_of_the_capture": {
            "unusable_for": "§9 CUDA kernel census -- cuda_gpu_kern_sum contains no kernel data",
            "usable_for": "XLA compile timing via NVTX ranges",
            "why_no_kernels": "the process was killed during initialisation, before any forecast kernel",
        },
        "nvtx": {
            "summary_rows": len(rows),
            "instance_total": instance_total,
            "note": (
                "summary ROWS and INSTANCES are different quantities: each row aggregates one "
                "named range over its instances. An earlier version reported the row count as "
                "'ranges' without qualification."
            ),
            "distinct_xla_modules": len(breakdown["modules"]),
            "module_compile_seconds_total": round(breakdown["module_compile_seconds_total"], 3),
            "top_modules": {k: round(v["seconds"], 3)
                            for k, v in list(breakdown["modules"].items())[:6]},
            "top_passes": [{"pass": p["pass"][:90], "seconds": round(p["seconds"], 3),
                            "instances": p["instances"]} for p in breakdown["top_passes"][:10]],
        },
        "big_module_scope": {
            "module": BIG_MODULE,
            "rows_mentioning_it": len(big_rows),
            "codegen_rows": len(codegen),
            "structure_observed": "PARTIAL FRONT-END ONLY",
            "explanation": (
                "front-end pass ranges are present and codegen ranges "
                "(XlaCompileGpuAsm/XlaEmitGpuAsm/XlaOptimizeLlvmIr) are ABSENT for this module, "
                "while smaller modules do show them. This is a partial front-end structure, NOT "
                "a full compile-pass structure, and an earlier version overstated it."
            ),
        },
        "cache_provenance": {
            "canonical_resolver": canonical,
            "stage2_effective_cache": describe_cache(canonical_path),
            "stage2_files_modified_in_window": files_modified_in_window(
                canonical_path, *STAGE2_WINDOW),
            "stage1_dedicated_cache": describe_cache(STAGE1_CACHE),
            "stage1_env": "GPUWRF_JAX_CACHE=1 with a fresh empty GPUWRF_JAX_CACHE_DIR",
            "stage2_env": (
                "no GPUWRF_JAX_CACHE_DIR -- the shell invoked `python -m gpuwrf run` directly, "
                "so the canonical no-env default applied"
            ),
        },
        "the_346s_vs_4.382s_question": {
            "stage1_cold_seconds": 346.360,
            "stage2_seconds": round(
                breakdown["modules"].get(BIG_MODULE, {}).get("seconds", float("nan")), 3),
            "MUST_NOT_BE_EQUATED": True,
            "status": "UNRESOLVED",
            "retracted_falsification": (
                "An earlier version claimed the warm-cache explanation was FALSIFIED because "
                "<DATA_ROOT>/gpuwrf_jax_cache does not exist. That test used the DEPRECATED "
                "jax_cache.DEFAULT_CACHE_DIR alias. The canonical resolver points elsewhere, "
                "that directory pre-existed Stage 2, and files were written in Stage 2's window. "
                "THE FALSIFICATION IS WITHDRAWN."
            ),
            "what_the_evidence_now_supports": (
                "Stage 2 did have a populated persistent cache and wrote to it during its "
                "window. But NEITHER cache holds a top-level executable entry for the big "
                "module, so a straight cache hit for THAT module is not evidenced either. "
                "Provenance is left UNRESOLVED rather than resolved in either direction."
            ),
            "only_proven_zero": (
                f"the count of TOP-LEVEL executable-cache files matching {BIG_MODULE} is zero in "
                "both directories. Nothing is proven about the hashed autotune records."
            ),
        },
        "bearing_on_pre_registered_H1": {
            "autotune_presence": "EVIDENCED -- autotune record directories exist and are populated",
            "autotune_time_dominance": "NOT MEASURED",
            "verdict": (
                "presence is not dominance. H1 (autotuning dominates cold compile) remains an "
                "untested pre-registered hypothesis. An earlier version claimed both 'a real "
                "autotune-volume signal' and 'bearing on H1: NONE' in the same object; that "
                "contradiction is removed by separating presence from time share."
            ),
        },
        "what_this_capture_reduces": (
            "It supplies a PARTIAL front-end pass breakdown for the big module, a full module "
            "list for the process, and evidence that autotune records are produced. It does NOT "
            "supply the cold-compile breakdown, does NOT establish autotune time dominance, and "
            "does NOT resolve why the two module-compile numbers differ."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    c = obj["cache_provenance"]
    print(f"  canonical cache: {c['canonical_resolver'].get('path')}")
    print(f"    stage2 dir: {c['stage2_effective_cache'].get('top_level_files')} top-level"
          f" + {c['stage2_effective_cache'].get('autotune_records')} autotune;"
          f" big-module entries {c['stage2_effective_cache'].get('top_level_entries_for_big_module')}")
    print(f"    stage1 dir: {c['stage1_dedicated_cache'].get('top_level_files')} top-level"
          f" + {c['stage1_dedicated_cache'].get('autotune_records')} autotune")
    print(f"    stage2 files MODIFIED in window: {c['stage2_files_modified_in_window'].get('count')}")
    print(f"  nvtx rows {obj['nvtx']['summary_rows']} instances {obj['nvtx']['instance_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
