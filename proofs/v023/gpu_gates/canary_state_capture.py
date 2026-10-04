"""Capture the short 3-domain canary state for digest/diff debugging.

This is proof infrastructure only.  Run it from a target worktree with
``PYTHONPATH=<target>/src`` so the imported model code comes from that commit.
It mirrors ``proofs/v021/canary_gate/perstep_timing_driver.py`` fixed-shape
mode, but can also dump the final state leaves to an NPZ for cross-commit
field-diff analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


def _truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _gpu_mem_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def _safe_path_part(part: Any) -> str:
    raw = getattr(part, "name", getattr(part, "key", getattr(part, "idx", part)))
    return str(raw).replace("/", "_").replace(" ", "")


def _state_leaf_items(states: Any):
    import jax
    import numpy as np

    for domain, state in sorted(states.items()):
        path_leaves, _treedef = jax.tree_util.tree_flatten_with_path(state)
        for idx, (path, leaf) in enumerate(path_leaves):
            arr = np.asarray(leaf)
            if not hasattr(arr, "dtype"):
                continue
            suffix = ".".join(_safe_path_part(part) for part in path) or f"leaf{idx}"
            key = f"{domain}/{idx:04d}/{suffix}"
            yield key, np.ascontiguousarray(arr)


def _write_state_digest(path: Path, *, tag: str, states: Any) -> dict[str, object]:
    h = hashlib.sha256()
    leaves = 0
    total_bytes = 0
    for key, arr in _state_leaf_items(states):
        leaves += 1
        total_bytes += int(arr.nbytes)
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps(tuple(int(v) for v in arr.shape)).encode("utf-8"))
        h.update(arr.tobytes(order="C"))
    payload = {
        "tag": tag,
        "sha256": h.hexdigest(),
        "leaves": leaves,
        "bytes": total_bytes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _write_state_npz(path: Path, meta_path: Path, *, tag: str, states: Any) -> None:
    import numpy as np

    arrays: dict[str, Any] = {}
    leaves: list[dict[str, object]] = []
    for idx, (key, arr) in enumerate(_state_leaf_items(states)):
        name = f"arr_{idx:04d}"
        arrays[name] = arr
        leaves.append(
            {
                "array": name,
                "key": key,
                "dtype": str(arr.dtype),
                "shape": [int(v) for v in arr.shape],
                "bytes": int(arr.nbytes),
            }
        )
    np.savez(path, **arrays)
    meta_path.write_text(
        json.dumps({"tag": tag, "npz": str(path), "leaves": leaves}, indent=2, sort_keys=True)
        + "\n"
    )


def main() -> None:
    import jax
    import numpy as np

    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        _load_domains,
        _nested_sync_mode_from_env,
        _output_cadence_steps_by_domain,
        domain_names_for,
    )
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.runtime.domain_tree import (
        DomainTree,
        maybe_prewarm_defused_nest,
        run_operational_domain_tree,
    )

    input_dir = Path(os.environ.get("CANARY_INPUT_DIR", "<DATA_ROOT>/wrf_downscale/runs/20250121/cpu"))
    tag = os.environ.get("CANARY_TAG", "canary_state_capture")
    maxdom = int(os.environ.get("CANARY_MAXDOM", os.environ.get("MAXDOM", "3")))
    fixed_root_steps = int(os.environ.get("CANARY_FIXED_ROOT_STEPS", "2"))
    fixed_repeats = int(os.environ.get("CANARY_FIXED_REPEATS", "3"))
    if fixed_root_steps <= 0:
        raise ValueError("CANARY_FIXED_ROOT_STEPS must be positive")
    if fixed_repeats < 2:
        raise ValueError("CANARY_FIXED_REPEATS must be at least 2")

    artifact_dir = Path(os.environ.get("CANARY_ARTIFACT_DIR", "proofs/v023/gpu_gates/canary_state"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir = artifact_dir / f"ts_output_{tag}"
    proof_dir = artifact_dir / f"ts_proof_{tag}"
    scratch_dir = artifact_dir / f"ts_scratch_{tag}"

    cfg = NestedPipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        proof_dir=proof_dir,
        hours=1,
        max_dom=maxdom,
        scratch_dir=scratch_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    proof_dir.mkdir(parents=True, exist_ok=True)

    names = domain_names_for(maxdom)
    print(
        f"MARKER:PROCESS_START epoch={time.time():.3f} tag={tag} maxdom={maxdom} "
        f"fixed_root_steps={fixed_root_steps} fixed_repeats={fixed_repeats} "
        f"input={input_dir} artifact_dir={artifact_dir} "
        f"defuse={os.environ.get('GPUWRF_NESTED_DEFUSE_COMPILE','0')} "
        f"fuse={os.environ.get('GPUWRF_NESTED_FUSE','(unset)')} "
        f"aot={os.environ.get('GPUWRF_NESTED_AOT','0')} "
        f"parallel={os.environ.get('GPUWRF_NESTED_PARALLEL_COMPILE','0')}",
        flush=True,
    )

    t_load0 = time.perf_counter()
    hierarchy, bundles, _meta, _run_start, dt_by_domain, initial_carries = _load_domains(cfg, names)
    root = names[0]
    cadence_run = Gen2Run(input_dir)
    output_cadence, _hist_min = _output_cadence_steps_by_domain(cadence_run, names, dt_by_domain)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=bool(cfg.feedback))
    nested_block_between, nested_root_sync_cadence = _nested_sync_mode_from_env()
    print(
        f"MARKER:LOADED wall_s={time.perf_counter() - t_load0:.1f} "
        f"root_dt={dt_by_domain[root]} names={names}",
        flush=True,
    )

    pc_t0 = time.perf_counter()
    pc = maybe_prewarm_defused_nest(tree, carries=initial_carries)
    print(
        f"MARKER:PREWARM_DONE wall_s={time.perf_counter() - pc_t0:.1f} "
        f"active={pc.get('active')} source={pc.get('source')}",
        flush=True,
    )

    def one_call(root_steps: int, label: str):
        t0 = time.perf_counter()
        result = run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=bool(cfg.feedback),
            output=None,
            output_cadence_steps=output_cadence,
            block_between=nested_block_between,
            root_sync_cadence=nested_root_sync_cadence,
            carries=initial_carries,
            initial_own_steps={n: 0 for n in names},
        )
        jax.block_until_ready(tuple(s.theta for s in result.states.values()))
        dt = time.perf_counter() - t0
        finite = all(bool(np.isfinite(np.asarray(s.theta)).all()) for s in result.states.values())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        vram = _gpu_mem_mib()
        print(
            f"CALL {label} root_steps={root_steps} wall_s={dt:.4f} finite={finite} "
            f"vram_mib={vram} rss_peak_mib={rss:.0f} d01_own={dict(result.own_steps).get('d01')}",
            flush=True,
        )
        return dt, result, finite, vram, rss

    call_times: list[float] = []
    finites: list[bool] = []
    vrams: list[int] = []
    rss_values: list[float] = []
    last_result = None
    for idx in range(fixed_repeats):
        label = "FIXED_FIRST" if idx == 0 else f"FIXED_{idx + 1}"
        dt_i, result_i, finite_i, vram_i, rss_i = one_call(fixed_root_steps, label)
        call_times.append(dt_i)
        finites.append(finite_i)
        vrams.append(vram_i)
        rss_values.append(rss_i)
        last_result = result_i
    assert last_result is not None

    wall_total = sum(call_times)
    s_per_step = (wall_total - call_times[0]) / ((fixed_repeats - 1) * fixed_root_steps)
    root_dt_s = float(dt_by_domain[root])
    s_per_fc_hour = s_per_step * (3600.0 / root_dt_s) if root_dt_s > 0 else float("nan")
    tail_vrams = vrams[1:] or vrams
    tail_rss = rss_values[1:] or rss_values
    summary = {
        "tag": tag,
        "input_dir": str(input_dir),
        "maxdom": maxdom,
        "fixed_root_steps": fixed_root_steps,
        "fixed_repeats": fixed_repeats,
        "call_times_s": call_times,
        "s_per_step": s_per_step,
        "s_per_fc_hour": s_per_fc_hour,
        "vram_peak_mib": max(vrams),
        "rss_peak_mib": max(rss_values),
        "vram_flat": (max(tail_vrams) - min(tail_vrams)) <= 50,
        "rss_flat": (max(tail_rss) - min(tail_rss)) <= 50,
        "all_finite": all(finites),
        "final_d01_own": dict(last_result.own_steps).get("d01"),
    }
    print(
        f"SUMMARY tag={tag} maxdom={maxdom} mode=fixed fixed_root_steps={fixed_root_steps} "
        f"fixed_repeats={fixed_repeats} first_s={call_times[0]:.4f} "
        f"wall_total_s={wall_total:.4f} call_times_s={','.join(f'{v:.4f}' for v in call_times)} "
        f"s_per_step={s_per_step:.4f} s_per_fc_hour={s_per_fc_hour:.4f} "
        f"vram_peak_mib={summary['vram_peak_mib']} rss_peak_mib={summary['rss_peak_mib']:.0f} "
        f"vram_flat={summary['vram_flat']} rss_flat={summary['rss_flat']} "
        f"all_finite={summary['all_finite']} final_d01_own={summary['final_d01_own']}",
        flush=True,
    )

    digest_path = artifact_dir / f"{tag}_state_digest.json"
    digest = _write_state_digest(digest_path, tag=tag, states=last_result.states)
    summary["digest"] = digest
    summary["digest_path"] = str(digest_path)
    print(
        f"BIT_ID tag={tag} digest_sha256={digest['sha256']} digest_leaves={digest['leaves']} "
        f"digest_bytes={digest['bytes']} digest_path={digest_path}",
        flush=True,
    )

    ref_digest = os.environ.get("CANARY_REF_DIGEST")
    if ref_digest:
        ref = json.loads(Path(ref_digest).read_text())
        ref_equal = (
            ref.get("sha256") == digest.get("sha256")
            and ref.get("leaves") == digest.get("leaves")
            and ref.get("bytes") == digest.get("bytes")
        )
        summary["ref_digest"] = ref_digest
        summary["ref_equal"] = ref_equal
        print(
            f"REF_COMPARE tag={tag} ref_digest={ref_digest} ref_equal={ref_equal} "
            f"ref_sha256={ref.get('sha256')} digest_sha256={digest.get('sha256')}",
            flush=True,
        )

    if _truthy("CANARY_DUMP_STATE", "0"):
        npz_path = artifact_dir / f"{tag}_state.npz"
        meta_path = artifact_dir / f"{tag}_state_meta.json"
        _write_state_npz(npz_path, meta_path, tag=tag, states=last_result.states)
        summary["state_npz"] = str(npz_path)
        summary["state_meta"] = str(meta_path)
        print(f"STATE_DUMP tag={tag} npz={npz_path} meta={meta_path}", flush=True)

    summary_path = artifact_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"SUMMARY_JSON tag={tag} path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
