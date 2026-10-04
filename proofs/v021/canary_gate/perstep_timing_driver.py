"""Per-root-step timing driver (TEST INFRA ONLY -- no model code).

Reuses the production nested_pipeline setup path (_load_domains, DomainTree,
prewarm) then drives the live nest via the SAME public entry the pipeline's
own segment loop uses (run_operational_domain_tree). The fused-vs-defuse path
is selected entirely by the env flags at compile time.

MEASUREMENT METHOD -- difference of two single calls (cancels the per-CALL
fixed cost so we isolate the steady per-root-step GPU dispatch cost):

  run_operational_domain_tree builds the cascade closures + loads the AOT
  executables ONCE per call, THEN loops `root_steps` internal steps over the
  already-compiled executable (cheap dispatch). The production pipeline calls
  it ONCE per output segment (~200 steps) so that per-call setup is amortized.
  Measuring per-step by calling it once-per-step would (wrongly) pay the setup
  N times. Instead we time ONE call with root_steps=K_WARM and ONE call with
  root_steps=1 on the SAME warm cache:

      s_per_step = (wall_Kwarm - wall_1) / (K_WARM - 1)

  The fixed per-call setup (AOT load, cascade build, first-step compile if any)
  cancels, leaving the true steady per-step cost. We run a separate cold call
  first to warm the persistent cache so both timed calls are warm.

Env: CANARY_MAXDOM/MAXDOM, CANARY_INPUT_DIR, CANARY_TAG, CANARY_KWARM (default 12).
Set CANARY_SINGLE_SHAPE_TIMING=1 for the 9-nest AOT gate: it repeats the same
root_steps call from the same initial clock/carries, avoiding one heavy compile
per timing shape or start-step key.
"""

from __future__ import annotations

import os
import resource
import subprocess
import time
import traceback
import hashlib
import json
from pathlib import Path

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

INPUT = os.environ.get("CANARY_INPUT_DIR", "<DATA_ROOT>/wrf_downscale/runs/20240901/cpu")
TAG = os.environ.get("CANARY_TAG", "fused_aot")
MAXDOM = int(os.environ.get("CANARY_MAXDOM", os.environ.get("MAXDOM", "3")))
KWARM = int(os.environ.get("CANARY_KWARM", "12"))
SINGLE_SHAPE_TIMING = os.environ.get("CANARY_SINGLE_SHAPE_TIMING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FIXED_ROOT_STEPS = int(os.environ.get("CANARY_FIXED_ROOT_STEPS", str(KWARM)))
FIXED_REPEATS = int(os.environ.get("CANARY_FIXED_REPEATS", "3"))
BASE = Path(__file__).resolve().parent


def _gpu_mem_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def _safe_path_part(part) -> str:
    raw = getattr(part, "name", getattr(part, "key", getattr(part, "idx", part)))
    return str(raw).replace("/", "_").replace(" ", "")


def _state_leaf_items(states):
    import jax
    import numpy as np

    for domain, state in sorted(states.items()):
        path_leaves, _treedef = jax.tree_util.tree_flatten_with_path(state)
        for idx, (path, leaf) in enumerate(path_leaves):
            arr = np.asarray(leaf)
            if not hasattr(arr, "dtype"):
                continue
            suffix = ".".join(_safe_path_part(part) for part in path) or f"leaf{idx}"
            yield f"{domain}/{idx:04d}/{suffix}", np.ascontiguousarray(arr)


def _write_state_digest(path: Path, *, tag: str, states) -> dict[str, object]:
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


def _compare_state_exact(lhs, rhs) -> tuple[bool, float, int]:
    import numpy as np

    lhs_items = list(_state_leaf_items(lhs))
    rhs_items = list(_state_leaf_items(rhs))
    if len(lhs_items) != len(rhs_items):
        return False, float("inf"), abs(len(lhs_items) - len(rhs_items))
    exact = True
    max_abs = 0.0
    mismatches = 0
    for (lhs_key, lhs_arr), (rhs_key, rhs_arr) in zip(lhs_items, rhs_items, strict=True):
        if lhs_key != rhs_key or lhs_arr.shape != rhs_arr.shape or lhs_arr.dtype != rhs_arr.dtype:
            exact = False
            mismatches += 1
            max_abs = float("inf")
            continue
        if np.array_equal(lhs_arr, rhs_arr):
            continue
        exact = False
        mismatches += 1
        if np.issubdtype(lhs_arr.dtype, np.number) and lhs_arr.size:
            diff = np.max(np.abs(lhs_arr.astype(np.float64) - rhs_arr.astype(np.float64)))
            max_abs = max(max_abs, float(diff))
        else:
            max_abs = float("inf")
    return exact, max_abs, mismatches


def main() -> None:
    import jax
    import numpy as np

    cfg = NestedPipelineConfig(
        input_dir=Path(INPUT),
        output_dir=BASE / f"ts_output_{TAG}",
        proof_dir=BASE / f"ts_proof_{TAG}",
        hours=1,
        max_dom=MAXDOM,
        scratch_dir=BASE / f"ts_scratch_{TAG}",
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.proof_dir.mkdir(parents=True, exist_ok=True)

    names = domain_names_for(MAXDOM)
    print(
        f"MARKER:PROCESS_START epoch={time.time():.3f} tag={TAG} maxdom={MAXDOM} kwarm={KWARM} "
        f"mode={'fixed' if SINGLE_SHAPE_TIMING else 'difference'} "
        f"fixed_root_steps={FIXED_ROOT_STEPS} fixed_repeats={FIXED_REPEATS} "
        f"defuse={os.environ.get('GPUWRF_NESTED_DEFUSE_COMPILE','0')} "
        f"fuse={os.environ.get('GPUWRF_NESTED_FUSE','(unset)')} "
        f"aot={os.environ.get('GPUWRF_NESTED_AOT','0')} "
        f"parallel={os.environ.get('GPUWRF_NESTED_PARALLEL_COMPILE','0')}",
        flush=True,
    )

    t_load0 = time.perf_counter()
    hierarchy, bundles, _meta, _run_start, dt_by_domain, initial_carries = _load_domains(cfg, names)
    root = names[0]
    cadence_run = Gen2Run(Path(cfg.input_dir))
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

    def one_call(root_steps, carries, own_steps, label):
        t0 = time.perf_counter()
        result = run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=bool(cfg.feedback),
            output=None,
            output_cadence_steps=output_cadence,
            block_between=nested_block_between,
            root_sync_cadence=nested_root_sync_cadence,
            carries=carries,
            initial_own_steps=own_steps,
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

    try:
        if SINGLE_SHAPE_TIMING:
            if FIXED_ROOT_STEPS <= 0:
                raise ValueError("CANARY_FIXED_ROOT_STEPS must be positive")
            if FIXED_REPEATS < 2:
                raise ValueError("CANARY_FIXED_REPEATS must be at least 2")

            fixed_own_steps = {n: 0 for n in names}
            call_times: list[float] = []
            finites: list[bool] = []
            vrams: list[int] = []
            rss_values: list[float] = []
            last_result = None
            for idx in range(FIXED_REPEATS):
                label = "FIXED_FIRST" if idx == 0 else f"FIXED_{idx + 1}"
                dt_i, result_i, finite_i, vram_i, rss_i = one_call(
                    FIXED_ROOT_STEPS,
                    initial_carries,
                    fixed_own_steps,
                    label,
                )
                call_times.append(dt_i)
                finites.append(finite_i)
                vrams.append(vram_i)
                rss_values.append(rss_i)
                last_result = result_i

            assert last_result is not None
            wall_total = sum(call_times)
            s_per_step = (wall_total - call_times[0]) / (
                (FIXED_REPEATS - 1) * FIXED_ROOT_STEPS
            )
            root_dt_s = float(dt_by_domain[root])
            s_per_fc_hour = s_per_step * (3600.0 / root_dt_s) if root_dt_s > 0 else float("nan")
            tail_vrams = vrams[1:] or vrams
            tail_rss = rss_values[1:] or rss_values
            vram_peak = max(vrams)
            rss_peak = max(rss_values)
            vram_flat = (max(tail_vrams) - min(tail_vrams)) <= 50
            rss_flat = (max(tail_rss) - min(tail_rss)) <= 50
            print(
                f"SUMMARY tag={TAG} maxdom={MAXDOM} mode=fixed "
                f"fixed_root_steps={FIXED_ROOT_STEPS} fixed_repeats={FIXED_REPEATS} "
                f"first_s={call_times[0]:.4f} wall_total_s={wall_total:.4f} "
                f"call_times_s={','.join(f'{v:.4f}' for v in call_times)} "
                f"s_per_step={s_per_step:.4f} s_per_fc_hour={s_per_fc_hour:.4f} "
                f"vram_peak_mib={vram_peak} rss_peak_mib={rss_peak:.0f} "
                f"vram_flat={vram_flat} rss_flat={rss_flat} "
                f"all_finite={all(finites)} final_d01_own={dict(last_result.own_steps).get('d01')}",
                flush=True,
            )

            digest_path = cfg.proof_dir / "warmK_state_digest.json"
            digest = _write_state_digest(digest_path, tag=TAG, states=last_result.states)
            print(
                f"BIT_ID tag={TAG} fixed_mode=True "
                f"digest_sha256={digest['sha256']} digest_leaves={digest['leaves']} "
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
                print(
                    f"REF_COMPARE tag={TAG} ref_digest={ref_digest} ref_equal={ref_equal} "
                    f"ref_sha256={ref.get('sha256')} digest_sha256={digest.get('sha256')}",
                    flush=True,
                )
            return

        cold_dt, _cold_result, cold_fin, _cold_vram, _cold_rss = one_call(
            2, initial_carries, {n: 0 for n in names}, "COLD"
        )
        print(f"MARKER:COLD_DONE cold_wall_s={cold_dt:.1f} finite={cold_fin}", flush=True)

        warm1_dt, _r1, f1, v1, rss1 = one_call(1, initial_carries, {n: 0 for n in names}, "WARM_1")
        warmK_dt, _rK, fK, vK, rssK = one_call(
            KWARM, initial_carries, {n: 0 for n in names}, "WARM_K"
        )
        warm1b_dt, _r1b, f1b, v1b, rss1b = one_call(
            1, initial_carries, {n: 0 for n in names}, "WARM_1b"
        )
        warmKb_dt, _rKb, fKb, vKb, rssKb = one_call(
            KWARM, initial_carries, {n: 0 for n in names}, "WARM_Kb"
        )

        if KWARM > 1:
            s_per_step = (warmK_dt - warm1_dt) / (KWARM - 1)
            s_per_step_b = (warmKb_dt - warm1b_dt) / (KWARM - 1)
        else:
            s_per_step = s_per_step_b = float("nan")

        vram_peak = max(v1, vK, v1b, vKb)
        rss_peak = max(rss1, rssK, rss1b, rssKb)
        vram_flat = (vKb - vK) <= 50 and (v1b - v1) <= 50
        rss_flat = (rssKb - rssK) <= 50 and (rss1b - rss1) <= 50

        print(
            f"SUMMARY tag={TAG} maxdom={MAXDOM} kwarm={KWARM} "
            f"warm1_s={warm1_dt:.4f} warmK_s={warmK_dt:.4f} "
            f"warm1b_s={warm1b_dt:.4f} warmKb_s={warmKb_dt:.4f} "
            f"s_per_step={s_per_step:.4f} s_per_step_b={s_per_step_b:.4f} "
            f"vram_peak_mib={vram_peak} rss_peak_mib={rss_peak:.0f} "
            f"vram_flat={vram_flat} rss_flat={rss_flat} "
            f"all_finite={f1 and fK and f1b and fKb}",
            flush=True,
        )

        repeat_exact, repeat_max_abs, repeat_mismatches = _compare_state_exact(
            _rK.states, _rKb.states
        )
        digest_path = cfg.proof_dir / "warmK_state_digest.json"
        digest = _write_state_digest(digest_path, tag=TAG, states=_rK.states)
        print(
            f"BIT_ID tag={TAG} repeat_exact={repeat_exact} "
            f"repeat_max_abs_diff={repeat_max_abs:.12g} repeat_mismatches={repeat_mismatches} "
            f"digest_sha256={digest['sha256']} digest_leaves={digest['leaves']} "
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
            print(
                f"REF_COMPARE tag={TAG} ref_digest={ref_digest} ref_equal={ref_equal} "
                f"ref_sha256={ref.get('sha256')} digest_sha256={digest.get('sha256')}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"DRIVER RAISED {type(exc).__name__}: {str(exc)[:1500]}", flush=True)
        traceback.print_exc()
    print(f"SUMMARY tag={TAG} done", flush=True)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
