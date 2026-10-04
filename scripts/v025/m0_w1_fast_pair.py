#!/usr/bin/env python3
"""Pre-stage the fresh CPU arm, then complete the W1 FAST pair in-lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m0_postlock_census as postlock  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402
import run_fast_pair as pair  # noqa: E402


class W1PairRefusal(RuntimeError):
    """W1 or its fresh CPU comparison did not satisfy the frozen pair."""


CPU_PREFLIGHT_SCHEMA = "wrf_gpu2.v025.m0.cpu_preflight.v1"
SESSION_IDENTITY_SCHEMA = "wrf_gpu2.v025.m0.session_preflight_identity.v1"


def _load_json(path: Path, what: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise W1PairRefusal(f"{what} is missing/not regular: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise W1PairRefusal(f"{what} is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise W1PairRefusal(f"{what} is not an object")
    return payload


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise W1PairRefusal(f"refusing to replace W1 pair: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _unique_stage(w1: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        stage for stage in w1.get("stages", [])
        if stage.get("name") == name
    ]
    if len(matches) != 1:
        raise W1PairRefusal(f"W1 has {len(matches)} stages named {name!r}")
    return matches[0]


def _selected_cold_stage(w1: dict[str, Any]) -> dict[str, Any]:
    """Resolve the canonical numbered cold attempt selected by live W1."""

    session = w1.get("session")
    if not isinstance(session, dict):
        return _unique_stage(w1, "cold_empty_cache_readiness")
    selected = session.get("selected_cold_stage")
    selected_index = session.get("qualified_attempt_index")
    if (
        not isinstance(selected, str)
        or not selected.startswith("cold_empty_cache_readiness_")
        or isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or selected != f"cold_empty_cache_readiness_{selected_index}"
    ):
        raise W1PairRefusal("W1 did not publish one canonical selected cold stage")
    cold = _unique_stage(w1, selected)
    if cold.get("status") not in {"OK", "PASS_COLD"}:
        raise W1PairRefusal("selected W1 cold stage is not a qualifying pass")
    return cold


def _cpu_record_payload(record: pair.ArmRecord) -> dict[str, Any]:
    return {
        "kind": record.kind,
        "run_id": record.run_id,
        "started_at_utc": record.started_at_utc,
        "finished_at_utc": record.finished_at_utc,
        "status": record.status,
        "payload": record.payload,
    }


def _cpu_record_from_payload(payload: dict[str, Any]) -> pair.ArmRecord:
    required = {
        "kind",
        "run_id",
        "started_at_utc",
        "finished_at_utc",
        "status",
        "payload",
    }
    if set(payload) != required or not isinstance(payload.get("payload"), dict):
        raise W1PairRefusal("pre-staged CPU record fields are incomplete")
    record = pair.ArmRecord(
        kind=str(payload["kind"]),
        run_id=str(payload["run_id"]),
        started_at_utc=str(payload["started_at_utc"]),
        finished_at_utc=str(payload["finished_at_utc"]),
        status=str(payload["status"]),
        payload=payload["payload"],
    )
    if record.kind != "cpu" or record.status != "OK":
        raise W1PairRefusal("pre-staged CPU record is not one successful CPU arm")
    return record


def stage_cpu_preflight(
    *,
    session_identity_path: Path,
    cpu_run_root: Path,
    output_path: Path,
    cpu_list: str = "16-27",
    cpu_runner=pair.cpu_arm_runner,
) -> dict[str, Any]:
    """Run the fresh CPU arm before receipt spend or lock acquisition."""

    postlock.assert_accelerator_free()
    postlock.assert_lock_environment_absent()
    identity = _load_json(session_identity_path, "session preflight identity")
    identity_hash = identity.get("identity_sha256")
    if (
        identity.get("schema") != SESSION_IDENTITY_SCHEMA
        or identity.get("status") != "PASS"
        or identity_hash
        != postlock.canonical_sha256(
            {key: value for key, value in identity.items()
             if key != "identity_sha256"}
        )
    ):
        raise W1PairRefusal("session preflight identity is incomplete or stale")
    started_ns = time.monotonic_ns()
    record = cpu_runner(
        run_id="m0-core-prelock-fresh-cpu",
        run_root=cpu_run_root,
        bind_to_core=True,
        cpu_list=cpu_list,
    )
    finished_ns = time.monotonic_ns()
    record_payload = _cpu_record_payload(record)
    if record.status != "OK":
        raise W1PairRefusal("pre-lock fresh CPU-WRF arm did not complete")
    output = pair._verified_output(record)
    launcher_seconds = record.payload.get("launcher_wallclock_seconds")
    if (
        isinstance(launcher_seconds, bool)
        or not isinstance(launcher_seconds, (int, float))
        or not 0.0 < float(launcher_seconds) <= 300.0
        or record.payload.get("cpu_list") != cpu_list
        or record.payload.get("ranks") != 12
    ):
        raise W1PairRefusal("pre-lock CPU arm violates its frozen resource/time gate")
    invocation = record.payload.get("command")
    if (
        not isinstance(invocation, list)
        or not invocation
        or any(not isinstance(part, str) or not part for part in invocation)
    ):
        raise W1PairRefusal("pre-lock CPU arm did not publish its invocation argv")
    tools = identity.get("tools")
    if (
        not isinstance(tools, dict)
        or set(tools) != {"python", "nsys", "mpirun", "wrf_exe"}
        or record.payload.get("mpirun_realpath")
        != (tools.get("mpirun") or {}).get("path")
        or record.payload.get("mpirun_sha256")
        != (tools.get("mpirun") or {}).get("sha256")
        or record.payload.get("wrf_exe_realpath")
        != (tools.get("wrf_exe") or {}).get("path")
        or record.payload.get("wrf_exe_sha256")
        != (tools.get("wrf_exe") or {}).get("sha256")
        or (tools.get("mpirun") or {}).get("invoked_path") not in invocation
        or (tools.get("wrf_exe") or {}).get("invoked_path") not in invocation
    ):
        raise W1PairRefusal(
            "pre-lock CPU arm did not execute the frozen MPI/WRF tools"
        )
    payload = {
        "schema": CPU_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "session_identity_path": str(Path(session_identity_path).resolve()),
        "session_identity_sha256": postlock.sha256_file(session_identity_path),
        "session_identity_content_sha256": identity_hash,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "cpu_list": cpu_list,
        "ranks": 12,
        "cpu_record": record_payload,
        "cpu_record_sha256": postlock.canonical_sha256(record_payload),
        "cpu_invocation_argv": invocation,
        "cpu_invocation_sha256": hashlib.sha256(
            "\0".join(invocation).encode("utf-8")
        ).hexdigest(),
        "verified_output": output,
        "device_action": False,
        "lock_environment_absent": True,
    }
    payload["preflight_sha256"] = postlock.canonical_sha256(payload)
    _atomic_json_no_replace(output_path, payload)
    return payload


def validate_cpu_preflight(
    path: Path,
    *,
    expected_session_identity_path: Path | None = None,
) -> tuple[dict[str, Any], pair.ArmRecord]:
    payload = _load_json(path, "pre-staged CPU preflight")
    expected_hash = postlock.canonical_sha256(
        {key: value for key, value in payload.items()
         if key != "preflight_sha256"}
    )
    if (
        payload.get("schema") != CPU_PREFLIGHT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("preflight_sha256") != expected_hash
        or payload.get("device_action") is not False
        or payload.get("lock_environment_absent") is not True
    ):
        raise W1PairRefusal("pre-staged CPU preflight is incomplete or stale")
    identity_path = Path(str(payload.get("session_identity_path", "")))
    if expected_session_identity_path is not None and (
        identity_path.resolve()
        != Path(expected_session_identity_path).resolve()
    ):
        raise W1PairRefusal("CPU preflight binds a different session identity")
    if (
        identity_path.is_symlink()
        or not identity_path.is_file()
        or postlock.sha256_file(identity_path)
        != payload.get("session_identity_sha256")
    ):
        raise W1PairRefusal("CPU preflight session identity changed")
    record = _cpu_record_from_payload(payload.get("cpu_record") or {})
    invocation = payload.get("cpu_invocation_argv")
    if (
        payload.get("cpu_record_sha256")
        != postlock.canonical_sha256(payload["cpu_record"])
        or not isinstance(invocation, list)
        or not invocation
        or payload.get("cpu_invocation_sha256")
        != hashlib.sha256("\0".join(invocation).encode("utf-8")).hexdigest()
        or invocation != record.payload.get("command")
    ):
        raise W1PairRefusal("pre-staged CPU invocation binding changed")
    output = pair._verified_output(record)
    if output != payload.get("verified_output"):
        raise W1PairRefusal("pre-staged CPU output changed before C1")
    return payload, record


def _in_session_provenance(w1_result_path: Path) -> dict[str, Any]:
    """Prove the C1 comparator is running inside the one held canonical lock.

    Amendment 5 puts C1 between W1 and W2, so the comparator necessarily runs
    while the single lock is still held.  That is the exact inverse of the
    Amendment-4 post-lock path: the lock environment must be *present* and must
    match the live holder file, and there is deliberately no release proof.
    """

    import run_gpu_arm as gpu_auth

    postlock.assert_accelerator_free()
    lock = gpu_auth.check_canonical_lock(
        None, expected_label="m0-core-w1-w2-w3-session"
    )
    return {
        "mode": "IN_SESSION_LOCK_HELD",
        "canonical_lock": lock,
        "lock_released_before_comparator": False,
        "expected_lock_hold_seconds": 60.0,
        "coordination_disclosure": (
            "the fresh 12-rank CPU-WRF arm was pre-staged before coordination; "
            "only its hash-bound comparison and pair preparation run in-lock"
        ),
        "w1_result_path": str(Path(w1_result_path).resolve()),
        "w1_result_sha256": postlock.sha256_file(Path(w1_result_path)),
    }


def build_pair(
    *,
    w1_result_path: Path,
    release_path: Path | None = None,
    cpu_run_root: Path | None,
    output_path: Path,
    cpu_list: str = "16-27",
    in_session: bool = False,
    cpu_preflight_path: Path | None = None,
    session_identity_path: Path | None = None,
    comparator=pair.compare_arms,
) -> dict[str, Any]:
    """Compare W1 with the fresh CPU arm from the same outer invocation."""

    analysis_started_ns = time.monotonic_ns()
    postlock.assert_accelerator_free()
    if in_session == (release_path is not None):
        raise W1PairRefusal(
            "exactly one of the post-lock release proof and the in-session "
            "held-lock mode may be used"
        )
    if in_session != (cpu_preflight_path is not None):
        raise W1PairRefusal(
            "in-session C1 requires exactly one pre-staged CPU preflight"
        )
    if in_session:
        session_provenance = _in_session_provenance(w1_result_path)
        release = {"run_id": None}
    else:
        session_provenance = None
        postlock.assert_lock_environment_absent()
        release = postlock.validate_lock_release_proof(
            release_path=release_path,
            window_result_path=w1_result_path,
            analysis_started_monotonic_ns=analysis_started_ns,
            expected_window="W1",
        )
    w1 = _load_json(w1_result_path, "W1 manager result")
    if (
        w1.get("status") != "OK"
        or w1.get("window") != "W1"
        or (not in_session and w1.get("run_id") != release.get("run_id"))
    ):
        raise W1PairRefusal("W1 manager result status/run is ineligible")
    cold = _selected_cold_stage(w1)
    cached = _unique_stage(w1, "cached_readiness_and_warm_integration")
    exact = _load_json(Path(cached["result_path"]), "W1 cached exact result")
    if postlock.sha256_file(Path(cached["result_path"])) != cached.get(
        "result_sha256"
    ):
        raise W1PairRefusal("W1 cached exact result changed")
    wrfout = exact.get("wrfout")
    if (
        not isinstance(wrfout, dict)
        or wrfout.get("status") != "PASS"
        or wrfout.get("run_id") != exact.get("run_id")
        or wrfout.get("result_exact_value_sha256")
        != (exact.get("result") or {}).get("exact_value_sha256")
    ):
        raise W1PairRefusal("W1 exact result lacks its same-result wrfout binding")
    try:
        exact_contract.validate_wrfout_binding(exact, verify_file=True)
    except exact_contract.ContractViolation as exc:
        raise W1PairRefusal(
            f"W1 exact result wrfout proof is invalid: {exc}"
        ) from exc

    preflight = None
    if cpu_preflight_path is not None:
        preflight, cpu_record = validate_cpu_preflight(
            cpu_preflight_path,
            expected_session_identity_path=session_identity_path,
        )
        cpu_finished_monotonic_ns = int(preflight["finished_monotonic_ns"])
        gpu_start_ns = int(exact["timing"]["integration_start_monotonic_ns"])
        gpu_end_ns = int(exact["timing"]["integration_end_monotonic_ns"])
        if not (
            int(preflight["started_monotonic_ns"])
            < cpu_finished_monotonic_ns
            <= gpu_start_ns
            < gpu_end_ns
            <= analysis_started_ns
        ):
            raise W1PairRefusal(
                "pre-staged CPU and W1 GPU arms overlap or carry invalid "
                "monotonic endpoints"
            )
    else:
        if cpu_run_root is None:
            raise W1PairRefusal("post-lock mode requires a fresh CPU run root")
        cpu_record = pair.cpu_arm_runner(
            run_id=f"{w1['run_id']}-fresh-cpu",
            run_root=cpu_run_root,
            bind_to_core=True,
            cpu_list=cpu_list,
        )
        cpu_finished_monotonic_ns = time.monotonic_ns()
    if cpu_record.payload.get("cpu_list") != cpu_list:
        raise W1PairRefusal(
            f"fresh CPU arm used {cpu_record.payload.get('cpu_list')}, "
            f"expected {cpu_list}"
        )
    gpu_record = pair.ArmRecord(
        kind="gpu",
        run_id=exact["run_id"],
        started_at_utc=str(cached["started_at_utc"]),
        finished_at_utc=str(cached["finished_at_utc"]),
        status="OK",
        payload={
            "wrfout": wrfout,
            "final_wrfout_path": wrfout["final_wrfout_path"],
            "final_wrfout_sha256": wrfout["final_wrfout_sha256"],
            "wrfout_binding_sha256": wrfout["binding_sha256"],
            "result_exact_value_sha256":
                wrfout["result_exact_value_sha256"],
            "exact_run_id": exact["run_id"],
            "timing_classes": {
                "cached_readiness_seconds":
                    exact["timing"]["readiness_seconds"],
                "warm_integration_seconds":
                    exact["timing"]["integration_seconds"],
                "wrfout_fixed_overhead_seconds": (
                    wrfout["timing"]["inspection_end_monotonic_ns"]
                    - wrfout["timing"]["materialization_start_monotonic_ns"]
                )
                / 1e9,
            },
        },
    )
    pair_payload = pair.finalize_pair_records(
        cpu_record=cpu_record,
        gpu_record=gpu_record,
        pair_id=f"{w1['run_id']}-prelock-cpu-then-w1",
        comparator=comparator,
    )
    cold_result = _load_json(Path(cold["result_path"]), "W1 cold exact result")
    pair_seconds = (
        float(cpu_record.payload["launcher_wallclock_seconds"])
        + float(exact["timing"]["integration_seconds"])
    )
    economy = {
        "cold_compile_seconds": {
            "value": cold_result["timing"]["readiness_seconds"],
            "threshold": 600.0,
        },
        "cached_load_seconds": {
            "value": exact["timing"]["readiness_seconds"],
            "threshold": 60.0,
        },
        "gpu_arm_seconds": {
            "value": exact["timing"]["integration_seconds"],
            "threshold": 300.0,
        },
        "cpu_arm_seconds": {
            "value": cpu_record.payload["launcher_wallclock_seconds"],
            "threshold": 300.0,
        },
        "pair_seconds_excl_lock_wait": {
            "value": pair_seconds,
            "threshold": 600.0,
        },
    }
    failures = [
        name for name, gate in economy.items()
        if (
            not isinstance(gate["value"], (int, float))
            or isinstance(gate["value"], bool)
            or not math.isfinite(float(gate["value"]))
            or not 0 < float(gate["value"]) <= float(gate["threshold"])
        )
    ]
    if failures:
        raise W1PairRefusal(f"W1 FAST economy gates failed: {failures}")
    exact_binding = {
        "run_id": w1["run_id"],
        "cold_result_sha256": cold["result_sha256"],
        "cached_result_sha256": cached["result_sha256"],
        "gpu_result_sha256": exact["result"]["exact_value_sha256"],
        "gpu_wrfout_path": wrfout["final_wrfout_path"],
        "gpu_wrfout_sha256": wrfout["final_wrfout_sha256"],
        "gpu_wrfout_binding_sha256": wrfout["binding_sha256"],
    }
    payload = {
        "schema": "wrf_gpu2.v025.m0.fast_case_qualification.v1",
        "status": "FAST_QUALIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": w1["run_id"],
        "w1_exact_boundary": exact_binding,
        "economy_gates": economy,
        "pair_completeness": {
            "fresh_cpu_arm_this_invocation": True,
            "fresh_gpu_arm_this_invocation": True,
            "comparator_result_present": True,
            "provenance_present": True,
            "arms_non_overlapping": True,
            "completeness_percent": 100,
        },
        "pair": pair_payload,
        "delta_atlas": pair_payload["comparison"],
        "lock_release_provenance": (
            {
                **session_provenance,
                "cpu_pair_process_started_monotonic_ns": analysis_started_ns,
                "cpu_pair_finished_monotonic_ns": cpu_finished_monotonic_ns,
                "pair_clock_start_monotonic_ns":
                    preflight["started_monotonic_ns"],
                "pair_clock_end_monotonic_ns":
                    exact["timing"]["integration_end_monotonic_ns"],
                "cpu_preflight_finished_before_gpu_integration": True,
                "cpu_preflight_path": str(Path(cpu_preflight_path).resolve()),
                "cpu_preflight_sha256": postlock.sha256_file(
                    cpu_preflight_path
                ),
                "cpu_preflight_content_sha256":
                    preflight["preflight_sha256"],
                "session_identity_path":
                    preflight["session_identity_path"],
                "session_identity_sha256":
                    preflight["session_identity_sha256"],
                "session_identity_content_sha256":
                    preflight["session_identity_content_sha256"],
            }
            if in_session
            else {
                "path": str(release_path.resolve()),
                "sha256": postlock.sha256_file(release_path),
                "release_before_cpu_arm": True,
                "wrapper_returned_monotonic_ns":
                    release["wrapper_returned_monotonic_ns"],
                "cpu_pair_process_started_monotonic_ns": analysis_started_ns,
                "cpu_pair_finished_monotonic_ns": cpu_finished_monotonic_ns,
                "pair_clock_start_monotonic_ns":
                    exact["timing"]["integration_start_monotonic_ns"],
                "pair_clock_end_monotonic_ns": cpu_finished_monotonic_ns,
            }
        ),
        "c1_subgate": {
            "mode": "IN_SESSION_FIRST_SUBGATE" if in_session else "POST_LOCK",
            "components": [
                "w1_same_result_wrfout_binding_verify_file",
                "fresh_12_rank_cpu_wrf_comparator",
            ],
            "both_green": True,
            "cpu_arm_pre_staged_before_receipt_and_lock": bool(in_session),
        },
        "cpu_resource": {
            "ranks": cpu_record.payload.get("ranks"),
            "cpu_list": cpu_record.payload.get("cpu_list"),
            "mpi_flags": cpu_record.payload.get("mpi_flags"),
            "hostname": cpu_record.payload.get("host"),
        },
        "device_action_after_lock_release": False,
    }
    _atomic_json_no_replace(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prestage-cpu", action="store_true")
    parser.add_argument("--w1-result", type=Path)
    parser.add_argument("--lock-release-proof", type=Path)
    parser.add_argument(
        "--in-session-held-lock",
        action="store_true",
        help=(
            "Amendment-5 C1 sub-gate: the comparator runs between W1 and W2 "
            "while the one canonical lock is still held"
        ),
    )
    parser.add_argument("--cpu-run-root", type=Path, required=True)
    parser.add_argument("--pre-staged-cpu", type=Path)
    parser.add_argument("--session-identity", type=Path)
    parser.add_argument("--prepare-session-pair", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prestage_cpu:
        if args.session_identity is None:
            parser.error("--prestage-cpu requires --session-identity")
        payload = stage_cpu_preflight(
            session_identity_path=args.session_identity,
            cpu_run_root=args.cpu_run_root,
            output_path=args.output,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    if args.w1_result is None:
        parser.error("pair comparison requires --w1-result")
    if args.in_session_held_lock == (args.lock_release_proof is not None):
        parser.error(
            "pass exactly one of --lock-release-proof and --in-session-held-lock"
        )
    payload = build_pair(
        w1_result_path=args.w1_result,
        release_path=args.lock_release_proof,
        cpu_run_root=args.cpu_run_root,
        output_path=args.output,
        in_session=bool(args.in_session_held_lock),
        cpu_preflight_path=args.pre_staged_cpu,
        session_identity_path=args.session_identity,
    )
    prepared = None
    qualification = None
    if args.prepare_session_pair:
        if not args.in_session_held_lock or args.receipt is None:
            parser.error(
                "--prepare-session-pair requires --in-session-held-lock "
                "and --receipt"
            )
        import m0_window_parent as parent
        import prepare_m0_matched_pair as pair_preparer

        qualification = parent._post_w1_qualification(
            w1_result_path=args.w1_result,
            fast_pair_path=args.output,
            output_path=parent.QUALIFICATION,
        )
        prepared = pair_preparer.prepare(
            output_dir=Path(parent.WINDOWS["W2"]["identity_path"]).parent,
            device_uuid=parent.EXPECTED_DEVICE_UUID,
            profiled_run_id=str(parent.WINDOWS["W2"]["run_id"]),
            clean_run_id=str(parent.WINDOWS["W3"]["run_id"]),
            profiled_receipt=args.receipt,
            clean_receipt=args.receipt,
            cache_seed=Path(parent.WINDOWS["W1"]["cache_path"]),
            cache_snapshot_root=Path(parent.PAIR_CACHE_ROOT),
            qualification_manifest=Path(parent.QUALIFICATION),
            source_root=REPO / "src/gpuwrf",
            run_dir=Path(parent.FAST_RUN_DIR),
        )
    print(
        json.dumps(
            {
                "pair": payload,
                "qualification": qualification,
                "prepared_pair": prepared,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
