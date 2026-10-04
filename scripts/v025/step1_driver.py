#!/usr/bin/env python3
"""The `baseline-census` window: orchestrate -> capture (locked) -> analyse (CPU).

Manager review on main `8d27c1b3` accepted the CPU plumbing and rejected the live
boundary. Four things were wrong and all four were about the *boundary*, not the
arithmetic: release was simulated, no `PreemptionGuard` or process-group kill, a
fourth non-frozen window label with `receipt.check` called directly, and an H1
ratio between unlike clocks.

The architecture is now three PROCESSES, because that is what makes the lock
lifecycle real rather than asserted:

    --orchestrate   (CPU, no lock)
        └─ scripts/with_gpu_lock.sh --label baseline-census
               └─ --capture   (holds the lock, touches the GPU, exits fast)
        │  wrapper returns  ->  THIS is the proof the lock was released
        └─ --analyse    (CPU-pinned, separate process, no lock)

`--capture` is deliberately short-lived: it authorises, runs one nsys stage,
writes its outcome and exits. Analysis is a different process launched after the
wrapper returns, so the device is never held while parsing CSVs. Nothing here
"claims" release: the object records the wrapper's exit status, and if the
wrapper did not return, no release is reported.

This module does NOT import `cpu_guard` -- it is a coordinated GPU entry point,
and §13 is enforced by `run_gpu_arm.authorise` (window whitelist + single-use
receipt + canonical lock), which is stronger than an environment variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

import application_attribution as aa  # noqa: E402
import baseline_census as bc  # noqa: E402
import nsys_export as nex  # noqa: E402
import h1_discriminator as h1  # noqa: E402
import m0_stale_pair as stale_pair  # noqa: E402
import m0_vram_sampler as mvs  # noqa: E402
import nvtx_exclusive as nx  # noqa: E402
import parse_profiler as pp  # noqa: E402
import run_gpu_arm as arm  # noqa: E402
import run_window as rw  # noqa: E402

#: FROZEN. One of `run_gpu_arm.WINDOWS`; not a fourth label invented here.
WINDOW_LABEL = "baseline-census"

LOCK_WRAPPER = REPO / "scripts/with_gpu_lock.sh"
ANALYSIS_AFFINITY = "0-3"

FAST_CASE = {"run_dir": "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1", "hours": 1}

BUDGET_CAPTURE_S = 1100.0
H1_DECISIVE_THRESHOLD_S = 420.98      # = (1 - 0.30) * 601.4
OVERHEAD_S = 180.0
DEADLINE_S = 1500.0
KILL_GRACE_S = 10.0
# ZERO-WAIT. After an explicit dual-manager handover the device is ours; if it is
# somehow held, that is a coordination failure and must fail NOW. Waiting would
# silently queue and then steal whatever gap opened next -- the §13 violation
# already committed once in this sprint.
LOCK_TIMEOUT_S = 0

OUT_BASE = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/step1-runs")
# Overridable so a test (or a reviewer) can direct emitted objects elsewhere.
# The analysis runs in a SEPARATE PROCESS, so this must travel by environment;
# patching the module attribute in the parent cannot reach the child.
PROOFS = Path(os.environ.get("GPUWRF_STEP1_PROOFS", str(REPO / "proofs/v025/m0")))
HLO_STAGE = "after_optimizations"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")


class Step1Blocked(RuntimeError):
    """An artifact or boundary condition failed. Verdicts are suppressed."""


def new_run_id(prefix: str = "step1") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise Step1Blocked(
            "run ID must be 8-128 safe filename characters and start alphanumeric"
        )
    return run_id


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------- #
# commands                                                                      #
# ---------------------------------------------------------------------------- #
def forecast_arm_command(*, out_root: Path) -> list[str]:
    return arm.build_forecast_command(run_dir=Path(FAST_CASE["run_dir"]),
                                      out_dir=out_root / "autotune0",
                                      hours=FAST_CASE["hours"])


def nsys_command(inner: Sequence[str], *, out_root: Path) -> list[str]:
    return [
        "nsys", "profile", "--trace", "nvtx,cuda,osrt", "--sample", "none",
        "--cpuctxsw", "none",
        # Required for the baseline-census VRAM evidence; without it the census
        # reports VRAM UNAVAILABLE rather than inventing a number.
        "--cuda-memory-usage", "true",
        "--force-overwrite", "true",
        "--output", str(out_root / "step1_autotune_off"), *inner,
    ]


def stage_environment(
    *,
    out_root: Path,
    evidence_identity: "mvs.EvidenceIdentity | None" = None,
    prepared_cache_dir: Path | None = None,
) -> dict[str, str]:
    environment = {
        "XLA_FLAGS": " ".join(["--xla_gpu_autotune_level=0",
                               f"--xla_dump_to={out_root / 'dump'}",
                               "--xla_dump_hlo_pass_re=.*"]),
        "GPUWRF_JAX_CACHE": "1",
        "GPUWRF_JAX_CACHE_DIR": str(prepared_cache_dir or (out_root / "cache")),
    }
    if evidence_identity is not None:
        environment.update(mvs.hook_environment(
            evidence_identity,
            allocator_sidecar=(out_root / "forecast_allocator.json").resolve(),
        ))
    return environment


def validate_prepared_cache(
    path: Path | None,
    expected_sha256: str | None,
) -> dict[str, Any] | None:
    """Fail closed on partial, aliased, empty, or mutated matched-pair caches."""

    if (path is None) != (expected_sha256 is None):
        raise Step1Blocked(
            "prepared cache path and SHA-256 must be supplied together"
        )
    if path is None:
        return None
    if not path.is_absolute():
        raise Step1Blocked("prepared cache path must be absolute")
    if path.is_symlink():
        raise Step1Blocked("prepared cache root may not be a symlink")
    if (
        not isinstance(expected_sha256, str)
        or not mvs.SHA256_PATTERN.fullmatch(expected_sha256)
    ):
        raise Step1Blocked("prepared cache identity must be a full lowercase SHA-256")
    try:
        observed = mvs.directory_tree_identity(path)
    except mvs.ResidencyEvidenceError as exc:
        raise Step1Blocked(str(exc)) from exc
    if observed["sha256"] != expected_sha256:
        raise Step1Blocked(
            "prepared cache content does not match its pre-registered SHA-256"
        )
    return observed


def capture_command(
    *,
    out_root: Path,
    receipt: Path,
    run_id: str,
    evidence_identity: Path | None = None,
    profiled: bool = True,
    prepared_cache_dir: Path | None = None,
    prepared_cache_sha256: str | None = None,
) -> list[str]:
    try:
        stale_pair.reject_stale_prepared_pair(
            run_id=run_id,
            identity_path=evidence_identity,
            prepared_cache_sha256=prepared_cache_sha256,
        )
    except stale_pair.StalePairError as exc:
        raise Step1Blocked(str(exc)) from exc
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--capture" if profiled else "--capture-clean",
        "--out-root",
        str(out_root),
        "--run-id",
        validate_run_id(run_id),
        "--receipt",
        str(receipt),
    ]
    if evidence_identity is not None:
        command.extend(["--evidence-identity", str(evidence_identity)])
    if prepared_cache_dir is not None:
        command.extend(["--prepared-cache-dir", str(prepared_cache_dir)])
    if prepared_cache_sha256 is not None:
        command.extend(["--prepared-cache-sha256", prepared_cache_sha256])
    return command


def locked_capture_command(*, out_root: Path, receipt: Path, run_id: str,
                           evidence_identity: Path | None = None,
                           profiled: bool = True,
                           prepared_cache_dir: Path | None = None,
                           prepared_cache_sha256: str | None = None,
                           lock_wrapper: Path = LOCK_WRAPPER) -> list[str]:
    return [str(lock_wrapper), "--timeout", str(LOCK_TIMEOUT_S), "--label", WINDOW_LABEL, "--",
            *capture_command(
                out_root=out_root,
                receipt=receipt,
                run_id=run_id,
                evidence_identity=evidence_identity,
                profiled=profiled,
                prepared_cache_dir=prepared_cache_dir,
                prepared_cache_sha256=prepared_cache_sha256,
            )]


def analysis_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """A clean CPU environment for the analysis child.

    Analysis is a CPU-only stage launched by a process that may itself have been
    started with an accelerator platform exported. The orchestrator sanitises
    rather than inherits, so a hostile caller environment cannot carry a GPU
    platform into the post-release stage.
    """
    env = dict(os.environ if base is None else base)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["GPUWRF_STEP1_PROOFS"] = str(PROOFS)
    return env


def analyse_command(*, out_root: Path, run_id: str) -> list[str]:
    return ["taskset", "-c", ANALYSIS_AFFINITY, sys.executable,
            str(Path(__file__).resolve()), "--analyse", "--out-root", str(out_root),
            "--run-id", validate_run_id(run_id)]


# ---------------------------------------------------------------------------- #
# capture: holds the lock, kills the whole group, exits                         #
# ---------------------------------------------------------------------------- #
def run_capture(
    *,
    out_root: Path,
    run_id: str,
    receipt_path: Path | None,
    evidence_identity_path: Path | None = None,
    profiled: bool = True,
    prepared_cache_dir: Path | None = None,
    prepared_cache_sha256: str | None = None,
    env: dict[str, str] | None = None,
    stub_stage: Sequence[str] | None = None,
    timeout_s: float = BUDGET_CAPTURE_S,
    authorise: Callable[..., dict[str, Any]] | None = None,
    ledger_hint: Path | None = None,
    guard: "arm.PreemptionGuard | None" = None,
    sampler_factory: Callable[..., "mvs.LockOwnerResidencySampler"] | None = None,
) -> dict[str, Any]:
    """Authorise, run ONE stage in its own process group, write the outcome, exit.

    The child is started with `start_new_session=True`, so nsys and the forecast
    it launches share a process group we can signal as a unit. Killing only the
    direct child would leave the forecast running on the GPU after the lock was
    released -- an orphan holding the device nobody is tracking.
    """
    env = dict(os.environ if env is None else env)
    run_id = validate_run_id(run_id)
    try:
        stale_pair.reject_stale_prepared_pair(
            run_id=run_id,
            identity_path=evidence_identity_path,
            prepared_cache_sha256=prepared_cache_sha256,
        )
    except stale_pair.StalePairError as exc:
        raise Step1Blocked(str(exc)) from exc
    evidence_variables = (
        "GPUWRF_M0_EVIDENCE",
        "GPUWRF_M0_EVIDENCE_PATH",
        "GPUWRF_M0_RUN_ID",
        "GPUWRF_M0_SOURCE_SHA256",
        "GPUWRF_M0_CONFIG_SHA256",
        "GPUWRF_M0_INPUT_MANIFEST_SHA256",
        "GPUWRF_M0_DEVICE_UUID",
    )
    if not profiled and any(env.get(name) is not None for name in evidence_variables):
        raise Step1Blocked(
            "clean matched-pair arm requires every GPUWRF_M0 evidence variable unset"
        )
    if evidence_identity_path is None and stub_stage is None:
        raise Step1Blocked(
            "production capture requires --evidence-identity before receipt spend"
        )
    evidence_identity = (
        mvs.load_identity(evidence_identity_path, expected_run_id=run_id)
        if evidence_identity_path is not None
        else None
    )
    if evidence_identity is not None and stub_stage is None:
        mvs.validate_identity_against_workload(
            evidence_identity,
            source_root=REPO / "src/gpuwrf",
            run_dir=Path(FAST_CASE["run_dir"]),
            hours=int(FAST_CASE["hours"]),
        )
    prepared_cache = validate_prepared_cache(
        prepared_cache_dir, prepared_cache_sha256
    )
    out_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise Step1Blocked(
            f"run directory already exists: {out_root}. Every capture requires a "
            "never-before-used empty directory."
        ) from exc
    if any(out_root.iterdir()):
        raise Step1Blocked(f"new run directory was unexpectedly non-empty: {out_root}")

    _atomic_json(out_root / "run_identity.json", {
        "schema": "wrf_gpu2.v025.m0.step1_run_identity.v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "directory_was_created_exclusively_and_empty": True,
    })
    if evidence_identity is not None:
        _atomic_json(
            out_root / "evidence_identity.json",
            {"schema": mvs.IDENTITY_SCHEMA, **evidence_identity.binding()},
        )
    outcome: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.step1_capture.v1",
        "run_id": run_id,
        "window": WINDOW_LABEL,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    # ---- canonical authorisation: whitelist + receipt + lock, then SPEND ----
    if receipt_path is None:
        raise Step1Blocked("no coordination receipt: refusing to open a GPU window")
    authorise = authorise or arm.authorise
    ledger_env = os.environ.get("GPUWRF_STEP1_LEDGER")
    ledger_path = ledger_hint or (Path(ledger_env) if ledger_env else None)
    authorisation = authorise(WINDOW_LABEL, receipt_path=receipt_path, env=env, consume=True,
                              ledger_path=ledger_path)
    outcome["authorisation"] = {
        "window": authorisation["window"],
        "receipt_fingerprint": authorisation.get("receipt_fingerprint"),
        "consumed": authorisation.get("consumed"),
        "lock": authorisation.get("lock"),
    }

    stage_env = dict(env)
    applied_environment = stage_environment(
        out_root=out_root,
        evidence_identity=evidence_identity if profiled else None,
        prepared_cache_dir=prepared_cache_dir,
    )
    stage_env.update(applied_environment)
    outcome["environment_applied"] = applied_environment
    outcome["evidence_hook"] = {
        "enabled": evidence_identity is not None and profiled,
        "default_off": evidence_identity is None or not profiled,
        "identity_path": (
            str(out_root / "evidence_identity.json")
            if evidence_identity is not None
            else None
        ),
    }
    outcome["prepared_cache"] = (
        {
            **prepared_cache,
            "path": str(prepared_cache_dir),
            "validated_before_receipt_spend": True,
        }
        if prepared_cache is not None
        else {
            "status": "fresh-per-run-cache",
            "path": str(out_root / "cache"),
        }
    )

    forecast_command = forecast_arm_command(out_root=out_root)
    command = (
        list(stub_stage)
        if stub_stage
        else (
            nsys_command(forecast_command, out_root=out_root)
            if profiled
            else forecast_command
        )
    )
    outcome["command"] = command
    outcome["capture_kind"] = (
        "stub"
        if stub_stage
        else ("production" if profiled else "production-clean-pair")
    )

    guard = guard or arm.PreemptionGuard()
    started = time.monotonic()
    killed_reason: str | None = None
    log_path = out_root / "capture.log"
    log_stream = log_path.open("x")
    sampler = None
    if evidence_identity is not None and profiled:
        sampler_factory = sampler_factory or mvs.LockOwnerResidencySampler
        sampler = sampler_factory(
            identity=evidence_identity,
            allocator_sidecar=out_root / "forecast_allocator.json",
            output_sidecar=out_root / "lock_owner_total_residency.json",
            stderr_path=out_root / "residency_sampler.log",
        )
        try:
            sampler.start()
        except Exception:
            log_stream.close()
            raise
    try:
        proc = subprocess.Popen(
            command, cwd=REPO, env=stage_env, start_new_session=True,
            stdout=log_stream, stderr=subprocess.STDOUT, text=True,
        )
    except Exception:
        if sampler is not None:
            sampler.abort()
        log_stream.close()
        raise
    pgid = os.getpgid(proc.pid)
    if sampler is not None:
        try:
            sampler.attach(root_pid=proc.pid, process_group_id=pgid)
        except Exception:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=KILL_GRACE_S)
                    break
                except subprocess.TimeoutExpired:
                    continue
            sampler.abort()
            log_stream.close()
            raise

    def kill_group(reason: str) -> None:
        nonlocal killed_reason
        killed_reason = reason
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=KILL_GRACE_S)
                return
            except subprocess.TimeoutExpired:
                continue

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda s, _f: kill_group(f"signal {signal.Signals(s).name}"))

    try:
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            elapsed = time.monotonic() - started
            if elapsed > timeout_s:
                kill_group(f"stage budget {timeout_s:.0f}s exceeded")
                break
            try:
                guard.check()
            except Exception as exc:  # noqa: BLE001 - any preemption signal
                kill_group(f"manager preemption: {exc}")
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if proc.poll() is None:
            kill_group(killed_reason or "capture exiting with the child still alive")
        log_stream.flush()
        os.fsync(log_stream.fileno())
        log_stream.close()

    sampler_error = None
    sampler_payload = None
    if sampler is not None:
        if proc.returncode == 0 and killed_reason is None:
            try:
                sampler_payload = sampler.finish()
            except Exception as exc:  # noqa: BLE001 - evidence failure blocks capture
                sampler_error = f"{type(exc).__name__}: {exc}"
        else:
            sampler.abort()

    elapsed = time.monotonic() - started
    outcome.update({
        "elapsed_seconds": elapsed,
        "returncode": proc.returncode,
        "killed": killed_reason is not None,
        "killed_reason": killed_reason,
        "orphans": _surviving_group(pgid),
        "status": (
            "OK"
            if (
                proc.returncode == 0
                and killed_reason is None
                and sampler_error is None
            )
            else "FAILED"
        ),
        "residency_evidence": {
            "status": (
                "OK"
                if sampler_payload is not None
                else ("FAILED" if sampler_error is not None else "NOT_ENABLED")
            ),
            "error": sampler_error,
        },
        "receipt_remains_spent": True,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "capture_log": {
            "path": str(log_path),
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        },
    })
    artifact_manifest = _capture_artifact_manifest(
        out_root=out_root, run_id=run_id, outcome=outcome
    )
    manifest_path = out_root / "capture_artifacts.json"
    _atomic_json(manifest_path, artifact_manifest)
    outcome["artifact_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    _atomic_json(out_root / "capture_outcome.json", outcome)
    return outcome


def _artifact_entry(path: Path, *, out_root: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(out_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _capture_artifact_manifest(
    *, out_root: Path, run_id: str, outcome: dict[str, Any]
) -> dict[str, Any]:
    """Freeze exactly what this capture produced before analysis begins."""
    artifacts: dict[str, Any] = {}
    role_paths = {
        "source_rep": out_root / "step1_autotune_off.nsys-rep",
        "capture_log": out_root / "capture.log",
        "candidate_coverage": out_root / "candidate_coverage.json",
        "production_reference": out_root / "production_reference.json",
        "evidence_identity": out_root / "evidence_identity.json",
        "forecast_allocator": out_root / "forecast_allocator.json",
        "lock_owner_total_residency": out_root / "lock_owner_total_residency.json",
        "residency_sampler_log": out_root / "residency_sampler.log",
        "pipeline_timing": (
            out_root / "autotune0/proofs/pipeline_run_20260521.json"
        ),
    }
    for role, path in role_paths.items():
        if path.is_file():
            artifacts[role] = _artifact_entry(path, out_root=out_root)

    hlo_candidates = []
    dump_dir = out_root / "dump"
    if dump_dir.is_dir():
        for path in sorted(dump_dir.iterdir()):
            if (
                path.is_file()
                and HLO_STAGE in path.name
                and path.suffix in {".txt", ".hlo"}
            ):
                hlo_candidates.append(_artifact_entry(path, out_root=out_root))

    return {
        "schema": "wrf_gpu2.v025.m0.step1_capture_artifacts.v1",
        "run_id": run_id,
        "capture_status": outcome.get("status"),
        "capture_kind": outcome.get("capture_kind"),
        "artifacts": artifacts,
        "hlo_candidates": hlo_candidates,
        "produced_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "analysis may consume only entries in this manifest and must recheck every hash; "
            "directory globs are not evidence selectors"
        ),
    }


def _surviving_group(pgid: int) -> list[int]:
    """Anything left in the group. A non-empty list is an orphan on the GPU."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return []
    survivors = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) == pgid:
                survivors.append(int(entry.name))
        except (ProcessLookupError, PermissionError):
            continue
    return survivors


# ---------------------------------------------------------------------------- #
# analysis: separate CPU process, after the lock is gone                        #
# ---------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: Any, out_root: Path) -> Any:
    """Replace the run directory with a stable token.

    A reviewer running `--dry-run` with a different `--out-root` must produce a
    byte-identical committed artifact. Without this, an independent review dirties
    the worktree just by verifying -- which is exactly what happened to the
    manager on main `8d27c1b3`.
    """
    root = str(out_root)
    if isinstance(value, str):
        return value.replace(root, "<OUT_ROOT>").replace(str(REPO), "<REPO>")
    if isinstance(value, dict):
        return {k: _normalise(v, out_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v, out_root) for v in value]
    return value


def export_pushpop_csv(rep: Path, csv_path: Path, *, runner=None) -> Path:
    runner = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True,
                                                   check=False, timeout=1800))
    if not rep.is_file():
        raise Step1Blocked(f"no nsys report at {rep}")
    proc = runner(["nsys", "stats", "--report", "nvtx_pushpop_trace", "--format", "csv",
                   "--output", "-", str(rep)])
    if proc.returncode != 0:
        raise Step1Blocked(f"nsys stats failed rc={proc.returncode}: {proc.stderr[-400:]}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(proc.stdout)
    if csv_path.stat().st_size == 0:
        raise Step1Blocked("nsys stats produced an empty trace")
    return csv_path


def _manifest_path(
    entry: dict[str, Any], *, out_root: Path, expected_run_id: str
) -> Path:
    relative = entry.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise Step1Blocked("capture artifact has no relative_path")
    path = (out_root / relative).resolve()
    root = out_root.resolve()
    if path != root and root not in path.parents:
        raise Step1Blocked(f"capture artifact escapes its run directory: {relative}")
    if not path.is_file():
        raise Step1Blocked(f"manifest-selected artifact is absent: {path}")
    actual = sha256_file(path)
    if actual != entry.get("sha256"):
        raise Step1Blocked(
            f"manifest-selected artifact hash changed for run {expected_run_id}: "
            f"{relative}; expected {entry.get('sha256')}, got {actual}"
        )
    return path


def _load_capture_manifest(
    *, out_root: Path, run_id: str, outcome: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = out_root / "capture_artifacts.json"
    if not manifest_path.is_file():
        raise Step1Blocked("no capture_artifacts.json: inputs are not run/hash bound")
    expected_hash = (outcome.get("artifact_manifest") or {}).get("sha256")
    actual_hash = sha256_file(manifest_path)
    if expected_hash != actual_hash:
        raise Step1Blocked(
            f"capture artifact manifest hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != run_id:
        raise Step1Blocked(
            f"capture manifest run ID {manifest.get('run_id')!r} != requested {run_id!r}"
        )
    if manifest.get("capture_kind") != outcome.get("capture_kind"):
        raise Step1Blocked(
            "capture manifest and outcome disagree on production/stub provenance"
        )
    return manifest


def _manifest_artifact(
    manifest: dict[str, Any], role: str, *, out_root: Path, run_id: str,
    required: bool = False,
) -> Path | None:
    entry = (manifest.get("artifacts") or {}).get(role)
    if entry is None:
        if required:
            raise Step1Blocked(f"capture manifest has no required {role!r} artifact")
        return None
    return _manifest_path(entry, out_root=out_root, expected_run_id=run_id)


def _load_manifest_json(
    manifest: dict[str, Any], role: str, *, out_root: Path, run_id: str,
) -> dict[str, Any] | None:
    path = _manifest_artifact(
        manifest, role, out_root=out_root, run_id=run_id, required=False
    )
    return json.loads(path.read_text()) if path is not None else None


def select_hlo_from_manifest(
    manifest: dict[str, Any], module: str, *, out_root: Path, run_id: str
) -> Path:
    candidates = []
    for entry in manifest.get("hlo_candidates") or []:
        relative = str(entry.get("relative_path") or "")
        name = Path(relative).name
        if module in name and HLO_STAGE in name and Path(name).suffix in {".txt", ".hlo"}:
            candidates.append(entry)
    if not candidates:
        raise Step1Blocked(
            f"no manifest-selected {HLO_STAGE!r} HLO for module {module!r}. Refusing to "
            "glob the dump directory or substitute another stage.")
    candidates.sort(
        key=lambda entry: re.sub(
            r"\d+", lambda match: match.group().zfill(8),
            str(entry.get("relative_path", "")),
        )
    )
    return _manifest_path(candidates[0], out_root=out_root, expected_run_id=run_id)


def run_analyse(*, out_root: Path, run_id: str, dry_run: bool = False) -> dict[str, Any]:
    """Consume the capture. Emits total objects or suppresses every verdict."""
    run_id = validate_run_id(run_id)
    record: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.step1_analysis.v1",
        "run_id": run_id,
        "window": WINDOW_LABEL,
        "dry_run": dry_run,
        "execution_status": "COMPLETED",
    }
    try:
        outcome_path = out_root / "capture_outcome.json"
        if not outcome_path.is_file():
            raise Step1Blocked("no capture_outcome.json: the capture did not complete")
        outcome = json.loads(outcome_path.read_text())
        record["capture_outcome"] = outcome
        if outcome.get("run_id") != run_id:
            raise Step1Blocked(
                f"capture run ID {outcome.get('run_id')!r} != requested {run_id!r}"
            )
        if outcome.get("status") != "OK":
            raise Step1Blocked(
                f"capture status {outcome.get('status')!r} "
                f"({outcome.get('killed_reason')}); refusing to derive verdicts")
        if outcome.get("orphans"):
            raise Step1Blocked(f"capture left orphan processes {outcome['orphans']}")
        if outcome.get("capture_kind") not in {"production", "stub"}:
            raise Step1Blocked(
                "capture outcome has no recognised production/stub provenance"
            )
        effective_dry = dry_run or outcome["capture_kind"] != "production"
        record["capture_kind"] = outcome["capture_kind"]
        record["production_evidence_eligible"] = not effective_dry
        capture_manifest = _load_capture_manifest(
            out_root=out_root, run_id=run_id, outcome=outcome
        )
        record["capture_artifact_manifest"] = {
            "path": str(out_root / "capture_artifacts.json"),
            "sha256": sha256_file(out_root / "capture_artifacts.json"),
            "run_id": run_id,
        }

        # ALWAYS invoke the real exporter. Previously only the pushpop trace was
        # exported and the census CSVs came from the stub, so the live path was
        # never exercised and a missing file read as an empty measurement.
        source_rep = _manifest_artifact(
            capture_manifest, "source_rep", out_root=out_root, run_id=run_id, required=True
        )
        source_entry = capture_manifest["artifacts"]["source_rep"]
        export = nex.export_all(
            source_rep, out_root, runner=_export_runner(), run_id=run_id,
            expected_source_sha256=source_entry["sha256"],
            allow_replace=True,
            legacy_dry_path=True,
        )
        record["nsys_export"] = export
        if export["status"] != "OK":
            raise Step1Blocked(
                f"nsys export unusable: {export['required_unusable']} produced no data rows. "
                f"An empty report is an ABSENT measurement, not a measurement of zero.")

        csv_path = out_root / nex.REPORTS["nvtx_pushpop_trace"]
        rows = nx.compute_exclusive(nx.parse_pushpop_trace(csv_path.read_text(errors="replace")))
        if not rows:
            raise Step1Blocked("the trace contains no NVTX ranges")

        dominant = nx.dominant_module(rows)
        if not dominant.get("module"):
            raise Step1Blocked("no dominant module could be identified")
        module = dominant["module"]

        readiness = nx.compile_readiness_seconds(rows)
        if readiness["status"] != "OK":
            raise Step1Blocked(readiness.get("reason", "no compile readiness"))

        record["H1"] = h1.verdict(
            readiness["t_off_seconds"], completed=True,
            t_off_basis=readiness["basis"],
            measurement_kind=readiness["measurement_kind"],
        )
        record["H1"]["t_off_derivation"] = readiness
        record["H1"]["SCOPE"] = (
            "DIAGNOSTIC ONLY. T_off is the nsys-session-origin readiness value: an UPPER bound "
            "on application-relative readiness, because the session origin precedes application "
            "launch and no executable-ready boundary has been mechanically identified. An "
            "overstated T_off understates the removable share, so it is conservative for the "
            "one-sided lower bound -- and for nothing else. It is NOT an exact product "
            "cold-compile time and MUST NOT be used to pass or fail the frozen §5.3 gate.")

        scoped = nx.scoped_pass_seconds(rows, module)
        record["H2"] = aa.h2_verdict(scoped["pass_exclusive_seconds"],
                                     scoped["module_compile_seconds"])
        record["H2"]["scope"] = {"module": module,
                                 "denominator_basis": scoped["denominator_basis"],
                                 "ranges_counted": scoped["ranges_counted"]}
        record["scoped_passes"] = scoped

        hlo_path = select_hlo_from_manifest(
            capture_manifest, module, out_root=out_root, run_id=run_id
        )
        attribution = aa.attribute_hlo(hlo_path.read_text(errors="replace"),
                                       aa.load_family_map())
        record["H3"] = aa.h3_verdict(attribution["per_family_share"],
                                     unattributed_share=attribution["unattributed_share"])
        record["H3"]["scope"] = {"module": module, "hlo_file": str(hlo_path),
                                 "hlo_sha256": sha256_file(hlo_path), "hlo_stage": HLO_STAGE}

        record["phase_attribution"] = nx.attribute(rows)
        window = nx.timestep_window(rows)
        record["timestep_window"] = window
        integration_scope = _integration_scope_from_nvtx(
            window,
            run_id=run_id,
            source_rep_sha256=source_entry["sha256"],
            dry_run=effective_dry,
        )
        record["integration_scope"] = integration_scope
        record["baseline_census"] = _census_from(
            out_root,
            capture_manifest=capture_manifest,
            run_id=run_id,
            integration_scope=integration_scope,
        )
        record["artifact_manifest"] = {
            "pushpop_csv": {"path": str(csv_path), "sha256": sha256_file(csv_path),
                            "bytes": csv_path.stat().st_size},
            "hlo": {"path": str(hlo_path), "sha256": record["H3"]["scope"]["hlo_sha256"]},
            "ranges": len(rows),
            "dominant_module": dominant,
        }
        # A blocked deliverable must not hide behind an OK run. The analysis
        # completing is not the same claim as everything having been measured.
        blocked = [name for name in ("H1", "H2", "H3")
                   if record[name].get("verdict") in {"BLOCKED", "SUPPRESSED"}]
        if record["baseline_census"].get("status") != "OK":
            blocked.append("baseline_census")
        record["deliverables_not_ok"] = blocked
        record["gate_status"] = "OK" if not blocked else "PARTIAL"
        record["status"] = record["gate_status"]
        record["verdicts_suppressed"] = False
        if blocked:
            record["partial_reason"] = (
                f"the analysis ran to completion, but {', '.join(blocked)} could not be "
                f"delivered from this capture. PARTIAL is not OK: each blocked deliverable "
                f"names the measurement it is missing.")
    except (Step1Blocked, nex.ExportFailed, pp.ProfilerParseError,
            json.JSONDecodeError, ValueError) as exc:
        record["gate_status"] = "BLOCKED"
        record["status"] = record["gate_status"]
        record["blocked_reason"] = str(exc)
        record["verdicts_suppressed"] = True
        for key in ("H1", "H2", "H3"):
            record[key] = {"verdict": "SUPPRESSED", "reason": str(exc)}
        record["baseline_census"] = {"status": "SUPPRESSED", "reason": str(exc)}

    suffix = "_dryrun" if dry_run else ""
    PROOFS.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        PROOFS / f"step1_attribution{suffix}.json",
        _normalise(record, out_root),
    )
    return record


def _integration_scope_from_nvtx(
    window: dict[str, Any],
    *,
    run_id: str,
    source_rep_sha256: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Bind the transfer interval to an NVTX range from this capture.

    A separately supplied JSON interval is not accepted: it could describe a
    different run or be a hand-authored scope.  Both NVTX and CUDA trace clocks
    come from the same nsys session, so this construction is mechanically
    placeable in ``cuda_gpu_trace``.
    """
    if window.get("status") != "OK":
        return {
            "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
            "status": "MISSING",
            "run_id": run_id,
            "capture_run_id": run_id,
            "source_rep_sha256": source_rep_sha256,
            "reason": window.get("reason", "no eligible NVTX integration range"),
            "markers": window.get("markers"),
            "dry_run_stub": dry_run,
        }
    start_ns, end_ns = window["window_ns"]
    return {
        "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
        "status": "OK",
        "run_id": run_id,
        "capture_run_id": run_id,
        "source_rep_sha256": source_rep_sha256,
        "production_derived": not dry_run,
        "mechanically_verified": True,
        "boundary_kind": "timestep",
        "boundary_source": "nvtx_pushpop_trace",
        "boundary_marker": window.get("name"),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "candidate_count": window.get("candidates"),
        "selection": window.get("selection"),
        "dry_run_stub": dry_run,
    }


def _census_from(
    out_root: Path,
    *,
    capture_manifest: dict[str, Any],
    run_id: str,
    integration_scope: dict[str, Any],
) -> dict[str, Any]:
    """Build the §9 census from the EXPORTED artifacts.

    `None` and `[]` mean different things here and are kept distinct: a missing
    file is an absent report, an empty file is an empty report, and neither is a
    measurement of zero.
    """
    def rows(report: str, parser):
        path = out_root / nex.REPORTS[report]
        return parser(path.read_text(errors="replace")) if path.is_file() else None

    try:
        cuda_rows = rows("cuda_gpu_trace", pp.parse_cuda_gpu_trace)
    except pp.ProfilerParseError:
        cuda_rows = None

    production_reference = _load_manifest_json(
        capture_manifest, "production_reference", out_root=out_root, run_id=run_id
    )
    candidate_coverage = _load_manifest_json(
        capture_manifest, "candidate_coverage", out_root=out_root, run_id=run_id
    )
    if candidate_coverage is not None:
        candidate_coverage = dict(candidate_coverage)
        candidate_coverage["capture_run_id"] = run_id
        candidate_coverage["source_rep_sha256"] = (
            capture_manifest.get("artifacts", {}).get("source_rep", {}).get("sha256")
        )

    allocator = _load_manifest_json(
        capture_manifest, "forecast_allocator", out_root=out_root, run_id=run_id
    )
    total_residency = _load_manifest_json(
        capture_manifest, "lock_owner_total_residency", out_root=out_root, run_id=run_id
    )
    residency = None
    if allocator is not None or total_residency is not None:
        residency = {
            "schema": "wrf_gpu2.v025.m0.vram_evidence.v1",
            "forecast_allocator": allocator,
            "lock_owner_total_residency": total_residency,
        }

    return bc.build(
        kernels=rows("cuda_gpu_kern_sum", pp.parse_nsys_kernel_summary),
        mem_rows=cuda_rows,
        integration_scope=integration_scope,
        production_reference=production_reference,
        candidate_coverage=candidate_coverage,
        # Both require measurements this window does not take. Declared MISSING
        # rather than approximated: a matched profiled/unprofiled pair needs a
        # second run, and peak residency needs a concurrent-allocation series.
        matched_profiler=None,
        residency=residency,
    )


def _export_runner():
    """Real `nsys stats`, or the dry harness's canned stand-in.

    The stub replaces only the `nsys` PROCESS. Command construction, file
    writing, hashing, empty-detection and the required-report gate all remain the
    production code path, so the dry run exercises the exporter rather than
    bypassing it.
    """
    canned = os.environ.get("GPUWRF_STEP1_STUB_EXPORT")
    if not canned:
        return None
    root = Path(canned)

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess:
        argv = list(command)
        if argv == ["nsys", "--version"]:
            return subprocess.CompletedProcess(
                argv, 0, "NVIDIA Nsight Systems version CPU-DRY-FIXTURE\n", ""
            )
        if "--force-export=true" in argv:
            # Exercise the production one-export/many-report boundary, including
            # its required private SQLite proof.  The canned runner replaces
            # only the nsys process, so it must reproduce the filesystem effect
            # of the one forced export as well as the report stdout.
            import sqlite3

            sqlite_path = Path(argv[argv.index("--sqlite") + 1])
            sqlite_path.unlink(missing_ok=True)
            with sqlite3.connect(sqlite_path) as database:
                database.execute(
                    "CREATE TABLE dry_fixture ("
                    "source TEXT NOT NULL, device_action INTEGER NOT NULL)"
                )
                database.execute(
                    "INSERT INTO dry_fixture VALUES (?, ?)",
                    ("CPU-DRY-FIXTURE", 0),
                )
        report = argv[argv.index("--report") + 1]
        path = root / f"{report}.csv"
        text = path.read_text() if path.is_file() else ""
        return subprocess.CompletedProcess(argv, 0, text, "")

    return runner


# ---------------------------------------------------------------------------- #
# orchestrate: lock wrapper around capture, then a separate analysis            #
# ---------------------------------------------------------------------------- #
def run_orchestrate(
    *,
    out_root: Path,
    run_id: str,
    receipt_path: Path | None = None,
    evidence_identity_path: Path | None = None,
    prepared_cache_dir: Path | None = None,
    prepared_cache_sha256: str | None = None,
    dry_run: bool = False,
    wrapper_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    analysis_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    base_env: dict[str, str] | None = None,
    capture_env: dict[str, str] | None = None,
    lock_wrapper: Path = LOCK_WRAPPER,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    try:
        stale_pair.reject_stale_prepared_pair(
            run_id=run_id,
            identity_path=evidence_identity_path,
            prepared_cache_sha256=prepared_cache_sha256,
        )
    except stale_pair.StalePairError as exc:
        raise Step1Blocked(str(exc)) from exc
    if not dry_run:
        if evidence_identity_path is None:
            raise Step1Blocked(
                "production orchestration requires --evidence-identity before lock acquisition"
            )
        evidence_identity = mvs.load_identity(
            evidence_identity_path, expected_run_id=run_id
        )
        mvs.validate_identity_against_workload(
            evidence_identity,
            source_root=REPO / "src/gpuwrf",
            run_dir=Path(FAST_CASE["run_dir"]),
            hours=int(FAST_CASE["hours"]),
        )
        validate_prepared_cache(prepared_cache_dir, prepared_cache_sha256)
    if out_root.exists():
        raise Step1Blocked(
            f"run directory already exists: {out_root}. Orchestration requires a unique "
            "never-before-used target."
        )
    out_root.parent.mkdir(parents=True, exist_ok=True)
    rw.validate_budget(
        [rw.Stage(name="capture", command=["true"], timeout_seconds=BUDGET_CAPTURE_S)],
        deadline_seconds=DEADLINE_S, overhead_seconds=OVERHEAD_S)

    analysis_runner = analysis_runner or (
        lambda cmd, env: subprocess.run(list(cmd), cwd=REPO, env=env, capture_output=True,
                                        text=True, timeout=DEADLINE_S))

    record: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.step1_orchestration.v1",
        "run_id": run_id,
        "window": WINDOW_LABEL,
        "dry_run": dry_run,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    locked = locked_capture_command(
        out_root=out_root, receipt=receipt_path or Path("NONE"), run_id=run_id,
        evidence_identity=evidence_identity_path,
        prepared_cache_dir=prepared_cache_dir,
        prepared_cache_sha256=prepared_cache_sha256,
        lock_wrapper=lock_wrapper,
    )
    record["locked_capture_command"] = " ".join(shlex.quote(c) for c in locked)
    pending_log = out_root.parent / f".{out_root.name}.lock-wrapper.log"
    if pending_log.exists():
        raise Step1Blocked(f"unique wrapper log target already exists: {pending_log}")
    if wrapper_runner is None:
        wrapper = _group_runner(
            locked, log_path=pending_log, env=capture_env
        )
    else:
        wrapper = wrapper_runner(locked)
        # Injected test runners do not own the production log path. Persist their
        # bounded output atomically so the schema remains the same.
        nex._atomic_write_text(pending_log, (wrapper.stdout or "") + (wrapper.stderr or ""))

    final_wrapper_log = (
        out_root / "lock_wrapper.log"
        if out_root.is_dir()
        else out_root.parent / f"{out_root.name}.lock-wrapper.failed.log"
    )
    os.replace(pending_log, final_wrapper_log)
    wrapper_log = {
        "path": str(final_wrapper_log),
        "bytes": final_wrapper_log.stat().st_size,
        "sha256": sha256_file(final_wrapper_log),
    }
    record["lock_wrapper"] = {
        "returncode": wrapper.returncode,
        "returned": True,
        "log_tail": (wrapper.stdout or "")[-1500:],
        "log": wrapper_log,
    }
    # The ONLY evidence of release: the wrapper process returned. Nothing here
    # asserts release on its own authority; a wrapper that never returns yields
    # no release record at all, because this line is never reached.
    record["gpu_released"] = {
        "released": True,
        "evidence": "scripts/with_gpu_lock.sh exited, which releases the flock it held",
        "wrapper_returncode": wrapper.returncode,
    }

    if wrapper.returncode != 0:
        record["execution_status"] = "FAILED"
        record["gate_status"] = "BLOCKED"
        record["status"] = "BLOCKED"
        record["blocked_reason"] = f"locked capture failed rc={wrapper.returncode}"
        record["analysis"] = {"status": "NOT_RUN"}
        _write_orchestration(record, out_root, dry_run)
        return record

    analysis = analysis_runner(analyse_command(out_root=out_root, run_id=run_id)
                               + (["--mark-dry"] if dry_run else []),
                               analysis_environment(base_env))
    analysis_path = PROOFS / (
        "step1_attribution_dryrun.json" if dry_run else "step1_attribution.json"
    )
    analysis_object = None
    analysis_parse_error = None
    try:
        analysis_object = json.loads(analysis_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        analysis_parse_error = str(exc)
    record["analysis"] = {"returncode": analysis.returncode,
                          "command": " ".join(analyse_command(
                              out_root=out_root, run_id=run_id)),
                          "ran_after_release": True,
                          "environment_sanitised_to_cpu": True,
                          "log_tail": ((analysis.stdout or "") + (analysis.stderr or ""))[-1500:],
                          "artifact": str(analysis_path),
                          "artifact_parse_error": analysis_parse_error}
    if analysis_object is None or analysis_object.get("run_id") != run_id:
        record["execution_status"] = "FAILED"
        record["gate_status"] = "BLOCKED"
        record["status"] = "BLOCKED"
        record["blocked_reason"] = (
            "analysis did not emit a parseable artifact bound to this run ID"
        )
    else:
        gate_status = analysis_object.get("gate_status") or analysis_object.get("status")
        expected_rc = 0 if gate_status == "OK" else 1
        if analysis.returncode != expected_rc:
            record["execution_status"] = "FAILED"
            record["gate_status"] = "BLOCKED"
            record["status"] = "BLOCKED"
            record["blocked_reason"] = (
                f"analysis rc/schema disagreement: gate_status={gate_status!r}, "
                f"returncode={analysis.returncode}, expected={expected_rc}"
            )
        else:
            record["execution_status"] = "COMPLETED"
            record["gate_status"] = gate_status
            record["status"] = gate_status
            record["analysis"]["gate_status"] = gate_status
    _write_orchestration(record, out_root, dry_run)
    return record


def _group_runner(command: Sequence[str],
                  timeout_s: float = DEADLINE_S,
                  *,
                  log_path: Path | None = None,
                  env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the lock wrapper in its OWN process group and kill the group on exit.

    `subprocess.run(timeout=...)` kills only the direct child. Here the direct
    child is `with_gpu_lock.sh`; killing it would leave the capture -- and the
    forecast under it -- alive on the GPU with nobody holding the lock. Exactly
    the orphan the inner boundary already guards against, one level up.
    """
    if log_path is None:
        fd, temporary = tempfile.mkstemp(prefix="v025-step1-group-", suffix=".log")
        os.close(fd)
        log_path = Path(temporary)
        log_mode = "w"
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "x"
    log_stream = log_path.open(log_mode)
    proc = subprocess.Popen(
        list(command), cwd=REPO, env=env, start_new_session=True,
        stdout=log_stream, stderr=subprocess.STDOUT, text=True,
    )
    pgid = os.getpgid(proc.pid)
    killed: str | None = None

    def kill_group(reason: str) -> None:
        nonlocal killed
        killed = reason
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=KILL_GRACE_S)
                return
            except subprocess.TimeoutExpired:
                continue

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda s, _f: kill_group(f"signal {signal.Signals(s).name}"))

    started = time.monotonic()
    try:
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() - started > timeout_s:
                kill_group(f"window deadline {timeout_s:.0f}s exceeded")
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if proc.poll() is None:
            kill_group("orchestrator exiting with the wrapper still alive")
        log_stream.flush()
        os.fsync(log_stream.fileno())
        log_stream.close()

    text = log_path.read_text(errors="replace")
    if killed:
        suffix = f"\n[orchestrator] killed the wrapper process group: {killed}\n"
        with log_path.open("a") as stream:
            stream.write(suffix)
        text += suffix
    returncode = proc.returncode if proc.returncode is not None else 1
    result = subprocess.CompletedProcess(list(command), returncode, text, "")
    result.log_path = str(log_path)
    result.log_sha256 = sha256_file(log_path)
    return result


def _write_orchestration(record: dict[str, Any], out_root: Path, dry_run: bool) -> None:
    PROOFS.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if dry_run else ""
    _atomic_json(
        PROOFS / f"step1_orchestration{suffix}.json",
        _normalise(record, out_root),
    )


def profiler_matched_pair_plan() -> dict[str, Any]:
    """Pre-register the separate pair; do not pretend it fits this window."""
    cached_load_cap_s = 60.0
    integration_cap_s = 300.0
    unprofiled_arm_cap_s = cached_load_cap_s + integration_cap_s
    total_cap_s = BUDGET_CAPTURE_S + OVERHEAD_S + unprofiled_arm_cap_s
    return {
        "schema": "wrf_gpu2.v025.m0.profiler_matched_pair_plan.v1",
        "status": "PREREGISTERED_SEPARATE_WINDOW",
        "evidence_status": "MISSING_NOT_RUN",
        "current_window_eligible": False,
        "current_window_deadline_seconds": DEADLINE_S,
        "sum_of_frozen_caps_seconds": total_cap_s,
        "calculation": {
            "profiled_capture_cap_seconds": BUDGET_CAPTURE_S,
            "fixed_overhead_seconds": OVERHEAD_S,
            "cached_load_cap_seconds": cached_load_cap_s,
            "unprofiled_integration_cap_seconds": integration_cap_s,
        },
        "reason": (
            f"{BUDGET_CAPTURE_S:.0f}+{OVERHEAD_S:.0f}+{unprofiled_arm_cap_s:.0f}"
            f"={total_cap_s:.0f}s exceeds the current {DEADLINE_S:.0f}s cap. "
            "Profiler perturbation remains MISSING here."
        ),
        "required_identity": [
            "distinct run IDs", "same workload_identity_sha256",
            "same integration_scope_sha256", "same event_mix_sha256",
            "same source/config/input hashes", "same physical GPU UUID",
            "byte-identical prepared-cache snapshots", "integration-only clocks",
            "recorded arm order",
        ],
        "arms": {
            "profiled": {
                "instrumentation": "nsys",
                "evidence_hook": "enabled",
                "clock_source": (
                    "allocator monotonic enclosure and pipeline wall_clock_per_hour_s[0]"
                ),
                "integration_only_gate_eligible": False,
                "resource_sidecars": "required",
            },
            "unprofiled": {
                "instrumentation": "none",
                "evidence_hook": "default-off/unset",
                "clock_source": "pipeline wall_clock_per_hour_s[0]",
                "integration_only_gate_eligible": False,
                "resource_sidecars": "forbidden",
            },
        },
        "fail_closed": [
            "either arm nonzero or missing pipeline payload",
            "identity/cache/event mismatch",
            "any GPUWRF_M0_* evidence variable in the clean arm",
            "same run ID or reordered/unrecorded execution",
        ],
        "allowed_next_path": (
            "the separately coordinated reproduction window, or a new real "
            "production-derived short pair whose identity/event coverage and sum-of-caps "
            "budget are accepted before execution"
        ),
    }


def vram_measurement_plan() -> dict[str, Any]:
    return {
        "schema": "wrf_gpu2.v025.m0.vram_measurement_plan.v1",
        "status": "IMPLEMENTED_CPU_PROVEN_NOT_RUN",
        "evidence_status": "MISSING_UNTIL_MANAGER_EXECUTOR_RUNS",
        "product_metric": (
            "baseline-subtracted peak total device residency sampled by the lock owner"
        ),
        "decomposition": (
            "forecast-process peak_bytes_in_use and peak_bytes_reserved sidecar"
        ),
        "bindings": [
            "run_id", "forecast_pid", "source_sha256",
            "config_sha256", "input_manifest_sha256", "timestamps",
        ],
        "sampler_requirements": [
            "lock_owner_parent role", "declared cadence", "sampling misses",
            "absolute and baseline-subtracted peaks", "unexpected competing contexts",
        ],
        "implementation": {
            "forecast_hook": "src/gpuwrf/runtime/operational_mode.py",
            "external_sampler": "scripts/v025/m0_vram_sampler.py",
            "capture_parent": "scripts/v025/step1_driver.py",
            "cadence_ms": mvs.DEFAULT_CADENCE_MS,
            "telemetry_process_model": (
                "one-shot prelaunch baseline plus persistent external streams"
            ),
        },
        "forbidden": (
            "the capture/orchestration parent must not import JAX or query allocator "
            "statistics; doing so would inspect or initialize the wrong allocator"
        ),
    }


def plan(out_root: Path | None = None, run_id: str | None = None) -> dict[str, Any]:
    run_id = validate_run_id(run_id or new_run_id("plan"))
    out_root = out_root or (OUT_BASE / run_id)
    evidence_identity = out_root.parent / f"{run_id}.evidence_identity.json"
    budget = rw.validate_budget(
        [rw.Stage(name="capture", command=["true"], timeout_seconds=BUDGET_CAPTURE_S)],
        deadline_seconds=DEADLINE_S, overhead_seconds=OVERHEAD_S)
    receipt = Path("<RECEIPT>")
    return {
        "window": WINDOW_LABEL,
        "budget": budget,
        "commands": {
            "1_locked_capture": " ".join(shlex.quote(c) for c in locked_capture_command(
                out_root=out_root,
                receipt=receipt,
                run_id=run_id,
                evidence_identity=evidence_identity,
            )),
            "2_analyse_after_release": " ".join(analyse_command(
                out_root=out_root, run_id=run_id)),
        },
        "run_id": run_id,
        "unique_run_directory": str(out_root),
        "evidence_identity": {
            "path": str(evidence_identity),
            "schema": mvs.IDENTITY_SCHEMA,
            "must_exist_and_validate_before_lock": True,
        },
        "environment_applied": stage_environment(out_root=out_root),
        "deliverables": ["H1", "H2", "H3", "baseline_census"],
        "hypotheses_this_window_can_answer": {
            "H1": "SUPPORTED / PARTIAL_SUPPORT / INCONCLUSIVE (never NOT_SUPPORTED), "
                  "and BLOCKED unless T_off is on the launch-to-readiness clock",
            "H2": "SUPPORTED or FALSIFIED, module-scoped exclusive pass seconds",
            "H3": "SUPPORTED or FALSIFIED, one after_optimizations HLO, structural proxy",
        },
        "profiler_matched_pair": profiler_matched_pair_plan(),
        "vram": vram_measurement_plan(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true",
                      help="run orchestrate -> capture -> analyse with CPU stubs")
    mode.add_argument("--orchestrate", action="store_true")
    mode.add_argument("--capture", action="store_true")
    mode.add_argument(
        "--capture-clean",
        action="store_true",
        help="separate matched-pair arm: direct forecast, evidence hook default-off",
    )
    mode.add_argument("--analyse", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--evidence-identity",
        type=Path,
        default=None,
        help="pre-launch hash/physical-device identity for the opt-in evidence arm",
    )
    parser.add_argument(
        "--prepared-cache-dir",
        type=Path,
        default=None,
        help="private writable cache snapshot pre-bound for one matched-pair arm",
    )
    parser.add_argument(
        "--prepared-cache-sha256",
        default=None,
        help="content identity required with --prepared-cache-dir",
    )
    parser.add_argument("--mark-dry", action="store_true",
                        help="label the emitted objects as dry-run artifacts "
                             "(compatible with --analyse; --dry-run is a MODE and is not)")
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id or new_run_id(
        "dryrun" if args.dry_run else "step1"
    ))
    out_root = args.out_root or (OUT_BASE / run_id)

    if args.print_plan:
        print(json.dumps(plan(out_root, run_id), indent=2, sort_keys=True, default=str))
        return 0
    if args.dry_run:
        import step1_dryrun
        record = step1_dryrun.run(out_root.parent, run_id=run_id)
        print(json.dumps({"status": record["status"]}, indent=2))
        return 0 if record["status"] == "OK" else 1
    if args.capture or args.capture_clean:
        # The dry harness passes the stub stage and a temp ledger through the
        # environment, because the lock wrapper sits between it and this process
        # and only environment survives that boundary.
        stub_stage = os.environ.get("GPUWRF_STEP1_STUB_STAGE")
        outcome = run_capture(
            out_root=out_root, run_id=run_id, receipt_path=args.receipt,
            evidence_identity_path=args.evidence_identity,
            profiled=not args.capture_clean,
            prepared_cache_dir=args.prepared_cache_dir,
            prepared_cache_sha256=args.prepared_cache_sha256,
            stub_stage=shlex.split(stub_stage) if stub_stage else None)
        return 0 if outcome["status"] == "OK" else 1
    if args.analyse:
        record = run_analyse(out_root=out_root, run_id=run_id, dry_run=args.mark_dry)
        for key in ("H1", "H2", "H3"):
            print(f"  {key}: {record[key].get('verdict')}")
        print(f"  baseline_census: {record.get('baseline_census', {}).get('status')}")
        return 0 if record["gate_status"] == "OK" else 1
    record = run_orchestrate(
        out_root=out_root,
        run_id=run_id,
        receipt_path=args.receipt,
        evidence_identity_path=args.evidence_identity,
        prepared_cache_dir=args.prepared_cache_dir,
        prepared_cache_sha256=args.prepared_cache_sha256,
    )
    return 0 if record["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
