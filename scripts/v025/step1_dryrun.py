#!/usr/bin/env python3
"""The dry run: outer wrapper stub -> real capture -> observed release -> analysis.

Manager requirement (main `8d27c1b3`): the dry tests must execute the outer
wrapper stub, then the capture, then an *observed* release, then a **separate**
hostile-env CPU analysis process.

What is stubbed is deliberately minimal:

* the **lock wrapper** -- a shell script that exports the lock variables the way
  `with_gpu_lock.sh` does, runs the capture, and exits, so its exit really is the
  release signal;
* the **receipt** -- a fresh two-manager receipt in a temp ledger, so no real
  coordination is consumed;
* the **GPU stage** -- `step1_stub.py`, which writes artifacts in nsys's real
  formats.

Everything else is production code: the same `run_capture` with its real
`PreemptionGuard` and process-group kill, the same authorisation path, and the
analysis launched as a genuinely separate CPU-pinned process under a hostile
`JAX_PLATFORMS=cuda` environment, so the CPU guard is exercised too.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

import step1_driver as drv  # noqa: E402
import step1_stub as stub  # noqa: E402


def write_receipt(path: Path) -> Path:
    """A well-formed two-manager receipt for the frozen window."""
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "window": drv.WINDOW_LABEL,
        "requested_at_utc": now.isoformat(),
        "request_text": "DRY RUN — no real coordination was requested for this receipt.",
        "replies": {
            "0:2": {"affirmative": True, "verbatim": "DRY RUN STUB — not a real reply.",
                    "received_at_utc": now.isoformat()},
            "0:3": {"affirmative": True, "verbatim": "DRY RUN STUB — not a real reply.",
                    "received_at_utc": now.isoformat()},
        },
    }, indent=2) + "\n")
    return path


def write_lock_wrapper(path: Path, holder: Path) -> Path:
    """A stand-in for `with_gpu_lock.sh`: exports the lock env, runs, exits.

    The point is the lifecycle, not the flock: the wrapper's **exit** is what the
    orchestrator reads as release, so the stub must genuinely exit rather than
    have the driver assume anything.
    """
    token = "dryrun-token-0001"
    holder.parent.mkdir(parents=True, exist_ok=True)
    # The holder RECORD format is load-bearing: `check_canonical_lock` looks for
    # `token=<TOKEN>` inside the file, not for the bare token. My first stub wrote
    # the bare token and the production check rejected it -- which is the dry run
    # doing its job, and the reason the stub must mirror `with_gpu_lock.sh`
    # rather than approximate it.
    path.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "label=''; while [[ $# -gt 0 ]]; do case \"$1\" in "
        "--label) label=\"$2\"; shift 2;; --) shift; break;; "
        "*) shift;; esac; done\n"
        f"printf 'holder=%s pid=%s since=%s token=%s cmd=%s\\n' "
        f"\"$label\" \"$$\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" "
        f"'{token}' \"$*\" > {holder}\n"
        f"export GPUWRF_GPU_LOCK_HELD=1 GPUWRF_GPU_LOCK_TOKEN={token} "
        f"GPUWRF_GPU_LOCK_HOLDER_FILE={holder} GPUWRF_GPU_LOCK_LABEL=\"$label\"\n"
        'echo "[stub-lock] acquired"\n'
        'set +e; "$@"; rc=$?; set -e\n'
        # the real wrapper truncates the holder file on release; mirroring it
        # means a post-hoc check can tell held from released.
        f": > {holder}\n"
        'echo "[stub-lock] released"\n'
        "exit $rc\n"
    )
    path.chmod(0o755)
    return path


def run(root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Execute the full three-process path with stubs. Returns the orchestration."""
    root.mkdir(parents=True, exist_ok=True)
    run_id = drv.validate_run_id(run_id or drv.new_run_id("dryrun"))
    out_root = root / run_id
    if out_root.exists():
        raise drv.Step1Blocked(f"dry-run target already exists: {out_root}")
    receipt = write_receipt(root / f"{run_id}.receipt.json")
    ledger = root / f"{run_id}.spend_ledger.json"
    wrapper = write_lock_wrapper(
        root / f"{run_id}.stub_lock.sh", root / f"{run_id}.holder.txt"
    )

    stub_stage = " ".join(stub.stub_stage_command(out_root, run_id=run_id))
    capture_env = dict(os.environ)
    capture_env.update({
        "GPUWRF_STEP1_STUB_STAGE": stub_stage,
        "GPUWRF_STEP1_LEDGER": str(ledger),
    })

    def analysis_runner(command, env):
        # point the exporter's stub runner at the canned tables the stage wrote
        env = dict(env, GPUWRF_STEP1_STUB_EXPORT=str(out_root / "canned_exports"))
        # a genuinely separate process. `env` is what the ORCHESTRATOR sanitised;
        # the hostile base below is what it had to sanitise away.
        return subprocess.run(list(command), cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=drv.DEADLINE_S)

    # HOSTILE base environment: an accelerator platform exported by the caller.
    # The orchestrator must not carry it into the post-release CPU stage.
    hostile = dict(os.environ, JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0")

    record = drv.run_orchestrate(
        out_root=out_root, run_id=run_id, receipt_path=receipt, dry_run=True,
        analysis_runner=analysis_runner, base_env=hostile, capture_env=capture_env,
        lock_wrapper=wrapper,
    )
    record["run_directory"] = str(out_root)
    return record


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/step1_dryrun")
    record = run(root)
    print(json.dumps({"status": record["status"],
                      "released": record["gpu_released"]["released"],
                      "analysis_rc": record["analysis"].get("returncode")}, indent=2))
    return 0 if record["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
