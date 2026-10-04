#!/usr/bin/env python3
"""Seal composed WRF-faithful LW+SW d03 land-TSK parity at the night branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
RAW_LW_PROOF = SPRINT / "rrtmg-composed-lw-replay-proof.json"
SW_REAL_WRF_PROOF = SPRINT / "rrtmg-sw-interface-real-wrf-proof.json"
COMPOSED_DELTA = SCRATCH / "rrtmg-composed-surface-delta-v1.npz"
SEALED_LW_DELTA = SCRATCH / "rrtmg-interface-surface-delta-fp32-v1.npz"
CARRY = STAGED / "inputs/last-healthy-d03-step-0.pkl"
WRF_SW_SOURCE = SCRATCH / "pristine-wrf-source/phys/module_ra_rrtmg_sw.F"

EXPECTED = {
    "raw_lw_proof": "5b6f36d6d2d2b60a18de8ab3ee261b0ebf5ed9112bb2d882be695f4164d8f20b",
    "raw_lw_canonical": "2ecffd35ef60a32fd3c33a7452ffc9a35f5a7429fadcf1081364b4f613ebb8e0",
    "raw_lw_head": "1178578722c702a3a16b885bea8c8e0aa35f90fd",
    "sw_real_wrf_proof": "cbbde365b3c429c4decae7ce4fa7c8df710b23bbcf426146c5c46d2288156ecb",
    "sw_real_wrf_canonical": "abfd701cc24922ae9973b302d5bbdcaa25c2a989dfc5080fb862fccf9c746015",
    "composed_delta": "ee3912b6dfe6df578edae765fbaa55c8b2134bb2aa21fdf8b69744e21d0d68ec",
    "sealed_lw_delta": "ee3912b6dfe6df578edae765fbaa55c8b2134bb2aa21fdf8b69744e21d0d68ec",
    "carry": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    "wrf_sw_source": "7f8af1da0ca1d25ce784a917bc68600300a7c569881c57e7de8501cd53496b59",
    "rrtmg_sw_source": "6a51dafcacca1c8b1593f73d0e71d06ed491a30ed437ba4be4d6b09a73bc372a",
    "coupler_source": "6b4a71cba59fc095cc00cbc028417dc8651acb5fc27b73d66800a1a29c15b21b",
}
DELTA_NAMES = ("theta_flux_delta", "qv_flux_delta", "fltv_delta", "t_skin_delta")


class ComposedFailure(RuntimeError):
    """Fail-closed composed proof error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode() if binary else result.stderr
        raise ComposedFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    digest = hashlib.sha256(disk).hexdigest()
    if disk != _git("show", f"HEAD:{relative}", binary=True):
        raise ComposedFailure(f"SOURCE_NOT_HEAD:{relative}")
    if expected_sha is not None and digest != expected_sha:
        raise ComposedFailure(f"SOURCE_HASH:{relative}:{digest}")
    return {
        "path": str(path),
        "sha256": digest,
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _canonical(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _checked_json(path: Path, expected_sha: str, expected_canonical: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha:
        raise ComposedFailure(f"PROOF_HASH:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("canonical_payload_sha256") != expected_canonical:
        raise ComposedFailure(f"PROOF_CANONICAL_PIN:{path}")
    if _canonical(value) != expected_canonical:
        raise ComposedFailure(f"PROOF_CANONICAL:{path}")
    return value


def _delta_identity() -> dict[str, Any]:
    if _sha256(COMPOSED_DELTA) != EXPECTED["composed_delta"]:
        raise ComposedFailure("COMPOSED_DELTA_HASH")
    if _sha256(SEALED_LW_DELTA) != EXPECTED["sealed_lw_delta"]:
        raise ComposedFailure("SEALED_LW_DELTA_HASH")
    records: dict[str, Any] = {}
    with (
        np.load(COMPOSED_DELTA, allow_pickle=False) as composed,
        np.load(SEALED_LW_DELTA, allow_pickle=False) as lw,
    ):
        if set(composed.files) != set(DELTA_NAMES) or set(lw.files) != set(DELTA_NAMES):
            raise ComposedFailure("DELTA_SCHEMA")
        for name in DELTA_NAMES:
            left = np.asarray(composed[name])
            right = np.asarray(lw[name])
            exact = bool(np.array_equal(left, right))
            if not exact:
                raise ComposedFailure(f"NIGHT_SW_DELTA_NOT_ZERO:{name}")
            records[name] = {
                "shape": list(left.shape),
                "dtype": left.dtype.str,
                "bitwise_identical_to_sealed_lw_delta": exact,
                "logical_c_bitpayload_sha256": hashlib.sha256(
                    np.ascontiguousarray(left).tobytes()
                ).hexdigest(),
            }
    return records


def run(output: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ComposedFailure("TRACKED_WORKTREE_NOT_CLEAN")
    raw = _checked_json(
        RAW_LW_PROOF, EXPECTED["raw_lw_proof"], EXPECTED["raw_lw_canonical"]
    )
    sw = _checked_json(
        SW_REAL_WRF_PROOF,
        EXPECTED["sw_real_wrf_proof"],
        EXPECTED["sw_real_wrf_canonical"],
    )
    if not (
        raw.get("passed") is True
        and raw.get("git", {}).get("head") == EXPECTED["raw_lw_head"]
        and raw.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve") is True
        and raw.get("tsk_parity", {}).get("water_bitwise_invariant") is True
        and sw.get("status") == "PASS"
        and sw.get("is_self_compare") is False
        and sw.get("gates", {}).get("real_wrf_swdnb_rms_and_max_strictly_improve")
        is True
    ):
        raise ComposedFailure("UPSTREAM_AUTHORITY")

    if _sha256(CARRY) != EXPECTED["carry"]:
        raise ComposedFailure("CARRY_HASH")
    with CARRY.open("rb") as handle:
        carry = pickle.load(handle)
    if type(carry).__name__ != "OperationalCarry" or carry.noahmp_rad is None:
        raise ComposedFailure("CARRY_SCHEMA")
    soldn, _lwdn, cosz = (np.asarray(value) for value in carry.noahmp_rad)
    night = bool(np.all(cosz <= 0.0))
    soldn_zero = bool(np.count_nonzero(soldn) == 0)
    if not night or not soldn_zero:
        raise ComposedFailure(f"D03_NOT_NIGHT:{night}:{soldn_zero}")

    if _sha256(WRF_SW_SOURCE) != EXPECTED["wrf_sw_source"]:
        raise ComposedFailure("WRF_SW_SOURCE_HASH")
    sw_source = WRF_SW_SOURCE.read_text(encoding="utf-8", errors="strict")
    night_expression = bool(
        re.search(
            r"if\s*\(coszrs\.le\.0\.0\)\s*dorrsw\s*=\s*\.false\.",
            sw_source,
            flags=re.IGNORECASE,
        )
    )
    if not night_expression:
        raise ComposedFailure("WRF_SW_NIGHT_EXPRESSION")

    delta_identity = _delta_identity()
    record = {
        "schema": "v0234-rrtmg-composed-lw-sw-night-d03-tsk-v1",
        "passed": True,
        "verdict": "WRF_COMPOSED_LW_SW_NIGHT_D03_TSK_PROVEN",
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git("status", "--porcelain", "--untracked-files=no"),
        },
        "execution": {
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "raw_lw_noahmp_replays": raw["execution"]["noahmp_replays"],
            "sw_d03_execution": "WRF exact night branch; no SW solver invocation",
        },
        "authority": {
            "script": _tracked(Path(__file__).resolve()),
            "rrtmg_sw_source": _tracked(
                REPO / "src/gpuwrf/physics/rrtmg_sw.py", EXPECTED["rrtmg_sw_source"]
            ),
            "coupler_source": _tracked(
                REPO / "src/gpuwrf/coupling/physics_couplers.py",
                EXPECTED["coupler_source"],
            ),
            "raw_lw_proof": {
                "path": str(RAW_LW_PROOF),
                "sha256": EXPECTED["raw_lw_proof"],
                "canonical_payload_sha256": EXPECTED["raw_lw_canonical"],
            },
            "sw_real_wrf_proof": {
                "path": str(SW_REAL_WRF_PROOF),
                "sha256": EXPECTED["sw_real_wrf_proof"],
                "canonical_payload_sha256": EXPECTED["sw_real_wrf_canonical"],
            },
            "carry": {"path": str(CARRY), "sha256": EXPECTED["carry"]},
            "wrf_sw_source": {
                "path": str(WRF_SW_SOURCE),
                "sha256": EXPECTED["wrf_sw_source"],
            },
        },
        "d03_shortwave_branch": {
            "wrf_expression": "if (coszrs.le.0.0) dorrsw = .false.",
            "source_expression_present": night_expression,
            "cosz_min": float(np.min(cosz)),
            "cosz_max": float(np.max(cosz)),
            "cosz_mean": float(np.mean(cosz)),
            "all_columns_night": night,
            "held_soldn_nonzero_count": int(np.count_nonzero(soldn)),
            "held_soldn_bitwise_zero": soldn_zero,
            "faithful_sw_surface_delta_bitwise_zero": True,
        },
        "composed_surface_delta": {
            "path": str(COMPOSED_DELTA),
            "sha256": EXPECTED["composed_delta"],
            "bitwise_identical_to_lw_delta_because_sw_is_inactive": True,
            "arrays": delta_identity,
        },
        "tsk_parity": {
            "accepted_qml_plus_top_buffer": raw["tsk_parity"]
            ["accepted_qml_plus_top_buffer"],
            "composed_wrf_faithful_lw_plus_sw": raw["tsk_parity"]
            ["candidate_plus_wrf_interfaces"],
            "land_rms_ratio": raw["tsk_parity"]["land_rms_ratio"],
            "land_rms_improvement_fraction": raw["tsk_parity"]
            ["land_rms_improvement_fraction"],
            "land_rms_and_max_strictly_improve": True,
            "water_bitwise_invariant": True,
            "remaining_status": "LAND_TSK_NOT_CLOSED_AFTER_COMPOSED_WRF_RADIATION_INTERFACES",
        },
        "gates": {
            "wrf_lw_real_source_proof": True,
            "wrf_sw_real_source_proof": True,
            "d03_wrf_shortwave_night_branch_exact": True,
            "composed_delta_bitwise_closed": True,
            "land_tsk_rms_and_max_strictly_improve": True,
            "water_bitwise_invariant": True,
            "accepted_pending_fresh_sp2_noise_gate": True,
        },
    }
    record["canonical_payload_sha256"] = _canonical(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        proof = run(args.output.resolve())
        print(
            json.dumps(
                {
                    "passed": proof["passed"],
                    "verdict": proof["verdict"],
                    "night": proof["d03_shortwave_branch"],
                    "land": proof["tsk_parity"]["composed_wrf_faithful_lw_plus_sw"]
                    ["land"],
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
