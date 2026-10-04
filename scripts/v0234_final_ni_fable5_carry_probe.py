"""Deterministic all-leaf localization of the v0.23.4 step-200 Ni blocker.

Authenticates the terminal2 failure namespace, then compares the retained
last-healthy step-0 and first-failed step-200 d03 carries leaf by leaf.  The
program is CPU-only, read-only on every retained artifact, and writes one
canonical proof JSON beside the sprint report.

Key predeclared discriminators:

- ``ph`` interior frozen: outside the walked spec ring the step-200 ``ph``
  must be BIT-IDENTICAL to step 0 if and only if the falsified wiring
  (`60659a2e`) discarded the interior ``advance_w`` update.
- Ni non-finite footprint: interior-wide with ring 0 finite predicts the
  observed first index (0, 1, 1); an isolated interior seed would instead
  reproduce the old step-9400 offshore mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np  # noqa: E402

SCHEMA = "gpuwrf.v0234.final-ni-fable5-carry-localization.v1"
NAMESPACE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal2"
)
FAILURE_DIR = NAMESPACE / "science-runtime" / "failure"
BLOCKER = NAMESPACE / "science-runtime" / "full-run-blocker.json"
BLOCKER_PROOF_SHA = "18e7936c95588d98f5100b35edf61591b9bd1835673de3a214590d36b6fa7584"
FAILURE_PROOF_SHA = "f1abd1c6e2c99af4bc2e74003c09d30898cf84c7efa73834d8ef9705f194fadf"
STEP0_SHA = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
STEP200_SHA = "3f439a40d6467d00d2acc5f719696da6a4cf5392572a704f864366cdbfcc0f55"

OUT_DIR = Path(__file__).resolve().parents[1] / (
    ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate() -> dict[str, Any]:
    blocker = json.loads(BLOCKER.read_text())
    if canonical_hash(blocker) != BLOCKER_PROOF_SHA:
        raise RuntimeError("blocker canonical hash mismatch")
    failure = json.loads((FAILURE_DIR / "failure-proof.json").read_text())
    if canonical_hash(failure) != FAILURE_PROOF_SHA:
        raise RuntimeError("failure proof canonical hash mismatch")
    step0 = FAILURE_DIR / "last-healthy-d03-step-0.pkl"
    step200 = FAILURE_DIR / "first-failed-d03-step-200.pkl"
    actual0 = sha256_file(step0)
    actual200 = sha256_file(step200)
    if actual0 != STEP0_SHA or actual200 != STEP200_SHA:
        raise RuntimeError("retained carry SHA mismatch")
    if failure["last_healthy"]["file_sha256"] != STEP0_SHA:
        raise RuntimeError("failure proof does not bind the step-0 carry")
    if failure["first_failed"]["file_sha256"] != STEP200_SHA:
        raise RuntimeError("failure proof does not bind the step-200 carry")
    detail = failure["failure"]["detail"]
    if "field=Ni level=0 step=200" not in detail or "(0, 1, 1)" not in detail:
        raise RuntimeError(f"unexpected failure detail: {detail}")
    return {
        "namespace": str(NAMESPACE),
        "blocker_proof_sha256": BLOCKER_PROOF_SHA,
        "failure_proof_sha256": FAILURE_PROOF_SHA,
        "step0_file_sha256": actual0,
        "step200_file_sha256": actual200,
        "failure_detail": detail,
    }


def ring_masks(ny: int, nx: int, max_ring: int) -> list[np.ndarray]:
    masks = []
    seen = np.zeros((ny, nx), dtype=bool)
    for r in range(max_ring):
        mask = np.zeros((ny, nx), dtype=bool)
        mask[r, :] = True
        mask[ny - 1 - r, :] = True
        mask[:, r] = True
        mask[:, nx - 1 - r] = True
        mask &= ~seen
        seen |= mask
        masks.append(mask)
    return masks


def _leaf_rows(c0: Any, c200: Any) -> list[dict[str, Any]]:
    import dataclasses

    rows: list[dict[str, Any]] = []

    def walk(name: str, a: Any, b: Any) -> None:
        if a is None or b is None:
            return
        if dataclasses.is_dataclass(a) and not isinstance(a, type):
            for field in dataclasses.fields(a):
                walk(f"{name}.{field.name}", getattr(a, field.name), getattr(b, field.name))
            return
        if isinstance(a, (tuple, list)):
            for index, (left, right) in enumerate(zip(a, b)):
                walk(f"{name}[{index}]", left, right)
            return
        if not hasattr(a, "shape") or not hasattr(a, "dtype"):
            return
        left = np.asarray(a)
        right = np.asarray(b)
        if not np.issubdtype(left.dtype, np.floating):
            return
        rows.append(
            {
                "leaf": name,
                "shape": list(left.shape),
                "nonfinite_step200": int(np.sum(~np.isfinite(right))),
                "size": int(right.size),
                "bit_identical": bool(np.array_equal(left, right)),
            }
        )

    # The State pytree is a non-dataclass container: walk its public arrays.
    walk("carry", c0, c200)
    s0, s200 = c0.state, c200.state
    for attr in sorted(dir(s0)):
        if attr.startswith("_"):
            continue
        left = getattr(s0, attr, None)
        right = getattr(s200, attr, None)
        if left is None or callable(left):
            continue
        walk(f"state.{attr}", left, right)
    return rows


def main() -> int:
    authority = authenticate()
    with (FAILURE_DIR / "last-healthy-d03-step-0.pkl").open("rb") as handle:
        c0 = pickle.load(handle)
    with (FAILURE_DIR / "first-failed-d03-step-200.pkl").open("rb") as handle:
        c200 = pickle.load(handle)

    ph0 = np.asarray(c0.state.ph)
    ph200 = np.asarray(c200.state.ph)
    ni200 = np.asarray(c200.state.Ni)
    nz, ny, nx = ph0.shape
    masks = ring_masks(ny, nx, 8)
    interior = np.ones((ny, nx), dtype=bool)
    for mask in masks:
        interior &= ~mask

    ph_rows = []
    for index, mask in enumerate(masks):
        ph_rows.append(
            {
                "ring": index,
                "max_abs_dph": float(np.max(np.abs(ph200 - ph0)[:, mask])),
                "bit_identical": bool(
                    np.array_equal(ph200[:, mask], ph0[:, mask])
                ),
            }
        )
    interior_identical = bool(np.array_equal(ph200[:, interior], ph0[:, interior]))
    beyond_ring0_identical = bool(
        np.array_equal(ph200[:, ~masks[0]], ph0[:, ~masks[0]])
    )

    ni_rows = []
    for index, mask in enumerate(masks[:3]):
        ni_rows.append(
            {
                "ring": index,
                "nonfinite": int(np.sum(~np.isfinite(ni200[:, mask]))),
                "cells": int(mask.sum()) * nz,
            }
        )
    scan = np.argwhere(~np.isfinite(ni200))
    first_bad = scan[np.lexsort((scan[:, 2], scan[:, 1], scan[:, 0]))][0]

    leaves = _leaf_rows(c0, c200)

    verdict_checks = {
        "ph_interior_bit_frozen_over_200_steps": interior_identical,
        "ph_change_confined_to_ring0": beyond_ring0_identical,
        "ni_first_nonfinite_scan_index_is_0_1_1": [int(v) for v in first_bad]
        == [0, 1, 1],
        "ni_ring1_nonfinite_dominant": ni_rows[1]["nonfinite"]
        > ni_rows[1]["cells"] * 0.9,
        "ni_ring0_mostly_finite": ni_rows[0]["nonfinite"]
        < ni_rows[0]["cells"] * 0.01,
    }
    verdict = (
        "INTERIOR_PH_FREEZE_CONFIRMED"
        if all(verdict_checks.values())
        else "LOCALIZATION_INCONCLUSIVE"
    )

    proof = {
        "schema": SCHEMA,
        "authority": authority,
        "grid": {"nz_ph": nz, "ny": ny, "nx": nx},
        "ph_rings": ph_rows,
        "ph_interior_bit_identical": interior_identical,
        "ph_beyond_ring0_bit_identical": beyond_ring0_identical,
        "ni_rings": ni_rows,
        "ni_first_nonfinite_scan_index": [int(v) for v in first_bad],
        "leaf_table": leaves,
        "verdict_checks": verdict_checks,
        "verdict": verdict,
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_carry_probe.py"
        ],
    }
    proof["proof_sha256"] = canonical_hash(proof)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "carry-localization-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={verdict}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0 if verdict == "INTERIOR_PH_FREEZE_CONFIRMED" else 1


if __name__ == "__main__":
    sys.exit(main())
