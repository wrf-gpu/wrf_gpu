#!/usr/bin/env python3
"""Fail-closed registry for the mechanically superseded pre-Amendment-3 pair."""
from __future__ import annotations

import hashlib
from pathlib import Path


STALE_RUN_IDS = frozenset(
    {
        "m0-pair-profiled-20260728-r1",
        "m0-pair-clean-20260728-r1",
    }
)
STALE_IDENTITY_SHA256 = frozenset(
    {
        "a11b9da29dd17a2151c653341c1a8c009aea8218167fa911113c612e4f3905f2",
        "bc794a2b093045ef44c5b23598968a179c7626053f1d065d6d35664bd79d1eb8",
    }
)
STALE_CACHE_SHA256 = frozenset(
    {
        "825980d919b383833d88c80e416c20cb3f47c0e36b434ed65167288d7409c618",
    }
)
STALE_PLAN_SHA256 = frozenset(
    {
        "213d09a6bc703294753c26613cfe93e14c2e6dcee453d92b4531a50fb6ab736e",
    }
)
SUPERSEDED_BY = (
    ".agent/patches/2026-07-28-v025-m0-autotune0-candidate-amendment.md"
)


class StalePairError(RuntimeError):
    """An attempt reused an identity, cache, plan, or command from the stale pair."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_stale_prepared_pair(
    *,
    run_id: str | None = None,
    identity_path: Path | None = None,
    prepared_cache_sha256: str | None = None,
    plan_path: Path | None = None,
) -> dict[str, object]:
    """Reject the old pair before output creation, lock, or receipt consumption."""

    reasons: list[str] = []
    if run_id in STALE_RUN_IDS:
        reasons.append(f"run_id={run_id}")
    if identity_path is not None and identity_path.is_file():
        identity_sha256 = _sha256(identity_path)
        if identity_sha256 in STALE_IDENTITY_SHA256:
            reasons.append(f"identity_sha256={identity_sha256}")
    if prepared_cache_sha256 in STALE_CACHE_SHA256:
        reasons.append(f"prepared_cache_sha256={prepared_cache_sha256}")
    if plan_path is not None and plan_path.is_file():
        plan_sha256 = _sha256(plan_path)
        if plan_sha256 in STALE_PLAN_SHA256:
            reasons.append(f"plan_sha256={plan_sha256}")
    if reasons:
        raise StalePairError(
            "mechanically superseded M0 prepared pair refused before receipt "
            f"consumption: {', '.join(reasons)}; superseded_by={SUPERSEDED_BY}"
        )
    return {
        "status": "CURRENT_OR_UNKNOWN_NOT_STALE",
        "checked_run_id": run_id,
        "identity_checked": identity_path is not None and identity_path.is_file(),
        "cache_identity_checked": prepared_cache_sha256 is not None,
        "plan_checked": plan_path is not None and plan_path.is_file(),
    }
