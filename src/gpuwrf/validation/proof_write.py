"""Shared gate for writing committed canonical proof artifacts.

Several validation engines (m5 tier1/tier2, KF oracle, the m6b synthetic
dryruns) compute a parity/invariant record AND, historically, dumped it to a
git-tracked canonical proof JSON on every call. The pytest suite calls these
engines to assert on the in-memory record, so every full-suite run re-wrote the
tracked proofs with floating-point / path / timing noise and left the worktree
dirty (a release-hygiene problem -- a clean checkout could not be reproduced
after `pytest`).

This module centralizes one honest rule: the tracked canonical proof is
overwritten ONLY on an explicit regeneration, signalled by either

  * a non-default ``out`` path (the caller routes the dump somewhere else,
    e.g. a pytest ``tmp_path``), or
  * the ``GPUWRF_WRITE_PROOFS`` environment flag being truthy (the
    ``scripts/m5_run_*`` / ``scripts/m6b*`` regenerators set it).

The numeric correctness signal is always the returned record / process exit
code -- never the side-effecting write -- so gating the write does not weaken
any test. It only stops the suite from clobbering committed proofs.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FLAG = "GPUWRF_WRITE_PROOFS"


def proof_writing_enabled() -> bool:
    """True iff the explicit proof-regeneration flag is set truthy."""

    return os.environ.get(_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def should_write_proof(out: Path, canonical: Path) -> bool:
    """Decide whether to write a canonical proof artifact.

    Writes when the regeneration flag is set, OR when ``out`` is not the
    committed canonical path (an explicit redirect, e.g. a tmp path). The
    default suite call -- canonical ``out`` with the flag unset -- returns False.
    """

    if proof_writing_enabled():
        return True
    try:
        return Path(out).resolve() != Path(canonical).resolve()
    except OSError:
        return str(out) != str(canonical)
