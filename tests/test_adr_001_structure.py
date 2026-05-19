"""Structural test for ADR-001 (M2-S8 contract AC #11)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / ".agent" / "decisions" / "ADR-001-backend-selection.md"

REQUIRED_TOKENS = ("Decision:", "Selected backend:", "Evidence summary", "Dissent")
SELECTED_BACKEND_REGEX = re.compile(
    r"^Selected backend: (jax|triton|gt4py|kokkos|cuda_tile|cupy_or_numba|hybrid:.+|deferred)$",
    re.MULTILINE,
)


def test_adr_exists() -> None:
    assert ADR.exists(), f"ADR-001 missing: {ADR}"


def test_adr_minimum_size() -> None:
    size = ADR.stat().st_size
    assert size >= 2000, f"ADR-001 too short: {size} < 2000 bytes"


def test_adr_contains_required_tokens() -> None:
    text = ADR.read_text(errors="replace")
    missing = [t for t in REQUIRED_TOKENS if t not in text]
    assert not missing, f"ADR-001 missing required tokens: {missing}"


def test_adr_selected_backend_line_matches_contract_regex() -> None:
    text = ADR.read_text(errors="replace")
    match = SELECTED_BACKEND_REGEX.search(text)
    assert match is not None, (
        "ADR-001 has no line matching contract regex "
        "^Selected backend: (jax|triton|gt4py|kokkos|cuda_tile|cupy_or_numba|hybrid:.+|deferred)$"
    )
    backend = match.group(1)
    assert backend in {
        "jax",
        "triton",
        "gt4py",
        "kokkos",
        "cuda_tile",
        "cupy_or_numba",
        "deferred",
    } or backend.startswith("hybrid:"), f"unrecognized backend: {backend!r}"
