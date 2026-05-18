from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_backend_is_not_locked_before_bakeoff() -> None:
    text = (ROOT / "ARCHITECTURE_PRINCIPLES.md").read_text(encoding="utf-8").lower()
    assert "bakeoff before locking" in text
    assert "jax" in text and "triton" in text and "gt4py" in text


def test_bitwise_is_not_default_requirement() -> None:
    text = (ROOT / "VALIDATION_STRATEGY.md").read_text(encoding="utf-8").lower()
    assert "no bitwise equality is required" in text
