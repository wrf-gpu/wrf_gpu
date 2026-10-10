"""v0.3.4: every honest per-scheme qualification note is reachable through classify_scheme.

Regression for a duplicate module-level ``_IMPLEMENTED_NOTES`` in io/scheme_catalog.py: the second
definition silently replaced the first, so km_opt=2 (an unqualified v022 scaffold) and km_opt=3
reported only the generic "Operationally wired into the GPU scan." reason.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gpuwrf.io import scheme_catalog
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

GENERIC = "Operationally wired into the GPU scan."

# (key, code) -> a phrase the scheme's honest caveat must carry
EXPECTED = {
    # km_opt=2/5: refused in v0.3.4 (unqualified, NaN under the release carry; manager on rel034
    # 18:08Z) -> RECOGNIZED_FAIL_CLOSED with that reason (tests/test_namelist_binding.py).
    ("km_opt", 3): "3-D Smagorinsky",
    ("cu_physics", 5): "Grell-3D",
    ("cu_physics", 93): "Grell-Devenyi",
    ("bl_pbl_physics", 9): "CAM-UW",
    # mp=40: demoted to REFERENCE_ONLY by o1-nlbind (CLI cannot build it under the release
    # defaults); its caveats moved into that reason (tests/test_v034_morrison_aero_wiring.py).
    ("sf_surface_physics", 3): "RUC LSM",
}


def test_implemented_notes_defined_once():
    tree = ast.parse(Path(scheme_catalog.__file__).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        names += [t.id for t in targets if isinstance(t, ast.Name)]
    assert names.count("_IMPLEMENTED_NOTES") == 1, names.count("_IMPLEMENTED_NOTES")


@pytest.mark.parametrize("key,code", sorted(EXPECTED))
def test_note_reachable_through_classify_scheme(key, code):
    support = classify_scheme(key, code)
    assert support.status is SupportStatus.IMPLEMENTED
    assert support.reason != GENERIC
    assert EXPECTED[(key, code)] in support.reason


@pytest.mark.parametrize("key,code", sorted(k for k in EXPECTED if k != ("km_opt", 2) and k != ("km_opt", 5)))
def test_v034_notes_state_gpu_qualification_pending(key, code):
    assert "qualification pending" in classify_scheme(key, code).reason
