"""Independent checks for the mechanically derived M0 authority guard set."""

from __future__ import annotations

import hashlib
import json
import ast
import sys
from dataclasses import asdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m0_authority_guard_inventory as inventory  # noqa: E402
import m0_exact_boundary_child as child  # noqa: E402


REVIEW12_OMISSIONS = {
    "stage_name_matches_handoff",
    "run_id_matches_handoff",
    "parent_launch_endpoint_is_integer",
    "parent_launch_endpoint_is_positive",
    "receipt_fingerprint_nonempty",
    "receipt_spent_timestamp_is_iso8601",
    "parent_pid_is_integer",
    "handoff_is_not_a_symlink",
    "handoff_is_valid_json",
    "handoff_is_a_json_object",
}


def test_actual_refusal_sites_equal_the_declared_inventory_both_ways():
    derived = set(inventory.derive_pre_device_guard_inventory(SCRIPTS))
    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    assert derived - declared == set()
    assert declared - derived == set()
    assert len(derived) == 35


def test_review12s_ten_independently_observed_omissions_are_all_derived():
    derived = set(inventory.derive_pre_device_guard_inventory(SCRIPTS))
    assert REVIEW12_OMISSIONS - derived == set()


def test_every_derived_guard_has_one_actual_source_site():
    sites = inventory.derive_pre_device_guard_sites(SCRIPTS)
    assert len(sites) == len({site.guard for site in sites}) == 35
    assert {site.provider for site in sites} == {
        "gpu_window_registry.py",
        "m0_exact_boundary_child.py",
        "run_gpu_arm.py",
    }
    assert sum(site.provider == "run_gpu_arm.py" for site in sites) == 7


def test_guard_inventory_digest_is_recomputed_from_site_provenance():
    sites = inventory.derive_pre_device_guard_sites(SCRIPTS)
    expected = hashlib.sha256(
        json.dumps(
            [asdict(site) for site in sites],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert inventory.guard_inventory_sha256(SCRIPTS) == expected
    imports = {
        alias.name
        for node in ast.walk(
            ast.parse(Path(inventory.__file__).read_text(encoding="utf-8"))
        )
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "jax" or name.startswith("jax.") for name in imports)
    assert not any(
        name == "gpuwrf" or name.startswith("gpuwrf.") for name in imports
    )
