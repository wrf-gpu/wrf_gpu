"""Committed ADR-036 validator attack matrix.

Each mutation is parseable Python and represents a reviewed bypass class from
the independent gate review.  A mutation survives only if the validator raises
``DiffValidationError``; generic source-byte pinning is the final backstop, not
the sole assertion being exercised.

ADR-038 re-alignment (2026-09-18): the semantic base moved to the adjudicated
ADR-038 default-entry flip (``validator.BASE_PRODUCTION_COMMIT``) and the
immutable candidate to the commit that re-binds the evidence hook to the
segmented default entry. Every attack class is re-verified against that pair.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts" / "v025"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_m0_instrumentation_diff as validator  # noqa: E402


OPERATIONAL_PATH = REPO / validator.EXPECTED_CHANGED_FILE


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"attack anchor is not unique: {old!r}"
    return source.replace(old, new, 1)


def _mutations(source: str) -> dict[str, str]:
    """Return the exact mandatory A1/A2/A3/A4/A6/A11/A12r/A14/A15 attacks."""

    a1 = _replace_once(
        source,
        "configure_jax_x64()\n",
        'configure_jax_x64()\nos.environ.setdefault("GPUWRF_ATTACK", "1")\n',
    )
    a2 = _replace_once(
        source,
        "configure_jax_x64()\n",
        'configure_jax_x64()\nos.environ["GPUWRF_ATTACK"] = "1"\n',
    )
    a3 = _replace_once(
        source,
        "    if evidence_flag is None:\n",
        "    if not evidence_flag:\n",
    )
    a4 = _replace_once(
        source,
        "    if evidence_flag != \"1\":\n",
        "    else:\n"
        "        return _m0_evidence_run(state, namelist, hours)\n"
        "    if evidence_flag != \"1\":\n",
    )
    a6 = _replace_once(
        source,
        "        os.link(temporary_path, path)\n",
        "        os.replace(temporary_path, path)\n",
    )
    a11 = _replace_once(
        source,
        '    if evidence_flag != "1":\n'
        "        raise RuntimeError(\n"
        "            f\"{_M0_EVIDENCE_FLAG} must be unset (default-off) or exactly '1'\"\n"
        "        )\n",
        '    if evidence_flag != "1":\n'
        "        pass\n",
    )
    a12r = _replace_once(
        source,
        "    evidence = _m0_evidence_config_from_env()\n",
        '    getattr(jax, "numpy")\n'
        "    evidence = _m0_evidence_config_from_env()\n",
    )
    a14 = _replace_once(
        source,
        "    evidence = _m0_evidence_config_from_env()\n",
        "    global _THETA_LIMITER_MIN_K\n"
        "    _THETA_LIMITER_MIN_K = 999.0\n"
        "    evidence = _m0_evidence_config_from_env()\n",
    )
    a15 = _replace_once(
        source,
        '_M0_EVIDENCE_RANGE = "GPUWRF_M0_FORECAST_INTEGRATION"\n',
        '_M0_EVIDENCE_RANGE = "GPUWRF_ATTACK_RANGE"\n',
    )
    return {
        "A1_import_time_call": a1,
        "A2_import_time_subscript_assignment": a2,
        "A3_default_branch_truthiness": a3,
        "A4_default_branch_else_hijack": a4,
        "A6_atomic_replace": a6,
        "A11_guard_body_removal": a11,
        "A12r_helper_dynamic_jax_access": a12r,
        "A14_helper_global_write": a14,
        "A15_constant_retarget": a15,
    }


@pytest.mark.parametrize(
    "attack_id",
    [
        "A1_import_time_call",
        "A2_import_time_subscript_assignment",
        "A3_default_branch_truthiness",
        "A4_default_branch_else_hijack",
        "A6_atomic_replace",
        "A11_guard_body_removal",
        "A12r_helper_dynamic_jax_access",
        "A14_helper_global_write",
        "A15_constant_retarget",
    ],
)
def test_mandatory_attack_matrix_rejects_every_parseable_mutation(attack_id):
    base = validator._git_file(validator.BASE_PRODUCTION_COMMIT, validator.EXPECTED_CHANGED_FILE)
    current = validator._git_file(
        validator.CANDIDATE_PRODUCTION_COMMIT, validator.EXPECTED_CHANGED_FILE
    )
    assert validator.validate_sources(base, current)["status"] == "PASS"
    mutated = _mutations(current)[attack_id]
    ast.parse(mutated)

    with pytest.raises(validator.DiffValidationError):
        validator.validate_sources(base, mutated)


def test_validator_pass_is_bound_without_a_self_referential_head_hash(
    tmp_path, monkeypatch
):
    # Replay the retained candidate from real git objects. Production has
    # advanced since this authority was frozen; none of its pins may move.
    frozen = tmp_path / "candidate"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO),
         str(frozen)], check=True,
    )
    subprocess.run(
        ["git", "-C", str(frozen), "checkout", validator.CANDIDATE_PRODUCTION_COMMIT,
         "--", "src/gpuwrf"], check=True,
    )
    monkeypatch.setattr(validator, "REPO", frozen)
    result = validator.validate_repository(
        validator.BASE_PRODUCTION_COMMIT,
        REPO / ".agent" / "decisions" / "ADR-036-v025-m0-evidence-hook.md",
    )
    assert result["status"] == "PASS"
    # ADR-038 re-alignment, re-bound to FINAL main: candidate = final main
    # HEAD (post seam 689beb792, post hook re-bind merge); src tree pinned so
    # any later src/gpuwrf drift fails closed.
    assert result["candidate_production_commit"] == (
        "062c808801a85b178c8a2aa20ec4b82552cf2035"
    )
    assert result["candidate_src_tree"] == (
        "d7f55214f309f61c15f7588401f01665b3d300cc"
    )
    assert result["working_production_source_matches_candidate"] is True
    assert len(result["validator_sha256"]) == 64
    assert "head" not in {key.lower() for key in result}
    operational = frozen / validator.EXPECTED_CHANGED_FILE
    operational.write_text(operational.read_text() + "\n# source drift\n")
    with pytest.raises(validator.DiffValidationError, match="working production source"):
        validator.validate_repository(
            validator.BASE_PRODUCTION_COMMIT,
            REPO / ".agent/decisions/ADR-036-v025-m0-evidence-hook.md",
        )
