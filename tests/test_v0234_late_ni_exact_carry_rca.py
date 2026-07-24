"""CPU/static gates for the late-Ni exact-carry runner."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from scripts.v0234_late_ni_exact_carry_rca import (
    EXACT_LIMIT_STEPS,
    GATE_STEPS,
    canonical_digest,
    namespace_for,
    segment_lengths,
    validate_authorization,
)


RUNNER = Path("scripts/v0234_late_ni_exact_carry_rca.py")
LAUNCHER = Path(
    ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause/"
    "v0234-late-ni-authorized.sh"
)


def test_segment_schedule_hits_gate_and_hard_limit_exactly():
    before = segment_lengths(0, GATE_STEPS["d01"])
    after = segment_lengths(GATE_STEPS["d01"], EXACT_LIMIT_STEPS["d01"])
    assert sum(before) == 1000
    assert sum(after) == 156
    assert max(before + after) <= 67
    assert GATE_STEPS == {"d01": 1000, "d02": 3000, "d03": 9000}
    assert EXACT_LIMIT_STEPS == {"d01": 1156, "d02": 3468, "d03": 10404}


def test_fresh_namespace_is_nonce_and_arm_bound():
    nonce = "a" * 64
    assert namespace_for(nonce, "smoke") == "v0234_gpt_late_ni_aaaaaaaaaaaaaaaa_smoke"
    assert namespace_for(nonce, "exact") == "v0234_gpt_late_ni_aaaaaaaaaaaaaaaa_exact"


def test_canonical_digest_excludes_only_embedded_digest():
    payload = {"schema": "x", "nonce": "a" * 64}
    digest = canonical_digest(payload)
    assert canonical_digest({**payload, "proof_sha256": digest}) == digest


def test_authorization_pins_a_committed_runner_ancestor(monkeypatch):
    committed = b"authorized runner bytes\n"
    runner_sha = hashlib.sha256(committed).hexdigest()
    monkeypatch.setattr(
        "scripts.v0234_late_ni_exact_carry_rca.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "scripts.v0234_late_ni_exact_carry_rca.subprocess.check_output",
        lambda *args, **kwargs: committed,
    )
    monkeypatch.setattr(
        "scripts.v0234_late_ni_exact_carry_rca.sha256_file",
        lambda _path: runner_sha,
    )
    nonce = "a" * 64
    payload = {
        "schema": "gpuwrf.v0234.late-ni-smoke-manager-authorization.v1",
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "arm": "smoke",
        "nonce": nonce,
        "namespace": namespace_for(nonce, "smoke"),
        "lock_label": "v0234-gpt-late-ni-rootcause",
        "authorized_model_processes": 1,
        "baseline_processes_authorized": 0,
        "runner_commit": "b" * 40,
        "runner_sha256": runner_sha,
        "candidate_model_commit": "3b81fb5b093639e70c12cce87d602c45b326b18b",
        "candidate_src_gpuwrf_tree": "e627605f6a8bc0dc23f5c474be4bb532b99297c1",
        "output_materialization": False,
        "existing_cpu_reference_only": True,
        "domain": "d03",
        "native_step_upper_bound": 1,
        "ordinary_native_step_evaluations": 1,
        "recorder_native_step_evaluations": 1,
        "total_native_step_evaluations": 2,
        "ordinary_recorder_identity_required": True,
    }
    payload["proof_sha256"] = canonical_digest(payload)
    assert validate_authorization(payload, arm="smoke") == payload["proof_sha256"]


def test_runner_is_import_safe_and_gpu_actions_stay_below_gpu_entry():
    source = RUNNER.read_text()
    tree = ast.parse(source)
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "jax" not in top_imports
    assert not any(name.startswith("gpuwrf") for name in top_imports | top_from)
    assert "import jax" in source[source.index("def _gpu_run"):]
    assert "scripts/with_gpu_lock.sh" not in source
    assert "nvidia-smi" not in source


def test_exact_path_binds_ordinary_identity_and_never_consumes_recorder():
    source = RUNNER.read_text()
    identity = source.index("full_prefix_comparison = compare(")
    admission = source.index(
        "admissible = bool(identity_admissible and first_prefix_bad is not None)"
    )
    assert identity < admission
    assert "rca.decode_records(" not in source
    assert source.count("advance_chunk_with_corrected_ni_rca.lower(") == 1
    assert '"ordinary_prefix_bisection": True' in source
    assert '"recorder_payload_authorized": False' in source
    assert '"executed": False' in source
    assert '"stop_at_first_red": True' in source
    assert '"output_materialization": False' in source
    assert "production _operational_force" in source
    assert '"runner_head"' not in source


def test_health_inventory_mirrors_guard_and_fails_before_recorder_compile():
    source = RUNNER.read_text()
    assert "_finite_check_candidates" in source
    assert "absent_optional_health_fields" in source
    assert "health_names_by_domain" in source
    assert "carry_health_by_domain" in source
    assert "active boundary health fields differ by domain" not in source
    ordinary_health = source.index("ordinary_health = decode_health(")
    recorder_lower = source.index(
        "lowered = advance_chunk_with_corrected_ni_rca.lower(", ordinary_health
    )
    assert ordinary_health < recorder_lower


def test_launcher_puts_model_process_inside_lock_without_gpu_query():
    source = LAUNCHER.read_text()
    lock = source.index('"$repo/scripts/with_gpu_lock.sh"')
    model = source.index("-m scripts.v0234_late_ni_exact_carry_rca")
    assert lock < model
    assert "nvidia-smi" not in source
    assert "JAX_PLATFORMS=cuda" in source
    assert "GPUWRF_LATE_NI_NONCE=\"$nonce\"" in source
    assert "test -z \"$(git -C \"$repo\" status --porcelain)\"" in source
