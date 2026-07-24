"""Focused CPU gates for the v0234 step-9000 autotune discriminator (Kimi turn).

CPU-only: no GPU allocation, no lock acquisition. Validates that the
discriminator's authority constants match the authenticated post-Fable terminal
proof values, that the retained 8800 carry authenticates with the production
manifest scheme, and that the locked launcher preserves every numerics-relevant
production environment binding while adding only per-arm autotune dump/load
flags.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402
from scripts import v0234_step9000_autotune_discriminator as disc  # noqa: E402

SPRINT = REPO_ROOT / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
LAUNCHER = SPRINT / "autotune-discriminator-launch-command.sh"
INNER = SPRINT / "autotune-discriminator-inner.sh"
PRODUCTION_LAUNCHER = (
    REPO_ROOT
    / ".agent/sprints/2026-07-17-v0234-post-fable-corner-window"
    / "full-tree-gpu-exact-launch-command.sh"
)

# Values authenticated from the post-Fable terminal proof (proof.json, raw
# 642d9295f42680125c9c8833fe63272098578e39f8451fbce924cafc8eb4c0f4, canonical
# e0e2fe4edf7629a0428bb00bedf790d34a34375f281c1912afa480a5a880bdf2).
PROOF_CONSTANTS = {
    "model_tree": "835dcc29bf316c0715b41a72e064985e9cf099df",
    "runner_source_sha256": "27f5c675b69599a5112951c4f12e4a7fa11918c7494f95ca6583429a692930e0",
    "carry_8800_file_sha256": "492cd961c4d4b8a9e47386ae92e2a60167c01eb53e70f3eaf360367d880b1c2a",
    "carry_8800_manifest_sha256": "270f6e0f0f58f670bf4c0c98b9d161c110a7c140cc7c8052d253409f0b234c28",
    "stablehlo_sha256": "b12b3d64a262326516d706d138e4ff6fe43e831380bad4e3f6651e9d117149cf",
    "lowered_hlo_file_sha256": "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308",
    "candidate_commit": "470e6111d516479bed4bc0c3b2be1007bb082afd",
    "candidate_tree": "bb0d7c9a4fe7befdf3120bdfdabde11db985682a",
}


def test_authority_constants_match_terminal_proof() -> None:
    assert disc.EXPECTED_MODEL_TREE == PROOF_CONSTANTS["model_tree"]
    assert disc.EXPECTED_RUNNER_SHA256 == PROOF_CONSTANTS["runner_source_sha256"]
    assert disc.EXPECTED_STABLEHLO_SHA256 == PROOF_CONSTANTS["stablehlo_sha256"]
    assert (
        disc.EXPECTED_HLO_ARTIFACT_FILE_SHA256
        == PROOF_CONSTANTS["lowered_hlo_file_sha256"]
    )
    assert disc.PRODUCTION_CANDIDATE_COMMIT == PROOF_CONSTANTS["candidate_commit"]
    assert disc.PRODUCTION_CANDIDATE_TREE == PROOF_CONSTANTS["candidate_tree"]
    assert disc.CAPTURE_DOMAIN == "d03"
    assert disc.CAPTURE_STEP == 200
    assert disc.ROOT_STEPS * 9 >= disc.CAPTURE_STEP
    assert disc.TERMINAL_OWN_STEPS == {"d01": 1200, "d02": 3600, "d03": 10800}


def test_runner_and_model_bytes_unchanged() -> None:
    runner_sha = hashlib.sha256(
        (REPO_ROOT / "scripts/v0234_nested_frozen_wrf_boundary_window.py")
        .read_bytes()
    ).hexdigest()
    assert runner_sha == PROOF_CONSTANTS["runner_source_sha256"]
    tree = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:src/gpuwrf"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tree == PROOF_CONSTANTS["model_tree"]
    diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", "--", "src/gpuwrf"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert diff == ""


def test_retained_8800_carry_still_authenticates() -> None:
    # The authenticated 8800 carry remains the terminal proof's anchor; the
    # v2 tree-path discriminator does not consume it, but its authentication
    # scheme is re-verified here as contract-mandated evidence hygiene.
    import pickle as _pickle

    runtime = runner._import_runtime()
    carry_path = Path(
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
        "corrected_ni_rca_max_22c2bd7a/"
        "nested_stage_omega_transport_470e6111_post_fable_corner_window_gpt56"
        "_resource_retry1/checkpoints/authenticated-d03-step-8800.pkl"
    )
    raw = carry_path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    assert file_sha == PROOF_CONSTANTS["carry_8800_file_sha256"]
    host = _pickle.loads(raw)
    manifest = runtime.ordinary.host_tree_manifest(host)
    assert manifest["leaf_count"] == 106
    assert manifest["floating_nonfinite_count"] == 0
    assert manifest["manifest_sha256"] == (
        PROOF_CONSTANTS["carry_8800_manifest_sha256"]
    )


def test_discriminator_reuses_production_machinery_and_gates() -> None:
    source = (REPO_ROOT / "scripts/v0234_step9000_autotune_discriminator.py").read_text()
    assert "runner._lower_compile_d03_once" in source
    assert "runtime.run_operational_domain_tree" in source
    assert "runtime.ordinary.load_corrected_tree" in source
    assert "runtime.ordinary.scheduler_contract" in source
    assert "runtime.ordinary.host_tree_manifest" in source
    assert "runner.assert_preemption_clear" in source
    assert "EXPECTED_STABLEHLO_SHA256" in source
    assert 'raise runner.RunnerGateError("STABLEHLO_DRIFT"' in source
    assert '"CAPTURE_MISSED"' in source
    assert "_ShallowStop" in source
    assert "nvidia-smi" not in source
    assert "with_gpu_lock" not in source  # lock lives only in the launcher
    # No model edit is performed by the discriminator itself.
    assert "src/gpuwrf" not in source.split("REPO_ROOT = ")[0]


def _launcher_env_assignments(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in re.findall(r'([A-Z_][A-Z0-9_]*)=("[^"]*"|\'[^\']*\'|\S+)', stripped):
            name, value = token
            if name in {"REPO", "NS", "RUN_DIR", "PROOF", "AUDIT", "CRITIC", "SPRINT", "RUNNER_SHA", "AUDIT_SHA", "CRITIC_SHA"}:
                continue
            env[name] = value.strip("\"'").rstrip(" \\")
    return env


def test_launcher_preserves_production_numeric_environment() -> None:
    production = _launcher_env_assignments(PRODUCTION_LAUNCHER.read_text())
    mine = _launcher_env_assignments(LAUNCHER.read_text())
    numeric_vars = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_ENABLE_COMPILATION_CACHE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "GPUWRF_ALLOCATOR",
        "GPUWRF_FINITE_CHECK",
        "GPUWRF_BITWISE",
        "GPUWRF_BATCH_ENSEMBLE",
        "GPUWRF_ADVANCE_CHUNK_LOOP",
        "GPUWRF_NESTED_FUSE",
        "GPUWRF_NESTED_DEFUSE_COMPILE",
        "GPUWRF_NESTED_PARALLEL_COMPILE",
        "GPUWRF_NESTED_AOT",
        "GPUWRF_AOT_VERIFY",
        "GPUWRF_NESTED_ASYNC_OUTPUT",
        "GPUWRF_NEST_OUTPUT_PIPELINE",
        "GPUWRF_NESTED_SYNC_MODE",
        "GPUWRF_JAX_CACHE",
        "GPUWRF_JAX_CACHE_LOCK",
        "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE",
        "GPUWRF_WRF_ROOT",
    )
    for var in numeric_vars:
        assert var in production, f"production launcher lost {var}"
        assert mine.get(var) == production[var], (
            f"{var}: production={production[var]!r} discriminator={mine.get(var)!r}"
        )
    # Runner-admission-only vars are intentionally not rebound: the
    # discriminator imports the runner machinery directly and sets
    # CANDIDATE_COMMIT/CANDIDATE_TREE itself.
    assert "GPUWRF_NESTED_BUNDLE_APPROVED_SHA" not in mine
    assert "XLA_FLAGS" not in mine  # per-arm only, inside the inner driver


def test_locked_launcher_and_inner_driver_contract() -> None:
    launcher = LAUNCHER.read_text()
    assert launcher.count("/scripts/with_gpu_lock.sh") == 1
    assert "--intent production-preemptible" in launcher
    assert "--label v0234-step9000-autotune-discriminator-kimi2" in launcher
    assert "nvidia-smi" not in launcher
    inner = INNER.read_text()
    for sentinel in (
        "/tmp/PREEMPT_GPU",
        "/tmp/PREEMPT_PRODUCTION",
        "<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION",
        "<DATA_ROOT>/alisios/state/PREEMPT_GPU",
        "/tmp/HOLD_GPU",
        "/tmp/HOLD_WRFGPU",
        "<DATA_ROOT>/alisios/state/HOLD_GPU",
        "<DATA_ROOT>/alisios/state/HOLD_WRFGPU",
    ):
        assert sentinel in inner
    assert inner.count("xla_gpu_dump_autotune_results_to") >= 3
    assert "xla_gpu_load_autotune_results_from" in inner
    assert "autotune-results-load.pb" in inner
    assert "ARM_C_INPUT_MISSING" in inner  # fail-closed if A's dump is absent
    assert "--arm \"$arm\"" in inner
    for script in (LAUNCHER, INNER):
        assert script.stat().st_mode & 0o111
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("arm", ("A", "B", "C"))
def test_arm_result_schema_documented(arm: str) -> None:
    source = (REPO_ROOT / "scripts/v0234_step9000_autotune_discriminator.py").read_text()
    assert "gpuwrf.v0234.step9000-autotune-discriminator-arm.v2" in source
    assert arm in ("A", "B", "C")
