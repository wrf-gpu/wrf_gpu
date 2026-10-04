"""CPU-only falsification suite for the Review-10 fallback assembly."""

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_window_registry as registry  # noqa: E402
import build_m0_review10_fallback_evidence as evidence_builder  # noqa: E402
import m0_cpu_boundary_preflight as cpu_preflight  # noqa: E402
import m0_exact_boundary_child as exact  # noqa: E402
import m0_review10_fallback_capability as capability  # noqa: E402
import run_gpu_arm as arm  # noqa: E402
import wrf_source_authority as wsa  # noqa: E402


FAST = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
CPU_ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def _stablehlo_segment(index: int, count: int, start: int) -> str:
    return f"""
    %iota_{index} = stablehlo.iota dim = 0 : tensor<{count}xi32>
    %base_scalar_{index} = stablehlo.constant dense<{start}> : tensor<i32>
    %base_vector_{index} = stablehlo.broadcast_in_dim %base_scalar_{index}, dims = [] : (tensor<i32>) -> tensor<{count}xi32>
    %indices_{index} = stablehlo.add %base_vector_{index}, %iota_{index} : tensor<{count}xi32>
    %result_{index}:2 = stablehlo.while(
      %iterArg_{index} = %zero, %xs_{index} = %indices_{index}
    ) : tensor<i32>, tensor<{count}xi32>
    cond {{
      %bound_{index} = stablehlo.constant dense<{count}> : tensor<i32>
      %pred_{index} = stablehlo.compare LT, %iterArg_{index}, %bound_{index}
      stablehlo.return %pred_{index} : tensor<i1>
    }} do {{
      stablehlo.return %iterArg_{index}, %xs_{index}
    }}
"""


def _stablehlo_text(counts=(179, 1, 179, 1)) -> str:
    segments = ""
    start = 1
    for index, count in enumerate(counts):
        segments += _stablehlo_segment(index, count, start)
        start += count
    return f"""
module {{
  func.func public @main() {{
    %zero = stablehlo.constant dense<0> : tensor<i32>
{segments}
    return
  }}
  func.func private @scan_body() {{
    return
  }}
}}
"""


def _boundary() -> dict:
    stablehlo_sha = "7" * 64
    observations = dict(capability.EXPECTED_BOUNDARY_OBSERVATIONS)
    payload = {
        "schema": "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1",
        "status": "PASS",
        "device_action": False,
        "platforms": ["cpu"],
        "authority": {"authority_sha256": "8" * 64},
        "native_bundle": {"field_count": 29},
        "observations": observations,
        "lowered_program": {
            "sha256": stablehlo_sha,
            "integration_trip_count": {
                "status": "PASS",
                "method": "exact-lowered-trip-count",
                "entry_function": "main",
                "loop_count": 4,
                "segment_trip_counts": [179, 1, 179, 1],
                "steps": 360,
                "segments": [],
                "stablehlo_sha256": stablehlo_sha,
                "configured_cross_check": {
                    "derivation":
                        "Fraction(str(hours))*3600/Fraction(str(dt_s))",
                    "hours": 1.0,
                    "forecast_interval_seconds": 3600.0,
                    "timestep_seconds": 10.0,
                    "exact_steps": 360,
                    "integral": True,
                },
                "count_matches_configuration": True,
            },
        },
    }
    payload["boundary_sha256"] = capability.canonical_sha256(payload)
    return payload


def _write_boundary(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "cpu_real_boundary_preflight.json"
    path.write_text(
        json.dumps(payload or _boundary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _resign(payload: dict) -> dict:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"content_address", "signature"}
    }
    digest = capability.canonical_sha256(unsigned)
    payload["content_address"] = {"algorithm": "sha256", "sha256": digest}
    payload["signature"] = {
        "scheme": "sha256-canonical-json-content-signature",
        "signed_sha256": digest,
    }
    return payload


def _receipt() -> arm.CoordinationReceipt:
    stamp = datetime.now(timezone.utc).isoformat()
    return arm.CoordinationReceipt(
        window=registry.SESSION_LABEL,
        requested_at_utc=stamp,
        replies={
            manager: {
                "affirmative": True,
                "verbatim": f"{manager} CPU fixture affirmative",
                "received_at_utc": stamp,
            }
            for manager in arm.REQUIRED_MANAGERS
        },
    )


def _lock_environment(
    tmp_path: Path,
    *,
    environment_token: str = "exact-token",
    holder_token: str = "exact-token",
    holder_label: str = registry.SESSION_LABEL,
    exported_label: str | None = registry.SESSION_LABEL,
    command: str = "cpu-only",
) -> dict[str, str]:
    holder = tmp_path / "holder"
    holder.write_text(
        f"holder={holder_label} pid=1 token={holder_token} cmd={command}\n",
        encoding="utf-8",
    )
    environment = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_TOKEN": environment_token,
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
    }
    if exported_label is not None:
        environment["GPUWRF_GPU_LOCK_LABEL"] = exported_label
    return environment


def test_exact_lowered_trip_count_is_mechanical_and_not_configured_observed():
    observed = exact.extract_exact_lowered_trip_count(_stablehlo_text())
    assert observed["method"] == "exact-lowered-trip-count"
    assert observed["segment_trip_counts"] == [179, 1, 179, 1]
    assert observed["steps"] == 360
    assert observed["loop_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    ["missing_main", "bound_mismatch", "uncarried_iota", "extra_while"],
)
def test_exact_lowered_trip_count_mutations_fail_closed(mutation):
    text = _stablehlo_text()
    if mutation == "missing_main":
        text = text.replace("public @main", "private @wrong")
    elif mutation == "bound_mismatch":
        text = text.replace("dense<179> : tensor<i32>", "dense<178> : tensor<i32>", 1)
    elif mutation == "uncarried_iota":
        text = text.replace("%xs_0 = %indices_0", "%xs_0 = %other")
    else:
        text = text.replace(
            "\n    return\n  }\n  func.func private",
            "\n    %extra = stablehlo.while() cond { } do { }\n"
            "    return\n  }\n  func.func private",
        )
    with pytest.raises(exact.BoundaryRefusal, match="exact-lowered-trip-count"):
        exact.extract_exact_lowered_trip_count(text)


@pytest.mark.parametrize(
    "field,value",
    [
        ("platforms", ["gpu"]),
        ("native_bundle", {"field_count": 28}),
        ("native_loader_calls", 0),
        ("exact_lower_calls", 0),
        ("compile_calls", 1),
        ("device_invocations", 1),
    ],
)
def test_r1_boundary_mutations_fail_closed(field, value):
    boundary = _boundary()
    if field in boundary["observations"]:
        boundary["observations"][field] = value
    else:
        boundary[field] = value
    with pytest.raises(capability.CapabilityRefusal):
        capability.validate_real_boundary(boundary)


def test_r1_source_authority_hash_and_import_order_mutations_fail_closed(
    tmp_path, monkeypatch
):
    environment = {
        variable: str(wsa.CANONICAL_ROOT)
        for variable in wsa.ROOT_ENV_VARS
    }
    authority = wsa.build_source_authority(
        namelist_path=FAST / "namelist.input", environ=environment
    )
    mutated = copy.deepcopy(authority)
    mutated["files"][0]["sha256"] = "0" * 64
    with pytest.raises(wsa.SourceAuthorityRefusal, match="hash"):
        wsa.validate_source_authority(mutated, environ=environment)

    path = tmp_path / "authority.json"
    wsa.write_authority(path, authority)
    child_environment = {
        **environment,
        wsa.AUTHORITY_ENV_VAR: str(path),
    }
    monkeypatch.setitem(sys.modules, "jax", object())
    with pytest.raises(wsa.SourceAuthorityRefusal, match="preceded authority"):
        wsa.assert_child_binding(environ=child_environment)


def test_r1_unset_and_split_roots_fail_closed():
    with pytest.raises(wsa.SourceAuthorityRefusal, match="unset"):
        wsa.resolve_authority_root(environ={})
    with pytest.raises(wsa.SourceAuthorityRefusal, match="conflicts"):
        wsa.resolve_authority_root(
            environ={
                wsa.ENV_VAR: str(wsa.CANONICAL_ROOT),
                wsa.SRC_ENV_VAR: str(REPO),
            }
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"JAX_PLATFORMS": "gpu", "CUDA_VISIBLE_DEVICES": ""},
        {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "0"},
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "GPUWRF_GPU_LOCK_TOKEN": "leaked",
        },
    ],
)
def test_r1_preflight_environment_mutations_fail_closed(environment):
    with pytest.raises(cpu_preflight.CpuBoundaryRefusal):
        cpu_preflight.validate_device_free_environment(environment)


def test_r2_exact_real_token_and_label_pass_without_spending(tmp_path):
    lock = arm.validate_spending_identity(
        window=registry.SESSION_LABEL,
        receipt=_receipt(),
        env=_lock_environment(tmp_path),
        ledger_path=tmp_path / "absent-ledger.json",
    )
    assert lock["token_matched"] is True
    assert lock["label_matched"] is True
    assert not (tmp_path / "absent-ledger.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "one_character_token",
        "strict_prefix_token",
        "strict_suffix_token",
        "token_only_inside_cmd",
        "wrong_holder_label",
        "missing_exported_label",
    ],
)
def test_r2_token_and_label_mutations_fail_without_spending(tmp_path, mutation):
    kwargs = {}
    if mutation == "one_character_token":
        kwargs["holder_token"] = "exact-tokem"
    elif mutation == "strict_prefix_token":
        kwargs["holder_token"] = "exact-token-extra"
    elif mutation == "strict_suffix_token":
        kwargs["holder_token"] = "prefix-exact-token"
    elif mutation == "token_only_inside_cmd":
        kwargs.update(
            holder_token="different-token",
            command="cpu-only token=exact-token",
        )
    elif mutation == "wrong_holder_label":
        kwargs["holder_label"] = "baseline-census"
    else:
        kwargs["exported_label"] = None
    with pytest.raises(arm.WindowNotAuthorised):
        arm.validate_spending_identity(
            window=registry.SESSION_LABEL,
            receipt=_receipt(),
            env=_lock_environment(tmp_path, **kwargs),
            ledger_path=tmp_path / "ledger.json",
        )
    assert not (tmp_path / "ledger.json").exists()


def test_r2_mutated_registry_and_stale_ledger_fail_without_spending(
    tmp_path, monkeypatch
):
    receipt = _receipt()
    environment = _lock_environment(tmp_path)
    monkeypatch.delitem(
        registry.REGISTERED_WINDOWS, registry.SESSION_LABEL
    )
    with pytest.raises(arm.WindowNotAuthorised, match="not registered"):
        arm.validate_spending_identity(
            window=registry.SESSION_LABEL,
            receipt=receipt,
            env=environment,
            ledger_path=tmp_path / "absent.json",
        )
    monkeypatch.undo()

    ledger = tmp_path / "stale.json"
    ledger.write_text(
        json.dumps(
            {
                "spent": {
                    receipt.fingerprint(): {
                        "spent_at_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": "CPU mutation fixture",
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(arm.WindowNotAuthorised, match="already spent"):
        arm.validate_spending_identity(
            window=registry.SESSION_LABEL,
            receipt=receipt,
            env=environment,
            ledger_path=ledger,
        )


def test_r2_spend_receipt_has_exactly_one_production_caller():
    tree = ast.parse(
        (SCRIPTS / "run_gpu_arm.py").read_text(encoding="utf-8")
    )
    callers = []
    for function in (
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "spend_receipt"
            for node in ast.walk(function)
        ):
            callers.append(function.name)
    assert callers == ["authorise"]


def test_c0_declaration_exact_sets_signature_clip_and_invariants(tmp_path):
    boundary = _boundary()
    path = _write_boundary(tmp_path, boundary)
    declaration = capability.build_declaration(
        boundary, boundary_path=path, environ={}
    )
    capability.validate_declaration(declaration)
    assert tuple(declaration["available"]) == capability.AVAILABLE_FIELDS
    assert tuple(declaration["missing"]) == capability.MISSING_FIELDS
    assert declaration["denominator"]["method"] == "exact-lowered-trip-count"
    assert declaration["denominator"]["steps"] == 360
    assert (
        declaration["integration_clip"]["emitter"]
        == "scripts/v025/m0_exact_boundary_child.py"
    )
    assert declaration["integration_clip"]["GPUWRF_M0_EVIDENCE"] == "UNSET"
    assert tuple(declaration["fallback_invariants"]) == (
        capability.FALLBACK_INVARIANT_NAMES
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_lowered_trip_count",
        "lowered_configured_count_mismatch",
        "wrong_denominator_method",
        "stablehlo_hash_mismatch",
    ],
)
def test_c0_denominator_mutations_fail_closed(tmp_path, mutation):
    boundary = _boundary()
    trip = boundary["lowered_program"]["integration_trip_count"]
    if mutation == "missing_lowered_trip_count":
        del boundary["lowered_program"]["integration_trip_count"]
    elif mutation == "lowered_configured_count_mismatch":
        trip["configured_cross_check"]["exact_steps"] = 359
    elif mutation == "wrong_denominator_method":
        trip["method"] = "configured-count-observed"
    else:
        trip["stablehlo_sha256"] = "6" * 64
    path = _write_boundary(tmp_path, boundary)
    with pytest.raises(capability.CapabilityRefusal):
        capability.build_declaration(boundary, boundary_path=path, environ={})


def test_c0_duplicate_production_range_fails_closed(tmp_path):
    boundary = _boundary()
    path = _write_boundary(tmp_path, boundary)
    with pytest.raises(capability.CapabilityRefusal, match="must remain unset"):
        capability.build_declaration(
            boundary,
            boundary_path=path,
            environ={"GPUWRF_M0_EVIDENCE": "1"},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "available_set_drift",
        "missing_set_drift",
        "integration_clip_wrong_emitter",
        "raw_capture_hash_binding_removed",
        "postlock_order_removed",
        "postcapture_readonly_preservation_removed",
        "w2_w3_reservation_removed",
        "extra_device_arm_after_w3",
    ],
)
def test_c0_declaration_and_fallback_mutations_fail_closed(
    tmp_path, mutation
):
    boundary = _boundary()
    path = _write_boundary(tmp_path, boundary)
    declaration = capability.build_declaration(
        boundary, boundary_path=path, environ={}
    )
    if mutation == "available_set_drift":
        declaration["available"].pop(next(iter(declaration["available"])))
    elif mutation == "missing_set_drift":
        declaration["missing"].pop(next(iter(declaration["missing"])))
    elif mutation == "integration_clip_wrong_emitter":
        declaration["integration_clip"]["emitter"] = (
            "src/gpuwrf/runtime/operational_mode.py"
        )
        declaration["integration_clip"][
            "production_operational_mode_emitter"
        ] = True
    else:
        invariant_for_mutation = {
            "raw_capture_hash_binding_removed":
                "raw_capture_hash_bound_before_postlock_analysis",
            "postlock_order_removed":
                "postcapture_analysis_after_lock_release",
            "postcapture_readonly_preservation_removed":
                "postcapture_failure_preserves_raw_capture_and_readonly_sqlite",
            "w2_w3_reservation_removed":
                "insufficient_w2_w3_time_stops_before_w2",
            "extra_device_arm_after_w3":
                "no_extra_device_arm_after_w2_hash_and_w3_stop",
        }
        declaration["fallback_invariants"][
            invariant_for_mutation[mutation]
        ]["status"] = "FAIL"
    with pytest.raises(capability.CapabilityRefusal):
        capability.validate_declaration(_resign(declaration))


def test_frozen_hard_gate_and_mutation_sets_are_exact():
    assert set(capability.HARD_GATE_MUTATIONS) == {"R1", "R2", "C0"}
    flattened = [
        mutation
        for gate in ("R1", "R2", "C0")
        for mutation in capability.HARD_GATE_MUTATIONS[gate]
    ]
    assert len(flattened) == len(set(flattened))


def test_evidence_builder_normalizes_relative_junit_and_artifact_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evidence_builder, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    junit = Path("suite.xml")
    junit.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="cpu" name="green" time="0.1"/>'
        "</testsuite>\n",
        encoding="utf-8",
    )
    parsed = evidence_builder._junit(junit)
    artifact = evidence_builder._artifact_identity(junit)
    assert parsed["path"] == "suite.xml"
    assert parsed["tests"] == 1
    assert artifact["path"] == "suite.xml"
    assert parsed["sha256"] == artifact["sha256"]


def test_evidence_builder_normalizes_absent_git_objects():
    assert (
        evidence_builder._git_optional_object(
            "HEAD:scripts/v025/definitely_absent_review10_fixture.py"
        )
        is None
    )
    assert evidence_builder._git_optional_object(
        "HEAD:scripts/v025/run_gpu_arm.py"
    )


def test_real_cpu_boundary_executes_loader_fast_builder_and_exact_lower(
    tmp_path, monkeypatch
):
    # The real boundary verifier freezes the terminal generation's callable
    # ASTs. Replay those source bytes with today's test/oracle infrastructure;
    # moving its digest pins to today's model would erase the archived claim.
    archived = tmp_path / "production"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO),
         str(archived)], check=True,
    )
    production = "ba9dc2331d14c7f0598b2fa9028c669236a79b6e"
    subprocess.run(
        ["git", "-C", str(archived), "update-ref", "HEAD", production], check=True,
    )
    subprocess.run(
        ["git", "-C", str(archived), "checkout", production, "--", "src/gpuwrf",
         "data/fixtures", "data/manifests", "scripts/extract_rrtmg_tables.py"],
        check=True,
    )
    shutil.copytree(SCRIPTS, archived / "scripts/v025",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (archived / "data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
    # The authority also binds HEAD^{tree} for checkout fallback identity.
    # Its producer and consumer must inspect the same archived repository.
    monkeypatch.setattr(wsa, "REPO_ROOT", archived)
    authority_environment = {
        variable: str(wsa.CANONICAL_ROOT)
        for variable in wsa.ROOT_ENV_VARS
    }
    authority = wsa.build_source_authority(
        namelist_path=FAST / "namelist.input",
        environ=authority_environment,
    )
    authority_path = tmp_path / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    output = tmp_path / "real-boundary.json"
    environment = {
        **CPU_ENV,
        **wsa.child_environment_binding(authority),
        wsa.AUTHORITY_ENV_VAR: str(authority_path),
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(archived / "scripts/v025/m0_cpu_boundary_preflight.py"),
            "--authority",
            str(authority_path),
            "--run-dir",
            str(FAST),
            "--output",
            str(output),
        ],
        cwd=archived,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    capability.validate_real_boundary(
        payload, expected_authority_sha256=authority["authority_sha256"]
    )
    trip = payload["lowered_program"]["integration_trip_count"]
    assert trip["method"] == "exact-lowered-trip-count"
    assert trip["segment_trip_counts"] == [179, 1, 179, 1]
    assert trip["steps"] == trip["configured_cross_check"]["exact_steps"] == 360
