from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.v0234_v10_pinned_replay_profile import (
    ALLOWED_RED_FIELDS,
    EXPECTED_COUNTS,
    MODEL_TREE,
    PREFIX_OWN_STEPS,
    PREFIX_SEGMENTS,
    STRICT_FIELDS,
    classify_scoped_v10_observation,
)
from scripts.v0234_v10_replay_evidence import (
    build_equality,
    build_pin_manifest,
    canonical_digest,
    load_self_hashed,
    main as evidence_main,
    write_self_hashed,
)


REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-21-v0234-gpt-v10-replay"


def _decisive(*, red: tuple[str, ...] = ()) -> dict:
    fields = {
        field: {
            "candidate_rmse": float(index + 1) / 10.0,
            "retry20_authority_exact": True,
            "no_worse": field not in red,
        }
        for index, field in enumerate(STRICT_FIELDS)
    }
    return {
        "passed": not red,
        "static_exact": True,
        "strict_fields": fields,
        "violations": [
            {"field": field, "gate": "rmse_no_worse_than_frozen_retry20"}
            for field in red
        ],
    }


def test_scoped_classifier_preserves_green_and_only_retains_v_v10_red() -> None:
    authority = {"authority_sha256": "a" * 64}
    green = classify_scoped_v10_observation(
        _decisive(), candidate_sha256="b" * 64, authority=authority,
    )
    assert green["passed"] is True
    assert green["classification"] == "STRICT_V_V10_GREEN"
    assert green["release_gate_green"] is True

    retained = classify_scoped_v10_observation(
        _decisive(red=("V", "V10")),
        candidate_sha256="c" * 64,
        authority=authority,
    )
    assert retained["passed"] is True
    assert retained["classification"] == "STRICT_V_V10_RED_RETAIN_AND_STOP"
    assert retained["release_gate_green"] is False
    assert set(retained["red_fields"]) == ALLOWED_RED_FIELDS
    assert retained["waiver_or_reclassification"] is False

    rejected = classify_scoped_v10_observation(
        _decisive(red=("T",)), candidate_sha256="d" * 64, authority=authority,
    )
    assert rejected["passed"] is False
    assert rejected["classification"] == "UNAUTHORIZED_SCIENTIFIC_RED"


def test_scoped_schedule_stops_exactly_at_step_9000() -> None:
    assert sum(PREFIX_SEGMENTS) == 1000
    assert PREFIX_OWN_STEPS == {"d01": 1000, "d02": 3000, "d03": 9000}
    assert EXPECTED_COUNTS == {"d01": 16, "d02": 16, "d03": 46}
    source = (REPO / "scripts/v0234_nested_frozen_wrf_boundary_window.py").read_text()
    assert 'globals().get("PROFILE_PREFIX_TERMINAL_HANDLER")' in source
    assert 'globals().get("PROFILE_EARLY_CAUSAL_HANDLER")' in source


def _rows(*, changed: tuple[str, int] | None = None) -> list[dict]:
    result = []
    for domain, count in EXPECTED_COUNTS.items():
        for ordinal in range(count):
            own_step = ordinal * (200 if domain == "d03" else 1)
            digest = hashlib.sha256(f"{domain}:{own_step}".encode()).hexdigest()
            if changed == (domain, own_step):
                digest = "f" * 64
            result.append({
                "domain": domain,
                "own_step": own_step,
                "candidate_sha256": digest,
            })
    return result


def _arm(*, mode: str, rows: list[dict] | None = None) -> dict:
    strict = {field: float(index + 1) for index, field in enumerate(STRICT_FIELDS)}
    return {
        "schema": "gpuwrf.v0234.v10-pinned-replay-arm.v1",
        "verdict": "V10_V_STRICT_RED_RETAINED",
        "mode": mode,
        "nonce": "1" * 64,
        "namespace": f"synthetic-{mode}",
        "src_gpuwrf_tree": MODEL_TREE,
        "reference_or_pinned_scoped_horizon_complete": True,
        "incremental_frame_pairs": {
            "counts": dict(EXPECTED_COUNTS),
            "rows": _rows() if rows is None else rows,
        },
        "checkpoint_carries": {
            step: {"carry": {"manifest": {"step": int(step), "leaves": ["a", "b"]}}}
            for step in ("8800", "9000")
        },
        "strict_rmse": strict,
        "red_fields": ["V", "V10"],
        "release_gate_green": False,
        "scope": {
            "d03_terminal_step": 9000,
            "late_ni_claimed": False,
            "tolerance_changed": False,
        },
    }


def test_pin_manifest_and_bit_equality_proofs_are_fail_closed(tmp_path: Path) -> None:
    reference_proof = tmp_path / "reference.json"
    pinned_proof = tmp_path / "pinned.json"
    reference_pin = tmp_path / "reference.pb"
    pinned_pin = tmp_path / "pinned.pb"
    manifest = tmp_path / "manifest.json"
    equality = tmp_path / "equality.json"
    write_self_hashed(reference_proof, _arm(mode="reference"))
    write_self_hashed(pinned_proof, _arm(mode="pinned"))
    reference_pin.write_bytes(b"authenticated autotune table")
    pinned_pin.write_bytes(reference_pin.read_bytes())

    pin_payload = build_pin_manifest(
        reference_proof=reference_proof,
        pin=reference_pin,
        output=manifest,
    )
    assert pin_payload["verdict"] == "V10_REFERENCE_PIN_AUTHENTICATED"
    manifest_payload, _ = load_self_hashed(manifest)
    assert manifest_payload["scientific_gate_reclassified"] is False

    equality_payload = build_equality(
        reference_proof=reference_proof,
        pinned_proof=pinned_proof,
        reference_pin=reference_pin,
        pinned_pin=pinned_pin,
        output=equality,
    )
    assert equality_payload["passed"] is True
    assert equality_payload["all_frames_bit_equal"] is True
    assert equality_payload["all_checkpoint_leaves_bit_equal"] is True
    assert equality_payload["autotune_dump_byte_equal"] is True
    loaded, _ = load_self_hashed(equality)
    unsigned = dict(loaded)
    embedded = unsigned.pop("proof_sha256")
    assert embedded == canonical_digest(unsigned)


def test_equality_proof_exposes_one_changed_frame(tmp_path: Path) -> None:
    reference_proof = tmp_path / "reference.json"
    pinned_proof = tmp_path / "pinned.json"
    reference_pin = tmp_path / "reference.pb"
    pinned_pin = tmp_path / "pinned.pb"
    equality = tmp_path / "equality.json"
    write_self_hashed(reference_proof, _arm(mode="reference"))
    changed = _rows(changed=("d03", 9000))
    write_self_hashed(pinned_proof, _arm(mode="pinned", rows=changed))
    reference_pin.write_bytes(b"same")
    pinned_pin.write_bytes(b"same")
    result = build_equality(
        reference_proof=reference_proof,
        pinned_proof=pinned_proof,
        reference_pin=reference_pin,
        pinned_pin=pinned_pin,
        output=equality,
    )
    assert result["passed"] is False
    changed_rows = [row for row in result["frame_comparisons"] if not row["bit_equal"]]
    assert [(row["domain"], row["own_step"]) for row in changed_rows] == [
        ("d03", 9000)
    ]


def test_evidence_cli_reports_the_signed_proof(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    reference_proof = tmp_path / "reference.json"
    reference_pin = tmp_path / "reference.pb"
    manifest = tmp_path / "manifest.json"
    write_self_hashed(reference_proof, _arm(mode="reference"))
    reference_pin.write_bytes(b"authenticated autotune table")
    monkeypatch.setattr(
        "sys.argv",
        [
            "v0234_v10_replay_evidence.py",
            "pin-manifest",
            "--reference-proof",
            str(reference_proof),
            "--pin",
            str(reference_pin),
            "--output",
            str(manifest),
        ],
    )

    assert evidence_main() == 0
    reported = json.loads(capsys.readouterr().out)
    signed, _ = load_self_hashed(manifest)
    assert reported["verdict"] == "V10_REFERENCE_PIN_AUTHENTICATED"
    assert reported["proof_sha256"] == signed["proof_sha256"]


def test_launcher_is_one_locked_dump_then_pin_command() -> None:
    launcher = SPRINT / "v10-pinned-replay-retry3.sh"
    text = launcher.read_text()
    assert text.count("/scripts/with_gpu_lock.sh") == 1
    assert text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
    assert "--intent production-preemptible" in text
    assert "--xla_gpu_dump_autotune_results_to=$DUMP_PIN" in text
    assert "--xla_gpu_load_autotune_results_from=$REFERENCE_PIN" in text
    assert "GPUWRF_NESTED_FUSE=0" in text
    assert "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=1" in text
    assert "nvidia-smi" not in text
    assert "rocm-smi" not in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_profile_static_schedule_and_launcher_audits_are_green() -> None:
    code = """
import json
from scripts import v0234_nested_frozen_wrf_boundary_window as runner
print(json.dumps({
    'schedule': runner.schedule_clock_oracle(),
    'static': runner.static_source_audit(),
    'launcher': runner.audit_exact_launch_command(runner.LAUNCH_COMMAND),
    'cpu_only_process_excluded': runner._declares_explicit_cpu_only_no_cuda([
        '/bin/bash', '-c',
        'env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= '
        "python -c 'from gpuwrf.runtime.domain_tree import run_domain_tree'",
    ]),
    'ambiguous_process_not_excluded': runner._declares_explicit_cpu_only_no_cuda([
        'python', '-m', 'gpuwrf.runtime.operational_mode',
    ]),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": f"{REPO}:{REPO / 'src'}",
        "GPUWRF_V10_PINNED_REPLAY": "1",
        "GPUWRF_V10_REPLAY_MODE": "reference",
        "GPUWRF_V10_REPLAY_NONCE": (
            "c17fca1e201ae106cf8ad77463686823c82226d59c7dea01e741c0be0ba7f6bc"
        ),
        "GPUWRF_NESTED_BUNDLE_APPROVED_SHA": (
            "1bfee4e94be5c614051563678d62e5293bb5426a"
        ),
    })
    completed = subprocess.run(
        ["<USER_HOME>/miniconda3/bin/python", "-c", code],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schedule"]["passed"] is True
    assert payload["static"]["passed"] is True
    assert payload["launcher"]["passed"] is True
    assert payload["cpu_only_process_excluded"] is True
    assert payload["ambiguous_process_not_excluded"] is False
