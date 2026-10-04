"""CPU-only proofs for the R5 generation and its terminal launch identity.

Every test here runs device-denied.  The R5 generation RAN once and is terminal
(retained cold rejection): no test may create a canonical mutable R5 root, and
no test may modify the retained R5 evidence.  The retired mutable launch roots
are proven vacant forever; the committed proof tree and session receipt are
proven byte-identical to their absolute pins.  Each one works in `tmp_path` or
reads the committed retained artifacts read-only.

The gate names are not declared here.  ``build_m0_r5_launch_readiness`` reads the
frozen mechanical set out of the sprint contract, and these tests assert against
that literal, so no candidate-authored declaration can define the reference set.

Re-bind provenance (2026-09-18 R5 freeze-guard re-bind sprint): the pins moved
from the retired R4/R5 candidate regime (HEAD ba9dc2331, src tree a6885ced) to
the authorized post-seam regime (final main 062c80880, src tree d7f55214);
pre-launch vacancy semantics became terminal-retention semantics.  Every moved
pin carries a tamper control in this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_m0_r5_launch_readiness as readiness  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import m0_window_parent as window_parent  # noqa: E402


@pytest.fixture(scope="module")
def frozen_h1_git(tmp_path_factory):
    # H1's synthetic output uses the terminal session's original tree pin.
    archived = tmp_path_factory.mktemp("m0-h1-source") / "repository.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", "--shared", str(REPO), str(archived)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(archived), "update-ref", "HEAD",
         "ba9dc2331d14c7f0598b2fa9028c669236a79b6e"], check=True,
    )
    assert subprocess.check_output(
        ["git", "-C", str(archived), "rev-parse", "HEAD:src/gpuwrf"], text=True,
    ).strip() == "a6885ceded260df2f5777d7366d75a5d38947cb7"
    return archived


# --------------------------------------------------------------------------- #
# R5 - the frozen gate set comes from the contract, never from the candidate    #
# --------------------------------------------------------------------------- #
def test_frozen_gate_set_is_read_from_the_contract_literal():
    frozen = readiness.contract_gate_set()
    assert frozen["members"] == sorted(
        [
            "H0_ROOTS",
            "H1_REAL_C1_BOUNDARY",
            "H2_PRODUCTION_ANALYSER",
            "H3_PACKET_BINDING",
            "H4_RETAINED_IDENTITY",
            "H5_DEVICE_DENIAL",
            "H6_SUITE_AND_TREE",
            "H7_KERNEL_LOCK_AUTHORITY",
        ]
    )
    assert frozen["source"].endswith("CONTRACT.md")
    # The literal really is the contract's own set expression, not a copy.
    assert frozen["literal"].startswith("{H0_ROOTS")
    assert readiness.sha256_file(readiness.CONTRACT_PATH) == frozen["source_sha256"]


def test_a_contract_without_the_set_refuses_rather_than_defaulting(tmp_path,
                                                                   monkeypatch):
    empty = tmp_path / "CONTRACT.md"
    empty.write_text("# no frozen set here\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "CONTRACT_PATH", empty)
    with pytest.raises(readiness.ReadinessRefusal):
        readiness.contract_gate_set()


# --------------------------------------------------------------------------- #
# H0 - the retired R5 generation: mutable roots vacant, evidence retained      #
# --------------------------------------------------------------------------- #
def test_h0_every_retired_mutable_r5_root_is_absent_non_symlink_and_unshadowed():
    vacancy = readiness.r5_generation_vacancy()
    assert vacancy["status"] == "PASS", json.dumps(vacancy, indent=2)
    observed = {item["target"] for item in vacancy["targets"]}
    assert observed == {
        readiness.R5_RAW_ROOT,
        readiness.R5_PAIR_CACHE_ROOT,
        readiness.R5_C1_CPU_RUN_ROOT,
    }
    for item in vacancy["targets"]:
        assert item["exists"] is False
        assert item["is_symlink"] is False
        assert item["symlink_ancestors_below_base"] == []


def test_h0_retained_terminal_r5_evidence_matches_its_absolute_pins():
    """The R5 generation ran and is terminal: its committed proof tree and
    session receipt are the generation's identity and must be byte-identical
    to the absolute pins -- present, real files, never edited."""

    retention = readiness.r5_terminal_retention()
    assert retention["status"] == "PASS", json.dumps(retention, indent=2)
    proof = retention["retained_proof_tree"]
    assert proof["exists"] is True
    assert proof["file_count"] > 0
    assert proof["tree_sha256"] == readiness.R5_PROOF_TREE_SHA256
    receipt = retention["retained_receipt"]
    assert receipt["exists"] is True
    assert receipt["sha256"] == readiness.R5_RECEIPT_SHA256


def test_h0_mutation_an_occupied_fixed_root_is_not_a_fresh_generation(monkeypatch):
    monkeypatch.setattr(readiness, "R5_RAW_ROOT", str(REPO / "scripts"))
    assert readiness.r5_generation_vacancy()["status"] == "BLOCKED"


def test_h0_mutation_a_tampered_retained_receipt_is_refused(tmp_path):
    """Negative control: one flipped byte in the retained receipt must fail
    the retention check closed."""

    retained = REPO / readiness.R5_RECEIPT_TEMPLATE
    tampered = tmp_path / "M0_CORE_W1_W2_W3_SESSION_RECEIPT_R5.json"
    tampered.write_bytes(retained.read_bytes().replace(b'"', b"'", 1))
    result = readiness.r5_terminal_retention(receipt_path=tampered)
    assert result["status"] == "BLOCKED"
    assert result["retained_receipt"]["matches_pin"] is False


def test_h0_mutation_a_tampered_retained_proof_tree_is_refused(tmp_path):
    """Negative control: one flipped byte anywhere in the retained proof tree
    must fail the retention check closed."""

    import shutil

    mirror = tmp_path / "autotune0_three_window_r5"
    shutil.copytree(
        REPO / readiness.R5_PROOF_ROOT, mirror, symlinks=False
    )
    victim = sorted(mirror.rglob("*.json"))[0]
    victim.write_bytes(victim.read_bytes() + b" ")
    result = readiness.r5_terminal_retention(proof_root=mirror)
    assert result["status"] == "BLOCKED"
    assert result["retained_proof_tree"]["matches_pin"] is False


def test_h0_parent_and_executor_name_one_r5_generation():
    constants = readiness.r5_constants_are_coherent()
    assert constants["status"] == "PASS", json.dumps(constants, indent=2)
    assert str(window_parent.RAW_ROOT) == readiness.R5_RAW_ROOT
    assert str(window_parent.PAIR_CACHE_ROOT) == readiness.R5_PAIR_CACHE_ROOT
    assert executor.C1_CPU_RUN_ROOT == readiness.R5_C1_CPU_RUN_ROOT
    assert executor.SESSION_RECEIPT == readiness.R5_RECEIPT_TEMPLATE
    assert executor.PROOF_ROOT == readiness.R5_PROOF_ROOT
    # Reviewed R3 run-id text is deliberately retained; freshness is the roots.
    assert all(
        run_id.endswith("-r3") for run_id in constants["run_ids_retain_reviewed_r3_text"]
    )
    assert constants["dead_fifth_root"]["status"] == (
        "UNREACHABLE_NOT_IN_HELD_SESSION"
    )


def test_h0_no_retired_mutable_r5_root_is_created_and_no_retained_evidence_moves():
    for target in (
        readiness.R5_RAW_ROOT,
        readiness.R5_PAIR_CACHE_ROOT,
        readiness.R5_C1_CPU_RUN_ROOT,
    ):
        assert not os.path.lexists(target)
    # The suite may not damage the retained terminal evidence either.
    assert readiness.r5_terminal_retention()["status"] == "PASS"


def test_h0_generation_identity_combines_vacancy_and_retention():
    identity = readiness.r5_generation_identity()
    assert identity["status"] == "PASS", json.dumps(identity, indent=2, default=str)
    assert identity["retired_mutable_roots_absent"] is True
    assert identity["retained_terminal_evidence_stable"] is True


# --------------------------------------------------------------------------- #
# H1 - the real outer -> held -> C1 boundary                                    #
# --------------------------------------------------------------------------- #
def test_h1_real_outer_owner_drives_the_real_c1_child_to_completion(
    tmp_path, monkeypatch, frozen_h1_git
):
    monkeypatch.setenv("GIT_DIR", str(frozen_h1_git))
    trace = readiness._h1_arm_in_fresh_process(
        tmp_path / "real", restore_r3_cpu_root=False
    )
    assert trace["status"] == "PASS", json.dumps(trace, indent=2, default=str)[:4000]
    # The traversal really reached the real child, with no C1 stage hook.
    child = trace["c1_child"]
    assert child["error"] is None
    assert child["returncode"] == 0
    assert child["supplied_cpu_run_root"] == readiness.R5_C1_CPU_RUN_ROOT
    assert child["cpu_preflight_root_supplied_by_owner"] == (
        readiness.R5_C1_CPU_RUN_ROOT
    )
    assert child["matches_fixed_r5_root"] is True
    # W2/W3 stayed reachable and the whole session passed.
    assert trace["stages_reached"][:2] == ["W1", "C1_SAME_RESULT_AND_FRESH_CPU_COMPARATOR"]
    assert trace["w2_and_w3_reachable"] is True
    assert trace["held_child"]["status"] == "PASS"
    assert trace["owner_status"] == "PASS"
    # Only the synthetic ledger was spent and no group leaked.
    assert trace["synthetic_ledger_spends"] >= 1
    assert trace["canonical_ledger_untouched"] is True
    assert trace["every_registered_process_group_empty"] is True
    # The real C1 command is the production one, not a replacement.
    command = trace["c1_command"]
    assert command[:3] == ["taskset", "-c", "16-27"]
    assert command[4].endswith("m0_w1_fast_pair.py")
    assert "--in-session-held-lock" in command
    assert "--prepare-session-pair" in command


def test_h1_mutation_restoring_a_fixed_r3_cpu_root_fails_for_that_reason(
    tmp_path, monkeypatch, frozen_h1_git
):
    monkeypatch.setenv("GIT_DIR", str(frozen_h1_git))
    trace = readiness._h1_arm_in_fresh_process(
        tmp_path / "r3", restore_r3_cpu_root=True
    )
    assert trace["status"] == "BLOCKED"
    child = trace["c1_child"]
    assert child["supplied_cpu_run_root"] == readiness.R3_C1_CPU_RUN_ROOT
    assert child["matches_fixed_r5_root"] is False
    # The failure is the root divergence, not some later unrelated refusal: the
    # owner pre-staged into R5 while the in-lock child was handed R3.
    assert trace["cpu_preflight_cpu_run_root"] == readiness.R5_C1_CPU_RUN_ROOT
    assert child["supplied_cpu_run_root"] != trace["cpu_preflight_cpu_run_root"]


def test_h1_isolation_is_declared_and_excludes_the_c1_stage(tmp_path):
    """A C1 stage hook, a replacement subprocess or a parser-only test cannot
    stand in for the boundary, so the attack declares exactly what it isolates."""

    source = (SCRIPTS / "build_m0_r5_launch_readiness.py").read_text(
        encoding="utf-8"
    )
    # The C1 branch calls the production CLI entrypoint with the real argv.
    assert "fast_pair.main(argv[argv.index(\"--w1-result\") :])" in source
    # And nothing installs a C1 stage hook.
    assert "core_session.C1_STAGE:" not in source
    assert "session.C1_STAGE: " not in source


# --------------------------------------------------------------------------- #
# H2 - the five production analyser relations                                  #
# --------------------------------------------------------------------------- #
def test_h2_all_five_production_relations_are_really_invoked(tmp_path):
    from _historical_artifacts import require_historical

    require_historical(readiness.REAL_BASELINE_SQLITE)
    rehearsal = readiness.production_analyser_rehearsal(tmp_path)
    assert rehearsal["status"] == "PASS", json.dumps(
        rehearsal["calls"], indent=2, default=str
    )[:4000]
    called = {call["relation"] for call in rehearsal["calls"]}
    assert called == set(readiness.PRODUCTION_RELATIONS)
    for call in rehearsal["calls"]:
        assert call["invoked"] is True
        assert call["is_generic_exception"] is False
        assert call["zero_substituted_for_missing"] is False
        # BLOCKED is an honest verdict too: a census that cannot attribute a
        # six-kernel fixture says so instead of substituting a zero.
        assert call["status"] in {"OK", "PASS", "MISSING", "BLOCKED"}
    by_relation = {call["relation"]: call for call in rehearsal["calls"]}
    assert by_relation["m0_postlock_census.derive_integration_scope"]["status"] == "OK"
    assert by_relation["m0_postlock_census.derive_step_census"]["status"] == "MISSING"
    analyzed = by_relation["m0_postlock_census.analyze_w2"]
    assert analyzed["status"] == "MISSING"
    assert analyzed["outcome"] == "NAMED_DOMAIN_REFUSAL"
    assert analyzed["refusal_class"] == "PostlockRefusal"
    # Which absent thing it names first depends on process state -- in a fresh
    # readiness process it is the missing release proof, inside the monolithic
    # suite it is the accelerator roots an earlier module imported.  Either way
    # it is a named domain refusal, never a generic exception or a zero.
    assert analyzed["named_refusal"]
    assert rehearsal["ast_call_inventory"]["status"] == "PASS"
    assert all(
        count > 0
        for count in rehearsal["ast_call_inventory"]["calls_per_relation"].values()
    )


def test_h2_missing_kernel_device_time_is_named_not_zeroed_or_crashed():
    from _historical_artifacts import require_historical

    require_historical(readiness.REAL_BASELINE_SQLITE)
    refusal = readiness.missing_kernel_table_refusal()
    assert refusal["status"] == "PASS"
    assert refusal["kernel_table_present"] is False
    named = refusal["named_refusal"]
    assert named["status"] == "MISSING"
    assert named["missing_quantity"] == "CUDA kernel device time"
    assert named["is_zero"] is False and named["is_crash"] is False
    assert refusal["artifact_sha256"] == readiness.REAL_BASELINE_SQLITE_SHA256


def test_h2_attribution_is_frozen_to_review10_source_named_leaves():
    attribution = readiness.deepest_unique_attribution()
    assert attribution["denominator_total_gpu_ops"] == 2_575_286
    assert attribution["all_leaves_total_gpu_ops"] == 2_574_776
    assert attribution["source_named_deepest_leaves_total_gpu_ops"] == 2_567_009
    assert attribution["binding_numerator"] == 2_567_009
    assert attribution["binding_numerator_is_source_named"] is True
    assert round(attribution["attributed_share"], 3) == 0.997
    assert attribution["reproduces_review10"] is True
    # The claim that Review 10 did not document the rule is retired.
    assert attribution["review10_documents_the_rule"] is True
    assert "OPUS_MANAGEMENT_REVIEW_10.md" in attribution["review10_source"]
    assert attribution["artifact_sha256"] == readiness.REAL_PROJ_CSV_SHA256


def test_h2_mutation_all_leaves_numerator_fails():
    mutated = readiness.deepest_unique_attribution(use_all_leaves=True)
    assert mutated["reproduces_review10"] is False
    assert mutated["binding_numerator"] == 2_574_776
    assert mutated["binding_numerator_is_source_named"] is False


def test_h2_event_fixture_is_byte_identical_to_the_committed_hash():
    assert readiness.EVENT_FIXTURE.is_file()
    assert (
        readiness.sha256_file(readiness.EVENT_FIXTURE)
        == readiness.EVENT_FIXTURE_SHA256
    )


# --------------------------------------------------------------------------- #
# H3 - the exact stable launch packet                                          #
# --------------------------------------------------------------------------- #
def test_h3_packet_binds_the_authorized_integration_line():
    packet = readiness.build_packet()
    binding = readiness.packet_binding(packet)
    assert binding["status"] == "PASS", json.dumps(binding, indent=2, default=str)
    assert packet["checkout"]["stable_integration_worktree"] == str(REPO)
    # The stable-checkout identity is the lineage, not a branch-name literal:
    # this checkout must descend from the authorized production head.
    checkout = packet["checkout"]
    assert checkout["authorized_head"] == readiness.AUTHORIZED_PRODUCTION_HEAD
    assert checkout["descends_from_authorized_head"] is True
    # The worker-worktree record is honest about whatever checkout ran.
    assert checkout["is_worker_worktree"] is (
        packet["checkout"]["branch"].startswith("worker/")
    )
    assert packet["session"]["label"] == "m0-core-w1-w2-w3-session"
    assert packet["session"]["stage_order"][0].startswith("W1")
    assert packet["grants_permission"] is False
    for entry in binding["commands"].values():
        assert entry["reproduces"] is True
        assert entry["every_repository_path_below_stable_worktree"] is True


def test_h3_mutation_a_checkout_off_the_authorized_line_is_not_stable(monkeypatch):
    """Negative control: a checkout that does not descend from the authorized
    production head must lose the stable-lineage binding.  The all-zeros sha
    is a commit that cannot be on any lineage, so the ancestry check must
    honestly report False rather than a hardcoded True."""

    foreign_head = "0" * 40
    monkeypatch.setattr(readiness, "AUTHORIZED_PRODUCTION_HEAD", foreign_head)
    rebind = readiness.build_packet()
    assert rebind["checkout"]["authorized_head"] == foreign_head
    assert rebind["checkout"]["descends_from_authorized_head"] is False


def test_h3_receipt_of_record_is_retained_terminal_evidence_and_the_ledger_is_the_authority():
    packet = readiness.build_packet()
    receipt = packet["receipt"]
    assert receipt["template_path"] == readiness.R5_RECEIPT_TEMPLATE
    # The R5 session ran: the receipt is retained terminal evidence whose
    # bytes are pinned; it still carries no authority by itself.
    assert receipt["exists_today"] is True
    assert receipt["retained_sha256_matches_pin"] is True
    assert receipt["is_template_not_authority"] is True
    assert receipt["post_hoc_content_fingerprint_proof_required"] is True
    assert packet["canonical_authority"]["global_spend_ledger"] == (
        readiness.CANONICAL_LEDGER
    )
    assert packet["canonical_authority"]["ledger_is_fingerprint_keyed"] is True
    assert packet["canonical_authority"]["zero_wait_lock_wrapper"] == (
        readiness.CANONICAL_LOCK_WRAPPER
    )
    kernel = packet["canonical_authority"]["kernel_lock_preflight"]
    assert kernel["lock_path"] == "/tmp/wrf_gpu2_gpu.lock"
    assert kernel["kernel_table"] == "/proc/locks"
    assert kernel["required_before_receipt_creation"] is True
    assert kernel["maximum_age_seconds"] == 120
    assert kernel["legacy_sidecar_fuser_lslocks_can_prove_vacancy"] is False
    assert kernel["vacant_snapshot_grants_permission"] is False


def test_packet_policy_carries_the_frozen_budget_and_stand_down():
    packet = readiness.build_packet()
    budget = packet["budget"]
    assert budget["prelock_cpu_maximum_seconds"] == 420
    assert budget["session_seconds"] == 4_220
    assert budget["session_ceiling_seconds"] == 4_500
    assert budget["within_ceiling"] is True
    assert budget["headroom_seconds"] == 280
    coordination = packet["coordination"]
    assert coordination["fresh_unabridged_affirmatives_required_from"] == ["0:2", "0:3"]
    assert coordination["explicit_handoff_required"] is True
    assert packet["nightly_return"]["wrf_gpu2_must_stand_down_without_release"] is True
    assert packet["nightly_return"]["grants_permission"] is False


def test_packet_policy_carries_current_default_owner_and_server_preemption():
    """The retired local-clock boundary cannot survive in the launch packet."""

    nightly = readiness.build_packet()["nightly_return"]
    assert nightly["policy"] == (
        "DEFAULT_OWNER_AND_SERVER_REQUESTED_PRODUCTION_PREEMPTION"
    )
    assert nightly["default_owner_outside_production"] == "0:2"
    assert nightly["production_ownership"]["active_nightly_owns_until"] == (
        "exact explicit RELEASE"
    )
    assert nightly["production_ownership"]["release_returns_immediately_to"] == "0:2"
    assert nightly["production_ownership"]["free_lock_between_stages_is_a_handoff"] is False
    assert nightly["wrf_gpu2_claim"]["requires_explicit_resource_manager_release"] is True
    assert nightly["wrf_gpu2_claim"]["free_lock_is_permission"] is False
    assert nightly["wrf_gpu2_claim"]["idle_device_is_permission"] is False
    assert nightly["preemption"]["trigger"] == "explicit server request"
    assert nightly["preemption"]["inferred_from_local_clock"] is False
    encoded = json.dumps(nightly, sort_keys=True)
    for retired_value in ("00:30", "00:55", "01:00", "23:14:40"):
        assert retired_value not in encoded
    assert "latest_safe_local_start" not in nightly
    assert nightly["supersedes"]["policy"] == "RECURRING_WEST_NIGHTLY_BOUNDARY"
    assert nightly["supersedes"]["manager_commit"] == "c1e446bb"
    assert nightly["supersedes"]["status"] == "RETIRED"


def test_packet_policy_records_current_provenance_and_default_owner():
    packet = readiness.build_packet()
    provenance = packet["nightly_return"]["override_provenance"]
    assert provenance["kind"] == "PRINCIPAL_OVERRIDE"
    assert provenance["manager_commit"] == (
        "e7d78373ebae2b530e496df01ed58f50a0158419"
    )
    assert ".agent/skills/locking-gpu/SKILL.md" in provenance["encoded_authority"]
    assert (
        ".agent/patches/2026-08-02-gpu-default-owner-server-preempt.md"
        in provenance["encoded_authority"]
    )
    assert "c1e446bb" in provenance["replaces"]
    assert "H0-H6 are unchanged" in provenance["scope"]
    assert packet["coordination"]["override_provenance"] == provenance
    # 0:2 is the permanent default owner; no date-scoped apparent gap survives.
    owner = packet["coordination"]["owner_priority"]
    assert owner["holder"] == "0:2"
    assert owner["role"] == "permanent default owner outside Production"
    assert owner["wrf_gpu2_claim_requires_explicit_resource_manager_release"] is True
    assert owner["free_lock_or_idle_device_is_permission"] is False
    assert owner["wrf_gpu2_takes_no_apparent_gap"] is True
    assert "not permission" in owner["rule"]
    assert "date" not in owner
    assert "duration" not in owner


def test_h3_packet_binds_the_four_fixed_r5_roots_and_their_terminal_identity():
    packet = readiness.build_packet()
    assert packet["roots"]["raw_root"] == readiness.R5_RAW_ROOT
    assert packet["roots"]["pair_cache_root"] == readiness.R5_PAIR_CACHE_ROOT
    assert packet["roots"]["c1_cpu_run_root"] == readiness.R5_C1_CPU_RUN_ROOT
    assert packet["roots"]["proof_root"] == readiness.R5_PROOF_ROOT
    assert packet["roots"]["retired_mutable_roots_absent"] is True
    assert packet["roots"]["retained_terminal_evidence_stable"] is True
    order = packet["session"]["stage_order"]
    assert order.index("C1_SAME_RESULT_AND_FRESH_CPU_COMPARATOR") < order.index(
        "W2_PROFILED_CAPTURE"
    )
    assert order.index("W2_PROFILED_CAPTURE") < order.index("W3_CLEAN_MATCHED_ARM")


def test_h3_mutation_a_worker_worktree_command_fails(tmp_path):
    """Negative control: the outer-owner command must fail its binding when it
    points at a real directory layout OUTSIDE the stable checkout.  The mirror
    is built live from the command's own repo paths so every refused path
    actually exists (a dead path would be skipped by the resolver and
    silently pass)."""

    worker = tmp_path / "worker-mirror"
    base = readiness.build_packet()
    mirrored = 0
    for part in base["commands"]["outer_owner"]["command"]:
        if isinstance(part, str) and part.startswith(str(REPO)):
            target = worker / Path(part).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# mirror of the production file\n", encoding="utf-8")
            mirrored += 1
    assert mirrored > 0
    mutated = readiness.build_packet(
        worker_worktree_mutation=True, worker_worktree=str(worker)
    )
    binding = readiness.packet_binding(mutated)
    assert binding["status"] == "BLOCKED"
    outer = binding["commands"]["outer_owner"]
    assert outer["reproduces"] is False or (
        outer["every_repository_path_below_stable_worktree"] is False
    )
    # The refusal is really about the escape, not a hash accident: the mutated
    # command resolves the mirrored script outside the stable worktree.
    assert any(
        not item["below_stable_worktree"] and not item["is_interpreter"]
        for item in outer["resolved_paths"]
    )


# --------------------------------------------------------------------------- #
# H4 / H5 / H6                                                                 #
# --------------------------------------------------------------------------- #
def test_h4_retained_r3_r4_and_r5_evidence_is_stable_within_one_process():
    before = readiness.retained_manifest()
    after = readiness.retained_manifest()
    assert before["aggregate_sha256"] == after["aggregate_sha256"]
    assert before["file_count"] > 0
    assert "proofs/v025/m0/autotune0_three_window" in before["roots"]
    assert "proofs/v025/m0/autotune0_three_window_r4" in before["roots"]
    assert (
        ".agent/sprints/2026-08-01-v0250-m0-r4-final-readiness"
        in before["roots"]
    )
    # The terminal R5 generation's evidence is retained alongside R3/R4.
    assert "proofs/v025/m0/autotune0_three_window_r5" in before["roots"]
    assert (
        ".agent/sprints/2026-08-09-v0250-m0-r5-kernel-lock-authority"
        in before["roots"]
    )


def test_h4_mutation_one_flipped_retained_byte_is_visible():
    before = readiness.retained_manifest()
    flipped = dict(before["files"])
    flipped[sorted(flipped)[0]] = "0" * 64
    assert readiness.canonical_sha256(flipped) != before["aggregate_sha256"]


def test_h5_no_accelerator_module_or_real_authority_action(tmp_path):
    """The claim is about the readiness process, so it is measured in one.

    This monolithic pytest process has already imported jaxlib through unrelated
    modules; asserting device denial here would measure pytest, not the builder.
    """

    import subprocess

    output = tmp_path / "denial.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json; sys.path.insert(0, %r);"
            "import build_m0_r5_launch_readiness as r;"
            "open(%r, 'w').write(json.dumps(r.device_denial()))"
            % (str(SCRIPTS), str(output)),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert output.is_file(), completed.stderr[-2000:]
    denial = json.loads(output.read_text(encoding="utf-8"))
    assert denial["status"] == "PASS", denial["accelerator_modules_loaded"]
    assert denial["accelerator_modules_loaded"] == []
    assert denial["environment"]["JAX_PLATFORMS"] == "cpu"
    assert denial["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert denial["canonical_lock_environment_absent"] is True
    assert denial["no_coordination_request"] is True


def test_h5_mutation_a_loaded_accelerator_module_blocks(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", object())
    assert readiness.device_denial()["status"] == "BLOCKED"


def test_h6_src_gpuwrf_tree_is_byte_identical_to_the_authorized_base(
    tmp_path, monkeypatch
):
    frozen = tmp_path / "authorized-base"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO),
         str(frozen)], check=True,
    )
    subprocess.run(
        ["git", "-C", str(frozen), "update-ref", "HEAD",
         readiness.AUTHORIZED_PRODUCTION_HEAD], check=True,
    )
    monkeypatch.setattr(readiness, "REPO", frozen)
    result = readiness.suite_and_tree(None)
    assert result["src_gpuwrf_tree"]["identical"] is True
    assert result["src_gpuwrf_tree"]["base_commit"] == (
        readiness.AUTHORIZED_PRODUCTION_HEAD
    )
    assert result["src_gpuwrf_tree"]["base_authorized"] == (
        result["src_gpuwrf_tree"]["head"]
    )
    assert (
        result["src_gpuwrf_tree"]["base_authorized"]
        == readiness.AUTHORIZED_SRC_GPUWRF_TREE
    )


def test_h6_mutation_a_src_tree_off_the_authorized_base_blocks(monkeypatch):
    """Negative control: a checkout whose src/gpuwrf diverges from the pinned
    authorized base must block, never pass open."""

    monkeypatch.setattr(
        readiness,
        "AUTHORIZED_PRODUCTION_HEAD",
        "24b128d539c101e941f659d37d6fe0aa3601850f",
    )
    assert readiness.suite_and_tree(None)["status"] == "BLOCKED"


def test_h6_mutation_one_failing_test_blocks(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite tests="3" failures="1" errors="0" skipped="0" time="1.0"/>\n',
        encoding="utf-8",
    )
    assert readiness.suite_and_tree(junit)["status"] == "BLOCKED"


def test_h7_real_private_flock_wins_over_false_free_legacy_views(tmp_path):
    gate = readiness.kernel_lock_authority_capability(tmp_path)
    assert gate["status"] == "PASS", json.dumps(gate, indent=2)[:8000]
    assert all(gate["checks"].values())
    occupied = gate["private_flock_control"]["occupied_snapshot"]
    assert occupied["verdict"] == "BLOCKED_OCCUPIED"
    assert occupied["matching_record_count"] == 1
    assert occupied["legacy_observations"]["holder_sidecar"]["text"] == ""
    assert occupied["legacy_observations"]["fuser"]["stdout"] == ""
    assert occupied["legacy_observations"]["lslocks"]["matching_lines"] == []
    assert gate["pre_receipt_rule"]["maximum_age_seconds"] == 120
    assert gate["pre_receipt_rule"]["vacancy_grants_permission"] is False
