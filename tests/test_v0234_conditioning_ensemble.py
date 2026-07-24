from __future__ import annotations

import numpy as np

from scripts import v0234_conditioning_ensemble as ensemble


def test_masks_are_balanced_reproducible_and_pairwise_orthogonal() -> None:
    shapes = {"U": (44, 93, 112), "V": (44, 94, 111)}
    first = ensemble.mask_audit(shapes)
    second = ensemble.mask_audit(shapes)

    assert first == second
    assert set(first["masks"]) == {"mask_a", "mask_b", "mask_c"}
    assert all(row["combined_balance_sum"] == 0 for row in first["masks"].values())
    assert all(value == 0 for value in first["pairwise_dot_products"].values())
    assert {row["combined_cell_count"] for row in first["masks"].values()} == {745_800}


def test_one_ulp_perturbation_changes_only_selected_cells() -> None:
    base = np.linspace(-20.0, 20.0, 44 * 13 * 14, dtype=np.float32).reshape(44, 13, 14)
    spec = ensemble.MASK_SPECS[0]
    signs = ensemble.mask_signs("U", base.shape, spec)

    plus, plus_stats = ensemble.apply_one_ulp(base, signs, 1)
    minus, minus_stats = ensemble.apply_one_ulp(base, signs, -1)

    selected = (slice(None), slice(5, -5), slice(5, -5))
    untouched = np.ones(base.shape, dtype=bool)
    untouched[selected] = False
    assert np.array_equal(plus[untouched], base[untouched])
    assert np.array_equal(minus[untouched], base[untouched])
    assert plus_stats["changed_count"] == signs.size
    assert minus_stats["changed_count"] == signs.size
    assert np.array_equal(
        plus[selected],
        np.nextafter(
            base[selected],
            np.where(signs > 0, np.float32(np.inf), np.float32(-np.inf)),
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        minus[selected],
        np.nextafter(
            base[selected],
            np.where(signs < 0, np.float32(np.inf), np.float32(-np.inf)),
            dtype=np.float32,
        ),
    )


def _statistics(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "literal_envelope_inside_fraction": 0.90,
        "milestone_gpu_to_member_max_ratio_max": 1.10,
        "gpu_over_material_factor_fraction": 0.0,
        "all_milestones_gpu_over_material_factor": False,
        "gpu_normalized_bias_p95": 0.04,
        "ensemble_normalized_bias_p95": 0.05,
        "gpu_ensemble_normalized_bias_p95_gap": 0.01,
        "magnitude_correlation_median": 0.55,
        "top5pct_jaccard_median": 0.20,
        "vertical_profile_correlation_median": 0.90,
        "gpu_terrain_correlation_median": 0.40,
        "ensemble_terrain_correlation_median": 0.35,
        "gpu_ensemble_terrain_median_gap": 0.05,
        "high_terrain_argmax_fraction": 0.75,
        "terrain_opposite_sign_hard_failure": False,
    }
    values.update(overrides)
    return values


def test_frozen_decision_rule_supports_only_when_all_support_gates_pass() -> None:
    verdict, gates = ensemble.classify_decision(_statistics())
    assert verdict == "WRF_CONDITIONING_ENSEMBLE_SUPPORTS_RELEASE_FRAMING"
    assert gates["primary_growth_support"] is True
    assert gates["zero_mean_pass"] is True
    assert gates["spatial_signature_pass"] is True
    assert gates["vertical_signature_pass"] is True


def test_frozen_decision_rule_falsifies_on_material_scale_gap() -> None:
    verdict, gates = ensemble.classify_decision(
        _statistics(
            literal_envelope_inside_fraction=0.0,
            milestone_gpu_to_member_max_ratio_max=3.0,
            gpu_over_material_factor_fraction=0.90,
            all_milestones_gpu_over_material_factor=True,
        )
    )
    assert verdict == "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING"
    assert gates["primary_scale_falsifier"] is True


def test_frozen_decision_rule_falsifies_on_two_hard_structure_failures() -> None:
    verdict, gates = ensemble.classify_decision(
        _statistics(
            literal_envelope_inside_fraction=0.30,
            milestone_gpu_to_member_max_ratio_max=1.5,
            magnitude_correlation_median=0.01,
            vertical_profile_correlation_median=0.10,
        )
    )
    assert verdict == "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING"
    assert gates["hard_structural_falsifier"] is True
    assert gates["hard_structural_failure_count"] >= 2


def test_frozen_decision_rule_is_inconclusive_in_the_gray_zone() -> None:
    verdict, gates = ensemble.classify_decision(
        _statistics(
            literal_envelope_inside_fraction=0.50,
            milestone_gpu_to_member_max_ratio_max=1.5,
            magnitude_correlation_median=0.20,
        )
    )
    assert verdict == "WRF_CONDITIONING_ENSEMBLE_INCONCLUSIVE"
    assert gates["primary_growth_support"] is False
    assert gates["hard_structural_falsifier"] is False


def _process(identity: str, physical: list[int], affinity: list[int] | None = None) -> dict[str, object]:
    return {
        "identity_sha256": identity,
        "cpu_affinity": affinity if affinity is not None else physical,
        "physical_cores": physical,
    }


def test_principal_override_freezes_exact_three_core_12_rank_layout() -> None:
    topology = {cpu: cpu for cpu in range(16)} | {cpu + 16: cpu for cpu in range(16)}
    logical = sorted(ensemble._parse_cpu_list(ensemble.CPUSET))

    assert ensemble.CPUSET == "13-15,29-31"
    assert ensemble.MPI_RANKS == 12
    assert sorted({topology[cpu] for cpu in logical}) == [13, 14, 15]
    assert set(ensemble.EXPECTED_PHYSICAL_CORES).isdisjoint(ensemble.PRODUCTION_PHYSICAL_CORES)


def test_systemd_boundary_freezes_exact_12_ranks_on_only_three_cores() -> None:
    placement = ensemble.verify_cpu_placement()
    payload = ensemble.wrf_payload_command("/bin/true")
    environment = ensemble.wrf_launch_environment(ensemble.SCRATCH / "test-dumps")
    command = ensemble.systemd_launch_command(
        "v0234-test-boundary", ensemble.REPO, ensemble.SCRATCH / "test-dumps", "/bin/true"
    )

    assert placement["rank_count"] == 12
    assert placement["per_rank_allowed_logical_processing_units"] == [13, 14, 15, 29, 30, 31]
    assert placement["allowed_physical_cores"] == [13, 14, 15]
    assert placement["average_rank_oversubscription_per_physical_core"] == 4.0
    assert payload[-7:] == [
        "--map-by", ":oversubscribe:hwtcpus", "--bind-to", "none", "-np", "12", "/bin/true"
    ]
    assert command[:3] == ["/usr/bin/taskset", "-c", "13-15,29-31"]
    assert "--property=CPUAffinity=13 14 15 29 30 31" in command
    assert "--property=SystemCallFilter=~sched_setaffinity" in command
    assert "--property=NoNewPrivileges=yes" in command
    assert command[-3:] == ["-np", "12", "/bin/true"]
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_affinity_unit_gate_fails_closed_on_filter_or_cpuset_drift() -> None:
    state = {
        "returncode": 0,
        "LoadState": "loaded",
        "CPUAffinity": "13-15 29-31",
        "SystemCallFilter": "~sched_setaffinity",
        "SystemCallErrorNumber": "1",
        "NoNewPrivileges": "yes",
        "KillMode": "control-group",
        "ControlGroup": "/user.slice/v0234.service",
        "MainPID": "123",
    }
    assert ensemble.affinity_unit_reasons(state) == []

    state["CPUAffinity"] = "0-31"
    state["SystemCallFilter"] = ""
    reasons = ensemble.affinity_unit_reasons(state)
    assert "affinity_unit_cpu_set_drift" in reasons
    assert "affinity_unit_sched_setaffinity_filter_drift" in reasons


def test_nightly_active_and_live_lock_alone_do_not_block_disjoint_production() -> None:
    topology = {cpu: cpu for cpu in range(16)} | {cpu + 16: cpu for cpu in range(16)}
    reasons = ensemble._admission_reasons(
        topology=topology,
        production_rows=[_process("prod", list(range(12)), list(range(12)))],
        ensemble_rows=[],
        nightly={"returncode": 0, "ActiveState": "active"},
        production_lock={"owner_alive": True, "owner_identity_matches": True},
        memory_gib=64.0,
        free_gib=100.0,
        allow_member_processes=False,
        ensemble_controller_affinity=[13, 14, 15, 29, 30, 31],
    )

    assert reasons == []


def test_resource_gate_rejects_production_overlap_and_member_affinity_escape() -> None:
    topology = {cpu: cpu for cpu in range(16)} | {cpu + 16: cpu for cpu in range(16)}
    reasons = ensemble._admission_reasons(
        topology=topology,
        production_rows=[_process("prod", [0, 13], [0, 13])],
        ensemble_rows=[_process("member", [13, 15, 16], [13, 15, 16])],
        nightly={"returncode": 0},
        production_lock={},
        memory_gib=64.0,
        free_gib=100.0,
        allow_member_processes=True,
        ensemble_controller_affinity=[13, 14, 15, 29, 30, 31],
    )

    assert "production_model_outside_cores_0_11" in reasons
    assert "production_ensemble_physical_core_overlap" in reasons
    assert "ensemble_model_affinity_escape" in reasons


def _snapshot(
    rows: list[dict[str, object]],
    *,
    controller: str = "controller",
    deny_reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "deny_reasons": deny_reasons or [],
        "alisios": {
            "production_model_processes": rows,
            "controller_identity": {"identity_sha256": controller},
        },
        "cpu_ownership": {"observed_physical_cores": [13, 14, 15]},
    }


def test_continuous_gate_detects_expansion_identity_and_affinity_drift() -> None:
    baseline = _snapshot([_process("a", [0], [0]), _process("b", [1], [1])])
    observed = _snapshot([
        _process("a", [0], [0]),
        _process("b", [1], [17]),
        _process("c", [2], [2]),
    ])
    reasons = ensemble.resource_drift_reasons(baseline, observed)  # type: ignore[arg-type]

    assert "production_model_process_expansion" in reasons
    assert "production_model_logical_affinity_drift" in reasons

    contracted = _snapshot([_process("a", [0], [0])], controller="replacement")
    reasons = ensemble.resource_drift_reasons(baseline, contracted)  # type: ignore[arg-type]
    assert "production_model_process_identity_drift" in reasons
    assert "production_controller_identity_drift" in reasons


def test_resource_gate_fails_closed_on_capacity_or_unreadable_affinity() -> None:
    topology = {cpu: cpu for cpu in range(16)} | {cpu + 16: cpu for cpu in range(16)}
    reasons = ensemble._admission_reasons(
        topology=topology,
        production_rows=[_process("prod", [], [])],
        ensemble_rows=[],
        nightly={"returncode": 0},
        production_lock={},
        memory_gib=31.9,
        free_gib=39.9,
        allow_member_processes=False,
        ensemble_controller_affinity=[13, 14, 15, 29, 30, 31],
    )

    assert "production_model_affinity_unreadable" in reasons
    assert "available_ram_below_32_gib" in reasons
    assert "mnt_data_free_below_40_gib" in reasons


def test_resource_gate_rejects_unpinned_ensemble_controller() -> None:
    topology = {cpu: cpu for cpu in range(16)} | {cpu + 16: cpu for cpu in range(16)}
    reasons = ensemble._admission_reasons(
        topology=topology,
        production_rows=[_process("prod", list(range(12)), list(range(12)))],
        ensemble_rows=[],
        nightly={"returncode": 0, "ActiveState": "active"},
        production_lock={"owner_alive": True, "owner_identity_matches": True},
        memory_gib=64.0,
        free_gib=100.0,
        allow_member_processes=False,
        ensemble_controller_affinity=list(range(32)),
    )

    assert "ensemble_controller_affinity_escape" in reasons
