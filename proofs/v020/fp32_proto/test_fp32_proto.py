from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acceptance_bands import check_metric_band, check_universal_hard_gates, field_group
from fp32_column_proto import (
    CASE_NAMES,
    accumulator_drift_experiment,
    build_case,
    cancellation_metrics,
    run_all_cases,
)


def test_acceptance_bands_are_importable_and_enforce_thresholds():
    assert field_group("U10") == "wind"
    assert field_group("T2") == "temperature"
    assert field_group("QVAPOR") == "qvapor"
    assert field_group("RAINNC") == "cumulative"

    assert check_metric_band("U10", 24.0, {"rmse_increase": 0.24}).passes
    assert not check_metric_band("U10", 24.0, {"rmse_increase": 0.26}).passes
    assert check_metric_band("T2", 72.0, {"rmse_increase": 0.99}).passes
    assert not check_metric_band("T2", 72.0, {"rmse_increase": 1.01}).passes
    assert check_metric_band("QVAPOR", 120.0, {"domain_mean_water_vapor_rel_diff": 0.49}).passes
    assert not check_metric_band("RAINNC", 72.0, {"domain_total_ratio": 4.2}).passes

    ok, failures = check_universal_hard_gates({"PSFC": np.array([90_000.0])})
    assert ok, failures
    ok, failures = check_universal_hard_gates({"PSFC": np.array([30_000.0])})
    assert not ok
    assert failures


def test_naive_total_fp32_reproduces_cancellation_failure():
    ratios = []
    for case_name in CASE_NAMES:
        metrics = cancellation_metrics(build_case(case_name))
        for field_name, field_metrics in metrics.items():
            ratio = field_metrics["naive_over_perturbation"]
            ratios.append(ratio)
            assert ratio > 27.0, (case_name, field_name, field_metrics)

    assert min(ratios) > 27.0
    assert np.median(np.asarray(ratios)) > 100.0


def test_perturbation_form_tracks_reference_over_many_substeps():
    results = run_all_cases(nsteps=3_000)

    assert results["summary"]["perturbation_cases_passing_bands"] == len(CASE_NAMES)
    assert results["summary"]["min_naive_over_perturbation"] > 27.0

    for case_result in results["cases"]:
        assert case_result["stability"]["all_finite"]["reference"]
        assert case_result["stability"]["all_finite"]["perturbation"]
        perturb = case_result["modes"]["perturbation"]
        assert perturb["all_band_checks_pass"], (case_result["case"], perturb["band_checks"])
        assert perturb["metrics"]["W"]["rmse_increase"] < 0.01
        assert perturb["metrics"]["P"]["psfc_rmse_increase"] < 1.0
        assert perturb["metrics"]["MU"]["dry_mass_relative_drift"] < 1.0e-5


def test_compensated_accumulator_cuts_drift_by_100x_plus():
    drift = accumulator_drift_experiment(nsteps=80_000)

    assert drift["naive_fp32_abs_err"] > 0.0
    assert drift["kahan_fp32_abs_err"] > 0.0
    assert drift["naive_over_kahan_err"] > 100.0
