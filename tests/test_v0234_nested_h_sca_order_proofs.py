"""Adversarial gates for the h_sca candidate's target-aware HLO policy."""

from __future__ import annotations

import pytest

from scripts import v0234_nested_h_sca_order_hlo_audit as audit


def _call(target: str) -> str:
    return f"module {{ %0 = stablehlo.custom_call @{target}() : () -> tensor<i32> }}"


def test_exact_cpu_lapack_tridiagonal_target_is_the_only_allowed_target() -> None:
    assert audit.ALLOWED_CPU_DEVICE_LIBRARY_TARGETS == frozenset({"lapack_dgtsv_ffi"})
    verdict = audit.evaluate_policy(_call("lapack_dgtsv_ffi"))
    assert verdict["passed"] is True
    assert verdict["extraction"]["targets"] == ["lapack_dgtsv_ffi"]


@pytest.mark.parametrize(
    "near_match",
    [
        "lapack_dgtsv",
        "lapack_dgtsv_ffi_extra",
        "lapack_sgtsv_ffi",
        "LAPACK_DGTSV_FFI",
        "xla.cpu.lapack_dgtsv_ffi",
    ],
)
def test_every_near_match_is_unknown_and_fails_closed(near_match: str) -> None:
    verdict = audit.evaluate_policy(_call(near_match))
    assert verdict["passed"] is False
    assert verdict["unknown_custom_targets"] == [near_match]


def test_allowed_library_target_does_not_mask_callback_sibling() -> None:
    stablehlo = (
        "module {\n"
        " %0 = stablehlo.custom_call @lapack_dgtsv_ffi() : () -> tensor<i32>\n"
        " %1 = stablehlo.custom_call @xla_python_cpu_callback() : () -> tensor<i32>\n"
        "}\n"
    )
    verdict = audit.evaluate_policy(stablehlo)
    assert verdict["passed"] is False
    assert verdict["forbidden_custom_targets"] == ["xla_python_cpu_callback"]
    assert verdict["unknown_custom_targets"] == ["xla_python_cpu_callback"]


@pytest.mark.parametrize("token", audit.FORBIDDEN_TEXT)
def test_every_callback_and_transfer_token_fails_closed(token: str) -> None:
    verdict = audit.evaluate_policy(f"module {{ // {token}\n }}")
    assert verdict["passed"] is False
    assert token in verdict["forbidden_text_tokens"]


def test_malformed_custom_call_is_not_silently_ignored() -> None:
    verdict = audit.evaluate_policy(
        "module { stablehlo.custom_call %arg0 : tensor<i32> }"
    )
    assert verdict["passed"] is False
    assert verdict["extraction"]["extraction_complete"] is False
