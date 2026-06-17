"""Shared helper for the operational-source no-host-transfer guards.

Several tests statically grep ``src/gpuwrf/runtime/operational_mode.py`` to
enforce the constitutional rule: NO host<->device transfer inside the per-step
timestep loop. v0.18 added one legitimate, OUT-OF-LOOP host pull:

    def _assert_nonzero_initial_mu_total(state: State) -> None:
        ...
        max_abs = float(jax.device_get(jnp.max(jnp.abs(jnp.asarray(state.mu_total)))))
        ...

This is a ONE-TIME fail-loud pre-flight check that pulls ``state.mu_total`` to
the host BEFORE the chunked host timestep loop is entered. Its two call sites
(``_committed_initial_carry_for_run`` and ``run_forecast_operational``) both run
once per forecast, never inside the compiled ``jax.lax.scan`` / ``_advance_chunk``
body (independently verified). It is the ONLY ``device_get`` in the file.

``strip_preflight_mu_total_check`` removes exactly that helper's function body
(from its ``def`` line to the next top-level ``def``) so the literal-token grep
becomes loop-precise. It does NOT touch any other code, so any other host
transfer anywhere else in the file still fails the guard. The constitutional
guarantee is unchanged; this only stops a false positive on a verified
one-time, out-of-loop check.
"""

from __future__ import annotations

import re

_PREFLIGHT_DEF = "def _assert_nonzero_initial_mu_total("


def strip_preflight_mu_total_check(source: str) -> str:
    """Return ``source`` with the one-time mu_total pre-flight helper removed.

    Asserts the helper exists (so the exemption can never silently widen to cover
    unrelated code if the helper is renamed/removed) and removes only its body.
    """

    assert _PREFLIGHT_DEF in source, (
        "expected the documented one-time _assert_nonzero_initial_mu_total helper; "
        "if it was renamed/removed, update this guard exemption deliberately"
    )
    # Match from the helper's def to (but not including) the next top-level def.
    pattern = re.compile(
        r"\ndef _assert_nonzero_initial_mu_total\(.*?(?=\ndef )",
        re.DOTALL,
    )
    stripped = pattern.sub("\n", source, count=1)
    # The exempted helper is the only legitimate device_get; nothing else should
    # remain that the grep would have to special-case.
    return stripped
