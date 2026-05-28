"""Comprehensive diagnostic harness for the operational forecast loop.

See ``.agent/sprints/2026-05-28-diagnostic-harness/design.md`` for the
schema, operator list, invariant list, and overhead budget.

The harness is **opt-in** via a static-argname ``diagnostic_on: bool`` flag so
that XLA dead-code-eliminates the entire instrumentation tree in production.
"""

from gpuwrf.diagnostics.comprehensive_harness import (
    DIAGNOSTIC_FIELD_INDEX,
    DIAGNOSTIC_OPERATORS,
    DIAGNOSTIC_SCHEMA_VERSION,
    DiagnosticAccumulator,
    DiagnosticReport,
    build_diagnostic_report,
    initial_diagnostic_accumulator,
    instrumented_physics_boundary_step,
    run_diagnostic_forecast,
)

__all__ = [
    "DIAGNOSTIC_FIELD_INDEX",
    "DIAGNOSTIC_OPERATORS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticAccumulator",
    "DiagnosticReport",
    "build_diagnostic_report",
    "initial_diagnostic_accumulator",
    "instrumented_physics_boundary_step",
    "run_diagnostic_forecast",
]
