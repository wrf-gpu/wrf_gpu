# Reviewer Report: V0.14 Grid-Delta Tolerance Envelope

Decision: ACCEPT.

The manifest follows the sprint contract and is suitable as the v0.14
pre-result tolerance candidate. It relies on documented existing hard RMSE bars
instead of tuning after seeing new validation output. It also preserves release
honesty by keeping `P`, `PH`, `MU`, `RAINC`, and broader diagnostics report-only
unless a separate pre-result threshold review freezes additional limits.

Important review findings:

- The old v0.12 red case still fails under the manifest, so this is not a
  permissive backfit.
- The static field group and parser-visible `fields` map are now internally
  consistent.
- The atlas tooling can parse the manifest in offline smoke mode.
- The manifest does not by itself enforce every mandatory-field policy; final
  scoring must preserve the atlas builder's default mandatory field list or pass
  explicit `--mandatory-field` options if that tool changes.

Residual risk:

- Pooled RMSE can hide localized drift. Final v0.14 release evidence must still
  include lead-time trend, p99/max, worst-cell maps, and inventory reporting.
- Report-only critical fields can still block release by manager decision if
  they show systematic physical drift even without frozen numeric thresholds.
