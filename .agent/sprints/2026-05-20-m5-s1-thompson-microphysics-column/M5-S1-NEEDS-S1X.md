# M5-S1 Needs S1.x - Thompson Table/Proxy Residuals

## Decision

Attempt 5 fixed the ice `lami` clamp numerator in `src/gpuwrf/physics/thompson_column.py:278-279` using `CIE2` from `src/gpuwrf/physics/thompson_constants.py:44`, then accepted the graupel `cge(11)/cgg(11)` source-truth correction at `src/gpuwrf/physics/thompson_constants.py:71-94` and `src/gpuwrf/physics/thompson_column.py:464,493`. The strict ADR-005 parity target still did not close. The current fixture manifest therefore keeps attempt carry-forward tolerances for the M5-S1 artifact gate at `fixtures/manifests/analytic-thompson-column-v1.yaml:157-254`, and the strict residuals below remain the scope for M5-S1.x.

## Strict Residual Fields

Strict ADR-005 tolerances are `abs=1e-10, rel=1e-8` for hydrometeor mixing ratios and `abs=1e-3, rel=1e-6` for `Ni/Nr` per `.agent/decisions/ADR-005-first-physics-suite.md:27`. Current post-fix residuals in `artifacts/m5/tier1_thompson_parity.json:15-35` still violate those thresholds:

| Field | Max abs | Max rel | Note |
|---|---:|---:|---|
| `qv` | `1.5091010112033235e-05` | `0.004311168982146883` | coupled vapor/process-order residual |
| `qc` | `0.0001519672247928526` | `0.9999988999202443` | cloud-water table/proxy residual |
| `qr` | `4.760876436193939e-06` | `45035996.27370496` | rain transfer residual at near-zero reference cells |
| `qi` | `0.00013714003800346464` | `0.30505020185045534` | ice table/proxy residual |
| `qs` | `0.0001447943623500526` | `1249.636404283526` | snow table/proxy residual |
| `qg` | `1.651181415760252e-05` | `986570506.4244672` | graupel-coupling table-proxy residual at near-zero reference cells |
| `Ni` | `126975.12500000041` | `0.3883326751538069` | ice-number residual |
| `Nr` | `67300.453125` | `45425309332488.81` | rain-number residual at near-zero reference cells |
| `T` | `0.04250859614040792` | `0.00015893814657304574` | process/table and graupel-coupling residual; manifest carry-forward tolerance only |

Near-zero-reference caveat: for `qr`, `qs`, `qg`, and `Nr` (and intermittently `qc`), the max relative errors are dominated by WRF reference cells that are zero or nearly zero. The max absolute errors are the load-bearing S1.x metric for those fields; relative errors remain in the table for completeness. This follows the validation framing in `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md`.

## Required S1.x Scope

M5-S1.x should export or otherwise match the WRF Thompson lookup/moment paths deferred by attempt 4: `t_Efrw`, `tps_iaus`, `tni_iaus`, rain-freezing tables, snow/graupel moments, graupel-coupling table-proxy residuals, and the related number/mass balance details. The M5-S1 attempt-6 gate is `GO_CARRYFORWARD` under carry-forward tolerances (`artifacts/m5/thompson_gate_result.json:2-9`), but this note is the strict-parity handoff and should be resolved before M5-S2 depends on exact Thompson parity.
