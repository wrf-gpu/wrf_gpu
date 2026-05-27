# Precision Downcast Audit

Sources:
- `src/gpuwrf/contracts/precision.py`
- `PRECISION_POLICY.md`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md`

Policy baseline: mixed precision requires explicit validation by variable and scheme. BF16/FP16 in physics is forbidden unless acceptance tests prove safety. The current precision registry already stores `u`, `v`, `theta`, `qv`, Thompson hydrometeors/number fields, `qke`, land masks, and wind/theta/qv lateral boundaries as FP32-gated fields. The audit below lists every current State FP64 field.

| Field | 1 km State MiB | Judgment | FP32 storage saving | BF16 judgment |
|---|---:|---|---:|---|
| `w` | 32.425 | Candidate-FP32 only after sound-wave and Tier-4 tests; currently not authorized. | 16.213 MiB | Not a BF16 candidate. |
| `p` | 31.705 | Must-stay-FP64; pressure-gradient/acoustic-sensitive. | 15.852 MiB rejected | Not a BF16 candidate. |
| `p_total` | 31.705 | Must-stay-FP64; authoritative pressure state. | 15.852 MiB rejected | Not a BF16 candidate. |
| `p_perturbation` | 31.705 | Must-stay-FP64; perturbation pressure feeds acoustic/PGF terms. | 15.852 MiB rejected | Not a BF16 candidate. |
| `ph` | 32.425 | Must-stay-FP64; geopotential/vertical metrics are pressure-gradient-sensitive. | 16.213 MiB rejected | Not a BF16 candidate. |
| `ph_total` | 32.425 | Must-stay-FP64. | 16.213 MiB rejected | Not a BF16 candidate. |
| `ph_perturbation` | 32.425 | Must-stay-FP64. | 16.213 MiB rejected | Not a BF16 candidate. |
| `mu` | 0.721 | Must-stay-FP64; dry-column mass. | 0.360 MiB rejected | Not a BF16 candidate. |
| `mu_total` | 0.721 | Must-stay-FP64; dry-column mass. | 0.360 MiB rejected | Not a BF16 candidate. |
| `mu_perturbation` | 0.721 | Must-stay-FP64; dry-column mass perturbation. | 0.360 MiB rejected | Not a BF16 candidate. |
| `ustar` | 0.721 | Candidate-FP32; surface handle, low memory value. | 0.360 MiB | Not worth BF16. |
| `theta_flux` | 0.721 | Candidate-FP32 after surface/PBL budget proof. | 0.360 MiB | Not worth BF16. |
| `qv_flux` | 0.721 | Candidate-FP32 after water-budget proof. | 0.360 MiB | Not worth BF16. |
| `tau_u` | 0.721 | Candidate-FP32 after near-surface wind proof. | 0.360 MiB | Not worth BF16. |
| `tau_v` | 0.721 | Candidate-FP32 after near-surface wind proof. | 0.360 MiB | Not worth BF16. |
| `rhosfc` | 0.721 | Candidate-FP32 after surface/PBL proof. | 0.360 MiB | Not worth BF16. |
| `fltv` | 0.721 | Candidate-FP32 after surface/PBL proof. | 0.360 MiB | Not worth BF16. |
| `t_skin` | 0.721 | Candidate-FP32; low risk relative to pressure fields, but land refresh and T2 skill must pass. | 0.360 MiB | Not worth BF16. |
| `soil_moisture` | 0.721 | Candidate-FP32; low memory value. | 0.360 MiB | Not worth BF16. |
| `roughness_m` | 0.721 | Candidate-FP32; prescribed/derived surface roughness. | 0.360 MiB | Not worth BF16. |
| `rain_acc` | 0.721 | Keep FP64 unless output accumulation tests authorize FP32. | 0.360 MiB low value | Not a BF16 candidate. |
| `snow_acc` | 0.721 | Keep FP64 unless output accumulation tests authorize FP32. | 0.360 MiB low value | Not a BF16 candidate. |
| `graupel_acc` | 0.721 | Keep FP64 unless output accumulation tests authorize FP32. | 0.360 MiB low value | Not a BF16 candidate. |
| `ice_acc` | 0.721 | Keep FP64 unless output accumulation tests authorize FP32. | 0.360 MiB low value | Not a BF16 candidate. |
| `ph_bdy` | 0.656 | Must-stay-FP64; geopotential boundary forcing. | 0.328 MiB rejected | Not a BF16 candidate. |
| `mu_bdy` | 0.015 | Must-stay-FP64; dry-mass boundary forcing. | 0.007 MiB rejected | Not a BF16 candidate. |

Aggregate memory implications at the derived full-domain 1 km shape:
- FP64 pressure/geopotential/mass fields locked by policy: 195.224 MiB.
- Plausible FP64-to-FP32 candidates, including `w` and surface handles: 42.513 MiB current storage, 21.257 MiB saving if all land.
- Surface-only FP64 candidates excluding `w`: 10.088 MiB current storage, 5.044 MiB saving. This is too small to drive a sprint by itself.
- Existing FP32 advected-scalar/hydrometeor/BF16 candidates would save ~95.436 MiB if moved from FP32 to BF16, but this is explicitly fail-closed by `PRECISION_POLICY.md` until physics acceptance tests prove safety.

Recommendation:
Precision is not the top bottleneck for current 3 km wall clock. The only large single FP64 candidate is `w`, and it touches acoustic stability. Downcast work should follow launch/fusion and memory-profile work, with a narrow proof plan per field.
