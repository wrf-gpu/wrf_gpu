# Release evidence and provenance

Measurements below use v0.3.0 (internal build dead7d004; internal src/gpuwrf tree 7a353f10ac1f). The curated public source preserves its numerical code. Every receipt retains its source, input hashes, clocks and original GPU proof. Raw strict FAIL reports and exact manager-approved disclosures are stored together.

CPU energy uses approximately 200 W on twelve cores (maintainer measurement); Swiss four-core energy is unmeasured. GPU energy is board power; host CPU is excluded. The [methods](METHODS.md) distinguish the six-hour fully concurrent benchmark from the separately labelled six-case 72 h FIFO production rate. B200 scenarios are inferred from old-version measurements.

| Result | Value | Evidence and label |
|---|---|---|
| Canary (PROD) GPU s per forecast hour, 24 h whole run | 4.12 | [M] [prod_publication_times.json](evidence/prod_publication_times.json) last file written − process start, ÷ 24 |
| Canary speedup vs CPU-WRF, 12 cores | 22.4 | [M] CPU-WRF 12 ranks 92.3 s/fc-h ÷ GPU |
| speedup vs the 2026-10-01 starting tree (3 h warm proxy) | 32 | [M] 107.6 s/fc-h on the starting tree ÷ S1 proxy 3.369 ([prod_bench.json](evidence/prod_bench.json) s1_warm_s_per_fc_h) |
| Tenerife one case, GPU s per simulated hour (steady) | 7.4 | [M] [wn3_solo_analysis.json](evidence/wn3_solo_analysis.json) throughput_stepping_s_per_case_h |
| Tenerife one-case speedup vs CPU-WRF, 4 cores | 49.7 | [M] CPU-WRF 4 ranks 370.15 s ÷ GPU |
| Tenerife cases run in parallel (max admitted) | 4 | [M] [wn3_parallel_analysis.json](evidence/wn3_parallel_analysis.json) |
| Tenerife GPU s per case-hour at N_max, 6 h benchmark | 9.68 | [M] arm start ([wn3_parallel_start.json](evidence/wn3_parallel_start.json)) → whole launcher completion, including output compression and cleanup ([wn3_receipt_20260614_18z_a1.json](evidence/wn3_receipt_20260614_18z_a1.json), [wn3_receipt_20260227_18z_a1.json](evidence/wn3_receipt_20260227_18z_a1.json), [wn3_receipt_20260502_18z_a1.json](evidence/wn3_receipt_20260502_18z_a1.json), [wn3_receipt_20260220_18z_a1.json](evidence/wn3_receipt_20260220_18z_a1.json)) ÷ (N × 6) |
| Tenerife throughput speedup vs CPU-WRF, 12 cores | 12.75 | [M] CPU-WRF 3 cases × 4 cores 123.38 s per case-hour ÷ GPU |
| Tenerife 72 h production batch, six cases in FIFO waves, s per case-hour | 6.28 | [M] [wn3_production_batch.json](evidence/wn3_production_batch.json) |
| 72 h production throughput vs CPU-WRF, 12 cores | 19.6 | [M] [wn3_production_batch.json](evidence/wn3_production_batch.json) |
| GPU memory per Tenerife case (process peak, GiB) | 4.4 | [M] [wn3_parallel_analysis.json](evidence/wn3_parallel_analysis.json) cases/per_case.vram_peak_mib (max) |
| Tenerife cases that fit one 32 GB card | 4 | [M] admission limit of the N sweep |
| CPU-WRF energy per case-hour (kJ) | 25 | [M] maintainer measurement: CPU-WRF on 12 cores ≈200 W |
| GPU board energy per case-hour at N_max (kJ) | 2.9 | [M] [wn3_parallel_dmon.json](evidence/wn3_parallel_dmon.json) energy_j ÷ (N × hours) |
| component energy ratio CPU package/GPU board | 8.6 | [M] CPU [M] ÷ GPU [M] |
| release-gate cases (24 h, all variables) | 6 | [M] [gate_d6_20260227_18z_a1_d6_24h.json](evidence/gate_d6_20260227_18z_a1_d6_24h.json), [gate_d6_20260502_18z_a1_d6_24h.json](evidence/gate_d6_20260502_18z_a1_d6_24h.json), [gate_d6_20260614_18z_a1_d6_24h.json](evidence/gate_d6_20260614_18z_a1_d6_24h.json), [gate_d6_20260220_18z_a1_d6_24h.json](evidence/gate_d6_20260220_18z_a1_d6_24h.json), [gate_d6_20260608_18z_a1_d6_24h.json](evidence/gate_d6_20260608_18z_a1_d6_24h.json), [gate_d6_20260120_18z_a1_d6_24h.json](evidence/gate_d6_20260120_18z_a1_d6_24h.json) |
| variables compared per domain | 375 | [M] inventory.common_variable_count (minimum over domains) |
| variables with numerical tolerances per domain | 62 | [M] fields with supplied frozen D6 bounds (minimum over domains) |
| release-gate tolerance failures | 0 | [M] sum of summaries.tolerance_failure_count |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_1.json](evidence/output_integrity_1.json); disclosed exception [output_integrity_exception_1.json](evidence/output_integrity_exception_1.json); approval [output_integrity_approval_1.json](evidence/output_integrity_approval_1.json) |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_2.json](evidence/output_integrity_2.json); disclosed exception [output_integrity_exception_2.json](evidence/output_integrity_exception_2.json); approval [output_integrity_approval_2.json](evidence/output_integrity_approval_2.json) |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_3.json](evidence/output_integrity_3.json); disclosed exception [output_integrity_exception_3.json](evidence/output_integrity_exception_3.json); approval [output_integrity_approval_3.json](evidence/output_integrity_approval_3.json) |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_4.json](evidence/output_integrity_4.json); disclosed exception [output_integrity_exception_4.json](evidence/output_integrity_exception_4.json); approval [output_integrity_approval_4.json](evidence/output_integrity_approval_4.json) |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_5.json](evidence/output_integrity_5.json); disclosed exception [output_integrity_exception_5.json](evidence/output_integrity_exception_5.json); approval [output_integrity_approval_5.json](evidence/output_integrity_approval_5.json) |
| WRF output integrity | raw FAIL; approved exception | [M] [output_integrity_6.json](evidence/output_integrity_6.json); disclosed exception [output_integrity_exception_6.json](evidence/output_integrity_exception_6.json); approval [output_integrity_approval_6.json](evidence/output_integrity_approval_6.json) |
| cases in the long-lead identity curves | 6 | [M] [identity_curves_wn3.json](evidence/identity_curves_wn3.json) |
| worst gate variable, % of its D6 limit, 0–24 h | 61 | [M] [identity_curves_wn3.json](evidence/identity_curves_wn3.json) |
| PROD worst core-field RMSE, % of reference limit | 78 | [M] [identity_curves_prod162.json](evidence/identity_curves_prod162.json) |
| B200 Tenerife case-hours per hour | 2,380 | [I] extrapolated from B200 runs on old version ([b200_extrapolation.json](evidence/b200_extrapolation.json); method docs/release/B200_EXTRAPOLATION.md) |
| B200 energy per case-hour (kJ) | 1.0 | [I] extrapolated from B200 runs on old version ([b200_extrapolation.json](evidence/b200_extrapolation.json); method docs/release/B200_EXTRAPOLATION.md) |
| Tenerife cases fitting a B200 | 30 | [I] 180 GB ÷ (case peak + 1 GiB) |

The [evidence index](evidence/INDEX.json) retains original artifact paths. The [validation report](VALIDATION.md) distinguishes all-six-case 24 h v0.3.0 results, the complete predecessor 72 h set, and dated v0.3.0 72 h addenda.
