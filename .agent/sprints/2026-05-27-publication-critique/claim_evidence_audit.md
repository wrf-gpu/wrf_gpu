# Claim-Evidence Audit

Scope: `publication/draft/paper.md`, cross-checked against `publication/draft/references.bib`, `publication/research_brief/english_brief.txt`, and `.agent/sprints/2026-05-27-publication-first-draft/honesty_audit.md`.

## Highest-Risk Claim Gaps

| Claim | Draft location | Support found | Verdict |
|---|---:|---|---|
| 24 h d02 pipeline completed in `324.78 s`; speedup `50.20x` | Abstract, contribution 2, Section 7, Discussion | Supported by `pipeline_run_20260521.json` and `honest_speedup_table.json`, but this is pre-fix incorrect-physics path | Backed but misleading as current headline |
| Corrected-physics post-fix run is `708.32 s` and `23.02x` | Section 7 table, Section 8.2 | Supported by `post_fix_speedup.json` and `pipeline_run_20260521.json` | Should be headline current result |
| "Wall-clock cost of running correct physics is `700.89 s` end-to-end" | Section 8.2 | Worker report says first run `700.885...` was superseded; current pipeline JSON says `708.317...` total and `700.731...` forecast-only | Fix metric name and number |
| "Architecture is correct" | Section 8.2 | Device residency/restart/D2H evidence exists, but skill still fails and known defects remain | Overclaim; soften |
| `7278 MiB` is "peak process memory" | Abstract, Section 6, Table 1 | `step_feasibility.json` records `nvidia-smi` memory.used after warm, not a peak allocator trace | Rephrase |
| Radiation cadence effectively disabled | Section 5 and Limitations | True for pre-fix namelist `999999`; post-fix path uses `180` | Mark as pre-fix only |
| Root-cause analysis remains open | Acknowledgements | RCA and partial-fix sprints now exist | Stale |
| `5-14x` AceCAST speedups | Related Work | Cited only to generic docs/manual from brief | Needs exact source or softer wording |
| `3.5-4x` Pace, `5.5x` ICON, `1.26 SYPD` SCREAM, `34x` NIM | Related Work | Bib entries exist; several rely on brief-extracted numbers | Needs final bibliographic verification |
| Python `3.13.11`, JAX `0.10.0` | Section 6, Limitations | First-draft honesty audit says command was run in that sprint | Backed but needs release manifest |
| Public code URL `github.com/<TBD>` | Reproducibility | Placeholder | Submission blocker |

## Supported Quantitative Claims

- Warm 1 h 20260521 wall time `5.706... s`: `wall_clock.json`, `reproducibility_v2.json`.
- Three-run warm CV `0.42 percent`: `reproducibility_v2.json` (`0.004248...`).
- CPU d02-only denominator `16305.31132 s`: `honest_speedup_table.json`.
- Five-domain aggregate framing `138.24x`, explicitly not apples-to-apples: `honest_speedup_table.json`.
- D2H inter-kernel count/bytes `0`: `d2h_audit_v2.json`. Caveat: post-fix worker report says the original `.nsys-rep` binary is absent in this checkout and the proof depends on parsed JSON.
- Restart continuity max delta `0.0`: `restart_continuity.json` and post-fix `invariant_preservation.json`.
- 24/24 wrfouts readable: `wrfout_inventory.json`.
- AEMET side-by-side counts: `73` stations, `24` common hours, `1747` joined rows in `gpu_vs_cpu_skill_diff.json`.
- Pre-fix RMSE values T2 `2.15/7.86 K`, U10 `2.31/11.31 m s-1`, V10 `2.75/9.44 m s-1`: `gpu_vs_cpu_skill_diff.json`.
- L2 d02 bounded failures T2 `4.07 K`, U10 `10.78 m s-1`, V10 `7.83 m s-1`: `tier4_rmse_l2_d02.json`.
- Post-fix skill remains outside tolerance: `post_fix_skill_diff.json`.

## Missing Or Weak Evidence Links

- The draft cites several proof paths in prose but not next to the grid/timestep values in Section 4.
- "Six of nine metrics improved" appears in the post-fix worker report, but the JSON queried during this sprint did not expose `metrics_improved` / `metrics_worsened` as top-level fields. Either cite the worker report directly or add the summary to JSON.
- "Still 3x above the 4-8x target" needs a precise local target citation and clearer math.
- "AEMET observations" uses a generic AEMET homepage citation. The observation source, product, license, station metadata, and retrieval date should be documented.

## Bottom Line

Most numbers are backed by local proof objects, but the main narrative uses the wrong current performance number. The safest framing is: pre-fix system proved device residency and high speed but failed skill; post-fix system preserves systems invariants, improves some metrics, costs `708.32 s` / `23.02x`, and still fails skill tolerance.
