# ARCHIVE MANIFEST — v0.16 fp32 speed-up-ceiling proof (Beweisführung)

Complete, self-contained public archive of the v0.16 fp32 make-or-break verdict.
Docs-only; no model code. Every artifact a reader needs lives in this folder.

**Verdict:** valid-numerics fp32 speed ceiling on the RTX 5090 = **~1.1×** (not
~4×/~6×), **0 % VRAM-peak reduction from precision alone**, double-confirmed (Opus
implementation + independent GPT reproduction, both *IMPOSSIBILITY CONFIRMED*).

## Index / entry point
- `README.md` — the authoritative Beweisführung: the verdict, the three measured
  pillars, why the old 4.3×/6× numbers are not the ceiling, the real wins, and a
  linked walk-through of every artifact below.

## Verdict reports (Markdown)
| File | Source (dev worktree) | Role |
|---|---|---|
| `opus-fullws-fp32-verdict.md` | (already in v0.16 public tree) | Opus full-working-set implementation + measured verdict (outcome (b)). **KEPT, unchanged.** |
| `gpt-fullws-fp32-crosscheck.md` | (already in v0.16 public tree) | Independent GPT reproduction + adversarial refutation (IMPOSSIBILITY CONFIRMED). **KEPT, unchanged.** |
| `compact-island-fp32.md` | `.wt-fp32-compact/.agent/reviews/2026-06-14-opus-compact-island-fp32.md` | The COMPACT explicit-fp64-island technique proof + the 4-way VRAM-peak root-cause control. |

## Roofline / ceiling derivation (Markdown)
| File | Source (dev worktree) | Role |
|---|---|---|
| `roofline-ceiling-and-levers.md` | `.wt-v017-gap/.agent/decisions/PERFORMANCE-CEILING-AND-LEVERS.md` | The roofline ceiling derivation + the full architecture-lever table (BouLac O(nz), fusion, multi-GPU scaling envelope). |
| `roofline-blind-crosscheck.md` | `.wt-v016-dash/.agent/reviews/2026-06-14-opus-roofline-check.md` | The BLIND independent roofline re-derivation (corrected naive "~2×" to the ~4.2–4.6× on-card / numerically-invalid ceiling; physically un-exceedable for AI≈1). |
| `reduced-precision-equivalence-criterion.md` | `.wt-v017-gap/.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md` | The binding equivalence-criterion / scientific-rigor methodology (long-horizon non-escalating divergence; conservation state stays fp64-locked). |

## Measured proof objects — `measurements/` (JSON)
| File | Source (dev worktree) | What it proves |
|---|---|---|
| `fullws_fp32_km_bench.json` | `.wt-fp32-full/proofs/perf/v016/` | GPU bench, real Switzerland d01: 16 k 1.107× / 65 k 1.110×, VRAM ratio 1.000; 147 k OOM. |
| `fullws_safe_km_bench.json` | `.wt-fp32-full/proofs/perf/v016/` | The numerically-defensible `safe` lane (pert+w only): 16 k 1.108×, VRAM 1.000, `[]`-unlock. |
| `fullws_base_absolute_oracle.json` | `.wt-fp32-full/proofs/perf/v016/` | **Pillar 2** — fp32 storage of base absolutes corrupts geopotential/PGF 26.85×/126.75× the gated-fp32 floor; `GATE_PASS=false`. |
| `fullws_boulac_onz_147k.json` | `.wt-fp32-full/proofs/perf/v016/` | The orthogonal 1 km-unlock: full-ws + BouLac-O(nz) runs 147 k cols finite at 21.31 GiB. |
| `gpt_fullws_reproduce.json` | `.wt-fp32-full/proofs/perf/v016/` | Independent GPT GPU reproduction: 1.105× / 1.111×, VRAM 1.000, `[]`-unlock. |
| `gpt_fullws_transient_memory_probe.json` | `.wt-fp32-full/proofs/perf/v016/` | **Pillar 1** — XLA GPU `memory_analysis`: transient `temp_size` 1975→2001 MiB unchanged; persistent `argument_size` 366→247 MiB shrank. |
| `gpt_fullws_transient_memory_probe_cpu.json` | `.wt-fp32-full/proofs/perf/v016/` | CPU-lowering cross-probe of Pillar 1 (`temp_size` 5305→5379 MiB unchanged). |
| `gpt_double_single_probe.json` | `.wt-fp32-full/proofs/perf/v016/` | Double-single recovery = fp64-equivalent storage + ~16× time. |
| `gpt_safe_alias_probe.json` | `.wt-fp32-full/proofs/perf/v016/` | Why the `safe` set must exclude `p`/`ph` (the `State.replace` alias-sync would re-demote totals). |
| `fp32_s2_mixed_ladder.json` | `proofs/perf/v016/` (main) | S2 mixed-perturb-fp32 ladder: 1.102×/1.113×, VRAM 1.000, 147 k OOM. |
| `compact_fp32_km_bench.json` | `.wt-fp32-full/proofs/perf/v016/` | Acoustic-only COMPACT bench (first pass; jit-cache-alias note). |
| `compact_fp32_km_bench_CORRECTED.json` | `.wt-fp32-full/proofs/perf/v016/` | COMPACT bench with per-precision cache clear: 1.109×/1.105×, VRAM 1.000, `[]`-unlock. |
| `native_fp32_km_bench.json` | `.wt-fp32-full/proofs/perf/v016/` | The earlier NATIVE/type-promotion attempt: 1.010× (cache-alias; +316% convert-scatter ops) — the convert-scatter failure mode. |
| `calc_p_rho_fp32_oracle.json` | `proofs/perf/v015/fp32_oracles/` | In-loop EOS island oracle — genuine-fp32 product within the fp32 noise floor when fed fp64 operands. |
| `advance_w_fp32_oracle.json` | `proofs/perf/v015/fp32_oracles/` | In-loop `advance_w` island oracle — finite / non-escalating on flat/moderate/steep terrain. |
| `compact_island_fp32_oracle.json` | `.wt-fp32-compact/proofs/perf/v016/` | The COMPACT design oracle (real b6 column): fp32 already-cancelled product matches the fp64-carrying form to the fp32 floor. |

## Top-level README link
The repo `README.md` Performance section now carries a prominent standalone callout
block ("📐 The full speed-up-ceiling proof (the *Beweisführung*)") pointing to
`proofs/v016/fp32_verdict/`, plus the in-table Performance-row pointer naming it the
Beweisführung. Both include the three-pillar one-liner.

## Provenance / hygiene
- All JSONs are small (< 25 KB each); the large `nsys-rep`/`sqlite`/`.npz` profiler
  blobs were intentionally **not** copied (the JSON summaries carry the numbers).
- The two pre-existing reports (`opus-fullws-fp32-verdict.md`,
  `gpt-fullws-fp32-crosscheck.md`) were kept byte-for-byte.
- Docs-only change; no `src/` modification.
