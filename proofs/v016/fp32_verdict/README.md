# v0.16 fp32 speed-up-ceiling proof — the full Beweisführung

This folder is the **complete, self-contained proof** of the v0.16 fp32 make-or-break:
the question of whether reduced precision (fp32) buys the project the large speed-up
and VRAM reduction it had hoped for on a single **RTX 5090**. It does not. This is the
evidence trail, end to end — roofline derivation, two independent cross-checks, the
numerical-soundness oracles, every measured benchmark JSON, and a double-confirmed
verdict.

Everything referenced here lives **inside this folder** (`measurements/` holds the JSON
proof objects), so the public release is self-contained — no dev-worktree paths.

---

## THE VERDICT (double-confirmed)

> **The valid-numerics fp32 speed ceiling on this RTX 5090 is `~1.1×`, NOT `~4×` and
> NOT `~6×`. fp32 delivers `0%` peak-VRAM reduction from precision alone. This is
> DOUBLE-CONFIRMED: an Opus full-working-set implementation and an independent GPT
> reproduction both reached "IMPOSSIBILITY CONFIRMED" — neither could find any
> valid-numerics lever past `~1.1×` / `1.000×` VRAM.**

Measured, real Switzerland d01, RTX 5090 ([`measurements/fullws_fp32_km_bench.json`](measurements/fullws_fp32_km_bench.json),
[`measurements/gpt_fullws_reproduce.json`](measurements/gpt_fullws_reproduce.json)):

| lane | 16,384 col | 65,536 col | VRAM ratio | unlocks 147 k (1 km)? |
|---|---|---|---|---|
| Opus full-working-set fp32 | **1.107×** | **1.110×** | **1.000** | no (`[]`) |
| GPT independent reproduce | **1.105×** | **1.111×** | **1.000** | no (`[]`) |
| numerically-defensible `safe` lane | **1.108×** | — | **1.000** | no (`[]`) |
| acoustic-only COMPACT lane | 1.109× | 1.105× | 1.000 | no (`[]`) |
| S2 mixed-perturb-fp32 ladder | 1.102× | 1.113× | 1.000 | no (OOM) |

Every numerically-valid fp32 lane lands at **~1.1× / 1.000× VRAM**. The lanes that do
better are not numerically valid (next section).

---

## THE THREE MEASURED PILLARS (why fp32 cannot do better with valid numerics)

### Pillar 1 — Demoting the **persistent State** to fp32 = `0%` VRAM (the peak is transient)

The full-working-set lever demoted the **entire broad persistent State family**
(`p`/`p_total`/`p_perturbation`/`ph`/`ph_total`/`ph_perturbation`/`w`) to fp32 — removing
**−240 MiB @16 k / −700 MiB @65 k** of carried fp64 3D arrays. **The peak VRAM did not
move: 3.73 GiB → 3.73 GiB @16 k, 11.43 GiB → 11.43 GiB @65 k (ratio 1.000).**

The reason is direct from XLA's own memory analysis (GPU,
[`measurements/gpt_fullws_transient_memory_probe.json`](measurements/gpt_fullws_transient_memory_probe.json)):
the **transient** high-water mark (`temp_size`) is **precision-insensitive** — it goes
**1975 → 2001 MiB** (fullws marginally *higher*) while the **persistent** `argument_size`
shrank **366 → 247 MiB**. The buffers that went fp32 are not the ones at the live-range
peak; the peak is set by fp64 compute that must stay fp64. (The CPU-lowering cross-probe,
[`measurements/gpt_fullws_transient_memory_probe_cpu.json`](measurements/gpt_fullws_transient_memory_probe_cpu.json),
agrees: `temp_size` 5305 → 5379 MiB unchanged while `argument_size` 366 → 247 MiB.) The
HLO float-buffer fp64 fraction genuinely dropped 99.6% → 83.8% — yet the peak did not
budge. **The persistent-storage lever is exhausted at ~0%.**

### Pillar 2 — The base absolutes `p_total`/`ph_total` (~1e5) **cannot be stored fp32**

The large persistent arrays *are* the base/total absolutes (~1e5 Pa pressure, ~2e5 m²/s²
geopotential). The conservation-pin oracle
([`measurements/fullws_base_absolute_oracle.json`](measurements/fullws_base_absolute_oracle.json),
`GATE_PASS = false`) measures what storing them fp32 does to the dynamics gradients:

| operator | fp64-store err | fp32-store err | × the perturbation's own gated-fp32 floor |
|---|---|---|---|
| geopotential layer-diff `dphi` (advance_w) | 0.0 | 1.31e-2 m²/s² | **26.85×** |
| pressure horizontal PGF (advance_uv) | 6.6e-12 Pa | 6.19e-3 Pa (0.124% of a ~5 Pa Δ) | **126.75×** |

Two load-bearing facts:

1. **fp32-store-with-fp64-island-difference == naive-all-fp32, byte-for-byte.** The
   explicit in-loop fp64 island gives **zero** benefit once the base is stored fp32,
   because ULP(1e5) ≈ 0.008 destroys the low bits **at storage** — widening the
   *difference* to fp64 cannot recover bits the *store* already threw away. This is the
   collapse of the whole "compute the cancellation in fp64" premise: the operands must
   still **have** the low bits.
2. The corruption is **27× / 127×** the gated-fp32 budget the forecast already accepts.

So the base/total absolutes are conservation-PINNED to fp64 **and** they are the large
arrays — closing the loop: there is **no valid-numerics State-precision VRAM win**. (The
in-loop island operators themselves are sound when fed fp64-stored operands —
[`measurements/calc_p_rho_fp32_oracle.json`](measurements/calc_p_rho_fp32_oracle.json) genuine-fp32
product within ~1.0–1.6× the p′ noise floor, ~6× better than naive;
[`measurements/advance_w_fp32_oracle.json`](measurements/advance_w_fp32_oracle.json) w finite /
non-escalating on flat/moderate/steep. The failure is specifically fp32 **storage of the
base**, not the islands.)

### Pillar 3 — The transient peak is precision-insensitive (fp64-pinned by cancellation + qke)

The transient high-water mark is dominated by buffers that are pinned to fp64 for
**correctness**, not by storage:

- the **cancellation islands** EOS/`calc_p_rho`, PGF/`advance_uv`, `advance_w` (the
  large-minus-large that detonates in naive fp32), and
- the **qke-pinned MYNN/PBL** physics. Direct evidence: at 1 km
  (`proofs/v090/d03_1km_validation.json`) the MYNN level-2.5 TKE budget in fp32 goes
  **non-finite after forecast hour 1, with `qke` the SOLE offender — 3036 non-finite
  cells** — while every other prognostic stayed finite. That is why the shipped precision
  matrix pins `qke`/`qsq` to fp64, and JAX type-promotion then widens every qke-touching
  MYNN intermediate to fp64.

Because these pinned fp64 buffers set the live-range peak, demoting `p`/`ph`/`w` to fp32
moves it by ~0.

---

## WHY THE OLD `~4.3×` / `~6×` NUMBERS ARE NOT THE CEILING

- **The `4.3×` "cost proxy" is numerically INVALID.** It was measured with `x64` globally
  off (`true_fp32_cost_proxy.json`, cited in the roofline docs) — i.e. **all-fp32
  including the compute at the cancellation/conservation sites**, with **no fp64 pins**.
  Its numerics are garbage by construction (it would corrupt mass conservation, the EOS/PGF
  cancellation, and detonate qke). It is a useful *cost/VRAM* proxy, **not** a production
  speed claim, and was never published as the achieved number.
- **"Double-single" does not buy it back.** Recovering the cancellation bits with a
  double-single representation costs **fp64-equivalent storage (two fp32 words/scalar) +
  ~16× the time** ([`measurements/gpt_double_single_probe.json`](measurements/gpt_double_single_probe.json):
  relative CPU time 16.0×, accuracy recovered to 2.3e-10 but at 8 bytes/scalar). There is
  no cheap way around the base-absolute pin.
- **`6×` exceeds the RTX 5090 roofline regardless.** The operational step has arithmetic
  intensity ≈ 1 FLOP/byte (pure stencil dycore, no GEMM). The independent blind
  re-derivation ([`roofline-blind-crosscheck.md`](roofline-blind-crosscheck.md)) puts the
  *both-terms-removed* on-card ceiling (vs our own fp64) at **~4.2–4.6× and physically
  un-exceedable** for this AI≈1 algorithm without raising arithmetic intensity — and that
  ceiling is the *numerically-invalid* all-fp32 figure. The *valid* fp32 lane that keeps
  the pins is the measured **~1.1×**. The roofline derivation
  ([`roofline-ceiling-and-levers.md`](roofline-ceiling-and-levers.md)) independently lands
  the valid production target at **~2–3× *vs CPU*** (not vs our fp64 GPU) and states a
  `6×` single-GPU-vs-CPU Canary-1 km figure is **not physically supported**.

So the numbers are reconciled, not contradictory: **on-card invalid all-fp32 ≈ 4.3×
(garbage numerics) → valid-numerics fp32 with the pins ≈ 1.1× (measured, this release).**

---

## THE REAL WINS (what v0.16 actually ships)

1. **The genuine ~1.1× fp32 lane** — a real ALU/bandwidth win on the non-cancellation
   compute (acoustic + safe State arithmetic), oracles pass, distinct compile.
2. **The 1 km-unlock — orthogonal to fp32.** The actual 1 km blocker is the **MYNN BouLac
   dense `(B,nz,nz)` mixing-length matrix** (a ~18.8 GiB single allocation at 147 k cols,
   identical for fp64/fp32/radiation-off — see [`compact-island-fp32.md`](compact-island-fp32.md)
   §5 4-way control). The chunked/O(nz) BouLac shape rewrite makes a **1 km single-domain
   fp64 forecast fit on one RTX 5090** ([`measurements/fullws_boulac_onz_147k.json`](measurements/fullws_boulac_onz_147k.json):
   147 k cols, **21.31 GiB, finite** — was OOM), **bit-identically** to the dense kernel.
   This is a memory-shape lever, not a precision lever.
3. **Honest fp64 parity carried forward** — the v0.15 final fp64 kernel + 72 h two-region
   WRF cell-for-cell identity are unchanged. fp64 GPU ≈ CPU-WRF parity is the GeForce
   `1/64` fp64 hardware law, not a defect.
4. **The cluster weak-scaling path** — the remaining genuine large-speedup levers are
   **algorithmic** (kernel-granularity/fusion to attack the ~70% non-roofline launch
   overhead the blind cross-check quantifies) and **multi-GPU weak scaling**
   ([`roofline-ceiling-and-levers.md`](roofline-ceiling-and-levers.md) §4: ~75–90% in-rack
   weak-scaling efficiency expected once production sharding + transfer audits land), NOT
   fp32.

---

## The evidence trail (every file in this folder)

### Verdict reports
| File | What it is |
|---|---|
| [`opus-fullws-fp32-verdict.md`](opus-fullws-fp32-verdict.md) | The Opus full-working-set implementation + measured verdict (outcome (b): valid-numerics ceiling ~1.1× + 0× VRAM, with the 3-pillar proof). |
| [`gpt-fullws-fp32-crosscheck.md`](gpt-fullws-fp32-crosscheck.md) | Independent GPT reproduction + adversarial refutation attempt. Verdict: **IMPOSSIBILITY CONFIRMED** (1.105× / 1.111×, VRAM 1.000; no valid lever toward >2× found). |
| [`compact-island-fp32.md`](compact-island-fp32.md) | The COMPACT explicit-fp64-island technique proof (numerically sound, genuinely moves the acoustic scan to fp32, ~1.1×) + the 4-way control that root-causes the VRAM peak to the broad fp64 working set + MYNN BouLac (NOT the acoustic scan). |

### Roofline / ceiling derivation
| File | What it is |
|---|---|
| [`roofline-ceiling-and-levers.md`](roofline-ceiling-and-levers.md) | The roofline ceiling derivation (HLO arithmetic-intensity ≈ 1; valid production target ~2–3× vs CPU; `6×` not physically supported) + the full architecture-lever table (BouLac O(nz), intermediate fusion, multi-GPU scaling envelope). |
| [`roofline-blind-crosscheck.md`](roofline-blind-crosscheck.md) | The BLIND independent roofline re-derivation that corrected the naive "~2× because memory-bound" reasoning to the **~4.2–4.6×** on-card (vs our own fp64) ceiling — and confirmed it is the *numerically-invalid* all-fp32 figure, physically un-exceedable for this AI≈1 algorithm. |
| [`reduced-precision-equivalence-criterion.md`](reduced-precision-equivalence-criterion.md) | The binding equivalence-criterion / scientific-rigor methodology (long-horizon non-escalating divergence; conservation-critical state stays fp64-locked; never publish the 4.3× cost proxy as the achieved number). |

### Measured proof objects — [`measurements/`](measurements/)
| File | What it proves |
|---|---|
| [`fullws_fp32_km_bench.json`](measurements/fullws_fp32_km_bench.json) | GPU bench, real Switzerland d01: 16 k **1.107×** / 65 k **1.110×**, **VRAM ratio 1.000**; 147 k OOM (fp32 alone does not unlock). |
| [`fullws_safe_km_bench.json`](measurements/fullws_safe_km_bench.json) | The numerically-defensible `safe` lane (`p_perturbation`/`ph_perturbation`/`w` only; keeps `p_total`/`ph_total` fp64): 16 k **1.108×**, VRAM **1.000** — even the valid lane is ~1.1× / 0% VRAM. |
| [`fullws_base_absolute_oracle.json`](measurements/fullws_base_absolute_oracle.json) | **Pillar 2** — storing the base absolutes fp32 corrupts the geopotential/PGF differences **26.85× / 126.75×** the gated-fp32 floor; `GATE_PASS=false`; fp32-store == naive-fp32 exactly (the fp64 island is powerless). |
| [`fullws_boulac_onz_147k.json`](measurements/fullws_boulac_onz_147k.json) | The **orthogonal** 1 km-unlock: full-ws + BouLac-O(nz) runs the 1 km / 147 k grid to **finite completion at 21.31 GiB**. |
| [`gpt_fullws_reproduce.json`](measurements/gpt_fullws_reproduce.json) | Independent GPT GPU reproduction (matches: **1.105× / 1.111×, VRAM 1.000, `[]`-unlock**). |
| [`gpt_fullws_transient_memory_probe.json`](measurements/gpt_fullws_transient_memory_probe.json) | **Pillar 1** — XLA GPU `memory_analysis`: transient `temp_size` **1975 → 2001 MiB unchanged** while persistent `argument_size` **366 → 247 MiB** shrank. |
| [`gpt_fullws_transient_memory_probe_cpu.json`](measurements/gpt_fullws_transient_memory_probe_cpu.json) | The CPU-lowering cross-probe (same conclusion: `temp_size` 5305 → 5379 MiB unchanged). |
| [`gpt_double_single_probe.json`](measurements/gpt_double_single_probe.json) | Double-single recovery costs **fp64-equivalent storage + ~16× time** — not a path to the ceiling. |
| [`gpt_safe_alias_probe.json`](measurements/gpt_safe_alias_probe.json) | Why the `safe` field set must exclude `p`/`ph` (the `State.replace` alias-sync would otherwise silently re-demote the totals to fp32). |
| [`fp32_s2_mixed_ladder.json`](measurements/fp32_s2_mixed_ladder.json) | The S2 mixed-perturb-fp32 ladder: **1.102× @16 k / 1.113× @65 k**, VRAM 1.000, 147 k OOM — the predecessor lane, same conclusion. |
| [`compact_fp32_km_bench.json`](measurements/compact_fp32_km_bench.json) · [`compact_fp32_km_bench_CORRECTED.json`](measurements/compact_fp32_km_bench_CORRECTED.json) | The acoustic-only COMPACT bench (CORRECTED clears the jit-cache alias so compact compiles distinctly): **1.109× / 1.105×**, VRAM 1.000, `[]`-unlock. |
| [`native_fp32_km_bench.json`](measurements/native_fp32_km_bench.json) | The earlier NATIVE/type-promotion attempt: **1.010×** (cache-alias; +316% convert-scatter ops, −10% fp64 arrays only), VRAM 1.000 — the documented convert-scatter failure mode. |
| [`calc_p_rho_fp32_oracle.json`](measurements/calc_p_rho_fp32_oracle.json) · [`advance_w_fp32_oracle.json`](measurements/advance_w_fp32_oracle.json) · [`compact_island_fp32_oracle.json`](measurements/compact_island_fp32_oracle.json) | The in-loop cancellation-island fp32 oracles — sound (within the fp32 noise floor / finite / non-escalating) when fed fp64-stored operands; isolate that the failure is fp32 **storage of the base**, not the island arithmetic. |

See also the honest performance panel `proofs/v016/dashboard/HONEST_PERF_PANEL.md` and
the release narrative [`RELEASE_NOTES_v0.16.0.md`](../../../RELEASE_NOTES_v0.16.0.md).
