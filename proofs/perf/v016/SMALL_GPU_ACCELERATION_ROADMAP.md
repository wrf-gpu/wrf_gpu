# The Future of Acceleration on Small (Consumer) GPUs — Grounded Synthesis & Roadmap

**Author:** wrf_gpu manager (Opus 4.8), 2026-06-14. **Rev 2** — incorporates the independent critical review + its GPU audit.
**Inputs synthesized:** (1) fresh GPU probes on the RTX 5090 (`eft_double_single_gpu_probe.*`, `fp64_fp32_throughput_probe.*`); (2) the external-AI risk analysis (`FP32_PERTURBATION_SPEEDUP_RISK_ANALYSIS.md`); (3) two grounded code+proof audits (acoustic-arena VRAM; megakernel + nested-case scaling); (4) **an independent critical review + GPU audit (`FP32_GRADIENT_CHALLENGE_CRITICAL_REVIEW.md`, `fp32_challenge_independent_audit.json`)** that re-ran the probes with different seeds/shapes and converges on the same path while correcting one number (below).
**Rule honored:** every number traces to a measurement or a source file. Where a number is a *target* (from an invalid proxy) and not a *measurement*, it is labeled.

---

## In einfacher Sprache (für eilige Leser)

**In drei Sätzen.** Auf einer kleinen Consumer-GPU ist realistisch **~1.8–2× schneller + ~2× weniger Speicher (1 km passt auf eine Karte)** drin — *nicht* 4–6×, weil die Last durch Kernel-Starts und Speicher begrenzt ist, nicht durch fp64-Rechnen. Den Gewinn gibt es nur über einen **perturbations-nativen fp32-Umbau** des Strömungskerns (kein breites Double-Single, das ist langsamer als fp64). Das größte Risiko ist nicht „geht nicht", sondern **„geht subtil daneben"**: Präzisionsänderungen im chaotischen Kern erzeugen leicht Abweichungen, die erst nach 72 h oder bei bestimmten Wetterlagen auftauchen und dann **dutzende Sprints** zum Einfangen kosten.

**Das Problem.** Wetterrechnen heißt: winzige Differenzen riesiger Zahlen (1 Pa Gradient auf 100 000 Pa). fp32 hat nur ~7 Stellen — nach der Subtraktion bleiben zu wenige. Lösung: die kleine Differenz *exakt mitführen* statt wegwerfen (Perturbationsform + gezielte Kompensation).

**Die wichtigste korrigierte Angst.** Die ganze fp32-Jagd startete mit „fp64 ist 64× langsamer". Gemessen (von mir und unabhängig gegengeprüft) stimmt das nicht — je nach Rechenform **~1.8× bis ~7×**, kein fester Wert. fp64 ist also *nicht* existenziell teuer; der belastbare Anker ist nicht der Mikrobench, sondern der Forecast-Proxy → **~1.82× im ganzen Schritt** (und auch das ist ein zu *erarbeitendes Ziel*, kein gemessenes Ergebnis).

**Drei Hebel, nicht einer — in verschiedenen Bauteilen.** Geschwindigkeit kommt v.a. aus der fp32-**Physik** (Strahlung/PBL, viele `exp`/`pow`), kaum aus dem Strömungskern. VRAM/1 km kommt aus dem fp32-**Dycore** — aber nur, wenn man dort wirklich *rechnet* in fp32, nicht nur *speichert* (reines Umspeichern = **0 %**, gemessen). Megakernel (Struktur) bringt nochmal etwas obendrauf. **Geschwindigkeit, VRAM und Gefahr sitzen in verschiedenen Teilen** — das ist der Planungsschlüssel.

### Master-Tabelle — welche Anpassung bringt was, wie schwer, wie gefährlich

Abkürzungen Fälle: **WS** = Workstation 128² (d01); **NEST** = Canary nested d01→d02→d03 (dein Fall); **1km** = große 1-km-Einzeldomäne; **CL** = Cluster/Multi-GPU. „Divergenz-Risiko" = Wahrscheinlichkeit, schwer auffindbare numerische Abweichungen zu erzeugen, + grobe Sprint-Kosten, sie einzufangen.

| Anpassung (Hebel) | Nutzen nach Fall | Magnitude (alle Analysen) | Komplexität | Divergenz-Risiko → Sprints | Urteil |
|---|---|---|---|---|---|
| **1. Scattered dtype-Edits** (heutige S2-Lane) | minimal überall; 0 VRAM | **1.05–1.2×** (gemessen 1.106×) | niedrig (fertig) | **niedrig** (gated, nur Storage) → ~0 | **STOP** — Sackgasse |
| **2. fp32-Physik** Strahlung/PBL (Fork C) | Speed WS/NEST/1km wo Strahlung läuft; skaliert mit Strahlungsanteil; wenig VRAM | **~1.2–1.5× Schritt** (Großteil der 4.29× Kern sitzt hier) | mittel (ADR-007 erlaubt; per-Scheme-Oracles existieren) | **mittel** — koppelt via Heizraten in die Dynamik zurück; bounded/gateable → **~1–3 Sprints** | **JETZT** — bankbar, geringes Risiko |
| **3. Perturbations-nativer fp32-Dycore** (Fork A, Hauptpfad) | **alle Fälle ~1.8–2× + ~2× VRAM**; Vorteil **wächst** mit Gittergröße (1.8→7× Sweep) → bestes für 1km/CL | **Ziel 1.8–2.9×** (aus ungültigem Proxy, zu erarbeiten) | **hoch** (Operator-Rewrite, 25 total-Materialisierungen, 5 Auslöschungs-Inseln) | **HOCH** — der chaotische Kern; subtile 72h-/Regime-Divergenz (Steilgelände, Starkwind = Projekt-Historie Venting/qke). **Das „dutzende-Sprints"-Risiko.** Mitigation: HLO-Gate + Conservation + Long-Horizon + Kahan | **Go/No-Go-Gaten** (Phase 2) |
| **4. Gezielte Kahan/Kompensation** | kein Speed; sichert **Langzeit-Gültigkeit** (gegen Drift) in NEST/1km/CL | Drift **167–200× besser** (Toy) | niedrig–mittel | **niedrig / risiko-SENKEND** (verbessert Genauigkeit); einziges Risiko: XLA bricht Kompensation → `optimization_barrier`+Selbsttest fängt es | **JA, nur gezielt** (nicht überall: sonst 4→8 Byte) |
| **5. Breites Double-Single** | **negativ** Speed; 0 VRAM (8 Byte) | **Verlangsamung** (25–32× fp32, 15–16× fp64) | mittel | mittel (Barrier-Fläche) + Perf-Desaster | **VERMEIDEN** — nur winzige Insel |
| **6. BouLac → O(nz) Shape-Fix** | kein Speed; **VRAM** → hilft 1km/NEST passen | 21.31 GiB finite @147k (vorher OOM) | mittel (Shape-Algebra) | **niedrig** (bit-identisch möglich, gegen fp64 gated) | **JA** — bestes Risiko/Nutzen für VRAM ohne Präzisionsrisiko |
| **7. Pallas Spalten-Megakernel** | ~zusätzlich; v.a. **1km/CL** (vertikal-gekoppelt) + WS (launch-bound) | **~1.3–1.6×** zusätzlich (ungemessen; 3× auf Thomas @512² per-op) | **hoch** (handgeschriebene Kernel) | **mittel** — Engineering-Bugs, aber bit-vs-XLA gateable (kein chaotisches Risiko) | **erst nach Fork A**, dann messen |
| **8. fp64 behalten + Multi-GPU Weak-Scaling** | **CL: der einzige Pfad zu >3×** (skaliert mit GPU-Zahl); Single-Card keine Beschleunigung | ungemessen (fake-mesh bit-identisch; echter Durchsatz offen) | hoch (NVLink/Cluster-Hardware) | **niedrig numerisch** (Partition-Invarianz bit-identisch), hoch Infra | **der echte Weg zu großen Faktoren** — braucht Hardware |
| **9. Stochastic Rounding** | theoretisch unbiased; Kahan ist stärker+deterministisch | unbewiesen | mittel | **HOCH für Reproduzierbarkeit** (Nichtdeterminismus bricht bit-identische-Restart-Disziplin) | **DEFER** (beide AIs einig) |

**Lesehilfe zur Risiko-Spalte:** Die einzige Anpassung mit echtem „dutzende-Sprints"-Divergenzrisiko ist **#3 (fp32-Dycore)** — genau deshalb wird sie hinter ein billiges, falsifizierbares HLO/Conservation-Gate gesetzt, *bevor* teure 72-h-Läufe starten. **#2, #4, #6** sind risikoarm und sofort wertvoll. **#5, #9** sind zu vermeiden. **#7, #8** kommen später bzw. brauchen Hardware.

---

## 0. Executive summary — three levers, repeatedly conflated

There is no single "fp32 lever." There are **three independent levers**, and almost every past confusion comes from mixing them:

| Lever | What it buys | From where (measured) | Cost / risk | Status |
|---|---|---|---|---|
| **L1 Precision → speed** | ~1.8–2× warm full-step | fp32 ALU **~1.8–7× (shape-dependent)** + transcendentals; mostly **radiation/PBL**, *not* dycore | medium–high | valid version **unbuilt** |
| **L2 Precision → VRAM/1 km** | ~2× VRAM, fits 1 km where fp64 OOMs | true-fp32 **compute** (not storage): 18.8 GiB OOM → **9.65 GiB fits** | high (dycore cancellation) | valid version **unbuilt** |
| **L3 Structure → speed** | ~1.3–1.6× incremental | column megakernel cuts ~3,300 D2D scan-carry copies/step | medium | **unbuilt, unmeasured** |

Two facts reframe everything below:

1. **The "RTX 5090 fp64 = 1/64" premise (ADR-007) is wrong in magnitude — but the exact penalty is NOT a single number.** 1/64 is the *peak-FLOPS* ratio. The effective penalty for XLA-generated code is **shape-dependent: ~1.8× to ~7.3×** (independent sweep, `fp32_challenge_independent_audit.json`: 1.79× at 4M elements → 7.26× at 16M; my own runs landed 3.4–4.3× on other shapes). The honest, double-checked conclusion is the *weaker* one: **fp64 is not 64× crippled, but a microbench ratio is not a forecast roofline.** The load-bearing anchor is therefore not the microbench but the **v015 true-fp32 cost proxy: dycore core 70.49 → 16.44 ms (4.29×), which Amdahls to ~1.82× on the full step** — and even that is a *cost ceiling from a numerically-invalid run*, i.e. a target to be earned, not a measured speedup. (My earlier "~4.3× / ~1× on the real step" framing in the intermediate Open-Challenge doc was too sharp on both ends and is superseded here.)

2. **Speed-source, VRAM-source, and danger-source are three different places.** The fp32 *speed* win is mostly **radiation/PBL transcendentals** (`pow`/`exp`, ~8.9× in fp32) — bounded fields, low cancellation risk, already ADR-007-authorized. The fp32 *VRAM* win is the **dycore acoustic arena** (the ~21.7 GiB transient peak). The *numerical danger* (catastrophic cancellation) is also in the **dycore acoustic** core. So the easy speed and the hard VRAM are in different subsystems — which is the central planning insight of this document.

---

## 1. Can the acoustic inner arena be made fp32-dominant? (the principal's Q1)

**Short answer: storage demotion = 0 %; the real VRAM win needs the full perturbation-native *compute* rewrite, which yields ~2× — but that is hard and unbuilt.** These are different things and the project only disproved the first.

**1a — Storage demotion alone is dead (measured, twice).** Making the perturbation work-arrays fp32 while the operators still *compute* in fp64 moves peak VRAM by **0 GiB** (`fp32_s2_mixed_ladder_final_2x2.json`: fp64 11.6455 GiB vs mixed-S2 11.6455 GiB, `vram_x = 1.000`; independently reproduced full-working-set, `2026-06-14-opus-fullws-fp32.md`). Reason: the peak is **transient compute** (XLA `temp_size` 5305→5379 MiB, *unchanged*), not persistent storage; and the base absolutes `p_total/ph_total` (~1e5/2e5) **cannot** be stored fp32 — the base-absolute oracle shows fp32 storage corrupts geopotential **26.85×** and PGF **126.75×** over budget, because `ULP(1e5)≈0.008` destroys the bits *at storage* before any difference (`GATE_PASS=False`). XLA's own rematerialization already proved the **fp64** live set is irreducible below **21.6 GiB**.

**1b — But true fp32 *compute* does halve VRAM (measured, invalid numerics).** When the compute itself runs fp32 (`true_fp32_cost_proxy.json`, x64 toggled off): VRAM **3.44 → 1.66 GiB = 2.07×**, and at 147,456 cols **fp64 OOMs at 18.8 GiB while fp32 fits at 9.65 GiB** — the literal difference between 1 km fitting on one card or not. The catch: that proxy is *numerically garbage by design* (it also demotes the cancellation/conservation sites). It proves the **cost ceiling**, not a valid path.

**1c — So the question reduces to: can the compute be made perturbation-native fp32 *validly*?** The machinery is **partially built**: ADR-031 "S2" already precomputes the ~5 cancellation-prone differences (`pb_grad`, `php_grad`, `dphb`) **once per stage in a tiny fp64 island** and feeds the small result into the fp32 loop (`acoustic.py:228-238`). What remains:
- **Never materialize an O(1e5) total inside the loop.** The static audit finds **25 `base = total − perturbation` sites** (`fp32_acoustic_static_audit.json`) and the legacy operational entry still `raise NotImplementedError` for the mixed acoustic core (`operational_mode.py:1287`). Totals must be reconstructed *only* at I/O / boundary / output.
- **Compensated accumulation for the time integration** (see §3) so the perturbation-native core does not drift over a 72 h forecast.
- The **classification is favorable**: most inner-loop temporaries (Thomas solve, `rhs`, `ph_next`, `t_2ave`, `al/p` outputs, flux accumulators, all of `w`) are perturbation-scale and fp32-safe; only ~5 operators are genuinely cancellation-critical, and those already have the fp64-island pattern.

**Quantified answer:** **~2× VRAM is achievable in principle** (the cost proxy proves the ceiling), but **only** by completing the perturbation-authoritative *operator* rewrite (not storage flags), eliminating the 25 total-materialization sites, and adding compensated accumulation. **Storage demotion gives 0 %** — that path is correctly closed. The single biggest obstacle is the base-absolute materialization, and it is an *operator-graph* problem, not a physics impossibility. **It stacks with — and is partly preceded by — the orthogonal MYNN-BouLac dense→O(nz) shape fix** (`GPUWRF_MYNN_BOULAC_ONZ=1`, measured 21.31 GiB finite at 147k), which removes a *different* big VRAM consumer with no precision risk.

---

## 2. Megakernel exclusion + grounded nested-Canary speedup (Q2)

### 2a — Was the megakernel correctly excluded? Partly. The *decision* is defensible; the *reasoning* is flawed.

The §5 justification claims a megakernel "mainly attacks launch/latency overhead — which only dominates the tiny case." **This conflates two mechanisms.** A hand-written column megakernel (Pallas/Triton) also keeps vertical-column intermediates in registers/shared memory, eliminating global-memory round-trips — a **memory-traffic** reduction independent of launch count that *grows* at large grids. The "FUSION ≈ 0 %" result that is cited as support actually measured **XLA re-fusion of an already-single-`jit` graph** (carry-split −0.14 %/+12 %) — a *different* mechanism. The project's own memory even admits "**only a megakernel or fp32 cuts bytes**," conceding the megakernel cuts bytes.

**Honest status:** the megakernel's byte-reduction benefit is **unmeasured, not refuted** (Pallas was never built; the only data point is ~3× on the *Thomas solve* at 512², a per-op result). The *magnitude estimate* — **~1.3–1.6× incremental over the ~2× the dynamic kernel already gives** (→ 2.08–4.14× vs CPU at 1 km) — is the reviewer's honest guess and is probably the right order. **Correct framing: "deferred and unquantified," not "dead on principle."** The decision to do fp32 first is still reasonable (fp32 cuts bytes ~2× by dtype width and is broader), but Pallas should be re-opened as a *measured* probe, not dismissed.

### 2b — Grounded speedup for the Canary nested d01→d02→d03 (9/3/1 km)

Measured grid + subcycling (`nesting_24h_v0110.json`):

| Domain | cols | steps/24 h | fp64 core ms/step¹ | fp32 core ms/step² |
|---|---:|---:|---:|---:|
| d01 9 km | 5,487 | 4,800 | 46.7 | 7.6 |
| d02 3 km | 10,494 | 14,400 | 71.4 | 11.6 |
| d03 1 km | 6,975 | 43,200 | 54.0 | 8.8 |

¹ measured fit `ms = 19.58 + 4.937·kcols` (`grid_scaling.json`, 7 real points). ² invalid-proxy fit `ms = 3.27 + 0.790·kcols`.

**Amdahl per domain** (core = dycore+rad+MYNN = 58.8 % of full step; non-core 41.2 % stays fp64; full = 1.7× core):
`speedup = 1.7·core_fp64 / (core_fp32 + 0.7·core_fp64)` → **~1.97× for every domain** (the core fraction is ~constant across these grids). So:

> **Grounded estimate: ~1.8–2.0× total-wall over the current fp64 GPU for the nested Canary case, IF the valid fp32 rewrite reproduces the proxy cost.** Up to ~2.4–2.9× if non-core (coupler/boundary/physics) also downcasts. **4× is not credible from precision alone.**

**Three honesty flags (why this is a target, not a measurement):**
1. The fp32 fits come from the **numerically-invalid** proxy; the valid rewrite must *earn* them (external-AI gate, §3).
2. The "6.25× marginal" the proxy leans on is **fit-vs-fit at large grids where fp64 OOM'd and was never measured** — the fp64 *slope* is real, but the paired large-grid point is not.
3. **There is no clean measured GPU-total-vs-CPU-total for the 3-domain nest.** The "9.86×" figure uses a **d02-step-only CPU extrapolation** (ignores d01+d03 CPU cost), and the two GPU walls for the same case disagree 3.2× (1,654 s vs 5,270 s). The defensible vs-CPU anchor is the **asymptotic 1.63× fp64 floor** (GPU 4.94 vs CPU-28 8.06 µs/col/step); fp32-valid would lift that to **~2.6–3.2× vs CPU-WRF**.

**Subtle but decisive:** most of that ~2× comes from **radiation/PBL fp32** (transcendentals), *not* the dycore acoustic (~8 % of wall, launch-bound, tiny win). So the speed lever and the VRAM lever pull on different subsystems.

---

## 3. Critique of the external-AI analysis (`FP32_PERTURBATION_SPEEDUP_RISK_ANALYSIS.md`)

**Verdict: strong, mostly correct, and I agree with its bottom line.** It correctly (i) separates partial-fp32 (10 %, dead) from true-fp32 (4.29× core); (ii) builds the right Amdahl ladder (1.82× core-only → 4.17× if everything downcasts); (iii) sets a HLO/profiler gate *before* expensive 72 h runs; (iv) puts the work at the JAX/operator level first, Pallas after; (v) lands "2× credible, 3× good, 4× stretch." Its identities (`1/(M_base+μ) = (1/M_base)/(1+μ/M_base)`, `p' = PB·expm1(s)`) are exactly the right perturbation-native rewrites.

**Three gaps I can close with ground truth:**

1. **It under-weights long-horizon drift.** Its "dead-end conditions" are all *storage/HLO-structure*; none addresses that a correct perturbation-native fp32 core can still **drift over a 72 h forecast** through swamped accumulation. My probe: naive fp32 accumulation drifts **200×** more than compensated over 200k steps (`eft_double_single_gpu_probe.json` E3: 0.83 vs 0.004). **The rewrite needs compensated (Kahan/Neumaier) summation on the time-integration and conservation sums — ~4 flops, cheap — or it will pass a short HLO gate and fail the 72 h conservation gate.** This is the one substantive thing it misses.

2. **It states the 6.25× marginal too cleanly.** Ground truth: the fp64 points at the grids where fp32 was measured **OOM'd** (`true_fp32_cost_proxy.json` `fp64_refs[1].oom=true`); the ratio there is fit-vs-fit. Real, but should be flagged as extrapolation (which is itself the scalability argument).

3. **It treats the 4.29× core as uniform "hardware upside."** Decomposed, it is **mostly radiation/PBL transcendentals**, not dycore — so the *risk-weighted* speedup is better than it looks (the big win is in the low-cancellation-risk physics; the high-risk dycore contributes little speed and is really the *VRAM* lever). It should split the 4.29× by subsystem.

It also correctly says compiler/register tricks can't recover lost mantissa bits — true; my one refinement is that `optimization_barrier` (a compiler construct) is *required* to **preserve** the error-free transformations, so the compiler layer is a necessary guard, not irrelevant.

### 3b. The independent critical review (`FP32_GRADIENT_CHALLENGE_CRITICAL_REVIEW.md`) — analyzed

A second model independently re-ran the probes (different seeds, separate script) against my intermediate Open-Challenge doc. **It converges with this roadmap on every path** (stop scattered edits; perturbation-native rewrite is the main path at 1.8–2.9×; targeted-not-broad Kahan; avoid broad DD; Pallas only after; remat only after) and reproduced the accuracy story (PGF 3.19e-3 → 1.59e-7 → 1.14e-10; Kahan 167× drift cut). It contributes **two things I adopt:**

1. **The decisive correction:** the fp64:fp32 microbench ratio is **not stable** — its shape sweep is **1.79× (4M elems) → 7.26× (16M elems)**. So my single "~4.3×" was one shape, not a law. Adopted throughout this Rev 2; the load-bearing anchor is now the cost-proxy Amdahl (~1.82×), explicitly labeled a *target from an invalid proxy*.
2. **Stochastic rounding is a poor fit** for this project specifically — nondeterminism conflicts with the bit-identical-restart / identity-proof discipline. Adopted (row 9 of the master table = DEFER).

**Where this roadmap stays broader than the review:** the review is almost entirely a *speed* (wall-clock) analysis; it barely engages the **VRAM/capability** axis. The deeper result here — storage-demotion = 0 % vs true-compute ≈ 2 ×, the transient-arena liveness finding, and the orthogonal BouLac→O(nz) lever (§1, §2a) — is the part that actually decides *1 km-on-one-card*, which for a small GPU is the more valuable prize. The two analyses are complementary, not contradictory: the review nails the speed ceiling and the "avoid broad DD" guardrail; this roadmap adds the capability axis, the per-domain nested-Canary numbers, and the divergence-risk weighting.

One honest divergence to flag (the kind the project's "always double-check kernel-speed" rule exists for): the **DD-vs-native-fp64 cost** differs between probes — my run ~6.7×, the review's ~15–16×. Both agree DD *loses* to fp64 (so "avoid broad DD" holds regardless), but the factor is methodology-dependent (DD-mul-by-scalar vs DD-mul-chain). Not worth resolving unless someone proposes a DD island, in which case measure that exact island.

---

## 4. The future of small-GPU acceleration, and the roadmap

### 4.1 What "small GPU" actually constrains
A consumer card (32 GB, fp64 throttled **shape-dependently ~1.8–7×** but **not** 64×) is bound first by **VRAM** (can the problem fit) and second by **launch overhead** (thousands of tiny kernels), and only third by fp64 arithmetic. So the highest-value target for small GPUs is **capability (fit 1 km / bigger domains)**, with speed a secondary ~2×. Raw single-card wall-clock will never be a huge multiple of 28-rank CPU here — the honest ceiling is ~2–3× — because the workload is launch/memory-shaped, not fp32-FLOP-shaped. The big multipliers (per-watt, whole-Earth) live in **weak-scaling across many GPUs**, not one consumer card.

### 4.2 The strategic fork (this is the principal's call)

**Fork A — Full perturbation-native fp32 (speed + VRAM + 1 km).** The only path to *both* ~2× wall and 1 km-on-one-card. Hard, multi-sprint, cancellation-risk in the dycore. Phase-gated so the risk is cheap to retire.

**Fork B — VRAM-only, keep fp64 (capability without the precision risk).** BouLac→O(nz) + the shipped RRTMG chunking → fit 1 km *in fp64*. Gets the capability and the ~1.6× fp64 floor, **skips** the ~2× speed and the cancellation risk. Lowest risk.

**Fork C — fp32 physics only (safe partial speed).** Downcast radiation + PBL (ADR-007 already authorizes; that is where most of the 4.29× lives), keep the dycore fp64. Medium risk, captures a good fraction of the speed, **no** VRAM/1 km. A strong "bankable" intermediate.

### 4.3 Recommended sequencing (phase-gated, falsifiable, cheap-to-kill)

- **Phase 0 — Mostly DONE; one piece remains.** The synthetic re-measurement was performed twice (mine + the independent audit) and **converges on "not 64×, but shape-dependent ~1.8–7×, so a microbench is not a roofline."** That already retires the "fp64 = 64× existential" premise and should reshape ADR-007's framing. The **one remaining ½-sprint task** is to measure the fp64 penalty on the **real dycore step** (not synthetic kernels) so the Amdahl uses a measured core fraction per resolution rather than the invalid cost-proxy. Cheap, removes the last unmeasured link in the speed argument.

- **Phase 1 — Fork C now (1–2 sprints, low risk, bankable).** fp32 radiation + MYNN-PBL behind a flag, oracle-gated. Captures the easy, low-cancellation slice of the speed win. Ship it; it is independently valuable and de-risks the toolchain (convert-tax, mixed-precision HLO).

- **Phase 2 — The fp32 go/no-go gate (1–2 sprints, the decision point).** Build the EFT/compensated primitives (already prototyped here, GPU-validated, bit-exact) + a perturbation-native prototype of the 5 dycore operators (PGF, EOS, advance_w, calc_p_rho, mu_t). Gate on the external-AI's HLO/profiler threshold at 256²: **temp_size materially below 7.57 GiB, top-3 kernels change dtype, convert-ops ≪ 697, no fp32 `total−perturbation` in-loop, AND conservation 1e-15 + bounded 200k-step drift with compensated accumulation.** If it crosses → Phase 3. If not → **stop; Fork A is a dead end and you spent ~2 sprints, not a milestone.**

- **Phase 3 — Full Fork A (3–5 sprints, only if Phase 2 passes).** Complete the operator rewrite, eliminate the 25 total-materialization sites, validate 72 h field-parity + conservation. Delivers ~1.8–2× and 1 km-on-one-card.

- **Phase 4 — Stacking levers (opportunistic).** BouLac→O(nz) (VRAM, low risk, do alongside Phase 2). One **measured** Pallas column megakernel on the vertical-coupled block (Thomas + column physics) to test the unquantified ~1.3–1.6× — *measure it before believing it*.

### 4.4 What does NOT work (so we stop revisiting it)
- **Double-single (DD) for speed:** dead. ~29× fp32 cost, ~6.7× slower than native fp64 on this GPU. DD/compensation is *only* for the few cancellation/accumulation points, never the bulk.
- **Storage-only fp32 demotion for VRAM:** dead. 0 % (measured twice). VRAM needs the *compute* rewrite.
- **XLA re-fusion / carry-split for speed:** dead (≈0 %, blind-double-confirmed). The remaining structural lever is a *hand-written* megakernel, which is a different mechanism.
- **Chasing single-card 4–6×:** not on this hardware. The workload is launch/memory-shaped; ~2–3× is the honest ceiling. Multipliers beyond that require multi-GPU weak-scaling.

---

## 5. Bottom line

The project's "fp32 impossible" verdict is **right about no-free-lunch and wrong about two load-bearing premises**: fp64 is **shape-dependent ~1.8–7×, not 64×** (the "64×-existential" framing is retired), and *true-fp32 compute* (not storage) does give ~2× speed and ~2× VRAM. The mathematical identity to make it valid exists and is GPU-proven (perturbation-native operators + error-free-transformation accumulation). The realistic prize for a small GPU is **~1.8–2× wall and 1 km-on-one-card** — a *target to be earned*, gated behind a cheap, falsifiable HLO/profiler threshold, not a measured result. The smart move is to **bank the easy physics-fp32 speed (Fork C) now**, do the **BouLac→O(nz) VRAM fix (low-risk capability)** alongside, and run the **fp32-dycore go/no-go gate** for ~2 sprints — committing to the full rewrite only if that gate passes. The single lever with real "dozens-of-sprints" divergence risk is the fp32 dycore; everything else is either low-risk-and-now (physics-fp32, Kahan, BouLac) or later/hardware (Pallas, multi-GPU).

### Sources
Probes: `proofs/perf/v016/eft_double_single_gpu_probe.{py,json}`, `fp64_fp32_throughput_probe.{py,json}`. Project evidence: `true_fp32_cost_proxy.json`, `fp32_s2_mixed_ladder_final_2x2.json`, `grid_scaling.json`, `km_feasibility_verdict.json`, `kernel_characterization.md`, `KERNEL-OPTIMIZATION-FINDINGS-FINAL.md` §5/§6/§8, `fp32_acoustic_static_audit.json`, `nesting_24h_v0110.json`, `2026-06-14-opus-fullws-fp32.md`, `ADR-007-precision-policy.md`, `ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md`, `FP32_PERTURBATION_SPEEDUP_RISK_ANALYSIS.md`, and `FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md`.
