# v0.3.3 — WRF-compatible forecasts on one NVIDIA GPU

**v0.3.3 technical validation record.** Final release source is `be0dd543c9f7f6d619a8b504348b11267ffe913f`, src `46fbb0d924f006941ee3c01dad1f6cf151072c3f`, package `33f3976306269d8eeeee9b7f534180ca0aebfb41`. Actual 7a executions retain their measured labels; final be0 reuse is explicit and bounded by the accepted native identity evidence, not a relabelled execution.

## Original CPU-WRF validation and disclosure rule

Final validation: G1 W3_0115 / W3R0227 / W3_0408; G2 W3P0227 / W3P_0115 / genuine W3Rf_0115; Monica72; shipped Swiss24. Seven72-hour forecasts and Swiss24 supply1,485 paired hourly domain-frames. Original CPU-WRF is fidelity truth. Each case has hourly heatmaps, RMSE/bias and spatial |error| p25–75/p5–95 bands; IC spread is empirical context. Replicas remain distinct and never replace the primary. Final strict24, full72 raw/annex, all-frame integrity and station table: [the final eight-arm table](#final-eight-arm-results). Final per-arm figures accompany the source-qualified historical galleries; no missing point or threshold is invented.

Owner decision `87e0ebccd` keeps every frozen gate, scorer, threshold and pairing unchanged: all run and report verbatim. Every miss is disclosed numerically. A miss may be nonblocking only when **all four** criteria hold: (1) near the bound—cumulative RMSE/caps within roughly10%, or trace amounts such as QGRAUP≤1e−9kg/kg; (2) trajectory-divergence signature—domain totals within a few%, displaced events, replicas/other cases pass—and no port bug found by the time-boxed hunts; a found bug is fixed; (3) no NaN/Inf, masking, unphysical values or lost hours, and F2 counts reported; (4) observation-based station skill is not materially worse than CPU-WRF; small significant changes such as0115 WS10≈+0.03m/s are disclosed. The manager records each edge-case decision in the release ledger. New failure classes, large or physically wrong fields, E41/S2/native failures and crashes still block. A disclosed raw FAIL remains a raw FAIL.

## F2: nonzero counts explicitly disclosed

The actual strict 7a three-domain0227R observation has200/600/1800 own-steps,12 history files and634,857 negative field-cell-phase visits (280 fewer than the historical635,137). These are visits, not a count of distinct cells. Nonfinite, moisture-cap, native-mass and new-class events are all zero. Extrema are qc−3.8382540879e−6kg/kg, Nr−0.9589087963#/kg and qv−4.2023111746e−6kg/kg. qc magnitude is0.418% above the old global minimum, Nr19.079% smaller; qv is13.976× the old minimum and is **explicitly disclosed**, with d01/d03 ratios145.884×/123.968×. d01 qc also grows from4.711e−13 to6.097e−8kg/kg despite the small global qc change.

The observed qc witness is already negative before native PD; captured source addition and pristine `solve_em:1932` use the same unclipped equation. This is SOURCE_PREADD, not proof of pure fp32 cancellation. Remaining4,897 non-post-RK visits are all ring0; negative interior visits are zero, and WRF-order MP cleanup clears interior negatives before entry/guards. “Would repair” diagnostics do not prove applied port masks. qv's larger historical ratio has no demonstrated causal attribution to the raw-qv fix. `8a88e29b` records **NONBLOCKING_WITH_DISCLOSURE for G1**, not a zero-event certificate, full pristine event oracle or whole-release GO. Native/runtime dispositions and all raw evidence remain separate.

## E41 root cause and final correction

Avoidable XLA input/transpose fusions crossed the unchanged post-compile E41 eligibility limit. E41 uses the original eligible HLO>500 rule, STACK>256B and non-leaf callee>1,000 SASS; a separate large-native-SASS diagnostic did not expand the gate. The correction uses the REAL native Noah-MP radiation path, literal column-local math, completed-field boundaries and `GPUWRF_LAYOUT_PIN=ac_carry_cols_cum`; source counts or barriers alone are not speed evidence. The final small-grid firewall isolates root-step phase seams only for mass-grid geometries with≤4,096 columns, with its predicate based on theta extent rather than staggered-u size.

Formal native E41 records: WN7a eligible>500=0 / STACK32B; Monica7a eligible>500=0 / STACK64B; Swissbe0 eligible>500=0 / STACK16B. Actual be0 WN d01/fused-d02 and Monica fused-d01 native ELF plus serialized KernelThunk sequences are byte-identical to formal7a (3/3); options/PIN and source/carry qualifications stay explicit. Swiss firewall overhead is approximately **+12% kernels / +8% ordinary-step cost on≤4,096-column grids only** [M, manager-scoped comparison]; it does not apply to the larger WN/Mon grids and is not a universal whole-run slowdown. Exact matched-profile/reference tuple: [manager ledger L3](RELEASE_LEDGER_v0.3.3.md). Added launches, traffic, allocations/peak memory and readiness remain disclosed where measured rather than inferred from the barrier.

## Snow semantics and radiation performance

The Thompson snow correction restores the original small-active-snow terminal-velocity denominator instead of an inappropriate numerical floor. Pristine-reference checks retain1,210/1,210 event passes, snow mixing-ratio maximum difference1.77636e−15 and the original positive fall-speed behavior. It adds no new physics flag, mask, ABI or full-grid intermediate. Correct WRF semantics take precedence over a small measured cost; whole-model memory/timing claims require the composed-source measurements. Snow patch54636f89 and source1f690812 remain linked to their review/oracle evidence.

The native REAL Noah-MP radiation kernel is the intended speed-up and fusion-boundary repair. A matched before/after kernel-family ratio is not separately quantified in this record; no such ratio is inferred from source structure. Kernel-family acceleration is not a whole-forecast claim; final whole-run rates, hardware, init/load/output/compression clocks, energy classification and peak memory are the explicitly dated v0.3.2 measurements in the README/performance guide. Dated v0.3.2 numbers remain labelled v0.3.2 until a verified final tuple is supplied. No new benchmark is required solely to fill a figure.

## Use, limitations and publication

The README user-guide link remains immediately below the title: https://wrf-gpu.github.io/wrf_gpu/. The guide keeps its user-first entry layout. Original source-labelled raw failures, near-threshold ledger decisions, replica flags and trace exceptions remain available: [the final ledger](RELEASE_LEDGER_v0.3.3.md). No universal observational equivalence, training GO or public release is inferred from completed runs. Source/docs/plots parity, approved history cleanup, tag/push, live site readback and manager-only Telegram/ALISIOS handoff remain final publication steps.


## Final eight-arm results

All eight original-CPU scoring readers are closed. The manager accepts the disclosed deviations under the Owner rule; raw failed checks remain failed.

| Comparison | Forecast | Strict 24 h (raw) | Full-window D6 (raw) | Interpretation |
|---|---:|---|---|---|
| Tenerife 0115 primary | 72 h | FAIL | FAIL | 61 floor-limited rows; one real V10 miss. Rain and cirrus lifetime differences disclosed (L4/L6/L7). |
| Tenerife 0227 primary | 72 h | PASS | FAIL | All eight breaches pass the frozen CPU-pair spread annex (L5). |
| Tenerife 0408 primary | 72 h | PASS | PASS | D6 passes; trace graupel integrity miss disclosed (L1). |
| 0227 IC member | 72 h | PASS | FAIL | All six breaches pass the frozen CPU-pair spread annex; integrity annex passes. |
| 0115 IC member | 72 h | FAIL | FAIL | 54 floor-limited rows; strict rain FAIL retained. Integrity annex passes; IC difference included. |
| 0115 same-IC replica | 72 h | PASS | FAIL | Eight floor-limited rows, two real V10 misses (h42/h44). Cirrus lifetime integrity FAIL disclosed; same-IC fresh-compile replica. |
| Storm Monica | 72 h | PASS | PASS | D6 and all-frame integrity pass. |
| Shipped Swiss | 24 h | PASS | PASS | D6 and all-frame integrity pass; 25 frames on one domain. |

All raw integrity and D6 reports, scoped annexes and their hashes are retained with the [final galleries](evidence/v033/final_w3/index.html) and [manifest](evidence/v033/final_w3/PLOT_MANIFEST.json). Strict24 uses 25 paired frames per domain; full72 uses 73. Swiss uses the same unchanged raw 24 h report for strict and full-window readings.

Primary 0115 all-frame integrity: d02 hour60, 50 horizontal columns; QICE ≤2.282946e-7 kg/kg, QNICE ≤2.4e4/kg versus GPU zero. Fresh-R replica: d03 hour63,112 columns; QICE ≤2.1445631048777614e-8 kg/kg,QNICE ≤45033.7421875/kg versus GPU zero. These are different realized slots in the disclosed cirrus-lifetime class, not identical flags or a rewritten PASS.

<a id="final-manager-ledger"></a>

### Final manager ledger (a8c0fefbd)

# v0.3.3 release ledger — manager edge-case decisions (rule: V033-OWNER-NEAR-THRESHOLD-DEVIATIONS-20261009.md)

Final source be0dd543 (src 46fbb0d9, pkg 33f39763). Science arms executed on 7a0c93ff, whose WN/Mon native executables are
byte-identical to be0 (compile 628de151); Swiss24 runs on be0. Raw FAILs stay recorded verbatim; nothing is renamed PASS.

| # | arm / gate | raw result | criteria (near-threshold · trajectory/no bug · finite/physical · stations) | manager disposition |
|---|---|---|---|---|
| L1 | W3_0408 all-frame integrity, d02 QGRAUP @2026-04-10_00, 5 ring cells | ANNEX FAIL: GPU exactly 0 vs CPU 2.46e-10 kg/kg > cap 1e-10 | trace amount (Owner rule names QGRAUP ≤ 1e-9) · same item as pre-final arms; GPU parent donors exactly 0 (FT02), no port bug found (hunt-mp) · all finite, 0 negative/unphysical · 0408 has no frozen station case | DISCLOSED, NOT BLOCKING (2026-10-09 07:04Z) |
| L2 | F2 0227R 3 h census (7a) | 634,857 negative visits; qc −3.84e-6, qv −4.2e-6 kg/kg | WRF-consistent frozen-source PREADD (pristine solve_em:1932), PD reduces, WRF-order MP cleans interior · 0 nonfinite/new classes | NONBLOCKING_WITH_DISCLOSURE (mass-opus 8a88e29b) |
| L3 | Swiss small-grid firewall cost | +12 % kernels, ~+8 % ordinary root on ≤4096-column grids only | required for E41 on small grids; WN/Mon unaffected (byte-identical) | ACCEPTED, DISCLOSED |
| L4 | W3_0115 strict24 d03 RAINNC h21–24 | FAIL 1.042–1.045 mm > 1.0 | +4.5 % (near-threshold) · IC spread exceeds the bound: frozen A3 proxy pair (GPU IC R vs IC P) d03 RAINNC h20–24 = 1.117–1.150 mm; IC-P replica W3P_0115 vs CPU 1.086–1.133 mm → primary error < intrinsic IC spread; FRESH-R replica (same IC R, same code, only fresh compile/autotune) = 0.845–0.886 mm PASS → same model lands on both sides of the bound from compile-level round-off alone = chaotic sensitivity proven; hunts found no port bug · finite · stations: 0115 criterion4 met (L8) | DISCLOSED, NOT BLOCKING (08:05Z; stations below) |
| L5 | W3R full72 d03 RAINNC h67+ | raw FAIL h67–h72: 1.099553, 1.291418, 1.803185, 2.269664, 2.430662, 2.464927 mm (up to 2.47 mm at h72) | frozen A3 annex: all 8 breaches FLOOR-LIMITED (CPU pair C/X .81–.95) | PASS via frozen annex (no manager waiver needed) |
| L6 | W3_0115 full72 A3 annex | 61 FLOOR-LIMITED, 1 REAL FAIL: d03 V10 lead 44 h RMSE 1.692 > 1.5 m/s (CPU pair C 1.207, C/X 0.713) | single frame, +12.8 % over bound (just above the ~10 % band) · coastal-wind topic: hunt-wind found difference upstream, no port bug · finite · IC-P replica d03 V10 h44 = 1.190 (passes); FRESH-R (same IC R) = 1.623 (also > 1.5; its A3: 8 FLOOR-LIMITED, 2 REAL FAIL = d03 V10 h42 1.725 / h44 1.623, same event) → reproducible under IC R: a ~0.3 m/s domain-mean V10 timing shift around h43–46 in one event, single lead, +8–13 %; proxy pair 1.207; mean V10 shift h43–46 present in both (−0.18…−0.29 m/s) · stations below | DISCLOSED, NOT BLOCKING (08:05Z) — persistent small coastal WS10 excess, v0.3.4 follow-up |
| L7 | W3_0115 all-frame integrity d02 QICE/QNICE @2026-01-18_12 (lead ~60 h), 50 columns | ANNEX REAL FAIL: GPU exactly 0 vs CPU QICE ≤2.28e-7 kg/kg, QNICE ≤2.4e4 /kg | interior (ring 44–52), 9.7–13 km MSL cirrus patch whose lifetime differs at a late lead = trajectory signature; d01 QNICE traces boundary but WRF-consistent; hunt-mp a507b389: no bug/patch · finite/physical (zero ice is a valid state) · station outcome recorded in L8. Same class in disclosed replica W3Rf_0115: d03 QICE ≤2.14e-8 / QNICE ≤4.5e4 @2026-01-18_15, 112 cells GPU exactly 0 (replica, not a release gate) | DISCLOSED, NOT BLOCKING (08:05Z) |
| L8 | Frozen station rules (alisios, 08:00Z) | 0227 raw DAMAGE / class CLEAR (score 94bf52ca) · 0115 raw DAMAGE / class SYSTEMATIC (score db0385fe) | 0115 GPU primary − CPU-R station RMSE: T2 −0.008…−0.012 K (better, B′ significant), RH2 −0.09…−0.10 (better), WD10 −0.27…−0.38° (better, n.s.), WS10 +0.022…+0.068 m/s (worse, significant) on absolute WS10 RMSE ≈3.69→3.76 m/s (+1.8 %); all previous GPU builds 3.71–3.77 → long-standing small coastal WS10 excess, not introduced by this release | criterion 4 MET (not materially worse); SYSTEMATIC WS10 DISCLOSED, NOT BLOCKING; v0.3.4 item: coastal 10 m wind speed |

**Manager release decision:** all eight scoring readers are closed; raw failures remain disclosed under the Owner rule, including the replica wind/cirrus slots above. Station criterion4 is met with the0115 coastal-wind difference disclosed.


Latest ledger supersedes the dated pre-scoring placeholders and old approximate 7% cost. The accepted Swiss ordinary-root cost is approximately 8%, kernel count +12%, on ≤4096-column grids only. WN/Mon byte identity is explicitly scoped; actual7a execution labels are never renamed be0.

## Exact 0227 late-rain range and statistical overview

The published raw report gives h67–h72: 1.099553, 1.291418, 1.803185, 2.269664, 2.430662, 2.464927 mm (up to 2.47 mm at h72). These values correct the earlier understated 1.10-to-1.29 range; the scoring itself is unchanged. All eight raw breaches are FLOOR-LIMITED under A3 (original CPU-pair C/X0.81–0.95). No new failure or gate waiver follows from this prose correction. The [six-run figure](evidence/v033/final_w3/stability_six_runs.png) and [methods/inputs](evidence/v033/final_w3/STABILITY_MANIFEST.json) show all 10 RMSE-bounded fields, d03, using captured metrics only; 0115 grey context is an older GPU proxy, not a CPU pair.
