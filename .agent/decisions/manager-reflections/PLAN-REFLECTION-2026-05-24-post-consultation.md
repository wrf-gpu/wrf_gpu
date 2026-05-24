# Plan Reflection — Post-Consultation (2026-05-24)

**Trigger**: External deep-AI consultation on the M6 dycore blocker, requested by the project principal after the HYBRID exit-rule fired and the operator bug-hunt returned `NO-BUG-LOCALIZED`. Consultation package: `/mnt/server/downloads/_andere/wrf_gpu2_consult.zip`.

## Consultation verdict (summary, in the consultant's framing)

> **Finish this as Option B, but make the savepoint harness the product for the next sprint.** Do **not** keep tuning ADR-023 with stabilizers. Do **not** start a clean ICON4Py/Dinosaur/Pace substrate rewrite yet. Use WRF itself as the numerical compiler: extract WRF small-step savepoints, reproduce them in JAX one operator at a time, then optimize the GPU path.

> **Stop trying to make a WRF-like dycore stable from the outside. Rebuild the WRF small-step from the inside, under savepoint parity, then optimize.**

## Diagnosis refinement

The consultation refined the manager's "architecture inadequacy" diagnosis:

> "I would not say 'WRF small-step is unportable.' WRF runs. I would say: **The current project architecture lacks the instrumentation needed to distinguish a wrong recurrence, wrong staging, missing WRF scratch state, and wrong source-equation coupling.**"

> "ADR-023's minimalist carry was a good scientific bet, but it has now failed the real d02 test. ADR-021's clamp-stripped failure does not prove that a full WRF small-step port is dead; it proves that a full WRF-shaped port **without savepoint parity** is also just guessing with more variables."

The key architectural lesson: **WRF compatibility cannot be validated only at 1 hour or 24 hours. At this point it must be validated at the acoustic substep level.**

## Adopted plan changes (user-approved 2026-05-24)

### 1. M6 split into M6a / M6b / M6c

- **M6a** — WRF small-step savepoint parity (sanitizer-off).
- **M6b** — Honest 1h Canary d02 (sanitizer-off, RMSE inside envelope).
- **M6c** — 6h/24h Gen2 statistical consistency (AceCAST-style probabilistic envelope).

### 2. Sprint sequencing — B-direct, savepoint-first

| Sprint | Scope |
|---|---|
| M6B0 | WRF `module_small_step_em` savepoint harness + JAX comparator + first coefficient parity |
| M6B1 | Coefficient parity (column → 16×16 → d02) |
| M6B2 | Tridiagonal solve parity |
| M6B3 | Scratch-state parity (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields) |
| M6B4 | Acoustic recurrence parity (one substep → all substeps in one RK stage) |
| M6B5 | Full dycore step parity (physics off, boundary off, sanitizer off) |
| M6B6 | Coupled step parity (physics on, boundary on, sanitizer off) |
| M6b | Honest 1h Canary d02 |
| M6c | Gen2 consistency 6h + 24h |
| M6-perf | Optimization (only after M6c) |

### 3. A-probe is NOT scheduled

The consultation's argument carries: a partial-scratch hybrid could improve T2 from 137 K to 40 K and trap the project in weeks of hidden-staging whack-a-mole. The savepoint harness gives surgical attribution that A-probe cannot. A-probe is allowed only as a single disposable sprint with a hard kill gate if a specific evidence-driven justification surfaces later.

### 4. Option C is fallback only

`C-primary = JAX reimplementation of ICON4Py/MPAS/WRF-proven vertical-implicit patterns, after B proves WRF-small-step parity is too expensive or structurally unsuitable.` Not before.

### 5. Option E (new) — optional shadow GPU-WRF lane

Research-scout sprint may evaluate AceCAST and `FahrenheitResearch/wrf-gpu-port` as shadow benchmarks / business-continuity backup. Not a replacement for B. Deferred unless principal explicitly authorizes.

### 6. ADR status updates

- ADR-023 → **SUPERSEDED-PROVISIONAL** (kept as scientific reference, not production architecture).
- ADR-024 → **ACCEPTED** (warm-bubble is permanently an operator-sanity diagnostic).
- ADR-025 (**TO BE DRAFTED**) → WRF savepoint harness + B-direct port ladder. Drafted during M6B0, finalized at M6B0 close.

### 7. Validation gates (binding)

| Gate | Binds |
|---|---|
| G1 Savepoint parity (sanitizer-off, per-operator delta) | M6B0–M6B6 |
| G2 10-step real d02 replay (no nonfinites, no caps) | end of M6B6 |
| G3 1h d02 honest forecast (RMSE inside envelope, theta bounded) | M6b |
| G4 6h/24h Gen2 probabilistic consistency | M6c |
| G5 Performance (wall-clock < 28-rank CPU WRF, 0 H2D/D2H in loop) | post-M6c |

### 8. Risk kill-gates

- M6B0 cannot patch WRF Fortran in ≤2 sprints → escalate; consider AceCAST instrumentation alternative.
- Comparator finds >15 savepoints diverging at step 2 even with WRF carries added → trigger external WRF-expert human review.
- No measurable speedup at M6B5 → reopen performance section before M6b dispatch.
- External WRF expert unobtainable → manager dispatches Codex critical-review as substitute and flags gap to principal.

### 9. M7 alignment

M7 (Canary operational v0) continues on CPU WRF as the operational backend until M6b passes, then migrates field-by-field as savepoint parity expands. Public messaging does not imply GPU-native operation before M6b. M7 may be backed by an E-lane shadow GPU-WRF for business continuity if principal authorizes.

## Total budget estimate

- **10–17 sprints** to M6 close (M6B0 through M6-perf).
- Wider than the prior 5–9 estimate. Each sprint produces hard per-operator parity evidence instead of forensics on billions of sanitized nonfinites.

## Out-of-band action item

Consultation recommends an **external human WRF expert review** at the M6B0 close (a two-hour review by someone who has touched WRF `module_small_step_em`, MPAS-A dycore, or ICON HEVI numerics). The review target is **"are these savepoints sufficient to prove the small-step port?"**, not "is JAX good?". The manager will flag this to the principal at M6B0 close.

## Principal approvals received (2026-05-24)

All 8 approval items from the manager presentation table answered affirmatively. Plan committed; M6B0 dispatched.
