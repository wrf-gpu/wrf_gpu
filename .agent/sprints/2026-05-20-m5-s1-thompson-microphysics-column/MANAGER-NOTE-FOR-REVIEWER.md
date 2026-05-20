# Manager Note for M5-S1 Attempt-4 Reviewer

Written 2026-05-20 by manager (Claude Opus 4.7 1M-context) after worker A4 closed attempt 4 with commit `4119d2a` and contractual handoff to `BLOCKER-m5-s1-attempt4-tolerance.md`. Three AI opinions now exist for this decision; please read all three before deciding.

## Attempt-4 outcome (confirmed in artifacts)

Worker A4 applied the diagnosis-prescribed narrow fixes from `diagnosis-report.md`:
1. Process-order refactor — **diagnosis predicted T 0.32K → 0.084K; actual result 0.32K → 0.0403K** (better than predicted).
2. Ni-deposition gated to sublimation branch — **Ni error 1.4M → 127k (91% reduction)**.
3. Real sedimentation bypass via locally-patched WRF Thompson object (replaces the `dz=1e30` hack).
4. ADR-005 strict tolerances restored.
5. MORNING-REPORT.md preserved.

Result against strict ADR-005 tolerances: still failing — `qc/qi/qs` max-abs ~1.5e-4 vs `abs=1e-10` target. T-error reduction was huge; mass-partition error remains. Worker correctly hit the contract's `BLOCKER-on-strict-tolerance` branch and stopped, requesting M5-S1.x for exact WRF table/moment export.

All other ACs pass: Fortran harness oracle (structural anti-tautology), 1 kernel launch, 0-byte HLO diff, Tier-2 conservation pass, `validate_agentos.py` clean, all governance constitutional rules hold.

## Three AI opinions on the table

### Opinion 1 — Manager (Claude Opus 4.7) prior, leaning Path A → softened after Gemini

Path A (close-and-defer + M5-S1.x parallel with M5-S2). Rationale: M5-S1's ADR-005-stated goal ("prove a real branchy WRF column scheme is implementable under JAX") is done; Fortran harness is the architectural win; the 87%/91% T/Ni error reductions are load-bearing physics evidence; coupled-run M6 is the right context to measure residual.

### Opinion 2 — Diagnosis Codex (read-only, gpt-5.5 xhigh)

Pre-attempt-4 read-only probe established the error budget (`diagnosis-report.md`):
- 55-65% from process-order — confirmed by attempt-4 result, fixed.
- 20-30% from lookup-table proxies — confirmed by attempt-4 result, NOT fixed (this is what blocker reports).
- 5-10% from Ni handling — confirmed by attempt-4 result, fixed.
- <10% other.

Recommended Path B with diagnosis-prescribed narrow fixes for attempt 4 (which worker A4 then executed). Has not weighed in on attempt-4 outcome yet — the diagnosis was pre-attempt-4.

### Opinion 3 — Gemini 3.5 high-flash (side-runner, read-only)

Full text: `gemini-third-opinion.md` (same folder). Summary:

**Recommends Path B (fix-cycle now)** despite acknowledging M5 momentum cost.

Load-bearing argument: cites `tier1_thompson_parity.json` relative errors directly — `qc` rel-err `0.999998`, `qr` `4.5e7`, `qs` `1249.6`, `qg` `9.8e8`. Argues the JAX kernel is "physically divergent" from WRF — accepting it bakes proxy-laden patterns into M5-S2/3/N kernels that subsequent developers copy. Cites ADR-005:29-30's "constrained subset that still claims progress" failure mode.

Counterargument (Gemini constructed against its own recommendation): trade-wind inversion + PBL discovery via M5-S2 MYNN is the actual Canary operational bottleneck; delaying that for microphysics tables is architectural misallocation.

**Novel reviewer check Gemini surfaced** (neither manager nor diagnosis codex flagged this): when lookup tables get baked into JAX trace, HLO may unroll into massive nested `select`/`conditional` trees → compile-OOM or trace-time blowup on the 4-core constrained worker (`sprint-contract.md:373`). Worth adding to whichever path's sprint contract you write.

Confidence: high. Track record: 1/3 toward Gemini role promotion (`.agent/references/dispatching-gemini.md`).

**Caveat**: Gemini is new to this project. Its opinion is one input among three, not a deciding vote. The reviewer (you) is the binding authority.

## Manager's question to you (the reviewer)

Given:
1. Worker attempt 4 satisfies the diagnosis-prescribed fixes (order/Ni/sedimentation) AS PREDICTED and HARDER than predicted — 0.32K → 0.04K is real, not noise.
2. Strict ADR-005 tolerances still violated by `qc/qi/qs` mass-partition errors dominated by lookup-table proxies.
3. The lookup-table-export work (~10-18h Fortran-table-reading + JAX transcription) is mechanical and well-scoped.
4. M5-S2 MYNN PBL is the next-scheme dependency.
5. Three AI opinions now lean: manager prior=A → softened, diagnosis=B (pre-result), Gemini=B (post-result with cited per-field rel-errors).

**Reviewer decision space**:
- **Path A**: Accept attempt-4 with documented residual debt + named M5-S1.x sub-sprint scope. Dispatch M5-S1.x + M5-S2 in parallel.
- **Path B**: Reject attempt-4. Dispatch attempt-5 with lookup-table export scope (~10-18h codex) before M5-S2.
- **Path C** (alternative): Accept-with-required-fixes where the fix is the M5-S1.x sub-sprint dispatched immediately as a contractual continuation of M5-S1, not a separate sprint — but M5-S2 is held until M5-S1.x closes (serial). Splits the difference: strict tolerances still get fixed, but the architectural M5-S1 milestone-progress claim is preserved.

**Reviewer guidance**: please name your decision (A/B/C/other), per-AC pass/fail/blocked, and which fixes are in-scope vs deferred. Independent of decision: please incorporate Gemini's HLO-unroll compile-OOM check into the sprint contract for whichever path you choose (manager will add it if you nominate it as a required check).

— Manager (Claude Opus 4.7 1M-context), 2026-05-20
