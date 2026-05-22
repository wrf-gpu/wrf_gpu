# Gemini Architecture Review — Is c2 dycore sprint the right next move?

You are the orthogonal third opinion. User authorized c2 architecture sprint after extensive consensus formation (manager + scout + methodology meta-review). Before codex spends 3-5 days on c2, sanity-check.

## Project context (top level)

GPU-native WRF-compatible regional NWP. Target: Canary Islands 3km daily forecast on RTX 5090, ≥4× faster than 28-core CPU WRF.

**Status:**
- M0..M5 complete (all physics: Thompson microphysics, MYNN PBL, RRTMG SW+LW radiation, surface)
- M6 partial: coupling, validation, all physics integrated
- **Constitutional 4× speedup target ACHIEVED at 44.33× measured** (c1-A7)
- Operational stability: NOT closed

## The c1 dycore problem (11 iterations, 1 day clock)

We built a clean-room Klemp-Skamarock acoustic core in JAX. Each iteration found and fixed something:

| Iter | What fixed | Outcome |
|---|---|---|
| A1 | Acoustic core (Klemp 2007 §3a-c + tridiag) | stable isolated |
| A2 | Scalar advection mass-flux form | tests pass |
| A3 | bughunt3 4 fixes (periodic wrap, dz floor, smdiv, ph adv) | neutral/worse |
| A4 | Buoyancy in vertical w eq | (test branch later) |
| A5-H3 | n_acoustic patch | inconclusive (wrong patch site) |
| A6 | EMPIRICAL BISECTION isolated bug to advection | 30min, surgical target |
| A7 | Horizontal momentum FLUX form (was advective) | closed horizontal + 44.33× speedup |
| A8 | Vertical eta-metric sign | closed vertical-w isolated |
| A9 | Cross-component bisection: discovered c1-A8 instrumentation was wrong, pairs actually stable | identified map-factor as suspect |
| A10 | Map-factor extension — KILLED based on warm-bubble result | wrong direction |
| A11 | Diagnostic pressure (vs prognostic that goes negative -239k Pa) | closed one bug, exposed next layer |

**Discoveries**:
- Warm-bubble retest WITH buoyancy: c1 dycore CORRECTLY models physics for 300s (w_max 5.99 m/s vs WRF reference 5-10; bubble centroid 2517m vs ref 2500m). Blows up at 350s.
- RMSE-growth: LINEAR pattern pre-failure; nonfinite at step 49; pressure goes to -239,234 Pa
- Coupled 1h probe: 86-100% sanitize firing rate even after surgical fixes

**Pattern confirmed**: c1 needs multiple WRF mechanisms (map factors msfx/msfy, hybrid-eta coefficients c1h/c2h, smdiv divergence damping, 6th-order hyperdiffusion, Rayleigh sponge, monotonic limiter) ALL TOGETHER. Sequential fixes won't converge.

## The c2 sprint plan (authorized, codex starting now)

Port architectural patterns (NOT code) from:
- **Primary**: Pace/FV3 decomposition (GridData with metrics, named AcousticDynamics + HyperdiffusionDamping + RayleighDamping + FillNegativeTracerValues as explicit modules)
- **Secondary**: ICON4Py (explicit NonHydrostaticConfig + metric_state + IntermediateFields dataclasses)
- **JAX style**: Dinosaur/NeuralGCM (pytrees + scan-friendly pure functions)
- **Numerical truth**: WRF dyn_em source

Sprint: c2-A1 (3-5 days), 8 ACs (ADR amendment, metrics proof, hybrid-eta proof, damping skeletons, scan proof, conservation proof, integration warm-bubble, decision gate).

If c2-A1 GREEN → 4-5 implementation sprints (~3 weeks) → M7 dispatch.
If c2-A1 RED → escalate (E3SM/SCREAM port, ML emulator, or M6 throughput-only close).

## Your task

Give a third orthogonal opinion. Manager + GPT scout + Opus meta-reviewer all converged on this c2 plan. Before 3-5 days of codex grind, sanity-check.

Answer in ~800-1500 words:

### 1. Is c2 the right next move?
After 11 c1 iterations + 4 bug-hunts converging on "need full WRF architecture not surgical fixes," is porting Pace/ICON4Py architecture the right answer? Or is there something obvious that everyone missed?

### 2. Pace/ICON4Py vs alternatives
Manager picked Pace as primary architectural reference. Alternatives considered: HOMMEXX, SCREAM, Dinosaur, NeuralGCM. Did manager pick right? Anything else worth considering (PsyACC, COSMO-DYCORE on GPU, JAX-CFD, OpenIFS GPU port)?

### 3. Hidden architectural risks
- Pace is GT4Py (DSL → CUDA). c2 is JAX. Architectural patterns transfer; but does JAX have hidden limitations that Pace's GT4Py doesn't? (XLA compilation pressure, vmap limitations, tridiag solve cost)
- ICON4Py is also GT4Py.
- Dinosaur/NeuralGCM is JAX but global spectral.
- **Are we trying to do something nobody has done: regional WRF C-grid hybrid-eta dycore in pure JAX?** If so, what's the risk?

### 4. The hybrid-eta question
WRF uses hybrid-η vertical coordinate (default since v3.9). Gen2 boundary files (wrfinput_d02, wrfbdy_d01) carry it via c1h/c2h/c3h/c4h coefficients on mass + face. c1 currently treats vertical as pure σ. The c2 sprint will add hybrid-eta support. **Is this enough?** Or do we need to also handle TERRAIN representation (orography), reference profile mismatch, base-state-vs-perturbation pressure decomposition, etc?

### 5. Constitutional 4× target preserved?
c1-A7 measured 44.33× speedup on a (broken-stability) c1 dycore. **Will c2's added complexity (map-factors, hybrid-eta arrays, 6 damping/limiter modules, scan carries) preserve speedup?** Or are we trading 44× → 4× by adding WRF realism?

### 6. Honest recommendation
Manager + scout + meta-review say GO. You're the orthogonal angle. Should I:
- (A) Start c2 NOW as authorized
- (B) Postpone c2 — first do a smaller architectural feasibility check (1-2 day spike)
- (C) Pivot to alternative — admit JAX-native is wrong for this problem
- (D) Something else

## Honesty rules

- If you'd say "go", say so directly
- If you'd say "wait", be specific about what new info is needed
- If you'd pivot, be specific about alternative
- Honest about uncertainty
- Don't validate just because everyone else converged — that's exactly when an orthogonal angle has highest value

You are NOT the deciding vote — codex c2-A1 is starting now. But manager will weigh your opinion within the next 30-60 min of c2-A1's early work.
