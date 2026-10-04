# d03 Ni non-finite nest instability — mechanism map + pursuit plan (manager 0:1, 2026-06-26)

the user directive (2026-06-26): a CORE dycore instability has priority — pursue the nest-K2 unlock with
**all resources until it works or is proven impossible** (Task #143). This is the carried "Mont-Blanc /
Thompson Ni NaN" class, now shown to be **more general than an extreme-terrain edge case**: the
**20250121 3-dom baseline** (normal terrain, default dt) goes **non-finite at ~step 720 / sim 4320 s /
72 min**, field **Ni** (Thompson ice number) on inner nest **d03 (1 km)**.

## Why it wasn't fixed-with-priority before (honest)
- The validated **production /goal case (20240901 9-nest canary)** ran **24 h finite + VRAM-stable in v0.19.1**
  → not release-blocking for v0.19/0.20/0.21.
- v0.21 **attempted** a fix (Opus+GPT convergent: mass-drain limiter 0.5·MUT + c2a alt-floor; WRF-faithful,
  identity-preserving, zero-regression) — but it **relocated, did not fully stabilize** the Mont-Blanc-extreme
  case → deep boundary fix deferred to v0.21.1.
- It was framed as an **extreme-terrain edge case**; the new 20250121-3-dom-@72min finding shows it is
  **more general** → re-prioritized to CORE now (the user's instinct correct).

## THE decisive first question (settles dynamics-root vs microphysics-symptom)
The v0.21 history says the chain was **boundary acoustic mass-pump → ph detonation → Thompson Ni NaN
DOWNSTREAM**. The manager-side Explore map (below) hypothesizes Ni-division as the ROOT. **These imply
opposite fixes.** So step 1 MUST be: instrument the 20250121 3-dom near step ~720 and capture the
**finite-onset ORDER** across `w, ph (geopotential), mu/muts, theta, p/p', alt(=1/rho)` vs `Ni`.
- If a **dynamics** field (w/ph/mu/theta) goes non-finite **first** → root is the acoustic mass-pump /
  vertical-solver → the REAL fix is in the dynamics (legitimate, WRF-faithful). A Thompson Ni clamp would
  only **mask** the NaN one step later → **FORBIDDEN by project rules (no masking clamps)**.
- If **Ni** goes non-finite **first** while dynamics stays finite → root is the Thompson microphysics path
  → a fix is legitimate ONLY if it matches **WRF mp_thompson's actual** Ni handling (WRF caps Nt_i; an
  arbitrary clamp is masking).

## CAVEAT on the map below
The Explore pass read the **MAIN checkout** `<USER_HOME>/src/wrf_gpu2` (may predate the v0.21-shipped
dycore work). All line numbers + the "v0.21 mass-drain guard NOT implemented / reaches-d03?" claims MUST be
**re-verified against the worktree** `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` (where v0.21 shipped).
The map is DIRECTION, not gospel.

## Candidate sites (verify line numbers in the worktree)
1. **Thompson Ni division** `physics/thompson_column.py` (~1214-1221): `Ni += where(ice_deposition<0,
   ice_deposition / max(xmi, XM0I), 0)`; `xmi = AM_I*xdi**3` can reach `XM0I=1e-12` on thin high-altitude
   d03 columns during sublimation → small negative deposition ÷ 1e-12 = huge negative Ni increment.
   No production upper-bound on Ni (the `assert_physical_bounds(...,1e12,enabled=debug)` is debug-only;
   the `999e3/rho` cap is init-only). **Check WRF: does mp_thompson bound Nt_i in the step?**
2. **Acoustic mass-continuity cap** `dynamics/acoustic_wrf.py` (~66-68, ~644-666):
   `TEMPORARY_MU_CONTINUITY_CFL_FRACTION = 1e-3` hardcoded, **not domain/dx-scaled** → on the 1 km d03
   (~1/27 the column mass of d01) the same fractional cap permits relatively ~27× larger mass oscillations.
3. **Vertical implicit acoustic solver** `dynamics/vertical_implicit_solver.py`: tridiagonal coeffs built
   identically for all domains (no dx scaling) — the path that should tolerate Cz>1. Verify whether the
   v0.21 c2a alt-floor / mass-drain guard is present here AND reaches the d03/nested path.
4. **Nest physics coupling** `coupling/physics_couplers.py` `thompson_adapter` (~1361-1380) receives `grid`
   but discards it (`del grid`) → no domain-aware behavior. Confirm whether ANY stability limiter is
   parent-only / keyed off domain index such that d03 runs unguarded.

## Plan (Task #143)
1. (0:3, GPU) Finite-onset ORDER diagnostic on 20250121 3-dom near step 720 → dynamics-root vs Ni-root.
2. (0:3 + manager) Root-cause the first-mover; verify against the WORKTREE code + WRF reference.
3. Fix WRF-faithfully (dynamics if dynamics-root; WRF-matched bound only if genuinely Ni-root). No masking.
4. Re-run the K2 nest dt-ladder (54→90 / n_sound 10→7) on the now-stable nest → the speedup prize.
5. Also settle: input-specific vs general vs a v0.20/0.21 regression vs the v0.19.1-24h-stable nest.
6. If a fundamental dycore limit with no faithful fix → report impossibility honestly.

Floor unaffected: v0.22 still ships K2 opt-in single-domain (1.80×) + findings; this is the bigger,
parallel prize.
