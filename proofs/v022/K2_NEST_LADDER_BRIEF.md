# K2 NEST dt-ladder — brief (next GPU task for 0:3, after the R0 synthesis)

K2's single-domain win (Switzerland-128: 1.8× lossless, dt 18/n_sound 7) is proven but **single-domain**. Before any
default change, validate it on the **NEST** — the real wallclock cost + the real CFL risk (steep terrain / Mont-Blanc)
live there. This is the v0.22 K2 gating follow-up (manager B-lite plan, `V022_OPENERS_RESULTS_AND_SHIP_PLAN.md`).

## Task
Run the dt/n_sound ladder on the **3-dom Canary** (NOT the 9-nest — too expensive; 3-dom is the tractable nest proxy).
Baseline = current 54/18/6 s per-domain dt, n_sound 10. Scale the ROOT dt up the same factors as the single-domain
ladder, preserving the parent_grid_ratio chain:

| Rung | root dt | d02 | d03 | n_sound |
|---|---|---|---|---|
| N1 | 54 | 18 | 6 | 10 |
| N2 | 60 | 20 | 6.67 | 9 |
| N3 | 72 | 24 | 8 | 8 |
| N4 | 90 | 30 | 10 | 7 |

## Per-rung gate (the real questions the single-domain run could not answer)
- **Per-domain max-Courant readout** — especially **inner nest d03 (1 km Alpine)**: report horizontal |U|·dt/dx AND
  vertical w·dt/dz. The single-domain run kept vertical Cz < 1 even at dt 18; the **nest over steep terrain is where
  Cz>1 / the Mont-Blanc mass-pump risk actually appears** — this is THE thing to find.
- finite + bounded + **mass/energy conservation** (the nest is the fragile path).
- operational tolerance vs N1 (T2/U10/V10/PSFC/T) — does the bigger-dt nest forecast stay acceptably close?
- s/fc-h win vs N1 (report s/fc-h, NOT s/step — the win is fewer steps).

## Cost note
Each dt rung is a fresh ~19 min 3-dom fused cold compile (dt is a compile-time constant) → ~1.5-2 h total; reuse the
AOT warm cache where the shape repeats. Time-box if a rung diverges early (a diverging rung IS the CFL-limit answer).

## Outcome → v0.22 decision
- If a higher-dt rung is **CFL-safe + stable + in-tolerance on the nest** → that is the **v0.22 K2 headline** (a real
  validated nest speedup); propose flipping it (or shipping it config-default) with the canary-benchmark gate.
- If steep-terrain CFL bites above some dt → ship K2 **opt-in** (config lever, lossless, documented single-domain win)
  and record the nest CFL ceiling as the reason. Either way it's an honest, shippable result.

Reuse `proofs/v022/k2_dt_ladder/` harness (run_rung.py / scale_namelist.py / compare_to_r1.py / courant_diag.py),
retargeted to the 3-dom Canary input. GPU via scripts/with_gpu_lock.sh; do NOT kill 0:2 corpus. Report to 0:1 per rung.
