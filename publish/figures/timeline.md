# Figure Spec: Milestone Timeline

Purpose: show the project progression from governance bootstrap to public release staging, with the current M7 skill-blocked status visible.

Canvas: wide landscape figure. X axis is milestone sequence. Y axis is evidence maturity.

ASCII layout:

```text
Evidence maturity
^                         M6: savepoint + Tier gates         M7: pipeline/perf done,
|                         parity ladder and M6 close         skill still blocked
|                  M4/M5: dycore + first physics             |
|           M2/M3: backend + state layout                    v
|    M1: fixtures/oracles                           M8: publish staging
| M0: governance
+---- M0 ---- M1 ---- M2 ---- M3 ---- M4 ---- M5 ---- M6 ---- M7 ---- M8 ---> milestone
```

Key annotations:
- M0: AgentOS / governance bootstrap.
- M1: WRF oracle and fixtures.
- M2: backend bakeoff decision path.
- M3: GPU state and grid skeleton.
- M4: minimal dycore.
- M5: first physics suite.
- M6: small-step parity, coupled short forecast, Tier-4 gates.
- M7: 3 km daily pipeline, D2H=0, restart, speedup, 1 km memory audit; forecast skill remains blocked.
- M8: release and publication package staging.

Rendering notes: color completed systems gates in green, blocked skill/corpus gates in amber/red, and future/release steps in gray. Do not imply M7 operational skill closure.
