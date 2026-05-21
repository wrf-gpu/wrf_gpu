# Sprint Tracker — Live Dashboard

Manager-maintained. 20-min cadence (AFK).

## Currently in flight (3 codex parallel — at cap, on critical-path)

| Window | Sprint | AI | Started | Wall | File ownership |
|---|---|---|---|---|---|
| `worker-s3zzzz` | M5-S3.zzzz SW cldprmc+spcvmc broadband | codex gpt-5.5 xhigh | 14:38 | 16-32h | `rrtmg_sw.py` only + harness `cldprmc_sw_*`+`spcvmc_*` records |
| `worker-s3zzzzz` | M5-S3.zzzzz LW cldprmc+rtrnmc broadband (NEW) | codex gpt-5.5 xhigh | 15:25 | 16-32h | `rrtmg_lw.py` only + harness `cldprmc_lw_*`+`rtrnmc_*` records |
| `worker-m6s3` | M6-S3 surface layer + bounded Noah-MP (NEW) | codex gpt-5.5 xhigh | 15:26 | 30-48h | NEW `physics/surface_layer.py, noah_mp.py`, `wrf_sfclay_harness.f90`, ADR-012+013 |

**3-way disjointness verified**:
- s3zzzz: rrtmg_sw.py + cldprmc_sw_/spcvmc_ harness records + lw_/sw_ validator names (interface-freeze)
- s3zzzzz: rrtmg_lw.py + cldprmc_lw_/rtrnmc_ harness records + lw_ validator names (interface-freeze)
- m6s3: NEW physics/surface_layer + noah_mp + sfclay harness; reads Gen2 via io.validation

## Closed this tick (2 BIG Opus verdicts)

| Sprint | Verdict |
|---|---|
| **M6-S2 coupled forecast driver** | Opus **ACCEPT-WITH-MINOR-FOLLOWUPS** (22 R-findings: 16P/6F-disclosed) |
| **M5-S3.zzz LW closeout** | Opus **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-LW-TAUMOL** (16/16 bands taug+fracs PASS intermediate; broadband still fails → S3.zzzzz binding) |

## M6-S2 reviewer 3 critical disclosures (binding downstream)

- R-16: **1s dycore_dt_s cap** is permanent guard, dynamics:physics 60× mismatch → **M6-S5 cannot use wall numbers; M6-S7 must lift cap for real RMSE**
- R-17: **Finite-state guard** saturates all clips at 24h → **M6-S4 must measure conservation BEFORE sanitize_state runs**
- R-18: **mu_bdy first-step-replay** (Gen2 has only wrfinput_d02) → **M6-S3 prereq F-S3-2: extend accessor for wrfout_d02_* OR document M6-S8 interior-only**

## M5 sprint table (live)

| Sprint | Status |
|---|---|
| M5-S0 through M5-S3.zzz (16 sprints) | ✓ all CLOSED |
| **M5-S3.zzzz SW cldprmc+spcvmc broadband** | 🟡 codex worker |
| **M5-S3.zzzzz LW cldprmc+rtrnmc broadband** | 🟡 codex worker (NEW, parallel under interface-freeze) |
| M5-S1.z Thompson collision tables | ⚪ optional |

**M5 RRTMG PARITY** = S3.zzzz + S3.zzzzz BOTH close (parallel ~24-48h instead of sequential 48-96h)

## M6 sprint table (live)

| Sprint | Status |
|---|---|
| M6 plan + S1 + S2a + S2 | ✓ all CLOSED |
| **M6-S3 surface + Noah-MP minimum** | 🟡 codex worker (NEW; F-S3-1/2/3 prereqs bundled) |
| M6-S4 Tier-2 coupled (must measure PRE-sanitize_state) | ⚪ queued |
| M6-S5 ADR-007 4× verdict (F-S5-1/2/3 prereqs: lift dycore cap, end-to-end wall, denominator choice) | ⚪ queued |
| M6-S6 Tier-3 TSC1.0 | ⚪ queued |
| M6-S7 Tier-4 probtest (must run with cap+guard disabled) | ⚪ queued |
| M6-S8 operational Gen2 + closeout | ⚪ queued |

## M7 sprint table

| Sprint | Status |
|---|---|
| M7 plan: scout + critic + manager-amendments | ✓ all CLOSED |
| M7-S0..S8 | ⚪ queued |

## Big-picture critical path (refined)

```
NOW (3 codex parallel)
  ├─ M5-S3.zzzz SW broadband → Opus → SW PARITY
  ├─ M5-S3.zzzzz LW broadband → Opus → LW PARITY (parallel under interface-freeze)
  └─ M6-S3 surface+Noah-MP → Opus → first operationally-meaningful U10/V10/T2/qv2

When (SW PARITY + LW PARITY): M5 RRTMG complete → ADR-009 final
When M6-S3 close: M6-S4/S5/S6/S7 4-way parallel → M6-S8 → M6 GREEN
M6 GREEN + M5 RRTMG complete → M6-S8 operational T2 binding gate meaningful → M7-S0 dispatch
M7 GREEN → M8 release
```

**Calendar refined**: M5 RRTMG PARITY ~24-48h (SW+LW parallel); M6 close 5-8 days; **end-goal landing ~2.5-3 weeks** (further accelerated by 3-way parallel)

## File-ownership snapshot

- `coupling/{driver, boundary_apply, physics_couplers}.py`: M6-S2 LANDED; M6-S3 will touch surface_adapter only
- `contracts/{state, precision}.py`: M6-S2 amended; FROZEN
- `physics/rrtmg_sw.py`: M5-S3.zzzz worker active
- `physics/rrtmg_lw.py`: M5-S3.zzzzz worker active (parallel, disjoint)
- `physics/surface_layer.py, noah_mp.py`: NEW M6-S3 worker active
- `scripts/wrf_rrtmg_harness.f90`: BOTH SW + LW workers extending (interface-freeze names enforced)
- `scripts/wrf_sfclay_harness.f90`: NEW M6-S3
- `io/**`: M6-S2a CLOSED + accessor; M6-S3 may extend for land state
- Other physics: CLOSED, frozen

## Recent ticks

- 14:58: M6-S2 + S3.zzz workers DONE; 2 Opus reviewers dispatched
- 15:25 (this tick): both Opus verdicts landed (ACCEPT + PARTIAL); **M6-S3 + M5-S3.zzzzz workers dispatched in parallel**; 3 codex parallel cap honored
- Next: 15:45 (20-min cadence)
