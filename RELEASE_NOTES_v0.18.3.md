# Release notes — wrf_gpu v0.18.3

**v0.18.3 is two bugfixes, bit-identical on the default path.** It makes the
all-7-island `--max-dom 9` nest compile + integrate + write output (it previously
hung forever in compile), and fixes the nested pipeline ignoring the namelist
`history_interval`. Default forecast numerics are unchanged.

## What v0.18.3 fixes

### 1. `--max-dom 9` (all-7-island nest) compile blowup

**Symptom:** an all-7-island `--max-dom 9` nested run pinned CPU+GPU for hours
emitting only `[Compiling module jit__advance_chunk] Very slow compile` /
`slow_operation_alarm` banners — it never finished compiling, never integrated,
wrote no output.

**Root cause (HLO-confirmed):** the Thompson microphysics sedimentation /
fall-speed scans materialized **static `s64[nz]` vertical-index arrays**
(`jnp.arange(nz)`) as scan operands. Across the **9 distinct domain shapes** of
the all-7 nest, XLA replicated and constant-folded those static index vectors
inside the large `_advance_chunk` programs and spent unbounded time folding /
lowering them. (Dump evidence + source mapping: `proofs/v018/maxdom9_fix/report.md`
— `s64[44] iota` in `thompson_column.py` `_fill_down` / `_sed_one_species`.)

**Fix:** the affected scans (`_sed_implicit_q`, `_fill_down`, `_sed_one_species`)
now thread a scalar `int32` index through the scan **carry** and scan over `None`
with an explicit `length`, so no static index vector is built. Iteration order and
values are identical, so the change is **bit-identical**.

**Proof (reference RTX 5090):**

| | Before v0.18.3 | After v0.18.3 |
|---|---|---|
| `--max-dom 9` `_advance_chunk` compile | never terminates (hours, no output) | all **9/9 domain-shape compiles bounded**: ≤ **409 s** cold, ≤ **22 s** warm cache-hit |
| Integration | never reached | reached, **stable**, mean GPU util **~85 %**, peak VRAM 13.8–14.2 GiB |

### 2. Nested output ignored `history_interval`

**Bug:** the nested pipeline hardcoded **hourly** output cadence
(`round(3600/dt)`) and ignored the namelist `time_control.history_interval`, so a
case configured for non-hourly output (e.g. 20-minute) would still only emit
hourly `wrfout`.

**Fix:** the writer now honors the per-domain `history_interval` and computes the
output valid time / `XTIME` from `own_step · dt`. **Hourly gates are unchanged**
— `ceil(3600/dt)` equals the previous `round(3600/dt)` for the v0.18.2 AC1_FIT
d01/d02/d03 timesteps (18/6/2 s → 200/600/1800 steps, first valid time unchanged).

**Proof:** a d03 `wrfout` written at forecast **+5 min** (`Times = …_18:05:00`,
`XTIME = 5.0`, **106 numeric fields finite**) under a `history_interval=5` namelist;
the old hardcoded-hourly path would not have emitted it.

## Scope and non-claims

- **Bit-identical default path.** 26/26 `examples/switzerland_d01` `wrfout` fields
  exact vs pre-fix, `max_abs_diff_overall = 0.0`. No physics thresholds, formulas,
  masks, or dtypes changed. 16 Thompson-column CPU tests pass.
- **Not a speed change.** The all-7 `--max-dom 9` forecast remains compute-heavy;
  this release makes it *compile and run at all* (bounded), it does not speed up
  the integration. Cold compile is a one-time cost; the persistent JIT cache makes
  later runs fast (warm cache-hit ≤ 22 s/shape).
- **MEASURED scope:** all-7 `--max-dom 9` staging case + AC1_FIT 9/3/1, reference
  RTX 5090 (32 GiB), fp64, mp8 Thompson / MYNN / Noah-MP / RRTMG.

## Proof objects

- `proofs/v018/maxdom9_fix/report.md` — full HLO root-cause + measured proofs
- `proofs/v018/maxdom9_fix/bit_identity_compare.json` — 26/26 exact, 0.0
- `proofs/v018/maxdom9_fix/history_cadence_static_check.json` — hourly non-regression
