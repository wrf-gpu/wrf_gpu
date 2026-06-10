# V0.14 Canary h08/h10/h18 Field-Drift Analysis (Fable)

Date: 2026-06-10. Sprint:
`.agent/sprints/2026-06-10-v014-fable-canary-h08-drift-analysis/sprint-contract.md`.
CPU-only; the active GPU run was not touched.

## 1. Verdict

**BLOCK — root cause found, proven, and fixed (uncommitted).** The growing
PSFC/MU/P/PH drift is a **real native-init lateral-boundary bug**, not
comparator/tolerance/writer noise and not bounded free-run divergence: the
standalone d01 consumes its 6-hourly `wrfbdy_d01` forcing at the hourly replay
cadence — **6x too fast** — and from h11 onward the lateral boundary is
**frozen at the t=66h boundary value for the remaining 61 hours**. The current
72h run can never pass the gate and its tail produces no usable parity
evidence. A 2-file, root-domain-only fix is implemented and CPU-proven
bit-exact (`proofs/v014/lbc_cadence_root_cause.*`, gate `rc=0`); the run is
already stopped (`gpu_rc=143` at h26) — review/merge, then relaunch. Canary
h08/h10/h18/h24 is the only current long-gate drift
evidence; there is NO completed v0.14 Switzerland GPU-vs-CPU field compare —
Switzerland CPU72 truth exists but the GPU run is pending, and it remains an
unrun cross-region falsifier after this decision (its `interval_seconds=10800`
would hit the same bug 3x-fast if launched unfixed).

## 2. Root-cause ranking

| # | Hypothesis | Evidence for | Evidence against | Next falsifier | Wall-clock |
|---|---|---|---|---|---|
| 1 | **d01 wrfbdy LBC cadence bug (3600 vs 21600 s)** — PROVEN | GPU spec-zone MU at h1..h10 == wrfbdy levels 1..10 to **0.0 Pa**; h11..h20 frozen at level 11 (1390.5); CPU truth == linear ramp to 0.05 Pa; domain-mean MU mirror (CPU +182 / GPU −176 Pa over seg 1); d01/d02 drift identical ⇒ parent origin | none | done — `proofs/v014/lbc_cadence_root_cause.py` rc=0 | done |
| 2 | GPU near-surface pressure vapor-light (~−210 Pa PSFC floor) — CONFIRMED secondary, pre-existing | CPU `PSFC−(p_top+MU+MUB)` = +203..220 Pa == vapor column weight; GPU ≈ 0±tens at every lead; both PSFCs exactly match their own written P/PH extrapolation (writer exonerated) | quasi-static, not the growing drift; present at h1 adjudication | rerun h1 falsifier post-cadence-fix; expect PSFC bias ≈ −210 ± small; then a dycore moist-pressure sprint | 1-2 h analysis post-rerun |
| 3 | Writer/output semantics (PSFC diagnostic) | — | GPU PSFC == extrapolation of own P/PH to 0.0 Pa; formula WRF-faithful | closed | — |
| 4 | Comparator/pairing/tolerance issue | — | CPU spec zone matches wrfbdy to 0.05 Pa through the comparator's own pairing; per-lead bias ≈ −RMSE (spatially uniform physical signal) | closed | — |
| 5 | Nest-coupling (d02 boundary package) | — | d01 shows the identical drift at every lead (dPSFC within ~5% of d02) | closed | — |
| 6 | Dynamics/physics kernel bug as drift driver | — | drift erratic per-hour tendencies track wrfbdy level jumps exactly; kernel residuals bounded (step-1 proofs) | post-fix 24h rerun slope | covered by rerun |
| 7 | Expected bounded free-run divergence | — | bias sign-consistency 1.0, spatially uniform −600 Pa surface pressure ≠ chaos; mass mirror is deterministic forcing error | closed | — |
| 8 | PB/MUB static spikes as part of drift | — | constant at every lead, 5-cell frame only (h18 split: interior RMSE 0.0003 vs frame 10-20) | already classified; separate writer/frame item | — |

## 3. Field-drift summary (d02, GPU−CPU; h1..h24 RMSE | bias)

Final trajectory note: the run was terminated with `gpu_rc=143` (SIGTERM) at
frame h26; the final 72h compare and grid-delta atlas never ran (MISSING), so
the h24 intermediate compare is the final drift evidence. The GPU d01 spec
zone is frozen at 1390.5 Pa (wrfbdy level 11) from h11 through the last frame
on disk (h26), exactly as the root cause predicts.

| Field | h1 | h4 | h8 | h12 | h18 | h24 |
|---|---|---|---|---|---|---|
| PSFC (Pa) | 124 \| −117 | 337 \| −335 | 607 \| −603 | 354 \| −341 | 541 \| −537 | 407 \| −403 |
| MU (Pa) | 98 \| +85 | 95 \| −89 | 356 \| −347 | 137 \| −116 | 317 \| −315 | 168 \| −163 |
| P (Pa) | 39 \| −8 | 137 \| −98 | 266 \| −178 | 139 \| −71 | 227 \| −117 | 162 \| −75 |
| PH (m2/s2) | 48 \| +25 | 96 \| −59 | 227 \| −170 | 165 \| −38 | 266 \| −141 | 239 \| −125 |
| PB (Pa) | 4.52 static, frame-only, constant all leads | | | | | |
| MUB (Pa) | 9.28 static, frame-only, constant all leads | | | | | |
| T (K) | 0.26 | 0.86 | 1.13 | 2.45 | 3.06 | 2.80 |
| U (m/s) | 0.34 | 2.70 | 4.20 | 6.27 | 5.55 | 4.52 |
| V (m/s) | 1.69 | 1.73 | 4.33 | 7.42 | 6.98 | 6.38 |
| QVAPOR (kg/kg) | 1.9e−4 | 5.2e−4 | 8.3e−4 | 1.0e−3 | 1.2e−3 | 1.3e−3 |
| T2 (K) | 0.39 | 1.16 | 1.45 | 1.61 | 0.76 | 0.59 |
| U10 (m/s) | 0.61 | 1.15 | 1.90 | 2.10 | 2.56 | 3.28 |
| V10 (m/s) | 1.23 | 1.28 | 2.36 | 4.28 | 2.76 | 2.64 |

Structure: PSFC bias/RMSE = −0.99 at EVERY lead h17–h24 (the error is a
spatially uniform mass offset, zero growth in spatial variance — deterministic
forcing error, not chaotic divergence); the same holds approximately for MU.
The h1–h10 oscillation tracks the wrfbdy level values being consumed 6x too
fast; from h11 the frozen-boundary regime makes the GPU a constant-LBC run, so
the GPU−CPU mass gap oscillates with the CPU's real synoptic cycle (PSFC RMSE
peaks 645 at h7, decays to 407 at h24 as CPU MU happens to swing back toward
the frozen value — NOT convergence; the next synoptic swing would reopen it).
Mass-field drift is domain-wide, not boundary-band (manager h18 split:
frame≈interior for PSFC/MU/P/PH/U/V). Winds/T degrade as the interior adjusts
to wrong-time forcing (V RMSE peaks 7.4 at h12, stays 6+ kept up by the frozen
boundary). dPSFC ≈ dMU − ~210 Pa at every lead: the constant gap is lane 2
(vapor-light GPU surface pressure), the growing/oscillating part is lane 1.

## 4. Decision on the 72h GPU job

**Stop now — already done** (`gpu_rc=143`, last frame h26): correct call.
Every frame was forced with wrong-time LBC and from h11 the boundary was
frozen, so h48/h72 would have added no gate-relevant information; the gate
verdict was already determined (FAIL). The h1–h26 frames preserved on disk are
sufficient diagnostic evidence (they bit-prove the bug). Do not launch
Switzerland GPU before the fix is merged (same bug, 3x-fast at its
`interval_seconds=10800`).

## 5. Exact next manager commands

```bash
# 1) run already stopped (gpu_rc=143). Confirm GPU free; clear a stale lock if any:
RUN_ROOT=/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z
nvidia-smi | head -15   # then: rm -f /tmp/wrf_gpu_validation_gpu.lock if stale

# 2) review + rerun the root-cause/fix gate (CPU-only, ~40 s)
git diff src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/integration/d02_replay.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 \
  python proofs/v014/lbc_cadence_root_cause.py   # expect rc=0, verdict ..._FIX_GATE_PASS

# 3) regression set already run green; optionally repeat
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 \
  python -m pytest tests/test_m6_boundary_apply.py tests/test_v013_tost_wrfbdy_fix.py \
  tests/test_p0_1a_nesting.py tests/test_gwd_operational_wiring.py -q

# 4) commit, then relaunch the 72h Canary gate with the SAME runbook command
#    (new $RUN_ROOT timestamp), then the h1 falsifier on the fresh output:
#    expect PSFC bias to collapse from −117 toward ≈ −210±small static
#    (lane 2), MU h1 bias from +85 toward ~0.
```

## 6. Proposed fix (implemented, uncommitted; manager merges)

- Files: `src/gpuwrf/integration/nested_pipeline.py`
  (`_root_boundary_cadence_override`, applied to the root domain only) and
  `src/gpuwrf/integration/d02_replay.py` (`load_wrfbdy_boundary_leaves`
  synthesized terminal level + strip helper refactor). Proof artifacts:
  `proofs/v014/lbc_cadence_root_cause.{py,json,md}`.
- WRF-faithful: WRF advances each boundary record with its `_BT*` tendency
  over `bdyfrq == interval_seconds`; linear interpolation between record
  values at that cadence is the identical forcing. Children keep
  `update_cadence_s == parent_dt`; hourly replay-history paths are untouched
  (no `interval_seconds` in their boundary meta).
- Proof gate: bug emulation == live GPU spec zone to 0.0 Pa (h1–h20); fixed
  cadence == CPU-WRF truth spec zone to 0.000 Pa (h1–h20) and 9.4e−6 Pa at
  h72; plumbing 3600→21600; 23 affected tests pass.

## 7. Context-sparing handoff

- objective: diagnose v0.14 Canary 72h h08 FAIL drift; verdict BLOCK+FIXED.
- root cause: d01 standalone consumed 6-hourly wrfbdy at 3600 s cadence (6x
  fast; frozen at the 66h record from h11) — live since v0.12.0 native-init;
  plausibly the long-standing KI-9 lead-time wind/mass divergence driver
  (TOST case-3 PSFC 525 Pa same path).
- proof: GPU spec-zone MU == wrfbdy levels to 0.0 Pa; fix == CPU truth to
  0.000 Pa; `proofs/v014/lbc_cadence_root_cause.*` rc=0.
- files changed: `nested_pipeline.py` (root cadence override),
  `d02_replay.py` (terminal wrfbdy leaf level), proof script/json/md, this
  report. All uncommitted; manager merges.
- commands run: CPU-only NetCDF analysis of d01/d02 h1–h20, wrfbdy decode,
  proof script, py_compile, 4 test files (23 passed, 1 skipped).
- secondary lane (NOT fixed): GPU near-surface pressure vapor-light ~−210 Pa
  (CPU PSFC−dry-column == vapor weight; GPU ≈ 0) — quasi-static, pre-existing,
  needs its own dycore sprint; sets the post-fix h1 PSFC expectation ≈ −210.
- known/static lanes unchanged: PB/MUB 5-cell frame, radiation timing ~−20
  min, bounded MYNN/RRTMG step-1 residuals.
- decision needed: merge the fix, relaunch the 72h gate (run already stopped,
  gpu_rc=143, no final compare/atlas); hold Switzerland GPU until merged
  (interval 10800 → 3x-fast same bug).
- Canary h08/h10/h18 is the only current long-gate drift evidence;
  Switzerland is an unrun cross-region falsifier.
- when the full 72h compare/atlas of the FIXED run arrives: check PSFC bias ≈
  flat ~−210 (lane 2), MU bias ≈ 0-slope, and whether U/V/T2 envelopes
  collapse toward h1-class values; growth-rate/variance structure then becomes
  the real free-run divergence measurement.
