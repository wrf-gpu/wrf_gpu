# P-BUNDLE Questions / Blocked Gates

## 2026-07-02 — P1 single-scan equivalence gate data missing

`proofs/perf/single_scan_equiv.py` cannot start in this worktree because `_build_real_case()` requires:

`data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`

The local `data/` tree only contains small fixtures/manifests, and the required run directory was not present under the expected path. The P1 knob itself is implemented with default OFF and unit-tested; the real-case equivalence JSON cannot be regenerated here until that run directory is restored or the gate is repointed to an available fixture.
