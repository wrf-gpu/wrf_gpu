# Worker Report — M7 Corpus Bridge

Summary: PARTIAL. Implemented the bounded Option A bridge at the requested harness paths and rebound `DEFAULT_M6_GEN2_RUN_DIR` to the surviving complete `20260521_18z_l3_24h_20260522T133443Z` cycle. The non-operational bridge preserves the operational N=10 default, lowers only explicit `--non-operational` runs to N=5, emits `PASS_PROBATIONARY` at N>=5, and emits `PASS_PROBATIONARY_PENDING` with `corpus_gate="BLOCKED_CORPUS"` when the current N=3 corpus is still below the probationary floor.

## Files Changed

- `src/gpuwrf/io/gen2_accessor.py`
- `src/gpuwrf/validation/tier4_rmse_harness.py`
- `scripts/m7_run_tier4_rmse_harness.py`
- `tests/test_m7_tier4_rmse_harness.py`
- `tests/test_m7_default_gen2_run_dir.py`
- `.agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json`
- `.agent/sprints/2026-05-27-m7-corpus-bridge/worker-report.md`

## Commands Run + Output

`find /mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z -maxdepth 1 -type f -name 'wrfout_d02_*' | wc -l`

Output:
```text
0
```

`find /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z -maxdepth 1 -type f -name 'wrfout_d02_*' | wc -l`

Output:
```text
25
```

`taskset -c 0-3 pytest -q tests/test_m7_tier4_rmse_harness.py tests/test_m7_default_gen2_run_dir.py`

Output:
```text
3 passed in 0.76s
```

`python -m compileall -q src/gpuwrf/validation/tier4_rmse_harness.py scripts/m7_run_tier4_rmse_harness.py`

Output:
```text
<no output; exit 0>
```

`taskset -c 0-3 python scripts/m7_run_tier4_rmse_harness.py --non-operational --ending-cycle 20260525_18z --output .agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json`

Key output:
```text
status=PASS_PROBATIONARY_PENDING
corpus_gate=BLOCKED_CORPUS
member_count=3
needed_members=2
finite_rmse_record_count=6
```

`taskset -c 0-3 pytest -q tests/test_m7_*.py`

Output:
```text
1 failed, 39 passed in 3.41s
FAILED tests/test_m7_s0a_schemas.py::test_aifs_ingest_manifest_validates_and_cites_existing_gen2_paths
AssertionError: /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/l3/namelist.wps
```

`if test -e /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/l3/namelist.wps; then echo present; else echo absent; fi`

Output:
```text
absent
```

## Proof Objects

- `.agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json`
- `tests/test_m7_default_gen2_run_dir.py`
- Focused pytest proof: `3 passed in 0.76s`

## Risks

- AC5 is not green in this environment because the full M7 test glob depends on an external WPS case path that is currently absent under `/mnt/data/canairy_meteo`. This worker did not modify `/mnt/data/canairy_meteo/**` or out-of-scope manifests/tests.
- The current probationary bridge remains pending, not pass, because the mounted corpus has N=3 and needs +2 complete pinned-grid members to reach the explicit non-operational N=5 floor.

## Handoff

Objective: bounded Option A bridge plus default Gen2 run-dir rebind.

Proof status: bridge implementation and focused tests pass; `probationary_smoke.json` produced; full M7 glob is PARTIAL due external data absence.

Next decision needed: restore/update the missing `20260520_18z_72h` WPS case artifact or revise that out-of-scope manifest, then rerun `taskset -c 0-3 pytest -q tests/test_m7_*.py`.
