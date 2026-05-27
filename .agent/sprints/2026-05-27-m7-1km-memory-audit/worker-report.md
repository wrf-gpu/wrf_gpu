Summary: Implemented the M7 1 km memory audit as a measurement-only sprint. Verdict: FITS_WITH_HEADROOM. The derived full-domain 1 km grid is (nz, ny, nx) = (44, 198, 477), 4,155,624 mass cells, exactly 9.0x the 3 km mass-cell count. Static State storage is 0.451 GiB across the 47 fields currently exposed by `src/gpuwrf/contracts/state.py`; the sprint text says 45 fields, so the audit follows the code source of truth and records the mismatch. Full-domain State+Tendencies+Grid resident allocation passed, and a warm one-RK-step synthetic 1 km operational probe passed with peak nvidia-smi memory 7278 MiB on a 32607 MiB RTX 5090.

Files changed:
- `scripts/m7_1km_memory_audit.py`
- `tests/test_m7_1km_memory_audit.py`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/grid_shape_1km.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/audit_command_summary.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_live.stdout.txt`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_live.stderr.txt`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_step.stdout.txt`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_step.stderr.txt`

Commands run and output:

1. `taskset -c 0-3 pytest -q tests/test_m7_1km_memory_audit.py`

```text
...                                                                      [100%]
3 passed in 0.79s
```

2. `taskset -c 0-3 python scripts/m7_1km_memory_audit.py`

```text
{"phase": "gaps", "verdict": "FITS_WITH_HEADROOM"}
{"phase_results": [{"cmd": ["/home/enric/miniconda3/bin/python", "/tmp/wrf_gpu2_1kmaudit/scripts/m7_1km_memory_audit.py", "--phase", "live", "--output-dir", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit"], "phase": "live", "returncode": 0, "stderr_path": "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_live.stderr.txt", "stdout_path": "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_live.stdout.txt"}, {"cmd": ["/home/enric/miniconda3/bin/python", "/tmp/wrf_gpu2_1kmaudit/scripts/m7_1km_memory_audit.py", "--phase", "step", "--output-dir", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit"], "phase": "step", "returncode": 0, "stderr_path": "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_step.stderr.txt", "stdout_path": "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/phase_step.stdout.txt"}], "proof_objects": ["/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/grid_shape_1km.json", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json", "/tmp/wrf_gpu2_1kmaudit/.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md"], "status": "PASS", "verdict": "FITS_WITH_HEADROOM"}
```

Child phase stdout/stderr:
- live stdout: `{"phase": "live", "status": "PASS"}`; live stderr: empty.
- step stdout: `{"phase": "step", "status": "PASS"}`; step stderr: empty.

3. `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`

```text
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
1383 MiB, 32607 MiB, 2 %
```

Proof objects produced:
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json`: AC1 PASS, 47 fields, total State 0.451 GiB, sanity total <= device VRAM.
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/grid_shape_1km.json`: AC2 PASS, source d04 wrfout from `wrf_l3`, derived full-domain 1 km shape (44, 198, 477). It records that the contract-named `wrf_l2` run has no d04/d05 wrfouts.
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json`: AC3 PASS, Gen2 d04 `build_replay_case` loader PASS, full-domain synthetic State+Tendencies+Grid known resident bytes 661,541,896, allocation sampler peak 3482 MiB.
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`: AC4 PASS, cold compile-inclusive wall 70.4196 s, warm one-step wall 0.1032 s, warm sampler peak 7278 MiB, transient estimate 6,969,994,232 bytes including allocator/runtime overhead.
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md`: AC5 produced; recommends transient/fusion work before precision changes.

Risks:
- The full-domain 1 km probe is synthetic because no full-domain 1 km `wrfinput`/`wrfbdy` source exists in the pinned run; Gen2 available 1 km nests are smaller d03/d04/d05 domains.
- The step probe uses nvidia-smi sampling, not Nsight/XLA heap profiling, so transient attribution is approximate.
- Persistent memory is not the bottleneck in this audit. Real IC/BC, full output, restart, and longer forecast windows can still add operational memory pressure.

Handoff:
- objective: document 1 km memory fit and operational gaps for M7 gate #8.
- files changed: listed above; no `src/gpuwrf/**` or governance files modified.
- commands run: listed above, all Python commands were pinned with `taskset -c 0-3`.
- proof objects produced: listed above.
- unresolved risks: synthetic full-domain source, approximate transient attribution, real IC/BC/output/restart memory not measured.
- next decision needed: decide whether M7 targets the derived full 3 km-domain-at-1 km forecast or the smaller existing Gen2 1 km nests, then run an Nsight/XLA memory-profile sprint if the full-domain path remains in scope.
