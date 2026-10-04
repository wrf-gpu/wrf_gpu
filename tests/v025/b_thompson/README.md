# Thompson kernel validation and measurement

`tests/v025/conftest.py` forces CPU. GPU regression uses
`asserted_backend_checks.py`, which directly invokes the checks without pytest
collection and asserts `jax.devices()[0].platform == "gpu"`.

`run_oracles.py --require-gpu` runs the copied, hash-bound Gate-2/L3a/L3b
oracles. `gpu_validation.py` adds the active precipitation and cold-mixed
oracles outside `tests/v025`, after an explicit GPU assertion. Copied source
provenance and unchanged tolerances/manifests are in `oracle_reference/`.

`prepare_columns.py` prepares actual day/night PROD workloads on CPU through
the retained coupler. These inputs are workload evidence; the retained WRF
column outputs provide independent expected results.

`component_probe.py` asserts GPU, compiles and warms before capture, checks
paired outputs for regression, and saves optimized HLO. `profile_calls.py`
captures CUDA graph nodes with nsys; `profile_metrics.py` counts all kernel,
memcpy and memset rows. `roofline_probe.py` estimates logical bytes from HLO
shapes and measured kernel durations while ncu counters are blocked (B18).
Its bandwidth estimates are **[I]**, not measured DRAM traffic.

Every GPU command must use the manager lease and `scripts/with_gpu_lock.sh`,
CPUs 12,13,28,29, nice 10, OMP 2 and `timeout`. Measurements use a detached
commit checkout. CPU preparation/analysis uses those CPUs and nice 19. Keep
all caches, temporary files and profiler artifacts under the lane disk path;
raw profiler directories are 0700 and files 0600.

The kept switches are default-off: `GPUWRF_THOMPSON_COLUMN_SED=1` selects
private column loops; `GPUWRF_THOMPSON_SED_FP32=1` selects WRF REAL flux work;
`GPUWRF_THOMPSON_COLUMN_LAYOUT=1` keeps the full body in `(column, level)`
layout and restores the retained interface shape. Trace-time flag changes
require clearing the caller's JIT cache or using a fresh process.

`GPUWRF_THOMPSON_NATIVE_REAL=1` assigns REAL work at process boundaries;
`GPUWRF_THOMPSON_FULL_COLUMN=1` requires that policy and fp32 private
sedimentation, and runs the explicit mp8 source/sink call in one column
kernel. It reuses retained equations with runtime table references. mp28
continues through its retained aerosol implementation. The experimental
`GPUWRF_THOMPSON_SED_PREP_FUSED` switch stays off when measuring full-column
fusion. Use `profile_calls.py --compare-full-column` for the matched pair.

`GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT=1` skips cold/ice transfer families
only when their column process gates exclude existing ice, freezing and
nucleation. It retains density assignments and number caps. The full branch
remains for active columns. `full_column(..., return_events=True)` provides
device counters for nonfinite input, inadmissible input and nonfinite work
before fallback. The probe reports these in a separate diagnostic call
outside the timing capture; production counter integration remains separate.

`GPUWRF_THOMPSON_FULL_COLUMN_WARM_EXIT` / `..._SED_EXIT` (default: the value of
`..._EARLY_EXIT`) skip warm-rain/evaporation rates and fall speeds in columns
without cloud/rain (or precipitating species) above R1; the retained update code
runs with exact zero rates/speeds, so outputs are bitwise unchanged (TH03).
Under native REAL, DOUBLE `x**n` with integral constant `n` uses products
(`_dpow`; `GPUWRF_THOMPSON_DPOW_PRODUCTS=0` restores libdevice pow for A/B).
`TH03`-style matched timing: `<USER_HOME>/wrf_gpu2_lanes/b-thompson/TH03/variants_probe.py`.
