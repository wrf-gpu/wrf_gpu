# Noah-MP native REAL validation

`GPUWRF_NOAHMP_NATIVE_REAL=1` selects WRF REAL work in the Noah-MP entries.
The default is off. Integer categories, Python loop bounds and inactive paths
remain unchanged. The active `opt_run=3` configuration has no pristine WRF
DOUBLE island; the sole `S_NODE` DOUBLE local belongs to excluded GROUNDWATER.

The frozen energy, phenology, water and coupled runners are loaded unchanged by
`run_savepoints.py`. Reports go to a private directory; fixture and runner SHA256
values are recorded. GPU validation uses direct Python with a literal backend
assertion; `tests/v025/conftest.py` must not select the backend.

The fp64 water gate remains `<1e-6 mm`. The manager registered the native rule
before measurement: each column's balance residual must be at most 1.5 times
pristine WRF REAL on the identical soil boundary, and at most WRF ERROR's
0.1 mm/timestep threshold. `run_water_real_gate.py` requires that registration
and a pristine oracle output. NP04's registration and source hashes are under
`<USER_HOME>/wrf_gpu2_lanes/b-noahmp/NP04/`.

`build_water_oracle.py` builds privately from pristine preprocessed Noah-MP
source, changing only SOILWATER's accessibility. Its caller reproduces WATER's
soil boundary, evaporation routing and runoff units with BTRANI=0, matching the
existing standalone gate. Routine bodies stay verbatim. The caller preserves
RUNSUB initialization despite the routine's OUT declaration. Full canopy/snow
evolution is outside that standalone water boundary.

`prepare_real.py` records real PROD day/night history inputs and loader defaults
for both shapes. These provide workloads; frozen pristine savepoints provide
expected fields. `component_probe.py` compiles the whole Noah-MP step, asserts
GPU plus lock ownership, preserves full outputs and optimized HLO, and captures
three calls per case with CUDA profiler and NVTX ranges. Its timing is a
component result, not whole-forecast S1 evidence.

Run only immutable snapshots. Set explicit JAX/autotune/CUDA/Triton/temp
directories below the lane artifact directory. CPU setup uses cores
12,13,28,29 at nice 19; GPU runs use the b-thompson lease at nice 10 and a timeout.
While `/tmp/wrf_gpu2_quiet` exists, defer builds, preparation and exports.
