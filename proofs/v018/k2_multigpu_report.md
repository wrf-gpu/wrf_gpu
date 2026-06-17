# v0.18 K2 Multi-GPU / Cluster Lab Report

- verdict: PASS (gated regions)
- feature status: EXPERIMENTAL, LAB-TESTED ONLY, default OFF, ACCEPT-AS-EXPERIMENTAL
- gate: `GPUWRF_K2_EXPERIMENTAL=1` (genuinely controls the feature; see Env Gate section)
- fake/local devices: 3 of 3
- flag-off graph unchanged (default path bit-identical): True
- halo exchange check: True
- sharded operator check: True
- ppermute+operator check: True
- real d02 one-step fake-mesh check (interior + shard seams): True
- operational bit-identical: False

## Honest Boundary-Condition Status (READ FIRST)

**Periodic decomposition validated; the physical (specified) boundary is NOT yet faithful.**

- Strict interior and internal shard seams reproduce the single-GPU reference **bit-for-bit at roundoff** (1e-13...1e-9). This proves the `lax.ppermute` periodic-halo substrate is correct.
- The **global physical x-boundary ring is NOT faithfully decomposed**: the periodic decomposition runs a periodic BC at the true domain edge, while the single-GPU reference uses WRF's specified/edge boundary treatment. They diverge **by design** by up to theta 0.036 K / p 2.89 Pa / mu 1.03 at x in {0,1,158}.
- This residual is a real periodic-vs-specified BC mismatch, **not** a seam bug and **not** roundoff.
- NOT-FAITHFUL: the periodic x-decomposition runs a periodic BC at the global physical edge, not WRF's specified/edge boundary; interior + internal shard seams ARE bit-for-bit vs the single-GPU reference
- K2 is physically valid for **periodic / idealized domains only, until specified-boundary decomposition lands**.
- The earlier draft widened the theta tolerance 1e-2 -> 4e-2 to make the boundary ring 'pass'. That has been **reverted**: theta atol is back to 1e-2, the boundary ring is **excluded from the pass gate** (not hidden behind a loosened tolerance), and the full-state allclose below is therefore expected `False` because it includes the boundary ring.
- full-state allclose at un-widened tolerance (includes boundary ring, expected False): False

## Step 0 v0.17 Ports

- Checked v0.18 trunk for the v0.17 nested performance fixes; the three commits were absent from ancestry and not patch-equivalent.
- Ported `209b8656` edge-only boundary interpolation, `ee016b1e` committed-seed churn fix, and `191bbd2a` root-async `block_between` sync.
- Validated the port with the domain-tree and edge-only boundary tests before the K2 lab proof.

## Design

- The default single-GPU path remains `run_forecast_operational`; disabled sharding selects that exact function object (proven bit-identical, default cannot regress).
- K2 uses x-domain decomposition over a named JAX `pmap` axis with `lax.ppermute` halo exchange.
- State, tendencies, metrics, and terrain are partitioned into device-resident x slabs; halo refreshes happen inside device computations.
- Column-local physics runs on local slabs. Horizontal dycore and acoustic scratch leaves refresh halos through the existing sharded context hooks.
- The decomposition is x-periodic; it does NOT reproduce WRF specified-boundary forcing (see boundary status above). `run_boundary=True` is intentionally rejected.

## Correctness (full-state max diffs vs single-GPU reference)

Note: theta/p/mu full-state max diffs are dominated by the NOT-FAITHFUL physical-boundary ring; see the Region Split for the gated (interior + seam) result that actually passes.

- `theta`: full-state max_abs=0.03551146842102071 atol=0.01 exact=False
- `u`: full-state max_abs=0.015713684528722238 atol=0.02 exact=False
- `v`: full-state max_abs=7.731312208125729e-05 atol=0.0001 exact=False
- `w`: full-state max_abs=0.003106231707939139 atol=0.004 exact=False
- `mu`: full-state max_abs=1.0320253684185445 atol=1.2 exact=False
- `p`: full-state max_abs=2.894869251467753 atol=3.5 exact=False
- `ph`: full-state max_abs=0.2353300628346915 atol=0.25 exact=False
- `qv`: full-state max_abs=0.0 atol=0.0 exact=True
- `qke`: full-state max_abs=5.421010862427522e-20 atol=0.0 exact=False
- `rain_acc`: full-state max_abs=0.0 atol=0.0 exact=True

## Region Split (interior + seams GATED; boundary ring UNGATED/NOT-FAITHFUL)

- `theta`: GATED interior max_abs=7.617018127348274e-12 pass=True; GATED seams max_abs=1.7053025658242404e-13 pass=True; UNGATED physical_boundary_ring max_abs=0.03551146842102071 (NOT-FAITHFUL, excluded from gate)
- `u`: GATED interior max_abs=3.340261400808231e-10 pass=True; GATED seams max_abs=4.0039083160081645e-12 pass=True; UNGATED physical_boundary_ring max_abs=0.015713684528722238 (NOT-FAITHFUL, excluded from gate)
- `v`: GATED interior max_abs=4.816924636941167e-12 pass=True; GATED seams max_abs=3.954651710269541e-12 pass=True; UNGATED physical_boundary_ring max_abs=7.731312208125729e-05 (NOT-FAITHFUL, excluded from gate)
- `w`: GATED interior max_abs=5.7559380872707067e-11 pass=True; GATED seams max_abs=5.810560366192874e-13 pass=True; UNGATED physical_boundary_ring max_abs=0.003106231707939139 (NOT-FAITHFUL, excluded from gate)
- `mu`: GATED interior max_abs=3.65253072232008e-09 pass=True; GATED seams max_abs=2.9103830456733704e-11 pass=True; UNGATED physical_boundary_ring max_abs=1.0320253684185445 (NOT-FAITHFUL, excluded from gate)
- `p`: GATED interior max_abs=2.4883775040507317e-09 pass=True; GATED seams max_abs=3.4924596548080444e-10 pass=True; UNGATED physical_boundary_ring max_abs=2.894869251467753 (NOT-FAITHFUL, excluded from gate)
- `ph`: GATED interior max_abs=5.195033736526966e-09 pass=True; GATED seams max_abs=2.9103830456733704e-11 pass=True; UNGATED physical_boundary_ring max_abs=0.2353300628346915 (NOT-FAITHFUL, excluded from gate)
- `qv`: GATED interior max_abs=0.0 pass=True; GATED seams max_abs=0.0 pass=True; UNGATED physical_boundary_ring max_abs=0.0 (NOT-FAITHFUL, excluded from gate)
- `qke`: GATED interior max_abs=5.421010862427522e-20 pass=True; GATED seams max_abs=5.421010862427522e-20 pass=True; UNGATED physical_boundary_ring max_abs=5.421010862427522e-20 (NOT-FAITHFUL, excluded from gate)
- `rain_acc`: GATED interior max_abs=0.0 pass=True; GATED seams max_abs=0.0 pass=True; UNGATED physical_boundary_ring max_abs=0.0 (NOT-FAITHFUL, excluded from gate)

## Env Gate (GPUWRF_K2_EXPERIMENTAL genuinely controls the feature)

- gate value this run: 1 (enabled); gate enforced: True; experimental path run: True
- With `GPUWRF_K2_EXPERIMENTAL` unset, `--check k2-lab/operational-forecast/d2` runs the **default-path flag-off proof only** and does NOT run the experimental sharded path.
- With `GPUWRF_K2_EXPERIMENTAL=1`, the experimental sharded path runs. `GPUWRF_K2_PARTITIONS` supplies the partition count when `--devices` is omitted.
- (`--no-require-env-gate` exists for internal/legacy regeneration; production callers should rely on the gate.)

## Multi-Node Status (wired, UN-EXERCISED)

- wired: True; exercised here: False; distributed_initialized this run: False
- single-node multi-GPU (pmap) only here. The multi-node init path (initialize_k2_distributed_from_env) is wired into main() before device enumeration and is invoked when GPUWRF_K2_MULTI_NODE=1 plus the coordinator/process env vars are set; it is UN-EXERCISED on this one-GPU lab box.
- No claim is made that multi-node works: it is **designed and wired but un-exercised** (one-GPU lab box). Single-node multi-GPU via `pmap` is what the fake mesh exercises.

## One-GPU / Fake-Mesh Limits

- hardware available here: one RTX 5090; fake CPU devices used for correctness
- weak-scaling storage shape: local_nx=53, halo_width=8, storage_multiplier=1.3018867924528301
- CPU fake-device wall time is not a GPU or cluster scaling measurement.
- Real NVLink/NVSwitch/NCCL behavior, compute/halo overlap, and multi-node InfiniBand behavior are unmeasured.

## NCAR / UCAR Run (runnable)

This command is runnable as written; it requires a d02 replay fixture passed via `--run-dir`.
On this workstation that fixture is the d02 replay case below; NCAR/UCAR must obtain or stage an
equivalent d02 replay run directory (a WRF run dir with the d02 met/state files the replay loader reads)
and point `--run-dir` at it. Set `--devices` (or `GPUWRF_K2_PARTITIONS`) to the number of visible devices.

```bash
GPUWRF_K2_EXPERIMENTAL=1 \
PYTHONPATH=src JAX_ENABLE_X64=true \
JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=3 \
python scripts/verify_multigpu_dgx_sim.py --check k2-lab --devices 3 \
  --run-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z \
  --forecast-halo-width 8 \
  --output proofs/v018/k2_multigpu_lab.json \
  --status-md proofs/v018/k2_multigpu_report.md
```

On a real N-GPU node, drop the CPU/XLA env vars and set `--devices N` (real GPUs):

```bash
GPUWRF_K2_EXPERIMENTAL=1 GPUWRF_K2_PARTITIONS=8 \
PYTHONPATH=src JAX_ENABLE_X64=true \
python scripts/verify_multigpu_dgx_sim.py --check k2-lab --devices 8 \
  --run-dir /path/to/d02_replay_run_dir \
  --forecast-halo-width 8 \
  --output proofs/v018/k2_multigpu_lab.json \
  --status-md proofs/v018/k2_multigpu_report.md
```

For multi-node, launch one process per host/GPU set and add `GPUWRF_K2_MULTI_NODE=1`, `GPUWRF_K2_COORDINATOR_ADDRESS`, `GPUWRF_K2_PROCESS_ID`, `GPUWRF_K2_PROCESS_COUNT`, and optional `GPUWRF_K2_LOCAL_DEVICE_IDS`. `initialize_k2_distributed_from_env` is invoked from `main()` before device enumeration when those are set. This path is wired but un-exercised here.
