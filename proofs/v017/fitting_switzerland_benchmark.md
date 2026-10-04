# Fitting Switzerland 3km GPU-vs-12-rank-CPU Benchmark (v0.17) — INTERIM

Status: **GPU 13h warm wall + FACTOR NOT YET MEASURED** — blocked overnight by
sustained sibling-agent GPU+CPU monopoly. Durable automation is armed and will
overwrite this file + `touch FIT_SWITZ_DONE` once the GPU frees. This interim
records the grids built, the measured CPU baseline, and the (important) fp64
fit-ceiling finding.

## Grids built (single-domain Switzerland 3km, FITS-candidates)
Same center/projection/physics as bigswiss: center 46.65N 8.55E, Lambert
(truelat 30/60, stand_lon 8.55), dx=3000 m, 45 vertical levels, p_top 5000.
Physics: mp_physics=8 (Thompson), ra_lw=ra_sw=4 (RRTMG), sf_sfclay=5 (MYNN),
sf_surface=4 (NoahMP), bl_pbl=5 (MYNN), cu=0. Built by re-running WPS
(geogrid/ungrib/metgrid) + real.exe (dmpar -np 4) on the bigswiss GFS
(2023-01-15_00Z, 0p50), 15h WPS window so wrfbdy covers the 13h run.

| e_we=e_sn | mass cols | wrfinput | status |
|-----------|-----------|----------|--------|
| 390 | 389×389 = 151,321 | built+validated | GPU OOM (fp64, this path) |
| 360 | 359×359 = 128,881 | built+validated | GPU OOM |
| 320 | 319×319 = 101,761 | built+validated | GPU OOM |
| 300 | 299×299 =  89,401 | built+validated | GPU OOM |
| 280 | 279×279 =  77,841 | built+validated | GPU retry pending |
| 256 | 255×255 =  65,025 | built+validated | GPU retry pending |
| 224 | 223×223 =  49,729 | built+validated | GPU retry pending |
| 192 | 191×191 =  36,481 | built+validated | GPU retry pending |

bigswiss (461×461 = 211k cols) OOMs single-card fp64 as expected.

## CPU reference (12-rank dmpar gfortran, cores 4-15) — MEASURED per-step rates
Runs were truncated by collision with a sibling 12-rank CPU job (ac1fit) on the
same cores, but the per-step integration rate is a valid measurement:
- **320 (101.8k cols): 1.587 s/step measured (200 steps) → 13h (2600 steps) CPU-mainloop ≈ 4126 s ≈ 1.15 h**
- **390 (151.3k cols): 2.329 s/step measured (1072 steps) → 13h CPU-mainloop ≈ 6057 s ≈ 1.68 h**
(total-wall adds MPI init + I/O, ~7-25 s/13h-run.) Clean uncontended 13h CPU
refs for the GPU-winning grid are queued (orchestrator, cores 4-15).

## fp64 FIT-CEILING FINDING (important; refines the memory anchor)
The v017-rc **standalone-native-init** forecast path OOMs at **≥300×300
(≈89k cols)** on this RTX 5090: the failing allocation is a **near-constant
~16.65–20.09 GiB single op, independent of grid size** (390 and 320 and 300 all
failed on the same ~20 GiB request) — consistent with an **XLA autotuning /
fusion scratch buffer**, NOT per-grid state. The card is **desktop-shared**
(~3.5 GiB used by Wayland/KDE/Chrome), leaving ~28.5 GiB, and the ~20 GiB
contiguous block cannot be carved under fragmentation. So the true fp64 fit
ceiling for THIS path is **well below the assumed ~167k cols** — the largest
fitting grid is being determined empirically (autotune-off retry of 390..300,
then descend 280..192).

Caveat: the memory anchor's "1km Canary fits one card" uses the nested/replay
chunked-BouLac path, which has a smaller per-call peak than this single-domain
standalone-native-init segmented path.

## Blocker (why FACTOR is pending)
GPU monopolized for hours by sibling agents (identity-72h-both, then
canary-8h-gpu, both ahead in the GPU lock queue) and CPU cores 4-15 by ac1fit.
Per the one-GPU-lock / no-CPU-collide rules I waited rather than preempting.
Background jobs/agents are reaped at turn boundaries in this harness; only tmux
windows + on-disk results persist. Automation (repo-persistent scripts in
`proofs/v017/fitswiss_auto/`, tmux windows `fitgpu`/`fitorch`) will pick the
largest fitting grid, run GPU-warm 13h + clean CPU 13h, compute the FACTOR +
spot equivalence (T2/U10/V10/PSFC), overwrite this file, and `touch
proofs/v017/FIT_SWITZ_DONE`.

## Re-run (manual, when GPU/CPU free)
```
# GPU: largest-fitting-grid sweep + 13h (wrapped in the GPU lock)
bash scripts/with_gpu_lock.sh --label fit-switz -- \
  bash proofs/v017/fitswiss_auto/run_gpu_multigrid.sh
# CPU: 12-rank 13h refs on cores 4-15
bash proofs/v017/fitswiss_auto/orchestrator.sh
# finalize (factor + equivalence + this file + FIT_SWITZ_DONE)
bash proofs/v017/fitswiss_auto/finalize.sh
```
