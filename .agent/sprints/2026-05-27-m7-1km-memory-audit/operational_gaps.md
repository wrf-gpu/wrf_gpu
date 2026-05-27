# M7 1 km Memory Audit Operational Gaps

Verdict: FITS_WITH_HEADROOM

Fit answer: yes for the measured/probed scope. Static State storage is 0.451 GiB on the derived full-domain 1 km grid [44, 198, 477]. Headroom is 77.68% using the strongest available basis (7631536128 bytes of 34190917632 bytes) if a live basis is available.

Grid provenance: the contract-named wrf_l2 run was checked. The sprint-named wrf_l2 run has no wrfout_d04/d05 files in this worktree. The proof object therefore records the available Gen2 l3 1 km nests and uses the sprint objective's full 3 km d02 -> 1 km horizontal scaling for the fit audit.

Top 5 resident State fields by VRAM:
- w: 32.43 MiB (float64, shape=[45, 198, 477])
- ph: 32.43 MiB (float64, shape=[45, 198, 477])
- ph_total: 32.43 MiB (float64, shape=[45, 198, 477])
- ph_perturbation: 32.43 MiB (float64, shape=[45, 198, 477])
- p: 31.70 MiB (float64, shape=[44, 198, 477])

Downcast candidates and constraints:
- Do not downcast `mu`, pressure, geopotential, or pressure-gradient/acoustic paths without a new reviewed precision artifact; those are FP64-locked.
- `u`, `v`, `theta`, `qv`, Thompson hydrometeors, number fields, `qke`, and lateral wind/theta/qv boundaries are already stored as FP32-gated fields in `contracts/precision.py`.
- `w` remains the main large candidate that is not already FP32, but ADR-007 requires a sound-wave and Tier-4 operational-impact test before changing it.
- Surface FP64 fields are small relative to 3D pressure/geopotential fields, so they are not first-order memory wins.

Kernel-fusion / transient-reduction candidates:
- Fuse or alias RK/acoustic save-family scratch (`*_save`, `t_2ave`, `ww`, `mudf`, `ph_tend`) where profiler evidence shows duplicated live ranges.
- Reduce pressure-gradient and vertical-solver temporaries by keeping coefficient construction inside the acoustic scan with XLA aliasing/buffer donation evidence.
- Batch Thompson/MYNN/surface physics without materializing independent full-field tendency copies beyond the persistent `Tendencies` contract.
- Keep boundary replay fused with the post-physics step so lateral-boundary padded arrays do not create separate full-domain staging buffers.

What would have to change to make 1 km operational:
- Replace the synthetic full-domain probe with a real full-domain 1 km `wrfinput`/`wrfbdy` source or explicitly scope M7 to the smaller Gen2 d03/d04/d05 nests.
- Capture an Nsight or XLA memory profile for the one-step full-domain probe before claiming transient headroom beyond this nvidia-smi estimate.
- If headroom tightens under real IC/BC, attack transient buffers first; persistent State storage alone is not the limiting footprint in this audit.
- Any precision change must cite ADR-007/Tier-4 evidence, not memory pressure alone.
