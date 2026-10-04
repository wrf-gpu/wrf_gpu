# CPU-WRF truth reference for the failing nest — exists, full 24h (2026-06-26)

**MAJOR FINDING (loop check #3):** a complete CPU-WRF reference for the EXACT failing case
(`20250121`, 3-dom nest) already exists on disk:

- **Location:** `<DATA_ROOT>/wrf_downscale/runs/20250121/cpu/`
- **Frames:** `wrfout_d01/d02/d03`, **73 frames each**, `2025-01-21_18:00:00 → 2025-01-22_18:00:00`
  = a **full 24-hour forecast**, every 20 min, all three nested domains.
- **Status:** `rsl.out.0000` ends `wrf: SUCCESS COMPLETE WRF` (12-rank CPU-WRF, clean finish).

## What this proves
1. **It is a GPU-PORT BUG, definitively** — CPU-WRF runs this exact nest (same terrain, same Cz>1:
   d02 Cz=1.40, d03 Cz=1.05) **finite for 24 h**, while our GPU port dies at **72 min**. NOT an input
   problem, NOT a CFL impossibility, NOT "never-validated → maybe unstable". The port has a real bug.
2. **We have the truth reference to validate the fix against** — exactly the user's no-shortcuts requirement
   ([[feedback_pod_runs_fully_valid_no_shortcuts_2026_06_26]]). The fixed GPU steep-terrain output must MATCH
   this CPU-WRF reference within operational tolerance (T2/U10/V10/PSFC/Q…), not merely "be finite".

## How it accelerates the fix
- **Divergence diagnostic:** compare the GPU d01/d02/d03 state to the CPU-WRF reference frames in the
  54-72 min window BEFORE the GPU blow-up → the FIRST field that drifts from truth (while still finite)
  pinpoints the bug and distinguishes microphysics-internal from dynamics. Fed to the trace subagent
  `a2030e2e9f1569136`.
- **Validation:** post-fix, GPU vs this reference over ≥72 min (ideally the full window) = the trustworthy
  proof for the AI-training pod data.

## Note
The GPU run starts at the same 18:00 with 18-min(d03)/20-min(history) cadence — GPU d03 frames at
18:18/18:36/18:54 before death; CPU-WRF d03 frames at 18:20/18:40/19:00/19:20(+80)/19:40(+100)... (the
slight cadence offset is the history-interval vs d03-step; align by sim-time when comparing).
