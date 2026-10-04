# G2 continuation (Fable-5, X-High, CPU-ONLY) — finish + mark DONE

Your prior G2 session already committed (on `worker/fable/g2-moving-nests @ 3aa3a563`, clean tree) the full WRF-faithful moving-nest implementation + adaptive-Δt + the `resolve_edge` stale-weights fix, plus **`G2_OPERATIONAL_GATE.json` = PASS** and **17 CPU pytest gates**. The window then died mid-way — a G2-specific crash (the F2/F3 lanes are fine, so it is NOT a quota issue; the isolated WRF `-DMOVE_NESTS` build is the likely culprit). Resume and FINISH — do not redo the done work:

1. **The one remaining oracle** is the external WRF-Fortran cross-check (`proofs/v023/feature_sprints/g2_wrf_fixture_compare.py` vs the `-DMOVE_NESTS` `em_quarter_ss` preset-move run). Check whether the isolated WRF `-DMOVE_NESTS` build completed.
   - If it built: run the compare, record the result in `G2_REPORT.md`.
   - **If the WRF build is broken / OOMs (it may be what crashed the prior session): do NOT rabbit-hole.** Document it honestly as a known limitation of the *fixture* prong, and rely on the three fixture-INDEPENDENT oracles that already PASS (operational gate, mass/energy conservation, analytic advection-through-move). Keep the WRF build off cores 0-3 (`taskset -c 4-31`, `-j<=4`) — those are 0:2's.
2. **Finalize `G2_REPORT.md`** (drop the "draft" label; state the verdict, coverage, and any honest limitation).
3. `touch proofs/v023/feature_sprints/G2_DONE` and report the verdict to `0:1` (`scripts/tmux_submit.sh 0:1 '<msg>'`).

CPU-ONLY (`JAX_PLATFORMS=cpu`). No masking/shortcuts. Once you can honestly close it, close it — don't over-work a 90%-done milestone.
