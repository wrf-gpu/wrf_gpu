# Manager Closeout: V0.14 Fable NoahMP Step-1 Closure

Merge Decision: ACCEPT AND COMMIT.

This sprint does not close v0.14 grid parity. It closes the prior
NoahMP-disabled Step-1 blocker and proves the next blocker with enough
specificity to keep the roadmap moving: NoahMP land-tile energy/HFX at the
strict worst cells. The small production fix is acceptable because it threads an
existing first-step flag into the already active NoahMP blend path and has a
focused regression test.

Accepted code/proof changes:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/noahmp_surface_hook.py`
- `src/gpuwrf/physics/noahmp_coupler.py`
- `tests/test_noahmp_coupler.py`
- Step-1 proof builders and rerun proof artifacts under `proofs/v014/`
- Fable review `.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md`

Roadmap decision:

- TOST, Switzerland, broad FP32, and long GPU validation remain paused.
- Next primary sprint should be the NoahMP land-tile energy closure. This is a
  hard kernel/physics task and should go to Fable/Mythos as a whole endpoint
  after `/compact`.
- Secondary parallel sprint can be GPT CPU-only RRTMG GLW/RTHRATEN parity if it
  does not touch the same source files.
