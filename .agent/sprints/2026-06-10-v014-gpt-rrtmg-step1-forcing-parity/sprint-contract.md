# Sprint Contract: V0.14 GPT RRTMG Step-1 Forcing Parity

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `43accdc6`

## Objective

Independently localize the secondary Step-1 RRTMG forcing residual without
touching production source while Fable works on the primary NoahMP land-tile
energy blocker.

Known residual from `proofs/v014/noahmp_step1_closure.*`:

- LWDN/GLW uniform clear-sky bias: about `+17.44 W/m2`.
- SWDOWN midpoint convention is mostly pinned: `+radt/2` seed RMSE `2.76 W/m2`
  vs lead-0 `56.43 W/m2`.
- Mass-coupled `RTHRATEN` vs WRF part2: max_abs `19.425`, RMSE `2.488`.

This is not the leading cause of the current land HFX residual, but it must be
closed or bounded for the strict Step-1 release gate.

## Required Work

1. Read the accepted NoahMP closure proof and RRTMG coupling code.
2. Produce a CPU-only WRF-anchored localization of the GLW/RTHRATEN residual:
   inputs, clock/solar geometry, surface properties, cloud/column optical
   inputs, layer ordering, flux-to-theta conversion, and mass-coupling.
3. Write a compact proof/report that says whether the likely fix is:
   - clock/geometry;
   - surface emissivity/albedo/land-state;
   - column thermodynamics/cloud/optics;
   - RRTMG LW kernel parity;
   - flux/tendency conversion or mass-coupling; or
   - another exact named boundary.
4. If a production fix is obvious, describe it precisely but do **not** apply it
   in this sprint. This keeps file ownership disjoint from the active Fable
   source sprint.

## Allowed Files

May write:

- `proofs/v014/rrtmg_step1_forcing_parity.py`
- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity.md`

Do not edit `src/gpuwrf/**`, tests, TOST, Switzerland, Grid-Delta Atlas, FP32,
memory, or NoahMP proof/source files.

## Gates

Required:

```bash
python -m py_compile proofs/v014/rrtmg_step1_forcing_parity.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_step1_forcing_parity.py
python -m json.tool proofs/v014/rrtmg_step1_forcing_parity.json >/tmp/rrtmg_step1_forcing_parity.validated.json
git diff --check
```

## Handoff Requirements

Write a concise review with objective, files changed, commands run, proof
objects, exact residual boundary, recommended fix if any, unresolved risks, and
whether this lane should block the next strict Step-1 attempt.

Completion marker:

```bash
tmux send-keys -t 0:2 'GPT RRTMG_STEP1_FORCING_PARITY DONE - see proofs/v014/rrtmg_step1_forcing_parity.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
