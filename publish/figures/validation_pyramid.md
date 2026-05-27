# Figure Spec: Validation Pyramid

Purpose: show the evidence hierarchy used for the WRF-compatible GPU rewrite.

Canvas: vertical pyramid with four tiers. Y axis is operational trust. X axis is corpus/runtime cost.

ASCII layout:

```text
                  Tier 4: ensemble / forecast-skill consistency
                 GPU vs CPU/Gen2 distributions and station skill
              ------------------------------------------------------
               Tier 3: short-run / timestep convergence envelope
            ----------------------------------------------------------
             Tier 2: physical invariants and bounds
          finite fields, mass/water checks, positivity, no hidden D2H
       ---------------------------------------------------------------
        Tier 1: fixture and savepoint parity
     analytic fixtures, WRF savepoints, negative perturbation tests
```

Axis labels:
- Vertical: operational trust increases upward.
- Horizontal: runtime, corpus size, and data-management cost increase to the right.

Key annotations:
- Tier 1 can prove local operator agreement but not operational forecast skill.
- Tier 2 prevents unphysical or architecturally invalid runs from advancing.
- Tier 3 catches timestep and short-run drift.
- Tier 4 is required before operational replacement claims; current M7 skill evidence remains red.

Rendering notes: make Tier 4 visually widest and mark the current M7 status as `blocked on skill/corpus` rather than closed.
