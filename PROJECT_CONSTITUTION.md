# Project Constitution

## Immutable End Goal

Build an open, professional-forkable, GPU-native regional NWP system that is WRF-compatible where useful and physics-identical-enough to WRF through explicit validation. The first operational target is Canary Islands 3 km and 1 km daily forecast runs.

## Non-Negotiables

- Physics correctness precedes speed claims.
- The high-frequency model state stays resident on the GPU after initialization.
- This is not a line-by-line WRF Fortran port.
- WRF compatibility means useful interfaces, variables, namelist mapping, fixtures, and validation behavior, not inherited architecture.
- Canary Islands operational value comes before general WRF replacement scope.
- The project must remain clear enough for professionals to fork, audit, and extend.
- Rules, memory, skills, and contracts are production assets. They may change only through patch, evidence, review, and versioned merge.
- Scope expansion and irreversible architecture decisions require human approval.

## Amendment Rule

Changing this constitution requires a written ADR, manager recommendation, independent review, and explicit human approval.
