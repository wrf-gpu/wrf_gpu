# Memory Patch: V0.14 Step-1 Live-Nest Theta/QV Wiring

Date: 2026-06-09

Reviewer Status: Pending. Opening sprint only.

Reason:

- Transient adjust-base helper proof closed the theta candidate, but production
  live-nest init is not yet wired to use it.
- This sprint may change `src/gpuwrf/integration/d02_replay.py`; record exact
  initialization and Step-1 comparison result after close.

Expected memory after close:

- Whether production live-nest theta/QV init closes.
- Whether full Step-1 16-field comparison closes or names the next field.
- Exact next manager decision for grid-parity chain.
