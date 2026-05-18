# M6 - Coupled Short Forecast

Goal: couple dycore and physics for short forecast windows.

Deliverables:

- short-run driver
- coupling diagnostics
- drift envelope
- initial ensemble consistency path

Acceptance gates:

- tier 3 passes
- tier 4 **small-ensemble prototype** (≈10 members) using probtest-style and/or PyCECT-style methods; establishes per-member runtime + storage cost
- verification-tooling research-scout sprint completes before M7 dispatch, with METplus-or-alternative ADR draft on disk
- transfer audit clean on coupled run
- surface/land coupling validated end-to-end if M5 selected a surface-coupled suite

Full ensemble (M7) is gated on the manager-approved cost model derived from this prototype. See `.agent/milestones/ROADMAP.md` M6 for full proof-object list.
