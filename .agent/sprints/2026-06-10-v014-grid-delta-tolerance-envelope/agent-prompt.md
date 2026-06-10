You are GPT-5.5 xhigh, analytical validation worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Sprint contract:
`.agent/sprints/2026-06-10-v014-grid-delta-tolerance-envelope/sprint-contract.md`

Objective:
Produce a reviewable candidate tolerance envelope for the v0.14 Grid-Delta
Atlas. This is analytical proof/docs work only. Do not edit model source and do
not start GPU/model runs.

Why:
The final v0.14 validation must include all-cell/all-field Grid-Delta Atlas plus
station TOST. The atlas tool supports `--tolerance-json`, but without a frozen
manifest it reports `REPORT_ONLY_NO_TOLERANCE_MANIFEST`. We need a candidate
manifest before final scoring so no tolerance is tuned after seeing new results.

Rules:
- No `src/gpuwrf/**` edits.
- No GPU.
- No TOST/Switzerland/model forecast launches.
- Do not touch Fable Step-1 files.
- Keep output concise and reviewable.
- Do not invent permissive thresholds to pass old failed runs. Prefer
  documented existing thresholds; label speculative fields report-only.

Deliver:
- `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
- `proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md`
- `.agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md`

Run required gates from the sprint contract.

When finished, send:

```bash
tmux send-keys -t 0:2 'GPT GRID_DELTA_TOLERANCE_ENVELOPE DONE - see proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
