You are GPT-5.5 xhigh acting as a read-only memory analyst for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-empirical-memory-map/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Produce an implementation-ready empirical/static memory map for remaining
non-radiation memory risks on the exact current branch. This is read-only
analysis. Do not edit production `src/`, do not run TOST, do not run
Switzerland validation, do not use the GPU, and do not start FP32 source work.

Important context:

- RRTMG column/band/optics tiling is already fixed and proved. Treat it as
  prior evidence.
- Current project priority remains grid-cell parity first. Your output should
  help decide which memory fixes, if any, are safe after that gate.
- Keep terminal output compact. Detailed tables belong in JSON.

Deliver:

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

Required validation:

```bash
python -m py_compile proofs/v014/empirical_memory_map.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/empirical_memory_map.py
python -m json.tool proofs/v014/empirical_memory_map.json \
  >/tmp/empirical_memory_map.validated.json
```

When done, print:

`GPT EMPIRICAL_MEMORY_MAP DONE`
