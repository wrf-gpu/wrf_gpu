# Sprint Contract: V0.14 Step-1 Part2 Source-Leaves Split

Date: 2026-06-09 23:40 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Split or fix the remaining Step-1 `T_TENDF` source-leaf divergence inside WRF
`first_rk_step_part2`.

The previous accepted proof:
`proofs/v014/step1_tendency_contract_split.{py,json,md}`.

Current exact boundary:

- patched-init `P/MU/W/PH` material frontiers are closed;
- first material field is full-domain `T_TENDF` at WRF
  `after_first_rk_step_part2`;
- `T_TENDF` max_abs versus the current JAX dry source bundle is
  `2457.5830078125`, RMSE `21.20870100357482`;
- source-save pre-addtend `T_TENDF` is also divergent;
- `rad_rk_tendf=1` is falsified as the dominant explanation;
- boundary/spec/acoustic explanations are too late for the first failure.

## Required Work

Use the fastest rigorous CPU-only path. Prefer disposable WRF instrumentation and
the existing v0.14 proof helpers over broad source edits.

At minimum:

1. Emit or consume WRF truth surfaces inside `first_rk_step_part2` after:
   - `calculate_phy_tend`;
   - `update_phy_ten`;
   - `conv_t_tendf_to_moist`.
2. Include raw theta source contributors needed to explain `T_TENDF`, including
   `RTH*TEN`, `T_HIST_SRC`, and any directly adjacent source/save leaves.
3. Compare the same surfaces against the current JAX dry physics/source bundle
   under the patched-init capture used by
   `step1_tendency_contract_split.py`.
4. If the manager's boundary hypothesis is wrong, do not stop. Rank likely
   alternatives, run cheap falsifiers, and return the next exact boundary.
5. If a narrow, performance-compatible source fix becomes obvious, you may
   implement it only with before/after proof. Otherwise make proof artifacts
   only.

## Output Artifacts

Required:

- `proofs/v014/step1_part2_source_leaves_split.py`
- `proofs/v014/step1_part2_source_leaves_split.json`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `.agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md`

If disposable WRF patching is used, also write:

- `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`

## Hard Constraints

- CPU-only. Do not use the GPU.
- No TOST.
- No Switzerland validation.
- No FP32/memory source work.
- No Hermes/Telegram.
- Do not use Fable/Mythos. This is a normal GPT-localization sprint unless it
  proves to be a hard unresolved core after completion.
- No broad production `src/gpuwrf/**` source edit. A source edit is allowed only
  if exact, narrow, WRF-anchored, and proven by before/after artifacts in this
  sprint.
- Keep top-level output compact. Large raw truth surfaces should live under
  `/mnt/data/wrf_gpu2/...` or in proof JSON; the markdown/review should be short.

## Validation Gates

Required manager-rerunnable commands:

```bash
python -m py_compile proofs/v014/step1_part2_source_leaves_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py
python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/step1_part2_source_leaves_split.validated.json
git diff --check
```

If source is edited, also provide a focused before/after command and expected
verdict in the markdown.

## Completion Marker

Print exactly:

`GPT STEP1_PART2_SOURCE_LEAVES_SPLIT DONE - see proofs/v014/step1_part2_source_leaves_split.md`
