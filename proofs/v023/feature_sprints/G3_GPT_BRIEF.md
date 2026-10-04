# G3 — Urban (BEP/BEM) + Lake (GPT, CPU-ONLY)

## Scope (v0.23 roadmap Table-E F-d)
Port faithfully, against the WRF-Fortran reference: the **urban canopy schemes** —
BEP (Building Effect Parameterization) + BEM (Building Energy Model),
`sf_urban_physics=2` and `=3` — and the **lake model** (`sf_lake_physics=1`).
gpuwrf has v0.22.0 fail-closed SCAFFOLDS for these → assess + replace with the REAL
implementation where tractable. WRF-faithful; no scheme silently substituted.

## CPU-ONLY + validation (no GPU)
The GPU belongs to another lane — do NOT use it. Run everything `taskset -c 4-31 env
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=` (cores 0-3 are 0:2's;
keep `-j<=4` on any build). Validate each scheme by **single-column / single-point
BIT-IDENTITY (or documented tolerance) vs the WRF-Fortran reference** — build the CPU
oracle if none exists (model it on F3's `camuw_oracle/` gfortran-vs-pristine-WRF harness).
Fail-closed: unsupported namelist options error before compute with a named reason.

## Rules
- No masking / nan_to_num / clip-to-hide-instability / fp32-self-compares. Real
  WRF-Fortran reference, honest bands.
- Each scheme ships opt-in + fail-closed; the default land/surface path stays byte-identical.
- Commit on branch `worker/gpt/g3-urban-lake`, clear messages.

## The honest-outcome rule (learned from F3, IMPORTANT)
If the existing scaffold is BROKEN, or the faithful port is a large dedicated effort:
**do NOT fake a passing port.** Instead — build the CPU oracle, PROVE the current
state honestly (verdict + evidence), flip the scheme to **REFERENCE_ONLY / fail-closed**
(namelist-accepted for oracle comparison, operational scan fail-closes with a named
reason, endpoint stub raises, default path byte-identical), and recommend the faithful
port as its own milestone. That is a fully valid, honest G3 outcome (exactly what F3 did
for CAM-UW). The oracle + honest seam is real value; a fabricated port is not.

## Deliverable
`proofs/v023/feature_sprints/G3_REPORT.md` (per-scheme: files, oracle result,
IMPLEMENTED vs REFERENCE_ONLY, coverage + any honest limitation) + `touch
proofs/v023/feature_sprints/G3_DONE` when done. Report to `0:1`
(`scripts/tmux_submit.sh 0:1 '<msg>'`) at each scheme done. If blocked → write
`proofs/v023/feature_sprints/G3_QUESTION.md` + ping `0:1`. Begin now.
