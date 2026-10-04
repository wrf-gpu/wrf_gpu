# F2 (ESCALATED from GPT) — Cumulus + advanced MP + LSM (Fable-5, X-High, CPU-ONLY)

## Why this is yours
GPT scoped F2 honestly and hit a real wall (it correctly REFUSED to fake a pass). The bundle needs real implementations + missing oracles — genuinely hard, coupled, multi-scheme. GPT's precise breakdown (read it, don't re-discover): `<USER_HOME>/src/wrf_gpu2_wt/v023-f2-gpt/proofs/v023/feature_sprints/F2_QUESTION.md` + `f2_missing_scheme_bundle_oracle_check.json`.

## Per-scheme state GPT established (your starting point)
- **New-Tiedtke cumulus (`cu_physics=16`):** 5 finite WRF savepoints exist at `proofs/v013/savepoints/cumulus/ntiedtke_case_*.json` → needs a source-specific JAX kernel + scan adapter validated against them. Distinct from old Tiedtke.
- **NSSL 2-moment MP (`mp_physics=18`):** WRF source + Registry present, but NO local single-column oracle → build the oracle, then the JAX port.
- **Morrison-aerosol MP (`mp_physics=40`):** WRF source + Registry present, no local oracle. Base Morrison infra exists (`src/gpuwrf/physics/_morrison_cold.py`, `microphysics_morrison.py`, and an oracle driver `proofs/v060/oracle/morrison_oracle_driver.f90` to model) → build the aerosol oracle + port on top.
- **RUC LSM (`sf_surface_physics=3`):** a preserved patch exists at `proofs/v018/ruc_lsm_port.patch`; v018 warm-land metrics were green → integrate + validate; trunk still fail-closed.

## The core
Build the WRF-Fortran single-column oracles where absent (mp18, mp40 — model the v060 Morrison oracle driver), then faithful JAX ports for all four, each validated by column bit-identity/tolerance vs its oracle. Integrate the RUC patch. WRF-faithful, coupled correctly.

## CPU-ONLY + rules
GPU is another lane's — do NOT use it (`JAX_PLATFORMS=cpu`). No masking/clamp/nan_to_num, no synthetic happy-paths, no fp32-self-compares. Each scheme opt-in + fail-closed; the default physics path stays byte-identical. Once you can prove a scheme, ACT — only report proven work; show the oracle result that proves it.

## Deliverable
Commit on branch `worker/fable/f2-physics`. Write `proofs/v023/feature_sprints/F2_REPORT.md` (per-scheme: files + oracle bit-identity/tolerance + coverage) and `touch proofs/v023/feature_sprints/F2_DONE` when the family is done+validated. Report to `0:1` (`scripts/tmux_submit.sh 0:1 '<msg>'`) at each scheme done. You MAY spawn GPT sub-agents for the mechanical oracle-plumbing. Hard blocker on a specific scheme → `F2_QUESTION.md` + ping `0:1`.
