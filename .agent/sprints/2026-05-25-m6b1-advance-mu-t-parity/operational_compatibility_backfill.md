# M6B1 — Operational-Compatibility Backfill (Critic Amendment #1)

This file backfills the operational-compatibility classification that M6B1's
original worker-report omitted. The audit memo
`.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/audit_memo.md` Part 5
flagged this as a VIOLATION of Critic Amendment #1 (which landed at commit
698fbde, before the M6B1 worker-report was written). M6B0-R is excused
(predates amendment); M6B2 has a compliant table; this file closes the M6B1
gap so the ladder-hygiene sprint can record full §14.5.1 coverage.

Classification rules (per `PROJECT_PLAN.md §14.5.1`):
- **validation-only**: lives only in the comparator / extractor / oracle path,
  never touched by production runtime code.
- **operational-approved-with-evidence**: enters the production timestep loop
  and has a profiler / fixture proof showing no regression.
- **undecided**: enters the production timestep loop or carry-state but the
  deferral is documented (gated on M6-perf-design or a later sprint).
- **NONE**: no production state changed.

Filed by sprint `2026-05-25-m6b-ladder-hygiene-cleanup`.

## Classification table

| Item | Classification | Evidence | Rationale |
|------|---------------|----------|-----------|
| `sp_advance_mu_t_pre` hook (Fortran) | validation-only | `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90:20-24` | Hook body is empty; emission is Python-orchestrated via `scripts/m6b0r_wrf_savepoint_extract.py`. Wrapper is gated on `#ifdef WRF_SAVEPOINT`, off in operational build (`/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`, SHA `1ec3815…` unchanged). |
| `sp_advance_mu_t_post` hook (Fortran) | validation-only | Same wrapper, lines 25-29 | Same as above; typed-args ABI exists but body is no-op. |
| `src/gpuwrf/dynamics/mu_t_advance.py::advance_mu_t_wrf` (callable) | validation-only | `grep -rn 'mu_t_advance' src/` returns only `src/gpuwrf/dynamics/mu_t_advance.py` itself; no production runtime importer | Helper is invoked exclusively by `scripts/m6b1_advance_mu_t_compare.py` and the comparator's regression test. Not wired into any `solve_em.F` replacement lane. |
| `src/gpuwrf/dynamics/mu_t_advance.py::AdvanceMuTInputs` dataclass | validation-only | Same grep evidence | Pure-Python dataclass; not consumed by JAX runtime. |
| New ladder entries `mu` (`tolerance_ladder.json` `abs=1e-8`) | validation-only | Loaded by comparator scripts only; tolerance reader test enforces presence but doesn't mutate runtime state | Pa-scale state; tolerance is the comparator pass-band. |
| New ladder entry `mudf` (`abs=1e-10`) | validation-only | Same | Pa-scale-ish mass-divergence damping carry; comparator pass-band only. |
| New ladder entry `muts` (`abs=1e-8`, `ulp=16` accumulation exception) | validation-only | Same; the `ulp=16` exception is documented as needed for accumulation | Pa-scale running mu-on-substep accumulator. |
| New ladder entry `muave` (`abs=1e-8`) | validation-only | Same | Pa-scale mu running average; carry-state field. |
| New ladder entry `ww` (`abs=1e-9`) | validation-only | Same | Pa s-1 vertical mass flux; carry-state field. |
| New ladder entry `theta` (`abs=1e-10`) | validation-only | Same | K-scale potential temperature carry. |
| New ladder entry `ph_tend` (`abs=1e-10`) | validation-only | Same | m2 s-2 geopotential tendency carry. |
| `advance_mu_t_pre/post` boundary tags (`VALID_BOUNDARIES`) | validation-only | `src/gpuwrf/validation/savepoint_schema.py:20-21` — only the savepoint-metadata validator consumes them | Schema enum for comparator dispatch; no operational consumer. |
| `advance_mu_t` operator tag (`VALID_OPERATORS`) | validation-only | `src/gpuwrf/validation/savepoint_schema.py:55` | Same reasoning. |
| Operational-mode carry semantics of `mu/muts/muave/ww/theta/ph_tend` | **undecided** | `PROJECT_PLAN.md §14.5.1` lists these in the operational-mode invariant table without a deferral; per ladder-audit Part 5 the operational-mode "carry inclusion" decision is deferred to M6-perf-design | The fields ARE consumed by the operational runtime today; this row classifies the M6-perf-design question "do we drop any of these from the operational-mode carry?" — not the M6B1 helper, which is validation-only by construction. |
| `kill_gate` status string (`PROCEED_TO_M6B2`) | validation-only | Returned only by `scripts/m6b1_advance_mu_t_compare.py` and surfaced in worker-report | Comparator-output string; no runtime consumer. |
| Default state-API impact | NONE | `git diff` shows no changes under `src/gpuwrf/state/` or `src/gpuwrf/dynamics/_pkg_init/__init__.py` | M6B1 introduced no fields into the operational state pytree. |
| Operational wrf.exe SHA | NONE | `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/proof_operational_sha256_post.txt` | Unchanged at `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`. |

## Aggregate "Undecided" items (M6-perf-design input)

- **Operational-mode carry of `mu/muts/muave/ww/theta/ph_tend`** — the M6B1
  helper is validation-only, but the underlying state fields are in the
  operational-mode invariant table and may be candidates for the operational
  GPU lane's "optional carry" pruning per `PROJECT_PLAN.md §14.5.1`. Decision
  deferred to M6-perf-design.

## Provenance

- Backfill author: opus tester (`tester/opus/m6b-ladder-hygiene-cleanup`).
- Audit source: `.agent/sprints/2026-05-25-m6b-ladder-cumulative-audit/audit_memo.md` Part 5.
- Original M6B1 worker-report (no §14.5.1 table): `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/worker-report.md`.
- Amendment text: critic ratification commit b7f1bde (§14.5.1 amendments
  ratified-with-amendments).
