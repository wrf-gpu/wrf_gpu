# Manager Response To Codex Cross-Model Review

Reviewer artifact: `.agent/decisions/REVIEW-codex-bootstrap-plan.md` (Codex, `gpt-5.5`, reasoning `xhigh`, session `019e3d47-70d6-7a22-b3d4-22bd52a883c1`).
Manager: Claude Opus 4.7 (1M context).
Status: **all findings accepted; patches applied to plan, roadmap, and sprint contract; no manager dissent recorded.**

## Decision

`Accept with required fixes` (Codex) → manager applies all 21 findings and all 5 dissents. Three unresolved items are escalated to the human arbiter as new explicit decisions in `PROJECT_PLAN.md §11`.

## Disposition table

| # | Severity | Codex finding (one-line) | Disposition | Patch location |
|---|---|---|---|---|
| 1 | blocker | M1 placeholder-Canary bypass contradicts proof-object rule | Accept | `PROJECT_PLAN.md` §7 (removed bypass), §10 (replaced row with human-arbiter scout authorization), `ROADMAP.md` M1 unchanged (was already correct) |
| 2 | major | M2 silently excludes CUDA Tile / explicit CUDA C++ | Accept | `PROJECT_PLAN.md` §5 (added candidate F), `ROADMAP.md` M2 (candidate list A–F), `§11` (new decision row) |
| 3 | major | CUDA Fortran exclusion turned into pre-judgement | Accept | `PROJECT_PLAN.md` §5 (reframed: excluded unless scout produces NVIDIA-only Canary v0 justification), `§11` (decision row updated) |
| 4 | major | PyCECT/probtest/Serialbox locked as tools, not families | Accept | `PROJECT_PLAN.md` §6 (table reworked: "tooling family / acceptance behavior" + non-binding candidates column + ADR trigger) |
| 5 | major | Full 100-member PyCECT mandated without cost gate | Accept | `PROJECT_PLAN.md` §6 / §10 (cost-model risk row added), `ROADMAP.md` M6 (small-ensemble prototype), `ROADMAP.md` M7 (full ensemble gated on M6 cost-model approval) |
| 6 | major | "Overrides per-milestone files where stricter" duplicates authority | Accept | `PROJECT_PLAN.md` §7 retitled "Proposed stricter gates pending milestone-file patch"; per-milestone files patched only after human approval |
| 7 | major | IC/BC not named proof object for regional path | Accept | `ROADMAP.md` M3 (BC metadata in `GridSpec`), `ROADMAP.md` M7 (IC/BC mapping proof object), `PROJECT_PLAN.md` §11 (new decision row #6) |
| 8 | major | Surface/land coupling missing | Accept | `ROADMAP.md` M5 (surface/land proof object when first suite requires it), `ROADMAP.md` M6 (surface coupling validated on coupled short run) |
| 9 | major | Map projection / terrain ingestion late | Accept | `ROADMAP.md` M3 (named, machine-readable fields for projection, terrain provenance with sanity check, vertical coords, halo width, BC metadata) |
| 10 | major | I/O / restart not falsifiable in M7 | Accept | `ROADMAP.md` M7 (`wrf{input,bdy,out,rst}` compatibility matrix + restart-continuity test as named proof objects) |
| 11 | major | S1 tolerance fields would be top-level | Accept | Sprint contract Acceptance Criteria rewritten: tolerances are per-variable only, top-level tolerances explicitly rejected by validator, `tier_overrides` mapping added, `tolerance_rationale` required |
| 12 | major | S1 file-ownership inconsistency (template + .gitignore) | Accept | Sprint contract File Ownership: added `fixtures/manifests/fixture-manifest-template.yaml` and `.gitignore` |
| 13 | major | S1 CLI under-specified for M2 reuse | Accept | Sprint contract: exact CLI argument names (5 flags) frozen; exact JSON output record schema frozen with field-by-field example |
| 14 | minor | `repo_status_snapshot.py runs clean` is not falsifiable | Accept | Sprint contract Acceptance Criteria #17 now asserts `dirty_files` is a subset of contract-owned paths and demands the parsed JSON in the worker report |
| 15 | minor | Branch name `worker/codex/...` contradicts policy | Accept | Sprint contract Handoff Requirements: branch is `worker/gpt/m1-s1-fixture-storage-policy` |
| 16 | major | KE spectrum slope listed as mandatory Tier-2 | Accept | `PROJECT_PLAN.md` §6 table: KE spectrum moved to "optional / scenario-specific"; mandatory Tier-2 is mass/positivity/bounds/NaN/Inf |
| 17 | major | Identity debug mode assumes NVHPC flags | Accept | `PROJECT_PLAN.md` §6: fault-isolation mechanism documented per backend in ADR-001/002, not assumed universal |
| 18 | major | M2 needs candidate-failure schema | Accept | `PROJECT_PLAN.md` §5 (per-candidate proof object adds the schema), `ROADMAP.md` M2 (full JSON schema written out) |
| 19 | major | Missing operational risks in risk register | Accept | `PROJECT_PLAN.md` §10 added five `[patch]`-tagged rows (IC/BC availability, terrain correctness, observation source, full-ensemble cost, RTX 5090 toolchain maturity, S1→S2 ordering) |
| 20 | note | S2 cannot run in parallel before S1 schema freeze | Accept | `PROJECT_PLAN.md` §10 (S1→S2 ordering row); `ROADMAP.md` M2 (S2 implementation gated on S1 review pass; read-only scout allowed) |
| 21 | note | Microphysics-first is assumption, not decision | Accept | `PROJECT_PLAN.md` §5 (M2 column analog explicitly not a commitment); `ROADMAP.md` M5 (new M5-S0 decision-gate sprint to select first physics suite) |

## Required interface freezes (from Codex)

All five freezes accepted:
1. **`FixtureManifest` source of truth** — frozen by S1 sprint contract Acceptance Criteria §Schema (criteria 1–5).
2. **`compare_fixture` CLI surface and JSON output** — frozen by S1 sprint contract Acceptance Criteria §Comparison CLI (criteria 9–12).
3. **M2 bakeoff candidate set + failure-artifact schema + agent-success metrics** — frozen by `PROJECT_PLAN.md` §5 and `ROADMAP.md` M2.
4. **M3 `GridSpec` fields + `State` residency semantics** — frozen at M3 entry per `ROADMAP.md` M3.
5. **M5/M6 `PhysicsColumnInput`/`PhysicsColumnOutput` for surface/land/SST + first-physics-suite selection** — frozen at M5-S0 decision-gate sprint per `ROADMAP.md` M5.

## Dissents (Codex → manager response)

- **D1 (M2 omits CUDA Tile):** accepted; candidate F added.
- **D2 (CUDA Fortran exclusion):** accepted; reframed in §5 and §11 as "excluded unless scout produces NVIDIA-only Canary v0 justification."
- **D3 (PyCECT + probtest hard-bound as tooling):** accepted; §6 table reworked to families, not tools.
- **D4 (M2 with placeholder Canary):** accepted; bypass removed; parallel-work alternative requires explicit human-arbiter approval and is read-only.
- **D5 (METplus deferred until M7):** accepted; verification-tooling research-scout sprint added to the end of M6 in `ROADMAP.md`.

**Manager records no counter-dissent.** Every Codex finding cross-checked against the cited governance files and the two research briefs; in every case the citation supports Codex's reading.

## Items escalated to human arbiter (new in `PROJECT_PLAN.md §11`)

Three were already present (WRF baseline source, fixture storage location, hard cap on analytic fixture sprints). Four added by this review:

- **§11.3** CUDA Fortran in M2 (default: excluded unless scout output).
- **§11.4** CUDA Tile / explicit CUDA C++ in M2 (recommended: include as candidate F; user can shrink the bakeoff to five).
- **§11.6** IC/BC source dataset (recommended: GFS or ERA5; impacts M3 GridSpec freeze).
- **§11.7** Approve M5-S0 decision-gate sprint as a sprint distinct from M5 implementation.

## What still needs to happen before S1 dispatch

1. ~~Human-arbiter approval of `PROJECT_PLAN.md` (especially §11 items 1–7).~~ **Done 2026-05-19**: user authorized M1 dispatch and delegated all operational and architecture-bakeoff decisions to the manager; §11 items 1–7 are recorded as manager decisions.
2. ✅ Per-milestone files (`M3-*.md`, `M5-*.md`, `M6-*.md`, `M7-*.md`) patched to reflect the §7 stricter gates.
3. ✅ `RISK_REGISTER.md` patched with the five `[patch]`-tagged rows.
4. Sprint S1 dispatches to Codex on branch `worker/gpt/m1-s1-fixture-storage-policy`, opened in a tmux window inside the user's session and closed on delivery — handled by `scripts/dispatch_sprint_worker.sh`.

**Dispatch authorized.**
