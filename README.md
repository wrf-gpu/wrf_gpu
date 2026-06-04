# wrf_gpu2

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF source. It is a clean JAX rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Current status — v0.9.0

**v0.9.0 is a standalone, JAX-native, single-GPU WRF v4 ARW forecast system for standard regional configurations.** It performs **native real-init** (assembles `wrfinput`/`wrfbdy` from met_em-stage forcing, no `real.exe` and no CPU-WRF artifact for the initial/boundary state), runs a **nonhydrostatic split-explicit ARW dycore** on the GPU, exposes a **WRF-compatible namelist** with a **GPU-operational physics menu** and a **fail-closed boundary** on everything not yet ported, and is **validated coupled vs CPU-WRF** on the Canary 3 km (d02) and 1 km (d03) cases.

This is a deliberate step beyond v0.1.0, which was a single-domain **replay** path that consumed CPU-WRF/Gen2 artifacts for initialization. v0.3.0 added native metgrid; v0.4.0 added native real-init (proven equivalent to `real.exe` at t=0); v0.6.0 expanded the operational physics menu; v0.9.0 consolidates these into a standalone forecast system.

> **Honesty note.** Two distinct claims are kept separate throughout this README and must not be conflated:
> 1. **Native init** is proven equivalent to `real.exe` at t=0 (savepoint parity) and produces a stable forecast.
> 2. The **coupled skill validation** vs CPU-WRF on d02/d03 is run through the **replay harness** (parent-history replay, which consumes a CPU-WRF `wrfout` for the boundary/skill comparison). The validated coupled-skill run is *not* a from-scratch native-init run.

### What v0.9.0 is — GPU-operational capability

**GPU-operational physics menu (scan-wired into the operational forecast loop, WRF-oracle-gated).** These are the schemes the operational scan actually dispatches; the exact wiring is in [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py) (`_SCAN_WIRED_OPTIONS`) and [`src/gpuwrf/coupling/scan_adapters.py`](src/gpuwrf/coupling/scan_adapters.py); the namelist-accepted matrix is in [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py).

| Family | Namelist key | GPU-operational options (scan-wired) |
|---|---|---|
| Microphysics | `mp_physics` | 1 Kessler, 2 Purdue-Lin, 3 WSM3, 4 WSM5, 6 WSM6, 8 Thompson, 10 Morrison, 16 WDM6 |
| PBL | `bl_pbl_physics` | 1 YSU, 5 MYNN, 7 ACM2, 8 BouLac |
| Surface layer | `sf_sfclay_physics` | 1 revised-MM5, 5 MYNN-SL, 7 Pleim-Xiu |
| Cumulus | `cu_physics` | 1 Kain-Fritsch, 2 BMJ (fp64), 3 Grell-Freitas (scale-aware), 6 Tiedtke |
| Radiation | `ra_sw_physics` / `ra_lw_physics` | RRTMG SW + LW (the operational radiation slot runs RRTMG; `ra_sw=4` / `ra_lw=4`) |
| Land surface | `sf_surface_physics` | 2 Noah classic (explicit static/land bundle), 4 Noah-MP (`use_noahmp=True`) |

`mp_physics=0` (passive vapor), `bl_pbl_physics=0`, `sf_sfclay_physics=0`, `cu_physics=0`, and `ra_*=0` are accepted as "disabled" slots.

**Parity-proven but fail-closed (recognized, loudly rejected if selected operationally).** These schemes pass per-scheme savepoint parity against an unmodified-WRF oracle but are **not** scan-wired into the GPU operational loop. Selecting one does **not** silently fall back or silently skip — it raises a specific, named error before any compute (`UnsupportedSchemeSelection` / `UnsupportedNamelistOption`):

- **MYJ PBL** (`bl_pbl_physics=2`) and its mandatory partner **Janjic-Eta surface layer** (`sf_sfclay_physics=2`) — savepoint-parity-proven CPU reference, GPU scan-wire is a post-0.9.0 item.
- **New-Tiedtke cumulus** (`cu_physics=16`) — interface-compatible/accepted but not separately source-gated by a distinct WRF path.
- **Dudhia shortwave** (`ra_sw_physics=1`) and **classic RRTM longwave** (`ra_lw_physics=1`) — isolated-savepoint parity-proven; the operational radiation slot runs RRTMG only, so these are not yet operationally selectable (post-0.9.0 jit/vmap rewrite + radiation-family dispatch).

**WRF-compatible namelist + fail-closed behavior.** The port reads WRF-exact namelist names and integer codes (`mp_physics`, `cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`, `ra_lw`, `ra_sw`, `diff_opt`, `km_opt`, `dyn_opt`, …) via `gpuwrf run --namelist namelist.input`. Option validation is **fail-closed before any compute** and reports one of three honest outcomes ([`src/gpuwrf/io/namelist_check.py`](src/gpuwrf/io/namelist_check.py)):

- **implemented** — accepted and operationally wired;
- **recognized-WRF-not-yet-implemented** — a real WRF v4 scheme the port names but does not yet wire (fail-closed, names the scheme);
- **invalid** — not a recognized WRF v4 option at all (fail-closed).

**Dynamics.** Nonhydrostatic ARW mass core, RK3 + split-explicit acoustic substepping, flux-form advection (h=5 / v=3), WRF `w_damping` + Rayleigh upper damping (`damp_opt=3`), monotonic 6th-order filter (`diff_6th_opt=2`), constant-K diffusion (`diff_opt=2`/`km_opt=1`), and the WRF real-data-default 2-D Smagorinsky path (`diff_opt=1`/`km_opt=4`). Idealized gates (Skamarock warm bubble, Straka density current) pass 6/6 against published references + pristine WRF v4.7.1 ground truth; the operational dycore is finite/stable over full d02/d03 forecasts. Full dycore record: [`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md).

**GPU-optimized operational mode.** Operational mode is GPU-optimized (gated fp32 downcast, kernel fusion, held-rate radiation cadence) by design. It is **not** bitwise-WRF — RMSE-equivalence is the operational bar; fp64 savepoint parity is the per-scheme validation-mode check.

### Validation (v0.9.0)

The validation lane produces the binding skill + speedup numbers; the placeholders below are **filled by the release worker from the validation burst** before the tag.

- **Native real-init equivalence.** Native `wrfinput`/`wrfbdy` assembly is savepoint-parity-proven equivalent to `real.exe` at t=0 (v0.4.0; one-cell categorical-LSM residual documented), and v0.3.0 native metgrid passed its gate. This removes the CPU-WRF dependency for the initial/boundary state.
- **Per-scheme savepoint parity.** Each GPU-operational scheme passes an fp64 math-faithfulness gate vs an unmodified-WRF oracle (regime-robustness insurance), under `proofs/`.
- **Coupled vs CPU-WRF, d02 (3 km).** Combined-physics GPU forecast vs CPU-WRF `wrfout` (T2 / HFX / U10 / V10 / PBLH / precip), radiation-ON, finite/stable. Skill summary: «FILL FROM VALIDATION BURST».
- **Coupled vs CPU-WRF, d03 (1 km).** Mandatory 1 km gate (every output timestep within scientifically-acceptable margins) with the v0.9.0 faithful physics. Skill summary: «FILL FROM VALIDATION BURST». *Note: the d03 1 km row may be carried over pending the cross-model release critic — the CPU 1 km reference is indicative (it ran inside a contended multi-domain nest, not a clean standalone).*
- **Powered TOST equivalence (n=15).** Statistical equivalence of 24–72 h RMSE on **T2 / U10 / V10** under TOST at the ADR-029 predeclared margins (10% of the local CPU-WRF benchmark RMSE: **T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s**). **n=15 is the binding floor and is honestly underpowered** relative to the ADR-029 target (n≈27 to detect a 10% RMSE difference at α=0.05, β=0.20). The result is labeled single-season (MAM) and underpowered, never an unqualified "equivalence PASS." Margins + power analysis: [`.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`](.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md). TOST result: «FILL FROM VALIDATION BURST».
- **End-to-end wall-clock speedup.** Honest command-to-finish wall-clock (CPU wall-clock ÷ GPU wall-clock), single RTX 5090 vs 28-rank CPU-WRF on the same workstation, compile-inclusive headline, reported for both the 9/3 km nested case and the 1 km case. Kernel-level per-step ratios are reported separately and are never the headline. Speedup: «FILL FROM VALIDATION BURST».

### Honest boundaries — what v0.9.0 does NOT claim

- **Not a universal WRF v4.** Standard regional ARW configs only. Exotic/rare features are README-TODO and fail-closed.
- **Not the full physics catalog.** WRF v4 has roughly 24 microphysics, 12 PBL, many surface-layer/LSM/cumulus/radiation options; v0.9.0 covers the common subset above. Everything else fails closed with a named reason.
- **Not terrain-faithful diffusion.** Both the constant-K and the new Smagorinsky paths are **flat-slab** (map-factor / coordinate-slope deformation terms dropped) — within tolerance for the Canary cases, not fully faithful over steep terrain. Terrain-slope diffusion is a post-0.9.0 refinement.
- **Not full two-way nesting.** v0.5.0 one-way nesting is operator-proven over a short window; full nested equivalence (24 h / two-way d03 feedback / radiation-in-loop) is a post-0.9.0 carry-over.
- **Not DFI / FDDA / spectral-nudging / adaptive-Δt** (fixed Δt only), **not aerosol-coupled microphysics** (Thompson-aerosol `mp=28`/Morrison-aerosol `mp=40`/NSSL fail closed; aerosol-State expansion is ADR-gated, post-0.9.0), and **not urban (BEP/BEM) / lake / WRF-Chem** (these are rejected, not roadmap).
- **Known bounded residual.** A documented near-surface westerly excess persists in the standalone 24 h forecast (T2 correct, stable/finite); after multi-round debugging it is ruled out vs WRF against every faithful ported operator and is characterized as dynamical, not a fidelity bug ([`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md), v0.4.0 carry-over). The daytime-T2 behavior is the WRF land HFX behavior plus the Noah-MP T2MB land-T2 overwrite (now implemented) — so the faithful default T2 may differ from the WRF `wrfout` T2 by that LSM-overwrite term.

A code-grounded, prioritized inventory of the remaining gap to a complete WRF v4 replacement lives in [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) and the v0.9.0+ full-port gap analysis under [`.agent/reviews/`](.agent/reviews/).

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No host/device transfers inside the timestep loop without an ADR. Fused timestep-scale kernels, not micro-kernel launch storms.
2. **Operational skill parity with CPU WRF v4** on Canary L2/L3 cases: 24–72 h RMSE on T2, U10, V10 statistically equivalent under TOST at predeclared operational margins on a seasonal ensemble (n=15 floor today; n≈27–30 is the powered target).
3. **Performance vs 28-rank CPU WRF** on the same workstation, re-certified after every correctness fix (no stale speedup claims). The headline is the honest command-to-finish wall-clock ratio; kernel-level ratios are reported separately, never as the headline.
4. **Validation against WRF, not bitwise reproducibility.** Tiered pyramid: micro fixture / savepoint parity → physical invariants → short-run / timestep convergence → station-RMSE TOST equivalence.
5. **Forkable and auditable.** Every claim has a proof object on disk. Every architecture decision has an ADR with cross-model review.
6. **Manager-led, agent-executed.** The user is consulted only at milestone closure and on genuine blockers. Sprint work runs autonomously, workers auto-notify the manager on exit.

## Where to look first (in this order)

| When you want to… | Read |
|---|---|
| Understand what cannot change | [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md), [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md), [`PROJECT_SPEC.md`](PROJECT_SPEC.md) |
| Understand the active plan | **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — the synthesis layer; updated when scope or strategy genuinely shifts |
| See the GPU-operational vs fail-closed physics matrix | [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py), [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py) (`_SCAN_WIRED_OPTIONS`) |
| Run a forecast | [`src/gpuwrf/cli.py`](src/gpuwrf/cli.py) — `gpuwrf run --namelist namelist.input …` |
| See milestone-by-milestone proof objects | [`.agent/milestones/ROADMAP.md`](.agent/milestones/ROADMAP.md) |
| See what's been decided so far | [`.agent/decisions/`](.agent/decisions/) — ADRs + cross-model reviews + milestone/version closeouts |
| Track sprint activity | [`.agent/sprints/`](.agent/sprints/) — one folder per sprint with contract + reports + closeout |
| Understand agent roles & rules | [`AGENTS.md`](AGENTS.md), [`.agent/roles/`](.agent/roles/), [`.agent/rules/`](.agent/rules/) |

## Live docs (updated as work progresses)

Trust the latest commit, not screenshots:

- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — status banner + manager decisions + escalations
- [`MILESTONES.md`](MILESTONES.md) — milestone gates
- [`.agent/milestones/ROADMAP.md`](.agent/milestones/ROADMAP.md) — proof-object checklists
- [`.agent/decisions/`](.agent/decisions/) — ADRs + version closeouts (`V0.4.0-CLOSE.md`, `V0.6.0-CLOSE.md`, …)
- [`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md) — dycore single-source-of-truth status
- [`RISK_REGISTER.md`](RISK_REGISTER.md) — living risk list
- **GPU port gaps / roadmap to a full WRF v4 replacement (honest TODO)** — [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md): code-grounded, prioritized (P0/P1/P2) inventory of what remains before this is a complete standalone WRF v4 replacement for all standard configs.

## Frozen governance (do not edit during sprints)

- [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md) — immutable end goal + non-negotiables
- [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) — agent operating rules
- [`ARCHITECTURE_PRINCIPLES.md`](ARCHITECTURE_PRINCIPLES.md), [`VALIDATION_STRATEGY.md`](VALIDATION_STRATEGY.md), [`PRECISION_POLICY.md`](PRECISION_POLICY.md), [`PERFORMANCE_TARGETS.md`](PERFORMANCE_TARGETS.md)
- [`INTERFACE_CONTRACTS.md`](INTERFACE_CONTRACTS.md), [`src/gpuwrf/contracts/`](src/gpuwrf/contracts/) — frozen State / grid / physics-registry contracts
- [`.agent/rules/*.md`](.agent/rules/), [`.agent/roles/*.md`](.agent/roles/), [`.agent/skills/*/SKILL.md`](.agent/skills/)

## How sprints run

1. Manager opens a milestone with a reviewed milestone plan (see `.agent/milestones/`).
2. Manager creates a sprint folder and writes a narrow contract with frozen interfaces + file ownership.
3. Manager dispatches in-process Opus subagents (and cross-model GPT/agy critics) for isolated work, reviews the diff, runs the acceptance gates, and merges.
4. Sprint closes when its `check_*_done` / acceptance gate returns `ok: true` against a proof object.
5. Manager integrates the sprint branch into the trunk via `git merge --no-ff`.

## Run

```bash
# Validate a WRF namelist fail-closed (no GPU / no compile needed):
gpuwrf run --namelist <input-dir>/namelist.input --input-dir <case-dir> \
    --output-dir runs/my_forecast --domain d02 --hours 1

# Repo / AgentOS checks:
python scripts/validate_agentos.py     # required files + skill metadata
pytest -q                              # full test suite
python scripts/repo_status_snapshot.py
```

## Layout

```
.
├── PROJECT_PLAN.md                  active plan (synthesis layer)
├── PROJECT_CONSTITUTION.md          immutable end goal
├── AGENTS.md / CLAUDE.md            agent operating rules
├── ARCHITECTURE_PRINCIPLES.md       backend / runtime principles
├── VALIDATION_STRATEGY.md           four-tier validation pyramid
├── PRECISION_POLICY.md              FP64/FP32/BF16 rules
├── PERFORMANCE_TARGETS.md           profiler JSON schema + transfer rules
├── INTERFACE_CONTRACTS.md           GridSpec, State, Tendencies
├── RISK_REGISTER.md                 living risk list
├── MILESTONES.md                    milestone gates
├── .agent/
│   ├── goals/                       per-milestone goal spec + manager runbook
│   ├── milestones/                  per-milestone files + ROADMAP.md
│   ├── decisions/                   ADRs, reviews, milestone/version closeouts
│   ├── sprints/                     one folder per sprint
│   ├── roles/                       role definitions
│   ├── rules/                       merge gates, branch policy, etc.
│   └── skills/                      project-local skills (authoritative)
├── docs/                            user-facing references
├── fixtures/                        manifest schemas + analytic samples + Canary slice
├── src/gpuwrf/                      implementation code
│   ├── contracts/                   frozen State / grid / physics_registry
│   ├── coupling/                    scan adapters + physics dispatch
│   ├── runtime/                     operational forecast loop
│   ├── physics/                     scheme kernels
│   ├── io/                          namelist check + wrfout/wrfinput I/O
│   └── integration/                 daily pipeline / native init
├── scripts/                         CLIs: check_*_done, validators
├── tests/                           pytest suite
├── proofs/                          per-milestone proof objects (JSON + reports)
└── publish/                         user-facing analysis + gaps TODO
```

## Do not do

- Do not implement model code without a sprint contract.
- Do not claim physics correctness without WRF-fixture / savepoint / analytic-oracle evidence.
- Do not claim GPU performance without profiler artifacts.
- Do not commit binary fixture data; large payloads live under `data/` (symlinked external storage).
- Do not edit governance files inside a sprint; raise an ADR or escalate at milestone close.
- Do not let a recognized-but-unported scheme silently fall back — it must fail closed and loud.
