# Release Notes — v0.23.1

A **usability + AI-native onboarding** release on top of v0.23.0. It changes **no
dynamics, no physics, and no default numerical result** — the default forecast is
**numerically identical to v0.23.0** (same compiled program on the compute path). It
makes the port easier to run, easier for an AI agent to operate, and closer to WRF's
"the namelist drives the run" ergonomics.

## Headline — a full HTML User's Guide

A complete, searchable **User's Guide** ships in `docs/` (static HTML, no build step)
covering installation, quickstart, the CLI, input/init, namelist compatibility, the
physics menu, the dynamical core, nesting, output, environment variables, performance,
validation, boundaries, version history, and a glossary — patterned after the official
WRF Users' Guide and honest about every open gate. It is designed to be served with
GitHub Pages and includes a client-side search index and a logo. Human-facing entry
point: `docs/index.html`.

## Headline — AI-native onboarding (clone → ask an agent)

The repository now ships an **AI operator skill** so a user can `git clone`, open an AI
coding agent (Claude Code / Codex / OpenCode), and say *"here is my input, get it
running."* The agent auto-loads the operator runbook and drives the whole job —
environment setup, GPU/driver check, a smoke test, then the user's case — **narrating
each step and its ETA** (venv → CUDA JAX → `nvidia-smi` → cold-compile estimate →
per-forecast-hour estimate → `wrfout`).

- `AI_OPERATOR.md` — the authoritative end-user operator runbook.
- `.claude/skills/run-wrf-gpu/` — a Claude Code skill that dispatches to it.
- A routing banner atop `AGENTS.md` / `CLAUDE.md` sends any agent helping an end user to
  the runbook (the rest of those files is the port's internal development protocol).

The operator skill relays the same honesty as the rest of the project: fp64 default,
opt-in modes stay opt-in, unsupported namelist options fail closed, no masking, and a
successful run does **not** close the open 24–72 h forecast-skill gate.

## WRF-parity CLI ergonomics

All backward-compatible — an explicit flag always wins, and existing invocations behave
identically. Only **omitted** flags gain WRF-faithful defaults, and every resolved value
is printed and recorded in the run payload.

- **`--hours` defaults from the namelist `&time_control`** (`run_days`/`run_hours`/
  `run_minutes`/`run_seconds`) when omitted, instead of silently defaulting to 1 h.
- **`--domain` defaults to `d01`** (WRF's root domain) for single-domain runs when
  omitted, instead of the surprising `d02`.
- **`--domains-from-namelist`** runs all domains the namelist declares (`max_dom`) — the
  explicit WRF-parity nested path (the default stays single-domain / explicit `--max-dom`
  for safety, since auto-running a deep nest can OOM or run long).
- **`gpuwrf namelist-support`** — a new offline subcommand that prints the scheme-support
  registry (operational / reference-only / fail-closed / disabled) without importing JAX
  or touching the GPU.
- **`--dry-run`** — validate the namelist, detect the input mode, resolve
  hours/domain/max_dom/scratch, print the effective plan, and exit without compiling.
- **Clearer `GPUWRF_WRF_ROOT` errors** — what was searched, why it is needed, and the
  exact `export` fix.
- **Effective-values in the run payload** — `namelist_path`, `namelist_max_dom`,
  `effective_max_dom`, `effective_domain`, `effective_hours`, `override_sources`.

See `docs/WRF_PARITY_ROADMAP.md` for the remaining parity candidates.

## Documentation-accuracy fixes

- Initialization is described precisely as the CLI implements it: standalone native-init
  loads the initial state from `wrfinput_<domain>` and lateral boundaries from
  `wrfbdy_d01` (no `real.exe` run and no CPU-WRF `wrfout` dependency); replay uses CPU-WRF
  `wrfout` history.
- The opt-in training subset is documented as **39 variables** (plus the mandatory
  coordinate variables), matching the code.
- Namelist tables mark `*=0` (disabled) slots as accepted, and clarify that
  reference-only schemes are parity-proven but fail closed operationally.

## Validation

- Default numerics: **numerically identical to v0.23.0** — the compute/dispatch path is
  unchanged; v0.23.1 only touches CLI argument resolution, two additive subcommands/flags,
  error text, run-payload metadata, and documentation.
- CLI changes covered by CPU unit tests; the forecast pipeline is unchanged.
- No masking, clamps, or `nan_to_num`. The open 24–72 h forecast-skill gate is **not**
  claimed closed.
