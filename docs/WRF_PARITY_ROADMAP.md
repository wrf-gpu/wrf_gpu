# WRF-usage-parity roadmap

**Goal:** make `wrf_gpu` as easy to use as the original WRF — ideally *identical
usage* where possible. A WRF user expects `namelist.input` to drive the run the way
`wrf.exe` does. This roadmap tracks the gaps between `gpuwrf run` and that mental
model, and what each release does about them.

The baseline: standard WRF runs as `wrf.exe` with `namelist.input` as the primary
runtime contract — forecast length (`run_days`, `run_hours`, …), domain count
(`max_dom`), and per-domain arrays are all read from the namelist.

> This file is about *usage/ergonomics* parity. For *feature/scheme coverage* parity — every
> scheme still reference-only or fail-closed, the 3-D turbulence closures, data assimilation,
> coupled models — see [`PORT_COMPLETION_ROADMAP.md`](PORT_COMPLETION_ROADMAP.md).

## Landed in v0.23.1

These backward-compatible CLI improvements shipped in v0.23.1 (an explicit flag
always wins; only *omitted* flags gained the new WRF-faithful defaults):

| # | Item | Status |
|---|------|--------|
| 1 | **`--hours` defaults from the namelist `&time_control`** (`run_days`/`run_hours`/`run_minutes`/`run_seconds`) when omitted; explicit `--hours` still overrides and is announced. | ✅ v0.23.1 |
| 3 | **`--domain` defaults to `d01`** (WRF's root domain) for single-domain runs when omitted, instead of the surprising `d02`. | ✅ v0.23.1 |
| — | **`--domains-from-namelist`** flag to run all domains the namelist declares (`max_dom`), the explicit WRF-parity nested path. | ✅ v0.23.1 |
| 6 | **`gpuwrf namelist-support`** subcommand — prints the scheme-support registry (operational / reference-only / fail-closed / disabled) offline, without importing JAX or touching the GPU. | ✅ v0.23.1 |
| 10 | **`--dry-run`** preflight — validates the namelist, detects the input mode, resolves hours/domain/max_dom/scratch, prints the effective plan, and exits without compiling or running. | ✅ v0.23.1 |
| 5 | **Clearer `GPUWRF_WRF_ROOT` errors** — what was searched, why it is needed, and the exact `export` fix. | ✅ v0.23.1 |
| 7 | **Effective-values in the run payload** — `namelist_path`, `namelist_max_dom`, `effective_max_dom`, `effective_domain`, `effective_hours`, and `override_sources`. | ✅ v0.23.1 |
| — | **AI-native onboarding** — an `AI_OPERATOR.md` runbook + a Claude Code skill so an agent can clone → set up → run a case for a user, narrating each step. | ✅ v0.23.1 |

## Landed in v0.23.4

The explicit nested CLI path is now correctness-accepted through `max_dom=9`
on the sealed one-hour all-physics fixture. This validates d01-d09 dispatch,
subcycling, boundaries, and output as a concrete capability; it does not make
deep nesting the implicit default or claim every namelist control is honored.
The README initialization wording is also reconciled with the shipped CLI:
standalone execution consumes prepared `wrfinput_*`/`wrfbdy_d01` artifacts; it
does not accept raw `met_em` as the direct run input.

## Planned / candidate (later)

| # | Item | Severity | Type | Difficulty |
|---|------|----------|------|-----------|
| 2 | Optionally **invert the nested default** so `max_dom > 1` in the namelist runs nested by default (currently opt-in via `--domains-from-namelist` for safety — auto-running a deep nest can OOM or take a long time). Needs a deliberate compatibility decision + a release note. | major | code+docs | M |
| 4 | A **one-command WRF-style invocation** (`gpuwrf run` inside a case directory with `namelist.input` present, deriving input/output/duration/domain), matching WRF's "run in the case dir" habit. | major | code+docs | M |
| 8 | **WRF-style restart ergonomics** (`restart`, `restart_interval`) surfaced through the CLI/namelist, not just the internal config. | minor | code+docs | M |
| 9 | **Honor or explicitly report more `&time_control` output-cadence keys** (frames-per-file, etc.) at run start, failing loud on accepted-but-not-honored keys. | minor | code+docs | M |
| — | **Honor nested `radt`** instead of the current fixed 30-minute target, and surface the effective cadence before compile. The v0.23.4 fixture requests `radt=9`, so this is a real disclosed usage/fidelity mismatch. | major | code+docs+physics validation | M |
| — | **Bind nested `topo_shading` / `slope_rad` faithfully** instead of accepting them while the runtime uses disabled values. | major | code+docs+oracle | M |
| — | **Honor `moist_adv_opt` / `scalar_adv_opt` in the single-domain daily pipeline.** It currently drops requested values and runs `0/0`; the accepted nested path already binds them. Surface effective values and add WRF-oracle coverage. | major | code+docs+oracle | M |
| — | **Reconcile omitted `time_step_sound` semantics and residual dry-mass behavior.** The runtime currently selects 10 acoustic substeps where pristine WRF derives 4 on the tracked fixture; a four-substep discriminator improved initial interior U/V but did not close the terminal gate. | major | code+docs+dycore validation | M–L |

## Principles for parity changes

- **Backward-compatible first.** An explicit flag must always win; existing scripts
  must keep working. New defaults only fill in *omitted* choices, and the resolved
  value is always printed.
- **Never trade safety for ergonomics.** Fail-closed namelist validation, the VRAM
  preflight, and the finite guard stay in force; a convenience default must not
  silently launch a run that could OOM or run for hours without the user's intent
  (hence the nested default stays explicit for now).
- **Say what you did.** Every resolved value (hours, domain, max_dom) and its source
  is echoed and recorded in the run payload.
