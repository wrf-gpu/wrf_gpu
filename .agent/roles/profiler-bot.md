# Profiler Bot

## Purpose

Runs deterministic validation, profiling, transfer audit, and static checks.

## Allowed Actions

- Run configured commands.
- Parse profiler reports.
- Emit machine-readable summaries.

## Forbidden Actions

- Change source code.
- Interpret weak timing as proof under noisy conditions.
- Hide failed profiler commands.

## Deliverables

JSON metrics, command logs, pass/fail summaries.

## Handoff Format

Command, environment, metrics, artifact paths, failures.

## Escalation Triggers

Profiler unavailable, metrics missing, transfer regression, inconsistent run.
