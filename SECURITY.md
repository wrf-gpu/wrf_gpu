# Security Policy

## Supported versions

`wrf_gpu` is a single-author, hobby-research project. There is no security-maintenance SLA. The most recent tagged release on the `main` branch is the only supported version. Older versions receive no fixes.

## Reporting a vulnerability

If you discover a security-relevant issue (a way to use this software to harm a user, leak data, or compromise a host beyond what the AGPL-3.0 warranty disclaimer covers), please **do not open a public issue**.

Instead, report via one of the following:

- GitHub private vulnerability reporting (preferred): use the "Report a vulnerability" link at <https://github.com/wrf-gpu/wrf_gpu/security/advisories/new>.
- Direct email: open a public GitHub discussion asking the maintainer for a private contact, or follow the contact information at the maintainer's organisation profile.

Please include:
- A description of the issue and a minimal reproduction
- The version of `wrf_gpu` affected (commit hash or tag)
- Whether the issue is theoretical or has been observed
- What you propose as a fix or mitigation (optional)

We aim to acknowledge security reports within 14 days. Because this is a hobby project, a "fix" may take the form of disabling the vulnerable code path or documenting the issue in the README rather than a coordinated patch release.

## Scope notes

- This project does not handle user authentication, network listeners, or third-party data uploads. The most likely security-relevant issue is an unsafe deserialisation in checkpoint loading or in the NetCDF reader, both of which use trusted-local-file assumptions.
- Forecast outputs from `wrf_gpu` are not validated for downstream decisions. Use of forecasts for safety-critical applications is explicitly disclaimed (see [LICENSE](LICENSE) §15-§16 and the README liability paragraph). This is not a security issue; it is a fitness-for-purpose statement.
