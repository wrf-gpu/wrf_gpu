# v0.14 GPT Moist-Theta Physics Consumer Audit

## Objective

Audit production physics/coupling consumers of `state.theta` and theta-to-temperature helpers for compatibility with runtime moist potential temperature (`theta_m`) and identify which paths need dry decoupling before WRF physics.

## Result

The compatibility issue is wider than the NoahMP `sfctmp` fallback. Boundary, feedback, and dycore transport should keep `state.theta` as moist theta, but most physics adapters that form `T`, dry theta, virtual theta, density, or dry theta tendencies need explicit dry conversion at adapter input and moist recoupling at writeback.

Already-good production pattern: grid-backed MYNN and the generic grid-backed surface column view decouple to dry theta before physics and recouple output dry theta to moist theta.

Highest-priority remaining consumers:

- NoahMP hook surface-layer view and `thx/fltv` path, even with Fable's active `sfctmp` fallback patch.
- Noah Classic forcing.
- Thompson and scan microphysics adapters.
- Radiation input builders and default held-radiation tendency application.
- Surface-layer scan adapters.
- PBL scan adapters and MYJ/Janjic.
- Cumulus scan adapters.
- GWDO temperature profile.
- Legacy WRF output diagnostics.

## Files Changed

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`
- `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`

No source files were edited by this audit.

## Commands Run

- Read project instructions and sprint contract:
  - `sed -n '1,220p' PROJECT_CONSTITUTION.md`
  - `sed -n '1,260p' AGENTS.md`
  - `sed -n '1,260p' .agent/decisions/V0140-RELEASE-CHECKLIST.md`
  - `sed -n '1,260p' .agent/sprints/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit/sprint-contract.md`
- Read local skills:
  - `.agent/skills/validating-physics/SKILL.md`
  - `.agent/skills/reporting-to-human/SKILL.md`
- Searched and inspected relevant `src/gpuwrf/**` files with `rg`, `git diff`, and `nl -ba`.
- Required gates:
  - `python -m json.tool proofs/v014/moist_theta_physics_consumer_audit.json >/tmp/moist_theta_physics_consumer_audit.validated.json`
  - `git diff --check`
- Completion marker:
  - `tmux send-keys -t 0:2 'GPT MOIST_THETA_CONSUMER_AUDIT DONE - see proofs/v014/moist_theta_physics_consumer_audit.md' Enter`

## Proof Objects Produced

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`

## Gate Results

- JSON validation: PASS.
- `git diff --check`: PASS.
- Completion marker: attempted, but `tmux` failed with `Operation not permitted`; proof files are authoritative.

## Unresolved Risks

- The worktree contained an uncommitted external patch in `src/gpuwrf/physics/noahmp_coupler.py` during this audit. It appears to decouple the NoahMP `sfctmp` fallback, but the raw NoahMP surface hook path still needs dry conversion.
- Radiation source-leaf mode has a dry-to-moist conversion, but the default held-radiation path still appears able to add dry `RTHRATEN` directly to moist `state.theta`.
- Dycore EOS qv-factor consistency is outside this physics-consumer fix and needs its own savepoint/oracle test before changing dynamics.
- No adapter source fixes were made in this read-only audit.

## Next Decision Needed

Adopt one shared dry/moist theta helper API and require all physics adapters to use a dry view on input and moist recoupling on output, while leaving prognostic boundary/dycore storage as moist theta.
