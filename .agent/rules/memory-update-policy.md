# Memory Update Policy

Agents do not self-update stable memory.

At sprint close:

1. Manager writes `memory-patch.md`.
2. Patch states scope, evidence, proposed destination, and reviewer status.
3. `python scripts/validate_memory_patch.py <patch>` passes.
4. Reviewer approves or rejects.
5. Approved patch updates stable memory or skills.
6. Skill changes run skill evals.

Milestone closeout includes memory hygiene.
