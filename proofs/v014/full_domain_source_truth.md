# V0.14 Full-Domain Source/Truth Surface

Verdict: `FULL_DOMAIN_TRUTH_SURFACE_BLOCKED_PATCH_ONLY_EXISTING_SURFACES`.

No strict same-input JAX comparison is authorized from the existing surfaces.

## Why

- Existing source/save output is patch-only.
- Existing post-RK/pre-halo truth is patch-only.
- The accepted source/save proof has only one conservative halo-valid mass cell.
- Full wrapper carry/boundary leaves were not emitted at the same boundary.

Next: use the staged early-step discriminator instead of another step-6000 wrapper micro-sprint.
