# Tester Report: V0.14 GPT Moist-Theta Physics Consumer Audit

Decision: PASS FOR READ-ONLY COMPATIBILITY AUDIT.

Contract gates reported by the worker:

```bash
python -m json.tool proofs/v014/moist_theta_physics_consumer_audit.json >/tmp/moist_theta_physics_consumer_audit.validated.json
git diff --check
```

Manager-side acceptance gates must validate the committed JSON, closeout
metadata, and whitespace before commit.

Evidence strength:

- Strong enough to update the v0.14 roadmap and require a broader
  moist-theta/dry-theta adapter boundary plan before long validation.
- Not a source fix.
- Not a proof that every listed adapter is currently failing in a free forecast;
  it is a source-contract audit identifying required gates.

Residual risks:

- The audit observed an active uncommitted Fable patch in
  `src/gpuwrf/physics/noahmp_coupler.py`; that patch is not part of this sprint.
- The audit broadens the correctness surface. The manager must avoid launching
  long TOST/Switzerland-GPU until the relevant θ_m boundary fixes are closed or
  explicitly scoped/demoted.

