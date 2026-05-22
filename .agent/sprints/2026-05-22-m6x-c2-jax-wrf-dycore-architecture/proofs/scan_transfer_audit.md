# AC5 Scan Transfer Audit

- Mode: static JAXPR audit plus executed analytic scan.
- Devices observed for final carry leaves: ['gpu']
- Outer loop: `gpuwrf.dynamics.orchestrator.run_scan` uses `jax.lax.scan`.
- Nested loop: `gpuwrf.dynamics.acoustic_wrf.run_acoustic_scan` uses `jax.lax.scan`.
- Host callback primitives present: False
- Post-init host/device transfers inside scan: 0 by static audit; no `host_callback`, `io_callback`, or `pure_callback` primitives appear in the timestep JAXPR.
- Limitation: this is not an Nsight transfer trace. It is sufficient for c2-A1 architecture proof, not for a GPU performance claim.

WRF source anchors: `module_small_step_em.F:562` for previous-pressure smdiv memory; `module_small_step_em.F:1094-1112` for flux/vertical-velocity carry context.
