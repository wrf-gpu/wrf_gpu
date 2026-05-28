# F2 Dycore Correctness Summary

Case statuses: {'warm_bubble': 'BLOCKED_GPU_UNAVAILABLE', 'density_current': 'BLOCKED_GPU_UNAVAILABLE'}. Case verdicts: {'warm_bubble': 'BLOCKED', 'density_current': 'BLOCKED'}.

Given the F2 outcomes, the most likely structural bug cannot be newly isolated from these idealized cases because the required JAX GPU backend was unavailable and both dycore integrations were blocked before timestep execution. The standing dycore-review hypothesis therefore remains the best-supported explanation: operational correctness is most at risk from large-step advection/RK coupling defects, acoustic dry-mass carry errors, and theta mass-decoupling mistakes. This summary is intentionally not a physics pass/fail claim.
