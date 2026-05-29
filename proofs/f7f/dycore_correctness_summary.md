# F2 Dycore Correctness Summary

Case statuses: {'warm_bubble': 'RAN_TO_COMPLETION', 'density_current': 'RAN_TO_COMPLETION'}. Case verdicts: {'warm_bubble': 'FAIL', 'density_current': 'FAIL'}.

Given the warm-bubble and density-current outcomes, the most likely structural bug is a missing or ineffective transport/buoyancy pathway in the operational RK/acoustic dycore: the reference cases require coherent vertical motion, lateral cold-pool propagation, and dry-mass consistency, so failures in rise/front/rotor checks point first to advection/RK coupling and then to acoustic mass/theta coupling. This is diagnostic evidence only; it does not repair the protected dycore files.
