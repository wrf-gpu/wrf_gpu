#!/bin/bash
# stub inputs step: records start, fails exactly like the R15 inputs refusal (real Python traceback, no gpuwrf/jax import)
touch "${SMK_MARK:?}/inputs_fail.started"
echo "inputs stub: phase inputs0219"
JAX_PLATFORMS=cpu python3 -c 'def require(ok, m):
    if not ok: raise ValueError(m)
require(False, "SCREEN6 explicit J2 result pin required")'
