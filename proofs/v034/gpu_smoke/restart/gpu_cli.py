"""Product CLI on GPU (release defaults) with platform assertion; prints its PID for external SIGKILL."""
import os
import sys

import jax

assert jax.devices()[0].platform == "gpu", jax.devices()
print(f"GPU_CLI_PID {os.getpid()} device {jax.devices()[0]}", file=sys.stderr, flush=True)
from gpuwrf import cli  # noqa: E402

sys.exit(cli.main(sys.argv[1:]))
