"""Direct GPU regression runner; never collects tests/v025/conftest.py."""
import importlib.util
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import pytest


def main():
    assert jax.devices()[0].platform == "gpu", jax.devices()
    jax.config.update("jax_enable_x64", True)
    path = Path(__file__).resolve().with_name("test_column_sedimentation.py")
    spec = importlib.util.spec_from_file_location("thompson_backend_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.monotonic()
    checks = 0
    for nz in (1, 44, 64):
        for dtype in (jnp.float32, jnp.float64):
            module.test_column_layout_and_flux(dtype, nz)
            checks += 1
    with pytest.MonkeyPatch.context() as monkeypatch:
        module.test_column_path_does_not_truncate_adaptive_steps(monkeypatch)
        checks += 1
    print(json.dumps({"asserted_platform": jax.devices()[0].platform,
                      "hardware": str(jax.devices()[0]), "checks": checks,
                      "elapsed_s": time.monotonic()-start,
                      "scope": "regression only; no WRF fidelity claim"}), flush=True)


if __name__ == "__main__":
    main()
