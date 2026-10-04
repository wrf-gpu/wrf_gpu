"""Fresh PROD d01+d02 inputs for b-core CPU tests (replaces integrate's common_inputs.pkl, deleted in the
2026-10-03 disk cleanup): the product loader nested_pipeline._load_domains, once per process, with the
State GPU guard pointed at the CPU device for the duration of the load only (conftest's GPU-skip hook
keeps working for every other test).  Skips when the PROD inputs are absent."""
import functools
from pathlib import Path

import pytest

PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


@functools.lru_cache(maxsize=1)
def _load():
    import jax

    import gpuwrf.contracts.state as state_module
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains

    assert jax.devices()[0].platform == "cpu"
    guard = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices()[0]
    try:
        # _load_domains reads only input_dir/feedback from the config; the output paths are never touched.
        config = NestedPipelineConfig(PROD, PROD / "unused_out", PROD / "unused_proof", hours=1, max_dom=2)
        return _load_domains(config, ("d01", "d02"))
    finally:
        state_module._gpu_device = guard


def prod_domains():
    """(hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries), the old pickle's tuple."""
    if not (PROD / "wrfinput_d02").exists():
        pytest.skip(f"PROD inputs unavailable: {PROD}")
    return _load()
