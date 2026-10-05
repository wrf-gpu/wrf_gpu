"""Run unchanged snapshot gates against one actual source-hunk deletion."""
import argparse
from pathlib import Path
import sys
ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['branch','producer','native']);a=ap.parse_args()
snap=Path('<USER_HOME>/wrf_gpu2_lanes/b-phys/BP80/snap')
sys.path.insert(0,str(snap/'src'));sys.path.insert(0,str(snap/'tests/v025/b_phys'))
sys.path.insert(0,str(snap/'scripts/v025'))
import cpu_guard  # must establish backend protection before any JAX import
import jax
assert jax.devices()[0].platform=='cpu'
if a.kind=='branch':
    from gpuwrf.physics import surface_layer as module
    old='snow_land = is_land & (snow_depth >= _lit(snow_depth, 0.1))'
    new='snow_land = jnp.zeros_like(is_land)'
    selector='test_real_swiss_snow_and_bare_match_pristine'
elif a.kind=='producer':
    from gpuwrf.physics import noahmp_coupler as module
    old='snowh=land_state.snowh'
    new='snowh=None'
    selector='test_noah_adapter_reads_entry_snow_depth'
else:
    from gpuwrf.physics.fp32 import surface_layer_real as module
    old='return _surface_layer_impl(state, first_timestep, jnp.float32, snowh=snowh)'
    new='return _surface_layer_impl(state, first_timestep, jnp.float32)'
    selector='test_real_swiss_snow_and_bare_match_pristine'
src=Path(module.__file__).read_text();assert src.count(old)==1
exec(compile(src.replace(old,new),module.__file__,'exec'),module.__dict__)
jax.clear_caches()
import pytest
sys.exit(pytest.main([str(snap/'tests/v025/b_phys/test_sfclay_snow_andreas.py'),'-k',selector,
                     '-q','--disable-warnings','--basetemp','<USER_HOME>/wrf_gpu2_lanes/b-phys/pytest',
                     '--junitxml','<USER_HOME>/wrf_gpu2_lanes/b-phys/BP80/mutant_'+a.kind+'.xml']))
