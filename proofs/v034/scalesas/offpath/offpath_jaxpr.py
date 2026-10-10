"""OFF-path identity: CPU jaxpr (source locations stripped) of the operational physics step.

Configs exercise the release suite seams that the o1-sas hunks touch (_initial_carry_for_run,
the cumulus slot, suite validation). Usage: python offpath_jaxpr.py <out.json>
"""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/o1-sas")
spec = importlib.util.spec_from_file_location("smoke", ROOT / "tests/test_v013_operational_smoke.py")
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
from gpuwrf.runtime.operational_mode import _initial_carry_for_run, _physics_step_forcing, _resolve_operational_suite
import gpuwrf._fast_defaults as fd

def strip(txt):
    txt = re.sub(r"name_and_src_info=[^\n]*?(?=\s\w+=|\])", "", txt)
    txt = re.sub(r"/[^\s:\"']+\.py:\d+(:\d+)?", "<loc>", txt)
    txt = re.sub(r"0x[0-9a-f]+", "0x", txt)
    return txt

CONFIGS = {
    "kf_thompson_mynn_rrtmg": dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=1, rad_rk_tendf=1),
    "cu0_thompson_mynn_rrtmg": dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0, rad_rk_tendf=1),
    "ntiedtke16": dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=16,
                       use_flux_advection=True, moist_adv_opt=2, rad_rk_tendf=1),
}
out = {"fast_defaults": {k: v for k, v in sorted(fd.FAST_PATH_DEFAULTS.items())} if hasattr(fd, "FAST_PATH_DEFAULTS") else None}
grid = smoke._grid()
state = smoke._convective_state(grid)
for name, over in CONFIGS.items():
    nml = smoke._namelist(grid, dt_s=60.0, **over)
    _resolve_operational_suite(nml)
    carry = _initial_carry_for_run(state, nml)
    struct = str(jax.tree_util.tree_structure(carry)) + str([(l.shape, str(l.dtype)) for l in jax.tree_util.tree_leaves(carry)])
    import dataclasses
    def step(c):
        f = _physics_step_forcing(c, nml, 0.0, run_radiation=True)
        dry = tuple(getattr(f.dry_tendencies, fl.name) for fl in dataclasses.fields(f.dry_tendencies))
        return f.state, f.carry, dry
    jx = jax.make_jaxpr(step)(carry)
    txt = strip(str(jx))
    out[name] = {"jaxpr_sha256": hashlib.sha256(txt.encode()).hexdigest(), "n_chars": len(txt),
                 "carry_sha256": hashlib.sha256(struct.encode()).hexdigest()}
    print(name, out[name], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
