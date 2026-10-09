"""RE03 deletion/behaviour mutants for GPUWRF_NEST_O3_FROM_PARENT (CPU, ONE fixture load).

Controls: every e2e test of tests/v025/fid_q2/test_nest_o3_from_parent.py passes on the candidate.  Mutants are
in-process source edits (exec of the edited function into its module, E151-style) or carry edits; each must make its
targeted test(s) fail.  jax.clear_caches() before every arm (E161).
usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 python mutants.py <worktree> <out.json>
"""
import __future__
import inspect
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "tests/v025/fid_q2"))
import jax  # noqa: E402
import pytest  # noqa: E402

import test_nest_o3_from_parent as T  # noqa: E402
from gpuwrf.coupling import physics_couplers as pc  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert T.__file__.startswith(str(WT)) and om.__file__.startswith(str(WT)), (T.__file__, om.__file__)
E2E = ("test_nested_carries_seed_a_held_real_o3rad", "test_root_refreshes_o3rad_only_at_its_radiation_calls",
       "test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it",
       "test_fused_cascade_forces_child_o3rad_like_the_eager_force", "test_override_reaches_both_rrtmg_column_states")


def run(name, nested):
    jax.clear_caches()
    fn = getattr(T, name)
    mp = pytest.MonkeyPatch()
    t0 = time.time()
    try:
        fn(nested, mp) if "monkeypatch" in inspect.signature(fn).parameters else fn(nested)
        status = "pass"
    except BaseException as exc:  # noqa: BLE001 - record any failure mode
        tb = traceback.extract_tb(exc.__traceback__)
        where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb) if "test_nest_o3" in f.filename), "?")
        status = f"FAIL {type(exc).__name__} at {where}: {str(exc)[:160]}"
    finally:
        mp.undo()
    return {"status": status, "s": round(time.time() - t0, 1)}


def exec_mutant(module, func, old, new):
    """Replace ``func`` in ``module`` by its source with exactly one ``old`` -> ``new`` edit; returns a restorer."""
    src = inspect.getsource(getattr(module, func))
    assert src.count(old) == 1, (func, old)
    saved = getattr(module, func)
    code = compile(src.replace(old, new), module.__file__, "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, module.__dict__)
    return lambda: setattr(module, func, saved)


SRC_MUTANTS = {
    "eager_force_hunk_deleted": (dt, "_operational_force",
        "    return force_child_carry_o3rad(\n        child.replace(state=forced_state), parent, edge.weights, "
        "parent_grid_ratio=int(edge.parent_grid_ratio)\n    )\n", "    return child.replace(state=forced_state)\n",
        ("test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it",)),
    "fused_force_hunk_deleted": (dt, "_build_fused_cascade_program",
        "            child_forced = force_child_carry_o3rad(\n                child_carry.replace(state=forced_state), "
        "parent_new, child_weights[idx],\n                parent_grid_ratio=int(child_ratios[idx]),\n            )\n",
        "            child_forced = child_carry.replace(state=forced_state)\n",
        ("test_fused_cascade_forces_child_o3rad_like_the_eager_force",)),
    "child_runs_own_cam": (om, "_refresh_rrtmg_driver", "            if nest_o3:\n", "            if False:\n",
        ("test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it",)),
    "root_uses_held_field": (om, "_refresh_rrtmg_driver",
        'nest_o3 = held_o3 is not None and not bool(getattr(namelist.boundary_config, "force_geopotential", True))',
        "nest_o3 = held_o3 is not None", ("test_root_refreshes_o3rad_only_at_its_radiation_calls",)),
    "root_does_not_store": (om, "_refresh_rrtmg_driver",
        "                    new_o3 = jnp.moveaxis(o3_columns, -1, 0).astype(held_o3.dtype)\n", "                    pass\n",
        ("test_root_refreshes_o3rad_only_at_its_radiation_calls",)),
    "root_refreshes_every_step": (om, "_refresh_rrtmg_driver",
        "        refreshed = refresh(None) if run_radiation else held\n",
        "        refreshed = refresh(None) if run_radiation else held[:3] + refresh(None)[3:]\n",
        ("test_root_refreshes_o3rad_only_at_its_radiation_calls",)),
    "override_not_passed": (om, "_refresh_rrtmg_driver", "            **o3_kwargs,\n", "",
        ("test_root_refreshes_o3rad_only_at_its_radiation_calls",
         "test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it")),
    "override_ignored_in_column_inputs": (pc, "_rrtmg_column_inputs_impl",
        "        ozone_vmr = ozone_vmr_override\n", "        ozone_vmr = ozone_vmr\n",
        ("test_override_reaches_both_rrtmg_column_states",)),
}

out = {"worktree": str(WT), "controls": {}, "mutants": {}}
t0 = time.time()
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE03") as tmp:
    nested = T.load_nested(Path(tmp))
out["load_s"] = round(time.time() - t0, 1)
for name in E2E:
    out["controls"][name] = run(name, nested)
    print("control", name, out["controls"][name], flush=True)

bundles, carries, tree = nested
no_seed = (bundles, {k: c.replace(o3rad=None) for k, c in carries.items()}, tree)
out["mutants"]["seed_hunk_deleted"] = {n: run(n, no_seed) for n in (E2E[0], E2E[2])}
print("seed_hunk_deleted", out["mutants"]["seed_hunk_deleted"], flush=True)
for mname, (module, func, old, new, targets) in SRC_MUTANTS.items():
    restore = exec_mutant(module, func, old, new)
    try:
        out["mutants"][mname] = {n: run(n, nested) for n in targets}
    finally:
        restore()
    print(mname, out["mutants"][mname], flush=True)

out["all_controls_pass"] = all(v["status"] == "pass" for v in out["controls"].values())
out["all_mutants_killed"] = all(all(r["status"] != "pass" for r in m.values()) for m in out["mutants"].values())
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print("all_controls_pass", out["all_controls_pass"], "all_mutants_killed", out["all_mutants_killed"])
