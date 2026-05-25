from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_MODE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"
OPERATIONAL_STATE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_state.py"
COMPARE_SCRIPT = ROOT / "scripts" / "m6b_real_ic_operational_compare.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_advance_mu_t_outputs_are_committed_to_resident_operational_state():
    mode = _source(OPERATIONAL_MODE)
    state = _source(OPERATIONAL_STATE)

    assert 'mu_new = advanced["mu"]' in mode
    assert 'theta_new = advanced["theta"] + theta_offset' in mode
    assert "theta=theta_new" in mode
    assert 'mudf_new=advanced["mudf"]' in mode
    assert 'muts_new=advanced["muts"]' in mode
    assert 'muave_new=advanced["muave"]' in mode
    assert 'ww_new=advanced["ww"]' in mode
    assert 'mudf=acoustic.mudf' in _source(COMPARE_SCRIPT)
    assert "mudf: jax.Array" in state
    assert "mu_new = state.mu_perturbation" not in mode
    assert "module_small_step_em.F:1102-1108" in mode
    assert "module_small_step_em.F:1141-1171" in mode


def test_w_coefficients_and_dt_sub_follow_contracted_acoustic_cadence():
    mode = _source(OPERATIONAL_MODE)

    assert "calc_coef_w_wrf_coefficients(\n        carry.muts" in mode
    assert "dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)" in mode
    assert "float(dt_stage) / float(scan_substeps)" not in mode
    assert "solve_em.F:2409-2717" in mode
    assert "solve_em.F:1472-1483" in mode


def test_ph_tend_matches_validation_bound_theta_delta_formula():
    mode = _source(OPERATIONAL_MODE)

    assert "theta_delta = jnp.asarray(theta_new) - jnp.asarray(theta_old)" in mode
    assert "set(0.01 * theta_delta)" in mode
    assert "module_small_step_em.F:1345-1395" in mode
    assert "new_state.ph) - jnp.asarray(old_state.ph" not in mode


def test_operational_fix_does_not_import_validation_only_composition():
    tree = ast.parse(_source(OPERATIONAL_MODE))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert "gpuwrf.dynamics.acoustic_loop" not in imported_modules
    assert "gpuwrf.dynamics.dycore_step" not in imported_modules
    assert "gpuwrf.dynamics.coupled_step" not in imported_modules
