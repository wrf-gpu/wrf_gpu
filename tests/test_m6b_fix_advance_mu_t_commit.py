from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_MODE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"
OPERATIONAL_STATE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_state.py"
COMPARE_SCRIPT = ROOT / "scripts" / "m6b_real_ic_operational_compare.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_advance_mu_t_outputs_are_committed_by_shared_core():
    mode = _source(OPERATIONAL_MODE)
    state = _source(OPERATIONAL_STATE)
    core = _source(ROOT / "src" / "gpuwrf" / "dynamics" / "core" / "acoustic.py")

    assert "acoustic_substep_core" in mode
    assert 'mu=advanced["mu"]' in core
    assert 'theta=advanced["theta"]' in core
    assert 'mudf=advanced["mudf"]' in core
    assert 'muts=advanced["muts"]' in core
    assert 'muave=advanced["muave"]' in core
    assert 'ww=scratch["ww"]' in core
    assert 'mudf=acoustic.mudf' in _source(COMPARE_SCRIPT)
    assert "mudf: jax.Array" in state
    assert "mu_new = state.mu_perturbation" not in mode
    assert "module_small_step_em.F:1533-1550" in core


def test_w_coefficients_and_dt_sub_follow_contracted_acoustic_cadence():
    mode = _source(OPERATIONAL_MODE)
    core = _source(ROOT / "src" / "gpuwrf" / "dynamics" / "core" / "acoustic.py")

    assert "calc_coef_w_wrf_coefficients(" in mode
    assert "acoustic.coef_mut if acoustic.coef_mut is not None else acoustic.muts" in mode
    assert "dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)" in mode
    assert "float(dt_stage) / float(scan_substeps)" not in mode
    assert "solve_em.F:2409-2717" in mode
    assert "solve_em.F:3065" in core


def test_ph_tend_matches_validation_bound_theta_delta_formula():
    mode = _source(OPERATIONAL_MODE)
    core = _source(ROOT / "src" / "gpuwrf" / "dynamics" / "core" / "acoustic.py")

    assert "theta_delta = jnp.asarray(theta_new) - jnp.asarray(theta_old)" in core
    assert "set(0.01 * theta_delta)" in core
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
