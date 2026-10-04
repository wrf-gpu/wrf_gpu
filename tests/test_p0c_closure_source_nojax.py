"""P0 closure (contract P0_CLOSURE_CONTRACT.md rev 2) source controls, JAX-free: never imports gpuwrf.

Each test binds one K item to its exact WRF form on the product SOURCE (AST / literal arithmetic) and names the mutant
it kills.  Numeric WRF-oracle evidence is the separate L2/L3 packages (O-B, O-2, O-C, O-D); these are cheap guards.
Run: python3 -m pytest -q -p no:cacheprovider tests/test_p0c_closure_source_nojax.py   (or python3 <file>)
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PC = ROOT / "src/gpuwrf/coupling/physics_couplers.py"
TC = ROOT / "src/gpuwrf/physics/thompson_column.py"
TAC = ROOT / "src/gpuwrf/physics/thompson_aero_column.py"
R1, R2 = 1.0e-12, 1.0e-6   # WRF module_mp_thompson.F:183-184 (port thompson_constants)


def _module_constant(path, name):
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return eval(compile(ast.Expression(n.value), str(path), "eval"), {})   # literal arithmetic only
    raise AssertionError(f"{name} not defined at module level in {path}")


def test_k1_coupler_exner_exponent_is_wrf_rcp_two_sevenths():
    """K1 (R1): WRF rcp = r_d/cp with cp = 7 r_d/2.  Kills the port's 287/1004 (0.2858566)."""
    kappa = _module_constant(PC, "R_D_OVER_CP")
    assert kappa == 2.0 / 7.0, kappa
    assert kappa != 287.0 / 1004.0



def _fn_src(path, name):
    tree = ast.parse(path.read_text())
    return ast.unparse(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))


THOMPSON_BUILDERS = ("_thompson_column_from_state", "_thompson_aero_column_from_state")


def test_k2_thompson_builders_pass_bottom_face_w():
    """K2 (R3): WRF w1d(k) = w(i,k,j), the bottom w-face (MPT :1224, :3416, :3657).  Kills mass-point w."""
    for name in THOMPSON_BUILDERS:
        src = _fn_src(PC, name)
        assert "w=_to_columns(state.w[:-1])" in src, name
        assert "_w_mass(" not in src, name


def test_k3_thompson_builders_use_wrf_physics_g_dz():
    """K3 (R2): Thompson dz = WRF dz8w from (PH+PHB)/9.81.  Kills the 9.80665 _column_dz_from_state (3.4e-4 rel)."""
    for name in THOMPSON_BUILDERS:
        src = _fn_src(PC, name)
        assert "dz_columns = _surface_dz_from_state(state)" in src, name
        assert "_column_dz_from_state(" not in src, name
    dz_src = _fn_src(PC, "_surface_dz_from_state")
    assert "WRF_PHYSICS_G_M_S2" in dz_src
    assert _module_constant(PC, "WRF_PHYSICS_G_M_S2") == 9.81


def _band_fn(path=None):
    """The PRODUCT _wrf_cloud_sed_band source, executed with numpy as jnp (the helper uses arange/cumsum/max/where)."""
    import numpy as np

    tree = ast.parse((path or TC).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_wrf_cloud_sed_band")
    ns = {"jnp": np, "R2": R2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "_wrf_cloud_sed_band", "exec"), ns)
    return ns["_wrf_cloud_sed_band"]


def _wrf_ksed_loop(rc, dz):
    """Literal transcription of WRF module_mp_thompson.F:3598, :3646-3653 (1-based -> 0-based), one column."""
    nz = len(rc)
    ksed, hgt_agl = 0, 0.0                      # ksed1(:) = 1 (kts)
    for k in range(0, nz - 1):                 # do k = kts, kte-1
        if rc[k] > R2:
            ksed = k
        hgt_agl += dz[k]
        if hgt_agl > 500.0:
            break                               # goto 151
    return [k <= ksed for k in range(nz)]


def _k4_cases():
    import numpy as np

    rng = np.random.default_rng(20260924)
    cases = [
        ([3e-4, 3e-4, 5e-7, 0.0, 0.0, 0.0], [80.0] * 6),          # critic case K: thin rc in (R1,R2] above ksed1
        ([5e-7, 5e-7, 5e-7, 5e-7, 0.0, 0.0], [80.0] * 6),         # no rc > R2 in band -> kts only
        ([0.0, 0.0, 0.0, 3e-4, 3e-4, 0.0], [100.0, 100.0, 100.0, 200.0, 100.0, 100.0]),   # height below k3 = 300
        ([0.0, 0.0, 0.0, 0.0, 0.0, 3e-4], [100.0] * 6),           # level 5 is kte: excluded (k <= kte-1)
        ([0.0, 3e-4, 3e-4, 0.0], [250.0, 250.0, 100.0, 100.0]),   # height below k2 = exactly 500 m -> visited
        ([3e-4, 3e-4, 3e-4, 3e-4], [600.0, 50.0, 50.0, 50.0]),    # first layer > 500 m -> kts only
    ]
    for _ in range(200):
        nz = int(rng.integers(3, 12))
        rc = np.where(rng.random(nz) < 0.5, 10.0 ** rng.uniform(-13, -3, nz), 0.0)
        dz = rng.choice([40.0, 60.0, 100.0, 125.0, 250.0], nz)
        cases.append((list(rc), list(dz)))
    return cases


def test_k4_ksed_band_matches_the_wrf_loop():
    """K4 (D1): kts..ksed1(5), ksed1(5) = highest level with height-below <= 500 m (inclusive) and rc > R2 among
    kts..kte-1, default kts.  Kills the old 'every level strictly below 500 m' mask (case K, exact-500 m)."""
    import numpy as np

    band = _band_fn()
    for rc, dz in _k4_cases():
        rc_w = np.maximum(np.asarray(rc), R1)
        got = band(rc_w[None, :], np.asarray(dz)[None, :])[0]
        assert list(map(bool, got)) == _wrf_ksed_loop(list(rc_w), list(dz)), (rc, dz)


def test_k4_both_cloud_sed_functions_use_the_band():
    for path, name in ((TC, "_sed_cloud_water"), (TAC, "_sed_cloud_water_aero")):
        src = _fn_src(path, name)
        assert "_wrf_cloud_sed_band(" in src, name
        assert "< 500.0" not in src and "below_500m" not in src, name


def _stages_fn():
    """PRODUCT _cloud_sed_rho_stages + density_from_pressure_temperature sources, executed with numpy as jnp."""
    import numpy as np

    tree = ast.parse(TC.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_cloud_sed_rho_stages", "density_from_pressure_temperature")]
    assert len(nodes) == 2
    ns = {"jnp": np, "R1": R1, "R_D": 287.0}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "_cloud_sed_rho_stages", "exec"), ns)
    return ns["_cloud_sed_rho_stages"]


def test_k5_density_stages_match_wrf():
    """K5 (D2): rc with pre-condensation rho (:3217/:3484); orho with the current rho (:3490/:3572 -> :3831); rhof
    from the current rho iff ANY(L_qr) (:3614), else the pre-condensation rho (:3194).  Kills 'post-adjust rho
    everywhere' (the old port) and an ungated rhof."""
    import numpy as np
    from types import SimpleNamespace

    stages = _stages_fn()
    p = np.array([[95000.0, 90000.0, 85000.0]] * 2)
    T = np.array([[285.0, 281.0, 277.0]] * 2)
    qv = np.array([[0.011, 0.009, 0.007]] * 2)
    rho_pre = np.array([[1.1601, 1.1102, 1.0603]] * 2)              # any distinct pre-condensation values
    qr_pre = np.array([[0.0, 5e-12, 0.0], [0.0, 0.0, 0.0]])         # column 0 has L_qr (5e-12 > R1), column 1 not
    rho_rc, rho_f, rho_o = stages(rho_pre, qr_pre, SimpleNamespace(p=p, T=T, qv=qv))
    cur = 0.622 * p / (287.0 * T * (qv + 0.622))                    # WRF :3490/:3572 form
    assert np.array_equal(rho_rc, rho_pre)
    assert np.allclose(rho_o, cur, rtol=0, atol=0)
    assert np.array_equal(rho_f[0], cur[0]) and np.array_equal(rho_f[1], rho_pre[1])


def _body_order(path, body):
    src = _fn_src(path, body)
    order = {k: src.find(k) for k in ("qr_pre_cond = state.qr", "_saturation_adjustment", "_rain_evaporation",
                                      "_cloud_sed_rho_stages(rho_pre_cond, qr_pre_cond, state)")}
    # The sedimentation CALL: the full-column dispatch guard (477831d36) names
    # ``_sedimentation.__module__`` earlier in the body and is not a stage.
    call = re.search(r"\b_sedimentation(?:_aero)?\(", src)
    order["_sedimentation"] = call.start() if call else -1
    return order


def test_k5_bodies_capture_and_pass_the_stages_in_wrf_order():
    for path, body, sed in ((TC, "_thompson_source_sink_body", "cloud_rho=cloud_rho"),
                            (TAC, "_thompson_aero_body", "cloud_sed_on, cloud_rho)")):
        o = _body_order(path, body)
        assert all(v >= 0 for v in o.values()), (body, o)
        assert o["qr_pre_cond = state.qr"] < o["_saturation_adjustment"] < o["_rain_evaporation"] \
            < o["_cloud_sed_rho_stages(rho_pre_cond, qr_pre_cond, state)"] < o["_sedimentation"], (body, o)
        assert sed in _fn_src(path, body), body
    for path, name in ((TC, "_sed_cloud_water"), (TAC, "_sed_cloud_water_aero")):
        src = _fn_src(path, name)
        assert "/ orho_rho *" in src and "rho_rc, rho_f, rho_o" in src, name


def test_k6a_mp28_evaporation_limiter_only_in_the_wrf_evaporation_branch():
    """K6a: WRF :3422/:3467 applies prw_vcd = MAX(-rc*0.99*orho*odt, prw_vcd) only when clap < -eps .and.
    ssatw < -1e-6 (is_aerosol_aware).  Kills the port's unqualified 'clap < 0'."""
    src = _fn_src(TAC, "_saturation_adjustment_aero")
    assert "evap_branch = (clap < -EPS) & (ssatw < -1e-06)" in src
    assert "prw_vcd = jnp.where(~full_evap & evap_branch, jnp.maximum(-rc * 0.99 / state.rho * odt, prw_vcd), prw_vcd)" in src
    assert "~full_evap & (clap < 0.0)" not in src


def test_k6b_mp28_fall_speed_uses_the_wrf_working_droplet_number():
    """K6b: the cloud fall speed uses nc = MAX(2, MIN(Nc*rho, Nt_c_max)) (MPT :3217, :3486).  Kills the re-applied
    entry rebalance _entry_cloud_number (MPT :1832-1842)."""
    src = _fn_src(TAC, "_cloud_water_fall_speeds_aero")
    assert "nc_m3 = jnp.maximum(2.0, jnp.minimum(state.Nc * rho, NT_C_MAX))" in src
    assert "_entry_cloud_number(" not in src


def test_k6c_mp28_number_flux_uses_the_clamped_working_nc():
    """K6c: sed_n = vtnck*nc with nc = MAX(2, MIN(Nc*rho, Nt_c_max)) (MPT :3826).  Kills the raw max(Nc*rho, 0)."""
    src = _fn_src(TAC, "_sed_cloud_water_aero")
    assert "nc = jnp.maximum(2.0, jnp.minimum(state.Nc * rho, NT_C_MAX))" in src
    assert "nc = jnp.maximum(state.Nc * rho, 0.0)" not in src


TESTS = ROOT / "tests"


def test_l1r1_noahmp_fixture_theta_uses_the_wrf_rcp():
    """p0cl1_r1 finding 1: a WRF theta_m state is built with rcp = 2/7.  The OLD fixture (surface_constants 287/1004) read
    back through the K1 coupler gives 293.000842 K instead of 293 K; the corrected fixture is exact.  surface_constants
    keeps 287/1004 (R1-B, separate release blocker) -- asserted unchanged here so no hidden waiver slips in."""
    t, p, p0 = 293.0, 98000.0, 1.0e5
    old = t * (p0 / p) ** (287.0 / 1004.0) * (p / p0) ** (2.0 / 7.0)
    new = t * (p0 / p) ** (2.0 / 7.0) * (p / p0) ** (2.0 / 7.0)
    assert abs(old - 293.0008422598) < 1e-9 and abs(old - t) > 1e-9            # reproduces the r1 failure value
    assert abs(new - t) < 1e-12
    src = (TESTS / "test_v014_noahmp_surface_hook_decoupling.py").read_text()
    assert "from gpuwrf.coupling.physics_couplers import R_D_OVER_CP as WRF_RCP" in src
    assert "theta_dry = t_dry * (P0_PA / p) ** WRF_RCP" in src and "naive_t = theta_m0 * (p[0] / P0_PA) ** WRF_RCP" in src
    assert "from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP" not in src
    assert "CP_D = 7.0 * R_D / 2.0" in (ROOT / "src/gpuwrf/physics/surface_constants.py").read_text()   # R1-B fixed by B39b (WRF cp)


def test_l1r1_m11_fixture_materialises_the_mp28_leaves():
    """p0cl1_r1 finding 2: the mp=28 builder reads state.nwfa / state.nifa directly, so an mp=8-shaped State (None) fails
    there; M11 now applies the production preparation (ensure_conditional_leaves(mp_physics=28)) and physical values."""
    builder = _fn_src(PC, "_thompson_aero_column_from_state")
    assert "_to_columns(state.nwfa)" in builder and "_to_columns(state.nifa)" in builder
    adapter = _fn_src(PC, "thompson_aero_adapter")
    assert "state.ensure_conditional_leaves(mp_physics=28)" in adapter        # what production does before the builder
    fix = _fn_src(TESTS / "test_p0c_closure_jax.py", "_state_with_faces")
    assert ".ensure_conditional_leaves(mp_physics=28)" in fix
    assert "nwfa=jnp.full_like(state.qc, 1000000000.0)" in fix and "nifa=jnp.full_like(state.qc, 1000000.0)" in fix


def test_l1r1_m13_passes_real_gwdo_statics():
    """p0cl1_r1 finding 3: gwdo_columns reads statics.var, so statics=None fails; M13 now passes a flat GWDOStatics built
    by build_gwdo_statics_from_wrf_fields (the tests/test_gwd_gwdo.py pattern) and keeps the KF + GWDO callback scan."""
    assert "var = statics.var.astype(dtype)" in _fn_src(ROOT / "src/gpuwrf/physics/gwd_gwdo.py", "gwdo_columns")
    m13 = _fn_src(TESTS / "test_p0c_closure_jax.py", "test_m13_kf_and_gwdo_adapters_add_no_host_callback")
    assert "pc.build_gwdo_statics_from_wrf_fields(" in m13 and "pc.gwdo_adapter(s, 54.0, statics, grid)" in m13
    assert "gwdo_adapter(s, 54.0, None" not in m13 and "sa.kf_adapter(" in m13 and "for primitive in CALLBACKS" in m13

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"PASS={len(fns)} FAIL=0")
    sys.exit(0)
