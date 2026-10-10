"""Anonymized jaxpr of _physics_step_forcing (E151) for OFF-path identity, plus carry leaf count."""
import dataclasses, hashlib, json, os, re, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import OperationalNamelist, _physics_step_forcing, _initial_carry_for_run

def grid(ny=4, nx=5, nz=10):
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    pr = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    tm = TerrainProvenance(source_path="offpath", sha256="offpath", shape=(ny, nx), units="m",
                           projection_transform="native-wrf-lambert", max_elevation_m=0.0, coastline_sanity_check_passed=True)
    m = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="offpath")
    return GridSpec(pr, tm, VerticalCoord("hybrid_eta", nz, 5000.0, eta), BCMetadata("ideal", (), 1, "linear", True), eta, jnp.zeros((ny, nx)), metrics=m)

def state(g):
    nz, ny, nx = g.nz, g.ny, g.nx
    f = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(g).items()}
    p = jnp.broadcast_to(jnp.linspace(97000.0, 25000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 10000.0 * 9.81, nz + 1)[:, None, None], (nz + 1, ny, nx))
    f.update(theta=jnp.full((nz, ny, nx), 300.0), p_total=p, ph_total=ph, mu_total=jnp.full((ny, nx), 90000.0),
             qv=jnp.full((nz, ny, nx), 6e-3), qc=jnp.full((nz, ny, nx), 1e-5), qke=jnp.full((nz, ny, nx), 0.5),
             u=jnp.full((nz, ny, nx + 1), 4.0), v=jnp.full((nz, ny + 1, nx), 1.0), t_skin=jnp.full((ny, nx), 302.0),
             xland=jnp.full((ny, nx), 1.0), mavail=jnp.full((ny, nx), 0.5), roughness_m=jnp.full((ny, nx), 0.1),
             ustar=jnp.full((ny, nx), 0.3), lu_index=jnp.zeros((ny, nx), dtype=jnp.int32))
    return State(**f)

def anon(txt):
    txt = re.sub(r"/[^\s\"':]+\.py:\d+(:\d+)?", "SRC", txt)
    txt = re.sub(r"name_and_src_info=[^,\]\)]*", "NSI", txt)
    return txt

out = {}
g = grid(); s = state(g)
z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)
tend = Tendencies(z((g.nz, g.ny, g.nx + 1)), z((g.nz, g.ny + 1, g.nx)), z((g.nz + 1, g.ny, g.nx)), z((g.nz, g.ny, g.nx)),
                  z((g.nz, g.ny, g.nx)), z((g.nz, g.ny, g.nx)), z((g.nz + 1, g.ny, g.nx)), z((g.ny, g.nx)))
for label, over in (("default_bl5", {}), ("ysu_bl1", dict(bl_pbl_physics=1, sf_sfclay_physics=1))):
    nml = dataclasses.replace(OperationalNamelist.from_grid(g, dt_s=10.0, tendencies=tend),
                              time_utc="2024-06-01T12:00:00Z", run_physics=True, use_noahmp=False, **over)
    carry = _initial_carry_for_run(s, nml)
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    def fn(c):
        f = _physics_step_forcing(c, nml, 0.0, run_radiation=False)
        d = f.dry_tendencies
        return f.state, f.carry, {k: v for k, v in vars(d).items() if v is not None and hasattr(v, "shape")}
    jx = jax.make_jaxpr(fn)(carry)
    a = anon(str(jx))
    out[label] = {"jaxpr_sha": hashlib.sha256(a.encode()).hexdigest(), "eqns": len(jx.jaxpr.eqns),
                  "carry_leaves": len(leaves), "carry_leaf_sig": hashlib.sha256(repr([(x.shape, str(x.dtype)) for x in leaves]).encode()).hexdigest()}
    open(os.path.join(sys.argv[2], f"{label}.jaxpr.txt"), "w").write(a)
json.dump(out, open(os.path.join(sys.argv[2], "summary.json"), "w"), indent=1)
print(sys.argv[1], json.dumps(out))
