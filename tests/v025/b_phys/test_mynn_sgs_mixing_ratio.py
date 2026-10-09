"""BP85: original WRF MYNN driver post-copy on real drizzle columns.

The fixture carries pristine CORE outputs: upstream condensation/plume error
cannot compensate for a wrong producer-seam unit. OFF checks are regression.
"""
import ctypes
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import mynn_pbl as P

FLAG = "GPUWRF_MYNN_SGS_MIXING_RATIO"
FIXTURE = Path(__file__).parent / "fixtures/mynn_sgs_drizzle_wrf.json"


@pytest.fixture(scope="module")
def pristine_postcopy(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    source = root / "data/wrf_pristine/WRF/phys/module_bl_mynnedmf_driver.F"
    text = source.read_text()
    start = text.index("!- Collect 3D ouput:")
    body = re.search(r"if \(icloud_bl > 0\) then.*?\n\s*endif", text[start:], re.S).group(0)
    # Exactly the original three source expressions, in a C ABI wrapper.
    assert "qc_bl1(k)/(1.0 - sqv1(k))" in body
    assert "qi_bl1(k)/(1.0 - sqv1(k))" in body
    assert "cldfra_bl(i,k,j) = cldfra_bl1(k)" in body
    generated = """subroutine bp85_copy(n,qv,qc_bl1,qi_bl1,cldfra_bl1,co,io,fo) bind(C)
use iso_c_binding
implicit none
integer(c_int),value :: n
real(c_float),intent(in) :: qv(n),qc_bl1(n),qi_bl1(n),cldfra_bl1(n)
real(c_float),intent(out) :: co(n),io(n),fo(n)
real(c_float) :: sqv1(n),qc_bl(1,n,1),qi_bl(1,n,1),cldfra_bl(1,n,1)
integer :: k,kts,kte,i,j,icloud_bl
kts=1;kte=n;i=1;j=1;icloud_bl=1
do k=1,n
 sqv1(k)=qv(k)/(1.0+qv(k))
enddo
""" + body + """
co=qc_bl(1,:,1);io=qi_bl(1,:,1);fo=cldfra_bl(1,:,1)
end subroutine
"""
    out = tmp_path_factory.mktemp("bp85_postcopy")
    (out / "copy.f90").write_text(generated)
    subprocess.run(["timeout", "60", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran",
                    "-O2", "-ffp-contract=off", "-fPIC", "-shared", "copy.f90", "-o", "copy.so"],
                   cwd=out, check=True, capture_output=True)
    (out / "provenance.json").write_text(json.dumps({
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(), "kind": 4,
        "library_sha256": hashlib.sha256((out / "copy.so").read_bytes()).hexdigest()}, indent=2))
    lib = ctypes.CDLL(str(out / "copy.so"))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.bp85_copy.argtypes = [ctypes.c_int] + [pointer] * 7
    lib.bp85_copy.restype = None
    def oracle(qv, qc, qi, cf):
        arrays = [np.ascontiguousarray(x, np.float32).reshape(-1) for x in [qv,qc,qi,cf]]
        outputs = [np.empty_like(arrays[0]) for _ in range(3)]
        lib.bp85_copy(len(arrays[0]), *arrays, *outputs)
        return [x.reshape(np.shape(qv)) for x in outputs]
    return oracle


def column_batch():
    fixture = json.loads(FIXTURE.read_text())
    before = np.asarray(fixture["input_values"], np.float32)
    ncol, nz = before.shape[:2]
    base = GridSpec.canary_3km_template()
    eta = jnp.linspace(1.,0.,nz+1,dtype=jnp.float32)
    grid = replace(base, projection=replace(base.projection,nx=ncol,ny=1),
                   vertical=replace(base.vertical,nz=nz,eta_levels=eta),eta_levels=eta,
                   terrain=replace(base.terrain,shape=(1,ncol)),terrain_height=jnp.zeros((1,ncol),jnp.float32),metrics=None)
    assert nz == grid.nz
    state = State(**{n:jnp.zeros(shape,jnp.float32) for n,shape in _state_field_shapes(grid).items()})
    mass = lambda x:jnp.asarray(np.asarray(x,np.float32).T[:,None,:])
    state = state.replace(_cast=False, qv=mass(before[:,:,4]),theta=mass(before[:,:,3]),
                          p_total=mass(before[:,:,6]),qc=mass(before[:,:,9]),qi=mass(before[:,:,10]),qs=mass(before[:,:,11]))
    fields = {n:jnp.asarray(np.asarray(v,np.float32)[None,...])
              for n,v in fixture["pristine_producer_output"].items()}
    out = P.MynnPBLColumnState(**fields)
    assert bool(jnp.any(out.qc_bl>0)) and bool(jnp.any(out.qi_bl>0))
    return state,out


def test_real_drizzle_sgs_producer_matches_pristine_postcopy(monkeypatch,pristine_postcopy):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS","1")
    monkeypatch.setenv("GPUWRF_MYNN_CLOUDMIX","1")
    monkeypatch.setenv(FLAG,"1")
    state,out = column_batch()
    got = C._state_from_mynn_output(state,out)
    qv = np.asarray(state.qv)
    raw = [np.asarray(C._from_columns(getattr(out,n))) for n in ["qc_bl","qi_bl","cldfra_bl"]]
    expected = pristine_postcopy(qv,*raw)
    for name,truth in zip(["qc_bl","qi_bl","cldfra_bl"],expected):
        np.testing.assert_array_max_ulp(np.asarray(getattr(got,name)),truth,maxulp=1)
    # Resolved cloud/vapor uses its existing producer, never SGS injection.
    for name in ["qc","qi","qv"]:
        np.testing.assert_array_equal(np.asarray(getattr(got,name)),np.asarray(C._from_columns(getattr(out,name))))
    monkeypatch.setenv(FLAG,"0")
    control = C._state_from_mynn_output(state,out)
    for name in State.__slots__:
        if name in {"qc_bl","qi_bl"}:continue
        lhs,rhs=getattr(got,name),getattr(control,name)
        if lhs is None:assert rhs is None
        else:np.testing.assert_array_equal(np.asarray(lhs),np.asarray(rhs))


def test_off_preserves_original_producer_values_and_lowered_program(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS","1")
    monkeypatch.setenv("GPUWRF_MYNN_CLOUDMIX","1")
    monkeypatch.setenv(FLAG,"0")
    state,out = column_batch()
    body,first = inspect.getsourcelines(C._state_from_mynn_output)
    source="".join(body)
    for name in ["qc_bl","qi_bl"]:
        source=source.replace(f'_mynn_sgs_dry_mixing(_from_columns(out.{name}), state)',f'_from_columns(out.{name})')
    namespace=dict(vars(C))
    exec(compile("\n"*(first-1)+source,C.__file__,"exec"),namespace)
    original=namespace["_state_from_mynn_output"]
    expected=original(state,out);got=C._state_from_mynn_output(state,out)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(got)):
        assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    from jaxlib import _jax as J
    texts=[]
    for fn in [original,C._state_from_mynn_output]:
        hlo=jax.jit(fn).lower(state,out).compiler_ir(dialect="hlo")
        module=J.HloModule.from_serialized_hlo_module_proto(hlo.as_serialized_hlo_module_proto())
        options=J.HloPrintOptions();options.print_metadata=False
        texts.append(module.to_string(options))
    assert texts[0]==texts[1],"OFF changed the source-neutral lowered program"
