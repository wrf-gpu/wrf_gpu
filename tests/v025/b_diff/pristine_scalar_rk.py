"""Unmodified pristine WRF scalar RK update chain for GPUWRF_PHYS_TEND_RK_WRF (D2/D3), C-callable.

Verbatim routines (extracted at build time, never edited): module_em.F ``rk_update_scalar`` and
``rk_update_scalar_pd``, module_physics_addtendc.F ``add_a2a``.  The driver ``chain`` reproduces
solve_em's caller wiring for one species: calculate_phy_tend couples the uncoupled physics tendency
with the time-n mass on the full mass tile; route 0 (moist species) adds it to the zeroed moist_tend
through update_phy_ten -> add_a2a (which skips the outermost ring on specified/nested domains);
route 1 (MYNN number scalars) is pbl_driver's direct ``scalar_tend(P_QNI) = RQNIBLTEN`` on the full
mass tile (module_pbl_driver.F:1866-1874), coupled in place (module_em.F:2473-2481).  The frozen RK1
diffusion + relax tendency is added after it and spec_bdy_scalar OVERWRITES rings < spec_zone of an
owned species (E62 wiring; the operand ``sc_other`` already carries the spec-zone value); then the three RK stages
call rk_update_scalar, preceded at rk_step == rk_order by rk_update_scalar_pd with
mu_old = mu_new = grid%mu_1 for positive-definite / monotonic families (solve_em.F:1932-2000).
``decouple`` is the literal phy_prep_part2 statement RTHRATEN = RTHRATEN/(c1*MUT+c2) with MUT = muts
(solve_em.F:3669, module_big_step_utilities_em.F:5098) applied to the calculate_phy_tend product.
"""
import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
ROUTINES = ("rk_update_scalar", "rk_update_scalar_pd")
ADDTEND_ROUTINES = ("add_a2a",)
FC = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")

DRIVER = r"""
subroutine chain(nx,ny,nz,spec_zone,nested,specified,pd,owned,route,q,phys,sc_other,adv,mu1,mus,mub,mut,c1,c2,msftx,msfty,dt,out) bind(c)
integer(c_int),value :: nx,ny,nz,spec_zone,nested,specified,pd,owned,route
real(c_float) :: q(nx+2,nz+1,ny+2),phys(nx+2,nz+1,ny+2),sc_other(nx+2,nz+1,ny+2),adv(nx+2,nz+1,ny+2,3)
real(c_float) :: mu1(nx+2,ny+2),mus(nx+2,ny+2,3),mub(nx+2,ny+2),mut(nx+2,ny+2),msftx(nx+2,ny+2),msfty(nx+2,ny+2)
real(c_float) :: c1(nz+1),c2(nz+1),out(nx+2,nz+1,ny+2)
real(c_float),value :: dt
real :: moist_tend(0:nx+1,1:nz+1,0:ny+1,1),moist_old(0:nx+1,1:nz+1,0:ny+1,1),moist(0:nx+1,1:nz+1,0:ny+1,1)
real :: advz(0:nx+1,1:nz+1,0:ny+1),h_t(0:nx+1,1:nz+1,0:ny+1),z_t(0:nx+1,1:nz+1,0:ny+1),adv_s(0:nx+1,1:nz+1,0:ny+1)
real :: rv(0:nx+1,1:nz+1,0:ny+1),lv(0:nx+1,1:nz+1,0:ny+1)
real :: dt_rk
integer :: i,k,j,rk_step,ring
type(grid_config_rec_type) :: config_flags
config_flags%nested=nested/=0
config_flags%specified=specified/=0
moist_tend=0.; advz=0.; h_t=0.; z_t=0.
moist(:,:,:,1)=q; moist_old(:,:,:,1)=0.
rv=0.; lv=0.
do j=1,ny; do k=1,nz; do i=1,nx
  ! calculate_phy_tend on its..itf / jts..jtf: RQ?BLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQ?BLTEN(I,K,J)
  rv(i,k,j)=(c1(k)*mut(i+1,j+1)+c2(k))*phys(i+1,k,j+1)
enddo; enddo; enddo
if(route==0)then
  ! update_phy_ten -> add_a2a(moist_tendf(ims,kms,jms,P_Q?),RQ?BLTEN,config_flags,...)
  call add_a2a(lv,rv,config_flags, 1,nx+1,1,ny+1,1,nz+1, 0,nx+1,0,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)
else
  ! pbl_driver: scalar_tend(i,k,j,P_QNI)=RQNIBLTEN(i,k,j) on i_start(ij)..i_end(ij), coupled in place
  lv(1:nx,1:nz,1:ny)=rv(1:nx,1:nz,1:ny)
endif
moist_tend(:,:,:,1)=lv
do j=1,ny; do k=1,nz; do i=1,nx
  ring=min(min(i-1,nx-i),min(j-1,ny-j))
  if(owned/=0 .and. ring<spec_zone)then
    moist_tend(i,k,j,1)=sc_other(i+1,k,j+1)
  else
    moist_tend(i,k,j,1)=moist_tend(i,k,j,1)+sc_other(i+1,k,j+1)
  endif
enddo; enddo; enddo
do rk_step=1,3
  if(rk_step==3)dt_rk=dt
  if(rk_step==2)dt_rk=0.5*dt
  if(rk_step==1)dt_rk=dt/3.
  adv_s=adv(:,:,:,rk_step)
  if(pd/=0 .and. rk_step==3)then
    call rk_update_scalar_pd(2,2,moist_old,moist_tend,c1,c2,mu1,mu1,mub,rk_step,dt_rk,spec_zone,config_flags, &
         1,nx+1,1,ny+1,1,nz+1, 0,nx+1,0,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)
  endif
  call rk_update_scalar(2,2,moist_old,moist,moist_tend,advz,advz,adv_s,h_t,z_t,msftx,msfty,c1,c2,mu1,mus(:,:,rk_step),mub, &
       rk_step,dt_rk,spec_zone,config_flags,.false., 1,nx+1,1,ny+1,1,nz+1, 0,nx+1,0,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)
enddo
out=moist(:,:,:,1)
end subroutine

subroutine pd_update(nx,ny,nz,spec_zone,nested,specified,q,sc,mu1,mub,c1,c2,dt,out) bind(c)
integer(c_int),value :: nx,ny,nz,spec_zone,nested,specified
real(c_float) :: q(nx+2,nz+1,ny+2),sc(nx+2,nz+1,ny+2),mu1(nx+2,ny+2),mub(nx+2,ny+2),c1(nz+1),c2(nz+1),out(nx+2,nz+1,ny+2)
real(c_float),value :: dt
real :: moist(0:nx+1,1:nz+1,0:ny+1,1),tend(0:nx+1,1:nz+1,0:ny+1,1)
type(grid_config_rec_type) :: config_flags
config_flags%nested=nested/=0
config_flags%specified=specified/=0
moist(:,:,:,1)=q; tend(:,:,:,1)=sc
call rk_update_scalar_pd(2,2,moist,tend,c1,c2,mu1,mu1,mub,3,dt,spec_zone,config_flags, &
     1,nx+1,1,ny+1,1,nz+1, 0,nx+1,0,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)
out=moist(:,:,:,1)
end subroutine

subroutine decouple(nx,ny,nz,r,mut,muts,c1,c2,out) bind(c)
integer(c_int),value :: nx,ny,nz
real(c_float) :: r(nx,nz,ny),mut(nx,ny),muts(nx,ny),c1(nz),c2(nz),out(nx,nz,ny)
integer :: i,k,j
do j=1,ny; do k=1,nz; do i=1,nx
  out(i,k,j)=(c1(k)*mut(i,j)+c2(k))*r(i,k,j)      ! calculate_phy_tend (module_em.F)
  out(i,k,j)=out(i,k,j)/(c1(k)*muts(i,j)+c2(k))   ! phy_prep_part2 (module_big_step_utilities_em.F:5098)
enddo; enddo; enddo
end subroutine
"""


def build(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    src = (WRF / "dyn_em/module_em.F").read_text()
    routines = {n: re.search(r"(?ims)^\s*SUBROUTINE " + n + r"\(.*?^\s*END SUBROUTINE " + n + r"\b", src).group(0)
                for n in ROUTINES}
    addtend = (WRF / "phys/module_physics_addtendc.F").read_text()
    routines.update({n: re.search(r"(?ims)^\s*SUBROUTINE " + n + r"\(.*?^\s*END SUBROUTINE " + n + r"\b", addtend).group(0)
                     for n in ADDTEND_ROUTINES})
    header = ("module wrf_scalar_rk_oracle\nuse iso_c_binding\nimplicit none\n"
              "type grid_config_rec_type\nlogical :: nested=.false.,specified=.false.,periodic_x=.false.,periodic_y=.false.\n"
              "integer :: rk_ord=3\nend type\ncontains\n")
    path = directory / "oracle.f90"
    path.write_text(header + "\n".join(routines.values()) + "\n" + DRIVER + "\nend module\n")
    cmd = [FC, "-O2", "-cpp", "-shared", "-fPIC", "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds",
           "-J", str(directory), str(path), "-o", str(directory / "oracle.so")]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr)
    (directory / "manifest.json").write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(src.encode()).hexdigest(),
        addtend_sha256=hashlib.sha256(addtend.encode()).hexdigest(),
        routine_sha256={n: hashlib.sha256(r.encode()).hexdigest() for n, r in routines.items()},
        caller_sha256=hashlib.sha256((WRF / "dyn_em/solve_em.F").read_bytes()).hexdigest(), command=cmd), indent=2))
    return directory / "oracle.so"


def _f(a, shape=None):
    a = np.asarray(a, np.float32)
    return np.asfortranarray(a if shape is None else a.reshape(shape))


def _pad3(a):  # (nz, ny, nx) -> fortran (nx+2, nz+1, ny+2), interior at 1..n
    nz, ny, nx = a.shape
    out = np.zeros((nx + 2, nz + 1, ny + 2), np.float32, order="F")
    out[1:nx + 1, :nz, 1:ny + 1] = np.asarray(a, np.float32).transpose(2, 0, 1)
    return out


def _pad2(a):
    ny, nx = a.shape
    out = np.zeros((nx + 2, ny + 2), np.float32, order="F")
    out[1:nx + 1, 1:ny + 1] = np.asarray(a, np.float32).T
    return out


def chain(lib: Path, *, q, phys, sc_other, adv, mu1, mus, mub, mut, c1, c2, msftx, msfty, dt,
          spec_zone, nested, specified, pd, owned, route):
    """WRF 3-stage scalar update of one species; all arrays (nz, ny, nx) / (ny, nx) REAL."""
    nz, ny, nx = q.shape
    so = ctypes.CDLL(str(lib))
    fn = so.chain
    fn.argtypes = [ctypes.c_int] * 9 + [ctypes.c_void_p] * 12 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = None
    adv4 = np.zeros((nx + 2, nz + 1, ny + 2, 3), np.float32, order="F")
    for s in range(3):
        adv4[..., s] = _pad3(adv[s])
    mus3 = np.zeros((nx + 2, ny + 2, 3), np.float32, order="F")
    for s in range(3):
        mus3[..., s] = _pad2(mus[s])
    vec = lambda v: np.asfortranarray(np.pad(np.asarray(v, np.float32), (0, 1), mode="edge"))
    out = np.zeros((nx + 2, nz + 1, ny + 2), np.float32, order="F")
    args = [_pad3(q), _pad3(phys), _pad3(sc_other), adv4, _pad2(mu1), mus3, _pad2(mub), _pad2(mut),
            vec(c1), vec(c2), _pad2(msftx), _pad2(msfty)]
    keep = args + [out]
    fn(nx, ny, nz, int(spec_zone), int(nested), int(specified), int(pd), int(owned), int(route),
       *[a.ctypes.data for a in args[:12]], ctypes.c_float(dt), out.ctypes.data)
    del keep
    return out[1:nx + 1, :nz, 1:ny + 1].transpose(1, 2, 0).copy()


def decouple(lib: Path, *, r, mut, muts, c1, c2):
    nz, ny, nx = r.shape
    so = ctypes.CDLL(str(lib))
    fn = so.decouple
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
    fn.restype = None
    rf = np.asfortranarray(np.asarray(r, np.float32).transpose(2, 0, 1))
    out = np.zeros_like(rf, order="F")
    a = [rf, _f(np.asarray(mut).T), _f(np.asarray(muts).T), _f(c1), _f(c2)]
    fn(nx, ny, nz, *[x.ctypes.data for x in a], out.ctypes.data)
    return out.transpose(1, 2, 0).copy()


def pd_update(lib: Path, *, q, sc, mu1, mub, c1, c2, dt, spec_zone, nested, specified):
    """Verbatim rk_update_scalar_pd (rk_step 3, mu_old = mu_new = mu_1) on one species."""
    nz, ny, nx = q.shape
    so = ctypes.CDLL(str(lib))
    fn = so.pd_update
    fn.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 6 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = None
    vec = lambda v: np.asfortranarray(np.pad(np.asarray(v, np.float32), (0, 1), mode="edge"))
    out = np.zeros((nx + 2, nz + 1, ny + 2), np.float32, order="F")
    args = [_pad3(q), _pad3(sc), _pad2(mu1), _pad2(mub), vec(c1), vec(c2)]
    fn(nx, ny, nz, int(spec_zone), int(nested), int(specified), *[a.ctypes.data for a in args],
       ctypes.c_float(dt), out.ctypes.data)
    return out[1:nx + 1, :nz, 1:ny + 1].transpose(1, 2, 0).copy()
