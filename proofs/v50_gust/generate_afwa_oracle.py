"""Compile verbatim pristine-WRF AFWA WSPD10MAX blocks into a REAL oracle; run it on real patches.

Usage (CPU only, assigned cores, QUIET respected by the caller):
  python proofs/v50_gust/generate_afwa_oracle.py <lane-build-dir>
Writes proofs/v50_gust/afwa_oracle.json. Reads CPU-WRF wrfout files READ-ONLY.
No port code is imported: operands, clock and Fortran wrapper are independent of gpuwrf.

Verbatim sources (asserted by exact text; SHA-256 recorded):
- phys/module_diag_afwa.F: ZAGL loop, time-step precipitation loop, reset + WSPD10MAX block;
- phys/module_diag_functions.F: FUNCTION uv_wind;
- dyn_em/module_big_step_utilities_em.F: phy_prep u_phy/v_phy statements (step-START winds) and
  moist_physics_prep_em z_at_w / z statements (step-END heights).
The alarm is an independent integer-second emulation of ESMF_Clock.F90 interval alarms
(PrevRingTime advances by the nominal interval) and WRF's CurrTime = step start during solve.
Per-step amounts are hourly CPU-WRF accumulations spread uniformly (real per-step RAINNCV is not
in history): prate = hourly rate. "scaled" patches multiply them to cover factor 0..1 and >150 mm/h.
Time levels are distinct (critic FIX 7): U/V/U10/V10 come from frame t, PH/PHB from frame t-1, so an
operand read from the wrong state changes the expected maximum on every heavy column.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
AFWA = WRF / "phys/module_diag_afwa.F"
FUNCS = WRF / "phys/module_diag_functions.F"
BIG = WRF / "dyn_em/module_big_step_utilities_em.F"
BUILD = Path(sys.argv[1]).resolve()
BUILD.mkdir(parents=True, exist_ok=True)


def between(text, start, end):
    a = text.index(start)
    b = text.index(end, a)
    return text[a:b]


afwa = AFWA.read_text()
zagl_block = between(afwa, "    ! ZAGL (height above ground)", "    ! Calculate relative humidity")
precip_block = between(afwa, "    ! Time-step precipitation (convective + nonconvective)", "    ! Calculate precipitable water")
severe_block = between(afwa, "    ! After each history dump, reset max/min value arrays\n    ! ----------------------------------------------------------------------\n    IF ( is_after_history_dump ) THEN\n      IF ( config_flags % afwa_severe_opt == 1 )",
                       "    ! Calculate 0-2000 foot (0 - 609.6 meter) shear.")
funcs = FUNCS.read_text()
uv_wind = between(funcs, "  FUNCTION uv_wind ( u, v ) result ( wind_speed )", "  END FUNCTION uv_wind") + "  END FUNCTION uv_wind\n"
big = BIG.read_text()
statements = {
    "u_phy": "      u_phy(i,k,j) = 0.5*(u(i,k,j)+u(i+1,k,j))\n",
    "v_phy": "      v_phy(i,k,j) = 0.5*(v(i,k,j)+v(i,k,j+1))\n",
    "z_at_w": "       z_at_w(i,k,j) = (ph(i,k,j)+phb(i,k,j))/g\n",
    "z": "       z(i,k,j) = 0.5*(z_at_w(i,k,j) +z_at_w(i,k+1,j) )\n",
}
for name, line in statements.items():
    assert big.count(line) >= 1, name

fortran = f'''module oracle_types
type domain_t
  real, allocatable :: wspd10max(:,:), afwa_llws(:,:), u10(:,:), v10(:,:), afwa_precip(:,:), afwa_totprecip(:,:)
  real, allocatable :: raincv(:,:), rainncv(:,:), rainc(:,:), rainnc(:,:), z(:,:,:), ht(:,:)
  real :: dt
end type
type config_t
  integer :: afwa_severe_opt
end type
end module
module diag_functions_oracle
contains
{uv_wind}end module
subroutine afwa_step(grid, config_flags, is_after_history_dump, u, v, ph, phb, nx, ny, nz)
use oracle_types
use diag_functions_oracle
implicit none
type(domain_t), intent(inout) :: grid
type(config_t), intent(in) :: config_flags
logical, intent(in) :: is_after_history_dump
integer, intent(in) :: nx, ny, nz
real, intent(in) :: u(1:nx+1,1:nz,1:ny), v(1:nx,1:nz,1:ny+1), ph(1:nx,1:nz+1,1:ny), phb(1:nx,1:nz+1,1:ny)
real, parameter :: g = 9.81
integer :: ims, ime, jms, jme, kms, kme, i, j, k
real, allocatable :: u_phy(:,:,:), v_phy(:,:,:), z_at_w(:,:,:), zagl(:,:,:)
real :: wind_vel, wind_blend, prate_mm_per_hr, factor, u1km, v1km, ublend, vblend
logical :: is_target_level
ims=1; ime=nx; jms=1; jme=ny; kms=1; kme=nz+1
allocate(u_phy(ims:ime,kms:kme,jms:jme), v_phy(ims:ime,kms:kme,jms:jme), z_at_w(ims:ime,kms:kme,jms:jme), zagl(ims:ime,kms:kme,jms:jme))
u_phy = 0.; v_phy = 0.; z_at_w = 0.; zagl = 0.; grid%z = 0.
u1km = 0.; v1km = 0.
do j = jms, jme
do k = 1, nz
do i = ims, ime
{statements["u_phy"]}{statements["v_phy"]}enddo
enddo
enddo
do j = jms, jme
do k = 1, nz+1
do i = ims, ime
{statements["z_at_w"]}enddo
enddo
enddo
do j = jms, jme
do k = 1, nz
do i = ims, ime
{statements["z"].replace("z(i,k,j)", "grid%z(i,k,j)", 1)}enddo
enddo
enddo
{zagl_block}{precip_block}{severe_block}end subroutine
program oracle
use oracle_types
implicit none
integer :: nx, ny, nz, nsteps, s, dt_s, hist_s, curr, prevring
type(domain_t) :: grid
type(config_t) :: config_flags
logical :: after
real, allocatable :: u(:,:,:), v(:,:,:), ph(:,:,:), phb(:,:,:)
open(10,file='input.bin',access='stream',form='unformatted',status='old')
open(11,file='output.bin',access='stream',form='unformatted',status='replace')
read(10) nx, ny, nz, nsteps, dt_s, hist_s
allocate(grid%wspd10max(nx,ny), grid%afwa_llws(nx,ny), grid%u10(nx,ny), grid%v10(nx,ny), grid%afwa_precip(nx,ny))
allocate(grid%afwa_totprecip(nx,ny), grid%raincv(nx,ny), grid%rainncv(nx,ny), grid%rainc(nx,ny), grid%rainnc(nx,ny))
allocate(grid%z(nx,nz+1,ny), grid%ht(nx,ny))
allocate(u(nx+1,nz,ny), v(nx,nz,ny+1), ph(nx,nz+1,ny), phb(nx,nz+1,ny))
config_flags%afwa_severe_opt = 1
grid%dt = real(dt_s)
read(10) grid%wspd10max
grid%afwa_llws = 0.; grid%rainc = 0.; grid%rainnc = 0.
prevring = 0
do s = 1, nsteps
  curr = (s-1)*dt_s
  ! ESMF interval alarm: advanced at clock advance to curr; PrevRingTime += RingInterval.
  if (s > 1 .and. prevring + hist_s <= curr) prevring = prevring + hist_s
  after = curr < prevring + dt_s
  read(10) grid%u10, grid%v10, grid%raincv, grid%rainncv, grid%ht, u, v, ph, phb
  call afwa_step(grid, config_flags, after, u, v, ph, phb, nx, ny, nz)
  write(11) merge(1,0,after), grid%wspd10max
enddo
close(10); close(11)
end program
'''
(BUILD / "oracle.f90").write_text(fortran)
FC = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
command = [FC, "-O2", "-ffp-contract=off", "-fno-fast-math", "-ffree-line-length-none", "oracle.f90", "-o", "oracle"]
subprocess.run(command, cwd=BUILD, check=True, timeout=60)

N148 = "<DATA_ROOT>/alisios/scratch/n148-cpuref-20261018_12z/cpu"
PROD = "<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case"
W227 = "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"
P = 6  # patch edge (columns)


def frame(path):
    with Dataset(path) as d:
        get = lambda n: np.asarray(d[n][0], np.float32)
        return dict(U=get("U"), V=get("V"), PH=get("PH"), PHB=get("PHB"), HGT=get("HGT"), U10=get("U10"),
                    V10=get("V10"), RAIN=get("RAINNC") + get("RAINC"), RAINC=get("RAINC"), RAINNC=get("RAINNC"))


def patch(f, y0, x0):
    s = np.s_[y0:y0 + P, x0:x0 + P]
    return dict(u=f["U"][:, y0:y0 + P, x0:x0 + P + 1], v=f["V"][:, y0:y0 + P + 1, x0:x0 + P],
                ph=f["PH"][(slice(None),) + s], phb=f["PHB"][(slice(None),) + s], ht=f["HGT"][s],
                u10=f["U10"][s], v10=f["V10"][s])


def pick(score, f):
    ny, nx = f["HGT"].shape
    best = None
    for y in range(1, ny - P - 1):
        for x in range(1, nx - P - 1):
            value = score(y, x)
            if best is None or value > best[0]:
                best = (value, y, x)
    return best[1], best[2]


def hourly(path_prev, path_now):
    a, b = frame(path_prev), frame(path_now)
    return (b, np.maximum(b["RAINNC"] - a["RAINNC"], 0).astype(np.float32),
            np.maximum(b["RAINC"] - a["RAINC"], 0).astype(np.float32), a)


def regime(name, files, score_fn, scale=1.0, synthetic=None, dt=54, hist_min=3, steps=12):
    seq = [hourly(files[i], files[i + 1]) for i in range(len(files) - 1)]
    f0, nc0, c0, _ = seq[-1]
    y0, x0 = pick(lambda y, x: score_fn(f0, nc0 + c0, y, x), f0)
    states = []
    for f, nc, c, f_prev in seq:
        p = patch(f, y0, x0)
        heights = patch(f_prev, y0, x0)  # a different time level than the winds
        p["ph"], p["phb"] = heights["ph"], heights["phb"]
        assert not np.array_equal(p["ph"], patch(f, y0, x0)["ph"])
        rnc, rc = nc[y0:y0 + P, x0:x0 + P] * np.float32(scale), c[y0:y0 + P, x0:x0 + P] * np.float32(scale)
        if synthetic is not None:
            rnc = synthetic.astype(np.float32)
            rc = np.zeros_like(rnc)
        # Uniform step amounts: prate = hourly rate (REAL per-step amount).
        p["rainncv"] = (rnc * np.float32(dt) / np.float32(3600)).astype(np.float32)
        p["raincv"] = (rc * np.float32(dt) / np.float32(3600)).astype(np.float32)
        states.append(p)
    nz = states[0]["u"].shape[0]
    initial = np.full((P, P), 3.0, np.float32)
    with (BUILD / "input.bin").open("wb") as fh:
        fh.write(np.asarray([P, P, nz, steps, dt, int(hist_min * 60)], np.int32).tobytes())
        fh.write(np.asfortranarray(initial.T).tobytes(order="F"))
        for s in range(steps):
            st = states[s % len(states)]
            for key in ("u10", "v10", "raincv", "rainncv", "ht"):
                fh.write(np.asfortranarray(st[key].T).tobytes(order="F"))
            for key in ("u", "v", "ph", "phb"):  # (z,y,x) -> Fortran (x,z,y)
                fh.write(np.asfortranarray(np.transpose(st[key], (2, 0, 1))).tobytes(order="F"))
    subprocess.run([str(BUILD / "oracle")], cwd=BUILD, check=True, timeout=60)
    raw = np.fromfile(BUILD / "output.bin", np.uint8)
    rec = 4 + 4 * P * P
    out = raw.reshape(steps, rec)
    reset = out[:, :4].copy().view(np.int32).ravel()
    wmax = out[:, 4:].copy().view(np.float32).reshape(steps, P, P)  # Fortran (x,y) -> C (y,x)
    assert np.isfinite(wmax).all()
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
    prate = [float((st["rainncv"] + st["raincv"]).max() / np.float32(dt) * 3600) for st in states]
    return dict(name=name, files=[str(x) for x in files], patch=[int(y0), int(x0), P], scale=scale,
                synthetic_precip=synthetic is not None, dt=dt, history_interval_min=hist_min, steps=steps, nz=nz,
                initial=enc(initial), reset=reset.tolist(), expected=enc(wmax), max_prate_mm_h=prate,
                states=[{k: enc(v) for k, v in st.items()} for st in states],
                shapes={k: list(v.shape) for k, v in states[0].items()})


def heavy_score(f, rain, y, x):
    return float(rain[y:y + P, x:x + P].sum())


def calm_score(f, rain, y, x):
    return -float(np.hypot(f["U10"], f["V10"])[y:y + P, x:x + P].max())


def shear_score(f, rain, y, x):
    k = 10  # ~1 km AGL in 44-level WN grids
    u = 0.5 * (f["U"][k, y:y + P, x:x + P] + f["U"][k, y:y + P, x + 1:x + P + 1])
    v = 0.5 * (f["V"][k, y:y + P, x:x + P] + f["V"][k, y + 1:y + P + 1, x:x + P])
    return float((np.hypot(u, v) - np.hypot(f["U10"], f["V10"])[y:y + P, x:x + P]).min())


d06 = [f"{N148}/wrfout_d06_2026-10-19_{h:02d}:00:00" for h in (16, 17, 18, 19, 20)]
d03 = [f"{N148}/wrfout_d03_2026-10-19_{h:02d}:00:00" for h in (3, 4, 5)]
prod = [f"{PROD}/wrfout_d02_2026-07-26_{h:02d}:00:00" for h in (2, 3, 4, 5)]
ramp = np.linspace(0., 220., P * P, dtype=np.float32).reshape(P, P)
regimes = [
    regime("convective_148_d06", d06, heavy_score),
    regime("convective_148_d06_scaled2p5", d06, heavy_score, scale=2.5),
    regime("calm_148_d03_night", d03, calm_score),
    regime("stable_prod_d02_night", prod, shear_score),
    regime("stable_prod_d02_night_ramp_precip", prod, shear_score, synthetic=ramp),
]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
fixture = dict(
    sources={str(p.relative_to(WRF)): sha(p) for p in (AFWA, FUNCS, BIG)},
    blocks_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in
                   dict(zagl=zagl_block, precip=precip_block, severe=severe_block, uv_wind=uv_wind).items()},
    compiler=command, compiler_version=subprocess.run([FC, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
    provenance=__doc__, precision="WRF REAL=4", tolerance="bit-exact (CPU interpret, IEEE ops)",
    regimes=regimes)
target = ROOT / "proofs/v50_gust/afwa_oracle.json"
target.write_text(json.dumps(fixture, indent=1) + "\n")
print(json.dumps(dict(fixture=str(target), regimes=[(r["name"], r["patch"], [round(x, 1) for x in r["max_prate_mm_h"]], r["reset"]) for r in regimes])))
