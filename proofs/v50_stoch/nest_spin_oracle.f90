! Original-WRF nested d01 RAND initialization (v50-stoch, built by proofs/u2_stoch/build_oracle.py --driver).
! start_em.F:316-339 -> INITIALIZE_STOCH (module_stoch.F:145-270) runs once from main BEFORE set_tiles
! (num_tiles=0: the ten 'R' spins have no tile body) and again from med_nest_initial
! (share/mediation_integrate.F:845) once per d01 child, now with tiles. Each call re-seeds (rand_seed)
! and re-runs SETUP_RAND_PERTURB, which keeps SPFORCS/SPFORCC. Records: state after round r=0..3,
! then d01's first RK 'T' update (module_first_rk_step_part2.F:240-260).
program oracle
use module_stoch
use module_configure
use module_domain
implicit none
integer,parameter::nx=18,ny=16,nz=5
type(grid_config_rec_type)::config
type(domain)::grid
integer::seed_dim,seed(8),n,r,ikmax,ilmax
real::amp(nx,ny),s(nx,ny),c(nx,ny),alpha
real::vc(nx,nz,ny),vs(nx,nz,ny),va(nz)
real::field(nx,nz,ny)
config%start_year=2026;config%start_month=2;config%start_day=28;config%start_hour=0;config%nens=3
grid%i_start=1;grid%i_end=nx;grid%j_start=1;grid%j_end=ny
allocate(grid%rand_real(nx,nz,ny),grid%rand_imag(nx,nz,ny))
grid%rand_real=0.;grid%rand_imag=0.
call random_seed(size=seed_dim)
if(seed_dim/=8)stop 8
open(20,file='oracle.bin',access='stream',form='unformatted',status='replace')
vc=1.;vs=0.;va=1.
s=0.;c=0.;field=0.
do r=0,3
grid%num_tiles=1
if(r==0)grid%num_tiles=0
call rand_seed(config,17,seed,1,seed_dim)
call setup_rand_perturb('R',0,.false.,amp,c,s,alpha,vc,vs,va, &
  1,1000000,1,1000000,ikmax,ilmax,54,9000.,9000.,0.5,15000.,21600.,1.e-6,10800.,-1.83, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
do n=1,10
call rand_pert_update(grid,'R',s,c,amp,alpha, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nz, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
  nz,nz,nz,nz,.false.,seed,seed_dim,9000.,9000.,0,field,2.,.5,vc,vs,va)
enddo
write(20)s,c,field,seed
enddo
grid%num_tiles=1
call rand_pert_update(grid,'T',s,c,amp,alpha, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nz, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
  nz,nz,nz,nz,.false.,seed,seed_dim,9000.,9000.,0,field,2.,.5,vc,vs,va)
write(20)s,c,field,seed
close(20)
end program
