program wrf_advance_uv_driver
  ! Standalone driver linking the UNMODIFIED WRF dyn_em advance_uv subroutine
  ! (module_small_step_em.F).  It reads a single-stage savepoint that supplies
  ! BOTH the live geopotential ``ph`` (= grid%ph_2) AND the STAGE-CONSTANT
  ! ``php`` (= grid%php from calc_php in rk_step_prep) as separate INTENT(IN)
  ! arrays, advances u/v one substep, and writes the resulting u/v back.  This
  ! is the WRF oracle for the v0.4.0 r5 split-explicit php-freeze fix: the JAX
  ! advance_uv_wrf must reproduce these u/v when it uses the frozen php for the
  ! 4th PGF term and the live ph for the first-3-terms gradient.
  use iso_fortran_env, only: int32
  use module_configure, only: grid_config_rec_type
  use module_small_step_em, only: advance_uv
  implicit none

  character(len=1024) :: input_path, output_path
  integer(int32) :: nx_i4, ny_i4, nz_i4
  integer :: nx, ny, nz, xd, yd, zd, unit_in, unit_out
  integer :: ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: spec_zone, real_bytes
  logical :: non_hydrostatic, top_lid
  real :: rdx, rdy, dts, cf1, cf2, cf3, emdiv
  type(grid_config_rec_type) :: config_flags

  real, allocatable :: u(:,:,:), ru_tend(:,:,:), v(:,:,:), rv_tend(:,:,:)
  real, allocatable :: p(:,:,:), pb(:,:,:), ph(:,:,:), php(:,:,:)
  real, allocatable :: alt(:,:,:), al(:,:,:), cqu(:,:,:), cqv(:,:,:)
  real, allocatable :: mu(:,:), muu(:,:), muv(:,:), mudf(:,:)
  real, allocatable :: msfux(:,:), msfuy(:,:), msfvx(:,:), msfvx_inv(:,:), msfvy(:,:)
  real, allocatable :: c1h(:), c2h(:), c1f(:), c2f(:), c3h(:), c4h(:), c3f(:), c4f(:)
  real, allocatable :: fnm(:), fnp(:), rdnw(:)

  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  if (len_trim(input_path) == 0 .or. len_trim(output_path) == 0) then
     error stop "usage: wrf_advance_uv_driver INPUT.bin OUTPUT.bin"
  end if

  open(newunit=unit_in, file=trim(input_path), form="unformatted", access="stream", &
       status="old", action="read", convert="little_endian")
  read(unit_in) nx_i4, ny_i4, nz_i4
  nx = int(nx_i4)
  ny = int(ny_i4)
  nz = int(nz_i4)
  xd = nx + 1
  yd = ny + 1
  zd = nz + 1

  allocate(u(xd,zd,yd), ru_tend(xd,zd,yd), v(xd,zd,yd), rv_tend(xd,zd,yd))
  allocate(p(xd,zd,yd), pb(xd,zd,yd), ph(xd,zd,yd), php(xd,zd,yd))
  allocate(alt(xd,zd,yd), al(xd,zd,yd), cqu(xd,zd,yd), cqv(xd,zd,yd))
  allocate(mu(xd,yd), muu(xd,yd), muv(xd,yd), mudf(xd,yd))
  allocate(msfux(xd,yd), msfuy(xd,yd), msfvx(xd,yd), msfvx_inv(xd,yd), msfvy(xd,yd))
  allocate(c1h(zd), c2h(zd), c1f(zd), c2f(zd), c3h(zd), c4h(zd), c3f(zd), c4f(zd))
  allocate(fnm(zd), fnp(zd), rdnw(zd))

  read(unit_in) rdx, rdy, dts, cf1, cf2, cf3, emdiv
  read(unit_in) u, ru_tend, v, rv_tend
  read(unit_in) p, pb, ph, php, alt, al, cqu, cqv
  read(unit_in) mu, muu, muv, mudf
  read(unit_in) msfux, msfuy, msfvx, msfvx_inv, msfvy
  read(unit_in) c1h, c2h, c1f, c2f, c3h, c4h, c3f, c4f
  read(unit_in) fnm, fnp, rdnw
  close(unit_in)

  config_flags%periodic_x = .false.
  config_flags%specified = .false.
  config_flags%nested = .false.
  config_flags%open_xs = .false.
  config_flags%open_xe = .false.
  config_flags%open_ys = .false.
  config_flags%open_ye = .false.
  config_flags%symmetric_xs = .false.
  config_flags%symmetric_xe = .false.
  config_flags%symmetric_ys = .false.
  config_flags%symmetric_ye = .false.
  config_flags%polar = .false.
  non_hydrostatic = .true.
  top_lid = .false.
  spec_zone = 1

  ids = 1; ide = nx + 1
  jds = 1; jde = ny + 1
  kds = 1; kde = nz + 1
  ims = 1; ime = nx + 1
  jms = 1; jme = ny + 1
  kms = 1; kme = nz + 1
  its = 1; ite = nx
  jts = 1; jte = ny
  kts = 1; kte = nz + 1

  call advance_uv ( u, ru_tend, v, rv_tend,        &
                    p, pb,                         &
                    ph, php, alt, al, mu,          &
                    muu, cqu, muv, cqv, mudf,      &
                    c1h, c2h, c1f, c2f,            &
                    c3h, c4h, c3f, c4f,            &
                    msfux, msfuy, msfvx,           &
                    msfvx_inv, msfvy,              &
                    rdx, rdy, dts,                 &
                    cf1, cf2, cf3, fnm, fnp,       &
                    emdiv,                         &
                    rdnw, config_flags, spec_zone, &
                    non_hydrostatic, top_lid,      &
                    ids, ide, jds, jde, kds, kde,  &
                    ims, ime, jms, jme, kms, kme,  &
                    its, ite, jts, jte, kts, kte  )

  real_bytes = storage_size(1.0) / 8
  open(newunit=unit_out, file=trim(output_path), form="unformatted", access="stream", &
       status="replace", action="write", convert="little_endian")
  write(unit_out) nx_i4, ny_i4, nz_i4, int(real_bytes, int32)
  write(unit_out) u, v
  close(unit_out)
end program wrf_advance_uv_driver
