program wrf_advance_mu_t_driver
  use iso_fortran_env, only: int32
  use module_configure, only: grid_config_rec_type
  use module_small_step_em, only: advance_mu_t
  implicit none

  character(len=1024) :: input_path, output_path
  integer(int32) :: nx_i4, ny_i4, nz_i4
  integer :: nx, ny, nz, xd, yd, zd, unit_in, unit_out
  integer :: ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: step, real_bytes
  real :: rdx, rdy, dts, epssm
  type(grid_config_rec_type) :: config_flags

  real, allocatable :: ww(:,:,:), ww_1(:,:,:), u(:,:,:), u_1(:,:,:)
  real, allocatable :: v(:,:,:), v_1(:,:,:), t(:,:,:), t_1(:,:,:)
  real, allocatable :: t_ave(:,:,:), ft(:,:,:), uam(:,:,:), vam(:,:,:), wwam(:,:,:)
  real, allocatable :: mu(:,:), mut(:,:), muave(:,:), muts(:,:), muu(:,:), muv(:,:), mudf(:,:)
  real, allocatable :: mu_tend(:,:), msfux(:,:), msfuy(:,:), msfvx(:,:), msfvx_inv(:,:)
  real, allocatable :: msfvy(:,:), msftx(:,:), msfty(:,:)
  real, allocatable :: c1h(:), c2h(:), c1f(:), c2f(:), c3h(:), c4h(:), c3f(:), c4f(:)
  real, allocatable :: dnw(:), fnm(:), fnp(:), rdnw(:)

  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  if (len_trim(input_path) == 0 .or. len_trim(output_path) == 0) then
     error stop "usage: wrf_advance_mu_t_driver INPUT.bin OUTPUT.bin"
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

  allocate(ww(xd,zd,yd), ww_1(xd,zd,yd), u(xd,zd,yd), u_1(xd,zd,yd))
  allocate(v(xd,zd,yd), v_1(xd,zd,yd), t(xd,zd,yd), t_1(xd,zd,yd))
  allocate(t_ave(xd,zd,yd), ft(xd,zd,yd), uam(xd,zd,yd), vam(xd,zd,yd), wwam(xd,zd,yd))
  allocate(mu(xd,yd), mut(xd,yd), muave(xd,yd), muts(xd,yd), muu(xd,yd), muv(xd,yd), mudf(xd,yd))
  allocate(mu_tend(xd,yd), msfux(xd,yd), msfuy(xd,yd), msfvx(xd,yd), msfvx_inv(xd,yd))
  allocate(msfvy(xd,yd), msftx(xd,yd), msfty(xd,yd))
  allocate(c1h(zd), c2h(zd), c1f(zd), c2f(zd), c3h(zd), c4h(zd), c3f(zd), c4f(zd))
  allocate(dnw(zd), fnm(zd), fnp(zd), rdnw(zd))

  read(unit_in) rdx, rdy, dts, epssm
  read(unit_in) ww, ww_1, u, u_1, v, v_1
  read(unit_in) mu, mut, muave, muts, muu, muv, mudf
  read(unit_in) c1h, c2h, c1f, c2f, c3h, c4h, c3f, c4f
  read(unit_in) uam, vam, wwam, t, t_1, t_ave, ft, mu_tend
  read(unit_in) dnw, fnm, fnp, rdnw
  read(unit_in) msfux, msfuy, msfvx, msfvx_inv, msfvy, msftx, msfty
  close(unit_in)

  config_flags%periodic_x = .false.
  config_flags%specified = .true.
  config_flags%nested = .false.
  step = 1
  ids = 1; ide = nx + 1
  jds = 1; jde = ny + 1
  kds = 1; kde = nz + 1
  ims = 1; ime = nx + 1
  jms = 1; jme = ny + 1
  kms = 1; kme = nz + 1
  its = 1; ite = nx
  jts = 1; jte = ny
  kts = 1; kte = nz + 1

  call advance_mu_t(ww, ww_1, u, u_1, v, v_1, &
       mu, mut, muave, muts, muu, muv, mudf, &
       c1h, c2h, c1f, c2f, c3h, c4h, c3f, c4f, &
       uam, vam, wwam, t, t_1, t_ave, ft, mu_tend, &
       rdx, rdy, dts, epssm, dnw, fnm, fnp, rdnw, &
       msfux, msfuy, msfvx, msfvx_inv, msfvy, msftx, msfty, &
       step, config_flags, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte)

  real_bytes = storage_size(1.0) / 8
  open(newunit=unit_out, file=trim(output_path), form="unformatted", access="stream", &
       status="replace", action="write", convert="little_endian")
  write(unit_out) nx_i4, ny_i4, nz_i4, int(real_bytes, int32)
  write(unit_out) mu, mudf, muts, muave, ww, t, t_ave
  close(unit_out)
end program wrf_advance_mu_t_driver
