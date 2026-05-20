program wrf_thompson_harness
  use module_mp_thompson, only: thompson_init, mp_gt_driver
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 12
  integer, parameter :: ids = 1, ide = 2, jds = 1, jde = 2, kds = 1, kde = nz
  integer, parameter :: ims = 1, ime = nx, jms = 1, jme = ny, kms = 1, kme = nz
  integer, parameter :: its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = nz
  integer :: k, narg
  character(len=512) :: input_path, output_path
  real :: dt, rho
  real, dimension(ims:ime,kms:kme,jms:jme) :: qv, qc, qr, qi, qs, qg, ni, nr
  real, dimension(ims:ime,kms:kme,jms:jme) :: th, pii, p, w, dz
  real, dimension(ims:ime,kms:kme,jms:jme) :: refl_10cm, re_cloud, re_ice, re_snow
  real, dimension(ims:ime,kms:kme,jms:jme) :: hgt
  real, dimension(ims:ime,jms:jme) :: rainnc, rainncv, sr

  narg = command_argument_count()
  if (narg /= 2) then
    write(*,'(A)') 'usage: wrf_thompson_harness <input.dat> <output.dat>'
    stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  qv = 0.0
  qc = 0.0
  qr = 0.0
  qi = 0.0
  qs = 0.0
  qg = 0.0
  ni = 0.0
  nr = 0.0
  th = 0.0
  pii = 1.0
  p = 0.0
  w = 0.0
  ! Sedimentation is bypassed in the locally patched Thompson object; keep
  ! physical layer depths so non-sedimentation thermodynamic paths see sane dz.
  dz = 1000.0
  hgt = 0.0
  refl_10cm = 0.0
  re_cloud = 0.0
  re_ice = 0.0
  re_snow = 0.0
  rainnc = 0.0
  rainncv = 0.0
  sr = 0.0

  do k = kms, kme
    hgt(:,k,:) = real(k - 1) * 1000.0
  enddo

  call thompson_init( &
    hgt=hgt, dx=3000.0, dy=3000.0, is_start=.true., &
    ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
    ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
    its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  open(unit=11, file=trim(input_path), status='old', action='read')
  read(11,*) k, dt
  if (k /= nz) stop 3
  do k = kts, kte
    read(11,*) th(1,k,1), p(1,k,1), qv(1,k,1), qc(1,k,1), qr(1,k,1), &
      qi(1,k,1), qs(1,k,1), qg(1,k,1), ni(1,k,1), nr(1,k,1)
  enddo
  close(11)

  call mp_gt_driver( &
    qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, &
    th=th, pii=pii, p=p, w=w, dz=dz, dt_in=dt, itimestep=1, &
    rainnc=rainnc, rainncv=rainncv, sr=sr, &
    refl_10cm=refl_10cm, diagflag=.false., ke_diag=kte, do_radar_ref=0, &
    re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
    has_reqc=0, has_reqi=0, has_reqs=0, &
    ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
    ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
    its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  open(unit=12, file=trim(output_path), status='replace', action='write')
  write(12,'(I6,1X,ES24.16E3)') nz, dt
  do k = kts, kte
    rho = 0.622 * p(1,k,1) / (287.04 * th(1,k,1) * (qv(1,k,1) + 0.622))
    write(12,'(10(1X,ES24.16E3))') th(1,k,1), p(1,k,1), qv(1,k,1), qc(1,k,1), &
      qr(1,k,1), qi(1,k,1), qs(1,k,1), qg(1,k,1), ni(1,k,1), nr(1,k,1)
  enddo
  close(12)
end program wrf_thompson_harness

subroutine nl_get_force_read_thompson(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_force_read_thompson

subroutine nl_get_write_thompson_tables(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_write_thompson_tables

subroutine nl_get_write_thompson_mp38table(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_write_thompson_mp38table

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_decomp1d(nitems, first_item, last_item)
  implicit none
  integer, intent(in) :: nitems
  integer, intent(out) :: first_item, last_item
  first_item = 0
  last_item = nitems - 1
end subroutine wrf_dm_decomp1d

subroutine wrf_dm_bcast_integer(values, nitems)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_double(values, nitems)
  implicit none
  double precision, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_double

subroutine wrf_dm_bcast_bytes(values, nitems)
  implicit none
  character(len=1), intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_gatherv(values, nitems, first_item, last_item, item_size)
  implicit none
  double precision, intent(inout) :: values(*)
  integer, intent(in) :: nitems, first_item, last_item, item_size
end subroutine wrf_dm_gatherv

real function module_dm_wrf_dm_max_real(value)
  implicit none
  real, intent(in) :: value
  module_dm_wrf_dm_max_real = value
end function module_dm_wrf_dm_max_real

subroutine module_timing_start_timing(label)
  implicit none
  character(len=*), intent(in) :: label
end subroutine module_timing_start_timing

subroutine module_timing_end_timing(label)
  implicit none
  character(len=*), intent(in) :: label
end subroutine module_timing_end_timing

subroutine wrf_abort()
  implicit none
  stop 99
end subroutine wrf_abort

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine set_wrf_debug_level(level)
  implicit none
  integer, intent(in) :: level
end subroutine set_wrf_debug_level
