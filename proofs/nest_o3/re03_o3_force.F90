! RE03: pristine WRF V4.7.1 nest ozone = force-down of the parent's o3rad (Registry o3rad rdf=(p2c)).
!   d01: o3input=2 (module_radiation_driver.F:1803, id 1 only): oznini(XLAT) -> ozn_time_int(julday, julian) -> ozn_p_int(p_hyd)
!   d02 = interp_fcn(d01 o3rad), d03 = interp_fcn(d02 o3rad): inc/nest_forcedown_interp.inc:265-280 p2c -> interp_fcn
!   (share/interp_fcn.F, interp_method_type default SINT, shw 2, imask_nostag = 1 (mediation_integrate.F:893), mass point).
! The CD argument is a parent sub-window around the nest (patch +-3, memory +-3 more), standing in for WRF's intermediate
! domain: interp_fcn_sint reads only cells around the masked nest footprint.
! I/O little-endian stream (library built -fconvert=big-endian).
program re03_o3_force
  use module_ra_cam_support, only: oznini
  use module_radiation_driver, only: ozn_time_int, ozn_p_int
  implicit none
  external interp_fcn
  integer, parameter :: levsiz = 59, nmon = 12
  integer :: hdr(12), nx1, ny1, nz, nx2, ny2, nx3, ny3, ip2, jp2, ip3, jp3, ratio, julday
  real :: julian
  real, allocatable :: lat1(:,:), p1(:,:,:), o31(:,:,:), o32(:,:,:), o33(:,:,:)
  real, allocatable :: ozmixm(:,:,:,:), ozmixt(:,:,:), pin(:)
  character(len=256) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', convert='native')
  read(10) hdr
  nx1 = hdr(1); ny1 = hdr(2); nz = hdr(3); nx2 = hdr(4); ny2 = hdr(5); nx3 = hdr(6); ny3 = hdr(7)
  ip2 = hdr(8); jp2 = hdr(9); ip3 = hdr(10); jp3 = hdr(11); ratio = hdr(12)
  read(10) julday, julian
  allocate(lat1(nx1+1, ny1+1), p1(nx1+1, nz+1, ny1+1), o31(nx1+1, nz+1, ny1+1))
  allocate(o32(nx2+1, nz+1, ny2+1), o33(nx3+1, nz+1, ny3+1))
  allocate(ozmixm(nx1+1, levsiz, ny1+1, nmon), ozmixt(nx1+1, levsiz, ny1+1), pin(levsiz))
  lat1 = 0.; p1 = 0.; o31 = 0.; o32 = 0.; o33 = 0.
  read(10) lat1(1:nx1, 1:ny1)
  read(10) p1(1:nx1, 1:nz, 1:ny1)
  close(10)

  call oznini(ozmixm, pin, levsiz, nmon, lat1, 1, nx1+1, 1, ny1+1, 1, nz+1, 1, nx1+1, 1, ny1+1, 1, nz+1, &
              1, nx1, 1, ny1, 1, nz)
  call ozn_time_int(julday, julian, ozmixm, ozmixt, levsiz, nmon, 1, nx1+1, 1, ny1+1, 1, nz+1, &
                    1, nx1+1, 1, ny1+1, 1, nz+1, 1, nx1, 1, ny1, 1, nz)
  call ozn_p_int(p1, pin, levsiz, ozmixt, o31, 1, nx1+1, 1, ny1+1, 1, nz+1, 1, nx1+1, 1, ny1+1, 1, nz+1, &
                 1, nx1, 1, ny1, 1, nz)
  call force(o31, nx1, ny1, o32, nx2, ny2, ip2, jp2)
  call force(o32, nx2, ny2, o33, nx3, ny3, ip3, jp3)

  open(20, file=trim(fout), access='stream', form='unformatted', status='replace', convert='native')
  write(20) o31(1:nx1, 1:nz, 1:ny1), o32(1:nx2, 1:nz, 1:ny2), o33(1:nx3, 1:nz, 1:ny3)
  close(20)

contains

  subroutine force(full, nxp, nyp, nfld, nxc, nyc, ipos, jpos)
    integer, intent(in) :: nxp, nyp, nxc, nyc, ipos, jpos
    real, intent(in) :: full(nxp+1, nz+1, nyp+1)
    real, intent(inout) :: nfld(nxc+1, nz+1, nyc+1)
    integer :: cits, cite, cjts, cjte, cims, cime, cjms, cjme
    integer, allocatable :: imask(:,:)
    real, allocatable :: cfld(:,:,:)
    cits = ipos - 3; cite = ipos + (nxc - 1) / ratio + 3
    cjts = jpos - 3; cjte = jpos + (nyc - 1) / ratio + 3
    cims = cits - 3; cime = cite + 3; cjms = cjts - 3; cjme = cjte + 3
    if (cims < 1 .or. cjms < 1 .or. cime > nxp .or. cjme > nyp) stop 'parent window leaves the parent interior'
    allocate(cfld(cims:cime, 1:nz+1, cjms:cjme), imask(1:nxc+1, 1:nyc+1))
    cfld = full(cims:cime, :, cjms:cjme)
    imask = 1
    call interp_fcn(cfld, 1, nxp+1, 1, nz+1, 1, nyp+1, cims, cime, 1, nz+1, cjms, cjme, cits, cite, 1, nz, cjts, cjte, &
                    nfld, 1, nxc+1, 1, nz+1, 1, nyc+1, 1, nxc+1, 1, nz+1, 1, nyc+1, 1, nxc, 1, nz, 1, nyc, &
                    2, imask, .false., .false., ipos, jpos, ratio, ratio)
  end subroutine force
end program re03_o3_force
