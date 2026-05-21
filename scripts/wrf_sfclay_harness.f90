program wrf_sfclay_harness
  use module_sf_sfclay, only: sfclay1d, sfclayinit
  implicit none

  integer :: n, i, argc
  character(len=512) :: input_path, output_path
  real, parameter :: cp = 1004.0
  real, parameter :: g = 9.80665
  real, parameter :: r = 287.0
  real, parameter :: rocp = 287.0 / 1004.0
  real, parameter :: xlv = 2.5e6
  real, parameter :: svp1 = 0.6112
  real, parameter :: svp2 = 17.67
  real, parameter :: svp3 = 29.65
  real, parameter :: svpt0 = 273.15
  real, parameter :: ep1 = 0.608
  real, parameter :: ep2 = 0.622
  real, parameter :: karman = 0.40
  real, parameter :: eomeg = 7.2921e-5
  real, parameter :: stbolt = 5.670374419e-8
  real, parameter :: p1000mb = 100000.0

  real, allocatable :: ux(:), vx(:), t1d(:), qv1d(:), p1d(:), dz8w1d(:)
  real, allocatable :: psfc(:), chs(:), chs2(:), cqs2(:), cpm(:), pblh(:), rmol(:)
  real, allocatable :: znt(:), ust(:), mavail(:), zol(:), mol(:), regime(:), psim(:), psih(:), fm(:), fh(:)
  real, allocatable :: xland(:), hfx(:), qfx(:), tsk(:), u10(:), v10(:), th2(:), t2(:), q2(:)
  real, allocatable :: flhc(:), flqc(:), qgh(:), qsfc(:), lh(:), gz1oz0(:), wspd(:), br(:), dx(:)
  real, allocatable :: lakemask(:), ustm(:), ck(:), cka(:), cd(:), cda(:)

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_sfclay_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) n
  call allocate_arrays(n)
  do i = 1, n
     read(10,*) ux(i), vx(i), t1d(i), qv1d(i), p1d(i), dz8w1d(i), tsk(i), xland(i), znt(i), mavail(i), ust(i), mol(i)
     psfc(i) = p1d(i)
     rmol(i) = 0.0
     pblh(i) = 1000.0
     dx(i) = 3000.0
  end do
  close(10)

  call sfclayinit(.true.)
  call sfclay1d(1, ux, vx, t1d, qv1d, p1d, dz8w1d, &
       cp, g, rocp, r, xlv, psfc, chs, chs2, cqs2, cpm, pblh, rmol, &
       znt, ust, mavail, zol, mol, regime, psim, psih, fm, fh, &
       xland, hfx, qfx, tsk, u10, v10, th2, t2, q2, flhc, flqc, qgh, &
       qsfc, lh, gz1oz0, wspd, br, 1, dx, &
       svp1, svp2, svp3, svpt0, ep1, ep2, karman, eomeg, stbolt, &
       p1000mb, lakemask, &
       1, n + 1, 1, 2, 1, 2, &
       1, n, 1, 1, 1, 1, &
       1, n, 1, 1, 1, 1, &
       0, 0, 0, ustm, ck, cka, cd, cda)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I8)') n
  do i = 1, n
     write(20,'(14(ES24.16E3,1X))') ust(i), hfx(i), qfx(i), u10(i), v10(i), th2(i), t2(i), q2(i), &
          flhc(i), flqc(i), br(i), zol(i), fm(i), fh(i)
  end do
  close(20)

contains

  subroutine allocate_arrays(m)
    integer, intent(in) :: m
    allocate(ux(m), vx(m), t1d(m), qv1d(m), p1d(m), dz8w1d(m))
    allocate(psfc(m), chs(m), chs2(m), cqs2(m), cpm(m), pblh(m), rmol(m))
    allocate(znt(m), ust(m), mavail(m), zol(m), mol(m), regime(m), psim(m), psih(m), fm(m), fh(m))
    allocate(xland(m), hfx(m), qfx(m), tsk(m), u10(m), v10(m), th2(m), t2(m), q2(m))
    allocate(flhc(m), flqc(m), qgh(m), qsfc(m), lh(m), gz1oz0(m), wspd(m), br(m), dx(m))
    allocate(lakemask(m), ustm(m), ck(m), cka(m), cd(m), cda(m))
    chs = 0.0; chs2 = 0.0; cqs2 = 0.0; cpm = cp; rmol = 0.0
    zol = 0.0; regime = 0.0; psim = 0.0; psih = 0.0; fm = 0.0; fh = 0.0
    hfx = 0.0; qfx = 0.0; flhc = 0.0; flqc = 0.0; qgh = 0.0; qsfc = 0.0; lh = 0.0
    gz1oz0 = 0.0; wspd = 0.0; br = 0.0; lakemask = 0.0; ustm = 0.0
    ck = 0.0; cka = 0.0; cd = 0.0; cda = 0.0
  end subroutine allocate_arrays

end program wrf_sfclay_harness
