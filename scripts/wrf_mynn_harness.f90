program wrf_mynn_harness
  use module_bl_mynnedmf_common, only: kind_phys, p608, p1000mb, rcp
  use module_bl_mynnedmf, only: get_pblh, mym_turbulence, mym_predict, &
       mynn_tendencies, retrieve_exchange_coeffs
  implicit none

  integer :: nz, k, argc, kpbl
  real(kind_phys) :: dt, dx, xland, closure, rmol, flt, fltv, flq, flqv, flqc
  real(kind_phys) :: ust, wind, psfc, wspd, uoce, voce, psig_bl, psig_shcu, pblh
  real(kind_phys), parameter :: bulk_cd = 1.3e-3_kind_phys
  real(kind_phys), parameter :: bulk_ch = 1.2e-3_kind_phys
  real(kind_phys), parameter :: bulk_cq = 1.2e-3_kind_phys
  real(kind_phys), parameter :: min_wind = 0.2_kind_phys
  character(len=512) :: input_path, output_path

  real(kind_phys), allocatable :: u(:), v(:), w(:), theta(:), qv(:), tke(:)
  real(kind_phys), allocatable :: p(:), rho(:), dz(:), zw(:), qke(:)
  real(kind_phys), allocatable :: thl(:), thv(:), thlv(:), tk(:), exner(:)
  real(kind_phys), allocatable :: ql(:), qw(:), qc(:), qi(:), qs(:), qnc(:), qni(:)
  real(kind_phys), allocatable :: sqv(:), sqc(:), sqi(:), sqs(:), sqw(:)
  real(kind_phys), allocatable :: qnwfa(:), qnifa(:), qnbca(:), ozone(:)
  real(kind_phys), allocatable :: tsq(:), qsq(:), cov(:), vt(:), vq(:)
  real(kind_phys), allocatable :: sh(:), sm(:), el(:), dfm(:), dfh(:), dfq(:)
  real(kind_phys), allocatable :: tcd(:), qcd(:), pdk(:), pdt(:), pdq(:), pdc(:)
  real(kind_phys), allocatable :: qwt(:), qshear(:), qbuoy(:), qdiss(:)
  real(kind_phys), allocatable :: cldfra(:), edmf_w(:), edmf_a(:), edmf_w_dd(:), edmf_a_dd(:)
  real(kind_phys), allocatable :: tkeprod_dn(:), tkeprod_up(:), pattern_spp(:), diss_heat(:)
  real(kind_phys), allocatable :: km(:), kh(:)
  real(kind_phys), allocatable :: du(:), dv(:), dth(:), dqv(:), dqc(:), dqi(:), dqs(:)
  real(kind_phys), allocatable :: dqnc(:), dqni(:), dqnwfa(:), dqnifa(:), dqnbca(:), dozone(:)
  real(kind_phys), allocatable :: sub_thl(:), sub_sqv(:), sub_u(:), sub_v(:)
  real(kind_phys), allocatable :: det_thl(:), det_sqv(:), det_sqc(:), det_u(:), det_v(:)
  real(kind_phys), allocatable :: s_aw1(:), s_awthl(:), s_awqt(:), s_awqv(:), s_awqc(:)
  real(kind_phys), allocatable :: s_awu(:), s_awv(:), s_awqnc(:), s_awqni(:)
  real(kind_phys), allocatable :: s_awqnwfa(:), s_awqnifa(:), s_awqnbca(:)
  real(kind_phys), allocatable :: sd_aw1(:), sd_awthl(:), sd_awqt(:), sd_awqv(:), sd_awqc(:)
  real(kind_phys), allocatable :: sd_awqi(:), sd_awqnc(:), sd_awqni(:)
  real(kind_phys), allocatable :: sd_awqnwfa(:), sd_awqnifa(:), sd_awu(:), sd_awv(:)

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_mynn_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) nz, dt
  call allocate_column(nz)
  do k = 1, nz
     read(10,*) u(k), v(k), w(k), theta(k), qv(k), tke(k), p(k), rho(k), dz(k)
  end do
  close(10)

  zw(1) = 0.0_kind_phys
  do k = 2, nz + 1
     zw(k) = zw(k-1) + dz(k-1)
  end do

  qke = 2.0_kind_phys * max(tke, 0.5_kind_phys * 1.0e-5_kind_phys)
  qv = max(qv, 0.0_kind_phys)
  thl = theta
  thv = theta * (1.0_kind_phys + p608 * qv)
  thlv = thv
  qw = qv
  sqv = qv
  sqw = qv
  tk = theta * (p / p1000mb) ** rcp
  exner = (p / p1000mb) ** rcp

  xland = 2.0_kind_phys
  dx = 3000.0_kind_phys
  closure = 2.5_kind_phys
  rmol = 0.0_kind_phys
  psig_bl = 1.0_kind_phys
  psig_shcu = 1.0_kind_phys
  psfc = p(1)
  wind = max(sqrt(u(1) * u(1) + v(1) * v(1)), min_wind)
  wspd = wind
  ust = sqrt(bulk_cd) * wind
  flt = bulk_ch * wind * 0.25_kind_phys
  flq = bulk_cq * wind * 1.0e-4_kind_phys
  flqv = flq
  flqc = 0.0_kind_phys
  fltv = (1.0_kind_phys + p608 * qv(1)) * flt + p608 * theta(1) * flq
  uoce = 0.0_kind_phys
  voce = 0.0_kind_phys

  call get_pblh(1, nz, pblh, thv, qke, zw, dz, xland, kpbl)

  call mym_turbulence(1, nz, xland, closure, dz, dx, zw, &
       u, v, thl, thv, thlv, ql, qw, qke, tsq, qsq, cov, vt, vq, &
       rmol, flt, fltv, flq, pblh, theta, sh, sm, el, dfm, dfh, dfq, &
       tcd, qcd, pdk, pdt, pdq, pdc, qwt, qshear, qbuoy, qdiss, &
       1, psig_bl, psig_shcu, cldfra, 2, edmf_w, edmf_a, edmf_w_dd, edmf_a_dd, &
       tkeprod_dn, tkeprod_up, 0, pattern_spp)

  call mym_predict(1, nz, closure, dt, dz, ust, flt, flq, &
       1.0_kind_phys, 1.0_kind_phys, el, dfq, rho, pdk, pdt, pdq, pdc, &
       qke, tsq, qsq, cov, s_aw1, s_awqv, 0, qwt, qdiss, 1)

  call mynn_tendencies(1, nz, 1, dt, dz, zw(1:nz), xland, rho, &
       u, v, theta, tk, qv, qc, qi, qs, qnc, qni, psfc, p, exner, &
       thl, sqv, sqc, sqi, sqs, sqw, qnwfa, qnifa, qnbca, ozone, &
       ust, flt, flq, flqv, flqc, wspd, uoce, voce, tsq, qsq, cov, &
       tcd, qcd, dfm, dfh, dfq, du, dv, dth, dqv, dqc, dqi, dqs, dqnc, dqni, &
       dqnwfa, dqnifa, dqnbca, dozone, diss_heat, s_aw1, s_awthl, s_awqt, &
       s_awqv, s_awqc, s_awu, s_awv, s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, &
       s_awqnbca, sd_aw1, sd_awthl, sd_awqt, sd_awqv, sd_awqc, sd_awqi, &
       sd_awqnc, sd_awqni, sd_awqnwfa, sd_awqnifa, sd_awu, sd_awv, &
       sub_thl, sub_sqv, sub_u, sub_v, det_thl, det_sqv, det_sqc, det_u, det_v, &
       .false., .false., .false., .false., .false., .false., .false., .false., &
       .false., cldfra, 0, 0, 0, 0, 0)

  call retrieve_exchange_coeffs(1, nz, dfm, dfh, dz, km, kh)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I6,1X,ES24.16E3)') nz, dt
  do k = 1, nz
     write(20,'(18(ES24.16E3,1X))') &
          u(k) + du(k) * dt, v(k) + dv(k) * dt, w(k), &
          theta(k) + dth(k) * dt, max(qv(k) + dqv(k) * dt, 0.0_kind_phys), &
          0.5_kind_phys * qke(k), p(k), rho(k), dz(k), km(k), kh(k), el(k), &
          qshear(k), qbuoy(k), qdiss(k), qwt(k), flt, flq
  end do
  close(20)

contains

  subroutine allocate_column(n)
    integer, intent(in) :: n
    allocate(u(n), v(n), w(n), theta(n), qv(n), tke(n), p(n), rho(n), dz(n), zw(n+1), qke(n))
    allocate(thl(n), thv(n), thlv(n), tk(n), exner(n), ql(n), qw(n), qc(n), qi(n), qs(n), qnc(n), qni(n))
    allocate(sqv(n), sqc(n), sqi(n), sqs(n), sqw(n), qnwfa(n), qnifa(n), qnbca(n), ozone(n))
    allocate(tsq(n), qsq(n), cov(n), vt(n), vq(n), sh(n), sm(n), el(n), dfm(n), dfh(n), dfq(n))
    allocate(tcd(n), qcd(n), pdk(n), pdt(n), pdq(n), pdc(n), qwt(n), qshear(n), qbuoy(n), qdiss(n))
    allocate(cldfra(n), edmf_w(n), edmf_a(n), edmf_w_dd(n), edmf_a_dd(n), tkeprod_dn(n), tkeprod_up(n))
    allocate(pattern_spp(n), diss_heat(n), km(n), kh(n), du(n), dv(n), dth(n), dqv(n), dqc(n), dqi(n), dqs(n))
    allocate(dqnc(n), dqni(n), dqnwfa(n), dqnifa(n), dqnbca(n), dozone(n))
    allocate(sub_thl(n), sub_sqv(n), sub_u(n), sub_v(n), det_thl(n), det_sqv(n), det_sqc(n), det_u(n), det_v(n))
    allocate(s_aw1(n+1), s_awthl(n+1), s_awqt(n+1), s_awqv(n+1), s_awqc(n+1), s_awu(n+1), s_awv(n+1))
    allocate(s_awqnc(n+1), s_awqni(n+1), s_awqnwfa(n+1), s_awqnifa(n+1), s_awqnbca(n+1))
    allocate(sd_aw1(n+1), sd_awthl(n+1), sd_awqt(n+1), sd_awqv(n+1), sd_awqc(n+1), sd_awqi(n+1))
    allocate(sd_awqnc(n+1), sd_awqni(n+1), sd_awqnwfa(n+1), sd_awqnifa(n+1), sd_awu(n+1), sd_awv(n+1))
    ql = 0.0_kind_phys; qc = 0.0_kind_phys; qi = 0.0_kind_phys; qs = 0.0_kind_phys
    qnc = 0.0_kind_phys; qni = 0.0_kind_phys; sqc = 0.0_kind_phys; sqi = 0.0_kind_phys; sqs = 0.0_kind_phys
    qnwfa = 0.0_kind_phys; qnifa = 0.0_kind_phys; qnbca = 0.0_kind_phys; ozone = 0.0_kind_phys
    tsq = 0.0_kind_phys; qsq = 0.0_kind_phys; cov = 0.0_kind_phys; vt = 0.0_kind_phys; vq = 0.0_kind_phys
    sh = 0.0_kind_phys; sm = 0.0_kind_phys; el = 0.0_kind_phys; dfm = 0.0_kind_phys; dfh = 0.0_kind_phys; dfq = 0.0_kind_phys
    tcd = 0.0_kind_phys; qcd = 0.0_kind_phys; pdk = 0.0_kind_phys; pdt = 0.0_kind_phys; pdq = 0.0_kind_phys; pdc = 0.0_kind_phys
    qwt = 0.0_kind_phys; qshear = 0.0_kind_phys; qbuoy = 0.0_kind_phys; qdiss = 0.0_kind_phys
    cldfra = 0.0_kind_phys; edmf_w = 0.0_kind_phys; edmf_a = 0.0_kind_phys
    edmf_w_dd = 0.0_kind_phys; edmf_a_dd = 0.0_kind_phys; tkeprod_dn = 0.0_kind_phys; tkeprod_up = 0.0_kind_phys
    pattern_spp = 0.0_kind_phys; diss_heat = 0.0_kind_phys; km = 0.0_kind_phys; kh = 0.0_kind_phys
    du = 0.0_kind_phys; dv = 0.0_kind_phys; dth = 0.0_kind_phys; dqv = 0.0_kind_phys
    dqc = 0.0_kind_phys; dqi = 0.0_kind_phys; dqs = 0.0_kind_phys; dqnc = 0.0_kind_phys; dqni = 0.0_kind_phys
    dqnwfa = 0.0_kind_phys; dqnifa = 0.0_kind_phys; dqnbca = 0.0_kind_phys; dozone = 0.0_kind_phys
    sub_thl = 0.0_kind_phys; sub_sqv = 0.0_kind_phys; sub_u = 0.0_kind_phys; sub_v = 0.0_kind_phys
    det_thl = 0.0_kind_phys; det_sqv = 0.0_kind_phys; det_sqc = 0.0_kind_phys; det_u = 0.0_kind_phys; det_v = 0.0_kind_phys
    s_aw1 = 0.0_kind_phys; s_awthl = 0.0_kind_phys; s_awqt = 0.0_kind_phys; s_awqv = 0.0_kind_phys; s_awqc = 0.0_kind_phys
    s_awu = 0.0_kind_phys; s_awv = 0.0_kind_phys; s_awqnc = 0.0_kind_phys; s_awqni = 0.0_kind_phys
    s_awqnwfa = 0.0_kind_phys; s_awqnifa = 0.0_kind_phys; s_awqnbca = 0.0_kind_phys
    sd_aw1 = 0.0_kind_phys; sd_awthl = 0.0_kind_phys; sd_awqt = 0.0_kind_phys; sd_awqv = 0.0_kind_phys
    sd_awqc = 0.0_kind_phys; sd_awqi = 0.0_kind_phys; sd_awqnc = 0.0_kind_phys; sd_awqni = 0.0_kind_phys
    sd_awqnwfa = 0.0_kind_phys; sd_awqnifa = 0.0_kind_phys; sd_awu = 0.0_kind_phys; sd_awv = 0.0_kind_phys
  end subroutine allocate_column
end program wrf_mynn_harness
