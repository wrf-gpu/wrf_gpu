program ferrier_oracle_driver
  use module_mp_fer_hires, only: fer_hires_init, FER_HIRES
  use module_mp_etanew, only: ETANEWinit, ETAMP_NEW
  implicit none

  integer, parameter :: kx = 40
  integer, parameter :: nxpvs = 7501
  integer, parameter :: ids = 1, ide = 2, jds = 1, jde = 2, kds = 1, kde = kx + 1
  integer, parameter :: ims = 1, ime = 1, jms = 1, jme = 1, kms = 1, kme = kx
  integer, parameter :: its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = kx

  character(len=32) :: scheme, arg
  integer :: cid
  real :: dt, dx, dy, gsmdt
  real :: dz8w(ims:ime,kms:kme,jms:jme)
  real :: rho_phy(ims:ime,kms:kme,jms:jme)
  real :: p_phy(ims:ime,kms:kme,jms:jme)
  real :: pi_phy(ims:ime,kms:kme,jms:jme)
  real :: th_phy(ims:ime,kms:kme,jms:jme)
  real :: qv(ims:ime,kms:kme,jms:jme)
  real :: qt(ims:ime,kms:kme,jms:jme)
  real :: qc(ims:ime,kms:kme,jms:jme)
  real :: qr(ims:ime,kms:kme,jms:jme)
  real :: qi(ims:ime,kms:kme,jms:jme)
  real :: qs(ims:ime,kms:kme,jms:jme)
  real :: f_ice_phy(ims:ime,kms:kme,jms:jme)
  real :: f_rain_phy(ims:ime,kms:kme,jms:jme)
  real :: f_rimef_phy(ims:ime,kms:kme,jms:jme)
  real :: th_in(ims:ime,kms:kme,jms:jme)
  real :: qv_in(ims:ime,kms:kme,jms:jme)
  real :: qt_in(ims:ime,kms:kme,jms:jme)
  real :: qc_in(ims:ime,kms:kme,jms:jme)
  real :: qr_in(ims:ime,kms:kme,jms:jme)
  real :: qi_in(ims:ime,kms:kme,jms:jme)
  real :: qs_in(ims:ime,kms:kme,jms:jme)
  real :: rainnc(ims:ime,jms:jme), rainncv(ims:ime,jms:jme), sr(ims:ime,jms:jme)
  integer :: lowlyr(ims:ime,jms:jme)
  real :: mp_restart_state(256), tbpvs_state(nxpvs), tbpvs0_state(nxpvs)

  call get_command_argument(1, scheme)
  call get_command_argument(2, arg)
  if (len_trim(scheme) == 0 .or. len_trim(arg) == 0) then
    print *, 'usage: ferrier_oracle_driver mp5|mp95 case_id'
    stop 2
  end if
  read(arg, *) cid

  dt = 60.0
  gsmdt = dt
  dx = 3000.0
  dy = 3000.0
  rainnc = 0.0
  rainncv = 0.0
  sr = 0.0
  lowlyr = 1
  f_ice_phy = 0.0
  f_rain_phy = 0.0
  f_rimef_phy = 1.0
  mp_restart_state = 0.0
  tbpvs_state = 0.0
  tbpvs0_state = 0.0

  if (trim(scheme) == 'mp5') then
    call fer_hires_init(gsmdt, dt, dx, dy, lowlyr, .false., .true., &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte, f_ice_phy, f_rain_phy, f_rimef_phy)
  else if (trim(scheme) == 'mp95') then
    call ETANEWinit(gsmdt, dt, dx, dy, lowlyr, .false., &
      f_ice_phy, f_rain_phy, f_rimef_phy, mp_restart_state, tbpvs_state, tbpvs0_state, .true., &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte)
  else
    print *, 'unknown scheme: ', trim(scheme)
    stop 2
  end if

  call fill_column(cid, trim(scheme), dz8w, rho_phy, p_phy, pi_phy, th_phy, qv, qt, qc, qr, qi, qs)
  th_in = th_phy
  qv_in = qv
  qt_in = qt
  qc_in = qc
  qr_in = qr
  qi_in = qi
  qs_in = qs

  if (trim(scheme) == 'mp5') then
    call FER_HIRES(1, dt, dx, dy, 1, rainnc, rainncv, dz8w, rho_phy, p_phy, pi_phy, th_phy, &
      qv, qt, lowlyr, sr, f_ice_phy, f_rain_phy, f_rimef_phy, qc, qr, qi, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte)
  else
    call ETAMP_NEW(1, dt, dx, dy, dz8w, rho_phy, p_phy, pi_phy, th_phy, qv, qt, &
      lowlyr, sr, f_ice_phy, f_rain_phy, f_rimef_phy, qc, qr, qs, &
      mp_restart_state, tbpvs_state, tbpvs0_state, rainnc, rainncv, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte)
  end if

  call dump_savepoint(trim(scheme), cid, dt, dz8w, rho_phy, p_phy, pi_phy, &
    th_in, qv_in, qt_in, qc_in, qr_in, qi_in, qs_in, th_phy, qv, qt, qc, qr, qi, qs, &
    f_ice_phy, f_rain_phy, f_rimef_phy, rainncv, sr)

contains

  subroutine fill_column(cid, scheme, dz, rho, p, pii, th, qv, qt, qc, qr, qi, qs)
    integer, intent(in) :: cid
    character(len=*), intent(in) :: scheme
    real, intent(out) :: dz(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme)
    real, intent(out) :: p(ims:ime,kms:kme,jms:jme), pii(ims:ime,kms:kme,jms:jme)
    real, intent(out) :: th(ims:ime,kms:kme,jms:jme), qv(ims:ime,kms:kme,jms:jme)
    real, intent(out) :: qt(ims:ime,kms:kme,jms:jme), qc(ims:ime,kms:kme,jms:jme)
    real, intent(out) :: qr(ims:ime,kms:kme,jms:jme), qi(ims:ime,kms:kme,jms:jme)
    real, intent(out) :: qs(ims:ime,kms:kme,jms:jme)
    integer :: k
    real :: z, temp, qbase, exn

    dz = 250.0
    qc = 0.0
    qr = 0.0
    qi = 0.0
    qs = 0.0

    do k = kts, kte
      z = (real(k) - 0.5) * dz(1,k,1)
      p(1,k,1) = 100000.0 * exp(-z / 8200.0)
      select case (cid)
      case (1)
        temp = 293.0 - 5.5e-3 * z
        qbase = 0.014 * exp(-z / 2500.0)
        if (k >= 3 .and. k <= 9) qc(1,k,1) = 6.0e-4
        if (k >= 2 .and. k <= 5) qr(1,k,1) = 1.5e-4
      case (2)
        temp = 284.0 - 6.8e-3 * z
        qbase = 0.008 * exp(-z / 3000.0)
        if (k >= 10 .and. k <= 22) qc(1,k,1) = 2.0e-4
        if (k >= 14 .and. k <= 26) qi(1,k,1) = 1.5e-4
        if (k >= 9 .and. k <= 16) qr(1,k,1) = 4.0e-5
        if (k >= 18 .and. k <= 30) qs(1,k,1) = 1.2e-4
      case (3)
        temp = 268.0 - 4.5e-3 * z
        qbase = 0.004 * exp(-z / 3500.0)
        if (k >= 12 .and. k <= 34) qi(1,k,1) = 2.5e-4
        if (k >= 18 .and. k <= 38) qs(1,k,1) = 3.0e-4
      case default
        temp = 278.0 - 6.0e-3 * z
        qbase = 0.003 * exp(-z / 2600.0)
        if (k >= 4 .and. k <= 12) qr(1,k,1) = 3.0e-4
        if (k >= 18 .and. k <= 32) qi(1,k,1) = 1.0e-4
        if (k >= 20 .and. k <= 36) qs(1,k,1) = 2.0e-4
      end select
      qv(1,k,1) = max(qbase, 1.0e-6)
      if (scheme == 'mp5') qs(1,k,1) = 0.0
      if (scheme == 'mp95') qi(1,k,1) = 0.0
      qt(1,k,1) = qc(1,k,1) + qr(1,k,1) + qi(1,k,1) + qs(1,k,1)
      exn = (p(1,k,1) / 100000.0) ** (287.04 / 1004.6)
      pii(1,k,1) = exn
      th(1,k,1) = temp / exn
      rho(1,k,1) = p(1,k,1) / (287.04 * temp * (1.0 + 0.61 * qv(1,k,1)))
    end do
  end subroutine fill_column

  subroutine dump_savepoint(scheme, cid, dt, dz, rho, p, pii, th0, qv0, qt0, qc0, qr0, qi0, qs0, &
      th1, qv1, qt1, qc1, qr1, qi1, qs1, fice, frain, frimef, rainncv, sr)
    character(len=*), intent(in) :: scheme
    integer, intent(in) :: cid
    real, intent(in) :: dt
    real, intent(in) :: dz(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: p(ims:ime,kms:kme,jms:jme), pii(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: th0(ims:ime,kms:kme,jms:jme), qv0(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qt0(ims:ime,kms:kme,jms:jme), qc0(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qr0(ims:ime,kms:kme,jms:jme), qi0(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qs0(ims:ime,kms:kme,jms:jme), th1(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qv1(ims:ime,kms:kme,jms:jme), qt1(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qc1(ims:ime,kms:kme,jms:jme), qr1(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: qi1(ims:ime,kms:kme,jms:jme), qs1(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: fice(ims:ime,kms:kme,jms:jme), frain(ims:ime,kms:kme,jms:jme)
    real, intent(in) :: frimef(ims:ime,kms:kme,jms:jme), rainncv(ims:ime,jms:jme), sr(ims:ime,jms:jme)

    write(*,'(A)') 'META source=pristine-wrf-v4'
    write(*,'(A,A)') 'META scheme=', trim(scheme)
    write(*,'(A,I0)') 'META case=', cid
    write(*,'(A,I0)') 'META kx=', kx
    write(*,'(A,ES24.16)') 'SCALAR DT ', dt
    write(*,'(A,ES24.16)') 'SCALAR RAINNCV ', rainncv(1,1)
    write(*,'(A,ES24.16)') 'SCALAR SR ', sr(1,1)
    call dump_field('DZ', dz)
    call dump_field('RHO', rho)
    call dump_field('P', p)
    call dump_field('PII', pii)
    call dump_field('TH_IN', th0)
    call dump_field('QV_IN', qv0)
    call dump_field('QT_IN', qt0)
    call dump_field('QC_IN', qc0)
    call dump_field('QR_IN', qr0)
    call dump_field('QI_IN', qi0)
    call dump_field('QS_IN', qs0)
    call dump_field('TH_OUT', th1)
    call dump_field('QV_OUT', qv1)
    call dump_field('QT_OUT', qt1)
    call dump_field('QC_OUT', qc1)
    call dump_field('QR_OUT', qr1)
    call dump_field('QI_OUT', qi1)
    call dump_field('QS_OUT', qs1)
    call dump_field('F_ICE_PHY_OUT', fice)
    call dump_field('F_RAIN_PHY_OUT', frain)
    call dump_field('F_RIMEF_PHY_OUT', frimef)
  end subroutine dump_savepoint

  subroutine dump_field(name, field)
    character(len=*), intent(in) :: name
    real, intent(in) :: field(ims:ime,kms:kme,jms:jme)
    integer :: k
    write(*,'(A,A)', advance='no') 'FIELD ', trim(name)
    do k = kts, kte
      write(*,'(1X,ES24.16)', advance='no') field(1,k,1)
    end do
    write(*,*)
  end subroutine dump_field
end program ferrier_oracle_driver
