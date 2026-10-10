! CAM01: pristine WRF V4.7.1 CAM radiation (ra_lw_physics = ra_sw_physics = 3) with the TRUE radiation_driver wiring
! on real CPU-WRF WN3 0227 columns.  Unchanged pristine routines from libwrflib.a; this driver only reproduces the caller.
!
! Caller wiring reproduced (pristine V4.7.1):
!   module_physics_init.F:2249-2269   camradinit(R_D,R_V,CP,G,STBOLT,EP_2,shalf=znu,pptop=p_top/1000 (z2sigma eta branch),
!                                     ozmixm,pin,levsiz=59,XLAT,n_ozmixm=13,m_ps_1,m_ps_2,m_hybi,aerosolc_1,aerosolc_2,
!                                     paerlev=29,n_aerosolc=13)          (module_check_a_mundo.F:3381-3384 dims)
!   module_radiation_driver.F:1601    doabsems = itimestep==1 .or. mod(itimestep,STEPABS)==1 (cam_abs_freq_s=21600)
!   module_radiation_driver.F:1972    CAMLWSCHEME: CAMRAD(dolw=.true.,dosw=.false.,...,n_cldadv=3, coszen=coszen)
!   module_radiation_driver.F:2533    CAMSWSCHEME: CAMRAD(dolw=.false.,dosw=.true.,...)  (separate call)
!   module_physics_init.F:929-941    ghg_input=1: read_CAMgases(julyr,float(julday),.true.,"CAM") at init (file read)
!   inc/scalar_indices.inc            aerosolc species indices P_sul..P_volc = 2..13 (package active)
!   radconst(xtime,declin,solcon,julian); calc_coszen(julian, xtime+radt/2, gmt)  (as RE01)
!   first_rk_step_part1: radiation P = p_hyd, P8W = p_hyd_w, PI = pi_phy, T = t_phy, RHO, Z, DZ8W
! Arms per column (one-column tiles inside one ncol-wide memory row, camradinit once for the row):
!   L = LW doabsems=.true.;  S = SW;  H = LW doabsems=.false. on a perturbed state using the HELD REAL abstot/absnxt/emstot of L
!   R8 = direct radctl calls with camrad's own preprocessing (r8 intermediates; consistency-checked against L/S REAL outputs)
! Input/Output: little-endian stream (library built -fconvert=big-endian -> CONVERT='native').
program cam01_driver
  use module_ra_cam, only: camrad, camradinit, param_cldoptics_calc, radctl, oznint, radozn, radinp, radtpl, radoz2, &
                           radems, radabs, get_aerosol
  use module_ra_cam_support, only: r8, ixcldliq, ixcldice, stebol, cpair, trcmix_clwrf, co2mmr, mxaerl, naer_all, &
                                   trcpth, ntoplw, nbands, bnd_nbr_LW, aqsat, get_int_scales, epsilo, naer
  use module_ra_clwrf_support, only: read_CAMgases
  use module_radiation_driver, only: radconst, calc_coszen
  use module_model_constants, only: R_D, R_V, CP, G, STBOLT, EP_2, DEGRAD, DPD
  use module_state_description, only: P_sul, P_sslt, P_dust1, P_dust2, P_dust3, P_dust4, P_ocpho, P_bcpho, &
                                      P_ocphi, P_bcphi, P_bg, P_volc
  implicit none

  integer, parameter :: levsiz = 59, n_ozmixm = 13, paerlev = 29, n_aerosolc = 13, cam_abs_dim1 = 4
  integer :: ncol, nz, ic, k, kk, hdr(2), yr, cam_abs_dim2, nmis_l, nmis_s, nmis_h
  real :: p_top, gmt, radt, dt, pptop, xtime, declin, solcon
  real, allocatable :: znu(:), julian(:), tsk(:), emiss(:), albedo(:), xland(:), xlat(:), xlong(:), snow(:), xice(:), tau(:)
  integer, allocatable :: julday(:)
  real, allocatable :: t(:,:), p(:,:), pii(:,:), rho(:,:), z(:,:), qv(:,:), qc(:,:), qr(:,:), qi(:,:), qs(:,:), qg(:,:), cf(:,:)
  real, allocatable :: p8w(:,:), dz8w(:,:)
  character(len=256) :: fin, fout, fint, fsint

  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  real, allocatable :: shalf(:), ozmixm(:,:,:,:), pin(:), m_psp(:,:), m_psn(:,:), m_hybi(:), aerc1(:,:,:,:), aerc2(:,:,:,:)
  real, allocatable :: g_xlat(:,:), g_xlong(:,:), g_alb(:,:), g_tsk(:,:), g_emiss(:,:), g_xland(:,:), g_xice(:,:), g_snow(:,:)
  real, allocatable :: g_cosz(:,:), g_hrang(:,:)
  real, allocatable :: g3_t(:,:,:), g3_p(:,:,:), g3_p8w(:,:,:), g3_z(:,:,:), g3_pi(:,:,:), g3_rho(:,:,:), g3_dz(:,:,:)
  real, allocatable :: g3_qv(:,:,:), g3_qc(:,:,:), g3_qr(:,:,:), g3_qi(:,:,:), g3_qs(:,:,:), g3_qg(:,:,:), g3_cf(:,:,:)
  real, allocatable :: g3_fice(:,:,:), g3_frain(:,:,:)
  real, allocatable :: hlw(:,:,:), hsw(:,:,:), hlwc(:,:,:), hswc(:,:,:), cemiss(:,:,:), taucldc(:,:,:), taucldi(:,:,:)
  real, allocatable :: abstot3(:,:,:,:), absnxt3(:,:,:,:), emstot3(:,:,:)
  real, allocatable, dimension(:,:) :: swupt, swuptc, swdnt, swdntc, lwupt, lwuptc, lwdnt, lwdntc, swupb, swupbc, swdnb, &
       swdnbc, lwupb, lwupbc, lwdnb, lwdnbc, swcf, lwcf, olr, coszr, gsw, glw, alvd, alvf, alnd, alnf, swvd, swvf, swnd, swnf, &
       swddir, swddni, swddif
  logical :: doabsems
  real(r8) :: gas0(5)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, fint)   ! LW intermediates (radinp/radtpl/radoz2/trcpth/radems/radabs), arm L state
  call get_command_argument(4, fsint)  ! SW intermediates (aqsat rh, get_aerosol AEROSOLt), arm S state
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', convert='native')
  read(10) hdr
  ncol = hdr(1); nz = hdr(2)
  allocate(znu(nz), julian(ncol), julday(ncol), tsk(ncol), emiss(ncol), albedo(ncol), xland(ncol), xlat(ncol), xlong(ncol), &
           snow(ncol), xice(ncol), tau(ncol))
  allocate(t(ncol,nz), p(ncol,nz), pii(ncol,nz), rho(ncol,nz), z(ncol,nz), qv(ncol,nz), qc(ncol,nz), qr(ncol,nz), qi(ncol,nz), &
           qs(ncol,nz), qg(ncol,nz), cf(ncol,nz), p8w(ncol,nz+1), dz8w(ncol,nz+1))
  read(10) p_top, gmt, radt, dt
  read(10) yr
  read(10) znu
  read(10) julian, julday, tsk, emiss, albedo, xland, xlat, xlong, snow, xice, tau
  read(10) t, p, pii, rho, z, qv, qc, qr, qi, qs, qg, cf
  read(10) p8w, dz8w
  close(10)

  ! one memory row of ncol columns (i), one j row; vertical memory = domain (kme = kde = e_vert)
  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  jts = 1; jte = 1; kts = 1; kte = nz
  cam_abs_dim2 = kde
  allocate(shalf(kms:kme), ozmixm(ims:ime,levsiz,jms:jme,n_ozmixm), pin(levsiz), m_psp(ims:ime,jms:jme), &
           m_psn(ims:ime,jms:jme), m_hybi(paerlev), aerc1(ims:ime,paerlev,jms:jme,n_aerosolc), &
           aerc2(ims:ime,paerlev,jms:jme,n_aerosolc))
  allocate(g_xlat(ims:ime,jms:jme), g_xlong(ims:ime,jms:jme), g_alb(ims:ime,jms:jme), g_tsk(ims:ime,jms:jme), &
           g_emiss(ims:ime,jms:jme), g_xland(ims:ime,jms:jme), g_xice(ims:ime,jms:jme), g_snow(ims:ime,jms:jme), &
           g_cosz(ims:ime,jms:jme), g_hrang(ims:ime,jms:jme))
  allocate(g3_t(ims:ime,kms:kme,jms:jme), g3_p(ims:ime,kms:kme,jms:jme), g3_p8w(ims:ime,kms:kme,jms:jme), &
           g3_z(ims:ime,kms:kme,jms:jme), g3_pi(ims:ime,kms:kme,jms:jme), g3_rho(ims:ime,kms:kme,jms:jme), &
           g3_dz(ims:ime,kms:kme,jms:jme), g3_qv(ims:ime,kms:kme,jms:jme), g3_qc(ims:ime,kms:kme,jms:jme), &
           g3_qr(ims:ime,kms:kme,jms:jme), g3_qi(ims:ime,kms:kme,jms:jme), g3_qs(ims:ime,kms:kme,jms:jme), &
           g3_qg(ims:ime,kms:kme,jms:jme), g3_cf(ims:ime,kms:kme,jms:jme), g3_fice(ims:ime,kms:kme,jms:jme), &
           g3_frain(ims:ime,kms:kme,jms:jme))
  allocate(hlw(ims:ime,kms:kme,jms:jme), hsw(ims:ime,kms:kme,jms:jme), hlwc(ims:ime,kms:kme,jms:jme), &
           hswc(ims:ime,kms:kme,jms:jme), cemiss(ims:ime,kms:kme,jms:jme), taucldc(ims:ime,kms:kme,jms:jme), &
           taucldi(ims:ime,kms:kme,jms:jme))
  allocate(abstot3(ims:ime,kms:kme,cam_abs_dim2,jms:jme), absnxt3(ims:ime,kms:kme,cam_abs_dim1,jms:jme), &
           emstot3(ims:ime,kms:kme,jms:jme))
  allocate(swupt(ims:ime,jms:jme), swuptc(ims:ime,jms:jme), swdnt(ims:ime,jms:jme), swdntc(ims:ime,jms:jme), &
           lwupt(ims:ime,jms:jme), lwuptc(ims:ime,jms:jme), lwdnt(ims:ime,jms:jme), lwdntc(ims:ime,jms:jme), &
           swupb(ims:ime,jms:jme), swupbc(ims:ime,jms:jme), swdnb(ims:ime,jms:jme), swdnbc(ims:ime,jms:jme), &
           lwupb(ims:ime,jms:jme), lwupbc(ims:ime,jms:jme), lwdnb(ims:ime,jms:jme), lwdnbc(ims:ime,jms:jme), &
           swcf(ims:ime,jms:jme), lwcf(ims:ime,jms:jme), olr(ims:ime,jms:jme), coszr(ims:ime,jms:jme), &
           gsw(ims:ime,jms:jme), glw(ims:ime,jms:jme), alvd(ims:ime,jms:jme), alvf(ims:ime,jms:jme), &
           alnd(ims:ime,jms:jme), alnf(ims:ime,jms:jme), swvd(ims:ime,jms:jme), swvf(ims:ime,jms:jme), &
           swnd(ims:ime,jms:jme), swnf(ims:ime,jms:jme), swddir(ims:ime,jms:jme), swddni(ims:ime,jms:jme), &
           swddif(ims:ime,jms:jme))

  ! WRF grid fields start at zero (Registry state); aerosol_init leaves BG/VOLC slices untouched
  shalf = 0.; shalf(1:nz) = znu(1:nz)
  pptop = p_top / 1000.
  ozmixm = 0.; pin = 0.; m_psp = 0.; m_psn = 0.; m_hybi = 0.; aerc1 = 0.; aerc2 = 0.
  do ic = 1, ncol
    g_xlat(ic,1) = xlat(ic); g_xlong(ic,1) = xlong(ic); g_alb(ic,1) = albedo(ic); g_tsk(ic,1) = tsk(ic)
    g_emiss(ic,1) = emiss(ic); g_xland(ic,1) = xland(ic); g_xice(ic,1) = xice(ic); g_snow(ic,1) = snow(ic)
  end do
  ! inc/scalar_indices.inc (set_scalar_indices_from_config, camlwscheme/camswscheme package active): the 4-D aerosolc
  ! species get indices 2..13 (slot 1 = PARAM_FIRST_SCALAR dummy); the module defaults are 1. camradinit copies them
  ! into idxSUL.. (found by the SW port, lane o1-camrad helper SWIDX).
  P_sul = 2; P_sslt = 3; P_dust1 = 4; P_dust2 = 5; P_dust3 = 6; P_dust4 = 7
  P_ocpho = 8; P_bcpho = 9; P_ocphi = 10; P_bcphi = 11; P_bg = 12; P_volc = 13
  ! module_physics_init.F:929-941: ghg_input=1 & real run -> read_CAMgases(julyr, float(julday), .true., "CAM") once at init
  call read_CAMgases(yr, real(julday(1)), .true., "CAM", gas0(1), gas0(2), gas0(3), gas0(4), gas0(5))
  its = 1; ite = ncol
  call camradinit(R_D, R_V, CP, G, STBOLT, EP_2, shalf, pptop, ozmixm, pin, levsiz, g_xlat, n_ozmixm, &
                  m_psp, m_psn, m_hybi, aerc1, aerc2, paerlev, n_aerosolc, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

  ! grid 3-D state (state A)
  g3_t = 0.; g3_p = 0.; g3_p8w = 0.; g3_z = 0.; g3_pi = 0.; g3_rho = 0.; g3_dz = 0.
  g3_qv = 0.; g3_qc = 0.; g3_qr = 0.; g3_qi = 0.; g3_qs = 0.; g3_qg = 0.; g3_cf = 0.; g3_fice = 0.; g3_frain = 0.
  do ic = 1, ncol
    g3_t(ic,1:nz,1) = t(ic,:); g3_p(ic,1:nz,1) = p(ic,:); g3_pi(ic,1:nz,1) = pii(ic,:); g3_rho(ic,1:nz,1) = rho(ic,:)
    g3_z(ic,1:nz,1) = z(ic,:); g3_p8w(ic,1:nz+1,1) = p8w(ic,:); g3_dz(ic,1:nz+1,1) = dz8w(ic,:)
    g3_qv(ic,1:nz,1) = qv(ic,:); g3_qc(ic,1:nz,1) = qc(ic,:); g3_qr(ic,1:nz,1) = qr(ic,:); g3_qi(ic,1:nz,1) = qi(ic,:)
    g3_qs(ic,1:nz,1) = qs(ic,:); g3_qg(ic,1:nz,1) = qg(ic,:); g3_cf(ic,1:nz,1) = cf(ic,:)
  end do
  alvd = g_alb; alvf = g_alb; alnd = g_alb; alnf = g_alb

  open(20, file=trim(fout), access='stream', form='unformatted', status='replace', convert='native')
  open(21, file=trim(fint), access='stream', form='unformatted', status='replace', convert='native')
  open(22, file=trim(fsint), access='stream', form='unformatted', status='replace', convert='native')
  write(20) ncol, nz, mxaerl
  write(20) pin, m_hybi
  nmis_l = 0; nmis_s = 0; nmis_h = 0
  abstot3 = 0.; absnxt3 = 0.; emstot3 = 0.
  do ic = 1, ncol
    its = ic; ite = ic
    xtime = tau(ic) * 60.0
    call radconst(xtime, declin, solcon, julian(ic), DEGRAD, DPD)
    call calc_coszen(ims, ime, jms, jme, its, ite, jts, jte, julian(ic), xtime + radt*0.5, gmt, declin, DEGRAD, &
                     g_xlong, g_xlat, g_cosz, g_hrang)
    call clear_outputs()
    ! ---- L: longwave, doabsems = .true. (itimestep 1 / every cam_abs_freq_s)
    doabsems = .true.
    call call_camrad(.true., .false.)
    call write_lw(.true.)
    write(20) abstot3(ic,1:nz+1,1:nz+1,1), absnxt3(ic,1:nz,1:4,1), emstot3(ic,1:nz+1,1)
    write(20) cemiss(ic,1:nz,1), taucldc(ic,1:nz,1), taucldi(ic,1:nz,1)
    ! ---- R8 LW (same state, same held arrays)
    call r8_dump(.true., .false., ic, nmis_l)
    ! ---- S: shortwave
    call call_camrad(.false., .true.)
    write(20) declin, solcon, g_cosz(ic,1)
    write(20) hsw(ic,1:nz,1), hswc(ic,1:nz,1)
    write(20) gsw(ic,1), swcf(ic,1), coszr(ic,1), swddir(ic,1), swddni(ic,1), swddif(ic,1), swupt(ic,1), swuptc(ic,1), &
              swdnt(ic,1), swdntc(ic,1), swupb(ic,1), swupbc(ic,1), swdnb(ic,1), swdnbc(ic,1)
    call r8_dump(.false., .true., ic, nmis_s)
    ! ---- H: held absorptivities (REAL state arrays from L) on a perturbed state
    do k = 1, nz
      g3_t(ic,k,1) = t(ic,k) + 0.8*sin(0.37*real(k))
      g3_qv(ic,k,1) = qv(ic,k) * (1.0 + 0.05*cos(0.23*real(k)))
    end do
    g_tsk(ic,1) = tsk(ic) + 0.5
    doabsems = .false.
    call call_camrad(.true., .false.)
    write(20) g3_t(ic,1:nz,1), g3_qv(ic,1:nz,1), g_tsk(ic,1)
    call write_lw(.false.)
    call r8_dump(.true., .false., ic, nmis_h)
    g3_t(ic,1:nz,1) = t(ic,:); g3_qv(ic,1:nz,1) = qv(ic,:); g_tsk(ic,1) = tsk(ic)
  end do
  close(20); close(21); close(22)
  print *, 'CAM01 columns', ncol, ' nz', nz, ' mxaerl', mxaerl, ' r8-vs-camrad REAL mismatches L/S/H:', nmis_l, nmis_s, nmis_h

contains

  subroutine clear_outputs()
    hlw = 0.; hsw = 0.; hlwc = 0.; hswc = 0.; cemiss = 0.; taucldc = 0.; taucldi = 0.
    swupt = 0.; swuptc = 0.; swdnt = 0.; swdntc = 0.; lwupt = 0.; lwuptc = 0.; lwdnt = 0.; lwdntc = 0.
    swupb = 0.; swupbc = 0.; swdnb = 0.; swdnbc = 0.; lwupb = 0.; lwupbc = 0.; lwdnb = 0.; lwdnbc = 0.
    swcf = 0.; lwcf = 0.; olr = 0.; coszr = 0.; gsw = 0.; glw = 0.; swvd = 0.; swvf = 0.; swnd = 0.; swnf = 0.
    swddir = 0.; swddni = 0.; swddif = 0.
  end subroutine clear_outputs

  subroutine call_camrad(dolw, dosw)
    logical, intent(in) :: dolw, dosw
    call camrad(RTHRATENLW=hlw, RTHRATENSW=hsw, RTHRATENLWC=hlwc, RTHRATENSWC=hswc, dolw=dolw, dosw=dosw, &
         SWUPT=swupt, SWUPTC=swuptc, SWDNT=swdnt, SWDNTC=swdntc, LWUPT=lwupt, LWUPTC=lwuptc, LWDNT=lwdnt, LWDNTC=lwdntc, &
         SWUPB=swupb, SWUPBC=swupbc, SWDNB=swdnb, SWDNBC=swdnbc, LWUPB=lwupb, LWUPBC=lwupbc, LWDNB=lwdnb, LWDNBC=lwdnbc, &
         swcf=swcf, lwcf=lwcf, olr=olr, cemiss=cemiss, taucldc=taucldc, taucldi=taucldi, coszr=coszr, &
         GSW=gsw, GLW=glw, XLAT=g_xlat, XLONG=g_xlong, ALBEDO=g_alb, t_phy=g3_t, TSK=g_tsk, EMISS=g_emiss, &
         QV3D=g3_qv, QC3D=g3_qc, QR3D=g3_qr, QI3D=g3_qi, QS3D=g3_qs, QG3D=g3_qg, &
         ALSWVISDIR=alvd, ALSWVISDIF=alvf, ALSWNIRDIR=alnd, ALSWNIRDIF=alnf, &
         SWVISDIR=swvd, SWVISDIF=swvf, SWNIRDIR=swnd, SWNIRDIF=swnf, sf_surface_physics=4, &
         SWDDIR=swddir, SWDDIF=swddif, SWDDNI=swddni, &
         F_QV=.true., F_QC=.true., F_QR=.true., F_QI=.true., F_QS=.true., F_QG=.true., &
         f_ice_phy=g3_fice, f_rain_phy=g3_frain, p_phy=g3_p, p8w=g3_p8w, z=g3_z, pi_phy=g3_pi, rho_phy=g3_rho, dz8w=g3_dz, &
         CLDFRA=g3_cf, XLAND=g_xland, XICE=g_xice, SNOW=g_snow, ozmixm=ozmixm, pin0=pin, levsiz=levsiz, &
         num_months=n_ozmixm, m_psp=m_psp, m_psn=m_psn, aerosolcp=aerc1, aerosolcn=aerc2, m_hybi0=m_hybi, &
         cam_abs_dim1=cam_abs_dim1, cam_abs_dim2=cam_abs_dim2, paerlev=paerlev, naer_c=n_aerosolc, &
         GMT=gmt, JULDAY=julday(ic), JULIAN=julian(ic), YR=yr, DT=dt, XTIME=xtime, DECLIN=declin, SOLCON=solcon, &
         RADT=radt, DEGRAD=DEGRAD, n_cldadv=3, abstot_3d=abstot3, absnxt_3d=absnxt3, emstot_3d=emstot3, &
         doabsems=doabsems, ghg_input=1, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, coszen=g_cosz)
  end subroutine call_camrad

  subroutine write_lw(with_extra)
    logical, intent(in) :: with_extra
    write(20) hlw(ic,1:nz,1), hlwc(ic,1:nz,1)
    write(20) glw(ic,1), olr(ic,1), lwcf(ic,1), lwupt(ic,1), lwuptc(ic,1), lwdnt(ic,1), lwdntc(ic,1), lwupb(ic,1), &
              lwupbc(ic,1), lwdnb(ic,1), lwdnbc(ic,1)
  end subroutine write_lw

  ! camrad's preprocessing for one column (module_ra_cam.F:512-700, copied loop by loop; ii=1, kk = kte-k+kts),
  ! then param_cldoptics_calc + radctl in r8, dumped and checked against camrad's REAL outputs of the same call.
  subroutine r8_dump(dolw, dosw, i, nmis)
    logical, intent(in) :: dolw, dosw
    integer, intent(in) :: i
    integer, intent(inout) :: nmis
    integer, parameter :: ncl = 1
    integer :: pver, pverp
    real(r8) :: coszrs(1), landfrac(1), landm(1), snowh(1), icefrac(1), lwups(1), asdir(1), asdif(1), aldir(1), aldif(1)
    real(r8) :: ps(1), clat(1), m_psjp(1), m_psjn(1), fsns(1), fsnt(1), flns(1), flnt(1), swcftoa(1), lwcftoa(1), olrtoa(1)
    real(r8) :: sols(1), soll(1), solsd(1), solld(1), fsds(1), fsdsdir(1), fsdsdif(1), flwds(1)
    real(r8) :: co2v, n2ov, ch4v, f11v, f12v
    real(r8), allocatable :: cld(:,:), pmid(:,:), lnpmid(:,:), pdel(:,:), zm(:,:), tt(:,:), pint(:,:), lnpint(:,:), q(:,:,:)
    real(r8), allocatable :: cicewp(:,:), cliqwp(:,:), tauxcl(:,:), tauxci(:,:), emis(:,:), rel(:,:), rei(:,:), pmxrgn(:,:)
    real(r8), allocatable :: fsup(:,:), fsupc(:,:), fsdn(:,:), fsdnc(:,:), fsdndir(:,:), fsdncdir(:,:), fsdndif(:,:)
    real(r8), allocatable :: fsdncdif(:,:), flup(:,:), flupc(:,:), fldn(:,:), fldnc(:,:), qrs(:,:), qrscs(:,:), qrl(:,:)
    real(r8), allocatable :: qrlcs(:,:), ozmixmj(:,:,:), ozmix(:,:), pin8(:), aerjp(:,:,:), aerjn(:,:,:), m_hybi8(:)
    real(r8), allocatable :: abstot(:,:,:), absnxt(:,:,:), emstot(:,:), o3vmr(:,:), n2o(:,:), ch4(:,:), cfc11(:,:), cfc12(:,:)
    integer :: nmxrgn(1), m, n, kk1
    real :: hreal
    pver = nz; pverp = nz + 1
    allocate(cld(1,pver), pmid(1,pver), lnpmid(1,pver), pdel(1,pver), zm(1,pver), tt(1,pver), pint(1,pverp), &
             lnpint(1,pverp), q(1,pver,3), cicewp(1,pver), cliqwp(1,pver), tauxcl(1,0:pver), tauxci(1,0:pver), &
             emis(1,pver), rel(1,pver), rei(1,pver), pmxrgn(1,pverp), fsup(1,pverp), fsupc(1,pverp), fsdn(1,pverp), &
             fsdnc(1,pverp), fsdndir(1,pverp), fsdncdir(1,pverp), fsdndif(1,pverp), fsdncdif(1,pverp), flup(1,pverp), &
             flupc(1,pverp), fldn(1,pverp), fldnc(1,pverp), qrs(1,pver), qrscs(1,pver), qrl(1,pver), qrlcs(1,pver), &
             ozmixmj(1,levsiz,n_ozmixm), ozmix(1,levsiz), pin8(levsiz), aerjp(1,paerlev,n_aerosolc), &
             aerjn(1,paerlev,n_aerosolc), m_hybi8(paerlev), abstot(1,pverp,pverp), absnxt(1,pver,4), emstot(1,pverp), &
             o3vmr(1,pver), n2o(1,pver), ch4(1,pver), cfc11(1,pver), cfc12(1,pver))
    call read_CAMgases(yr, julian(i), .false., "CAM", co2v, n2ov, ch4v, f11v, f12v)
    clat(1) = g_xlat(i,1)*DEGRAD
    coszrs(1) = g_cosz(i,1)
    do k = kts, kte
      kk = kte - k + kts
      q(1,kk,1) = max(1.e-10, g3_qv(i,k,1)/(1.+g3_qv(i,k,1)))
      q(1,kk,ixcldliq) = max(0., g3_qc(i,k,1)/(1.+g3_qv(i,k,1)))
      q(1,kk,ixcldice) = max(0., (g3_qi(i,k,1)+g3_qs(i,k,1))/(1.+g3_qv(i,k,1)))
      cld(1,kk) = g3_cf(i,k,1)
    end do
    landfrac(1) = 2.-g_xland(i,1); landm(1) = landfrac(1); snowh(1) = 0.001*g_snow(i,1); icefrac(1) = g_xice(i,1)
    ozmixmj = 0.
    do m = 1, n_ozmixm-1
      do k = 1, levsiz
        ozmixmj(1,k,m) = ozmixm(i,k,1,m+1)
      end do
    end do
    m_psjp(1) = m_psp(i,1); m_psjn(1) = m_psn(i,1)
    do n = 1, n_aerosolc
      do k = 1, paerlev
        aerjp(1,k,n) = aerc1(i,k,1,n); aerjn(1,k,n) = aerc2(i,k,1,n)
      end do
    end do
    lwups(1) = stebol*g_emiss(i,1)*g_tsk(i,1)**4
    do k = kts, kte+1
      kk = kte - k + kts + 1
      pint(1,kk) = g3_p8w(i,k,1)
      if (k .eq. kts) ps(1) = pint(1,kk)
      lnpint(1,kk) = log(pint(1,kk))
    end do
    if (.not. doabsems .and. dolw) then
      do kk = 1, cam_abs_dim2
        do kk1 = kts, kte+1
          abstot(1,kk1,kk) = abstot3(i,kk1,kk,1)
        end do
      end do
      do kk = 1, cam_abs_dim1
        do kk1 = kts, kte
          absnxt(1,kk1,kk) = absnxt3(i,kk1,kk,1)
        end do
      end do
      do kk = kts, kte+1
        emstot(1,kk) = emstot3(i,kk,1)
      end do
    endif
    do k = kts, kte
      kk = kte - k + kts
      pmid(1,kk) = g3_p(i,k,1)
      lnpmid(1,kk) = log(pmid(1,kk))
      lnpint(1,kk) = log(pint(1,kk))
      pdel(1,kk) = pint(1,kk+1) - pint(1,kk)
      tt(1,kk) = g3_t(i,k,1)
      zm(1,kk) = g3_z(i,k,1)
    end do
    call param_cldoptics_calc(ncl, ncl, pver, pverp, pver, pverp, 3, q, cld, landfrac, landm, icefrac, &
                              pdel, tt, ps, pmid, pint, cicewp, cliqwp, emis, rel, rei, pmxrgn, nmxrgn, snowh)
    asdir(1) = g_alb(i,1); asdif(1) = g_alb(i,1); aldir(1) = g_alb(i,1); aldif(1) = g_alb(i,1)
    pin8 = pin; m_hybi8 = m_hybi
    tauxcl = 0.; tauxci = 0.
    call radctl(1, 1, ncl, ncl, pver, pverp, pver, pverp, 3, 3, lwups, emis, pmid, &
                pint, lnpmid, lnpint, pdel, tt, q, cld, cicewp, cliqwp, tauxcl, tauxci, coszrs, clat, asdir, asdif, &
                aldir, aldif, solcon, gmt, julday(i), julian(i), dt, xtime, &
                pin8, ozmixmj, ozmix, levsiz, n_ozmixm, 1, &
                m_psjp, m_psjn, aerjp, aerjn, m_hybi8, paerlev, n_aerosolc, pmxrgn, nmxrgn, &
                dolw, dosw, doabsems, abstot, absnxt, emstot, &
                fsup, fsupc, fsdn, fsdnc, fsdndir, fsdncdir, fsdndif, fsdncdif, &
                flup, flupc, fldn, fldnc, swcftoa, lwcftoa, olrtoa, &
                fsns, fsnt, flns, flnt, qrs, qrscs, qrl, qrlcs, flwds, rel, rei, &
                sols, soll, solsd, solld, n2ov, ch4v, f11v, f12v, landfrac, zm, fsds, fsdsdir, fsdsdif)
    ! ozone and trace gases as radctl sees them (oznint/radozn/trcmix_clwrf are pure)
    call oznint(julday(i), julian(i), dt, gmt, xtime, ozmixmj, ozmix, levsiz, n_ozmixm, 1)
    call radozn(1, 1, 1, pver, pmid, pin8, levsiz, ozmix, o3vmr)
    call trcmix_clwrf(1, 1, 1, pver, pmid, clat, n2ov, ch4v, f11v, f12v, n2o, ch4, cfc11, cfc12)
    if (dosw) call sw_intermediates(pver, pverp, tt, q, pmid, pint, m_psjp, m_psjn, aerjp, aerjn, m_hybi8)
    if (dolw .and. doabsems) call lw_intermediates(pver, pverp, lwups, tt, q, o3vmr, pmid, pint, lnpmid, lnpint, &
                                                     n2o, ch4, cfc11, cfc12)
    write(20) co2v, n2ov, ch4v, f11v, f12v, co2mmr, lwups(1), coszrs(1)
    write(20) q(1,:,1), q(1,:,2), q(1,:,3), cld(1,:), pmid(1,:), pint(1,:), tt(1,:)
    write(20) cicewp(1,:), cliqwp(1,:), emis(1,:), rel(1,:), rei(1,:), pmxrgn(1,:), nmxrgn(1)
    write(20) o3vmr(1,:), n2o(1,:), ch4(1,:), cfc11(1,:), cfc12(1,:)
    if (dolw) then
      write(20) qrl(1,:), qrlcs(1,:), flup(1,:), flupc(1,:), fldn(1,:), fldnc(1,:), flwds(1), flns(1), flnt(1), &
                lwcftoa(1), olrtoa(1)
      write(20) abstot(1,:,:), absnxt(1,:,:), emstot(1,:)
      do k = kts, kte
        kk = kte - k + kts
        hreal = 1.e4*qrl(1,kk)/(cpair*g3_pi(i,k,1))
        if (hreal /= hlw(i,k,1)) nmis = nmis + 1
      end do
    end if
    if (dosw) then
      write(20) qrs(1,:), qrscs(1,:), fsup(1,:), fsupc(1,:), fsdn(1,:), fsdnc(1,:), fsdndir(1,:), fsdndif(1,:), &
                fsns(1), fsds(1), fsdsdir(1), fsdsdif(1), swcftoa(1), sols(1), soll(1), solsd(1), solld(1)
      write(20) tauxcl(1,1:pver), tauxci(1,1:pver)
      do k = kts, kte
        kk = kte - k + kts
        hreal = 1.e4*qrs(1,kk)/(cpair*g3_pi(i,k,1))
        if (hreal /= hsw(i,k,1)) nmis = nmis + 1
      end do
    end if
  end subroutine r8_dump

  ! radclwmx's own call sequence (module_ra_cam.F radclwmx: aer_pth, radtpl, radoz2, trcpth, aer_trn, radems, radabs) on the
  ! radctl operands (radinp pmidrd/pintrd = pbr/pnm, pmln/piln = lnpmid/lnpint), every output dumped for unit parity.
  subroutine lw_intermediates(pver, pverp, lwups, tt, q, o3vmr, pmid, pint, lnpmid, lnpint, n2o, ch4, cfc11, cfc12)
    integer, intent(in) :: pver, pverp
    real(r8), intent(in) :: lwups(1), tt(1,pver), q(1,pver,3), o3vmr(1,pver), pmid(1,pver), pint(1,pverp)
    real(r8), intent(in) :: lnpmid(1,pver), lnpint(1,pverp), n2o(1,pver), ch4(1,pver), cfc11(1,pver), cfc12(1,pver)
    real(r8) :: pbr(1,pver), pnm(1,pverp), eccf, o3mmr(1,pver)
    real(r8) :: plco2(1,pverp), plh2o(1,pverp), tplnka(1,pverp), s2c(1,pverp), tcg(1,pverp), w(1,pverp), tplnke(1)
    real(r8) :: tint(1,pverp), tint4(1,pverp), tlayr(1,pverp), tlayr4(1,pverp), plh2ob(nbands,1,pverp), wb(nbands,1,pverp)
    real(r8) :: plol(1,pverp), plos(1,pverp)
    real(r8) :: ucfc11(1,pverp), ucfc12(1,pverp), un2o0(1,pverp), un2o1(1,pverp), uch4(1,pverp), uco211(1,pverp)
    real(r8) :: uco212(1,pverp), uco213(1,pverp), uco221(1,pverp), uco222(1,pverp), uco223(1,pverp), bn2o0(1,pverp)
    real(r8) :: bn2o1(1,pverp), bch4(1,pverp), uptype(1,pverp)
    real(r8) :: co2em(1,pverp), co2eml(1,pver), co2t(1,pverp), h2otr(1,pverp), abplnk1(14,1,pverp), abplnk2(14,1,pverp)
    real(r8) :: emstot(1,pverp), abstot(1,pverp,pverp), absnxt(1,pver,4), aer_mpp(1,pverp)
    real(r8), allocatable :: aer_trn_ttl(:,:,:,:)
    allocate(aer_trn_ttl(1,pverp,pverp,bnd_nbr_LW))
    aer_trn_ttl = 1.0      ! aer_trn with strat_volcanic = .false.
    aer_mpp = 0.0          ! aer_pth of the zero volcanic mass (get_aerosol: AEROSOLt(:,:,idxVOLC) = 0)
    call radinp(1, 1, 1, pver, pverp, pmid, pint, o3vmr, pbr, pnm, eccf, o3mmr)
    call radtpl(1, 1, 1, pver, pverp, tt, lwups, q(:,:,1), pnm, plco2, plh2o, tplnka, s2c, tcg, w, tplnke, &
                tint, tint4, tlayr, tlayr4, lnpmid, lnpint, plh2ob, wb)
    call radoz2(1, 1, 1, pver, pverp, o3vmr, pnm, plol, plos, ntoplw)
    call trcpth(1, 1, 1, pver, pverp, tt, pnm, cfc11, cfc12, n2o, ch4, q(:,:,1), ucfc11, ucfc12, un2o0, &
                un2o1, uch4, uco211, uco212, uco213, uco221, uco222, uco223, bn2o0, bn2o1, bch4, uptype)
    call radems(1, 1, 1, pver, pverp, s2c, tcg, w, tplnke, plh2o, pnm, plco2, tint, tint4, tlayr, tlayr4, plol, plos, &
                ucfc11, ucfc12, un2o0, un2o1, uch4, uco211, uco212, uco213, uco221, uco222, uco223, uptype, &
                bn2o0, bn2o1, bch4, co2em, co2eml, co2t, h2otr, abplnk1, abplnk2, emstot, plh2ob, wb, aer_trn_ttl)
    call radabs(1, 1, 1, pver, pverp, pbr, pnm, co2em, co2eml, tplnka, s2c, tcg, w, h2otr, plco2, plh2o, co2t, &
                tint, tlayr, plol, plos, lnpmid, lnpint, ucfc11, ucfc12, un2o0, un2o1, uch4, uco211, uco212, &
                uco213, uco221, uco222, uco223, uptype, bn2o0, bn2o1, bch4, abplnk1, abplnk2, abstot, absnxt, &
                plh2ob, wb, aer_mpp, aer_trn_ttl)
    write(21) pbr(1,:), pnm(1,:), eccf, o3mmr(1,:)
    write(21) plco2(1,:), plh2o(1,:), tplnka(1,:), s2c(1,:), tcg(1,:), w(1,:), tplnke(1), tint(1,:), tint4(1,:), &
              tlayr(1,:), tlayr4(1,:), plh2ob(:,1,:), wb(:,1,:)
    write(21) plol(1,:), plos(1,:)
    write(21) ucfc11(1,:), ucfc12(1,:), un2o0(1,:), un2o1(1,:), uch4(1,:), uco211(1,:), uco212(1,:), uco213(1,:), &
              uco221(1,:), uco222(1,:), uco223(1,:), bn2o0(1,:), bn2o1(1,:), bch4(1,:), uptype(1,:)
    write(21) co2em(1,:), co2eml(1,:), co2t(1,:), h2otr(1,:), abplnk1(:,1,:), abplnk2(:,1,:), emstot(1,:)
    write(21) abstot(1,:,:), absnxt(1,:,:)
  end subroutine lw_intermediates

  ! radctl's SW operands: aqsat -> rh (module_ra_cam.F radctl), get_int_scales + get_aerosol(naer passed as naer_c).
  subroutine sw_intermediates(pver, pverp, tt, q, pmid, pint, m_psjp, m_psjn, aerjp, aerjn, m_hybi8)
    integer, intent(in) :: pver, pverp
    real(r8), intent(in) :: tt(1,pver), q(1,pver,3), pmid(1,pver), pint(1,pverp), m_psjp(1), m_psjn(1)
    real(r8), intent(in) :: aerjp(1,paerlev,n_aerosolc), aerjn(1,paerlev,n_aerosolc), m_hybi8(paerlev)
    real(r8) :: esat(1,pver), qsat(1,pver), rh(1,pver), scales(naer_all), aerosol(1,pver,naer_all)
    call aqsat(tt, pmid, esat, qsat, 1, 1, pver, 1, pver)
    rh(1,1:pver) = q(1,1:pver,1) / qsat(1,1:pver) * ((1.0 - epsilo) * qsat(1,1:pver) + epsilo) / &
                   ((1.0 - epsilo) * q(1,1:pver,1) + epsilo)
    call get_int_scales(scales)
    aerosol = 0.
    call get_aerosol(1, julday(ic), julian(ic), dt, gmt, xtime, m_psjp, m_psjn, aerjp, aerjn, m_hybi8, paerlev, naer, &
                     pint, 1, pver, pverp, pver, pverp, aerosol, scales)
    write(22) esat(1,:), qsat(1,:), rh(1,:), scales, aerosol(1,:,:)
  end subroutine sw_intermediates

end program cam01_driver
