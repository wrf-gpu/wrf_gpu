program wrf_rrtmg_harness
  use module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad
  use module_ra_rrtmg_lw, only: rrtmg_lwinit, rrtmg_lwrad
  implicit none

  integer, parameter :: rk = kind(1.0)
  real(rk), parameter :: cp_air = 1004.0_rk
  real(rk), parameter :: gravity = 9.80665_rk
  real(rk), parameter :: rd_air = 287.04_rk
  real(rk), parameter :: rv_over_rd_minus_one = 0.608_rk
  real(rk), parameter :: stefan_boltzmann = 5.670374419e-8_rk
  real(rk), parameter :: lw_init_ptop_pa = 400.0_rk

  integer :: nz, k, argc
  character(len=512) :: input_path, output_path
  real(rk) :: surface_albedo, coszen, surface_temperature, surface_emissivity
  real(rk), allocatable :: temp(:), press(:), qv(:), qc(:), qi(:), qs(:), qg(:), cldfra(:), dz(:), rho(:)
  real(rk), allocatable :: sw_heat(:), lw_heat(:), layer_mass_p(:)
  real(rk), allocatable :: sw_down(:), sw_up(:), lw_down(:), lw_up(:)
  real(rk) :: sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_rrtmg_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) nz
  read(10,*) surface_albedo, coszen, surface_temperature, surface_emissivity
  allocate(temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz))
  allocate(sw_heat(nz), lw_heat(nz), layer_mass_p(nz))
  allocate(sw_down(0:nz+1), sw_up(0:nz+1), lw_down(0:nz+1), lw_up(0:nz+1))
  do k = 1, nz
     read(10,*) temp(k), press(k), qv(k), qc(k), qi(k), qs(k), qg(k), cldfra(k), dz(k), rho(k)
  end do
  close(10)

  call rrtmg_swinit(.true., 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz)
  call rrtmg_lwinit(lw_init_ptop_pa, .true., 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz)

  call run_wrf_rrtmg_drivers(nz, surface_albedo, coszen, surface_temperature, surface_emissivity, &
                             temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                             sw_heat, lw_heat, sw_down, sw_up, lw_down, lw_up, &
                             sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I6)') nz
  write(20,'(6(ES24.16E3,1X))') sw_down(nz+1), sw_up(nz+1), sw_down(0), sw_up(0), sw_col_abs, sw_sfc_abs
  write(20,'(6(ES24.16E3,1X))') lw_down(nz+1), lw_up(nz+1), lw_down(0), lw_up(0), lw_col_heat, lw_sfc_emit
  do k = 1, nz
     write(20,'(3(ES24.16E3,1X))') sw_heat(k), lw_heat(k), layer_mass_p(k)
  end do
  do k = 0, nz+1
     write(20,'(4(ES24.16E3,1X))') sw_down(k), sw_up(k), lw_down(k), lw_up(k)
  end do
  close(20)

contains

  subroutine run_wrf_rrtmg_drivers(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                   temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                   sw_heat, lw_heat, sw_down, sw_up, lw_down, lw_up, &
                                   sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(out) :: sw_heat(nz), lw_heat(nz), sw_down(0:nz+1), sw_up(0:nz+1), lw_down(0:nz+1), lw_up(0:nz+1)
    real(rk), intent(out) :: sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p(nz)

    integer :: ids, ide, jds, jde, kds, kde
    integer :: ims, ime, jms, jme, kms, kme
    integer :: its, ite, jts, jte, kts, kte
    integer :: no_src, sf_surface_physics, yr, julday, ghg_input
    integer :: icloud, cldovrlp, idcor, o3input, mp_physics
    integer :: has_reqc, has_reqi, has_reqs, aer_opt, aer_ra_feedback, calc_clean_atm_diag, progn
    real(rk) :: xtime, gmt, radt, degrad, declin, solcon, julian, r_d, grav
    real(rk) :: net_surface, net_top
    logical :: warm_rain, is_cammgmp_used, f_qv, f_qc, f_qr, f_qi, f_qs, f_qg, proceed_cmaq_sw
    real(rk), pointer :: tauaer3d_sw(:,:,:,:), ssaaer3d_sw(:,:,:,:), asyaer3d_sw(:,:,:,:)

    real(rk), allocatable :: t3d(:,:,:), t8w(:,:,:), p3d(:,:,:), p8w(:,:,:), pi3d(:,:,:), rho3d(:,:,:), dz8w(:,:,:)
    real(rk), allocatable :: cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), qi3d(:,:,:), qs3d(:,:,:), qg3d(:,:,:)
    real(rk), allocatable :: o33d(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
    real(rk), allocatable :: rthratensw(:,:,:), rthratenswc(:,:,:), rthratenlw(:,:,:), rthratenlwc(:,:,:)
    real(rk), allocatable :: swupflx(:,:,:), swupflxc(:,:,:), swdnflx(:,:,:), swdnflxc(:,:,:)
    real(rk), allocatable :: lwupflx(:,:,:), lwupflxc(:,:,:), lwdnflx(:,:,:), lwdnflxc(:,:,:)
    real(rk), allocatable :: xlat(:,:), xlong(:,:), xland(:,:), xice(:,:), snow(:,:), tsk(:,:), albedo(:,:), emiss(:,:)
    real(rk), allocatable :: coszr(:,:), xcoszen(:,:), obscur(:,:)
    real(rk), allocatable :: gsw(:,:), swcf(:,:), glw(:,:), olr(:,:), lwcf(:,:)
    real(rk), allocatable :: swupt(:,:), swuptc(:,:), swuptcln(:,:), swdnt(:,:), swdntc(:,:), swdntcln(:,:)
    real(rk), allocatable :: swupb(:,:), swupbc(:,:), swupbcln(:,:), swdnb(:,:), swdnbc(:,:), swdnbcln(:,:)
    real(rk), allocatable :: lwupt(:,:), lwuptc(:,:), lwuptcln(:,:), lwdnt(:,:), lwdntc(:,:), lwdntcln(:,:)
    real(rk), allocatable :: lwupb(:,:), lwupbc(:,:), lwupbcln(:,:), lwdnb(:,:), lwdnbc(:,:), lwdnbcln(:,:)
    real(rk), allocatable :: swvisdir(:,:), swvisdif(:,:), swnirdir(:,:), swnirdif(:,:)
    real(rk), allocatable :: swddir(:,:), swddni(:,:), swddif(:,:), swdownc(:,:), swddnic(:,:), swddirc(:,:)
    real(rk), allocatable :: alswvisdir(:,:), alswvisdif(:,:), alswnirdir(:,:), alswnirdif(:,:)

    ids = 1; ide = 1; jds = 1; jde = 1; kds = 1; kde = nz + 1
    ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
    its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz
    no_src = 6
    sf_surface_physics = 0
    yr = 2026
    julday = 142
    julian = 142.0_rk
    ghg_input = 0
    xtime = 720.0_rk
    gmt = 12.0_rk
    radt = 30.0_rk
    degrad = 0.017453292519943295_rk
    declin = 0.0_rk
    solcon = 1368.22_rk
    r_d = rd_air
    grav = gravity
    icloud = 1
    cldovrlp = 1
    idcor = 0
    o3input = 0
    mp_physics = 8
    has_reqc = 1
    has_reqi = 1
    has_reqs = 1
    aer_opt = 0
    aer_ra_feedback = 0
    calc_clean_atm_diag = 0
    progn = 0
    warm_rain = .false.
    is_cammgmp_used = .false.
    f_qv = .true.; f_qc = .true.; f_qr = .true.; f_qi = .true.; f_qs = .true.; f_qg = .true.
    proceed_cmaq_sw = .false.

    nullify(tauaer3d_sw)
    nullify(ssaaer3d_sw)
    nullify(asyaer3d_sw)

    allocate(t3d(1,kms:kme,1), t8w(1,kms:kme,1), p3d(1,kms:kme,1), p8w(1,kms:kme,1), pi3d(1,kms:kme,1), rho3d(1,kms:kme,1), dz8w(1,kms:kme,1))
    allocate(cldfra3d(1,kms:kme,1), qv3d(1,kms:kme,1), qc3d(1,kms:kme,1), qr3d(1,kms:kme,1), qi3d(1,kms:kme,1), qs3d(1,kms:kme,1), qg3d(1,kms:kme,1))
    allocate(o33d(1,kms:kme,1), re_cloud(1,kms:kme,1), re_ice(1,kms:kme,1), re_snow(1,kms:kme,1))
    allocate(rthratensw(1,kms:kme,1), rthratenswc(1,kms:kme,1), rthratenlw(1,kms:kme,1), rthratenlwc(1,kms:kme,1))
    allocate(swupflx(1,kms:kme+2,1), swupflxc(1,kms:kme+2,1), swdnflx(1,kms:kme+2,1), swdnflxc(1,kms:kme+2,1))
    allocate(lwupflx(1,kms:kme+2,1), lwupflxc(1,kms:kme+2,1), lwdnflx(1,kms:kme+2,1), lwdnflxc(1,kms:kme+2,1))
    allocate(xlat(1,1), xlong(1,1), xland(1,1), xice(1,1), snow(1,1), tsk(1,1), albedo(1,1), emiss(1,1))
    allocate(coszr(1,1), xcoszen(1,1), obscur(1,1))
    allocate(gsw(1,1), swcf(1,1), glw(1,1), olr(1,1), lwcf(1,1))
    allocate(swupt(1,1), swuptc(1,1), swuptcln(1,1), swdnt(1,1), swdntc(1,1), swdntcln(1,1))
    allocate(swupb(1,1), swupbc(1,1), swupbcln(1,1), swdnb(1,1), swdnbc(1,1), swdnbcln(1,1))
    allocate(lwupt(1,1), lwuptc(1,1), lwuptcln(1,1), lwdnt(1,1), lwdntc(1,1), lwdntcln(1,1))
    allocate(lwupb(1,1), lwupbc(1,1), lwupbcln(1,1), lwdnb(1,1), lwdnbc(1,1), lwdnbcln(1,1))
    allocate(swvisdir(1,1), swvisdif(1,1), swnirdir(1,1), swnirdif(1,1))
    allocate(swddir(1,1), swddni(1,1), swddif(1,1), swdownc(1,1), swddnic(1,1), swddirc(1,1))
    allocate(alswvisdir(1,1), alswvisdif(1,1), alswnirdir(1,1), alswnirdif(1,1))

    call initialize_column_arrays(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                  temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                  t3d, t8w, p3d, p8w, pi3d, rho3d, dz8w, cldfra3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, &
                                  o33d, re_cloud, re_ice, re_snow, xlat, xlong, xland, xice, snow, tsk, albedo, emiss, &
                                  coszr, xcoszen, obscur, alswvisdir, alswvisdif, alswnirdir, alswnirdif, layer_mass_p)

    rthratensw = 0.0_rk; rthratenswc = 0.0_rk; rthratenlw = 0.0_rk; rthratenlwc = 0.0_rk
    swupflx = 0.0_rk; swupflxc = 0.0_rk; swdnflx = 0.0_rk; swdnflxc = 0.0_rk
    lwupflx = 0.0_rk; lwupflxc = 0.0_rk; lwdnflx = 0.0_rk; lwdnflxc = 0.0_rk
    gsw = 0.0_rk; swcf = 0.0_rk; glw = 0.0_rk; olr = 0.0_rk; lwcf = 0.0_rk
    swupt = 0.0_rk; swuptc = 0.0_rk; swuptcln = 0.0_rk; swdnt = 0.0_rk; swdntc = 0.0_rk; swdntcln = 0.0_rk
    swupb = 0.0_rk; swupbc = 0.0_rk; swupbcln = 0.0_rk; swdnb = 0.0_rk; swdnbc = 0.0_rk; swdnbcln = 0.0_rk
    lwupt = 0.0_rk; lwuptc = 0.0_rk; lwuptcln = 0.0_rk; lwdnt = 0.0_rk; lwdntc = 0.0_rk; lwdntcln = 0.0_rk
    lwupb = 0.0_rk; lwupbc = 0.0_rk; lwupbcln = 0.0_rk; lwdnb = 0.0_rk; lwdnbc = 0.0_rk; lwdnbcln = 0.0_rk
    swvisdir = 0.0_rk; swvisdif = 0.0_rk; swnirdir = 0.0_rk; swnirdif = 0.0_rk
    swddir = 0.0_rk; swddni = 0.0_rk; swddif = 0.0_rk; swdownc = 0.0_rk; swddnic = 0.0_rk; swddirc = 0.0_rk

    call rrtmg_swrad( &
         rthratensw=rthratensw, rthratenswc=rthratenswc, &
         swupt=swupt, swuptc=swuptc, swuptcln=swuptcln, swdnt=swdnt, swdntc=swdntc, swdntcln=swdntcln, &
         swupb=swupb, swupbc=swupbc, swupbcln=swupbcln, swdnb=swdnb, swdnbc=swdnbc, swdnbcln=swdnbcln, &
         swcf=swcf, gsw=gsw, xtime=xtime, gmt=gmt, xlat=xlat, xlong=xlong, radt=radt, degrad=degrad, declin=declin, &
         coszr=coszr, julday=julday, solcon=solcon, albedo=albedo, t3d=t3d, t8w=t8w, tsk=tsk, p3d=p3d, p8w=p8w, pi3d=pi3d, rho3d=rho3d, &
         dz8w=dz8w, cldfra3d=cldfra3d, lradius=re_cloud, iradius=re_ice, is_cammgmp_used=is_cammgmp_used, r=r_d, g=grav, &
         re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
         icloud=icloud, warm_rain=warm_rain, cldovrlp=cldovrlp, idcor=idcor, f_ice_phy=qi3d, f_rain_phy=qr3d, &
         xland=xland, xice=xice, snow=snow, qv3d=qv3d, qc3d=qc3d, qr3d=qr3d, qi3d=qi3d, qs3d=qs3d, qg3d=qg3d, &
         o3input=o3input, o33d=o33d, aer_opt=aer_opt, no_src=no_src, &
         alswvisdir=alswvisdir, alswvisdif=alswvisdif, alswnirdir=alswnirdir, alswnirdif=alswnirdif, &
         swvisdir=swvisdir, swvisdif=swvisdif, swnirdir=swnirdir, swnirdif=swnirdif, sf_surface_physics=sf_surface_physics, &
         f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, aer_ra_feedback=aer_ra_feedback, progn=progn, &
         calc_clean_atm_diag=calc_clean_atm_diag, mp_physics=mp_physics, ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
         swupflx=swupflx, swupflxc=swupflxc, swdnflx=swdnflx, swdnflxc=swdnflxc, tauaer3d_sw=tauaer3d_sw, ssaaer3d_sw=ssaaer3d_sw, asyaer3d_sw=asyaer3d_sw, &
         swddir=swddir, swddni=swddni, swddif=swddif, swdownc=swdownc, swddnic=swddnic, swddirc=swddirc, &
         xcoszen=xcoszen, yr=yr, julian=julian, ghg_input=ghg_input, obscur=obscur, proceed_cmaq_sw=proceed_cmaq_sw)

    call rrtmg_lwrad( &
         rthratenlw=rthratenlw, rthratenlwc=rthratenlwc, &
         lwupt=lwupt, lwuptc=lwuptc, lwuptcln=lwuptcln, lwdnt=lwdnt, lwdntc=lwdntc, lwdntcln=lwdntcln, &
         lwupb=lwupb, lwupbc=lwupbc, lwupbcln=lwupbcln, lwdnb=lwdnb, lwdnbc=lwdnbc, lwdnbcln=lwdnbcln, &
         glw=glw, olr=olr, lwcf=lwcf, emiss=emiss, p8w=p8w, p3d=p3d, pi3d=pi3d, dz8w=dz8w, tsk=tsk, t3d=t3d, t8w=t8w, rho3d=rho3d, r=r_d, g=grav, &
         icloud=icloud, warm_rain=warm_rain, cldfra3d=cldfra3d, cldovrlp=cldovrlp, idcor=idcor, xlat=xlat, &
         lradius=re_cloud, iradius=re_ice, is_cammgmp_used=is_cammgmp_used, f_ice_phy=qi3d, f_rain_phy=qr3d, &
         xland=xland, xice=xice, snow=snow, qv3d=qv3d, qc3d=qc3d, qr3d=qr3d, qi3d=qi3d, qs3d=qs3d, qg3d=qg3d, &
         o3input=o3input, o33d=o33d, f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, &
         re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
         aer_ra_feedback=aer_ra_feedback, progn=progn, calc_clean_atm_diag=calc_clean_atm_diag, yr=yr, julian=julian, ghg_input=ghg_input, mp_physics=mp_physics, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, lwupflx=lwupflx, lwupflxc=lwupflxc, lwdnflx=lwdnflx, lwdnflxc=lwdnflxc)

    do k = 1, nz
       sw_heat(k) = rthratensw(1,k,1)
       lw_heat(k) = rthratenlw(1,k,1)
    enddo
    do k = 0, nz+1
       sw_up(k) = swupflx(1,k+1,1)
       sw_down(k) = swdnflx(1,k+1,1)
       lw_up(k) = lwupflx(1,k+1,1)
       lw_down(k) = lwdnflx(1,k+1,1)
    enddo

    sw_down(nz+1) = swdnt(1,1)
    sw_up(nz+1) = swupt(1,1)
    lw_down(nz+1) = lwdnt(1,1)
    lw_up(nz+1) = lwupt(1,1)

    net_surface = sw_down(0) - sw_up(0)
    net_top = sw_down(nz+1) - sw_up(nz+1)
    sw_col_abs = net_top - net_surface
    sw_sfc_abs = net_surface
    net_surface = lw_down(0) - lw_up(0)
    net_top = lw_down(nz) - lw_up(nz)
    lw_col_heat = net_top - net_surface
    lw_sfc_emit = stefan_boltzmann * max(0.0_rk, min(1.0_rk, surface_emissivity)) * max(surface_temperature, 120.0_rk)**4
  end subroutine run_wrf_rrtmg_drivers

  subroutine initialize_column_arrays(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                      temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                      t3d, t8w, p3d, p8w, pi3d, rho3d, dz8w, cldfra3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, &
                                      o33d, re_cloud, re_ice, re_snow, xlat, xlong, xland, xice, snow, tsk, albedo, emiss, &
                                      coszr, xcoszen, obscur, alswvisdir, alswvisdif, alswnirdir, alswnirdif, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(inout) :: t3d(:,:,:), t8w(:,:,:), p3d(:,:,:), p8w(:,:,:), pi3d(:,:,:), rho3d(:,:,:), dz8w(:,:,:)
    real(rk), intent(inout) :: cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), qi3d(:,:,:), qs3d(:,:,:), qg3d(:,:,:)
    real(rk), intent(inout) :: o33d(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
    real(rk), intent(inout) :: xlat(:,:), xlong(:,:), xland(:,:), xice(:,:), snow(:,:), tsk(:,:), albedo(:,:), emiss(:,:)
    real(rk), intent(inout) :: coszr(:,:), xcoszen(:,:), obscur(:,:), alswvisdir(:,:), alswvisdif(:,:), alswnirdir(:,:), alswnirdif(:,:)
    real(rk), intent(out) :: layer_mass_p(nz)
    integer :: k

    t3d = 0.0_rk; t8w = 0.0_rk; p3d = 0.0_rk; p8w = 0.0_rk; pi3d = 1.0_rk; rho3d = 0.0_rk; dz8w = 0.0_rk
    cldfra3d = 0.0_rk; qv3d = 0.0_rk; qc3d = 0.0_rk; qr3d = 0.0_rk; qi3d = 0.0_rk; qs3d = 0.0_rk; qg3d = 0.0_rk
    o33d = 0.0_rk
    re_cloud = 10.0e-6_rk
    re_ice = 30.0e-6_rk
    re_snow = 75.0e-6_rk

    xlat(1,1) = 28.3_rk
    xlong(1,1) = -16.5_rk
    xland(1,1) = 1.0_rk
    xice(1,1) = 0.0_rk
    snow(1,1) = 0.0_rk
    tsk(1,1) = surface_temperature
    albedo(1,1) = surface_albedo
    emiss(1,1) = surface_emissivity
    coszr(1,1) = coszen_scalar
    xcoszen(1,1) = coszen_scalar
    obscur(1,1) = 0.0_rk
    alswvisdir(1,1) = surface_albedo
    alswvisdif(1,1) = surface_albedo
    alswnirdir(1,1) = surface_albedo
    alswnirdif(1,1) = surface_albedo

    do k = 1, nz
       t3d(1,k,1) = temp(k)
       p3d(1,k,1) = press(k)
       rho3d(1,k,1) = rho(k)
       dz8w(1,k,1) = dz(k)
       qv3d(1,k,1) = max(qv(k), 1.0e-12_rk)
       qc3d(1,k,1) = max(qc(k), 0.0_rk)
       qi3d(1,k,1) = max(qi(k), 0.0_rk)
       qs3d(1,k,1) = max(qs(k), 0.0_rk)
       qg3d(1,k,1) = max(qg(k), 0.0_rk)
       cldfra3d(1,k,1) = min(1.0_rk, max(0.0_rk, cldfra(k)))
    enddo

    t3d(1,nz+1,1) = temp(nz)
    p3d(1,nz+1,1) = max(1.0_rk, press(nz) * 0.5_rk)
    rho3d(1,nz+1,1) = max(1.0e-6_rk, p3d(1,nz+1,1) / (rd_air * temp(nz) * (1.0_rk + rv_over_rd_minus_one * max(qv(nz), 0.0_rk))))
    dz8w(1,nz+1,1) = dz(nz)
    qv3d(1,nz+1,1) = max(qv(nz), 1.0e-12_rk)

    call fill_interface_profiles(nz, temp, press, t8w, p8w, layer_mass_p)
  end subroutine initialize_column_arrays

  subroutine fill_interface_profiles(nz, temp, press, t8w, p8w, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: temp(nz), press(nz)
    real(rk), intent(inout) :: t8w(:,:,:), p8w(:,:,:)
    real(rk), intent(out) :: layer_mass_p(nz)
    integer :: k
    real(rk) :: dp_bottom, dp_top

    if (nz == 1) then
       p8w(1,1,1) = press(1) + 50.0_rk
       p8w(1,2,1) = max(1.0_rk, press(1) - 50.0_rk)
       t8w(1,1,1) = temp(1)
       t8w(1,2,1) = temp(1)
    else
       dp_bottom = max(10.0_rk, press(1) - press(2))
       p8w(1,1,1) = press(1) + 0.5_rk * dp_bottom
       t8w(1,1,1) = temp(1)
       do k = 2, nz
          p8w(1,k,1) = 0.5_rk * (press(k-1) + press(k))
          t8w(1,k,1) = 0.5_rk * (temp(k-1) + temp(k))
       enddo
       dp_top = max(10.0_rk, press(nz-1) - press(nz))
       p8w(1,nz+1,1) = max(400.0_rk, press(nz) - 0.5_rk * dp_top)
       t8w(1,nz+1,1) = temp(nz)
    endif
    do k = 1, nz
       layer_mass_p(k) = max(1.0e-6_rk, (p8w(1,k,1) - p8w(1,k+1,1)) / gravity)
    enddo
  end subroutine fill_interface_profiles

end program wrf_rrtmg_harness

subroutine read_camgases_stub() bind(C, name="__module_ra_clwrf_support_MOD_read_camgases")
  use iso_c_binding
  implicit none
end subroutine read_camgases_stub

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(values, nbytes)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nbytes
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_bcast_integer(values, nitems)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_real(values, nitems)
  implicit none
  real, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_real

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(*,*) trim(message)
  stop 99
end subroutine wrf_error_fatal

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
end subroutine wrf_message
