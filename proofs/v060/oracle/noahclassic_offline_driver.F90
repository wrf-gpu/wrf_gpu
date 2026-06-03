! ============================================================================
!  WRF Noah-classic (sf_surface_physics=2) offline savepoint driver
!  v0.6.0 lane 14 oracle harness.
! ----------------------------------------------------------------------------
!  Faithful EXTERNAL ORACLE (NOT a self-compare). Links the COMPILED pristine
!  WRF object module_sf_noahlsm.o, reads the real WRF run/ tables via
!  SOIL_VEG_GEN_PARM (so the module-level REDPRM parameter arrays BB/SATDK/
!  MAXSMC/NROTBL/... are populated EXACTLY as in a WRF run), reconstructs the
!  per-cell forcing prep done by module_sf_noahdrv.F:lsm (TH2 via Exner, Q2SAT
!  from QGH-style saturation, DQSDT2, ZLVL, SOLNET, FFROZP), then calls the
!  exact SFLX orchestrator on real Canary d03 land columns under the FROZEN
!  WRF-coupled option set:
!      LOCAL=.false., UA_PHYS=.false., RDLAI2D=.false., USEMONALB=.false.,
!      OPT_THCND=1, FASDAS=0, ICE=0 (non-glacial land), SLOPETYP=1.
!  CH/CM are brought IN from the surface layer (as WRF does; SFLX does NOT call
!  SFCDIF in the coupled path). The IN snapshot + OUT snapshot is a per-column
!  input->output savepoint for the JAX port parity gate (TSK/TSLB/SMOIS/SH2O/
!  HFX/QFX/GRDFLX/...).
!
!  I/O contract (ASCII, paired with build_noahclassic_savepoints.py):
!    - reads  : noahclassic_columns.in
!    - writes : noahclassic_savepoints.out  (TAG value ... lines, per column)
! ============================================================================
program noahclassic_offline_driver
  use module_sf_noahlsm, only : sflx, redprm
  use module_sf_noahdrv, only : soil_veg_gen_parm
  implicit none

  integer, parameter :: NSOIL = 4
  integer, parameter :: IIN = 32, IOUT = 31

  ! per-run configuration
  character(len=64) :: dataset
  integer           :: ncol, slopetyp, soilcolor, isurban, isice, iswater, ic
  real              :: dt
  real              :: sldpth(NSOIL), zsoil(NSOIL)

  ! per-column scalars (read)
  integer :: vegtyp, soiltyp
  real    :: lat, julian, cosz, dx, dz8w, zlvl
  real    :: shdfac, shmin, shmax, tbot
  real    :: sfctmp, sfcprs, psfc, uu, vv, q2k, qc, soldn, glw
  real    :: rainbl, srflag, snoalb1
  real    :: t1, chk, cmk
  real    :: snow_mm, snowhk, sneqv, sncovr, snotime1, ribb
  real    :: albbck_in, z0brd_in, emiss_in
  real    :: stc(NSOIL), smc(NSOIL), swc(NSOIL)
  integer :: yearlen

  ! reconstructed forcing (mirror of module_sf_noahdrv.F:lsm)
  real    :: th2, q2sat, dqsdt2, lwdn, solnet, prcp, ffrozp
  real    :: apes, apelm, sfcth2
  real    :: e2sat, q2sati, sfctsno, qgh
  real    :: capa, cp_loc, rovcp
  real    :: dummy
  integer :: ns
  ! save-in snapshots
  real    :: t1_in, stc_in(NSOIL), smc_in(NSOIL), swc_in(NSOIL)
  real    :: cmk_in, snowhk_in, sneqv_in, sncovr_in

  ! SFLX outputs / inouts
  real :: albedok, embrd, z0brd, z0k, emissi
  real :: eta, sheat, eta_kinematic, fdown
  real :: ec, edir, et(NSOIL), ett, esnow, drip, dew
  real :: beta, etp, ssoil
  real :: flx1, flx2, flx3, flx4, fvb, fbur, fgsn
  real :: snomlt
  real :: runoff1, runoff2, runoff3
  real :: rc, pc, rsmin, xlai, rcs, rct, rcq, rcsoil
  real :: soilw, soilm, q1, smav(NSOIL)
  real :: smcwlt, smcdry, smcref, smcmax
  integer :: nroot
  real :: sfhead1rt, infxs1rt, etpnd1
  real :: aoasis
  real :: xsda_qfx, hfx_phy, qfx_phy, xqnorm, hcpct_fasdas
  integer :: fasdas, opt_thcnd
  logical :: local, ua_phys, rdlai2d, usemonalb
  character(len=256) :: llanduse, lsoil

  real, parameter :: a2 = 17.67, a3 = 273.15, a4 = 29.65
  real, parameter :: a23m4 = a2*(a3-a4)
  real            :: hfx_grid, qfx_grid, grdflx_grid

  ! REDPRM parameter block (dumped per column so the JAX port consumes WRF's
  ! exact derived parameters; the port then implements the physics solve).
  real :: r_cfactr, r_cmcmax, r_rsmax, r_topt, r_refkdt, r_kdt, r_sbeta
  real :: r_shdfac, r_rsmin, r_rgl, r_hs, r_zbot, r_frzx, r_psisat, r_slope
  real :: r_snup, r_salp, r_bexp, r_dksat, r_dwsat, r_smcmax, r_smcwlt
  real :: r_smcref, r_smcdry, r_f1, r_quartz, r_fxexp, r_czil, r_csoil, r_ptu
  real :: r_laimin, r_laimax, r_emissmin, r_emissmax, r_albedomin, r_albedomax
  real :: r_z0min, r_z0max, r_lvcoef, r_ztopv, r_zbotv
  real :: r_rtdis(NSOIL), r_shdfac_io
  real :: res_alb, res_embrd, res_xlai, res_z0
  integer :: r_nroot

  ! WRF model constants (module_model_constants): r_d=287, cp=7*r_d/2
  rovcp  = 287.0/1004.0   ! drv RCP used for column air-temp Exner reconstruction
  cp_loc = 1004.5
  capa   = 287.04/cp_loc  ! drv CAPA = rovcp1 actually uses model rcp; drv uses CAPA=r_d/cp

  ! --------------------------------------------------------------------------
  !  Read configuration
  ! --------------------------------------------------------------------------
  open(IIN, file='noahclassic_columns.in', status='old', form='formatted', action='read')
  read(IIN,*) dataset
  read(IIN,*) ncol, slopetyp, soilcolor, dt, isurban, isice, iswater
  read(IIN,*) (sldpth(ns), ns=1,NSOIL)

  ! Read WRF tables -> populates REDPRM module-level parameter arrays.
  call soil_veg_gen_parm(trim(dataset), 'STAS')

  ! Frozen WRF-coupled option set
  local     = .false.
  ua_phys   = .false.
  rdlai2d   = .false.
  usemonalb = .false.
  opt_thcnd = 1
  fasdas    = 0
  aoasis    = 1.0
  llanduse  = trim(dataset)
  lsoil     = 'STAS'
  capa      = 287.04/cp_loc

  open(IOUT, file='noahclassic_savepoints.out', status='replace', form='formatted', action='write')
  write(IOUT,'(A,I0)') 'NCOL ', ncol

  ! ZSOIL = cumulative negative interface depth from SLDPTH
  zsoil(1) = -sldpth(1)
  do ns = 2, NSOIL
     zsoil(ns) = -sldpth(ns) + zsoil(ns-1)
  end do

  do ic = 1, ncol
     ! ---- read one column record ----
     read(IIN,*) vegtyp, soiltyp
     read(IIN,*) lat, julian, yearlen, cosz, dx, dz8w, zlvl
     read(IIN,*) shdfac, shmin, shmax, tbot
     read(IIN,*) sfctmp, sfcprs, psfc, uu, vv, q2k, qc, soldn, glw
     read(IIN,*) rainbl, srflag, snoalb1
     read(IIN,*) t1, chk, cmk
     read(IIN,*) snow_mm, snowhk, sncovr, snotime1, ribb
     read(IIN,*) albbck_in, z0brd_in, emiss_in
     read(IIN,*) (stc(ns), ns=1,NSOIL)
     read(IIN,*) (smc(ns), ns=1,NSOIL)
     read(IIN,*) (swc(ns), ns=1,NSOIL)

     ! --------------------------------------------------------------------
     !  Reconstruct the per-cell forcing prep from module_sf_noahdrv.F:lsm
     !  (lines ~805-907 of module_sf_noahdrv.F).
     ! --------------------------------------------------------------------
     emissi = emiss_in
     lwdn   = glw * emissi
     solnet = soldn * (1.0 - albbck_in)   ! SOLNET via background albedo (drv uses ALBEDO(I,J))
     prcp   = rainbl / dt

     ! TH2 via Exner (drv: APES/APELM)
     apes   = (1.0e5/psfc)**capa
     apelm  = (1.0e5/sfcprs)**capa
     sfcth2 = sfctmp*apelm
     th2    = sfcth2/apes

     ! Q2SAT (spec humidity) from QGH-style: drv computes QGH then Q2SAT=QGH/(1+QGH).
     ! Reconstruct QGH from saturation over the SKIN temp T1 the way sfclay does:
     !   es = 611.2*exp(17.67*(T1-273.15)/(T1-29.65)); qgh = 0.622*es/(psfc-es).
     e2sat  = 611.2*exp(a2*(t1-a3)/(t1-a4))
     qgh    = 0.622*e2sat/(psfc-e2sat)
     q2sat  = qgh/(1.0+qgh)
     dqsdt2 = q2sat*a23m4/(sfctmp-a4)**2

     ! snow saturation blend (drv block) when snow on surface
     if (snow_mm > 0.0) then
        sfctsno = sfctmp
        e2sat   = 611.2*exp(6174.*(1./273.15 - 1./sfctsno))
        q2sati  = 0.622*e2sat/(sfcprs-e2sat)
        q2sati  = q2sati/(1.0+q2sati)
        if (t1 .gt. 273.14) then
           q2sat  = q2sat*(1.-sncovr) + q2sati*sncovr
           dqsdt2 = dqsdt2*(1.-sncovr) + q2sati*6174./(sfctsno**2)*sncovr
        else
           q2sat  = q2sati
           dqsdt2 = q2sati*6174./(sfctsno**2)
        end if
        if (t1 .gt. 273. .and. sncovr .gt. 0. .and. soldn .gt. 10.) dqsdt2 = dqsdt2*(1.-sncovr)
     end if

     ! FFROZP from 1st-level air temp (no SR plumbed)
     if (sfctmp <= 273.15) then
        ffrozp = 1.0
     else
        ffrozp = 0.0
     end if

     ! convert SNOW (mm) -> SNEQV (m); CMC from CANWAT handled in python (CMK passed in m)
     sneqv  = snow_mm*0.001
     z0brd  = z0brd_in
     embrd  = emiss_in

     ! reset diagnostics
     sfhead1rt = 0.; infxs1rt = 0.; etpnd1 = 0.
     xsda_qfx = 0.; hfx_phy = 0.; qfx_phy = 0.; xqnorm = 0.; hcpct_fasdas = 0.
     xlai = 0.0

     ! --------------------------------------------------------------------
     !  Call REDPRM directly to expose the full derived-parameter block for
     !  this column (same call SFLX makes internally; SHDFAC is INOUT).
     ! --------------------------------------------------------------------
     r_shdfac_io = shdfac
     call redprm (vegtyp, soiltyp, slopetyp, r_cfactr, r_cmcmax, r_rsmax,    &
                  r_topt, r_refkdt, r_kdt, r_sbeta, r_shdfac_io, r_rsmin,    &
                  r_rgl, r_hs, r_zbot, r_frzx, r_psisat, r_slope, r_snup,    &
                  r_salp, r_bexp, r_dksat, r_dwsat, r_smcmax, r_smcwlt,      &
                  r_smcref, r_smcdry, r_f1, r_quartz, r_fxexp, r_rtdis,      &
                  sldpth, zsoil, r_nroot, NSOIL, r_czil, r_laimin, r_laimax, &
                  r_emissmin, r_emissmax, r_albedomin, r_albedomax,          &
                  r_z0min, r_z0max, r_csoil, r_ptu, llanduse, lsoil,         &
                  local, r_lvcoef, r_ztopv, r_zbotv)

     ! Resolve ALB/XLAI/EMBRD/Z0BRD via the SFLX shdfac-interp block (the exact
     ! values SFLX hands to the physics; USEMONALB=.false., RDLAI2D=.false.).
     block
       real :: interp_fraction, alb_r, xlai_r, embrd_r, z0_r
       if (r_shdfac_io >= shmax) then
          embrd_r = r_emissmax; xlai_r = r_laimax; alb_r = r_albedomin; z0_r = r_z0max
       else if (r_shdfac_io <= shmin) then
          embrd_r = r_emissmin; xlai_r = r_laimin; alb_r = r_albedomax; z0_r = r_z0min
       else if (shmax > shmin) then
          interp_fraction = min(max((r_shdfac_io - shmin)/(shmax - shmin), 0.0), 1.0)
          embrd_r = (1.0-interp_fraction)*r_emissmin + interp_fraction*r_emissmax
          xlai_r  = (1.0-interp_fraction)*r_laimin   + interp_fraction*r_laimax
          alb_r   = (1.0-interp_fraction)*r_albedomax + interp_fraction*r_albedomin
          z0_r    = (1.0-interp_fraction)*r_z0min     + interp_fraction*r_z0max
       else
          embrd_r = 0.5*r_emissmin + 0.5*r_emissmax
          xlai_r  = 0.5*r_laimin   + 0.5*r_laimax
          alb_r   = 0.5*r_albedomin + 0.5*r_albedomax
          z0_r    = 0.5*r_z0min    + 0.5*r_z0max
       end if
       albbck_in = alb_r
       z0brd_in  = z0_r
       emiss_in  = embrd_r
       xlai      = xlai_r
       res_alb = alb_r; res_embrd = embrd_r; res_xlai = xlai_r; res_z0 = z0_r
     end block

     ! save IN snapshots (SFLX mutates the prognostics in place)
     t1_in = t1; cmk_in = cmk; snowhk_in = snowhk; sneqv_in = sneqv; sncovr_in = sncovr
     do ns = 1, NSOIL
        stc_in(ns) = stc(ns); smc_in(ns) = smc(ns); swc_in(ns) = swc(ns)
     end do

     ! --------------------------------------------------------------------
     !  CALL SFLX (mirror of module_sf_noahdrv.F:lsm argument order, ICE=0)
     ! --------------------------------------------------------------------
     call sflx (ic, 1, ffrozp, isurban, dt, zlvl, NSOIL, sldpth,        & !C
                local,                                                  & !L
                llanduse, lsoil,                                        & !CL
                lwdn, soldn, solnet, sfcprs, prcp, sfctmp, q2k, dummy,  & !F
                dummy, dummy, dummy,                                    & !F
                th2, q2sat, dqsdt2,                                     & !I
                vegtyp, soiltyp, slopetyp, shdfac, shmin, shmax,        & !I
                albbck_in, snoalb1, tbot, z0brd, z0k, emissi, embrd,    & !S
                cmk, t1, stc, smc, swc, snowhk, sneqv, albedok, chk, cmk,& !H
                eta, sheat, eta_kinematic, fdown,                       & !O
                ec, edir, et, ett, esnow, drip, dew,                    & !O
                beta, etp, ssoil,                                       & !O
                flx1, flx2, flx3,                                       & !O
                flx4, fvb, fbur, fgsn, ua_phys,                         & !UA
                snomlt, sncovr,                                         & !O
                runoff1, runoff2, runoff3,                              & !O
                rc, pc, rsmin, xlai, rcs, rct, rcq, rcsoil,             & !O
                soilw, soilm, q1, smav,                                 & !D
                rdlai2d, usemonalb,                                     &
                snotime1,                                               &
                ribb,                                                   &
                smcwlt, smcdry, smcref, smcmax, nroot,                  &
                sfhead1rt,                                              & !I
                infxs1rt, etpnd1, opt_thcnd, aoasis                     & !P
               ,xsda_qfx, hfx_phy, qfx_phy, xqnorm                      & !fasdas
               ,fasdas, hcpct_fasdas)

     ! grid-level fluxes as module_sf_noahdrv.F:lsm assigns post-SFLX
     hfx_grid    = sheat
     qfx_grid    = eta_kinematic   ! kg m-2 s-1
     grdflx_grid = ssoil

     ! --------------------------------------------------------------------
     !  Write savepoint for this column
     ! --------------------------------------------------------------------
     write(IOUT,'(A)') 'COL'
     write(IOUT,'(A,2I6)') 'CAT ', vegtyp, soiltyp
     write(IOUT,'(A,7ES20.10)') 'FORCING ', sfctmp, sfcprs, q2k, soldn, glw, uu, vv
     write(IOUT,'(A,5ES20.10)') 'FORCING2 ', th2, q2sat, dqsdt2, prcp, solnet
     write(IOUT,'(A,3ES20.10)') 'FORCING3 ', ffrozp, zlvl, lwdn
     write(IOUT,'(A,3ES20.10)') 'CHCM_IN ', chk, cmk_in, ribb
     ! prognostic IN
     write(IOUT,'(A,ES20.10)')  'T1_IN ', t1_in
     write(IOUT,'(A,4ES20.10)') 'STC_IN ', (stc_in(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SMC_IN ', (smc_in(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SH2O_IN ', (swc_in(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SNOW_IN ', sneqv_in, snowhk_in, sncovr_in, cmk_in
     ! prognostic OUT
     write(IOUT,'(A,ES20.10)')  'T1_OUT ', t1
     write(IOUT,'(A,4ES20.10)') 'STC_OUT ', (stc(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SMC_OUT ', (smc(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SH2O_OUT ', (swc(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'SNOW_OUT ', sneqv, snowhk, sncovr, cmk
     ! fluxes / diagnostics
     write(IOUT,'(A,5ES20.10)') 'FLUX ', hfx_grid, qfx_grid, eta, grdflx_grid, etp
     write(IOUT,'(A,5ES20.10)') 'DIAG ', albedok, emissi, z0k, q1, snomlt
     write(IOUT,'(A,5ES20.10)') 'EVAP ', edir, ec, ett, esnow, dew
     write(IOUT,'(A,4ES20.10)') 'FLXX ', flx1, flx2, flx3, beta
     write(IOUT,'(A,3ES20.10)') 'RUNOFF ', runoff1, runoff2, runoff3
     write(IOUT,'(A,5ES20.10)') 'PARM ', smcmax, smcwlt, smcref, smcdry, real(nroot)
     ! full REDPRM derived-parameter block (the JAX port consumes these)
     write(IOUT,'(A,6ES20.10)') 'RP1 ', r_bexp, r_dksat, r_dwsat, r_psisat, r_quartz, r_f1
     write(IOUT,'(A,6ES20.10)') 'RP2 ', r_smcmax, r_smcwlt, r_smcref, r_smcdry, r_kdt, r_frzx
     write(IOUT,'(A,6ES20.10)') 'RP3 ', r_slope, r_snup, r_salp, r_czil, r_sbeta, r_csoil
     write(IOUT,'(A,6ES20.10)') 'RP4 ', r_fxexp, r_zbot, r_cfactr, r_cmcmax, r_rsmax, r_topt
     write(IOUT,'(A,5ES20.10)') 'RP5 ', r_rgl, r_hs, r_rsmin, r_lvcoef, r_shdfac_io
     write(IOUT,'(A,6ES20.10)') 'RPV ', r_laimin, r_laimax, r_emissmin, r_emissmax, r_albedomin, r_albedomax
     write(IOUT,'(A,2ES20.10)') 'RPZ ', r_z0min, r_z0max
     write(IOUT,'(A,I6)')       'RPNROOT ', r_nroot
     write(IOUT,'(A,4ES20.10)') 'RPRTDIS ', (r_rtdis(ns), ns=1,NSOIL)
     write(IOUT,'(A,4ES20.10)') 'RESOLVED ', res_alb, res_embrd, res_xlai, res_z0
     write(IOUT,'(A)') 'ENDCOL'
  end do

  close(IIN)
  close(IOUT)
  write(*,'(A)') 'NOAHCLASSIC_OFFLINE_OK'
end program noahclassic_offline_driver
