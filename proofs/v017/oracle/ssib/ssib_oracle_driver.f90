PROGRAM ssib_oracle
  !---------------------------------------------------------------------------
  ! Standalone fp64 oracle for the WRF SSiB land-surface model
  ! (sf_surface_physics=8).  Calls the UNMODIFIED WRF subroutine SSIB
  ! (phys/module_sf_ssib.F:885) directly, once per land column, replicating
  ! the per-column argument preparation of the WRF surface driver
  ! (module_surface_driver.F:4356 `CALL ssib(...)`, the XLAND<1.5 / XICE<thresh
  ! land branch) for a dry / no-multilayer-snow cold start.
  !
  ! SSiB is self-contained: VEGOUT (module_sf_ssib.F:1995) reads ALL vegetation
  ! and soil parameters from module-level DATA arrays indexed by (ITYPE, MONTH);
  ! there is NO SSIBPARM.TBL / LANDUSE.TBL / SSIB_INIT dependency, no
  ! preprocessor blocks, and no `USE module_model_constants`.  The only external
  ! symbols are wrf_message / wrf_error_fatal / wrf_debug (linker stubs).
  !
  ! The module has NO `IMPLICIT NONE` and NO INTENT on the SSIB dummy args, so
  ! they are typed implicitly: names starting I,J,K,L,M,N are INTEGER, all else
  ! REAL.  Under -fdefault-real-8 every implicit REAL is REAL(8); this driver
  ! declares the matching actual-argument kinds.
  !
  ! Snow cold start: ISNOW=1 (single-layer/no-multilayer snow), SWE=0,
  ! SNOWDEPTH=0, all 4-level snow sub-state arrays 0.  With SNOA(snow)=0 the
  ! ITIME=1 block leaves CAPAC small and keeps ISNOW=1, so the multi-layer snow
  ! solver (TEMRS2/SNOW_1ST/LAYERN) is NOT exercised; the TEMRS1 single-layer
  ! path runs.
  !
  ! sw_physics=1  => SSIB uses the empirical CLOUD->RADFRAC split (the SALBxx /
  ! RADFRACxx 4-component inputs are then overwritten internally), so we feed
  ! representative values and let SSIB recompute.
  !
  ! Emits flat `VAR[i]=value` lines (inputs + INOUT carry + outputs) for parsing
  ! into JSON savepoints.
  !---------------------------------------------------------------------------
  USE module_sf_ssib, ONLY : SSIB
  IMPLICIT NONE

  ! fp64 working kind for all REAL actuals handed to SSIB (matches -fdefault-real-8).
  INTEGER, PARAMETER :: WP = KIND(1.0)          ! becomes 8 under -fdefault-real-8
  INTEGER, PARAMETER :: N  = 5                  ! columns per case

  CHARACTER(LEN=4) :: MMINLU = 'USGS'           ! USGS landuse -> IVUSGS mapping

  ! ---- Per-column scalar inputs (the SSIB argument set, land branch) ----------
  REAL(WP), DIMENSION(N) :: DDTT_A, ZLAT_A, SUNANGLE_A, PPL_A, PPC_A, RLWDOWN_A
  REAL(WP), DIMENSION(N) :: ZWIND2_A, UMM_A, VMM_A, QM_A, TM_A, PM_A, PSUR_A
  REAL(WP), DIMENSION(N) :: SWDOWN1_A, DAY_A, CLOUD_A
  REAL(WP), DIMENSION(N) :: SALB11_A, SALB12_A, SALB21_A, SALB22_A
  REAL(WP), DIMENSION(N) :: RADFRAC11_A, RADFRAC12_A, RADFRAC21_A, RADFRAC22_A
  INTEGER,  DIMENSION(N) :: IVGTYP_A, ITIME_A, SW_PHYSICS_A

  ! ---- INOUT carry: initial values saved + working copies --------------------
  REAL(WP), DIMENSION(N) :: WWW1_A, WWW2_A, WWW3_A, TC_A, TGS_A, TD_A
  REAL(WP), DIMENSION(N) :: SNOA_A, ROFF_A, SNOB_A, TA_A
  REAL(WP), DIMENSION(N) :: WWW1_0, WWW2_0, WWW3_0, TC_0, TGS_0, TD_0
  REAL(WP), DIMENSION(N) :: SNOA_0, SNOB_0, TA_0
  ! albedo carry (INOUT in driver; passed as BEDO)
  REAL(WP), DIMENSION(N) :: BEDO_A, BEDO_0

  ! ---- OUTPUT scalars (post-call) --------------------------------------------
  REAL(WP), DIMENSION(N) :: XHSFLX_A, ELATEN_A, GHTFLX_A, XHLFLX_A, TGEFF_A
  REAL(WP), DIMENSION(N) :: USTAR_A, RIB_A, FM_A, FH_A, CM_A
  REAL(WP), DIMENSION(N) :: XLHF_A, XSHF_A, XGHF_A, XEGS_A, XECI_A, XECT_A
  REAL(WP), DIMENSION(N) :: XEGI_A, XEGT_A, XSDN_A, XSUP_A, XLDN_A, XLUP_A
  REAL(WP), DIMENSION(N) :: XWAT_A, XHCX_A, XHGX_A, XZLT_A, XVCF_A, XXZ0_A
  REAL(WP), DIMENSION(N) :: XVEG_A, XDD_A
  REAL(WP), DIMENSION(N) :: Q2M_A, UV10_A

  ! ---- 4-level snow sub-state (cold start = 0; INOUT) ------------------------
  INTEGER,  DIMENSION(N) :: ISNOW_A
  REAL(WP), DIMENSION(N) :: SWE_A, SNOWDEN_A, SNOWDEPTH_A, TKAIR_A
  REAL(WP), DIMENSION(N) :: DZO1_A,WO1_A,TSSN1_A,TSSNO1_A,BWO1_A,BTO1_A,CTO1_A,FIO1_A,FLO1_A,BIO1_A,BLO1_A,HO1_A
  REAL(WP), DIMENSION(N) :: DZO2_A,WO2_A,TSSN2_A,TSSNO2_A,BWO2_A,BTO2_A,CTO2_A,FIO2_A,FLO2_A,BIO2_A,BLO2_A,HO2_A
  REAL(WP), DIMENSION(N) :: DZO3_A,WO3_A,TSSN3_A,TSSNO3_A,BWO3_A,BTO3_A,CTO3_A,FIO3_A,FLO3_A,BIO3_A,BLO3_A,HO3_A
  REAL(WP), DIMENSION(N) :: DZO4_A,WO4_A,TSSN4_A,TSSNO4_A,BWO4_A,BTO4_A,CTO4_A,FIO4_A,FLO4_A,BIO4_A,BLO4_A,HO4_A
  ! saved initial snow carry (for the cold-start ISNOW)
  INTEGER,  DIMENSION(N) :: ISNOW_0

  INTEGER :: case_id, i
  CHARACTER(LEN=32)  :: arg, precision_mode
  CHARACTER(LEN=64)  :: regime_name

  ! scratch SSIB INOUT/OUTPUT scalars for one column call
  REAL(WP) :: io_www1, io_www2, io_www3, io_tc, io_tgs, io_td
  REAL(WP) :: io_snoa, io_roff, io_snob, io_ta, io_bedo
  REAL(WP) :: o_xhsflx, o_elaten, o_ghtflx, o_xhlflx, o_tgeff
  REAL(WP) :: o_ustar, o_rib, o_fm, o_fh, o_cm
  REAL(WP) :: o_xlhf, o_xshf, o_xghf, o_xegs, o_xeci, o_xect
  REAL(WP) :: o_xegi, o_xegt, o_xsdn, o_xsup, o_xldn, o_xlup
  REAL(WP) :: o_xwat, o_xhcx, o_xhgx, o_xzlt, o_xvcf, o_xxz0
  REAL(WP) :: o_xveg, o_xdd, o_q2m, o_uv10
  REAL(WP) :: io_salb11, io_salb12, io_salb21, io_salb22
  REAL(WP) :: io_rf11, io_rf12, io_rf21, io_rf22, io_cloud
  INTEGER  :: io_isnow
  REAL(WP) :: io_swe, io_snowden, io_snowdepth, io_tkair
  REAL(WP) :: dzo1,wo1,tssn1,tssno1,bwo1,bto1,cto1,fio1,flo1,bio1,blo1,ho1
  REAL(WP) :: dzo2,wo2,tssn2,tssno2,bwo2,bto2,cto2,fio2,flo2,bio2,blo2,ho2
  REAL(WP) :: dzo3,wo3,tssn3,tssno3,bwo3,bto3,cto3,fio3,flo3,bio3,blo3,ho3
  REAL(WP) :: dzo4,wo4,tssn4,tssno4,bwo4,bto4,cto4,fio4,flo4,bio4,blo4,ho4

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
     CALL GET_COMMAND_ARGUMENT(1, arg)
     READ(arg,*) case_id
  ELSE
     case_id = 1
  END IF
  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN
     CALL GET_COMMAND_ARGUMENT(2, precision_mode)
  ELSE
     precision_mode = 'fp64'
  END IF

  CALL build_case(case_id, regime_name)

  !-- Per-column SSIB driver (mirrors module_surface_driver.F:4356 land branch) -
  DO i = 1, N
     ! init the SSIB scratch INOUT carry from the case state
     io_www1 = WWW1_A(i) ; io_www2 = WWW2_A(i) ; io_www3 = WWW3_A(i)
     io_tc   = TC_A(i)   ; io_tgs  = TGS_A(i)  ; io_td   = TD_A(i)
     io_snoa = SNOA_A(i) ; io_roff = ROFF_A(i) ; io_snob = SNOB_A(i)
     io_ta   = TA_A(i)   ; io_bedo = BEDO_A(i)
     io_salb11 = SALB11_A(i) ; io_salb12 = SALB12_A(i)
     io_salb21 = SALB21_A(i) ; io_salb22 = SALB22_A(i)
     io_rf11 = RADFRAC11_A(i) ; io_rf12 = RADFRAC12_A(i)
     io_rf21 = RADFRAC21_A(i) ; io_rf22 = RADFRAC22_A(i)
     io_cloud = CLOUD_A(i)

     io_isnow = ISNOW_A(i) ; io_swe = SWE_A(i)
     io_snowden = SNOWDEN_A(i) ; io_snowdepth = SNOWDEPTH_A(i)
     io_tkair = TKAIR_A(i)
     dzo1=DZO1_A(i);wo1=WO1_A(i);tssn1=TSSN1_A(i);tssno1=TSSNO1_A(i);bwo1=BWO1_A(i);bto1=BTO1_A(i)
     cto1=CTO1_A(i);fio1=FIO1_A(i);flo1=FLO1_A(i);bio1=BIO1_A(i);blo1=BLO1_A(i);ho1=HO1_A(i)
     dzo2=DZO2_A(i);wo2=WO2_A(i);tssn2=TSSN2_A(i);tssno2=TSSNO2_A(i);bwo2=BWO2_A(i);bto2=BTO2_A(i)
     cto2=CTO2_A(i);fio2=FIO2_A(i);flo2=FLO2_A(i);bio2=BIO2_A(i);blo2=BLO2_A(i);ho2=HO2_A(i)
     dzo3=DZO3_A(i);wo3=WO3_A(i);tssn3=TSSN3_A(i);tssno3=TSSNO3_A(i);bwo3=BWO3_A(i);bto3=BTO3_A(i)
     cto3=CTO3_A(i);fio3=FIO3_A(i);flo3=FLO3_A(i);bio3=BIO3_A(i);blo3=BLO3_A(i);ho3=HO3_A(i)
     dzo4=DZO4_A(i);wo4=WO4_A(i);tssn4=TSSN4_A(i);tssno4=TSSNO4_A(i);bwo4=BWO4_A(i);bto4=BTO4_A(i)
     cto4=CTO4_A(i);fio4=FIO4_A(i);flo4=FLO4_A(i);bio4=BIO4_A(i);blo4=BLO4_A(i);ho4=HO4_A(i)

     CALL SSIB( i, 1, DDTT_A(i), ITIME_A(i), ZLAT_A(i), SUNANGLE_A(i),     &  ! in
                PPL_A(i), PPC_A(i), RLWDOWN_A(i), ZWIND2_A(i),             &  ! in
                io_www1, io_www2, io_www3,                                 &  ! inout soil moisture
                io_tc, io_tgs, io_td,                                      &  ! inout temps
                io_snoa, io_roff,                                          &  ! inout snow/runoff
                UMM_A(i), VMM_A(i), QM_A(i), TM_A(i),                      &  ! in forcing
                PM_A(i), PSUR_A(i), IVGTYP_A(i),                           &  ! in
                SWDOWN1_A(i), io_snob,                                     &  ! in SW, inout canwat
                io_salb11, io_salb12, io_salb21, io_salb22,                &  ! inout 4-comp albedo
                io_rf11, io_rf12, io_rf21, io_rf22,                        &  ! in/inout radfrac
                o_xhsflx, o_elaten, o_ghtflx, o_xhlflx, o_tgeff,           &  ! out hfx/lh/grd/qfx/tsk
                o_ustar, o_rib, o_fm, o_fh, o_cm,                          &  ! out ust/br/fm/fh/cm
                o_xlhf, o_xshf, o_xghf, o_xegs, o_xeci, o_xect,            &  ! out
                o_xegi, o_xegt, o_xsdn, o_xsup, o_xldn, o_xlup,            &  ! out
                o_xwat, o_xhcx, o_xhgx, o_xzlt, o_xvcf, o_xxz0,            &  ! out
                o_xveg, o_xdd,                                             &  ! out
                io_isnow, io_swe, io_snowden, io_snowdepth, io_tkair,      &  ! snow
                dzo1,wo1,tssn1,tssno1,bwo1,bto1,cto1,fio1,flo1,bio1,blo1,ho1, &  ! snow
                dzo2,wo2,tssn2,tssno2,bwo2,bto2,cto2,fio2,flo2,bio2,blo2,ho2, &  ! snow
                dzo3,wo3,tssn3,tssno3,bwo3,bto3,cto3,fio3,flo3,bio3,blo3,ho3, &  ! snow
                dzo4,wo4,tssn4,tssno4,bwo4,bto4,cto4,fio4,flo4,bio4,blo4,ho4, &  ! snow
                DAY_A(i), io_cloud, o_q2m, io_ta, io_bedo, o_uv10,         &  ! day/cloud/q2/ta/albedo/uv10
                SW_PHYSICS_A(i), MMINLU )                                     ! sw choice, landuse map

     !-- store outputs ---------------------------------------------------------
     XHSFLX_A(i)=o_xhsflx ; ELATEN_A(i)=o_elaten ; GHTFLX_A(i)=o_ghtflx
     XHLFLX_A(i)=o_xhlflx ; TGEFF_A(i)=o_tgeff
     USTAR_A(i)=o_ustar ; RIB_A(i)=o_rib ; FM_A(i)=o_fm ; FH_A(i)=o_fh ; CM_A(i)=o_cm
     XLHF_A(i)=o_xlhf ; XSHF_A(i)=o_xshf ; XGHF_A(i)=o_xghf
     XEGS_A(i)=o_xegs ; XECI_A(i)=o_xeci ; XECT_A(i)=o_xect
     XEGI_A(i)=o_xegi ; XEGT_A(i)=o_xegt
     XSDN_A(i)=o_xsdn ; XSUP_A(i)=o_xsup ; XLDN_A(i)=o_xldn ; XLUP_A(i)=o_xlup
     XWAT_A(i)=o_xwat ; XHCX_A(i)=o_xhcx ; XHGX_A(i)=o_xhgx
     XZLT_A(i)=o_xzlt ; XVCF_A(i)=o_xvcf ; XXZ0_A(i)=o_xxz0
     XVEG_A(i)=o_xveg ; XDD_A(i)=o_xdd
     Q2M_A(i)=o_q2m ; UV10_A(i)=o_uv10
     ! updated INOUT carry
     WWW1_A(i)=io_www1 ; WWW2_A(i)=io_www2 ; WWW3_A(i)=io_www3
     TC_A(i)=io_tc ; TGS_A(i)=io_tgs ; TD_A(i)=io_td
     SNOA_A(i)=io_snoa ; ROFF_A(i)=io_roff ; SNOB_A(i)=io_snob
     TA_A(i)=io_ta ; BEDO_A(i)=io_bedo
     ISNOW_A(i)=io_isnow ; SWE_A(i)=io_swe
     SNOWDEN_A(i)=io_snowden ; SNOWDEPTH_A(i)=io_snowdepth
     ! capture updated 4-comp albedo carry
     SALB11_A(i)=io_salb11 ; SALB12_A(i)=io_salb12
     SALB21_A(i)=io_salb21 ; SALB22_A(i)=io_salb22
  END DO

  !-- Emit savepoint -----------------------------------------------------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,A)')  'REGIME_NAME=', TRIM(regime_name)
  WRITE(*,'(A,A)')  'PRECISION_MODE=', TRIM(precision_mode)
  WRITE(*,'(A,I0)') 'N=', N
  WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0

  DO i = 1, N
     ! --- INPUTS (as fed to SSIB) ---
     CALL dump('DDTT',     i, DDTT_A(i))
     CALL dumpi('ITIME',   i, ITIME_A(i))
     CALL dump('ZLAT',     i, ZLAT_A(i))
     CALL dump('SUNANGLE', i, SUNANGLE_A(i))
     CALL dump('PPL',      i, PPL_A(i))
     CALL dump('PPC',      i, PPC_A(i))
     CALL dump('RLWDOWN',  i, RLWDOWN_A(i))
     CALL dump('ZWIND2',   i, ZWIND2_A(i))
     CALL dump('UMM',      i, UMM_A(i))
     CALL dump('VMM',      i, VMM_A(i))
     CALL dump('QM',       i, QM_A(i))
     CALL dump('TM',       i, TM_A(i))
     CALL dump('PM',       i, PM_A(i))
     CALL dump('PSUR',     i, PSUR_A(i))
     CALL dumpi('IVGTYP',  i, IVGTYP_A(i))
     CALL dump('SWDOWN1',  i, SWDOWN1_A(i))
     CALL dump('DAY',      i, DAY_A(i))
     CALL dumpi('SW_PHYSICS', i, SW_PHYSICS_A(i))
     ! --- INITIAL INOUT carry (kernel needs these as inputs) ---
     CALL dump('WWW1_IN',  i, WWW1_0(i))
     CALL dump('WWW2_IN',  i, WWW2_0(i))
     CALL dump('WWW3_IN',  i, WWW3_0(i))
     CALL dump('TC_IN',    i, TC_0(i))
     CALL dump('TGS_IN',   i, TGS_0(i))
     CALL dump('TD_IN',    i, TD_0(i))
     CALL dump('TA_IN',    i, TA_0(i))
     CALL dump('SNOA_IN',  i, SNOA_0(i))
     CALL dump('SNOB_IN',  i, SNOB_0(i))
     CALL dump('BEDO_IN',  i, BEDO_0(i))
     CALL dumpi('ISNOW_IN',i, ISNOW_0(i))
     CALL dump('CLOUD_IN', i, CLOUD_A(i))
     ! --- OUTPUTS (fluxes / diagnostics) ---
     CALL dump('HFX',      i, XHSFLX_A(i))   ! XHSFLX -> hfx
     CALL dump('LH',       i, ELATEN_A(i))   ! ELATEN -> lh
     CALL dump('GRDFLX',   i, GHTFLX_A(i))   ! GHTFLX -> grdflx
     CALL dump('QFX',      i, XHLFLX_A(i))   ! XHLFLX -> qfx
     CALL dump('TGEFF',    i, TGEFF_A(i))    ! TGEFF  -> tsk
     CALL dump('USTAR',    i, USTAR_A(i))
     CALL dump('RIB',      i, RIB_A(i))
     CALL dump('FM',       i, FM_A(i))
     CALL dump('FH',       i, FH_A(i))
     CALL dump('CM',       i, CM_A(i))
     CALL dump('XLHF',     i, XLHF_A(i))
     CALL dump('XSHF',     i, XSHF_A(i))
     CALL dump('XGHF',     i, XGHF_A(i))
     CALL dump('XEGS',     i, XEGS_A(i))
     CALL dump('XECI',     i, XECI_A(i))
     CALL dump('XECT',     i, XECT_A(i))
     CALL dump('XEGI',     i, XEGI_A(i))
     CALL dump('XEGT',     i, XEGT_A(i))
     CALL dump('XSDN',     i, XSDN_A(i))
     CALL dump('XSUP',     i, XSUP_A(i))
     CALL dump('XLDN',     i, XLDN_A(i))
     CALL dump('XLUP',     i, XLUP_A(i))
     CALL dump('XWAT',     i, XWAT_A(i))
     CALL dump('XHCX',     i, XHCX_A(i))
     CALL dump('XHGX',     i, XHGX_A(i))
     CALL dump('XZLT',     i, XZLT_A(i))
     CALL dump('XVCF',     i, XVCF_A(i))
     CALL dump('XXZ0',     i, XXZ0_A(i))
     CALL dump('XVEG',     i, XVEG_A(i))
     CALL dump('XDD',      i, XDD_A(i))
     CALL dump('Q2M',      i, Q2M_A(i))
     CALL dump('UV10',     i, UV10_A(i))
     CALL dump('SALB11',   i, SALB11_A(i))
     CALL dump('SALB12',   i, SALB12_A(i))
     CALL dump('SALB21',   i, SALB21_A(i))
     CALL dump('SALB22',   i, SALB22_A(i))
     CALL dump('BEDO',     i, BEDO_A(i))
     ! --- UPDATED INOUT carry (state out) ---
     CALL dump('WWW1',     i, WWW1_A(i))
     CALL dump('WWW2',     i, WWW2_A(i))
     CALL dump('WWW3',     i, WWW3_A(i))
     CALL dump('TC',       i, TC_A(i))
     CALL dump('TGS',      i, TGS_A(i))
     CALL dump('TD',       i, TD_A(i))
     CALL dump('TA',       i, TA_A(i))
     CALL dump('SNOA',     i, SNOA_A(i))
     CALL dump('SNOB',     i, SNOB_A(i))
     CALL dump('ROFF',     i, ROFF_A(i))
     CALL dumpi('ISNOW',   i, ISNOW_A(i))
     CALL dump('SWE',      i, SWE_A(i))
     CALL dump('SNOWDEN',  i, SNOWDEN_A(i))
     CALL dump('SNOWDEPTH',i, SNOWDEPTH_A(i))
  END DO

CONTAINS

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL(WP), INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE dumpi(name, idx, ivalue)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx, ivalue
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', REAL(ivalue, WP)
  END SUBROUTINE dumpi

  ! Set one land column.  PSUR in Pa (converted to mb internally by SSIB),
  ! PM the full-level pressure (Pa), precip in mm/step, ZWIND2 = full lowest
  ! model-layer thickness (m).  WWW1/2/3 are volumetric soil moisture (m3/m3).
  SUBROUTINE setcol(idx, ddtt, itime, zlat, sunangle, ppl, ppc, rlwdown,    &
                    zwind2, umm, vmm, qm, tm, pm, psur, ivgtyp, swdown1,    &
                    day, cloud, sw_phys,                                    &
                    www1, www2, www3, tc, tgs, td, ta, snoa, snob, bedo,    &
                    roff)
    INTEGER,  INTENT(IN) :: idx, itime, ivgtyp, sw_phys
    REAL(WP), INTENT(IN) :: ddtt, zlat, sunangle, ppl, ppc, rlwdown
    REAL(WP), INTENT(IN) :: zwind2, umm, vmm, qm, tm, pm, psur, swdown1
    REAL(WP), INTENT(IN) :: day, cloud
    REAL(WP), INTENT(IN) :: www1, www2, www3, tc, tgs, td, ta, snoa, snob, bedo, roff

    DDTT_A(idx)=ddtt ; ITIME_A(idx)=itime ; ZLAT_A(idx)=zlat
    SUNANGLE_A(idx)=sunangle ; PPL_A(idx)=ppl ; PPC_A(idx)=ppc
    RLWDOWN_A(idx)=rlwdown ; ZWIND2_A(idx)=zwind2
    UMM_A(idx)=umm ; VMM_A(idx)=vmm ; QM_A(idx)=qm ; TM_A(idx)=tm
    PM_A(idx)=pm ; PSUR_A(idx)=psur ; IVGTYP_A(idx)=ivgtyp
    SWDOWN1_A(idx)=swdown1 ; DAY_A(idx)=day ; SW_PHYSICS_A(idx)=sw_phys
    CLOUD_A(idx)=cloud
    ! representative 4-component albedo / radfrac inputs (overwritten internally
    ! for sw_physics=1 path, but must be finite).
    SALB11_A(idx)=0.10_WP ; SALB12_A(idx)=0.10_WP ; SALB21_A(idx)=0.20_WP ; SALB22_A(idx)=0.20_WP
    RADFRAC11_A(idx)=0.30_WP ; RADFRAC12_A(idx)=0.10_WP
    RADFRAC21_A(idx)=0.40_WP ; RADFRAC22_A(idx)=0.20_WP
    ! INOUT carry + saved initials
    WWW1_A(idx)=www1 ; WWW1_0(idx)=www1
    WWW2_A(idx)=www2 ; WWW2_0(idx)=www2
    WWW3_A(idx)=www3 ; WWW3_0(idx)=www3
    TC_A(idx)=tc ; TC_0(idx)=tc
    TGS_A(idx)=tgs ; TGS_0(idx)=tgs
    TD_A(idx)=td ; TD_0(idx)=td
    TA_A(idx)=ta ; TA_0(idx)=ta
    SNOA_A(idx)=snoa ; SNOA_0(idx)=snoa
    SNOB_A(idx)=snob ; SNOB_0(idx)=snob
    BEDO_A(idx)=bedo ; BEDO_0(idx)=bedo
    ROFF_A(idx)=roff
    ! 4-level snow cold start: ISNOW=1 (single layer / no multilayer snow), all 0
    ISNOW_A(idx)=1 ; ISNOW_0(idx)=1
    SWE_A(idx)=0.0_WP ; SNOWDEN_A(idx)=0.0_WP ; SNOWDEPTH_A(idx)=0.0_WP ; TKAIR_A(idx)=tm
    CALL zero_snow_layers(idx)
  END SUBROUTINE setcol

  SUBROUTINE zero_snow_layers(idx)
    INTEGER, INTENT(IN) :: idx
    DZO1_A(idx)=0;WO1_A(idx)=0;TSSN1_A(idx)=0;TSSNO1_A(idx)=0;BWO1_A(idx)=0;BTO1_A(idx)=0
    CTO1_A(idx)=0;FIO1_A(idx)=0;FLO1_A(idx)=0;BIO1_A(idx)=0;BLO1_A(idx)=0;HO1_A(idx)=0
    DZO2_A(idx)=0;WO2_A(idx)=0;TSSN2_A(idx)=0;TSSNO2_A(idx)=0;BWO2_A(idx)=0;BTO2_A(idx)=0
    CTO2_A(idx)=0;FIO2_A(idx)=0;FLO2_A(idx)=0;BIO2_A(idx)=0;BLO2_A(idx)=0;HO2_A(idx)=0
    DZO3_A(idx)=0;WO3_A(idx)=0;TSSN3_A(idx)=0;TSSNO3_A(idx)=0;BWO3_A(idx)=0;BTO3_A(idx)=0
    CTO3_A(idx)=0;FIO3_A(idx)=0;FLO3_A(idx)=0;BIO3_A(idx)=0;BLO3_A(idx)=0;HO3_A(idx)=0
    DZO4_A(idx)=0;WO4_A(idx)=0;TSSN4_A(idx)=0;TSSNO4_A(idx)=0;BWO4_A(idx)=0;BTO4_A(idx)=0
    CTO4_A(idx)=0;FIO4_A(idx)=0;FLO4_A(idx)=0;BIO4_A(idx)=0;BLO4_A(idx)=0;HO4_A(idx)=0
  END SUBROUTINE zero_snow_layers

  SUBROUTINE build_case(cid, name)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    REAL(WP), PARAMETER :: DT = 180.0_WP        ! surface timestep (s)
    REAL(WP), PARAMETER :: DAYJ = 196.0_WP      ! mid-July (NH summer)
    ! USGS veg types -> IVUSGS -> SSiB ITYPE (1..13):
    !  IVGTYP=2 (dryland crop) -> IVUSGS(2)=12 (crop)
    !  IVGTYP=7 (grassland)    -> IVUSGS(7)=7  (grass)
    !  IVGTYP=10 (savanna)     -> IVUSGS(10)=6
    !  IVGTYP=11 (decid broad) -> IVUSGS(11)=2 (broadleaf)
    !  IVGTYP=14 (evergreen ndl)-> IVUSGS(14)=4
    INTEGER, PARAMETER :: VG_GRASS=7, VG_CROP=2, VG_BROAD=11, VG_SAVAN=10, VG_NEEDL=14
    SELECT CASE (cid)
    CASE (1)
       name = 'unstable_day_land'
       !          idx ddtt itime zlat  sunang ppl  ppc  rlwdn  zwind2 umm  vmm  qm      tm     pm       psur     ivg      swdn  day    cloud sw   www1  www2  www3  tc     tgs    td     ta     snoa snob bedo  roff
       CALL setcol(1, DT, 1, 35.0_WP, 0.85_WP,0._WP,0._WP,350._WP,60._WP,3.0_WP,1.5_WP,0.010_WP,300._WP,98000._WP,99500._WP, VG_GRASS,800._WP,DAYJ,0.3_WP,1, 0.25_WP,0.27_WP,0.28_WP,301._WP,302._WP,299._WP,300._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(2, DT, 1, 30.0_WP, 0.90_WP,0._WP,0._WP,360._WP,55._WP,4.0_WP,2.0_WP,0.012_WP,303._WP,96000._WP,97000._WP, VG_CROP, 850._WP,DAYJ,0.2_WP,1, 0.22_WP,0.24_WP,0.26_WP,304._WP,305._WP,301._WP,303._WP,0._WP,0._WP,0.17_WP,0._WP)
       CALL setcol(3, DT, 1, 40.0_WP, 0.75_WP,0._WP,0._WP,345._WP,65._WP,2.5_WP,1.0_WP,0.009_WP,298._WP,92000._WP,93000._WP, VG_BROAD,720._WP,DAYJ,0.4_WP,1, 0.30_WP,0.31_WP,0.32_WP,299._WP,300._WP,297._WP,298._WP,0._WP,0._WP,0.16_WP,0._WP)
       CALL setcol(4, DT, 1, 25.0_WP, 0.95_WP,0._WP,0._WP,365._WP,50._WP,5.0_WP,2.5_WP,0.013_WP,305._WP,100500._WP,101000._WP,VG_SAVAN,880._WP,DAYJ,0.1_WP,1, 0.18_WP,0.20_WP,0.22_WP,306._WP,307._WP,303._WP,305._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(5, DT, 1, 45.0_WP, 0.70_WP,0._WP,0._WP,340._WP,70._WP,3.5_WP,1.5_WP,0.008_WP,296._WP,90000._WP,91000._WP, VG_NEEDL,680._WP,DAYJ,0.5_WP,1, 0.28_WP,0.29_WP,0.30_WP,297._WP,298._WP,295._WP,296._WP,0._WP,0._WP,0.15_WP,0._WP)
    CASE (2)
       name = 'stable_night_land'
       !  night: SUNANGLE ~ 0 (clamped to 0.01746 internally), SWDOWN1 ~ 0
       CALL setcol(1, DT, 1, 35.0_WP, 0.0_WP, 0._WP,0._WP,300._WP,60._WP,1.0_WP,0.5_WP,0.006_WP,288._WP,98000._WP,99500._WP, VG_GRASS,0.1_WP,DAYJ,0.2_WP,1, 0.25_WP,0.27_WP,0.28_WP,286._WP,285._WP,290._WP,287._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(2, DT, 1, 30.0_WP, 0.0_WP, 0._WP,0._WP,290._WP,55._WP,1.5_WP,0.8_WP,0.005_WP,291._WP,96000._WP,97000._WP, VG_CROP, 0.1_WP,DAYJ,0.1_WP,1, 0.22_WP,0.24_WP,0.26_WP,289._WP,288._WP,293._WP,290._WP,0._WP,0._WP,0.17_WP,0._WP)
       CALL setcol(3, DT, 1, 40.0_WP, 0.0_WP, 0._WP,0._WP,285._WP,65._WP,0.8_WP,0.4_WP,0.004_WP,284._WP,92000._WP,93000._WP, VG_BROAD,0.1_WP,DAYJ,0.3_WP,1, 0.30_WP,0.31_WP,0.32_WP,282._WP,281._WP,286._WP,283._WP,0._WP,0._WP,0.16_WP,0._WP)
       CALL setcol(4, DT, 1, 25.0_WP, 0.0_WP, 0._WP,0._WP,305._WP,50._WP,2.0_WP,1.0_WP,0.007_WP,293._WP,100500._WP,101000._WP,VG_SAVAN,0.1_WP,DAYJ,0.1_WP,1, 0.18_WP,0.20_WP,0.22_WP,291._WP,290._WP,295._WP,292._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(5, DT, 1, 45.0_WP, 0.0_WP, 0._WP,0._WP,295._WP,70._WP,1.2_WP,0.6_WP,0.005_WP,290._WP,90000._WP,91000._WP, VG_NEEDL,0.1_WP,DAYJ,0.4_WP,1, 0.28_WP,0.29_WP,0.30_WP,288._WP,287._WP,292._WP,289._WP,0._WP,0._WP,0.15_WP,0._WP)
    CASE (3)
       name = 'wet_precip_land'
       !  precip in mm/step (ppl large-scale, ppc convective)
       CALL setcol(1, DT, 1, 35.0_WP, 0.60_WP,0.5_WP,0.2_WP,360._WP,60._WP,2.0_WP,1.0_WP,0.014_WP,295._WP,98000._WP,99500._WP, VG_GRASS,300._WP,DAYJ,0.9_WP,1, 0.34_WP,0.35_WP,0.36_WP,296._WP,296._WP,294._WP,295._WP,0._WP,0._WP,0.16_WP,0._WP)
       CALL setcol(2, DT, 1, 30.0_WP, 0.55_WP,0.8_WP,0.4_WP,358._WP,55._WP,2.5_WP,1.2_WP,0.015_WP,296._WP,96000._WP,97000._WP, VG_CROP, 250._WP,DAYJ,0.95_WP,1,0.36_WP,0.37_WP,0.38_WP,297._WP,297._WP,295._WP,296._WP,0._WP,0._WP,0.15_WP,0._WP)
       CALL setcol(3, DT, 1, 40.0_WP, 0.50_WP,1.0_WP,0.6_WP,355._WP,65._WP,1.5_WP,0.8_WP,0.013_WP,293._WP,92000._WP,93000._WP, VG_BROAD,200._WP,DAYJ,1.0_WP,1, 0.38_WP,0.39_WP,0.40_WP,294._WP,294._WP,292._WP,293._WP,0._WP,0._WP,0.14_WP,0._WP)
       CALL setcol(4, DT, 1, 25.0_WP, 0.65_WP,0.3_WP,0.5_WP,362._WP,50._WP,3.0_WP,1.5_WP,0.015_WP,297._WP,100500._WP,101000._WP,VG_SAVAN,350._WP,DAYJ,0.85_WP,1,0.40_WP,0.40_WP,0.41_WP,298._WP,298._WP,296._WP,297._WP,0._WP,0._WP,0.16_WP,0._WP)
       CALL setcol(5, DT, 1, 45.0_WP, 0.45_WP,0.6_WP,0.3_WP,359._WP,70._WP,1.8_WP,0.9_WP,0.016_WP,298._WP,90000._WP,91000._WP, VG_NEEDL,280._WP,DAYJ,0.9_WP,1, 0.36_WP,0.38_WP,0.39_WP,299._WP,299._WP,297._WP,298._WP,0._WP,0._WP,0.14_WP,0._WP)
    CASE (4)
       name = 'near_neutral_land'
       CALL setcol(1, DT, 1, 35.0_WP, 0.40_WP,0._WP,0._WP,330._WP,60._WP,4.0_WP,2.0_WP,0.009_WP,292._WP,98000._WP,99500._WP, VG_GRASS,300._WP,DAYJ,0.5_WP,1, 0.25_WP,0.27_WP,0.28_WP,292._WP,292._WP,291._WP,292._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(2, DT, 1, 30.0_WP, 0.45_WP,0._WP,0._WP,325._WP,55._WP,5.0_WP,2.5_WP,0.010_WP,294._WP,96000._WP,97000._WP, VG_CROP, 280._WP,DAYJ,0.5_WP,1, 0.22_WP,0.24_WP,0.26_WP,294._WP,294._WP,293._WP,294._WP,0._WP,0._WP,0.17_WP,0._WP)
       CALL setcol(3, DT, 1, 40.0_WP, 0.35_WP,0._WP,0._WP,320._WP,65._WP,3.5_WP,1.8_WP,0.008_WP,289._WP,92000._WP,93000._WP, VG_BROAD,250._WP,DAYJ,0.6_WP,1, 0.30_WP,0.31_WP,0.32_WP,289._WP,289._WP,288._WP,289._WP,0._WP,0._WP,0.16_WP,0._WP)
       CALL setcol(4, DT, 1, 25.0_WP, 0.50_WP,0._WP,0._WP,335._WP,50._WP,6.0_WP,3.0_WP,0.011_WP,296._WP,100500._WP,101000._WP,VG_SAVAN,320._WP,DAYJ,0.4_WP,1, 0.18_WP,0.20_WP,0.22_WP,296._WP,296._WP,295._WP,296._WP,0._WP,0._WP,0.18_WP,0._WP)
       CALL setcol(5, DT, 1, 45.0_WP, 0.30_WP,0._WP,0._WP,332._WP,70._WP,4.5_WP,2.2_WP,0.009_WP,300._WP,90000._WP,91000._WP, VG_NEEDL,220._WP,DAYJ,0.6_WP,1, 0.28_WP,0.29_WP,0.30_WP,300._WP,300._WP,299._WP,300._WP,0._WP,0._WP,0.15_WP,0._WP)
    CASE DEFAULT
       name = 'warm_dry_grass'
       CALL setcol(1, DT, 1, 33.0_WP, 0.80_WP,0._WP,0._WP,355._WP,60._WP,3.0_WP,1.5_WP,0.007_WP,302._WP,97000._WP,98000._WP, VG_GRASS,820._WP,DAYJ,0.2_WP,1, 0.12_WP,0.14_WP,0.16_WP,303._WP,304._WP,300._WP,302._WP,0._WP,0._WP,0.20_WP,0._WP)
       CALL setcol(2, DT, 1, 33.0_WP, 0.80_WP,0._WP,0._WP,355._WP,60._WP,3.0_WP,1.5_WP,0.007_WP,302._WP,97000._WP,98000._WP, VG_CROP, 820._WP,DAYJ,0.2_WP,1, 0.12_WP,0.14_WP,0.16_WP,303._WP,304._WP,300._WP,302._WP,0._WP,0._WP,0.20_WP,0._WP)
       CALL setcol(3, DT, 1, 33.0_WP, 0.80_WP,0._WP,0._WP,355._WP,60._WP,3.0_WP,1.5_WP,0.007_WP,302._WP,97000._WP,98000._WP, VG_BROAD,820._WP,DAYJ,0.2_WP,1, 0.12_WP,0.14_WP,0.16_WP,303._WP,304._WP,300._WP,302._WP,0._WP,0._WP,0.20_WP,0._WP)
       CALL setcol(4, DT, 1, 33.0_WP, 0.80_WP,0._WP,0._WP,355._WP,60._WP,3.0_WP,1.5_WP,0.007_WP,302._WP,97000._WP,98000._WP, VG_SAVAN,820._WP,DAYJ,0.2_WP,1, 0.12_WP,0.14_WP,0.16_WP,303._WP,304._WP,300._WP,302._WP,0._WP,0._WP,0.20_WP,0._WP)
       CALL setcol(5, DT, 1, 33.0_WP, 0.80_WP,0._WP,0._WP,355._WP,60._WP,3.0_WP,1.5_WP,0.007_WP,302._WP,97000._WP,98000._WP, VG_NEEDL,820._WP,DAYJ,0.2_WP,1, 0.12_WP,0.14_WP,0.16_WP,303._WP,304._WP,300._WP,302._WP,0._WP,0._WP,0.20_WP,0._WP)
    END SELECT
  END SUBROUTINE build_case

END PROGRAM ssib_oracle
