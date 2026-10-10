! CAM-UW PBL (bl_pbl_physics=9) pristine-WRF column oracle, v0.3.4 (lane o1-camuw).
!
! Calls the UNMODIFIED prebuilt WRF objects (phys/module_bl_camuwpbl_driver.o etc.).
! Fixes the v023 F3 driver defect: that driver never called WRF's CAM init, so the
! saturation-vapour-pressure table (wv_saturation:estbl, built by esinti/gestbl in
! module_physics_init.F:5109) stayed zero -> every qsat/gam was garbage.
! This driver reproduces WRF's CAM_INIT order (module_physics_init.F:5040-5165):
!   pcnst=5; esinti(epsilo,latvap,latice,rh2o,cpair,tmelt); cnst_add Q/CLDLIQ/CLDICE/NUMLIQ/NUMICE
! then camuwpblinit (module_physics_init.F:3828) and camuwpbl exactly as
! module_pbl_driver.F:2033 (keyword call, itimestep, kvm3d/kvh3d/taures carried as REAL).
! Columns are read from a text file (one column = one CAM chunk, ncol=1 like WRF).
PROGRAM camuw_oracle
  USE shr_kind_mod, ONLY : r8 => shr_kind_r8
  USE module_bl_camuwpbl_driver, ONLY : camuwpblinit, camuwpbl
  USE module_cam_support, ONLY : pcnst_runtime, pcnst_mp
  USE module_cam_esinti, ONLY : esinti
  USE wv_saturation, ONLY : estblf
  USE constituents, ONLY : cnst_add, cnst_name, cnst_longname, cnst_cp, cnst_cv, &
                           cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
                           cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, &
                           dcconnam, fixcnam, tendnam, ptendnam, dmetendnam, &
                           sflxnam, tottnam
  USE physconst, ONLY : mwh2o, cpwv, epsilo, latvap, latice, rh2o, cpair, tmelt, mwdry, &
                        gravit, rair, zvir, karman
  IMPLICIT NONE

  INTEGER :: ncol_in, kx, nstep, ic, istep, k, mm, n
  REAL :: dt
  INTEGER :: ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: U,V,TH,RHO,QV,QC,QI,P8W,P,Z,T,ZW
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: CLDFRA_OLD,CLDFRA,RTHRATENLW,EXNER,QNC,QNI,WSEDL3D
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQIBLTEN,RQNIBLTEN,RQVBLTEN,RQCBLTEN
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: KVM3D,KVH3D,SMAW3D,TURBTYPE3D,TKE_PBL
  REAL, DIMENSION(1,1) :: HFX,QFX,USTAR,HT,TAURESX2D,TAURESY2D,TPERT2D,QPERT2D,WPERT2D,PBLH2D
  INTEGER, DIMENSION(1,1) :: KPBL2D
  LOGICAL :: restart, is_cammgmp_used
  CHARACTER(LEN=256) :: fin
  REAL(r8) :: tt

  CALL GET_COMMAND_ARGUMENT(1, fin)
  OPEN(10, FILE=TRIM(fin), STATUS='OLD', ACTION='READ')
  READ(10,*) ncol_in, kx, nstep, dt

  ids=1; ide=2; jds=1; jde=2; kds=1; kde=kx+1
  ims=1; ime=1; jms=1; jme=1; kms=1; kme=kx+1
  its=1; ite=1; jts=1; jte=1; kts=1; kte=kx
  ALLOCATE(U(1,kme,1),V(1,kme,1),TH(1,kme,1),RHO(1,kme,1),QV(1,kme,1),QC(1,kme,1),QI(1,kme,1), &
           P8W(1,kme,1),P(1,kme,1),Z(1,kme,1),T(1,kme,1),ZW(1,kme,1),CLDFRA_OLD(1,kme,1), &
           CLDFRA(1,kme,1),RTHRATENLW(1,kme,1),EXNER(1,kme,1),QNC(1,kme,1),QNI(1,kme,1), &
           WSEDL3D(1,kme,1),RUBLTEN(1,kme,1),RVBLTEN(1,kme,1),RTHBLTEN(1,kme,1),RQIBLTEN(1,kme,1), &
           RQNIBLTEN(1,kme,1),RQVBLTEN(1,kme,1),RQCBLTEN(1,kme,1),KVM3D(1,kme,1),KVH3D(1,kme,1), &
           SMAW3D(1,kme,1),TURBTYPE3D(1,kme,1),TKE_PBL(1,kme,1))

  ! ---- WRF CAM_INIT (module_physics_init.F:5100-5165), non-CAMMGMP, non-chem ----
  pcnst_runtime = 5
  pcnst_mp = 5
  call esinti(epsilo, latvap, latice, rh2o, cpair, tmelt)
  ALLOCATE(cnst_name(5),cnst_longname(5),cnst_cp(5),cnst_cv(5),cnst_mw(5),cnst_type(5), &
           cnst_rgas(5),qmin(5),qmincg(5),cnst_fixed_ubc(5),apcnst(5),bpcnst(5),hadvnam(5), &
           vadvnam(5),dcconnam(5),fixcnam(5),tendnam(5),ptendnam(5),dmetendnam(5),sflxnam(5),tottnam(5))
  cnst_fixed_ubc(:) = .false.
  call cnst_add('Q', mwh2o, cpwv, 1.E-12_r8, mm, longname='Specific humidity', readiv=.true. )
  call cnst_add('CLDLIQ', mwdry, cpair, 0._r8, mm, longname='Grid box averaged cloud liquid amount')
  call cnst_add('CLDICE', mwdry, cpair, 0._r8, mm, longname='Grid box averaged cloud ice amount'   )
  call cnst_add('NUMLIQ', mwdry, cpair, 0._r8, mm, longname='Grid box averaged cloud liquid number')
  call cnst_add('NUMICE', mwdry, cpair, 0._r8, mm, longname='Grid box averaged cloud ice number'   )

  restart = .FALSE.
  is_cammgmp_used = .FALSE.
  CALL camuwpblinit(RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN, restart,TKE_PBL,is_cammgmp_used, &
       ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte)

  ! constants actually used (full precision) + estbl table (estblf at table nodes is exact)
  WRITE(*,'(A,ES26.17E3)') 'CONST cpair=', cpair
  WRITE(*,'(A,ES26.17E3)') 'CONST gravit=', gravit
  WRITE(*,'(A,ES26.17E3)') 'CONST rair=', rair
  WRITE(*,'(A,ES26.17E3)') 'CONST zvir=', zvir
  WRITE(*,'(A,ES26.17E3)') 'CONST latvap=', latvap
  WRITE(*,'(A,ES26.17E3)') 'CONST latice=', latice
  WRITE(*,'(A,ES26.17E3)') 'CONST karman=', karman
  WRITE(*,'(A,ES26.17E3)') 'CONST epsilo=', epsilo
  WRITE(*,'(A,ES26.17E3)') 'CONST rh2o=', rh2o
  WRITE(*,'(A,ES26.17E3)') 'CONST tmelt=', tmelt
  DO n = 1, 204
     tt = 173.16_r8 + REAL(n-1, r8)
     WRITE(*,'(A,I0,A,ES26.17E3)') 'ESTBL[', n, ']=', estblf(tt)
  END DO

  DO ic = 1, ncol_in
     CALL read_column()
     KVM3D = 0.0; KVH3D = 0.0; TAURESX2D = 0.0; TAURESY2D = 0.0
     DO istep = 1, nstep
        RUBLTEN = 0.0; RVBLTEN = 0.0; RTHBLTEN = 0.0; RQIBLTEN = 0.0
        RQNIBLTEN = 0.0; RQVBLTEN = 0.0; RQCBLTEN = 0.0
        SMAW3D = 0.0; TURBTYPE3D = 0.0; TKE_PBL = 0.0
        TPERT2D = 0.0; QPERT2D = 0.0; WPERT2D = 0.0; PBLH2D = 0.0; KPBL2D = 0
        WRITE(*,'(A,I0,A,I0)') 'BEGIN col=', ic, ' step=', istep
        CALL dump_in()
        CALL camuwpbl(DT=dt,U_PHY=U,V_PHY=V,TH_PHY=TH,RHO=RHO                       &
             ,QV_CURR=QV,HFX=HFX,QFX=QFX,USTAR=USTAR,P8W=P8W,P_PHY=P                &
             ,Z=Z,T_PHY=T,QC_CURR=QC,QI_CURR=QI,Z_AT_W=ZW                           &
             ,CLDFRA_OLD_mp=CLDFRA_OLD,CLDFRA=CLDFRA,HT=HT                          &
             ,RTHRATENLW=RTHRATENLW,EXNER=EXNER                                     &
             ,is_CAMMGMP_used=is_cammgmp_used                                       &
             ,ITIMESTEP=istep,QNC_CURR=QNC,QNI_CURR=QNI                             &
             ,WSEDL3D=WSEDL3D                                                       &
             ,IDS=ids,IDE=ide, JDS=jds,JDE=jde, KDS=kds,KDE=kde                     &
             ,IMS=ims,IME=ime, JMS=jms,JME=jme, KMS=kms,KME=kme                     &
             ,ITS=its,ITE=ite, JTS=jts,JTE=jte, KTS=kts,KTE=kte                     &
             ,TAURESX2D=TAURESX2D,TAURESY2D=TAURESY2D                               &
             ,RUBLTEN=RUBLTEN,RVBLTEN=RVBLTEN,RTHBLTEN=RTHBLTEN                     &
             ,RQIBLTEN=RQIBLTEN,RQNIBLTEN=RQNIBLTEN,RQVBLTEN=RQVBLTEN               &
             ,RQCBLTEN=RQCBLTEN,KVM3D=KVM3D,KVH3D=KVH3D                             &
             ,TPERT2D=TPERT2D,QPERT2D=QPERT2D,WPERT2D=WPERT2D,SMAW3D=SMAW3D         &
             ,TURBTYPE3D=TURBTYPE3D                                                 &
             ,TKE_pbl=TKE_PBL,PBLH2D=PBLH2D,KPBL2D=KPBL2D                           )
        CALL dump_out()
        WRITE(*,'(A)') 'END'
        ! evolve the column with the PBL tendencies only (REAL arithmetic) so the next
        ! step exercises the carried kvh/kvm/taures path on a changed state
        DO k = kts, kte
           U(1,k,1)   = U(1,k,1)   + dt*RUBLTEN(1,k,1)
           V(1,k,1)   = V(1,k,1)   + dt*RVBLTEN(1,k,1)
           TH(1,k,1)  = TH(1,k,1)  + dt*RTHBLTEN(1,k,1)
           QV(1,k,1)  = MAX(QV(1,k,1) + dt*RQVBLTEN(1,k,1), 0.0)
           QC(1,k,1)  = MAX(QC(1,k,1) + dt*RQCBLTEN(1,k,1), 0.0)
           QI(1,k,1)  = MAX(QI(1,k,1) + dt*RQIBLTEN(1,k,1), 0.0)
           QNI(1,k,1) = MAX(QNI(1,k,1) + dt*RQNIBLTEN(1,k,1), 0.0)
           T(1,k,1)   = TH(1,k,1)*EXNER(1,k,1)
           RHO(1,k,1) = P(1,k,1)/(287.0*T(1,k,1)*(1.0+0.608*QV(1,k,1)))
        END DO
     END DO
  END DO

CONTAINS

  SUBROUTINE read_column()
    READ(10,*) HFX(1,1), QFX(1,1), USTAR(1,1), HT(1,1)
    READ(10,*) (U(1,k,1), k=kts,kte)
    READ(10,*) (V(1,k,1), k=kts,kte)
    READ(10,*) (TH(1,k,1), k=kts,kte)
    READ(10,*) (P(1,k,1), k=kts,kte)
    READ(10,*) (T(1,k,1), k=kts,kte)
    READ(10,*) (EXNER(1,k,1), k=kts,kte)
    READ(10,*) (RHO(1,k,1), k=kts,kte)
    READ(10,*) (QV(1,k,1), k=kts,kte)
    READ(10,*) (QC(1,k,1), k=kts,kte)
    READ(10,*) (QI(1,k,1), k=kts,kte)
    READ(10,*) (QNC(1,k,1), k=kts,kte)
    READ(10,*) (QNI(1,k,1), k=kts,kte)
    READ(10,*) (CLDFRA(1,k,1), k=kts,kte)
    READ(10,*) (RTHRATENLW(1,k,1), k=kts,kte)
    READ(10,*) (WSEDL3D(1,k,1), k=kts,kte)
    READ(10,*) (Z(1,k,1), k=kts,kte)
    READ(10,*) (P8W(1,k,1), k=kts,kte+1)
    READ(10,*) (ZW(1,k,1), k=kts,kte+1)
    CLDFRA_OLD = CLDFRA
  END SUBROUTINE read_column

  SUBROUTINE d1(nm, arr, n)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(1,kme,1), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: n
    INTEGER :: kk
    DO kk = 1, n
      WRITE(*,'(A,A,I0,A,ES26.17E3)') nm, '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE d1

  SUBROUTINE dump_in()
    WRITE(*,'(A,ES26.17E3)') 'IN HFX=', HFX(1,1)
    WRITE(*,'(A,ES26.17E3)') 'IN QFX=', QFX(1,1)
    WRITE(*,'(A,ES26.17E3)') 'IN UST=', USTAR(1,1)
    WRITE(*,'(A,ES26.17E3)') 'IN HT=', HT(1,1)
    WRITE(*,'(A,ES26.17E3)') 'IN TAURESX=', TAURESX2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'IN TAURESY=', TAURESY2D(1,1)
    CALL d1('IN U', U, kx); CALL d1('IN V', V, kx); CALL d1('IN TH', TH, kx)
    CALL d1('IN P', P, kx); CALL d1('IN T', T, kx); CALL d1('IN EXNER', EXNER, kx)
    CALL d1('IN RHO', RHO, kx); CALL d1('IN QV', QV, kx); CALL d1('IN QC', QC, kx)
    CALL d1('IN QI', QI, kx); CALL d1('IN QNC', QNC, kx); CALL d1('IN QNI', QNI, kx)
    CALL d1('IN CLDFRA', CLDFRA, kx); CALL d1('IN RTHRATENLW', RTHRATENLW, kx)
    CALL d1('IN WSEDL', WSEDL3D, kx); CALL d1('IN Z', Z, kx)
    CALL d1('IN P8W', P8W, kx+1); CALL d1('IN ZW', ZW, kx+1)
    CALL d1('IN KVM', KVM3D, kx+1); CALL d1('IN KVH', KVH3D, kx+1)
  END SUBROUTINE dump_in

  SUBROUTINE dump_out()
    WRITE(*,'(A,ES26.17E3)') 'OUT PBLH=', PBLH2D(1,1)
    WRITE(*,'(A,I0)') 'OUT KPBL=', KPBL2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'OUT TPERT=', TPERT2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'OUT QPERT=', QPERT2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'OUT WPERT=', WPERT2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'OUT TAURESX=', TAURESX2D(1,1)
    WRITE(*,'(A,ES26.17E3)') 'OUT TAURESY=', TAURESY2D(1,1)
    CALL d1('OUT RUBLTEN', RUBLTEN, kx); CALL d1('OUT RVBLTEN', RVBLTEN, kx)
    CALL d1('OUT RTHBLTEN', RTHBLTEN, kx); CALL d1('OUT RQVBLTEN', RQVBLTEN, kx)
    CALL d1('OUT RQCBLTEN', RQCBLTEN, kx); CALL d1('OUT RQIBLTEN', RQIBLTEN, kx)
    CALL d1('OUT RQNIBLTEN', RQNIBLTEN, kx)
    CALL d1('OUT TKE_PBL', TKE_PBL, kx+1); CALL d1('OUT KVM', KVM3D, kx+1)
    CALL d1('OUT KVH', KVH3D, kx+1); CALL d1('OUT SMAW', SMAW3D, kx+1)
    CALL d1('OUT TURBTYPE', TURBTYPE3D, kx+1)
  END SUBROUTINE dump_out
END PROGRAM camuw_oracle
