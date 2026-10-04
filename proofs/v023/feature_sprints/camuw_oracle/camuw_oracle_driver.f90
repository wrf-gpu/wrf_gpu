! Standalone CAM-UW PBL oracle driver for F3.
!
! This calls the unmodified WRF CAM-UW wrapper
! phys/module_bl_camuwpbl_driver.F:camuwpbl on one synthetic column and emits
! flat key=value output for JSON parsing.  It intentionally mirrors the
! v0.13/v0.60 oracle style used elsewhere in this repository.
PROGRAM camuw_oracle
  USE module_bl_camuwpbl_driver, ONLY : camuwpblinit, camuwpbl
  USE module_cam_support, ONLY : pcnst_runtime, pcnst_mp, pcnst_non_chem_modal_aero, iulog
  USE constituents, ONLY : cnst_add, cnst_name, cnst_longname, cnst_cp, cnst_cv, &
                           cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
                           cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, &
                           dcconnam, fixcnam, tendnam, ptendnam, dmetendnam, &
                           sflxnam, tottnam
  USE physconst, ONLY : mwh2o, mwdry, cpwv, cpair
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.80665
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: CP_D = 1004.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: P1000 = 1.0E5
  REAL, PARAMETER :: ROVCP = R_D / CP_D
  REAL, PARAMETER :: EP1 = R_V / R_D - 1.0
  REAL, PARAMETER :: DT = 60.0
  INTEGER, PARAMETER :: KX = 16

  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,TH,RHO,QV,QC,QI,P8W,P,Z,T,ZW
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CLDFRA_OLD,CLDFRA,RTHRATENLW,EXNER
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: QNC,QNI,WSEDL3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RQNIBLTEN,RQVBLTEN,RQCBLTEN,KVM3D,KVH3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: SMAW3D,TURBTYPE3D,TKE_PBL
  REAL, DIMENSION(ims:ime,jms:jme) :: HFX,QFX,USTAR,HT,TAURESX2D,TAURESY2D
  REAL, DIMENSION(ims:ime,jms:jme) :: TPERT2D,QPERT2D,WPERT2D,PBLH2D
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL2D

  CHARACTER(LEN=32) :: arg, name
  INTEGER :: case_id
  LOGICAL :: restart, is_cammgmp_used

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL init_cam_constituents()
  CALL build_case(case_id, name, U, V, TH, RHO, QV, QC, QI, P8W, P, Z, T, ZW, &
                  CLDFRA_OLD, CLDFRA, RTHRATENLW, EXNER, QNC, QNI, WSEDL3D, &
                  HFX, QFX, USTAR, HT)

  RUBLTEN = 0.0; RVBLTEN = 0.0; RTHBLTEN = 0.0; RQIBLTEN = 0.0
  RQNIBLTEN = 0.0; RQVBLTEN = 0.0; RQCBLTEN = 0.0
  KVM3D = 0.0; KVH3D = 0.0
  SMAW3D = 0.0; TURBTYPE3D = 0.0; TKE_PBL = 0.0
  TAURESX2D = 0.0; TAURESY2D = 0.0
  TPERT2D = 0.0; QPERT2D = 0.0; WPERT2D = 0.0
  PBLH2D = 0.0; KPBL2D = 0
  restart = .FALSE.
  is_cammgmp_used = .FALSE.

  CALL camuwpblinit(RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN, &
       restart,TKE_PBL,is_cammgmp_used, &
       ids,ide,jds,jde,kds,kde, &
       ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte)

  CALL camuwpbl(DT,U,V,TH,RHO,QV,HFX,QFX,USTAR,P8W, &
       P,Z,T,QC,QI,ZW,CLDFRA_OLD,CLDFRA,HT, &
       RTHRATENLW,EXNER,is_cammgmp_used, &
       1,QNC,QNI,WSEDL3D, &
       ids,ide,jds,jde,kds,kde, &
       ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte, &
       TAURESX2D,TAURESY2D, &
       RUBLTEN,RVBLTEN,RTHBLTEN,RQIBLTEN,RQNIBLTEN,RQVBLTEN,RQCBLTEN, &
       KVM3D,KVH3D, &
       TPERT2D,QPERT2D,WPERT2D,SMAW3D,TURBTYPE3D, &
       TKE_PBL,PBLH2D,KPBL2D)

  CALL dump_case(case_id, name, U, V, TH, QV, QC, QI, P, P8W, Z, ZW, T, EXNER, RHO, &
                 HFX, QFX, USTAR, PBLH2D, KPBL2D, TPERT2D, QPERT2D, WPERT2D, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, RQIBLTEN, &
                 TKE_PBL, KVM3D, KVH3D, SMAW3D, TURBTYPE3D)

CONTAINS

  SUBROUTINE init_cam_constituents()
    INTEGER :: ind
    LOGICAL :: readiv

    pcnst_runtime = 5
    pcnst_mp = 5
    pcnst_non_chem_modal_aero = 5
    iulog = ''
    CALL allocate_constituent_arrays(pcnst_runtime)
    readiv = .TRUE.

    CALL cnst_add('Q', mwh2o, cpwv, 1.0D-12, ind, longname='water vapor', &
                  readiv=readiv, mixtype='wet')
    CALL cnst_add('CLDLIQ', mwdry, cpair, 0.0D0, ind, longname='cloud liquid', &
                  readiv=readiv, mixtype='wet')
    CALL cnst_add('CLDICE', mwdry, cpair, 0.0D0, ind, longname='cloud ice', &
                  readiv=readiv, mixtype='wet')
    CALL cnst_add('NUMLIQ', mwdry, cpair, 0.0D0, ind, longname='cloud liquid number', &
                  readiv=readiv, mixtype='wet')
    CALL cnst_add('NUMICE', mwdry, cpair, 0.0D0, ind, longname='cloud ice number', &
                  readiv=readiv, mixtype='wet')
  END SUBROUTINE init_cam_constituents

  SUBROUTINE allocate_constituent_arrays(n)
    INTEGER, INTENT(IN) :: n
    IF (.NOT. ALLOCATED(cnst_name)) ALLOCATE(cnst_name(n))
    IF (.NOT. ALLOCATED(cnst_longname)) ALLOCATE(cnst_longname(n))
    IF (.NOT. ALLOCATED(cnst_cp)) ALLOCATE(cnst_cp(n))
    IF (.NOT. ALLOCATED(cnst_cv)) ALLOCATE(cnst_cv(n))
    IF (.NOT. ALLOCATED(cnst_mw)) ALLOCATE(cnst_mw(n))
    IF (.NOT. ALLOCATED(cnst_type)) ALLOCATE(cnst_type(n))
    IF (.NOT. ALLOCATED(cnst_rgas)) ALLOCATE(cnst_rgas(n))
    IF (.NOT. ALLOCATED(qmin)) ALLOCATE(qmin(n))
    IF (.NOT. ALLOCATED(qmincg)) ALLOCATE(qmincg(n))
    IF (.NOT. ALLOCATED(cnst_fixed_ubc)) ALLOCATE(cnst_fixed_ubc(n))
    IF (.NOT. ALLOCATED(apcnst)) ALLOCATE(apcnst(n))
    IF (.NOT. ALLOCATED(bpcnst)) ALLOCATE(bpcnst(n))
    IF (.NOT. ALLOCATED(hadvnam)) ALLOCATE(hadvnam(n))
    IF (.NOT. ALLOCATED(vadvnam)) ALLOCATE(vadvnam(n))
    IF (.NOT. ALLOCATED(dcconnam)) ALLOCATE(dcconnam(n))
    IF (.NOT. ALLOCATED(fixcnam)) ALLOCATE(fixcnam(n))
    IF (.NOT. ALLOCATED(tendnam)) ALLOCATE(tendnam(n))
    IF (.NOT. ALLOCATED(ptendnam)) ALLOCATE(ptendnam(n))
    IF (.NOT. ALLOCATED(dmetendnam)) ALLOCATE(dmetendnam(n))
    IF (.NOT. ALLOCATED(sflxnam)) ALLOCATE(sflxnam(n))
    IF (.NOT. ALLOCATED(tottnam)) ALLOCATE(tottnam(n))
    cnst_name = ''
    cnst_longname = ''
    cnst_cp = 0.0
    cnst_cv = 0.0
    cnst_mw = 0.0
    cnst_type = 'wet'
    cnst_rgas = 0.0
    qmin = 0.0
    qmincg = 0.0
    cnst_fixed_ubc = .FALSE.
    apcnst = ''
    bpcnst = ''
    hadvnam = ''
    vadvnam = ''
    dcconnam = ''
    fixcnam = ''
    tendnam = ''
    ptendnam = ''
    dmetendnam = ''
    sflxnam = ''
    tottnam = ''
  END SUBROUTINE allocate_constituent_arrays

  SUBROUTINE build_case(cid, nm, Uu, Vv, Th, Rhoo, Qq, Qc, Qi, Pint, Pmid, Zmid, Temp, Zint, &
                        CfOld, Cf, LwTen, Exn, Qnc, Qni, Wsed, HfxA, QfxA, UstA, HtA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Th,Rhoo,Qq,Qc,Qi,Pint,Pmid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Zmid,Temp,Zint,CfOld,Cf,LwTen,Exn
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Qnc,Qni,Wsed
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: HfxA,QfxA,UstA,HtA
    REAL, DIMENSION(KX+1) :: zi, pi
    REAL, DIMENSION(KX) :: zm, theta, qprof, qcprof, qiprof, uprof, vprof, tmp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, hfx0, qfx0, ust0
    REAL :: z, tv, exner_mass
    INTEGER :: k

    ztop = 9000.0
    zi(1) = 0.0
    DO k = 1, KX
      zi(k+1) = ztop * (REAL(k) / REAL(KX)) ** 1.15
      zm(k) = 0.5 * (zi(k) + zi(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      nm = 'unstable_moist_cloud'
      psfc0=100000.0; theta0=300.0; zml=950.0; lapse_ml=0.0002; lapse_ft=0.0040
      q0=0.0140; qscale=2200.0; shear=0.0018; hfx0=280.0; qfx0=1.0E-4; ust0=0.45
    CASE (2)
      nm = 'stable_nocturnal'
      psfc0=100800.0; theta0=288.0; zml=160.0; lapse_ml=0.0100; lapse_ft=0.0045
      q0=0.0060; qscale=1600.0; shear=0.0050; hfx0=-40.0; qfx0=0.0; ust0=0.22
    CASE DEFAULT
      nm = 'neutral_low_cloud'
      psfc0=101200.0; theta0=294.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0032
      q0=0.0100; qscale=2400.0; shear=0.0020; hfx0=15.0; qfx0=2.0E-5; ust0=0.35
    END SELECT

    pi(1) = psfc0
    DO k = 1, KX
      z = zm(k)
      IF (z <= zml) THEN
        theta(k) = theta0 + lapse_ml * z
      ELSE
        theta(k) = theta0 + lapse_ml * zml + lapse_ft * (z - zml)
      END IF
      qprof(k) = MAX(q0 * EXP(-z / qscale), 2.0E-5)
      qcprof(k) = 0.0
      qiprof(k) = 0.0
      IF (z > 900.0 .AND. z < 1800.0 .AND. cid /= 2) qcprof(k) = 3.0E-5
      uprof(k) = 5.0 + shear * z
      vprof(k) = 1.0 + 0.3 * shear * z
      exner_mass = (MAX(pi(k), 1000.0) / P1000) ** ROVCP
      tmp(k) = theta(k) * exner_mass
      tv = tmp(k) * (1.0 + EP1 * qprof(k))
      pi(k+1) = pi(k) * EXP(-G * (zi(k+1) - zi(k)) / (R_D * MAX(tv, 180.0)))
      pfull(k) = 0.5 * (pi(k) + pi(k+1))
      exner_mass = (pfull(k) / P1000) ** ROVCP
      tmp(k) = theta(k) * exner_mass
    END DO

    Uu=0.0; Vv=0.0; Th=0.0; Rhoo=0.0; Qq=0.0; Qc=0.0; Qi=0.0
    Pint=0.0; Pmid=0.0; Zmid=0.0; Temp=0.0; Zint=0.0
    CfOld=0.0; Cf=0.0; LwTen=0.0; Exn=0.0; Qnc=0.0; Qni=0.0; Wsed=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Th(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Qc(1,k,1) = qcprof(k)
      Qi(1,k,1) = qiprof(k)
      Pmid(1,k,1) = pfull(k)
      Pint(1,k,1) = pi(k)
      Zmid(1,k,1) = zm(k)
      Zint(1,k,1) = zi(k)
      Temp(1,k,1) = tmp(k)
      Exn(1,k,1) = (pfull(k) / P1000) ** ROVCP
      Rhoo(1,k,1) = pfull(k) / (R_D * tmp(k) * (1.0 + 0.608 * qprof(k) - qcprof(k) - qiprof(k)))
      IF (qcprof(k) > 0.0) THEN
        Cf(1,k,1) = 0.6
        CfOld(1,k,1) = 0.6
      END IF
      Qnc(1,k,1) = 1.0E8
      Qni(1,k,1) = 1.0E6
    END DO
    Pint(1,KX+1,1) = pi(KX+1)
    Zint(1,KX+1,1) = zi(KX+1)
    HfxA(1,1)=hfx0
    QfxA(1,1)=qfx0
    UstA(1,1)=ust0
    HtA(1,1)=0.0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(nm, arr, n)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: n
    INTEGER :: kk
    DO kk = 1, n
      WRITE(*,'(A,A,I0,A,ES23.15)') nm, '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid, nm, Uu, Vv, Th, Qq, Qc, Qi, Pmid, Pint, Zmid, Zint, Temp, Exn, Rhoo, &
                       HfxA, QfxA, UstA, PblhA, KpblA, TpertA, QpertA, WpertA, &
                       Uten, Vten, Thten, Qvten, Qcten, Qiten, TkeA, KvmA, KvhA, SmawA, TurbA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Th,Qq,Qc,Qi,Pmid,Pint,Zmid,Zint,Temp,Exn,Rhoo
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten,Qiten,TkeA,KvmA,KvhA,SmawA,TurbA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: HfxA,QfxA,UstA,PblhA,TpertA,QpertA,WpertA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(nm)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', DT
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'USTAR=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'PBLH=', PblhA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    WRITE(*,'(A,ES23.15)') 'TPERT=', TpertA(1,1)
    WRITE(*,'(A,ES23.15)') 'QPERT=', QpertA(1,1)
    WRITE(*,'(A,ES23.15)') 'WPERT=', WpertA(1,1)
    CALL dump_col('U', Uu, KX)
    CALL dump_col('V', Vv, KX)
    CALL dump_col('TH', Th, KX)
    CALL dump_col('QV', Qq, KX)
    CALL dump_col('QC', Qc, KX)
    CALL dump_col('QI', Qi, KX)
    CALL dump_col('T', Temp, KX)
    CALL dump_col('P', Pmid, KX)
    CALL dump_col('P8W', Pint, KX+1)
    CALL dump_col('Z', Zmid, KX)
    CALL dump_col('Z_AT_W', Zint, KX+1)
    CALL dump_col('EXNER', Exn, KX)
    CALL dump_col('RHO', Rhoo, KX)
    CALL dump_col('RUBLTEN', Uten, KX)
    CALL dump_col('RVBLTEN', Vten, KX)
    CALL dump_col('RTHBLTEN', Thten, KX)
    CALL dump_col('RQVBLTEN', Qvten, KX)
    CALL dump_col('RQCBLTEN', Qcten, KX)
    CALL dump_col('RQIBLTEN', Qiten, KX)
    CALL dump_col('TKE_PBL', TkeA, KX+1)
    CALL dump_col('KVM3D', KvmA, KX+1)
    CALL dump_col('KVH3D', KvhA, KX+1)
    CALL dump_col('SMAW3D', SmawA, KX+1)
    CALL dump_col('TURBTYPE3D', TurbA, KX+1)
  END SUBROUTINE dump_case
END PROGRAM camuw_oracle
