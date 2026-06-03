! =====================================================================
! v0.6.0 single-column MYJ (Mellor-Yamada-Janjic) PBL oracle driver.
!
! bl_pbl_physics=2 REQUIRES sf_sfclay_physics=2: the MYJ PBL consumes the
! surface-layer exchange coefficients/fluxes produced by the Janjic Eta
! surface layer (USTAR, AKHS, AKMS, THZ0, QZ0, UZ0, VZ0, QSFC, CT,
! CHKLOWQ, ELFLX). To make the pairing faithful, this driver first calls
! the UNMODIFIED WRF MYJSFC (+ MYJSFCINIT) to populate those coupling
! fields, then calls the UNMODIFIED WRF MYJPBL. The dump captures the PBL
! TKE, exchange coefficient EXCH_H, U/V/theta/qv tendencies, PBLH, KPBL,
! and mixing length EL_MYJ for JAX savepoint parity.
!
! Real WRF-module oracle, not a JAX self-compare; not a full wrf.exe run.
! =====================================================================
PROGRAM myjpbl_oracle
  USE MODULE_SF_MYJSFC, ONLY : MYJSFC, MYJSFCINIT
  USE MODULE_BL_MYJPBL, ONLY : MYJPBL, MYJPBLINIT
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: DT     = 60.0
  INTEGER, PARAMETER :: STEPBL = 1

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  ! Shared profile / surface state
  REAL,DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,TH,QV,QC,QI,PMID,PINT,DZ,EXNER,RHO
  REAL,DIMENSION(ims:ime,jms:jme) :: HT,MAVAIL,TSK,XLAND,Z0BASE,SICE,SNOW
  REAL,DIMENSION(ims:ime,jms:jme) :: QSFC,THZ0,QZ0,UZ0,VZ0
  REAL,DIMENSION(ims:ime,jms:jme) :: USTAR,ZNT,PBLH,RMOL,AKHS,AKMS,RIB
  REAL,DIMENSION(ims:ime,jms:jme) :: CT,CHKLOWQ,ELFLX,MIXHT
  ! Surface-layer extra outputs (unused by PBL but required MYJSFC args)
  REAL,DIMENSION(ims:ime,kms:kme,jms:jme) :: Q2SFC
  REAL,DIMENSION(ims:ime,jms:jme) :: CHS,CHS2,CQS2,HFX,QFX,FLX_LH,FLHC,FLQC,QGH,CPM
  REAL,DIMENSION(ims:ime,jms:jme) :: U10,V10,T02,TH02,TSHLTR,TH10,Q02,QSHLTR,Q10,PSHLTR,U10E,V10E
  REAL,DIMENSION(ims:ime,jms:jme) :: SEAMASKD,XICE
  ! PBL prognostic/output fields
  REAL,DIMENSION(ims:ime,kms:kme,jms:jme) :: TKE_MYJ,EXCH_H,EL_MYJ
  REAL,DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN,RQIBLTEN
  INTEGER,DIMENSION(ims:ime,jms:jme) :: LOWLYR,IVGTYP,KPBL
  INTEGER :: ISURBAN, IZ0TLND, ITIMESTEP, case_id, k, kflip
  CHARACTER(LEN=256) :: regime
  CHARACTER(LEN=32) :: arg
  LOGICAL :: restart, allowed_to_read

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, regime, U, V, T, TH, QV, QC, QI, PMID, PINT, DZ, &
                  EXNER, RHO, Q2SFC, HT, MAVAIL, TSK, XLAND, Z0BASE, &
                  QSFC, THZ0, QZ0, UZ0, VZ0, USTAR, ZNT, IVGTYP, SICE, SNOW)

  ISURBAN = 1
  IZ0TLND = 0
  ITIMESTEP = 2
  restart = .FALSE.
  allowed_to_read = .FALSE.
  LOWLYR = 1
  RMOL = 0.0
  AKHS = 0.0; AKMS = 0.0
  PBLH = -1.0
  CT = 0.0
  SEAMASKD = XLAND
  XICE = 0.0

  ! --- Surface-layer half of the pair (produces the PBL coupling fields) ---
  CALL MYJSFCINIT(LOWLYR,USTAR,ZNT,SEAMASKD,XICE,IVGTYP,restart, &
                  allowed_to_read, &
                  ids,ide,jds,jde,kds,kde, &
                  ims,ime,jms,jme,kms,kme, &
                  its,ite,jts,jte,kts,kte)

  CALL MYJSFC(ITIMESTEP,HT,DZ, &
              PMID,PINT,TH,T,QV,QC,U,V,Q2SFC, &
              TSK,QSFC,THZ0,QZ0,UZ0,VZ0, &
              LOWLYR,XLAND,IVGTYP,ISURBAN,IZ0TLND, &
              USTAR,ZNT,Z0BASE,PBLH,MAVAIL,RMOL, &
              AKHS,AKMS, &
              RIB, &
              CHS,CHS2,CQS2,HFX,QFX,FLX_LH,FLHC,FLQC, &
              QGH,CPM,CT, &
              U10,V10,T02,TH02,TSHLTR,TH10,Q02,QSHLTR,Q10,PSHLTR, &
              P1000,U10E,V10E, &
              ids,ide,jds,jde,kds,kde, &
              ims,ime,jms,jme,kms,kme, &
              its,ite,jts,jte,kts,kte)

  ! CHKLOWQ = lowest-layer moisture-availability factor; LSMs set 1.0
  ! (module_surface_driver.F). ELFLX = surface latent heat flux = FLX_LH.
  CHKLOWQ = 1.0
  ELFLX = FLX_LH

  ! --- PBL half of the pair ---
  CALL MYJPBLINIT(RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN, &
                  TKE_MYJ,EXCH_H,restart,allowed_to_read, &
                  ids,ide,jds,jde,kds,kde, &
                  ims,ime,jms,jme,kms,kme, &
                  its,ite,jts,jte,kts,kte)
  ! Seed the column TKE from the same profile used by the SL Q2 input so the
  ! PBL starts from a physically consistent TKE state (TKE_MYJ = 0.5*q**2).
  DO k = kts, kte
    TKE_MYJ(1,k,1) = Q2SFC(1,k,1)
  END DO

  RQCBLTEN = 0.0; RQIBLTEN = 0.0
  EL_MYJ = 0.0
  KPBL = 0
  MIXHT = 0.0

  CALL MYJPBL(DT,STEPBL,HT,DZ, &
              PMID,PINT,TH,T,EXNER,QV,QC,QI,QC,QC,QC, &
              U,V,RHO,TSK,QSFC,CHKLOWQ,THZ0,QZ0,UZ0,VZ0, &
              LOWLYR,XLAND,SICE,SNOW, &
              TKE_MYJ,EXCH_H,USTAR,ZNT,EL_MYJ,PBLH,KPBL,CT, &
              AKHS,AKMS,ELFLX,MIXHT, &
              RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN, &
              RQIBLTEN,RQCBLTEN,RQCBLTEN,RQCBLTEN, &
              ids,ide,jds,jde,kds,kde, &
              ims,ime,jms,jme,kms,kme, &
              its,ite,jts,jte,kts,kte)

  CALL dump_case(case_id, regime, U, V, T, TH, QV, QC, PMID, PINT, DZ, EXNER, RHO, &
                 HT, TSK, XLAND, QSFC, THZ0, QZ0, UZ0, VZ0, USTAR, ZNT, &
                 CHKLOWQ, ELFLX, AKHS, AKMS, CT, PBLH, KPBL, MIXHT, &
                 TKE_MYJ, EXCH_H, EL_MYJ, RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, &
                 ITIMESTEP)

CONTAINS

  SUBROUTINE build_case(cid, name, Uu, Vv, Tt, Tht, Qq, Qcw, Qice, Pp, Pii, Dzz, &
                        Exn, Rhoo, Q2a, Htt, Mav, Tsk_a, Xl, Z0b, &
                        Qsf, Thz, Qz, Uz, Vz, Ust_a, Znt_a, Ivg, Sice_a, Snow_a)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    REAL,DIMENSION(ims:ime,kms:kme,jms:jme),INTENT(OUT) :: Uu,Vv,Tt,Tht,Qq,Qcw,Qice,Pp,Pii,Dzz,Exn,Rhoo,Q2a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Htt,Mav,Tsk_a,Xl,Z0b,Qsf,Thz,Qz,Uz,Vz
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Ust_a,Znt_a,Sice_a,Snow_a
    INTEGER,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Ivg
    REAL,DIMENSION(KX+1) :: zint, pint
    REAL,DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull, q2prof
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass, qmix
    REAL :: tsk0, ust0, znt0, xland0, mav0, q2sfc0, dtheta_sfc
    INTEGER :: k, ivgtyp0

    ztop = 12000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.18
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      name='unstable_daytime_land'
      psfc0=100000.0; theta0=300.0; zml=1100.0; lapse_ml=0.0003; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; tsk0=305.0; dtheta_sfc=5.0
      ust0=0.45; znt0=0.10; xland0=1.0; mav0=0.6; q2sfc0=0.6; ivgtyp0=10
    CASE (2)
      name='strong_unstable_land'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; tsk0=310.0; dtheta_sfc=8.0
      ust0=0.60; znt0=0.08; xland0=1.0; mav0=0.5; q2sfc0=1.0; ivgtyp0=7
    CASE (3)
      name='stable_nocturnal_land'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; tsk0=283.0; dtheta_sfc=-4.0
      ust0=0.20; znt0=0.05; xland0=1.0; mav0=0.4; q2sfc0=0.05; ivgtyp0=10
    CASE (4)
      name='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; tsk0=288.0; dtheta_sfc=-1.0
      ust0=0.18; znt0=0.001; xland0=2.0; mav0=1.0; q2sfc0=0.1; ivgtyp0=16
    CASE (5)
      name='neutral_land'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; tsk0=296.5; dtheta_sfc=0.5
      ust0=0.40; znt0=0.08; xland0=1.0; mav0=0.6; q2sfc0=0.3; ivgtyp0=10
    CASE DEFAULT
      name='unstable_marine'
      psfc0=101000.0; theta0=293.0; zml=600.0; lapse_ml=0.0010; lapse_ft=0.0035
      q0=0.0150; qscale=2400.0; shear=0.0015; tsk0=296.0; dtheta_sfc=3.0
      ust0=0.25; znt0=0.0015; xland0=2.0; mav0=1.0; q2sfc0=0.4; ivgtyp0=16
    END SELECT

    pint(1) = psfc0
    DO k = 1, KX
      z = zmid(k)
      IF (z <= zml) THEN
        theta(k) = theta0 + lapse_ml*z
      ELSE
        theta(k) = theta0 + lapse_ml*zml + lapse_ft*(z-zml)
      END IF
      qprof(k) = MAX(q0*EXP(-z/qscale), 1.0E-5)
      uprof(k) = 4.0 + shear*z
      vprof(k) = 1.0 + 0.25*shear*z
      pi_mass = (MAX(pint(k), 1000.0)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
      tv = temp(k)*(1.0 + EP1*qprof(k))
      pint(k+1) = pint(k)*EXP(-G*(zint(k+1)-zint(k))/(R_D*MAX(tv, 180.0)))
      pfull(k) = 0.5*(pint(k)+pint(k+1))
      pi_mass = (pfull(k)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
      q2prof(k) = MAX(q2sfc0*EXP(-z/600.0), 1.0E-3)
    END DO

    Uu=0.; Vv=0.; Tt=0.; Tht=0.; Qq=0.; Qcw=0.; Qice=0.; Pp=0.; Pii=0.
    Dzz=0.; Exn=0.; Rhoo=0.; Q2a=0.
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Tht(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Qcw(1,k,1) = 0.0
      Qice(1,k,1) = 0.0
      Pp(1,k,1) = pfull(k)
      Pii(1,k,1) = pint(k)
      Dzz(1,k,1) = zint(k+1)-zint(k)
      Exn(1,k,1) = (pfull(k)/P1000)**ROVCP
      ! moist density: rho = p / (R_d * T * (1 + 0.608 q_mix - q_cond))
      qmix = qprof(k)/(1.+qprof(k))
      Rhoo(1,k,1) = pfull(k)/(R_D*temp(k)*(1.+0.608*qmix))
      Q2a(1,k,1) = 0.5*q2prof(k)   ! TKE_MYJ = 0.5*q**2; we store TKE here
    END DO
    Pii(1,KX+1,1) = pint(KX+1)

    Htt(1,1)=0.0
    Mav(1,1)=mav0
    Tsk_a(1,1)=theta0*( (pint(1)/P1000)**ROVCP ) + dtheta_sfc
    Xl(1,1)=xland0
    Z0b(1,1)=znt0
    Ust_a(1,1)=ust0
    Znt_a(1,1)=znt0
    Ivg(1,1)=ivgtyp0
    Sice_a(1,1)=0.0
    Snow_a(1,1)=0.0
    Qsf(1,1)=qprof(1)
    Thz(1,1)=theta(1)
    Qz(1,1)=qprof(1)/(1.+qprof(1))
    Uz(1,1)=0.0
    Vz(1,1)=0.0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col_interface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col_interface

  SUBROUTINE dump_case(cid, name, Uu, Vv, Tt, Tht, Qq, Qcw, Pp, Pii, Dzz, Exn, Rhoo, &
                       Htt, Tsk_a, Xl, Qsf, Thz, Qz, Uz, Vz, Ust_a, Znt_a, &
                       Chk, Elf, Akhs_a, Akms_a, Ct_a, Pblh_a, Kpbl_a, Mixht_a, &
                       Tke_a, Exch_a, El_a, Uten, Vten, Thten, Qvten, itime)
    INTEGER, INTENT(IN) :: cid, itime
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL,DIMENSION(ims:ime,kms:kme,jms:jme),INTENT(IN) :: Uu,Vv,Tt,Tht,Qq,Qcw,Pp,Pii,Dzz,Exn,Rhoo
    REAL,DIMENSION(ims:ime,kms:kme,jms:jme),INTENT(IN) :: Tke_a,Exch_a,El_a,Uten,Vten,Thten,Qvten
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Htt,Tsk_a,Xl,Qsf,Thz,Qz,Uz,Vz,Ust_a,Znt_a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Chk,Elf,Akhs_a,Akms_a,Ct_a,Pblh_a,Mixht_a
    INTEGER,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Kpbl_a
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'ITIMESTEP=', itime
    WRITE(*,'(A,ES23.15)') 'DT=', DT
    WRITE(*,'(A,I0)') 'STEPBL=', STEPBL
    WRITE(*,'(A,ES23.15)') 'HT=', Htt(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', Tsk_a(1,1)
    WRITE(*,'(A,ES23.15)') 'XLAND=', Xl(1,1)
    ! Surface-layer coupling fields handed to the PBL
    WRITE(*,'(A,ES23.15)') 'USTAR=', Ust_a(1,1)
    WRITE(*,'(A,ES23.15)') 'ZNT=', Znt_a(1,1)
    WRITE(*,'(A,ES23.15)') 'AKHS=', Akhs_a(1,1)
    WRITE(*,'(A,ES23.15)') 'AKMS=', Akms_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CHKLOWQ=', Chk(1,1)
    WRITE(*,'(A,ES23.15)') 'ELFLX=', Elf(1,1)
    WRITE(*,'(A,ES23.15)') 'THZ0=', Thz(1,1)
    WRITE(*,'(A,ES23.15)') 'QZ0=', Qz(1,1)
    WRITE(*,'(A,ES23.15)') 'UZ0=', Uz(1,1)
    WRITE(*,'(A,ES23.15)') 'VZ0=', Vz(1,1)
    WRITE(*,'(A,ES23.15)') 'QSFC=', Qsf(1,1)
    WRITE(*,'(A,ES23.15)') 'CT=', Ct_a(1,1)
    ! PBL outputs
    WRITE(*,'(A,ES23.15)') 'PBLH=', Pblh_a(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', Kpbl_a(1,1)
    WRITE(*,'(A,ES23.15)') 'MIXHT=', Mixht_a(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('TH', Tht)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', Qcw)
    CALL dump_col('PMID', Pp)
    CALL dump_col_interface('PINT', Pii)
    CALL dump_col('DZ', Dzz)
    CALL dump_col('EXNER', Exn)
    CALL dump_col('RHO', Rhoo)
    CALL dump_col('TKE_MYJ', Tke_a)
    CALL dump_col('EXCH_H', Exch_a)
    CALL dump_col('EL_MYJ', El_a)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
  END SUBROUTINE dump_case
END PROGRAM myjpbl_oracle
