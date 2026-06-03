! =====================================================================
! v0.6.0 single-column BouLac PBL oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_bl_boulac.F column wrapper on six
! prescribed short soundings spanning unstable, stable, and neutral regimes.
! The dump captures inputs plus WRF BouLac tendencies/diagnostics for JAX
! savepoint parity. This is a real WRF-module oracle, not a JAX self-compare;
! it is not a full coupled wrf.exe run.
! =====================================================================
PROGRAM boulac_oracle
  USE module_bl_boulac, ONLY : boulac
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: CP = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: ROVCP = R_D/CP
  REAL, PARAMETER :: P1000 = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DZ,U,V,TH,RHO,QV,QC
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TKE,TKE_IN,DLK,WU,WV,WT,WQ,EXCH_H,EXCH_M
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: A_U,A_V,A_T,A_Q,A_E,B_U,B_V,B_T,B_Q,B_E
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DLG_BEP,DL_U_BEP,SF_BEP,VL_BEP
  REAL, DIMENSION(ims:ime,jms:jme) :: HFX,QFX,USTAR,PBLH,FRC_URB2D
  CHARACTER(LEN=32) :: arg, regime
  INTEGER :: case_id
  INTEGER :: idiff
  LOGICAL :: flag_bep

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, regime, DZ, U, V, TH, RHO, QV, QC, TKE, HFX, QFX, USTAR)
  TKE_IN = TKE

  RUBLTEN=0.0; RVBLTEN=0.0; RTHBLTEN=0.0; RQVBLTEN=0.0; RQCBLTEN=0.0
  DLK=0.0; WU=0.0; WV=0.0; WT=0.0; WQ=0.0; EXCH_H=0.0; EXCH_M=0.0
  A_U=0.0; A_V=0.0; A_T=0.0; A_Q=0.0; A_E=0.0
  B_U=0.0; B_V=0.0; B_T=0.0; B_Q=0.0; B_E=0.0
  DLG_BEP=0.0; DL_U_BEP=0.0; SF_BEP=1.0; VL_BEP=1.0
  PBLH=0.0; FRC_URB2D=0.0
  idiff = 0
  flag_bep = .FALSE.

  CALL boulac(frc_urb2d=FRC_URB2D,idiff=idiff,flag_bep=flag_bep,dz8w=DZ,dt=60.0, &
              u_phy=U,v_phy=V,th_phy=TH,rho=RHO,qv_curr=QV,qc_curr=QC, &
              hfx=HFX,qfx=QFX,ustar=USTAR,cp=CP,g=G, &
              rublten=RUBLTEN,rvblten=RVBLTEN,rthblten=RTHBLTEN, &
              rqvblten=RQVBLTEN,rqcblten=RQCBLTEN, &
              tke=TKE,dlk=DLK,wu=WU,wv=WV,wt=WT,wq=WQ, &
              exch_h=EXCH_H,exch_m=EXCH_M,pblh=PBLH, &
              a_u_bep=A_U,a_v_bep=A_V,a_t_bep=A_T,a_q_bep=A_Q, &
              a_e_bep=A_E,b_u_bep=B_U,b_v_bep=B_V,b_t_bep=B_T, &
              b_q_bep=B_Q,b_e_bep=B_E,dlg_bep=DLG_BEP,dl_u_bep=DL_U_BEP, &
              sf_bep=SF_BEP,vl_bep=VL_BEP, &
              ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
              ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
              its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)

  CALL dump_case(case_id, regime, DZ, U, V, TH, RHO, QV, QC, TKE_IN, TKE, DLK, &
                 HFX, QFX, USTAR, PBLH, EXCH_H, EXCH_M, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, WU, WV, WT, WQ)

CONTAINS

  SUBROUTINE build_case(cid, name, Dz, Uu, Vv, Thh, RhoA, Qq, QcA, TkeA, HfxA, QfxA, UstA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Dz,Uu,Vv,Thh,RhoA,Qq,QcA,TkeA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: HfxA,QfxA,UstA
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, rho_prof, tke_prof
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear
    REAL :: hfx0, qfx0, ust0, z, pi_mass, tv
    INTEGER :: k

    ztop = 10000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.18
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      name='unstable_daytime'
      psfc0=100000.0; theta0=300.0; zml=1000.0; lapse_ml=0.0002; lapse_ft=0.0045
      q0=0.0140; qscale=2200.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55
    CASE (2)
      name='strong_unstable'
      psfc0=100800.0; theta0=302.0; zml=1450.0; lapse_ml=0.0001; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80
    CASE (3)
      name='stable_nocturnal'
      psfc0=100500.0; theta0=287.0; zml=180.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1500.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25
    CASE (4)
      name='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=260.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18
    CASE (5)
      name='neutral'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0000; lapse_ft=0.0000
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40
    CASE DEFAULT
      name='weakly_unstable_low_ust'
      psfc0=99500.0; theta0=295.0; zml=700.0; lapse_ml=0.0008; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=20.0; qfx0=1.00E-5; ust0=0.08
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
      tv = temp(k)*(1.0 + (R_V/R_D - 1.0)*qprof(k))
      pint(k+1) = pint(k)*EXP(-G*(zint(k+1)-zint(k))/(R_D*MAX(tv, 180.0)))
      rho_prof(k) = 0.5*(pint(k)+pint(k+1))/(R_D*temp(k)*(1.0 + 0.61*qprof(k)))
      tke_prof(k) = MAX(0.04*EXP(-z/1500.0) + 0.02*ABS(shear)*z, 1.0E-4)
    END DO

    Dz=0.0; Uu=0.0; Vv=0.0; Thh=0.0; RhoA=0.0; Qq=0.0; QcA=0.0; TkeA=0.0
    DO k = 1, KX
      Dz(1,k,1) = zint(k+1)-zint(k)
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Thh(1,k,1) = theta(k)
      RhoA(1,k,1) = rho_prof(k)
      Qq(1,k,1) = qprof(k)
      QcA(1,k,1) = 0.0
      TkeA(1,k,1) = tke_prof(k)
    END DO
    HfxA(1,1)=hfx0; QfxA(1,1)=qfx0; UstA(1,1)=ust0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES25.17)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid, name, Dz, Uu, Vv, Thh, RhoA, Qq, QcA, TkeIn, TkeOut, DlkA, &
                       HfxA, QfxA, UstA, PblhA, Kh, Km, &
                       Uten, Vten, Thten, Qvten, Qcten, WuA, WvA, WtA, WqA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Dz,Uu,Vv,Thh,RhoA,Qq,QcA,TkeIn,TkeOut,DlkA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Kh,Km,Uten,Vten,Thten,Qvten,Qcten,WuA,WvA,WtA,WqA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: HfxA,QfxA,UstA,PblhA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES25.17)') 'DT=', 60.0
    WRITE(*,'(A,ES25.17)') 'CP=', CP
    WRITE(*,'(A,ES25.17)') 'G=', G
    WRITE(*,'(A,ES25.17)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES25.17)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES25.17)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES25.17)') 'PBLH=', PblhA(1,1)
    CALL dump_col('DZ', Dz)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('TH', Thh)
    CALL dump_col('RHO', RhoA)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', QcA)
    CALL dump_col('TKE_IN', TkeIn)
    CALL dump_col('TKE', TkeOut)
    CALL dump_col('DLK', DlkA)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
    CALL dump_col('EXCH_H', Kh)
    CALL dump_col('EXCH_M', Km)
    CALL dump_col('WU', WuA)
    CALL dump_col('WV', WvA)
    CALL dump_col('WT', WtA)
    CALL dump_col('WQ', WqA)
  END SUBROUTINE dump_case
END PROGRAM boulac_oracle
