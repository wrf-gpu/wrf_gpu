! Dev-only adversarial oracle for calcnfromq / smallvalues / radardd02 / calc_eff_radius.
! Uses the pristine WRF module with ONE visibility line added (public statement), init as the
! v034 column oracle, and calls each routine exactly like nssl_2mom_driver (density-scaled an).
PROGRAM cleanup_adv
  USE module_mp_nssl_2mom, ONLY : nssl_2mom_init, calcnfromq, smallvalues, radardd02, calc_eff_radius
  IMPLICIT NONE
  INTEGER, PARAMETER :: na = 18, nx = 1
  INTEGER :: nz, k, il, ccn_is_ccna
  REAL, ALLOCATABLE :: an0(:,:,:,:), an(:,:,:,:), dn(:,:,:), t77(:,:,:), t0(:,:,:), w(:,:,:), dbz(:,:,:)
  REAL, ALLOCATABLE :: t1(:,:,:), t2(:,:,:), t3(:,:,:), t4(:,:,:), t5(:,:,:), t6(:,:,:)
  REAL :: nssl_params(20), dtp
  REAL :: row(na+2)
  OPEN(11, FILE='adv_in.txt', STATUS='old')
  READ(11,*) nz
  ALLOCATE(an0(nx,1,nz,na), an(nx,1,nz,na), dn(nx,1,nz+1), t77(nx,1,nz), t0(nx,1,nz), w(nx,1,nz), dbz(nx,1,nz))
  ALLOCATE(t1(nx,1,nz), t2(nx,1,nz), t3(nx,1,nz), t4(nx,1,nz), t5(nx,1,nz), t6(nx,1,nz))
  DO k = 1, nz
    READ(11,*) row
    an0(1,1,k,1:na) = row(1:na)
    dn(1,1,k) = row(na+1)
    t77(1,1,k) = row(na+2)
  END DO
  dn(1,1,nz+1) = 1.0
  w = 0.0
  dtp = 54.0
  nssl_params(:) = 0.0
  nssl_params(1) = 0.5E9; nssl_params(2) = 0.0; nssl_params(3) = 1.0; nssl_params(4) = 4.0E5
  nssl_params(5) = 4.0E4; nssl_params(6) = 8.0E5; nssl_params(7) = 3.0E6; nssl_params(8) = 500.0
  nssl_params(9) = 900.0; nssl_params(10) = 100.0; nssl_params(11) = 0.0; nssl_params(12) = 12.0
  ccn_is_ccna = 0
  CALL nssl_2mom_init( nssl_params=nssl_params, ipctmp=5, mixphase=0,  &
       nssl_density_on=.TRUE., nssl_hail_on=.TRUE., nssl_ccn_on=.TRUE., &
       nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=ccn_is_ccna )
  ! calcnfromq (driver line 3116)
  an = an0
  CALL calcnfromq(nx,1,nz,an,na,0,0,dn)
  CALL dump('CQ', an)
  ! smallvalues (driver lines 3258-3263)
  an = an0
  t0 = an0(:,:,:,1)*t77
  CALL smallvalues(nx,1,nz,na,1,0,0,dtp,nx,t0,an,dn,w,t77,.FALSE.)
  CALL dump('SV', an)
  DO k = 1, nz
    WRITE(*,'(A,I0,A,ES26.17E3)') 'SVT0:', k, '=', t0(1,1,k)
  END DO
  ! radardd02 on the raw adversarial state (driver lines 3319-3329; cnoh0t/hwdn1t dead for ZVD)
  an = an0
  dbz = 0.0
  t0 = an0(:,:,:,1)*t77
  CALL radardd02(nx,1,nz,0,na,an,t0,dbz,dn,nz,4.0E5,500.0,5,nz,0)
  DO k = 1, nz
    WRITE(*,'(A,I0,A,ES26.17E3)') 'DBZ:', k, '=', dbz(1,1,k)
  END DO
  ! calc_eff_radius (driver lines 3360-3378)
  t1 = 2.51E-6; t2 = 10.01E-6; t3 = 25.E-6; t4 = 50.e-6; t5 = 0.0; t6 = 0.0
  CALL calc_eff_radius(nx,1,nz,na,1,0,0,t1=t1,t2=t2,t3=t3,t4=t4,t5=t5,t6=t6, &
       f_t4=.FALSE.,f_t5=.FALSE.,f_t6=.FALSE.,an=an,dn=dn)
  DO k = 1, nz
    WRITE(*,'(A,I0,A,3ES26.17E3)') 'RE:', k, '=', MAX(2.51E-6, MIN(t1(1,1,k), 50.E-6)), &
         MAX(10.01E-6, MIN(t2(1,1,k), 125.E-6)), MAX(25.E-6, MIN(t3(1,1,k), 999.E-6))
  END DO
CONTAINS
  SUBROUTINE dump(tag, a)
    CHARACTER(LEN=*) :: tag
    REAL :: a(:,:,:,:)
    INTEGER :: kk, ii
    DO kk = 1, SIZE(a,3)
      WRITE(*,'(A,A,I0,A,*(ES26.17E3,1X))') tag, ':', kk, '=', (a(1,1,kk,ii), ii=1,SIZE(a,4))
    END DO
  END SUBROUTINE dump
END PROGRAM cleanup_adv
