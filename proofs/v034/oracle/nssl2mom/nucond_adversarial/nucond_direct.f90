! o1-nssl dev oracle: call the PRISTINE NUCOND directly on columns read from a text file.
! Init mirrors the v023/v034 column oracle (WRF default mp=18). Input file (list-directed):
!   ncol nz dtp
!   an(ix,1,kz,il) for il=1..18, kz=1..nz, ix=1..ncol   (density-scaled moments, as inside the driver)
!   dn, t77, pn, w   (each (ix,kz) ordered kz-major per il-less layout: kz outer, ix inner)
! Output: an, t0, ssfilt after NUCOND (same ordering), ES26.17E3.
PROGRAM nucond_direct
  USE module_mp_nssl_2mom, ONLY : nssl_2mom_init, nucond
  IMPLICIT NONE
  INTEGER, PARAMETER :: NA = 18
  REAL, ALLOCATABLE :: an(:,:,:,:), dn(:,:,:), t77(:,:,:), pn(:,:,:), w(:,:,:), t0(:,:,:), t9(:,:,:)
  REAL, ALLOCATABLE :: ssfilt(:,:,:), t00(:,:,:), dz(:,:,:), axtra(:,:,:,:)
  REAL :: dtp, nssl_params(20)
  INTEGER :: ncol, nz, il, kz, ix, ccn_is_ccna
  CHARACTER(LEN=256) :: fin
  CALL GET_COMMAND_ARGUMENT(1, fin)
  nssl_params(:) = 0.0
  nssl_params(1) = 0.5E9; nssl_params(2) = 0.0; nssl_params(3) = 1.0; nssl_params(4) = 4.0E5
  nssl_params(5) = 4.0E4; nssl_params(6) = 8.0E5; nssl_params(7) = 3.0E6; nssl_params(8) = 500.0
  nssl_params(9) = 900.0; nssl_params(10) = 100.0; nssl_params(11) = 0.0; nssl_params(12) = 12.0
  ccn_is_ccna = 0
  CALL nssl_2mom_init( nssl_params=nssl_params, ipctmp=5, mixphase=0,  &
       nssl_density_on=.TRUE., nssl_hail_on=.TRUE., nssl_ccn_on=.TRUE., &
       nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=ccn_is_ccna )
  OPEN(11, FILE=TRIM(fin), STATUS='old')
  READ(11,*) ncol, nz, dtp
  ALLOCATE(an(ncol,1,nz,NA), dn(ncol,1,nz), t77(ncol,1,nz), pn(ncol,1,nz), w(ncol,1,nz), t0(ncol,1,nz))
  ALLOCATE(t9(ncol,1,nz), ssfilt(ncol,1,nz), t00(ncol,1,nz), dz(ncol,1,nz), axtra(ncol,1,nz,1))
  READ(11,*) (((an(ix,1,kz,il), ix=1,ncol), kz=1,nz), il=1,NA)
  READ(11,*) ((dn(ix,1,kz), ix=1,ncol), kz=1,nz)
  READ(11,*) ((t77(ix,1,kz), ix=1,ncol), kz=1,nz)
  READ(11,*) ((pn(ix,1,kz), ix=1,ncol), kz=1,nz)
  READ(11,*) ((w(ix,1,kz), ix=1,ncol), kz=1,nz)
  CLOSE(11)
  t0 = 0.0; t9 = 0.0; ssfilt = 0.0; axtra = 0.0; dz = 200.0
  t00 = 380.0/pn
  CALL NUCOND(ncol,1,nz,NA,1, 0,0,dtp,ncol, dz, t0,t9, an,dn,t77, pn,w, 64, axtra,.false., &
              ssfilt,t00,t77,.false.)
  WRITE(*,'(*(ES26.17E3,1X))') (((an(ix,1,kz,il), ix=1,ncol), kz=1,nz), il=1,NA)
  WRITE(*,'(*(ES26.17E3,1X))') ((t0(ix,1,kz), ix=1,ncol), kz=1,nz)
  WRITE(*,'(*(ES26.17E3,1X))') ((ssfilt(ix,1,kz), ix=1,ncol), kz=1,nz)
END PROGRAM nucond_direct
