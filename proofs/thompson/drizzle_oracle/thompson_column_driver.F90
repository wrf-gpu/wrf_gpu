! LL01 copy of TH08 (hgt3d padded to 2 j-rows: thompson_init reads hgt(its+1,k,jts+1), :473-476, outside a 1-row tile).
! TH08: pristine WRF mp_gt_driver (Thompson mp8, not aerosol-aware) on N adversarial columns.
! Input  (stream, native endian): int32 ncol, nlev; real32 dt; then 13 real32 arrays (ncol,nlev) in order
!        qv qc qr qi qs qg ni nr th pii p w dz8w   (i fastest, Fortran order)
! Output (stream): the same 9 prognostics after one step + rainncv snowncv graupelncv sr (ncol each).
PROGRAM thompson_column_driver
  USE module_mp_thompson, ONLY : thompson_init, mp_gt_driver
  IMPLICIT NONE
  INTEGER :: ncol, nlev, i, k
  REAL :: dt
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: qv, qc, qr, qi, qs, qg, ni, nr, th, pii, p, w, dz8w
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: refl_10cm, re_cloud, re_ice, re_snow, hgt3d
  REAL, ALLOCATABLE, DIMENSION(:,:) :: RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAUPELNC, GRAUPELNCV, SR
  INTEGER :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  INTEGER :: itimestep, has_reqc, has_reqi, has_reqs, do_radar_ref, ke_diag
  LOGICAL :: diagflag
  CHARACTER(LEN=512) :: fin, fout
  CALL get_command_argument(1, fin); CALL get_command_argument(2, fout)
  OPEN(10, FILE=TRIM(fin), ACCESS='stream', FORM='unformatted', STATUS='old')
  READ(10) ncol, nlev, dt
  ids=1; ide=ncol+1; jds=1; jde=2; kds=1; kde=nlev+1
  ims=1; ime=ncol; jms=1; jme=1; kms=1; kme=nlev
  its=1; ite=ncol; jts=1; jte=1; kts=1; kte=nlev
  ALLOCATE(qv(ncol,nlev,1), qc(ncol,nlev,1), qr(ncol,nlev,1), qi(ncol,nlev,1), qs(ncol,nlev,1), qg(ncol,nlev,1))
  ALLOCATE(ni(ncol,nlev,1), nr(ncol,nlev,1), th(ncol,nlev,1), pii(ncol,nlev,1), p(ncol,nlev,1), w(ncol,nlev,1), dz8w(ncol,nlev,1))
  ALLOCATE(refl_10cm(ncol,nlev,1), re_cloud(ncol,nlev,1), re_ice(ncol,nlev,1), re_snow(ncol,nlev,1), hgt3d(ncol,nlev,2))
  ALLOCATE(RAINNC(ncol,1), RAINNCV(ncol,1), SNOWNC(ncol,1), SNOWNCV(ncol,1), GRAUPELNC(ncol,1), GRAUPELNCV(ncol,1), SR(ncol,1))
  READ(10) qv, qc, qr, qi, qs, qg, ni, nr, th, pii, p, w, dz8w
  CLOSE(10)
  refl_10cm = 0.; re_cloud = 0.; re_ice = 0.; re_snow = 0.
  hgt3d = 0.
  DO i = 1, ncol
    hgt3d(i,1,1) = 0.
    DO k = 2, nlev
      hgt3d(i,k,1) = hgt3d(i,k-1,1) + dz8w(i,k-1,1)
    END DO
  END DO
  RAINNC = 0.; RAINNCV = 0.; SNOWNC = 0.; SNOWNCV = 0.; GRAUPELNC = 0.; GRAUPELNCV = 0.; SR = 0.
  itimestep = 1; diagflag = .FALSE.; do_radar_ref = 0; ke_diag = nlev; has_reqc = 0; has_reqi = 0; has_reqs = 0
  CALL thompson_init(HGT=hgt3d, IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
       IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte)
  CALL mp_gt_driver(QV=qv, QC=qc, QR=qr, QI=qi, QS=qs, QG=qg, NI=ni, NR=nr, TH=th, PII=pii, P=p, W=w, DZ=dz8w, &
       DT_IN=dt, ITIMESTEP=itimestep, RAINNC=RAINNC, RAINNCV=RAINNCV, SNOWNC=SNOWNC, SNOWNCV=SNOWNCV, &
       GRAUPELNC=GRAUPELNC, GRAUPELNCV=GRAUPELNCV, SR=SR, refl_10cm=refl_10cm, diagflag=diagflag, &
       ke_diag=ke_diag, do_radar_ref=do_radar_ref, re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
       has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
  OPEN(11, FILE=TRIM(fout), ACCESS='stream', FORM='unformatted', STATUS='replace')
  WRITE(11) qv, qc, qr, qi, qs, qg, ni, nr, th, RAINNCV, SNOWNCV, GRAUPELNCV, SR
  CLOSE(11)
END PROGRAM thompson_column_driver
