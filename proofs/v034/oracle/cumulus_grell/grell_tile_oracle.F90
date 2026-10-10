! v0.3.4 o1-grell: multi-column pristine-WRF oracle for cu_physics=5 (G3DRV +
! conv_grell_spread3d, exactly the module_cumulus_driver call sequence) and
! cu_physics=93 (GRELLDRV).  Calls the UNMODIFIED pristine WRF Fortran only.
!
! Build with -DSCHEME_G3 or -DSCHEME_GD; REAL kind follows the compiler default
! (fp64 savepoints: -fdefault-real-8 -fdefault-double-8; fp32: WRF REAL).
!
! Input  (stream, little endian): int32 header(8) = nx ny kx ishallow cugd_avedx
!         ichoice periodic itimestep; float64 dt dx; then float64 3-D arrays
!         (nx,kx+1,ny) in Fortran order: u v w t q p pi rho dz8w p8w rthften
!         rqvften rthraten rthblten rqvblten; then float64 2-D (nx,ny): ht xland
!         gsw kpbl htop_in hbot_in.
! Output (stream): float64 arrays in the fixed order of write_outputs below.
program grell_tile_oracle
#ifdef SCHEME_G3
  use module_cu_g3, only: G3DRV, conv_grell_spread3d
#endif
#ifdef SCHEME_GD
  use module_cu_gd, only: GRELLDRV
#endif
  implicit none
  integer, parameter :: d8 = selected_real_kind(15)
  integer, parameter :: ensdim = 144, maxiens = 1, maxens = 3, maxens2 = 3, maxens3 = 16
  integer, parameter :: halo = 3
  integer :: hdr(8), nx, ny, kx, ishallow, cugd_avedx, ichoice, iperiodic, itimestep
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, i, j, k, n, u_in, u_out
  real(d8) :: rd2(2)
  real :: dt, dx, xlv, cp, grav, r_v
  logical :: periodic
  character(len=512) :: fin, fout
  real, allocatable, dimension(:,:,:) :: u, v, w, t, q, p, pi, rho, dz8w, p8w
  real, allocatable, dimension(:,:,:) :: rthften, rqvften, rthraten, rthblten, rqvblten
  real, allocatable, dimension(:,:,:) :: rthcuten, rqvcuten, rqccuten, rqicuten
  real, allocatable, dimension(:,:,:) :: cugd_tten, cugd_qvten, cugd_qcten, cugd_ttens, cugd_qvtens
  real, allocatable, dimension(:,:,:) :: gdc, gdc2, xf_ens, pr_ens
  real, allocatable, dimension(:,:) :: raincv, pratec, htop, hbot, apr_gr, apr_w, apr_mc, apr_st
  real, allocatable, dimension(:,:) :: apr_as, apr_capma, apr_capme, apr_capmi, mass_flux
  real, allocatable, dimension(:,:) :: ht, xland, gsw, edt_out, xmb_shallow
  real, allocatable, dimension(:,:) :: g3_raincv, g3_pratec
  integer, allocatable, dimension(:,:) :: kpbl, k22_shallow, kbcon_shallow, ktop_shallow, ktop_deep
  logical, allocatable, dimension(:,:) :: cu_act_flag
  real(d8), allocatable :: buf3(:,:,:), buf2(:,:)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u_in, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u_in) hdr
  nx = hdr(1); ny = hdr(2); kx = hdr(3); ishallow = hdr(4); cugd_avedx = hdr(5)
  ichoice = hdr(6); iperiodic = hdr(7); itimestep = hdr(8)
  periodic = (iperiodic /= 0)
  read(u_in) rd2
  dt = real(rd2(1)); dx = real(rd2(2))

  ! WRF module_model_constants values handed to the cumulus driver.
  xlv = 2.5e6; cp = 7.*287./2.; grav = 9.81; r_v = 461.6

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = kx + 1
  ims = ids - halo; ime = ide + halo; jms = jds - halo; jme = jde + halo; kms = 1; kme = kx + 1
  its = 1; ite = nx; jts = 1; jte = ny; kts = 1; kte = kx

  allocate(buf3(nx, kx + 1, ny), buf2(nx, ny))
  call alloc3(u); call alloc3(v); call alloc3(w); call alloc3(t); call alloc3(q)
  call alloc3(p); call alloc3(pi); call alloc3(rho); call alloc3(dz8w); call alloc3(p8w)
  call alloc3(rthften); call alloc3(rqvften); call alloc3(rthraten); call alloc3(rthblten)
  call alloc3(rqvblten); call alloc3(rthcuten); call alloc3(rqvcuten); call alloc3(rqccuten)
  call alloc3(rqicuten); call alloc3(cugd_tten); call alloc3(cugd_qvten); call alloc3(cugd_qcten)
  call alloc3(cugd_ttens); call alloc3(cugd_qvtens); call alloc3(gdc); call alloc3(gdc2)
  allocate(xf_ens(ims:ime, jms:jme, ensdim), pr_ens(ims:ime, jms:jme, ensdim))
  xf_ens = 0.; pr_ens = 0.
  call alloc2(raincv); call alloc2(pratec); call alloc2(htop); call alloc2(hbot)
  call alloc2(apr_gr); call alloc2(apr_w); call alloc2(apr_mc); call alloc2(apr_st)
  call alloc2(apr_as); call alloc2(apr_capma); call alloc2(apr_capme); call alloc2(apr_capmi)
  call alloc2(mass_flux); call alloc2(ht); call alloc2(xland); call alloc2(gsw)
  call alloc2(edt_out); call alloc2(xmb_shallow); call alloc2(g3_raincv); call alloc2(g3_pratec)
  allocate(kpbl(ims:ime, jms:jme), k22_shallow(ims:ime, jms:jme), kbcon_shallow(ims:ime, jms:jme))
  allocate(ktop_shallow(ims:ime, jms:jme), ktop_deep(ims:ime, jms:jme), cu_act_flag(ims:ime, jms:jme))
  kpbl = 1; k22_shallow = 0; kbcon_shallow = 0; ktop_shallow = 0; ktop_deep = 0
  cu_act_flag = .true.

  call read3(u); call read3(v); call read3(w); call read3(t); call read3(q)
  call read3(p); call read3(pi); call read3(rho); call read3(dz8w); call read3(p8w)
  call read3(rthften); call read3(rqvften); call read3(rthraten); call read3(rthblten)
  call read3(rqvblten)
  call read2(ht); call read2(xland); call read2(gsw)
  read(u_in) buf2
  kpbl(1:nx, 1:ny) = nint(buf2)
  call read2(htop); call read2(hbot)
  close(u_in)

#ifdef SCHEME_G3
  ! module_cumulus_driver.F:876-881 (EM_CORE, G3SCHEME) pre-scales the forcing.
  do j = jts, min(jte, jde - 1)
    do k = kts, kte
      do i = its, min(ite, ide - 1)
        rthften(i,k,j) = (rthften(i,k,j) + rthraten(i,k,j) + rthblten(i,k,j)) * pi(i,k,j)
        rqvften(i,k,j) = rqvften(i,k,j) + rqvblten(i,k,j)
      end do
    end do
  end do
  call G3DRV(DT=dt, ITIMESTEP=itimestep, DX=dx, U=u, V=v, T=t, W=w, RHO=rho, &
       P=p, PI=pi, Q=q, RAINCV=raincv, DZ8W=dz8w, P8W=p8w, XLV=xlv, CP=cp, G=grav, R_V=r_v, &
       APR_GR=apr_gr, APR_W=apr_w, APR_MC=apr_mc, APR_ST=apr_st, APR_AS=apr_as, PRATEC=pratec, &
       APR_CAPMA=apr_capma, APR_CAPME=apr_capme, APR_CAPMI=apr_capmi, MASS_FLUX=mass_flux, &
       XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht, xland=xland, gsw=gsw, edt_out=edt_out, &
       GDC=gdc, GDC2=gdc2, kpbl=kpbl, k22_shallow=k22_shallow, kbcon_shallow=kbcon_shallow, &
       ktop_shallow=ktop_shallow, xmb_shallow=xmb_shallow, ktop_deep=ktop_deep, &
       cugd_tten=cugd_tten, cugd_qvten=cugd_qvten, cugd_ttens=cugd_ttens, cugd_qvtens=cugd_qvtens, &
       cugd_qcten=cugd_qcten, cugd_avedx=cugd_avedx, imomentum=0, ishallow_g3=ishallow, &
       ENSDIM=ensdim, MAXIENS=maxiens, MAXENS=maxens, MAXENS2=maxens2, MAXENS3=maxens3, &
       ichoice=ichoice, htop=htop, hbot=hbot, CU_ACT_FLAG=cu_act_flag, warm_rain=.false., &
       IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
       IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, &
       IPS=its, IPE=ite, JPS=jts, JPE=jte, KPS=kts, KPE=kte, &
       ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
       PERIODIC_X=periodic, PERIODIC_Y=periodic, &
       RTHCUTEN=rthcuten, RTHFTEN=rthften, RQICUTEN=rqicuten, RQVFTEN=rqvften, &
       rqvblten=rqvblten, rthblten=rthblten, RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, &
       F_QV=.true., F_QC=.true., F_QR=.true., F_QI=.true., F_QS=.true.)
  g3_raincv = raincv
  g3_pratec = pratec
  ! module_cumulus_driver.F:1630-1652 (cu_physics == 5, single tile).
  call conv_grell_spread3d(rthcuten=rthcuten, rqvcuten=rqvcuten, &
       rqccuten=rqccuten, raincv=raincv, cugd_avedx=cugd_avedx, &
       cugd_tten=cugd_tten, cugd_qvten=cugd_qvten, rqicuten=rqicuten, &
       cugd_ttens=cugd_ttens, cugd_qvtens=cugd_qvtens, &
       cugd_qcten=cugd_qcten, pi_phy=pi, moist_qv=q, &
       PRATEC=pratec, dt=dt, num_tiles=1, imomentum=0, &
       F_QV=.true., F_QC=.true., F_QR=.true., F_QI=.true., F_QS=.true., &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
       ips=its, ipe=ite, jps=jts, jpe=jte, kps=kts, kpe=kte, &
       ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
#endif
#ifdef SCHEME_GD
  call GRELLDRV(DT=dt, ITIMESTEP=itimestep, DX=dx, RHO=rho, RAINCV=raincv, PRATEC=pratec, &
       U=u, V=v, T=t, W=w, Q=q, P=p, PI=pi, DZ8W=dz8w, P8W=p8w, XLV=xlv, CP=cp, G=grav, R_V=r_v, &
       HTOP=htop, HBOT=hbot, KTOP_DEEP=ktop_deep, CU_ACT_FLAG=cu_act_flag, WARM_RAIN=.false., &
       APR_GR=apr_gr, APR_W=apr_w, APR_MC=apr_mc, APR_ST=apr_st, APR_AS=apr_as, &
       APR_CAPMA=apr_capma, APR_CAPME=apr_capme, APR_CAPMI=apr_capmi, MASS_FLUX=mass_flux, &
       XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht, XLAND=xland, GSW=gsw, GDC=gdc, GDC2=gdc2, &
       ENSDIM=ensdim, MAXIENS=maxiens, MAXENS=maxens, MAXENS2=maxens2, MAXENS3=maxens3, &
       IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, IMS=ims, IME=ime, JMS=jms, JME=jme, &
       KMS=kms, KME=kme, ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
       PERIODIC_X=periodic, PERIODIC_Y=periodic, RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, &
       RQICUTEN=rqicuten, RQVFTEN=rqvften, RQVBLTEN=rqvblten, RTHFTEN=rthften, &
       RTHCUTEN=rthcuten, RTHRATEN=rthraten, RTHBLTEN=rthblten, F_QV=.true., F_QC=.true., &
       F_QR=.true., F_QI=.true., F_QS=.true.)
  g3_raincv = raincv
  g3_pratec = pratec
#endif

  open(newunit=u_out, file=trim(fout), access='stream', form='unformatted', status='replace')
  call write3(rthcuten); call write3(rqvcuten); call write3(rqccuten); call write3(rqicuten)
  call write3(cugd_tten); call write3(cugd_qvten); call write3(cugd_qcten)
  call write3(cugd_ttens); call write3(cugd_qvtens); call write3(gdc); call write3(gdc2)
  call write2(raincv); call write2(pratec); call write2(g3_raincv); call write2(g3_pratec)
  call write2(htop); call write2(hbot); call write2(mass_flux); call write2(edt_out)
  call write2(apr_gr); call write2(apr_w); call write2(apr_mc); call write2(apr_st)
  call write2(apr_as); call write2(apr_capma); call write2(apr_capme); call write2(apr_capmi)
  call write2(xmb_shallow)
  call write2i(ktop_deep); call write2i(k22_shallow); call write2i(kbcon_shallow); call write2i(ktop_shallow)
  do n = 1, ensdim
    buf2 = real(xf_ens(1:nx, 1:ny, n), d8)
    write(u_out) buf2
  end do
  do n = 1, ensdim
    buf2 = real(pr_ens(1:nx, 1:ny, n), d8)
    write(u_out) buf2
  end do
  close(u_out)

contains
  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme))
    a = 0.
  end subroutine alloc3
  subroutine alloc2(a)
    real, allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime, jms:jme))
    a = 0.
  end subroutine alloc2
  subroutine read3(a)
    real, intent(inout) :: a(ims:ime, kms:kme, jms:jme)
    read(u_in) buf3
    a(1:nx, 1:kx+1, 1:ny) = real(buf3)
  end subroutine read3
  subroutine read2(a)
    real, intent(inout) :: a(ims:ime, jms:jme)
    read(u_in) buf2
    a(1:nx, 1:ny) = real(buf2)
  end subroutine read2
  subroutine write3(a)
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
    buf3 = real(a(1:nx, 1:kx+1, 1:ny), d8)
    write(u_out) buf3
  end subroutine write3
  subroutine write2(a)
    real, intent(in) :: a(ims:ime, jms:jme)
    buf2 = real(a(1:nx, 1:ny), d8)
    write(u_out) buf2
  end subroutine write2
  subroutine write2i(a)
    integer, intent(in) :: a(ims:ime, jms:jme)
    buf2 = real(a(1:nx, 1:ny), d8)
    write(u_out) buf2
  end subroutine write2i
end program grell_tile_oracle
