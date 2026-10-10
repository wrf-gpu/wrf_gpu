! SWIDX: pristine WRF V4.7.1 CAM shortwave (radctl dosw block, radforce = .false.) on given radctl operands, with the
! Registry aerosolc scalar indices either left at their declared default 1 (mode 0, = CAM01 driver state) or set as
! set_scalar_indices_from_config does for ra_sw_physics = 3 (mode 1: P_SUL..P_VOLC = 2..13).
! Unchanged pristine routines from libwrflib.a (camradinit, aqsat, get_int_scales, get_aerosol, radinp, radcswmx);
! this driver reproduces radctl's SW block literally (rh expression, call order) and dumps radcswmx's raw (cgs) outputs.
! Input/Output: little-endian stream (library built -fconvert=big-endian -> CONVERT='native').
program swidx_driver
  use module_ra_cam, only: camradinit, radcswmx, get_aerosol, radinp
  use module_ra_cam_support, only: r8, aqsat, get_int_scales, epsilo, naer, naer_all, mxaerl, co2mmr, scon
  use module_state_description, only: P_sul, P_sslt, P_dust1, P_dust2, P_dust3, P_dust4, P_ocpho, P_bcpho, &
                                      P_ocphi, P_bcphi, P_bg, P_volc
  use module_model_constants, only: R_D, R_V, CP, G, STBOLT, EP_2
  implicit none
  integer, parameter :: levsiz = 59, n_ozmixm = 13, paerlev = 29, n_aerosolc = 13, nspint = 19, naer_groups = 7
  integer :: ncol, nz, mode, ic, pver, pverp, hdr(3), nmx1(1), julday
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  real :: p_top, pptop, gmt, dt, xtime
  real, allocatable :: znu(:), shalf(:), ozmixm(:,:,:,:), pin(:), m_psp(:,:), m_psn(:,:), m_hybi(:), aerc1(:,:,:,:), &
                       aerc2(:,:,:,:), xlat(:,:)
  real :: solcon, alb, julian
  real(r8) :: co2in, eccf
  real(r8), allocatable :: q1(:,:), cld(:,:), pmid(:,:), pint(:,:), tt(:,:), cicewp(:,:), cliqwp(:,:), rel(:,:), &
                           rei(:,:), pmxrgn(:,:), o3vmr(:,:), esat(:,:), qsat(:,:), rh(:,:), aerosol(:,:,:), &
                           pbr(:,:), pnm(:,:), o3mmr(:,:), tauxcl(:,:), tauxci(:,:), qrs(:,:), qrscs(:,:), &
                           fsup(:,:), fsupc(:,:), fsdn(:,:), fsdnc(:,:), fsdndir(:,:), fsdncdir(:,:), fsdndif(:,:), &
                           fsdncdif(:,:), aerjp(:,:,:), aerjn(:,:,:), m_hybi8(:)
  real(r8) :: coszrs(1), asdir(1), asdif(1), aldir(1), aldif(1), solin(1), fsnt(1), fsntc(1), fsntoa(1), fsntoac(1), &
              fsnirt(1), fsnrtc(1), fsnirtsq(1), fsns(1), fsnsc(1), fsdsc(1), fsds(1), sols(1), soll(1), solsd(1), &
              solld(1), frc_day(1), fsdsdir(1), fsdsdif(1), fsdscdir(1), fsdscdif(1), m_psjp(1), m_psjn(1), &
              scales(naer_all)
  real(r8) :: aertau(1,nspint,naer_groups), aerssa(1,nspint,naer_groups), aerasm(1,nspint,naer_groups), &
              aerfwd(1,nspint,naer_groups)
  integer :: k, n
  character(len=256) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', convert='native')
  read(10) hdr
  ncol = hdr(1); nz = hdr(2); mode = hdr(3)
  pver = nz; pverp = nz + 1
  allocate(znu(nz))
  read(10) p_top
  read(10) znu

  if (mode == 1) then   ! set_scalar_indices_from_config, ra_sw_physics = 3 (aerosolc: dummy slot 1, then 2..13)
    P_sul = 2; P_sslt = 3; P_dust1 = 4; P_dust2 = 5; P_dust3 = 6; P_dust4 = 7
    P_ocpho = 8; P_bcpho = 9; P_ocphi = 10; P_bcphi = 11; P_bg = 12; P_volc = 13
  end if

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz
  allocate(shalf(kms:kme), ozmixm(ims:ime,levsiz,jms:jme,n_ozmixm), pin(levsiz), m_psp(ims:ime,jms:jme), &
           m_psn(ims:ime,jms:jme), m_hybi(paerlev), aerc1(ims:ime,paerlev,jms:jme,n_aerosolc), &
           aerc2(ims:ime,paerlev,jms:jme,n_aerosolc), xlat(ims:ime,jms:jme))
  shalf = 0.; shalf(1:nz) = znu(1:nz)
  pptop = p_top / 1000.
  ozmixm = 0.; pin = 0.; m_psp = 0.; m_psn = 0.; m_hybi = 0.; aerc1 = 0.; aerc2 = 0.; xlat = 28.
  call camradinit(R_D, R_V, CP, G, STBOLT, EP_2, shalf, pptop, ozmixm, pin, levsiz, xlat, n_ozmixm, &
                  m_psp, m_psn, m_hybi, aerc1, aerc2, paerlev, n_aerosolc, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

  allocate(q1(1,pver), cld(1,pver), pmid(1,pver), pint(1,pverp), tt(1,pver), cicewp(1,pver), cliqwp(1,pver), &
           rel(1,pver), rei(1,pver), pmxrgn(1,pverp), o3vmr(1,pver), esat(1,pver), qsat(1,pver), rh(1,pver), &
           aerosol(1,pver,naer_all), pbr(1,pver), pnm(1,pverp), o3mmr(1,pver), tauxcl(1,0:pver), tauxci(1,0:pver), &
           qrs(1,pver), qrscs(1,pver), fsup(1,pverp), fsupc(1,pverp), fsdn(1,pverp), fsdnc(1,pverp), &
           fsdndir(1,pverp), fsdncdir(1,pverp), fsdndif(1,pverp), fsdncdif(1,pverp), &
           aerjp(1,paerlev,n_aerosolc), aerjn(1,paerlev,n_aerosolc), m_hybi8(paerlev))
  do n = 1, n_aerosolc
    do k = 1, paerlev
      aerjp(1,k,n) = aerc1(1,k,1,n); aerjn(1,k,n) = aerc2(1,k,1,n)
    end do
  end do
  m_psjp(1) = m_psp(1,1); m_psjn(1) = m_psn(1,1)
  m_hybi8 = m_hybi
  julday = 1; gmt = 0.; dt = 18.; xtime = 0.

  open(20, file=trim(fout), access='stream', form='unformatted', status='replace', convert='native')
  write(20) ncol, nz, mode, mxaerl
  write(20) m_hybi
  write(20) aerc1(1,:,1,:)
  do ic = 1, ncol
    read(10) q1(1,:), cld(1,:), pmid(1,:), pint(1,:), tt(1,:), cicewp(1,:), cliqwp(1,:), rel(1,:), rei(1,:), &
             pmxrgn(1,:), o3vmr(1,:)
    read(10) nmx1(1)
    read(10) coszrs(1), co2in
    read(10) solcon, alb, julian
    co2mmr = co2in
    asdir(1) = alb; asdif(1) = alb; aldir(1) = alb; aldif(1) = alb
    tauxcl = 0.; tauxci = 0.
    ! ---- radctl SW block (module_ra_cam.F:1785-1849, radforce = .false.)
    call aqsat(tt, pmid, esat, qsat, 1, 1, pver, 1, pver)
    rh(1:1,1:pver) = q1(1:1,1:pver) / qsat(1:1,1:pver) * &
       ((1.0 - epsilo) * qsat(1:1,1:pver) + epsilo) / &
       ((1.0 - epsilo) * q1(1:1,1:pver) + epsilo)
    call get_int_scales(scales)
    aerosol = 0.
    call get_aerosol(1, julday, julian, dt, gmt, xtime, m_psjp, m_psjn, aerjp, aerjn, m_hybi8, paerlev, naer, &
                     pint, 1, pver, pverp, pver, pverp, aerosol, scales)
    call radinp(1, 1, 1, pver, pverp, pmid, pint, o3vmr, pbr, pnm, eccf, o3mmr)
    call radcswmx(1, 1, 1, 1, pver, pverp, &
                  pnm, pbr, q1, rh, o3mmr, &
                  aerosol, cld, cicewp, cliqwp, rel, &
                  rei, tauxcl, tauxci, eccf, coszrs, scon, solin, solcon, &
                  asdir, asdif, aldir, aldif, nmx1, &
                  pmxrgn, qrs, qrscs, fsnt, fsntc, fsntoa, &
                  fsntoac, fsnirt, fsnrtc, fsnirtsq, fsns, &
                  fsnsc, fsdsc, fsds, sols, soll, &
                  solsd, solld, frc_day, &
                  fsup, fsupc, fsdn, fsdnc, &
                  fsdndir, fsdncdir, fsdndif, fsdncdif, &
                  fsdsdir, fsdsdif, fsdscdir, fsdscdif, &
                  aertau, aerssa, aerasm, aerfwd)
    write(20) rh(1,:), aerosol(1,:,:)
    write(20) qrs(1,:), qrscs(1,:), fsup(1,:), fsupc(1,:), fsdn(1,:), fsdnc(1,:), fsdndir(1,:), fsdndif(1,:), &
              fsdncdir(1,:), fsdncdif(1,:), tauxcl(1,1:pver), tauxci(1,1:pver)
    write(20) fsns(1), fsntoa(1), fsntoac(1), fsds(1), fsdsdir(1), fsdsdif(1), sols(1), soll(1), solsd(1), solld(1), &
              solin(1), nmx1(1), pmxrgn(1,1)
  end do
  close(10); close(20)
  print *, 'SWIDX columns', ncol, ' nz', nz, ' mode', mode, ' mxaerl', mxaerl
end program swidx_driver
