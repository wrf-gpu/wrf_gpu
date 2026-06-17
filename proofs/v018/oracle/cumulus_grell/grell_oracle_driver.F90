program grell_oracle_driver
  ! v0.18 standalone oracle harness; calls pristine WRF Fortran only.
  use, intrinsic :: iso_fortran_env, only: real64
#ifdef SCHEME_G3
  use module_cu_g3, only: G3DRV
#endif
#ifdef SCHEME_GD
  use module_cu_gd, only: GRELLDRV
#endif
  implicit none

  integer, parameter :: rk = real64
  integer, parameter :: kx = 45
  integer, parameter :: ids = 1, ide = 2, jds = 1, jde = 2, kds = 1, kde = kx + 1
  integer, parameter :: ims = 1, ime = 1, jms = 1, jme = 1, kms = 1, kme = kx
  integer, parameter :: ips = 1, ipe = 1, jps = 1, jpe = 1, kps = 1, kpe = kx
  integer, parameter :: its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = kx
  integer, parameter :: ensdim = 144, maxiens = 1, maxens = 3, maxens2 = 3, maxens3 = 16

  integer :: case_id, k
  character(len=16) :: arg
  real(rk) :: dt, dx, xlv, cp, grav, r_v, p0
  real(rk), dimension(kx) :: zlev
  character(len=48) :: regime
  real(rk), dimension(ims:ime,kms:kme,jms:jme) :: rho, u, v, t, w, q, p, pi, dz8w, p8w
  real(rk), dimension(ims:ime,kms:kme,jms:jme) :: rthcuten, rqvcuten, rqccuten, rqicuten
  real(rk), dimension(ims:ime,kms:kme,jms:jme) :: rthften, rqvften, rthblten, rqvblten, rthraten
  real(rk), dimension(ims:ime,kms:kme,jms:jme) :: cugd_tten, cugd_qvten, cugd_qcten, cugd_ttens, cugd_qvtens
  real(rk), dimension(ims:ime,kms:kme,jms:jme) :: gdc, gdc2
  real(rk), dimension(ims:ime,jms:jme) :: raincv, pratec, htop, hbot, apr_gr, apr_w, apr_mc
  real(rk), dimension(ims:ime,jms:jme) :: apr_st, apr_as, apr_capma, apr_capme, apr_capmi
  real(rk), dimension(ims:ime,jms:jme) :: mass_flux, ht, xland, gsw, edt_out, xmb_shallow
  real(rk), dimension(ims:ime,jms:jme,1:ensdim) :: xf_ens, pr_ens
  integer, dimension(ims:ime,jms:jme) :: kpbl, k22_shallow, kbcon_shallow, ktop_shallow, ktop_deep
  logical, dimension(ims:ime,jms:jme) :: cu_act_flag

  case_id = 1
  call get_command_argument(1, arg)
  if (len_trim(arg) > 0) read(arg, *) case_id

  xlv = 2.5e6_rk
  cp = 1004.0_rk
  grav = 9.81_rk
  r_v = 461.6_rk
  p0 = 100000.0_rk

  rho = 0.0_rk; u = 0.0_rk; v = 0.0_rk; t = 0.0_rk; w = 0.0_rk
  q = 0.0_rk; p = 0.0_rk; pi = 0.0_rk; dz8w = 0.0_rk; p8w = 0.0_rk
  rthcuten = 0.0_rk; rqvcuten = 0.0_rk; rqccuten = 0.0_rk; rqicuten = 0.0_rk
  rthften = 0.0_rk; rqvften = 0.0_rk; rthblten = 0.0_rk; rqvblten = 0.0_rk; rthraten = 0.0_rk
  cugd_tten = 0.0_rk; cugd_qvten = 0.0_rk; cugd_qcten = 0.0_rk; cugd_ttens = 0.0_rk; cugd_qvtens = 0.0_rk
  gdc = 0.0_rk; gdc2 = 0.0_rk
  raincv = 0.0_rk; pratec = 0.0_rk; htop = 0.0_rk; hbot = 0.0_rk
  apr_gr = 0.0_rk; apr_w = 0.0_rk; apr_mc = 0.0_rk; apr_st = 0.0_rk; apr_as = 0.0_rk
  apr_capma = 0.0_rk; apr_capme = 0.0_rk; apr_capmi = 0.0_rk
  mass_flux = 0.0_rk; ht = 0.0_rk; xland = 1.0_rk; gsw = 700.0_rk; edt_out = 0.0_rk; xmb_shallow = 0.0_rk
  xf_ens = 0.0_rk; pr_ens = 0.0_rk
  kpbl = 6; k22_shallow = 0; kbcon_shallow = 0; ktop_shallow = 0; ktop_deep = 0
  cu_act_flag = .true.

  call build_sounding(case_id, t, q, p, dz8w, rho, u, v, w, zlev, dt, dx, kpbl(1,1), xland(1,1), regime)
  do k = kts, kte
    pi(1,k,1) = (p(1,k,1) / p0) ** (287.0_rk / cp)
    p8w(1,k,1) = p(1,k,1)
  end do
  do k = 1, min(kpbl(1,1), kx)
    if (case_id == 2) then
      rthblten(1,k,1) = 1.2e-4_rk
      rqvblten(1,k,1) = 1.0e-8_rk
    else if (case_id == 1 .or. case_id == 4 .or. case_id == 5) then
      rthblten(1,k,1) = 7.5e-5_rk
      rqvblten(1,k,1) = 8.0e-9_rk
    end if
  end do

#ifdef SCHEME_G3
  call G3DRV(DT=dt, ITIMESTEP=case_id, DX=dx, RHO=rho, RAINCV=raincv, PRATEC=pratec, &
       U=u, V=v, T=t, W=w, Q=q, P=p, PI=pi, DZ8W=dz8w, P8W=p8w, XLV=xlv, CP=cp, G=grav, R_V=r_v, &
       HTOP=htop, HBOT=hbot, CU_ACT_FLAG=cu_act_flag, WARM_RAIN=.false., APR_GR=apr_gr, APR_W=apr_w, &
       APR_MC=apr_mc, APR_ST=apr_st, APR_AS=apr_as, APR_CAPMA=apr_capma, APR_CAPME=apr_capme, &
       APR_CAPMI=apr_capmi, MASS_FLUX=mass_flux, XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht, XLAND=xland, &
       GSW=gsw, EDT_OUT=edt_out, GDC=gdc, GDC2=gdc2, KPBL=kpbl, K22_SHALLOW=k22_shallow, &
       KBCON_SHALLOW=kbcon_shallow, KTOP_SHALLOW=ktop_shallow, XMB_SHALLOW=xmb_shallow, &
       KTOP_DEEP=ktop_deep, CUGD_TTEN=cugd_tten, CUGD_QVTEN=cugd_qvten, CUGD_QCTEN=cugd_qcten, &
       CUGD_TTENS=cugd_ttens, CUGD_QVTENS=cugd_qvtens, CUGD_AVEDX=1, IMOMENTUM=0, &
       ENSDIM=ensdim, MAXIENS=maxiens, MAXENS=maxens, MAXENS2=maxens2, MAXENS3=maxens3, ICHOICE=0, &
       ISHALLOW_G3=0, IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
       IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, IPS=ips, IPE=ipe, JPS=jps, JPE=jpe, &
       KPS=kps, KPE=kpe, ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
       PERIODIC_X=.true., PERIODIC_Y=.true., RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, &
       RQICUTEN=rqicuten, RQVFTEN=rqvften, RTHFTEN=rthften, RTHCUTEN=rthcuten, &
       RQVBLTEN=rqvblten, RTHBLTEN=rthblten, F_QV=.true., F_QC=.true., F_QR=.true., F_QI=.true., F_QS=.true.)
#endif
#ifdef SCHEME_GD
  call GRELLDRV(DT=dt, ITIMESTEP=case_id, DX=dx, RHO=rho, RAINCV=raincv, PRATEC=pratec, &
       U=u, V=v, T=t, W=w, Q=q, P=p, PI=pi, DZ8W=dz8w, P8W=p8w, XLV=xlv, CP=cp, G=grav, R_V=r_v, &
       HTOP=htop, HBOT=hbot, KTOP_DEEP=ktop_deep, CU_ACT_FLAG=cu_act_flag, WARM_RAIN=.false., &
       APR_GR=apr_gr, APR_W=apr_w, APR_MC=apr_mc, APR_ST=apr_st, APR_AS=apr_as, &
       APR_CAPMA=apr_capma, APR_CAPME=apr_capme, APR_CAPMI=apr_capmi, MASS_FLUX=mass_flux, &
       XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht, XLAND=xland, GSW=gsw, GDC=gdc, GDC2=gdc2, &
       ENSDIM=ensdim, MAXIENS=maxiens, MAXENS=maxens, MAXENS2=maxens2, MAXENS3=maxens3, &
       IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, IMS=ims, IME=ime, JMS=jms, JME=jme, &
       KMS=kms, KME=kme, ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
       PERIODIC_X=.true., PERIODIC_Y=.true., RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, &
       RQICUTEN=rqicuten, RQVFTEN=rqvften, RQVBLTEN=rqvblten, RTHFTEN=rthften, &
       RTHCUTEN=rthcuten, RTHRATEN=rthraten, RTHBLTEN=rthblten, F_QV=.true., F_QC=.true., &
       F_QR=.true., F_QI=.true., F_QS=.true.)
#endif

  print '(A,I0)', 'CASE ', case_id
  print '(A,A)', 'REGIME ', trim(regime)
#ifdef SCHEME_G3
  print '(A)', 'SCHEME G3'
#endif
#ifdef SCHEME_GD
  print '(A)', 'SCHEME GD'
#endif
  print '(A,1PE24.16)', 'SCALAR RAINCV ', raincv(1,1)
  print '(A,1PE24.16)', 'SCALAR PRATEC ', pratec(1,1)
  print '(A,1PE24.16)', 'SCALAR HTOP ', htop(1,1)
  print '(A,1PE24.16)', 'SCALAR HBOT ', hbot(1,1)
  call print_column('T', t)
  call print_column('QV', q)
  call print_column('P', p)
  call print_column('RTHCUTEN', rthcuten)
  call print_column('RQVCUTEN', rqvcuten)
  call print_column('RQCCUTEN', rqccuten)
  call print_column('RQICUTEN', rqicuten)
  call print_column('CUGD_TTEN', cugd_tten)
  call print_column('CUGD_QVTEN', cugd_qvten)
  call print_column('CUGD_QCTEN', cugd_qcten)
  call print_column('CUGD_TTENS', cugd_ttens)
  call print_column('CUGD_QVTENS', cugd_qvtens)

contains
  subroutine print_column(name, arr)
    character(len=*), intent(in) :: name
    real(rk), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: arr
    integer :: kk
    print '(A,A)', 'COLUMN ', trim(name)
    do kk = kts, kte
      print '(I0,1X,1PE24.16)', kk, arr(1,kk,1)
    end do
  end subroutine print_column

  real(rk) function qsat_liq(temp, pres)
    real(rk), intent(in) :: temp, pres
    real(rk) :: es
    es = 611.2_rk * exp(17.67_rk * (temp - 273.15_rk) / (temp - 29.65_rk))
    es = min(es, 0.95_rk * pres)
    qsat_liq = 0.622_rk * es / (pres - es)
  end function qsat_liq

  subroutine build_sounding(cid, tt, qq, pp, dz, rr, uu, vv, ww, zz, dt_out, dx_out, kpbl_out, xland_out, regime_out)
    integer, intent(in) :: cid
    real(rk), dimension(ims:ime,kms:kme,jms:jme), intent(out) :: tt, qq, pp, dz, rr, uu, vv, ww
    real(rk), dimension(kx), intent(out) :: zz
    real(rk), intent(out) :: dt_out, dx_out, xland_out
    integer, intent(out) :: kpbl_out
    character(len=48), intent(out) :: regime_out
    real(rk) :: psfc, tsfc, zml, lapse_theta, upper_lapse, upper_start
    real(rk) :: rh_ml, rh_free, ushr, wpeak, cap_amp
    real(rk) :: theta_sfc, ztop, z_k, th_k, t_k, q_k, p_k, tv_k, rh_k, qs
    integer :: kk

    ztop = 19000.0_rk
    do kk = 1, kx
      zz(kk) = ztop * ((real(kk, rk) - 0.5_rk) / real(kx, rk)) ** 1.18_rk
    end do

    select case (cid)
    case (1)
      regime_out = 'deep_convective'
      psfc = 101200.0_rk; tsfc = 310.0_rk; zml = 1800.0_rk; lapse_theta = 1.8e-3_rk
      upper_lapse = 1.25e-2_rk; upper_start = 9000.0_rk
      rh_ml = 0.98_rk; rh_free = 0.82_rk; ushr = 8.0_rk; wpeak = 4.0_rk; cap_amp = 0.0_rk
      dx_out = 12000.0_rk; dt_out = 120.0_rk; kpbl_out = 9; xland_out = 1.0_rk
    case (2)
      regime_out = 'shallow_convective'
      psfc = 100500.0_rk; tsfc = 301.0_rk; zml = 900.0_rk; lapse_theta = 4.7e-3_rk
      upper_lapse = 8.0e-3_rk; upper_start = 4500.0_rk
      rh_ml = 0.89_rk; rh_free = 0.38_rk; ushr = 3.0_rk; wpeak = 0.42_rk; cap_amp = 3.2_rk
      dx_out = 9000.0_rk; dt_out = 54.0_rk; kpbl_out = 5; xland_out = 1.0_rk
    case (3)
      regime_out = 'stable_nontriggering'
      psfc = 100000.0_rk; tsfc = 286.0_rk; zml = 250.0_rk; lapse_theta = 1.05e-2_rk
      upper_lapse = 0.0_rk; upper_start = 19000.0_rk
      rh_ml = 0.36_rk; rh_free = 0.12_rk; ushr = 0.0_rk; wpeak = -0.08_rk; cap_amp = 0.0_rk
      dx_out = 9000.0_rk; dt_out = 54.0_rk; kpbl_out = 2; xland_out = 1.0_rk
    case (4)
      regime_out = 'scale_aware_coarse_15km'
      psfc = 101200.0_rk; tsfc = 310.0_rk; zml = 1800.0_rk; lapse_theta = 1.8e-3_rk
      upper_lapse = 1.25e-2_rk; upper_start = 9000.0_rk
      rh_ml = 0.98_rk; rh_free = 0.82_rk; ushr = 8.0_rk; wpeak = 4.0_rk; cap_amp = 0.0_rk
      dx_out = 15000.0_rk; dt_out = 150.0_rk; kpbl_out = 9; xland_out = 1.0_rk
    case (5)
      regime_out = 'scale_aware_fine_3km'
      psfc = 101200.0_rk; tsfc = 310.0_rk; zml = 1800.0_rk; lapse_theta = 1.8e-3_rk
      upper_lapse = 1.25e-2_rk; upper_start = 9000.0_rk
      rh_ml = 0.98_rk; rh_free = 0.82_rk; ushr = 8.0_rk; wpeak = 4.0_rk; cap_amp = 0.0_rk
      dx_out = 3000.0_rk; dt_out = 36.0_rk; kpbl_out = 9; xland_out = 1.0_rk
    case default
      regime_out = 'default_deep'
      psfc = 100800.0_rk; tsfc = 302.0_rk; zml = 1200.0_rk; lapse_theta = 3.5e-3_rk
      upper_lapse = 8.0e-3_rk; upper_start = 9000.0_rk
      rh_ml = 0.92_rk; rh_free = 0.62_rk; ushr = 8.0_rk; wpeak = 1.1_rk; cap_amp = 0.0_rk
      dx_out = 9000.0_rk; dt_out = 54.0_rk; kpbl_out = 6; xland_out = 1.0_rk
    end select

    theta_sfc = tsfc * (p0 / psfc) ** (287.0_rk / cp)
    p_k = psfc
    do kk = 1, kx
      z_k = zz(kk)
      if (z_k <= zml) then
        th_k = theta_sfc
        rh_k = rh_ml
      else
        th_k = theta_sfc + lapse_theta * (z_k - zml)
        if (cap_amp > 0.0_rk .and. z_k > 1100.0_rk .and. z_k < 2600.0_rk) then
          th_k = th_k + cap_amp * exp(-((z_k - 1700.0_rk) / 550.0_rk) ** 2)
        end if
        if (upper_lapse > 0.0_rk .and. z_k > upper_start) then
          th_k = th_k + upper_lapse * (z_k - upper_start)
        end if
        rh_k = rh_free + (rh_ml - rh_free) * exp(-(z_k - zml) / 2300.0_rk)
      end if

      if (kk == 1) then
        t_k = th_k * (psfc / p0) ** (287.0_rk / cp)
        tv_k = t_k
        p_k = psfc * exp(-grav * zz(1) / (287.0_rk * tv_k))
      else
        t_k = th_k * (p_k / p0) ** (287.0_rk / cp)
        tv_k = t_k * (1.0_rk + 0.608_rk * qq(1,kk-1,1))
        p_k = p_k * exp(-grav * (zz(kk) - zz(kk-1)) / (287.0_rk * tv_k))
      end if

      t_k = th_k * (p_k / p0) ** (287.0_rk / cp)
      qs = qsat_liq(t_k, p_k)
      q_k = max(1.0e-7_rk, rh_k * qs)
      tv_k = t_k * (1.0_rk + 0.608_rk * q_k)

      tt(1,kk,1) = t_k
      qq(1,kk,1) = q_k
      pp(1,kk,1) = p_k
      rr(1,kk,1) = p_k / (287.0_rk * tv_k)
      if (kk == 1) then
        dz(1,kk,1) = 2.0_rk * zz(1)
      else
        dz(1,kk,1) = zz(kk) - zz(kk-1)
      end if
      uu(1,kk,1) = ushr * min(1.0_rk, z_k / 10000.0_rk)
      vv(1,kk,1) = 0.0_rk
      ww(1,kk,1) = wpeak * exp(-((z_k - 1700.0_rk) / 1300.0_rk) ** 2)
    end do
  end subroutine build_sounding
end program grell_oracle_driver
