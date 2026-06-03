program mynn_edmf_oracle
  ! WRF-vs-JAX MYNN-EDMF qv oracle.
  !
  ! Calls the REAL WRF DMP_mf + mynn_tendencies (linked from the pristine
  ! module_bl_mynnedmf.o) on a real d03 daytime-land column. Dumps the mass-flux
  ! solver arrays (s_aw, s_awqv, sub_sqv, det_sqv), the eddy-diffusivities
  ! (dfm/dfh/dfq), and the qv tendency Dqv -- for both the full-EDMF config
  ! (bl_mynn_edmf=1, WRF default) and an "ED-only" config (mass-flux arrays
  ! zeroed) that reproduces what the current gpuwrf mynn_pbl.py does.
  !
  ! The difference Dqv(EDMF) - Dqv(ED-only) is the missing-MF signal.
  !
  ! Inputs are read from a flat key=value text file (column_d03_12z.flat),
  ! written by emit_flat.py. Output: oracle_out.txt (flat key=value), parsed
  ! by the python proof reducer.
  use module_bl_mynnedmf_common, only: kind_phys, cp, r_d, p608, p1000mb, rcp, xlvcp
  use module_bl_mynnedmf, only: dmp_mf, mynn_tendencies
  implicit none

  integer, parameter :: NZMAX = 200
  integer :: kts, kte, nz, k, i, j
  real(kind_phys) :: delt, dx, ust, pblh, psfc, wspd, xland, flt, flq, flqv, flqc
  real(kind_phys) :: fltv, ts, uoce, voce, ps, th_sfc

  ! column profiles (kts:kte)
  real(kind_phys), dimension(NZMAX) :: u,v,w,th,tk,p,exner,rho,dz,qv,qc,qi,qs,qke
  real(kind_phys), dimension(NZMAX) :: thl,sqv,sqc,sqi,sqs,sqw
  real(kind_phys), dimension(NZMAX) :: qnc,qni,qnwfa,qnifa,qnbca,ozone
  real(kind_phys), dimension(NZMAX) :: tsq,qsq,cov,tcd,qcd,dfm,dfh,dfq,diss_heat
  real(kind_phys), dimension(NZMAX) :: thv, cldfra_bl1, qc_bl1, qi_bl1
  real(kind_phys), dimension(NZMAX) :: cldfra_bl1_old, qc_bl1_old
  real(kind_phys), dimension(NZMAX) :: vt1, vq1, sgm1, el1, tkeprod_up
  real(kind_phys), dimension(NZMAX+1) :: zw

  ! tendencies
  real(kind_phys), dimension(NZMAX) :: Du,Dv,Dth,Dqv,Dqc,Dqi,Dqs,Dqnc,Dqni
  real(kind_phys), dimension(NZMAX) :: Dqnwfa,Dqnifa,Dqnbca,Dozone

  ! mass-flux solver arrays (kts:kte+1)
  real(kind_phys), dimension(NZMAX+1) :: s_aw,s_awthl,s_awqt,s_awqv,s_awqc
  real(kind_phys), dimension(NZMAX+1) :: s_awu,s_awv,s_awqke,s_awqnc,s_awqni
  real(kind_phys), dimension(NZMAX+1) :: s_awqnwfa,s_awqnifa,s_awqnbca
  real(kind_phys), dimension(NZMAX+1) :: sd_aw,sd_awthl,sd_awqt,sd_awqv,sd_awqc
  real(kind_phys), dimension(NZMAX+1) :: sd_awqi,sd_awqnc,sd_awqni,sd_awqnwfa
  real(kind_phys), dimension(NZMAX+1) :: sd_awqnifa,sd_awu,sd_awv
  real(kind_phys), dimension(NZMAX) :: sub_thl,sub_sqv,sub_u,sub_v
  real(kind_phys), dimension(NZMAX) :: det_thl,det_sqv,det_sqc,det_u,det_v
  real(kind_phys), dimension(NZMAX+1) :: kmdz_eff

  ! edmf updraft outputs
  real(kind_phys), dimension(NZMAX) :: edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc
  integer :: ktop_plume
  real(kind_phys) :: maxmf, ztop_plume, maxwidth, Psig_shcu

  ! chem (unused)
  integer, parameter :: nchem = 1
  real(kind_phys), dimension(NZMAX, nchem) :: chem1
  real(kind_phys), dimension(NZMAX+1, nchem) :: s_awchem1
  logical :: mix_chem

  ! spp (unused)
  real(kind_phys), dimension(NZMAX) :: pattern_spp_pbl
  integer :: spp_pbl

  ! config
  integer :: bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_edmf_tke
  integer :: bl_mynn_mixscalars, bl_mynn_cloudmix, bl_mynn_mixqt
  logical :: FLAG_QC, FLAG_QI, FLAG_QNC, FLAG_QNI, FLAG_QS
  logical :: FLAG_QNWFA, FLAG_QNIFA, FLAG_QNBCA, FLAG_OZONE
  integer :: kpbl, mode

  character(len=256) :: infile, outfile
  integer :: u_out

  infile  = 'column_d03_12z.flat'
  outfile = 'oracle_out.txt'

  call read_column(infile, nz, delt, dx, ust, pblh, psfc, wspd, xland, &
       ts, ps, u, v, w, th, tk, p, exner, rho, dz, qv, qc, qi, qke, &
       flt, flq, flqv, fltv, th_sfc, &
       bl_mynn_edmf, bl_mynn_edmf_mom, bl_mynn_edmf_tke, &
       bl_mynn_mixscalars, bl_mynn_cloudmix, bl_mynn_mixqt)

  kts = 1
  kte = nz

  open(newunit=u_out, file=outfile, status='replace', action='write')

  ! mode 1: full EDMF (WRF default). mode 0: ED-only (mass-flux arrays zeroed).
  do mode = 1, 0, -1
     call run_once(mode, u_out)
  end do

  close(u_out)
  print *, 'oracle done, nz=', nz, ' wrote ', trim(outfile)

contains

  subroutine run_once(mode, u_out)
    integer, intent(in) :: mode, u_out
    real(kind_phys) :: dzk
    character(len=8) :: tag

    if (mode == 1) then
       tag = 'edmf'
    else
       tag = 'edonly'
    end if

    ! ---- derive specific humidities & thl/sqw exactly as WRF main mynn does
    ! (module_bl_mynnedmf.F:828-840 and mynnedmf_pre_run sqv=qv/(1+qv))
    zw(kts) = 0.0_kind_phys
    do k = kts, kte
       zw(k+1)   = zw(k) + dz(k)
       sqv(k)    = qv(k)/(1.0_kind_phys + qv(k))
       sqc(k)    = qc(k)/(1.0_kind_phys + qv(k))
       sqi(k)    = qi(k)/(1.0_kind_phys + qv(k))
       sqs(k)    = 0.0_kind_phys
       sqw(k)    = sqv(k) + sqc(k) + sqi(k)
       thl(k)    = th(k) - xlvcp/exner(k)*sqc(k)
       thv(k)    = th(k)*(1.0_kind_phys + p608*sqv(k))
       qnc(k)    = 0.0_kind_phys
       qni(k)    = 0.0_kind_phys
       qnwfa(k)  = 0.0_kind_phys
       qnifa(k)  = 0.0_kind_phys
       qnbca(k)  = 0.0_kind_phys
       ozone(k)  = 0.0_kind_phys
       tsq(k)    = 0.0_kind_phys
       qsq(k)    = 0.0_kind_phys
       cov(k)    = 0.0_kind_phys
       cldfra_bl1(k) = 0.0_kind_phys
       qc_bl1(k) = 0.0_kind_phys
       qi_bl1(k) = 0.0_kind_phys
       cldfra_bl1_old(k) = 0.0_kind_phys
       qc_bl1_old(k) = 0.0_kind_phys
       vt1(k) = 0.0_kind_phys
       vq1(k) = 0.0_kind_phys
       sgm1(k) = 0.0_kind_phys
       diss_heat(k) = 0.0_kind_phys
       el1(k) = 50.0_kind_phys
       tkeprod_up(k) = 0.0_kind_phys
       pattern_spp_pbl(k) = 0.0_kind_phys
    end do

    ! ---- compute dfm/dfh/dfq the simple WRF-consistent way.
    ! We use a representative eddy diffusivity profile derived from a neutral/
    ! convective mixing-length closure. To make the WRF-vs-JAX comparison sharp
    ! and isolate the MASS-FLUX contribution, both the Fortran oracle and the JAX
    ! port consume the SAME dfh/dfm/dfq (provided here), so any Dqv difference is
    ! purely the s_awqv/sub/det mass-flux terms -- not a turbulence-closure diff.
    ! Kh(k) ~ karman*ust*z*(1-z/pblh)^2 capped, on interfaces (k>=kts+1).
    dfm(kts) = 0.0_kind_phys
    dfh(kts) = 0.0_kind_phys
    dfq(kts) = 0.0_kind_phys
    do k = kts+1, kte
       dzk = 0.5_kind_phys*(dz(k)+dz(k-1))
       call kh_profile(zw(k), pblh, ust, dfh(k))
       dfh(k) = dfh(k)/dzk           ! dfh has units K/dz (== Kh/dz) per WRF
       dfm(k) = dfh(k)
       dfq(k) = dfh(k)
    end do

    ! ---- init solver & edmf arrays
    s_aw = 0.0_kind_phys; s_awthl = 0.0_kind_phys; s_awqt = 0.0_kind_phys
    s_awqv = 0.0_kind_phys; s_awqc = 0.0_kind_phys; s_awu = 0.0_kind_phys
    s_awv = 0.0_kind_phys; s_awqke = 0.0_kind_phys; s_awqnc = 0.0_kind_phys
    s_awqni = 0.0_kind_phys; s_awqnwfa = 0.0_kind_phys; s_awqnifa = 0.0_kind_phys
    s_awqnbca = 0.0_kind_phys
    sd_aw = 0.0_kind_phys; sd_awthl = 0.0_kind_phys; sd_awqt = 0.0_kind_phys
    sd_awqv = 0.0_kind_phys; sd_awqc = 0.0_kind_phys; sd_awqi = 0.0_kind_phys
    sd_awqnc = 0.0_kind_phys; sd_awqni = 0.0_kind_phys; sd_awqnwfa = 0.0_kind_phys
    sd_awqnifa = 0.0_kind_phys; sd_awu = 0.0_kind_phys; sd_awv = 0.0_kind_phys
    sub_thl = 0.0_kind_phys; sub_sqv = 0.0_kind_phys; sub_u = 0.0_kind_phys
    sub_v = 0.0_kind_phys; det_thl = 0.0_kind_phys; det_sqv = 0.0_kind_phys
    det_sqc = 0.0_kind_phys; det_u = 0.0_kind_phys; det_v = 0.0_kind_phys
    edmf_a = 0.0_kind_phys; edmf_w = 0.0_kind_phys; edmf_qt = 0.0_kind_phys
    edmf_thl = 0.0_kind_phys; edmf_ent = 0.0_kind_phys; edmf_qc = 0.0_kind_phys
    tcd = 0.0_kind_phys; qcd = 0.0_kind_phys
    s_awchem1 = 0.0_kind_phys; chem1 = 0.0_kind_phys
    ktop_plume = 0; maxmf = 0.0_kind_phys; ztop_plume = 0.0_kind_phys
    maxwidth = 0.0_kind_phys
    Psig_shcu = 1.0_kind_phys
    spp_pbl = 0
    mix_chem = .false.
    kpbl = 2

    FLAG_QC = .true.;  FLAG_QI = .true.;  FLAG_QS = .false.
    FLAG_QNC = .false.; FLAG_QNI = .false.
    FLAG_QNWFA = .false.; FLAG_QNIFA = .false.; FLAG_QNBCA = .false.
    FLAG_OZONE = .false.
    bl_mynn_edmf_mom = 0
    bl_mynn_edmf_tke = 0
    bl_mynn_mixscalars = 1
    bl_mynn_cloudmix = 1
    bl_mynn_mixqt = 0
    uoce = 0.0_kind_phys; voce = 0.0_kind_phys; flqc = 0.0_kind_phys

    if (mode == 1) then
       ! REAL WRF DMP_mf, with qt1=sqw, qv1=sqv, qc1=sqc (specific contents)
       call DMP_mf(1,1, kts,kte, delt, zw, dz, p, rho, &
            bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, &
            u,v,w,th,thl,thv,tk, sqw,sqv,sqc,qke, &
            qnc,qni,qnwfa,qnifa,qnbca, exner,vt1,vq1,sgm1, &
            ust,flt,fltv,flq,flqv, pblh,kpbl,dx, xland,ts, &
            edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc, &
            s_aw,s_awthl,s_awqt,s_awqv,s_awqc, &
            s_awu,s_awv,s_awqke, s_awqnc,s_awqni, &
            s_awqnwfa,s_awqnifa,s_awqnbca, &
            sub_thl,sub_sqv,sub_u,sub_v, &
            det_thl,det_sqv,det_sqc,det_u,det_v, &
            nchem,chem1,s_awchem1, mix_chem, &
            qc_bl1,cldfra_bl1, qc_bl1_old,cldfra_bl1_old, &
            FLAG_QC,FLAG_QI, FLAG_QNC,FLAG_QNI, &
            FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA, Psig_shcu, &
            maxwidth,ktop_plume,maxmf,ztop_plume, &
            spp_pbl,pattern_spp_pbl, tkeprod_up,el1)
       bl_mynn_edmf = 1
    else
       ! ED-only: leave all s_aw*/sub/det at zero (mimics current gpuwrf path)
       bl_mynn_edmf = 0
    end if

    call calc_kmdz_eff(kmdz_eff)

    ! REAL WRF mynn_tendencies
    call mynn_tendencies(kts,kte,1, delt,dz,zw,xland,rho, &
         u,v,th,tk,qv,qc,qi,qs,qnc,qni, psfc,p,exner, &
         thl,sqv,sqc,sqi,sqs,sqw, qnwfa,qnifa,qnbca,ozone, &
         ust,flt,flq,flqv,flqc,wspd, uoce,voce, tsq,qsq,cov, tcd,qcd, &
         dfm,dfh,dfq, &
         Du,Dv,Dth,Dqv,Dqc,Dqi,Dqs,Dqnc,Dqni, &
         Dqnwfa,Dqnifa,Dqnbca,Dozone, diss_heat, &
         s_aw,s_awthl, s_awqt,s_awqv,s_awqc, s_awu,s_awv, &
         s_awqnc,s_awqni, s_awqnwfa,s_awqnifa,s_awqnbca, &
         sd_aw,sd_awthl,sd_awqt,sd_awqv, sd_awqc,sd_awqi, &
         sd_awqnc,sd_awqni, sd_awqnwfa,sd_awqnifa, sd_awu,sd_awv, &
         sub_thl,sub_sqv, sub_u,sub_v, det_thl,det_sqv,det_sqc, &
         det_u,det_v, &
         FLAG_QC,FLAG_QI,FLAG_QNC,FLAG_QNI, FLAG_QS, &
         FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA, FLAG_OZONE, &
         cldfra_bl1, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf, &
         bl_mynn_edmf_mom, bl_mynn_mixscalars)

    ! ---- dump
    call wr_arr(u_out, trim(tag)//'_s_aw',     s_aw,    kte+1)
    call wr_arr(u_out, trim(tag)//'_s_awqv',   s_awqv,  kte+1)
    call wr_arr(u_out, trim(tag)//'_s_awqt',   s_awqt,  kte+1)
    call wr_arr(u_out, trim(tag)//'_s_awqc',   s_awqc,  kte+1)
    call wr_arr(u_out, trim(tag)//'_sub_sqv',  sub_sqv, kte)
    call wr_arr(u_out, trim(tag)//'_det_sqv',  det_sqv, kte)
    call wr_arr(u_out, trim(tag)//'_dfh',      dfh,     kte)
    call wr_arr(u_out, trim(tag)//'_dfm',      dfm,     kte)
    call wr_arr(u_out, trim(tag)//'_kmdz_eff', kmdz_eff, kte+1)
    call wr_arr(u_out, trim(tag)//'_edmf_a',   edmf_a,  kte)
    call wr_arr(u_out, trim(tag)//'_edmf_w',   edmf_w,  kte)
    call wr_arr(u_out, trim(tag)//'_Du',       Du,      kte)
    call wr_arr(u_out, trim(tag)//'_Dv',       Dv,      kte)
    call wr_arr(u_out, trim(tag)//'_Dqv',      Dqv,     kte)
    call wr_arr(u_out, trim(tag)//'_Dth',      Dth,     kte)
    call wr_arr(u_out, trim(tag)//'_sqv_post', sqv,     kte)  ! sqv updated by tendencies? no -- sqv is inout but tendencies returns Dqv; sqv unchanged for mixqt=0 path except thl. dump anyway
    write(u_out,'(A,I0)') trim(tag)//'_ktop_plume=', ktop_plume
    write(u_out,'(A,ES16.8)') trim(tag)//'_maxmf=', maxmf
    write(u_out,'(A,ES16.8)') trim(tag)//'_ztop_plume=', ztop_plume
  end subroutine run_once

  subroutine calc_kmdz_eff(out)
    real(kind_phys), intent(out) :: out(:)
    real(kind_phys), dimension(NZMAX+1) :: rhoz_local
    integer :: kk

    rhoz_local(kts) = rho(kts)
    out(kts) = rhoz_local(kts)*dfm(kts)
    do kk = kts+1, kte
       rhoz_local(kk) = (rho(kk)*dz(kk-1) + rho(kk-1)*dz(kk))/(dz(kk-1)+dz(kk))
       rhoz_local(kk) = max(rhoz_local(kk), 1.0E-4_kind_phys)
       out(kk) = rhoz_local(kk)*dfm(kk)
    end do
    rhoz_local(kte+1) = rhoz_local(kte)
    out(kte+1) = rhoz_local(kte+1)*dfm(kte)

    do kk = kts+1, kte-1
       out(kk) = max(out(kk), 0.5_kind_phys*(s_aw(kk)+sd_aw(kk)))
       out(kk) = max(out(kk), -0.5_kind_phys*(s_aw(kk)-s_aw(kk+1)) &
                            -0.5_kind_phys*(sd_aw(kk)-sd_aw(kk+1)))
    end do
  end subroutine calc_kmdz_eff

  subroutine kh_profile(z, pblh, ust, kh)
    real(kind_phys), intent(in) :: z, pblh, ust
    real(kind_phys), intent(out) :: kh
    real(kind_phys) :: zr
    real(kind_phys), parameter :: karman = 0.4_kind_phys
    if (z < pblh) then
       zr = z/pblh
       kh = karman*max(ust,0.1_kind_phys)*z*(1.0_kind_phys-zr)*(1.0_kind_phys-zr)
    else
       kh = 0.1_kind_phys
    end if
    kh = max(min(kh, 500.0_kind_phys), 0.01_kind_phys)
  end subroutine kh_profile

  subroutine wr_arr(u_out, name, arr, n)
    integer, intent(in) :: u_out, n
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: arr(:)
    integer :: kk
    character(len=16) :: buf
    write(u_out,'(A)',advance='no') name//'='
    do kk = 1, n
       write(buf,'(ES15.7)') arr(kk)
       if (kk < n) then
          write(u_out,'(A,A)',advance='no') trim(adjustl(buf)), ','
       else
          write(u_out,'(A)') trim(adjustl(buf))
       end if
    end do
  end subroutine wr_arr

  subroutine read_column(fname, nz, delt, dx, ust, pblh, psfc, wspd, xland, &
       ts, ps, u, v, w, th, tk, p, exner, rho, dz, qv, qc, qi, qke, &
       flt, flq, flqv, fltv, th_sfc, &
       e_edmf, e_mom, e_tke, e_mixs, e_cmix, e_mixqt)
    character(len=*), intent(in) :: fname
    integer, intent(out) :: nz, e_edmf, e_mom, e_tke, e_mixs, e_cmix, e_mixqt
    real(kind_phys), intent(out) :: delt, dx, ust, pblh, psfc, wspd, xland
    real(kind_phys), intent(out) :: ts, ps, flt, flq, flqv, fltv, th_sfc
    real(kind_phys), intent(out) :: u(:),v(:),w(:),th(:),tk(:),p(:),exner(:)
    real(kind_phys), intent(out) :: rho(:),dz(:),qv(:),qc(:),qi(:),qke(:)
    integer :: uin, ios, eqpos, kk
    character(len=20000) :: line
    character(len=64) :: key
    character(len=20000) :: val

    open(newunit=uin, file=fname, status='old', action='read')
    do
       read(uin,'(A)',iostat=ios) line
       if (ios /= 0) exit
       if (len_trim(line) == 0) cycle
       eqpos = index(line, '=')
       if (eqpos == 0) cycle
       key = adjustl(line(1:eqpos-1))
       val = adjustl(line(eqpos+1:))
       select case (trim(key))
       case ('nz');    read(val,*) nz
       case ('delt');  read(val,*) delt
       case ('dx');    read(val,*) dx
       case ('ust');   read(val,*) ust
       case ('pblh');  read(val,*) pblh
       case ('psfc');  read(val,*) psfc
       case ('wspd');  read(val,*) wspd
       case ('xland'); read(val,*) xland
       case ('ts');    read(val,*) ts
       case ('ps');    read(val,*) ps
       case ('flt');   read(val,*) flt
       case ('flq');   read(val,*) flq
       case ('flqv');  read(val,*) flqv
       case ('fltv');  read(val,*) fltv
       case ('th_sfc'); read(val,*) th_sfc
       case ('e_edmf'); read(val,*) e_edmf
       case ('e_mom');  read(val,*) e_mom
       case ('e_tke');  read(val,*) e_tke
       case ('e_mixs'); read(val,*) e_mixs
       case ('e_cmix'); read(val,*) e_cmix
       case ('e_mixqt'); read(val,*) e_mixqt
       case ('u');     read(val,*) (u(kk), kk=1,nz)
       case ('v');     read(val,*) (v(kk), kk=1,nz)
       case ('w');     read(val,*) (w(kk), kk=1,nz)
       case ('th');    read(val,*) (th(kk), kk=1,nz)
       case ('tk');    read(val,*) (tk(kk), kk=1,nz)
       case ('p');     read(val,*) (p(kk), kk=1,nz)
       case ('exner'); read(val,*) (exner(kk), kk=1,nz)
       case ('rho');   read(val,*) (rho(kk), kk=1,nz)
       case ('dz');    read(val,*) (dz(kk), kk=1,nz)
       case ('qv');    read(val,*) (qv(kk), kk=1,nz)
       case ('qc');    read(val,*) (qc(kk), kk=1,nz)
       case ('qi');    read(val,*) (qi(kk), kk=1,nz)
       case ('qke');   read(val,*) (qke(kk), kk=1,nz)
       end select
    end do
    close(uin)
  end subroutine read_column

end program mynn_edmf_oracle
