! ============================================================================
!  WRF Noah-MP offline savepoint driver  (v0.2.0 P0-3 oracle harness, Sprint 0b)
! ----------------------------------------------------------------------------
!  Faithful EXTERNAL ORACLE (NOT a self-compare): links the COMPILED pristine
!  WRF objects (module_sf_noahmplsm.o / module_sf_noahmpdrv.o), reads the real
!  WRF run/ tables via NOAHMP_TABLES, builds the per-column `parameters` with
!  TRANSFER_MP_PARAMETERS, and calls the exact NOAHMP_SFLX orchestrator on real
!  Canary land columns. The call brackets every component (phenology, energy,
!  soil-thermo, snow-water, soil-water), so its IN snapshot + OUT snapshot is a
!  per-component input->output savepoint for S1/S2/S3/S4/S5.
!
!  Active-option scope (ADR-NOAHMP-INTERFACES.md): dveg=4, opt_crs=1, opt_btr=1,
!  opt_run=3 (Schaake), opt_sfc=1, opt_frz=1, opt_inf=1, opt_rad=3, opt_alb=2,
!  opt_snf=1, opt_tbot=2, opt_stc=1. ICE=0, IST=1, CROPTYPE=0, IRRFRA=0 (land).
!
!  I/O contract (ASCII, paired with build_noahmp_savepoints.py):
!    - reads  : noahmp_columns.in   (one record per column; see READ block)
!    - writes : noahmp_savepoints.out  (key value ... lines, per column)
!  The Python wrapper provides the column records (extracted from the corpus
!  wrfinput/wrfout) and parses the output into the canonical per-component JSON.
! ============================================================================
program noahmp_offline_driver
  use module_sf_noahmplsm, only : noahmp_parameters, noahmp_sflx, noahmp_options
  use noahmp_tables, only : read_mp_veg_parameters, read_mp_soil_parameters, &
       read_mp_rad_parameters, read_mp_global_parameters
  use module_sf_noahmpdrv, only : transfer_mp_parameters
  implicit none

  integer, parameter :: NSOIL = 4, NSNOW = 3
  integer, parameter :: IOUT = 31, IIN = 32

  ! --- per-run configuration read from header ---
  character(len=64)  :: dataset
  integer            :: ncol, slopetyp, soilcolor

  ! --- column scalars (read) ---
  integer :: vegtyp, isltyp
  real    :: lat, julian, cosz, dt, dx, dz8w, zlvl
  real    :: shdfac, shdmax, tbot
  real    :: sfctmp, sfcprs, psfc, uu, vv, q2, qc, soldn, lwdn
  real    :: prcpconv, prcpnonc, prcpsnow, prcpgrpl, prcphail
  integer :: yearlen, ic

  ! --- prognostic land state (in/out) ---
  real    :: albold, sneqvo, eah, tah, fwet, canliq, canice, tv, tg, qsfc
  real    :: qsnow, qrain, snowh, sneqv, zwt, wa, wt, wslake, smcwtd
  real    :: deeprech, rech, cm, ch, tauss, grain, gdd
  integer :: isnow, pgs
  real    :: stc(-NSNOW+1:NSOIL), sh2o(NSOIL), smc(NSOIL)
  real    :: zsnso(-NSNOW+1:NSOIL), snice(-NSNOW+1:0), snliq(-NSNOW+1:0)
  real    :: zsoil(NSOIL), smceq(NSOIL), ficeold(-NSNOW+1:0)
  real    :: lfmass, rtmass, stmass, wood, stblcp, fastcp, lai, sai
  integer :: soiltype(NSOIL)

  ! --- carbon / unused-but-required ---
  real    :: co2air, o2air, foln
  real    :: gecros1d(60), qtldrn, tdfracmp
  real    :: irrfra, sifra, mifra, fifra
  integer :: ircntsi, ircntmi, ircntfi
  real    :: iramtsi, iramtmi, iramtfi, irsirate, irmirate, irfirate, firr, eirr
  character(len=256) :: llanduse

  ! --- outputs ---
  real :: z0wrf, fsa, fsr, fira, fsh, ssoil, fcev, fgev, fctr
  real :: ecan, etran, edir, trad, tgb, tgv, t2mv, t2mb, q2v, q2b
  real :: runsrf, runsub, apar, psn, sav, sag, fsno, nee, gpp, npp, fveg, albedo
  real :: qsnbot, ponding, ponding1, ponding2, rssun, rssha
  real :: albsnd(2), albsni(2), bgap, wgap, chv, chb, emissi
  real :: shg, shc, shb, evg, evb, ghv, ghb, irg, irc, irb, tr, evc
  real :: chleaf, chuc, chv2, chb2, fpice, pahv, pahg, pahb, pah
  real :: laisun, laisha, rb, qints, qintr, qdrips, qdripr, qthros, qthror
  real :: qsnsub, qsnfro, qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc, qmelt
  real :: rain, snow_o, acc_ssoil, acc_qinsur, acc_qseva
  real :: acc_etrani(NSOIL), hcpct(-NSNOW+1:NSOIL), eflxb, canhs
  real :: acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir

  ! save IN snapshots for the savepoint (the SFLX call mutates the prognostics)
  real :: tv_in, tg_in, tah_in, eah_in, lai_in, sai_in, sneqv_in, snowh_in
  real :: stc_in(-NSNOW+1:NSOIL), sh2o_in(NSOIL), smc_in(NSOIL)
  integer :: isnow_in
  real :: canliq_in, canice_in, fwet_in, qsfc_in, smcwtd_in, albold_in

  type(noahmp_parameters) :: parameters
  integer :: k

  ! --------------------------------------------------------------------------
  !  Read configuration + tables
  ! --------------------------------------------------------------------------
  open(IIN, file='noahmp_columns.in', status='old', form='formatted', action='read')
  read(IIN,*) dataset
  read(IIN,*) ncol, slopetyp, soilcolor, dt
  read(IIN,*) (zsoil(k), k=1,NSOIL)

  call read_mp_veg_parameters(trim(dataset))
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()

  ! FROZEN active-option set (ADR-NOAHMP-INTERFACES.md §1). Order matches
  ! noahmp_options(idveg, opt_crs, opt_btr, opt_run, opt_sfc, opt_frz, opt_inf,
  ! opt_rad, opt_alb, opt_snf, opt_tbot, opt_stc, opt_rsf, opt_soil, opt_pedo,
  ! opt_crop, opt_irr, opt_irrm, opt_infdv, opt_tdrn).
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, &
                      1, 1, 1, 0, 0, 0, 1, 0)

  open(IOUT, file='noahmp_savepoints.out', status='replace', form='formatted', action='write')
  write(IOUT,'(A,I0)') 'NCOL ', ncol

  llanduse = trim(dataset)

  ! --------------------------------------------------------------------------
  !  Per-column loop
  ! --------------------------------------------------------------------------
  do ic = 1, ncol
    ! ---- read one column record (free-format, key order fixed) ----
    read(IIN,*) vegtyp, isltyp
    read(IIN,*) lat, julian, yearlen, cosz, dx, dz8w, zlvl
    read(IIN,*) shdfac, shdmax, tbot
    read(IIN,*) sfctmp, sfcprs, psfc, uu, vv, q2, qc, soldn, lwdn
    read(IIN,*) prcpconv, prcpnonc, prcpsnow, prcpgrpl, prcphail
    read(IIN,*) (stc(k), k=-NSNOW+1,NSOIL)
    read(IIN,*) (smc(k), k=1,NSOIL)
    read(IIN,*) (sh2o(k), k=1,NSOIL)
    read(IIN,*) tv, tg, tah, eah, canliq, canice, fwet, qsfc
    read(IIN,*) lai, sai, snowh, sneqv, sneqvo, albold, tauss
    read(IIN,*) isnow
    read(IIN,*) (zsnso(k), k=-NSNOW+1,NSOIL)
    read(IIN,*) (snice(k), k=-NSNOW+1,0)
    read(IIN,*) (snliq(k), k=-NSNOW+1,0)
    read(IIN,*) cm, ch, smcwtd

    ! ---- fixed land config + benign defaults for cut paths ----
    soiltype = isltyp
    do k = 1, NSOIL
      smceq(k) = smc(k)         ! opt_run=3 does not evolve SMCEQ; seed = SMC
    end do
    ficeold = 0.0
    foln = 1.0
    co2air = 395.0e-6 * sfcprs   ! Pa partial pressure (parameters%CO2 * SFCPRS)
    o2air  = 0.209   * sfcprs
    qsnow = 0.0; qrain = 0.0; zwt = -2.0; wa = 4900.0; wt = 4900.0
    wslake = 0.0; deeprech = 0.0; rech = 0.0; grain = 0.0; gdd = 0.0; pgs = 0
    lfmass = max(lai,0.05)/0.035; rtmass = 500.0; stmass = max(sai,0.05)/0.003
    wood = 500.0; stblcp = 1000.0; fastcp = 1000.0
    gecros1d = 0.0; qtldrn = 0.0; tdfracmp = 0.0
    irrfra = 0.0; sifra = 0.0; mifra = 0.0; fifra = 0.0
    ircntsi = 0; ircntmi = 0; ircntfi = 0
    iramtsi = 0.0; iramtmi = 0.0; iramtfi = 0.0
    acc_ssoil = 0.0; acc_qinsur = 0.0; acc_qseva = 0.0; acc_etrani = 0.0
    acc_dwater = 0.0; acc_prcp = 0.0; acc_ecan = 0.0; acc_etran = 0.0; acc_edir = 0.0

    call transfer_mp_parameters(4, vegtyp, soiltype, slopetyp, soilcolor, 0, parameters)

    ! ---- snapshot the IN state ----
    tv_in=tv; tg_in=tg; tah_in=tah; eah_in=eah; lai_in=lai; sai_in=sai
    sneqv_in=sneqv; snowh_in=snowh; isnow_in=isnow
    canliq_in=canliq; canice_in=canice; fwet_in=fwet; qsfc_in=qsfc
    smcwtd_in=smcwtd; albold_in=albold
    stc_in=stc; sh2o_in=sh2o; smc_in=smc

    ! ---- THE WRF NOAH-MP CALL (orchestrates all components) ----
    call noahmp_sflx(parameters, ic, 1, lat, yearlen, julian, cosz, &
         dt, dx, dz8w, NSOIL, zsoil, NSNOW, &
         shdfac, shdmax, vegtyp, 0, 1, 0, &
         smceq, &
         sfctmp, sfcprs, psfc, uu, vv, q2, &
         qc, soldn, lwdn, &
         prcpconv, prcpnonc, 0.0, prcpsnow, prcpgrpl, prcphail, &
         tbot, co2air, o2air, foln, ficeold, zlvl, &
         irrfra, sifra, mifra, fifra, llanduse, &
         albold, sneqvo, &
         stc, sh2o, smc, tah, eah, fwet, &
         canliq, canice, tv, tg, qsfc, qsnow, &
         qrain, &
         isnow, zsnso, snowh, sneqv, snice, snliq, &
         zwt, wa, wt, wslake, lfmass, rtmass, &
         stmass, wood, stblcp, fastcp, lai, sai, &
         cm, ch, tauss, &
         grain, gdd, pgs, &
         smcwtd, deeprech, rech, &
         gecros1d, &
         qtldrn, tdfracmp, &
         z0wrf, &
         ircntsi, ircntmi, ircntfi, iramtsi, iramtmi, iramtfi, &
         irsirate, irmirate, irfirate, firr, eirr, &
         fsa, fsr, fira, fsh, ssoil, fcev, &
         fgev, fctr, ecan, etran, edir, trad, &
         tgb, tgv, t2mv, t2mb, q2v, q2b, &
         runsrf, runsub, apar, psn, sav, sag, &
         fsno, nee, gpp, npp, fveg, albedo, &
         qsnbot, ponding, ponding1, ponding2, rssun, rssha, &
         albsnd, albsni, &
         bgap, wgap, chv, chb, emissi, &
         shg, shc, shb, evg, evb, ghv, &
         ghb, irg, irc, irb, tr, evc, &
         chleaf, chuc, chv2, chb2, fpice, pahv, &
         pahg, pahb, pah, laisun, laisha, rb, &
         qints, qintr, qdrips, qdripr, qthros, qthror, &
         qsnsub, qsnfro, qsubc, qfroc, qfrzc, qmeltc, &
         qevac, qdewc, qmelt, &
         rain, snow_o, acc_ssoil, acc_qinsur, acc_qseva, &
         acc_etrani, hcpct, eflxb, canhs, &
         acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir)

    ! ---- dump the savepoint (IN then OUT, per component) ----
    write(IOUT,'(A,I0)') 'COL ', ic
    write(IOUT,'(A,2I6)')   'CAT vegtyp isltyp ', vegtyp, isltyp
    ! forcing echo
    write(IOUT,'(A,9ES16.8)') 'FORCING sfctmp sfcprs psfc uu vv q2 qc soldn lwdn ', &
         sfctmp, sfcprs, psfc, uu, vv, q2, qc, soldn, lwdn
    write(IOUT,'(A,5ES16.8)') 'FORCING2 cosz julian shdfac shdmax tbot ', &
         cosz, julian, shdfac, shdmax, tbot
    ! --- PHENOLOGY (S5) ---
    write(IOUT,'(A,6ES16.8)') 'PHEN_IN lai sai ', lai_in, sai_in
    write(IOUT,'(A,3ES16.8)') 'PHEN_OUT lai sai fveg ', lai, sai, fveg
    ! --- ENERGY (S1) — the HFX-fix component ---
    write(IOUT,'(A,4ES16.8)') 'ENERGY_IN tv tg tah eah ', tv_in, tg_in, tah_in, eah_in
    write(IOUT,'(A,13ES16.8)') 'ENERGY_OUT fsh fcev fgev fctr ssoil fira trad emissi z0wrf chv chb sav sag ', &
         fsh, fcev, fgev, fctr, ssoil, fira, trad, emissi, z0wrf, chv, chb, sav, sag
    write(IOUT,'(A,7ES16.8)') 'ENERGY_STATE tv tg tah eah albedo fsno fsa ', &
         tv, tg, tah, eah, albedo, fsno, fsa
    write(IOUT,'(A,5ES16.8)') 'ET ecan etran edir qsnow qmelt ', ecan, etran, edir, qsnow, qmelt
    ! --- SOIL THERMO (S2) — STC over snow+soil column ---
    write(IOUT,'(A,7ES16.8)') 'STC_IN ', (stc_in(k), k=-NSNOW+1,NSOIL)
    write(IOUT,'(A,7ES16.8)') 'STC_OUT ', (stc(k), k=-NSNOW+1,NSOIL)
    write(IOUT,'(A,7ES16.8)') 'HCPCT ', (hcpct(k), k=-NSNOW+1,NSOIL)
    write(IOUT,'(A,1ES16.8)') 'TBOTOUT eflxb ', eflxb
    ! --- SNOW WATER (S3) ---
    write(IOUT,'(A,3I6)')      'SNOW_ISNOW in out ', isnow_in, isnow
    write(IOUT,'(A,4ES16.8)')  'SNOW_IN snowh sneqv sneqvo albold ', snowh_in, sneqv_in, sneqvo, albold_in
    write(IOUT,'(A,5ES16.8)')  'SNOW_OUT snowh sneqv qsnbot fsno albold ', snowh, sneqv, qsnbot, fsno, albold
    write(IOUT,'(A,7ES16.8)')  'ZSNSO_OUT ', (zsnso(k), k=-NSNOW+1,NSOIL)
    write(IOUT,'(A,3ES16.8)')  'SNICE_OUT ', (snice(k), k=-NSNOW+1,0)
    write(IOUT,'(A,3ES16.8)')  'SNLIQ_OUT ', (snliq(k), k=-NSNOW+1,0)
    ! --- WATER / SCHAAKE (S4) ---
    write(IOUT,'(A,4ES16.8)')  'SMC_IN ', (smc_in(k), k=1,NSOIL)
    write(IOUT,'(A,4ES16.8)')  'SMC_OUT ', (smc(k), k=1,NSOIL)
    write(IOUT,'(A,4ES16.8)')  'SH2O_IN ', (sh2o_in(k), k=1,NSOIL)
    write(IOUT,'(A,4ES16.8)')  'SH2O_OUT ', (sh2o(k), k=1,NSOIL)
    write(IOUT,'(A,5ES16.8)')  'WATER_OUT runsrf runsub smcwtd canliq canice ', &
         runsrf, runsub, smcwtd, canliq, canice
    ! --- DRIVER FLUX MAPPING (the coupler outputs) ---
    write(IOUT,'(A,5ES16.8)')  'DRIVER hfx lh qfx grdflx tsk ', &
         fsh, (fcev+fgev+fctr), (ecan+edir+etran), ssoil, trad
    write(IOUT,'(A)') 'ENDCOL'
  end do

  close(IIN)
  close(IOUT)
  write(*,'(A,I0,A)') 'NOAHMP_OFFLINE_OK ', ncol, ' columns'
end program noahmp_offline_driver
