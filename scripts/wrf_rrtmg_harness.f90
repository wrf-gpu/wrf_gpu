program wrf_rrtmg_harness
  ! Output format:
  !   Existing formatted records are preserved:
  !     line 1: nz
  !     line 2: SW broadband scalars
  !     line 3: LW broadband scalars
  !     next nz lines: SW heating, LW heating, pressure layer mass
  !     next nz+2 lines: SW down/up, LW down/up interface fluxes
  !   M5-S3.z appends a marker line "#RRTMG_ORACLE_V1_BINARY", followed by
  !   a stream-unformatted binary payload.  The Python parser treats this as
  !   native WRF intermediate oracle record v1:
  !     int32[6] = nlay_sw, nlay_lw, max_sw_g, max_lw_g, nbndsw, nbndlw
  !     real[4,14] SW per-band fluxes in order toa_up, toa_down, sfc_up, sfc_down
  !     real[4,16] LW per-band fluxes in the same order
  !     int SW jp/jt/jt1/indself/indfor, each length nlay_sw
  !     real SW fac00/fac01/fac10/fac11/selffac/forfac, each length nlay_sw
  !     real SW colmol[nlay_sw,6] = H2O, CO2, O3, CH4, N2O, O2 columns
  !     real SW taug[nlay_sw,max_sw_g,14], taur[nlay_sw,max_sw_g,14],
  !       sfluxzen[max_sw_g,14]
  !     int LW jp/jt, each length nlay_lw
  !     real LW planklay[nlay_lw,16], planklev[0:nlay_lw,16],
  !       plankbnd[16], taug[nlay_lw,max_lw_g,16],
  !       fracs[nlay_lw,max_lw_g,16], secdiff[16],
  !       dplankup[nlay_lw,16], dplankdn[nlay_lw,16]
  !     M5-S3.zzzz appends SW cloudy boundary/stage records:
  !       cldprmc pcldfmc/ptaucmc/pasycmc/pomgcmc/ptaormc[nlay_sw,max_sw_g,14]
  !       spcvmc zref/ztra/zrefd/ztrad clear/cloud/blended[nlay_sw+1,max_sw_g,14]
  !       spcvmc direct_trans[nlay_sw,max_sw_g,14],
  !       raw zfd/zfu[nlay_sw+1,max_sw_g,14], and weighted flux zfd/zfu[nlay_sw+1,max_sw_g,14]
  use module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad
  use module_ra_rrtmg_lw, only: rrtmg_lwinit, rrtmg_lwrad
  use parrrsw, only: sw_nbnd => nbndsw, sw_ngpt => ngptsw, sw_jpband => jpband, &
                     sw_jpb1 => jpb1, sw_mxmol => mxmol
  use parrrtm, only: lw_nbnd => nbndlw, lw_ngpt => ngptlw, lw_mxmol => mxmol, &
                     lw_maxxsec => maxxsec
  use rrtmg_sw_setcoef, only: setcoef_sw
  use rrtmg_sw_taumol, only: taumol_sw
  use rrtmg_sw_spcvmc, only: spcvmc_sw
  use mcica_subcol_gen_sw, only: mcica_subcol_sw
  use rrtmg_sw_cldprmc, only: cldprmc_sw
  use rrtmg_sw_reftra, only: reftra_sw
  use rrtmg_sw_vrtqdr, only: vrtqdr_sw
  use rrsw_wvn, only: sw_ngc => ngc
  use rrsw_tbl, only: tblint, bpade, od_lo, exp_tbl
  use rrtmg_lw_setcoef, only: setcoef
  use rrtmg_lw_taumol, only: taumol
  use rrtmg_lw_rtrnmc, only: rtrnmc
  implicit none

  integer, parameter :: rk = kind(1.0)
  integer, parameter :: sw_max_g = 12
  integer, parameter :: lw_max_g = 16
  real(rk), parameter :: cp_air = 1004.0_rk
  real(rk), parameter :: gravity = 9.80665_rk
  real(rk), parameter :: rd_air = 287.04_rk
  real(rk), parameter :: rv_over_rd_minus_one = 0.608_rk
  real(rk), parameter :: stefan_boltzmann = 5.670374419e-8_rk
  real(rk), parameter :: lw_init_ptop_pa = 400.0_rk
  real(rk), parameter :: avogadro = 6.02214199e23_rk
  real(rk), parameter :: dry_air_mw = 28.9660_rk
  real(rk), parameter :: water_mw = 18.0160_rk
  real(rk), parameter :: h2o_mmr_to_vmr = 1.607793_rk
  real(rk), parameter :: o3_mmr_to_vmr = 0.603461_rk
  real(rk), parameter :: co2_vmr_default = (280.0_rk + 90.0_rk * exp(0.02_rk * (2026.0_rk - 2000.0_rk))) * 1.0e-6_rk
  real(rk), parameter :: ch4_vmr_default = 1774.0e-9_rk
  real(rk), parameter :: n2o_vmr_default = 319.0e-9_rk
  real(rk), parameter :: o2_vmr_default = 0.209488_rk
  real(rk), parameter :: o3_vmr_default = 8.0e-8_rk
  real(rk), parameter :: cfc11_vmr_default = 0.251e-9_rk
  real(rk), parameter :: cfc12_vmr_default = 0.538e-9_rk
  real(rk), parameter :: cfc22_vmr_default = 0.169e-9_rk
  real(rk), parameter :: ccl4_vmr_default = 0.093e-9_rk

  integer :: nz, k, argc
  character(len=512) :: input_path, output_path
  real(rk) :: surface_albedo, coszen, surface_temperature, surface_emissivity
  real(rk), allocatable :: temp(:), press(:), qv(:), qc(:), qi(:), qs(:), qg(:), cldfra(:), dz(:), rho(:)
  real(rk), allocatable :: sw_heat(:), lw_heat(:), layer_mass_p(:)
  real(rk), allocatable :: sw_down(:), sw_up(:), lw_down(:), lw_up(:)
  real(rk) :: sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit
  integer, allocatable :: sw_jp(:), sw_jt(:), sw_jt1(:), sw_indself(:), sw_indfor(:)
  integer, allocatable :: lw_jp(:), lw_jt(:)
  real(rk), allocatable :: sw_fac00(:), sw_fac01(:), sw_fac10(:), sw_fac11(:), sw_selffac(:), sw_forfac(:)
  real(rk), allocatable :: sw_colmol(:,:), sw_taug(:,:,:), sw_taur(:,:,:), sw_sfluxzen(:,:)
  real(rk), allocatable :: sw_cld_pcldfmc(:,:,:), sw_cld_ptaucmc(:,:,:), sw_cld_pasycmc(:,:,:), sw_cld_pomgcmc(:,:,:), sw_cld_ptaormc(:,:,:)
  real(rk), allocatable :: sw_spc_zref(:,:,:), sw_spc_ztra(:,:,:), sw_spc_zrefd(:,:,:), sw_spc_ztrad(:,:,:)
  real(rk), allocatable :: sw_spc_zref_clear(:,:,:), sw_spc_ztra_clear(:,:,:), sw_spc_zrefd_clear(:,:,:), sw_spc_ztrad_clear(:,:,:)
  real(rk), allocatable :: sw_spc_zref_cloud(:,:,:), sw_spc_ztra_cloud(:,:,:), sw_spc_zrefd_cloud(:,:,:), sw_spc_ztrad_cloud(:,:,:)
  real(rk), allocatable :: sw_spc_direct_trans(:,:,:), sw_spc_zfd(:,:,:), sw_spc_zfu(:,:,:), sw_spc_zfd_flux(:,:,:), sw_spc_zfu_flux(:,:,:)
  real(rk), allocatable :: lw_planklay(:,:), lw_planklev(:,:), lw_plankbnd(:)
  real(rk), allocatable :: lw_taug(:,:,:), lw_fracs(:,:,:), lw_secdiff(:), lw_dplankup(:,:), lw_dplankdn(:,:)
  real(rk), allocatable :: sw_band_flux(:,:), lw_band_flux(:,:)

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_rrtmg_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) nz
  read(10,*) surface_albedo, coszen, surface_temperature, surface_emissivity
  allocate(temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz))
  allocate(sw_heat(nz), lw_heat(nz), layer_mass_p(nz))
  allocate(sw_down(0:nz+1), sw_up(0:nz+1), lw_down(0:nz+1), lw_up(0:nz+1))
  allocate(sw_jp(nz+1), sw_jt(nz+1), sw_jt1(nz+1), sw_indself(nz+1), sw_indfor(nz+1))
  allocate(lw_jp(nz+1), lw_jt(nz+1))
  allocate(sw_fac00(nz+1), sw_fac01(nz+1), sw_fac10(nz+1), sw_fac11(nz+1), sw_selffac(nz+1), sw_forfac(nz+1))
  allocate(sw_colmol(nz+1,6), sw_taug(nz+1,sw_max_g,sw_nbnd), sw_taur(nz+1,sw_max_g,sw_nbnd), sw_sfluxzen(sw_max_g,sw_nbnd))
  allocate(sw_cld_pcldfmc(nz+1,sw_max_g,sw_nbnd), sw_cld_ptaucmc(nz+1,sw_max_g,sw_nbnd), sw_cld_pasycmc(nz+1,sw_max_g,sw_nbnd), sw_cld_pomgcmc(nz+1,sw_max_g,sw_nbnd), sw_cld_ptaormc(nz+1,sw_max_g,sw_nbnd))
  allocate(sw_spc_zref(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad(nz+2,sw_max_g,sw_nbnd))
  allocate(sw_spc_zref_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad_clear(nz+2,sw_max_g,sw_nbnd))
  allocate(sw_spc_zref_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad_cloud(nz+2,sw_max_g,sw_nbnd))
  allocate(sw_spc_direct_trans(nz+1,sw_max_g,sw_nbnd), sw_spc_zfd(nz+2,sw_max_g,sw_nbnd), sw_spc_zfu(nz+2,sw_max_g,sw_nbnd), sw_spc_zfd_flux(nz+2,sw_max_g,sw_nbnd), sw_spc_zfu_flux(nz+2,sw_max_g,sw_nbnd))
  allocate(lw_planklay(nz+1,lw_nbnd), lw_planklev(0:nz+1,lw_nbnd), lw_plankbnd(lw_nbnd))
  allocate(lw_taug(nz+1,lw_max_g,lw_nbnd), lw_fracs(nz+1,lw_max_g,lw_nbnd), lw_secdiff(lw_nbnd))
  allocate(lw_dplankup(nz+1,lw_nbnd), lw_dplankdn(nz+1,lw_nbnd))
  allocate(sw_band_flux(4,sw_nbnd), lw_band_flux(4,lw_nbnd))
  do k = 1, nz
     read(10,*) temp(k), press(k), qv(k), qc(k), qi(k), qs(k), qg(k), cldfra(k), dz(k), rho(k)
  end do
  close(10)

  call rrtmg_swinit(.true., 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz)
  call rrtmg_lwinit(lw_init_ptop_pa, .true., 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz)

  call run_wrf_rrtmg_drivers(nz, surface_albedo, coszen, surface_temperature, surface_emissivity, &
                             temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                             sw_heat, lw_heat, sw_down, sw_up, lw_down, lw_up, &
                             sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p)
  call compute_intermediate_oracle(nz, surface_albedo, coszen, surface_temperature, surface_emissivity, &
                                   temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                   sw_band_flux, lw_band_flux, sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                   sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                   sw_taug, sw_taur, sw_sfluxzen, sw_cld_pcldfmc, sw_cld_ptaucmc, sw_cld_pasycmc, &
                                   sw_cld_pomgcmc, sw_cld_ptaormc, sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad, &
                                   sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear, &
                                   sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud, &
                                   sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux, &
                                   lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                   lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I6)') nz
  write(20,'(6(ES24.16E3,1X))') sw_down(nz+1), sw_up(nz+1), sw_down(0), sw_up(0), sw_col_abs, sw_sfc_abs
  write(20,'(6(ES24.16E3,1X))') lw_down(nz+1), lw_up(nz+1), lw_down(0), lw_up(0), lw_col_heat, lw_sfc_emit
  do k = 1, nz
     write(20,'(3(ES24.16E3,1X))') sw_heat(k), lw_heat(k), layer_mass_p(k)
  end do
  do k = 0, nz+1
     write(20,'(4(ES24.16E3,1X))') sw_down(k), sw_up(k), lw_down(k), lw_up(k)
  end do
  write(20,'(A)') '#RRTMG_ORACLE_V1_BINARY'
  close(20)
  call append_intermediate_oracle(output_path, nz+1, nz+1, sw_band_flux, lw_band_flux, &
                                  sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                  sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                  sw_taug, sw_taur, sw_sfluxzen, sw_cld_pcldfmc, sw_cld_ptaucmc, sw_cld_pasycmc, &
                                  sw_cld_pomgcmc, sw_cld_ptaormc, sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad, &
                                  sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear, &
                                  sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud, &
                                  sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux, &
                                  lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                  lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn)

contains

  subroutine compute_intermediate_oracle(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                         temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                         sw_band_flux, lw_band_flux, sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                         sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                         sw_taug_band, sw_taur_band, sw_sflux_band, sw_cld_pcldfmc, sw_cld_ptaucmc, sw_cld_pasycmc, &
                                         sw_cld_pomgcmc, sw_cld_ptaormc, sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad, &
                                         sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear, &
                                         sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud, &
                                         sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux, &
                                         lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                         lw_plankbnd, lw_taug_band, lw_fracs_band, lw_secdiff, lw_dplankup, lw_dplankdn)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(out) :: sw_band_flux(4,sw_nbnd), lw_band_flux(4,lw_nbnd)
    integer, intent(out) :: sw_jp(nz+1), sw_jt(nz+1), sw_jt1(nz+1), sw_indself(nz+1), sw_indfor(nz+1)
    integer, intent(out) :: lw_jp(nz+1), lw_jt(nz+1)
    real(rk), intent(out) :: sw_fac00(nz+1), sw_fac01(nz+1), sw_fac10(nz+1), sw_fac11(nz+1), sw_selffac(nz+1), sw_forfac(nz+1)
    real(rk), intent(out) :: sw_colmol(nz+1,6), sw_taug_band(nz+1,sw_max_g,sw_nbnd), sw_taur_band(nz+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_sflux_band(sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_cld_pcldfmc(nz+1,sw_max_g,sw_nbnd), sw_cld_ptaucmc(nz+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_cld_pasycmc(nz+1,sw_max_g,sw_nbnd), sw_cld_pomgcmc(nz+1,sw_max_g,sw_nbnd), sw_cld_ptaormc(nz+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_spc_zref(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad(nz+2,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_spc_zref_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd_clear(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad_clear(nz+2,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_spc_zref_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_ztra_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_zrefd_cloud(nz+2,sw_max_g,sw_nbnd), sw_spc_ztrad_cloud(nz+2,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_spc_direct_trans(nz+1,sw_max_g,sw_nbnd), sw_spc_zfd(nz+2,sw_max_g,sw_nbnd), sw_spc_zfu(nz+2,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_spc_zfd_flux(nz+2,sw_max_g,sw_nbnd), sw_spc_zfu_flux(nz+2,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: lw_planklay(nz+1,lw_nbnd), lw_planklev(0:nz+1,lw_nbnd), lw_plankbnd(lw_nbnd)
    real(rk), intent(out) :: lw_taug_band(nz+1,lw_max_g,lw_nbnd), lw_fracs_band(nz+1,lw_max_g,lw_nbnd)
    real(rk), intent(out) :: lw_secdiff(lw_nbnd), lw_dplankup(nz+1,lw_nbnd), lw_dplankdn(nz+1,lw_nbnd)

    integer, parameter :: sw_counts(sw_nbnd) = (/6,12,8,8,10,10,2,10,8,6,6,8,6,12/)
    integer, parameter :: lw_counts(lw_nbnd) = (/10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2/)
    integer :: nlay, k, b, g, ig, start_g, laytrop, layswtch, laylow, ncbands
    integer :: irng_sw, icld_sw, inflag_sw, iceflag_sw, liqflag_sw, permuteseed_sw, idcor_sw, juldat_sw
    real(rk) :: pavel(nz+1), tavel(nz+1), pz(0:nz+1), tz(0:nz+1)
    real(rk) :: coldry(nz+1), wkl(sw_mxmol,nz+1), wkl_lw(lw_mxmol,nz+1), wx(lw_maxxsec,nz+1)
    real(rk) :: wbroad(nz+1), h2ovmr(nz+1), o3vmr(nz+1), amm, summol, amttl, wvttl, wvsh, pwvcm
    real(rk) :: co2mult(nz+1), colh2o(nz+1), colco2(nz+1), colo3(nz+1), coln2o(nz+1), colch4(nz+1), colo2(nz+1), colmol(nz+1)
    real(rk) :: selffrac(nz+1), forfrac(nz+1)
    real(rk) :: taug_sw(nz+1,sw_ngpt), taur_sw(nz+1,sw_ngpt), sflux_sw(sw_ngpt)
    real(rk) :: albdif(sw_nbnd), albdir(sw_nbnd), adjflux(sw_jpband)
    real(rk) :: cldf_sw(nz+1,sw_ngpt), taucmc_sw(nz+1,sw_ngpt), taormc_sw(nz+1,sw_ngpt), asy_sw(nz+1,sw_ngpt), omg_sw(nz+1,sw_ngpt)
    real(rk) :: cldfmc_mc(sw_ngpt,nz+1), ciwpmc_mc(sw_ngpt,nz+1), clwpmc_mc(sw_ngpt,nz+1), cswpmc_mc(sw_ngpt,nz+1)
    real(rk) :: taormc_mc(sw_ngpt,nz+1), taucmc_mc(sw_ngpt,nz+1), ssacmc_mc(sw_ngpt,nz+1), asmcmc_mc(sw_ngpt,nz+1), fsfcmc_mc(sw_ngpt,nz+1)
    real(rk) :: reicmc_mc(nz+1), relqmc_mc(nz+1), resnmc_mc(nz+1)
    real(rk) :: play_mc(1,nz+1), cldfrac_mc(1,nz+1), clwpth_mc(1,nz+1), ciwpth_mc(1,nz+1), cswpth_mc(1,nz+1)
    real(rk) :: rei_mc(1,nz+1), rel_mc(1,nz+1), res_mc(1,nz+1), hgt_mc(1,nz+1)
    real(rk) :: taucld_mc(sw_nbnd,1,nz+1), ssacld_mc(sw_nbnd,1,nz+1), asmcld_mc(sw_nbnd,1,nz+1), fsfcld_mc(sw_nbnd,1,nz+1)
    real(rk) :: cldfmcl_mc(sw_ngpt,1,nz+1), ciwpmcl_mc(sw_ngpt,1,nz+1), clwpmcl_mc(sw_ngpt,1,nz+1), cswpmcl_mc(sw_ngpt,1,nz+1)
    real(rk) :: taucmcl_mc(sw_ngpt,1,nz+1), ssacmcl_mc(sw_ngpt,1,nz+1), asmcmcl_mc(sw_ngpt,1,nz+1), fsfcmcl_mc(sw_ngpt,1,nz+1)
    real(rk) :: reicmcl_mc(1,nz+1), relqmcl_mc(1,nz+1), resnmcl_mc(1,nz+1)
    real(rk) :: pdel_pa, cloud_safe, dzsum, snow_mass_factor
    real(rk) :: taua_sw(nz+1,sw_nbnd), asya_sw(nz+1,sw_nbnd), omga_sw(nz+1,sw_nbnd)
    real(rk) :: bbfd(nz+2), bbfu(nz+2), bbcd(nz+2), bbcu(nz+2), uvfd(nz+2), uvcd(nz+2), nifd(nz+2), nicd(nz+2)
    real(rk) :: bbfddir(nz+2), bbcddir(nz+2), uvfddir(nz+2), uvcddir(nz+2), nifddir(nz+2), nicddir(nz+2)
    real(rk) :: colco(nz+1), colbrd(nz+1), rat_h2oco2(nz+1), rat_h2oco2_1(nz+1), rat_h2oo3(nz+1), rat_h2oo3_1(nz+1)
    real(rk) :: rat_h2on2o(nz+1), rat_h2on2o_1(nz+1), rat_h2och4(nz+1), rat_h2och4_1(nz+1), rat_n2oco2(nz+1), rat_n2oco2_1(nz+1)
    real(rk) :: rat_o3co2(nz+1), rat_o3co2_1(nz+1), minorfrac(nz+1), scaleminor(nz+1), scaleminorn2(nz+1)
    integer :: indminor(nz+1), indself_lw(nz+1), indfor_lw(nz+1), jt1_lw(nz+1)
    real(rk) :: fac00_lw(nz+1), fac01_lw(nz+1), fac10_lw(nz+1), fac11_lw(nz+1), selffac_lw(nz+1), selffrac_lw(nz+1), forfac_lw(nz+1), forfrac_lw(nz+1)
    real(rk) :: semiss(lw_nbnd), fracs_lw(nz+1,lw_ngpt), taug_lw(nz+1,lw_ngpt), taut_lw(nz+1,lw_ngpt)
    real(rk) :: cldf_lw(lw_ngpt,nz+1), taucmc_lw(lw_ngpt,nz+1)
    real(rk) :: totuflux(0:nz+1), totdflux(0:nz+1), fnet(0:nz+1), htr(0:nz+1), totuclfl(0:nz+1), totdclfl(0:nz+1), fnetc(0:nz+1), htrc(0:nz+1)

    nlay = nz + 1
    call build_rrtmg_profile(nz, temp, press, qv, pavel, tavel, pz, tz, coldry, h2ovmr)
    call fill_o3_profile(nlay, pz, o3vmr)
    wkl = 0.0_rk
    wkl_lw = 0.0_rk
    wx = 0.0_rk
    amttl = 0.0_rk
    wvttl = 0.0_rk
    do k = 1, nlay
       wkl(1,k) = coldry(k) * h2ovmr(k)
       wkl(2,k) = coldry(k) * co2_vmr_default
       wkl(3,k) = coldry(k) * o3vmr(k)
       wkl(4,k) = coldry(k) * n2o_vmr_default
       wkl(6,k) = coldry(k) * ch4_vmr_default
       wkl(7,k) = coldry(k) * o2_vmr_default
       wkl_lw(:,k) = wkl(:,k)
       wkl_lw(3,k) = coldry(k) * o3_vmr_default
       summol = co2_vmr_default + o3_vmr_default + n2o_vmr_default + ch4_vmr_default + o2_vmr_default
       wbroad(k) = coldry(k) * max(0.0_rk, 1.0_rk - summol)
       wx(1,k) = coldry(k) * ccl4_vmr_default * 1.0e-20_rk
       wx(2,k) = coldry(k) * cfc11_vmr_default * 1.0e-20_rk
       wx(3,k) = coldry(k) * cfc12_vmr_default * 1.0e-20_rk
       wx(4,k) = coldry(k) * cfc22_vmr_default * 1.0e-20_rk
       amttl = amttl + coldry(k) + wkl_lw(1,k)
       wvttl = wvttl + wkl_lw(1,k)
    enddo

    call setcoef_sw(nlay, pavel, tavel, pz, tz, surface_temperature, coldry, wkl, &
                    laytrop, layswtch, laylow, sw_jp, sw_jt, sw_jt1, &
                    co2mult, colch4, colco2, colh2o, colmol, coln2o, &
                    colo2, colo3, sw_fac00, sw_fac01, sw_fac10, sw_fac11, &
                    sw_selffac, selffrac, sw_indself, sw_forfac, forfrac, sw_indfor)
    call taumol_sw(nlay, colh2o, colco2, colch4, colo2, colo3, colmol, &
                   laytrop, sw_jp, sw_jt, sw_jt1, sw_fac00, sw_fac01, sw_fac10, sw_fac11, &
                   sw_selffac, selffrac, sw_indself, sw_forfac, forfrac, sw_indfor, sflux_sw, taug_sw, taur_sw)
    sw_colmol(:,1) = colh2o
    sw_colmol(:,2) = colco2
    sw_colmol(:,3) = colo3
    sw_colmol(:,4) = colch4
    sw_colmol(:,5) = coln2o
    sw_colmol(:,6) = colo2
    sw_taug_band = 0.0_rk
    sw_taur_band = 0.0_rk
    sw_sflux_band = 0.0_rk
    start_g = 1
    do b = 1, sw_nbnd
       do g = 1, sw_counts(b)
          sw_taug_band(:,g,b) = taug_sw(:,start_g+g-1)
          sw_taur_band(:,g,b) = taur_sw(:,start_g+g-1)
          sw_sflux_band(g,b) = sflux_sw(start_g+g-1)
       enddo
       start_g = start_g + sw_counts(b)
    enddo

    albdif = surface_albedo
    albdir = surface_albedo
    adjflux = 1.0_rk
    sw_cld_pcldfmc = 0.0_rk
    sw_cld_ptaucmc = 0.0_rk
    sw_cld_pasycmc = 0.0_rk
    sw_cld_pomgcmc = 1.0_rk
    sw_cld_ptaormc = 0.0_rk
    sw_spc_zref = 0.0_rk
    sw_spc_ztra = 0.0_rk
    sw_spc_zrefd = 0.0_rk
    sw_spc_ztrad = 0.0_rk
    sw_spc_zref_clear = 0.0_rk
    sw_spc_ztra_clear = 0.0_rk
    sw_spc_zrefd_clear = 0.0_rk
    sw_spc_ztrad_clear = 0.0_rk
    sw_spc_zref_cloud = 0.0_rk
    sw_spc_ztra_cloud = 0.0_rk
    sw_spc_zrefd_cloud = 0.0_rk
    sw_spc_ztrad_cloud = 0.0_rk
    sw_spc_direct_trans = 0.0_rk
    sw_spc_zfd = 0.0_rk
    sw_spc_zfu = 0.0_rk
    sw_spc_zfd_flux = 0.0_rk
    sw_spc_zfu_flux = 0.0_rk
    taua_sw = 0.0_rk
    asya_sw = 0.0_rk
    omga_sw = 1.0_rk

    icld_sw = 1
    inflag_sw = 5
    iceflag_sw = 5
    liqflag_sw = 1
    irng_sw = 0
    permuteseed_sw = 1
    idcor_sw = 0
    juldat_sw = 142
    play_mc = 0.0_rk
    cldfrac_mc = 0.0_rk
    clwpth_mc = 0.0_rk
    ciwpth_mc = 0.0_rk
    cswpth_mc = 0.0_rk
    rei_mc = 10.0_rk
    rel_mc = 10.0_rk
    res_mc = 10.0_rk
    hgt_mc = 0.0_rk
    taucld_mc = 0.0_rk
    ssacld_mc = 1.0_rk
    asmcld_mc = 0.0_rk
    fsfcld_mc = 0.0_rk
    dzsum = 0.0_rk
    do k = 1, nz
       pdel_pa = (pz(k-1) - pz(k)) * 100.0_rk
       cloud_safe = max(0.01_rk, min(1.0_rk, max(0.0_rk, cldfra(k))))
       play_mc(1,k) = pavel(k)
       cldfrac_mc(1,k) = min(1.0_rk, max(0.0_rk, cldfra(k)))
       clwpth_mc(1,k) = max(qc(k), 0.0_rk) * pdel_pa / gravity * 1000.0_rk / cloud_safe
       ciwpth_mc(1,k) = max(qi(k), 0.0_rk) * pdel_pa / gravity * 1000.0_rk / cloud_safe
       snow_mass_factor = 0.99_rk
       cswpth_mc(1,k) = max(qs(k), 0.0_rk) * snow_mass_factor * pdel_pa / gravity * 1000.0_rk / cloud_safe
       rel_mc(1,k) = 10.0_rk
       rei_mc(1,k) = 30.0_rk
       res_mc(1,k) = 75.0_rk
       hgt_mc(1,k) = dzsum + 0.5_rk * dz(k)
       dzsum = dzsum + dz(k)
    enddo
    play_mc(1,nlay) = pavel(nlay)
    cldfrac_mc(1,nlay) = 0.0_rk
    clwpth_mc(1,nlay) = 0.0_rk
    ciwpth_mc(1,nlay) = 0.0_rk
    cswpth_mc(1,nlay) = 0.0_rk
    rel_mc(1,nlay) = 10.0_rk
    rei_mc(1,nlay) = 10.0_rk
    res_mc(1,nlay) = 10.0_rk
    hgt_mc(1,nlay) = dzsum + 0.5_rk * dz(nz)

    call mcica_subcol_sw(1, 1, nlay, icld_sw, permuteseed_sw, irng_sw, play_mc, &
                         cldfrac_mc, ciwpth_mc, clwpth_mc, cswpth_mc, rei_mc, rel_mc, res_mc, &
                         taucld_mc, ssacld_mc, asmcld_mc, fsfcld_mc, hgt_mc, idcor_sw, juldat_sw, 28.3_rk, &
                         cldfmcl_mc, ciwpmcl_mc, clwpmcl_mc, cswpmcl_mc, reicmcl_mc, relqmcl_mc, resnmcl_mc, &
                         taucmcl_mc, ssacmcl_mc, asmcmcl_mc, fsfcmcl_mc)
    do k = 1, nlay
       reicmc_mc(k) = reicmcl_mc(1,k)
       relqmc_mc(k) = relqmcl_mc(1,k)
       resnmc_mc(k) = resnmcl_mc(1,k)
       do ig = 1, sw_ngpt
          cldfmc_mc(ig,k) = cldfmcl_mc(ig,1,k)
          ciwpmc_mc(ig,k) = ciwpmcl_mc(ig,1,k)
          clwpmc_mc(ig,k) = clwpmcl_mc(ig,1,k)
          cswpmc_mc(ig,k) = cswpmcl_mc(ig,1,k)
          taucmc_mc(ig,k) = taucmcl_mc(ig,1,k)
          ssacmc_mc(ig,k) = ssacmcl_mc(ig,1,k)
          asmcmc_mc(ig,k) = asmcmcl_mc(ig,1,k)
          fsfcmc_mc(ig,k) = fsfcmcl_mc(ig,1,k)
       enddo
    enddo
    call cldprmc_sw(nlay, inflag_sw, iceflag_sw, liqflag_sw, cldfmc_mc, &
                    ciwpmc_mc, clwpmc_mc, cswpmc_mc, reicmc_mc, relqmc_mc, resnmc_mc, &
                    taormc_mc, taucmc_mc, ssacmc_mc, asmcmc_mc, fsfcmc_mc)
    do k = 1, nlay
       do ig = 1, sw_ngpt
          cldf_sw(k,ig) = cldfmc_mc(ig,k)
          taucmc_sw(k,ig) = taucmc_mc(ig,k)
          taormc_sw(k,ig) = taormc_mc(ig,k)
          asy_sw(k,ig) = asmcmc_mc(ig,k)
          omg_sw(k,ig) = ssacmc_mc(ig,k)
       enddo
    enddo
    start_g = 1
    do b = 1, sw_nbnd
       do g = 1, sw_counts(b)
          ig = start_g + g - 1
          sw_cld_pcldfmc(:,g,b) = cldf_sw(:,ig)
          sw_cld_ptaucmc(:,g,b) = taucmc_sw(:,ig)
          sw_cld_pasycmc(:,g,b) = asy_sw(:,ig)
          sw_cld_pomgcmc(:,g,b) = omg_sw(:,ig)
          sw_cld_ptaormc(:,g,b) = taormc_sw(:,ig)
       enddo
       start_g = start_g + sw_counts(b)
    enddo

    call compute_spcvmc_sw_stage_oracle(nlay, surface_albedo, max(coszen_scalar, 1.0e-10_rk), &
                                        taug_sw, taur_sw, sflux_sw, cldf_sw, taucmc_sw, asy_sw, omg_sw, taormc_sw, &
                                        taua_sw, asya_sw, omga_sw, &
                                        sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad, &
                                        sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear, &
                                        sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud, &
                                        sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux)

    do b = 1, sw_nbnd
       call spcvmc_sw(nlay, sw_jpb1 + b - 1, sw_jpb1 + b - 1, 1, 0, &
                      pavel, tavel, pz, tz, surface_temperature, albdif, albdir, &
                      cldf_sw, taucmc_sw, asy_sw, omg_sw, taormc_sw, &
                      taua_sw, asya_sw, omga_sw, max(coszen_scalar, 1.0e-10_rk), coldry, wkl, adjflux, &
                      laytrop, layswtch, laylow, sw_jp, sw_jt, sw_jt1, &
                      co2mult, colch4, colco2, colh2o, colmol, coln2o, colo2, colo3, &
                      sw_fac00, sw_fac01, sw_fac10, sw_fac11, &
                      sw_selffac, selffrac, sw_indself, sw_forfac, forfrac, sw_indfor, &
                      bbfd, bbfu, bbcd, bbcu, uvfd, uvcd, nifd, nicd, &
                      bbfddir, bbcddir, uvfddir, uvcddir, nifddir, nicddir)
       sw_band_flux(1,b) = bbfu(nlay+1)
       sw_band_flux(2,b) = bbfd(nlay+1)
       sw_band_flux(3,b) = bbfu(1)
       sw_band_flux(4,b) = bbfd(1)
    enddo

    semiss = surface_emissivity
    call setcoef(nlay, 1, pavel, tavel, tz, surface_temperature, semiss, coldry, wkl_lw, wbroad, &
                 laytrop, lw_jp, lw_jt, jt1_lw, lw_planklay, lw_planklev, lw_plankbnd, &
                 colh2o, colco2, colo3, coln2o, colco, colch4, colo2, colbrd, &
                 fac00_lw, fac01_lw, fac10_lw, fac11_lw, &
                 rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
                 rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
                 rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
                 selffac_lw, selffrac_lw, indself_lw, forfac_lw, forfrac_lw, indfor_lw, &
                 minorfrac, scaleminor, scaleminorn2, indminor)
    call taumol(nlay, pavel, wx, coldry, laytrop, lw_jp, lw_jt, jt1_lw, lw_planklay, lw_planklev, lw_plankbnd, &
                colh2o, colco2, colo3, coln2o, colco, colch4, colo2, colbrd, &
                fac00_lw, fac01_lw, fac10_lw, fac11_lw, &
                rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
                rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
                rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
                selffac_lw, selffrac_lw, indself_lw, forfac_lw, forfrac_lw, indfor_lw, &
                minorfrac, scaleminor, scaleminorn2, indminor, fracs_lw, taug_lw)
    taut_lw = taug_lw
    lw_taug_band = 0.0_rk
    lw_fracs_band = 0.0_rk
    start_g = 1
    do b = 1, lw_nbnd
       do g = 1, lw_counts(b)
          lw_taug_band(:,g,b) = taug_lw(:,start_g+g-1)
          lw_fracs_band(:,g,b) = fracs_lw(:,start_g+g-1)
       enddo
       start_g = start_g + lw_counts(b)
    enddo
    do b = 1, lw_nbnd
       do k = 1, nlay
          lw_dplankup(k,b) = lw_planklev(k,b) - lw_planklay(k,b)
          lw_dplankdn(k,b) = lw_planklev(k-1,b) - lw_planklay(k,b)
       enddo
    enddo
    wvsh = (water_mw * wvttl) / max(dry_air_mw * amttl, 1.0e-30_rk)
    pwvcm = wvsh * (1.0e3_rk * pz(0)) / (1.0e2_rk * gravity)
    call lw_diffusivity(pwvcm, lw_secdiff)
    cldf_lw = 0.0_rk
    taucmc_lw = 0.0_rk
    ncbands = lw_nbnd
    do b = 1, lw_nbnd
       call rtrnmc(nlay, b, b, 0, pz, semiss, ncbands, cldf_lw, taucmc_lw, &
                   lw_planklay, lw_planklev, lw_plankbnd, pwvcm, fracs_lw, taut_lw, &
                   totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, htrc)
       lw_band_flux(1,b) = totuflux(nlay)
       lw_band_flux(2,b) = totdflux(nlay)
       lw_band_flux(3,b) = totuflux(0)
       lw_band_flux(4,b) = totdflux(0)
    enddo
  end subroutine compute_intermediate_oracle

  subroutine compute_spcvmc_sw_stage_oracle(nlay, surface_albedo, prmu0, ztaug, ztaur, zsflxzen, &
                                            pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc, &
                                            ptaua, pasya, pomga, zref_out, ztra_out, zrefd_out, ztrad_out, &
                                            zref_clear_out, ztra_clear_out, zrefd_clear_out, ztrad_clear_out, &
                                            zref_cloud_out, ztra_cloud_out, zrefd_cloud_out, ztrad_cloud_out, &
                                            direct_trans_out, zfd_out, zfu_out, zfd_flux_out, zfu_flux_out)
    integer, intent(in) :: nlay
    real(rk), intent(in) :: surface_albedo, prmu0
    real(rk), intent(in) :: ztaug(nlay,sw_ngpt), ztaur(nlay,sw_ngpt), zsflxzen(sw_ngpt)
    real(rk), intent(in) :: pcldfmc(nlay,sw_ngpt), ptaucmc(nlay,sw_ngpt), pasycmc(nlay,sw_ngpt), pomgcmc(nlay,sw_ngpt), ptaormc(nlay,sw_ngpt)
    real(rk), intent(in) :: ptaua(nlay,sw_nbnd), pasya(nlay,sw_nbnd), pomga(nlay,sw_nbnd)
    real(rk), intent(out) :: zref_out(nlay+1,sw_max_g,sw_nbnd), ztra_out(nlay+1,sw_max_g,sw_nbnd), zrefd_out(nlay+1,sw_max_g,sw_nbnd), ztrad_out(nlay+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: zref_clear_out(nlay+1,sw_max_g,sw_nbnd), ztra_clear_out(nlay+1,sw_max_g,sw_nbnd), zrefd_clear_out(nlay+1,sw_max_g,sw_nbnd), ztrad_clear_out(nlay+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: zref_cloud_out(nlay+1,sw_max_g,sw_nbnd), ztra_cloud_out(nlay+1,sw_max_g,sw_nbnd), zrefd_cloud_out(nlay+1,sw_max_g,sw_nbnd), ztrad_cloud_out(nlay+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: direct_trans_out(nlay,sw_max_g,sw_nbnd), zfd_out(nlay+1,sw_max_g,sw_nbnd), zfu_out(nlay+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: zfd_flux_out(nlay+1,sw_max_g,sw_nbnd), zfu_flux_out(nlay+1,sw_max_g,sw_nbnd)

    integer, parameter :: sw_counts(sw_nbnd) = (/6,12,8,8,10,10,2,10,8,6,6,8,6,12/)
    logical :: lrtchkclr(nlay), lrtchkcld(nlay)
    integer :: b, jg, jk, ikl, iw, igt, itind
    real(rk) :: tblind, ze1, zclear, zcloud, zdbtmc, zdbtmo, zf, zwf, tauorig, zincflx
    real(rk) :: zdbt(nlay+1), zdbt_nodel(nlay+1), zdbtc(nlay+1), zdbtc_nodel(nlay+1)
    real(rk) :: ztdbt(nlay+1), ztdbt_nodel(nlay+1), ztdbtc(nlay+1), ztdbtc_nodel(nlay+1)
    real(rk) :: zgcc(nlay), zgco(nlay), zomcc(nlay), zomco(nlay), ztauc(nlay), ztauo(nlay)
    real(rk) :: zref(nlay+1), zrefc(nlay+1), zrefo(nlay+1), zrefd(nlay+1), zrefdc(nlay+1), zrefdo(nlay+1)
    real(rk) :: ztra(nlay+1), ztrac(nlay+1), ztrao(nlay+1), ztrad(nlay+1), ztradc(nlay+1), ztrado(nlay+1)
    real(rk) :: zrdnd(nlay+1), zrdndc(nlay+1), zrup(nlay+1), zrupd(nlay+1), zrupc(nlay+1), zrupdc(nlay+1)
    real(rk) :: zcd(nlay+1,sw_ngpt), zcu(nlay+1,sw_ngpt), zfd(nlay+1,sw_ngpt), zfu(nlay+1,sw_ngpt)

    zref_out = 0.0_rk
    ztra_out = 0.0_rk
    zrefd_out = 0.0_rk
    ztrad_out = 0.0_rk
    zref_clear_out = 0.0_rk
    ztra_clear_out = 0.0_rk
    zrefd_clear_out = 0.0_rk
    ztrad_clear_out = 0.0_rk
    zref_cloud_out = 0.0_rk
    ztra_cloud_out = 0.0_rk
    zrefd_cloud_out = 0.0_rk
    ztrad_cloud_out = 0.0_rk
    direct_trans_out = 0.0_rk
    zfd_out = 0.0_rk
    zfu_out = 0.0_rk
    zfd_flux_out = 0.0_rk
    zfu_flux_out = 0.0_rk
    zcd = 0.0_rk
    zcu = 0.0_rk
    zfd = 0.0_rk
    zfu = 0.0_rk

    iw = 0
    do b = 1, sw_nbnd
       igt = sw_counts(b)
       do jg = 1, igt
          iw = iw + 1
          zincflx = zsflxzen(iw) * prmu0

          ztdbtc(1) = 1.0_rk
          ztdbtc_nodel(1) = 1.0_rk
          zdbtc(nlay+1) = 0.0_rk
          ztrac(nlay+1) = 0.0_rk
          ztradc(nlay+1) = 0.0_rk
          zrefc(nlay+1) = surface_albedo
          zrefdc(nlay+1) = surface_albedo
          zrupc(nlay+1) = surface_albedo
          zrupdc(nlay+1) = surface_albedo

          ztdbt(1) = 1.0_rk
          ztdbt_nodel(1) = 1.0_rk
          zdbt(nlay+1) = 0.0_rk
          ztra(nlay+1) = 0.0_rk
          ztrad(nlay+1) = 0.0_rk
          zref(nlay+1) = surface_albedo
          zrefd(nlay+1) = surface_albedo
          zrup(nlay+1) = surface_albedo
          zrupd(nlay+1) = surface_albedo
          zrefo(nlay+1) = surface_albedo
          zrefdo(nlay+1) = surface_albedo
          ztrao(nlay+1) = 0.0_rk
          ztrado(nlay+1) = 0.0_rk

          do jk = 1, nlay
             ikl = nlay + 1 - jk
             lrtchkclr(jk) = .true.
             lrtchkcld(jk) = (pcldfmc(ikl,iw) > 1.0e-12_rk)

             ztauc(jk) = ztaur(ikl,iw) + ztaug(ikl,iw) + ptaua(ikl,b)
             zomcc(jk) = ztaur(ikl,iw) + ptaua(ikl,b) * pomga(ikl,b)
             zgcc(jk) = pasya(ikl,b) * pomga(ikl,b) * ptaua(ikl,b) / zomcc(jk)
             zomcc(jk) = zomcc(jk) / ztauc(jk)

             zclear = 1.0_rk - pcldfmc(ikl,iw)
             zcloud = pcldfmc(ikl,iw)
             ze1 = ztauc(jk) / prmu0
             if (ze1 .le. od_lo) then
                zdbtmc = 1.0_rk - ze1 + 0.5_rk * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5_rk
                zdbtmc = exp_tbl(itind)
             endif
             zdbtc_nodel(jk) = zdbtmc
             ztdbtc_nodel(jk+1) = zdbtc_nodel(jk) * ztdbtc_nodel(jk)

             tauorig = ztauc(jk) + ptaormc(ikl,iw)
             ze1 = tauorig / prmu0
             if (ze1 .le. od_lo) then
                zdbtmo = 1.0_rk - ze1 + 0.5_rk * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5_rk
                zdbtmo = exp_tbl(itind)
             endif
             zdbt_nodel(jk) = zclear * zdbtmc + zcloud * zdbtmo
             ztdbt_nodel(jk+1) = zdbt_nodel(jk) * ztdbt_nodel(jk)
          enddo

          do jk = 1, nlay
             zf = zgcc(jk) * zgcc(jk)
             zwf = zomcc(jk) * zf
             ztauc(jk) = (1.0_rk - zwf) * ztauc(jk)
             zomcc(jk) = (zomcc(jk) - zwf) / (1.0_rk - zwf)
             zgcc(jk) = (zgcc(jk) - zf) / (1.0_rk - zf)
          enddo

          do jk = 1, nlay
             ikl = nlay + 1 - jk
             ztauo(jk) = ztauc(jk) + ptaucmc(ikl,iw)
             zomco(jk) = ztauc(jk) * zomcc(jk) + ptaucmc(ikl,iw) * pomgcmc(ikl,iw)
             zgco(jk) = (ptaucmc(ikl,iw) * pomgcmc(ikl,iw) * pasycmc(ikl,iw) + ztauc(jk) * zomcc(jk) * zgcc(jk)) / zomco(jk)
             zomco(jk) = zomco(jk) / ztauo(jk)
          enddo

          call reftra_sw(nlay, lrtchkclr, zgcc, prmu0, ztauc, zomcc, zrefc, zrefdc, ztrac, ztradc)
          call reftra_sw(nlay, lrtchkcld, zgco, prmu0, ztauo, zomco, zrefo, zrefdo, ztrao, ztrado)

          do jk = 1, nlay
             ikl = nlay + 1 - jk
             zclear = 1.0_rk - pcldfmc(ikl,iw)
             zcloud = pcldfmc(ikl,iw)
             zref(jk) = zclear * zrefc(jk) + zcloud * zrefo(jk)
             zrefd(jk) = zclear * zrefdc(jk) + zcloud * zrefdo(jk)
             ztra(jk) = zclear * ztrac(jk) + zcloud * ztrao(jk)
             ztrad(jk) = zclear * ztradc(jk) + zcloud * ztrado(jk)

             ze1 = ztauc(jk) / prmu0
             if (ze1 .le. od_lo) then
                zdbtmc = 1.0_rk - ze1 + 0.5_rk * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5_rk
                zdbtmc = exp_tbl(itind)
             endif
             zdbtc(jk) = zdbtmc
             ztdbtc(jk+1) = zdbtc(jk) * ztdbtc(jk)

             ze1 = ztauo(jk) / prmu0
             if (ze1 .le. od_lo) then
                zdbtmo = 1.0_rk - ze1 + 0.5_rk * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5_rk
                zdbtmo = exp_tbl(itind)
             endif
             zdbt(jk) = zclear * zdbtmc + zcloud * zdbtmo
             ztdbt(jk+1) = zdbt(jk) * ztdbt(jk)
          enddo

          call vrtqdr_sw(nlay, iw, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, zcd, zcu)
          call vrtqdr_sw(nlay, iw, zref, zrefd, ztra, ztrad, zdbt, zrdnd, zrup, zrupd, ztdbt, zfd, zfu)

          zref_out(:,jg,b) = zref
          ztra_out(:,jg,b) = ztra
          zrefd_out(:,jg,b) = zrefd
          ztrad_out(:,jg,b) = ztrad
          zref_clear_out(:,jg,b) = zrefc
          ztra_clear_out(:,jg,b) = ztrac
          zrefd_clear_out(:,jg,b) = zrefdc
          ztrad_clear_out(:,jg,b) = ztradc
          zref_cloud_out(:,jg,b) = zrefo
          ztra_cloud_out(:,jg,b) = ztrao
          zrefd_cloud_out(:,jg,b) = zrefdo
          ztrad_cloud_out(:,jg,b) = ztrado
          direct_trans_out(:,jg,b) = zdbt(1:nlay)
          zfd_out(:,jg,b) = zfd(:,iw)
          zfu_out(:,jg,b) = zfu(:,iw)
          do jk = 1, nlay + 1
             ikl = nlay + 2 - jk
             zfd_flux_out(ikl,jg,b) = zincflx * zfd(jk,iw)
             zfu_flux_out(ikl,jg,b) = zincflx * zfu(jk,iw)
          enddo
       enddo
    enddo
  end subroutine compute_spcvmc_sw_stage_oracle

  subroutine build_rrtmg_profile(nz, temp, press, qv, pavel, tavel, pz, tz, coldry, h2ovmr)
    integer, intent(in) :: nz
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz)
    real(rk), intent(out) :: pavel(nz+1), tavel(nz+1), pz(0:nz+1), tz(0:nz+1), coldry(nz+1), h2ovmr(nz+1)
    integer :: k, nlay
    real(rk) :: p8(0:nz), t8(0:nz), dp_bottom, dp_top, amm

    nlay = nz + 1
    dp_bottom = max(10.0_rk, press(1) - press(2))
    p8(0) = press(1) + 0.5_rk * dp_bottom
    t8(0) = temp(1)
    do k = 1, nz-1
       p8(k) = 0.5_rk * (press(k) + press(k+1))
       t8(k) = 0.5_rk * (temp(k) + temp(k+1))
    enddo
    dp_top = max(10.0_rk, press(nz-1) - press(nz))
    p8(nz) = max(400.0_rk, press(nz) - 0.5_rk * dp_top)
    t8(nz) = temp(nz)

    pz(0:nz) = p8(0:nz) * 0.01_rk
    tz(0:nz) = t8(0:nz)
    pavel(1:nz) = press(1:nz) * 0.01_rk
    tavel(1:nz) = temp(1:nz)
    pavel(nlay) = 0.5_rk * pz(nz)
    tavel(nlay) = tz(nz)
    pz(nlay) = 1.0e-5_rk
    tz(nlay) = tz(nz)
    do k = 1, nz
       h2ovmr(k) = max(qv(k), 1.0e-12_rk) * h2o_mmr_to_vmr
    enddo
    h2ovmr(nlay) = h2ovmr(nz)
    do k = 1, nlay
       amm = (1.0_rk - h2ovmr(k)) * dry_air_mw + h2ovmr(k) * water_mw
       coldry(k) = max(pz(k-1) - pz(k), 1.0e-12_rk) * 1.0e3_rk * avogadro / &
                   (1.0e2_rk * gravity * amm * (1.0_rk + h2ovmr(k)))
    enddo
  end subroutine build_rrtmg_profile

  subroutine fill_o3_profile(nlay, pz, o3vmr)
    integer, intent(in) :: nlay
    real(rk), intent(in) :: pz(0:nlay)
    real(rk), intent(out) :: o3vmr(nlay)
    real(rk), parameter :: o3sum(31) = (/ &
         5.297e-8_rk, 5.852e-8_rk, 6.579e-8_rk, 7.505e-8_rk, &
         8.577e-8_rk, 9.895e-8_rk, 1.175e-7_rk, 1.399e-7_rk, &
         1.677e-7_rk, 2.003e-7_rk, 2.571e-7_rk, 3.325e-7_rk, &
         4.438e-7_rk, 6.255e-7_rk, 8.168e-7_rk, 1.036e-6_rk, &
         1.366e-6_rk, 1.855e-6_rk, 2.514e-6_rk, 3.240e-6_rk, &
         4.033e-6_rk, 4.854e-6_rk, 5.517e-6_rk, 6.089e-6_rk, &
         6.689e-6_rk, 1.106e-5_rk, 1.462e-5_rk, 1.321e-5_rk, &
         9.856e-6_rk, 5.960e-6_rk, 5.960e-6_rk /)
    real(rk), parameter :: ppsum(31) = (/ &
         955.890_rk, 850.532_rk, 754.599_rk, 667.742_rk, 589.841_rk, &
         519.421_rk, 455.480_rk, 398.085_rk, 347.171_rk, 301.735_rk, &
         261.310_rk, 225.360_rk, 193.419_rk, 165.490_rk, 141.032_rk, &
         120.125_rk, 102.689_rk, 87.829_rk, 75.123_rk, 64.306_rk, &
         55.086_rk, 47.209_rk, 40.535_rk, 34.795_rk, 29.865_rk, &
         19.122_rk, 9.277_rk, 4.660_rk, 2.421_rk, 1.294_rk, 0.647_rk /)
    real(rk), parameter :: o3win(31) = (/ &
         4.629e-8_rk, 4.686e-8_rk, 5.017e-8_rk, 5.613e-8_rk, &
         6.871e-8_rk, 8.751e-8_rk, 1.138e-7_rk, 1.516e-7_rk, &
         2.161e-7_rk, 3.264e-7_rk, 4.968e-7_rk, 7.338e-7_rk, &
         1.017e-6_rk, 1.308e-6_rk, 1.625e-6_rk, 2.011e-6_rk, &
         2.516e-6_rk, 3.130e-6_rk, 3.840e-6_rk, 4.703e-6_rk, &
         5.486e-6_rk, 6.289e-6_rk, 6.993e-6_rk, 7.494e-6_rk, &
         8.197e-6_rk, 9.632e-6_rk, 1.113e-5_rk, 1.146e-5_rk, &
         9.389e-6_rk, 6.135e-6_rk, 6.135e-6_rk /)
    real(rk), parameter :: ppwin(31) = (/ &
         955.747_rk, 841.783_rk, 740.199_rk, 649.538_rk, 568.404_rk, &
         495.815_rk, 431.069_rk, 373.464_rk, 322.354_rk, 277.190_rk, &
         237.635_rk, 203.433_rk, 174.070_rk, 148.949_rk, 127.408_rk, &
         108.915_rk, 93.114_rk, 79.551_rk, 67.940_rk, 58.072_rk, &
         49.593_rk, 42.318_rk, 36.138_rk, 30.907_rk, 26.362_rk, &
         16.423_rk, 7.583_rk, 3.620_rk, 1.807_rk, 0.938_rk, 0.469_rk /)
    integer :: k, jj
    real(rk) :: o3ann(31), ppwrkh(32), pb1, pb2, pt1, pt2, o3mmr

    o3ann(1) = 0.5_rk * (o3sum(1) + o3win(1))
    do k = 2, 31
       o3ann(k) = o3win(k-1) + (o3win(k) - o3win(k-1)) / (ppwin(k) - ppwin(k-1)) * (ppsum(k) - ppwin(k-1))
       o3ann(k) = 0.5_rk * (o3ann(k) + o3sum(k))
    enddo
    ppwrkh(1) = 1100.0_rk
    do k = 2, 31
       ppwrkh(k) = 0.5_rk * (ppsum(k) + ppsum(k-1))
    enddo
    ppwrkh(32) = 0.0_rk

    do k = 1, nlay
       o3mmr = 0.0_rk
       do jj = 1, 31
          if (pz(k-1) <= ppwrkh(jj)) then
             pb1 = 0.0_rk
          else
             pb1 = pz(k-1) - ppwrkh(jj)
          endif
          if (pz(k-1) <= ppwrkh(jj+1)) then
             pb2 = 0.0_rk
          else
             pb2 = pz(k-1) - ppwrkh(jj+1)
          endif
          if (pz(k) <= ppwrkh(jj)) then
             pt1 = 0.0_rk
          else
             pt1 = pz(k) - ppwrkh(jj)
          endif
          if (pz(k) <= ppwrkh(jj+1)) then
             pt2 = 0.0_rk
          else
             pt2 = pz(k) - ppwrkh(jj+1)
          endif
          o3mmr = o3mmr + (pb2 - pb1 - pt2 + pt1) * o3ann(jj)
       enddo
       o3vmr(k) = o3mmr / max(pz(k-1) - pz(k), 1.0e-12_rk) * o3_mmr_to_vmr
    enddo
  end subroutine fill_o3_profile

  subroutine lw_diffusivity(pwvcm, secdiff)
    real(rk), intent(in) :: pwvcm
    real(rk), intent(out) :: secdiff(lw_nbnd)
    real(rk), parameter :: a0(lw_nbnd) = (/1.66_rk,1.55_rk,1.58_rk,1.66_rk,1.54_rk,1.454_rk,1.89_rk,1.33_rk,1.668_rk,1.66_rk,1.66_rk,1.66_rk,1.66_rk,1.66_rk,1.66_rk,1.66_rk/)
    real(rk), parameter :: a1(lw_nbnd) = (/0.0_rk,0.25_rk,0.22_rk,0.0_rk,0.13_rk,0.446_rk,-0.10_rk,0.40_rk,-0.006_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk/)
    real(rk), parameter :: a2(lw_nbnd) = (/0.0_rk,-12.0_rk,-11.7_rk,0.0_rk,-0.72_rk,-0.243_rk,0.19_rk,-0.062_rk,0.414_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk,0.0_rk/)
    integer :: b
    do b = 1, lw_nbnd
       if (b == 1 .or. b == 4 .or. b >= 10) then
          secdiff(b) = 1.66_rk
       else
          secdiff(b) = a0(b) + a1(b) * exp(a2(b) * pwvcm)
          secdiff(b) = min(1.80_rk, max(1.50_rk, secdiff(b)))
       endif
    enddo
  end subroutine lw_diffusivity

  subroutine append_intermediate_oracle(output_path, nlay_sw, nlay_lw, sw_band_flux, lw_band_flux, &
                                        sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                        sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                        sw_taug, sw_taur, sw_sfluxzen, sw_cld_pcldfmc, sw_cld_ptaucmc, sw_cld_pasycmc, &
                                        sw_cld_pomgcmc, sw_cld_ptaormc, sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad, &
                                        sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear, &
                                        sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud, &
                                        sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux, &
                                        lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                        lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn)
    character(len=*), intent(in) :: output_path
    integer, intent(in) :: nlay_sw, nlay_lw
    real(rk), intent(in) :: sw_band_flux(4,sw_nbnd), lw_band_flux(4,lw_nbnd)
    integer, intent(in) :: sw_jp(nlay_sw), sw_jt(nlay_sw), sw_jt1(nlay_sw), sw_indself(nlay_sw), sw_indfor(nlay_sw)
    integer, intent(in) :: lw_jp(nlay_lw), lw_jt(nlay_lw)
    real(rk), intent(in) :: sw_fac00(nlay_sw), sw_fac01(nlay_sw), sw_fac10(nlay_sw), sw_fac11(nlay_sw), sw_selffac(nlay_sw), sw_forfac(nlay_sw)
    real(rk), intent(in) :: sw_colmol(nlay_sw,6), sw_taug(nlay_sw,sw_max_g,sw_nbnd), sw_taur(nlay_sw,sw_max_g,sw_nbnd), sw_sfluxzen(sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_cld_pcldfmc(nlay_sw,sw_max_g,sw_nbnd), sw_cld_ptaucmc(nlay_sw,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_cld_pasycmc(nlay_sw,sw_max_g,sw_nbnd), sw_cld_pomgcmc(nlay_sw,sw_max_g,sw_nbnd), sw_cld_ptaormc(nlay_sw,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_spc_zref(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztra(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_zrefd(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztrad(nlay_sw+1,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_spc_zref_clear(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztra_clear(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_zrefd_clear(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztrad_clear(nlay_sw+1,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_spc_zref_cloud(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztra_cloud(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_zrefd_cloud(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_ztrad_cloud(nlay_sw+1,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_spc_direct_trans(nlay_sw,sw_max_g,sw_nbnd), sw_spc_zfd(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_zfu(nlay_sw+1,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: sw_spc_zfd_flux(nlay_sw+1,sw_max_g,sw_nbnd), sw_spc_zfu_flux(nlay_sw+1,sw_max_g,sw_nbnd)
    real(rk), intent(in) :: lw_planklay(nlay_lw,lw_nbnd), lw_planklev(0:nlay_lw,lw_nbnd), lw_plankbnd(lw_nbnd)
    real(rk), intent(in) :: lw_taug(nlay_lw,lw_max_g,lw_nbnd), lw_fracs(nlay_lw,lw_max_g,lw_nbnd), lw_secdiff(lw_nbnd)
    real(rk), intent(in) :: lw_dplankup(nlay_lw,lw_nbnd), lw_dplankdn(nlay_lw,lw_nbnd)
    integer :: dims(6)

    dims = (/nlay_sw, nlay_lw, sw_max_g, lw_max_g, sw_nbnd, lw_nbnd/)
    open(unit=21, file=trim(output_path), status='old', access='stream', form='unformatted', position='append', action='write')
    write(21) dims
    write(21) sw_band_flux
    write(21) lw_band_flux
    write(21) sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor
    write(21) sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac
    write(21) sw_colmol, sw_taug, sw_taur, sw_sfluxzen
    write(21) lw_jp, lw_jt
    write(21) lw_planklay, lw_planklev, lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn
    write(21) sw_cld_pcldfmc, sw_cld_ptaucmc, sw_cld_pasycmc, sw_cld_pomgcmc, sw_cld_ptaormc
    write(21) sw_spc_zref, sw_spc_ztra, sw_spc_zrefd, sw_spc_ztrad
    write(21) sw_spc_zref_clear, sw_spc_ztra_clear, sw_spc_zrefd_clear, sw_spc_ztrad_clear
    write(21) sw_spc_zref_cloud, sw_spc_ztra_cloud, sw_spc_zrefd_cloud, sw_spc_ztrad_cloud
    write(21) sw_spc_direct_trans, sw_spc_zfd, sw_spc_zfu, sw_spc_zfd_flux, sw_spc_zfu_flux
    close(21)
  end subroutine append_intermediate_oracle

  subroutine run_wrf_rrtmg_drivers(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                   temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                   sw_heat, lw_heat, sw_down, sw_up, lw_down, lw_up, &
                                   sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(out) :: sw_heat(nz), lw_heat(nz), sw_down(0:nz+1), sw_up(0:nz+1), lw_down(0:nz+1), lw_up(0:nz+1)
    real(rk), intent(out) :: sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit, layer_mass_p(nz)

    integer :: ids, ide, jds, jde, kds, kde
    integer :: ims, ime, jms, jme, kms, kme
    integer :: its, ite, jts, jte, kts, kte
    integer :: no_src, sf_surface_physics, yr, julday, ghg_input
    integer :: icloud, cldovrlp, idcor, o3input, mp_physics
    integer :: has_reqc, has_reqi, has_reqs, aer_opt, aer_ra_feedback, calc_clean_atm_diag, progn
    real(rk) :: xtime, gmt, radt, degrad, declin, solcon, julian, r_d, grav
    real(rk) :: net_surface, net_top
    logical :: warm_rain, is_cammgmp_used, f_qv, f_qc, f_qr, f_qi, f_qs, f_qg, proceed_cmaq_sw
    real(rk), pointer :: tauaer3d_sw(:,:,:,:), ssaaer3d_sw(:,:,:,:), asyaer3d_sw(:,:,:,:)

    real(rk), allocatable :: t3d(:,:,:), t8w(:,:,:), p3d(:,:,:), p8w(:,:,:), pi3d(:,:,:), rho3d(:,:,:), dz8w(:,:,:)
    real(rk), allocatable :: cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), qi3d(:,:,:), qs3d(:,:,:), qg3d(:,:,:)
    real(rk), allocatable :: o33d(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
    real(rk), allocatable :: rthratensw(:,:,:), rthratenswc(:,:,:), rthratenlw(:,:,:), rthratenlwc(:,:,:)
    real(rk), allocatable :: swupflx(:,:,:), swupflxc(:,:,:), swdnflx(:,:,:), swdnflxc(:,:,:)
    real(rk), allocatable :: lwupflx(:,:,:), lwupflxc(:,:,:), lwdnflx(:,:,:), lwdnflxc(:,:,:)
    real(rk), allocatable :: xlat(:,:), xlong(:,:), xland(:,:), xice(:,:), snow(:,:), tsk(:,:), albedo(:,:), emiss(:,:)
    real(rk), allocatable :: coszr(:,:), xcoszen(:,:), obscur(:,:)
    real(rk), allocatable :: gsw(:,:), swcf(:,:), glw(:,:), olr(:,:), lwcf(:,:)
    real(rk), allocatable :: swupt(:,:), swuptc(:,:), swuptcln(:,:), swdnt(:,:), swdntc(:,:), swdntcln(:,:)
    real(rk), allocatable :: swupb(:,:), swupbc(:,:), swupbcln(:,:), swdnb(:,:), swdnbc(:,:), swdnbcln(:,:)
    real(rk), allocatable :: lwupt(:,:), lwuptc(:,:), lwuptcln(:,:), lwdnt(:,:), lwdntc(:,:), lwdntcln(:,:)
    real(rk), allocatable :: lwupb(:,:), lwupbc(:,:), lwupbcln(:,:), lwdnb(:,:), lwdnbc(:,:), lwdnbcln(:,:)
    real(rk), allocatable :: swvisdir(:,:), swvisdif(:,:), swnirdir(:,:), swnirdif(:,:)
    real(rk), allocatable :: swddir(:,:), swddni(:,:), swddif(:,:), swdownc(:,:), swddnic(:,:), swddirc(:,:)
    real(rk), allocatable :: alswvisdir(:,:), alswvisdif(:,:), alswnirdir(:,:), alswnirdif(:,:)

    ids = 1; ide = 1; jds = 1; jde = 1; kds = 1; kde = nz + 1
    ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
    its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz
    no_src = 6
    sf_surface_physics = 0
    yr = 2026
    julday = 142
    julian = 142.0_rk
    ghg_input = 0
    xtime = 720.0_rk
    gmt = 12.0_rk
    radt = 30.0_rk
    degrad = 0.017453292519943295_rk
    declin = 0.0_rk
    solcon = 1368.22_rk
    r_d = rd_air
    grav = gravity
    icloud = 1
    cldovrlp = 1
    idcor = 0
    o3input = 0
    mp_physics = 8
    has_reqc = 1
    has_reqi = 1
    has_reqs = 1
    aer_opt = 0
    aer_ra_feedback = 0
    calc_clean_atm_diag = 0
    progn = 0
    warm_rain = .false.
    is_cammgmp_used = .false.
    f_qv = .true.; f_qc = .true.; f_qr = .true.; f_qi = .true.; f_qs = .true.; f_qg = .true.
    proceed_cmaq_sw = .false.

    nullify(tauaer3d_sw)
    nullify(ssaaer3d_sw)
    nullify(asyaer3d_sw)

    allocate(t3d(1,kms:kme,1), t8w(1,kms:kme,1), p3d(1,kms:kme,1), p8w(1,kms:kme,1), pi3d(1,kms:kme,1), rho3d(1,kms:kme,1), dz8w(1,kms:kme,1))
    allocate(cldfra3d(1,kms:kme,1), qv3d(1,kms:kme,1), qc3d(1,kms:kme,1), qr3d(1,kms:kme,1), qi3d(1,kms:kme,1), qs3d(1,kms:kme,1), qg3d(1,kms:kme,1))
    allocate(o33d(1,kms:kme,1), re_cloud(1,kms:kme,1), re_ice(1,kms:kme,1), re_snow(1,kms:kme,1))
    allocate(rthratensw(1,kms:kme,1), rthratenswc(1,kms:kme,1), rthratenlw(1,kms:kme,1), rthratenlwc(1,kms:kme,1))
    allocate(swupflx(1,kms:kme+2,1), swupflxc(1,kms:kme+2,1), swdnflx(1,kms:kme+2,1), swdnflxc(1,kms:kme+2,1))
    allocate(lwupflx(1,kms:kme+2,1), lwupflxc(1,kms:kme+2,1), lwdnflx(1,kms:kme+2,1), lwdnflxc(1,kms:kme+2,1))
    allocate(xlat(1,1), xlong(1,1), xland(1,1), xice(1,1), snow(1,1), tsk(1,1), albedo(1,1), emiss(1,1))
    allocate(coszr(1,1), xcoszen(1,1), obscur(1,1))
    allocate(gsw(1,1), swcf(1,1), glw(1,1), olr(1,1), lwcf(1,1))
    allocate(swupt(1,1), swuptc(1,1), swuptcln(1,1), swdnt(1,1), swdntc(1,1), swdntcln(1,1))
    allocate(swupb(1,1), swupbc(1,1), swupbcln(1,1), swdnb(1,1), swdnbc(1,1), swdnbcln(1,1))
    allocate(lwupt(1,1), lwuptc(1,1), lwuptcln(1,1), lwdnt(1,1), lwdntc(1,1), lwdntcln(1,1))
    allocate(lwupb(1,1), lwupbc(1,1), lwupbcln(1,1), lwdnb(1,1), lwdnbc(1,1), lwdnbcln(1,1))
    allocate(swvisdir(1,1), swvisdif(1,1), swnirdir(1,1), swnirdif(1,1))
    allocate(swddir(1,1), swddni(1,1), swddif(1,1), swdownc(1,1), swddnic(1,1), swddirc(1,1))
    allocate(alswvisdir(1,1), alswvisdif(1,1), alswnirdir(1,1), alswnirdif(1,1))

    call initialize_column_arrays(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                  temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                  t3d, t8w, p3d, p8w, pi3d, rho3d, dz8w, cldfra3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, &
                                  o33d, re_cloud, re_ice, re_snow, xlat, xlong, xland, xice, snow, tsk, albedo, emiss, &
                                  coszr, xcoszen, obscur, alswvisdir, alswvisdif, alswnirdir, alswnirdif, layer_mass_p)

    rthratensw = 0.0_rk; rthratenswc = 0.0_rk; rthratenlw = 0.0_rk; rthratenlwc = 0.0_rk
    swupflx = 0.0_rk; swupflxc = 0.0_rk; swdnflx = 0.0_rk; swdnflxc = 0.0_rk
    lwupflx = 0.0_rk; lwupflxc = 0.0_rk; lwdnflx = 0.0_rk; lwdnflxc = 0.0_rk
    gsw = 0.0_rk; swcf = 0.0_rk; glw = 0.0_rk; olr = 0.0_rk; lwcf = 0.0_rk
    swupt = 0.0_rk; swuptc = 0.0_rk; swuptcln = 0.0_rk; swdnt = 0.0_rk; swdntc = 0.0_rk; swdntcln = 0.0_rk
    swupb = 0.0_rk; swupbc = 0.0_rk; swupbcln = 0.0_rk; swdnb = 0.0_rk; swdnbc = 0.0_rk; swdnbcln = 0.0_rk
    lwupt = 0.0_rk; lwuptc = 0.0_rk; lwuptcln = 0.0_rk; lwdnt = 0.0_rk; lwdntc = 0.0_rk; lwdntcln = 0.0_rk
    lwupb = 0.0_rk; lwupbc = 0.0_rk; lwupbcln = 0.0_rk; lwdnb = 0.0_rk; lwdnbc = 0.0_rk; lwdnbcln = 0.0_rk
    swvisdir = 0.0_rk; swvisdif = 0.0_rk; swnirdir = 0.0_rk; swnirdif = 0.0_rk
    swddir = 0.0_rk; swddni = 0.0_rk; swddif = 0.0_rk; swdownc = 0.0_rk; swddnic = 0.0_rk; swddirc = 0.0_rk

    call rrtmg_swrad( &
         rthratensw=rthratensw, rthratenswc=rthratenswc, &
         swupt=swupt, swuptc=swuptc, swuptcln=swuptcln, swdnt=swdnt, swdntc=swdntc, swdntcln=swdntcln, &
         swupb=swupb, swupbc=swupbc, swupbcln=swupbcln, swdnb=swdnb, swdnbc=swdnbc, swdnbcln=swdnbcln, &
         swcf=swcf, gsw=gsw, xtime=xtime, gmt=gmt, xlat=xlat, xlong=xlong, radt=radt, degrad=degrad, declin=declin, &
         coszr=coszr, julday=julday, solcon=solcon, albedo=albedo, t3d=t3d, t8w=t8w, tsk=tsk, p3d=p3d, p8w=p8w, pi3d=pi3d, rho3d=rho3d, &
         dz8w=dz8w, cldfra3d=cldfra3d, lradius=re_cloud, iradius=re_ice, is_cammgmp_used=is_cammgmp_used, r=r_d, g=grav, &
         re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
         icloud=icloud, warm_rain=warm_rain, cldovrlp=cldovrlp, idcor=idcor, f_ice_phy=qi3d, f_rain_phy=qr3d, &
         xland=xland, xice=xice, snow=snow, qv3d=qv3d, qc3d=qc3d, qr3d=qr3d, qi3d=qi3d, qs3d=qs3d, qg3d=qg3d, &
         o3input=o3input, o33d=o33d, aer_opt=aer_opt, no_src=no_src, &
         alswvisdir=alswvisdir, alswvisdif=alswvisdif, alswnirdir=alswnirdir, alswnirdif=alswnirdif, &
         swvisdir=swvisdir, swvisdif=swvisdif, swnirdir=swnirdir, swnirdif=swnirdif, sf_surface_physics=sf_surface_physics, &
         f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, aer_ra_feedback=aer_ra_feedback, progn=progn, &
         calc_clean_atm_diag=calc_clean_atm_diag, mp_physics=mp_physics, ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
         swupflx=swupflx, swupflxc=swupflxc, swdnflx=swdnflx, swdnflxc=swdnflxc, tauaer3d_sw=tauaer3d_sw, ssaaer3d_sw=ssaaer3d_sw, asyaer3d_sw=asyaer3d_sw, &
         swddir=swddir, swddni=swddni, swddif=swddif, swdownc=swdownc, swddnic=swddnic, swddirc=swddirc, &
         xcoszen=xcoszen, yr=yr, julian=julian, ghg_input=ghg_input, obscur=obscur, proceed_cmaq_sw=proceed_cmaq_sw)

    call rrtmg_lwrad( &
         rthratenlw=rthratenlw, rthratenlwc=rthratenlwc, &
         lwupt=lwupt, lwuptc=lwuptc, lwuptcln=lwuptcln, lwdnt=lwdnt, lwdntc=lwdntc, lwdntcln=lwdntcln, &
         lwupb=lwupb, lwupbc=lwupbc, lwupbcln=lwupbcln, lwdnb=lwdnb, lwdnbc=lwdnbc, lwdnbcln=lwdnbcln, &
         glw=glw, olr=olr, lwcf=lwcf, emiss=emiss, p8w=p8w, p3d=p3d, pi3d=pi3d, dz8w=dz8w, tsk=tsk, t3d=t3d, t8w=t8w, rho3d=rho3d, r=r_d, g=grav, &
         icloud=icloud, warm_rain=warm_rain, cldfra3d=cldfra3d, cldovrlp=cldovrlp, idcor=idcor, xlat=xlat, &
         lradius=re_cloud, iradius=re_ice, is_cammgmp_used=is_cammgmp_used, f_ice_phy=qi3d, f_rain_phy=qr3d, &
         xland=xland, xice=xice, snow=snow, qv3d=qv3d, qc3d=qc3d, qr3d=qr3d, qi3d=qi3d, qs3d=qs3d, qg3d=qg3d, &
         o3input=o3input, o33d=o33d, f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, &
         re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
         aer_ra_feedback=aer_ra_feedback, progn=progn, calc_clean_atm_diag=calc_clean_atm_diag, yr=yr, julian=julian, ghg_input=ghg_input, mp_physics=mp_physics, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, lwupflx=lwupflx, lwupflxc=lwupflxc, lwdnflx=lwdnflx, lwdnflxc=lwdnflxc)

    do k = 1, nz
       sw_heat(k) = rthratensw(1,k,1)
       lw_heat(k) = rthratenlw(1,k,1)
    enddo
    do k = 0, nz+1
       sw_up(k) = swupflx(1,k+1,1)
       sw_down(k) = swdnflx(1,k+1,1)
       lw_up(k) = lwupflx(1,k+1,1)
       lw_down(k) = lwdnflx(1,k+1,1)
    enddo

    sw_down(nz+1) = swdnt(1,1)
    sw_up(nz+1) = swupt(1,1)
    lw_down(nz+1) = lwdnt(1,1)
    lw_up(nz+1) = lwupt(1,1)

    net_surface = sw_down(0) - sw_up(0)
    net_top = sw_down(nz+1) - sw_up(nz+1)
    sw_col_abs = net_top - net_surface
    sw_sfc_abs = net_surface
    net_surface = lw_down(0) - lw_up(0)
    net_top = lw_down(nz) - lw_up(nz)
    lw_col_heat = net_top - net_surface
    lw_sfc_emit = stefan_boltzmann * max(0.0_rk, min(1.0_rk, surface_emissivity)) * max(surface_temperature, 120.0_rk)**4
  end subroutine run_wrf_rrtmg_drivers

  subroutine initialize_column_arrays(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                      temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                      t3d, t8w, p3d, p8w, pi3d, rho3d, dz8w, cldfra3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, &
                                      o33d, re_cloud, re_ice, re_snow, xlat, xlong, xland, xice, snow, tsk, albedo, emiss, &
                                      coszr, xcoszen, obscur, alswvisdir, alswvisdif, alswnirdir, alswnirdif, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(inout) :: t3d(:,:,:), t8w(:,:,:), p3d(:,:,:), p8w(:,:,:), pi3d(:,:,:), rho3d(:,:,:), dz8w(:,:,:)
    real(rk), intent(inout) :: cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), qi3d(:,:,:), qs3d(:,:,:), qg3d(:,:,:)
    real(rk), intent(inout) :: o33d(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
    real(rk), intent(inout) :: xlat(:,:), xlong(:,:), xland(:,:), xice(:,:), snow(:,:), tsk(:,:), albedo(:,:), emiss(:,:)
    real(rk), intent(inout) :: coszr(:,:), xcoszen(:,:), obscur(:,:), alswvisdir(:,:), alswvisdif(:,:), alswnirdir(:,:), alswnirdif(:,:)
    real(rk), intent(out) :: layer_mass_p(nz)
    integer :: k

    t3d = 0.0_rk; t8w = 0.0_rk; p3d = 0.0_rk; p8w = 0.0_rk; pi3d = 1.0_rk; rho3d = 0.0_rk; dz8w = 0.0_rk
    cldfra3d = 0.0_rk; qv3d = 0.0_rk; qc3d = 0.0_rk; qr3d = 0.0_rk; qi3d = 0.0_rk; qs3d = 0.0_rk; qg3d = 0.0_rk
    o33d = 0.0_rk
    re_cloud = 10.0e-6_rk
    re_ice = 30.0e-6_rk
    re_snow = 75.0e-6_rk

    xlat(1,1) = 28.3_rk
    xlong(1,1) = -16.5_rk
    xland(1,1) = 1.0_rk
    xice(1,1) = 0.0_rk
    snow(1,1) = 0.0_rk
    tsk(1,1) = surface_temperature
    albedo(1,1) = surface_albedo
    emiss(1,1) = surface_emissivity
    coszr(1,1) = coszen_scalar
    xcoszen(1,1) = coszen_scalar
    obscur(1,1) = 0.0_rk
    alswvisdir(1,1) = surface_albedo
    alswvisdif(1,1) = surface_albedo
    alswnirdir(1,1) = surface_albedo
    alswnirdif(1,1) = surface_albedo

    do k = 1, nz
       t3d(1,k,1) = temp(k)
       p3d(1,k,1) = press(k)
       rho3d(1,k,1) = rho(k)
       dz8w(1,k,1) = dz(k)
       qv3d(1,k,1) = max(qv(k), 1.0e-12_rk)
       qc3d(1,k,1) = max(qc(k), 0.0_rk)
       qi3d(1,k,1) = max(qi(k), 0.0_rk)
       qs3d(1,k,1) = max(qs(k), 0.0_rk)
       qg3d(1,k,1) = max(qg(k), 0.0_rk)
       cldfra3d(1,k,1) = min(1.0_rk, max(0.0_rk, cldfra(k)))
    enddo

    t3d(1,nz+1,1) = temp(nz)
    p3d(1,nz+1,1) = max(1.0_rk, press(nz) * 0.5_rk)
    rho3d(1,nz+1,1) = max(1.0e-6_rk, p3d(1,nz+1,1) / (rd_air * temp(nz) * (1.0_rk + rv_over_rd_minus_one * max(qv(nz), 0.0_rk))))
    dz8w(1,nz+1,1) = dz(nz)
    qv3d(1,nz+1,1) = max(qv(nz), 1.0e-12_rk)

    call fill_interface_profiles(nz, temp, press, t8w, p8w, layer_mass_p)
  end subroutine initialize_column_arrays

  subroutine fill_interface_profiles(nz, temp, press, t8w, p8w, layer_mass_p)
    integer, intent(in) :: nz
    real(rk), intent(in) :: temp(nz), press(nz)
    real(rk), intent(inout) :: t8w(:,:,:), p8w(:,:,:)
    real(rk), intent(out) :: layer_mass_p(nz)
    integer :: k
    real(rk) :: dp_bottom, dp_top

    if (nz == 1) then
       p8w(1,1,1) = press(1) + 50.0_rk
       p8w(1,2,1) = max(1.0_rk, press(1) - 50.0_rk)
       t8w(1,1,1) = temp(1)
       t8w(1,2,1) = temp(1)
    else
       dp_bottom = max(10.0_rk, press(1) - press(2))
       p8w(1,1,1) = press(1) + 0.5_rk * dp_bottom
       t8w(1,1,1) = temp(1)
       do k = 2, nz
          p8w(1,k,1) = 0.5_rk * (press(k-1) + press(k))
          t8w(1,k,1) = 0.5_rk * (temp(k-1) + temp(k))
       enddo
       dp_top = max(10.0_rk, press(nz-1) - press(nz))
       p8w(1,nz+1,1) = max(400.0_rk, press(nz) - 0.5_rk * dp_top)
       t8w(1,nz+1,1) = temp(nz)
    endif
    do k = 1, nz
       layer_mass_p(k) = max(1.0e-6_rk, (p8w(1,k,1) - p8w(1,k+1,1)) / gravity)
    enddo
  end subroutine fill_interface_profiles

end program wrf_rrtmg_harness

subroutine read_camgases_stub() bind(C, name="__module_ra_clwrf_support_MOD_read_camgases")
  use iso_c_binding
  implicit none
end subroutine read_camgases_stub

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(values, nbytes)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nbytes
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_bcast_integer(values, nitems)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_real(values, nitems)
  implicit none
  real, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_real

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(*,*) trim(message)
  stop 99
end subroutine wrf_error_fatal

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
end subroutine wrf_message
