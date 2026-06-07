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
  use module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad
  use module_ra_rrtmg_lw, only: rrtmg_lwinit, rrtmg_lwrad, inirad
  use parrrsw, only: sw_nbnd => nbndsw, sw_ngpt => ngptsw, sw_jpband => jpband, &
                     sw_jpb1 => jpb1, sw_mxmol => mxmol
  use parrrtm, only: lw_nbnd => nbndlw, lw_ngpt => ngptlw, lw_mxmol => mxmol, &
                     lw_maxxsec => maxxsec
  use rrtmg_sw_setcoef, only: setcoef_sw
  use rrtmg_sw_taumol, only: taumol_sw
  use rrtmg_sw_spcvmc, only: spcvmc_sw
  use mcica_subcol_gen_lw, only: mcica_subcol_lw
  use rrtmg_lw_cldprmc, only: cldprmc
  use rrtmg_lw_setcoef, only: setcoef
  use rrtmg_lw_taumol, only: taumol
  use rrtmg_lw_rtrnmc, only: rtrnmc
  use rrlw_con, only: lw_fluxfac => fluxfac
  use rrlw_tbl, only: lw_tblint => tblint, lw_bpade => bpade, lw_tau_tbl => tau_tbl, &
                      lw_exp_tbl => exp_tbl, lw_tfn_tbl => tfn_tbl
  use rrlw_wvn, only: lw_delwave => delwave, lw_ngb => ngb, lw_ngs => ngs
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
  real(rk), allocatable :: lw_planklay(:,:), lw_planklev(:,:), lw_plankbnd(:)
  real(rk), allocatable :: lw_taug(:,:,:), lw_fracs(:,:,:), lw_secdiff(:), lw_dplankup(:,:), lw_dplankdn(:,:)
  real(rk), allocatable :: lw_cldprmc_cldfmc(:,:,:), lw_cldprmc_taucmc(:,:,:)
  real(rk), allocatable :: lw_rtrnmc_pfracs(:,:,:), lw_rtrnmc_plansum(:,:), lw_rtrnmc_tfn_tbl_output(:,:,:)
  real(rk), allocatable :: lw_rtrnmc_zfd_per_gpoint(:,:,:), lw_rtrnmc_zfu_per_gpoint(:,:,:)
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
  allocate(lw_planklay(nz+1,lw_nbnd), lw_planklev(0:nz+1,lw_nbnd), lw_plankbnd(lw_nbnd))
  allocate(lw_taug(nz+1,lw_max_g,lw_nbnd), lw_fracs(nz+1,lw_max_g,lw_nbnd), lw_secdiff(lw_nbnd))
  allocate(lw_dplankup(nz+1,lw_nbnd), lw_dplankdn(nz+1,lw_nbnd))
  allocate(lw_cldprmc_cldfmc(nz+1,lw_max_g,lw_nbnd), lw_cldprmc_taucmc(nz+1,lw_max_g,lw_nbnd))
  allocate(lw_rtrnmc_pfracs(nz+1,lw_max_g,lw_nbnd), lw_rtrnmc_plansum(nz+1,lw_nbnd), lw_rtrnmc_tfn_tbl_output(nz+1,lw_max_g,lw_nbnd))
  allocate(lw_rtrnmc_zfd_per_gpoint(nz+2,lw_max_g,lw_nbnd), lw_rtrnmc_zfu_per_gpoint(nz+2,lw_max_g,lw_nbnd))
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
                                   sw_taug, sw_taur, sw_sfluxzen, lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                   lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn, &
                                   lw_cldprmc_cldfmc, lw_cldprmc_taucmc, lw_rtrnmc_pfracs, lw_rtrnmc_plansum, &
                                   lw_rtrnmc_tfn_tbl_output, lw_rtrnmc_zfd_per_gpoint, lw_rtrnmc_zfu_per_gpoint)

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
                                  sw_taug, sw_taur, sw_sfluxzen, lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                  lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn, &
                                  lw_cldprmc_cldfmc, lw_cldprmc_taucmc, lw_rtrnmc_pfracs, lw_rtrnmc_plansum, &
                                  lw_rtrnmc_tfn_tbl_output, lw_rtrnmc_zfd_per_gpoint, lw_rtrnmc_zfu_per_gpoint)

contains

  subroutine compute_intermediate_oracle(nz, surface_albedo, coszen_scalar, surface_temperature, surface_emissivity, &
                                         temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                                         sw_band_flux, lw_band_flux, sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                         sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                         sw_taug_band, sw_taur_band, sw_sflux_band, lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                         lw_plankbnd, lw_taug_band, lw_fracs_band, lw_secdiff, lw_dplankup, lw_dplankdn, &
                                         lw_cldprmc_cldfmc_band, lw_cldprmc_taucmc_band, lw_rtrnmc_pfracs_band, &
                                         lw_rtrnmc_plansum_band, lw_rtrnmc_tfn_tbl_output_band, &
                                         lw_rtrnmc_zfd_per_gpoint_band, lw_rtrnmc_zfu_per_gpoint_band)
    integer, intent(in) :: nz
    real(rk), intent(in) :: surface_albedo, coszen_scalar, surface_temperature, surface_emissivity
    real(rk), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(rk), intent(out) :: sw_band_flux(4,sw_nbnd), lw_band_flux(4,lw_nbnd)
    integer, intent(out) :: sw_jp(nz+1), sw_jt(nz+1), sw_jt1(nz+1), sw_indself(nz+1), sw_indfor(nz+1)
    integer, intent(out) :: lw_jp(nz+1), lw_jt(nz+1)
    real(rk), intent(out) :: sw_fac00(nz+1), sw_fac01(nz+1), sw_fac10(nz+1), sw_fac11(nz+1), sw_selffac(nz+1), sw_forfac(nz+1)
    real(rk), intent(out) :: sw_colmol(nz+1,6), sw_taug_band(nz+1,sw_max_g,sw_nbnd), sw_taur_band(nz+1,sw_max_g,sw_nbnd)
    real(rk), intent(out) :: sw_sflux_band(sw_max_g,sw_nbnd)
    real(rk), intent(out) :: lw_planklay(nz+1,lw_nbnd), lw_planklev(0:nz+1,lw_nbnd), lw_plankbnd(lw_nbnd)
    real(rk), intent(out) :: lw_taug_band(nz+1,lw_max_g,lw_nbnd), lw_fracs_band(nz+1,lw_max_g,lw_nbnd)
    real(rk), intent(out) :: lw_secdiff(lw_nbnd), lw_dplankup(nz+1,lw_nbnd), lw_dplankdn(nz+1,lw_nbnd)
    real(rk), intent(out) :: lw_cldprmc_cldfmc_band(nz+1,lw_max_g,lw_nbnd), lw_cldprmc_taucmc_band(nz+1,lw_max_g,lw_nbnd)
    real(rk), intent(out) :: lw_rtrnmc_pfracs_band(nz+1,lw_max_g,lw_nbnd), lw_rtrnmc_plansum_band(nz+1,lw_nbnd)
    real(rk), intent(out) :: lw_rtrnmc_tfn_tbl_output_band(nz+1,lw_max_g,lw_nbnd)
    real(rk), intent(out) :: lw_rtrnmc_zfd_per_gpoint_band(nz+2,lw_max_g,lw_nbnd), lw_rtrnmc_zfu_per_gpoint_band(nz+2,lw_max_g,lw_nbnd)

    integer, parameter :: sw_counts(sw_nbnd) = (/6,12,8,8,10,10,2,10,8,6,6,8,6,12/)
    integer, parameter :: lw_counts(lw_nbnd) = (/10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2/)
    integer :: nlay, k, b, g, start_g, laytrop, layswtch, laylow, ncbands
    integer :: icld_lw, irng_lw, permuteseed_lw
    real(rk) :: pavel(nz+1), tavel(nz+1), tavel_lw(nz+1), pz(0:nz+1), tz(0:nz+1), tz_lw(0:nz+1)
    real(rk) :: coldry(nz+1), wkl(sw_mxmol,nz+1), wkl_lw(lw_mxmol,nz+1), wx(lw_maxxsec,nz+1)
    real(rk) :: wbroad(nz+1), h2ovmr(nz+1), amm, summol, summol_lw, amttl, wvttl, wvsh, pwvcm
    real(rk) :: o3mmr_lw(nz+1), o3vmr_lw(nz+1), plev_o3(1:nz+2)
    real(rk) :: co2mult(nz+1), colh2o(nz+1), colco2(nz+1), colo3(nz+1), coln2o(nz+1), colch4(nz+1), colo2(nz+1), colmol(nz+1)
    real(rk) :: selffrac(nz+1), forfrac(nz+1)
    real(rk) :: taug_sw(nz+1,sw_ngpt), taur_sw(nz+1,sw_ngpt), sflux_sw(sw_ngpt)
    real(rk) :: albdif(sw_nbnd), albdir(sw_nbnd), adjflux(sw_jpband)
    real(rk) :: cldf_sw(nz+1,sw_ngpt), taucmc_sw(nz+1,sw_ngpt), taormc_sw(nz+1,sw_ngpt), asy_sw(nz+1,sw_ngpt), omg_sw(nz+1,sw_ngpt)
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
    real(rk) :: ciwpmc_lw(lw_ngpt,nz+1), clwpmc_lw(lw_ngpt,nz+1), cswpmc_lw(lw_ngpt,nz+1)
    real(rk) :: reicmc_lw(nz+1), relqmc_lw(nz+1), resnmc_lw(nz+1)
    real(rk) :: play_lw(1,nz+1), hgt_lw(1,nz+1), cldfrac_lw(1,nz+1)
    real(rk) :: ciwp_lw(1,nz+1), clwp_lw(1,nz+1), cswp_lw(1,nz+1), tauc_lw(lw_nbnd,1,nz+1)
    real(rk) :: rei_lw(1,nz+1), rel_lw(1,nz+1), res_lw(1,nz+1)
    real(rk) :: cldfmcl_lw(lw_ngpt,1,nz+1), ciwpmcl_lw(lw_ngpt,1,nz+1), clwpmcl_lw(lw_ngpt,1,nz+1)
    real(rk) :: cswpmcl_lw(lw_ngpt,1,nz+1), taucmcl_lw(lw_ngpt,1,nz+1)
    real(rk) :: reicmcl_lw(1,nz+1), relqmcl_lw(1,nz+1), resnmcl_lw(1,nz+1)
    real(rk) :: layer_mass_ext(nz+1), dzsum, cloud_safe, snow_mass_factor
    real(rk) :: totuflux(0:nz+1), totdflux(0:nz+1), fnet(0:nz+1), htr(0:nz+1), totuclfl(0:nz+1), totdclfl(0:nz+1), fnetc(0:nz+1), htrc(0:nz+1)

    nlay = nz + 1
    call build_rrtmg_profile(nz, temp, press, qv, pavel, tavel, pz, tz, coldry, h2ovmr)
    call adjust_lw_buffer_temperature(nz, pz, tavel, tz, tavel_lw, tz_lw)
    do k = 1, nlay + 1
       plev_o3(k) = pz(k-1)
    enddo
    o3mmr_lw = 0.0_rk
    call inirad(o3mmr_lw, plev_o3, 1, nlay - 1)
    do k = 1, nlay
       o3vmr_lw(k) = o3mmr_lw(k) * o3_mmr_to_vmr
    enddo
    wkl = 0.0_rk
    wkl_lw = 0.0_rk
    wx = 0.0_rk
    amttl = 0.0_rk
    wvttl = 0.0_rk
    do k = 1, nlay
       wkl(1,k) = coldry(k) * h2ovmr(k)
       wkl(2,k) = coldry(k) * co2_vmr_default
       ! KI-6: the SW intermediate oracle must use the SAME WRF O3DATA climatology
       ! ozone (o3input=0 path in RRTMG_SWRAD: o3vmr = o3mmr*amdo) that the real WRF
       ! shortwave driver and the JAX SW kernel both use. The previous constant
       ! o3_vmr_default (8e-8) was a harness simplification that left the O3-dependent
       ! UV bands (9,10,12,13) disagreeing with the climatology JAX taug — most at the
       ! extra model-top layer where the climatology ozone column is ~28x the constant.
       wkl(3,k) = coldry(k) * o3vmr_lw(k)
       wkl(4,k) = coldry(k) * n2o_vmr_default
       wkl(6,k) = coldry(k) * ch4_vmr_default
       wkl(7,k) = coldry(k) * o2_vmr_default
       wkl_lw(:,k) = wkl(:,k)
       wkl_lw(3,k) = coldry(k) * o3vmr_lw(k)
       summol = co2_vmr_default + o3_vmr_default + n2o_vmr_default + ch4_vmr_default + o2_vmr_default
       summol_lw = co2_vmr_default + o3vmr_lw(k) + n2o_vmr_default + ch4_vmr_default + o2_vmr_default
       wbroad(k) = coldry(k) * max(0.0_rk, 1.0_rk - summol_lw)
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
    cldf_sw = 0.0_rk
    taucmc_sw = 0.0_rk
    taormc_sw = 0.0_rk
    asy_sw = 0.0_rk
    omg_sw = 1.0_rk
    taua_sw = 0.0_rk
    asya_sw = 0.0_rk
    omga_sw = 1.0_rk
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
    call setcoef(nlay, 1, pavel, tavel_lw, tz_lw, surface_temperature, semiss, coldry, wkl_lw, wbroad, &
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
    ciwpmc_lw = 0.0_rk
    clwpmc_lw = 0.0_rk
    cswpmc_lw = 0.0_rk
    reicmc_lw = 10.0_rk
    relqmc_lw = 10.0_rk
    resnmc_lw = 10.0_rk
    play_lw = 0.0_rk
    hgt_lw = 0.0_rk
    cldfrac_lw = 0.0_rk
    ciwp_lw = 0.0_rk
    clwp_lw = 0.0_rk
    cswp_lw = 0.0_rk
    tauc_lw = 0.0_rk
    rei_lw = 10.0_rk
    rel_lw = 10.0_rk
    res_lw = 10.0_rk
    do k = 1, nlay
       layer_mass_ext(k) = max(1.0e-6_rk, (pz(k-1) - pz(k)) * 100.0_rk / gravity)
       play_lw(1,k) = pavel(k)
    enddo
    dzsum = 0.0_rk
    do k = 1, nz
       hgt_lw(1,k) = dzsum + 0.5_rk * dz(k)
       dzsum = dzsum + dz(k)
       cldfrac_lw(1,k) = min(1.0_rk, max(0.0_rk, cldfra(k)))
       cloud_safe = max(0.01_rk, cldfrac_lw(1,k))
       clwp_lw(1,k) = max(qc(k), 0.0_rk) * layer_mass_ext(k) * 1000.0_rk / cloud_safe
       ciwp_lw(1,k) = max(qi(k), 0.0_rk) * layer_mass_ext(k) * 1000.0_rk / cloud_safe
       snow_mass_factor = 0.99_rk
       cswp_lw(1,k) = max(qs(k), 0.0_rk) * snow_mass_factor * layer_mass_ext(k) * 1000.0_rk / cloud_safe
       rel_lw(1,k) = 10.0_rk
       rei_lw(1,k) = 30.0_rk
       res_lw(1,k) = 75.0_rk
    enddo
    hgt_lw(1,nlay) = dzsum + 0.5_rk * dz(nz)
    icld_lw = 1
    irng_lw = 0
    permuteseed_lw = 150
    call mcica_subcol_lw(1, 1, nlay, icld_lw, permuteseed_lw, irng_lw, play_lw, &
                         cldfrac_lw, ciwp_lw, clwp_lw, cswp_lw, rei_lw, rel_lw, res_lw, tauc_lw, &
                         hgt_lw, 0, 142, 28.3_rk, cldfmcl_lw, ciwpmcl_lw, clwpmcl_lw, cswpmcl_lw, &
                         reicmcl_lw, relqmcl_lw, resnmcl_lw, taucmcl_lw)
    do k = 1, nlay
       do g = 1, lw_ngpt
          cldf_lw(g,k) = cldfmcl_lw(g,1,k)
          ciwpmc_lw(g,k) = ciwpmcl_lw(g,1,k)
          clwpmc_lw(g,k) = clwpmcl_lw(g,1,k)
          cswpmc_lw(g,k) = cswpmcl_lw(g,1,k)
          taucmc_lw(g,k) = taucmcl_lw(g,1,k)
       enddo
       reicmc_lw(k) = reicmcl_lw(1,k)
       relqmc_lw(k) = relqmcl_lw(1,k)
       resnmc_lw(k) = resnmcl_lw(1,k)
    enddo
    ncbands = lw_nbnd
    call cldprmc(nlay, 5, 5, 1, cldf_lw, ciwpmc_lw, clwpmc_lw, cswpmc_lw, reicmc_lw, relqmc_lw, resnmc_lw, ncbands, taucmc_lw)
    lw_cldprmc_cldfmc_band = 0.0_rk
    lw_cldprmc_taucmc_band = 0.0_rk
    lw_rtrnmc_pfracs_band = 0.0_rk
    lw_rtrnmc_plansum_band = 0.0_rk
    lw_rtrnmc_tfn_tbl_output_band = 0.0_rk
    lw_rtrnmc_zfd_per_gpoint_band = 0.0_rk
    lw_rtrnmc_zfu_per_gpoint_band = 0.0_rk
    start_g = 1
    do b = 1, lw_nbnd
       do g = 1, lw_counts(b)
          lw_cldprmc_cldfmc_band(:,g,b) = cldf_lw(start_g+g-1,:)
          lw_cldprmc_taucmc_band(:,g,b) = taucmc_lw(start_g+g-1,:)
          lw_rtrnmc_pfracs_band(:,g,b) = fracs_lw(:,start_g+g-1)
       enddo
       lw_rtrnmc_plansum_band(:,b) = sum(lw_rtrnmc_pfracs_band(:,:,b), dim=2) * lw_planklay(:,b)
       start_g = start_g + lw_counts(b)
    enddo
    do b = 1, lw_nbnd
       call capture_lw_rtrnmc_band(nlay, b, pz, semiss, cldf_lw, taucmc_lw, &
                   lw_planklay, lw_planklev, lw_plankbnd, pwvcm, fracs_lw, taut_lw, &
                   lw_rtrnmc_tfn_tbl_output_band(:,:,b), lw_rtrnmc_zfd_per_gpoint_band(:,:,b), &
                   lw_rtrnmc_zfu_per_gpoint_band(:,:,b))
       call rtrnmc(nlay, b, b, 1, pz, semiss, ncbands, cldf_lw, taucmc_lw, &
                   lw_planklay, lw_planklev, lw_plankbnd, pwvcm, fracs_lw, taut_lw, &
                   totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, htrc)
       lw_band_flux(1,b) = totuflux(nlay)
       lw_band_flux(2,b) = totdflux(nlay)
       lw_band_flux(3,b) = totuflux(0)
       lw_band_flux(4,b) = totdflux(0)
    enddo
  end subroutine compute_intermediate_oracle

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

  subroutine adjust_lw_buffer_temperature(nz, pz, tavel, tz, tavel_lw, tz_lw)
    integer, intent(in) :: nz
    real(rk), intent(in) :: pz(0:nz+1), tavel(nz+1), tz(0:nz+1)
    real(rk), intent(out) :: tavel_lw(nz+1), tz_lw(0:nz+1)
    integer, parameter :: nproflevs = 60
    real(rk), parameter :: pprof(nproflevs) = (/ &
       1000.00_rk,855.47_rk,731.82_rk,626.05_rk,535.57_rk,458.16_rk, &
       391.94_rk,335.29_rk,286.83_rk,245.38_rk,209.91_rk,179.57_rk, &
       153.62_rk,131.41_rk,112.42_rk,96.17_rk,82.27_rk,70.38_rk, &
       60.21_rk,51.51_rk,44.06_rk,37.69_rk,32.25_rk,27.59_rk, &
       23.60_rk,20.19_rk,17.27_rk,14.77_rk,12.64_rk,10.81_rk, &
       9.25_rk,7.91_rk,6.77_rk,5.79_rk,4.95_rk,4.24_rk, &
       3.63_rk,3.10_rk,2.65_rk,2.27_rk,1.94_rk,1.66_rk, &
       1.42_rk,1.22_rk,1.04_rk,0.89_rk,0.76_rk,0.65_rk, &
       0.56_rk,0.48_rk,0.41_rk,0.35_rk,0.30_rk,0.26_rk, &
       0.22_rk,0.19_rk,0.16_rk,0.14_rk,0.12_rk,0.10_rk /)
    real(rk), parameter :: tprof(nproflevs) = (/ &
       286.96_rk,281.07_rk,275.16_rk,268.11_rk,260.56_rk,253.02_rk, &
       245.62_rk,238.41_rk,231.57_rk,225.91_rk,221.72_rk,217.79_rk, &
       215.06_rk,212.74_rk,210.25_rk,210.16_rk,210.69_rk,212.14_rk, &
       213.74_rk,215.37_rk,216.82_rk,217.94_rk,219.03_rk,220.18_rk, &
       221.37_rk,222.64_rk,224.16_rk,225.88_rk,227.63_rk,229.51_rk, &
       231.50_rk,233.73_rk,236.18_rk,238.78_rk,241.60_rk,244.44_rk, &
       247.35_rk,250.33_rk,253.32_rk,256.30_rk,259.22_rk,262.12_rk, &
       264.80_rk,266.50_rk,267.59_rk,268.44_rk,268.69_rk,267.76_rk, &
       266.13_rk,263.96_rk,261.54_rk,258.93_rk,256.15_rk,253.23_rk, &
       249.89_rk,246.67_rk,243.48_rk,240.25_rk,236.66_rk,233.86_rk /)
    integer :: l, ll, klev, nlay
    real(rk) :: varint(nz+2), plev, wght, vark, vark1

    nlay = nz + 1
    tavel_lw = tavel
    tz_lw = tz
    do l = 1, nlay + 1
       plev = pz(l-1)
       if (l == nlay + 1) plev = 0.0_rk
       if (pprof(nproflevs) .lt. plev) then
          klev = nproflevs
          do ll = 2, nproflevs
             if (pprof(ll) .lt. plev) then
                klev = ll - 1
                exit
             endif
          enddo
       else
          klev = nproflevs
       endif
       if (klev .ne. nproflevs) then
          vark = tprof(klev)
          vark1 = tprof(klev+1)
          wght = (plev - pprof(klev)) / (pprof(klev+1) - pprof(klev))
       else
          vark = tprof(klev)
          vark1 = tprof(klev)
          wght = 0.0_rk
       endif
       varint(l) = wght * (vark1 - vark) + vark
    enddo

    do l = nz + 1, nlay + 1
       tz_lw(l-1) = varint(l) + (tz(nz-1) - varint(nz))
       tavel_lw(l-1) = 0.5_rk * (tz_lw(l-1) + tz_lw(l-2))
    enddo
  end subroutine adjust_lw_buffer_temperature

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

  subroutine capture_lw_rtrnmc_band(nlayers, iband, pz, semiss, cldfmc, taucmc, &
                                    planklay, planklev, plankbnd, pwvcm, fracs, taut, &
                                    tfn_out, zfd_out, zfu_out)
    integer, intent(in) :: nlayers, iband
    real(rk), intent(in) :: pz(0:), semiss(:), cldfmc(:,:), taucmc(:,:)
    real(rk), intent(in) :: planklay(:,:), planklev(0:,:), plankbnd(:), pwvcm, fracs(:,:), taut(:,:)
    real(rk), intent(out) :: tfn_out(:,:), zfd_out(:,:), zfu_out(:,:)

    real(rk) :: secdiff(lw_nbnd), odcld(nlayers,lw_ngpt), abscld(nlayers,lw_ngpt), efclfrac(nlayers,lw_ngpt)
    real(rk) :: atot(nlayers), atrans(nlayers), bbugas(nlayers), bbutot(nlayers)
    real(rk) :: transcld, radld, radclrd, plfrac, blay, dplankup, dplankdn
    real(rk) :: odepth, odtot, odepth_rec, odtot_rec, gassrc
    real(rk) :: tblind, tfactot, bbd, bbdtot, tfacgas, transc, tausfac
    real(rk) :: rad0, reflect, radlu, radclru, scale
    integer :: icldlyr(nlayers), iclddn
    integer :: ib, lay, lev, ig, igc, g_local, start_g, end_g
    integer :: ittot, itgas, itr

    call lw_diffusivity(pwvcm, secdiff)
    odcld = 0.0_rk
    abscld = 0.0_rk
    efclfrac = 0.0_rk
    icldlyr = 0
    do lay = 1, nlayers
       do ig = 1, lw_ngpt
          if (cldfmc(ig,lay) .eq. 1.0_rk) then
             ib = lw_ngb(ig)
             odcld(lay,ig) = secdiff(ib) * taucmc(ig,lay)
             transcld = exp(-odcld(lay,ig))
             abscld(lay,ig) = 1.0_rk - transcld
             efclfrac(lay,ig) = abscld(lay,ig) * cldfmc(ig,lay)
             icldlyr(lay) = 1
          endif
       enddo
    enddo

    tfn_out = 0.0_rk
    zfd_out = 0.0_rk
    zfu_out = 0.0_rk
    start_g = 1
    if (iband .ge. 2) start_g = lw_ngs(iband-1) + 1
    end_g = lw_ngs(iband)
    scale = 0.5_rk * lw_delwave(iband) * lw_fluxfac

    do igc = start_g, end_g
       g_local = igc - start_g + 1
       radld = 0.0_rk
       radclrd = 0.0_rk
       iclddn = 0
       zfd_out(nlayers+1,g_local) = 0.0_rk

       do lev = nlayers, 1, -1
          plfrac = fracs(lev,igc)
          blay = planklay(lev,iband)
          dplankup = planklev(lev,iband) - blay
          dplankdn = planklev(lev-1,iband) - blay
          odepth = secdiff(iband) * taut(lev,igc)
          if (odepth .lt. 0.0_rk) odepth = 0.0_rk
          if (icldlyr(lev) .eq. 1) then
             iclddn = 1
             odtot = odepth + odcld(lev,igc)
             if (odtot .lt. 0.06_rk) then
                atrans(lev) = odepth - 0.5_rk * odepth * odepth
                odepth_rec = odepth / 6.0_rk
                tfn_out(lev,g_local) = odepth_rec
                gassrc = plfrac * (blay + dplankdn * odepth_rec) * atrans(lev)
                atot(lev) = odtot - 0.5_rk * odtot * odtot
                odtot_rec = odtot / 6.0_rk
                bbdtot = plfrac * (blay + dplankdn * odtot_rec)
                bbd = plfrac * (blay + dplankdn * odepth_rec)
                radld = radld - radld * (atrans(lev) + efclfrac(lev,igc) * (1.0_rk - atrans(lev))) + &
                        gassrc + cldfmc(igc,lev) * (bbdtot * atot(lev) - gassrc)
                bbugas(lev) = plfrac * (blay + dplankup * odepth_rec)
                bbutot(lev) = plfrac * (blay + dplankup * odtot_rec)
             elseif (odepth .le. 0.06_rk) then
                atrans(lev) = odepth - 0.5_rk * odepth * odepth
                odepth_rec = odepth / 6.0_rk
                tfn_out(lev,g_local) = odepth_rec
                gassrc = plfrac * (blay + dplankdn * odepth_rec) * atrans(lev)
                odtot = odepth + odcld(lev,igc)
                tblind = odtot / (lw_bpade + odtot)
                ittot = lw_tblint * tblind + 0.5_rk
                tfactot = lw_tfn_tbl(ittot)
                bbdtot = plfrac * (blay + tfactot * dplankdn)
                bbd = plfrac * (blay + dplankdn * odepth_rec)
                atot(lev) = 1.0_rk - lw_exp_tbl(ittot)
                radld = radld - radld * (atrans(lev) + efclfrac(lev,igc) * (1.0_rk - atrans(lev))) + &
                        gassrc + cldfmc(igc,lev) * (bbdtot * atot(lev) - gassrc)
                bbugas(lev) = plfrac * (blay + dplankup * odepth_rec)
                bbutot(lev) = plfrac * (blay + tfactot * dplankup)
             else
                tblind = odepth / (lw_bpade + odepth)
                itgas = lw_tblint * tblind + 0.5_rk
                odepth = lw_tau_tbl(itgas)
                atrans(lev) = 1.0_rk - lw_exp_tbl(itgas)
                tfacgas = lw_tfn_tbl(itgas)
                tfn_out(lev,g_local) = tfacgas
                gassrc = atrans(lev) * plfrac * (blay + tfacgas * dplankdn)
                odtot = odepth + odcld(lev,igc)
                tblind = odtot / (lw_bpade + odtot)
                ittot = lw_tblint * tblind + 0.5_rk
                tfactot = lw_tfn_tbl(ittot)
                bbdtot = plfrac * (blay + tfactot * dplankdn)
                bbd = plfrac * (blay + tfacgas * dplankdn)
                atot(lev) = 1.0_rk - lw_exp_tbl(ittot)
                radld = radld - radld * (atrans(lev) + efclfrac(lev,igc) * (1.0_rk - atrans(lev))) + &
                        gassrc + cldfmc(igc,lev) * (bbdtot * atot(lev) - gassrc)
                bbugas(lev) = plfrac * (blay + tfacgas * dplankup)
                bbutot(lev) = plfrac * (blay + tfactot * dplankup)
             endif
          else
             if (odepth .le. 0.06_rk) then
                atrans(lev) = odepth - 0.5_rk * odepth * odepth
                odepth = odepth / 6.0_rk
                tfn_out(lev,g_local) = odepth
                bbd = plfrac * (blay + dplankdn * odepth)
                bbugas(lev) = plfrac * (blay + dplankup * odepth)
             else
                tblind = odepth / (lw_bpade + odepth)
                itr = lw_tblint * tblind + 0.5_rk
                transc = lw_exp_tbl(itr)
                atrans(lev) = 1.0_rk - transc
                tausfac = lw_tfn_tbl(itr)
                tfn_out(lev,g_local) = tausfac
                bbd = plfrac * (blay + tausfac * dplankdn)
                bbugas(lev) = plfrac * (blay + tausfac * dplankup)
             endif
             atot(lev) = atrans(lev)
             bbutot(lev) = bbugas(lev)
             radld = radld + (bbd - radld) * atrans(lev)
          endif

          zfd_out(lev,g_local) = radld * scale
          if (iclddn .eq. 1) then
             radclrd = radclrd + (bbd - radclrd) * atrans(lev)
          else
             radclrd = radld
          endif
       enddo

       rad0 = fracs(1,igc) * plankbnd(iband)
       reflect = 1.0_rk - semiss(iband)
       radlu = rad0 + reflect * radld
       radclru = rad0 + reflect * radclrd
       zfu_out(1,g_local) = radlu * scale

       do lev = 1, nlayers
          if (icldlyr(lev) .eq. 1) then
             gassrc = bbugas(lev) * atrans(lev)
             radlu = radlu - radlu * (atrans(lev) + efclfrac(lev,igc) * (1.0_rk - atrans(lev))) + &
                     gassrc + cldfmc(igc,lev) * (bbutot(lev) * atot(lev) - gassrc)
          else
             radlu = radlu + (bbugas(lev) - radlu) * atrans(lev)
          endif
          zfu_out(lev+1,g_local) = radlu * scale
          if (iclddn .eq. 1) then
             radclru = radclru + (bbugas(lev) - radclru) * atrans(lev)
          else
             radclru = radlu
          endif
       enddo
    enddo
  end subroutine capture_lw_rtrnmc_band

  subroutine append_intermediate_oracle(output_path, nlay_sw, nlay_lw, sw_band_flux, lw_band_flux, &
                                        sw_jp, sw_jt, sw_jt1, sw_indself, sw_indfor, &
                                        sw_fac00, sw_fac01, sw_fac10, sw_fac11, sw_selffac, sw_forfac, sw_colmol, &
                                        sw_taug, sw_taur, sw_sfluxzen, lw_jp, lw_jt, lw_planklay, lw_planklev, &
                                        lw_plankbnd, lw_taug, lw_fracs, lw_secdiff, lw_dplankup, lw_dplankdn, &
                                        lw_cldprmc_cldfmc, lw_cldprmc_taucmc, lw_rtrnmc_pfracs, lw_rtrnmc_plansum, &
                                        lw_rtrnmc_tfn_tbl_output, lw_rtrnmc_zfd_per_gpoint, lw_rtrnmc_zfu_per_gpoint)
    character(len=*), intent(in) :: output_path
    integer, intent(in) :: nlay_sw, nlay_lw
    real(rk), intent(in) :: sw_band_flux(4,sw_nbnd), lw_band_flux(4,lw_nbnd)
    integer, intent(in) :: sw_jp(nlay_sw), sw_jt(nlay_sw), sw_jt1(nlay_sw), sw_indself(nlay_sw), sw_indfor(nlay_sw)
    integer, intent(in) :: lw_jp(nlay_lw), lw_jt(nlay_lw)
    real(rk), intent(in) :: sw_fac00(nlay_sw), sw_fac01(nlay_sw), sw_fac10(nlay_sw), sw_fac11(nlay_sw), sw_selffac(nlay_sw), sw_forfac(nlay_sw)
    real(rk), intent(in) :: sw_colmol(nlay_sw,6), sw_taug(nlay_sw,sw_max_g,sw_nbnd), sw_taur(nlay_sw,sw_max_g,sw_nbnd), sw_sfluxzen(sw_max_g,sw_nbnd)
    real(rk), intent(in) :: lw_planklay(nlay_lw,lw_nbnd), lw_planklev(0:nlay_lw,lw_nbnd), lw_plankbnd(lw_nbnd)
    real(rk), intent(in) :: lw_taug(nlay_lw,lw_max_g,lw_nbnd), lw_fracs(nlay_lw,lw_max_g,lw_nbnd), lw_secdiff(lw_nbnd)
    real(rk), intent(in) :: lw_dplankup(nlay_lw,lw_nbnd), lw_dplankdn(nlay_lw,lw_nbnd)
    real(rk), intent(in) :: lw_cldprmc_cldfmc(nlay_lw,lw_max_g,lw_nbnd), lw_cldprmc_taucmc(nlay_lw,lw_max_g,lw_nbnd)
    real(rk), intent(in) :: lw_rtrnmc_pfracs(nlay_lw,lw_max_g,lw_nbnd), lw_rtrnmc_plansum(nlay_lw,lw_nbnd)
    real(rk), intent(in) :: lw_rtrnmc_tfn_tbl_output(nlay_lw,lw_max_g,lw_nbnd)
    real(rk), intent(in) :: lw_rtrnmc_zfd_per_gpoint(nlay_lw+1,lw_max_g,lw_nbnd), lw_rtrnmc_zfu_per_gpoint(nlay_lw+1,lw_max_g,lw_nbnd)
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
    write(21) lw_cldprmc_cldfmc, lw_cldprmc_taucmc
    write(21) lw_rtrnmc_pfracs, lw_rtrnmc_plansum, lw_rtrnmc_tfn_tbl_output, lw_rtrnmc_zfd_per_gpoint, lw_rtrnmc_zfu_per_gpoint
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
