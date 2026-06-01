! Standalone pristine-WRF Noah-MP snow oracle (Sprint S3 parity).
!
! The snow subroutines in snow_routines.inc are transcribed VERBATIM from
!   /home/enric/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F
! (SNOWWATER/SNOWFALL/COMBINE/DIVIDE/COMBO/COMPACT/SNOWH2O/SNOW_AGE/
!  SNOWALB_CLASS), with the parameter derived-type reduced to the snow subset
! and constants pinned to the WRF PARAMETER values (:204-220).  No physics is
! altered.  A driver runs deterministic single-column scenarios across
! accumulation / compaction / melt / sublimation and prints every state field
! so the JAX port can be checked field-for-field.
!
! Build (wrfbuild conda env):
!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 snow_oracle.f90 -o snow_oracle

module noahmp_snow_oracle
  implicit none

  integer, parameter :: NSNOW = 3
  integer, parameter :: NSOIL = 4

  real, parameter :: GRAV   = 9.80616
  real, parameter :: TFRZ   = 273.16
  real, parameter :: HFUS   = 0.3336E06
  real, parameter :: CWAT   = 4.188E06
  real, parameter :: CICE   = 2.094E06
  real, parameter :: DENH2O = 1000.0
  real, parameter :: DENICE = 917.0

  type noahmp_parameters
    real :: SSI          = 0.03
    real :: SNOW_RET_FAC = 5.e-5
    real :: SWEMX        = 1.00
    real :: TAU0         = 1.e6
    real :: GRAIN_GROWTH = 5000.
    real :: EXTRA_GROWTH = 10.
    real :: DIRT_SOOT    = 0.3
  end type noahmp_parameters

contains

  include 'snow_routines.inc'

  subroutine setup_scenario(s, isnow, snowh, sneqv, sneqvo, tauss, albold, tg, &
       sfctmp, dt, snice, snliq, stc, zsnso, dzsnso, sh2o, sice, &
       ficeold, qsnow, snowhin, qsnfro, qsnsub, qrain, imelt, cosz, bdfall)
    integer, intent(in) :: s
    integer, intent(out) :: isnow
    real, intent(out) :: snowh, sneqv, sneqvo, tauss, albold, tg, sfctmp, dt, cosz, bdfall
    real, intent(out) :: snice(-NSNOW+1:0), snliq(-NSNOW+1:0)
    real, intent(out) :: stc(-NSNOW+1:NSOIL), zsnso(-NSNOW+1:NSOIL), dzsnso(-NSNOW+1:NSOIL)
    real, intent(out) :: sh2o(1:NSOIL), sice(1:NSOIL)
    real, intent(out) :: ficeold(-NSNOW+1:0), qsnow, snowhin, qsnfro, qsnsub, qrain
    integer, intent(out) :: imelt(-NSNOW+1:0)
    real :: zsoil(1:NSOIL)
    integer :: iz

    zsoil = (/ -0.1, -0.4, -1.0, -2.0 /)
    ! defaults: zero snow, cold soil
    isnow = 0; snowh = 0.0; sneqv = 0.0; sneqvo = 0.0; tauss = 0.0; albold = 0.65
    tg = 270.0; sfctmp = 270.0; dt = 1800.0; cosz = 0.5
    snice = 0.0; snliq = 0.0; stc = 0.0; snice = 0.0; snliq = 0.0
    do iz = 1, NSOIL
      stc(iz) = 280.0
    end do
    dzsnso = 0.0
    dzsnso(1) = zsoil(1)
    do iz = 2, NSOIL
      dzsnso(iz) = zsoil(iz) - zsoil(iz-1)
    end do
    zsnso = 0.0
    zsnso(1) = dzsnso(1)
    do iz = 2, NSOIL
      zsnso(iz) = zsnso(iz-1) + dzsnso(iz)
    end do
    do iz = 1, NSOIL
      dzsnso(iz) = -dzsnso(iz)
    end do
    sh2o = 0.25; sice = 0.05
    ficeold = 0.0; qsnow = 0.0; qsnfro = 0.0; qsnsub = 0.0; qrain = 0.0
    imelt = 0
    bdfall = min(120.0, 67.92 + 51.25*exp((sfctmp - TFRZ)/2.59))
    snowhin = 0.0

    select case (s)
    case (1)   ! zero-snow no-op (the common Canary case)
      qsnow = 0.0
    case (2)   ! light snowfall onto bare ground -> shallow bulk (no layer)
      sfctmp = 268.0; qsnow = 0.005   ! mm/s -> 9 mm over 1800s, < 0.025 m depth
      bdfall = min(120.0, 67.92 + 51.25*exp((sfctmp - TFRZ)/2.59))
      snowhin = qsnow/bdfall
    case (3)   ! heavier snowfall onto bare -> creates first layer (ISNOW=-1)
      sfctmp = 265.0; qsnow = 0.02    ! 36 mm over 1800s
      bdfall = min(120.0, 67.92 + 51.25*exp((sfctmp - TFRZ)/2.59))
      snowhin = qsnow/bdfall
    case (4)   ! existing single layer + snowfall, cold, compaction
      isnow = -1; snowh = 0.30; sneqv = 30.0
      snice(0) = 30.0; snliq(0) = 0.0; stc(0) = 263.0
      dzsnso(0) = 0.30
      sfctmp = 264.0; qsnow = 0.01
      bdfall = min(120.0, 67.92 + 51.25*exp((sfctmp - TFRZ)/2.59))
      snowhin = qsnow/bdfall
    case (5)   ! deep single layer -> DIVIDE into 2 layers (dz>0.05 thick, heavy)
      isnow = -1; snowh = 0.60; sneqv = 120.0
      snice(0) = 120.0; snliq(0) = 0.0; stc(0) = 268.0
      dzsnso(0) = 0.60
      sfctmp = 266.0; qsnow = 0.0
    case (6)   ! two layers, melt phase change -> COMPACT melt + liquid
      isnow = -2; snowh = 0.40; sneqv = 90.0
      snice(-1) = 40.0; snliq(-1) = 2.0; stc(-1) = 272.0
      snice(0)  = 50.0; snliq(0)  = 5.0; stc(0)  = 273.0
      dzsnso(-1) = 0.18; dzsnso(0) = 0.22
      ficeold(-1) = 0.95; ficeold(0) = 0.90
      imelt(-1) = 1; imelt(0) = 1
      sfctmp = 274.0; tg = 273.16; qrain = 0.002
    case (7)   ! three layers, mixed
      isnow = -3; snowh = 0.80; sneqv = 200.0
      snice(-2) = 60.0; snliq(-2) = 1.0; stc(-2) = 260.0
      snice(-1) = 70.0; snliq(-1) = 3.0; stc(-1) = 265.0
      snice(0)  = 65.0; snliq(0)  = 8.0; stc(0)  = 272.0
      dzsnso(-2) = 0.30; dzsnso(-1) = 0.30; dzsnso(0) = 0.20
      ficeold(-2) = 0.98; ficeold(-1) = 0.96; ficeold(0) = 0.89
      sfctmp = 270.0; tg = 270.0; qrain = 0.001
    case (8)   ! thin top layer triggers COMBINE collapse (SNICE<=0.1 surface)
      isnow = -2; snowh = 0.20; sneqv = 30.0
      snice(-1) = 30.0; snliq(-1) = 1.0; stc(-1) = 266.0
      snice(0)  = 0.05; snliq(0)  = 0.5; stc(0)  = 271.0
      dzsnso(-1) = 0.19; dzsnso(0) = 0.01
      sfctmp = 268.0
    case (9)   ! shallow snow sublimation to zero
      isnow = 0; snowh = 0.01; sneqv = 5.0
      sfctmp = 263.0; qsnsub = 0.004; tg = 262.0
    case (10)  ! single layer, heavy sublimation -> WGDIF<1e-6 -> COMBINE
      isnow = -1; snowh = 0.05; sneqv = 4.0
      snice(0) = 4.0; snliq(0) = 0.2; stc(0) = 268.0
      dzsnso(0) = 0.05
      sfctmp = 266.0; qsnsub = 0.003
    case (11)  ! frost onto existing shallow snow (no layer)
      isnow = 0; snowh = 0.02; sneqv = 8.0
      sfctmp = 264.0; qsnfro = 0.002; tg = 263.0
    case (12)  ! warm fresh snowfall -> albedo refresh (cosz>0)
      isnow = -1; snowh = 0.10; sneqv = 12.0
      snice(0) = 12.0; snliq(0) = 0.0; stc(0) = 271.0
      dzsnso(0) = 0.10
      sfctmp = 271.0; qsnow = 0.01; albold = 0.60; cosz = 0.7
      bdfall = min(120.0, 67.92 + 51.25*exp((sfctmp - TFRZ)/2.59))
      snowhin = qsnow/bdfall
    case (13)  ! nighttime aging only (cosz=0), no snowfall
      isnow = -1; snowh = 0.15; sneqv = 20.0
      snice(0) = 20.0; snliq(0) = 0.0; stc(0) = 260.0
      dzsnso(0) = 0.15
      sfctmp = 258.0; tg = 258.0; albold = 0.70; cosz = -0.1
    case (14)  ! two thin layers -> phase-1 collapse to one, then check resum
      isnow = -2; snowh = 0.03; sneqv = 0.6
      snice(-1) = 0.08; snliq(-1) = 0.0; stc(-1) = 269.0
      snice(0)  = 0.5;  snliq(0)  = 0.05; stc(0) = 271.0
      dzsnso(-1) = 0.015; dzsnso(0) = 0.015
      sfctmp = 268.0
    end select

    ! Build a consistent ZSNSO for the active snow layers from DZSNSO (as a prior
    ! WRF step would have stored it): cumulative NEGATIVE depth from the topmost
    ! active snow layer down to the surface, then continued through the soil
    ! (offset by the surface snow depth).
    do iz = isnow+1, 0
      if (iz == isnow+1) then
        zsnso(iz) = -dzsnso(iz)
      else
        zsnso(iz) = zsnso(iz-1) - dzsnso(iz)
      end if
    end do
    if (isnow < 0) then
      do iz = 1, NSOIL
        zsnso(iz) = zsnso(0) + zsoil(iz)
      end do
    else
      do iz = 1, NSOIL
        zsnso(iz) = zsoil(iz)
      end do
    end if

    sneqvo = sneqv   ! SNEQVO captured BEFORE SNOWWATER advances SNEQV
  end subroutine setup_scenario

  subroutine dump_state(u, s, tag, isnow, snowh, sneqv, sneqvo, tauss, albold, &
       snice, snliq, stc, zsnso, dzsnso, sh2o, sice, qsnbot, snoflow, p1, p2)
    integer, intent(in) :: u, s, isnow
    character(len=*), intent(in) :: tag
    real, intent(in) :: snowh, sneqv, sneqvo, tauss, albold, qsnbot, snoflow, p1, p2
    real, intent(in) :: snice(-NSNOW+1:0), snliq(-NSNOW+1:0)
    real, intent(in) :: stc(-NSNOW+1:NSOIL), zsnso(-NSNOW+1:NSOIL), dzsnso(-NSNOW+1:NSOIL)
    real, intent(in) :: sh2o(1:NSOIL), sice(1:NSOIL)
    integer :: iz
    write(u,'(A,I3,1X,A)') 'SCEN', s, tag
    write(u,'(A,I4)') 'ISNOW ', isnow
    write(u,'(A,4ES24.16)') 'SCAL ', snowh, sneqv, sneqvo, tauss
    write(u,'(A,ES24.16)') 'ALBOLD ', albold
    write(u,'(A,4ES24.16)') 'OUTS ', qsnbot, snoflow, p1, p2
    do iz = -NSNOW+1, 0
      write(u,'(A,I3,3ES24.16)') 'SNL ', iz, snice(iz), snliq(iz), stc(iz)
    end do
    do iz = -NSNOW+1, NSOIL
      write(u,'(A,I3,2ES24.16)') 'ZD ', iz, zsnso(iz), dzsnso(iz)
    end do
    do iz = 1, NSOIL
      write(u,'(A,I3,2ES24.16)') 'SOIL ', iz, sh2o(iz), sice(iz)
    end do
  end subroutine dump_state

end module noahmp_snow_oracle


program snow_oracle
  use noahmp_snow_oracle
  implicit none

  type(noahmp_parameters) :: parameters
  integer :: s
  character(len=256) :: outpath
  integer, parameter :: NF = 14

  integer :: isnow
  real :: snowh, sneqv, sneqvo, tauss, albold, tg, sfctmp, dt, cosz, bdfall
  real :: snice(-NSNOW+1:0), snliq(-NSNOW+1:0)
  real :: stc(-NSNOW+1:NSOIL), zsnso(-NSNOW+1:NSOIL), dzsnso(-NSNOW+1:NSOIL)
  real :: sh2o(1:NSOIL), sice(1:NSOIL), zsoil(1:NSOIL)
  real :: ficeold(-NSNOW+1:0), qsnow, snowhin, qsnfro, qsnsub, qrain
  integer :: imelt(-NSNOW+1:0)
  real :: qsnbot, snoflow, ponding1, ponding2, fage, alb
  real :: albsnd(2), albsni(2)

  call get_command_argument(1, outpath)
  if (len_trim(outpath) == 0) outpath = 'snow_oracle_out.txt'
  open(unit=20, file=trim(outpath), status='replace', action='write')

  zsoil = (/ -0.1, -0.4, -1.0, -2.0 /)

  do s = 1, NF
    call setup_scenario(s, isnow, snowh, sneqv, sneqvo, tauss, albold, tg, &
         sfctmp, dt, snice, snliq, stc, zsnso, dzsnso, sh2o, sice, &
         ficeold, qsnow, snowhin, qsnfro, qsnsub, qrain, imelt, cosz, bdfall)

    call dump_state(20, s, 'PRE', isnow, snowh, sneqv, sneqvo, tauss, albold, &
         snice, snliq, stc, zsnso, dzsnso, sh2o, sice, 0.0, 0.0, 0.0, 0.0)

    call SNOWWATER(parameters, NSNOW, NSOIL, imelt, dt, zsoil, sfctmp, snowhin, &
         qsnow, qsnfro, qsnsub, qrain, ficeold, 1, 1, &
         isnow, snowh, sneqv, snice, snliq, sh2o, sice, stc, zsnso, dzsnso, &
         qsnbot, snoflow, ponding1, ponding2)

    call SNOW_AGE(parameters, dt, tg, sneqvo, sneqv, tauss, fage)
    alb = albold
    if (cosz > 0.0) then
      call SNOWALB_CLASS(parameters, 2, qsnow, dt, alb, albold, albsnd, albsni, 1, 1)
      albold = alb
    end if

    call dump_state(20, s, 'POST', isnow, snowh, sneqv, sneqvo, tauss, albold, &
         snice, snliq, stc, zsnso, dzsnso, sh2o, sice, qsnbot, snoflow, ponding1, ponding2)
    write(20,'(A,I3,A,ES24.16)') 'SCEN', s, ' FAGE ', fage
  end do

  close(20)
  write(*,*) 'snow oracle wrote ', trim(outpath)
end program snow_oracle
