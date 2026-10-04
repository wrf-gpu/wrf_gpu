! Independent pristine-WRF REAL oracle, linked to unchanged RWORDSIZE=4 objects.
! Caller wiring follows WATER :6199-6255 for opt_run=3, soil_update_steps=1.
! This is the frozen standalone gate's soil boundary: BTRANI=0, no precip/melt,
! snow evaporation removed before SOILWATER. Snow/canopy evolution is separately
! gated; no synthetic output is substituted for the pristine soil solver.
program water_real_driver
  use module_sf_noahmplsm, only: noahmp_parameters, noahmp_options, soilwater
  use noahmp_tables, only: read_mp_veg_parameters, read_mp_soil_parameters, &
       read_mp_rad_parameters, read_mp_global_parameters
  use module_sf_noahmpdrv, only: transfer_mp_parameters
  implicit none
  integer, parameter :: nsoil=4, nsnow=3
  integer :: ncol, i, vegtyp, isltyp, soiltype(nsoil), k
  real :: dt, zsoil(nsoil), dzsnso(-nsnow+1:nsoil), smc(nsoil), sh2o(nsoil), sice(nsoil)
  real :: smc_in(nsoil), qseva, qinsur, etrani(nsoil), tg, sneqv, edir, dx
  real :: zwt, smcwtd, deeprech, runsrf, qdrain, runsub, wcnd(nsoil), fcrmax, qtldrn
  ! SOILWATER declares RUNSUB OUT yet subtracts from the caller's zero value.
  ! WATER supplies that zero locally. Preserve its initialization across this
  ! external call; gfortran can otherwise remove it on the OUT declaration.
  volatile :: runsub
  real :: runtime_residual, beginning, ending
  type(noahmp_parameters) :: parameters
  if (storage_size(dt) /= 32) stop 'oracle must be compiled WRF REAL (32 bits)'
  call read_mp_veg_parameters('MODIFIED_IGBP_MODIS_NOAH')
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()
  call noahmp_options(4,1,1,3,1,1,1,3,2,1,2,1,1,1,1,0,0,0,1,0)
  open(31,file='water_columns.in',status='old')
  open(32,file='water_columns.out',status='replace')
  read(31,*) ncol
  do i=1,ncol
    read(31,*) vegtyp, isltyp, dt, dx, tg, sneqv, edir, smcwtd
    read(31,*) zsoil
    read(31,*) dzsnso(1:nsoil)
    read(31,*) smc
    read(31,*) sh2o
    smc_in=smc
    soiltype=isltyp
    call transfer_mp_parameters(4,vegtyp,soiltype,1,4,0,parameters)
    sice=max(0.0,smc-sh2o)
    dzsnso(-nsnow+1:0)=0.0
    ! WATER :6128-6168 (same frozen standalone interface). EDIR is net ground ET.
    qseva=max(edir,0.0)
    if(sneqv>0.0) qseva=qseva-min(qseva,sneqv/dt)
    qinsur=max(-edir,0.0)
    if(sneqv>0.0) qinsur=0.0
    if(tg<=273.16) then
      sice(1)=sice(1)+(qinsur-qseva)*dt/(dzsnso(1)*1000.0)
      if(sice(1)<0.0) then
        sh2o(1)=sh2o(1)+sice(1)
        sice(1)=0.0
      endif
      smc(1)=sh2o(1)+sice(1)
      qseva=0.0
      qinsur=0.0
    endif
    qseva=qseva*0.001
    qinsur=qinsur*0.001
    etrani=0.0
    zwt=-2.0
    deeprech=0.0
    qtldrn=0.0
    runsub=0.0
    call soilwater(parameters,nsoil,nsnow,dt,zsoil,dzsnso, &
         qinsur,qseva,etrani,sice,i,1,0.0,dx,sh2o,smc,zwt,vegtyp, &
         smcwtd,deeprech,runsrf,qdrain,runsub,wcnd,fcrmax,qtldrn)
    ! WATER :6224-6253 explicitly adds drainage and reconstitutes total moisture.
    runsub=runsub+qdrain
    smc=sh2o+sice
    runsrf=runsrf*dt
    runsub=runsub*dt
    ! Also disclose the ERROR-style REAL accumulation, distinct from the common
    ! external diagnostic used to score both implementations.
    beginning=0.0
    ending=0.0
    do k=1,nsoil
      beginning=beginning+smc_in(k)*dzsnso(k)*1000.0
      ending=ending+smc(k)*dzsnso(k)*1000.0
    enddo
    runtime_residual=ending-beginning+runsrf+runsub
    if(sneqv<=0.0) runtime_residual=runtime_residual+edir*dt
    write(32,'(I4,11ES25.16)') i,smc,sh2o,runsrf,runsub,runtime_residual
  enddo
  close(31)
  close(32)
end program water_real_driver
