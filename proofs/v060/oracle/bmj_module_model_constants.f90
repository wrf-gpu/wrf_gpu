! Minimal standalone constants module for compiling pristine module_cu_bmj.F.
!
! WRF's full share/module_model_constants.F exports names such as CP/G/D608 that
! collide with BMJDRV dummy arguments under a direct standalone gfortran build.
! The pristine BMJ source only needs the saturation-table constants below from
! the USE-associated module; the remaining physical constants are passed through
! BMJDRV/BMJ dummy arguments exactly as WRF does.
MODULE module_model_constants
  IMPLICIT NONE
  REAL, PARAMETER :: pq0 = 379.90516
  REAL, PARAMETER :: a2 = 17.2693882
  REAL, PARAMETER :: a3 = 273.16
  REAL, PARAMETER :: a4 = 35.86
  REAL, PARAMETER :: epsq = 1.0e-12
END MODULE module_model_constants
