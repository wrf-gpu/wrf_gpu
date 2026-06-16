! Minimal stub of module_model_constants for the WDM6 single-column oracle.
!
! module_mp_wdm6.F uses ONLY three names from module_model_constants:
!   use module_model_constants, only : RE_QC_BG, RE_QI_BG, RE_QS_BG
! These are the effective-radius BACKGROUND values used in the wdm6() wrapper
! and effectRad_wdm6. We provide them with the EXACT values from the pristine
! WRF share/module_model_constants.F so the effective-radius diagnostics match
! the real scheme. (The driver also re-applies the same literals as bounds.)
!
! Provenance: WRF share/module_model_constants.F
!   real, parameter :: RE_QC_BG = 2.49E-6
!   real, parameter :: RE_QI_BG = 4.99E-6
!   real, parameter :: RE_QS_BG = 9.99E-6
module module_model_constants
  implicit none
  public
  real, parameter :: RE_QC_BG = 2.49E-6
  real, parameter :: RE_QI_BG = 4.99E-6
  real, parameter :: RE_QS_BG = 9.99E-6
end module module_model_constants
