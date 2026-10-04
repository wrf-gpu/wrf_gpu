! =====================================================================
! Minimal standalone stub modules so the UNMODIFIED WRF Morrison source
! (module_mp_morr_two_moment.F) and its real dependency module_mp_radar.F
! can be compiled in isolation, without pulling in ESMF / the full WRF
! framework.
!
! Neither module_mp_morr_two_moment.F nor module_mp_radar.F reference any
! MODULE-SCOPE name from module_wrf_error; they only `USE module_wrf_error`
! (blanket) and call the GLOBAL subroutine wrf_debug. So an empty stub
! module + a no-op global wrf_debug is sufficient. The Morrison and radar
! scheme sources are copied VERBATIM and are NOT edited.
!
! module_model_constants is NOT stubbed: the real WRF share/module_model_constants.F
! is pure PARAMETERs (CP,G,R_D,R_V,EP_2,...) with no heavy deps, so the build
! script compiles the real file and this stub set does NOT redefine it.
! =====================================================================

MODULE module_wrf_error
  IMPLICIT NONE
  ! Morrison/radar only `USE` this module blanket; they reference no
  ! module-scope symbol from it. Provide a couple of harmless names that
  ! the real module also exports, in case of future ONLY-clause use.
  INTEGER :: wrf_debug_level = 0
  CHARACTER(LEN=512) :: wrf_err_message = ' '
END MODULE module_wrf_error

! Global no-op wrf_debug (radar calls CALL wrf_debug(...) at module scope).
SUBROUTINE wrf_debug( level, str )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: str
  ! intentionally silent
END SUBROUTINE wrf_debug

! Global wrf_error_fatal (not referenced by Morrison/radar in the path we
! exercise, but provided for link safety).
SUBROUTINE wrf_error_fatal( str )
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(*,'(A)') 'WRF_ERROR_FATAL: '//TRIM(str)
  STOP 9
END SUBROUTINE wrf_error_fatal
