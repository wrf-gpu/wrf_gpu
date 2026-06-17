! =====================================================================
! v0.17 Goddard GCE 4-ice (WRF mp_physics=7, GSFCGCE_4ICE_NUWRF) oracle
! stub set.
!
! Minimal standalone stubs so the UNMODIFIED WRF Goddard 4-ice source
! (phys/module_mp_gsfcgce_4ice_nuwrf.F) and its real dependency
! module_mp_radar.F can be compiled + run on a single column in isolation,
! without ESMF / the full WRF framework.
!
! The Goddard source is copied VERBATIM (unmodified). Beyond the real
! module_mp_radar / module_model_constants, it needs ONLY:
!   * module_wrf_error  -- USEd blanket by module_mp_radar; the Goddard
!                          module references the module-scope name
!                          `wrf_err_message` at line 4647 (auto_conversion
!                          error path). Provide both wrf_debug_level and
!                          wrf_err_message (the real module exports both).
!   * GLOBAL wrf_debug      (no-op logger; radar_init calls it).
!   * GLOBAL wrf_error_fatal (print+stop; Goddard calls it on a bad xland
!                          or a strange auto-conversion R6).
!   * GLOBAL wrf_message    (print; provided for link safety).
!   * GLOBAL wrf_dm_on_monitor (=.TRUE.; serial single rank IS the monitor;
!                          declared `LOGICAL, EXTERNAL` at module line 38 and
!                          called at lines 2434 / 2534 to gate diagnostic
!                          prints to the monitor rank only).
!
! NOTE: the Goddard module does NOT `USE module_dm` (line 2092 is a COMMENT),
! does NOT call any nl_get_*, does NOT use module_timing/module_domain, and
! references no VREC/VSQRT (so frame/libmassv.F is not needed). module_mp_radar
! likewise needs no libmassv here. Hence this stub is deliberately smaller
! than the Thompson stub.
!
! NOTHING here touches scheme physics: every stub is either the canonical
! serial reduction identity (single rank IS the monitor) or a silent/aborting
! logger. The microphysics is the UNMODIFIED Fortran.
! =====================================================================

MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: wrf_debug_level = 0
  CHARACTER(LEN=512) :: wrf_err_message = ' '
END MODULE module_wrf_error

! ----- global (non-module) symbols the scheme references ------------------

SUBROUTINE wrf_debug( level, str )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: str
  ! intentionally silent
END SUBROUTINE wrf_debug

SUBROUTINE wrf_message( str )
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(0,'(A)') TRIM(str)
END SUBROUTINE wrf_message

SUBROUTINE wrf_error_fatal( str )
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(0,'(A)') 'WRF_ERROR_FATAL: '//TRIM(str)
  STOP 9
END SUBROUTINE wrf_error_fatal

! Serial: the single rank IS the monitor rank.
LOGICAL FUNCTION wrf_dm_on_monitor()
  IMPLICIT NONE
  wrf_dm_on_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor
