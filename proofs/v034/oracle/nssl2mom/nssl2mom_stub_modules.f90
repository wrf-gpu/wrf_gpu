! =====================================================================
! Minimal standalone stubs so the UNMODIFIED WRF NSSL 2-moment source
! (phys/module_mp_nssl_2mom.F) can be compiled/linked in isolation,
! without pulling in ESMF / the full WRF framework.
!
! module_mp_nssl_2mom.F has NO `USE` statements at all (fully
! self-contained module). Its only external references are:
!   - wrf_dm_on_monitor()  (logical function; nssl_2mom_init lines 1459,
!     1466 of phys/module_mp_nssl_2mom.F) -- used ONLY to decide whether
!     to (a) print a namelist warning (under #ifdef WRF_ELEC) and
!     (b) append the nssl_mp_params namelist to an existing
!     'namelist.output' file. Pure logging. The stub returns .FALSE.
!     (i.e. behaves like a non-monitor MPI rank), which skips the
!     namelist.output append; physics is untouched.
!   - wrf_error_fatal(str)  (nssl_2mom_init line 1868, only reached for
!     an invalid ipctmp) -- abort shim, prints and stops.
!
! These stubs are no-op logging/abort shims ONLY. No physics is stubbed.
! =====================================================================

LOGICAL FUNCTION wrf_dm_on_monitor()
  IMPLICIT NONE
  ! Behave like a non-monitor rank: skips the namelist.output append in
  ! nssl_2mom_init (logging only; no physics impact).
  wrf_dm_on_monitor = .FALSE.
END FUNCTION wrf_dm_on_monitor

SUBROUTINE wrf_error_fatal( str )
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(*,'(A)') 'WRF_ERROR_FATAL: '//TRIM(str)
  STOP 9
END SUBROUTINE wrf_error_fatal
