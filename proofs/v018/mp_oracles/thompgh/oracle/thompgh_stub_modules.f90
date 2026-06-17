! =====================================================================
! v0.17 Thompson graupel-hail (mp_physics=38, THOMPSONGH) oracle stub set.
!
! Minimal standalone stubs so the UNMODIFIED WRF Thompson source
! (phys/module_mp_thompson.F) can be compiled + run on a single column in
! isolation, without ESMF / the full WRF framework, while still exercising
! the REAL is_hail_aware (mp=38) variable-density-graupel path.
!
! The Thompson source is copied VERBATIM (unmodified). It only needs, beyond
! the real module_model_constants / module_mp_radar / libmassv:
!   * module_wrf_error              -- USEd blanket; provides wrf_debug_level /
!                                      wrf_err_message names only.
!   * module_dm (ONLY: wrf_dm_max_real) -- the scheme `USE module_dm, ONLY:
!                                      wrf_dm_max_real`; serial = identity.
!   * module_timing                 -- the qr_acr_qg/qr_acr_qs table generators
!                                      `USE module_timing` (start/end/init).
!   * GLOBAL wrf_debug / wrf_message / wrf_error_fatal (no-op / print+stop).
!   * GLOBAL wrf_dm_on_monitor (=.TRUE.; serial single rank IS the monitor).
!   * GLOBAL wrf_dm_decomp1d / wrf_dm_gatherv -- the table generators call these
!                                      UNCONDITIONALLY (not DM_PARALLEL-guarded).
!                                      Serial semantics: decomp1d returns the
!                                      FULL 0-based [0, n-1] range; gatherv is a
!                                      no-op (the single rank already holds the
!                                      whole array).  These are the canonical
!                                      serial reductions, NOT a physics change.
!   * GLOBAL nl_get_force_read_thompson / nl_get_write_thompson_tables /
!     nl_get_write_thompson_mp38table -- namelist getters the init calls.
!       force_read_thompson      = .FALSE. (compute, do not require a .dat)
!       write_thompson_tables    = .FALSE. (do not write .dat files to disk)
!       write_thompson_mp38table = .TRUE.  (REQUIRED: lets the mp=38 path
!                                           COMPUTE the qr_acr_qg_mp38V1 table
!                                           in-memory instead of fatal-aborting
!                                           on the missing .dat -- the gate at
!                                           module_mp_thompson.F:4163).
!
! NOTHING here touches scheme physics: every stub is either a serial reduction
! identity, a namelist control value matching the WRF mp=38 production default,
! or a silent logger.  The microphysics + lookup tables are the UNMODIFIED
! Fortran.
! =====================================================================

MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: wrf_debug_level = 0
  CHARACTER(LEN=512) :: wrf_err_message = ' '
END MODULE module_wrf_error

MODULE module_dm
  IMPLICIT NONE
CONTAINS
  ! Serial: the max over one rank is the value itself.
  REAL FUNCTION wrf_dm_max_real ( inval )
    IMPLICIT NONE
    REAL, INTENT(IN) :: inval
    wrf_dm_max_real = inval
  END FUNCTION wrf_dm_max_real
END MODULE module_dm

! table_ccnAct does a BLANKET `USE module_domain` but references NO symbol from
! it (it only calls the EXTERNAL wrf_dm_on_monitor + reads CCN_ACTIVATE.BIN).
! An empty stub satisfies the USE without touching physics.
MODULE module_domain
  IMPLICIT NONE
END MODULE module_domain

MODULE module_timing
  IMPLICIT NONE
  INTEGER, PARAMETER :: cnmax = 30
  INTEGER, PRIVATE :: cn = 0
  REAL, PRIVATE :: elapsed_seconds_start(cnmax) = 0.0
CONTAINS
  SUBROUTINE init_module_timing
    cn = 0
  END SUBROUTINE init_module_timing
  SUBROUTINE start_timing
    ! no-op (we do not need wall-clock instrumentation in the oracle)
  END SUBROUTINE start_timing
  SUBROUTINE end_timing ( string )
    CHARACTER(LEN=*), INTENT(IN) :: string
    ! no-op
  END SUBROUTINE end_timing
END MODULE module_timing

! ----- global (non-module) symbols the scheme references ------------------

SUBROUTINE wrf_debug( level, str )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: str
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

LOGICAL FUNCTION wrf_dm_on_monitor()
  IMPLICIT NONE
  wrf_dm_on_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor

! Serial 1-D decomposition: rank 0 owns the whole [0, total-1] range.
! WRF returns ZERO-BASED bounds (see module_mp_thompson.F comment "values
! returned from wrf_dm_decomp1d are zero-based, add 1").
SUBROUTINE wrf_dm_decomp1d( nt, km_s, km_e )
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: nt
  INTEGER, INTENT(OUT) :: km_s, km_e
  km_s = 0
  km_e = nt - 1
END SUBROUTINE wrf_dm_decomp1d

! Serial gatherv: the single rank already holds the full array, so this is a
! no-op.  Signature matches the Thompson call
!   CALL wrf_dm_gatherv(table, nz, km_s, km_e, R8SIZE)
! where table is a REAL(KIND=8) array, nz/km_s/km_e are INTEGERs and the last
! arg is the element byte size (8).  We take the buffer as assumed-size so any
! rank/shape binds; nothing is moved.
SUBROUTINE wrf_dm_gatherv( v, nz, km_s, km_e, size )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nz, km_s, km_e, size
  REAL(KIND=8), INTENT(INOUT) :: v(*)
  ! serial: no inter-rank gather required
END SUBROUTINE wrf_dm_gatherv

! ----- namelist getters (mp=38 production control values) -----------------

SUBROUTINE nl_get_force_read_thompson( idomain, value )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: idomain
  LOGICAL, INTENT(OUT) :: value
  value = .FALSE.
END SUBROUTINE nl_get_force_read_thompson

SUBROUTINE nl_get_write_thompson_tables( idomain, value )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: idomain
  LOGICAL, INTENT(OUT) :: value
  ! .TRUE. so the FIRST init WRITES the (heavy, 9-density-plane) collision
  ! tables to .dat files; subsequent cases INQUIRE+READ them instead of
  ! recomputing. This is what makes a 6-case oracle build practical -- the
  ! mp=38 qr_acr_qg table generation is ~10 CPU-min and would otherwise rerun
  ! per case. The written .dat is the scheme's OWN output (bit-identical to the
  ! in-memory table), not a physics change.
  value = .TRUE.
END SUBROUTINE nl_get_write_thompson_tables

SUBROUTINE nl_get_write_thompson_mp38table( idomain, value )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: idomain
  LOGICAL, INTENT(OUT) :: value
  value = .TRUE.
END SUBROUTINE nl_get_write_thompson_mp38table
