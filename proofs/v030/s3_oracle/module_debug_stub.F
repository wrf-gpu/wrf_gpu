!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! MODULE_DEBUG (STUB) for the S3 interp oracle harness.
!
! The real WPS module_debug.F depends on parallel_module (MPI). For the offline
! interp-kernel oracle we only need the WARN/ERROR level constants and an
! mprintf that prints nothing meaningful (the interp kernels only call mprintf
! for unrecognized-method warnings, which the analytic oracle test never hits).
! Interface MATCHES the real module_debug mprintf signature exactly so
! interp_module.F / bitarray_module.F / queue_module.F compile unchanged.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module module_debug

   integer, parameter :: QUIET=-100, LOGFILE=-2, DEBUG=0, INFORM=1, &
                         WARN=2, ERROR=3, STDOUT=100

   contains

   subroutine mprintf(assertion, level, fmtstring, &
                      newline, &
                      i1, i2, i3, i4, i5, i6, &
                      f1, f2, f3, f4, f5, f6, &
                      s1, s2, s3, s4, s5, s6, &
                      l1, l2, l3, l4, l5, l6)

      implicit none

      integer, intent(in) :: level
      logical, intent(in) :: assertion
      character (len=*), intent(in) :: fmtstring
      logical, intent(in), optional :: newline
      integer, intent(in), optional :: i1, i2, i3, i4, i5, i6
      real, intent(in), optional :: f1, f2, f3, f4, f5, f6
      logical, intent(in), optional :: l1, l2, l3, l4, l5, l6
      character (len=*), intent(in), optional :: s1, s2, s3, s4, s5, s6

      ! Only act on a hard ERROR assertion (mirror the real one stopping run);
      ! for the oracle we just print and stop so a bad test is loud.
      if (assertion .and. level == ERROR) then
         write(0,*) 'STUB module_debug ERROR: ', trim(fmtstring)
         stop 1
      end if

   end subroutine mprintf

end module module_debug
