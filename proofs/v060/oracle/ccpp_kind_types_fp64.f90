 module ccpp_kind_types
! fp64 override of the WRF physics kind for an fp64 cross-check oracle build.
! The pristine WRF ccpp_kind_types.f90 sets kind_phys=selected_real_kind(6)
! (single precision); this override sets it to selected_real_kind(15) (double)
! ONLY to demonstrate that trace-cell effective-radius floor flips in the fp32
! reference are single-precision detection-threshold dust, not a JAX port bug.
! The scheme source (mp_wsm6.F90) is otherwise UNMODIFIED.
   integer, parameter :: kind_phys = selected_real_kind(15)
   integer, parameter :: kind_dyn  = selected_real_kind(15)
   integer, parameter :: kind_io4  = selected_real_kind(6)
   integer, parameter :: kind_io8  = selected_real_kind(15)
   integer, parameter :: kind_grid = selected_real_kind(15)
   integer, parameter :: kind_int8 = selected_int_kind(15)
   integer, parameter :: kind_int4 = selected_int_kind(8)
 end module ccpp_kind_types
