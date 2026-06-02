!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! S3 interp ORACLE DRIVER
!
! C-callable wrappers around the REAL WPS interp_module.F kernels. Python
! (ctypes) drives these to produce the ground-truth values the JAX port is
! graded against (sprint AC: each kernel <= 1e-6 rel vs interp_module.F).
!
! The slab is passed as a flat (nx*ny) array in Fortran column-major order, i.e.
! slab1d(i + (j-1)*nx) == slab(i,j) with i in 1..nx (first/x axis), j in 1..ny
! (second/y axis) -- exactly interp_module.F's (start_x:end_x, start_y:end_y)
! convention with start_x=start_y=1.
!
! interp_list is an integer array of method ids (terminated by a 0), built on
! the Python side to match a parsed METGRID.TBL string. interp_opts carries the
! per-method option (search depth); 0 elsewhere.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module oracle_driver

   use iso_c_binding
   use interp_module

   implicit none

   contains

   ! Run interp_sequence at npts target points against a single 2-D slab.
   ! has_mask=0 -> no mask; has_mask=1 -> mask_array given, equality exclusion
   !   when relational==' ' (passed as ichar of a 1-char code: 32=' ',60='<',62='>').
   subroutine oracle_interp(slab1d, nx, ny, &
                            rx, ry, npts, &
                            method_ids, nmeth, opts, &
                            msgval, &
                            has_mask, mask1d, maskval, rel_code, &
                            out) bind(C, name="oracle_interp")

      integer(c_int), value :: nx, ny, npts, nmeth, has_mask, rel_code
      real(c_float), intent(in) :: slab1d(nx*ny)
      real(c_float), intent(in) :: rx(npts), ry(npts)
      integer(c_int), intent(in) :: method_ids(nmeth), opts(nmeth)
      real(c_float), value :: msgval, maskval
      real(c_float), intent(in) :: mask1d(*)
      real(c_float), intent(out) :: out(npts)

      ! Local
      real, allocatable :: slab(:,:,:)
      real, allocatable :: mask(:,:)
      integer, allocatable :: ilist(:), iopts(:)
      character(len=1) :: rel
      integer :: i, j, k

      allocate(slab(1:nx, 1:ny, 1:1))
      do j = 1, ny
         do i = 1, nx
            slab(i, j, 1) = slab1d(i + (j-1)*nx)
         end do
      end do

      ! interp_list / interp_opts in interp_module are 1-based, 0-terminated;
      ! interp_sequence reads interp_list(idx) and stops at the first 0.
      allocate(ilist(nmeth+1), iopts(nmeth+1))
      ilist = 0
      iopts = 0
      do k = 1, nmeth
         ilist(k) = method_ids(k)
         iopts(k) = opts(k)
      end do

      rel = char(rel_code)

      if (has_mask == 1) then
         allocate(mask(1:nx, 1:ny))
         do j = 1, ny
            do i = 1, nx
               mask(i, j) = mask1d(i + (j-1)*nx)
            end do
         end do
         do k = 1, npts
            out(k) = interp_sequence(rx(k), ry(k), 1, slab, 1, nx, 1, ny, 1, 1, &
                                     msgval, ilist, iopts, 1, rel, maskval, mask)
         end do
         deallocate(mask)
      else
         do k = 1, npts
            out(k) = interp_sequence(rx(k), ry(k), 1, slab, 1, nx, 1, ny, 1, 1, &
                                     msgval, ilist, iopts, 1)
         end do
      end if

      deallocate(slab, ilist, iopts)

   end subroutine oracle_interp

   ! Direct oned() exposure for the 1-D unit oracle.
   subroutine oracle_oned(x, a, b, c, d, n, out) bind(C, name="oracle_oned")
      integer(c_int), value :: n
      real(c_float), intent(in) :: x(n), a(n), b(n), c(n), d(n)
      real(c_float), intent(out) :: out(n)
      integer :: k
      do k = 1, n
         out(k) = oned(x(k), a(k), b(k), c(k), d(k))
      end do
   end subroutine oracle_oned

end module oracle_driver
