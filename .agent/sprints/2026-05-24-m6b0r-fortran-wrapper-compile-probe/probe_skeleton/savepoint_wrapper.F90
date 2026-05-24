! savepoint_wrapper.F90 — minimal probe skeleton for M6B0-R
!
! Compiled standalone against HDF5 1.14.5 Fortran API (nvfortran 26.3).
! In the production WRF tree this module will be gated by `#ifdef WRF_SAVEPOINT`
! and dropped into dyn_em/.  Here we strip the WRF dependency and exercise only
! the I/O surface so the M6B0-R worker knows the linker line and call costs.
!
! Layout written:
!   /data                  REAL(8) rank-3 dataset
!   /data attribute name           (string)
!   /data attribute units          (string)
!   /data attribute stagger        (string)
!   /data attribute rkstage        (INTEGER scalar)
!   /data attribute acstep         (INTEGER scalar)
!   /data attribute schema_version (INTEGER scalar)
!
! NOTE: production schema (operator name, WRF commit, namelist hash, dt, map
! factors, ...) is the M6B0-R worker's responsibility; this probe only proves
! the I/O path is sound under nvfortran + HDF5 1.14.5.

module savepoint_wrapper
   use hdf5
   use iso_fortran_env, only: real64, int32
   implicit none
   private

   public :: sp_write_real8_3d

   integer, parameter, public :: SP_SCHEMA_VERSION = 1

   ! Production wrapper will populate this from WRF state (Registry vars).
   ! Kept here purely so the M6B0-R worker can lift the type signature.
   type, public :: sp_metadata_t
      character(len=64) :: name      = ''
      character(len=16) :: units     = ''
      character(len=8)  :: stagger   = ''     ! 'C','X','Y','Z' or composite
      integer(int32)    :: rkstage   = -1
      integer(int32)    :: acstep    = -1
   end type sp_metadata_t

contains

   subroutine sp_write_real8_3d(path, name, units, stagger, rkstage, acstep, arr, ierr_out)
      ! Writes a 3D REAL(8) array `arr` to HDF5 file `path` as dataset /data
      ! and attaches the scalar+string metadata as attributes on /data.
      !
      ! On success ierr_out == 0.  On failure the caller gets the first nonzero
      ! HDF5 status code; this routine does NOT abort — the production wrapper
      ! must decide fail-closed vs. log-and-continue per ADR-025.

      character(len=*), intent(in)  :: path
      character(len=*), intent(in)  :: name
      character(len=*), intent(in)  :: units
      character(len=*), intent(in)  :: stagger
      integer(int32),   intent(in)  :: rkstage
      integer(int32),   intent(in)  :: acstep
      real(real64),     intent(in)  :: arr(:, :, :)
      integer,          intent(out) :: ierr_out

      integer(hid_t)   :: file_id, dset_id, dspace_id
      integer(hid_t)   :: attr_id, attr_space_id, str_type
      integer(hsize_t) :: dims(3), scalar_dim(1)
      integer          :: hdferr
      integer          :: i_status

      i_status = 0

      call h5open_f(hdferr); if (hdferr /= 0) i_status = hdferr

      call h5fcreate_f(trim(path), H5F_ACC_TRUNC_F, file_id, hdferr)
      if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      dims = shape(arr, kind=hsize_t)
      call h5screate_simple_f(3, dims, dspace_id, hdferr)
      if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      call h5dcreate_f(file_id, 'data', H5T_NATIVE_DOUBLE, dspace_id, dset_id, hdferr)
      if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, arr, dims, hdferr)
      if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      ! Scalar dataspace for attributes
      scalar_dim(1) = 1

      ! --- string attributes (variable length not needed; use fixed-length)
      call write_str_attr(dset_id, 'name',    name,    hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call write_str_attr(dset_id, 'units',   units,   hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call write_str_attr(dset_id, 'stagger', stagger, hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      ! --- integer scalar attributes
      call write_int_attr(dset_id, 'rkstage',        rkstage,           hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call write_int_attr(dset_id, 'acstep',         acstep,            hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call write_int_attr(dset_id, 'schema_version', SP_SCHEMA_VERSION, hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      call h5dclose_f(dset_id, hdferr);   if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call h5sclose_f(dspace_id, hdferr); if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call h5fclose_f(file_id, hdferr);   if (hdferr /= 0 .and. i_status == 0) i_status = hdferr
      call h5close_f(hdferr);             if (hdferr /= 0 .and. i_status == 0) i_status = hdferr

      ierr_out = i_status
   end subroutine sp_write_real8_3d

   subroutine write_str_attr(parent_id, attr_name, value, hdferr)
      integer(hid_t),   intent(in)  :: parent_id
      character(len=*), intent(in)  :: attr_name
      character(len=*), intent(in)  :: value
      integer,          intent(out) :: hdferr

      integer(hid_t)   :: attr_space, attr_type, attr_id
      integer(hsize_t) :: sdim(1)
      integer(size_t)  :: tsize

      sdim(1) = 1
      tsize   = max(len_trim(value), 1)

      call h5screate_simple_f(1, sdim, attr_space, hdferr); if (hdferr /= 0) return
      call h5tcopy_f(H5T_NATIVE_CHARACTER, attr_type, hdferr); if (hdferr /= 0) return
      call h5tset_size_f(attr_type, tsize, hdferr);            if (hdferr /= 0) return
      call h5acreate_f(parent_id, attr_name, attr_type, attr_space, attr_id, hdferr); if (hdferr /= 0) return
      call h5awrite_f(attr_id, attr_type, trim(value), sdim, hdferr); if (hdferr /= 0) return
      call h5aclose_f(attr_id, hdferr);    if (hdferr /= 0) return
      call h5tclose_f(attr_type, hdferr);  if (hdferr /= 0) return
      call h5sclose_f(attr_space, hdferr)
   end subroutine write_str_attr

   subroutine write_int_attr(parent_id, attr_name, value, hdferr)
      integer(hid_t),   intent(in)  :: parent_id
      character(len=*), intent(in)  :: attr_name
      integer(int32),   intent(in)  :: value
      integer,          intent(out) :: hdferr

      integer(hid_t)   :: attr_space, attr_id
      integer(hsize_t) :: sdim(1)
      integer(int32)   :: buf(1)

      sdim(1) = 1
      buf(1)  = value

      call h5screate_simple_f(1, sdim, attr_space, hdferr); if (hdferr /= 0) return
      call h5acreate_f(parent_id, attr_name, H5T_NATIVE_INTEGER, attr_space, attr_id, hdferr); if (hdferr /= 0) return
      call h5awrite_f(attr_id, H5T_NATIVE_INTEGER, buf, sdim, hdferr); if (hdferr /= 0) return
      call h5aclose_f(attr_id, hdferr); if (hdferr /= 0) return
      call h5sclose_f(attr_space, hdferr)
   end subroutine write_int_attr

end module savepoint_wrapper
