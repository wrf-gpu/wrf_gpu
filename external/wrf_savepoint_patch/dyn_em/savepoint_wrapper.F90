program wrf_savepoint_instrumented
  use hdf5
  implicit none
  integer :: hdferr
  call h5open_f(hdferr)
  call h5close_f(hdferr)
  print *, "M6B0-R WRF_SAVEPOINT CPU emission shim"
  print *, "Savepoint extraction is orchestrated by scripts/m6b0r_wrf_savepoint_extract.py"
end program wrf_savepoint_instrumented

module savepoint_wrapper
  implicit none
contains
  subroutine sp_calc_coef_w_pre()
  end subroutine sp_calc_coef_w_pre
  subroutine sp_calc_coef_w_post()
  end subroutine sp_calc_coef_w_post
  subroutine sp_small_step_prep_post()
  end subroutine sp_small_step_prep_post
  subroutine sp_advance_mu_t_pre(rkstage, acstep, mu, mut, mudf, muts, muave, ww_in, theta_in)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww_in(:, :, :), theta_in(:, :, :)
  end subroutine sp_advance_mu_t_pre
  subroutine sp_advance_mu_t_post(rkstage, acstep, mu, mut, mudf, muts, muave, ww_out, theta_out, ph_tend)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww_out(:, :, :), theta_out(:, :, :), ph_tend(:, :, :)
  end subroutine sp_advance_mu_t_post
  subroutine sp_advance_uv_post()
  end subroutine sp_advance_uv_post
  subroutine sp_advance_w_rhs_ready()
  end subroutine sp_advance_w_rhs_ready
  subroutine sp_advance_w_raw_w()
  end subroutine sp_advance_w_raw_w
  subroutine sp_advance_w_tridiag_fwd()
  end subroutine sp_advance_w_tridiag_fwd
  subroutine sp_advance_w_tridiag_back()
  end subroutine sp_advance_w_tridiag_back
  subroutine sp_advance_w_rayleigh()
  end subroutine sp_advance_w_rayleigh
  subroutine sp_advance_w_ph_final()
  end subroutine sp_advance_w_ph_final
  subroutine sp_calc_p_rho_post()
  end subroutine sp_calc_p_rho_post
  subroutine sp_small_step_finish_post()
  end subroutine sp_small_step_finish_post
  subroutine sp_acoustic_substep_boundary()
  end subroutine sp_acoustic_substep_boundary
  subroutine sp_rk_stage_boundary()
  end subroutine sp_rk_stage_boundary
end module savepoint_wrapper
