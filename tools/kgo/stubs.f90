! Minimal stubs to compile the COSP radar path standalone (no netCDF/RTTOV).
module cosp_kinds
  implicit none
  integer,parameter :: wp = selected_real_kind(15)
  integer,parameter :: dp = wp
end module cosp_kinds

module mod_cosp_error
  implicit none
contains
  subroutine errorMessage(msg)
    character(len=*),intent(in) :: msg
    write(0,*) trim(msg)
  end subroutine errorMessage
end module mod_cosp_error

module mod_cosp_rttov_util
  implicit none
  type rttov_cfg
    integer :: dummy = 0
  end type rttov_cfg
end module mod_cosp_rttov_util
