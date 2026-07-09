program drv_radar
  ! Standalone driver over the reference COSP radar path, used as the
  ! known-good-output (KGO) generator for validating jcosp.
  !
  ! Reads a plain-text case file:
  !   nprof ngate
  !   p(Pa) T(K) sh(kg/kg) hgt(km)                    [ngate lines, TOA first]
  !   hm(g/kg) x 9, re(um) x 9                        [nprof*ngate lines]
  !   frac_out(int) x ngate, ls_p(1) , cv_p(1) flags  [for prec_scops: nprof rows]
  ! and writes z_vol, kr_vol, g_vol, dBZe, Ze_non (nprof x ngate) plus the
  ! prec_scops output, all in ES24.16.
  use cosp_kinds,          only: wp
  use mod_cosp_config,     only: R_UNDEF, N_HYDRO
  use mod_cosp_stats,      only: radar_cfg
  use mod_cosp_cloudsat_interface, only: cosp_cloudsat_init
  use mod_quickbeam_optics, only: size_distribution, hydro_class_init, &
                                  quickbeam_optics, quickbeam_optics_init, gases
  use quickbeam,           only: quickbeam_subcolumn
  use mod_prec_scops,      only: prec_scops
  implicit none

  integer :: nprof, ngate, i, k, j
  real(wp), allocatable :: p(:), t(:), sh(:), hgt(:)
  real(wp), allocatable :: hm(:,:,:), re(:,:,:), np_mat(:,:,:)
  real(wp), allocatable :: pmat(:,:), tmat(:,:), shmat(:,:), hgtmat(:,:)
  real(wp), allocatable :: z_vol(:,:), kr_vol(:,:), g_vol(:,:)
  real(wp), allocatable :: dbze(:,:), ze_non(:,:)
  real(wp), allocatable :: frac_out(:,:,:), prec_frac(:,:,:)
  real(wp), allocatable :: ls_p(:,:), cv_p(:,:)
  type(size_distribution) :: sd
  character(len=64) :: scheme_name
  type(radar_cfg) :: rcfg

  open(10, file='kgo_input.txt', status='old')
  read(10,*) nprof, ngate
  allocate(p(ngate), t(ngate), sh(ngate), hgt(ngate))
  do k = 1, ngate
     read(10,*) p(k), t(k), sh(k), hgt(k)
  end do
  allocate(hm(nprof,ngate,N_HYDRO), re(nprof,ngate,N_HYDRO), &
           np_mat(nprof,ngate,N_HYDRO))
  np_mat = 0._wp
  do i = 1, nprof
     do k = 1, ngate
        read(10,*) (hm(i,k,j), j=1,N_HYDRO), (re(i,k,j), j=1,N_HYDRO)
     end do
  end do
  ! prec_scops inputs: subcolumn cloud map + gridbox precip-rate flags.
  allocate(frac_out(1,nprof,ngate), prec_frac(1,nprof,ngate))
  allocate(ls_p(1,ngate), cv_p(1,ngate))
  do i = 1, nprof
     read(10,*) (frac_out(1,i,k), k=1,ngate)
  end do
  read(10,*) (ls_p(1,k), k=1,ngate)
  read(10,*) (cv_p(1,k), k=1,ngate)
  close(10)

  call quickbeam_optics_init()
  call hydro_class_init(.true., .false., sd)
  scheme_name = 'kgo_test_scheme'
  call cosp_cloudsat_init(94._wp, -1._wp, 1, 0, R_UNDEF, N_HYDRO, 0, rcfg, &
                          scheme_name, load_LUT=.false.)

  allocate(pmat(nprof,ngate), tmat(nprof,ngate), shmat(nprof,ngate), &
           hgtmat(nprof,ngate), z_vol(nprof,ngate), kr_vol(nprof,ngate), &
           g_vol(nprof,ngate), dbze(nprof,ngate), ze_non(nprof,ngate))
  do i = 1, nprof
     pmat(i,:) = p; tmat(i,:) = t; shmat(i,:) = sh; hgtmat(i,:) = hgt
  end do

  z_vol = 0._wp; kr_vol = 0._wp
  call quickbeam_optics(sd, rcfg, nprof, ngate, R_UNDEF, hm, re, np_mat, &
                        pmat, tmat, shmat, z_vol, kr_vol)
  ! Gaseous attenuation exactly as the reference driver computes it.
  do i = 1, nprof
     do k = 1, ngate
        g_vol(i,k) = gases(pmat(i,k), tmat(i,k), shmat(i,k), rcfg%freq)
     end do
  end do
  call quickbeam_subcolumn(rcfg, nprof, ngate, hgtmat, z_vol, kr_vol, g_vol, &
                           dbze, ze_non)
  call prec_scops(1, ngate, nprof, ls_p, cv_p, frac_out, prec_frac)

  open(11, file='kgo_output.txt', status='replace')
  do i = 1, nprof
     do k = 1, ngate
        write(11,'(5ES24.16)') z_vol(i,k), kr_vol(i,k), g_vol(i,k), &
                               dbze(i,k), ze_non(i,k)
     end do
  end do
  do i = 1, nprof
     write(11,'(*(I3))') (int(prec_frac(1,i,k)), k=1,ngate)
  end do
  close(11)
end program drv_radar
