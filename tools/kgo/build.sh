#!/bin/bash
# Build the standalone COSP radar-path reference binary for KGO validation.
# Requires gfortran and a COSPv2.0 checkout (env COSP_REF_DIR, or it is
# cloned into ./cosp-src).
set -euo pipefail
cd "$(dirname "$0")"
REF="${COSP_REF_DIR:-cosp-src}"
if [ ! -d "$REF" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/CFMIP/COSPv2.0 "$REF"
fi
SUB="$REF/subsample_and_optics_example"
mkdir -p build && cd build
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c ../stubs.f90
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$REF/src/cosp_constants.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$REF/src/cosp_config.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$REF/src/cosp_stats.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/optics/quickbeam_optics/mrgrnk.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/optics/quickbeam_optics/array_lib.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/optics/quickbeam_optics/math_lib.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/optics/quickbeam_optics/optics_lib.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$REF/src/simulator/quickbeam/quickbeam.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/optics/quickbeam_optics/quickbeam_optics.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$REF/src/simulator/cosp_cloudsat_interface.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c "$SUB/subcol/prec_scops.F90"
gfortran -O1 -ffree-line-length-none -fcheck=bounds -c ../drv_radar.f90
gfortran -o ../drv_radar ./*.o
echo "built: $(cd .. && pwd)/drv_radar"
