#!/bin/bash
# Build heFFTe.  Usage: build_heffte.sh [avx2|avx512]
#
# heFFTe compiles its stock/pack kernels with -mavx512 when Heffte_ENABLE_AVX512=ON
# and does NOT runtime-dispatch, so such a build SIGILLs on the Haswell login node
# and on the sm_75 prod nodes.  We therefore keep two installs:
#   ext/install          AVX2  -- login node, prod/devel/long
#   ext/install/avx512   AVX512 -- axxxl, a100l, a100r
set -e
source /home/lqcd/wdetmold/fft/env.sh
VARIANT=${1:-avx2}
case $VARIANT in
  avx2)   PREFIX=$FFT_PREFIX;          AVX512=OFF ;;
  avx512) PREFIX=$FFT_PREFIX/avx512;   AVX512=ON  ;;
  *) echo "usage: $0 [avx2|avx512]"; exit 1 ;;
esac
for i in $(seq 1 120); do [ -f $FFT_PREFIX/lib/libfftw3f.so ] && break; sleep 10; done
cd $FFT_SRC/heffte
d=build-$VARIANT; rm -rf $d && mkdir $d && cd $d
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PREFIX \
  -DBUILD_SHARED_LIBS=ON \
  -DHeffte_ENABLE_FFTW=ON  -DFFTW_ROOT=$FFT_PREFIX \
  -DHeffte_ENABLE_MKL=ON   -DMKL_ROOT=$MKLROOT \
  -DHeffte_ENABLE_CUDA=ON  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DCMAKE_CUDA_FLAGS="-I$OPENMPI_TOP/include" \
  -DHeffte_ENABLE_AVX=ON -DHeffte_ENABLE_AVX512=$AVX512 \
  -DCMAKE_CXX_COMPILER=mpicxx \
  > cmake.log 2>&1
make -j12 > build.log 2>&1
make install >> build.log 2>&1
echo "heFFTe($VARIANT) installed to $PREFIX (fftw, mkl, cufft, stock/avx$( [ $AVX512 = ON ] && echo 512 || echo 2))"
