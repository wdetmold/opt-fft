#!/bin/bash
# Build heFFTe with tracing ON, for the ONE measurement the distributed-FFT literature does
# not contain: the communication-vs-compute split of a distributed 3D FFT at 1-2 CPU nodes.
# See docs/literature_dist/00-SURVEY.md section 3 -- every published breakdown is >=4 nodes
# (local 1D FFT 9.3% at 168 Power9 cores, 11-13% across five libraries at 16 ranks), which
# caps a local-kernel-only drop-in at ~1.07x; nobody has measured the small-node end, and it
# is the number our whole distributed pitch depends on.
#
# Heffte_ENABLE_TRACING makes every transform log its events with MPI_Wtime and write one
# text file per rank.  The instrumented events (src/heffte_compute_transform.cpp) are exactly
# the split we want: "fft-1d" is the local batched-1D executor call -- the slot our kernel
# would occupy -- and "reshape" is the transpose (pack + MPI + unpack together; this build
# cannot separate those three, unlike the ICL Vampir traces, and the report must say so).
#
# Deliberately NOT installed: the benchmark binary in the build tree is all we need, so the
# production ext/install and ext/install/avx512 trees are left untouched.  CUDA is off -- this
# is a CPU measurement and nvcc doubles the build time for nothing.
#
# Two build settings here are not cosmetic:
#   * MKL is OFF.  This is an FFTW-backend CPU measurement, and the module's MKL resolves to
#     libmkl_*.so.2, which is not on the runtime path -- linking it only breaks exec.
#   * Heffte_ENABLE_TRACING is ALSO forced through CMAKE_CXX_FLAGS, not just the cmake option.
#     FFTW_ROOT puts ext/install/include on the include path, and that directory holds the
#     heffte_config.h of the earlier NON-tracing install; the library picked that one up and
#     compiled the trace globals (event_log/log_filename, defined in src/heffte_reshape3d.cpp)
#     out, while the benchmark saw the build-tree config and expected them -- a link-time
#     "undefined symbol: heffte::log_filename".  Defining the macro on the command line makes
#     the setting independent of which config header wins the include race.
#
# Usage: build_heffte_trace.sh [avx2|avx512]
#   avx2   -- login node, prod/devel/long   (heFFTe does NOT runtime-dispatch: an avx512
#   avx512 -- axxxl, a100l, a100r            build SIGILLs on Haswell and on the prod nodes)
set -e
source /home/lqcd/wdetmold/fft/env.sh
VARIANT=${1:-avx512}
case $VARIANT in
  avx2)   AVX512=OFF ;;
  avx512) AVX512=ON  ;;
  *) echo "usage: $0 [avx2|avx512]"; exit 1 ;;
esac
cd $FFT_SRC/heffte
d=build-trace-$VARIANT; rm -rf $d && mkdir $d && cd $d
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DHeffte_ENABLE_TRACING=ON \
  -DHeffte_ENABLE_FFTW=ON  -DFFTW_ROOT=$FFT_PREFIX \
  -DHeffte_ENABLE_MKL=OFF \
  -DHeffte_ENABLE_CUDA=OFF \
  -DHeffte_ENABLE_AVX=ON -DHeffte_ENABLE_AVX512=$AVX512 \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CXX_FLAGS="-DHeffte_ENABLE_TRACING" \
  > cmake.log 2>&1
make -j16 speed3d_c2c > build.log 2>&1
echo "traced speed3d_c2c($VARIANT): $(pwd)/benchmarks/speed3d_c2c"
