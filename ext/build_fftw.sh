#!/bin/bash
set -e
source /home/lqcd/wdetmold/fft/env.sh
cd $FFT_SRC
[ -d fftw-3.3.10 ] || tar xzf fftw-3.3.10.tar.gz
# SIMD variants are runtime-dispatched by FFTW, so no -march=native: binaries stay
# usable on the older/newer cluster nodes while still using AVX2+FMA codelets here.
COMMON="--prefix=$FFT_PREFIX --enable-shared --enable-static --enable-threads \
        --enable-openmp --enable-mpi --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 --enable-fma"
CF="-O3 -fno-math-errno -fPIC"
for prec in double single; do
  d=$FFT_SRC/build-fftw-$prec; rm -rf $d; mkdir -p $d; cd $d
  extra=""; [ $prec = single ] && extra="--enable-float"
  ../fftw-3.3.10/configure $COMMON $extra CC=gcc MPICC=mpicc CFLAGS="$CF" > configure.log 2>&1
  make -j16 > build.log 2>&1
  make install >> build.log 2>&1
  echo "FFTW $prec installed"
done
