# 3D FFT project environment.  Usage:  source env.sh
source /usr/share/modules/init/bash

module load cmake/3.26.3
module load cuda/12.2.1                 # nvcc 12.2, cuFFT 11.0.8
module load openmpi/4.1.5-cuda-11.8     # CUDA-aware OpenMPI + UCX + slurm/pmix
module load mkl/latest                  # oneMKL 2022.0.2 (DFTI, cluster DFT)
module load python/3.12.11              # the system python3.12 ships no dev headers

export FFT_ROOT=/home/lqcd/wdetmold/fft
export FFT_PREFIX=$FFT_ROOT/ext/install
export FFT_SRC=$FFT_ROOT/ext/src

export PATH=$FFT_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$FFT_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$FFT_PREFIX:$CMAKE_PREFIX_PATH
export PKG_CONFIG_PATH=$FFT_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$FFT_PREFIX/include:$CPATH

# GPU targets: A100 (login/a80n0/a81n2) = sm_80, cluster nodes 2080 Ti = sm_75
export CUDA_ARCHS="75;80"   # 2080 Ti (prod) and A100 (a100l/a100r, 8 GPU/node)
export VKFFT_DIR=$FFT_SRC/VkFFT
export MATHDX_DIR=$FFT_ROOT/venv/lib/python3.12/site-packages/nvidia/mathdx/include   # cuFFTDx

# python venv for prototyping / benchmarking (cupy, pyvkfft, pyfftw, ducc0)
if [ -f $FFT_ROOT/venv/bin/activate ]; then
    source $FFT_ROOT/venv/bin/activate
fi
# Must end on a success: this file is sourced by scripts running under `set -e`.
true
