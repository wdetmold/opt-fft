#!/bin/bash
set -e
source /usr/share/modules/init/bash
module load cuda/12.2.1
module load openmpi/4.1.5-cuda-11.8
module load python/3.12.11          # system python3.12 has no dev headers; this one does
cd /home/lqcd/wdetmold/fft
rm -rf venv
python3 -m venv venv
source venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q numpy scipy matplotlib pandas
# CPU FFT bindings: ducc0 (pocketfft successor), pyFFTW (FFTW3), mkl_fft (oneMKL DFTI)
python -m pip install -q ducc0 pyfftw mkl-fft
# GPU: cupy -> cuFFT on A100/2080Ti;  pyvkfft -> VkFFT (CUDA backend only, no OpenCL here)
python -m pip install -q cupy-cuda12x
VKFFT_BACKEND=cuda python -m pip install -q pyvkfft
# device-level cuFFTDx headers (fused small 3D FFTs) + multi-GPU cuFFTMp/NVSHMEM
python -m pip install -q nvidia-mathdx nvidia-cufftmp-cu12 nvidia-nvshmem-cu12
# MPI from source against the CUDA-aware OpenMPI 4.1.5 in use here
python -m pip install -q --no-binary=mpi4py mpi4py
echo "--- versions ---"
python - <<'PY'
import importlib
mods = ["numpy","scipy","ducc0","pyfftw","mkl_fft","cupy","pyvkfft.version","mpi4py"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod,'__version__','(ok)')}")
    except Exception as e:
        print(f"  {m:16s} FAILED: {type(e).__name__}: {e}")
PY
