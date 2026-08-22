/* SOTA reference: cuFFT, batched 3D Z2Z via cufftPlanMany -- the library the competition
 * has to beat. Planning happens in create() and is therefore excluded from the timed
 * region, exactly as FFTW's planning is on the CPU side. */
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

struct fft3d_gpu_plan {
    cufftHandle handle;
    int L, batch;
};

extern "C" const char *fft3d_gpu_name(void) { return "cufft"; }
extern "C" const char *fft3d_gpu_description(void) { return "cuFFT 11.0 cufftPlanMany Z2Z, batched"; }
extern "C" int fft3d_gpu_supports(int L) { return L > 0; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    int n[3] = { L, L, L };
    const int dist = L * L * L;
    if (cufftPlanMany(&p->handle, 3, n, NULL, 1, dist, NULL, 1, dist, CUFFT_Z2Z, batch)
        != CUFFT_SUCCESS) {
        free(p);
        return NULL;
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    cufftExecZ2Z(p->handle, (cufftDoubleComplex *)in, (cufftDoubleComplex *)out, CUFFT_FORWARD);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    cufftDestroy(p->handle);
    free(p);
}
