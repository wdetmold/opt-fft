/* SOTA baseline: split the batch across GPUs, cuFFT on each.
 *
 * This is the realistic library baseline for a large batch, and the one a practitioner would
 * actually write: independent volumes need no communication, so the only question is whether
 * cuFFT is fast per GPU. Any distributed decomposition the panel builds has to beat THIS,
 * not just beat single-GPU cuFFT.
 *
 * Its weakness is structural and worth stating: with a batch smaller than the GPU count it
 * leaves GPUs idle, and at B=1 it uses exactly one. That is the regime where a slab or
 * pencil decomposition must earn the cost of its redistribution.
 */
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_mgpu_api.h"

#define MAXG 16

struct fft3d_mgpu_plan {
    int L, batch, ngpus;
    int dev[MAXG];
    int nvol[MAXG];
    long off[MAXG];
    double2 *in[MAXG];
    double2 *out[MAXG];
    cufftHandle plan[MAXG];
    int has_plan[MAXG];
};

extern "C" const char *fft3d_mgpu_name(void) { return "cufft_batchsplit"; }
extern "C" const char *fft3d_mgpu_description(void)
{ return "batch split across GPUs, cuFFT Z2Z per GPU, no communication"; }
extern "C" int fft3d_mgpu_supports(int L, int ngpus)
{ return L > 0 && ngpus > 0 && ngpus <= MAXG; }

extern "C" fft3d_mgpu_plan *fft3d_mgpu_create(int L, int batch, int ngpus, const int *devices)
{
    fft3d_mgpu_plan *p = (fft3d_mgpu_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch; p->ngpus = ngpus;
    const size_t vol = (size_t)L * L * L;
    const int dist = L * L * L;
    int n[3] = { L, L, L };

    long given = 0;
    for (int g = 0; g < ngpus; ++g) {
        p->dev[g] = devices[g];
        p->nvol[g] = batch / ngpus + (g < batch % ngpus ? 1 : 0);
        p->off[g] = given;
        given += p->nvol[g];
        if (p->nvol[g] == 0) continue;

        if (cudaSetDevice(p->dev[g]) != cudaSuccess) return NULL;
        size_t nb = (size_t)p->nvol[g] * vol * sizeof(double2);
        if (cudaMalloc((void **)&p->in[g], nb) != cudaSuccess) return NULL;
        if (cudaMalloc((void **)&p->out[g], nb) != cudaSuccess) return NULL;
        if (cufftPlanMany(&p->plan[g], 3, n, NULL, 1, dist, NULL, 1, dist,
                          CUFFT_Z2Z, p->nvol[g]) != CUFFT_SUCCESS) return NULL;
        p->has_plan[g] = 1;
    }
    return p;
}

extern "C" void fft3d_mgpu_upload(fft3d_mgpu_plan *p, const double2 *host)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int g = 0; g < p->ngpus; ++g) {
        if (p->nvol[g] == 0) continue;
        cudaSetDevice(p->dev[g]);
        cudaMemcpy(p->in[g], host + (size_t)p->off[g] * vol,
                   (size_t)p->nvol[g] * vol * sizeof(double2), cudaMemcpyHostToDevice);
    }
}

extern "C" void fft3d_mgpu_execute(fft3d_mgpu_plan *p)
{
    for (int g = 0; g < p->ngpus; ++g) {
        if (p->nvol[g] == 0) continue;
        cudaSetDevice(p->dev[g]);
        /* out-of-place, so repeated calls stay idempotent */
        cufftExecZ2Z(p->plan[g], (cufftDoubleComplex *)p->in[g],
                     (cufftDoubleComplex *)p->out[g], CUFFT_FORWARD);
    }
}

extern "C" void fft3d_mgpu_download(fft3d_mgpu_plan *p, double2 *host)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int g = 0; g < p->ngpus; ++g) {
        if (p->nvol[g] == 0) continue;
        cudaSetDevice(p->dev[g]);
        cudaMemcpy(host + (size_t)p->off[g] * vol, p->out[g],
                   (size_t)p->nvol[g] * vol * sizeof(double2), cudaMemcpyDeviceToHost);
    }
}

extern "C" void fft3d_mgpu_destroy(fft3d_mgpu_plan *p)
{
    if (!p) return;
    for (int g = 0; g < p->ngpus; ++g) {
        cudaSetDevice(p->dev[g]);
        if (p->has_plan[g]) cufftDestroy(p->plan[g]);
        cudaFree(p->in[g]);
        cudaFree(p->out[g]);
    }
    free(p);
}
