/* Library-free floor for the multi-GPU competition: split the BATCH across GPUs and run a
 * naive O(L^4) row-column transform on each.
 *
 * There is deliberately no communication at all. Independent volumes are embarrassingly
 * parallel, so for a large batch this is the obvious first thing to try, and it is the
 * honest baseline every cleverer decomposition has to beat. What it cannot do is help a
 * SMALL batch of LARGE volumes -- with B < ngpus some GPUs sit idle, and with B=1 seven of
 * eight do nothing. That is exactly the regime where a slab or pencil decomposition has to
 * earn its redistribution, and where this floor should lose badly.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "../fft3d_mgpu_api.h"

#define MAXG 16

struct fft3d_mgpu_plan {
    int L, batch, ngpus;
    int dev[MAXG];
    int nvol[MAXG];        /* volumes owned by each GPU */
    long off[MAXG];        /* first volume index owned by each GPU */
    double2 *in[MAXG];
    double2 *out[MAXG];
    double2 *scratch[MAXG];
    double2 *tw[MAXG];
};

__global__ void mg_dft_axis(const double2 *__restrict__ in, double2 *__restrict__ out,
                            const double2 *__restrict__ w, int L, long outer, long inner)
{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = outer * (long)L * inner;
    if (tid >= total) return;
    long i = tid % inner;
    long k = (tid / inner) % L;
    long o = tid / (inner * (long)L);
    const double2 *base = in + o * (long)L * inner + i;
    double re = 0.0, im = 0.0;
    for (int j = 0; j < L; ++j) {
        double2 x = base[(long)j * inner];
        double2 t = w[(int)(((long)j * k) % L)];
        re += x.x * t.x - x.y * t.y;
        im += x.x * t.y + x.y * t.x;
    }
    double2 r; r.x = re; r.y = im;
    out[o * (long)L * inner + k * inner + i] = r;
}

extern "C" const char *fft3d_mgpu_name(void) { return "baseline_mgpu"; }
extern "C" const char *fft3d_mgpu_description(void)
{ return "batch split across GPUs, dense O(L) per output, no communication"; }
extern "C" int fft3d_mgpu_supports(int L, int ngpus)
{ return L > 0 && ngpus > 0 && ngpus <= MAXG; }

extern "C" fft3d_mgpu_plan *fft3d_mgpu_create(int L, int batch, int ngpus, const int *devices)
{
    fft3d_mgpu_plan *p = (fft3d_mgpu_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch; p->ngpus = ngpus;
    const size_t vol = (size_t)L * L * L;

    double2 *h = (double2 *)malloc(L * sizeof(double2));
    if (!h) { free(p); return NULL; }
    for (int j = 0; j < L; ++j) {
        double phase = -2.0 * M_PI * (double)j / (double)L;
        h[j].x = cos(phase); h[j].y = sin(phase);
    }

    long given = 0;
    for (int g = 0; g < ngpus; ++g) {
        p->dev[g] = devices[g];
        /* spread the remainder so no GPU is more than one volume behind */
        p->nvol[g] = batch / ngpus + (g < batch % ngpus ? 1 : 0);
        p->off[g] = given;
        given += p->nvol[g];

        if (cudaSetDevice(p->dev[g]) != cudaSuccess) { free(h); return NULL; }
        size_t nb = (size_t)p->nvol[g] * vol * sizeof(double2);
        if (p->nvol[g] > 0) {
            if (cudaMalloc((void **)&p->in[g], nb) != cudaSuccess ||
                cudaMalloc((void **)&p->out[g], nb) != cudaSuccess ||
                cudaMalloc((void **)&p->scratch[g], nb) != cudaSuccess) { free(h); return NULL; }
        }
        if (cudaMalloc((void **)&p->tw[g], L * sizeof(double2)) != cudaSuccess) { free(h); return NULL; }
        cudaMemcpy(p->tw[g], h, L * sizeof(double2), cudaMemcpyHostToDevice);
    }
    free(h);
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
    const int L = p->L;
    const long vol = (long)L * L * L;
    const int threads = 256;
    for (int g = 0; g < p->ngpus; ++g) {
        if (p->nvol[g] == 0) continue;
        cudaSetDevice(p->dev[g]);
        long total = vol * p->nvol[g];
        long blocks = (total + threads - 1) / threads;
        /* x, then y, then z; out-of-place throughout so execute() is idempotent */
        mg_dft_axis<<<blocks, threads>>>(p->in[g], p->scratch[g], p->tw[g], L,
                                        p->nvol[g], (long)L * L);
        mg_dft_axis<<<blocks, threads>>>(p->scratch[g], p->out[g], p->tw[g], L,
                                        (long)p->nvol[g] * L, L);
        mg_dft_axis<<<blocks, threads>>>(p->out[g], p->scratch[g], p->tw[g], L,
                                        (long)p->nvol[g] * L * L, 1);
        cudaMemcpyAsync(p->out[g], p->scratch[g], (size_t)total * sizeof(double2),
                        cudaMemcpyDeviceToDevice);
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
        cudaFree(p->in[g]); cudaFree(p->out[g]); cudaFree(p->scratch[g]); cudaFree(p->tw[g]);
    }
    free(p);
}
