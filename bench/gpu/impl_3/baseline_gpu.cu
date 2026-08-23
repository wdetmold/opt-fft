/* Library-free floor for the GPU competition: row-column with a dense O(L) sum per output
 * element, i.e. O(L^4) per volume per axis and no FFT factorization at all. The GPU analogue
 * of the CPU phase's baseline_matrix, and it exists for the same two reasons: it validates
 * the harness end to end before the panel delivers anything, and it gives every later result
 * something library-free to be measured against.
 *
 * One thread per output element; each thread walks one line. Correct, trivially parallel,
 * and deliberately not clever.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

struct fft3d_gpu_plan {
    int L, batch;
    double2 *twiddle;   /* device: W[j] = exp(-2 pi i j / L) */
    double2 *scratch;   /* device: one full batch of intermediate */
};

/* Treat the volume as (outer, L, inner) for the axis being transformed. */
__global__ void dft_axis(const double2 *__restrict__ in, double2 *__restrict__ out,
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
    double2 r;
    r.x = re;
    r.y = im;
    out[o * (long)L * inner + k * inner + i] = r;
}

extern "C" const char *fft3d_gpu_name(void) { return "baseline_gpu"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "row-column dense O(L) per output, no factorization"; }
extern "C" int fft3d_gpu_supports(int L) { return L > 0; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    double2 *h = (double2 *)malloc((size_t)L * sizeof(double2));
    if (!h) { free(p); return NULL; }
    for (int j = 0; j < L; ++j) {
        double phase = -2.0 * M_PI * (double)j / (double)L;
        h[j].x = cos(phase);
        h[j].y = sin(phase);
    }
    if (cudaMalloc((void **)&p->twiddle, (size_t)L * sizeof(double2)) != cudaSuccess) {
        free(h); free(p); return NULL;
    }
    cudaMemcpy(p->twiddle, h, (size_t)L * sizeof(double2), cudaMemcpyHostToDevice);
    free(h);

    size_t elems = (size_t)L * L * L * batch;
    if (cudaMalloc((void **)&p->scratch, elems * sizeof(double2)) != cudaSuccess) {
        cudaFree(p->twiddle); free(p); return NULL;
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const int L = p->L;
    const long vol = (long)L * L * L;
    const long total = vol * p->batch;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;

    /* axis 0 (slowest): outer = batch, inner = L*L */
    dft_axis<<<blocks, threads>>>(in, p->scratch, p->twiddle, L, p->batch, (long)L * L);
    /* axis 1: outer = batch*L, inner = L */
    dft_axis<<<blocks, threads>>>(p->scratch, out, p->twiddle, L, (long)p->batch * L, L);
    /* axis 2 (fastest): outer = batch*L*L, inner = 1 */
    dft_axis<<<blocks, threads>>>(out, p->scratch, p->twiddle, L, (long)p->batch * L * L, 1);
    cudaMemcpyAsync(out, p->scratch, (size_t)total * sizeof(double2), cudaMemcpyDeviceToDevice);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    cudaFree(p->twiddle);
    cudaFree(p->scratch);
    free(p);
}
