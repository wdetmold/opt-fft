/* SOTA reference: FFTW3 through the GURU SPLIT-ARRAY interface -- the strongest way to
 * drive FFTW on this hardware. The standard interleaved interface pays shuffle traffic
 * to pair re/im on every butterfly; the guru split API lets FFTW work on separate
 * re/im arrays, the same layout every winning hand-written kernel here uses.
 *
 * To be the honest strongest baseline it also exports fft3d_chain: deinterleave ONCE,
 * run the whole m-step chain split (FFT + exact map on split data), interleave once --
 * the same conversion amortization our fused kernels enjoy. Without this, per-step
 * conversions would bury the split advantage and the baseline would be a strawman.
 *
 * Planning (FFTW_MEASURE) happens in fft3d_create(), excluded from timing as always.
 */
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../fft3d_api.h"

struct fft3d_plan {
    fftw_plan plan;              /* guru split, in-place on (ri,ii) -> (ro,io) */
    int L, batch;
    size_t n;                    /* batch * L^3 */
    double *ri, *ii, *ro, *io;   /* 64-byte aligned split buffers */
};

const char *fft3d_name(void) { return "fftw3_guru"; }
const char *fft3d_description(void)
{ return "FFTW 3.3.10 guru split-array dft, FFTW_MEASURE, fused split chain"; }
int fft3d_supports(int L) { return L > 0; }

fft3d_plan *fft3d_create(int L, int batch)
{
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;
    p->n = (size_t)batch * L * L * L;
    p->ri = fftw_alloc_real(p->n); p->ii = fftw_alloc_real(p->n);
    p->ro = fftw_alloc_real(p->n); p->io = fftw_alloc_real(p->n);
    if (!p->ri || !p->ii || !p->ro || !p->io) { free(p); return NULL; }

    fftw_iodim dims[3], many[1];
    int V = L * L * L;
    dims[0].n = L; dims[0].is = L * L; dims[0].os = L * L;
    dims[1].n = L; dims[1].is = L;     dims[1].os = L;
    dims[2].n = L; dims[2].is = 1;     dims[2].os = 1;
    many[0].n = batch; many[0].is = V; many[0].os = V;
    /* Forward transform: for split arrays FFTW has no sign argument -- passing
     * (ri, ii, ro, io) in this order IS the forward (FFTW_FORWARD) transform. */
    p->plan = fftw_plan_guru_split_dft(3, dims, 1, many,
                                       p->ri, p->ii, p->ro, p->io, FFTW_MEASURE);
    if (!p->plan) { free(p); return NULL; }
    return p;
}

static void deint(const double _Complex *z, double *re, double *im, size_t n)
{
    const double *v = (const double *)z;
    for (size_t i = 0; i < n; ++i) { re[i] = v[2*i]; im[i] = v[2*i+1]; }
}
static void inter(const double *re, const double *im, double _Complex *z, size_t n)
{
    double *v = (double *)z;
    for (size_t i = 0; i < n; ++i) { v[2*i] = re[i]; v[2*i+1] = im[i]; }
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    deint(in, p->ri, p->ii, p->n);
    fftw_execute_split_dft(p->plan, p->ri, p->ii, p->ro, p->io);
    inter(p->ro, p->io, out, p->n);
}

/* The whole graded chain, split throughout: z = FFT3(x) + c; x <- z / (1 + |z|). */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const double *cv = (const double *)c;
    deint(x0, p->ri, p->ii, p->n);
    for (int s = 0; s < m; ++s) {
        fftw_execute_split_dft(p->plan, p->ri, p->ii, p->ro, p->io);
        for (size_t i = 0; i < p->n; ++i) {
            double zr = p->ro[i] + cv[2*i];
            double zi = p->io[i] + cv[2*i+1];
            double sc = 1.0 / (1.0 + sqrt(zr*zr + zi*zi));
            p->ri[i] = zr * sc;
            p->ii[i] = zi * sc;
        }
    }
    inter(p->ri, p->ii, final_out, p->n);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    fftw_destroy_plan(p->plan);
    fftw_free(p->ri); fftw_free(p->ii); fftw_free(p->ro); fftw_free(p->io);
    free(p);
}
