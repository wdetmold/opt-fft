/* fftw1d_custom: FFTW's OWN generator (genfft) asked for a monolithic straight-line
 * codelet at each exact size -- a configuration no shipped FFTW provides. In 1D this is
 * the natural form: the genfft n1_<L> codelet IS a 1D DFT, no axis passes, no transposes.
 *
 * Reported as its OWN column, never as a stock library: it answers "how good could
 * library-generated code be", not "what a user gets".
 *
 * Two compile-time variants of the SAME generated codelet (genfft_shim.h):
 *   default        R = double   split arrays, scalar DAG + compiler autovectorization
 *   -DCUSTOM_SOA   R = v8       SoA batch-lane: 8 transforms per vector lane-slot, the
 *                               layout the survey names as the top under-used lever
 * The codelet natively takes (v, ivs, ovs), so the batch is expressed inside one call.
 * Both export fft1d_chain: convert to split ONCE, run all m steps split, convert back.
 * Sizes are limited to what genfft can emit as straight-line code (1024+ is infeasible).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#ifdef CUSTOM_SOA
typedef double v8 __attribute__((vector_size(64), aligned(64)));
#define R v8
#define LANES 8
#else
#define LANES 1
#endif
#include "genfft_shim.h"
#include "codelets/n1_13.c"
#include "codelets/n1_31.c"
#include "codelets/n1_32.c"
#include "codelets/n1_60.c"
#include "codelets/n1_64.c"
#include "codelets/n1_128.c"

typedef void (*codelet_fn)(const R *ri, const R *ii, R *ro, R *io,
                           stride is, stride os, INT v, INT ivs, INT ovs);
static codelet_fn pick(int L)
{
    switch (L) {
    case 13: return n1_13;  case 31: return n1_31;  case 32: return n1_32;
    case 60: return n1_60;  case 64: return n1_64;  case 128: return n1_128;
    }
    return 0;
}

struct fft1d_plan {
    int L, batch;
    size_t g;                 /* R-groups per buffer = L*batch/LANES */
    codelet_fn cod;
    R *re, *im, *ro, *io;
};

const char *fft1d_name(void)
{
#ifdef CUSTOM_SOA
    return "fftw1d_custom_soa";
#else
    return "fftw1d_custom";
#endif
}
const char *fft1d_description(void)
{
#ifdef CUSTOM_SOA
    return "genfft monolithic codelet, SoA 8-transform batch-lane split-complex";
#else
    return "genfft monolithic codelet, split arrays, scalar DAG + autovec";
#endif
}
int fft1d_supports(int L)
{
    if (!pick(L)) return 0;
#ifdef CUSTOM_SOA
    return 1;   /* batch multiple checked in create */
#else
    return 1;
#endif
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!pick(L)) return NULL;
    if (batch % LANES) return NULL;              /* SoA needs whole lane groups */
    fft1d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch; p->cod = pick(L);
    p->g = (size_t)L * batch / LANES;
    if (posix_memalign((void **)&p->re, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->im, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->ro, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->io, 64, p->g * sizeof(R))) { free(p); return NULL; }
    return p;
}

/* interleaved (b,L) complex  <->  split, SoA-grouped when LANES>1 */
static void load_split(fft1d_plan *p, const double _Complex *z)
{
    const double *v = (const double *)z;
    double *re = (double *)p->re, *im = (double *)p->im;
    int L = p->L, B = p->batch;
#if LANES > 1
    for (int g = 0; g < B / LANES; ++g)
        for (int e = 0; e < L; ++e)
            for (int l = 0; l < LANES; ++l) {
                size_t src = ((size_t)(g * LANES + l)) * L + e;
                size_t dst = ((size_t)g * L + e) * LANES + l;
                re[dst] = v[2 * src]; im[dst] = v[2 * src + 1];
            }
#else
    for (size_t i = 0; i < (size_t)L * B; ++i) { re[i] = v[2*i]; im[i] = v[2*i+1]; }
#endif
}
static void store_split(fft1d_plan *p, const R *sre, const R *sim, double _Complex *z)
{
    double *v = (double *)z;
    const double *re = (const double *)sre, *im = (const double *)sim;
    int L = p->L, B = p->batch;
#if LANES > 1
    for (int g = 0; g < B / LANES; ++g)
        for (int e = 0; e < L; ++e)
            for (int l = 0; l < LANES; ++l) {
                size_t dst = ((size_t)(g * LANES + l)) * L + e;
                size_t src = ((size_t)g * L + e) * LANES + l;
                v[2 * dst] = re[src]; v[2 * dst + 1] = im[src];
            }
#else
    for (size_t i = 0; i < (size_t)L * B; ++i) { v[2*i] = re[i]; v[2*i+1] = im[i]; }
#endif
}

/* one batched transform on the split planes: the codelet's own (v, ivs, ovs) does the batch */
static void run_batch(fft1d_plan *p, const R *ri, const R *ii, R *ro, R *io)
{
    INT v = p->batch / LANES;                    /* transforms (groups) in this call */
    p->cod(ri, ii, ro, io, 1, 1, v, (INT)p->L, (INT)p->L);
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    load_split(p, in);
    run_batch(p, p->re, p->im, p->ro, p->io);
    store_split(p, p->ro, p->io, out);
}

void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    R *cre, *cim;
    if (posix_memalign((void **)&cre, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&cim, 64, p->g * sizeof(R))) return;
    R *hr = p->re, *hi = p->im;
    load_split(p, c);
    memcpy(cre, p->re, p->g * sizeof(R)); memcpy(cim, p->im, p->g * sizeof(R));
    p->re = hr; p->im = hi;
    load_split(p, x0);
    double *fr = (double *)cre, *fi = (double *)cim;
    size_t n = p->g * LANES;
    for (int s = 0; s < m; ++s) {
        run_batch(p, p->re, p->im, p->ro, p->io);
        double *zr = (double *)p->ro, *zi = (double *)p->io;
        double *xr = (double *)p->re, *xi = (double *)p->im;
        for (size_t i = 0; i < n; ++i) {
            double a = zr[i] + fr[i], b = zi[i] + fi[i];
            double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
            xr[i] = a * sc; xi[i] = b * sc;
        }
    }
    store_split(p, p->re, p->im, final_out);
    free(cre); free(cim);
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->re); free(p->im); free(p->ro); free(p->io); free(p);
}
