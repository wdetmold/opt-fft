/* fftw3_custom: FFTW's OWN code generator (genfft) asked for direct codelets at the
 * prime sizes its shipped set lacks (17, 23, 31) -- the "how good could FFTW be if
 * someone did the work" baseline. Reported in its own column: no packaged FFTW ships
 * these codelets, so this is NOT a stock-library number.
 *
 * Two compile-time variants of the SAME generated codelets (see genfft_shim.h):
 *   default        R = double        split arrays, scalar DAG + compiler autovec
 *   -DCUSTOM_SOA   R = v8 (8 doubles) SoA batch-lane: 8 volumes per vector lane-slot,
 *                                     the layout our winning kernels use -- built to
 *                                     answer whether genfft output enjoys the same
 *                                     split-complex SoA benefit (Will's question).
 * Both export fft3d_chain (split/SoA layout held across the whole chain, converted once).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft3d_api.h"

#ifdef CUSTOM_SOA
typedef double v8 __attribute__((vector_size(64), aligned(64)));
#define R v8
#define LANES 8
#else
#define LANES 1
#endif
#include "genfft_shim.h"
#include "codelets/n1_10.c"
#include "codelets/n1_12.c"
#include "codelets/n1_15.c"
#include "codelets/n1_17.c"
#include "codelets/n1_20.c"
#include "codelets/n1_23.c"
#include "codelets/n1_25.c"
#include "codelets/n1_27.c"
#include "codelets/n1_31.c"
#include "codelets/n1_32.c"
#include "codelets/n1_40.c"
#include "codelets/n1_50.c"

typedef void (*codelet_fn)(const R *ri, const R *ii, R *ro, R *io,
                           stride is, stride os, INT v, INT ivs, INT ovs);
static codelet_fn pick(int L)
{
    switch (L) {
    case 10: return n1_10;  case 12: return n1_12;  case 15: return n1_15;
    case 17: return n1_17;  case 20: return n1_20;  case 23: return n1_23;
    case 25: return n1_25;  case 27: return n1_27;  case 31: return n1_31;
    case 32: return n1_32;  case 40: return n1_40;  case 50: return n1_50;
    }
    return 0;
}

struct fft3d_plan {
    int L, batch;
    size_t n;               /* batch * L^3 complex elements */
    size_t g;               /* element groups: n / LANES     */
    codelet_fn cod;
    R *re, *im, *sre, *sim; /* split (or SoA-split) work + scratch planes */
};

const char *fft3d_name(void)
{
#ifdef CUSTOM_SOA
    return "fftw3_custom_soa";
#else
    return "fftw3_custom";
#endif
}
const char *fft3d_description(void)
{
#ifdef CUSTOM_SOA
    return "genfft custom codelets (17/23/31), SoA 8-volume batch-lane split-complex";
#else
    return "genfft custom codelets (17/23/31), split arrays, scalar DAG + autovec";
#endif
}
int fft3d_supports(int L) { return pick(L) != 0; }

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch % LANES) return NULL;
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;
    p->n = (size_t)batch * L * L * L;
    p->g = p->n / LANES;
    p->cod = pick(L);
    if (posix_memalign((void **)&p->re,  64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->im,  64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->sre, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&p->sim, 64, p->g * sizeof(R))) { free(p); return NULL; }
    return p;
}

/* interleaved complex <-> split (or SoA batch-lane split) */
static void load_split(fft3d_plan *p, const double _Complex *z)
{
    const double *v = (const double *)z;
    int L = p->L; size_t V = (size_t)L * L * L;
#ifdef CUSTOM_SOA
    double *re = (double *)p->re, *im = (double *)p->im;
    for (size_t e = 0; e < V; ++e)
        for (int g = 0; g < p->batch / LANES; ++g)
            for (int l = 0; l < LANES; ++l) {
                size_t src = ((size_t)(g * LANES + l)) * V + e;
                size_t dst = (e + (size_t)g * V) * LANES + l;
                re[dst] = v[2 * src]; im[dst] = v[2 * src + 1];
            }
#else
    double *re = (double *)p->re, *im = (double *)p->im;
    for (size_t i = 0; i < p->n; ++i) { re[i] = v[2 * i]; im[i] = v[2 * i + 1]; }
#endif
}
static void store_split(fft3d_plan *p, double _Complex *z)
{
    double *v = (double *)z;
    int L = p->L; size_t V = (size_t)L * L * L;
#ifdef CUSTOM_SOA
    const double *re = (const double *)p->re, *im = (const double *)p->im;
    for (size_t e = 0; e < V; ++e)
        for (int g = 0; g < p->batch / LANES; ++g)
            for (int l = 0; l < LANES; ++l) {
                size_t dst = ((size_t)(g * LANES + l)) * V + e;
                size_t src = (e + (size_t)g * V) * LANES + l;
                v[2 * dst] = re[src]; v[2 * dst + 1] = im[src];
            }
#else
    const double *re = (const double *)p->re, *im = (const double *)p->im;
    for (size_t i = 0; i < p->n; ++i) { v[2 * i] = re[i]; v[2 * i + 1] = im[i]; }
#endif
}

/* one full 3D transform on the split/SoA planes: 3 strided codelet passes */
static void fft3(fft3d_plan *p)
{
    int L = p->L; size_t V = (size_t)L * L * L;
    size_t vols = p->g / V;              /* volume-groups resident in R lanes */
    /* z axis: stride 1, pencils are contiguous runs           */
    p->cod(p->re, p->im, p->sre, p->sim, 1, 1, (INT)(vols * L * L), L, L);
    /* y axis: stride L                                        */
    for (size_t g = 0; g < vols; ++g)
        for (int x = 0; x < L; ++x)
            p->cod(p->sre + g * V + (size_t)x * L * L, p->sim + g * V + (size_t)x * L * L,
                   p->re  + g * V + (size_t)x * L * L, p->im  + g * V + (size_t)x * L * L,
                   L, L, L, 1, 1);
    /* x axis: stride L*L                                      */
    for (size_t g = 0; g < vols; ++g)
        p->cod(p->re + g * V, p->im + g * V, p->sre + g * V, p->sim + g * V,
               (INT)L * L, (INT)L * L, L * L, 1, 1);
    /* result now in sre/sim; swap into primary */
    R *t;
    t = p->re; p->re = p->sre; p->sre = t;
    t = p->im; p->im = p->sim; p->sim = t;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    load_split(p, in);
    fft3(p);
    store_split(p, out);
}

void fft3d_chain(fft3d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    /* c in matching layout, loaded once */
    R *cre, *cim;
    if (posix_memalign((void **)&cre, 64, p->g * sizeof(R)) ||
        posix_memalign((void **)&cim, 64, p->g * sizeof(R))) return;
    R *hre = p->re, *him = p->im;
    load_split(p, c);
    memcpy(cre, p->re, p->g * sizeof(R)); memcpy(cim, p->im, p->g * sizeof(R));
    p->re = hre; p->im = him;
    load_split(p, x0);
    double *fre = (double *)cre, *fim = (double *)cim;
    for (int s = 0; s < m; ++s) {
        fft3(p);
        double *zr = (double *)p->re, *zi = (double *)p->im;
        size_t nn = p->g * LANES;
        for (size_t i = 0; i < nn; ++i) {
            double a = zr[i] + fre[i], b = zi[i] + fim[i];
            double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
            zr[i] = a * sc; zi[i] = b * sc;
        }
    }
    store_split(p, final_out);
    free(cre); free(cim);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->re); free(p->im); free(p->sre); free(p->sim); free(p);
}
