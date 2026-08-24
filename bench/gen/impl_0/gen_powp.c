/* Library-free baseline: row-column with a dense L x L DFT matrix per axis.
 *
 * This is the floor, not a contender.  It exists so the harness can be validated
 * end to end before the implementer panel delivers anything, and so every later
 * result has something to be measured against besides the libraries.  O(L^4) per
 * volume per axis, no FFT factorization at all -- deliberately the same algorithm
 * as slow_dft.dft3d_separable, in C.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>

#include "../fft3d_api.h"

struct fft3d_plan {
    int L, batch;
    double _Complex *w;      /* L x L DFT matrix, row-major */
    double _Complex *tmp;    /* one volume of scratch */
};

const char *fft3d_name(void) { return "gen_powp"; }
const char *fft3d_description(void) { return "prime-power CT with general twiddles: OWN 25,27,50,100 -- the campaign's center of gravity (STUB: dense floor)"; }
int fft3d_supports(int L) { return L == 25 || L == 27 || L == 50 || L == 100; }

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->w = malloc((size_t)L * L * sizeof *p->w);
    p->tmp = malloc((size_t)L * L * L * sizeof *p->tmp);
    if (!p->w || !p->tmp) { free(p->w); free(p->tmp); free(p); return NULL; }

    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            double phase = -2.0 * M_PI * (double)((k * j) % L) / (double)L;
            p->w[(size_t)k * L + j] = cos(phase) + I * sin(phase);
        }
    return p;
}

/* out[k][b][c] = sum_j w[k][j] * in[j][b][c]  -- contract the slowest axis of a
 * (L, mid, fast) block, which is all three axes in turn once the caller permutes. */
static void contract_slowest(const double _Complex *w, int L,
                             const double _Complex *in, double _Complex *out,
                             int inner)
{
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            double _Complex acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    const size_t volume = (size_t)L * L * L;

    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *src = in + (size_t)b * volume;
        double _Complex *dst = out + (size_t)b * volume;

        /* axis 0: slowest, inner = L*L */
        contract_slowest(p->w, L, src, dst, L * L);

        /* axis 1: for each fixed x, the (y, z) plane has y slowest */
        for (int x = 0; x < L; ++x)
            contract_slowest(p->w, L, dst + (size_t)x * L * L,
                             p->tmp + (size_t)x * L * L, L);

        /* axis 2: fastest; treat each length-L row as its own block (inner = 1) */
        for (size_t row = 0; row < volume / (size_t)L; ++row)
            contract_slowest(p->w, L, p->tmp + row * (size_t)L,
                             dst + row * (size_t)L, 1);
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->w);
    free(p->tmp);
    free(p);
}
