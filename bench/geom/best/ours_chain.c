/* Head-to-head adapter: OUR kernels driving the rivals' exact graded chain.
 *
 *   step:  z = FFT(state);  state = (z + c) / (1 + |z + c|)
 *
 * The FFT is fft3d_best (create() races all competition entries on this machine); the map
 * is a plain separate pass -- deliberately NOT fused, because that is the honest current
 * state of our code versus theirs, which fuses the map into its passes.
 * Returns the state after step 1 in `one` and after step m in `final`, exactly like the
 * rivals' _run(), so the two implementations can be cross-checked element by element.
 */
#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fft3d_best.h"

void ours_chain(int L, long B, long m,
                const double _Complex *x0, const double _Complex *c,
                double _Complex *one, double _Complex *final_out)
{
    const long n = B * (long)L * L * L;
    fft3d_best_plan *plan = fft3d_best_create(L, (int)B);
    if (!plan) return;
    /* The kernel ABI requires 64-byte alignment; plain malloc gives 16 and some kernels
       use aligned 512-bit loads -- that was a real segfault at the larger sizes. */
    double _Complex *cur = NULL, *z = NULL;
    if (posix_memalign((void **)&cur, 64, (size_t)n * sizeof *cur) != 0 ||
        posix_memalign((void **)&z,   64, (size_t)n * sizeof *z) != 0) {
        free(cur); free(z); fft3d_best_destroy(plan); return;
    }
    memcpy(cur, x0, (size_t)n * sizeof *cur);

    for (long s = 0; s < m; ++s) {
        fft3d_best_execute(plan, cur, z);
        /* Vectorizable map: plain sqrt on squares. Scalar cabs (libm hypot, ~40 cyc/elem)
           dwarfed the FFT at 4e8 map points and measured the glue, not the kernels. */
        {
            const double *zr = (const double *)z;
            const double *cr = (const double *)c;
            double *o = (double *)cur;
            for (long i = 0; i < n; ++i) {
                double re = zr[2*i] + cr[2*i];
                double im = zr[2*i+1] + cr[2*i+1];
                double scale = 1.0 / (1.0 + sqrt(re*re + im*im));
                o[2*i] = re * scale;
                o[2*i+1] = im * scale;
            }
        }
        if (s == 0) memcpy(one, cur, (size_t)n * sizeof *cur);
    }
    memcpy(final_out, cur, (size_t)n * sizeof *cur);
    free(cur);
    free(z);
    fft3d_best_destroy(plan);
}

/* Same, but with the plan made once outside the timed region (theirs precomputes in setup
 * too). Timing calls this. */
static fft3d_best_plan *g_plan[65];
void ours_setup(int L, long B) { if (!g_plan[L]) g_plan[L] = fft3d_best_create(L, (int)B); }
void ours_run(int L, long B, long m,
              const double _Complex *x0, const double _Complex *c,
              double _Complex *one, double _Complex *final_out)
{
    const long n = B * (long)L * L * L;
    fft3d_best_plan *plan = g_plan[L];
    if (!plan) return;
    static double _Complex *cur = 0, *z = 0; static long cap = 0;
    if (n > cap) {
        free(cur); free(z); cur = z = 0;
        if (posix_memalign((void **)&cur, 64, (size_t)n*16) != 0 ||
            posix_memalign((void **)&z,   64, (size_t)n*16) != 0) return;
        cap = n;
    }
    memcpy(cur, x0, (size_t)n * 16);
    for (long s = 0; s < m; ++s) {
        fft3d_best_execute(plan, cur, z);
        /* Vectorizable map: plain sqrt on squares. Scalar cabs (libm hypot, ~40 cyc/elem)
           dwarfed the FFT at 4e8 map points and measured the glue, not the kernels. */
        {
            const double *zr = (const double *)z;
            const double *cr = (const double *)c;
            double *o = (double *)cur;
            for (long i = 0; i < n; ++i) {
                double re = zr[2*i] + cr[2*i];
                double im = zr[2*i+1] + cr[2*i+1];
                double scale = 1.0 / (1.0 + sqrt(re*re + im*im));
                o[2*i] = re * scale;
                o[2*i+1] = im * scale;
            }
        }
        if (s == 0) memcpy(one, cur, (size_t)n * 16);
    }
    memcpy(final_out, cur, (size_t)n * 16);
}
