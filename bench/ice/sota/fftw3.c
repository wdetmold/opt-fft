/* SOTA reference: FFTW3, batched 3D c2c via the advanced (plan_many) interface.
 *
 * Compiled three times with different planner effort (-DPLAN_FLAG/-DPLAN_NAME), since
 * FFTW's planner effort is the whole point of comparing against it: ESTIMATE is what a
 * careless caller gets, PATIENT is close to the best the library can do for a fixed size,
 * which is exactly the regime our hand-written implementations are competing in.
 *
 * Planning happens in fft3d_create() and is therefore excluded from the timed region --
 * reported separately as setup_seconds, where PATIENT's cost is plainly visible.
 */
#include <fftw3.h>
#include <stdint.h>
#include <stdlib.h>

#include "../fft3d_api.h"

#ifndef PLAN_FLAG
#define PLAN_FLAG FFTW_ESTIMATE
#endif
#ifndef PLAN_NAME
#define PLAN_NAME "fftw3_estimate"
#endif

struct fft3d_plan {
    fftw_plan plan;
    int L, batch;
};

const char *fft3d_name(void) { return PLAN_NAME; }
const char *fft3d_description(void) { return "FFTW 3.3.10 plan_many_dft, " PLAN_NAME; }
int fft3d_supports(int L) { return L > 0; }

fft3d_plan *fft3d_create(int L, int batch)
{
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    const int n[3] = { L, L, L };
    const int dist = L * L * L;
    /* Plan on scratch with the same 64-byte alignment the driver uses, then execute on
     * the driver's buffers with the new-array interface. */
    fftw_complex *in = fftw_alloc_complex((size_t)dist * batch);
    fftw_complex *out = fftw_alloc_complex((size_t)dist * batch);
    if (!in || !out) { free(p); return NULL; }

    p->plan = fftw_plan_many_dft(3, n, batch,
                                 in, NULL, 1, dist,
                                 out, NULL, 1, dist,
                                 FFTW_FORWARD, PLAN_FLAG);
    fftw_free(in);
    fftw_free(out);
    if (!p->plan) { free(p); return NULL; }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    fftw_execute_dft(p->plan, (fftw_complex *)(void *)(uintptr_t)in, (fftw_complex *)out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    fftw_destroy_plan(p->plan);
    free(p);
}
