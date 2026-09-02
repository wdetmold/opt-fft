#include <fftw3.h>
#include <stdint.h>
#include <stdlib.h>
#include "../fft1d_api.h"
#ifndef PLAN_FLAG
#define PLAN_FLAG FFTW_ESTIMATE
#endif
#ifndef PLAN_NAME
#define PLAN_NAME "fftw1d_estimate"
#endif
struct fft1d_plan { fftw_plan plan; int L, batch; };
const char *fft1d_name(void){ return PLAN_NAME; }
const char *fft1d_description(void){ return "FFTW 3.3.10 plan_many_dft 1D, " PLAN_NAME; }
int fft1d_supports(int L){ return L > 0; }
fft1d_plan *fft1d_create(int L, int batch){
    fft1d_plan *p = malloc(sizeof *p); if(!p) return NULL; p->L=L; p->batch=batch;
    int n[1] = { L };
    fftw_complex *in = fftw_alloc_complex((size_t)L*batch), *out = fftw_alloc_complex((size_t)L*batch);
    if(!in||!out){ free(p); return NULL; }
    p->plan = fftw_plan_many_dft(1, n, batch, in, NULL, 1, L, out, NULL, 1, L, FFTW_FORWARD, PLAN_FLAG);
    fftw_free(in); fftw_free(out);
    if(!p->plan){ free(p); return NULL; } return p;
}
void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    fftw_execute_dft(p->plan, (fftw_complex*)(void*)(uintptr_t)in, (fftw_complex*)out);
}
void fft1d_destroy(fft1d_plan *p){ if(!p) return; fftw_destroy_plan(p->plan); free(p); }
