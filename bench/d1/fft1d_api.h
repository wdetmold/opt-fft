#ifndef FFT1D_API_H
#define FFT1D_API_H
#include <complex.h>
typedef struct fft1d_plan fft1d_plan;
const char *fft1d_name(void);
const char *fft1d_description(void);
int fft1d_supports(int L);
fft1d_plan *fft1d_create(int L, int batch);
void fft1d_execute(fft1d_plan *plan, const double _Complex *in, double _Complex *out);
/* optional weak: own the whole m-step map chain (fused). */
void fft1d_chain(fft1d_plan *plan, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m);
void fft1d_destroy(fft1d_plan *plan);
#endif
