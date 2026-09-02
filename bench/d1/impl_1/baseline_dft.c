#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "../fft1d_api.h"
struct fft1d_plan { int L, batch; double _Complex *w; };
const char *fft1d_name(void){ return "baseline_dft"; }
const char *fft1d_description(void){ return "dense L x L DFT matrix, O(L^2)/vector"; }
int fft1d_supports(int L){ return L > 0; }
fft1d_plan *fft1d_create(int L, int batch){
    fft1d_plan *p = malloc(sizeof *p); if(!p) return NULL; p->L=L; p->batch=batch;
    p->w = malloc((size_t)L*L*sizeof *p->w); if(!p->w){ free(p); return NULL; }
    for(int k=0;k<L;k++) for(int j=0;j<L;j++){ double ph=-2.0*M_PI*((k*j)%L)/L; p->w[(size_t)k*L+j]=cos(ph)+I*sin(ph); }
    return p;
}
void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    int L=p->L;
    for(int b=0;b<p->batch;b++){ const double _Complex *x=in+(size_t)b*L; double _Complex *y=out+(size_t)b*L;
        for(int k=0;k<L;k++){ double _Complex s=0; const double _Complex *wr=p->w+(size_t)k*L; for(int j=0;j<L;j++) s+=wr[j]*x[j]; y[k]=s; } }
}
void fft1d_destroy(fft1d_plan *p){ if(!p) return; free(p->w); free(p); }
