#include <mkl_dfti.h>
#include <stdlib.h>
#include "../fft1d_api.h"
struct fft1d_plan { DFTI_DESCRIPTOR_HANDLE h; int L, batch; };
const char *fft1d_name(void){ return "mkl1d_dfti"; }
const char *fft1d_description(void){ return "oneMKL DFTI 1D, sequential, batched"; }
int fft1d_supports(int L){ return L > 0; }
fft1d_plan *fft1d_create(int L, int batch){
    fft1d_plan *p = malloc(sizeof *p); if(!p) return NULL; p->L=L; p->batch=batch;
    DftiCreateDescriptor(&p->h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)L);
    DftiSetValue(p->h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)batch);
    DftiSetValue(p->h, DFTI_INPUT_DISTANCE, (MKL_LONG)L);
    DftiSetValue(p->h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)L);
    DftiSetValue(p->h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(p->h); return p;
}
void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    DftiComputeForward(p->h, (void*)in, out);
}
void fft1d_destroy(fft1d_plan *p){ if(!p) return; DftiFreeDescriptor(&p->h); free(p); }
