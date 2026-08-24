/* SOTA reference: Intel oneMKL DFTI, batched 3D c2c, sequential (single-threaded) MKL.
 *
 * MKL's batch is native (DFTI_NUMBER_OF_TRANSFORMS + distances), so this is the library
 * doing the same job in the same layout, with no per-volume call overhead.
 */
#include <mkl_dfti.h>
#include <stdint.h>
#include <stdlib.h>

#include "../fft3d_api.h"

/* Two MKL generations are benchmarked from this one source: the 2022 module build and
 * the current wheel, selected by the name/description macros. */
#ifndef MKL_BACKEND_NAME
#define MKL_BACKEND_NAME "mkl_dfti"
#endif
#ifndef MKL_BACKEND_DESC
#define MKL_BACKEND_DESC "oneMKL 2022.0.2 DFTI, sequential, batched"
#endif

struct fft3d_plan {
    DFTI_DESCRIPTOR_HANDLE handle;
};

const char *fft3d_name(void) { return MKL_BACKEND_NAME; }
const char *fft3d_description(void) { return MKL_BACKEND_DESC; }
int fft3d_supports(int L) { return L > 0; }

fft3d_plan *fft3d_create(int L, int batch)
{
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;

    MKL_LONG dims[3] = { L, L, L };
    MKL_LONG status = DftiCreateDescriptor(&p->handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
    if (status != DFTI_NO_ERROR) { free(p); return NULL; }

    MKL_LONG strides[4] = { 0, (MKL_LONG)L * L, L, 1 };
    const MKL_LONG dist = (MKL_LONG)L * L * L;
    int bad = 0;
    bad |= DftiSetValue(p->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != DFTI_NO_ERROR;
    bad |= DftiSetValue(p->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)batch) != DFTI_NO_ERROR;
    bad |= DftiSetValue(p->handle, DFTI_INPUT_DISTANCE, dist) != DFTI_NO_ERROR;
    bad |= DftiSetValue(p->handle, DFTI_OUTPUT_DISTANCE, dist) != DFTI_NO_ERROR;
    bad |= DftiSetValue(p->handle, DFTI_INPUT_STRIDES, strides) != DFTI_NO_ERROR;
    bad |= DftiSetValue(p->handle, DFTI_OUTPUT_STRIDES, strides) != DFTI_NO_ERROR;
    bad |= DftiCommitDescriptor(p->handle) != DFTI_NO_ERROR;
    if (bad) { DftiFreeDescriptor(&p->handle); free(p); return NULL; }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    DftiComputeForward(p->handle, (void *)(uintptr_t)in, out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    DftiFreeDescriptor(&p->handle);
    free(p);
}
