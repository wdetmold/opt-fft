/* SOTA reference: ducc0 (pocketfft's successor, Reinecke), single-threaded.
 *
 * ducc0 needs no planning step at all, which is exactly why it is interesting here:
 * for small fixed sizes it competes with FFTW's PATIENT plans while paying nothing at
 * setup.  The whole batch is handed over as a 4D array with the transform taken over
 * the last three axes, so the library sees the same batched job as everyone else.
 */
#include <complex>
#include <vector>

#include "ducc0/fft/fft.h"

extern "C" {
#include "../fft3d_api.h"
}

struct fft3d_plan {
    int L;
    int batch;
    std::vector<size_t> shape;
    std::vector<size_t> axes;
};

extern "C" const char *fft3d_name(void) { return "ducc0_c2c"; }
extern "C" const char *fft3d_description(void) { return "ducc0 0.41 c2c, no planning, 32 threads"; }
extern "C" int fft3d_supports(int L) { return L > 0; }

extern "C" fft3d_plan *fft3d_create(int L, int batch)
{
    auto *p = new fft3d_plan;
    p->L = L;
    p->batch = batch;
    p->shape = { (size_t)batch, (size_t)L, (size_t)L, (size_t)L };
    p->axes = { 1, 2, 3 };
    return p;
}

extern "C" void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    using cd = std::complex<double>;
    ducc0::cfmav<cd> vin(reinterpret_cast<const cd *>(in), p->shape);
    ducc0::vfmav<cd> vout(reinterpret_cast<cd *>(out), p->shape);
    ducc0::c2c(vin, vout, p->axes, /*forward=*/true, /*fct=*/1.0, /*nthreads=*/32);
}

extern "C" void fft3d_destroy(fft3d_plan *p) { delete p; }
