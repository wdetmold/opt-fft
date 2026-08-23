// Experimentation-only MKL reference (never shipped in graded path)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include "mkl_dfti.h"
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }

static void map_inter(double* z, const double* c, long n){ // interleaved complex, n complex
    const __m512d vone = _mm512_set1_pd(1.0);
    const __m512d vhalf = _mm512_set1_pd(0.5);
    const __m512d v15 = _mm512_set1_pd(1.5);
    const __m512d vtiny = _mm512_set1_pd(1e-300);
    for (long i = 0; i < n; i += 4){ // 4 complex = 8 doubles
        __m512d v = _mm512_loadu_pd(z + 2*i) + _mm512_loadu_pd(c + 2*i);
        __m512d sw = _mm512_permute_pd(v, 0x55);
        __m512d r2 = _mm512_max_pd(v*v + sw*sw, vtiny);
        __m512d e = _mm512_rsqrt14_pd(r2);
        __m512d h = r2 * vhalf;
        e = e * (v15 - h*e*e);
        e = e * (v15 - h*e*e);
        __m512d s = _mm512_div_pd(vone, vone + r2*e);
        _mm512_storeu_pd(z + 2*i, v*s);
    }
}
int main(int argc, char** argv){
    long L = atol(argv[1]), B = atol(argv[2]), m = atol(argv[3]);
    long n = L*L*L;
    double* x = aligned_alloc(64, (size_t)B*n*16);
    double* c = aligned_alloc(64, (size_t)B*n*16);
    for (long i = 0; i < B*n*2; ++i){ x[i] = 0.1 + 1e-6*(i%37); c[i] = 0.01; }
    DFTI_DESCRIPTOR_HANDLE h;
    MKL_LONG sizes[3] = {L, L, L};
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 3, sizes);
    DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)B);
    DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)n);
    DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)n);
    DftiCommitDescriptor(h);
    // warm
    DftiComputeForward(h, x);
    map_inter(x, c, B*n);
    double best = 1e9;
    for (int r = 0; r < 6; ++r){
        double t0 = now();
        for (long it = 0; it < m; ++it){
            DftiComputeForward(h, x);
            map_inter(x, c, B*n);
        }
        double t1 = now();
        if (t1-t0 < best) best = t1-t0;
    }
    printf("MKL L=%ld B=%ld m=%ld: %8.1f us/vol-iter\n", L, B, m, best/(B*m)*1e6);
    return 0;
}
