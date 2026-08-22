/* Self-contained check of fft3d_best: every geometry against a naive reference.
 *
 * The reference here is the definition, transcribed -- a dense O(L) sum per output element,
 * applied along each axis, with no factorization anywhere. It is deliberately not clever, so
 * that agreement means the fast kernel is right rather than that two clever things agree.
 * (The same kernels are also checked against numpy in the benchmark harness, and numpy's own
 * agreement with the definition is established in python/test_fft3d.py.)
 *
 *   make verify
 */
#define _POSIX_C_SOURCE 200809L

/* M_PI is not in strict POSIX; spell the constant out rather than relax the feature test. */
#define PI_CONST 3.14159265358979323846

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft3d_best.h"

static const int GEOMETRIES[] = { 6, 8, 13, 17, 23, 36, 45, 64 };
static const int NGEOM = (int)(sizeof GEOMETRIES / sizeof GEOMETRIES[0]);

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *aligned_or_die(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
        exit(2);
    }
    return p;
}

/* Deterministic, reproducible test field: a cheap 64-bit mixer, so no library RNG is needed
 * and the same numbers come out on any machine. */
static void fill(double _Complex *x, size_t n, unsigned long long seed)
{
    unsigned long long s = seed * 0x9E3779B97F4A7C15ULL + 0xDEADBEEFULL;
    for (size_t i = 0; i < n; ++i) {
        s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
        s ^= s >> 27; s *= 0x94D049BB133111EBULL;
        s ^= s >> 31;
        double re = (double)((s >> 11) & 0x1FFFFF) / 1048576.0 - 1.0;
        s ^= s << 17;
        double im = (double)((s >> 13) & 0x1FFFFF) / 1048576.0 - 1.0;
        x[i] = re + im * I;
    }
}

/* Contract one axis of a (outer, L, inner) block by the dense DFT matrix. */
static void axis_reference(const double _Complex *in, double _Complex *out,
                           int L, long outer, long inner)
{
    for (long o = 0; o < outer; ++o)
        for (int k = 0; k < L; ++k)
            for (long i = 0; i < inner; ++i) {
                double _Complex acc = 0.0;
                for (int j = 0; j < L; ++j) {
                    double phase = -2.0 * PI_CONST * (double)((j * k) % L) / (double)L;
                    acc += in[(o * L + j) * inner + i] * (cos(phase) + I * sin(phase));
                }
                out[(o * L + k) * inner + i] = acc;
            }
}

static void reference_transform(const double _Complex *in, double _Complex *out,
                                int L, int batch, double _Complex *scratch)
{
    const long vol = (long)L * L * L;
    for (int b = 0; b < batch; ++b) {
        const double _Complex *src = in + (long)b * vol;
        double _Complex *dst = out + (long)b * vol;
        axis_reference(src, dst, L, 1, (long)L * L);      /* x: slowest */
        axis_reference(dst, scratch, L, L, L);            /* y */
        axis_reference(scratch, dst, L, (long)L * L, 1);  /* z: fastest */
    }
}

int main(void)
{
    printf("fft3d_best -- checking every geometry against the definition\n\n");
    printf("  %-4s %-18s %10s %12s   %s\n", "L", "kernel", "rel L2", "us/transform", "verdict");

    int failures = 0;
    for (int g = 0; g < NGEOM; ++g) {
        const int L = GEOMETRIES[g];
        /* One volume is enough for correctness; the harness measures throughput at scale. */
        const int batch = (L <= 17) ? 3 : 1;
        const size_t count = (size_t)L * L * L * batch;
        const size_t bytes = count * sizeof(double _Complex);

        if (!fft3d_best_supports(L)) {
            printf("  %-4d %-18s %10s %12s   NO KERNEL\n", L, "-", "-", "-");
            failures++;
            continue;
        }

        double _Complex *in = aligned_or_die(bytes);
        double _Complex *got = aligned_or_die(bytes);
        double _Complex *want = aligned_or_die(bytes);
        double _Complex *scratch = aligned_or_die((size_t)L * L * L * sizeof(double _Complex));
        fill(in, count, (unsigned long long)L);

        fft3d_best_plan *plan = fft3d_best_create(L, batch);
        if (!plan) {
            printf("  %-4d %-18s %10s %12s   CREATE FAILED\n", L,
                   fft3d_best_kernel_name(L), "-", "-");
            failures++;
            free(in); free(got); free(want); free(scratch);
            continue;
        }

        /* Warm, then time a few calls -- and calling repeatedly also proves the kernel is
           repeatable rather than consuming its own output. */
        fft3d_best_execute(plan, in, got);
        const int reps = (L <= 17) ? 200 : 20;
        double t0 = now_seconds();
        for (int r = 0; r < reps; ++r) fft3d_best_execute(plan, in, got);
        double per_transform = (now_seconds() - t0) / reps / batch;

        reference_transform(in, want, L, batch, scratch);

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double _Complex d = got[i] - want[i];
            num += creal(d) * creal(d) + cimag(d) * cimag(d);
            den += creal(want[i]) * creal(want[i]) + cimag(want[i]) * cimag(want[i]);
        }
        double rel = den > 0 ? sqrt(num / den) : sqrt(num);
        int ok = rel < 1e-12 && isfinite(rel);
        if (!ok) failures++;

        printf("  %-4d %-18s %10.2e %12.3f   %s\n", L, fft3d_best_kernel_name(L),
               rel, per_transform * 1e6, ok ? "ok" : "FAIL");

        fft3d_best_destroy(plan);
        free(in); free(got); free(want); free(scratch);
    }

    printf("\n%d of %d geometries correct\n", NGEOM - failures, NGEOM);
    if (failures == 0) {
        printf("\nkernels:\n");
        for (int g = 0; g < NGEOM; ++g)
            printf("  L=%-3d %s\n", GEOMETRIES[g],
                   fft3d_best_kernel_description(GEOMETRIES[g]));
    }
    return failures ? 1 : 0;
}
