/* Measure the primary kernel against its alternate, here, on this machine.
 *
 * The competition measured these on one exclusive Xeon Gold 5218. Which of two close
 * kernels wins at a given batch size is a property of the machine, not of the algorithm, so
 * a caller on different hardware should check rather than inherit our answer -- and this is
 * the tool for it.
 *
 *   make pick_check && ./pick_check          # every geometry that has an alternate
 *   ./pick_check 64 1 2 8 64                 # one geometry, at chosen batch sizes
 */
#define _POSIX_C_SOURCE 200809L
#define PI_CONST 3.14159265358979323846

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The renamed entry points, declared directly: this tool deliberately bypasses the
 * dispatcher so it can time both sides of a choice the dispatcher makes for you. */
#define DECLARE_KERNEL(tag)                                                              \
    const char *tag##_name(void);                                                        \
    int tag##_supports(int L);                                                           \
    void *tag##_create(int L, int batch);                                                \
    void tag##_execute(void *plan, const double _Complex *in, double _Complex *out);     \
    void tag##_destroy(void *plan);

DECLARE_KERNEL(k8)   DECLARE_KERNEL(k8b)
DECLARE_KERNEL(k36)  DECLARE_KERNEL(k36b)
DECLARE_KERNEL(k64)  DECLARE_KERNEL(k64b)

struct pair {
    int L;
    const char *(*a_name)(void); void *(*a_create)(int,int);
    void (*a_exec)(void*,const double _Complex*,double _Complex*); void (*a_destroy)(void*);
    const char *(*b_name)(void); void *(*b_create)(int,int);
    void (*b_exec)(void*,const double _Complex*,double _Complex*); void (*b_destroy)(void*);
};

static const struct pair PAIRS[] = {
    {  8, k8_name,  k8_create,  k8_execute,  k8_destroy,
         k8b_name,  k8b_create, k8b_execute, k8b_destroy },
    { 36, k36_name, k36_create, k36_execute, k36_destroy,
         k36b_name, k36b_create, k36b_execute, k36b_destroy },
    { 64, k64_name, k64_create, k64_execute, k64_destroy,
         k64b_name, k64b_create, k64b_execute, k64b_destroy },
};
static const int NPAIRS = (int)(sizeof PAIRS / sizeof PAIRS[0]);

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *aligned_or_die(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) { fprintf(stderr, "oom\n"); exit(2); }
    return p;
}

static double measure(void *(*create)(int,int),
                      void (*exec)(void*,const double _Complex*,double _Complex*),
                      void (*destroy)(void*),
                      int L, int batch, const double _Complex *in, double _Complex *out)
{
    void *plan = create(L, batch);
    if (!plan) return -1.0;
    for (int i = 0; i < 3; ++i) exec(plan, in, out);      /* warm */
    /* Enough repetitions to cover ~50 ms, so a single scheduling hiccup does not decide it. */
    double t0 = now_seconds();
    exec(plan, in, out);
    double one = now_seconds() - t0;
    long reps = one > 0 ? (long)(0.05 / one) + 1 : 20;
    if (reps > 2000) reps = 2000;
    double best = 1e30;
    for (int s = 0; s < 5; ++s) {
        t0 = now_seconds();
        for (long r = 0; r < reps; ++r) exec(plan, in, out);
        double per = (now_seconds() - t0) / (double)reps / (double)batch;
        if (per < best) best = per;
    }
    destroy(plan);
    return best;
}

int main(int argc, char **argv)
{
    int only_L = argc > 1 ? atoi(argv[1]) : 0;
    int batches[16], nb = 0;
    if (argc > 2) { for (int i = 2; i < argc && nb < 16; ++i) batches[nb++] = atoi(argv[i]); }
    else { batches[nb++] = 1; batches[nb++] = 2; batches[nb++] = 8; batches[nb++] = 64; }

    printf("primary vs alternate, measured here (us per transform, best of 5)\n\n");
    printf("  %-4s %-7s %-18s %-18s %s\n", "L", "batch", "primary", "alternate", "verdict");

    for (int p = 0; p < NPAIRS; ++p) {
        const struct pair *pr = &PAIRS[p];
        if (only_L && pr->L != only_L) continue;
        for (int i = 0; i < nb; ++i) {
            int batch = batches[i];
            size_t count = (size_t)pr->L * pr->L * pr->L * batch;
            size_t bytes = count * sizeof(double _Complex);
            double _Complex *in = aligned_or_die(bytes);
            double _Complex *out = aligned_or_die(bytes);
            for (size_t j = 0; j < count; ++j)
                in[j] = (double)((j * 2654435761u) % 1000) / 500.0 - 1.0;

            double a = measure(pr->a_create, pr->a_exec, pr->a_destroy, pr->L, batch, in, out);
            double b = measure(pr->b_create, pr->b_exec, pr->b_destroy, pr->L, batch, in, out);
            const char *verdict = "both failed";
            if (a > 0 && b > 0)
                verdict = (a <= b) ? "primary" : "ALTERNATE";
            printf("  %-4d %-7d %-8s %9.3f %-8s %9.3f  %s",
                   pr->L, batch, pr->a_name(), a * 1e6, pr->b_name(), b * 1e6, verdict);
            if (a > 0 && b > 0) printf("  (%.2fx)", a > b ? a / b : b / a);
            printf("\n");
            free(in); free(out);
        }
    }
    printf("\nThe library's built-in choice is in fft3d_best.c's GEOMETRIES table, and by\n"
           "default it races the two at create() time rather than trusting a table.\n");
    return 0;
}
