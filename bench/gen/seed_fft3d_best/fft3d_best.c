/* Dispatcher: one API over the eight winning kernels.
 *
 * Each kernel is compiled as its own translation unit with its six API symbols renamed by
 * -D (see the Makefile), because the kernels were written independently and share nine
 * static helper names between them -- concatenating them into one file would collide, and
 * one of them (L17_matrixsimd) instantiates itself by #including its own file, which a
 * merge would break. Keeping them as separate units means the code here is byte-identical
 * to what was measured, which is the point of a summary.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft3d_best.h"

/* Each kernel's renamed entry points. The plan type is opaque and private to each unit, so
 * it is carried as void * here. */
#define DECLARE_KERNEL(tag)                                                              \
    const char *tag##_name(void);                                                        \
    const char *tag##_description(void);                                                 \
    int tag##_supports(int L);                                                           \
    void *tag##_create(int L, int batch);                                                \
    void tag##_execute(void *plan, const double _Complex *in, double _Complex *out);     \
    void tag##_destroy(void *plan);

DECLARE_KERNEL(k6)
DECLARE_KERNEL(k8)
DECLARE_KERNEL(k13)
DECLARE_KERNEL(k17)
DECLARE_KERNEL(k23)
DECLARE_KERNEL(k36)
DECLARE_KERNEL(k45)
DECLARE_KERNEL(k64)
/* Every other competition entry at each geometry. Which one wins is a property of the
 * MACHINE (measured: the L=64 pair inverts by 4.3x between Haswell and Cascade Lake, and on
 * Ice Lake the grading winner differed from the Cascade Lake winner at three of eight
 * sizes), so create() races all of them and keeps the fastest here. */
DECLARE_KERNEL(k6b)
DECLARE_KERNEL(k8b)
DECLARE_KERNEL(k13b)
DECLARE_KERNEL(k17b)
DECLARE_KERNEL(k23b)
DECLARE_KERNEL(k36b)
DECLARE_KERNEL(k36c)
DECLARE_KERNEL(k45b)
DECLARE_KERNEL(k64b)

struct kernel_entry {
    int L;
    const char *(*name)(void);
    const char *(*description)(void);
    int (*supports)(int);
    void *(*create)(int, int);
    void (*execute)(void *, const double _Complex *, double _Complex *);
    void (*destroy)(void *);
};

#define ENTRY(l, tag) { l, tag##_name, tag##_description, tag##_supports,                \
                        tag##_create, tag##_execute, tag##_destroy }

/* One row per geometry. `alt` is a second kernel used from `alt_from_batch` upwards, and is
 * populated ONLY where the measurements actually support a crossover.
 *
 * Where it is NULL, an alternate exists in kernels/ but did not earn selection: at L=8 and
 * L=36 the two candidates sat within 1-4% of each other at every batch point measured and
 * the winner ALTERNATED with batch, which is smaller than the ~1.4% run-to-run spread at
 * those sizes. Selecting on batch there would encode measurement noise. L=64 is the one real
 * crossover: L64_radix8 trails at B=1 but wins by 2% at B=2 and 6% at B=8, monotonically.
 *
 * Numbers behind this are in ../results/panel_r11/leaderboard.txt; see README.md. */
#define MAX_CANDIDATES 3
struct geometry_row {
    int L;
    /* candidates[0] is the primary: the Cascade Lake competition winner, used when the
     * race is disabled. The rest are raced against it at create() time. */
    const struct kernel_entry *candidates[MAX_CANDIDATES];
};

static const struct kernel_entry K6   = ENTRY(6,  k6);
static const struct kernel_entry K6B  = ENTRY(6,  k6b);
static const struct kernel_entry K13B = ENTRY(13, k13b);
static const struct kernel_entry K17B = ENTRY(17, k17b);
static const struct kernel_entry K23B = ENTRY(23, k23b);
static const struct kernel_entry K36C = ENTRY(36, k36c);
static const struct kernel_entry K45B = ENTRY(45, k45b);
static const struct kernel_entry K8   = ENTRY(8,  k8);
static const struct kernel_entry K13  = ENTRY(13, k13);
static const struct kernel_entry K17  = ENTRY(17, k17);
static const struct kernel_entry K23  = ENTRY(23, k23);
static const struct kernel_entry K36  = ENTRY(36, k36);
static const struct kernel_entry K45  = ENTRY(45, k45);
static const struct kernel_entry K64  = ENTRY(64, k64);
static const struct kernel_entry K8B  = ENTRY(8,  k8b);
static const struct kernel_entry K36B = ENTRY(36, k36b);
static const struct kernel_entry K64B = ENTRY(64, k64b);

static const struct geometry_row GEOMETRIES[] = {
    {  6, { &K6,  &K6B,  NULL  } },   /* pfa | unrolled: unrolled won the Ice Lake grading */
    {  8, { &K8,  &K8B,  NULL  } },   /* batchsimd | fusedaxes: fusedaxes won on Ice Lake */
    { 13, { &K13, &K13B, NULL  } },   /* direct | rader */
    { 17, { &K17, &K17B, NULL  } },   /* matrixsimd | rader */
    { 23, { &K23, &K23B, NULL  } },   /* rader | matrixsimd: dead even on Cascade Lake */
    { 36, { &K36, &K36B, &K36C } },   /* mixedradix | pencilfused | pfa: pfa won on Ice Lake */
    { 45, { &K45, &K45B, NULL  } },   /* pfa | mixedradix */
    { 64, { &K64, &K64B, NULL  } },   /* blocked | radix8: inverts 4.3x between machines */
};
static const int NGEOM = (int)(sizeof GEOMETRIES / sizeof GEOMETRIES[0]);

struct fft3d_best_plan {
    const struct kernel_entry *k;
    void *inner;
};

static const struct geometry_row *row_of(int L)
{
    for (int i = 0; i < NGEOM; ++i)
        if (GEOMETRIES[i].L == L) return &GEOMETRIES[i];
    return NULL;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Time one candidate briefly. Returns seconds per execute, or -1 if it will not plan. */
static double trial(const struct kernel_entry *k, int L, int batch,
                    const double _Complex *in, double _Complex *out)
{
    void *plan = k->create(L, batch);
    if (!plan) return -1.0;
    for (int i = 0; i < 3; ++i) k->execute(plan, in, out);
    double best = 1e30;
    for (int s = 0; s < 3; ++s) {
        double t0 = now_seconds();
        for (int r = 0; r < 4; ++r) k->execute(plan, in, out);
        double per = (now_seconds() - t0) / 4.0;
        if (per < best) best = per;
    }
    k->destroy(plan);
    return best;
}

/* Which kernel serves (L, batch)?
 *
 * Where a geometry has two close candidates, the answer is a property of the MACHINE, not of
 * the algorithm: on the Xeon Gold 5218 the competition ran on, L64_radix8 beats L64_blocked
 * by 2-6% from B=2 upwards, while on an AVX2 Haswell it is 4.1-4.3x SLOWER at every batch
 * size (measured -- see pick_check). A hardcoded threshold would therefore hand a 4x
 * regression to anyone on the wrong hardware.
 *
 * So the choice is measured here instead, in create(), which every phase of this project
 * excludes from the reported time. The race runs at a capped batch (the ranking was stable
 * across batch on a given machine) to keep its memory and time bounded.
 *
 * FFT3D_BEST_NO_RACE=1 skips it and takes the primary; FFT3D_BEST_FORCE_ALT=1 takes the
 * alternate. Both exist so the choice can be pinned when measuring something else. */
static const struct kernel_entry *choose(int L, int batch)
{
    const struct geometry_row *g = row_of(L);
    if (!g) return NULL;
    if (!g->candidates[1]) return g->candidates[0];
    if (getenv("FFT3D_BEST_NO_RACE")) return g->candidates[0];

    /* Cap the race at ~32 MB per buffer so a huge batch does not make planning expensive. */
    const long vol = (long)L * L * L;
    int race_batch = (int)(2097152L / vol);
    if (race_batch < 1) race_batch = 1;
    if (race_batch > batch) race_batch = batch;

    size_t count = (size_t)vol * race_batch;
    size_t bytes = count * sizeof(double _Complex);
    double _Complex *in = NULL, *out = NULL;
    if (posix_memalign((void **)&in, 64, bytes) != 0 || !in) return g->candidates[0];
    if (posix_memalign((void **)&out, 64, bytes) != 0 || !out) { free(in); return g->candidates[0]; }
    memset(out, 0, bytes);
    for (size_t j = 0; j < count; ++j)
        in[j] = (double)((j * 2654435761u) % 1000) / 500.0 - 1.0;

    const struct kernel_entry *winner = NULL;
    double best_t = 0.0;
    for (int c = 0; c < MAX_CANDIDATES && g->candidates[c]; ++c) {
        double t = trial(g->candidates[c], L, race_batch, in, out);
        if (t > 0 && (!winner || t < best_t)) { winner = g->candidates[c]; best_t = t; }
    }
    free(in);
    free(out);
    return winner ? winner : g->candidates[0];
}

/* Reporting only: what the table would say without measuring. */
static const struct kernel_entry *lookup(int L, int batch)
{
    const struct geometry_row *g = row_of(L);
    (void)batch;
    return g ? g->candidates[0] : NULL;
}

/* The alternate kernel at this geometry, if the library carries one (NULL otherwise).
 * Exposed so a caller can measure the choice on its own machine rather than trusting ours. */
static const struct kernel_entry *alternate_of(int L)
{
    const struct geometry_row *g = row_of(L);
    return g ? g->candidates[1] : NULL;
}

int fft3d_best_supports(int L)
{
    const struct kernel_entry *k = lookup(L, 0);
    /* Ask the kernel too: it is the authority on what it accepts. */
    return k && k->supports(L);
}

const char *fft3d_best_kernel_name(int L)
{
    const struct kernel_entry *k = lookup(L, 0);
    return k ? k->name() : NULL;
}

const char *fft3d_best_kernel_description(int L)
{
    const struct kernel_entry *k = lookup(L, 0);
    return k ? k->description() : NULL;
}

const char *fft3d_best_kernel_name_for(int L, int batch)
{
    const struct kernel_entry *k = choose(L, batch);
    return k ? k->name() : NULL;
}

const char *fft3d_best_selected_name(const fft3d_best_plan *plan)
{
    return plan ? plan->k->name() : NULL;
}

const char *fft3d_best_alternate_name(int L)
{
    const struct kernel_entry *k = alternate_of(L);
    return k ? k->name() : NULL;
}

fft3d_best_plan *fft3d_best_create(int L, int batch)
{
    const struct kernel_entry *k = choose(L, batch);
    if (!k || !k->supports(L) || batch <= 0) return NULL;
    fft3d_best_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->k = k;
    p->inner = k->create(L, batch);
    if (!p->inner) { free(p); return NULL; }
    return p;
}

void fft3d_best_execute(fft3d_best_plan *p, const double _Complex *in, double _Complex *out)
{
    p->k->execute(p->inner, in, out);
}

void fft3d_best_destroy(fft3d_best_plan *p)
{
    if (!p) return;
    p->k->destroy(p->inner);
    free(p);
}
