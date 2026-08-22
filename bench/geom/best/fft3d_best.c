/* Dispatcher: one API over the eight winning kernels.
 *
 * Each kernel is compiled as its own translation unit with its six API symbols renamed by
 * -D (see the Makefile), because the kernels were written independently and share nine
 * static helper names between them -- concatenating them into one file would collide, and
 * one of them (L17_matrixsimd) instantiates itself by #including its own file, which a
 * merge would break. Keeping them as separate units means the code here is byte-identical
 * to what was measured, which is the point of a summary.
 */
#include <stdlib.h>

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

static const struct kernel_entry KERNELS[] = {
    ENTRY(6,  k6),
    ENTRY(8,  k8),
    ENTRY(13, k13),
    ENTRY(17, k17),
    ENTRY(23, k23),
    ENTRY(36, k36),
    ENTRY(45, k45),
    ENTRY(64, k64),
};
static const int NKERNELS = (int)(sizeof KERNELS / sizeof KERNELS[0]);

struct fft3d_best_plan {
    const struct kernel_entry *k;
    void *inner;
};

static const struct kernel_entry *lookup(int L)
{
    for (int i = 0; i < NKERNELS; ++i)
        if (KERNELS[i].L == L) return &KERNELS[i];
    return NULL;
}

int fft3d_best_supports(int L)
{
    const struct kernel_entry *k = lookup(L);
    /* Ask the kernel too: it is the authority on what it accepts. */
    return k && k->supports(L);
}

const char *fft3d_best_kernel_name(int L)
{
    const struct kernel_entry *k = lookup(L);
    return k ? k->name() : NULL;
}

const char *fft3d_best_kernel_description(int L)
{
    const struct kernel_entry *k = lookup(L);
    return k ? k->description() : NULL;
}

fft3d_best_plan *fft3d_best_create(int L, int batch)
{
    const struct kernel_entry *k = lookup(L);
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
