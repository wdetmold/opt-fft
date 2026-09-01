/* benchFFT benchee for the lqcd generalize library (the gen_race trunk).
 * Community-standard harness (Frigo & Johnson): their calibrated timing, their
 * 5 N log2 N "mflops" convention, their arbitrary-precision accuracy check.
 * Single-transform (vrank 0) => our B=1 paths; vector problems (v<N>) => batched,
 * laid out as consecutive volumes, matching our API directly.
 */
#include "bench-user.h"
#include <stdlib.h>

BEGIN_BENCH_DOC
BENCH_DOC("name", "lqcd-gen")
BENCH_DOC("author", "generalize campaign panel")
BENCH_DOC("year", "2026")
BENCH_DOC("notes", "arbitrary-L 3D c2c fp64 trunk: factorization planner + plan-time race, AVX-512")
END_BENCH_DOC

typedef struct fft3d_plan fft3d_plan;
extern fft3d_plan *fft3d_create(int L, int batch);
extern void fft3d_execute(fft3d_plan *p, const void *in, void *out);
extern void fft3d_destroy(fft3d_plan *p);
extern int fft3d_supports(int L);

static fft3d_plan *the_plan;

int can_do(struct problem *p)
{
     return p->kind == PROBLEM_COMPLEX
         && p->rank == 3
         && p->n[0] == p->n[1] && p->n[1] == p->n[2]
         && !p->in_place && !p->split
         && p->sign == -1
         && p->vrank <= 1
         && fft3d_supports((int)p->n[0]);
}

void copy_h2c(struct problem *p, bench_complex *out) { cacopy((bench_complex *)p->in, out, p->size * p->vsize); }
void copy_c2h(struct problem *p, bench_complex *in)  { cacopy(in, (bench_complex *)p->in, p->size * p->vsize); }
void copy_r2c(struct problem *p, bench_complex *out) { UNUSED(p); UNUSED(out); BENCH_ASSERT(0); }
void copy_c2r(struct problem *p, bench_complex *in)  { UNUSED(p); UNUSED(in); BENCH_ASSERT(0); }

void setup(struct problem *p)
{
     BENCH_ASSERT(can_do(p));
     the_plan = fft3d_create((int)p->n[0], p->vrank ? (int)p->vn[0] : 1);
     BENCH_ASSERT(the_plan);
}

void doit(int iter, struct problem *p)
{
     int i;
     for (i = 0; i < iter; ++i)
          fft3d_execute(the_plan, p->in, p->out);
}

void done(struct problem *p)
{
     UNUSED(p);
     fft3d_destroy(the_plan);
     the_plan = 0;
}
