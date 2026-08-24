#include <stdint.h>
#include <math.h>
#include <string.h>
#include "zig_tables.h"

typedef __uint128_t u128;
typedef struct { u128 state, inc; } pcg64_t;

#define PCG_MUL ((((u128)2549297995355413924ULL) << 64) | 4865540595714422341ULL)

static inline void pcg_step(pcg64_t* r){ r->state = r->state * PCG_MUL + r->inc; }
static inline uint64_t rotr64(uint64_t v, unsigned rot){ return (v >> rot) | (v << ((64u - rot) & 63u)); }
static inline uint64_t pcg_out(u128 s){
  return rotr64((uint64_t)(s >> 64) ^ (uint64_t)s, (unsigned)(s >> 122));
}
static inline uint64_t pcg_next64(pcg64_t* r){ pcg_step(r); return pcg_out(r->state); }
static inline double pcg_nextd(pcg64_t* r){ return (double)(pcg_next64(r) >> 11) * (1.0/9007199254740992.0); }

static void pcg_seed(pcg64_t* r, const uint64_t* w){
  u128 initstate = (((u128)w[0]) << 64) | w[1];
  u128 initseq   = (((u128)w[2]) << 64) | w[3];
  r->state = 0; r->inc = (initseq << 1) | 1;
  pcg_step(r);
  r->state += initstate;
  pcg_step(r);
}

static const double* wi_double = (const double*)wi_double_bits;
static const double* fi_double = (const double*)fi_double_bits;

static double znorm(pcg64_t* rng){
  for (;;){
    uint64_t r = pcg_next64(rng);
    int idx = (int)(r & 0xff);
    uint64_t r9 = r >> 9;
    uint64_t rabs = r9 & 0x000fffffffffffffULL;
    double x = (double)rabs * wi_double[idx];
    if (r & 0x100) x = -x;
    if (rabs < ki_double[idx]) return x;
    if (idx == 0){
      const double nor_r = *(const double*)&zig_nor_r_bits;
      const double neg_inv = *(const double*)&zig_neg_inv_bits;
      for (;;){
        double xx = neg_inv * log1p(-pcg_nextd(rng));
        double yy = -log1p(-pcg_nextd(rng));
        if (yy + yy > xx * xx){
          double v = xx + nor_r;
          return (r9 & 0x100) ? -v : v;
        }
      }
    } else {
      if ((fi_double[idx-1] - fi_double[idx]) * pcg_nextd(rng) + fi_double[idx] < exp(-0.5*x*x))
        return x;
    }
  }
}

void fill_raw(const uint64_t* w, long n, uint64_t* out){
  pcg64_t r; pcg_seed(&r, w);
  for (long i = 0; i < n; i++) out[i] = pcg_next64(&r);
}
void fill_normals(const uint64_t* w, long n, double* out){
  pcg64_t r; pcg_seed(&r, w);
  for (long i = 0; i < n; i++) out[i] = znorm(&r);
}
