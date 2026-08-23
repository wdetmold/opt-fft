src = open('implementation.c').read()
old = src[src.index("// ---------------------------------------------------------------- conversions"):src.index("// ---------------------------------------------------------------- init")]
new = '''// ---------------------------------------------------------------- conversions
// interleaved complex <-> planar, vectorized (2 permutes per 8 complexes)
static const long long IDX_RE[8] = {0, 2, 4, 6, 8, 10, 12, 14};
static const long long IDX_IM[8] = {1, 3, 5, 7, 9, 11, 13, 15};
static const long long IDX_LO[8] = {0, 8, 1, 9, 2, 10, 3, 11};
static const long long IDX_HI[8] = {4, 12, 5, 13, 6, 14, 7, 15};

static inline void row_deinter(const double *p, double *qr, double *qi, long L) {
  __m512i ire = _mm512_loadu_si512(IDX_RE), iim = _mm512_loadu_si512(IDX_IM);
  long z = 0;
  for (; z + 8 <= L; z += 8) {
    __m512d v0 = _mm512_loadu_pd(p + 2 * z), v1 = _mm512_loadu_pd(p + 2 * z + 8);
    _mm512_store_pd(qr + z, _mm512_permutex2var_pd(v0, ire, v1));
    _mm512_store_pd(qi + z, _mm512_permutex2var_pd(v0, iim, v1));
  }
  for (; z < L; z++) { qr[z] = p[2 * z]; qi[z] = p[2 * z + 1]; }
}
static inline void row_inter(double *p, const double *qr, const double *qi, long L) {
  __m512i ilo = _mm512_loadu_si512(IDX_LO), ihi = _mm512_loadu_si512(IDX_HI);
  long z = 0;
  for (; z + 8 <= L; z += 8) {
    __m512d vr = _mm512_load_pd(qr + z), vi = _mm512_load_pd(qi + z);
    _mm512_storeu_pd(p + 2 * z, _mm512_permutex2var_pd(vr, ilo, vi));
    _mm512_storeu_pd(p + 2 * z + 8, _mm512_permutex2var_pd(vr, ihi, vi));
  }
  for (; z < L; z++) { p[2 * z] = qr[z]; p[2 * z + 1] = qi[z]; }
}
static void load_vol(long L, long R, long SS, const double *xin, double *dre, double *dim) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++)
      row_deinter(xin + (x * L + y) * L * 2, dre + x * SS + y * R, dim + x * SS + y * R, L);
}
static void store_vol(long L, long R, long SS, const double *sre, const double *sim, double *xout) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++)
      row_inter(xout + (x * L + y) * L * 2, sre + x * SS + y * R, sim + x * SS + y * R, L);
}

'''
src = src.replace(old, new)
open('implementation.c','w').write(src)
print("ok")
