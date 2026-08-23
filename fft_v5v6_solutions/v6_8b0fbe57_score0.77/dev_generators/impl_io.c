static double W2re_[12192 + 64] __attribute__((aligned(64)));
static double W2im_[12192 + 64] __attribute__((aligned(64)));
static double Wre_[91152 + 64] __attribute__((aligned(64)));
static double Wim_[91152 + 64] __attribute__((aligned(64)));
// ---------------------------------------------------------------------------
// import / export between interleaved complex128 and SoA
// ---------------------------------------------------------------------------
static void soa_import(double* restrict re, double* restrict im, const double* restrict src, long n)
{
    const __m512i idxe = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxo = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    long i = 0;
    for (; i + 8 <= n; i += 8) {
        VD v0 = LD(src + 2*i), v1 = LD(src + 2*i + 8);
        ST(re + i, _mm512_permutex2var_pd(v0, idxe, v1));
        ST(im + i, _mm512_permutex2var_pd(v0, idxo, v1));
    }
    for (; i < n; i++) { re[i] = src[2*i]; im[i] = src[2*i+1]; }
}
static void soa_export(const double* restrict re, const double* restrict im, double* restrict dst, long n)
{
    const __m512i idxl = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxh = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    long i = 0;
    if ((((uintptr_t)dst) & 63) == 0) {
        for (; i + 8 <= n; i += 8) {
            VD r = LD(re + i), m = LD(im + i);
            _mm512_stream_pd(dst + 2*i,     _mm512_permutex2var_pd(r, idxl, m));
            _mm512_stream_pd(dst + 2*i + 8, _mm512_permutex2var_pd(r, idxh, m));
        }
        _mm_sfence();
    } else {
        for (; i + 8 <= n; i += 8) {
            VD r = LD(re + i), m = LD(im + i);
            ST(dst + 2*i,     _mm512_permutex2var_pd(r, idxl, m));
            ST(dst + 2*i + 8, _mm512_permutex2var_pd(r, idxh, m));
        }
    }
    for (; i < n; i++) { dst[2*i] = re[i]; dst[2*i+1] = im[i]; }
}

