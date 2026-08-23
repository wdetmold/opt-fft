# add MAP_STYLE 3 (rsqrt14+2NR+div) and 4 (float-seeded rsqrt + 2NR + div) to merged_alt's mapv
src = open('impl_merged_alt.c').read()
old = '''#else
    __m512d r2m = _mm512_max_pd((__m512d)r2, _mm512_set1_pd(1e-300));
    __m512d t = _mm512_rsqrt14_pd(r2m);
    __m512d hr = _mm512_mul_pd(r2m, _mm512_set1_pd(0.5));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    __m512d mag = _mm512_mul_pd(r2m, t);
    __m512d den = _mm512_add_pd(mag, _mm512_set1_pd(1.0));
    __m512d u = _mm512_rcp14_pd(den);
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
#endif'''
new = '''#elif MAP_STYLE == 3
    __m512d t = _mm512_rsqrt14_pd((__m512d)r2);
    __m512d hr = _mm512_mul_pd((__m512d)r2, _mm512_set1_pd(0.5));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    __m512d den = _mm512_fmadd_pd((__m512d)r2, t, _mm512_set1_pd(1.0));
    __m512d u = _mm512_div_pd(_mm512_set1_pd(1.0), den);
#elif MAP_STYLE == 4
    __m512d t = _mm512_cvtps_pd(_mm256_rsqrt14_ps(_mm512_cvtpd_ps((__m512d)r2)));
    __m512d hr = _mm512_mul_pd((__m512d)r2, _mm512_set1_pd(0.5));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    __m512d den = _mm512_fmadd_pd((__m512d)r2, t, _mm512_set1_pd(1.0));
    __m512d u = _mm512_div_pd(_mm512_set1_pd(1.0), den);
#else
    __m512d r2m = _mm512_max_pd((__m512d)r2, _mm512_set1_pd(1e-300));
    __m512d t = _mm512_rsqrt14_pd(r2m);
    __m512d hr = _mm512_mul_pd(r2m, _mm512_set1_pd(0.5));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    __m512d mag = _mm512_mul_pd(r2m, t);
    __m512d den = _mm512_add_pd(mag, _mm512_set1_pd(1.0));
    __m512d u = _mm512_rcp14_pd(den);
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
#endif'''
assert old in src
src = src.replace(old, new)
open('impl_merged_alt.c','w').write(src)
print('ok')
