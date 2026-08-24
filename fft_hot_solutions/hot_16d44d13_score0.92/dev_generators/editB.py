src = open('impl_merged_alt.c').read()
src = src.replace('''#elif MAP_STYLE == 3
    __m512d t = _mm512_rsqrt14_pd((__m512d)r2);
    __m512d hr = _mm512_mul_pd((__m512d)r2, _mm512_set1_pd(0.5));''',
'''#elif MAP_STYLE == 3
    __m512d r2g = _mm512_max_pd((__m512d)r2, _mm512_set1_pd(1e-300));
    __m512d t = _mm512_rsqrt14_pd(r2g);
    __m512d hr = _mm512_mul_pd(r2g, _mm512_set1_pd(0.5));''')
src = src.replace('''    __m512d den = _mm512_fmadd_pd((__m512d)r2, t, _mm512_set1_pd(1.0));
    __m512d u = _mm512_div_pd(_mm512_set1_pd(1.0), den);
#elif MAP_STYLE == 4''',
'''    __m512d den = _mm512_fmadd_pd(r2g, t, _mm512_set1_pd(1.0));
    __m512d u = _mm512_div_pd(_mm512_set1_pd(1.0), den);
#elif MAP_STYLE == 4''')
src = src.replace('#define MAP_STYLE 2 ', '#define MAP_STYLE 3 ')
open('impl_merged_alt.c','w').write(src)
print('ok')
