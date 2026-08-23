import re
src = open('implementation.c').read()
CFG = {36: dict(PS=3208, ROW=40, IMOFF=1600, KC=5, JB=5, NR=36),
       45: dict(PS=5384, ROW=56, IMOFF=2688, KC=6, JB=6, NR=45),
       64: dict(PS=9224, ROW=72, IMOFF=4608, KC=8, JB=8, NR=64)}
for N,g in CFG.items():
    PS,ROW,IMOFF,KC,JB,NR = g['PS'],g['ROW'],g['IMOFF'],g['KC'],g['JB'],g['NR']
    STRIP = NR*8
    # 1. buildc: replace cnat stores with CP stores (column-major-by-(j,kc), then i)
    sB = src.index(f"static void buildc_{N}(")
    eB = src.index("\nstatic", sB)
    body = src[sB:eB]
    old = f"""                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*{ROW} + kb*8, RE[r]);
                    _mm512_store_pd(npre + {IMOFF} + (long)(jb*8+r)*{ROW} + kb*8, IM[r]);"""
    assert old in body, (N, 'buildc cnat store')
    new = f"""                    DEINT(lo, hi, RE[r], IM[r]);
                    {{ long e = (long)(jb*8+r)*{KC} + kb;
                       _mm512_store_pd(cnat + e*{N*16} + (long)i*16, RE[r]);
                       _mm512_store_pd(cnat + e*{N*16} + (long)i*16 + 8, IM[r]); }}"""
    body = body.replace(old, new)
    body = body.replace(f"double* npre = cnat + (long)i*{PS};", "")
    src = src[:sB] + body + src[eB:]
    # 2. mapcol: sequential c reads from CP; keep x prefetch (distance-1)
    sM = src.index(f"static void mapcol_{N}(")
    eM = src.index("\nstatic", sM)
    oddtail = ""
    if N % 2 == 1:
        oddtail = f"""
    {{ int i = {N-1};
        __m512d xr0 = _mm512_load_pd(pr + (long)i*{PS});
        __m512d xi0 = _mm512_load_pd(pi + (long)i*{PS});
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*{PS}, xr0);
        _mm512_store_pd(pi + (long)i*{PS}, xi0);
    }}"""
    new_mapcol = f"""static void mapcol_{N}(double* X, const double* C, int j, int kc, int jn, int kcn){{
    double* pr = X + (long)j*{ROW} + kc*8;
    double* pi = pr + {IMOFF};
    const double* cp = C + ((long)j*{KC} + kc)*{N*16};
    const char* npr = (const char*)(X + (long)jn*{ROW} + kcn*8);
    for(int i=0;i+2<={N};i+=2){{
        _mm_prefetch(npr + (long)i*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)i*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        __m512d xr0 = _mm512_load_pd(pr + (long)i*{PS});
        __m512d xi0 = _mm512_load_pd(pi + (long)i*{PS});
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*{PS});
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*{PS});
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*{PS}, xr0);
        _mm512_store_pd(pi + (long)i*{PS}, xi0);
        _mm512_store_pd(pr + (long)(i+1)*{PS}, xr1);
        _mm512_store_pd(pi + (long)(i+1)*{PS}, xi1);
    }}{oddtail}
}}"""
    src = src[:sM] + new_mapcol + src[eM:]
open('implementation.c','w').write(src)
print("patch14 applied")