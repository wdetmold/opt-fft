import re
src = open('implementation.c').read()

CFG = {36: dict(PS=3208, ROW=40, IMOFF=1600, KC=5, JB=5, CROW=72, CPL=2592),
       45: dict(PS=5384, ROW=56, IMOFF=2688, KC=6, JB=6, CROW=90, CPL=4050),
       64: dict(PS=9224, ROW=72, IMOFF=4608, KC=8, JB=8, CROW=128, CPL=8192)}

for N, g in CFG.items():
    PS, ROW, IMOFF, KC, JB = g['PS'], g['ROW'], g['IMOFF'], g['KC'], g['JB']
    # ---- 1. new mapvec (split, aligned, no masks) ----
    i = src.index(f"static inline void mapvec_{N}(")
    j = src.index("\n}", i) + 2
    new_mapvec = f"""static inline void mapvec_{N}(double* pre, double* pim, const double* cre, const double* cim){{
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}}
"""
    src = src[:i] + new_mapvec + src[j:]
    # ---- 2. S_N map call sites ----
    # pattern in S_N:
    #   int mj0 = MKJ0_36[jb], mj1 = MKJ1_36[jb];
    #   ... crow = Csw + ((long)i*36 + k)*72 + jb*16;
    #   mapvec_36(TSre, TSim, crow, mj0, mj1);  (x2 unrolled + tail)
    sS = src.index(f"static void S_{N}(")
    eS = src.index("\nstatic", sS)
    body = src[sS:eS]
    body = body.replace(f"int mj0 = MKJ0_{N}[jb], mj1 = MKJ1_{N}[jb];", "")
    body = re.sub(rf"const double\* crow = Csw \+ \(\(long\)i\*{N} \+ k\)\*{g['CROW']} \+ jb\*16;",
                  f"const double* crow = Csw + (long)i*{PS} + (long)k*{ROW} + jb*8;", body)
    body = body.replace(f"mapvec_{N}(&TS_{N}[0] + (long)k*{ROW} + jb*8, &TS_{N}[0] + {IMOFF} + (long)k*{ROW} + jb*8, crow, mj0, mj1);",
                        f"mapvec_{N}(&TS_{N}[0] + (long)k*{ROW} + jb*8, &TS_{N}[0] + {IMOFF} + (long)k*{ROW} + jb*8, crow, crow + {IMOFF});")
    body = body.replace(f"mapvec_{N}(&TS_{N}[0] + (long)(k+1)*{ROW} + jb*8, &TS_{N}[0] + {IMOFF} + (long)(k+1)*{ROW} + jb*8, crow + {g['CROW']}, mj0, mj1);",
                        f"mapvec_{N}(&TS_{N}[0] + (long)(k+1)*{ROW} + jb*8, &TS_{N}[0] + {IMOFF} + (long)(k+1)*{ROW} + jb*8, crow + {ROW}, crow + {ROW} + {IMOFF});")
    # S prefetch block for csw (points into old interleaved layout) -> new layout
    body = re.sub(rf"const char\* cq = \(const char\*\)\(Csw \+ \(long\)i\*{g['CPL']} \+ jb\*16\) \+ \(long\)kb\*8\*{g['CROW']}\*8;",
                  f"const char* cq = (const char*)(Csw + (long)i*{PS} + jb*8) + (long)kb*8*{ROW}*8;", body)
    body = re.sub(rf"_mm_prefetch\(cq \+ \(long\)kk\*{g['CROW']}\*8, _MM_HINT_T0\); _mm_prefetch\(cq \+ \(long\)kk\*{g['CROW']}\*8 \+ 64, _MM_HINT_T0\);",
                  f"_mm_prefetch(cq + (long)kk*{ROW}*8, _MM_HINT_T0); _mm_prefetch(cq + (long)kk*{ROW}*8 + {IMOFF}*8, _MM_HINT_T0);", body)
    src = src[:sS] + body + src[eS:]
    # ---- 3. mapcol_N rewrite ----
    sM = src.index(f"static void mapcol_{N}(")
    eM = src.index("\nstatic", sM)
    oddtail = ""
    if N % 2 == 1:
        oddtail = f"""
    {{ int i = {N-1};
        __m512d xr0 = _mm512_load_pd(pr + (long)i*{PS});
        __m512d xi0 = _mm512_load_pd(pi + (long)i*{PS});
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*{PS}));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*{PS} + {IMOFF}));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*{PS}, xr0);
        _mm512_store_pd(pi + (long)i*{PS}, xi0);
    }}"""
    new_mapcol = f"""static void mapcol_{N}(double* X, const double* C, int j, int kc, int jn, int kcn){{
    double* pr = X + (long)j*{ROW} + kc*8;
    double* pi = pr + {IMOFF};
    const double* cp = C + (long)j*{ROW} + kc*8;
    const char* npr = (const char*)(X + (long)jn*{ROW} + kcn*8);
    const char* ncp = (const char*)(C + (long)jn*{ROW} + kcn*8);
    for(int i=0;i+2<={N};i+=2){{
        _mm_prefetch(npr + (long)i*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)i*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        _mm_prefetch(ncp + (long)i*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(ncp + (long)i*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        _mm_prefetch(ncp + (long)(i+1)*{PS}*8, _MM_HINT_T0);
        _mm_prefetch(ncp + (long)(i+1)*{PS}*8 + {IMOFF}*8, _MM_HINT_T0);
        __m512d xr0 = _mm512_load_pd(pr + (long)i*{PS});
        __m512d xi0 = _mm512_load_pd(pi + (long)i*{PS});
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*{PS}));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*{PS} + {IMOFF}));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*{PS});
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*{PS});
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*{PS}));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*{PS} + {IMOFF}));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*{PS}, xr0);
        _mm512_store_pd(pi + (long)i*{PS}, xi0);
        _mm512_store_pd(pr + (long)(i+1)*{PS}, xr1);
        _mm512_store_pd(pi + (long)(i+1)*{PS}, xi1);
    }}{oddtail}
}}"""
    src = src[:sM] + new_mapcol + src[eM:]
    # ---- 4. buildc: produce split cnat and split csw from interleaved c ----
    sB = src.index(f"static void buildcsw_{N}(")
    eB = src.index("\nstatic", sB)
    new_build = f"""static void buildc_{N}(const double* c, double* cnat, double* csw, int build_sw){{
    for(int i=0;i<{N};i++){{
        const double* cp = c + (long)i*{N*N*2};
        double* npre = cnat + (long)i*{PS};
        double* spre = csw + (long)i*{PS};
        for(int jb=0;jb<{JB};jb++){{
            int jn = {N} - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<{KC};kb++){{
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){{
                    const double* row = cp + ((long)(jb*8+r)*{N} + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_{N}[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_{N}[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*{ROW} + kb*8, RE[r]);
                    _mm512_store_pd(npre + {IMOFF} + (long)(jb*8+r)*{ROW} + kb*8, IM[r]);
                }}
                if(build_sw){{
                    for(int r=jn;r<8;r++){{ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }}
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(spre + (long)(kb*8+0)*{ROW} + jb*8, o0);
                    _mm512_store_pd(spre + (long)(kb*8+1)*{ROW} + jb*8, o1);
                    _mm512_store_pd(spre + (long)(kb*8+2)*{ROW} + jb*8, o2);
                    _mm512_store_pd(spre + (long)(kb*8+3)*{ROW} + jb*8, o3);
                    _mm512_store_pd(spre + (long)(kb*8+4)*{ROW} + jb*8, o4);
                    _mm512_store_pd(spre + (long)(kb*8+5)*{ROW} + jb*8, o5);
                    _mm512_store_pd(spre + (long)(kb*8+6)*{ROW} + jb*8, o6);
                    _mm512_store_pd(spre + (long)(kb*8+7)*{ROW} + jb*8, o7);
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+0)*{ROW} + jb*8, o0);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+1)*{ROW} + jb*8, o1);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+2)*{ROW} + jb*8, o2);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+3)*{ROW} + jb*8, o3);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+4)*{ROW} + jb*8, o4);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+5)*{ROW} + jb*8, o5);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+6)*{ROW} + jb*8, o6);
                    _mm512_store_pd(spre + {IMOFF} + (long)(kb*8+7)*{ROW} + jb*8, o7);
                }}
            }}
        }}
    }}
}}"""
    src = src[:sB] + new_build + src[eB:]
    # ---- 5. run_N updates ----
    sR = src.index(f"void run_{N}(")
    eRn = src.find("\nstatic", sR)
    eRv = src.find("\nvoid", sR+10)
    cands = [x for x in (eRn, eRv, len(src)) if x > 0]
    eR = min(cands)
    body = src[sR:eR]
    old = f"CSW_{N} = alloc_huge_st((long){N**3}*16 + 4096); CNAT_{N} = alloc_huge_st((long){N**3}*16 + 4096);"
    assert old in body, (N, 'alloc')
    body = body.replace(old, f"CSW_{N} = alloc_huge_st((long){N}*{PS}*8 + 4096); CNAT_{N} = alloc_huge_st((long){N}*{PS}*8 + 4096);")
    old = f"memcpy(CNAT_{N}, c + v*{2*N**3}, (long){N**3}*16);\n        const double* cx = CNAT_{N};\n        if(m >= 3) buildcsw_{N}(cx, CSW_{N});"
    assert old in body, (N, 'memcpy', body[:400])
    body = body.replace(old, f"buildc_{N}(c + v*{2*N**3}, CNAT_{N}, CSW_{N}, m >= 3);\n        const double* cx = CNAT_{N};")
    src = src[:sR] + body + src[eR:]

open('implementation.c','w').write(src)
print("patch3 applied")