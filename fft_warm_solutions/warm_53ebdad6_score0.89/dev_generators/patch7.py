import re
src = open('implementation.c').read()
# t-path medicine for 13/17/23: split-c mapvec/mapcol, strip-order csw, strip-S with flat map
CFG = {13: dict(PS=520, ROW=16, IMOFF=256, KC=2, JB=2, NR=13),
       17: dict(PS=1160, ROW=24, IMOFF=576, KC=3, JB=3, NR=17),
       23: dict(PS=1160, ROW=24, IMOFF=576, KC=3, JB=3, NR=23)}

for N, g in CFG.items():
    PS, ROW, IMOFF, KC, JB, NR = g['PS'], g['ROW'], g['IMOFF'], g['KC'], g['JB'], g['NR']
    STRIP = NR*8
    T = f"{N}t"
    # 1. mapvec -> split
    i = src.index(f"static inline void mapvec_{T}(")
    j = src.index("\n}", i) + 2
    src = src[:i] + f"""static inline void mapvec_{T}(double* pre, double* pim, const double* cre, const double* cim){{
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}}
""" + src[j:]
    # 2. new S (strip style, flat csw map, no prefetch)
    sS = src.index(f"static void S_{T}(double* X")
    eS = src.index("\nstatic", sS+10)
    new = f"""static void S_{T}(double* X, const double* Csw, int do_map, int do_next){{
    for(int i=0;i<{N};i++){{
        double* pl = X + (long)i*{PS};
        for(int kc=0;kc<{KC};kc++){{ dftp{N}_v(pl + kc*8, pl + {IMOFF} + kc*8, {ROW}); }}
        for(int jb=0;jb<{JB};jb++){{
            const double* rb = pl + (long)jb*8*{ROW};
            for(int kb=0;kb<{KC};kb++){{
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*{ROW}+kb*8); r1=_mm512_load_pd(rb+1*{ROW}+kb*8);
                r2=_mm512_load_pd(rb+2*{ROW}+kb*8); r3=_mm512_load_pd(rb+3*{ROW}+kb*8);
                r4=_mm512_load_pd(rb+4*{ROW}+kb*8); r5=_mm512_load_pd(rb+5*{ROW}+kb*8);
                r6=_mm512_load_pd(rb+6*{ROW}+kb*8); r7=_mm512_load_pd(rb+7*{ROW}+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_{T}[0] + (long)kb*8*{ROW} + jb*8;
                _mm512_store_pd(tb+0*{ROW}, o0); _mm512_store_pd(tb+1*{ROW}, o1);
                _mm512_store_pd(tb+2*{ROW}, o2); _mm512_store_pd(tb+3*{ROW}, o3);
                _mm512_store_pd(tb+4*{ROW}, o4); _mm512_store_pd(tb+5*{ROW}, o5);
                _mm512_store_pd(tb+6*{ROW}, o6); _mm512_store_pd(tb+7*{ROW}, o7);
                r0=_mm512_load_pd(rb+{IMOFF}+0*{ROW}+kb*8); r1=_mm512_load_pd(rb+{IMOFF}+1*{ROW}+kb*8);
                r2=_mm512_load_pd(rb+{IMOFF}+2*{ROW}+kb*8); r3=_mm512_load_pd(rb+{IMOFF}+3*{ROW}+kb*8);
                r4=_mm512_load_pd(rb+{IMOFF}+4*{ROW}+kb*8); r5=_mm512_load_pd(rb+{IMOFF}+5*{ROW}+kb*8);
                r6=_mm512_load_pd(rb+{IMOFF}+6*{ROW}+kb*8); r7=_mm512_load_pd(rb+{IMOFF}+7*{ROW}+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += {IMOFF};
                _mm512_store_pd(tb+0*{ROW}, o0); _mm512_store_pd(tb+1*{ROW}, o1);
                _mm512_store_pd(tb+2*{ROW}, o2); _mm512_store_pd(tb+3*{ROW}, o3);
                _mm512_store_pd(tb+4*{ROW}, o4); _mm512_store_pd(tb+5*{ROW}, o5);
                _mm512_store_pd(tb+6*{ROW}, o6); _mm512_store_pd(tb+7*{ROW}, o7);
            }}
            dftp{N}_v(&TS_{T}[0] + jb*8, &TS_{T}[0] + {IMOFF} + jb*8, {ROW});
            if(do_map){{
                const double* cre = Csw + (long)i*{PS} + (long)jb*{STRIP};
                double* tr = &TS_{T}[0] + jb*8;
                double* ti = &TS_{T}[0] + {IMOFF} + jb*8;
                int k=0;
                for(; k+2<={NR}; k+=2){{
                    mapvec_{T}(tr + (long)k*{ROW}, ti + (long)k*{ROW}, cre + k*8, cre + {IMOFF} + k*8);
                    mapvec_{T}(tr + (long)(k+1)*{ROW}, ti + (long)(k+1)*{ROW}, cre + k*8 + 8, cre + {IMOFF} + k*8 + 8);
                }}
                for(; k<{NR}; k++) mapvec_{T}(tr + (long)k*{ROW}, ti + (long)k*{ROW}, cre + k*8, cre + {IMOFF} + k*8);
            }}
            if(do_next) dftp{N}_v(&TS_{T}[0] + jb*8, &TS_{T}[0] + {IMOFF} + jb*8, {ROW});
            {{
                double* rb2 = pl + (long)jb*8*{ROW};
                for(int kb=0;kb<{KC};kb++){{
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_{T}[0] + (long)kb*8*{ROW} + jb*8;
                    r0=_mm512_load_pd(tb+0*{ROW}); r1=_mm512_load_pd(tb+1*{ROW});
                    r2=_mm512_load_pd(tb+2*{ROW}); r3=_mm512_load_pd(tb+3*{ROW});
                    r4=_mm512_load_pd(tb+4*{ROW}); r5=_mm512_load_pd(tb+5*{ROW});
                    r6=_mm512_load_pd(tb+6*{ROW}); r7=_mm512_load_pd(tb+7*{ROW});
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*{ROW}+kb*8, o0); _mm512_store_pd(rb2+1*{ROW}+kb*8, o1);
                    _mm512_store_pd(rb2+2*{ROW}+kb*8, o2); _mm512_store_pd(rb2+3*{ROW}+kb*8, o3);
                    _mm512_store_pd(rb2+4*{ROW}+kb*8, o4); _mm512_store_pd(rb2+5*{ROW}+kb*8, o5);
                    _mm512_store_pd(rb2+6*{ROW}+kb*8, o6); _mm512_store_pd(rb2+7*{ROW}+kb*8, o7);
                    const double* tb2 = tb + {IMOFF};
                    r0=_mm512_load_pd(tb2+0*{ROW}); r1=_mm512_load_pd(tb2+1*{ROW});
                    r2=_mm512_load_pd(tb2+2*{ROW}); r3=_mm512_load_pd(tb2+3*{ROW});
                    r4=_mm512_load_pd(tb2+4*{ROW}); r5=_mm512_load_pd(tb2+5*{ROW});
                    r6=_mm512_load_pd(tb2+6*{ROW}); r7=_mm512_load_pd(tb2+7*{ROW});
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+{IMOFF}+0*{ROW}+kb*8, o0); _mm512_store_pd(rb2+{IMOFF}+1*{ROW}+kb*8, o1);
                    _mm512_store_pd(rb2+{IMOFF}+2*{ROW}+kb*8, o2); _mm512_store_pd(rb2+{IMOFF}+3*{ROW}+kb*8, o3);
                    _mm512_store_pd(rb2+{IMOFF}+4*{ROW}+kb*8, o4); _mm512_store_pd(rb2+{IMOFF}+5*{ROW}+kb*8, o5);
                    _mm512_store_pd(rb2+{IMOFF}+6*{ROW}+kb*8, o6); _mm512_store_pd(rb2+{IMOFF}+7*{ROW}+kb*8, o7);
                }}
            }}
        }}
        if(do_next){{
            for(int kc=0;kc<{KC};kc++){{ dftp{N}_v(pl + kc*8, pl + {IMOFF} + kc*8, {ROW}); }}
        }}
    }}
}}"""
    src = src[:sS] + new + src[eS:]
    # 3. mapcol -> split (no prefetch)
    sM = src.index(f"static void mapcol_{T}(")
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
    new_mapcol = f"""static void mapcol_{T}(double* X, const double* C, int j, int kc, int jn, int kcn){{
    double* pr = X + (long)j*{ROW} + kc*8;
    double* pi = pr + {IMOFF};
    const double* cp = C + (long)j*{ROW} + kc*8;
    for(int i=0;i+2<={N};i+=2){{
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
    # 4. buildcsw -> buildc split (nat X-geom + csw strip order)
    sB = src.index(f"static void buildcsw_{T}(")
    eB = src.index("\nstatic", sB)
    stores_sw = "\n".join(
        f"""                    if(kb*8+{r} < {NR}) _mm512_store_pd(spre + (long)jb*{STRIP} + (long)(kb*8+{r})*8, o{r});""" for r in range(8))
    stores_sw_im = "\n".join(
        f"""                    if(kb*8+{r} < {NR}) _mm512_store_pd(spre + {IMOFF} + (long)jb*{STRIP} + (long)(kb*8+{r})*8, oo{r});""" for r in range(8))
    new_build = f"""static void buildc_{T}(const double* c, double* cnat, double* csw, int build_sw){{
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
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_{T}[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_{T}[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*{ROW} + kb*8, RE[r]);
                    _mm512_store_pd(npre + {IMOFF} + (long)(jb*8+r)*{ROW} + kb*8, IM[r]);
                }}
                if(build_sw){{
                    for(int r=jn;r<8;r++){{ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }}
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
{stores_sw}
                    __m512d oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7;
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7);
{stores_sw_im}
                }}
            }}
        }}
    }}
}}"""
    src = src[:sB] + new_build + src[eB:]
    # 5. run_t: alloc + buildc
    old = f"CSW_{T} = alloc_huge_st((long){N**3}*16 + 4096); CNAT_{T} = alloc_huge_st((long){N**3}*16 + 4096);"
    assert old in src, (N, 'allocT')
    src = src.replace(old, f"CSW_{T} = alloc_huge_st((long){N}*{PS}*8 + 4096); CNAT_{T} = alloc_huge_st((long){N}*{PS}*8 + 4096);")
    old = f"memcpy(CNAT_{T}, c + v*{2*N**3}, (long){N**3}*16);\n        const double* cx = CNAT_{T};\n        if(m >= 3) buildcsw_{T}(cx, CSW_{T});"
    assert old in src, (N, 'memcpyT')
    src = src.replace(old, f"buildc_{T}(c + v*{2*N**3}, CNAT_{T}, CSW_{T}, m >= 3);\n        const double* cx = CNAT_{T};")

open('implementation.c','w').write(src)
print("patch7 applied")