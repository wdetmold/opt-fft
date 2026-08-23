import re
src = open('implementation.c').read()

CFG = {36: dict(PS=3208, ROW=40, IMOFF=1600, KC=5, JB=5, NR=36, PF=False),
       45: dict(PS=5384, ROW=56, IMOFF=2688, KC=6, JB=6, NR=45, PF=False),
       64: dict(PS=9224, ROW=72, IMOFF=4608, KC=8, JB=8, NR=64, PF=True)}

for N, g in CFG.items():
    PS, ROW, IMOFF, KC, JB, NR = g['PS'], g['ROW'], g['IMOFF'], g['KC'], g['JB'], g['NR']
    STRIP = NR*8
    # ---------- new S_N ----------
    sS = src.index(f"static void S_{N}(double* X")
    eS = src.index("\nstatic", sS+10)
    tilepf = ""
    outpf = ""
    xplanepf = ""
    if g['PF']:
        PFJ = (PS*8)//JB + 64
        xplanepf = f"""
            {{ const char* q = npl + (long)jb*{PFJ};
               for(long b=0;b<{PFJ};b+=64) _mm_prefetch(q+b, _MM_HINT_T1); }}"""
    new = f"""static void S_{N}(double* X, const double* Csw, int do_map, int do_next){{
    for(int i=0;i<{N};i++){{
        double* pl = X + (long)i*{PS};
        const char* npl = (const char*)(X + (long)(i+1 < {N} ? i+1 : 0)*{PS});
        const char* ncw = (const char*)(Csw + (long)i*{PS});
        (void)npl; (void)ncw;
        for(int kc=0;kc<{KC};kc++){{ dft{N}_v(pl + kc*8, pl + {IMOFF} + kc*8, {ROW}); }}
        for(int jb=0;jb<{JB};jb++){{
            const double* rb = pl + (long)jb*8*{ROW};
            for(int kb=0;kb<{KC};kb++){{
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*{ROW}+kb*8); r1=_mm512_load_pd(rb+1*{ROW}+kb*8);
                r2=_mm512_load_pd(rb+2*{ROW}+kb*8); r3=_mm512_load_pd(rb+3*{ROW}+kb*8);
                r4=_mm512_load_pd(rb+4*{ROW}+kb*8); r5=_mm512_load_pd(rb+5*{ROW}+kb*8);
                r6=_mm512_load_pd(rb+6*{ROW}+kb*8); r7=_mm512_load_pd(rb+7*{ROW}+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_{N}[0] + (long)kb*8*{ROW} + jb*8;
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
                _mm512_store_pd(tb+6*{ROW}, o6); _mm512_store_pd(tb+7*{ROW}, o7);{tilepf}
            }}
            dft{N}_v(&TS_{N}[0] + jb*8, &TS_{N}[0] + {IMOFF} + jb*8, {ROW});{xplanepf}
            if(do_map){{
                const double* cre = Csw + (long)i*{PS} + (long)jb*{STRIP};
                long jb2 = jb + 2;
                const char* cnx = (jb2 < {JB}) ? (const char*)(cre + 2*{STRIP})
                                  : (const char*)(Csw + (long)(i+1 < {N} ? i+1 : 0)*{PS} + (long)(jb2-{JB})*{STRIP});
                double* tr = &TS_{N}[0] + jb*8;
                double* ti = &TS_{N}[0] + {IMOFF} + jb*8;
                for(int k=0; k+2<={NR}; k+=2){{
                    _mm_prefetch(cnx + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + {IMOFF}*8 + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + k*64 + 64, _MM_HINT_T0);
                    _mm_prefetch(cnx + {IMOFF}*8 + k*64 + 64, _MM_HINT_T0);
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*{ROW}), xi0 = _mm512_load_pd(ti + (long)k*{ROW});
                    __m512d xr1 = _mm512_load_pd(tr + (long)(k+1)*{ROW}), xi1 = _mm512_load_pd(ti + (long)(k+1)*{ROW});
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + {IMOFF} + k*8));
                    __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cre + k*8 + 8));
                    __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cre + {IMOFF} + k*8 + 8));
                    map2(zr0, zi0, &xr0, &xi0);
                    map2(zr1, zi1, &xr1, &xi1);
                    _mm512_store_pd(tr + (long)k*{ROW}, xr0); _mm512_store_pd(ti + (long)k*{ROW}, xi0);
                    _mm512_store_pd(tr + (long)(k+1)*{ROW}, xr1); _mm512_store_pd(ti + (long)(k+1)*{ROW}, xi1);
                }}"""
    if NR % 2 == 1:
        new += f"""
                {{ int k = {NR-1};
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*{ROW}), xi0 = _mm512_load_pd(ti + (long)k*{ROW});
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + {IMOFF} + k*8));
                    map2(zr0, zi0, &xr0, &xi0);
                    _mm512_store_pd(tr + (long)k*{ROW}, xr0); _mm512_store_pd(ti + (long)k*{ROW}, xi0);
                }}"""
    new += f"""
            }}
            if(do_next) dft{N}_v(&TS_{N}[0] + jb*8, &TS_{N}[0] + {IMOFF} + jb*8, {ROW});
            {{
                double* rb2 = pl + (long)jb*8*{ROW};
                for(int kb=0;kb<{KC};kb++){{
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_{N}[0] + (long)kb*8*{ROW} + jb*8;
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
                    _mm512_store_pd(rb2+{IMOFF}+6*{ROW}+kb*8, o6); _mm512_store_pd(rb2+{IMOFF}+7*{ROW}+kb*8, o7);{outpf}
                }}
            }}
        }}
        if(do_next){{
            for(int kc=0;kc<{KC};kc++){{ dft{N}_v(pl + kc*8, pl + {IMOFF} + kc*8, {ROW}); }}
        }}
    }}
}}"""
    src = src[:sS] + new + src[eS:]
    # ---------- buildc: csw in strip order ----------
    old1 = f"""                if(build_sw){{
                    for(int r=jn;r<8;r++){{ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }}
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);"""
    assert old1 in src, (N, 'buildc1')
    # replace the csw store addressing: spre + (kb*8+r)*ROW + jb*8  ->  spre + jb*STRIP + (kb*8+r)*8
    sB = src.index(f"static void buildc_{N}(")
    eB = src.index("\nstatic", sB)
    body = src[sB:eB]
    for r in range(8):
        o = f"_mm512_store_pd(spre + (long)(kb*8+{r})*{ROW} + jb*8, o{r});"
        n = f"if(kb*8+{r} < {NR}) _mm512_store_pd(spre + (long)jb*{STRIP} + (long)(kb*8+{r})*8, o{r});"
        assert o in body, (N, 'csw store', r)
        body = body.replace(o, n)
        o = f"_mm512_store_pd(spre + {IMOFF} + (long)(kb*8+{r})*{ROW} + jb*8, o{r});"
        n = f"if(kb*8+{r} < {NR}) _mm512_store_pd(spre + {IMOFF} + (long)jb*{STRIP} + (long)(kb*8+{r})*8, o{r});"
        assert o in body, (N, 'csw store im', r)
        body = body.replace(o, n)
    src = src[:sB] + body + src[eB:]

open('implementation.c','w').write(src)
print("patch5 applied")