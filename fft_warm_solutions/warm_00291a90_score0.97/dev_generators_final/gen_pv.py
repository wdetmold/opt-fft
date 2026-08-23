"""Per-volume (within-volume) drivers for family-A sizes: used for batch remainders.
Reuses cd{L}_p/m/mn/mns codelets (runtime element stride es)."""

def gen_pv(L, cd=None, asmcd=False):
    if cd is None: cd = f'cd{L}' 
    NV = (L + 7) // 8
    RS = NV * 16
    YP = NV * 8
    PS = YP * RS
    if (PS * 8) % 4096 == 0: PS += 16
    NYB = YP // 8
    NZS = NV * 8
    L2, L3 = L*L, L*L*L
    if asmcd:
        CALL_Y = f"cd{L}_w_p(pl + v*16)"
        CALL_ZP = f"cd{L}_z_p(ZSP_{L})"
        CALL_ZM = f"cd{L}_z_m(ZSP_{L}, CZT + ((long)i*{NYB}+yb)*{L}*16)"
        CALL_ZMN = f"cd{L}_z_mn(ZSP_{L}, CZT + ((long)i*{NYB}+yb)*{L}*16)"
        CALL_XM = f"cd{L}_v_m(p, pc)"
        CALL_XMN = f"cd{L}_v_mn(p, pc)"
        CALL_XMNS = f"cd{L}_v_mns(p, pc, SNAP + y*{RS} + v*16)"
    else:
        CALL_Y = f"{cd}_p(pl + v*16, {RS})"
        CALL_ZP = f"{cd}_p(ZSP_{L}, 16)"
        CALL_ZM = f"{cd}_m(ZSP_{L}, 16, CZT + ((long)i*{NYB}+yb)*{L}*16)"
        CALL_ZMN = f"{cd}_mn(ZSP_{L}, 16, CZT + ((long)i*{NYB}+yb)*{L}*16)"
        CALL_XM = f"{cd}_m(p, {PS}, pc)"
        CALL_XMN = f"{cd}_mn(p, {PS}, pc)"
        CALL_XMNS = f"{cd}_mns(p, {PS}, pc, SNAP + y*{RS} + v*16, {PS})"
    return f"""
// ---------------- per-volume driver, L={L} (NV={NV} RS={RS} YP={YP} PS={PS}) ----------------
static double ZSP_{L}[{NZS}*16+8] ALIGN64;
static double* XP_{L} = 0;
static double* CGP_{L} = 0;
static double* CZTP_{L} = 0;
static double* CPP2_{L} = 0;
static double* SNP_{L} = 0;
static inline void trinp_{L}(const double* rb, double* ZS){{
    for(int v=0; v<{NV}; v++){{
        __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
        for(int h=0; h<2; h++){{
            r0=_mm512_load_pd(rb + 0*{RS} + v*16+8*h); r1=_mm512_load_pd(rb + 1*{RS} + v*16+8*h);
            r2=_mm512_load_pd(rb + 2*{RS} + v*16+8*h); r3=_mm512_load_pd(rb + 3*{RS} + v*16+8*h);
            r4=_mm512_load_pd(rb + 4*{RS} + v*16+8*h); r5=_mm512_load_pd(rb + 5*{RS} + v*16+8*h);
            r6=_mm512_load_pd(rb + 6*{RS} + v*16+8*h); r7=_mm512_load_pd(rb + 7*{RS} + v*16+8*h);
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(ZS + (v*8+0)*16+8*h, o0); _mm512_store_pd(ZS + (v*8+1)*16+8*h, o1);
            _mm512_store_pd(ZS + (v*8+2)*16+8*h, o2); _mm512_store_pd(ZS + (v*8+3)*16+8*h, o3);
            _mm512_store_pd(ZS + (v*8+4)*16+8*h, o4); _mm512_store_pd(ZS + (v*8+5)*16+8*h, o5);
            _mm512_store_pd(ZS + (v*8+6)*16+8*h, o6); _mm512_store_pd(ZS + (v*8+7)*16+8*h, o7);
        }}
    }}
}}
static inline void troutp_{L}(double* rb, const double* ZS){{
    for(int v=0; v<{NV}; v++){{
        __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
        for(int h=0; h<2; h++){{
            r0=_mm512_load_pd(ZS + (v*8+0)*16+8*h); r1=_mm512_load_pd(ZS + (v*8+1)*16+8*h);
            r2=_mm512_load_pd(ZS + (v*8+2)*16+8*h); r3=_mm512_load_pd(ZS + (v*8+3)*16+8*h);
            r4=_mm512_load_pd(ZS + (v*8+4)*16+8*h); r5=_mm512_load_pd(ZS + (v*8+5)*16+8*h);
            r6=_mm512_load_pd(ZS + (v*8+6)*16+8*h); r7=_mm512_load_pd(ZS + (v*8+7)*16+8*h);
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(rb + 0*{RS} + v*16+8*h, o0); _mm512_store_pd(rb + 1*{RS} + v*16+8*h, o1);
            _mm512_store_pd(rb + 2*{RS} + v*16+8*h, o2); _mm512_store_pd(rb + 3*{RS} + v*16+8*h, o3);
            _mm512_store_pd(rb + 4*{RS} + v*16+8*h, o4); _mm512_store_pd(rb + 5*{RS} + v*16+8*h, o5);
            _mm512_store_pd(rb + 6*{RS} + v*16+8*h, o6); _mm512_store_pd(rb + 7*{RS} + v*16+8*h, o7);
        }}
    }}
}}
static void SP_{L}(double* X, const double* CZT, int mode){{
    for(int i=0;i<{L};i++){{
        double* pl = X + (long)i*{PS};
        _Pragma("GCC unroll 1") for(int v=0;v<{NV};v++) {CALL_Y};
        _Pragma("GCC unroll 1") for(int yb=0; yb<{NYB}; yb++){{
            trinp_{L}(pl + (long)yb*8*{RS}, ZSP_{L});
            if(mode == 0) {CALL_ZP};
            else if(mode == 1) {CALL_ZM};
            else {CALL_ZMN};
            troutp_{L}(pl + (long)yb*8*{RS}, ZSP_{L});
        }}
        if(mode == 2){{
            _Pragma("GCC unroll 1") for(int v=0;v<{NV};v++) {CALL_Y};
        }}
    }}
}}
static void PP_{L}(double* X, const double* CP, int mode, double* SNAP){{
    _Pragma("GCC unroll 1") for(long e=0;e<{L}*{NV};e++){{
        long y = e / {NV}, v = e % {NV};
        double* p = X + y*{RS} + v*16;
        const double* pc = CP + e*{L}*16;
        if(mode == 1) {CALL_XM};
        else if(mode == 2) {CALL_XMN};
        else {CALL_XMNS};
    }}
}}
static void convinp_{L}(const double* src, double* X){{
    for(long i=0;i<{L};i++) for(long y=0;y<{L};y++){{
        const double* s = src + 2*((i*{L}+y)*{L});
        double* d = X + i*{PS} + y*{RS};
        long v=0;
        for(; v<{L}/8; v++){{
            __m512d lo=_mm512_loadu_pd(s + v*16), hi=_mm512_loadu_pd(s + v*16 + 8);
            __m512d re, im;
            DEINT(lo, hi, re, im);
            _mm512_store_pd(d + v*16, re); _mm512_store_pd(d + v*16 + 8, im);
        }}
        for(long z=v*8; z<{L}; z++){{ d[(z/8)*16 + (z%8)] = s[2*z]; d[(z/8)*16 + 8 + (z%8)] = s[2*z+1]; }}
    }}
}}
static void convoutp_{L}(const double* X, double* dst){{
    for(long i=0;i<{L};i++) for(long y=0;y<{L};y++){{
        double* s = dst + 2*((i*{L}+y)*{L});
        const double* d = X + i*{PS} + y*{RS};
        long v=0;
        for(; v<{L}/8; v++){{
            __m512d re=_mm512_load_pd(d + v*16), im=_mm512_load_pd(d + v*16 + 8);
            __m512d lo, hi;
            INTER(re, im, lo, hi);
            _mm512_storeu_pd(s + v*16, lo); _mm512_storeu_pd(s + v*16 + 8, hi);
        }}
        for(long z=v*8; z<{L}; z++){{ s[2*z] = d[(z/8)*16 + (z%8)]; s[2*z+1] = d[(z/8)*16 + 8 + (z%8)]; }}
    }}
}}
static void run_{L}pv(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!XP_{L}){{
        XP_{L}  = alloc_huge_st((long){L}*{PS}*8);
        CGP_{L} = alloc_huge_st((long){L}*{PS}*8);
        CZTP_{L} = alloc_huge_st((long){L}*{NYB}*{L}*16*8);
        CPP2_{L} = alloc_huge_st((long){L}*{NV}*{L}*16*8);
        SNP_{L} = alloc_huge_st((long){L}*{PS}*8);
    }}
    for(long b=0; b<B; b++){{
        convinp_{L}(x0 + b*2*{L3}, XP_{L});
        convinp_{L}(c + b*2*{L3}, CGP_{L});
        for(int i=0;i<{L};i++) for(int yb=0; yb<{NYB}; yb++){{
            trinp_{L}(CGP_{L} + (long)i*{PS} + (long)yb*8*{RS}, ZSP_{L});
            memcpy(CZTP_{L} + ((long)i*{NYB} + yb)*{L}*16, ZSP_{L}, {L}*16*8);
        }}
        for(long e=0;e<{L}*{NV};e++){{
            long y = e / {NV}, v = e % {NV};
            for(long q=0;q<{L};q++){{
                _mm512_store_pd(CPP2_{L} + e*{L}*16 + q*16,     _mm512_load_pd(CGP_{L} + q*{PS} + y*{RS} + v*16));
                _mm512_store_pd(CPP2_{L} + e*{L}*16 + q*16 + 8, _mm512_load_pd(CGP_{L} + q*{PS} + y*{RS} + v*16 + 8));
            }}
        }}
        SP_{L}(XP_{L}, CZTP_{L}, 0);
        if(m == 1){{
            PP_{L}(XP_{L}, CPP2_{L}, 1, 0);
            convoutp_{L}(XP_{L}, out1 + b*2*{L3});
            convoutp_{L}(XP_{L}, outm + b*2*{L3});
            continue;
        }}
        PP_{L}(XP_{L}, CPP2_{L}, 3, SNP_{L});
        convoutp_{L}(SNP_{L}, out1 + b*2*{L3});
        long t = 2;
        while(1){{
            SP_{L}(XP_{L}, CZTP_{L}, t==m ? 1 : 2);
            if(t == m) break;
            t++;
            PP_{L}(XP_{L}, CPP2_{L}, t==m ? 1 : 2, 0);
            if(t == m) break;
            t++;
        }}
        convoutp_{L}(XP_{L}, outm + b*2*{L3});
    }}
}}
"""
