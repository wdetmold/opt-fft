import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps, cmul_const, W
from genb2 import facs, maps

def p1_stage(L, name, which, RW, SRW):
    """pass1 row-stage over one plane: which=1: DFT_N1 rows inm[:,j2] -> SP rows (j2*N1+k1);
       which=2: DFT_N2 SP rows (j2*N1+k1) -> plane rows outm[k1][k2]. slots unrolled (ZB)."""
    N1,N2,mode = facs(L)
    inm, outm = maps(L)
    ZB = (L+7)//8
    body=[]
    if which==1:
        for j2 in range(N2):
            u = Emit(f'a{j2}_')
            for cc in range(ZB):
                xs=[(u.v(f"_mm512_load_pd(P + {inm[j1][j2]*RW + cc*16})"), u.v(f"_mm512_load_pd(P + {inm[j1][j2]*RW + cc*16+8})")) for j1 in range(N1)]
                ys = DFTS[N1](u, xs)
                for k1 in range(N1):
                    y = ys[k1]
                    if mode=='ct' and (k1*j2)%64: y = cmul_const(u, y, *W(64,k1*j2))
                    u.raw(f"_mm512_store_pd(SP + {(j2*N1+k1)*SRW + cc*16}, {y[0]});")
                    u.raw(f"_mm512_store_pd(SP + {(j2*N1+k1)*SRW + cc*16+8}, {y[1]});")
            decls = "\n        ".join(u.const_decls())
            body.append(f"    {{\n        {decls}\n{u.code(indent='        ')}\n    }}\n    __asm__ volatile(\"\" ::: \"memory\");")
        return f"static void __attribute__((noinline)) {name}(const double* restrict P, double* restrict SP){{\n" + "\n".join(body) + "\n}\n"
    else:
        for k1 in range(N1):
            u = Emit(f'b{k1}_')
            for cc in range(ZB):
                xs=[(u.v(f"_mm512_load_pd(SP + {(j2*N1+k1)*SRW + cc*16})"), u.v(f"_mm512_load_pd(SP + {(j2*N1+k1)*SRW + cc*16+8})")) for j2 in range(N2)]
                ys = DFTS[N2](u, xs)
                for k2 in range(N2):
                    k = outm[k1][k2]
                    u.raw(f"_mm512_store_pd(P + {k*RW + cc*16}, {ys[k2][0]});")
                    u.raw(f"_mm512_store_pd(P + {k*RW + cc*16+8}, {ys[k2][1]});")
            decls = "\n        ".join(u.const_decls())
            body.append(f"    {{\n        {decls}\n{u.code(indent='        ')}\n    }}\n    __asm__ volatile(\"\" ::: \"memory\");")
        return f"static void __attribute__((noinline)) {name}(double* restrict P, const double* restrict SP){{\n" + "\n".join(body) + "\n}\n"

def p2_stage(L, name, which, PL, CPL, TRW, WS):
    """pass2 striped plane-stage: which=1: src V planes inm[:,j2] -> T rows (j2*N1+k1) [+ twiddle];
       which=2: T rows -> V planes outm[k1][k2] with MAPSTORE and c planes (k1*N2+k2).
       pos loop runtime over WS slots. TRW = T row stride (doubles) = WS*16."""
    N1,N2,mode = facs(L)
    inm, outm = maps(L)
    body=[]
    if which==1:
        for j2 in range(N2):
            u = Emit(f'c{j2}_')
            for j1 in range(N1):
                u.raw(f"_mm_prefetch((const char*)(V + {inm[j1][j2]}*{PL} + pos*16 + {WS*16}), _MM_HINT_T0);")
                u.raw(f"_mm_prefetch((const char*)(V + {inm[j1][j2]}*{PL} + pos*16 + {WS*16+8}), _MM_HINT_T0);")
            xs=[(u.v(f"_mm512_load_pd(V + {inm[j1][j2]}*{PL} + pos*16)"), u.v(f"_mm512_load_pd(V + {inm[j1][j2]}*{PL} + pos*16+8)")) for j1 in range(N1)]
            ys = DFTS[N1](u, xs)
            for k1 in range(N1):
                y = ys[k1]
                if mode=='ct' and (k1*j2)%64: y = cmul_const(u, y, *W(64,k1*j2))
                u.raw(f"_mm512_store_pd(T + {(j2*N1+k1)*TRW} + pos*16, {y[0]});")
                u.raw(f"_mm512_store_pd(T + {(j2*N1+k1)*TRW} + pos*16+8, {y[1]});")
            decls = "\n        ".join(u.const_decls())
            body.append(f"    {{\n        {decls}\n        for(long pos=0; pos<{WS}; pos++){{\n{u.code(indent='            ')}\n        }}\n    }}\n    __asm__ volatile(\"\" ::: \"memory\");")
        return f"static void __attribute__((noinline)) {name}(const double* restrict V, double* restrict T){{\n" + "\n".join(body) + "\n}\n"
    else:
        for k1 in range(N1):
            u = Emit(f'd{k1}_')
            xs=[(u.v(f"_mm512_load_pd(T + {(j2*N1+k1)*TRW} + pos*16)"), u.v(f"_mm512_load_pd(T + {(j2*N1+k1)*TRW} + pos*16+8)")) for j2 in range(N2)]
            ys = DFTS[N2](u, xs)
            for k2 in range(N2):
                k = outm[k1][k2]
                u.raw(f"MAPSTORE2({ys[k2][0]}, {ys[k2][1]}, V, {k}*{PL} + pos*16, cb, {(k1*N2+k2)}*{CPL} + pos*16);")
            decls = "\n        ".join(u.const_decls())
            body.append(f"    {{\n        {decls}\n        for(long pos=0; pos<{WS}; pos++){{\n{u.code(indent='            ')}\n        }}\n    }}\n    __asm__ volatile(\"\" ::: \"memory\");")
        return f"static void __attribute__((noinline)) {name}(double* restrict V, const double* restrict T, const double* restrict cb){{\n" + "\n".join(body) + "\n}\n"

def gen_size3(L):
    N1,N2,mode = facs(L)
    inm, outm = maps(L)
    ZB = (L+7)//8
    RW = ZB*16 + (8 if L==64 else 0)
    NR = 8*ZB
    PL = NR*RW + 8
    VOL = L*PL
    SRW = RW            # scratch plane row stride
    NSL = L*ZB          # real slots per plane
    WS = {36:36, 45:30, 64:32}[L]
    NST = NSL//WS if NSL%WS==0 else None
    assert NST, (L,NSL,WS)
    TRW = WS*16
    CPL = NR*ZB*16 + 8  # c plane stride (padded rows like V planes, zeroed)
    L3 = L*L*L
    # plane permutation for c: PERM[k] = stage slot of plane k
    PERM = [0]*L
    for k1 in range(N1):
        for k2 in range(N2):
            PERM[outm[k1][k2]] = k1*N2+k2
    s=[]
    s.append(p1_stage(L, f"p1a_{L}", 1, RW, SRW))
    s.append(p1_stage(L, f"p1b_{L}", 2, RW, SRW))
    s.append(p2_stage(L, f"p2a_{L}", 1, PL, CPL, TRW, WS))
    s.append(p2_stage(L, f"p2b_{L}", 2, PL, CPL, TRW, WS))
    # reuse ptr_, conv_in_, conv_out_ generated elsewhere (same names as v8b)
    s.append(f"""
static double SPL_{L}[{N1*N2}*{SRW}] ALIGN64;
static double TB_{L}[{N1*N2}*{TRW} + 64] ALIGN64;
static const int CPERM_{L}[{L}] = {{ {", ".join(str(PERM[k]) for k in range(L))} }};
static void step3_{L}(double* restrict V, const double* restrict CP){{
    for(int x=0;x<{L};x++){{
        double* P = V + (long)x*{PL};
        p1a_{L}(P, SPL_{L});
        p1b_{L}(P, SPL_{L});
        ptr_{L}(P);
        p1a_{L}(P, SPL_{L});
        p1b_{L}(P, SPL_{L});
    }}
    for(int st=0; st<{NST}; st++){{
        long o = (long)st*{WS}*16;
        p2a_{L}(V + o, TB_{L});
        p2b_{L}(V + o, TB_{L}, CP + o);
    }}
}}
// c conv: plane-major, stage-permuted, both parities; packed rows (no pad rows inside plane)
static void conv_c3_{L}(const double* restrict src, double* restrict CC0, double* restrict CC1){{
    for(int k=0;k<{L};k++){{
        double* d0 = CC0 + (long)CPERM_{L}[k]*{CPL};
        double* d1 = CC1 + (long)CPERM_{L}[k]*{CPL};
        for(int r=0;r<{L};r++){{
            // parity0 plane layout: row r=y, lanes z
            const double* sp = src + (((long)k*{L}+r)*{L})*2;
            for(int g=0; g<{ZB}; g++){{
                if(8*g+8 <= {L}){{
                    __m512d a = _mm512_loadu_pd(sp + 16*g);
                    __m512d b = _mm512_loadu_pd(sp + 16*g + 8);
                    _mm512_store_pd(d0 + ((long)r*{ZB}+g)*16,     _mm512_permutex2var_pd(a, IDX_RE, b));
                    _mm512_store_pd(d0 + ((long)r*{ZB}+g)*16 + 8, _mm512_permutex2var_pd(a, IDX_IM, b));
                }} else {{
                    double tre[8] ALIGN64={{0,0,0,0,0,0,0,0}}, tim[8] ALIGN64={{0,0,0,0,0,0,0,0}};
                    for(int t=0; 8*g+t<{L}; t++){{ tre[t]=sp[2*(8*g+t)]; tim[t]=sp[2*(8*g+t)+1]; }}
                    _mm512_store_pd(d0 + ((long)r*{ZB}+g)*16, _mm512_load_pd(tre));
                    _mm512_store_pd(d0 + ((long)r*{ZB}+g)*16 + 8, _mm512_load_pd(tim));
                }}
            }}
        }}
        for(int r={L}; r<{NR}; r++){{ memset(d0 + (long)r*{ZB}*16, 0, {ZB}*16*8); memset(d1 + (long)r*{ZB}*16, 0, {ZB}*16*8); }}
        // parity1: row r=z, lanes y: transpose of parity0 plane: CC1[r*ZB+g][l] = CC0[(8g+l)*ZB + r/8][r%8]
        for(int Rb=0; Rb<{ZB}; Rb++)
            for(int g=0; g<{ZB}; g++){{
                const double* s0 = d0 + ((long)(8*g)*{ZB} + Rb)*16;
                double* dd = d1 + ((long)(8*Rb)*{ZB} + g)*16;
                const long ES = {ZB}*16;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                for(int half=0; half<2; half++){{
                    r0=_mm512_load_pd(s0+half*8); r1=_mm512_load_pd(s0+ES+half*8); r2=_mm512_load_pd(s0+2*ES+half*8); r3=_mm512_load_pd(s0+3*ES+half*8);
                    r4=_mm512_load_pd(s0+4*ES+half*8); r5=_mm512_load_pd(s0+5*ES+half*8); r6=_mm512_load_pd(s0+6*ES+half*8); r7=_mm512_load_pd(s0+7*ES+half*8);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(dd+half*8,o0); _mm512_store_pd(dd+ES+half*8,o1); _mm512_store_pd(dd+2*ES+half*8,o2); _mm512_store_pd(dd+3*ES+half*8,o3);
                    _mm512_store_pd(dd+4*ES+half*8,o4); _mm512_store_pd(dd+5*ES+half*8,o5); _mm512_store_pd(dd+6*ES+half*8,o6); _mm512_store_pd(dd+7*ES+half*8,o7);
                }}
            }}
    }}
}}
static double* V3_{L}; static double* D0_{L}; static double* D1_{L};
void run3_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!V3_{L}){{ V3_{L} = alloc_arena({VOL}*8 + 4096); D0_{L} = alloc_arena((long){L}*{CPL}*8 + 4096)+64; D1_{L} = alloc_arena((long){L}*{CPL}*8 + 4096)+128; }}
    for(long b=0;b<B;b++){{
        long off = b*(long){L3}*2;
        conv_in_{L}(x0 + off, V3_{L});
        conv_c3_{L}(c + off, D0_{L}, D1_{L});
        for(long t=0;t<m;t++){{
            step3_{L}(V3_{L}, (t%2==0)? D1_{L} : D0_{L});
            if(t==0 && m>1) conv_out_{L}(V3_{L}, out1 + off, 1);
        }}
        conv_out_{L}(V3_{L}, outm + off, (int)(m%2));
        if(m==1) memcpy(out1 + off, outm + off, (long){L3}*16);
    }}
}}
""")
    return "\n".join(s)
