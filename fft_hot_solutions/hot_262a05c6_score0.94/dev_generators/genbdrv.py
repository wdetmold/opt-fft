import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps, cmul_const, W
from genprime import MAP_MACRO

def col_dft(L, name, fuse_map, stride, cstride=None, ncol=1):
    """straight-line two-stage DFT-L on a vec column; strides BAKED."""
    if L == 36: N1,N2,mode = 4,9,'pfa'
    elif L == 45: N1,N2,mode = 5,9,'pfa'
    elif L == 64: N1,N2,mode = 8,8,'ct'
    else: raise ValueError
    if mode=='pfa':
        inm, outm = pfa_maps(N1,N2)
    else:
        inm = [[8*j1+j2 for j2 in range(N2)] for j1 in range(N1)]
        outm = [[k1+8*k2 for k2 in range(N2)] for k1 in range(N1)]
    e1 = Emit('a')
    for j2 in range(N2):
        for cc in range(ncol):
            xs = []
            for j1 in range(N1):
                off = inm[j1][j2]*stride + 16*cc
                if fuse_map:
                    e1.raw(f"_mm_prefetch((const char*)(col + {off} + 16), _MM_HINT_T0);")
                    e1.raw(f"_mm_prefetch((const char*)(col + {off} + 24), _MM_HINT_T0);")
                xs.append((e1.v(f"_mm512_load_pd(col + {off})"), e1.v(f"_mm512_load_pd(col + {off} + 8)")))
            ys = DFTS[N1](e1, xs)
            for k1 in range(N1):
                y = ys[k1]
                if mode=='ct' and (k1*j2)%64 != 0:
                    c,s = W(64, k1*j2)
                    y = cmul_const(e1, y, c, s)
                e1.raw(f"_mm512_store_pd(SC_{name} + {((j2*N1+k1)*ncol+cc)*16}, {y[0]});")
                e1.raw(f"_mm512_store_pd(SC_{name} + {((j2*N1+k1)*ncol+cc)*16+8}, {y[1]});")
    e2 = Emit('b')
    for k1 in range(N1):
        for cc in range(ncol):
            xs = []
            for j2 in range(N2):
                off = ((j2*N1+k1)*ncol+cc)*16
                xs.append((e2.v(f"_mm512_load_pd(SC_{name} + {off})"), e2.v(f"_mm512_load_pd(SC_{name} + {off} + 8)")))
            ys = DFTS[N2](e2, xs)
            for k2 in range(N2):
                k = outm[k1][k2]
                y = ys[k2]
                if fuse_map:
                    e2.raw(f"MAPSTORE2({y[0]}, {y[1]}, col, {k*stride + 16*cc}, cb, {k*cstride + 16*cc});")
                else:
                    e2.raw(f"_mm512_store_pd(col + {k*stride + 16*cc}, {y[0]});")
                    e2.raw(f"_mm512_store_pd(col + {k*stride + 16*cc} + 8, {y[1]});")
    args = "double* restrict col"
    if fuse_map: args += ", const double* restrict cb"
    decls1 = "\n    ".join(e1.const_decls())
    decls2 = "\n    ".join(e2.const_decls())
    return f"""
static double SC_{name}[{N2}*{N1}*{ncol}*16] ALIGN64;
static void __attribute__((noinline)) {name}({args}){{
    {{
    {decls1}
{e1.code()}
    }}
    __asm__ volatile("" ::: "memory");
    {{
    {decls2}
{e2.code()}
    }}
}}
"""

def col_dft_pipe(L, name, fuse_map, stride, cstride=None):
    """pipelined: stage2 of CURRENT column (from SCA) interleaved with stage1 of NEXT column (into SCB).
       emits two variants: name_ab (SCA->read, SCB->write) and name_ba."""
    if L == 36: N1,N2,mode = 4,9,'pfa'
    elif L == 45: N1,N2,mode = 5,9,'pfa'
    elif L == 64: N1,N2,mode = 8,8,'ct'
    else: raise ValueError
    if mode=='pfa':
        inm, outm = pfa_maps(N1,N2)
    else:
        inm = [[8*j1+j2 for j2 in range(N2)] for j1 in range(N1)]
        outm = [[k1+8*k2 for k2 in range(N2)] for k1 in range(N1)]
    def emit_s1_unit(e, j2, scw):
        xs = []
        for j1 in range(N1):
            off = inm[j1][j2]*stride
            xs.append((e.v(f"_mm512_load_pd(nxt + {off})"), e.v(f"_mm512_load_pd(nxt + {off} + 8)")))
        ys = DFTS[N1](e, xs)
        for k1 in range(N1):
            y = ys[k1]
            if mode=='ct' and (k1*j2)%64 != 0:
                c,s = W(64, k1*j2)
                y = cmul_const(e, y, c, s)
            e.raw(f"_mm512_store_pd({scw} + {(j2*N1+k1)*16}, {y[0]});")
            e.raw(f"_mm512_store_pd({scw} + {(j2*N1+k1)*16+8}, {y[1]});")
    def emit_s2_unit(e, k1, scr):
        xs = []
        for j2 in range(N2):
            off = (j2*N1+k1)*16
            xs.append((e.v(f"_mm512_load_pd({scr} + {off})"), e.v(f"_mm512_load_pd({scr} + {off} + 8)")))
        ys = DFTS[N2](e, xs)
        for k2 in range(N2):
            k = outm[k1][k2]
            y = ys[k2]
            if fuse_map:
                e.raw(f"MAPSTORE2({y[0]}, {y[1]}, col, {k*stride}, cb, {k*cstride});")
            else:
                e.raw(f"_mm512_store_pd(col + {k*stride}, {y[0]});")
                e.raw(f"_mm512_store_pd(col + {k*stride} + 8, {y[1]});")
    out = []
    for (sfx, scr, scw) in (("ab", f"SCA_{name}", f"SCB_{name}"), ("ba", f"SCB_{name}", f"SCA_{name}")):
        e = Emit('p')
        # interleave: schedule S1 units among S2 units
        s1q = list(range(N2))
        per = (N2 + N1 - 1)//N1
        for k1 in range(N1):
            emit_s2_unit(e, k1, scr)
            for _ in range(per):
                if s1q: emit_s1_unit(e, s1q.pop(0), scw)
        while s1q: emit_s1_unit(e, s1q.pop(0), scw)
        decls = "\n    ".join(e.const_decls())
        args = "double* restrict col, const double* restrict nxt"
        if fuse_map: args += ", const double* restrict cb"
        out.append(f"""static void __attribute__((noinline)) {name}_{sfx}({args}){{
    {decls}
{e.code()}
}}
""")
    # also a starter (stage1 only into SCA) and finisher (stage2 only from given sc)
    e = Emit('q')
    for j2 in range(N2):
        emit_s1_unit(e, j2, f"SCA_{name}")
    decls = "\n    ".join(e.const_decls())
    out.append(f"""static void __attribute__((noinline)) {name}_st(const double* restrict nxt){{
    {decls}
{e.code()}
}}
""")
    for (sfx, scr) in (("fa", f"SCA_{name}"), ("fb", f"SCB_{name}")):
        e = Emit('r')
        for k1 in range(N1):
            emit_s2_unit(e, k1, scr)
        decls = "\n    ".join(e.const_decls())
        args = "double* restrict col"
        if fuse_map: args += ", const double* restrict cb"
        out.append(f"""static void __attribute__((noinline)) {name}_{sfx}({args}){{
    {decls}
{e.code()}
}}
""")
    pre = f"static double SCA_{name}[{L}*16] ALIGN64;\nstatic double SCB_{name}[{L}*16] ALIGN64;\n"
    return pre + "\n".join(out)

def gen_size(L):
    ZB = (L+7)//8
    RW = ZB*16   # packed rows (pass structure is row-streamed; no column sweeps)
    NR = 8*ZB           # padded rows per plane
    PL = NR*RW + 8      # plane stride (pad: breaks 4K aliasing across planes)
    VOL = L*PL
    s = []
    s.append(col_dft(L, f"cdft{L}", False, RW))
    s.append(col_dft(L, f"cdft{L}m", True, PL, 16))
    s.append(col_dft_pipe(L, f"pw{L}", False, RW))
    s.append(col_dft_pipe(L, f"px{L}", True, PL, 16))
    # plane transpose: swap (row r, lane of group g) <-> ...
    s.append(f"""
static void __attribute__((noinline)) ptr_{L}(double* restrict P){{
    for(int a=0;a<{ZB};a++){{
        // diagonal tile
        {{
            double* ta = P + (long)8*a*{RW} + a*16;
            __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
            // re
            r0=_mm512_load_pd(ta); r1=_mm512_load_pd(ta+{RW}); r2=_mm512_load_pd(ta+2*{RW}); r3=_mm512_load_pd(ta+3*{RW});
            r4=_mm512_load_pd(ta+4*{RW}); r5=_mm512_load_pd(ta+5*{RW}); r6=_mm512_load_pd(ta+6*{RW}); r7=_mm512_load_pd(ta+7*{RW});
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(ta,o0); _mm512_store_pd(ta+{RW},o1); _mm512_store_pd(ta+2*{RW},o2); _mm512_store_pd(ta+3*{RW},o3);
            _mm512_store_pd(ta+4*{RW},o4); _mm512_store_pd(ta+5*{RW},o5); _mm512_store_pd(ta+6*{RW},o6); _mm512_store_pd(ta+7*{RW},o7);
            // im
            r0=_mm512_load_pd(ta+8); r1=_mm512_load_pd(ta+{RW}+8); r2=_mm512_load_pd(ta+2*{RW}+8); r3=_mm512_load_pd(ta+3*{RW}+8);
            r4=_mm512_load_pd(ta+4*{RW}+8); r5=_mm512_load_pd(ta+5*{RW}+8); r6=_mm512_load_pd(ta+6*{RW}+8); r7=_mm512_load_pd(ta+7*{RW}+8);
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(ta+8,o0); _mm512_store_pd(ta+{RW}+8,o1); _mm512_store_pd(ta+2*{RW}+8,o2); _mm512_store_pd(ta+3*{RW}+8,o3);
            _mm512_store_pd(ta+4*{RW}+8,o4); _mm512_store_pd(ta+5*{RW}+8,o5); _mm512_store_pd(ta+6*{RW}+8,o6); _mm512_store_pd(ta+7*{RW}+8,o7);
        }}
        for(int b=a+1;b<{ZB};b++){{
            double* ta = P + (long)8*a*{RW} + b*16;
            double* tb = P + (long)8*b*{RW} + a*16;
            for(int half=0; half<2; half++){{
                double* pa = ta + 8*half; double* pb = tb + 8*half;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                __m512d q0,q1,q2,q3,q4,q5,q6,q7,p0,p1,p2,p3,p4,p5,p6,p7;
                r0=_mm512_load_pd(pa); r1=_mm512_load_pd(pa+{RW}); r2=_mm512_load_pd(pa+2*{RW}); r3=_mm512_load_pd(pa+3*{RW});
                r4=_mm512_load_pd(pa+4*{RW}); r5=_mm512_load_pd(pa+5*{RW}); r6=_mm512_load_pd(pa+6*{RW}); r7=_mm512_load_pd(pa+7*{RW});
                q0=_mm512_load_pd(pb); q1=_mm512_load_pd(pb+{RW}); q2=_mm512_load_pd(pb+2*{RW}); q3=_mm512_load_pd(pb+3*{RW});
                q4=_mm512_load_pd(pb+4*{RW}); q5=_mm512_load_pd(pb+5*{RW}); q6=_mm512_load_pd(pb+6*{RW}); q7=_mm512_load_pd(pb+7*{RW});
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                TR8(q0,q1,q2,q3,q4,q5,q6,q7,p0,p1,p2,p3,p4,p5,p6,p7);
                _mm512_store_pd(pb,o0); _mm512_store_pd(pb+{RW},o1); _mm512_store_pd(pb+2*{RW},o2); _mm512_store_pd(pb+3*{RW},o3);
                _mm512_store_pd(pb+4*{RW},o4); _mm512_store_pd(pb+5*{RW},o5); _mm512_store_pd(pb+6*{RW},o6); _mm512_store_pd(pb+7*{RW},o7);
                _mm512_store_pd(pa,p0); _mm512_store_pd(pa+{RW},p1); _mm512_store_pd(pa+2*{RW},p2); _mm512_store_pd(pa+3*{RW},p3);
                _mm512_store_pd(pa+4*{RW},p4); _mm512_store_pd(pa+5*{RW},p5); _mm512_store_pd(pa+6*{RW},p6); _mm512_store_pd(pa+7*{RW},p7);
            }}
        }}
    }}
}}
#define COLS_{L}(P) do{{ \
    pw{L}_st((P)); \
    int _g = 0; \
    for(; _g+2<={ZB}-1; _g+=2){{ pw{L}_ab((P)+_g*16, (P)+(_g+1)*16); pw{L}_ba((P)+(_g+1)*16, (P)+(_g+2)*16); }} \
    if(_g < {ZB}-1){{ pw{L}_ab((P)+_g*16, (P)+(_g+1)*16); pw{L}_fb((P)+(_g+1)*16); }} \
    else pw{L}_fa((P)+_g*16); \
}}while(0)
static double* PCOL_{L}[{L}*{ZB}+1];
static void init_pcol_{L}(double* V){{
    long n=0;
    for(int r=0;r<{L};r++) for(int g=0;g<{ZB};g++) PCOL_{L}[n++] = V + (long)r*{RW} + g*16;
    PCOL_{L}[n] = PCOL_{L}[n-1];
}}
static void step_{L}(double* restrict V, const double* restrict CP){{
    for(int x=0;x<{L};x++){{
        double* P = V + (long)x*{PL};
        COLS_{L}(P);
        ptr_{L}(P);
        COLS_{L}(P);
    }}
    {{
        px{L}_st(PCOL_{L}[0]);
        long i = 0;
        const long NCOL = (long){L}*{ZB};
        for(; i+2 <= NCOL-1; i += 2){{
            px{L}_ab(PCOL_{L}[i], PCOL_{L}[i+1], CP + i*{L}*16);
            px{L}_ba(PCOL_{L}[i+1], PCOL_{L}[i+2], CP + (i+1)*{L}*16);
        }}
        if(i < NCOL-1){{ px{L}_ab(PCOL_{L}[i], PCOL_{L}[i+1], CP + i*{L}*16); px{L}_fb(PCOL_{L}[NCOL-1], CP + (NCOL-1)*{L}*16); }}
        else px{L}_fa(PCOL_{L}[NCOL-1], CP + (NCOL-1)*{L}*16);
    }}
}}
// conv: numpy volume (complex interleaved, C order) <-> internal [x][row][g][16]
static void conv_in_{L}(const double* restrict src, double* restrict V){{
    for(int x=0;x<{L};x++){{
        for(int y=0;y<{L};y++){{
            const double* s = src + ((long)x*{L}+y)*{L}*2;
            double* d = V + (long)x*{PL} + (long)y*{RW};
            int z=0;
            for(; z+8<={L}; z+=8){{
                __m512d a = _mm512_loadu_pd(s + 2*z);
                __m512d b = _mm512_loadu_pd(s + 2*z + 8);
                __m512d re = _mm512_permutex2var_pd(a, IDX_RE, b);
                __m512d im = _mm512_permutex2var_pd(a, IDX_IM, b);
                _mm512_store_pd(d + (z/8)*16, re);
                _mm512_store_pd(d + (z/8)*16 + 8, im);
            }}
            if(z<{L}){{
                double tre[8]={{0,0,0,0,0,0,0,0}}, tim[8]={{0,0,0,0,0,0,0,0}};
                for(int t=0; z+t<{L}; t++){{ tre[t]=s[2*(z+t)]; tim[t]=s[2*(z+t)+1]; }}
                _mm512_store_pd(d + (z/8)*16, _mm512_loadu_pd(tre));
                _mm512_store_pd(d + (z/8)*16 + 8, _mm512_loadu_pd(tim));
            }}
        }}
        // zero pad rows
        for(int y={L}; y<{NR}; y++) memset(V + (long)x*{PL} + (long)y*{RW}, 0, {RW}*8);
    }}
}}
static void conv_out_{L}(const double* restrict V, double* restrict dst, int parity){{
    // parity 0: rows=y lanes=z ; parity 1: rows=z lanes=y
    for(int x=0;x<{L};x++){{
        const double* P = V + (long)x*{PL};
        if(parity==0){{
            for(int y=0;y<{L};y++){{
                double* d = dst + ((long)x*{L}+y)*{L}*2;
                const double* p = P + (long)y*{RW};
                int z=0;
                for(; z+8<={L}; z+=8){{
                    __m512d re = _mm512_load_pd(p + (z/8)*16);
                    __m512d im = _mm512_load_pd(p + (z/8)*16 + 8);
                    __m512d a = _mm512_permutex2var_pd(re, IDX_ILA, im);
                    __m512d b = _mm512_permutex2var_pd(re, IDX_ILB, im);
                    _mm512_storeu_pd(d + 2*z, a);
                    _mm512_storeu_pd(d + 2*z + 8, b);
                }}
                for(; z<{L}; z++){{ d[2*z] = p[(z/8)*16 + (z%8)]; d[2*z+1] = p[(z/8)*16 + 8 + (z%8)]; }}
            }}
        }} else {{
            // rows=z lanes=y: process 8z x 8y tiles: transpose then interleave-store per y
            for(int Z=0; Z+8<={L}; Z+=8){{
                for(int G=0; G*8<{L}; G++){{
                    const double* p0 = P + (long)Z*{RW} + G*16;
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,R0,R1,R2,R3,R4,R5,R6,R7;
                    __m512d i0,i1,i2,i3,i4,i5,i6,i7,I0,I1,I2,I3,I4,I5,I6,I7;
                    r0=_mm512_load_pd(p0);          r1=_mm512_load_pd(p0+{RW});   r2=_mm512_load_pd(p0+2*{RW}); r3=_mm512_load_pd(p0+3*{RW});
                    r4=_mm512_load_pd(p0+4*{RW});   r5=_mm512_load_pd(p0+5*{RW}); r6=_mm512_load_pd(p0+6*{RW}); r7=_mm512_load_pd(p0+7*{RW});
                    i0=_mm512_load_pd(p0+8);        i1=_mm512_load_pd(p0+{RW}+8); i2=_mm512_load_pd(p0+2*{RW}+8); i3=_mm512_load_pd(p0+3*{RW}+8);
                    i4=_mm512_load_pd(p0+4*{RW}+8); i5=_mm512_load_pd(p0+5*{RW}+8); i6=_mm512_load_pd(p0+6*{RW}+8); i7=_mm512_load_pd(p0+7*{RW}+8);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,R0,R1,R2,R3,R4,R5,R6,R7);
                    TR8(i0,i1,i2,i3,i4,i5,i6,i7,I0,I1,I2,I3,I4,I5,I6,I7);
                    int ylim = {L} - 8*G; if(ylim>8) ylim=8;
                    double* db = dst + (((long)x*{L} + 8*G)*{L} + Z)*2;
                    __m512d A,Bv;
                    #define OUTY(t, RR, II) if(t<ylim){{ \
                        A = _mm512_permutex2var_pd(RR, IDX_ILA, II); \
                        Bv= _mm512_permutex2var_pd(RR, IDX_ILB, II); \
                        _mm512_storeu_pd(db + (long)t*{L}*2, A); \
                        _mm512_storeu_pd(db + (long)t*{L}*2 + 8, Bv); }}
                    OUTY(0,R0,I0) OUTY(1,R1,I1) OUTY(2,R2,I2) OUTY(3,R3,I3)
                    OUTY(4,R4,I4) OUTY(5,R5,I5) OUTY(6,R6,I6) OUTY(7,R7,I7)
                    #undef OUTY
                }}
            }}
            for(int z={L} - ({L}%8); z<{L}; z++){{
                const double* p = P + (long)z*{RW};
                for(int y=0;y<{L};y++){{
                    double* d = dst + (((long)x*{L}+y)*{L} + z)*2;
                    d[0] = p[(y/8)*16 + (y%8)];
                    d[1] = p[(y/8)*16 + 8 + (y%8)];
                }}
            }}
        }}
    }}
}}

static void conv_cc_{L}(const double* restrict src, double* restrict CC0, double* restrict CC1){{
    // parity 0 (used when pass2 input rows=y, lanes=z): element (x=k, y=r, z=8g+l)
    for(int r=0;r<{L};r++)
        for(int g=0;g<{ZB};g++){{
            double* d = CC0 + (((long)r*{ZB}+g)*{L})*16;
            for(int k=0;k<{L};k++){{
                const double* s = src + (((long)k*{L}+r)*{L} + 8*g)*2;
                if(8*g+8 <= {L}){{
                    __m512d a = _mm512_loadu_pd(s);
                    __m512d b = _mm512_loadu_pd(s+8);
                    _mm512_store_pd(d + (long)k*16,     _mm512_permutex2var_pd(a, IDX_RE, b));
                    _mm512_store_pd(d + (long)k*16 + 8, _mm512_permutex2var_pd(a, IDX_IM, b));
                }} else {{
                    double tre[8]={{0,0,0,0,0,0,0,0}}, tim[8]={{0,0,0,0,0,0,0,0}};
                    for(int t=0; 8*g+t<{L}; t++){{ tre[t]=s[2*t]; tim[t]=s[2*t+1]; }}
                    _mm512_store_pd(d + (long)k*16, _mm512_loadu_pd(tre));
                    _mm512_store_pd(d + (long)k*16 + 8, _mm512_loadu_pd(tim));
                }}
            }}
        }}
    // parity 1: CC1[r][g][k][l] = c(k, 8g+l, r) = CC0[8g+l][r/8][k][r%8] -> 8x8 lane/row transposes
    {{
        const long ES = (long){ZB}*{L}*16;   // stride between consecutive "r" rows in CC layout
        for(int R=0; R<{ZB}; R++)            // r-block
            for(int g=0; g<{ZB}; g++)
                for(int k=0; k<{L}; k++){{
                    const double* s0 = CC0 + ((long)(8*g)*{ZB} + R)*{L}*16 + (long)k*16;
                    double* d0 = CC1 + ((long)(8*R)*{ZB} + g)*{L}*16 + (long)k*16;
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    r0=_mm512_load_pd(s0); r1=_mm512_load_pd(s0+ES); r2=_mm512_load_pd(s0+2*ES); r3=_mm512_load_pd(s0+3*ES);
                    r4=_mm512_load_pd(s0+4*ES); r5=_mm512_load_pd(s0+5*ES); r6=_mm512_load_pd(s0+6*ES); r7=_mm512_load_pd(s0+7*ES);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(d0,o0); _mm512_store_pd(d0+ES,o1); _mm512_store_pd(d0+2*ES,o2); _mm512_store_pd(d0+3*ES,o3);
                    _mm512_store_pd(d0+4*ES,o4); _mm512_store_pd(d0+5*ES,o5); _mm512_store_pd(d0+6*ES,o6); _mm512_store_pd(d0+7*ES,o7);
                    r0=_mm512_load_pd(s0+8); r1=_mm512_load_pd(s0+ES+8); r2=_mm512_load_pd(s0+2*ES+8); r3=_mm512_load_pd(s0+3*ES+8);
                    r4=_mm512_load_pd(s0+4*ES+8); r5=_mm512_load_pd(s0+5*ES+8); r6=_mm512_load_pd(s0+6*ES+8); r7=_mm512_load_pd(s0+7*ES+8);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(d0+8,o0); _mm512_store_pd(d0+ES+8,o1); _mm512_store_pd(d0+2*ES+8,o2); _mm512_store_pd(d0+3*ES+8,o3);
                    _mm512_store_pd(d0+4*ES+8,o4); _mm512_store_pd(d0+5*ES+8,o5); _mm512_store_pd(d0+6*ES+8,o6); _mm512_store_pd(d0+7*ES+8,o7);
                }}
    }}
}}
static double* V_{L}; static double* C0_{L}; static double* C1_{L}; static double* CC0_{L}; static double* CC1_{L};
void bench_pass1_{L}(long reps){{
    for(long r=0;r<reps;r++)
        for(int x=0;x<{L};x++){{
            double* P = V_{L} + (long)x*{PL};
            COLS_{L}(P);
            ptr_{L}(P);
            COLS_{L}(P);
        }}
}}
void bench_cols_{L}(long reps){{
    for(long r=0;r<reps;r++)
        for(int x=0;x<{L};x++){{
            double* P = V_{L} + (long)x*{PL};
            COLS_{L}(P);
        }}
}}
void bench_tr_{L}(long reps){{
    for(long r=0;r<reps;r++)
        for(int x=0;x<{L};x++) ptr_{L}(V_{L} + (long)x*{PL});
}}
void bench_pass2_{L}(long reps){{
    for(long r=0;r<reps;r++)
        for(int rr=0;rr<{L};rr++){{
            for(int g=0; g<{ZB}; g++)
                cdft{L}m(V_{L} + (long)rr*{RW} + g*16, CC0_{L} + ((long)rr*{ZB} + g)*{L}*16);
        }}
}}
void run_{L}_g(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!V_{L}){{ V_{L} = alloc_arena({VOL}*8 + 4096); C0_{L} = alloc_arena({VOL}*8 + 4096); C1_{L} = alloc_arena({VOL}*8 + 4096); CC0_{L} = alloc_arena((long){L}*{L}*{ZB}*16*8 + 4096)+64; CC1_{L} = alloc_arena((long){L}*{L}*{ZB}*16*8 + 4096)+128; }}
    init_pcol_{L}(V_{L});
    for(long b=0;b<B;b++){{
        long off = b*(long){L*L*L}*2;
        conv_in_{L}(x0 + off, V_{L});
        conv_cc_{L}(c + off, CC0_{L}, CC1_{L});
        for(long t=0;t<m;t++){{
            step_{L}(V_{L}, (t%2==0)? CC1_{L} : CC0_{L});
            if(t==0 && m>1) conv_out_{L}(V_{L}, out1 + off, 1);
        }}
        conv_out_{L}(V_{L}, outm + off, (int)(m%2));
        if(m==1) memcpy(out1 + off, outm + off, (long){L*L*L}*16);
    }}
}}
""")
    return "\n".join(s)

PRE2 = r'''
static __m512i IDX_RE, IDX_IM, IDX_ILA, IDX_ILB;
static __m512d V_15B;
__attribute__((constructor)) static void init_idx(void){
    IDX_RE = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    IDX_IM = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    IDX_ILA = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    IDX_ILB = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    V_15B = _mm512_set1_pd(1.5);
}
#define MAPSTORE2(xr_, xi_, dst, off, cbase, coff) do{ \
    __m512d zr = _mm512_add_pd(xr_, _mm512_load_pd((cbase)+(coff))); \
    __m512d zi = _mm512_add_pd(xi_, _mm512_load_pd((cbase)+(coff)+8)); \
    __m512d mm = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, V_TINY)); \
    __m512d r0 = _mm512_rsqrt14_pd(mm); \
    __m512d mg0= _mm512_mul_pd(mm, r0); \
    __m512d t_ = _mm512_mul_pd(mg0, r0); \
    __m512d e_ = _mm512_fnmadd_pd(t_, V_HALF, V_15B); \
    __m512d mg1= _mm512_mul_pd(mg0, e_); \
    __m512d r1 = _mm512_mul_pd(r0, e_); \
    __m512d e3 = _mm512_fnmadd_pd(mg1, mg1, mm); \
    __m512d hr = _mm512_mul_pd(r1, V_HALF); \
    __m512d u  = _mm512_add_pd(V_ONE, mg1); \
    u = _mm512_fmadd_pd(e3, hr, u); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e4 = _mm512_fnmadd_pd(u, w0, V_ONE); \
    __m512d w1 = _mm512_fmadd_pd(w0, e4, w0); \
    __m512d ee = _mm512_mul_pd(e4, e4); \
    __m512d w2 = _mm512_fmadd_pd(w1, ee, w1); \
    _mm512_store_pd((dst)+(off),   _mm512_mul_pd(zr, w2)); \
    _mm512_store_pd((dst)+(off)+8, _mm512_mul_pd(zi, w2)); \
}while(0)
'''

if __name__ == "__main__":
    from gendrive import PRELUDE
    parts = [PRELUDE, PRE2]
    for L in (36,45,64):
        parts.append(gen_size(L))
    src = "\n".join(parts)
    open("v8b.c","w").write(src)
    print("wrote v8b.c", len(src))
