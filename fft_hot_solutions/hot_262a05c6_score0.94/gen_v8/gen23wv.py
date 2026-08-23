import numpy as np
from genprime2 import emit_prime_axis, tables

def gen_23_wv():
    p = 23
    ZB = 3; RW = ZB*16          # 48 doubles
    NR = 24
    PL = NR*RW + 8              # plane stride 1160 doubles
    VOL = p*PL
    L3 = p*p*p
    s = []
    # column codelets, baked strides
    s.append(emit_prime_axis(p, f"wz_{p}", RW))                                  # pass1 column (stride RW)
    s.append(emit_prime_axis(p, f"wx_{p}", PL, fuse_map=True, cstride=PL))       # pass2 pencil (stride PL), c same layout
    # plane transpose 24x24 grid of (rows x lane-groups): tiles 3x3, reuse TR8
    s.append(f"""
static void wtr_{p}(double* restrict P){{
    for(int a=0;a<{ZB};a++){{
        {{
            double* ta = P + (long)8*a*{RW} + a*16;
            for(int half=0; half<2; half++){{
                double* pa = ta + 8*half;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(pa); r1=_mm512_load_pd(pa+{RW}); r2=_mm512_load_pd(pa+2*{RW}); r3=_mm512_load_pd(pa+3*{RW});
                r4=_mm512_load_pd(pa+4*{RW}); r5=_mm512_load_pd(pa+5*{RW}); r6=_mm512_load_pd(pa+6*{RW}); r7=_mm512_load_pd(pa+7*{RW});
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                _mm512_store_pd(pa,o0); _mm512_store_pd(pa+{RW},o1); _mm512_store_pd(pa+2*{RW},o2); _mm512_store_pd(pa+3*{RW},o3);
                _mm512_store_pd(pa+4*{RW},o4); _mm512_store_pd(pa+5*{RW},o5); _mm512_store_pd(pa+6*{RW},o6); _mm512_store_pd(pa+7*{RW},o7);
            }}
        }}
        for(int b=a+1;b<{ZB};b++){{
            double* ta = P + (long)8*a*{RW} + b*16;
            double* tb = P + (long)8*b*{RW} + a*16;
            for(int half=0; half<2; half++){{
                double* pa = ta + 8*half; double* pb = tb + 8*half;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                __m512d q0,q1,q2,q3,q4,q5,q6,q7,s0,s1,s2,s3,s4,s5,s6,s7;
                r0=_mm512_load_pd(pa); r1=_mm512_load_pd(pa+{RW}); r2=_mm512_load_pd(pa+2*{RW}); r3=_mm512_load_pd(pa+3*{RW});
                r4=_mm512_load_pd(pa+4*{RW}); r5=_mm512_load_pd(pa+5*{RW}); r6=_mm512_load_pd(pa+6*{RW}); r7=_mm512_load_pd(pa+7*{RW});
                q0=_mm512_load_pd(pb); q1=_mm512_load_pd(pb+{RW}); q2=_mm512_load_pd(pb+2*{RW}); q3=_mm512_load_pd(pb+3*{RW});
                q4=_mm512_load_pd(pb+4*{RW}); q5=_mm512_load_pd(pb+5*{RW}); q6=_mm512_load_pd(pb+6*{RW}); q7=_mm512_load_pd(pb+7*{RW});
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                TR8(q0,q1,q2,q3,q4,q5,q6,q7,s0,s1,s2,s3,s4,s5,s6,s7);
                _mm512_store_pd(pb,o0); _mm512_store_pd(pb+{RW},o1); _mm512_store_pd(pb+2*{RW},o2); _mm512_store_pd(pb+3*{RW},o3);
                _mm512_store_pd(pb+4*{RW},o4); _mm512_store_pd(pb+5*{RW},o5); _mm512_store_pd(pb+6*{RW},o6); _mm512_store_pd(pb+7*{RW},o7);
                _mm512_store_pd(pa,s0); _mm512_store_pd(pa+{RW},s1); _mm512_store_pd(pa+2*{RW},s2); _mm512_store_pd(pa+3*{RW},s3);
                _mm512_store_pd(pa+4*{RW},s4); _mm512_store_pd(pa+5*{RW},s5); _mm512_store_pd(pa+6*{RW},s6); _mm512_store_pd(pa+7*{RW},s7);
            }}
        }}
    }}
}}
static void wstep_{p}(double* restrict V, const double* restrict CP){{
    for(int x=0;x<{p};x++){{
        double* P = V + (long)x*{PL};
        for(int g=0;g<{ZB};g++) wz_{p}(P + g*16);
        wtr_{p}(P);
        for(int g=0;g<{ZB};g++) wz_{p}(P + g*16);
    }}
    for(int r=0;r<{p};r++)
        for(int g=0;g<{ZB};g++)
            wx_{p}(V + (long)r*{RW} + g*16, CP + (long)r*{RW} + g*16);
}}
static void wconv_in_{p}(const double* restrict src, double* restrict V){{
    for(int x=0;x<{p};x++){{
        for(int y=0;y<{p};y++){{
            const double* sp = src + ((long)x*{p}+y)*{p}*2;
            double* d = V + (long)x*{PL} + (long)y*{RW};
            int z=0;
            for(; z+8<={p}; z+=8){{
                __m512d a = _mm512_loadu_pd(sp + 2*z);
                __m512d b = _mm512_loadu_pd(sp + 2*z + 8);
                __m512d re = _mm512_permutex2var_pd(a, IDX_RE, b);
                __m512d im = _mm512_permutex2var_pd(a, IDX_IM, b);
                _mm512_store_pd(d + (z/8)*16, re);
                _mm512_store_pd(d + (z/8)*16 + 8, im);
            }}
            {{
                double tre[8]={{0,0,0,0,0,0,0,0}}, tim[8]={{0,0,0,0,0,0,0,0}};
                for(int t=0; z+t<{p}; t++){{ tre[t]=sp[2*(z+t)]; tim[t]=sp[2*(z+t)+1]; }}
                _mm512_store_pd(d + (z/8)*16, _mm512_loadu_pd(tre));
                _mm512_store_pd(d + (z/8)*16 + 8, _mm512_loadu_pd(tim));
            }}
        }}
        for(int y={p}; y<{NR}; y++) memset(V + (long)x*{PL} + (long)y*{RW}, 0, {RW}*8);
        memset(V + (long)x*{PL} + (long){NR}*{RW}, 0, ({PL}-{NR}*{RW})*8);
    }}
}}
static void wconv_out_{p}(const double* restrict V, double* restrict dst, int parity){{
    for(int x=0;x<{p};x++){{
        const double* P = V + (long)x*{PL};
        if(parity==0){{
            for(int y=0;y<{p};y++){{
                double* d = dst + ((long)x*{p}+y)*{p}*2;
                const double* pp = P + (long)y*{RW};
                for(int z=0; z<{p}; z++){{ d[2*z] = pp[(z/8)*16 + (z%8)]; d[2*z+1] = pp[(z/8)*16 + 8 + (z%8)]; }}
            }}
        }} else {{
            for(int z=0;z<{p};z++){{
                const double* pp = P + (long)z*{RW};
                for(int y=0;y<{p};y++){{
                    double* d = dst + (((long)x*{p}+y)*{p} + z)*2;
                    d[0] = pp[(y/8)*16 + (y%8)];
                    d[1] = pp[(y/8)*16 + 8 + (y%8)];
                }}
            }}
        }}
    }}
}}
static double* WV_{p} = 0; static double* WC0_{p} = 0; static double* WC1_{p} = 0;
void run3_{p}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!WV_{p}){{ WV_{p} = alloc_arena({VOL}*8+4096); WC0_{p} = alloc_arena({VOL}*8+4096)+64; WC1_{p} = alloc_arena({VOL}*8+4096)+128; }}
    for(long b=0;b<B;b++){{
        long off = b*(long){L3}*2;
        wconv_in_{p}(x0 + off, WV_{p});
        wconv_in_{p}(c + off, WC0_{p});
        memcpy(WC1_{p}, WC0_{p}, (long){VOL}*8);
        for(int x=0;x<{p};x++) wtr_{p}(WC1_{p} + (long)x*{PL});
        for(long t=0;t<m;t++){{
            wstep_{p}(WV_{p}, (t%2==0)? WC1_{p} : WC0_{p});
            if(t==0 && m>1) wconv_out_{p}(WV_{p}, out1 + off, 1);
        }}
        wconv_out_{p}(WV_{p}, outm + off, (int)(m%2));
        if(m==1) memcpy(out1 + off, outm + off, (long){L3}*16);
    }}
}}
""")
    return "\n".join(s)
