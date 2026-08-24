import numpy as np
from genlib import hexd
from genb import Emit, DFTS, cmul_const, W
from genbdrv import col_dft, col_dft_pipe  # baked-stride column machinery


def y_stages_64(PLX):
    """y-DFT over rows of one a-plane, row-streamed.
       stage1 (in-place): for y2 in 0..7: vertical DFT8 across rows {y2+8t}, + twiddle W64^{k1*y2},
           written back to rows {8k1+y2}... NOTE: write row set == read row set (same residue class)
           BUT positions permuted: row (y2+8t) gets T[k1=t]? we store T[k1] to row y2+8*k1: same set, ok in-place per slot.
       stage2 (to scratch SP): for k1: vertical DFT8 across rows {8k1+y2: y2}, output X[k1+8k2] -> SP row k1+8k2.
       Then plane <- SP (the z-pass of NEXT step reads rows; we instead make stage2 write back... simpler: copy via the caller using SP as the new plane? We write stage2 straight back to the plane rows k1+8k2 via a barrier: collision! -> use SP then memcpy-free: pass2 reads from SP? No: keep SP static and have stage2 of y be the LAST op of pass1, writing to the plane is required...
       Resolution: stage2 writes SP; then a copy sweep SP->plane (contiguous, streaming)."""
    from genb import Emit, DFTS, cmul_const, W
    out = []
    # stage1: loop over zz slots; unrolled y2 inside? emit fn(P): for y2 blocks: for zz loop
    e = Emit('g')
    body = []
    for y2 in range(8):
        u = Emit(f'h{y2}_')
        xs = []
        for t in range(8):
            off = (y2 + 8*t)*64*16
            u.raw(f"_mm_prefetch((const char*)(P + {off} + zz*16 + 128), _MM_HINT_T0);")
            xs.append((u.v(f"_mm512_load_pd(P + {off} + zz*16)"), u.v(f"_mm512_load_pd(P + {off} + zz*16 + 8)")))
        ys = DFTS[8](u, xs)
        for k1 in range(8):
            y = ys[k1]
            if (k1*y2) % 64:
                y = cmul_const(u, y, *W(64, k1*y2))
            u.raw(f"_mm512_store_pd(P + {(y2 + 8*k1)*64*16} + zz*16, {y[0]});")
            u.raw(f"_mm512_store_pd(P + {(y2 + 8*k1)*64*16} + zz*16 + 8, {y[1]});")
        decls = "\n        ".join(u.const_decls())
        body.append(f"    {{\n        {decls}\n        for(long zz=0; zz<64; zz++){{\n{u.code(indent='            ')}\n        }}\n    }}")
    out.append("static void __attribute__((noinline)) ys1_64(double* restrict P){\n" + "\n".join(body) + "\n}\n")
    body = []
    for k1 in range(8):
        u = Emit(f'i{k1}_')
        xs = []
        for y2 in range(8):
            off = (y2 + 8*k1)*64*16
            xs.append((u.v(f"_mm512_load_pd(P + {off} + zz*16)"), u.v(f"_mm512_load_pd(P + {off} + zz*16 + 8)")))
        ys = DFTS[8](u, xs)
        for k2 in range(8):
            off = (k1 + 8*k2)*64*16
            u.raw(f"_mm512_store_pd(SP + {off} + zz*16, {ys[k2][0]});")
            u.raw(f"_mm512_store_pd(SP + {off} + zz*16 + 8, {ys[k2][1]});")
        decls = "\n        ".join(u.const_decls())
        body.append(f"    {{\n        {decls}\n        for(long zz=0; zz<64; zz++){{\n{u.code(indent='            ')}\n        }}\n    }}")
    out.append("static void __attribute__((noinline)) ys2_64(const double* restrict P, double* restrict SP){\n" + "\n".join(body) + "\n}\n")
    return "\n".join(out)

def gen_64x():
    L = 64
    PLX = 64*64*16 + 8          # a-plane stride in doubles (+pad vs 4K: 65536+64B -> 65600B % 4096 = 64)
    VOL = 8*PLX
    L3 = L*L*L
    s = []
    # z columns: contiguous vec-pts (stride 16); y columns: stride 64*16
    s.append(col_dft(64, "xz64", False, 16))
    s.append(col_dft_pipe(64, "qz64", False, 16))
    s.append(y_stages_64(PLX))
    # x-octet codelet: 8 vec-pts at stride PLX; twiddle table TWX[ka][lane]
    e = Emit('x')
    # twiddle vector table: for ka=1..7: W64^{ka*l} for l=0..7 (re,im vectors)
    rows = []
    for ka in range(8):
        re = [W(64, ka*l)[0] for l in range(8)]
        im = [W(64, ka*l)[1] for l in range(8)]
        rows.append("{" + ", ".join(hexd(v) for v in re) + "}")
        rows.append("{" + ", ".join(hexd(v) for v in im) + "}")
    twtab = "static const double TWX64[16][8] ALIGN64 = {\n" + ",\n".join(rows) + "};\n"
    # emit function body
    body = Emit('x')
    xs = []
    for a in range(8):
        body.raw(f"_mm_prefetch((const char*)(px + {a}*{PLX} + 64), _MM_HINT_T0);")
        body.raw(f"_mm_prefetch((const char*)(px + {a}*{PLX} + 72), _MM_HINT_T0);")
        body.raw(f"_mm_prefetch((const char*)(cb + {a}*{PLX} + 64), _MM_HINT_T0);")
        body.raw(f"_mm_prefetch((const char*)(cb + {a}*{PLX} + 72), _MM_HINT_T0);")
        xs.append((body.v(f"_mm512_load_pd(px + {a}*{PLX})"), body.v(f"_mm512_load_pd(px + {a}*{PLX} + 8)")))
    ys = DFTS[8](body, xs)     # DFT8 over a (vertical)
    # twiddle: y[ka] *= W64^{ka*l} (vector constants)
    tw = []
    for ka in range(8):
        if ka == 0:
            tw.append(ys[0])
        else:
            cr = body.v(f"_mm512_load_pd(TWX64[{2*ka}])")
            ci = body.v(f"_mm512_load_pd(TWX64[{2*ka+1}])")
            t0 = body.v(f"_mm512_mul_pd({cr},{ys[ka][0]})")
            t0 = body.v(f"_mm512_fnmadd_pd({ci},{ys[ka][1]},{t0})")
            t1 = body.v(f"_mm512_mul_pd({cr},{ys[ka][1]})")
            t1 = body.v(f"_mm512_fmadd_pd({ci},{ys[ka][0]},{t1})")
            tw.append((t0,t1))
    # transpose 8x8 re and im separately: input regs tw[ka] lanes=l -> output regs n[l] lanes=ka
    res = [t[0] for t in tw]; ims = [t[1] for t in tw]
    body.raw("__m512d " + ",".join(f"nr{l}" for l in range(8)) + ";")
    body.raw("__m512d " + ",".join(f"ni{l}" for l in range(8)) + ";")
    body.raw(f"TR8({','.join(res)},{','.join(f'nr{l}' for l in range(8))});")
    body.raw(f"TR8({','.join(ims)},{','.join(f'ni{l}' for l in range(8))});")
    xs2 = [(f"nr{l}", f"ni{l}") for l in range(8)]
    ys2 = DFTS[8](body, xs2)   # DFT8 over l (vertical after transpose)
    for kl in range(8):
        body.raw(f"MAPST({ys2[kl][0]}, {ys2[kl][1]}, dst, {kl}*{PLX}, cb, {kl}*{PLX});")
    decls = "\n    ".join(body.const_decls())
    s.append(twtab + f"""
static void __attribute__((noinline)) ex64(const double* restrict px, double* restrict dst, const double* restrict cb){{
    {decls}
{body.code()}
}}
""")
    # conv helpers: numpy C-order <-> VX layout [a][y][z][16] lanes x=8a+l
    s.append(f"""
static void convx_in_64p(const double* restrict src, double* restrict V){{
    // pencil-major: V[((y*64+z)*8 + a)*16]
    for(int a=0;a<8;a++)
        for(int y=0;y<64;y++){{
            double* d = V + ((long)y*64*8 + a)*16;
            for(int zb=0; zb<8; zb++){{
                __m512d re[8], im[8], tre[8], tim[8];
                for(int l=0;l<8;l++){{
                    const double* sp = src + (((long)(8*a+l)*64 + y)*64 + 8*zb)*2;
                    __m512d q0 = _mm512_loadu_pd(sp);
                    __m512d q1 = _mm512_loadu_pd(sp+8);
                    re[l] = _mm512_permutex2var_pd(q0, IDX_RE, q1);
                    im[l] = _mm512_permutex2var_pd(q0, IDX_IM, q1);
                }}
                TR8(re[0],re[1],re[2],re[3],re[4],re[5],re[6],re[7],tre[0],tre[1],tre[2],tre[3],tre[4],tre[5],tre[6],tre[7]);
                TR8(im[0],im[1],im[2],im[3],im[4],im[5],im[6],im[7],tim[0],tim[1],tim[2],tim[3],tim[4],tim[5],tim[6],tim[7]);
                for(int t=0;t<8;t++){{
                    _mm512_store_pd(d + (long)(8*zb+t)*128, tre[t]);
                    _mm512_store_pd(d + (long)(8*zb+t)*128 + 8, tim[t]);
                }}
            }}
        }}
}}
static void convx_in_64(const double* restrict src, double* restrict V){{
    for(int a=0;a<8;a++)
        for(int y=0;y<64;y++){{
            double* d = V + (long)a*{PLX} + (long)y*64*16;
            for(int zb=0; zb<8; zb++){{
                __m512d re[8], im[8], tre[8], tim[8];
                for(int l=0;l<8;l++){{
                    const double* sp = src + (((long)(8*a+l)*64 + y)*64 + 8*zb)*2;
                    __m512d q0 = _mm512_loadu_pd(sp);
                    __m512d q1 = _mm512_loadu_pd(sp+8);
                    re[l] = _mm512_permutex2var_pd(q0, IDX_RE, q1);
                    im[l] = _mm512_permutex2var_pd(q0, IDX_IM, q1);
                }}
                TR8(re[0],re[1],re[2],re[3],re[4],re[5],re[6],re[7],tre[0],tre[1],tre[2],tre[3],tre[4],tre[5],tre[6],tre[7]);
                TR8(im[0],im[1],im[2],im[3],im[4],im[5],im[6],im[7],tim[0],tim[1],tim[2],tim[3],tim[4],tim[5],tim[6],tim[7]);
                for(int t=0;t<8;t++){{
                    _mm512_store_pd(d + (long)(8*zb+t)*16, tre[t]);
                    _mm512_store_pd(d + (long)(8*zb+t)*16 + 8, tim[t]);
                }}
            }}
        }}
}}
static void convx_out_64(const double* restrict V, double* restrict dst){{
    for(int a=0;a<8;a++)
        for(int y=0;y<64;y++){{
            const double* p = V + (long)a*{PLX} + (long)y*64*16;
            for(int zb=0; zb<8; zb++){{
                __m512d re[8], im[8], tre[8], tim[8];
                for(int t=0;t<8;t++){{
                    re[t] = _mm512_load_pd(p + (long)(8*zb+t)*16);
                    im[t] = _mm512_load_pd(p + (long)(8*zb+t)*16 + 8);
                }}
                TR8(re[0],re[1],re[2],re[3],re[4],re[5],re[6],re[7],tre[0],tre[1],tre[2],tre[3],tre[4],tre[5],tre[6],tre[7]);
                TR8(im[0],im[1],im[2],im[3],im[4],im[5],im[6],im[7],tim[0],tim[1],tim[2],tim[3],tim[4],tim[5],tim[6],tim[7]);
                for(int l=0;l<8;l++){{
                    double* dp = dst + (((long)(8*a+l)*64 + y)*64 + 8*zb)*2;
                    _mm512_storeu_pd(dp,   _mm512_permutex2var_pd(tre[l], IDX_ILA, tim[l]));
                    _mm512_storeu_pd(dp+8, _mm512_permutex2var_pd(tre[l], IDX_ILB, tim[l]));
                }}
            }}
        }}
}}
static void xstep_64(double* restrict V, double* restrict V2, const double* restrict CP){{
    for(int a=0;a<8;a++){{
        double* P = V + (long)a*{PLX};
        for(int i2=0;i2<64;i2++) xz64(P + (long)i2*1024);
        ys1_64(P);
        ys2_64(P, V2 + (long)a*{PLX});
    }}
    for(long e=0; e<64*64; e++)
        ex64(V2 + e*16, V + e*16, CP + e*16);
}}
static double* XV_64 = 0; static double* XV2_64 = 0; static double* XC_64 = 0;
void xrun_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!XV_64){{ XV_64 = alloc_arena({VOL}*8 + 4096); XV2_64 = alloc_arena({VOL}*8 + 4096) + 128; XC_64 = alloc_arena({VOL}*8 + 4096) + 64; }}
    for(long b=0;b<B;b++){{
        long off = b*(long){L3}*2;
        convx_in_64(x0 + off, XV_64);
        convx_in_64(c + off, XC_64);
        for(long t=0;t<m;t++){{
            xstep_64(XV_64, XV2_64, XC_64);
            if(t==0 && m>1) convx_out_64(XV_64, out1 + off);
        }}
        convx_out_64(XV_64, outm + off);
        if(m==1) memcpy(out1 + off, outm + off, (long){L3}*16);
    }}
}}
""")
    return "\n".join(s)
