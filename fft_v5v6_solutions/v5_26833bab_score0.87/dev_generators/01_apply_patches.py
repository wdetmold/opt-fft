#!/usr/bin/env python3
"""Regenerate the FINAL graded /workdir/implementation.c of Taiga attempt
26833bab (fft3d-fixed-geometry-opt-20260821 v5, score 0.8729) by replaying the
session's in-place patch chain on top of 00_base_implementation.c (the full
rewrite the agent made at 01:27 UTC, log line ~2095).

Each step below corresponds to one bash action in the session log
(attempt_26833bab_score0.87.log); log line numbers are noted. Edits whose net
effect was zero (NTA<->T0 prefetch-hint flips, the BC next-slab prefetch block
added then deleted, the prefetch-distance sweep restored to 8, the batch-36/45
experiment reverted, the interleaved-passBC experiment replaced by the "plain"
version) are collapsed: the interleave experiment is applied as the single
region replacement that produced the final "plain" passBC, the others are
skipped. Every string edit asserts its target exists, so any transcription
drift fails loudly instead of silently diverging.

Usage: python3 01_apply_patches.py [outfile]
       (default outfile: ../implementation.c relative to this script)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '00_base_implementation.c')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'implementation.c')

src = open(BASE).read()

def rep(old, new, count=None):
    global src
    n = src.count(old)
    assert n > 0, "target not found:\n" + old[:200]
    if count is not None:
        assert n == count, f"expected {count} occurrences, found {n}: " + old[:120]
    src = src.replace(old, new)

# ---------------------------------------------------------------- P1
# log ~2726: LPADC padding of the fast axis; passes -> passA + fused passBC;
# conversions -> LP layout; bench hook CB -> passBC; drop bench_passC.
rep('''static void mk_bufs(long L){
    long SLAB = L*L + 8;
    size_t plane = (size_t)(L*SLAB + 16);''',
'''#define LPADC(L) (((L)+7)&~7)
static void mk_bufs(long L){
    long LP = LPADC(L);
    long SLAB = L*LP + 8;
    size_t plane = (size_t)(L*SLAB + 16);''')

start = src.index('/* ------------------------------------------------ passes ---------- */')
end = src.index('/* ------------------------------------------- conversions ----------- */')
new_passes = r'''/* ------------------------------------------------ passes ---------- */
#define DEF_PASSES(L) \
static void passA_##L(const double* restrict ir, const double* restrict ii, \
                      double* restrict orr, double* restrict oi){ \
    const long LP = LPADC(L); const long NN = (long)(L)*LP; const long SLAB = NN + 8; \
    for (long c0 = 0; c0 < NN; c0 += 8) \
        core##L(ir + c0, ii + c0, SLAB, 0xFF, 0, orr + c0, oi + c0, SLAB, 0xFF, 0, 0); \
} \
/* fused: for each slab x: axis-2 transform into slab scratch, then axis-3
   (transposed) + c + map into state (axes 2/3 swapped) */ \
static void passBC_##L(const double* restrict ir, const double* restrict ii, \
                       double* restrict orr, double* restrict oi, \
                       const double* restrict cre, const double* restrict cim, \
                       double* restrict t2r, double* restrict t2i){ \
    const long LP = LPADC(L); const long SLAB = (long)(L)*LP + 8; \
    vd KR[LPADC(L)], KI[LPADC(L)]; \
    for (long x = 0; x < L; x++){ \
        const double *br = ir + x*SLAB, *bi = ii + x*SLAB; \
        double *dr = orr + x*SLAB, *di = oi + x*SLAB; \
        const double *qr = cre + x*SLAB, *qi = cim + x*SLAB; \
        for (long z0 = 0; z0 < LP; z0 += 8) \
            core##L(br + z0, bi + z0, LP, 0xFF, 0, t2r + z0, t2i + z0, LP, 0xFF, 0, 0); \
        for (long y0 = 0; y0 < L; y0 += 8){ \
            int ny = (int)((L - y0) < 8 ? (L - y0) : 8); \
            __mmask8 my = (__mmask8)(ny == 8 ? 0xFF : ((1u<<ny)-1u)); \
            if (ny == 8){ \
                for (int jb = 0; jb < LP; jb += 8){ \
                    vd RR[8], SS[8]; \
                    for (int t = 0; t < 8; t++){ \
                        RR[t] = VLOADU(t2r + (y0+t)*(long)LP + jb); \
                        SS[t] = VLOADU(t2i + (y0+t)*(long)LP + jb); \
                    } \
                    transp8_from(RR, KR + jb); transp8_from(SS, KI + jb); \
                } \
            } else { \
                for (int jb = 0; jb < LP; jb += 8){ \
                    vd RR[8], SS[8]; \
                    for (int t = 0; t < 8; t++){ \
                        if (t < ny){ \
                            RR[t] = VLOADU(t2r + (y0+t)*(long)LP + jb); \
                            SS[t] = VLOADU(t2i + (y0+t)*(long)LP + jb); \
                        } else { RR[t] = _mm512_setzero_pd(); SS[t] = _mm512_setzero_pd(); } \
                    } \
                    transp8_from(RR, KR + jb); transp8_from(SS, KI + jb); \
                } \
            } \
            core##L((const double*)KR, (const double*)KI, 8, 0xFF, 0, \
                    dr + y0, di + y0, LP, my, qr + y0, qi + y0); \
        } \
    } \
}

DEF_PASSES(6)  DEF_PASSES(8)  DEF_PASSES(13) DEF_PASSES(17)
DEF_PASSES(23) DEF_PASSES(36) DEF_PASSES(45) DEF_PASSES(64)

'''
src = src[:start] + new_passes + src[end:]

start = src.index('/* ------------------------------------------- conversions ----------- */')
end = src.index('/* ------------------------------------------------ drivers ---------- */')
new_conv = r'''/* ------------------------------------------- conversions ----------- */
static void split_in(long L, const double *x, double *sre, double *sim){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    const __m512i idxA = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxB = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    for (long u = 0; u < L; u++){
        for (long y = 0; y < L; y++){
            const double *src = x + (u*L + y)*L*2;
            double *dr = sre + u*SLAB + y*LP, *di = sim + u*SLAB + y*LP;
            long w = 0;
            for (; w + 8 <= L; w += 8){
                vd q0 = _mm512_loadu_pd(src + 2*w);
                vd q1 = _mm512_loadu_pd(src + 2*w + 8);
                _mm512_storeu_pd(dr + w, _mm512_permutex2var_pd(q0, idxA, q1));
                _mm512_storeu_pd(di + w, _mm512_permutex2var_pd(q0, idxB, q1));
            }
            for (; w < L; w++){ dr[w] = src[2*w]; di[w] = src[2*w+1]; }
        }
    }
}

static void flip_copy(long L, const double *src, double *dst){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    for (long x = 0; x < L; x++){
        const double *s = src + x*SLAB; double *d = dst + x*SLAB;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++)
                d[z*LP + y] = s[y*LP + z];
    }
}

static void snap_nat(long L, const double *sre, const double *sim, double *out){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    const __m512i idxI0 = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxI1 = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    for (long u = 0; u < L; u++){
        for (long y = 0; y < L; y++){
            const double *dr = sre + u*SLAB + y*LP, *di = sim + u*SLAB + y*LP;
            double *dst = out + (u*L + y)*L*2;
            long w = 0;
            for (; w + 8 <= L; w += 8){
                vd a = _mm512_loadu_pd(dr + w);
                vd b = _mm512_loadu_pd(di + w);
                _mm512_storeu_pd(dst + 2*w,     _mm512_permutex2var_pd(a, idxI0, b));
                _mm512_storeu_pd(dst + 2*w + 8, _mm512_permutex2var_pd(a, idxI1, b));
            }
            for (; w < L; w++){ dst[2*w] = dr[w]; dst[2*w+1] = di[w]; }
        }
    }
}

static void snap_flip(long L, const double *sre, const double *sim, double *out){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    for (long x = 0; x < L; x++){
        const double *sr = sre + x*SLAB, *si = sim + x*SLAB;
        double *dst = out + x*L*L*2;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++){
                dst[(y*L+z)*2]   = sr[z*LP + y];
                dst[(y*L+z)*2+1] = si[z*LP + y];
            }
    }
}

'''
src = src[:start] + new_conv + src[end:]

rep('''#define CB(N) passB_##N(bf->sre,bf->sim,bf->t1re,bf->t1im)
    BENCH_SWITCH(CB)''',
'''#define CB(N) passBC_##N(bf->sre,bf->sim,bf->t1re,bf->t1im,bf->cre,bf->cim,bf->t2re,bf->t2im)
    BENCH_SWITCH(CB)''')
start = src.index('double bench_passC(')
end = src.index('void fill_state(')
src = src[:start] + src[end:]

# ---------------------------------------------------------------- P2
# log ~3047: batch-vectorized mode (SIMD lanes = 8 volumes) for L<=23.
rep("typedef struct { double *sre,*sim,*t1re,*t1im,*t2re,*t2im,*cre,*cim,*cfre,*cfim; } bufs_t;",
    "typedef struct { double *sre,*sim,*t1re,*t1im,*t2re,*t2im,*cre,*cim,*cfre,*cfim,*gre,*gim,*gcre,*gcim; } bufs_t;")
rep('''    for (int i = 0; i < 10; i++){
        *fields[i] = (double*)aligned_alloc(64, sizeof(double)*plane);
        memset(*fields[i], 0, sizeof(double)*plane);
    }
}''',
'''    for (int i = 0; i < 10; i++){
        *fields[i] = (double*)aligned_alloc(64, sizeof(double)*plane);
        memset(*fields[i], 0, sizeof(double)*plane);
    }
    if (L <= 23){
        size_t gsz = (size_t)L*L*L*8 + 64;
        double **gf[4] = {&b->gre,&b->gim,&b->gcre,&b->gcim};
        for (int i = 0; i < 4; i++){
            *gf[i] = (double*)aligned_alloc(64, sizeof(double)*gsz);
            memset(*gf[i], 0, sizeof(double)*gsz);
        }
    }
}''')

anchor = "/* ------------------------------------------------ drivers ---------- */"
batch_code = r'''/* =================== batch-vectorized mode (lanes = 8 volumes) ========== */
/* group layout: plane[e*8 + l], e in [0,L^3), lane l = volume index in group */
#define DEF_BATCH(L) \
static void bpassX_##L(double* restrict gr, double* restrict gi){ \
    const long NN = (long)L*L; \
    for (long u = 0; u < NN; u++) \
        core##L(gr + u*8, gi + u*8, NN*8, 0xFF, 0, gr + u*8, gi + u*8, NN*8, 0xFF, 0, 0); \
} \
static void bpassY_##L(double* restrict gr, double* restrict gi){ \
    const long NN = (long)L*L; \
    for (long x = 0; x < L; x++) \
        for (long z = 0; z < L; z++){ \
            long base = (x*NN + z)*8; \
            core##L(gr + base, gi + base, (long)L*8, 0xFF, 0, gr + base, gi + base, (long)L*8, 0xFF, 0, 0); \
        } \
} \
static void bpassZ_##L(double* restrict gr, double* restrict gi, \
                       const double* restrict cr, const double* restrict ci){ \
    const long NN = (long)L*L; \
    for (long u = 0; u < NN; u++){ \
        long base = u*(long)L*8; \
        core##L(gr + base, gi + base, 8, 0xFF, 0, gr + base, gi + base, 8, 0xFF, \
                cr + base, ci + base); \
    } \
}

DEF_BATCH(6)  DEF_BATCH(8)  DEF_BATCH(13) DEF_BATCH(17) DEF_BATCH(23)

/* interleave nv (<=8) volumes (complex interleaved, n=L^3 each) into group planes */
static void conv_in_batch(long L, const double *x, long stride2n, long nv, double *gr, double *gi){
    const long n = L*L*L;
    long e = 0;
    for (; e + 8 <= n; e += 8){
        vd RE[8], IM[8];
        const __m512i idxA = _mm512_set_epi64(14,12,10,8,6,4,2,0);
        const __m512i idxB = _mm512_set_epi64(15,13,11,9,7,5,3,1);
        for (int l = 0; l < 8; l++){
            if (l < nv){
                vd q0 = _mm512_loadu_pd(x + l*stride2n + 2*e);
                vd q1 = _mm512_loadu_pd(x + l*stride2n + 2*e + 8);
                RE[l] = _mm512_permutex2var_pd(q0, idxA, q1);
                IM[l] = _mm512_permutex2var_pd(q0, idxB, q1);
            } else { RE[l] = _mm512_setzero_pd(); IM[l] = _mm512_setzero_pd(); }
        }
        transp8_from(RE, (vd*)(gr + e*8));
        transp8_from(IM, (vd*)(gi + e*8));
    }
    for (; e < n; e++){
        for (int l = 0; l < 8; l++){
            gr[e*8+l] = (l < nv) ? x[l*stride2n + 2*e]   : 0.0;
            gi[e*8+l] = (l < nv) ? x[l*stride2n + 2*e+1] : 0.0;
        }
    }
}

/* extract nv volumes from group planes to interleaved complex out */
static void snap_batch(long L, const double *gr, const double *gi, double *out, long stride2n, long nv){
    const long n = L*L*L;
    const __m512i idxI0 = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxI1 = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    long e = 0;
    for (; e + 8 <= n; e += 8){
        vd RE[8], IM[8];
        transp8_from((const vd*)(gr + e*8), RE);
        transp8_from((const vd*)(gi + e*8), IM);
        for (int l = 0; l < nv; l++){
            _mm512_storeu_pd(out + l*stride2n + 2*e,     _mm512_permutex2var_pd(RE[l], idxI0, IM[l]));
            _mm512_storeu_pd(out + l*stride2n + 2*e + 8, _mm512_permutex2var_pd(RE[l], idxI1, IM[l]));
        }
    }
    for (; e < n; e++){
        for (int l = 0; l < nv; l++){
            out[l*stride2n + 2*e]   = gr[e*8+l];
            out[l*stride2n + 2*e+1] = gi[e*8+l];
        }
    }
}

''' + anchor
rep(anchor, batch_code, count=1)

rep('''void fill_state(long L, const double* x, const double* c){''',
'''double bench_batch(long L, long iters){
    bufs_t *bf = &BUFS[L];
    unsigned long long t0 = __rdtsc();
    switch(L){
        case 6: for(long i=0;i<iters;i++){ bpassX_6(bf->gre,bf->gim); bpassY_6(bf->gre,bf->gim); bpassZ_6(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 8: for(long i=0;i<iters;i++){ bpassX_8(bf->gre,bf->gim); bpassY_8(bf->gre,bf->gim); bpassZ_8(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 13: for(long i=0;i<iters;i++){ bpassX_13(bf->gre,bf->gim); bpassY_13(bf->gre,bf->gim); bpassZ_13(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 17: for(long i=0;i<iters;i++){ bpassX_17(bf->gre,bf->gim); bpassY_17(bf->gre,bf->gim); bpassZ_17(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 23: for(long i=0;i<iters;i++){ bpassX_23(bf->gre,bf->gim); bpassY_23(bf->gre,bf->gim); bpassZ_23(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
    }
    unsigned long long t1 = __rdtsc();
    return (double)(t1-t0)/iters/(double)(L*L*L*8);
}
void fill_batch(long L, const double* x, const double* c){
    conv_in_batch(L, x, 0, 1, BUFS[L].gre, BUFS[L].gim); /* broadcast same volume */
    conv_in_batch(L, c, 0, 1, BUFS[L].gcre, BUFS[L].gcim);
}
void fill_state(long L, const double* x, const double* c){''')

# ---------------------------------------------------------------- P3
# log ~3238: RUN_SLAB_BODY + DEF_RUN_BATCH/DEF_RUN_SLAB (subsumes the log's
# intermediate DEF_RUN batch edit at ~3157, which this slice overwrote).
start = src.index("#define DEF_RUN(L)")
end = src.index("/* ---- dev microbenchmarks")
new_run = r'''#define RUN_SLAB_BODY(L) \
    for (long b = b0; b < B; b++){ \
        split_in(L, x0 + b*2*n, bf->sre, bf->sim); \
        split_in(L, cc + b*2*n, bf->cre, bf->cim); \
        flip_copy(L, bf->cre, bf->cfre); flip_copy(L, bf->cim, bf->cfim); \
        int flip = 0; \
        for (long it = 0; it < m; it++){ \
            passA_##L(bf->sre, bf->sim, bf->t1re, bf->t1im); \
            if (!flip) passBC_##L(bf->t1re, bf->t1im, bf->sre, bf->sim, bf->cfre, bf->cfim, bf->t2re, bf->t2im); \
            else       passBC_##L(bf->t1re, bf->t1im, bf->sre, bf->sim, bf->cre,  bf->cim,  bf->t2re, bf->t2im); \
            flip ^= 1; \
            if (it == 0){ \
                if (flip) snap_flip(L, bf->sre, bf->sim, out1 + b*2*n); \
                else      snap_nat (L, bf->sre, bf->sim, out1 + b*2*n); \
            } \
        } \
        if (flip) snap_flip(L, bf->sre, bf->sim, outm + b*2*n); \
        else      snap_nat (L, bf->sre, bf->sim, outm + b*2*n); \
    }

#define DEF_RUN_BATCH(L) \
void run_##L(long B, long m, const double *x0, const double *cc, double *out1, double *outm){ \
    bufs_t *bf = &BUFS[L]; \
    const long n = (long)L*L*L; \
    long b0 = 0; \
    if (B >= 4){ \
        for (; b0 + 4 <= B; b0 += 8){ \
            long nv = (B - b0 >= 8) ? 8 : (B - b0); \
            conv_in_batch(L, x0 + b0*2*n, 2*n, nv, bf->gre, bf->gim); \
            conv_in_batch(L, cc + b0*2*n, 2*n, nv, bf->gcre, bf->gcim); \
            for (long it = 0; it < m; it++){ \
                bpassX_##L(bf->gre, bf->gim); \
                bpassY_##L(bf->gre, bf->gim); \
                bpassZ_##L(bf->gre, bf->gim, bf->gcre, bf->gcim); \
                if (it == 0) snap_batch(L, bf->gre, bf->gim, out1 + b0*2*n, 2*n, nv); \
            } \
            snap_batch(L, bf->gre, bf->gim, outm + b0*2*n, 2*n, nv); \
            if (nv < 8){ b0 = B; break; } \
        } \
    } \
    RUN_SLAB_BODY(L) \
}

#define DEF_RUN_SLAB(L) \
void run_##L(long B, long m, const double *x0, const double *cc, double *out1, double *outm){ \
    bufs_t *bf = &BUFS[L]; \
    const long n = (long)L*L*L; \
    long b0 = 0; \
    RUN_SLAB_BODY(L) \
}

DEF_RUN_BATCH(6)  DEF_RUN_BATCH(8)  DEF_RUN_BATCH(13) DEF_RUN_BATCH(17)
DEF_RUN_BATCH(23) DEF_RUN_SLAB(36) DEF_RUN_SLAB(45) DEF_RUN_SLAB(64)

'''
src = src[:start] + new_run + src[end:]

# ---------------------------------------------------------------- P4
# log ~3427: software prefetch of the next chunk's rows in passA. (The same
# patch's passBC B-part prefetch is subsumed by the "plain" passBC below.)
rep(r'''    for (long c0 = 0; c0 < NN; c0 += 8) \
        core##L(ir + c0, ii + c0, SLAB, 0xFF, 0, orr + c0, oi + c0, SLAB, 0xFF, 0, 0); \
''',
r'''    for (long c0 = 0; c0 < NN; c0 += 8){ \
        for (int r = 0; r < L; r++){ \
            _mm_prefetch((const char*)(ir + (long)r*SLAB + c0 + 8), _MM_HINT_T0); \
            _mm_prefetch((const char*)(ii + (long)r*SLAB + c0 + 8), _MM_HINT_T0); \
        } \
        core##L(ir + c0, ii + c0, SLAB, 0xFF, 0, orr + c0, oi + c0, SLAB, 0xFF, 0, 0); \
    } \
''', count=1)

# ---------------------------------------------------------------- P5
# log ~3737: prefetch of upcoming c rows inside OUT; in-place slab pipeline.
rep(r'''#define OUT(k, vr, vi) do{ \
    if (cq){ \
        vd cr_ = VLOADZ(sm, cq + (long)(k)*os); \
''',
r'''#define OUT(k, vr, vi) do{ \
    if (cq){ \
        _mm_prefetch((const char*)(cq  + (long)(k)*os + 4*os), _MM_HINT_T0); \
        vd cr_ = VLOADZ(sm, cq + (long)(k)*os); \
''')
rep(r'''        for (long it = 0; it < m; it++){ \
            passA_##L(bf->sre, bf->sim, bf->t1re, bf->t1im); \
            if (!flip) passBC_##L(bf->t1re, bf->t1im, bf->sre, bf->sim, bf->cfre, bf->cfim, bf->t2re, bf->t2im); \
            else       passBC_##L(bf->t1re, bf->t1im, bf->sre, bf->sim, bf->cre,  bf->cim,  bf->t2re, bf->t2im); \
''',
r'''        for (long it = 0; it < m; it++){ \
            passA_##L(bf->sre, bf->sim, bf->sre, bf->sim); \
            if (!flip) passBC_##L(bf->sre, bf->sim, bf->sre, bf->sim, bf->cfre, bf->cfim, bf->t2re, bf->t2im); \
            else       passBC_##L(bf->sre, bf->sim, bf->sre, bf->sim, bf->cre,  bf->cim,  bf->t2re, bf->t2im); \
''')
rep("#define CA(N) passA_##N(bf->sre,bf->sim,bf->t1re,bf->t1im)",
    "#define CA(N) passA_##N(bf->sre,bf->sim,bf->sre,bf->sim)")
rep("#define CB(N) passBC_##N(bf->sre,bf->sim,bf->t1re,bf->t1im,bf->cre,bf->cim,bf->t2re,bf->t2im)",
    "#define CB(N) passBC_##N(bf->sre,bf->sim,bf->sre,bf->sim,bf->cre,bf->cim,bf->t2re,bf->t2im)")

# ---------------------------------------------------------------- P6
# log ~3860: fuse d = 1 + m*y in the nonlinear map. (Same patch's TS scratch
# stride is subsumed by the "plain" passBC below.)
rep('''    vd s = VMUL(m, y);
    vd d = VADD(VSET1(1.0), s);''',
'''    vd d = VFMA(m, y, VSET1(1.0));''')

# ---------------------------------------------------------------- P7
# log ~3938: fuse batch Y+Z passes per x-slice (call sites + bench; the pass
# bodies themselves are wholesale-replaced again at P17).
rep(r'''                bpassX_##L(bf->gre, bf->gim); \
                bpassY_##L(bf->gre, bf->gim); \
                bpassZ_##L(bf->gre, bf->gim, bf->gcre, bf->gcim); \
''',
r'''                bpassX_##L(bf->gre, bf->gim); \
                bpassYZ_##L(bf->gre, bf->gim, bf->gcre, bf->gcim); \
''')
rep('''        case 6: for(long i=0;i<iters;i++){ bpassX_6(bf->gre,bf->gim); bpassY_6(bf->gre,bf->gim); bpassZ_6(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 8: for(long i=0;i<iters;i++){ bpassX_8(bf->gre,bf->gim); bpassY_8(bf->gre,bf->gim); bpassZ_8(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 13: for(long i=0;i<iters;i++){ bpassX_13(bf->gre,bf->gim); bpassY_13(bf->gre,bf->gim); bpassZ_13(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 17: for(long i=0;i<iters;i++){ bpassX_17(bf->gre,bf->gim); bpassY_17(bf->gre,bf->gim); bpassZ_17(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 23: for(long i=0;i<iters;i++){ bpassX_23(bf->gre,bf->gim); bpassY_23(bf->gre,bf->gim); bpassZ_23(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;''',
'''        case 6: for(long i=0;i<iters;i++){ bpassX_6(bf->gre,bf->gim); bpassYZ_6(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 8: for(long i=0;i<iters;i++){ bpassX_8(bf->gre,bf->gim); bpassYZ_8(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 13: for(long i=0;i<iters;i++){ bpassX_13(bf->gre,bf->gim); bpassYZ_13(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 17: for(long i=0;i<iters;i++){ bpassX_17(bf->gre,bf->gim); bpassYZ_17(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;
        case 23: for(long i=0;i<iters;i++){ bpassX_23(bf->gre,bf->gim); bpassYZ_23(bf->gre,bf->gim,bf->gcre,bf->gcim);} break;''')

# ---------------------------------------------------------------- P8
# log ~4187 (interleave experiment) resolved at ~4278/4366 by restoring the
# "plain" fused passBC (prefetched B-part, TS-strided scratch): the A/B test
# showed plain wins, and impl_plain.c was copied back over implementation.c.
start = src.index('static void passBC_##L(')
end = src.index('DEF_PASSES(6)')
plain = r'''static void passBC_##L(const double* restrict ir, const double* restrict ii, \
                       double* restrict orr, double* restrict oi, \
                       const double* restrict cre, const double* restrict cim, \
                       double* restrict t2r, double* restrict t2i){ \
    const long LP = LPADC(L); const long SLAB = (long)(L)*LP + 8; \
    const long TS = LP + 8; \
    vd KR[LPADC(L)], KI[LPADC(L)]; \
    for (long x = 0; x < L; x++){ \
        const double *br = ir + x*SLAB, *bi = ii + x*SLAB; \
        double *dr = orr + x*SLAB, *di = oi + x*SLAB; \
        const double *qr = cre + x*SLAB, *qi = cim + x*SLAB; \
        for (long z0 = 0; z0 < LP; z0 += 8){ \
            for (int y = 0; y < L; y++){ \
                _mm_prefetch((const char*)(br + (long)y*LP + z0 + 8), _MM_HINT_T0); \
                _mm_prefetch((const char*)(bi + (long)y*LP + z0 + 8), _MM_HINT_T0); \
            } \
            core##L(br + z0, bi + z0, LP, 0xFF, 0, t2r + z0, t2i + z0, TS, 0xFF, 0, 0); \
        } \
        for (long y0 = 0; y0 < L; y0 += 8){ \
            int ny = (int)((L - y0) < 8 ? (L - y0) : 8); \
            __mmask8 my = (__mmask8)(ny == 8 ? 0xFF : ((1u<<ny)-1u)); \
            if (ny == 8){ \
                for (int jb = 0; jb < LP; jb += 8){ \
                    vd RR[8], SS[8]; \
                    for (int t = 0; t < 8; t++){ \
                        RR[t] = VLOADU(t2r + (y0+t)*TS + jb); \
                        SS[t] = VLOADU(t2i + (y0+t)*TS + jb); \
                    } \
                    transp8_from(RR, KR + jb); transp8_from(SS, KI + jb); \
                } \
            } else { \
                for (int jb = 0; jb < LP; jb += 8){ \
                    vd RR[8], SS[8]; \
                    for (int t = 0; t < 8; t++){ \
                        if (t < ny){ \
                            RR[t] = VLOADU(t2r + (y0+t)*TS + jb); \
                            SS[t] = VLOADU(t2i + (y0+t)*TS + jb); \
                        } else { RR[t] = _mm512_setzero_pd(); SS[t] = _mm512_setzero_pd(); } \
                    } \
                    transp8_from(RR, KR + jb); transp8_from(SS, KI + jb); \
                } \
            } \
            core##L((const double*)KR, (const double*)KI, 8, 0xFF, 0, \
                    dr + y0, di + y0, LP, my, qr + y0, qi + y0); \
        } \
    } \
}

'''
src = src[:start] + plain + src[end:]

# ---------------------------------------------------------------- P10
# log ~4408: per-L partial-group thresholds for batch vs slab remainder.
rep(r'''#define DEF_RUN_BATCH(L) \
void run_##L(long B, long m, const double *x0, const double *cc, double *out1, double *outm){ \
    bufs_t *bf = &BUFS[L]; \
    const long n = (long)L*L*L; \
    long b0 = 0; \
    if (B >= 4){ \
        for (; b0 + 4 <= B; b0 += 8){ \
            long nv = (B - b0 >= 8) ? 8 : (B - b0); \
''',
r'''#define DEF_RUN_BATCH(L, THRESH) \
void run_##L(long B, long m, const double *x0, const double *cc, double *out1, double *outm){ \
    bufs_t *bf = &BUFS[L]; \
    const long n = (long)L*L*L; \
    long b0 = 0; \
    if (B >= THRESH){ \
        for (; b0 + THRESH <= B; b0 += 8){ \
            long nv = (B - b0 >= 8) ? 8 : (B - b0); \
''')
rep('''DEF_RUN_BATCH(6)  DEF_RUN_BATCH(8)  DEF_RUN_BATCH(13) DEF_RUN_BATCH(17)
DEF_RUN_BATCH(23) DEF_RUN_SLAB(36) DEF_RUN_SLAB(45) DEF_RUN_SLAB(64)''',
'''DEF_RUN_BATCH(6,5)  DEF_RUN_BATCH(8,7)  DEF_RUN_BATCH(13,6) DEF_RUN_BATCH(17,6)
DEF_RUN_BATCH(23,8) DEF_RUN_SLAB(36) DEF_RUN_SLAB(45) DEF_RUN_SLAB(64)''')

# ---------------------------------------------------------------- P11
# log ~4528 (E17): m==1 fast paths (batch-36/45 experiment of ~4475 reverted
# by the same patch, so only the memcpy edits survive).
rep(r'''            snap_batch(L, bf->gre, bf->gim, outm + b0*2*n, 2*n, nv); \
            if (nv < 8){ b0 = B; break; } \
''',
r'''            if (m == 1) memcpy(outm + b0*2*n, out1 + b0*2*n, sizeof(double)*2*n*nv); \
            else snap_batch(L, bf->gre, bf->gim, outm + b0*2*n, 2*n, nv); \
            if (nv < 8){ b0 = B; break; } \
''')
rep(r'''        if (flip) snap_flip(L, bf->sre, bf->sim, outm + b*2*n); \
        else      snap_nat (L, bf->sre, bf->sim, outm + b*2*n); \
    }''',
r'''        if (m == 1) memcpy(outm + b*2*n, out1 + b*2*n, sizeof(double)*2*n); \
        else if (flip) snap_flip(L, bf->sre, bf->sim, outm + b*2*n); \
        else      snap_nat (L, bf->sre, bf->sim, outm + b*2*n); \
    }''')

# ---------------------------------------------------------------- P13
# log ~4741: vectorize flip_copy and snap_flip with 8x8 block transposes.
rep('''static void flip_copy(long L, const double *src, double *dst){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    for (long x = 0; x < L; x++){
        const double *s = src + x*SLAB; double *d = dst + x*SLAB;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++)
                d[z*LP + y] = s[y*LP + z];
    }
}''',
'''static void flip_copy(long L, const double *src, double *dst){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    for (long x = 0; x < L; x++){
        const double *s = src + x*SLAB; double *d = dst + x*SLAB;
        for (long z0 = 0; z0 < L; z0 += 8){
            int zn = (int)((L - z0) < 8 ? (L - z0) : 8);
            __mmask8 mz = (__mmask8)(zn == 8 ? 0xFF : ((1u<<zn)-1u));
            for (long y0 = 0; y0 < L; y0 += 8){
                int yn = (int)((L - y0) < 8 ? (L - y0) : 8);
                __mmask8 my = (__mmask8)(yn == 8 ? 0xFF : ((1u<<yn)-1u));
                vd T[8];
                for (int t = 0; t < 8; t++)
                    T[t] = (t < yn) ? VLOADZ(mz, s + (y0+t)*LP + z0) : _mm512_setzero_pd();
                vd R[8];
                transp8_from(T, R);
                for (int t = 0; t < zn; t++)
                    VSTOREM(d + (z0+t)*LP + y0, my, R[t]);
            }
        }
    }
}''')
rep('''static void snap_flip(long L, const double *sre, const double *sim, double *out){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    for (long x = 0; x < L; x++){
        const double *sr = sre + x*SLAB, *si = sim + x*SLAB;
        double *dst = out + x*L*L*2;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++){
                dst[(y*L+z)*2]   = sr[z*LP + y];
                dst[(y*L+z)*2+1] = si[z*LP + y];
            }
    }
}''',
'''static void snap_flip(long L, const double *sre, const double *sim, double *out){
    const long LP = LPADC(L), SLAB = L*LP + 8;
    const __m512i idxI0 = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxI1 = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    for (long x = 0; x < L; x++){
        const double *sr = sre + x*SLAB, *si = sim + x*SLAB;
        double *dst = out + x*L*L*2;
        for (long y0 = 0; y0 < L; y0 += 8){
            int yn = (int)((L - y0) < 8 ? (L - y0) : 8);
            __mmask8 my = (__mmask8)(yn == 8 ? 0xFF : ((1u<<yn)-1u));
            for (long z0 = 0; z0 < L; z0 += 8){
                int zn = (int)((L - z0) < 8 ? (L - z0) : 8);
                unsigned zn2 = (unsigned)(2*zn);
                __mmask8 m0 = (__mmask8)(zn2 >= 8 ? 0xFF : ((1u<<zn2)-1u));
                __mmask8 m1 = (__mmask8)(zn2 <= 8 ? 0 : ((1u<<(zn2-8))-1u));
                vd TR[8], TI[8];
                for (int t = 0; t < 8; t++){
                    TR[t] = (t < zn) ? VLOADZ(my, sr + (z0+t)*LP + y0) : _mm512_setzero_pd();
                    TI[t] = (t < zn) ? VLOADZ(my, si + (z0+t)*LP + y0) : _mm512_setzero_pd();
                }
                vd RR[8], RI[8];
                transp8_from(TR, RR); transp8_from(TI, RI);
                for (int t = 0; t < yn; t++){
                    double *o = dst + ((y0+t)*L + z0)*2;
                    VSTOREM(o,     m0, _mm512_permutex2var_pd(RR[t], idxI0, RI[t]));
                    VSTOREM(o + 8, m1, _mm512_permutex2var_pd(RR[t], idxI1, RI[t]));
                }
            }
        }
    }
}''')

# ---------------------------------------------------------------- P14
# log ~4875: nonlinear map via hardware vsqrtpd (divider pipe) + rcp14/Newton.
rep('''static AI void map8(vd zr, vd zi, vd *xr, vd *xi){
    vd m = VFMA(zr, zr, VMUL(zi, zi));
    vd y = _mm512_rsqrt14_pd(m);
    vd hm = VMUL(m, VSET1(0.5));
    vd th = VSET1(1.5);
    y = VMUL(y, VFNMA(VMUL(hm, y), y, th));
    y = VMUL(y, VFNMA(VMUL(hm, y), y, th));
    vd d = VFMA(m, y, VSET1(1.0));
    vd r = _mm512_rcp14_pd(d);
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    __mmask8 k0 = _mm512_cmp_pd_mask(m, VSET1(2.2250738585072014e-308), _CMP_LT_OQ);
    r = _mm512_mask_blend_pd(k0, r, VSET1(1.0));
    *xr = VMUL(zr, r);
    *xi = VMUL(zi, r);
}''',
'''static AI void map8(vd zr, vd zi, vd *xr, vd *xi){
    vd m = VFMA(zr, zr, VMUL(zi, zi));
    vd s = _mm512_sqrt_pd(m);              /* divider pipe, hidden under FFT ALU */
    vd d = VADD(s, VSET1(1.0));
    vd r = _mm512_rcp14_pd(d);
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    *xr = VMUL(zr, r);
    *xi = VMUL(zi, r);
}''')

# ---------------------------------------------------------------- P15
# log ~4929: 2MB-aligned, hugepage-advised allocations.
rep('''#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>''',
'''#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>''')
rep('''static void mk_bufs(long L){
    long LP = LPADC(L);
    long SLAB = L*LP + 8;
    size_t plane = (size_t)(L*SLAB + 16);
    bufs_t *b = &BUFS[L];
    double **fields[10] = {&b->sre,&b->sim,&b->t1re,&b->t1im,&b->t2re,&b->t2im,&b->cre,&b->cim,&b->cfre,&b->cfim};
    for (int i = 0; i < 10; i++){
        *fields[i] = (double*)aligned_alloc(64, sizeof(double)*plane);
        memset(*fields[i], 0, sizeof(double)*plane);
    }
    if (L <= 23){
        size_t gsz = (size_t)L*L*L*8 + 64;
        double **gf[4] = {&b->gre,&b->gim,&b->gcre,&b->gcim};
        for (int i = 0; i < 4; i++){
            *gf[i] = (double*)aligned_alloc(64, sizeof(double)*gsz);
            memset(*gf[i], 0, sizeof(double)*gsz);
        }
    }
}''',
'''static double* big_alloc(size_t bytes){
    size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
    void *p = aligned_alloc(2u<<20, sz);
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);
    return (double*)p;
}
static void mk_bufs(long L){
    long LP = LPADC(L);
    long SLAB = L*LP + 8;
    size_t plane = (size_t)(L*SLAB + 16);
    bufs_t *b = &BUFS[L];
    double **fields[10] = {&b->sre,&b->sim,&b->t1re,&b->t1im,&b->t2re,&b->t2im,&b->cre,&b->cim,&b->cfre,&b->cfim};
    for (int i = 0; i < 10; i++)
        *fields[i] = big_alloc(sizeof(double)*plane);
    if (L <= 23){
        size_t gsz = (size_t)L*L*L*8 + 64;
        double **gf[4] = {&b->gre,&b->gim,&b->gcre,&b->gcim};
        for (int i = 0; i < 4; i++)
            *gf[i] = big_alloc(sizeof(double)*gsz);
    }
}''')

# ---------------------------------------------------------------- P16
# log ~4998: stagger cache-set offsets across the 2MB-aligned buffers.
rep('''static double* big_alloc(size_t bytes){
    size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
    void *p = aligned_alloc(2u<<20, sz);
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);
    return (double*)p;
}''',
'''static long stagger_ctr = 0;
static double* big_alloc(size_t bytes){
    long off = (stagger_ctr++ % 13) * 4224;   /* stagger cache-set offsets */
    size_t sz = (bytes + off + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
    char *p = (char*)aligned_alloc(2u<<20, sz);
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);
    return (double*)(p + off);
}''')

# ---------------------------------------------------------------- P17
# log ~5179: padded per-x-slice group stride GS = L*L*8 + 8; new stagger.
rep("long off = (stagger_ctr++ % 13) * 4224;   /* stagger cache-set offsets */",
    "long off = (stagger_ctr++ % 31) * 832;   /* stagger cache-set offsets */")
rep('''    if (L <= 23){
        size_t gsz = (size_t)L*L*L*8 + 64;''',
'''    if (L <= 23){
        size_t gsz = (size_t)L*((size_t)L*L*8 + 8) + 64;''')
start = src.index('#define DEF_BATCH(L)')
end = src.index('/* interleave nv')
new_batch = r'''#define DEF_BATCH(L) \
static void bpassX_##L(double* restrict gr, double* restrict gi){ \
    const long NN = (long)L*L; const long GS = NN*8 + 8; \
    for (long u = 0; u < NN; u++){ \
        for (int r = 0; r < L; r++){ \
            _mm_prefetch((const char*)(gr + (long)r*GS + u*8 + 8), _MM_HINT_T0); \
            _mm_prefetch((const char*)(gi + (long)r*GS + u*8 + 8), _MM_HINT_T0); \
        } \
        core##L(gr + u*8, gi + u*8, GS, 0xFF, 0, gr + u*8, gi + u*8, GS, 0xFF, 0, 0); \
    } \
} \
static void bpassYZ_##L(double* restrict gr, double* restrict gi, \
                        const double* restrict cr, const double* restrict ci){ \
    const long NN = (long)L*L; const long GS = NN*8 + 8; \
    for (long x = 0; x < L; x++){ \
        double *sr = gr + x*GS, *si = gi + x*GS; \
        const double *tr = cr + x*GS, *ti = ci + x*GS; \
        for (long z = 0; z < L; z++){ \
            long base = z*8; \
            _mm_prefetch((const char*)(tr + z*64), _MM_HINT_T0); \
            _mm_prefetch((const char*)(ti + z*64), _MM_HINT_T0); \
            core##L(sr + base, si + base, (long)L*8, 0xFF, 0, sr + base, si + base, (long)L*8, 0xFF, 0, 0); \
        } \
        for (long y = 0; y < L; y++){ \
            long base = y*(long)L*8; \
            core##L(sr + base, si + base, 8, 0xFF, 0, sr + base, si + base, 8, 0xFF, \
                    tr + base, ti + base); \
        } \
    } \
}

DEF_BATCH(6)  DEF_BATCH(8)  DEF_BATCH(13) DEF_BATCH(17) DEF_BATCH(23)

'''
src = src[:start] + new_batch + src[end:]

start = src.index('/* interleave nv')
end = src.index('/* extract nv')
new_conv2 = '''/* interleave nv (<=8) volumes (complex interleaved, n=L^3 each) into group planes */
static void conv_in_batch(long L, const double *x, long stride2n, long nv, double *gr, double *gi){
    const long NN = L*L, GS = NN*8 + 8;
    const __m512i idxA = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxB = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    for (long xs = 0; xs < L; xs++){
        double *dr = gr + xs*GS, *di = gi + xs*GS;
        const double *sb = x + xs*NN*2;
        long w = 0;
        for (; w + 8 <= NN; w += 8){
            vd RE[8], IM[8];
            for (int l = 0; l < 8; l++){
                if (l < nv){
                    vd q0 = _mm512_loadu_pd(sb + l*stride2n + 2*w);
                    vd q1 = _mm512_loadu_pd(sb + l*stride2n + 2*w + 8);
                    RE[l] = _mm512_permutex2var_pd(q0, idxA, q1);
                    IM[l] = _mm512_permutex2var_pd(q0, idxB, q1);
                } else { RE[l] = _mm512_setzero_pd(); IM[l] = _mm512_setzero_pd(); }
            }
            transp8_from(RE, (vd*)(dr + w*8));
            transp8_from(IM, (vd*)(di + w*8));
        }
        for (long t = 0, r = NN & 7; t < r; t++, w++){
            for (int l = 0; l < 8; l++){
                dr[w*8+l] = (l < nv) ? sb[l*stride2n + 2*w]   : 0.0;
                di[w*8+l] = (l < nv) ? sb[l*stride2n + 2*w+1] : 0.0;
            }
        }
    }
}

'''
src = src[:start] + new_conv2 + src[end:]

start = src.index('/* extract nv')
end = src.index('/* ------------------------------------------------ drivers ---------- */')
new_snap2 = '''/* extract nv volumes from group planes to interleaved complex out */
static void snap_batch(long L, const double *gr, const double *gi, double *out, long stride2n, long nv){
    const long NN = L*L, GS = NN*8 + 8;
    const __m512i idxI0 = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxI1 = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    for (long xs = 0; xs < L; xs++){
        const double *dr = gr + xs*GS, *di = gi + xs*GS;
        double *ob = out + xs*NN*2;
        long w = 0;
        for (; w + 8 <= NN; w += 8){
            vd RE[8], IM[8];
            transp8_from((const vd*)(dr + w*8), RE);
            transp8_from((const vd*)(di + w*8), IM);
            for (int l = 0; l < nv; l++){
                _mm512_storeu_pd(ob + l*stride2n + 2*w,     _mm512_permutex2var_pd(RE[l], idxI0, IM[l]));
                _mm512_storeu_pd(ob + l*stride2n + 2*w + 8, _mm512_permutex2var_pd(RE[l], idxI1, IM[l]));
            }
        }
        for (long t = 0, r = NN & 7; t < r; t++, w++){
            for (int l = 0; l < nv; l++){
                ob[l*stride2n + 2*w]   = dr[w*8+l];
                ob[l*stride2n + 2*w+1] = di[w*8+l];
            }
        }
    }
}

'''
src = src[:start] + new_snap2 + src[end:]

# ---------------------------------------------------------------- P18
# log ~5500: allocation fallback if the 2MB-aligned request fails.
rep('''    char *p = (char*)aligned_alloc(2u<<20, sz);
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);''',
'''    char *p = (char*)aligned_alloc(2u<<20, sz);
    if (!p) p = (char*)aligned_alloc(4096, sz);
#ifdef MADV_HUGEPAGE
    if (p) madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);''')

# ---------------------------------------------------------------- P19
# log ~5614: remove all restrict qualifiers (UB on the in-place aliased paths).
assert '* restrict ' in src
src = src.replace('* restrict ', '* ')
assert 'restrict' not in src

# ---------------------------------------------------------------- P20
# log ~5916: audit-friendly header.
rep('// Batched 3D complex-to-complex FFT + nonlinear iteration; AVX-512; 1 thread.',
'''// Iterated batched 3D complex FFTs for L in {6,8,13,17,23,36,45,64}.
// All transform arithmetic is implemented from scratch here (no FFT library
// of any kind): PFA/Cooley-Tukey codelets for composite sizes, symmetric
// half-matrix kernels for the primes 13/17/23, AVX-512 intrinsics throughout,
// strictly single-threaded.''')

open(OUT, 'w').write(src)
print(f"wrote {os.path.abspath(OUT)}: {len(src)} bytes, {src.count(chr(10))} lines")
