import numpy as np
from ir import *
from emit import emit_codelet
from prime_gen import gen_prime_codelet

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
PERVOL = (13, 17, 23, 36, 45, 64)   # sizes with per-volume mode
BATCH  = (6, 8, 13, 17, 23, 36, 45) # sizes with batched mode

HEADER = r'''// Auto-generated: specialized iterated batched 3D complex FFT for L in {6,8,13,17,23,36,45,64}.
//
// All transform arithmetic is original, generated code (no FFT library code of
// any kind). Algorithms used, all specialized at generation time with baked
// double-precision twiddle constants (computed in long double):
//   - composite sizes: prime-factor (Good-Thomas) and Cooley-Tukey splits
//     (6=2x3, 8=2*4, 36=9x4 PFA, 45=9x5 PFA, 64=8x8 CT with fused twiddles)
//   - prime sizes 13/17/23: direct symmetric-folded DFT (cos/sin half-matrix
//     dot products), register-tiled over output pairs
// Two execution strategies:
//   - "batched" lanes-of-8-volumes layout for small/medium L (SIMD across
//     volumes, zero shuffles; zero-padded tail lanes)
//   - "per-volume" slice pipeline for large L (64: digit-transposed (pi)
//     storage so the slice transpose fuses into the radix-8x8 codelet;
//     36/45: unit-dim padded to a multiple of 8, transposes via 8x8 tiles)
// The elementwise map z/(1+|z|) uses rsqrt14+Newton (2 steps) for |z| and a
// balanced divide/Newton-reciprocal mix, accurate to ~2 ulp.
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static void* big_alloc(size_t bytes){
    size_t HP = (size_t)2<<20;
    size_t sz = (bytes + HP - 1) & ~(HP-1);
    void* p = mmap(0, sz + HP, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { void* q = 0; if (posix_memalign(&q, 64, bytes)) return 0; return q; }
    uintptr_t a = ((uintptr_t)p + HP - 1) & ~(uintptr_t)(HP-1);
    size_t head = a - (uintptr_t)p;
    if (head) munmap(p, head);
    size_t tail = HP - head;
    if (tail) munmap((void*)(a + sz), tail);
    madvise((void*)a, sz, MADV_HUGEPAGE);
    memset((void*)a, 0, sz);   // touch to materialize
    return (void*)a;
}
// skewed allocation: avoid identical cache-set phase across arrays
static double* skew_alloc(size_t doubles, int idx){
    double* p = (double*)big_alloc((doubles + 1024)*sizeof(double));
    return p + (size_t)(idx & 7)*72;
}

#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    __m512d _t0 = _mm512_unpacklo_pd(r0, r1); \
    __m512d _t1 = _mm512_unpackhi_pd(r0, r1); \
    __m512d _t2 = _mm512_unpacklo_pd(r2, r3); \
    __m512d _t3 = _mm512_unpackhi_pd(r2, r3); \
    __m512d _t4 = _mm512_unpacklo_pd(r4, r5); \
    __m512d _t5 = _mm512_unpackhi_pd(r4, r5); \
    __m512d _t6 = _mm512_unpacklo_pd(r6, r7); \
    __m512d _t7 = _mm512_unpackhi_pd(r6, r7); \
    __m512d _u0 = _mm512_shuffle_f64x2(_t0, _t2, 0x88); \
    __m512d _u1 = _mm512_shuffle_f64x2(_t1, _t3, 0x88); \
    __m512d _u2 = _mm512_shuffle_f64x2(_t0, _t2, 0xdd); \
    __m512d _u3 = _mm512_shuffle_f64x2(_t1, _t3, 0xdd); \
    __m512d _u4 = _mm512_shuffle_f64x2(_t4, _t6, 0x88); \
    __m512d _u5 = _mm512_shuffle_f64x2(_t5, _t7, 0x88); \
    __m512d _u6 = _mm512_shuffle_f64x2(_t4, _t6, 0xdd); \
    __m512d _u7 = _mm512_shuffle_f64x2(_t5, _t7, 0xdd); \
    r0 = _mm512_shuffle_f64x2(_u0, _u4, 0x88); \
    r1 = _mm512_shuffle_f64x2(_u1, _u5, 0x88); \
    r2 = _mm512_shuffle_f64x2(_u2, _u6, 0x88); \
    r3 = _mm512_shuffle_f64x2(_u3, _u7, 0x88); \
    r4 = _mm512_shuffle_f64x2(_u0, _u4, 0xdd); \
    r5 = _mm512_shuffle_f64x2(_u1, _u5, 0xdd); \
    r6 = _mm512_shuffle_f64x2(_u2, _u6, 0xdd); \
    r7 = _mm512_shuffle_f64x2(_u3, _u7, 0xdd); \
} while(0)

static const long long IDX_RE[8] __attribute__((aligned(64))) = {0,2,4,6,8,10,12,14};
static const long long IDX_IM[8] __attribute__((aligned(64))) = {1,3,5,7,9,11,13,15};
static const long long IDX_LO[8] __attribute__((aligned(64))) = {0,8,1,9,2,10,3,11};
static const long long IDX_HI[8] __attribute__((aligned(64))) = {4,12,5,13,6,14,7,15};

#define DEINT(p, re, im) do { \
    __m512d _a = _mm512_loadu_pd((p)); \
    __m512d _b = _mm512_loadu_pd((p)+8); \
    re = _mm512_permutex2var_pd(_a, _mm512_load_si512((const void*)IDX_RE), _b); \
    im = _mm512_permutex2var_pd(_a, _mm512_load_si512((const void*)IDX_IM), _b); \
} while(0)
#define INTST(p, re, im) do { \
    _mm512_storeu_pd((p),   _mm512_permutex2var_pd(re, _mm512_load_si512((const void*)IDX_LO), im)); \
    _mm512_storeu_pd((p)+8, _mm512_permutex2var_pd(re, _mm512_load_si512((const void*)IDX_HI), im)); \
} while(0)
#define INTST_NT(p, re, im) do { \
    _mm512_stream_pd((p),   _mm512_permutex2var_pd(re, _mm512_load_si512((const void*)IDX_LO), im)); \
    _mm512_stream_pd((p)+8, _mm512_permutex2var_pd(re, _mm512_load_si512((const void*)IDX_HI), im)); \
} while(0)

// laneize: cnt complex elements starting at src (per volume, volumes vstride2 doubles apart)
static void laneize(const double* src, long cnt, long vstride2, long nl, double* DR, double* DI){
    long i = 0;
    if (nl == 8){
        for (; i + 8 <= cnt; i += 8){
            __m512d R[8], I[8];
            for (int v = 0; v < 8; ++v) DEINT(src + (long)v*vstride2 + i*2, R[v], I[v]);
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            TR8(I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]);
            for (int k = 0; k < 8; ++k){
                _mm512_store_pd(DR + (i+k)*8, R[k]);
                _mm512_store_pd(DI + (i+k)*8, I[k]);
            }
        }
    }
    for (; i < cnt; ++i)
        for (long v = 0; v < nl; ++v){
            DR[i*8+v] = src[v*vstride2 + i*2];
            DI[i*8+v] = src[v*vstride2 + i*2 + 1];
        }
    if (nl < 8){
        for (long ii = 0; ii < cnt; ++ii)
            for (long v = nl; v < 8; ++v){ DR[ii*8+v] = 0.0; DI[ii*8+v] = 0.0; }
    }
}

static void delaneize(const double* SR, const double* SI, long cnt, long vstride2, long nl, double* dst){
    long i = 0;
    if (nl == 8){
        int nt = (((uintptr_t)dst & 63) == 0) && ((vstride2 & 7) == 0);
        if (nt)
        for (; i + 8 <= cnt; i += 8){
            __m512d R[8], I[8];
            for (int k = 0; k < 8; ++k){
                R[k] = _mm512_load_pd(SR + (i+k)*8);
                I[k] = _mm512_load_pd(SI + (i+k)*8);
            }
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            TR8(I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]);
            for (int v = 0; v < 8; ++v) INTST_NT(dst + (long)v*vstride2 + i*2, R[v], I[v]);
        }
        else
        for (; i + 8 <= cnt; i += 8){
            __m512d R[8], I[8];
            for (int k = 0; k < 8; ++k){
                R[k] = _mm512_load_pd(SR + (i+k)*8);
                I[k] = _mm512_load_pd(SI + (i+k)*8);
            }
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            TR8(I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]);
            for (int v = 0; v < 8; ++v) INTST(dst + (long)v*vstride2 + i*2, R[v], I[v]);
        }
    }
    for (; i < cnt; ++i)
        for (long v = 0; v < nl; ++v){
            dst[v*vstride2 + i*2]     = SR[i*8+v];
            dst[v*vstride2 + i*2 + 1] = SI[i*8+v];
        }
}

static void split_n(const double* src, long n, double* dr, double* di){
    long i = 0;
    for (; i + 8 <= n; i += 8){
        __m512d re, im;
        DEINT(src + 2*i, re, im);
        _mm512_storeu_pd(dr + i, re);
        _mm512_storeu_pd(di + i, im);
    }
    for (; i < n; ++i){ dr[i] = src[2*i]; di[i] = src[2*i+1]; }
}

static void merge_n(const double* sr, const double* si, long n, double* dst){
    long i = 0;
    for (; i + 8 <= n; i += 8){
        __m512d re = _mm512_loadu_pd(sr + i), im = _mm512_loadu_pd(si + i);
        INTST(dst + 2*i, re, im);
    }
    for (; i < n; ++i){ dst[2*i] = sr[i]; dst[2*i+1] = si[i]; }
}

// transpose one LxL plane of doubles: d[b*L + a] = s[a*L + b]
static void transpose_plane(const double* sp, double* dp, long L){
    for (long a0 = 0; a0 < L; a0 += 8){
        long na = (L - a0 < 8) ? (L - a0) : 8;
        __mmask8 amask = (__mmask8)((1u << na) - 1u);
        for (long b0 = 0; b0 < L; b0 += 8){
            long nb = (L - b0 < 8) ? (L - b0) : 8;
            __mmask8 bmask = (__mmask8)((1u << nb) - 1u);
            __m512d R[8];
            for (long t = 0; t < na; ++t)
                R[t] = _mm512_maskz_loadu_pd(bmask, sp + (a0+t)*L + b0);
            for (long t = na; t < 8; ++t) R[t] = _mm512_setzero_pd();
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            for (long t = 0; t < nb; ++t)
                _mm512_mask_storeu_pd(dp + (b0+t)*L + a0, amask, R[t]);
        }
    }
}

// move row r -> pi(r) and apply digit-transpose (pi) to row contents; self-inverse composition. s != d.
static void unperm_slice64(const double* s, double* d){
    for (long r = 0; r < 64; ++r){
        const double* sp = s + r*64;
        __m512d R0 = _mm512_loadu_pd(sp),    R1 = _mm512_loadu_pd(sp+8),
                R2 = _mm512_loadu_pd(sp+16), R3 = _mm512_loadu_pd(sp+24),
                R4 = _mm512_loadu_pd(sp+32), R5 = _mm512_loadu_pd(sp+40),
                R6 = _mm512_loadu_pd(sp+48), R7 = _mm512_loadu_pd(sp+56);
        TR8(R0,R1,R2,R3,R4,R5,R6,R7);
        double* dp = d + ((r & 7)*8 + (r >> 3))*64;
        _mm512_storeu_pd(dp, R0);    _mm512_storeu_pd(dp+8, R1);
        _mm512_storeu_pd(dp+16, R2); _mm512_storeu_pd(dp+24, R3);
        _mm512_storeu_pd(dp+32, R4); _mm512_storeu_pd(dp+40, R5);
        _mm512_storeu_pd(dp+48, R6); _mm512_storeu_pd(dp+56, R7);
    }
}

// fused: deinterleave one complex slice (64x64) and apply unperm (row r -> pi(r), contents TR8)
static void split_unperm64(const double* src, double* dR, double* dI){
    for (long r = 0; r < 64; ++r){
        const double* sp = src + r*128;
        __m512d R0,R1,R2,R3,R4,R5,R6,R7, I0,I1,I2,I3,I4,I5,I6,I7;
        DEINT(sp,     R0, I0); DEINT(sp+16,  R1, I1);
        DEINT(sp+32,  R2, I2); DEINT(sp+48,  R3, I3);
        DEINT(sp+64,  R4, I4); DEINT(sp+80,  R5, I5);
        DEINT(sp+96,  R6, I6); DEINT(sp+112, R7, I7);
        TR8(R0,R1,R2,R3,R4,R5,R6,R7);
        TR8(I0,I1,I2,I3,I4,I5,I6,I7);
        double* dr = dR + ((r & 7)*8 + (r >> 3))*64;
        double* di = dI + ((r & 7)*8 + (r >> 3))*64;
        _mm512_storeu_pd(dr, R0);    _mm512_storeu_pd(dr+8, R1);
        _mm512_storeu_pd(dr+16, R2); _mm512_storeu_pd(dr+24, R3);
        _mm512_storeu_pd(dr+32, R4); _mm512_storeu_pd(dr+40, R5);
        _mm512_storeu_pd(dr+48, R6); _mm512_storeu_pd(dr+56, R7);
        _mm512_storeu_pd(di, I0);    _mm512_storeu_pd(di+8, I1);
        _mm512_storeu_pd(di+16, I2); _mm512_storeu_pd(di+24, I3);
        _mm512_storeu_pd(di+32, I4); _mm512_storeu_pd(di+40, I5);
        _mm512_storeu_pd(di+48, I6); _mm512_storeu_pd(di+56, I7);
    }
}
// fused: unperm (pi) + interleave to complex; inverse of split_unperm64
static void merge_unperm64(const double* sR, const double* sI, double* dst){
    for (long r = 0; r < 64; ++r){
        const double* sr = sR + r*64; const double* si = sI + r*64;
        __m512d R0 = _mm512_loadu_pd(sr),    R1 = _mm512_loadu_pd(sr+8),
                R2 = _mm512_loadu_pd(sr+16), R3 = _mm512_loadu_pd(sr+24),
                R4 = _mm512_loadu_pd(sr+32), R5 = _mm512_loadu_pd(sr+40),
                R6 = _mm512_loadu_pd(sr+48), R7 = _mm512_loadu_pd(sr+56);
        __m512d I0 = _mm512_loadu_pd(si),    I1 = _mm512_loadu_pd(si+8),
                I2 = _mm512_loadu_pd(si+16), I3 = _mm512_loadu_pd(si+24),
                I4 = _mm512_loadu_pd(si+32), I5 = _mm512_loadu_pd(si+40),
                I6 = _mm512_loadu_pd(si+48), I7 = _mm512_loadu_pd(si+56);
        TR8(R0,R1,R2,R3,R4,R5,R6,R7);
        TR8(I0,I1,I2,I3,I4,I5,I6,I7);
        double* dp = dst + ((r & 7)*8 + (r >> 3))*128;
        if (((uintptr_t)dp & 63) == 0){
            INTST_NT(dp,     R0, I0); INTST_NT(dp+16,  R1, I1);
            INTST_NT(dp+32,  R2, I2); INTST_NT(dp+48,  R3, I3);
            INTST_NT(dp+64,  R4, I4); INTST_NT(dp+80,  R5, I5);
            INTST_NT(dp+96,  R6, I6); INTST_NT(dp+112, R7, I7);
        } else {
            INTST(dp,     R0, I0); INTST(dp+16,  R1, I1);
            INTST(dp+32,  R2, I2); INTST(dp+48,  R3, I3);
            INTST(dp+64,  R4, I4); INTST(dp+80,  R5, I5);
            INTST(dp+96,  R6, I6); INTST(dp+112, R7, I7);
        }
    }
}
// transpose 64x64 (split re/im) and interleave-store
static void transpose_merge64(const double* sR, const double* sI, double* dst){
    for (long a0 = 0; a0 < 64; a0 += 8){
        for (long b0 = 0; b0 < 64; b0 += 8){
            __m512d R[8], I[8];
            for (int t = 0; t < 8; ++t){
                R[t] = _mm512_loadu_pd(sR + (a0+t)*64 + b0);
                I[t] = _mm512_loadu_pd(sI + (a0+t)*64 + b0);
            }
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            TR8(I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]);
            if (((uintptr_t)dst & 63) == 0)
                for (int t = 0; t < 8; ++t) INTST_NT(dst + ((b0+t)*64 + a0)*2, R[t], I[t]);
            else
                for (int t = 0; t < 8; ++t) INTST(dst + ((b0+t)*64 + a0)*2, R[t], I[t]);
        }
    }
}
'''

PI64 = [ (k % 8)*8 + k//8 for k in range(64) ]


PI64 = [ (k % 8)*8 + k//8 for k in range(64) ]

def gen_codelets():
    parts = []
    for L in SIZES:
        if L == 64:
            parts.append("#define TROW 64")
            parts.append(emit_codelet(64, "fft64_pT", in_perm=PI64, tileT=True))
            parts.append(emit_codelet(64, "fft64_pT_f1", tileT=True, interleaved_in=True))
            parts.append("#undef TROW")
            parts.append(emit_codelet(64, "fft64_pp", in_perm=PI64, out_perm=PI64))
            parts.append(emit_codelet(64, "fft64_pp_f1", out_perm=PI64))
            parts.append(emit_codelet(64, "fft64_f"))
        elif L == 17:
            parts.append(gen_prime_codelet(17, "fft17_f", KB=4))
        elif L == 23:
            parts.append(gen_prime_codelet(23, "fft23_f", KB=6))
        else:
            parts.append(emit_codelet(L, f"fft{L}_f", pf=8 if L in (36, 45) else 0))
    return "\n\n".join(parts)

MAPSLAB = r"""
// elementwise z = y + c ; y = z/(1+|z|) over n doubles (vector width 8)
static void map_slab(double* yr, double* yi, const double* cr, const double* ci, long n){
    const __m512d vone = _mm512_set1_pd(1.0);
    const __m512d vhalf = _mm512_set1_pd(0.5);
    const __m512d v15 = _mm512_set1_pd(1.5);
    const __m512d v2 = _mm512_set1_pd(2.0);
    const __m512d vtiny = _mm512_set1_pd(1e-300);
    long t = 0;
    for (; t + 16 <= n; t += 16){
        _mm_prefetch((const char*)(cr + t + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(ci + t + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(cr + t + 136), _MM_HINT_T0);
        _mm_prefetch((const char*)(ci + t + 136), _MM_HINT_T0);
        __m512d zr0 = _mm512_load_pd(yr + t) + _mm512_load_pd(cr + t);
        __m512d zi0 = _mm512_load_pd(yi + t) + _mm512_load_pd(ci + t);
        __m512d zr1 = _mm512_load_pd(yr + t + 8) + _mm512_load_pd(cr + t + 8);
        __m512d zi1 = _mm512_load_pd(yi + t + 8) + _mm512_load_pd(ci + t + 8);
        __m512d r20 = _mm512_max_pd(zr0*zr0 + zi0*zi0, vtiny);
        __m512d r21 = _mm512_max_pd(zr1*zr1 + zi1*zi1, vtiny);
        __m512d e0 = _mm512_rsqrt14_pd(r20);
        __m512d e1 = _mm512_rsqrt14_pd(r21);
        __m512d h0 = r20 * vhalf, h1 = r21 * vhalf;
        e0 = e0 * (v15 - h0*e0*e0);
        e1 = e1 * (v15 - h1*e1*e1);
        e0 = e0 * (v15 - h0*e0*e0);
        e1 = e1 * (v15 - h1*e1*e1);
        __m512d dn0 = vone + r20*e0;
        __m512d dn1 = vone + r21*e1;
        __m512d rc0 = _mm512_div_pd(vone, dn0);       // divider unit
        __m512d rc1 = _mm512_rcp14_pd(dn1);           // FMA path
        rc1 = rc1 * (v2 - dn1*rc1);
        rc1 = rc1 * (v2 - dn1*rc1);
        _mm512_store_pd(yr + t, zr0*rc0);
        _mm512_store_pd(yi + t, zi0*rc0);
        _mm512_store_pd(yr + t + 8, zr1*rc1);
        _mm512_store_pd(yi + t + 8, zi1*rc1);
    }
    for (; t < n; t += 8){
        __m512d zr = _mm512_load_pd(yr + t) + _mm512_load_pd(cr + t);
        __m512d zi = _mm512_load_pd(yi + t) + _mm512_load_pd(ci + t);
        __m512d r2 = _mm512_max_pd(zr*zr + zi*zi, vtiny);
        __m512d e = _mm512_rsqrt14_pd(r2);
        __m512d h = r2 * vhalf;
        e = e * (v15 - h*e*e);
        e = e * (v15 - h*e*e);
        __m512d rc = _mm512_div_pd(vone, vone + r2*e);
        _mm512_store_pd(yr + t, zr*rc);
        _mm512_store_pd(yi + t, zi*rc);
    }
}
"""

def gen_batched(L):
    LL = L*L; LLL = L*L*L
    SLAB = LL*8 + 8
    s = f"""
// ---------------- batched-mode, L={L} (lanes = volumes) ----------------
static void iter_b{L}(double* SR, double* SI, const double* CR, const double* CI){{
    for (long r = 0; r < {LL}; ++r){{
        double* pr = SR + r*8; double* pi_ = SI + r*8;
        fft{L}_f(pr, pi_, pr, pi_, {SLAB}, {SLAB});
    }}
    for (long x = 0; x < {L}; ++x){{
        double* sr = SR + x*{SLAB}; double* si = SI + x*{SLAB};
        for (long z = 0; z < {L}; ++z)
            fft{L}_f(sr + z*8, si + z*8, sr + z*8, si + z*8, {L*8}, {L*8});
        for (long y = 0; y < {L}; ++y)
            fft{L}_f(sr + y*{L*8}, si + y*{L*8}, sr + y*{L*8}, si + y*{L*8}, 8, 8);
        if (x)
            map_slab(SR + (x-1)*{SLAB}, SI + (x-1)*{SLAB}, CR + (x-1)*{SLAB}, CI + (x-1)*{SLAB}, {LL*8});
    }}
    map_slab(SR + {L-1}*{SLAB}, SI + {L-1}*{SLAB}, CR + {L-1}*{SLAB}, CI + {L-1}*{SLAB}, {LL*8});
}}
static void run_small_{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
    static double *SR, *SI, *CR, *CI;
    if (!SR){{
        SR = skew_alloc((size_t){L}*{SLAB}, 0);
        SI = skew_alloc((size_t){L}*{SLAB}, 1);
        CR = skew_alloc((size_t){L}*{SLAB}, 2);
        CI = skew_alloc((size_t){L}*{SLAB}, 3);
    }}
    long G = (B + 7) / 8;
    for (long g = 0; g < G; ++g){{
        long nl = B - g*8; if (nl > 8) nl = 8;
        const double* xg = x0 + g*8*(long){LLL}*2;
        const double* cg = c  + g*8*(long){LLL}*2;
        for (long x = 0; x < {L}; ++x){{
            laneize(xg + x*{LL}*2, {LL}, {LLL}*2, nl, SR + x*{SLAB}, SI + x*{SLAB});
            laneize(cg + x*{LL}*2, {LL}, {LLL}*2, nl, CR + x*{SLAB}, CI + x*{SLAB});
        }}
        for (long it = 1; it <= m; ++it){{
            iter_b{L}(SR, SI, CR, CI);
            if (it == 1){{
                double* o = out1 + g*8*(long){LLL}*2;
                for (long x = 0; x < {L}; ++x)
                    delaneize(SR + x*{SLAB}, SI + x*{SLAB}, {LL}, {LLL}*2, nl, o + x*{LL}*2);
            }}
        }}
        double* o = outm + g*8*(long){LLL}*2;
        for (long x = 0; x < {L}; ++x)
            delaneize(SR + x*{SLAB}, SI + x*{SLAB}, {LL}, {LLL}*2, nl, o + x*{LL}*2);
    }}
}}
"""
    return s


def gen_pervol_pad(L):
    """Per-volume mode for L in {36,45}: unit dim padded to Lp (multiple of 8), zero-pad lanes.
    Layout: [x][a][b] with a,b in [0,Lp), slice stride SLs = Lp*Lp+8. No masks anywhere."""
    Lp = (L + 7) // 8 * 8
    LL = L*L; LLL = L*L*L
    SLs = Lp*Lp + 8
    NCH = Lp // 8
    return f"""
// ---------------- per-volume padded mode, L={L} (Lp={Lp}) ----------------
static void transpose_pad{L}(const double* s, double* d){{
    for (long a0 = 0; a0 < {Lp}; a0 += 8)
        for (long b0 = 0; b0 < {Lp}; b0 += 8){{
            __m512d R[8];
            for (int t = 0; t < 8; ++t) R[t] = _mm512_loadu_pd(s + (a0+t)*{Lp} + b0);
            TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
            for (int t = 0; t < 8; ++t) _mm512_storeu_pd(d + (b0+t)*{Lp} + a0, R[t]);
        }}
}}
static void iter_vp{L}(double* XR, double* XI, const double* cr, const double* ci){{
    static double SCR[{Lp}*8] __attribute__((aligned(64))), SCI[{Lp}*8] __attribute__((aligned(64)));   // zero-init; rows L..Lp-1 stay zero
    static double SLR[{Lp}*{Lp}] __attribute__((aligned(64))), SLI[{Lp}*{Lp}] __attribute__((aligned(64)));
    for (long x = 0; x < {L}; ++x){{
        double* xr = XR + x*{SLs}; double* xi = XI + x*{SLs};
        for (long uc = 0; uc < {NCH}; ++uc){{
            fft{L}_f(xr + uc*8, xi + uc*8, SCR, SCI, {Lp}, 8);
            for (long kb = 0; kb < {NCH}; ++kb){{
                __m512d R[8], I[8];
                for (int t = 0; t < 8; ++t){{ R[t] = _mm512_load_pd(SCR + (kb*8+t)*8); I[t] = _mm512_load_pd(SCI + (kb*8+t)*8); }}
                TR8(R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7]);
                TR8(I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]);
                for (int u = 0; u < 8; ++u){{
                    _mm512_store_pd(SLR + (uc*8+u)*{Lp} + kb*8, R[u]);
                    _mm512_store_pd(SLI + (uc*8+u)*{Lp} + kb*8, I[u]);
                }}
            }}
        }}
        for (long uc = 0; uc < {NCH}; ++uc)
            fft{L}_f(SLR + uc*8, SLI + uc*8, xr + uc*8, xi + uc*8, {Lp}, {Lp});
    }}
    for (long a = 0; a < {L}; ++a){{
        for (long uc = 0; uc < {NCH}; ++uc){{
            long base = a*{Lp} + uc*8;
            fft{L}_f(XR+base, XI+base, XR+base, XI+base, {SLs}, {SLs});
        }}
        if (a){{
            long ap = a - 1;
            for (long x = 0; x < {L}; ++x)
                map_slab(XR + x*{SLs} + ap*{Lp}, XI + x*{SLs} + ap*{Lp}, cr + x*{SLs} + ap*{Lp}, ci + x*{SLs} + ap*{Lp}, {Lp});
        }}
    }}
    {{ long ap = {L}-1;
    for (long x = 0; x < {L}; ++x)
        map_slab(XR + x*{SLs} + ap*{Lp}, XI + x*{SLs} + ap*{Lp}, cr + x*{SLs} + ap*{Lp}, ci + x*{SLs} + ap*{Lp}, {Lp}); }}
}}
static void run_pv{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
    static double *AR, *AI, *C0R, *C0I, *C1R, *C1I, *T1, *T2;
    if (!AR){{
        AR = skew_alloc((size_t){L}*{SLs}, 0); AI = skew_alloc((size_t){L}*{SLs}, 1);
        C0R = skew_alloc((size_t){L}*{SLs}, 4); C0I = skew_alloc((size_t){L}*{SLs}, 5);
        C1R = skew_alloc((size_t){L}*{SLs}, 6); C1I = skew_alloc((size_t){L}*{SLs}, 7);
        T1 = skew_alloc({Lp}*{Lp}, 2); T2 = skew_alloc({Lp}*{Lp}, 5);
    }}
    for (long b = 0; b < B; ++b){{
        const double* xv = x0 + b*(long){LLL}*2;
        const double* cv = c  + b*(long){LLL}*2;
        for (long x = 0; x < {L}; ++x){{
            for (long y = 0; y < {L}; ++y){{
                split_n(xv + (x*{LL} + y*{L})*2, {L}, AR + x*{SLs} + y*{Lp}, AI + x*{SLs} + y*{Lp});
                split_n(cv + (x*{LL} + y*{L})*2, {L}, C0R + x*{SLs} + y*{Lp}, C0I + x*{SLs} + y*{Lp});
            }}
            transpose_pad{L}(C0R + x*{SLs}, C1R + x*{SLs});
            transpose_pad{L}(C0I + x*{SLs}, C1I + x*{SLs});
        }}
        for (long it = 1; it <= m; ++it){{
            const double* ccr = (it & 1) ? C1R : C0R;
            const double* cci = (it & 1) ? C1I : C0I;
            iter_vp{L}(AR, AI, ccr, cci);
            if (it == 1 && m > 1){{
                double* o = out1 + b*(long){LLL}*2;
                for (long x = 0; x < {L}; ++x){{
                    transpose_pad{L}(AR + x*{SLs}, T1);
                    transpose_pad{L}(AI + x*{SLs}, T2);
                    for (long y = 0; y < {L}; ++y)
                        merge_n(T1 + y*{Lp}, T2 + y*{Lp}, {L}, o + (x*{LL} + y*{L})*2);
                }}
            }}
        }}
        double* om = outm + b*(long){LLL}*2;
        if (m & 1){{
            for (long x = 0; x < {L}; ++x){{
                transpose_pad{L}(AR + x*{SLs}, T1);
                transpose_pad{L}(AI + x*{SLs}, T2);
                for (long y = 0; y < {L}; ++y)
                    merge_n(T1 + y*{Lp}, T2 + y*{Lp}, {L}, om + (x*{LL} + y*{L})*2);
            }}
        }} else {{
            for (long x = 0; x < {L}; ++x)
                for (long y = 0; y < {L}; ++y)
                    merge_n(AR + x*{SLs} + y*{Lp}, AI + x*{SLs} + y*{Lp}, {L}, om + (x*{LL} + y*{L})*2);
        }}
        if (m == 1) memcpy(out1 + b*(long){LLL}*2, om, (size_t){LLL}*2*sizeof(double));
    }}
}}
"""

def gen_pervol64():
    LL = 64*64; LLL = 64*64*64; SL = LL + 8
    return f"""
// ---------------- per-volume-mode, L=64, pi-permuted storage, slice-scratch ----------------
static void iter_v64(double* XR, double* XI, const double* cr, const double* ci){{
    static double SPR[2][{LL}] __attribute__((aligned(64)));
    static double SPI[2][{LL}] __attribute__((aligned(64)));
    {{
        double* xr = XR; double* xi = XI;
        for (long uc = 0; uc < 8; ++uc)
            fft64_pT(xr + uc*8, xi + uc*8, SPR[0] + uc*512, SPI[0] + uc*512, 64, 0);
        int cb = 0;
        for (long x = 0; x < 63; ++x){{
            double* nr = XR + (x+1)*{SL}; double* ni = XI + (x+1)*{SL};
            double* or_ = XR + x*{SL}; double* oi = XI + x*{SL};
            for (long uc = 0; uc < 8; ++uc)
                fft64_pT(nr + uc*8, ni + uc*8, SPR[cb^1] + uc*512, SPI[cb^1] + uc*512, 64, 0);
            for (long uc = 0; uc < 8; ++uc)
                fft64_pp(SPR[cb] + uc*8, SPI[cb] + uc*8, or_ + uc*8, oi + uc*8, 64, 64);
            cb ^= 1;
        }}
        double* or_ = XR + 63*{SL}; double* oi = XI + 63*{SL};
        for (long uc = 0; uc < 8; ++uc)
            fft64_pp(SPR[cb] + uc*8, SPI[cb] + uc*8, or_ + uc*8, oi + uc*8, 64, 64);
    }}
    for (long a = 0; a < 64; ++a){{
        for (long uc = 0; uc < 8; ++uc){{
            long base = a*64 + uc*8;
            fft64_f(XR+base, XI+base, XR+base, XI+base, {SL}, {SL});
        }}
        if (a){{
            long ap = a - 1;
            for (long x = 0; x < 64; ++x)
                map_slab(XR + x*{SL} + ap*64, XI + x*{SL} + ap*64, cr + x*{SL} + ap*64, ci + x*{SL} + ap*64, 64);
        }}
    }}
    for (long x = 0; x < 64; ++x)
        map_slab(XR + x*{SL} + 63*64, XI + x*{SL} + 63*64, cr + x*{SL} + 63*64, ci + x*{SL} + 63*64, 64);
}}
static void iter_v64_first(const double* xv, double* XR, double* XI,
                           const double* cr, const double* ci){{
    static double SPR[{LL}] __attribute__((aligned(64)));
    static double SPI[{LL}] __attribute__((aligned(64)));
    for (long x = 0; x < 64; ++x){{
        const double* src = xv + x*{LL}*2;
        double* xr = XR + x*{SL}; double* xi = XI + x*{SL};
        for (long uc = 0; uc < 8; ++uc)
            fft64_pT_f1(src + uc*16, (const double*)0, SPR + uc*512, SPI + uc*512, 64, 0);
        for (long uc = 0; uc < 8; ++uc)
            fft64_pp_f1(SPR + uc*8, SPI + uc*8, xr + uc*8, xi + uc*8, 64, 64);
    }}
    for (long a = 0; a < 64; ++a){{
        for (long uc = 0; uc < 8; ++uc){{
            long base = a*64 + uc*8;
            fft64_f(XR+base, XI+base, XR+base, XI+base, {SL}, {SL});
        }}
        if (a){{
            long ap = a - 1;
            for (long x = 0; x < 64; ++x)
                map_slab(XR + x*{SL} + ap*64, XI + x*{SL} + ap*64, cr + x*{SL} + ap*64, ci + x*{SL} + ap*64, 64);
        }}
    }}
    for (long x = 0; x < 64; ++x)
        map_slab(XR + x*{SL} + 63*64, XI + x*{SL} + 63*64, cr + x*{SL} + 63*64, ci + x*{SL} + 63*64, 64);
}}
static void export64(const double* curR, const double* curI, long parity, double* o,
                     double* T1, double* T2){{
    for (long x = 0; x < 64; ++x){{
        if (parity & 1){{
            unperm_slice64(curR + x*{SL}, T1);
            unperm_slice64(curI + x*{SL}, T2);
            transpose_merge64(T1, T2, o + x*{LL}*2);
        }} else {{
            merge_unperm64(curR + x*{SL}, curI + x*{SL}, o + x*{LL}*2);
        }}
    }}
}}
static void run_big_64(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
    static double *AR, *AI, *C0R, *C0I, *C1R, *C1I, *T1, *T2;
    if (!AR){{
        AR = skew_alloc((size_t)64*{SL}, 0); AI = skew_alloc((size_t)64*{SL}, 1);
        C0R = skew_alloc((size_t)64*{SL}, 4); C0I = skew_alloc((size_t)64*{SL}, 5);
        C1R = skew_alloc((size_t)64*{SL}, 6); C1I = skew_alloc((size_t)64*{SL}, 7);
        T1 = skew_alloc({LL}*2, 2); T2 = skew_alloc({LL}*2, 5);
    }}
    for (long b = 0; b < B; ++b){{
        const double* xv = x0 + b*(long){LLL}*2;
        const double* cv = c  + b*(long){LLL}*2;
        for (long x = 0; x < 64; ++x){{
            split_unperm64(cv + x*{LL}*2, C0R + x*{SL}, C0I + x*{SL});
            transpose_plane(C0R + x*{SL}, C1R + x*{SL}, 64);
            transpose_plane(C0I + x*{SL}, C1I + x*{SL}, 64);
        }}
        iter_v64_first(xv, AR, AI, C1R, C1I);
        if (m > 1)
            export64(AR, AI, 1, out1 + b*(long){LLL}*2, T1, T2);
        for (long it = 2; it <= m; ++it){{
            const double* ccr = (it & 1) ? C1R : C0R;
            const double* cci = (it & 1) ? C1I : C0I;
            iter_v64(AR, AI, ccr, cci);
        }}
        export64(AR, AI, m & 1, outm + b*(long){LLL}*2, T1, T2);
        if (m == 1) memcpy(out1 + b*(long){LLL}*2, outm + b*(long){LLL}*2, (size_t){LLL}*2*sizeof(double));
    }}
}}
"""

def gen_dispatch():
    lines = ["void run_size(long L, long B, long m, const double* x0, const double* c,",
             "              double* out1, double* outm, long mode){",
             "    (void)mode;",
             "    switch (L){"]
    for L in SIZES:
        lines.append(f"    case {L}:")
        if L == 64:
            lines.append("        run_big_64(B, m, x0, c, out1, outm);")
        elif L in (13, 17, 23, 36, 45):
            LLL = L**3
            thresh = 99 if L in (23, 45) else 7   # 23/45: per-volume always better for partial groups
            lines.append(f"        {{ long bf = (mode == 2) ? B : ((mode == 3) ? 0 : (B / 8) * 8);")
            lines.append(f"          long tail = B - bf;")
            lines.append(f"          if (tail && tail >= {thresh} && mode == 0){{ bf = B; tail = 0; }}")
            lines.append(f"          if (bf) run_small_{L}(bf, m, x0, c, out1, outm);")
            lines.append(f"          if (tail) run_pv{L}(tail, m, x0 + bf*(long){LLL}*2, c + bf*(long){LLL}*2,")
            lines.append(f"                              out1 + bf*(long){LLL}*2, outm + bf*(long){LLL}*2); }}")
        else:
            lines.append(f"        run_small_{L}(B, m, x0, c, out1, outm);")
        lines.append("        break;")
    lines.append("    }")
    lines.append("    _mm_sfence();")
    lines.append("}")
    return "\n".join(lines)

def main():
    parts = [HEADER, gen_codelets(), MAPSLAB]
    for L in SIZES:
        if L != 64:
            parts.append(gen_batched(L))
    for Lx in (13, 17, 23, 36, 45):
        parts.append(gen_pervol_pad(Lx))
    parts.append(gen_pervol64())
    parts.append(gen_dispatch())
    src = "\n".join(parts)
    open("implementation.c", "w").write(src)
    print("lines:", len(src.splitlines()))

if __name__ == "__main__":
    main()
