"""Shared emitter utilities."""
import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')

def hexd(x):
    """double -> C hex literal"""
    f = float(x)
    return float.hex(f)

def trig(L):
    """exact mod-L twiddles: (cos, sin) of -2*pi*t/L as doubles, t=0..L-1"""
    cs = []
    for t in range(L):
        ang = (-2*PI) * LD(t) / LD(L)
        cs.append((float(np.cos(ang)), float(np.sin(ang))))
    return cs

class Emit:
    def __init__(self):
        self.lines = []
        self.ind = 0
    def __call__(self, s=""):
        for ln in s.split("\n"):
            self.lines.append("    "*self.ind + ln)
    def text(self):
        return "\n".join(self.lines) + "\n"

PRELUDE = r"""
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#define ALIGN64 __attribute__((aligned(64)))
typedef __m512d V;
#define VL(p)      _mm512_load_pd(p)
#define VLU(p)     _mm512_loadu_pd(p)
#define VS(p,v)    _mm512_store_pd(p,v)
#define VSU(p,v)   _mm512_storeu_pd(p,v)
#define VADD(a,b)  _mm512_add_pd(a,b)
#define VSUB(a,b)  _mm512_sub_pd(a,b)
#define VMUL(a,b)  _mm512_mul_pd(a,b)
#define VFMA(a,b,c)  _mm512_fmadd_pd(a,b,c)
#define VFMS(a,b,c)  _mm512_fmsub_pd(a,b,c)
#define VFNMA(a,b,c) _mm512_fnmadd_pd(a,b,c)
#define VSET1(x)   _mm512_set1_pd(x)
#define VRSQRT(x)  _mm512_rsqrt14_pd(x)
#define VRCP(x)    _mm512_rcp14_pd(x)


static const long long IDXR_[8] ALIGN64 = {0,2,4,6,8,10,12,14};
static const long long IDXI_[8] ALIGN64 = {1,3,5,7,9,11,13,15};
static const long long IDXLO_[8] ALIGN64 = {0,8,1,9,2,10,3,11};
static const long long IDXHI_[8] ALIGN64 = {4,12,5,13,6,14,7,15};
#define PERM2(a,idx,b) _mm512_permutex2var_pd(a, _mm512_load_si512((const void*)idx), b)
#define PERM2Z(m,a,idx,b) _mm512_maskz_permutex2var_pd(m, a, _mm512_load_si512((const void*)idx), b)

// ---- map: given zr,zi (post +c), produce zr*q, zi*q with q = 1/(1+sqrt(zr^2+zi^2))
// rsqrt14 seed + 2 Newton for rsqrt; m = s*r; rcp14 seed + 2 Newton for 1/(1+m).
// all-FMA, ~1ulp.  s floored at tiny to avoid rsqrt(0)=inf -> NaN (pad lanes).
#define MAP2(zr, zi) do { \
    V s_ = VFMA(zr, zr, _mm512_mul_pd(zi, zi)); \
    V r_ = VRSQRT(s_); \
    V h_ = VMUL(s_, VSET1(0.5)); \
    V e_ = VFNMA(VMUL(r_, r_), h_, VSET1(0.5)); \
    r_ = VFMA(r_, e_, r_); \
    e_ = VFNMA(VMUL(r_, r_), h_, VSET1(0.5)); \
    r_ = VFMA(r_, e_, r_); \
    V m_ = VMUL(s_, r_); \
    V u_ = VADD(VSET1(1.0), m_); \
    V q_ = VRCP(u_); \
    V t_ = VFNMA(u_, q_, VSET1(1.0)); \
    q_ = VFMA(q_, t_, q_); \
    t_ = VFNMA(u_, q_, VSET1(1.0)); \
    q_ = VFMA(q_, t_, q_); \
    zr = VMUL(zr, q_); \
    zi = VMUL(zi, q_); \
} while(0)
"""
