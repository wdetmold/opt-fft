#!/usr/bin/env python3
"""Regenerate implementation.c from the reconstructed gen.py, emulating the
x86-64 Linux container's numpy longdouble (80-bit x87 extended) semantics
with mpmath, so the emitted twiddle constants match the graded file.

Emulation of  float(np.cos((-2*PI)*kk/N))  on x86:
  - PI parsed to 64-bit-mantissa extended
  - each arithmetic op rounded to 64-bit mantissa (mp.prec=64)
  - cosl computed (assumed correctly rounded to 64-bit mantissa)
  - result rounded 64 -> 53 bits (double rounding preserved)
Residual risk: glibc cosl/sinl is not always correctly rounded at 80-bit;
constants whose 80-bit value sits within ~1 ulp(2^-63) of a 53-bit rounding
boundary could differ (expected: a handful at most, <=1 ulp in double).
"""
import sys, os
from mpmath import mp, mpf, cos, sin

HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.path.join(HERE, 'workdir')
sys.path.insert(0, WD)

import gen

def r64(x):
    mp.prec = 64
    return +x

def r53(x):
    mp.prec = 53
    y = +x
    mp.prec = 64
    return y

PI64 = r64(mpf('3.14159265358979323846264338327950288'))

def ext_angle(sign, kk, N):
    mp.prec = 64
    t = r64(sign * 2 * PI64)      # (-2*PI) or (2*PI), rounded to extended
    t = r64(t * kk)
    t = r64(t / N)
    return t

def cos64(ang):
    mp.prec = 200
    v = cos(ang)
    return float(r53(r64(v)))

def sin64(ang):
    mp.prec = 200
    v = sin(ang)
    return float(r53(r64(v)))

def omega(N, k):
    kk = k % N
    ang = ext_angle(-1, kk, N)
    c = cos64(ang); s = sin64(ang)
    if (4 * kk) % N == 0:
        c = float(round(c)); s = float(round(s))
    return (c, s)

def trig(N, j, k, fn):
    ang = ext_angle(+1, (j * k) % N, N)
    return cos64(ang) if fn == 'c' else sin64(ang)

# sanity: compare against direct-double path to count changed constants
import math
def dbl_omega(N, k):
    kk = k % N
    ang = (-2 * math.pi) * kk / N
    c = math.cos(ang); s = math.sin(ang)
    if (4 * kk) % N == 0:
        c = float(round(c)); s = float(round(s))
    return (c, s)

gen.omega = omega
gen.trig = trig

outpath = os.path.join(WD, 'implementation.c')
_real_open = open
def open_redirect(path, mode='r', *a, **k):
    if path == '/workdir/implementation.c':
        path = outpath
    return _real_open(path, mode, *a, **k)
gen.open = open_redirect

gen.main()
print("regenerated with x86-extended emulation ->", outpath)
