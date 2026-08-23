import numpy as np, mpmath
mpmath.mp.prec = 120
def hexd(v): return float(np.float64(v)).hex()
def cosv(N,t): return float(mpmath.cos(2*mpmath.pi*t/N))
def sinv(N,t): return float(mpmath.sin(2*mpmath.pi*t/N))
def tidx(j,k,N):
    h=(N-1)//2
    t=(j*k)%N
    return (t,1) if t<=h else (N-t,-1)

def gen_quad2(N, fname):
    h=(N-1)//2
    L=[]; W=L.append
    W(f"static const double CC_{N}[{2*h}] ALIGN64 = {{" + ",".join(hexd(cosv(N,t+1)) for t in range(h)) + "," + ",".join(hexd(sinv(N,t+1)) for t in range(h)) + "};")
    W(f"static void {fname}(double* re, double* im, long es){{")
    W(f"  double scr[{(3*h+1)*8}] __attribute__((aligned(64)));")  # o_re[h], e_im[h], x0i, ar-park[h]
    PARK = (2*h+1)*8
    # Sweep 1: Ar + stage o_re + X0r
    W("  {")
    for t in range(h): W(f"  const __m512d C{t} = _mm512_set1_pd(CC_{N}[{t}]);")
    W(f"  __m512d x0r = _mm512_load_pd(re);")
    W(f"  __m512d s0r = x0r;")
    for k in range(1,h+1): W(f"  __m512d ar{k} = x0r;")
    for j in range(1,h+1):
        W(f"  {{ __m512d p = _mm512_load_pd(re + {j}*es), q = _mm512_load_pd(re + {N-j}*es);")
        W(f"    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);")
        W(f"    _mm512_store_pd(scr + {(j-1)*8}, o); s0r = _mm512_add_pd(s0r, e);")
        for k in range(1,h+1):
            t,_ = tidx(j,k,N)
            W(f"    ar{k} = _mm512_fmadd_pd(C{t-1}, e, ar{k});")
        W("  }")
    W(f"  _mm512_store_pd(re, s0r);")
    for k in range(1,h+1): W(f"  _mm512_store_pd(scr + {PARK + (k-1)*8}, ar{k});")
    W("  }")
    # Sweep 2: Bi + stage e_im + x0i; combine RE
    W("  {")
    for t in range(h): W(f"  const __m512d S{t} = _mm512_set1_pd(CC_{N}[{h+t}]);")
    W(f"  __m512d x0i = _mm512_load_pd(im);")
    W(f"  _mm512_store_pd(scr + {2*h*8}, x0i);")
    W(f"  __m512d s0i = x0i;")
    for k in range(1,h+1): W(f"  __m512d bi{k} = _mm512_setzero_pd();")
    for j in range(1,h+1):
        W(f"  {{ __m512d p = _mm512_load_pd(im + {j}*es), q = _mm512_load_pd(im + {N-j}*es);")
        W(f"    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);")
        W(f"    _mm512_store_pd(scr + {(h+j-1)*8}, e); s0i = _mm512_add_pd(s0i, e);")
        for k in range(1,h+1):
            t,s = tidx(j,k,N)
            op = "fmadd" if s>0 else "fnmadd"
            W(f"    bi{k} = _mm512_{op}_pd(S{t-1}, o, bi{k});")
        W("  }")
    W(f"  _mm512_store_pd(im, s0i);")
    for k in range(1,h+1):
        W(f"  {{ __m512d a = _mm512_load_pd(scr + {PARK + (k-1)*8});")
        W(f"    _mm512_store_pd(re + {k}*es, _mm512_add_pd(a, bi{k}));")
        W(f"    _mm512_store_pd(re + {N-k}*es, _mm512_sub_pd(a, bi{k})); }}")
    W("  }")
    # Sweep 3: Ai; park
    W("  {")
    for t in range(h): W(f"  const __m512d C{t} = _mm512_set1_pd(CC_{N}[{t}]);")
    W(f"  __m512d x0i = _mm512_load_pd(scr + {2*h*8});")
    for k in range(1,h+1): W(f"  __m512d ai{k} = x0i;")
    for j in range(1,h+1):
        W(f"  {{ __m512d e = _mm512_load_pd(scr + {(h+j-1)*8});")
        for k in range(1,h+1):
            t,_ = tidx(j,k,N)
            W(f"    ai{k} = _mm512_fmadd_pd(C{t-1}, e, ai{k});")
        W("  }")
    for k in range(1,h+1): W(f"  _mm512_store_pd(scr + {PARK + (k-1)*8}, ai{k});")
    W("  }")
    # Sweep 4: Br; combine IM
    W("  {")
    for t in range(h): W(f"  const __m512d S{t} = _mm512_set1_pd(CC_{N}[{h+t}]);")
    for k in range(1,h+1): W(f"  __m512d br{k} = _mm512_setzero_pd();")
    for j in range(1,h+1):
        W(f"  {{ __m512d o = _mm512_load_pd(scr + {(j-1)*8});")
        for k in range(1,h+1):
            t,s = tidx(j,k,N)
            op = "fmadd" if s>0 else "fnmadd"
            W(f"    br{k} = _mm512_{op}_pd(S{t-1}, o, br{k});")
        W("  }")
    for k in range(1,h+1):
        W(f"  {{ __m512d a = _mm512_load_pd(scr + {PARK + (k-1)*8});")
        W(f"    _mm512_store_pd(im + {k}*es, _mm512_sub_pd(a, br{k}));")
        W(f"    _mm512_store_pd(im + {N-k}*es, _mm512_add_pd(a, br{k})); }}")
    W("  }")
    W("}")
    return "\n".join(L)

parts = []
for N in (13,17,23):
    parts.append(gen_quad2(N, f"dftp{N}_v"))
open("primekerns.h","w").write("\n".join(parts))
print("generated")