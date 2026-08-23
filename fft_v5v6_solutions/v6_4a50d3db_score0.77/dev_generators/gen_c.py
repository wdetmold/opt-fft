#!/usr/bin/env python3
"""Emit implementation.c from generated DFT programs."""
import math
import gen_fft as G

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

# zP (row spacing), yP (rows per x-plane) per size; zCW=8*ceil(L/8) compute width
PADS = {6:(8,6), 8:(8,8), 13:(16,13), 17:(24,17), 23:(24,23), 36:(40,36), 45:(48,45), 64:(72,64)}
BPATH = {6,8,13,17,23,36,45}   # sizes with lane-batch variants
TOUCH_T = {36,45,64}           # T-path P1 plane touch-prefetch
TOUCH_B = set()                # B-path by-codelet next-plane prefetch
TZQT = {6,8,64}                # sizes using fzqT+normstoreT in T-path
BTHR = {6:5, 8:9, 13:6, 17:7, 23:9, 36:9, 45:7}  # min tail size for padded lane-batch

G.PLAN.update({2:('base',),3:('base',),4:('base',),5:('dsym',),
               6:('pfa',2,3),8:('ct',2),9:('ct',3),10:('pfa',2,5),
               11:('dsym',),12:('pfa',4,3),13:('rader',),16:('ct',4),
               17:('rader',),22:('pfa',2,11),23:('rader',),
               36:('ct',6),45:('pfa',5,9),64:('ct',4)})

def chex(v):
    if v == int(v) and abs(v) < 1e15:
        return f"{v:.1f}"
    return v.hex()

def build(n):
    P = G.Prog()
    xs = [(P.inp(('r',j)), P.inp(('i',j))) for j in range(n)]
    X = G.dft(P, n, xs)
    return P, xs, X

def live_set(P, X):
    seen=set(); stack=[i for pr in X for i in pr]
    while stack:
        i=stack.pop()
        if i in seen: continue
        seen.add(i)
        op,a,b,c=P.nodes[i]
        if op in ('add','sub'): stack += [a,b]
        elif op in ('neg','mul'): stack.append(i*0+a)
    return seen

def emit_body(P, X, invar, out):
    """invar: node id -> C var name for inputs. Returns (lines, outvars list of (re,im))."""
    live = live_set(P, X)
    lines = []
    name = {}
    for i in sorted(live):
        op,a,b,c = P.nodes[i]
        if op == 'in':
            name[i] = invar[i]
            continue
        v = f"v{i}"
        name[i] = v
        if op == 'add':   lines.append(f"VD {v} = {name[a]} + {name[b]};")
        elif op == 'sub': lines.append(f"VD {v} = {name[a]} - {name[b]};")
        elif op == 'neg': lines.append(f"VD {v} = -{name[a]};")
        elif op == 'mul': lines.append(f"VD {v} = {name[a]} * SET1({chex(c)});")
    outv = [(name[r], name[m]) for (r,m) in X]
    return lines, outv

def gen_strided(L, fname, stride, norm, inplace, P, xs, X, pf=None, stride_st=None):
    """codelet with strided vector loads/stores.
       norm: fused c-add + normalize (args xr,xi,cr,ci) in-place.
       else: src/dst args (sr,si,dr,di); if inplace: single (r,i) args."""
    lines = []
    if norm:
        args = "double*restrict xr, double*restrict xi, const double*restrict cr, const double*restrict ci"
        lr, li, sr_, si_ = "xr","xi","xr","xi"
    elif inplace:
        args = "double*restrict xr, double*restrict xi"
        lr, li, sr_, si_ = "xr","xi","xr","xi"
    else:
        args = "const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di"
        lr, li, sr_, si_ = "sr","si","dr","di"
    lines.append(f"static void {fname}({args}){{")
    invar = {}
    for j in range(L):
        off = j*stride
        lines.append(f"VD r{j} = vload({lr} + {off}); VD i{j} = vload({li} + {off});")
        invar[P.inp(('r',j))] = f"r{j}"
        invar[P.inp(('i',j))] = f"i{j}"
    if pf is not None:
        for j in range(L):
            if (j*stride*8) % 128 == 0:
                lines.append(f"__builtin_prefetch({lr} + {j*stride + pf}, 0, 3); __builtin_prefetch({li} + {j*stride + pf}, 0, 3);")
    body, outv = emit_body(P, X, invar, X)
    lines += body
    sst = stride if stride_st is None else stride_st
    order = sorted(range(L), key=lambda j: max(X[j][0], X[j][1]))
    for j in order:
        off = j*sst
        orr, oii = outv[j]
        if norm:
            lines.append(f"{{ VD zr = {orr} + vload(cr + {off}); VD zi = {oii} + vload(ci + {off});"
                         f" VD sc = nscale(zr*zr + zi*zi);"
                         f" vstore(xr + {off}, zr*sc); vstore(xi + {off}, zi*sc); }}")
        else:
            lines.append(f"vstore({sr_} + {off}, {orr}); vstore({si_} + {off}, {oii});")
    lines.append("}")
    return "\n".join(lines)

def gen_rows(L, fname, zp, P, xs, X, inplace=False, nrows=8):
    """z-lines codelet: 8 rows (stride zp), transposed load/store; stores only nrows rows."""
    nb = (L + 7)//8
    if inplace:
        lines = [f"static void {fname}(double* qr, double* qi){{",
                 "const double* sr = qr; const double* si = qi; double* dr = qr; double* di = qi;"]
    else:
        lines = [f"static void {fname}(const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di){{"]
    invar = {}
    # loads + transpose per component
    for comp, base in (("r","sr"), ("i","si")):
        for jb in range(nb):
            for r in range(8):
                lines.append(f"VD {comp}a{jb}_{r} = vload({base} + {r*zp + jb*8});")
            outs = ",".join(f"{comp}{jb*8+t}" for t in range(8))
            lines.append("VD " + ",".join(f"{comp}{jb*8+t}" for t in range(8)) + ";")
            ins = ",".join(f"{comp}a{jb}_{r}" for r in range(8))
            lines.append(f"TP8({outs},{ins});")
    for j in range(L):
        invar[P.inp(('r',j))] = f"r{j}"
        invar[P.inp(('i',j))] = f"i{j}"
    body, outv = emit_body(P, X, invar, X)
    lines += body
    for comp, base, sel in (("r","dr",0), ("i","di",1)):
        for jb in range(nb):
            ins = ",".join(outv[jb*8+t][sel] if jb*8+t < L else outv[0][sel] for t in range(8))
            outs = ",".join(f"{comp}o{jb}_{r}" for r in range(8))
            lines.append("VD " + outs + ";")
            lines.append(f"TP8({outs},{ins});")
            for r in range(nrows):
                lines.append(f"vstore({base} + {r*zp + jb*8}, {comp}o{jb}_{r});")
    lines.append("}")
    return "\n".join(lines)

HEADER = r'''/* generated by gen_c.py -- do not edit */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef double VD __attribute__((vector_size(64), aligned(64)));
#define SET1(c) ((VD){(c),(c),(c),(c),(c),(c),(c),(c)})
static inline VD vload(const double*p){ return *(const VD*)p; }
static inline void vstore(double*p, VD v){ *(VD*)p = v; }

#define TP8(o0,o1,o2,o3,o4,o5,o6,o7,i0,i1,i2,i3,i4,i5,i6,i7) do{ \
  __m512d _t0=_mm512_unpacklo_pd((__m512d)(i0),(__m512d)(i1)); \
  __m512d _t1=_mm512_unpackhi_pd((__m512d)(i0),(__m512d)(i1)); \
  __m512d _t2=_mm512_unpacklo_pd((__m512d)(i2),(__m512d)(i3)); \
  __m512d _t3=_mm512_unpackhi_pd((__m512d)(i2),(__m512d)(i3)); \
  __m512d _t4=_mm512_unpacklo_pd((__m512d)(i4),(__m512d)(i5)); \
  __m512d _t5=_mm512_unpackhi_pd((__m512d)(i4),(__m512d)(i5)); \
  __m512d _t6=_mm512_unpacklo_pd((__m512d)(i6),(__m512d)(i7)); \
  __m512d _t7=_mm512_unpackhi_pd((__m512d)(i6),(__m512d)(i7)); \
  __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88); \
  __m512d _u1=_mm512_shuffle_f64x2(_t0,_t2,0xDD); \
  __m512d _u2=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
  __m512d _u3=_mm512_shuffle_f64x2(_t1,_t3,0xDD); \
  __m512d _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88); \
  __m512d _u5=_mm512_shuffle_f64x2(_t4,_t6,0xDD); \
  __m512d _u6=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
  __m512d _u7=_mm512_shuffle_f64x2(_t5,_t7,0xDD); \
  o0=(VD)_mm512_shuffle_f64x2(_u0,_u4,0x88); \
  o4=(VD)_mm512_shuffle_f64x2(_u0,_u4,0xDD); \
  o1=(VD)_mm512_shuffle_f64x2(_u2,_u6,0x88); \
  o5=(VD)_mm512_shuffle_f64x2(_u2,_u6,0xDD); \
  o2=(VD)_mm512_shuffle_f64x2(_u1,_u5,0x88); \
  o6=(VD)_mm512_shuffle_f64x2(_u1,_u5,0xDD); \
  o3=(VD)_mm512_shuffle_f64x2(_u3,_u7,0x88); \
  o7=(VD)_mm512_shuffle_f64x2(_u3,_u7,0xDD); \
}while(0)

static inline VD nscale(VD s){
  __m512d S=(__m512d)(s + SET1(1e-300));
#ifdef EXACT_NORM
  __m512d mag=_mm512_sqrt_pd(S);
  __m512d den=_mm512_add_pd(_mm512_set1_pd(1.0),mag);
  return (VD)_mm512_div_pd(_mm512_set1_pd(1.0),den);
#else
  __m512d y=_mm512_rsqrt14_pd(S);
  __m512d h=_mm512_mul_pd(S,_mm512_set1_pd(0.5));
  __m512d t;
  t=_mm512_fnmadd_pd(h,_mm512_mul_pd(y,y),_mm512_set1_pd(1.5)); y=_mm512_mul_pd(y,t);
  t=_mm512_fnmadd_pd(h,_mm512_mul_pd(y,y),_mm512_set1_pd(1.5)); y=_mm512_mul_pd(y,t);
  __m512d mag=_mm512_mul_pd(S,y);
#ifdef PRECISE_NORM
  mag=_mm512_fmadd_pd(_mm512_fnmadd_pd(mag,mag,S),_mm512_mul_pd(_mm512_set1_pd(0.5),y),mag);
#endif
  __m512d den=_mm512_add_pd(_mm512_set1_pd(1.0),mag);
  __m512d v=_mm512_rcp14_pd(den);
  v=_mm512_mul_pd(v,_mm512_fnmadd_pd(den,v,_mm512_set1_pd(2.0)));
#ifdef PRECISE_NORM
  v=_mm512_fmadd_pd(v,_mm512_fnmadd_pd(den,v,_mm512_set1_pd(1.0)),v);
#else
  v=_mm512_mul_pd(v,_mm512_fnmadd_pd(den,v,_mm512_set1_pd(2.0)));
#endif
  return (VD)v;
#endif
}

static inline void deint8(const double*restrict src, double*restrict dre, double*restrict dim){
  __m512d a=_mm512_loadu_pd(src), b=_mm512_loadu_pd(src+8);
  const __m512i er=_mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i oi=_mm512_set_epi64(15,13,11,9,7,5,3,1);
  _mm512_store_pd(dre,_mm512_permutex2var_pd(a,er,b));
  _mm512_store_pd(dim,_mm512_permutex2var_pd(a,oi,b));
}
static inline void intl8(const double*restrict re, const double*restrict im, double*restrict dst){
  __m512d r=_mm512_load_pd(re), i=_mm512_load_pd(im);
  const __m512i lo=_mm512_set_epi64(11,3,10,2,9,1,8,0);
  const __m512i hi=_mm512_set_epi64(15,7,14,6,13,5,12,4);
  _mm512_storeu_pd(dst,_mm512_permutex2var_pd(r,lo,i));
  _mm512_storeu_pd(dst+8,_mm512_permutex2var_pd(r,hi,i));
}
static void row_deint(const double*restrict src, double*restrict dre, double*restrict dim, long n){
  long k=0;
  for(; k+8<=n; k+=8) deint8(src+2*k, dre+k, dim+k);
  for(; k<n; k++){ dre[k]=src[2*k]; dim[k]=src[2*k+1]; }
}
static void normloop(double*restrict qr, double*restrict qi, const double*restrict cr, const double*restrict ci, long n){
  for(long i=0;i<n;i+=8){
    __builtin_prefetch(cr+i+128, 0, 3);
    __builtin_prefetch(ci+i+128, 0, 3);
    VD zr = vload(qr+i) + vload(cr+i);
    VD zi = vload(qi+i) + vload(ci+i);
    VD sc = nscale(zr*zr+zi*zi);
    vstore(qr+i, zr*sc); vstore(qi+i, zi*sc);
  }
}
static void row_intl(const double*restrict re, const double*restrict im, double*restrict dst, long n){
  long k=0;
  for(; k+8<=n; k+=8) intl8(re+k, im+k, dst+2*k);
  for(; k<n; k++){ dst[2*k]=re[k]; dst[2*k+1]=im[k]; }
}
'''

def gen_copy(L, zp, yp):
    """pass-B copy-in/copy-out helpers for size L (T path)."""
    zg = (L+7)//8
    zcw = zg*8
    qp = zcw + (8 if (zcw//8) % 8 == 0 else 0)
    t=[]
    a=t.append
    a(f"static void copyin_{L}(const double*restrict vr, const double*restrict vi, double*restrict qr, double*restrict qi){{")
    a(f"  for(long x=0;x<{L};x++){{")
    a(f"    const double* sr = vr + x*{yp*zp}; const double* si = vi + x*{yp*zp};")
    for k in range(zg):
        a(f"    __builtin_prefetch(sr + {4*yp*zp} + {8*k}, 0, 3); __builtin_prefetch(si + {4*yp*zp} + {8*k}, 0, 3);")
    for k in range(zg):
        a(f"    vstore(qr + x*{qp} + {8*k}, vload(sr + {8*k})); vstore(qi + x*{qp} + {8*k}, vload(si + {8*k}));")
    a("  }")
    for xr in range(L, zg*8):
        for k in range(zg):
            a(f"  vstore(qr + {xr*qp + 8*k}, SET1(0.0)); vstore(qi + {xr*qp + 8*k}, SET1(0.0));")
    a("}")
    a(f"static void normstore_{L}(const double*restrict cr, const double*restrict ci, double*restrict vr, double*restrict vi){{")
    a(f"  const double* qr = QR_{L}; const double* qi = QI_{L};")
    a(f"  for(long x=0;x<{L};x++){{")
    a(f"    double* dr = vr + x*{yp*zp}; double* di = vi + x*{yp*zp};")
    for k in range(zg):
        a(f"    __builtin_prefetch(cr + x*{zcw} + {2*zcw} + {8*k}, 0, 3); __builtin_prefetch(ci + x*{zcw} + {2*zcw} + {8*k}, 0, 3);")
    for k in range(zg):
        a(f"    __builtin_prefetch(dr + {zp} + {8*k}, 1, 3); __builtin_prefetch(di + {zp} + {8*k}, 1, 3);")
    for k in range(zg):
        a(f"    VD zr{k} = vload(qr + x*{qp} + {8*k}) + vload(cr + x*{zcw} + {8*k});")
        a(f"    VD zi{k} = vload(qi + x*{qp} + {8*k}) + vload(ci + x*{zcw} + {8*k});")
    for k in range(zg):
        a(f"    VD sc{k} = nscale(zr{k}*zr{k}+zi{k}*zi{k});")
    for k in range(zg):
        a(f"    vstore(dr + {8*k}, zr{k}*sc{k}); vstore(di + {8*k}, zi{k}*sc{k});")
    a("  }")
    a("}")
    if L in TZQT:
        a(gen_normstoreT(L, zp, yp, zcw, qp))
    a(f"static void copyout_{L}(const double*restrict qr, const double*restrict qi, double*restrict vr, double*restrict vi){{")
    a(f"  for(long x=0;x<{L};x++){{")
    a(f"    double* dr = vr + x*{yp*zp}; double* di = vi + x*{yp*zp};")
    for k in range(zg):
        a(f"    vstore(dr + {8*k}, vload(qr + x*{qp} + {8*k})); vstore(di + {8*k}, vload(qi + x*{qp} + {8*k}));")
    a("  }")
    a("}")
    return "\n".join(t)



def emit_blocks(P, blocks, fname, args, ldexpr, stexpr, pre=None):
    """blocks: list of (in_slots, X_outputs, out_slots). Emits one function."""
    lines=[f"static void {fname}({args}){{"]
    if pre: lines += pre
    for bi,(ins,X,outs) in enumerate(blocks):
        invar={}
        for slot in ins:
            lines.append(f"VD r{bi}_{slot} = {ldexpr('r',slot)}; VD i{bi}_{slot} = {ldexpr('i',slot)};")
            invar[P.inp(('r',(bi,slot)))]=f"r{bi}_{slot}"
            invar[P.inp(('i',(bi,slot)))]=f"i{bi}_{slot}"
        body,outv = emit_body(P,X,invar,X)
        # rename collisions across blocks: emit_body names are v{id}, ids unique within P -> fine
        lines += body
        for (orr,oii),slot in zip(outv,outs):
            lines.append(stexpr(orr,oii,slot))
    lines.append("}")
    return "\n".join(lines)

def gen_4step64(qp):
    import gen_fft as G
    out=[]
    # stage 1: for r in 0..7: DFT8 of slots r+8t -> same slots
    P=G.Prog()
    blocks=[]
    for r in range(8):
        slots=[r+8*t for t in range(8)]
        xs=[(P.inp(('r',(r,sl))), P.inp(('i',(r,sl)))) for sl in slots]
        X=G.dft(P,8,xs)
        blocks.append((slots,X,slots))
    # blocks use bi=r in emit: emit_blocks uses bi=loop idx; ensure input names keyed (bi,slot): rebuild with bi index
    P=G.Prog(); blocks=[]
    for bi in range(8):
        slots=[bi+8*t for t in range(8)]
        xs=[(P.inp(('r',(bi,sl))), P.inp(('i',(bi,sl)))) for sl in slots]
        X=G.dft(P,8,xs)
        blocks.append((slots,X,slots))
    out.append(emit_blocks(P, blocks, "fx8s1_64", "double*restrict xr, double*restrict xi",
        lambda c,sl: f"vload(x{c} + {sl*qp})",
        lambda r,i,sl: f"vstore(xr + {sl*qp}, {r}); vstore(xi + {sl*qp}, {i});"))
    # stage 2: for a: load slots 8a+r (=Y_r[a]), twiddle w64^{ra}, DFT8 -> out slots a+8b
    P=G.Prog(); blocks=[]
    for a in range(8):
        slots=[8*a+r for r in range(8)]
        xs=[(P.inp(('r',(a,sl))), P.inp(('i',(a,sl)))) for sl in slots]
        ts=[]
        for r in range(8):
            cr,ci = G.twid(64, r*a)
            ts.append(G.cmulc(P, cr, ci, xs[r]))
        X=G.dft(P,8,ts)
        blocks.append((slots,X,[a+8*b for b in range(8)]))
    out.append(emit_blocks(P, blocks, "fx8s2_64", "const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di",
        lambda c,sl: f"vload(s{c} + {sl*qp})",
        lambda r,i,sl: f"vstore(dr + {sl*qp}, {r}); vstore(di + {sl*qp}, {i});"))
    return "\n".join(out)


def gen_rows_notout(L, fname, zp, P, xs, X):
    """rows codelet: transposed loads, body, UNtransposed stores to 2nd buffer [z][x-chunk]."""
    nb = (L + 7)//8
    lines = [f"static void {fname}(const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di){{"]
    invar = {}
    for comp, base in (("r","sr"), ("i","si")):
        for jb in range(nb):
            for r in range(8):
                lines.append(f"VD {comp}a{jb}_{r} = vload({base} + {r*zp + jb*8});")
            outs = ",".join(f"{comp}{jb*8+t}" for t in range(8))
            lines.append("VD " + outs + ";")
            ins = ",".join(f"{comp}a{jb}_{r}" for r in range(8))
            lines.append(f"TP8({outs},{ins});")
    for j in range(L):
        invar[P.inp(('r',j))] = f"r{j}"
        invar[P.inp(('i',j))] = f"i{j}"
    body, outv = emit_body(P, X, invar, X)
    lines += body
    for j in range(L):
        lines.append(f"vstore(dr + {j*zp}, {outv[j][0]}); vstore(di + {j*zp}, {outv[j][1]});")
    lines.append("}")
    return "\n".join(lines)

def gen_normstoreT(L, zp, yp, zcw, qp):
    """normstore reading transposed Q2 [z][x]; handles partial x-chunks."""
    zg=(L+7)//8
    rem = L - (zg-1)*8   # rows in last chunk (1..8)
    t=[]; a=t.append
    a(f"static void normstoreT_{L}(const double*restrict cr, const double*restrict ci, double*restrict vr, double*restrict vi){{")
    a(f"  const double* qr = Q2R_{L}; const double* qi = Q2I_{L};")
    def chunk(xbexpr, nrows, tag):
        for jb in range(zg):
            a(f"    {{ // {tag} block jb={jb}")
            for tt in range(8):
                a(f"      VD qa{tt} = vload(qr + {(jb*8+tt)*qp} + {xbexpr}); VD qb{tt} = vload(qi + {(jb*8+tt)*qp} + {xbexpr});")
            a(f"      VD xr0,xr1,xr2,xr3,xr4,xr5,xr6,xr7; TP8(xr0,xr1,xr2,xr3,xr4,xr5,xr6,xr7,qa0,qa1,qa2,qa3,qa4,qa5,qa6,qa7);")
            a(f"      VD xi0,xi1,xi2,xi3,xi4,xi5,xi6,xi7; TP8(xi0,xi1,xi2,xi3,xi4,xi5,xi6,xi7,qb0,qb1,qb2,qb3,qb4,qb5,qb6,qb7);")
            for tt in range(nrows):
                a(f"      {{ double* dr = vr + ({xbexpr}+{tt})*{yp*zp}; double* di = vi + ({xbexpr}+{tt})*{yp*zp};")
                a(f"        __builtin_prefetch(cr + ({xbexpr}+{tt})*{zcw} + {2*zcw} + {jb*8}, 0, 3); __builtin_prefetch(ci + ({xbexpr}+{tt})*{zcw} + {2*zcw} + {jb*8}, 0, 3);")
                a(f"        __builtin_prefetch(dr + {zp} + {jb*8}, 1, 3); __builtin_prefetch(di + {zp} + {jb*8}, 1, 3);")
                a(f"        VD zr = xr{tt} + vload(cr + ({xbexpr}+{tt})*{zcw} + {jb*8});")
                a(f"        VD zi = xi{tt} + vload(ci + ({xbexpr}+{tt})*{zcw} + {jb*8});")
                a(f"        VD sc = nscale(zr*zr+zi*zi);")
                a(f"        vstore(dr + {jb*8}, zr*sc); vstore(di + {jb*8}, zi*sc); }}")
            a("    }")
    nfull = zg if rem == 8 else zg-1
    if nfull > 0:
        a(f"  for(long xc=0;xc<{nfull};xc++){{")
        a(f"    const size_t xb = xc*8;")
        chunk("xb", 8, "full")
        a("  }")
    if rem < 8:
        a(f"  {{")
        chunk(f"{(zg-1)*8}", rem, "partial")
        a("  }")
    a("}")
    return "\n".join(t)

def gen_fused_small(L):
    """L in {6,8}: per-plane fused B-path codelets (all lines in one call)."""
    import gen_fft as G
    out=[]
    # bzP: all y rows of one x-plane: block y: loads (y*L*8 + j*8), in-place
    P=G.Prog(); blocks=[]
    for y in range(L):
        slots=[y*L*8 + j*8 for j in range(L)]
        xs=[(P.inp(('r',(y,sl))), P.inp(('i',(y,sl)))) for sl in slots]
        X=G.dft(P,L,xs)
        blocks.append((slots,X,slots))
    out.append(emit_blocks(P, blocks, f"bzP_{L}", "double*restrict xr, double*restrict xi",
        lambda c,sl: f"vload(x{c} + {sl})",
        lambda r,i,sl: f"vstore(xr + {sl}, {r}); vstore(xi + {sl}, {i});"))
    # byP: all z of one x-plane: block z: loads (j*L*8 + z*8), in-place
    P=G.Prog(); blocks=[]
    for z in range(L):
        slots=[j*L*8 + z*8 for j in range(L)]
        xs=[(P.inp(('r',(z,sl))), P.inp(('i',(z,sl)))) for sl in slots]
        X=G.dft(P,L,xs)
        blocks.append((slots,X,slots))
    out.append(emit_blocks(P, blocks, f"byP_{L}", "double*restrict xr, double*restrict xi",
        lambda c,sl: f"vload(x{c} + {sl})",
        lambda r,i,sl: f"vstore(xr + {sl}, {r}); vstore(xi + {sl}, {i});"))
    # bxqP: same index pattern as byP (x along j at stride L*8), on Q
    P=G.Prog(); blocks=[]
    for z in range(L):
        slots=[j*L*8 + z*8 for j in range(L)]
        xs=[(P.inp(('r',(z,sl))), P.inp(('i',(z,sl)))) for sl in slots]
        X=G.dft(P,L,xs)
        blocks.append((slots,X,slots))
    out.append(emit_blocks(P, blocks, f"bxqP_{L}", "double*restrict xr, double*restrict xi",
        lambda c,sl: f"vload(x{c} + {sl})",
        lambda r,i,sl: f"vstore(xr + {sl}, {r}); vstore(xi + {sl}, {i});"))
    return "\n".join(out)

def driver(L):
    zp, yp = PADS[L]
    yc = (L+7)//8
    zg = (L+7)//8
    L3 = L*L*L
    vol = L*yp*zp
    pl  = yc*8*zp
    useb = L in BPATH
    zcw = zg*8
    qp = zcw + (8 if (zcw//8) % 8 == 0 else 0)
    t = []
    a = t.append
    a(f"/* ================= size {L} ================= */")
    a(f"static double *XR_{L},*XI_{L},*CR_{L},*CI_{L},*PR_{L},*PI_{L},*QR_{L},*QI_{L},*Q2R_{L},*Q2I_{L};")
    if useb:
        a(f"static double *BR_{L},*BI_{L},*BCR_{L},*BCI_{L},*BQR_{L},*BQI_{L};")
    a(gen_copy(L, zp, yp))
    a(f"static void convin_{L}(const double*restrict src, double*restrict vr, double*restrict vi){{")
    a(f"  for(long x=0;x<{L};x++) for(long y=0;y<{L};y++)")
    a(f"    row_deint(src + 2*((x*{L}+y)*{L}), vr + (x*{yp}+y)*{zp}, vi + (x*{yp}+y)*{zp}, {L});")
    a("}")
    a(f"static void convc_{L}(const double*restrict src, double*restrict cr, double*restrict ci){{")
    a(f"  for(long y=0;y<{L};y++) for(long x=0;x<{L};x++){{")
    a(f"    __builtin_prefetch(src + 2*((x*{L}+y)*{L}) + {4*2*L*L}, 0, 2);")
    a(f"    row_deint(src + 2*((x*{L}+y)*{L}), cr + (y*{L}+x)*{zcw}, ci + (y*{L}+x)*{zcw}, {L});")
    a("  }")
    a("}")
    a(f"static void convout_{L}(const double*restrict vr, const double*restrict vi, double*restrict dst){{")
    a(f"  for(long x=0;x<{L};x++) for(long y=0;y<{L};y++)")
    a(f"    row_intl(vr + (x*{yp}+y)*{zp}, vi + (x*{yp}+y)*{zp}, dst + 2*((x*{L}+y)*{L}), {L});")
    a("}")
    # one iteration, T path
    a(f"static void iterT_{L}(void){{")
    a(f"  for(long x=0;x<{L};x++){{")
    a(f"    const size_t pb = (size_t)x*{yp*zp};")
    if L in TOUCH_T:
        a(f"    if(x==0){{ const double* tr = XR_{L}; const double* ti = XI_{L};")
        a(f"      for(long k=0;k<{L*zp};k+=8){{ __builtin_prefetch(tr+k,0,3); __builtin_prefetch(ti+k,0,3); }} }}")
    a(f"    for(long z=0;z<{zg};z++) fyi_{L}(XR_{L}+pb+z*8, XI_{L}+pb+z*8);")
    a("  }")
    a(f"  for(long y=0;y<{L};y++){{")
    a(f"    if(y==0){{ for(long x=0;x<{L};x++){{ const double* tr=XR_{L}+x*{yp*zp}; const double* ti=XI_{L}+x*{yp*zp};")
    a(f"      for(long k=0;k<{zg*8};k+=8){{ __builtin_prefetch(tr+k,0,3); __builtin_prefetch(ti+k,0,3); }} }} }}")
    a(f"    for(long z=0;z<{zg};z++)")
    a(f"      fxqd_{L}(XR_{L}+(size_t)y*{zp}+z*8, XI_{L}+(size_t)y*{zp}+z*8, QR_{L}+z*8, QI_{L}+z*8);")
    if L in TZQT:
        a(f"    for(long xc=0;xc<{zg};xc++)")
        a(f"      fzqT_{L}(QR_{L}+xc*{8*qp}, QI_{L}+xc*{8*qp}, Q2R_{L}+xc*8, Q2I_{L}+xc*8);")
        a(f"    normstoreT_{L}(CR_{L}+(size_t)y*{L*zcw}, CI_{L}+(size_t)y*{L*zcw}, XR_{L}+(size_t)y*{zp}, XI_{L}+(size_t)y*{zp});")
    elif L % 8:
        a(f"    for(long xc=0;xc<{zg-1};xc++)")
        a(f"      fzq_{L}(QR_{L}+xc*{8*qp}, QI_{L}+xc*{8*qp});")
        a(f"    fzqL_{L}(QR_{L}+{(zg-1)*8*qp}, QI_{L}+{(zg-1)*8*qp});")
        a(f"    normstore_{L}(CR_{L}+(size_t)y*{L*zcw}, CI_{L}+(size_t)y*{L*zcw}, XR_{L}+(size_t)y*{zp}, XI_{L}+(size_t)y*{zp});")
    else:
        a(f"    for(long xc=0;xc<{zg};xc++)")
        a(f"      fzq_{L}(QR_{L}+xc*{8*qp}, QI_{L}+xc*{8*qp});")
        a(f"    normstore_{L}(CR_{L}+(size_t)y*{L*zcw}, CI_{L}+(size_t)y*{L*zcw}, XR_{L}+(size_t)y*{zp}, XI_{L}+(size_t)y*{zp});")
    a("  }")
    a("}")
    a(f"static void runT_{L}(const double* xin, const double* cin, double* o1, double* om, long m){{")
    a(f"  convin_{L}(xin, XR_{L}, XI_{L});")
    a(f"  convc_{L}(cin, CR_{L}, CI_{L});")
    a(f"  if(m<=0){{ convout_{L}(XR_{L}, XI_{L}, o1); convout_{L}(XR_{L}, XI_{L}, om); return; }}")
    a(f"  for(long it=0; it<m; it++){{")
    a(f"    iterT_{L}();")
    a(f"    if(it==0) convout_{L}(XR_{L}, XI_{L}, o1);")
    a("  }")
    a(f"  convout_{L}(XR_{L}, XI_{L}, om);")
    a("}")
    if useb:
        a(f"static void bconvin_{L}(const double* xin, double*restrict vr, double*restrict vi, long vc){{")
        a(f"  const double* s[8]; for(int v=0;v<8;v++) s[v]=xin+(size_t)(v<vc?v:vc-1)*{2*L3};")
        a(f"  long idx=0;")
        a(f"  for(; idx+4<=({L3}/4)*4; idx+=4){{")
        a(f"    __m512d m0=_mm512_loadu_pd(s[0]+2*idx), m1=_mm512_loadu_pd(s[1]+2*idx), m2=_mm512_loadu_pd(s[2]+2*idx), m3=_mm512_loadu_pd(s[3]+2*idx);")
        a(f"    __m512d m4=_mm512_loadu_pd(s[4]+2*idx), m5=_mm512_loadu_pd(s[5]+2*idx), m6=_mm512_loadu_pd(s[6]+2*idx), m7=_mm512_loadu_pd(s[7]+2*idx);")
        a(f"    VD c0,c1,c2,c3,c4,c5,c6,c7;")
        a(f"    TP8(c0,c1,c2,c3,c4,c5,c6,c7,(VD)m0,(VD)m1,(VD)m2,(VD)m3,(VD)m4,(VD)m5,(VD)m6,(VD)m7);")
        a(f"    vstore(vr+idx*8,c0); vstore(vi+idx*8,c1); vstore(vr+idx*8+8,c2); vstore(vi+idx*8+8,c3);")
        a(f"    vstore(vr+idx*8+16,c4); vstore(vi+idx*8+16,c5); vstore(vr+idx*8+24,c6); vstore(vi+idx*8+24,c7);")
        a("  }")
        if L3 % 4:
            a(f"  for(; idx<{L3}; idx++) for(int v=0;v<8;v++){{ vr[idx*8+v]=s[v][2*idx]; vi[idx*8+v]=s[v][2*idx+1]; }}")
        a("}")
        a(f"static void bconvout_{L}(const double*restrict vr, const double*restrict vi, double* dst, long vc){{")
        a(f"  static double scratch[{2*L3}];")
        a(f"  double* d[8]; for(int v=0;v<8;v++) d[v]= v<vc ? dst+(size_t)v*{2*L3} : scratch;")
        a(f"  long idx=0;")
        a(f"  for(; idx+4<=({L3}/4)*4; idx+=4){{")
        a(f"    VD c0,c1,c2,c3,c4,c5,c6,c7;")
        a(f"    TP8(c0,c1,c2,c3,c4,c5,c6,c7,vload(vr+idx*8),vload(vi+idx*8),vload(vr+idx*8+8),vload(vi+idx*8+8),vload(vr+idx*8+16),vload(vi+idx*8+16),vload(vr+idx*8+24),vload(vi+idx*8+24));")
        a(f"    _mm512_storeu_pd(d[0]+2*idx,(__m512d)c0); _mm512_storeu_pd(d[1]+2*idx,(__m512d)c1); _mm512_storeu_pd(d[2]+2*idx,(__m512d)c2); _mm512_storeu_pd(d[3]+2*idx,(__m512d)c3);")
        a(f"    _mm512_storeu_pd(d[4]+2*idx,(__m512d)c4); _mm512_storeu_pd(d[5]+2*idx,(__m512d)c5); _mm512_storeu_pd(d[6]+2*idx,(__m512d)c6); _mm512_storeu_pd(d[7]+2*idx,(__m512d)c7);")
        a("  }")
        if L3 % 4:
            a(f"  for(; idx<{L3}; idx++) for(int v=0;v<8;v++){{ d[v][2*idx]=vr[idx*8+v]; d[v][2*idx+1]=vi[idx*8+v]; }}")
        a("}")
        a(f"static void bcopyin_{L}(const double*restrict vr, const double*restrict vi, double*restrict qr, double*restrict qi){{")
        a(f"  for(long x=0;x<{L};x++){{")
        a(f"    const double* sr=vr+(size_t)x*{L*L*8}; const double* si=vi+(size_t)x*{L*L*8};")
        a(f"    for(long k=0;k<{L};k++){{ __builtin_prefetch(sr+{2*L*L*8}+k*8,0,3); __builtin_prefetch(si+{2*L*L*8}+k*8,0,3); vstore(qr+x*{L*8}+k*8, vload(sr+k*8)); vstore(qi+x*{L*8}+k*8, vload(si+k*8)); }}")
        a("  }")
        a("}")
        a(f"static void bnormstore_{L}(const double*restrict cr, const double*restrict ci, double*restrict vr, double*restrict vi){{")
        a(f"  const double* qr = BQR_{L}; const double* qi = BQI_{L};")
        a(f"  for(long x=0;x<{L};x++){{")
        a(f"    double* dr = vr + (size_t)x*{L*L*8}; double* di = vi + (size_t)x*{L*L*8};")
        a(f"    long k=0;")
        a(f"    for(;k<{(L//4)*32};k+=32){{")
        pfc = 2*L*8
        a(f"      __builtin_prefetch(cr + x*{L*8} + {pfc} + k, 0, 3); __builtin_prefetch(ci + x*{L*8} + {pfc} + k, 0, 3);")
        a(f"      __builtin_prefetch(cr + x*{L*8} + {pfc} + k + 16, 0, 3); __builtin_prefetch(ci + x*{L*8} + {pfc} + k + 16, 0, 3);")
        a(f"      __builtin_prefetch(dr + {L*8} + k, 1, 3); __builtin_prefetch(di + {L*8} + k, 1, 3);")
        a(f"      __builtin_prefetch(dr + {L*L*8} + k + 16, 1, 3); __builtin_prefetch(di + {L*L*8} + k + 16, 1, 3);")
        a(f"      VD zr0 = vload(qr+x*{L*8}+k   ) + vload(cr+x*{L*8}+k   ), zi0 = vload(qi+x*{L*8}+k   ) + vload(ci+x*{L*8}+k   );")
        a(f"      VD zr1 = vload(qr+x*{L*8}+k+8 ) + vload(cr+x*{L*8}+k+8 ), zi1 = vload(qi+x*{L*8}+k+8 ) + vload(ci+x*{L*8}+k+8 );")
        a(f"      VD zr2 = vload(qr+x*{L*8}+k+16) + vload(cr+x*{L*8}+k+16), zi2 = vload(qi+x*{L*8}+k+16) + vload(ci+x*{L*8}+k+16);")
        a(f"      VD zr3 = vload(qr+x*{L*8}+k+24) + vload(cr+x*{L*8}+k+24), zi3 = vload(qi+x*{L*8}+k+24) + vload(ci+x*{L*8}+k+24);")
        a(f"      VD sc0 = nscale(zr0*zr0+zi0*zi0), sc1 = nscale(zr1*zr1+zi1*zi1), sc2 = nscale(zr2*zr2+zi2*zi2), sc3 = nscale(zr3*zr3+zi3*zi3);")
        a(f"      vstore(dr+k   , zr0*sc0); vstore(di+k   , zi0*sc0);")
        a(f"      vstore(dr+k+8 , zr1*sc1); vstore(di+k+8 , zi1*sc1);")
        a(f"      vstore(dr+k+16, zr2*sc2); vstore(di+k+16, zi2*sc2);")
        a(f"      vstore(dr+k+24, zr3*sc3); vstore(di+k+24, zi3*sc3);")
        a("    }")
        a(f"    for(;k<{L*8};k+=8){{")
        a(f"      VD zr = vload(qr+x*{L*8}+k) + vload(cr+x*{L*8}+k), zi = vload(qi+x*{L*8}+k) + vload(ci+x*{L*8}+k);")
        a(f"      VD sc = nscale(zr*zr+zi*zi);")
        a(f"      vstore(dr+k, zr*sc); vstore(di+k, zi*sc);")
        a("    }")
        a("  }")
        a("}")
        a(f"static void bcopyout_{L}(const double*restrict qr, const double*restrict qi, double*restrict vr, double*restrict vi){{")
        a(f"  for(long x=0;x<{L};x++){{")
        a(f"    double* dr=vr+(size_t)x*{L*L*8}; double* di=vi+(size_t)x*{L*L*8};")
        a(f"    for(long k=0;k<{L};k++){{ vstore(dr+k*8, vload(qr+x*{L*8}+k*8)); vstore(di+k*8, vload(qi+x*{L*8}+k*8)); }}")
        a("  }")
        a("}")
        a(f"static void iterB_{L}(void){{")
        a(f"  for(long x=0;x<{L};x++){{")
        a(f"    const size_t pb=(size_t)x*{L*L*8};")
        if L in (6,8):
            a(f"    byP_{L}(BR_{L}+pb, BI_{L}+pb);")
        else:
            a(f"    for(long z=0;z<{L};z++) by_{L}(BR_{L}+pb+z*8, BI_{L}+pb+z*8);")
        a("  }")
        a(f"  for(long y=0;y<{L};y++){{")
        if L in (6,8):
            a(f"    bcopyin_{L}(BR_{L}+(size_t)y*{L*8}, BI_{L}+(size_t)y*{L*8}, BQR_{L}, BQI_{L});")
            a(f"    bxqP_{L}(BQR_{L}, BQI_{L});")
            a(f"    bzP_{L}(BQR_{L}, BQI_{L});")
        else:
            a(f"    if(y==0){{ for(long x=0;x<{L};x++){{ const double* tr=BR_{L}+(size_t)x*{L*L*8}; const double* ti=BI_{L}+(size_t)x*{L*L*8};")
            a(f"      for(long k=0;k<{L*8};k+=8){{ __builtin_prefetch(tr+k,0,3); __builtin_prefetch(ti+k,0,3); }} }} }}")
            a(f"    for(long z=0;z<{L};z++)")
            a(f"      bxqd_{L}(BR_{L}+(size_t)y*{L*8}+z*8, BI_{L}+(size_t)y*{L*8}+z*8, BQR_{L}+z*8, BQI_{L}+z*8);")
            a(f"    for(long x=0;x<{L};x++)")
            a(f"      bz_{L}(BQR_{L}+(size_t)x*{L*8}, BQI_{L}+(size_t)x*{L*8});")
        a(f"    bnormstore_{L}(BCR_{L}+(size_t)y*{L*L*8}, BCI_{L}+(size_t)y*{L*L*8}, BR_{L}+(size_t)y*{L*8}, BI_{L}+(size_t)y*{L*8});")
        a("  }")
        a("}")
        a(f"static void bconvc_{L}(const double* cin, double*restrict vr, double*restrict vi, long vc){{")
        a(f"  const double* s[8]; for(int v=0;v<8;v++) s[v]=cin+(size_t)(v<vc?v:vc-1)*{2*L3};")
        a(f"  for(long x=0;x<{L};x++)for(long y=0;y<{L};y++){{")
        a(f"    const size_t db=((size_t)y*{L}+x)*{L*8}; const size_t sb=2*((x*{L}+y)*{L});")
        a(f"    long z=0;")
        a(f"    for(; z+4<={L}; z+=4){{")
        a(f"      __m512d m0=_mm512_loadu_pd(s[0]+sb+2*z), m1=_mm512_loadu_pd(s[1]+sb+2*z), m2=_mm512_loadu_pd(s[2]+sb+2*z), m3=_mm512_loadu_pd(s[3]+sb+2*z);")
        a(f"      __m512d m4=_mm512_loadu_pd(s[4]+sb+2*z), m5=_mm512_loadu_pd(s[5]+sb+2*z), m6=_mm512_loadu_pd(s[6]+sb+2*z), m7=_mm512_loadu_pd(s[7]+sb+2*z);")
        a(f"      VD c0,c1,c2,c3,c4,c5,c6,c7;")
        a(f"      TP8(c0,c1,c2,c3,c4,c5,c6,c7,(VD)m0,(VD)m1,(VD)m2,(VD)m3,(VD)m4,(VD)m5,(VD)m6,(VD)m7);")
        a(f"      vstore(vr+db+z*8,c0); vstore(vi+db+z*8,c1); vstore(vr+db+z*8+8,c2); vstore(vi+db+z*8+8,c3);")
        a(f"      vstore(vr+db+z*8+16,c4); vstore(vi+db+z*8+16,c5); vstore(vr+db+z*8+24,c6); vstore(vi+db+z*8+24,c7);")
        a("    }")
        a(f"    for(; z<{L}; z++) for(int v=0;v<8;v++){{ vr[db+z*8+v]=s[v][sb+2*z]; vi[db+z*8+v]=s[v][sb+2*z+1]; }}")
        a("  }")
        a("}")
        a(f"static void runB_{L}(const double* xin, const double* cin, double* o1, double* om, long m, long vc){{")
        a(f"  bconvin_{L}(xin, BR_{L}, BI_{L}, vc);")
        a(f"  bconvc_{L}(cin, BCR_{L}, BCI_{L}, vc);")
        a(f"  if(m<=0){{ bconvout_{L}(BR_{L}, BI_{L}, o1, vc); bconvout_{L}(BR_{L}, BI_{L}, om, vc); return; }}")
        a(f"  for(long it=0; it<m; it++){{")
        a(f"    iterB_{L}();")
        a(f"    if(it==0) bconvout_{L}(BR_{L}, BI_{L}, o1, vc);")
        a("  }")
        a(f"  bconvout_{L}(BR_{L}, BI_{L}, om, vc);")
        a("}")
    a(f"static void run_{L}(long B, long m, const double* xin, const double* cin, double* o1, double* om){{")
    a(f"  long b=0;")
    if useb:
        thr = BTHR.get(L, 9)
        a(f"  if(useB[{SIZES.index(L)}]){{")
        a(f"    for(; b+8<=B; b+=8)")
        a(f"      runB_{L}(xin+(size_t)b*{2*L3}, cin+(size_t)b*{2*L3}, o1+(size_t)b*{2*L3}, om+(size_t)b*{2*L3}, m, 8);")
        a(f"    if(B-b >= {thr}){{")
        a(f"      runB_{L}(xin+(size_t)b*{2*L3}, cin+(size_t)b*{2*L3}, o1+(size_t)b*{2*L3}, om+(size_t)b*{2*L3}, m, B-b);")
        a(f"      b = B;")
        a("    }")
        a("  }")
    a(f"  for(; b<B; b++)")
    a(f"    runT_{L}(xin+(size_t)b*{2*L3}, cin+(size_t)b*{2*L3}, o1+(size_t)b*{2*L3}, om+(size_t)b*{2*L3}, m);")
    a("}")
    return "\n".join(t)

def main():
    parts = [HEADER]
    parts.append("static int useB[8] = {1,1,1,1,1,1,1,0}; /* per-size lane-batch enable */")
    parts.append("void set_useB(const long* f){ for(int i=0;i<8;i++) useB[i]=(int)f[i]; }")
    for L in SIZES:
        zp, yp = PADS[L]
        P, xs, X = build(L)
        parts.append(f"/* ---- codelets L={L} (zp={zp}, yp={yp}) ---- */")
        zcw = ((L+7)//8)*8
        qp = zcw + (8 if (zcw//8) % 8 == 0 else 0)
        if L in TZQT:
            parts.append(gen_rows_notout(L, f"fzqT_{L}", qp, P, xs, X))
        else:
            parts.append(gen_rows(L, f"fzq_{L}", qp, P, xs, X, inplace=True))
            if L % 8:
                parts.append(gen_rows(L, f"fzqL_{L}", qp, P, xs, X, inplace=True, nrows=(L % 8)))
        parts.append(gen_strided(L, f"fyi_{L}", zp, False, True, P, xs, X, pf=(yp*zp if L in TOUCH_T else None)))
        parts.append(gen_strided(L, f"fxqd_{L}", yp*zp, False, False, P, xs, X, stride_st=qp))

        if L in BPATH:
            if L in (6,8):
                parts.append(gen_fused_small(L))
            else:
                parts.append(gen_strided(L, f"bz_{L}", 8, False, True, P, xs, X))
                parts.append(gen_strided(L, f"by_{L}", 8*L, False, True, P, xs, X, pf=(L*L*8 if L in TOUCH_B else None)))
                parts.append(gen_strided(L, f"bxqd_{L}", 8*L*L, False, False, P, xs, X, stride_st=8*L))
    for L in SIZES:
        parts.append(driver(L))
    # init + dispatch
    init = ["int impl_init(void){", "  size_t total=0; size_t off[64]; int k=0;"]
    allocs = []
    for L in SIZES:
        zp, yp = PADS[L]
        vol = L*yp*zp; pl = ((L+7)//8)*8*zp; zcw = ((L+7)//8)*8; qp = zcw + (8 if (zcw//8) % 8 == 0 else 0)
        for nm in (f"XR_{L}",f"XI_{L}"):
            allocs.append((nm, vol))
        for nm in (f"CR_{L}",f"CI_{L}"):
            allocs.append((nm, L*L*zcw))
        for nm in (f"PR_{L}",f"PI_{L}"):
            allocs.append((nm, pl))
        for nm in (f"QR_{L}",f"QI_{L}",f"Q2R_{L}",f"Q2I_{L}"):
            allocs.append((nm, zcw*qp))
        if L in BPATH:
            for nm in (f"BR_{L}",f"BI_{L}",f"BCR_{L}",f"BCI_{L}",):
                allocs.append((nm, L*L*L*8))
            for nm in (f"BQR_{L}",f"BQI_{L}"):
                allocs.append((nm, L*L*8))
    for nm, sz in allocs:
        init.append(f"  off[k++] = total; total += {((sz*8 + 63)//64*64)};")
    init.append("  size_t align = 1<<21;")
    init.append("  total = (total + align - 1) & ~(align-1);")
    init.append("  void* base = mmap(0, total + align, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);")
    init.append("  if(base == MAP_FAILED) return -1;")
    init.append("  char* p = (char*)(((size_t)base + align - 1) & ~(align-1));")
    init.append("  madvise(p, total, MADV_HUGEPAGE);")
    init.append("  memset(p, 0, total);")
    init.append("  k=0;")
    for nm, sz in allocs:
        init.append(f"  {nm} = (double*)(p + off[k++]);")
    init.append("  return 0;")
    init.append("}")
    parts.append("\n".join(init))
    disp = ["void run_size(long L, long B, long m, const double* x, const double* c, double* o1, double* om){",
            "  switch(L){"]
    for L in SIZES:
        disp.append(f"    case {L}: run_{L}(B,m,x,c,o1,om); break;")
    disp.append("  }")
    disp.append("}")
    parts.append("\n".join(disp))
    src = "\n".join(parts) + "\n"
    with open("implementation.c","w") as f:
        f.write(src)
    print(f"wrote implementation.c: {len(src)} bytes, {src.count(chr(10))} lines")

if __name__ == '__main__':
    main()
