# SoA batch-lane engines for primes 13/17/23 (8 volumes in zmm lanes).
# State: RE[L^3][8], IM[L^3][8]. Steps: z-pass, y-pass, x-pass(+c+map), all in-place.
# Prime DFT: folded symmetric, k-tiled accumulators, register-resident constants
# (broadcast once per tile from table), FMA sign-baked.
import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
from gen_fg import emit_map, emit_transpose8
import numpy as np
import os
EMB = os.environ.get('EMB','0')=='1'

PI = np.longdouble('3.14159265358979323846264338327950288')

def prime_engine(L, K):
    h = (L-1)//2
    em = Emitter()
    # constant tables: cos/sin magnitudes with signs for (j,k): c[j][k] = cos(2pi jk/L)
    cs = np.zeros((h+1, h+1)); sn = np.zeros((h+1, h+1))
    for j in range(1, h+1):
        for k in range(1, h+1):
            ang = 2*PI*np.longdouble((j*k) % L)/np.longdouble(L)
            cs[j][k] = float(np.cos(ang)); sn[j][k] = float(np.sin(ang))
    em(f'''
// ============ L = {L} prime SoA engine ============
static double *S{L}_re, *S{L}_im, *C{L}_re, *C{L}_im;
static double UV{L}[{4*((L-1)//2)*8}] __attribute__((aligned(64)));
void init_{L}(void){{
  S{L}_re = huge_alloc({L**3}*8*8); S{L}_im = huge_alloc({L**3}*8*8);
  C{L}_re = huge_alloc({L**3}*8*8); C{L}_im = huge_alloc({L**3}*8*8);
}}''')
    # twiddle tables in consumption order per k-tile: for tile t, for j, the K (c,s) pairs
    tiles = [list(range(1+K*t, min(1+K*(t+1), h+1))) for t in range((h+K-1)//K)]
    tbl = []
    for t, ks in enumerate(tiles):
        for j in range(1, h+1):
            for k in ks:
                tbl += [cs[j][k], sn[j][k]]
    em(f'static const double W{L}[{len(tbl)}] __attribute__((aligned(64))) = {{')
    em(','.join(chex(v) for v in tbl) + '};')

    # ---- pencil codelet generator ----
    def pencil(pe, C, stride, sink):
        """stride in rows. sink(pe, k, re_name, im_name) emits the consume/store for output k."""
        pe(f'const double* w = W{L};')
        x0r = T(pe, f'_mm512_load_pd(br)')
        x0i = T(pe, f'_mm512_load_pd(bi)')
        dcr, dci = x0r, x0i
        for t, ks in enumerate(tiles):
            accs = {}
            for k in ks:
                accs[k] = (f'a{t}_{k}r', f'a{t}_{k}i', f'b{t}_{k}r', f'b{t}_{k}i')
                pe(f'__m512d a{t}_{k}r = {x0r}, a{t}_{k}i = {x0i}, b{t}_{k}r = _mm512_setzero_pd(), b{t}_{k}i = _mm512_setzero_pd();')
            for j in range(1, h+1):
                if t == 0 and stride > 1:
                    pe(f'_mm_prefetch((const char*)(br+{j*stride}*8+16), _MM_HINT_T0);')
                    pe(f'_mm_prefetch((const char*)(bi+{j*stride}*8+16), _MM_HINT_T0);')
                    pe(f'_mm_prefetch((const char*)(br+{(L-j)*stride}*8+16), _MM_HINT_T0);')
                    pe(f'_mm_prefetch((const char*)(bi+{(L-j)*stride}*8+16), _MM_HINT_T0);')
                if t == 0:
                    xjr = T(pe, f'_mm512_load_pd(br+{j*stride}*8)')
                    xji = T(pe, f'_mm512_load_pd(bi+{j*stride}*8)')
                    yjr = T(pe, f'_mm512_load_pd(br+{(L-j)*stride}*8)')
                    yji = T(pe, f'_mm512_load_pd(bi+{(L-j)*stride}*8)')
                    ur = ADD(pe, xjr, yjr); ui = ADD(pe, xji, yji)
                    vr = SUB(pe, xjr, yjr); vi = SUB(pe, xji, yji)
                    if len(tiles) > 1:
                        pe(f'_mm512_store_pd(UV{L}+{(4*(j-1)+0)*8}, {ur});')
                        pe(f'_mm512_store_pd(UV{L}+{(4*(j-1)+1)*8}, {ui});')
                        pe(f'_mm512_store_pd(UV{L}+{(4*(j-1)+2)*8}, {vr});')
                        pe(f'_mm512_store_pd(UV{L}+{(4*(j-1)+3)*8}, {vi});')
                    dcr = ADD(pe, dcr, ur); dci = ADD(pe, dci, ui)
                else:
                    ur = T(pe, f'_mm512_load_pd(UV{L}+{(4*(j-1)+0)*8})')
                    ui = T(pe, f'_mm512_load_pd(UV{L}+{(4*(j-1)+1)*8})')
                    vr = T(pe, f'_mm512_load_pd(UV{L}+{(4*(j-1)+2)*8})')
                    vi = T(pe, f'_mm512_load_pd(UV{L}+{(4*(j-1)+3)*8})')
                for k in ks:
                    pos = (sum(len(tiles[tt]) for tt in range(t))*h + (j-1)*len(ks) + ks.index(k))*2
                    ar, ai, brr, bri = accs[k]
                    if EMB:
                        pe(f'FMAB({ar}, {ur}, w[{pos}]);')
                        pe(f'FMAB({ai}, {ui}, w[{pos}]);')
                        pe(f'FMAB({brr}, {vi}, w[{pos}+1]);')
                        pe(f'FMAB({bri}, {vr}, w[{pos}+1]);')
                    else:
                        cv = T(pe, f'bcastv(w+{pos})')
                        sv = T(pe, f'bcastv(w+{pos}+1)')
                        pe(f'{ar} = _mm512_fmadd_pd({cv},{ur},{ar});')
                        pe(f'{ai} = _mm512_fmadd_pd({cv},{ui},{ai});')
                        pe(f'{brr} = _mm512_fmadd_pd({sv},{vi},{brr});')
                        pe(f'{bri} = _mm512_fmadd_pd({sv},{vr},{bri});')
            if t == 0:
                sink(pe, 0, dcr, dci)
            for k in ks:
                ar, ai, brr, bri = accs[k]
                Xkr = ADD(pe, ar, brr); Xki = SUB(pe, ai, bri)
                Ykr = SUB(pe, ar, brr); Yki = ADD(pe, ai, bri)
                sink(pe, k, Xkr, Xki)
                sink(pe, L-k, Ykr, Yki)

    CC = Consts()
    body = Emitter()
    # z-pass: stride 1; pencil base = (x*L+y)*L
    body(f'static void pass_z_{L}(void){{')
    body('%CONSTS%')
    body(f'for(int o=0;o<{L*L};o++){{')
    body(f'double* br = S{L}_re + (size_t)o*{L}*8; double* bi = S{L}_im + (size_t)o*{L}*8;')
    pe = Emitter()
    def sink_z(p, k, r, i):
        p(f'_mm512_store_pd(br+{k}*8, {r});')
        p(f'_mm512_store_pd(bi+{k}*8, {i});')
    pencil(pe, CC, 1, sink_z)
    body(pe.out()); body('}'); body('}')
    # y-pass: stride L; outer (x, z)
    body(f'static void pass_y_{L}(void){{')
    body('%CONSTS%')
    body(f'for(int x=0;x<{L};x++)for(int z=0;z<{L};z++){{')
    body(f'double* br = S{L}_re + ((size_t)x*{L*L}+z)*8; double* bi = S{L}_im + ((size_t)x*{L*L}+z)*8;')
    pe = Emitter()
    def sink_y(p, k, r, i):
        p(f'_mm512_store_pd(br+{k*L}*8, {r});')
        p(f'_mm512_store_pd(bi+{k*L}*8, {i});')
    pencil(pe, CC, L, sink_y)
    body(pe.out()); body('}'); body('}')
    # x-pass + c + map: stride L^2; outer (y,z)
    body(f'static void pass_x_{L}(void){{')
    body('%CONSTS%')
    body(f'for(int o=0;o<{L*L};o++){{')
    body(f'double* br = S{L}_re + (size_t)o*8; double* bi = S{L}_im + (size_t)o*8;')
    body(f'const double* cr = C{L}_re + (size_t)o*8; const double* ci = C{L}_im + (size_t)o*8;')
    body(f'for(int pk=0;pk<{L};pk++){{ _mm_prefetch((const char*)(cr+pk*{L*L}*8+8), _MM_HINT_T0); _mm_prefetch((const char*)(ci+pk*{L*L}*8+8), _MM_HINT_T0); }}')
    pe = Emitter()
    def sink_x(p, k, r, i):
        zr = T(p, f'_mm512_add_pd({r}, _mm512_load_pd(cr+{k*L*L}*8))')
        zi = T(p, f'_mm512_add_pd({i}, _mm512_load_pd(ci+{k*L*L}*8))')
        p(f'__m512d fr{k}, fi{k};')
        emit_map(p, CC, zr, zi, f'fr{k}', f'fi{k}')
        p(f'_mm512_store_pd(br+{k*L*L}*8, fr{k});')
        p(f'_mm512_store_pd(bi+{k*L*L}*8, fi{k});')
    pencil(pe, CC, L*L, sink_x)
    body(pe.out()); body('}'); body('}')
    em(body.out().replace('%CONSTS%', CC.loads()))

    # conversions: 8-vol groups with zero-padded lanes
    em(f'''
static void convin_{L}(const double* x0, long nb){{
  // nb volumes (<=8) -> lanes; pad lanes zeroed
  for(int e=0;e<{L**3};e++){{
    for(long v=0;v<nb;v++){{ S{L}_re[e*8+v] = x0[v*{2*L**3}+2*e]; S{L}_im[e*8+v] = x0[v*{2*L**3}+2*e+1]; }}
    for(long v=nb;v<8;v++){{ S{L}_re[e*8+v]=0.0; S{L}_im[e*8+v]=0.0; }}
  }}
}}
static void convc_{L}(const double* c, long nb){{
  for(int e=0;e<{L**3};e++){{
    for(long v=0;v<nb;v++){{ C{L}_re[e*8+v] = c[v*{2*L**3}+2*e]; C{L}_im[e*8+v] = c[v*{2*L**3}+2*e+1]; }}
    for(long v=nb;v<8;v++){{ C{L}_re[e*8+v]=0.0; C{L}_im[e*8+v]=0.0; }}
  }}
}}
static void convout_{L}(double* out, long nb){{
  for(int e=0;e<{L**3};e++){{
    for(long v=0;v<nb;v++){{ out[v*{2*L**3}+2*e] = S{L}_re[e*8+v]; out[v*{2*L**3}+2*e+1] = S{L}_im[e*8+v]; }}
  }}
}}
uint64_t bench_{L}(int which, long reps){{
  uint64_t t0=__rdtsc();
  for(long r=0;r<reps;r++){{
    if(which==0) pass_z_{L}();
    else if(which==1) pass_y_{L}();
    else pass_x_{L}();
  }}
  return __rdtsc()-t0;
}}
void run_{L}(const double* x0, const double* c, double* one, double* fin, long B, long m){{
  for(long g=0;g<B;g+=8){{
    long nb = B-g < 8 ? B-g : 8;
    convin_{L}(x0 + g*{2*L**3}, nb);
    convc_{L}(c + g*{2*L**3}, nb);
    for(long s=0;s<m;s++){{
      pass_z_{L}(); pass_y_{L}(); pass_x_{L}();
      if(s==0) convout_{L}(one + g*{2*L**3}, nb);
    }}
    convout_{L}(fin + g*{2*L**3}, nb);
  }}
}}''')
    return em.out()

if __name__ == '__main__':
    import os
    PRELUDE = open('/tmp/g/impl_fg.c').read().split('// =================')[0]
    from genlib import BCASTV_DEF
    src = PRELUDE + '#include <math.h>\n#include <x86intrin.h>\n' + BCASTV_DEF + '''
#define FMAB(acc, mul, mem) __asm__("vfmadd231pd %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(mul), "m"(mem))
'''
    Ks = {13: int(os.environ.get('K13', 5)), 17: int(os.environ.get('K17', 5)), 23: int(os.environ.get('K23', 5))}
    for L in (13, 17, 23):
        src += prime_engine(L, Ks[L])
    open('/tmp/g/impl_p.c','w').write(src)
    print('wrote impl_p.c', len(src))
