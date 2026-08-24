cd /workdir/dev && cat > gen.py <<'PYEOF'
# Generates implementation.c
import codelets, kernels

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
SCHEME = {6:'soa', 8:'soa', 13:'soa', 17:'slab', 23:'slab', 36:'slab', 45:'slab', 64:'slab'}
def r8(x): return (x + 7) // 8 * 8

HDR = r'''// AUTO-GENERATED -- batched iterated 3D DFT specialized kernels
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
typedef double V __attribute__((vector_size(64), aligned(64)));
#define VC(x) ((V){(x),(x),(x),(x),(x),(x),(x),(x)})
static inline V vrsqrt14(V x){ return (V)_mm512_rsqrt14_pd((__m512d)x); }
static inline V vrcp14(V x){ return (V)_mm512_rcp14_pd((__m512d)x); }
static inline V vmaxv(V a, V b){ return (V)_mm512_max_pd((__m512d)a,(__m512d)b); }

// f = 1 / (1 + sqrt(s)), s = zr^2 + zi^2, via Newton (no divider unit)
static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  s = vmaxv(s, VC(2.2250738585072014e-308));
  V r = vrsqrt14(s);
  V h = s * VC(0.5);
  r = r * (VC(1.5) - h*r*r);
  r = r * (VC(1.5) - h*r*r);
  V t = s * r;              // ~= sqrt(s)
  V u = VC(1.0) + t;
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}
#define STPW(k, a, b_) do { long _k = (k); V zr = (a) + cr[_k*os_]; V zi = (b_) + ci[_k*os_]; \
  V f = pw_factor(zr, zi); yr[_k*os_] = zr*f; yi[_k*os_] = zi*f; } while (0)

// 8x8 double transpose
static inline __attribute__((always_inline)) void tr8x8(const V* a, V* b){
  V u0 = __builtin_shufflevector(a[0], a[1], 0, 8, 2, 10, 4, 12, 6, 14);
  V u1 = __builtin_shufflevector(a[0], a[1], 1, 9, 3, 11, 5, 13, 7, 15);
  V u2 = __builtin_shufflevector(a[2], a[3], 0, 8, 2, 10, 4, 12, 6, 14);
  V u3 = __builtin_shufflevector(a[2], a[3], 1, 9, 3, 11, 5, 13, 7, 15);
  V u4 = __builtin_shufflevector(a[4], a[5], 0, 8, 2, 10, 4, 12, 6, 14);
  V u5 = __builtin_shufflevector(a[4], a[5], 1, 9, 3, 11, 5, 13, 7, 15);
  V u6 = __builtin_shufflevector(a[6], a[7], 0, 8, 2, 10, 4, 12, 6, 14);
  V u7 = __builtin_shufflevector(a[6], a[7], 1, 9, 3, 11, 5, 13, 7, 15);
  V w0 = __builtin_shufflevector(u0, u2, 0, 1, 8, 9, 4, 5, 12, 13);
  V w1 = __builtin_shufflevector(u1, u3, 0, 1, 8, 9, 4, 5, 12, 13);
  V w2 = __builtin_shufflevector(u0, u2, 2, 3, 10, 11, 6, 7, 14, 15);
  V w3 = __builtin_shufflevector(u1, u3, 2, 3, 10, 11, 6, 7, 14, 15);
  V w4 = __builtin_shufflevector(u4, u6, 0, 1, 8, 9, 4, 5, 12, 13);
  V w5 = __builtin_shufflevector(u5, u7, 0, 1, 8, 9, 4, 5, 12, 13);
  V w6 = __builtin_shufflevector(u4, u6, 2, 3, 10, 11, 6, 7, 14, 15);
  V w7 = __builtin_shufflevector(u5, u7, 2, 3, 10, 11, 6, 7, 14, 15);
  b[0] = __builtin_shufflevector(w0, w4, 0, 1, 2, 3, 8, 9, 10, 11);
  b[1] = __builtin_shufflevector(w1, w5, 0, 1, 2, 3, 8, 9, 10, 11);
  b[2] = __builtin_shufflevector(w2, w6, 0, 1, 2, 3, 8, 9, 10, 11);
  b[3] = __builtin_shufflevector(w3, w7, 0, 1, 2, 3, 8, 9, 10, 11);
  b[4] = __builtin_shufflevector(w0, w4, 4, 5, 6, 7, 12, 13, 14, 15);
  b[5] = __builtin_shufflevector(w1, w5, 4, 5, 6, 7, 12, 13, 14, 15);
  b[6] = __builtin_shufflevector(w2, w6, 4, 5, 6, 7, 12, 13, 14, 15);
  b[7] = __builtin_shufflevector(w3, w7, 4, 5, 6, 7, 12, 13, 14, 15);
}
'''

def gen_soa(L, use_cols):
    L2, L3 = L*L, L*L*L
    if use_cols:
        passz = f"fft{L}_cols(sr + y*{L}, si + y*{L}, sr + y*{L}, si + y*{L}, 1, 1);"
        passy = f"fft{L}_cols(sr + z, si + z, sr + z, si + z, {L}, {L});"
        passx = f"""fft{L}_colspw(soa{L}_xr + u, soa{L}_xi + u, soa{L}_xr + u, soa{L}_xi + u, {L2}, {L2},
                     soa{L}_cr + u, soa{L}_ci + u);"""
    else:
        passz = f"fft{L}_core(sr + y*{L}, si + y*{L}, sr + y*{L}, si + y*{L}, 1, 1);"
        passy = f"fft{L}_core(sr + z, si + z, sr + z, si + z, {L}, {L});"
        passx = f"""{{
          V tr[{L}], ti[{L}];
          fft{L}_core(soa{L}_xr + u, soa{L}_xi + u, tr, ti, {L2}, 1);
          for (long i = 0; i < {L}; i++) {{
            long idx = u + i*{L2};
            V zr = tr[i] + soa{L}_cr[idx];
            V zi = ti[i] + soa{L}_ci[idx];
            V f = pw_factor(zr, zi);
            soa{L}_xr[idx] = zr*f; soa{L}_xi[idx] = zi*f;
          }}
        }}"""
    return f'''
// ---------------- L={L} : SoA across 8 volumes ----------------
static V soa{L}_xr[{L3}], soa{L}_xi[{L3}], soa{L}_cr[{L3}], soa{L}_ci[{L3}];
static V soa{L}_or[{L3}], soa{L}_oi[{L3}];

void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  for (long g0 = 0; g0 < B; g0 += 8) {{
    int lanes = (B - g0) < 8 ? (int)(B - g0) : 8;
    for (long i = 0; i < {L3}; i++) {{
      V xr = VC(0.0), xi = VC(0.0), cr = VC(1.0), ci = VC(0.0);
      for (int l = 0; l < lanes; l++) {{
        long src = 2*((g0+l)*{L3} + i);
        xr[l] = x0[src]; xi[l] = x0[src+1];
        cr[l] = c[src];  ci[l] = c[src+1];
      }}
      soa{L}_xr[i] = xr; soa{L}_xi[i] = xi; soa{L}_cr[i] = cr; soa{L}_ci[i] = ci;
    }}
    for (long it = 0; it < m; it++) {{
      for (long x = 0; x < {L}; x++) {{
        V *sr = soa{L}_xr + x*{L2}, *si = soa{L}_xi + x*{L2};
        for (long y = 0; y < {L}; y++)
          {passz}
        for (long z = 0; z < {L}; z++)
          {passy}
      }}
      for (long u = 0; u < {L2}; u++) {{
        {passx}
      }}
      if (it == 0) {{
        memcpy(soa{L}_or, soa{L}_xr, sizeof(soa{L}_or));
        memcpy(soa{L}_oi, soa{L}_xi, sizeof(soa{L}_oi));
      }}
    }}
    for (int l = 0; l < lanes; l++) {{
      double *d1 = out1 + 2*(g0+l)*{L3}, *dm = outm + 2*(g0+l)*{L3};
      for (long i = 0; i < {L3}; i++) {{
        d1[2*i] = soa{L}_or[i][l]; d1[2*i+1] = soa{L}_oi[i][l];
        dm[2*i] = soa{L}_xr[i][l]; dm[2*i+1] = soa{L}_xi[i][l];
      }}
    }}
  }}
}}
'''

def gen_slab(L):
    P1 = r8(L)
    P2 = L*P1 + (8 if L == 64 else 0)
    P1V, P2V = P1//8, P2//8
    NB = P1//8
    L3 = L*L*L
    return f'''
// ---------------- L={L} : per-volume slab scheme (P1={P1}, P2={P2}) ----------------
static V slab{L}_xr[{L*P2V}], slab{L}_xi[{L*P2V}];
static V slab{L}_cer[{L*P2V}], slab{L}_cei[{L*P2V}];
static V slab{L}_cor[{L*P2V}], slab{L}_coi[{L*P2V}];
static V slab{L}_1r[{L*P2V}], slab{L}_1i[{L*P2V}];
static V slab{L}_tsr[{P1*P1V}], slab{L}_tsi[{P1*P1V}];
static V slab{L}_br[{P1}], slab{L}_bi[{P1}];

static void slab{L}_iter(long m) {{
  for (long j = {L}; j < {P1}; j++) {{ slab{L}_br[j] = VC(0.0); slab{L}_bi[j] = VC(0.0); }}
  for (long it = 0; it < m; it++) {{
    for (long x = 0; x < {L}; x++) {{
      V *sr = slab{L}_xr + x*{P2V}, *si = slab{L}_xi + x*{P2V};
      for (long cc = 0; cc < {NB}; cc++) {{
        fft{L}_cols(sr + cc, si + cc, slab{L}_br, slab{L}_bi, {P1V}, 1);
        for (long rb = 0; rb < {P1V}; rb++) {{
          V tb[8];
          tr8x8(slab{L}_br + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsr[(cc*8+q)*{P1V} + rb] = tb[q];
          tr8x8(slab{L}_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsi[(cc*8+q)*{P1V} + rb] = tb[q];
        }}
      }}
      for (long cc = 0; cc < {NB}; cc++)
        fft{L}_cols(slab{L}_tsr + cc, slab{L}_tsi + cc, sr + cc, si + cc, {P1V}, {P1V});
    }}
    const V *pcr = (it & 1) ? slab{L}_cer : slab{L}_cor;
    const V *pci = (it & 1) ? slab{L}_cei : slab{L}_coi;
    for (long o = 0; o < {L*P1V}; o++)
      fft{L}_colspw(slab{L}_xr + o, slab{L}_xi + o, slab{L}_xr + o, slab{L}_xi + o, {P2V}, {P2V},
                    pcr + o, pci + o);
    if (it == 0) {{
      memcpy(slab{L}_1r, slab{L}_xr, sizeof(slab{L}_1r));
      memcpy(slab{L}_1i, slab{L}_xi, sizeof(slab{L}_1i));
    }}
  }}
}}

static void slab{L}_extract(const V*restrict ar, const V*restrict ai, int parity, double*restrict dst) {{
  const double *r = (const double*)ar, *im = (const double*)ai;
  for (long x = 0; x < {L}; x++)
    for (long y = 0; y < {L}; y++) {{
      const long rowr = x*{P2} + (parity ? y : y*{P1});
      const long step = parity ? {P1} : 1;
      double *d = dst + 2*((x*{L} + y)*{L});
      for (long z = 0; z < {L}; z++) {{
        d[2*z]   = r[rowr + z*step];
        d[2*z+1] = im[rowr + z*step];
      }}
    }}
}}

void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  double *xr = (double*)slab{L}_xr, *xi = (double*)slab{L}_xi;
  double *cer = (double*)slab{L}_cer, *cei = (double*)slab{L}_cei;
  double *cor = (double*)slab{L}_cor, *coi = (double*)slab{L}_coi;
  for (long v = 0; v < B; v++) {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};
    for (long x = 0; x < {L}; x++)
      for (long y = 0; y < {L}; y++) {{
        long row = x*{P2} + y*{P1};
        const double *px = sx + 2*((x*{L}+y)*{L});
        const double *pc = sc + 2*((x*{L}+y)*{L});
        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cer[row+z] = pc[2*z]; cei[row+z] = pc[2*z+1];
          cor[x*{P2} + z*{P1} + y] = pc[2*z];
          coi[x*{P2} + z*{P1} + y] = pc[2*z+1];
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cer[row+z] = 1.0; cei[row+z] = 0.0;
          cor[row+z] = 1.0; coi[row+z] = 0.0;
        }}
      }}
    slab{L}_iter(m);
    slab{L}_extract(slab{L}_1r, slab{L}_1i, 1, out1 + 2*v*{L3});
    slab{L}_extract(slab{L}_xr, slab{L}_xi, (int)(m & 1), outm + 2*v*{L3});
  }}
}}
'''

def main():
    parts = [HDR]
    for L in SIZES:
        if SCHEME[L] == 'soa':
            if L == 13:
                parts.append(kernels.gen_kernels(13))
                parts.append(gen_soa(13, True))
            else:
                code, _ = codelets.gen_core(L)
                parts.append(code)
                parts.append(gen_soa(L, False))
        else:
            parts.append(kernels.gen_kernels(L))
            parts.append(gen_slab(L))
    src = "\n".join(parts)
    open("../implementation.c", "w").write(src)
    print(f"wrote implementation.c: {len(src.splitlines())} lines")

if __name__ == "__main__":
    main()
PYEOF
python3 gen.py && rm -f ../implementation.so && cd /workdir && time gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py