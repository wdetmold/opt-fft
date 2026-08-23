import numpy as np
LD = np.longdouble
PI_LD = LD('3.14159265358979323846264338327950288')

def gen_prime_codelet(N, name, KB=4, UNROLL2=False):
    h = (N-1)//2
    # constant tables CT[k][j], ST[k][j], k,j in 1..h (row padded to h+1)
    W = h + 2
    ct_rows = []
    st_rows = []
    for k in range(1, h+1):
        crow = ['0.0']; srow = ['0.0']
        for j in range(1, h+1):
            ang = (LD(2)*PI_LD)*LD((k*j) % N)/LD(N)
            crow.append(float(np.cos(ang)).hex())
            srow.append(float(np.sin(ang)).hex())
        crow.append('0.0'); srow.append('0.0')
        ct_rows.append("{" + ",".join(crow) + "}")
        st_rows.append("{" + ",".join(srow) + "}")
    out = []
    out.append(f"static const double CT{N}[{h}][{W}] __attribute__((aligned(64))) = {{{','.join(ct_rows)}}};")
    out.append(f"static const double ST{N}[{h}][{W}] __attribute__((aligned(64))) = {{{','.join(st_rows)}}};")
    out.append(f"static void {name}(const double* pxr, const double* pxi, double* pyr, double* pyi, long sx, long sy){{")
    out.append(f"    __attribute__((aligned(64))) double PR[{(h+2)*8}], PI_[{(h+2)*8}], QR[{(h+2)*8}], QI[{(h+2)*8}];")
    out.append( "    __m512d x0r = _mm512_loadu_pd(pxr), x0i = _mm512_loadu_pd(pxi);")
    out.append( "    __m512d s0r = x0r, s0i = x0i, s1r = _mm512_setzero_pd(), s1i = _mm512_setzero_pd();")
    out.append(f"    for (long j = 1; j <= {h}; ++j){{")
    out.append(f"        __m512d a = _mm512_loadu_pd(pxr + j*sx), b = _mm512_loadu_pd(pxr + ({N}-j)*sx);")
    out.append(f"        __m512d c = _mm512_loadu_pd(pxi + j*sx), d = _mm512_loadu_pd(pxi + ({N}-j)*sx);")
    out.append( "        __m512d pr = a+b, qr = a-b, pi = c+d, qi = c-d;")
    out.append( "        _mm512_store_pd(PR+j*8, pr); _mm512_store_pd(QR+j*8, qr);")
    out.append( "        _mm512_store_pd(PI_+j*8, pi); _mm512_store_pd(QI+j*8, qi);")
    out.append( "        if (j & 1){ s0r = s0r + pr; s0i = s0i + pi; } else { s1r = s1r + pr; s1i = s1i + pi; }")
    out.append( "    }")
    out.append( "    _mm512_storeu_pd(pyr, s0r+s1r); _mm512_storeu_pd(pyi, s0i+s1i);")
    out.append(f"    _mm512_store_pd(PR+{(h+1)*8}, _mm512_setzero_pd()); _mm512_store_pd(QR+{(h+1)*8}, _mm512_setzero_pd());")
    out.append(f"    _mm512_store_pd(PI_+{(h+1)*8}, _mm512_setzero_pd()); _mm512_store_pd(QI+{(h+1)*8}, _mm512_setzero_pd());")
    # k tiles
    k = 1
    while k <= h:
        kb = min(KB, h - k + 1)
        ks = list(range(k, k+kb))
        decl = []
        for t in ks:
            decl.append(f"ar{t} = x0r, ai{t} = x0i, br{t} = _mm512_setzero_pd(), bi{t} = _mm512_setzero_pd()")
        out.append(f"    {{ __m512d {', '.join(decl)};")
        if UNROLL2:
            out.append(f"    for (long j = 1; j <= {h}; j += 2){{")
            out.append( "        __m512d pr = _mm512_load_pd(PR+j*8), pi = _mm512_load_pd(PI_+j*8);")
            out.append( "        __m512d qr = _mm512_load_pd(QR+j*8), qi = _mm512_load_pd(QI+j*8);")
            out.append( "        __m512d pr2 = _mm512_load_pd(PR+j*8+8), pi2 = _mm512_load_pd(PI_+j*8+8);")
            out.append( "        __m512d qr2 = _mm512_load_pd(QR+j*8+8), qi2 = _mm512_load_pd(QI+j*8+8);")
            for t in ks:
                out.append(f"        {{ __m512d c = _mm512_set1_pd(CT{N}[{t-1}][j]), s = _mm512_set1_pd(ST{N}[{t-1}][j]);")
                out.append(f"          __m512d c2 = _mm512_set1_pd(CT{N}[{t-1}][j+1]), s2 = _mm512_set1_pd(ST{N}[{t-1}][j+1]);")
                out.append(f"          ar{t} = _mm512_fmadd_pd(c, pr, ar{t}); ai{t} = _mm512_fmadd_pd(c, pi, ai{t});")
                out.append(f"          br{t} = _mm512_fmadd_pd(s, qr, br{t}); bi{t} = _mm512_fmadd_pd(s, qi, bi{t});")
                out.append(f"          ar{t} = _mm512_fmadd_pd(c2, pr2, ar{t}); ai{t} = _mm512_fmadd_pd(c2, pi2, ai{t});")
                out.append(f"          br{t} = _mm512_fmadd_pd(s2, qr2, br{t}); bi{t} = _mm512_fmadd_pd(s2, qi2, bi{t}); }}")
            out.append( "    }")
        else:
            out.append(f"    for (long j = 1; j <= {h}; ++j){{")
            out.append( "        __m512d pr = _mm512_load_pd(PR+j*8), pi = _mm512_load_pd(PI_+j*8);")
            out.append( "        __m512d qr = _mm512_load_pd(QR+j*8), qi = _mm512_load_pd(QI+j*8);")
            for t in ks:
                out.append(f"        {{ __m512d c = _mm512_set1_pd(CT{N}[{t-1}][j]), s = _mm512_set1_pd(ST{N}[{t-1}][j]);")
                out.append(f"          ar{t} = _mm512_fmadd_pd(c, pr, ar{t}); ai{t} = _mm512_fmadd_pd(c, pi, ai{t});")
                out.append(f"          br{t} = _mm512_fmadd_pd(s, qr, br{t}); bi{t} = _mm512_fmadd_pd(s, qi, bi{t}); }}")
            out.append( "    }")
        for t in ks:
            out.append(f"    _mm512_storeu_pd(pyr + {t}*sy, ar{t}+bi{t});")
            out.append(f"    _mm512_storeu_pd(pyi + {t}*sy, ai{t}-br{t});")
            out.append(f"    _mm512_storeu_pd(pyr + {N-t}*sy, ar{t}-bi{t});")
            out.append(f"    _mm512_storeu_pd(pyi + {t and (N-t)}*sy, ai{t}+br{t});")
        out.append("    }")
        k += kb
    out.append("}")
    return "\n".join(out)

if __name__ == "__main__":
    print(gen_prime_codelet(13, "fft13_t2")[:500])
