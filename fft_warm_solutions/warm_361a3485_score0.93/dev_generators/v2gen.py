import pgen

def crt(L, N1, r1, r9):
    # plane p in [0,L) with p%N1==r1, p%9==r9
    for p in range(L):
        if p % N1 == r1 % N1 and p % 9 == r9 % 9:
            return p
    raise ValueError

def gen_stageA_ip(L, N1, name, PS):
    """in-place x stage A: for j2, dft-N1 across planes.
       input j1 at plane (9*j1+N1*j2)%L ; output k1 at plane CRT(k1, N1*j2)."""
    import numpy as np
    from pgen import E, dft4e, dft5e, hexd, PI2, LD
    lines = []
    A = lines.append
    consts = {}
    if N1 == 5:
        a5 = -PI2/LD(5)
        consts = {'C51':hexd(float(np.cos(a5))),'C52':hexd(float(np.cos(2*a5))),
                  'S51':hexd(float(np.sin(a5))),'S52':hexd(float(np.sin(2*a5)))}
        A(f"static const double SAT{L}[4] __attribute__((aligned(64))) = {{" + ",".join(consts.values()) + "};")
    # plane tables per j2
    inpl = [[ (9*j1+N1*j2) % L for j1 in range(N1)] for j2 in range(9)]
    outpl = [[ crt(L, N1, k1, N1*j2) for k1 in range(N1)] for j2 in range(9)]
    A(f"static const int SA{L}_IN[9][{N1}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in inpl) + "};")
    A(f"static const int SA{L}_OUT[9][{N1}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in outpl) + "};")
    A(f"static void {name}(vd *S, int j2) {{")
    kmap={}
    for i,nm in enumerate(consts):
        A(f"  vd K_{nm} = VCA(SAT{L}[{i}]);")
        kmap[nm]=f"K_{nm}"
    for j1 in range(N1):
        A(f"  const vd * i{j1} = S + 2*(ptrdiff_t)SA{L}_IN[j2][{j1}]*{PS};")
    for k1 in range(N1):
        A(f"  vd * o{k1} = S + 2*(ptrdiff_t)SA{L}_OUT[j2][{k1}]*{PS};")
    A(f"  for (long q = 0; q < {PS}; q++) {{")
    e = E()
    xs = [(e.v(f"i{j1}[2*q]"), e.v(f"i{j1}[2*q+1]")) for j1 in range(N1)]
    ys = dft4e(e, xs) if N1==4 else dft5e(e, xs, kmap)
    lines.extend(["  "+x for x in e.lines])
    for k1 in range(N1):
        A(f"    o{k1}[2*q] = {ys[k1][0]}; o{k1}[2*q+1] = {ys[k1][1]};")
    A("  }")
    A("}")
    return "\n".join(lines)

def gen_stageB_ip(L, N1, name, PS):
    """in-place x stage B: for k1, dft9 across planes.
       input j2 at plane CRT(k1, N1*j2); output k2 at plane CRT(k1, k2)."""
    import numpy as np
    from pgen import E, dft9e, hexd, PI2, LD
    lines=[]; A=lines.append
    consts={'H':hexd(-0.5),'S3':hexd(float(np.sin(PI2*LD(1)/LD(6))))}
    for idx in (1,2,4):
        a = -PI2*LD(idx)/LD(9)
        consts[f'C9_{idx}']=hexd(float(np.cos(a))); consts[f'S9_{idx}']=hexd(float(np.sin(a)))
    A(f"static const double SBT{L}[{len(consts)}] __attribute__((aligned(64))) = {{" + ",".join(consts.values()) + "};")
    inpl = [[ crt(L, N1, k1, N1*j2) for j2 in range(9)] for k1 in range(N1)]
    outpl = [[ crt(L, N1, k1, k2) for k2 in range(9)] for k1 in range(N1)]
    A(f"static const int SB{L}_IN[{N1}][9] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in inpl) + "};")
    A(f"static const int SB{L}_OUT[{N1}][9] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in outpl) + "};")
    A(f"static void {name}(vd *S, int k1) {{")
    kmap={}
    for i,nm in enumerate(consts):
        A(f"  vd K_{nm} = VCA(SBT{L}[{i}]);")
        kmap[nm]=f"K_{nm}"
    for j2 in range(9):
        A(f"  const vd * i{j2} = S + 2*(ptrdiff_t)SB{L}_IN[k1][{j2}]*{PS};")
    for k2 in range(9):
        A(f"  vd * o{k2} = S + 2*(ptrdiff_t)SB{L}_OUT[k1][{k2}]*{PS};")
    A(f"  for (long q = 0; q < {PS}; q++) {{")
    e = E()
    xs = [(e.v(f"i{j2}[2*q]"), e.v(f"i{j2}[2*q+1]")) for j2 in range(9)]
    ys = dft9e(e, xs, kmap)
    lines.extend(["  "+x for x in e.lines])
    for k2 in range(9):
        A(f"    o{k2}[2*q] = {ys[k2][0]}; o{k2}[2*q+1] = {ys[k2][1]};")
    A("  }")
    A("}")
    return "\n".join(lines)

def gen_v2(L):
    N1 = 4 if L == 36 else 5
    PV = (L + 7) // 8
    PS = L * PV
    s = []
    s.append(gen_stageA_ip(L, N1, f"st{L}a", PS))
    s.append(gen_stageB_ip(L, N1, f"st{L}b", PS))
    s.append(pgen.gen_pfa_io(L, N1, f"kgio{L}"))
    s.append(f"""
// -------- family V2 (single-arena) L={L}: PV={PV}, PS={PS} --------
static vd *VS{L}_, *CA{L}_, *CB{L}_;
static void v2init_{L}(void) {{
    if (VS{L}_) return;
    size_t bytes = (size_t)2*{PS}*{L}*sizeof(vd) + 4096;
    VS{L}_ = (vd*)((char*)big_alloc(bytes) + 0);
    CA{L}_ = (vd*)((char*)big_alloc(bytes) + 1024);
    CB{L}_ = (vd*)((char*)big_alloc(bytes) + 2048);
}}
static void conv_in_B{L}(const double *restrict src, vd *restrict X) {{
    for (int x = 0; x < {L}; x++) {{
      for (int yb = 0; yb < {PV}; yb++) {{
        for (int zb = 0; zb < {PV}; zb++) {{
            vd tR[8], tI[8];
            for (int r = 0; r < 8; r++) {{
                int y = 8*yb + r;
                if (y < {L}) {{
                    const double *p = src + 2*((ptrdiff_t)(x*{L}+y)*{L} + 8*zb);
                    int nz = {L} - 8*zb; if (nz > 8) nz = 8;
                    double rr[8] __attribute__((aligned(64))), ii[8] __attribute__((aligned(64)));
                    for (int cc = 0; cc < nz; cc++) {{ rr[cc]=p[2*cc]; ii[cc]=p[2*cc+1]; }}
                    for (int cc = nz; cc < 8; cc++) {{ rr[cc]=0; ii[cc]=0; }}
                    tR[r] = *(const vd*)rr; tI[r] = *(const vd*)ii;
                }} else {{ tR[r] = (vd)_mm512_setzero_pd(); tI[r] = (vd)_mm512_setzero_pd(); }}
            }}
            tr8(tR); tr8(tI);
            for (int cc = 0; cc < 8; cc++) {{
                int z = 8*zb + cc;
                if (z < {L}) {{
                    X[2*((ptrdiff_t)(x*{L}+z)*{PV} + yb)]     = tR[cc];
                    X[2*((ptrdiff_t)(x*{L}+z)*{PV} + yb) + 1] = tI[cc];
                }}
            }}
        }}
      }}
    }}
}}
static void conv_out_B{L}(const vd *restrict X, double *restrict dst) {{
    for (int x = 0; x < {L}; x++) {{
      for (int zb = 0; zb < {PV}; zb++) {{
        for (int yb = 0; yb < {PV}; yb++) {{
            vd tR[8], tI[8];
            for (int r = 0; r < 8; r++) {{
                int z = 8*zb + r;
                if (z < {L}) {{
                    tR[r] = X[2*((ptrdiff_t)(x*{L}+z)*{PV} + yb)];
                    tI[r] = X[2*((ptrdiff_t)(x*{L}+z)*{PV} + yb) + 1];
                }} else {{ tR[r] = (vd)_mm512_setzero_pd(); tI[r] = tR[r]; }}
            }}
            tr8(tR); tr8(tI);
            for (int cc = 0; cc < 8; cc++) {{
                int y = 8*yb + cc;
                if (y >= {L}) break;
                double *p = dst + 2*((ptrdiff_t)(x*{L}+y)*{L} + 8*zb);
                int nz = {L} - 8*zb; if (nz > 8) nz = 8;
                const double *rr = (const double*)&tR[cc], *ii = (const double*)&tI[cc];
                for (int q2 = 0; q2 < nz; q2++) {{ p[2*q2] = rr[q2]; p[2*q2+1] = ii[q2]; }}
            }}
        }}
      }}
    }}
}}
static void planepass_{L}(vd *restrict S, const vd *restrict CT) {{
    vd scr[2*{PS}] __attribute__((aligned(64)));
    for (int wc = 0; wc < {PV}; wc++) {{
        const vd *p = S + 2*wc;
        kgio{L}(p, p+1, 2*{PV}, scr + 2*wc, scr + 2*wc + 1, 2*{PV}, p, p+1, 0);
    }}
    for (int ub = 0; ub < {PV}; ub++) {{
        vd bufR[{PV}*8], bufI[{PV}*8];
        for (int blk = 0; blk < {PV}; blk++) {{
            vd tR[8], tI[8];
            for (int r = 0; r < 8; r++) {{
                int u = 8*ub + r;
                if (u < {L}) {{ tR[r] = scr[2*((ptrdiff_t)u*{PV}+blk)]; tI[r] = scr[2*((ptrdiff_t)u*{PV}+blk)+1]; }}
                else {{ tR[r] = (vd)_mm512_setzero_pd(); tI[r] = tR[r]; }}
            }}
            tr8(tR); tr8(tI);
            for (int cc = 0; cc < 8; cc++) {{ bufR[8*blk+cc] = tR[cc]; bufI[8*blk+cc] = tI[cc]; }}
        }}
        kgio{L}(bufR, bufI, 1, S + 2*ub, S + 2*ub + 1, 2*{PV}, CT + 2*ub, CT + 2*ub + 1, 1);
    }}
}}
void run{L}(long long Bll, long long mll, const double *x0, const double *c0,
           double *out1, double *outm) {{
    const long NV = {L}*{L}*{L}; const long B = (long)Bll;
    long m = (long)mll; if (m < 1) m = 1;
    v2init_{L}();
    for (long v = 0; v < B; v++) {{
        conv_in_vol(x0 + (long long)v*NV*2, {L}, {PV}, VS{L}_);
        conv_in_vol(c0 + (long long)v*NV*2, {L}, {PV}, CA{L}_);
        conv_in_B{L}(c0 + (long long)v*NV*2, CB{L}_);
        int inA = 1;
        for (long it = 1; it <= m; it++) {{
            const vd *CT = inA ? CB{L}_ : CA{L}_;
            for (int j2 = 0; j2 < 9; j2++) st{L}a(VS{L}_, j2);
            for (int k1 = 0; k1 < {N1}; k1++) st{L}b(VS{L}_, k1);
            for (int x = 0; x < {L}; x++)
                planepass_{L}(VS{L}_ + (ptrdiff_t)x*2*{PS}, CT + (ptrdiff_t)x*2*{PS});
            inA ^= 1;
            if (it == 1 || it == m) {{
                if (it == 1) {{ if (inA) conv_out_vol(VS{L}_, {L}, {PV}, out1 + (long long)v*NV*2); else conv_out_B{L}(VS{L}_, out1 + (long long)v*NV*2); }}
                if (it == m) {{ if (inA) conv_out_vol(VS{L}_, {L}, {PV}, outm + (long long)v*NV*2); else conv_out_B{L}(VS{L}_, outm + (long long)v*NV*2); }}
            }}
        }}
    }}
}}
""")
    return "\n".join(s)
