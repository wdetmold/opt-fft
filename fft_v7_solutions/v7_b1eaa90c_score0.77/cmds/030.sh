cd /workdir/dev && python3 - <<'EOF'
src = open('codelets.py').read()
# Replace gen_core with a staged version for composite sizes
old = '''def gen_core(N):'''
new = '''def gen_staged(N, fac, pfa):
    """two-stage core with explicit intermediate arrays; fac=(P,Q).
    pfa=True: PFA maps, no twiddles. Loads xr[i*is_], stores yr[k*os_]."""
    P, Q = fac
    assert P*Q == N
    segs = []
    # stage1: for each second-index b in [0,Q): P-point DFT
    for b in range(Q):
        e = E()
        if pfa:
            idxs = [(Q*a + P*b) % N for a in range(P)]
        else:
            idxs = [Q*t + b for t in range(P)]
        x = [(e.t(f"xr[{i}*is_]"), e.t(f"xi[{i}*is_]")) for i in idxs]
        out = fft_any(e, x, P)
        if not pfa:
            out2 = []
            for k1 in range(P):
                c, s = tw(b*k1, N)
                out2.append(cmul_w(e, out[k1], c, s))
            out = out2
        stores = "\\n".join(f"    Ar[{b*P+k1}] = {out[k1][0]}; Ai[{b*P+k1}] = {out[k1][1]};" for k1 in range(P))
        segs.append("  {\\n" + e.code(indent="    ") + "\\n" + stores + "\\n  }")
    # stage2: for each k1 in [0,P): Q-point DFT over b
    for k1 in range(P):
        e = E()
        x = [(e.t(f"Ar[{b*P+k1}]"), e.t(f"Ai[{b*P+k1}]")) for b in range(Q)]
        out = fft_any(e, x, Q)
        stores = []
        for k2 in range(Q):
            if pfa:
                # X[k] for the unique k with k%P==?? : stage1 over 'a' computed DFT_P -> index k%P ; stage2 -> k%Q
                k = [kk for kk in range(N) if kk % P == k1 and kk % Q == k2][0]
            else:
                k = P*k2 + k1
            stores.append(f"    ST({k}, {out[k2][0]}, {out[k2][1]});")
        segs.append("  {\\n" + e.code(indent="    ") + "\\n" + "\\n".join(stores) + "\\n  }")
    body = "\\n".join(segs)
    return body

def gen_core(N):'''
src = src.replace(old, new)

# rewrite gen_core dispatch to use staged for 36,45,64 with ST macro interface
old2 = '''    e = E()
    x = []
    for i in range(N):
        x.append((e.t(f"xr[{i}*is_]"), e.t(f"xi[{i}*is_]")))
    if N == 6:    out = fft_pfa(e, x, 2, 3)
    elif N == 8:  out = fft8(e, x)
    elif N in (13, 17, 23): out = fft_halfmatrix(e, x, N)
    elif N == 36: out = fft_pfa(e, x, 4, 9)
    elif N == 45: out = fft_pfa(e, x, 9, 5)
    elif N == 64: out = fft_ct(e, x, 8, 8)
    elif N in (2,3,4,5,9): out = fft_any(e, x, N)
    else: raise ValueError(N)
    body = e.code()
    stores = "\\n".join(f"  yr[{k}*os_] = {out[k][0]}; yi[{k}*os_] = {out[k][1]};" for k in range(N))
    code = f"""static inline __attribute__((always_inline)) void fft{N}_core(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
{body}
{stores}
}}
"""
    return code, e.n'''
new2 = '''    if N in (36, 45, 64):
        fac = {36:(4,9), 45:(9,5), 64:(8,8)}[N]
        pfa = N != 64
        body = gen_staged(N, fac, pfa)
        code = f"""static inline __attribute__((always_inline)) void fft{N}_core(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
  V Ar[{N}], Ai[{N}];
#define ST(k, a, b) do {{ yr[(k)*os_] = (a); yi[(k)*os_] = (b); }} while (0)
{body}
#undef ST
}}
"""
        return code, 0
    e = E()
    x = []
    for i in range(N):
        x.append((e.t(f"xr[{i}*is_]"), e.t(f"xi[{i}*is_]")))
    if N == 6:    out = fft_pfa(e, x, 2, 3)
    elif N == 8:  out = fft8(e, x)
    elif N in (13, 17, 23): out = fft_halfmatrix(e, x, N)
    elif N in (2,3,4,5,9): out = fft_any(e, x, N)
    else: raise ValueError(N)
    body = e.code()
    stores = "\\n".join(f"  yr[{k}*os_] = {out[k][0]}; yi[{k}*os_] = {out[k][1]};" for k in range(N))
    code = f"""static inline __attribute__((always_inline)) void fft{N}_core(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
{body}
{stores}
}}
"""
    return code, e.n'''
src = src.replace(old2, new2)
open('codelets.py','w').write(src)
print("patched")
EOF
python3 test_cores.py 2>/dev/null | tail -5