"""asm looped two-stage DFT for family B: dftL_a(src, dst, tin, tout) using GPR offset tables."""
from gen_asm import A
from gen_asm_b import Tab, Ctx, adft_graph, acmulw, arel, tw_ld
from gen_b2 import fact
from genlib import hexd

def gen_dft_a(L):
    N1, N2, mode = fact(L)
    tab = Tab(f"A{L}")
    a = A()
    ctx = Ctx(a, tab)
    # prewarm constants used by both graphs (keep resident across loops)
    # collect by dry-run on a scratch emitter
    dry_a = A(); dry_ctx = Ctx(dry_a, Tab("dry"))
    xs = [(dry_a.alloc(), dry_a.alloc()) for _ in range(max(N1,N2))]
    try:
        adft_graph(dry_ctx, xs[:N1], N1)
    except Exception:
        pass
    # simpler: just emit; consts inside loop get cached per loop body (reloaded per iter)
    a = A()
    ctx = Ctx(a, tab)
    gprs = ["r8", "r9"]
    a.ins(f"mov ${N2}, %[cnt]")
    a.ins("1:")
    x = []
    for n1 in range(N1):
        g = gprs[n1 % 2]
        a.ins(f"movq {n1*8}(%[tin]), %%{g}")
        xr = a.alloc(); xi = a.alloc()
        a.ins(f"vmovapd (%[src],%%{g}), %%zmm{xr}")
        a.ins(f"vmovapd 64(%[src],%%{g}), %%zmm{xi}")
        x.append((xr, xi))
    y = adft_graph(ctx, x, N1)
    if mode == 'ct':
        # twiddle rows from twv (advances per iter). row k1-1 at (k1-1)*128
        y2 = [y[0]]
        for k1 in range(1, N1):
            wr = a.alloc(); wi = a.alloc()
            a.ins(f"vmovapd {(k1-1)*128}(%[twv]), %%zmm{wr}")
            a.ins(f"vmovapd {(k1-1)*128+64}(%[twv]), %%zmm{wi}")
            xr, xi = y[k1]
            t = a.mul(xr, wr)
            a.fnma(t, xi, wi)
            u = a.mul(xi, wr)
            a.fma(u, xr, wi)
            a.rel(xr); a.rel(xi); a.rel(wr); a.rel(wi)
            y2.append((t, u))
        y = y2
    for k1 in range(N1):
        a.st('SC', k1*N2*128, y[k1][0]); a.st('SC', k1*N2*128+64, y[k1][1])
        arel(ctx, y[k1])
    ctx.flush()
    a.ins(f"add ${N1*8}, %[tin]")
    a.ins("add $128, %[SC]")
    if mode == 'ct':
        a.ins(f"add ${(N1-1)*128}, %[twv]")
    a.ins("dec %[cnt]")
    a.ins("jnz 1b")
    a.ins(f"sub ${N2*128}, %[SC]")
    a.ins(f"mov ${N1}, %[cnt]")
    a.ins("2:")
    x2 = [(a.ld('SC', n2*128), a.ld('SC', n2*128+64)) for n2 in range(N2)]
    yy = adft_graph(ctx, x2, N2)
    for k2 in range(N2):
        a.ins(f"movq {k2*8}(%[tout]), %%r8")
        a.ins(f"vmovapd %%zmm{yy[k2][0]}, (%[dst],%%r8)")
        a.ins(f"vmovapd %%zmm{yy[k2][1]}, 64(%[dst],%%r8)")
        arel(ctx, yy[k2])
    ctx.flush()
    a.ins(f"add ${N2*128}, %[SC]")
    a.ins(f"add ${N2*8}, %[tout]")
    a.ins("dec %[cnt]")
    a.ins("jnz 2b")
    assert not a.live, a.live
    body = "\\n\\t".join(a.lines)
    ops = ['[src]"r"(src)', '[dst]"r"(dst)', '[tin]"+r"(tin)', '[tout]"+r"(tout)',
           '[SC]"+r"(sc)', '[cnt]"=&r"(cnt)', f'[ctv]"r"(CTV_A{L})']
    if mode == 'ct':
        ops.append('[twv]"+r"(twv)')
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', ' + ", ".join(f'"{g}"' for g in set(gprs) | {"r8"}) + ', "memory", "cc"'
    twdecl = ""
    twsetup = ""
    if mode == 'ct':
        rows = []
        for n2 in range(N2):
            for k1 in range(1, N1):
                wr, wi = tw_ld(64, k1*n2)
                rows.append("{" + ",".join([hexd(wr)]*8) + "," + ",".join([hexd(wi)]*8) + "}")
        twdecl = f"static const double TWVA_{L}[{N2*(N1-1)}][16] ALIGN64 = {{ {', '.join(rows)} }};\n"
        twsetup = f"const double* twv = &TWVA_{L}[0][0];"
    return twdecl + tab.decl() + f"""static double SCA_{L}[{L}][16] ALIGN64;
static void __attribute__((noinline)) dft{L}_a(const double* src, double* dst, const long* tin, const long* tout){{
    double* sc = SCA_{L}[0];
    long cnt;
    {twsetup}
    __asm__ volatile("{body}"
    : {", ".join(o for o in ops if '+' in o or '=' in o)}
    : {", ".join(o for o in ops if '+' not in o and '=' not in o)}
    : {clob});
}}
"""
