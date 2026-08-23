# Generator for L = F*G engines (36 = 4*9, 45 = 5*9): 6D-PFA digit-state engine.
import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
import os
import numpy as np

def emit_transpose8(em, inv, outv):
    """8x8 double transpose: inv/outv lists of 8 __m512d names."""
    t=[None]*8; s=[None]*8
    for i in range(4):
        t[2*i]   = T(em, f'_mm512_unpacklo_pd({inv[2*i]},{inv[2*i+1]})')
        t[2*i+1] = T(em, f'_mm512_unpackhi_pd({inv[2*i]},{inv[2*i+1]})')
    # stage2: 4x4 lane blocks: shuffle_f64x2 imm combos
    for i in range(2):
        for j in range(2):
            s[4*i+j]   = T(em, f'_mm512_shuffle_f64x2({t[4*i+j]},{t[4*i+j+2]},0x88)')
            s[4*i+j+2] = T(em, f'_mm512_shuffle_f64x2({t[4*i+j]},{t[4*i+j+2]},0xdd)')
    for j in range(4):
        em(f'{outv[j]} = _mm512_shuffle_f64x2({s[j]},{s[j+4]},0x88);')
        em(f'{outv[j+4]} = _mm512_shuffle_f64x2({s[j]},{s[j+4]},0xdd);')

def emit_map(em, C, zr, zi, outr, outi):
    """z/(1+|z|). q-form NR rsqrt (scale-carried), rcp14+2NR. 22 ops."""
    three = C.get('C_three',3.0); twelve = C.get('C_twelve',12.0)
    s16 = C.get('C_s16',0.0625)
    one = C.get('C_one',1.0); two = C.get('C_two',2.0); tiny = C.get('C_tiny',1e-30)
    t0 = T(em, f'_mm512_fmadd_pd({zi},{zi},_mm512_mul_pd({zr},{zr}))')
    t  = T(em, f'_mm512_max_pd({t0},{tiny})')
    p0 = T(em, f'_mm512_rsqrt14_pd({t})')
    q1 = T(em, f'_mm512_mul_pd({t},{p0})')
    e1 = T(em, f'_mm512_fnmadd_pd({q1},{p0},{three})')
    p1 = T(em, f'_mm512_mul_pd({p0},{e1})')
    q2 = T(em, f'_mm512_mul_pd({t},{p1})')
    e2 = T(em, f'_mm512_fnmadd_pd({q2},{p1},{twelve})')
    r  = T(em, f'_mm512_mul_pd({q2},{e2})')
    d  = T(em, f'_mm512_fmadd_pd({r},{s16},{one})')
    w  = T(em, f'_mm512_rcp14_pd({d})')
    e3 = T(em, f'_mm512_fnmadd_pd({d},{w},{two})')
    w1 = T(em, f'_mm512_mul_pd({w},{e3})')
    e4 = T(em, f'_mm512_fnmadd_pd({d},{w1},{two})')
    w2 = T(em, f'_mm512_mul_pd({w1},{e4})')
    em(f'{outr} = _mm512_mul_pd({zr},{w2});')
    em(f'{outi} = _mm512_mul_pd({zi},{w2});')

def modinv(a, m):
    for x in range(1, m):
        if (a*x) % m == 1: return x
    raise ValueError

def gen_engine(F, G):
    N = F*G
    FG3 = F**3
    LANES = ((FG3 + 7)//8)*8          # u-lane count padded (64 for 36, 128 for 45)
    LANESP = LANES + 8                 # row stride in doubles (4K-alias skew)
    ROWS = G**3                        # 729
    ROWSP = ((ROWS + 7)//8)*8          # 736
    Ginv = modinv(G % F, F) if F > 1 else 0
    Finv = modinv(F % G, G)
    sF = [ (k*Ginv) % F for k in range(F) ]   # DFT_F output k -> digit slot
    sG = [ (k*Finv) % G for k in range(G) ]
    em = Emitter(); C = Consts()
    name = f'{N}'
    if F == 4:
        SLANE = '(((u2>>1)*4+u3)*8 + u1*2+(u2&1))'
        if os.environ.get('INLANE','0') == '1':
            CLANE = SLANE
        else:
            CLANE = '((u1*2+(u2&1))*8 + (u2>>1)*4+u3)'
    else:
        SLANE = 'ul'
        CLANE = 'ul'
    REALL = FG3
    NF8 = (N**3//8)*8
    RB = LANES*8                       # row bytes per (re or im) = LANES doubles
    em(f'''
// ================= L = {N} = {F} x {G} PFA digit engine =================
#define N{N} {N}
static double *S{N}_re, *S{N}_im;        // state: ROWSP x LANES doubles
static double *CT{N}_re, *CT{N}_im;      // c tiles: (ROWSP/8) x LANES x 8
static int32_t *SPOS{N};                 // flat spatial -> state pos
static int32_t *TPOS{N};
static int32_t *FS{N};                   // state pos (dense incl pads) -> 2*f (pads -> 0)
static int32_t *FT{N};                   // ctile pos -> 2*f (pads -> 0)
static double SCR{N}_re[{LANES}*8] __attribute__((aligned(64)));
static double SCR{N}_im[{LANES}*8] __attribute__((aligned(64)));
void init_{N}(void){{
  S{N}_re = huge_alloc({ROWSP}*{LANESP}*8);
  S{N}_im = huge_alloc({ROWSP}*{LANESP}*8);
  CT{N}_re = huge_alloc({ROWSP}*{LANESP}*8);
  CT{N}_im = huge_alloc({ROWSP}*{LANESP}*8);
  SPOS{N} = (int32_t*)huge_alloc({N**3}*4);
  TPOS{N} = (int32_t*)huge_alloc({N**3}*4);
  FS{N} = (int32_t*)huge_alloc({ROWSP}*{LANESP}*4);
  FT{N} = (int32_t*)huge_alloc({ROWSP}*{LANESP}*4);
  memset(FS{N}, 0, {ROWSP}*{LANESP}*4);
  memset(FT{N}, 0, {ROWSP}*{LANESP}*4);
  memset(S{N}_re, 0, {ROWSP}*{LANESP}*8);
  memset(S{N}_im, 0, {ROWSP}*{LANESP}*8);
  memset(CT{N}_re, 0, {ROWSP}*{LANESP}*8);
  memset(CT{N}_im, 0, {ROWSP}*{LANESP}*8);
  for(int x=0;x<{N};x++)for(int y=0;y<{N};y++)for(int z=0;z<{N};z++){{
    int u1=((x%{F})*{Ginv})%{F}, v1=((x%{G})*{Finv})%{G};
    int u2=((y%{F})*{Ginv})%{F}, v2=((y%{G})*{Finv})%{G};
    int u3=((z%{F})*{Ginv})%{F}, v3=((z%{G})*{Finv})%{G};
    int vlin=(v1*{G}+v2)*{G}+v3, ul=(u1*{F}+u2)*{F}+u3;
    int f=(x*{N}+y)*{N}+z;
    SPOS{N}[f] = vlin*{LANESP} + {SLANE};
    TPOS{N}[f] = vlin*{LANESP} + {CLANE};
    FS{N}[vlin*{LANESP} + {SLANE}] = 2*f;
    FT{N}[vlin*{LANESP} + {CLANE}] = 2*f;
  }}
}}''')
    # ---------------- phase A: three DFT_G passes, lane-chunked ----------------
    # pass over axis a (stride st rows), pencils for all other-row combos
    em(f'static void phaseA_{N}(void){{')
    em(C.loads() if False else '')
    body = Emitter()
    CA = Consts()
    # one codelet: given base row pointer expr and row stride (in doubles), lane offset
    body(f'''
  for(int lc=0;lc<{LANES};lc+=8){{
    // axis v3: stride 1 row; (v1,v2) outer
    for(int o=0;o<{G*G};o++){{
      double* br = S{N}_re + (size_t)o*{G}*{LANESP} + lc;
      double* bi = S{N}_im + (size_t)o*{G}*{LANESP} + lc;''')
    def pencil(body, stride_rows, pf=False):
        x=[]
        for j in range(G):
            if pf:
                body(f'_mm_prefetch((const char*)(br+{(j+G)*stride_rows}*{LANESP}), _MM_HINT_T0);')
                body(f'_mm_prefetch((const char*)(bi+{(j+G)*stride_rows}*{LANESP}), _MM_HINT_T0);')
            r = T(body, f'_mm512_load_pd(br+{j*stride_rows}*{LANESP})')
            i = T(body, f'_mm512_load_pd(bi+{j*stride_rows}*{LANESP})')
            x.append((r,i))
        X = dft9(body, CA, x)
        for k in range(G):
            body(f'_mm512_store_pd(br+{sG[k]*stride_rows}*{LANESP}, {X[k][0]});')
            body(f'_mm512_store_pd(bi+{sG[k]*stride_rows}*{LANESP}, {X[k][1]});')
    p1 = Emitter(); pencil(p1, 1, pf=(N==45 and os.environ.get('PF','1')=='1')); body(p1.out()); body('}')
    body(f'''
    // axis v2: stride {G} rows; outer (v1, v3)
    for(int v1=0;v1<{G};v1++)for(int v3=0;v3<{G};v3++){{
      double* br = S{N}_re + ((size_t)v1*{G*G}+v3)*{LANESP} + lc;
      double* bi = S{N}_im + ((size_t)v1*{G*G}+v3)*{LANESP} + lc;''')
    p2 = Emitter(); pencil(p2, G); body(p2.out()); body('}')
    body(f'''
    // axis v1: stride {G*G} rows; outer (v2, v3)
    for(int o=0;o<{G*G};o++){{
      double* br = S{N}_re + (size_t)o*{LANESP} + lc;
      double* bi = S{N}_im + (size_t)o*{LANESP} + lc;''')
    p3 = Emitter(); pencil(p3, G*G); body(p3.out()); body('}')
    body('}')
    em(CA.loads())
    em(body.out())
    em('}')
    # ---------------- phase B (in-register variant for F=4) ----------------
    if F == 4:
        CB = Consts()
        bb = Emitter()
        bb(f'''
  for(int r=0;r<{ROWS};r++){{
    double* pr = S{N}_re + (size_t)r*{LANESP};
    double* pi = S{N}_im + (size_t)r*{LANESP};
    const double* cr = CT{N}_re + (size_t)r*{LANESP};
    const double* ci = CT{N}_im + (size_t)r*{LANESP};''')
        # load row: 8 re + 8 im
        R = [T(bb, f'_mm512_load_pd(pr+{g}*8)') for g in range(8)]
        I = [T(bb, f'_mm512_load_pd(pi+{g}*8)') for g in range(8)]
        # u3-DFT4 vertical on groups {0..3} and {4..7}
        for b in range(2):
            X = dft4(bb, CB, [(R[b*4+j], I[b*4+j]) for j in range(4)])
            for k in range(4):
                R[b*4+k], I[b*4+k] = X[k]
        # u2-DFT4 pair trick on (g, g+4), lane parity
        for g in range(4):
            Pr, Pi, Qr, Qi = R[g], I[g], R[g+4], I[g+4]
            Sr = ADD(bb, Pr, Qr); Si = ADD(bb, Pi, Qi)
            Dr = SUB(bb, Pr, Qr); Di = SUB(bb, Pi, Qi)
            swSr = T(bb, f'_mm512_permute_pd({Sr}, 0x55)')
            swSi = T(bb, f'_mm512_permute_pd({Si}, 0x55)')
            swDr = T(bb, f'_mm512_permute_pd({Dr}, 0x55)')
            swDi = T(bb, f'_mm512_permute_pd({Di}, 0x55)')
            # masked combines: even lanes from S/swS, odd lanes from swD/D
            t1 = T(bb, f'_mm512_mask_add_pd({Sr}, 0x55, {Sr}, {swSr})')
            R[g] = T(bb, f'_mm512_mask_add_pd({t1}, 0xAA, {swDr}, {Di})')
            t2 = T(bb, f'_mm512_mask_add_pd({Si}, 0x55, {Si}, {swSi})')
            I[g] = T(bb, f'_mm512_mask_sub_pd({t2}, 0xAA, {swDi}, {Dr})')
            t3 = T(bb, f'_mm512_mask_sub_pd({Sr}, 0x55, {Sr}, {swSr})')
            R[g+4] = T(bb, f'_mm512_mask_sub_pd({t3}, 0xAA, {swDr}, {Di})')
            t4 = T(bb, f'_mm512_mask_sub_pd({Si}, 0x55, {Si}, {swSi})')
            I[g+4] = T(bb, f'_mm512_mask_add_pd({t4}, 0xAA, {swDi}, {Dr})')
        if os.environ.get('INLANE','0') == '1':
            # u1-DFT4 entirely in-lane: lanes q = u1*2 + u2lo
            s1 = CB.get('dummy_s1', 1.0)  # placeholder to keep Consts non-empty
            bb('const __m512d SGN1 = _mm512_set_pd(-1.,-1.,-1.,-1.,1.,1.,1.,1.);')
            bb('const __m512d SGNA = _mm512_set_pd(-1.,-1.,1.,1.,-1.,-1.,1.,1.);')
            bb('const __m512d SGNBre = _mm512_set_pd(1.,1.,-1.,-1.,-1.,-1.,1.,1.);')
            bb('const __m512d SGNBim = _mm512_set_pd(-1.,-1.,-1.,-1.,1.,1.,1.,1.);')
            bb('const __m512i IDXA = _mm512_set_epi64(5,4,1,0,5,4,1,0);')
            bb('const __m512i IDXB = _mm512_set_epi64(15,14,3,2,15,14,3,2);')
            for g in range(8):
                vr, vi = R[g], I[g]
                swr = T(bb, f'_mm512_shuffle_f64x2({vr},{vr},0x4E)')
                swi = T(bb, f'_mm512_shuffle_f64x2({vi},{vi},0x4E)')
                Mr = T(bb, f'_mm512_fmadd_pd({swr}, SGN1, {vr})')
                Mi = T(bb, f'_mm512_fmadd_pd({swi}, SGN1, {vi})')
                ar = T(bb, f'_mm512_permutexvar_pd(IDXA, {Mr})')
                ai = T(bb, f'_mm512_permutexvar_pd(IDXA, {Mi})')
                br = T(bb, f'_mm512_permutex2var_pd({Mr}, IDXB, {Mi})')
                bi = T(bb, f'_mm512_permutex2var_pd({Mi}, IDXB, {Mr})')
                R[g] = T(bb, f'_mm512_fmadd_pd({ar}, SGNA, _mm512_mul_pd({br}, SGNBre))')
                I[g] = T(bb, f'_mm512_fmadd_pd({ai}, SGNA, _mm512_mul_pd({bi}, SGNBim))')
            # c + map in A-convention, store directly
            subs = []
            for g in range(8):
                bu = Emitter()
                zr = T(bu, f'_mm512_add_pd({R[g]}, _mm512_load_pd(cr+{g}*8))')
                zi = T(bu, f'_mm512_add_pd({I[g]}, _mm512_load_pd(ci+{g}*8))')
                bu(f'__m512d mr{g}, mi{g};')
                emit_map(bu, CB, zr, zi, f'mr{g}', f'mi{g}')
                bu(f'_mm512_store_pd(pr+{g}*8, mr{g});')
                bu(f'_mm512_store_pd(pi+{g}*8, mi{g});')
                subs.append((bu.out().split(chr(10)), g))
            mx = max(len(s) for s,_ in subs)
            for rr in range(mx):
                for s,_ in subs:
                    if rr < len(s): bb(s[rr])
            bb('}')
        else:
            # transpose re, im
            TR = [f'tr{g}' for g in range(8)]; TI = [f'ti{g}' for g in range(8)]
            for v in TR+TI: bb(f'__m512d {v};')
            emit_transpose8(bb, R, TR)
            emit_transpose8(bb, I, TI)
            # u1-DFT4 vertical: groups {0,2,4,6} and {1,3,5,7}
            for par in range(2):
                X = dft4(bb, CB, [(TR[par+2*j], TI[par+2*j]) for j in range(4)])
                for k in range(4):
                    TR[par+2*k], TI[par+2*k] = X[k]
            # c + map (c in B-convention)
            subs = []
            for g in range(8):
                bu = Emitter()
                zr = T(bu, f'_mm512_add_pd({TR[g]}, _mm512_load_pd(cr+{g}*8))')
                zi = T(bu, f'_mm512_add_pd({TI[g]}, _mm512_load_pd(ci+{g}*8))')
                bu(f'__m512d mr{g}, mi{g};')
                emit_map(bu, CB, zr, zi, f'mr{g}', f'mi{g}')
                subs.append((bu.out().split(chr(10)), g))
            mx = max(len(s) for s,_ in subs)
            for rr in range(mx):
                for s,_ in subs:
                    if rr < len(s): bb(s[rr])
            MR = [f'mr{g}' for g in range(8)]; MI = [f'mi{g}' for g in range(8)]
            # transpose back and store
            OR_ = [f'or{g}' for g in range(8)]; OI = [f'oi{g}' for g in range(8)]
            for v in OR_+OI: bb(f'__m512d {v};')
            emit_transpose8(bb, MR, OR_)
            emit_transpose8(bb, MI, OI)
            for g in range(8):
                bb(f'_mm512_store_pd(pr+{g}*8, {OR_[g]});')
                bb(f'_mm512_store_pd(pi+{g}*8, {OI[g]});')
            bb('}')
        em(f'static void phaseB_{N}(void){{')
        em(CB.loads())
        em(bb.out())
        em('}')

    # ---------------- phase B ----------------
    if F == 4:
        CBx = None
    CB = Consts()
    bb = Emitter()
    nch = ROWSP//8
    bb(f'''
  for(int vc=0;vc<{nch};vc++){{
    double* pr = S{N}_re + (size_t)vc*8*{LANESP};
    double* pi = S{N}_im + (size_t)vc*8*{LANESP};
    const double* cr = CT{N}_re + (size_t)vc*8*{LANESP};
    const double* ci = CT{N}_im + (size_t)vc*8*{LANESP};''')
    dftF = {4:dft4, 5:dft5}[F]
    # --- build tin blocks (unrolled over ub) and u3 pencils, then weave ---
    tin_blocks = []
    for ub in range(LANES//8):
        be = Emitter()
        for arr in ('re','im'):
            sp = 'pr' if arr=='re' else 'pi'
            tin = [T(be, f'_mm512_load_pd({sp}+{r}*{LANESP}+{ub}*8)') for r in range(8)]
            outv = [f'o{arr}{ub}_{j}' for j in range(8)]
            for o in outv: be(f'__m512d {o};')
            emit_transpose8(be, tin, outv)
            for j in range(8):
                be(f'_mm512_store_pd(SCR{N}_{arr}+({ub*8+j})*8, {outv[j]});')
        tin_blocks.append(be.out().split(chr(10)))
    u3_pencils = []
    for p in range(F*F):
        be = Emitter()
        base = p*F
        x=[]
        for j in range(F):
            r = T(be, f'_mm512_load_pd(SCR{N}_re+{(j+base)}*8)')
            i = T(be, f'_mm512_load_pd(SCR{N}_im+{(j+base)}*8)')
            x.append((r,i))
        X = dftF(be, CB, x)
        for k in range(F):
            tgt = sF[k]*1
            be(f'_mm512_store_pd(SCR{N}_re+({tgt}+{base})*8, {X[k][0]});')
            be(f'_mm512_store_pd(SCR{N}_im+({tgt}+{base})*8, {X[k][1]});')
        u3_pencils.append((base+F-1, be.out().split(chr(10))))  # ready when rows <= base+F-1 transposed
    # weave: maintain two cursors; a pencil can start once tin blocks covering its rows are emitted
    tin_idx = 0; pen_idx = 0
    tin_line = 0; pen_line = 0
    cover = -1  # highest scratch row transposed
    out_lines = []
    while tin_idx < len(tin_blocks) or pen_idx < len(u3_pencils):
        # alternate: emit a few tin lines then a few pencil lines (if ready)
        can_pen = pen_idx < len(u3_pencils) and u3_pencils[pen_idx][0] <= cover
        if tin_idx < len(tin_blocks):
            blk = tin_blocks[tin_idx]
            n = 3 if can_pen else len(blk) - tin_line
            for _ in range(n):
                if tin_line < len(blk):
                    out_lines.append(blk[tin_line]); tin_line += 1
            if tin_line >= len(blk):
                tin_idx += 1; tin_line = 0; cover = tin_idx*8 - 1
        if can_pen:
            blk = u3_pencils[pen_idx][1]
            for _ in range(4):
                if pen_line < len(blk):
                    out_lines.append(blk[pen_line]); pen_line += 1
            if pen_line >= len(blk):
                pen_idx += 1; pen_line = 0
        elif tin_idx >= len(tin_blocks) and pen_idx < len(u3_pencils):
            # flush remaining pencils
            blk = u3_pencils[pen_idx][1]
            out_lines.extend(blk[pen_line:]); pen_idx += 1; pen_line = 0
    for ln in out_lines: bb(ln)
    def fpass(bb, stride, fuse_map):
        x=[]
        for j in range(F):
            r = T(bb, f'_mm512_load_pd(SCR{N}_re+({j}*{stride}+base)*8)')
            i = T(bb, f'_mm512_load_pd(SCR{N}_im+({j}*{stride}+base)*8)')
            x.append((r,i))
        X = dftF(bb, CB, x)
        if not fuse_map:
            for k in range(F):
                tgt = sF[k]*stride
                bb(f'_mm512_store_pd(SCR{N}_re+({tgt}+base)*8, {X[k][0]});')
                bb(f'_mm512_store_pd(SCR{N}_im+({tgt}+base)*8, {X[k][1]});')
        else:
            subs = []
            for k in range(F):
                tgt = sF[k]*stride
                bu = Emitter()
                zr = T(bu, f'_mm512_add_pd({X[k][0]}, _mm512_load_pd(cr+({tgt}+base)*8))')
                zi = T(bu, f'_mm512_add_pd({X[k][1]}, _mm512_load_pd(ci+({tgt}+base)*8))')
                bu(f'__m512d mr{k}, mi{k};')
                emit_map(bu, CB, zr, zi, f'mr{k}', f'mi{k}')
                bu(f'_mm512_store_pd(SCR{N}_re+({tgt}+base)*8, mr{k});')
                bu(f'_mm512_store_pd(SCR{N}_im+({tgt}+base)*8, mi{k});')
                subs.append(bu.out().split(chr(10)))
            mx = max(len(s) for s in subs)
            for row in range(mx):
                for s in subs:
                    if row < len(s): bb(s[row])
    # u3 pass woven into tin above

    # pass u2: stride F, base = u1*F*F + u3
    bb(f'for(int u1=0;u1<{F};u1++)for(int u3=0;u3<{F};u3++){{ int base=u1*{F*F}+u3;')
    fpass(bb, F, False); bb('}')
    # pass u1: stride F*F, base = u2*F+u3
    bb(f'for(int o=0;o<{F*F};o++){{ int base=o;')
    fpass(bb, F*F, False); bb('}')
    # transpose back + c + map fused; software-pipelined across ub blocks
    tb_blocks = []
    map_blocks = []
    for ub in range(LANES//8):
        be = Emitter()
        for arr in ('re','im'):
            tin = [T(be, f'_mm512_load_pd(SCR{N}_{arr}+({ub*8+j})*8)') for j in range(8)]
            outv = [f'b{arr}{ub}_{j}' for j in range(8)]
            for o in outv: be(f'__m512d {o};')
            emit_transpose8(be, tin, outv)
        tb_blocks.append(be.out().split(chr(10)))
        me_ = Emitter()
        msubs = []
        for r in range(8):
            bu = Emitter()
            zr = T(bu, f'_mm512_add_pd(bre{ub}_{r}, _mm512_load_pd(cr+{r}*{LANESP}+{ub}*8))')
            zi = T(bu, f'_mm512_add_pd(bim{ub}_{r}, _mm512_load_pd(ci+{r}*{LANESP}+{ub}*8))')
            bu(f'__m512d fr{ub}_{r}, fi{ub}_{r};')
            emit_map(bu, CB, zr, zi, f'fr{ub}_{r}', f'fi{ub}_{r}')
            bu(f'_mm512_store_pd(pr+{r}*{LANESP}+{ub}*8, fr{ub}_{r});')
            bu(f'_mm512_store_pd(pi+{r}*{LANESP}+{ub}*8, fi{ub}_{r});')
            msubs.append(bu.out().split(chr(10)))
        mmx = max(len(s) for s in msubs)
        for row in range(mmx):
            for s in msubs:
                if row < len(s): me_(s[row])
        map_blocks.append(me_.out().split(chr(10)))
    import os as _os
    WEAVE = _os.environ.get('WEAVE','0')=='1'
    if WEAVE:
        for ln in tb_blocks[0]: bb(ln)
        for u in range(1, LANES//8):
            a = tb_blocks[u]; b2 = map_blocks[u-1]
            ia = ib = 0
            while ia < len(a) or ib < len(b2):
                for _ in range(2):
                    if ia < len(a): bb(a[ia]); ia += 1
                for _ in range(3):
                    if ib < len(b2): bb(b2[ib]); ib += 1
        for ln in map_blocks[LANES//8-1]: bb(ln)
    else:
        for u in range(LANES//8):
            for ln in tb_blocks[u]: bb(ln)
            for ln in map_blocks[u]: bb(ln)
    bb('}')
    if F != 4:
        em(f'static void phaseB_{N}(void){{')
        em(CB.loads())
        em(bb.out())
        em('}')
    # ---------------- conversions + driver ----------------
    em(f'''
static void convin_{N}(const double* x0){{
  for(int f=0;f<{N**3};f++){{
    int sp = SPOS{N}[f];
    _mm_prefetch((const char*)(S{N}_re + SPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    _mm_prefetch((const char*)(S{N}_im + SPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    S{N}_re[sp] = x0[2*f];
    S{N}_im[sp] = x0[2*f+1];
  }}
}}
static void convc_{N}(const double* c){{
  for(int f=0;f<{N**3};f++){{
    int tp = TPOS{N}[f];
    _mm_prefetch((const char*)(CT{N}_re + TPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    _mm_prefetch((const char*)(CT{N}_im + TPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    CT{N}_re[tp] = c[2*f];
    CT{N}_im[tp] = c[2*f+1];
  }}
}}
static void convout_{N}(double* out){{
  for(int f=0;f<{N**3};f++){{
    int sp = SPOS{N}[f];
    _mm_prefetch((const char*)(S{N}_re + SPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    _mm_prefetch((const char*)(S{N}_im + SPOS{N}[f+24 < {N**3} ? f+24 : f]), _MM_HINT_T0);
    out[2*f]   = S{N}_re[sp];
    out[2*f+1] = S{N}_im[sp];
  }}
}}
uint64_t bench_{N}(int which, long reps){{
  uint64_t t0 = __rdtsc();
  for(long r=0;r<reps;r++){{ if(which==0) phaseA_{N}(); else phaseB_{N}(); }}
  return __rdtsc() - t0;
}}
void run_{N}(const double* x0, const double* c, double* one, double* fin, long B, long m){{
  for(long b=0;b<B;b++){{
    convin_{N}(x0 + b*2*{N**3});
    convc_{N}(c + b*2*{N**3});
    for(long s=0;s<m;s++){{
      phaseA_{N}();
      phaseB_{N}();
      if(s==0) convout_{N}(one + b*2*{N**3});
    }}
    convout_{N}(fin + b*2*{N**3});
  }}
}}''')
    pre = Emitter()
    C.decl(pre); CA.decl(pre); CB.decl(pre)
    return pre.out() + em.out()

PRELUDE = r'''
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
static double* huge_alloc(size_t bytes){
  size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20)-1);
  void* p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if(p == MAP_FAILED){ abort(); }
  madvise(p, sz, MADV_HUGEPAGE);
  memset(p, 0, sz);
  return (double*)p;
}
'''

if __name__ == '__main__':
    src = PRELUDE
    src += gen_engine(4, 9)
    src += gen_engine(5, 9)
    open('/tmp/g/impl_fg.c','w').write(src)
    print('wrote impl_fg.c', len(src))
