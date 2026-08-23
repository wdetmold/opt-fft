import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')

def hexd(v):
    return float(v).hex()

def tw(N, k):
    # exp(-2*pi*i*k/N) with exact mod-N reduction, longdouble precision
    a = (-2*PI) * LD(k % N) / LD(N)
    return float(np.cos(a)), float(np.sin(a))

class E:
    def __init__(self):
        self.lines = []
        self.n = 0
    def v(self, init=None):
        self.n += 1
        name = f"t{self.n}"
        if init is not None:
            self.lines.append(f"__m512d {name} = {init};")
        return name
    def raw(self, s):
        self.lines.append(s)

def setc(e, val):
    return f"_mm512_set1_pd({hexd(val)})"

def add(e,a,b): return e.v(f"_mm512_add_pd({a}, {b})")
def sub(e,a,b): return e.v(f"_mm512_sub_pd({a}, {b})")
def mul(e,a,b): return e.v(f"_mm512_mul_pd({a}, {b})")
def fmadd(e,a,b,c): return e.v(f"_mm512_fmadd_pd({a}, {b}, {c})")
def fnmadd(e,a,b,c): return e.v(f"_mm512_fnmadd_pd({a}, {b}, {c})")
def fmsub(e,a,b,c): return e.v(f"_mm512_fmsub_pd({a}, {b}, {c})")

def cadd(e,x,y): return (add(e,x[0],y[0]), add(e,x[1],y[1]))
def csub(e,x,y): return (sub(e,x[0],y[0]), sub(e,x[1],y[1]))

def cmulw(e, x, N, k):
    # y = x * exp(-2 pi i k / N)
    k = k % N
    if k == 0: return x
    wr, wi = tw(N,k)
    if abs(wr-1)<1e-30 and abs(wi)<1e-30: return x
    xr, xi = x
    # yr = wr*xr - wi*xi ; yi = wr*xi + wi*xr
    WR = setc(e, wr); WI = setc(e, wi)
    t  = mul(e, xr, WR)
    yr = fnmadd(e, xi, WI, t) if wi != 0.0 else t
    u  = mul(e, xi, WR)
    yi = fmadd(e, xr, WI, u) if wi != 0.0 else u
    return (yr, yi)

def cmuli(e, x, sign):
    # y = sign * i * x  -> (−sign*xi, sign*xr) conceptually: i*x = (-xi, xr)
    # return names without ops (caller combines); here realize via tuple swap markers
    raise RuntimeError("use combine forms")

# ---- small DFT builders on lists of complex pairs ----
def dft2(e, x):
    return [cadd(e,x[0],x[1]), csub(e,x[0],x[1])]

def dft3(e, x):
    # y0 = x0+t, y1 = v + i*s*u ... forward with w3 = c + i*s, c=-1/2, s=-sqrt(3)/2
    c = -0.5
    S = float(np.sqrt(LD(3))/2)  # positive sqrt3/2 ; w = c - i*S (forward)
    t = cadd(e, x[1], x[2])
    u = csub(e, x[1], x[2])
    y0 = cadd(e, x[0], t)
    C = setc(e, c)
    vr = fmadd(e, C, t[0], x[0][0]); vi = fmadd(e, C, t[1], x[0][1])
    # y1 = v - i*S*u : re = vr + (-i S u).re = vr - S*(i u).re... i*u = (-ui, ur); -i S u = (S*ui, -S*ur)
    SS = setc(e, S)
    y1r = fmadd(e, SS, u[1], vr); y1i = fnmadd(e, SS, u[0], vi)
    y2r = fnmadd(e, SS, u[1], vr); y2i = fmadd(e, SS, u[0], vi)
    return [y0, (y1r,y1i), (y2r,y2i)]

def dft4(e, x):
    t0 = cadd(e,x[0],x[2]); t1 = csub(e,x[0],x[2])
    t2 = cadd(e,x[1],x[3]); t3 = csub(e,x[1],x[3])
    y0 = cadd(e,t0,t2); y2 = csub(e,t0,t2)
    # y1 = t1 - i t3 ; y3 = t1 + i t3 ; (-i t3) = (t3i, -t3r)
    y1 = (add(e,t1[0],t3[1]), sub(e,t1[1],t3[0]))
    y3 = (sub(e,t1[0],t3[1]), add(e,t1[1],t3[0]))
    return [y0,y1,y2,y3]

def dft5(e, x):
    c1, s1 = tw(5,1); c2, s2 = tw(5,2)  # s negative values (sin of -2pi k/5)
    # use classic: w^k = c_k + i s_k with s_k = sin(-2pi k/5) (negative)
    t1 = cadd(e,x[1],x[4]); t2 = cadd(e,x[2],x[3])
    t3 = csub(e,x[1],x[4]); t4 = csub(e,x[2],x[3])
    y0 = cadd(e, x[0], cadd(e,t1,t2))
    C1 = setc(e,c1); C2 = setc(e,c2)
    a1r = fmadd(e,C2,t2[0], fmadd(e,C1,t1[0],x[0][0]))
    a1i = fmadd(e,C2,t2[1], fmadd(e,C1,t1[1],x[0][1]))
    a2r = fmadd(e,C1,t2[0], fmadd(e,C2,t1[0],x[0][0]))
    a2i = fmadd(e,C1,t2[1], fmadd(e,C2,t1[1],x[0][1]))
    S1 = setc(e,s1); S2 = setc(e,s2)
    # B1 = s1*t3 + s2*t4 (complex, times i): y1 = A1 + i*B1, y4 = A1 - i*B1
    b1r = fmadd(e,S2,t4[0], mul(e,S1,t3[0])); b1i = fmadd(e,S2,t4[1], mul(e,S1,t3[1]))
    b2r = fnmadd(e,S1,t4[0], mul(e,S2,t3[0])); b2i = fnmadd(e,S1,t4[1], mul(e,S2,t3[1]))
    # i*B = (-Bi, Br)
    y1 = (sub(e,a1r,b1i), add(e,a1i,b1r))
    y4 = (add(e,a1r,b1i), sub(e,a1i,b1r))
    y2 = (sub(e,a2r,b2i), add(e,a2i,b2r))
    y3 = (add(e,a2r,b2i), sub(e,a2i,b2r))
    return [y0,y1,y2,y3,y4]

def dft8(e, x):
    Ev = dft4(e, [x[0],x[2],x[4],x[6]])
    Od = dft4(e, [x[1],x[3],x[5],x[7]])
    y = [None]*8
    y[0] = cadd(e,Ev[0],Od[0]); y[4] = csub(e,Ev[0],Od[0])
    # w8^1 * O1 : w = s - i s , s = sqrt(2)/2  (cos(-pi/4)=s, sin(-pi/4)=-s)
    s = float(np.sqrt(LD(2))/2)
    S = setc(e, s)
    o1r, o1i = Od[1]
    w1r = mul(e, S, add(e,o1r,o1i))         # s*(or+oi)
    w1i = mul(e, S, sub(e,o1i,o1r))         # s*(oi-or)
    y[1] = cadd(e,Ev[1],(w1r,w1i)); y[5] = csub(e,Ev[1],(w1r,w1i))
    # w8^2 = -i : (-i O2) = (o2i, -o2r)
    o2r,o2i = Od[2]
    y[2] = (add(e,Ev[2][0],o2i), sub(e,Ev[2][1],o2r))
    y[6] = (sub(e,Ev[2][0],o2i), add(e,Ev[2][1],o2r))
    # w8^3 = -s - i s : w3*O3 = -s(or - oi) - i s(oi + or) => re = s*(oi-or), im = -s*(or+oi)
    o3r,o3i = Od[3]
    w3r = mul(e, S, sub(e,o3i,o3r))
    w3i = mul(e, S, add(e,o3r,o3i))  # then negate by using sub in combine
    y[3] = (add(e,Ev[3][0],w3r), sub(e,Ev[3][1],w3i))
    y[7] = (sub(e,Ev[3][0],w3r), add(e,Ev[3][1],w3i))
    return y

def dft9(e, x):
    # 3x3 CT: A_r = dft3(x[r],x[r+3],x[r+6]) ; B_r[d] = A_r[d]*w9^{rd} ; out[3c+d] = dft3_c(B_0[d],B_1[d],B_2[d])
    A = [dft3(e, [x[r],x[r+3],x[r+6]]) for r in range(3)]
    B = [[cmulw(e, A[r][d], 9, r*d) for d in range(3)] for r in range(3)]
    y = [None]*9
    for d in range(3):
        col = dft3(e, [B[0][d],B[1][d],B[2][d]])
        for c in range(3):
            y[3*c+d] = col[c]
    return y

def dft_generic(e, x, N):
    if N==2: return dft2(e,x)
    if N==3: return dft3(e,x)
    if N==4: return dft4(e,x)
    if N==5: return dft5(e,x)
    if N==8: return dft8(e,x)
    if N==9: return dft9(e,x)
    raise RuntimeError(N)

# ---------- codelet function generators ----------

def loadv(e, base, off):
    if off == 0: return e.v(f"_mm512_load_pd({base})")
    return e.v(f"_mm512_load_pd({base} + {off})")

def storev(e, base, off, v):
    if off == 0: e.raw(f"_mm512_store_pd({base}, {v});")
    else: e.raw(f"_mm512_store_pd({base} + {off}, {v});")

def gen_pfa2(name, N, N1, N2, ai=False):
    # N = N1*N2 coprime. in index n=(a1*n1 + a2*n2)%N with a1 = N2*? ... use n=(N2*n1+N1*n2)%N? need exponent match:
    # standard: n = (N2*n1 + N1*n2) % N ; k = (N2*u1*k1 + N1*u2*k2) % N with N2*u1 = 1 mod N1, N1*u2 = 1 mod N2
    # check: n*k = N2^2 u1 n1 k1 + N1^2 u2 n2 k2 (mod N cross terms vanish: N1*N2*stuff)
    #   N2^2 u1 n1k1 mod N: we need = N2*n1k1*? hmm verify numerically below instead.
    u1 = pow(N2, -1, N1)
    u2 = pow(N1, -1, N2)
    def nin(n1,n2): return (N2*n1 + N1*n2) % N
    def kout(k1,k2): return (N2*u1*k1 + N1*u2*k2) % N
    # verify PFA identity numerically on indices
    for k1 in range(N1):
        for k2 in range(N2):
            for n1 in range(N1):
                for n2 in range(N2):
                    lhs = (nin(n1,n2)*kout(k1,k2)) % N
                    rhs = (N2*u1*N2*n1*k1 + N1*u2*N1*n2*k2) % N
                    # exponent must equal n1k1/N1 + n2k2/N2 (mod 1): i.e. (N2*n1k1 + N1*n2k2)*? 
            # skip - verified by separate functional test
    e = E()
    e.raw(f"static __attribute__((always_inline)) inline void {name}(double* re, double* im, long es){{")
    # load all
    x = [None]*N
    for n1 in range(N1):
        for n2 in range(N2):
            n = nin(n1,n2)
            xr = loadv(e, "re", f"{n}*es"); xi = loadv(e, "im", f"{n}*es")
            if x[0] is None and False: pass
            if n1==0 and n2==0: pass
            x[n1*N2+n2] = (xr,xi)   # store by (n1,n2)
    # first stage: DFT_N1 over n1 for each n2  -- but exponent: verify mapping works via w_N1^{n1 k1 * (N2 u1 N2 /N1?)}...
    # We rely on standard PFA with these maps; validate externally.
    A = [[None]*N2 for _ in range(N1)]
    for n2 in range(N2):
        col = dft_generic(e, [x[n1*N2+n2] for n1 in range(N1)], N1)
        for k1 in range(N1):
            A[k1][n2] = col[k1]
    for k1 in range(N1):
        col = dft_generic(e, A[k1], N2)
        for k2 in range(N2):
            k = kout(k1,k2)
            storev(e, "re", f"{k}*es", col[k2][0])
            storev(e, "im", f"{k}*es", col[k2][1])
    e.raw("}")
    return "\n".join(e.lines)




def gen_hartley(name, N, chunks, scratch):
    h = (N-1)//2
    e = E()
    tab = []
    for kc in chunks:
        for j in range(1,h+1):
            for k in kc:
                ang = 2*PI*LD((k*j) % N)/LD(N)
                tab.append(float(np.cos(ang)))
                tab.append(float(np.sin(ang)))
    e.raw(f"static const double HT_{N}[{len(tab)}] ALIGN64 = {{" + ",".join(hexd(v) for v in tab) + "};")
    if scratch:
        e.raw(f"static double HS_{N}[{h}][32] ALIGN64;")  # s_re s_im d_re d_im per j
    e.raw(f"static void {name}(double* re, double* im, long es){{")
    x0r = loadv(e,"re","0"); x0i = loadv(e,"im","0")
    if scratch:
        o0r, o0i = None, None
        for j in range(1,h+1):
            ar = loadv(e,"re",f"{j}*es"); ai = loadv(e,"im",f"{j}*es")
            br = loadv(e,"re",f"{N-j}*es"); bi = loadv(e,"im",f"{N-j}*es")
            sr = add(e,ar,br); si = add(e,ai,bi)
            dr = sub(e,ar,br); di = sub(e,ai,bi)
            e.raw(f"_mm512_store_pd(HS_{N}[{j-1}], {sr}); _mm512_store_pd(HS_{N}[{j-1}]+8, {si});")
            e.raw(f"_mm512_store_pd(HS_{N}[{j-1}]+16, {dr}); _mm512_store_pd(HS_{N}[{j-1}]+24, {di});")
            if o0r is None: o0r, o0i = sr, si
            else: o0r = add(e,o0r,sr); o0i = add(e,o0i,si)
        o0r = add(e,o0r,x0r); o0i = add(e,o0i,x0i)
        storev(e,"re","0",o0r); storev(e,"im","0",o0i)
        e.raw('__asm__ volatile("" ::: "memory");')
    ti = 0
    first = True
    for kc in chunks:
        if scratch: e.raw('__asm__ volatile("" ::: "memory");')
        accnames = {}
        for k in kc:
            ar_ = f"accAr{k}"; ai_ = f"accAi{k}"; br_ = f"accBr{k}"; bi_ = f"accBi{k}"
            e.raw(f"__m512d {ar_} = {x0r}, {ai_} = {x0i}, {br_}, {bi_};")
            accnames[k] = (ar_, ai_, br_, bi_)
        if (not scratch) and first:
            e.raw(f"__m512d o0r = {x0r}, o0i = {x0i};")
        for j in range(1,h+1):
            if scratch:
                sr = e.v(f"_mm512_load_pd(HS_{N}[{j-1}])"); si = e.v(f"_mm512_load_pd(HS_{N}[{j-1}]+8)")
                dr = e.v(f"_mm512_load_pd(HS_{N}[{j-1}]+16)"); di = e.v(f"_mm512_load_pd(HS_{N}[{j-1}]+24)")
            else:
                ar = loadv(e,"re",f"{j}*es"); ai = loadv(e,"im",f"{j}*es")
                br = loadv(e,"re",f"{N-j}*es"); bi = loadv(e,"im",f"{N-j}*es")
                sr = add(e,ar,br); si = add(e,ai,bi)
                dr = sub(e,ar,br); di = sub(e,ai,bi)
                if first:
                    e.raw(f"o0r = _mm512_add_pd(o0r, {sr}); o0i = _mm512_add_pd(o0i, {si});")
            import os as _os
            style = _os.environ.get("HSTYLE","bcastv")
            for idx,k in enumerate(kc):
                ar_, ai_, br_, bi_ = accnames[k]
                if style == "embed":
                    e.raw(f"FMA_BC({ar_}, {sr}, HT_{N}[{ti}]);")
                    e.raw(f"FMA_BC({ai_}, {si}, HT_{N}[{ti}]);")
                    if j==1:
                        e.raw(f"MUL_BC({br_}, {di}, HT_{N}[{ti+1}]);")
                        e.raw(f"MUL_BC({bi_}, {dr}, HT_{N}[{ti+1}]);")
                    else:
                        e.raw(f"FMA_BC({br_}, {di}, HT_{N}[{ti+1}]);")
                        e.raw(f"FMA_BC({bi_}, {dr}, HT_{N}[{ti+1}]);")
                    ti += 2
                    continue
                bc = f"bc{j}_{k}_c"; bs = f"bc{j}_{k}_s"
                e.raw(f"__m512d {bc}, {bs};")
                if style == "bcastv":
                    e.raw(f"BCASTV({bc}, HT_{N}[{ti}]); BCASTV({bs}, HT_{N}[{ti+1}]);")
                else:
                    e.raw(f"BCAST({bc}, HT_{N}[{ti}]); BCAST({bs}, HT_{N}[{ti+1}]);")
                e.raw(f"{ar_} = _mm512_fmadd_pd({bc}, {sr}, {ar_});")
                e.raw(f"{ai_} = _mm512_fmadd_pd({bc}, {si}, {ai_});")
                if j==1:
                    e.raw(f"{br_} = _mm512_mul_pd({bs}, {di});")
                    e.raw(f"{bi_} = _mm512_mul_pd({bs}, {dr});")
                else:
                    e.raw(f"{br_} = _mm512_fmadd_pd({bs}, {di}, {br_});")
                    e.raw(f"{bi_} = _mm512_fmadd_pd({bs}, {dr}, {bi_});")
                ti += 2
        if (not scratch) and first:
            e.raw(f'storev_dummy();')
            e.lines.pop()
            e.raw(f"_mm512_store_pd(re, o0r); _mm512_store_pd(im, o0i);")
        first = False
        for k in kc:
            ar_, ai_, br_, bi_ = accnames[k]
            yr = e.v(f"_mm512_add_pd({ar_}, {br_})"); yi = e.v(f"_mm512_sub_pd({ai_}, {bi_})")
            zr = e.v(f"_mm512_sub_pd({ar_}, {br_})"); zi = e.v(f"_mm512_add_pd({ai_}, {bi_})")
            storev(e,"re",f"{k}*es",yr); storev(e,"im",f"{k}*es",yi)
            storev(e,"re",f"{N-k}*es",zr); storev(e,"im",f"{N-k}*es",zi)
    e.raw("}")
    return "\n".join(e.lines)



def gen_hartley_pair(name, N, ck=2):
    """two-column hartley via s/d scratch; columns at +po doubles."""
    h = (N-1)//2
    e = E()
    ks = list(range(1,h+1))
    chunks = [ks[i:i+ck] for i in range(0,h,ck)]
    tab = []
    for kc in chunks:
        for j in range(1,h+1):
            for k in kc:
                ang = 2*PI*LD((k*j) % N)/LD(N)
                tab.append(float(np.cos(ang)))
                tab.append(float(np.sin(ang)))
    e.raw(f"static const double HP_{N}[{len(tab)}] ALIGN64 = {{" + ",".join(hexd(v) for v in tab) + "};")
    e.raw(f"static double HPS_{N}[{h}][64] ALIGN64;")
    e.raw(f"static void {name}(double* re, double* im, long es, long po){{")
    for col in (0,1):
        off = "" if col==0 else "+po"
        e.raw(f"__m512d x0r{col} = _mm512_load_pd(re{off});")
        e.raw(f"__m512d x0i{col} = _mm512_load_pd(im{off});")
        e.raw(f"__m512d o0r{col} = x0r{col}, o0i{col} = x0i{col};")
    for j in range(1,h+1):
        for col in (0,1):
            off = "" if col==0 else " + po"
            ar = e.v(f"_mm512_load_pd(re + {j}*es{off})"); ai = e.v(f"_mm512_load_pd(im + {j}*es{off})")
            br = e.v(f"_mm512_load_pd(re + {N-j}*es{off})"); bi = e.v(f"_mm512_load_pd(im + {N-j}*es{off})")
            sr = add(e,ar,br); si = add(e,ai,bi)
            dr = sub(e,ar,br); di = sub(e,ai,bi)
            e.raw(f"_mm512_store_pd(HPS_{N}[{j-1}]+{col*32}, {sr}); _mm512_store_pd(HPS_{N}[{j-1}]+{col*32}+8, {si});")
            e.raw(f"_mm512_store_pd(HPS_{N}[{j-1}]+{col*32}+16, {dr}); _mm512_store_pd(HPS_{N}[{j-1}]+{col*32}+24, {di});")
            e.raw(f"o0r{col} = _mm512_add_pd(o0r{col}, {sr}); o0i{col} = _mm512_add_pd(o0i{col}, {si});")
    e.raw("_mm512_store_pd(re, o0r0); _mm512_store_pd(im, o0i0);")
    e.raw("_mm512_store_pd(re + po, o0r1); _mm512_store_pd(im + po, o0i1);")
    ti = 0
    for kc in chunks:
        for k in kc:
            for col in (0,1):
                e.raw(f"__m512d Ar{k}_{col} = x0r{col}, Ai{k}_{col} = x0i{col}, Br{k}_{col}, Bi{k}_{col};")
        for j in range(1,h+1):
            names = {}
            for col in (0,1):
                sr = e.v(f"_mm512_load_pd(HPS_{N}[{j-1}]+{col*32})"); si = e.v(f"_mm512_load_pd(HPS_{N}[{j-1}]+{col*32}+8)")
                dr = e.v(f"_mm512_load_pd(HPS_{N}[{j-1}]+{col*32}+16)"); di = e.v(f"_mm512_load_pd(HPS_{N}[{j-1}]+{col*32}+24)")
                names[col] = (sr,si,dr,di)
            for k in kc:
                e.raw(f"__m512d bc{j}_{k}, bs{j}_{k};")
                e.raw(f"BCAST(bc{j}_{k}, HP_{N}[{ti}]); BCAST(bs{j}_{k}, HP_{N}[{ti+1}]);")
                for col in (0,1):
                    sr,si,dr,di = names[col]
                    e.raw(f"Ar{k}_{col} = _mm512_fmadd_pd(bc{j}_{k}, {sr}, Ar{k}_{col});")
                    e.raw(f"Ai{k}_{col} = _mm512_fmadd_pd(bc{j}_{k}, {si}, Ai{k}_{col});")
                    if j==1:
                        e.raw(f"Br{k}_{col} = _mm512_mul_pd(bs{j}_{k}, {di});")
                        e.raw(f"Bi{k}_{col} = _mm512_mul_pd(bs{j}_{k}, {dr});")
                    else:
                        e.raw(f"Br{k}_{col} = _mm512_fmadd_pd(bs{j}_{k}, {di}, Br{k}_{col});")
                        e.raw(f"Bi{k}_{col} = _mm512_fmadd_pd(bs{j}_{k}, {dr}, Bi{k}_{col});")
                ti += 2
        for k in kc:
            for col in (0,1):
                off = "" if col==0 else " + po"
                e.raw(f"_mm512_store_pd(re + {k}*es{off}, _mm512_add_pd(Ar{k}_{col}, Br{k}_{col}));")
                e.raw(f"_mm512_store_pd(im + {k}*es{off}, _mm512_sub_pd(Ai{k}_{col}, Bi{k}_{col}));")
                e.raw(f"_mm512_store_pd(re + {N-k}*es{off}, _mm512_sub_pd(Ar{k}_{col}, Br{k}_{col}));")
                e.raw(f"_mm512_store_pd(im + {N-k}*es{off}, _mm512_add_pd(Ai{k}_{col}, Bi{k}_{col}));")
    e.raw("}")
    return "\n".join(e.lines)


# ================= C file assembly =================

PRELUDE = r'''
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN64 __attribute__((aligned(64)))
#include <sys/mman.h>
static double* alloc_huge(long bytes){
    long HP = (long)2<<20;
    bytes = (bytes + HP - 1) & ~(HP-1);
    void* p = mmap(0, bytes + HP, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED) { void* q = aligned_alloc(64, bytes); memset(q,0,bytes); return (double*)q; }
    char* a = (char*)(((unsigned long)p + HP - 1) & ~((unsigned long)HP-1));
    if(a > (char*)p) munmap(p, a - (char*)p);
    long tail = ((char*)p + bytes + HP) - (a + bytes);
    if(tail > 0) munmap(a + bytes, tail);
    madvise(a, bytes, MADV_HUGEPAGE);
    memset(a, 0, bytes);
    return (double*)a;
}
static long stagger_ctr = 0;
static double* alloc_huge_st(long bytes){
    // distinct sub-page stagger per buffer to avoid cross-buffer set aliasing
    long st = ((stagger_ctr++ % 29) + 1) * 4672;   // 4672 = 73*64 bytes, odd line multiple
    double* p = alloc_huge(bytes + st + 4096);
    return (double*)((char*)p + st);
}

static const __m512d VONE_ = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
#define VONE VONE_
static const __m512d VHALF_ = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
#define VHALF VHALF_

#define FMA_BC(acc, s, mem)  __asm__("vfmadd231pd %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(s), "m"(mem))
#define FNMA_BC(acc, s, mem) __asm__("vfnmadd231pd %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(s), "m"(mem))
#define MUL_BC(dst, s, mem)  __asm__("vmulpd %2%{1to8%}, %1, %0" : "=v"(dst) : "v"(s), "m"(mem))
#define BCAST(dst, mem) __asm__("vbroadcastsd %1, %0" : "=v"(dst) : "m"(mem))
#define BCASTV(dst, mem) dst = _mm512_set1_pd(*(volatile const double*)&(mem))

static inline void map2(__m512d zr, __m512d zi, __m512d* oxr, __m512d* oxi){
    const __m512d TINY = _mm512_set1_pd(1e-30);
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY));
    __m512d r0 = _mm512_rsqrt14_pd(m);
    __m512d t  = _mm512_mul_pd(m, r0);
    __m512d hr = _mm512_mul_pd(r0, VHALF);
    __m512d eh = _mm512_fnmadd_pd(t, hr, VHALF);
    __m512d r1 = _mm512_fmadd_pd(r0, eh, r0);
    __m512d mg0= _mm512_mul_pd(m, r1);
    __m512d hr1= _mm512_mul_pd(r1, VHALF);
    __m512d e2 = _mm512_fnmadd_pd(mg0, mg0, m);
    __m512d mag= _mm512_fmadd_pd(e2, hr1, mg0);
    __m512d u  = _mm512_add_pd(VONE, mag);
    __m512d w0 = _mm512_rcp14_pd(u);
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE);
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0);
    __m512d ee = _mm512_mul_pd(e3, e3);
    __m512d w2 = _mm512_fmadd_pd(a, ee, a);
    *oxr = _mm512_mul_pd(zr, w2);
    *oxi = _mm512_mul_pd(zi, w2);
}

// map over contiguous element-vec range, layout [e][2][8]; x and c same layout
static inline void map_range(double* x, const double* c, long n){
    long e=0;
    for(; e+2<=n; e+=2){
        __m512d xr0 = _mm512_load_pd(x + e*16);
        __m512d xi0 = _mm512_load_pd(x + e*16 + 8);
        __m512d xr1 = _mm512_load_pd(x + e*16 + 16);
        __m512d xi1 = _mm512_load_pd(x + e*16 + 24);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(c + e*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(c + e*16 + 8));
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(c + e*16 + 16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(c + e*16 + 24));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(x + e*16, xr0);
        _mm512_store_pd(x + e*16 + 8, xi0);
        _mm512_store_pd(x + e*16 + 16, xr1);
        _mm512_store_pd(x + e*16 + 24, xi1);
    }
    for(; e<n; e++){
        __m512d xr = _mm512_load_pd(x + e*16);
        __m512d xi = _mm512_load_pd(x + e*16 + 8);
        __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(c + e*16));
        __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(c + e*16 + 8));
        map2(zr, zi, &xr, &xi);
        _mm512_store_pd(x + e*16, xr);
        _mm512_store_pd(x + e*16 + 8, xi);
    }
}

// 8x8 double transpose: in r0..r7 -> out o0..o7
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7) do{ \
    __m512d _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    __m512d _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    __m512d _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    __m512d _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x44), _u1=_mm512_shuffle_f64x2(_t4,_t6,0x44); \
    __m512d _u2=_mm512_shuffle_f64x2(_t0,_t2,0xee), _u3=_mm512_shuffle_f64x2(_t4,_t6,0xee); \
    __m512d _u4=_mm512_shuffle_f64x2(_t1,_t3,0x44), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x44); \
    __m512d _u6=_mm512_shuffle_f64x2(_t1,_t3,0xee), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xee); \
    o0=_mm512_shuffle_f64x2(_u0,_u1,0x88); o2=_mm512_shuffle_f64x2(_u0,_u1,0xdd); \
    o4=_mm512_shuffle_f64x2(_u2,_u3,0x88); o6=_mm512_shuffle_f64x2(_u2,_u3,0xdd); \
    o1=_mm512_shuffle_f64x2(_u4,_u5,0x88); o3=_mm512_shuffle_f64x2(_u4,_u5,0xdd); \
    o5=_mm512_shuffle_f64x2(_u6,_u7,0x88); o7=_mm512_shuffle_f64x2(_u6,_u7,0xdd); \
}while(0)

static const __m512i IDX_EVEN_ = {0,2,4,6,8,10,12,14};
static const __m512i IDX_ODD_  = {1,3,5,7,9,11,13,15};
#define DEINT(lo,hi,re,im) do{ re=_mm512_permutex2var_pd(lo, IDX_EVEN_, hi); im=_mm512_permutex2var_pd(lo, IDX_ODD_, hi);}while(0)
static const __m512i IDX_ILO_ = {0,8,1,9,2,10,3,11};
static const __m512i IDX_IHI_ = {4,12,5,13,6,14,7,15};
#define INTER(re,im,lo,hi) do{ lo=_mm512_permutex2var_pd(re, IDX_ILO_, im); hi=_mm512_permutex2var_pd(re, IDX_IHI_, im);}while(0)
'''


def gen_familyA(L, codelet_name, pair_name=None, PF=0, btail=None, bthr=6, tag=None):
    if tag is None: tag = str(L)
    """across-volume SoA kernels. layout: plane i at offset i*PS*16; within plane e=j*L+k at +e*16."""
    L2 = L*L; L3 = L*L*L
    PS = L2 if (L2 % 2 == 1) else L2+1   # odd element stride to avoid 4K aliasing
    PSZ = PS*16  # doubles per plane
    PFS = ""
    if PF:
        # during ax2 (second phase of rep0), prefetch next slab (T1) and this slab's c (T0)
        pfn = (PSZ*8 + L - 1)//L
        PFS = (f"if(rep==0){{ const char* q = (const char*)(X + (long)(i+1<{L}?i+1:0)*{PSZ}) + (long)j*{pfn};"
               f" for(long b=0;b<{pfn};b+=64) _mm_prefetch(q+b, _MM_HINT_T1);"
               f" if(do_map){{ const char* cq = (const char*)(C + (long)i*{PSZ}) + (long)j*{pfn};"
               f" for(long b=0;b<{pfn};b+=64) _mm_prefetch(cq+b, _MM_HINT_T0); }} }}")
    if pair_name:
        assert (L*L) % 2 == 1
        AX1 = f"int k=0; for(;k+2<={L};k+=2) {pair_name}(sl + k*16, sl + k*16 + 8, {L*16}, 16); for(;k<{L};k++) {codelet_name}(sl + k*16, sl + k*16 + 8, {L*16});"
        AX2 = f"int j=0; for(;j+2<={L};j+=2) {pair_name}(sl + (long)j*{L*16}, sl + (long)j*{L*16} + 8, 16, {L*16}); for(;j<{L};j++) {codelet_name}(sl + (long)j*{L*16}, sl + (long)j*{L*16} + 8, 16);"
    else:
        AX1 = f"for(int k=0;k<{L};k++) {codelet_name}(sl + k*16, sl + k*16 + 8, {L*16});"
        AX2 = f"for(int j=0;j<{L};j++){{ {codelet_name}(sl + (long)j*{L*16}, sl + (long)j*{L*16} + 8, 16); {PFS} }}"
    if pair_name:
        PCALL = f"if(e+2<={L*L}) {pair_name}(p, p+8, {PSZ}, 16); else {codelet_name}(p, p+8, {PSZ});"
    else:
        PCALL = f"{codelet_name}(p, p+8, {PSZ});"
    ESTEP = f"(e+2<={L*L} ? 2 : 1)" if pair_name else "1"
    if btail:
        BTAIL = (f"if(rem && rem < {bthr}){{ long Bm = B - rem; "
                 f"if(Bm) run_{btail}_decl_dummy(); "
                 f"run_{btail}(x0 + Bm*2*{L*L*L}, c + Bm*2*{L*L*L}, out1 + Bm*2*{L*L*L}, outm + Bm*2*{L*L*L}, rem, m); "
                 f"B = Bm; G = B/8; if(!B) return; }}")
        BTAIL = BTAIL.replace(f"if(Bm) run_{btail}_decl_dummy(); ", "")
    else:
        BTAIL = ""
    s = []
    s.append(f"""
// ---------------- family A, L={L} (PS={PS}) ----------------
static void S_{L}(double* X, const double* C, int do_map, int do_next){{
    for(int i=0;i<{L};i++){{
        double* sl = X + (long)i*{PSZ};
        for(int rep=0;;rep++){{
            {AX1}
            {AX2}
            if(rep==0){{
                if(do_map) map_range(sl, C + (long)i*{PSZ}, {L2});
                if(do_next) continue;
            }}
            break;
        }}
    }}
}}
static void P_{L}(double* X, const double* C, int do_next){{
    for(long e=0;e<{L2};e+={ESTEP}){{
        double* p = X + e*16;
        const double* cp = C + e*16;
        {PCALL}
        for(long ee=0; ee<{ESTEP}; ee++){{
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<={L}; t+=2){{
            __m512d xr0 = _mm512_load_pd(p2 + t*{PSZ});
            __m512d xi0 = _mm512_load_pd(p2 + t*{PSZ} + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*{PSZ} + {PSZ});
            __m512d xi1 = _mm512_load_pd(p2 + t*{PSZ} + {PSZ} + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*{PSZ}));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*{PSZ} + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*{PSZ} + {PSZ}));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*{PSZ} + {PSZ} + 8));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*{PSZ}, xr0);
            _mm512_store_pd(p2 + t*{PSZ} + 8, xi0);
            _mm512_store_pd(p2 + t*{PSZ} + {PSZ}, xr1);
            _mm512_store_pd(p2 + t*{PSZ} + {PSZ} + 8, xi1);
        }}
        for(; t<{L}; t++){{
            __m512d xr = _mm512_load_pd(p2 + t*{PSZ});
            __m512d xi = _mm512_load_pd(p2 + t*{PSZ} + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*{PSZ}));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*{PSZ} + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*{PSZ}, xr);
            _mm512_store_pd(p2 + t*{PSZ} + 8, xi);
        }}
        }}
        if(do_next) {{ {PCALL} }}
    }}
}}
static void convin_{L}(const double* const* src, double* G){{
    for(int i=0;i<{L};i++){{
        double* gp = G + (long)i*{PSZ};
        long base = (long)i*{L2};
        long e=0;
        for(; e+4<={L2}; e+=4){{
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }}
        for(; e<{L2}; e++){{
            for(int v=0;v<8;v++){{ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }}
        }}
    }}
}}
static void convout_{L}(const double* G, double* const* dst, int nv){{
    for(int i=0;i<{L};i++){{
        const double* gp = G + (long)i*{PSZ};
        long base = (long)i*{L2};
        long e=0;
        for(; e+4<={L2}; e+=4){{
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {{o0,o1,o2,o3,o4,o5,o6,o7}};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }}
        for(; e<{L2}; e++){{
            for(int v=0;v<nv;v++){{ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }}
        }}
    }}
}}
static double* XG_{L} = 0;
static double* CG_{L} = 0;
void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(m < 1) m = 1;
    if(!XG_{L}){{ XG_{L} = alloc_huge_st((long){L}*{PSZ}*8); CG_{L} = alloc_huge_st((long){L}*{PSZ}*8); }}
    long G = (B + 7) / 8;
    long rem = B % 8;
    {BTAIL}
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){{
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){{
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*{L3};
        }}
        convin_{L}(srcs, XG_{L});
        for(int v=0; v<8; v++){{
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*{L3};
        }}
        convin_{L}(srcs, CG_{L});
        S_{L}(XG_{L}, CG_{L}, 0, 0);
        P_{L}(XG_{L}, CG_{L}, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*{L3};
        convout_{L}(XG_{L}, dsts, nv);
        if(m >= 2){{
            S_{L}(XG_{L}, CG_{L}, 0, 0);
            for(long t=2; t<=m; t++){{
                if((t & 1) == 0) P_{L}(XG_{L}, CG_{L}, t<m);
                else             S_{L}(XG_{L}, CG_{L}, 1, t<m);
            }}
        }}
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*{L3};
        convout_{L}(XG_{L}, dsts, nv);
    }}
}}
""")
    code = "\n".join(s)
    if tag != str(L):
        for ident in ("S_", "P_", "convin_", "convout_", "XG_", "CG_", "run_"):
            for sep in ("(", "[", " ", ",", ";", ")", "\n"):
                code = code.replace(ident + str(L) + sep, ident + tag + sep)
    return code


def build_A_sizes():
    out = []
    # codelets
    out.append(gen_pfa2("dft6_v", 6, 2, 3))
    # direct dft8 codelet wrapper
    e = E()
    e.raw("static __attribute__((always_inline)) inline void dft8_v(double* re, double* im, long es){")
    x = []
    for t in range(8):
        x.append((loadv(e,"re",f"{t}*es"), loadv(e,"im",f"{t}*es")))
    y = dft8(e, x)
    for t in range(8):
        storev(e,"re",f"{t}*es",y[t][0]); storev(e,"im",f"{t}*es",y[t][1])
    e.raw("}")
    out.append("\n".join(e.lines))
    import os
    H13 = os.environ.get("H13","reg6")
    if H13=="reg6": out.append(gen_hartley("dft13_v", 13, [[1,2,3,4,5,6]], scratch=False))
    elif H13=="s4": out.append(gen_hartley("dft13_v", 13, [[1,2,3,4],[5,6]], scratch=True))
    else: out.append(gen_hartley("dft13_v", 13, [[1,2,3],[4,5,6]], scratch=True))
    H17 = os.environ.get("H17","s44")
    if H17=="s44": out.append(gen_hartley("dft17_v", 17, [[1,2,3,4],[5,6,7,8]], scratch=True))
    elif H17=="s53": out.append(gen_hartley("dft17_v", 17, [[1,2,3,4,5],[6,7,8]], scratch=True))
    else: out.append(gen_hartley("dft17_v", 17, [[1,2],[3,4],[5,6],[7,8]], scratch=True))
    H23 = os.environ.get("H23","s443")
    if H23=="s443": out.append(gen_hartley("dft23_v", 23, [[1,2,3,4],[5,6,7,8],[9,10,11]], scratch=True))
    elif H23=="s65": out.append(gen_hartley("dft23_v", 23, [[1,2,3,4,5,6],[7,8,9,10,11]], scratch=True))
    else: out.append(gen_hartley("dft23_v", 23, [[1,2,3],[4,5,6],[7,8,9],[10,11]], scratch=True))
    out.append(gen_familyB(13, 16, 16, "dft13_v", PF=0, tag="13t"))
    out.append(gen_familyB(17, 24, 24, "dft17_v", PF=0, tag="17t"))
    out.append(gen_familyB(23, 24, 24, "dft23_v", PF=0, tag="23t"))
    out.append(gen_familyA(6, "dft6_v"))
    out.append(gen_familyA(8, "dft8_v"))
    out.append(gen_familyA(13, "dft13_v", PF=0, btail="13t", bthr=6))
    out.append(gen_familyA(17, "dft17_v", PF=0, btail="17t", bthr=5))
    out.append(gen_familyA(23, "dft23_v", PF=0, btail="23t", bthr=8))
    return "\n".join(out)



def gen_twostage_pair(name, N, N1, N2, mode):
    """two adjacent zmm columns (+8 doubles) per call."""
    e = E()
    e.raw(f"static double SC2_{N}[{N}][32] ALIGN64;")
    def ld2(expr_base):
        a = e.v(f"_mm512_load_pd({expr_base})")
        b = e.v(f"_mm512_load_pd({expr_base} + 8)")
        return a,b
    if mode == 'pfa':
        u1 = pow(N2, -1, N1); u2 = pow(N1, -1, N2)
        IN = [[(N2*n1 + N1*n2) % N for n1 in range(N1)] for n2 in range(N2)]
        OUT = [[(N2*u1*k1 + N1*u2*k2) % N for k2 in range(N2)] for k1 in range(N1)]
        e.raw(f"static const int IN2_{N}[{N2}][{N1}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in IN) + "};")
        e.raw(f"static const int OUT2_{N}[{N1}][{N2}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in OUT) + "};")
        e.raw(f"static void {name}(double* re, double* im, long es){{")
        e.raw(f"for(int n2=0;n2<{N2};n2++){{")
        e.raw(f"const int* off = IN2_{N}[n2];")
        xs = [[],[]]
        for n1 in range(N1):
            e.raw(f"long o{n1} = (long)off[{n1}]*es;")
            r0 = e.v(f"_mm512_load_pd(re + o{n1})"); i0 = e.v(f"_mm512_load_pd(im + o{n1})")
            r1 = e.v(f"_mm512_load_pd(re + o{n1} + 8)"); i1 = e.v(f"_mm512_load_pd(im + o{n1} + 8)")
            xs[0].append((r0,i0)); xs[1].append((r1,i1))
        ys = [dft_generic(e, xs[0], N1), dft_generic(e, xs[1], N1)]
        for k1 in range(N1):
            e.raw(f"_mm512_store_pd(SC2_{N}[{k1}*{N2}+n2], {ys[0][k1][0]}); _mm512_store_pd(SC2_{N}[{k1}*{N2}+n2]+8, {ys[0][k1][1]});")
            e.raw(f"_mm512_store_pd(SC2_{N}[{k1}*{N2}+n2]+16, {ys[1][k1][0]}); _mm512_store_pd(SC2_{N}[{k1}*{N2}+n2]+24, {ys[1][k1][1]});")
        e.raw("}")
        e.raw(f"for(int k1=0;k1<{N1};k1++){{")
        e.raw(f"const int* off = OUT2_{N}[k1];")
        xs = [[],[]]
        for n2 in range(N2):
            a = e.v(f"_mm512_load_pd(SC2_{N}[k1*{N2}+{n2}])"); b = e.v(f"_mm512_load_pd(SC2_{N}[k1*{N2}+{n2}]+8)")
            c_ = e.v(f"_mm512_load_pd(SC2_{N}[k1*{N2}+{n2}]+16)"); d = e.v(f"_mm512_load_pd(SC2_{N}[k1*{N2}+{n2}]+24)")
            xs[0].append((a,b)); xs[1].append((c_,d))
        ys = [dft_generic(e, xs[0], N2), dft_generic(e, xs[1], N2)]
        for k2 in range(N2):
            e.raw(f"long q{k2} = (long)off[{k2}]*es;")
            e.raw(f"_mm512_store_pd(re + q{k2}, {ys[0][k2][0]}); _mm512_store_pd(im + q{k2}, {ys[0][k2][1]});")
            e.raw(f"_mm512_store_pd(re + q{k2} + 8, {ys[1][k2][0]}); _mm512_store_pd(im + q{k2} + 8, {ys[1][k2][1]});")
        e.raw("}")
        e.raw("}")
    else:
        assert N1 == N2
        R = N1
        e.raw(f"static void {name}(double* re, double* im, long es){{")
        e.raw(f"for(int r=0;r<{R};r++){{")
        e.raw(f"const double* twp = TW_{N} + r*{2*R};")
        e.raw("double* rb = re + (long)r*es; double* ib = im + (long)r*es;")
        xs=[[],[]]
        for q in range(R):
            a = e.v(f"_mm512_load_pd(rb + (long){q*R}*es)"); b = e.v(f"_mm512_load_pd(ib + (long){q*R}*es)")
            c_ = e.v(f"_mm512_load_pd(rb + (long){q*R}*es + 8)"); d = e.v(f"_mm512_load_pd(ib + (long){q*R}*es + 8)")
            xs[0].append((a,b)); xs[1].append((c_,d))
        ys = [dft_generic(e, xs[0], R), dft_generic(e, xs[1], R)]
        e.raw(f"_mm512_store_pd(&SC2_{N}[0][0] + (0*{R}+r)*32, {ys[0][0][0]}); _mm512_store_pd(&SC2_{N}[0][0] + (0*{R}+r)*32+8, {ys[0][0][1]});")
        e.raw(f"_mm512_store_pd(&SC2_{N}[0][0] + (0*{R}+r)*32+16, {ys[1][0][0]}); _mm512_store_pd(&SC2_{N}[0][0] + (0*{R}+r)*32+24, {ys[1][0][1]});")
        for d_ in range(1,R):
            e.raw(f"__m512d wr{d_}, wi{d_};")
            e.raw(f"BCAST(wr{d_}, twp[{2*d_}]); BCAST(wi{d_}, twp[{2*d_+1}]);")
            for col in (0,1):
                yr, yi = ys[col][d_]
                vr = f"tvr{d_}_{col}"; vi = f"tvi{d_}_{col}"
                e.raw(f"__m512d {vr} = _mm512_fnmadd_pd({yi}, wi{d_}, _mm512_mul_pd({yr}, wr{d_}));")
                e.raw(f"__m512d {vi} = _mm512_fmadd_pd({yr}, wi{d_}, _mm512_mul_pd({yi}, wr{d_}));")
            e.raw(f"_mm512_store_pd(&SC2_{N}[0][0] + ({d_}*{R}+r)*32, tvr{d_}_0); _mm512_store_pd(&SC2_{N}[0][0] + ({d_}*{R}+r)*32+8, tvi{d_}_0);")
            e.raw(f"_mm512_store_pd(&SC2_{N}[0][0] + ({d_}*{R}+r)*32+16, tvr{d_}_1); _mm512_store_pd(&SC2_{N}[0][0] + ({d_}*{R}+r)*32+24, tvi{d_}_1);")
        e.raw("}")
        e.raw(f"for(int d=0;d<{R};d++){{")
        e.raw(f"double* rb = re + (long)d*es; double* ib = im + (long)d*es;")
        xs=[[],[]]
        for r in range(R):
            a = e.v(f"_mm512_load_pd(&SC2_{N}[0][0] + (d*{R}+{r})*32)"); b = e.v(f"_mm512_load_pd(&SC2_{N}[0][0] + (d*{R}+{r})*32+8)")
            c_ = e.v(f"_mm512_load_pd(&SC2_{N}[0][0] + (d*{R}+{r})*32+16)"); dd = e.v(f"_mm512_load_pd(&SC2_{N}[0][0] + (d*{R}+{r})*32+24)")
            xs[0].append((a,b)); xs[1].append((c_,dd))
        ys = [dft_generic(e, xs[0], R), dft_generic(e, xs[1], R)]
        for c_ in range(R):
            e.raw(f"_mm512_store_pd(rb + (long){c_*R}*es, {ys[0][c_][0]}); _mm512_store_pd(ib + (long){c_*R}*es, {ys[0][c_][1]});")
            e.raw(f"_mm512_store_pd(rb + (long){c_*R}*es + 8, {ys[1][c_][0]}); _mm512_store_pd(ib + (long){c_*R}*es + 8, {ys[1][c_][1]});")
        e.raw("}")
        e.raw("}")
    return "\n".join(e.lines)


# ============== family B: within-volume SoA, V=1 ==============


def gen_twostage(name, N, N1, N2, mode):
    e = E()
    e.raw(f"static double SC_{N}[{N}][16] ALIGN64;")
    if mode == 'pfa':
        u1 = pow(N2, -1, N1); u2 = pow(N1, -1, N2)
        IN = [[(N2*n1 + N1*n2) % N for n1 in range(N1)] for n2 in range(N2)]
        OUT = [[(N2*u1*k1 + N1*u2*k2) % N for k2 in range(N2)] for k1 in range(N1)]
        e.raw(f"static const int IN_{N}[{N2}][{N1}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in IN) + "};")
        e.raw(f"static const int OUT_{N}[{N1}][{N2}] = {{" + ",".join("{"+",".join(map(str,row))+"}" for row in OUT) + "};")
        e.raw(f"static void {name}(double* re, double* im, long es){{")
        e.raw(f"for(int n2=0;n2<{N2};n2++){{")
        e.raw(f"const int* off = IN_{N}[n2];")
        x = []
        for n1 in range(N1):
            xr = e.v(f"_mm512_load_pd(re + (long)off[{n1}]*es)")
            xi = e.v(f"_mm512_load_pd(im + (long)off[{n1}]*es)")
            x.append((xr,xi))
        y = dft_generic(e, x, N1)
        for k1 in range(N1):
            e.raw(f"_mm512_store_pd(SC_{N}[{k1}*{N2}+n2], {y[k1][0]}); _mm512_store_pd(SC_{N}[{k1}*{N2}+n2]+8, {y[k1][1]});")
        e.raw("}")
        e.raw(f"for(int k1=0;k1<{N1};k1++){{")
        e.raw(f"const int* off = OUT_{N}[k1];")
        x = []
        for n2 in range(N2):
            xr = e.v(f"_mm512_load_pd(SC_{N}[k1*{N2}+{n2}])"); xi = e.v(f"_mm512_load_pd(SC_{N}[k1*{N2}+{n2}]+8)")
            x.append((xr,xi))
        y = dft_generic(e, x, N2)
        for k2 in range(N2):
            e.raw(f"_mm512_store_pd(re + (long)off[{k2}]*es, {y[k2][0]}); _mm512_store_pd(im + (long)off[{k2}]*es, {y[k2][1]});")
        e.raw("}")
        e.raw("}")
    else:
        assert N1 == N2
        R = N1
        # twiddle table: TW[r][d] pairs, r=0..R-1 (r=0 trivial but keep uniform)
        vals = []
        for r in range(R):
            for d in range(R):
                wr, wi = tw(N, r*d)
                vals += [wr, wi]
        e.raw(f"static const double TW_{N}[{len(vals)}] ALIGN64 = {{" + ",".join(hexd(v) for v in vals) + "};")
        e.raw(f"static void {name}(double* re, double* im, long es){{")
        e.raw(f"for(int r=0;r<{R};r++){{")
        e.raw(f"const double* twp = TW_{N} + r*{2*R};")
        e.raw("double* rb = re + (long)r*es; double* ib = im + (long)r*es;")
        x = []
        for q in range(R):
            xr = e.v(f"_mm512_load_pd(rb + (long){q*R}*es)")
            xi = e.v(f"_mm512_load_pd(ib + (long){q*R}*es)")
            x.append((xr,xi))
        y = dft_generic(e, x, R)
        # d=0: no twiddle
        e.raw(f"_mm512_store_pd(SC_{N}[0*{R}]+r*16 - r*16 + (0*{R}+r)*0, {y[0][0]});")
        e.lines.pop()
        e.raw(f"_mm512_store_pd(&SC_{N}[0][0] + (0*{R}+r)*16, {y[0][0]}); _mm512_store_pd(&SC_{N}[0][0] + (0*{R}+r)*16+8, {y[0][1]});")
        for d in range(1,R):
            yr, yi = y[d]
            vr = f"tvr{d}"; vi = f"tvi{d}"; wr = f"twr{d}"; wi = f"twi{d}"
            e.raw(f"__m512d {vr}, {vi}, {wr}, {wi};")
            e.raw(f"BCASTV({wr}, twp[{2*d}]); BCASTV({wi}, twp[{2*d+1}]);")
            e.raw(f"{vr} = _mm512_fnmadd_pd({yi}, {wi}, _mm512_mul_pd({yr}, {wr}));")
            e.raw(f"{vi} = _mm512_fmadd_pd({yr}, {wi}, _mm512_mul_pd({yi}, {wr}));")
            e.raw(f"_mm512_store_pd(&SC_{N}[0][0] + ({d}*{R}+r)*16, {vr}); _mm512_store_pd(&SC_{N}[0][0] + ({d}*{R}+r)*16+8, {vi});")
        e.raw("}")
        e.raw(f"for(int d=0;d<{R};d++){{")
        e.raw(f"double* rb = re + (long)d*es; double* ib = im + (long)d*es;")
        x = []
        for r in range(R):
            xr = e.v(f"_mm512_load_pd(&SC_{N}[0][0] + (d*{R}+{r})*16)")
            xi = e.v(f"_mm512_load_pd(&SC_{N}[0][0] + (d*{R}+{r})*16+8)")
            x.append((xr,xi))
        y = dft_generic(e, x, R)
        for c in range(R):
            e.raw(f"_mm512_store_pd(rb + (long){c*R}*es, {y[c][0]}); _mm512_store_pd(ib + (long){c*R}*es, {y[c][1]});")
        e.raw("}")
        e.raw("}")
    return "\n".join(e.lines)


def gen_familyB(L, LPAD, LJP, codelet, PF=1, CPAD=None, pair=None, tag=None):
    if tag is None: tag = str(L)
    KB = (L + 7)//8       # chunks along k (valid incl tail pad)
    JB = LJP//8           # blocks along j
    if CPAD is None: CPAD = LPAD
    KROWS = KB*8
    IMOFF = LJP*LPAD
    IMOFF_T = KROWS*CPAD
    PPS = 2*LJP*LPAD + 8
    L2 = L*L; L3 = L*L*L
    masks = []
    for kc in range(KB):
        vk = min(8, L - kc*8)
        m0 = (1 << (2*min(vk,4))) - 1
        m1 = (1 << (2*max(vk-4,0))) - 1
        masks.append((m0,m1))
    mk0 = ",".join(str(m[0]) for m in masks)
    mk1 = ",".join(str(m[1]) for m in masks)
    pre = f"static const int MK0_{L}[{KB}] = {{{mk0}}};\nstatic const int MK1_{L}[{KB}] = {{{mk1}}};\nstatic const int MKJ0_{L}[{KB}] = {{{mk0}}};\nstatic const int MKJ1_{L}[{KB}] = {{{mk1}}};\n"
    lines = [f"#define MASKS_{L}(crow_, pre_, pim_) do{{ \\"]
    for kc in range(KB):
        m0, m1 = masks[kc]
        lines.append(f"  mapvec_{L}((pre_)+{kc*8}, (pim_)+{kc*8}, (crow_)+{kc*16}, {m0}, {m1}); \\")
    lines.append("}while(0)")
    pre += "\n".join(lines) + "\n"
    if PF:
        PF1 = ""
        PF2 = ""
        PFKB = """
                    {{ // interleaved prefetch: next plane slice (T1) + c_sw slice (T0)
                        long pfn = ({PPS}*8)/({JB}*{KB}) + 64;
                        const char* q = npl + ((long)jb*{KB}+kb)*pfn;
                        for(long b=0;b<pfn;b+=64) _mm_prefetch(q+b, _MM_HINT_T1);
                        if(do_map){{
                            const char* cq = (const char*)(Csw + (long)i*{TL2} + jb*16) + (long)kb*{KSL}*{TL}*8;
                            for(int kk=0;kk<{KSL};kk++){{ _mm_prefetch(cq + (long)kk*{TL}*8, _MM_HINT_T0); _mm_prefetch(cq + (long)kk*{TL}*8 + 64, _MM_HINT_T0); }}
                        }}
                    }}""".format(TL2=2*L*L, TL=2*L, JB=JB, KB=KB, PPS=PPS, KSL=(L+KB-1)//KB)
        PF3 = """                _mm_prefetch(npr + (long)i*{PPS}*8, _MM_HINT_T1);
                _mm_prefetch(npr + (long)i*{PPS}*8 + {IMOFF}*8, _MM_HINT_T1);
                _mm_prefetch(ncp + (long)i*{TL2}*8, _MM_HINT_T1);
                _mm_prefetch(ncp + (long)i*{TL2}*8 + 64, _MM_HINT_T1);
                _mm_prefetch(npr + (long)(i+1)*{PPS}*8, _MM_HINT_T1);
                _mm_prefetch(npr + (long)(i+1)*{PPS}*8 + {IMOFF}*8, _MM_HINT_T1);
                _mm_prefetch(ncp + (long)(i+1)*{TL2}*8, _MM_HINT_T1);
                _mm_prefetch(ncp + (long)(i+1)*{TL2}*8 + 64, _MM_HINT_T1);
""".format(PPS=PPS, IMOFF=IMOFF, TL2=2*L*L)
    else:
        PF1 = ""; PF2 = ""; PF3 = ""; PFKB = ""
    TRIN = """
            {{
                const double* rb = pl + (long)jb*8*{LPAD};
                for(int kb=0;kb<{KB};kb++){{
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    r0=_mm512_load_pd(rb+0*{LPAD}+kb*8); r1=_mm512_load_pd(rb+1*{LPAD}+kb*8);
                    r2=_mm512_load_pd(rb+2*{LPAD}+kb*8); r3=_mm512_load_pd(rb+3*{LPAD}+kb*8);
                    r4=_mm512_load_pd(rb+4*{LPAD}+kb*8); r5=_mm512_load_pd(rb+5*{LPAD}+kb*8);
                    r6=_mm512_load_pd(rb+6*{LPAD}+kb*8); r7=_mm512_load_pd(rb+7*{LPAD}+kb*8);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    double* tb = &TS_{L}[0] + (long)kb*8*{CPAD} + jb*8;
                    _mm512_store_pd(tb+0*{CPAD}, o0); _mm512_store_pd(tb+1*{CPAD}, o1);
                    _mm512_store_pd(tb+2*{CPAD}, o2); _mm512_store_pd(tb+3*{CPAD}, o3);
                    _mm512_store_pd(tb+4*{CPAD}, o4); _mm512_store_pd(tb+5*{CPAD}, o5);
                    _mm512_store_pd(tb+6*{CPAD}, o6); _mm512_store_pd(tb+7*{CPAD}, o7);
                    r0=_mm512_load_pd(rb+{IMOFF}+0*{LPAD}+kb*8); r1=_mm512_load_pd(rb+{IMOFF}+1*{LPAD}+kb*8);
                    r2=_mm512_load_pd(rb+{IMOFF}+2*{LPAD}+kb*8); r3=_mm512_load_pd(rb+{IMOFF}+3*{LPAD}+kb*8);
                    r4=_mm512_load_pd(rb+{IMOFF}+4*{LPAD}+kb*8); r5=_mm512_load_pd(rb+{IMOFF}+5*{LPAD}+kb*8);
                    r6=_mm512_load_pd(rb+{IMOFF}+6*{LPAD}+kb*8); r7=_mm512_load_pd(rb+{IMOFF}+7*{LPAD}+kb*8);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    tb += {IMOFF_T};
                    _mm512_store_pd(tb+0*{CPAD}, o0); _mm512_store_pd(tb+1*{CPAD}, o1);
                    _mm512_store_pd(tb+2*{CPAD}, o2); _mm512_store_pd(tb+3*{CPAD}, o3);
                    _mm512_store_pd(tb+4*{CPAD}, o4); _mm512_store_pd(tb+5*{CPAD}, o5);
                    _mm512_store_pd(tb+6*{CPAD}, o6); _mm512_store_pd(tb+7*{CPAD}, o7);
{PFKB}
                }}
            }}""".format(JB=JB, KB=KB, LPAD=LPAD, CPAD=CPAD, IMOFF=IMOFF, IMOFF_T=IMOFF_T, L=L, PFKB=PFKB)
    TROUT = """
            {{
                double* rb = pl + (long)jb*8*{LPAD};
                for(int kb=0;kb<{KB};kb++){{
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_{L}[0] + (long)kb*8*{CPAD} + jb*8;
                    r0=_mm512_load_pd(tb+0*{CPAD}); r1=_mm512_load_pd(tb+1*{CPAD});
                    r2=_mm512_load_pd(tb+2*{CPAD}); r3=_mm512_load_pd(tb+3*{CPAD});
                    r4=_mm512_load_pd(tb+4*{CPAD}); r5=_mm512_load_pd(tb+5*{CPAD});
                    r6=_mm512_load_pd(tb+6*{CPAD}); r7=_mm512_load_pd(tb+7*{CPAD});
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb+0*{LPAD}+kb*8, o0); _mm512_store_pd(rb+1*{LPAD}+kb*8, o1);
                    _mm512_store_pd(rb+2*{LPAD}+kb*8, o2); _mm512_store_pd(rb+3*{LPAD}+kb*8, o3);
                    _mm512_store_pd(rb+4*{LPAD}+kb*8, o4); _mm512_store_pd(rb+5*{LPAD}+kb*8, o5);
                    _mm512_store_pd(rb+6*{LPAD}+kb*8, o6); _mm512_store_pd(rb+7*{LPAD}+kb*8, o7);
                    tb += {IMOFF_T};
                    r0=_mm512_load_pd(tb+0*{CPAD}); r1=_mm512_load_pd(tb+1*{CPAD});
                    r2=_mm512_load_pd(tb+2*{CPAD}); r3=_mm512_load_pd(tb+3*{CPAD});
                    r4=_mm512_load_pd(tb+4*{CPAD}); r5=_mm512_load_pd(tb+5*{CPAD});
                    r6=_mm512_load_pd(tb+6*{CPAD}); r7=_mm512_load_pd(tb+7*{CPAD});
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb+{IMOFF}+0*{LPAD}+kb*8, o0); _mm512_store_pd(rb+{IMOFF}+1*{LPAD}+kb*8, o1);
                    _mm512_store_pd(rb+{IMOFF}+2*{LPAD}+kb*8, o2); _mm512_store_pd(rb+{IMOFF}+3*{LPAD}+kb*8, o3);
                    _mm512_store_pd(rb+{IMOFF}+4*{LPAD}+kb*8, o4); _mm512_store_pd(rb+{IMOFF}+5*{LPAD}+kb*8, o5);
                    _mm512_store_pd(rb+{IMOFF}+6*{LPAD}+kb*8, o6); _mm512_store_pd(rb+{IMOFF}+7*{LPAD}+kb*8, o7);
                }}
            }}""".format(JB=JB, KB=KB, LPAD=LPAD, CPAD=CPAD, IMOFF=IMOFF, IMOFF_T=IMOFF_T, L=L)
    TL2 = 2*L*L; TL = 2*L
    if pair:
        calls = []
        kc = 0
        while kc < KB:
            if kc+2 <= KB:
                calls.append(f"{pair}(pl + {kc*8}, pl + {IMOFF} + {kc*8}, {LPAD});")
                kc += 2
            else:
                calls.append(f"{codelet}(pl + {kc*8}, pl + {IMOFF} + {kc*8}, {LPAD});")
                kc += 1
        AX1CALLS = " ".join(calls)
        # P: pair codelet only when kc even and kc+1 < KB; P loop iterates per kc for the map, so run pair at even kc, single at odd leftover... simplest: call pair at even kc (covers kc,kc+1), nothing at odd kc except when it had no pair.
        PCODE1 = f"if(two) {pair}(pr, pi, {PPS}); else {codelet}(pr, pi, {PPS});"
        PCODE2 = f"if(two) {pair}(pr, pi, {PPS}); else {codelet}(pr, pi, {PPS});"
    else:
        AX1CALLS = f"for(int kc=0;kc<{KB};kc++){{ {codelet}(pl + kc*8, pl + {IMOFF} + kc*8, {LPAD}); }}"
        PCODE1 = f"{codelet}(pr, pi, {PPS}); if(two) {codelet}(pr+8, pi+8, {PPS});"
        PCODE2 = f"{codelet}(pr, pi, {PPS}); if(two) {codelet}(pr+8, pi+8, {PPS});"
    s = []
    s.append(f"""
// ---------------- family B, L={L} (LPAD={LPAD}, LJP={LJP}, CPAD={CPAD}, PPS={PPS}) ----------------
static double TS_{L}[{2*KROWS*CPAD}] ALIGN64;
static inline void mapvec_{L}(double* pre, double* pim, const double* crow, int m0, int m1){{
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)m0, crow);
    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)m1, crow+8);
    __m512d cr, ci;
    DEINT(lo, hi, cr, ci);
    __m512d zr = _mm512_add_pd(xr, cr), zi = _mm512_add_pd(xi, ci);
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}}
static void S_{L}(double* X, const double* Csw, int do_map, int do_next){{
    for(int i=0;i<{L};i++){{
        double* pl = X + (long)i*{PPS};
        const char* npl = (const char*)(X + (long)(i+1 < {L} ? i+1 : 0)*{PPS});
        {{
            int rep = 0;
            {AX1CALLS}
        }}
        for(int jb=0;jb<{JB};jb++){{
{PF2}
            {TRIN}
            {codelet}(&TS_{L}[0] + jb*8, &TS_{L}[0] + {IMOFF_T} + jb*8, {CPAD});
            if(do_map){{
                for(int k=0;k<{L};k++){{
                    const double* crow = Csw + ((long)i*{L} + k)*{TL} + jb*16;
                    mapvec_{L}(&TS_{L}[0] + (long)k*{CPAD} + jb*8, &TS_{L}[0] + {IMOFF_T} + (long)k*{CPAD} + jb*8, crow, MKJ0_{L}[jb], MKJ1_{L}[jb]);
                }}
            }}
            if(do_next) {codelet}(&TS_{L}[0] + jb*8, &TS_{L}[0] + {IMOFF_T} + jb*8, {CPAD});
            {TROUT}
        }}
        if(do_next){{
            {AX1CALLS}
        }}
    }}
}}
static void mapcol_{L}(double* X, const double* C, int j, int kc, int jn, int kcn){{
    double* pr = X + (long)j*{LPAD} + kc*8;
    double* pi = pr + {IMOFF};
    int m0 = MK0_{L}[kc], m1 = MK1_{L}[kc];
    const double* cp = C + (long)j*{2*L} + kc*16;
    const char* npr = (const char*)(X + (long)jn*{LPAD} + kcn*8);
    const char* ncp = (const char*)(C + (long)jn*{2*L} + kcn*16);
    for(int i=0;i<{L};i+=2){{
{PF3}        __m512d xr = _mm512_load_pd(pr + (long)i*{PPS});
        __m512d xi = _mm512_load_pd(pi + (long)i*{PPS});
        __m512d lo = _mm512_maskz_loadu_pd((__mmask8)m0, cp + (long)i*{2*L2});
        __m512d hi = _mm512_maskz_loadu_pd((__mmask8)m1, cp + (long)i*{2*L2} + 8);
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*{PPS});
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*{PPS});
        __m512d lo1 = _mm512_maskz_loadu_pd((__mmask8)m0, cp + (long)(i+1)*{2*L2});
        __m512d hi1 = _mm512_maskz_loadu_pd((__mmask8)m1, cp + (long)(i+1)*{2*L2} + 8);
        __m512d cr, ci, cr1, ci1;
        DEINT(lo, hi, cr, ci);
        DEINT(lo1, hi1, cr1, ci1);
        __m512d zr = _mm512_add_pd(xr, cr), zi = _mm512_add_pd(xi, ci);
        __m512d zr1 = _mm512_add_pd(xr1, cr1), zi1 = _mm512_add_pd(xi1, ci1);
        map2(zr, zi, &xr, &xi);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*{PPS}, xr);
        _mm512_store_pd(pi + (long)i*{PPS}, xi);
        _mm512_store_pd(pr + (long)(i+1)*{PPS}, xr1);
        _mm512_store_pd(pi + (long)(i+1)*{PPS}, xi1);
    }}
}}
static void P_{L}(double* X, const double* C, int do_next){{
    for(int j=0;j<{L};j++){{
        for(int kc=0;kc<{KB};kc++){{
            double* pr = X + (long)j*{LPAD} + kc*8;
            double* pi = pr + {IMOFF};
            int kc2 = kc+1, j2 = j;
            if(kc2 >= {KB}){{ kc2 = 0; j2 = (j+1<{L}) ? j+1 : 0; }}
            {codelet}(pr, pi, {PPS});
            mapcol_{L}(X, C, j, kc, j2, kc2);
            if(do_next) {codelet}(pr, pi, {PPS});
        }}
    }}
}}
static void convin_{L}(const double* src, double* X){{
    for(int i=0;i<{L};i++){{
        for(int j=0;j<{L};j++){{
            const double* row = src + ((long)i*{L}+j)*{2*L};
            double* pre = X + (long)i*{PPS} + (long)j*{LPAD};
            for(int kc=0;kc<{KB};kc++){{
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_{L}[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_{L}[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + {IMOFF} + kc*8, im);
            }}
        }}
    }}
}}
static void convout_{L}(const double* X, double* dst){{
    for(int i=0;i<{L};i++){{
        for(int j=0;j<{L};j++){{
            double* row = dst + ((long)i*{L}+j)*{2*L};
            const double* pre = X + (long)i*{PPS} + (long)j*{LPAD};
            for(int kc=0;kc<{KB};kc++){{
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + {IMOFF} + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_{L}[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_{L}[kc], hi);
            }}
        }}
    }}
}}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildcsw_{L}(const double* c, double* csw){{
    for(int i=0;i<{L};i++){{
        const double* cp = c + (long)i*{2*L2};
        double* sp = csw + (long)i*{2*L2};
        for(int jb=0;jb<{KB};jb++){{
            int jn = {L} - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<{KB};kb++){{
                int kn = {L} - kb*8; if(kn>8) kn=8;
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){{
                    const double* row = cp + ((long)(jb*8+r)*{L} + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_{L}[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_{L}[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                }}
                for(int r=jn;r<8;r++){{ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }}
                __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                __m512d TRE[8] = {{o0,o1,o2,o3,o4,o5,o6,o7}};
                TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],o0,o1,o2,o3,o4,o5,o6,o7);
                __m512d TIM[8] = {{o0,o1,o2,o3,o4,o5,o6,o7}};
                for(int r=0;r<kn;r++){{
                    double* row = sp + ((long)(kb*8+r)*{L} + jb*8)*2;
                    __m512d lo, hi;
                    INTER(TRE[r], TIM[r], lo, hi);
                    _mm512_mask_storeu_pd(row, (__mmask8)MK0_{L}[jb], lo);
                    _mm512_mask_storeu_pd(row+8, (__mmask8)MK1_{L}[jb], hi);
                }}
            }}
        }}
    }}
}}
static double* XV_{L} = 0;
static double* CSW_{L} = 0;
static double* CNAT_{L} = 0;
void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(m < 1) m = 1;
    if(!XV_{L}){{ XV_{L} = alloc_huge_st((long){L}*{PPS}*8 + 4096); CSW_{L} = alloc_huge_st((long){L3}*16 + 4096); CNAT_{L} = alloc_huge_st((long){L3}*16 + 4096); }}
    for(long v=0; v<B; v++){{
        memcpy(CNAT_{L}, c + v*{2*L3}, (long){L3}*16);
        const double* cx = CNAT_{L};
        if(m >= 3) buildcsw_{L}(cx, CSW_{L});
        convin_{L}(x0 + v*{2*L3}, XV_{L});
        S_{L}(XV_{L}, CSW_{L}, 0, 0);
        P_{L}(XV_{L}, cx, 0);
        convout_{L}(XV_{L}, out1 + v*{2*L3});
        if(m >= 2){{
            S_{L}(XV_{L}, CSW_{L}, 0, 0);
            for(long t=2; t<=m; t++){{
                if((t & 1) == 0) P_{L}(XV_{L}, cx, t<m);
                else             S_{L}(XV_{L}, CSW_{L}, 1, t<m);
            }}
        }}
        convout_{L}(XV_{L}, outm + v*{2*L3});
    }}
}}
""")
    code = pre + s[0]
    if tag != str(L):
        for ident in ("TS_", "mapvec_", "S_", "P_", "convin_", "convout_", "buildcsw_", "XV_", "CSW_", "CNAT_", "run_", "MK0_", "MK1_", "MKJ0_", "MKJ1_", "MASKS_", "mapcol_"):
            code = code.replace(ident + str(L) + "[", ident + tag + "[")
            code = code.replace(ident + str(L) + "(", ident + tag + "(")
            code = code.replace(ident + str(L) + " ", ident + tag + " ")
            code = code.replace(ident + str(L) + ",", ident + tag + ",")
            code = code.replace(ident + str(L) + ";", ident + tag + ";")
            code = code.replace(ident + str(L) + ")", ident + tag + ")")
            code = code.replace(ident + str(L) + "\n", ident + tag + "\n")
    return code

def build_B_sizes():
    out = []
    out.append(gen_twostage("dft36_v", 36, 4, 9, 'pfa'))
    out.append(gen_twostage("dft45_v", 45, 5, 9, 'pfa'))
    out.append(gen_twostage("dft64_v", 64, 8, 8, 'ct'))
    out.append(gen_familyB(36, 40, 40, "dft36_v", PF=1))
    out.append(gen_familyB(45, 56, 48, "dft45_v"))
    out.append(gen_familyB(64, 72, 64, "dft64_v"))
    return "\n".join(out)


HEADER = r""""""[1:-1]
HEADER = '/*\n * Iterated batched 3D complex-to-complex DFTs for fixed cube sizes\n *   L in {6, 8, 13, 17, 23, 36, 45, 64},   z = FFT3(x) + c ; x <- z / (1 + |z|)\n *\n * All transform arithmetic is our own (no FFT libraries of any kind):\n *  - small-N codelets generated at build time: PFA 6=2x3, 36=4x9, 45=5x9 (twiddle-free),\n *    Cooley-Tukey 64=8x8 with exact mod-N twiddles, direct symmetric ("Hartley-split")\n *    DFTs for the primes 13/17/23 (2h^2 fused-multiply-adds per length-N pencil),\n *  - all twiddle/trig constants precomputed in extended precision and baked as hex\n *    literals (exact mod-N reduction, <=0.5 ulp),\n *  - two data layouts: sizes 6..23 process 8 volumes per AVX-512 lane group\n *    (pure vertical SIMD, zero shuffles in the transform); sizes 36/45/64 are\n *    within-volume split re/im planes with 8x8 register transposes for the\n *    contiguous axis,\n *  - one fused sweep per iteration step in steady state (slab pass and pencil pass\n *    alternate, each finishing step t and starting step t+1),\n *  - elementwise map z/(1+|z|) via rsqrt14+Newton+Heron and rcp14+4th-order\n *    Newton (~1 ulp, full IEEE double precision throughout),\n *  - single-threaded, AVX-512, huge-page backed buffers.\n */\n'

if __name__ == "__main__":
    src = HEADER + PRELUDE + build_A_sizes() + build_B_sizes()
    with open("implementation.c","w") as f:
        f.write(src)
    print("wrote implementation.c", len(src), "bytes")
