import numpy as np
from genlib import hexd, LD, PI
from gen_a import prime_tables, fold_idx

class A:
    """asm emitter with explicit register pool (zmm0..zmm31)"""
    def __init__(self):
        self.lines = []
        self.free = list(range(32))
        self.live = set()
    def alloc(self):
        assert self.free, "register pool exhausted"
        r = max(self.free)
        self.free.remove(r)
        self.live.add(r)
        return r
    def alloc_low(self):
        lows = [r for r in self.free if r < 16]
        assert lows, "no low registers free"
        r = lows[0]
        self.free.remove(r)
        self.live.add(r)
        return r
    def rel(self, r):
        assert r in self.live, f"release of non-live {r}"
        self.live.discard(r)
        self.free.insert(0, r)
    def ins(self, s):
        self.lines.append(s)
    # memory ops: base is operand name string, off in bytes
    def ld(self, base, off):
        r = self.alloc()
        self.ins(f"vmovapd {off}(%[{base}]), %%zmm{r}")
        return r
    def st(self, base, off, r):
        self.ins(f"vmovapd %%zmm{r}, {off}(%[{base}])")
    def bcast(self, base, off):
        r = self.alloc()
        self.ins(f"vbroadcastsd {off}(%[{base}]), %%zmm{r}")
        return r
    # arithmetic; all return NEW reg unless inplace
    def add(self, a, b, kill=()):
        d = self._dst(kill)
        self.ins(f"vaddpd %%zmm{b}, %%zmm{a}, %%zmm{d}")
        return d
    def sub(self, a, b, kill=()):
        d = self._dst(kill)
        self.ins(f"vsubpd %%zmm{b}, %%zmm{a}, %%zmm{d}")
        return d
    def mul(self, a, b, kill=()):
        d = self._dst(kill)
        self.ins(f"vmulpd %%zmm{b}, %%zmm{a}, %%zmm{d}")
        return d
    def _dst(self, kill):
        for k in kill:
            self.rel(k)
        return self.alloc()
    def mov(self, a):
        d = self.alloc()
        self.ins(f"vmovapd %%zmm{a}, %%zmm{d}")
        return d
    def fma(self, acc, a, b):   # acc += a*b (in place)
        self.ins(f"vfmadd231pd %%zmm{b}, %%zmm{a}, %%zmm{acc}")
    def fnma(self, acc, a, b):  # acc -= a*b
        self.ins(f"vfnmadd231pd %%zmm{b}, %%zmm{a}, %%zmm{acc}")
    def fma213(self, d, m, a):  # d = d*m + a
        self.ins(f"vfmadd213pd %%zmm{a}, %%zmm{m}, %%zmm{d}")
    def fnma213(self, d, m, a): # d = -d*m + a
        self.ins(f"vfnmadd213pd %%zmm{a}, %%zmm{m}, %%zmm{d}")
    def rsqrt(self, a, kill=()):
        d = self._dst(kill)
        self.ins(f"vrsqrt14pd %%zmm{a}, %%zmm{d}")
        return d
    def rcp(self, a, kill=()):
        d = self._dst(kill)
        self.ins(f"vrcp14pd %%zmm{a}, %%zmm{d}")
        return d

def emit_map_asm(a, zr, zi, mc):
    """map z/(1+|z|); zr,zi regs (killed); mc = dict of const regs TINY,ONE,HALF.
    returns (xr, xi) new regs."""
    m  = a.mul(zr, zr)              # m = zr*zr
    a.fma(m, zi, zi)                # m += zi*zi
    self_tiny = mc['TINY']
    a.ins(f"vaddpd %%zmm{self_tiny}, %%zmm{m}, %%zmm{m}")  # m += tiny
    r0 = a.rsqrt(m)
    t  = a.mul(m, r0)
    hr = a.mul(r0, mc['HALF'])
    # eh = 0.5 - t*hr
    eh = a.mov(mc['HALF'])
    a.fnma(eh, t, hr); a.rel(t); a.rel(hr)
    # r1 = r0 + r0*eh
    a.fma213(eh, r0, r0)            # eh = eh*r0 + r0 -> r1
    r1 = eh
    a.rel(r0)
    mg0 = a.mul(m, r1)
    hr1 = a.mul(r1, mc['HALF'], kill=(r1,))
    # e2 = m - mg0*mg0
    a.fnma213(mg0, mg0, m)  # careful: mg0 = -(mg0*mg0) + m ... destroys mg0!
    # need mg0 afterwards -> recompute differently:
    # we did it wrong; redo: e2 = m - mg0^2 then mag = mg0 + e2*hr1
    # implement: e2 = mov(m); fnma(e2, mg0, mg0); mag = mg0; fma(mag, e2, hr1)
    raise RuntimeError("see emit_map_asm2")

def emit_map_asm2(a, zr, zi, mc):
    m  = a.mul(zr, zr)
    a.fma(m, zi, zi)
    a.ins(f"vaddpd %%zmm{mc['TINY']}, %%zmm{m}, %%zmm{m}")
    r0 = a.rsqrt(m)
    t  = a.mul(m, r0)
    hr = a.mul(r0, mc['HALF'])
    eh = a.mov(mc['HALF'])
    a.fnma(eh, t, hr)
    a.rel(t); a.rel(hr)
    r1 = a.mov(r0)
    a.fma(r1, r0, eh)
    a.rel(r0); a.rel(eh)
    mg0 = a.mul(m, r1)
    hr1 = a.mul(r1, mc['HALF'], kill=(r1,))
    e2 = a.mov(m)
    a.fnma(e2, mg0, mg0)
    a.rel(m)
    mag = mg0
    a.fma(mag, e2, hr1)
    a.rel(e2); a.rel(hr1)
    u = a.add(mag, mc['ONE'], kill=(mag,))
    w0 = a.rcp(u)
    e3 = a.mov(mc['ONE'])
    a.fnma(e3, u, w0)
    a.rel(u)
    w1 = a.mov(w0)
    a.fma(w1, w0, e3)
    a.rel(w0)
    ee = a.mul(e3, e3, kill=(e3,))
    w2 = a.mov(w1)
    a.fma(w2, w1, ee)
    a.rel(w1); a.rel(ee)
    xr = a.mul(zr, w2, kill=(zr,))
    xi = a.mul(zi, w2, kill=(zi,))
    a.rel(w2)
    return xr, xi
