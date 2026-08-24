# v8 generator library: emit AVX-512 C codelets
import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')

def hexd(x):
    return float(np.double(x)).hex()

def trig(L):
    """exact mod-L cos/sin tables of -2pi j/L in longdouble -> double"""
    j = np.arange(L)
    ang = (-2*PI) * j.astype(LD) / LD(L)
    return np.cos(ang), np.sin(ang)

class E:
    """emitter with simple SSA naming"""
    def __init__(self, pfx="t"):
        self.lines = []
        self.n = 0
        self.pfx = pfx
    def v(self, expr):
        name = f"{self.pfx}{self.n}"; self.n += 1
        self.lines.append(f"__m512d {name} = {expr};")
        return name
    def raw(self, s):
        self.lines.append(s)
    def code(self, indent="    "):
        return "\n".join(indent + l for l in self.lines)

# complex vector ops on (re,im) name pairs
def cadd(e,a,b): return (e.v(f"_mm512_add_pd({a[0]},{b[0]})"), e.v(f"_mm512_add_pd({a[1]},{b[1]})"))
def csub(e,a,b): return (e.v(f"_mm512_sub_pd({a[0]},{b[0]})"), e.v(f"_mm512_sub_pd({a[1]},{b[1]})"))
def cmulir(e,a):  # multiply by -i: (re,im) -> (im, -re)  [since -i*(r+ii) = i_ - i r]
    return (a[1], e.v(f"_mm512_sub_pd(_mm512_setzero_pd(),{a[0]})"))

