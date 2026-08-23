# S07 (log 01:24:50): vmap -> rsqrt-chain + exact divide
src = open('impl_head.c').read()
old = src[src.index('AI void vmap'):src.index('// ---------------------------------------------------------------- small DFT')]
new = '''AI void vmap(v8 *zr, v8 *zi){
    v8 s = *zr * *zr + *zi * *zi;
    s = (v8)_mm512_max_pd((__m512d)s, (__m512d)splat(1e-300));
    v8 r = (v8)_mm512_rsqrt14_pd((__m512d)s);
    v8 hs = s*splat(0.5);
    r = r*(splat(1.5) - hs*r*r);
    r = r*(splat(1.5) - hs*r*r);
    v8 t = splat(1.0)/(splat(1.0) + s*r);   // 1/(1+|z|), exact divide
    *zr *= t; *zi *= t;
}

'''
src = src.replace(old, new)
open('impl_head.c','w').write(src)
print("vmap updated")
