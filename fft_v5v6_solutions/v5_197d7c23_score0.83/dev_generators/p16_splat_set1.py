# B14 (log 02:25:00): splat via _mm512_set1_pd so gcc folds table loads into
# embedded-broadcast FMA operands
src=open('impl_head.c').read()
old='AI v8 splat(double x){ return (v8){x,x,x,x,x,x,x,x}; }'
new='AI v8 splat(double x){ return (v8)_mm512_set1_pd(x); }'
assert old in src
src=src.replace(old,new)
open('impl_head.c','w').write(src)
