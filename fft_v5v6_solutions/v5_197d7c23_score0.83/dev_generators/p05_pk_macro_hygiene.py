# S06 (log 01:22:18): macro hygiene - pass HH/LL into helper macros
src = open('impl_head.c').read()
src = src.replace('#define PK_ACC4(T0,T1,T2,T3) \\', '#define PK_ACC4(HH,T0,T1,T2,T3) \\')
src = src.replace('#define PK_OUT(K, UR,UI,VR,VI) \\', '#define PK_OUT(LL,K, UR,UI,VR,VI) \\')
src = src.replace('PK_ACC4(t0,t1,t2,t3)', 'PK_ACC4(HH,t0,t1,t2,t3)')
for a,b in [('PK_OUT(k, ','PK_OUT(LL,k, '),('PK_OUT(k,   ','PK_OUT(LL,k,   '),('PK_OUT(k+1, ','PK_OUT(LL,k+1, '),('PK_OUT(k+2, ','PK_OUT(LL,k+2, '),('PK_OUT(k+3, ','PK_OUT(LL,k+3, ')]:
    src = src.replace(a,b)
open('impl_head.c','w').write(src)
