# B16 (log 02:25:57): fully unroll the prime-kernel j-loops
src=open('impl_head.c').read()
src=src.replace('''#define PK_ACC4(HH,T0,T1,T2,T3) \\
        for(int j=1;j<=HH;j++){ \\''','''#define PK_ACC4(HH,T0,T1,T2,T3) \\
        _Pragma("GCC unroll 16") \\
        for(int j=1;j<=HH;j++){ \\''')
# also unroll the pre-loop and K2/K1 loops
src=src.replace('''    v8 o0r=x0r, o0i=x0i; \\
    for(int j=1;j<=HH;j++){ \\
        v8 ar=p[2*st*j], ai=p[2*st*j+1]; \\''','''    v8 o0r=x0r, o0i=x0i; \\
    _Pragma("GCC unroll 16") \\
    for(int j=1;j<=HH;j++){ \\
        v8 ar=p[2*st*j], ai=p[2*st*j+1]; \\''')
src=src.replace('''        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\
        for(int j=1;j<=HH;j++){ \\''','''        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\
        _Pragma("GCC unroll 16") \\
        for(int j=1;j<=HH;j++){ \\''')
src=src.replace('''        v8 ur=x0r, ui=x0i, vr=VZERO, vi=VZERO; \\
        for(int j=1;j<=HH;j++){ \\''','''        v8 ur=x0r, ui=x0i, vr=VZERO, vi=VZERO; \\
        _Pragma("GCC unroll 16") \\
        for(int j=1;j<=HH;j++){ \\''')
open('impl_head.c','w').write(src)
