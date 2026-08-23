# B60 (log 03:11:02): add a 3-wide k-block to PRIME_KERNEL so k23 (HH=11) runs
# K4+K4+K3 instead of K4+K4+K2+K1. Last change before the final /workdir copy.
src=open('impl_head.c').read()
anchor = '''    for(; k+1<=HH; k+=2){ \\
        const double *t0 = TABLE + (k-1)*HH*2; \\
        const double *t1 = t0 + HH*2; \\
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \\
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\'''
k3block = '''    for(; k+2<=HH; k+=3){ \\
        const double *t0 = TABLE + (k-1)*HH*2; \\
        const double *t1 = t0 + HH*2; \\
        const double *t2 = t1 + HH*2; \\
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \\
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\
        v8 ur2=x0r, ui2=x0i, vr2=VZERO, vi2=VZERO; \\
        _Pragma("GCC unroll 16") \\
        for(int j=1;j<=HH;j++){ \\
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \\
            v8 c0=splat(t0[2*j-2]), s0=splat(t0[2*j-1]); \\
            v8 c1=splat(t1[2*j-2]), s1=splat(t1[2*j-1]); \\
            v8 c2=splat(t2[2*j-2]), s2=splat(t2[2*j-1]); \\
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \\
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \\
            ur2+=c2*_sr; ui2+=c2*_si; vr2+=s2*_dr; vi2+=s2*_di; \\
        } \\
        PK_OUT(LL,k,   ur0,ui0,vr0,vi0) \\
        PK_OUT(LL,k+1, ur1,ui1,vr1,vi1) \\
        PK_OUT(LL,k+2, ur2,ui2,vr2,vi2) \\
    } \\
''' + anchor
assert anchor in src
src=src.replace(anchor, k3block)
open('impl_head.c','w').write(src)
print("K3 block added")
