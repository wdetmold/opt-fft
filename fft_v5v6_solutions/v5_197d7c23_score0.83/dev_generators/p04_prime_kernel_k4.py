# S05 (log 01:22:06): replace PRIME_KERNEL with 4-way k-blocked version
src = open('impl_head.c').read()
# Replace PRIME_KERNEL with 4-way k-blocked version
old_start = src.index('// direct symmetric prime kernel')
old_end = src.index('PRIME_KERNEL(k13, 13, 6, tab13)')
new = '''// direct symmetric prime kernel, 4-way k-blocked (FMA-bound)
#define PK_ACC4(T0,T1,T2,T3) \\
        for(int j=1;j<=HH;j++){ \\
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \\
            v8 c0=splat(T0[2*j-2]), s0=splat(T0[2*j-1]); \\
            v8 c1=splat(T1[2*j-2]), s1=splat(T1[2*j-1]); \\
            v8 c2=splat(T2[2*j-2]), s2=splat(T2[2*j-1]); \\
            v8 c3=splat(T3[2*j-2]), s3=splat(T3[2*j-1]); \\
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \\
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \\
            ur2+=c2*_sr; ui2+=c2*_si; vr2+=s2*_dr; vi2+=s2*_di; \\
            ur3+=c3*_sr; ui3+=c3*_si; vr3+=s3*_dr; vi3+=s3*_di; \\
        }
#define PK_OUT(K, UR,UI,VR,VI) \\
        p[2*st*(K)]=UR+VI;        p[2*st*(K)+1]=UI-VR; \\
        p[2*st*(LL-(K))]=UR-VI;   p[2*st*(LL-(K))+1]=UI+VR;

#define PRIME_KERNEL(NAME, LL, HH, TABLE) \\
AI void NAME(v8 *restrict p, const long st){ \\
    v8 sr[HH+1], si[HH+1], dr[HH+1], di[HH+1]; \\
    const v8 x0r=p[0], x0i=p[1]; \\
    v8 o0r=x0r, o0i=x0i; \\
    for(int j=1;j<=HH;j++){ \\
        v8 ar=p[2*st*j], ai=p[2*st*j+1]; \\
        v8 br=p[2*st*(LL-j)], bi=p[2*st*(LL-j)+1]; \\
        sr[j]=ar+br; si[j]=ai+bi; dr[j]=ar-br; di[j]=ai-bi; \\
        o0r+=sr[j]; o0i+=si[j]; \\
    } \\
    p[0]=o0r; p[1]=o0i; \\
    int k=1; \\
    for(; k+3<=HH; k+=4){ \\
        const double *t0 = TABLE + (k-1)*HH*2 - 0; \\
        const double *t1 = t0 + HH*2; \\
        const double *t2 = t1 + HH*2; \\
        const double *t3 = t2 + HH*2; \\
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \\
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\
        v8 ur2=x0r, ui2=x0i, vr2=VZERO, vi2=VZERO; \\
        v8 ur3=x0r, ui3=x0i, vr3=VZERO, vi3=VZERO; \\
        PK_ACC4(t0,t1,t2,t3) \\
        PK_OUT(k,   ur0,ui0,vr0,vi0) \\
        PK_OUT(k+1, ur1,ui1,vr1,vi1) \\
        PK_OUT(k+2, ur2,ui2,vr2,vi2) \\
        PK_OUT(k+3, ur3,ui3,vr3,vi3) \\
    } \\
    for(; k+1<=HH; k+=2){ \\
        const double *t0 = TABLE + (k-1)*HH*2; \\
        const double *t1 = t0 + HH*2; \\
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \\
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \\
        for(int j=1;j<=HH;j++){ \\
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \\
            v8 c0=splat(t0[2*j-2]), s0=splat(t0[2*j-1]); \\
            v8 c1=splat(t1[2*j-2]), s1=splat(t1[2*j-1]); \\
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \\
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \\
        } \\
        PK_OUT(k,   ur0,ui0,vr0,vi0) \\
        PK_OUT(k+1, ur1,ui1,vr1,vi1) \\
    } \\
    for(; k<=HH; k++){ \\
        const double *t0 = TABLE + (k-1)*HH*2; \\
        v8 ur=x0r, ui=x0i, vr=VZERO, vi=VZERO; \\
        for(int j=1;j<=HH;j++){ \\
            v8 c=splat(t0[2*j-2]), s=splat(t0[2*j-1]); \\
            ur+=c*sr[j]; ui+=c*si[j]; vr+=s*dr[j]; vi+=s*di[j]; \\
        } \\
        PK_OUT(k, ur,ui,vr,vi) \\
    } \\
}

'''
src = src[:old_start] + new + src[old_end:]
open('impl_head.c','w').write(src)
print("done")
