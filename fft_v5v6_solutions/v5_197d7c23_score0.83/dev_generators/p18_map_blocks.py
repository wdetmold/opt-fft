# B19 (log 02:29:28): 4-way-ILP rcp-chain map (map_blocks); used in both step schemes
src=open('impl_head.c').read()
old = src[src.index('AI void vmap'):src.index('// ---------------------------------------------------------------- small DFT')]
new = '''AI void vmap(v8 *zr, v8 *zi){
    v8 s = *zr * *zr + *zi * *zi;
    s = (v8)_mm512_max_pd((__m512d)s, (__m512d)splat(1e-300));
    v8 r = (v8)_mm512_rsqrt14_pd((__m512d)s);
    v8 hs = s*splat(0.5);
    r = r*(splat(1.5) - hs*r*r);
    r = r*(splat(1.5) - hs*r*r);
    v8 u = splat(1.0) + s*r;
    v8 t = (v8)_mm512_rcp14_pd((__m512d)u);
    t = t*(splat(2.0) - u*t);
    t = t*(splat(2.0) - u*t);
    *zr *= t; *zi *= t;
}
// 4-way ILP map over n blocks: p <- (p+c)/(1+|p+c|)
AI void map_blocks(v8 *restrict p, const v8 *restrict c, long n){
    long i=0;
    for(; i+4<=n; i+=4){
        v8 zr0=p[2*i+0]+c[2*i+0], zi0=p[2*i+1]+c[2*i+1];
        v8 zr1=p[2*i+2]+c[2*i+2], zi1=p[2*i+3]+c[2*i+3];
        v8 zr2=p[2*i+4]+c[2*i+4], zi2=p[2*i+5]+c[2*i+5];
        v8 zr3=p[2*i+6]+c[2*i+6], zi3=p[2*i+7]+c[2*i+7];
        v8 s0=zr0*zr0+zi0*zi0, s1=zr1*zr1+zi1*zi1, s2=zr2*zr2+zi2*zi2, s3=zr3*zr3+zi3*zi3;
        s0=(v8)_mm512_max_pd((__m512d)s0,(__m512d)splat(1e-300));
        s1=(v8)_mm512_max_pd((__m512d)s1,(__m512d)splat(1e-300));
        s2=(v8)_mm512_max_pd((__m512d)s2,(__m512d)splat(1e-300));
        s3=(v8)_mm512_max_pd((__m512d)s3,(__m512d)splat(1e-300));
        v8 r0=(v8)_mm512_rsqrt14_pd((__m512d)s0), r1=(v8)_mm512_rsqrt14_pd((__m512d)s1);
        v8 r2=(v8)_mm512_rsqrt14_pd((__m512d)s2), r3=(v8)_mm512_rsqrt14_pd((__m512d)s3);
        v8 h0=s0*splat(0.5), h1=s1*splat(0.5), h2=s2*splat(0.5), h3=s3*splat(0.5);
        r0=r0*(splat(1.5)-h0*r0*r0); r1=r1*(splat(1.5)-h1*r1*r1);
        r2=r2*(splat(1.5)-h2*r2*r2); r3=r3*(splat(1.5)-h3*r3*r3);
        r0=r0*(splat(1.5)-h0*r0*r0); r1=r1*(splat(1.5)-h1*r1*r1);
        r2=r2*(splat(1.5)-h2*r2*r2); r3=r3*(splat(1.5)-h3*r3*r3);
        v8 u0=splat(1.0)+s0*r0, u1=splat(1.0)+s1*r1, u2=splat(1.0)+s2*r2, u3=splat(1.0)+s3*r3;
        v8 t0=(v8)_mm512_rcp14_pd((__m512d)u0), t1=(v8)_mm512_rcp14_pd((__m512d)u1);
        v8 t2=(v8)_mm512_rcp14_pd((__m512d)u2), t3=(v8)_mm512_rcp14_pd((__m512d)u3);
        t0=t0*(splat(2.0)-u0*t0); t1=t1*(splat(2.0)-u1*t1);
        t2=t2*(splat(2.0)-u2*t2); t3=t3*(splat(2.0)-u3*t3);
        t0=t0*(splat(2.0)-u0*t0); t1=t1*(splat(2.0)-u1*t1);
        t2=t2*(splat(2.0)-u2*t2); t3=t3*(splat(2.0)-u3*t3);
        p[2*i+0]=zr0*t0; p[2*i+1]=zi0*t0;
        p[2*i+2]=zr1*t1; p[2*i+3]=zi1*t1;
        p[2*i+4]=zr2*t2; p[2*i+5]=zi2*t2;
        p[2*i+6]=zr3*t3; p[2*i+7]=zi3*t3;
    }
    for(; i<n; i++){
        v8 zr=p[2*i]+c[2*i], zi=p[2*i+1]+c[2*i+1];
        vmap(&zr,&zi);
        p[2*i]=zr; p[2*i+1]=zi;
    }
}

'''
src = src.replace(old, new)
open('impl_head.c','w').write(src)
# use map_blocks in the step macros
src=open('impl_tail.c').read()
src=src.replace('''        for(long i=0;i<L2;i++){ \\
            v8 zr=pl[2*i]+cpl[2*i], zi=pl[2*i+1]+cpl[2*i+1]; \\
            vmap(&zr,&zi); \\
            pl[2*i]=zr; pl[2*i+1]=zi; \\
        } \\''','''        map_blocks(pl, cpl, L2); \\''')
src=src.replace('''        for(long i=0;i<(long)LL*RB;i++){                         /* +c, map */ \\
            v8 zr=pl[2*i]+cpl[2*i], zi=pl[2*i+1]+cpl[2*i+1]; \\
            vmap(&zr,&zi); \\
            pl[2*i]=zr; pl[2*i+1]=zi; \\
        } \\''','''        map_blocks(pl, cpl, (long)LL*RB);                        /* +c, map */ \\''')
open('impl_tail.c','w').write(src)
print("ok")
