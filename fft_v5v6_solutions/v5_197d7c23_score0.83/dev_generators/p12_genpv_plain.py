# B1/T1 (log 02:10:34, first action after the container restart): revert GEN_PV to the
# clean version (no prefetch, separate map pass). This is the macro that ships (with
# p18's map_blocks substitution and p24's KFND parameter applied on top).
src = open('impl_tail.c').read()
old_start = src.index('#define GEN_PV(SFX, LL, RSY, RB, KFN, PF)')
old_end = src.index('static void pv_in(')
new = r'''#define GEN_PV(SFX, LL, RSY, RB, KFN, PF) \
static void pv_step_##SFX(v8 *restrict V, const v8 *restrict C){ \
    const long PB=(long)RSY*RB, NSTR=((LL)+7)/8; \
    for(long y=0;y<LL;y++) \
        for(long zc=0; zc<RB; zc++) KFN(V + (y*RB+zc)*2, PB);   /* x lines */ \
    for(long x=0;x<LL;x++){ \
        v8 *pl = V + x*PB*2; \
        const v8 *cpl = C + x*PB*2; \
        for(long zc=0; zc<RB; zc++) KFN(pl + zc*2, RB);         /* y lines */ \
        for(long s=0; s<NSTR; s++){                              /* z lines */ \
            const long y0 = (s<NSTR-1)? 8*s : (long)RSY-8; \
            const long skip = (s<NSTR-1)? 0 : 8*NSTR-(long)RSY; \
            v8 buf[RB*8*2]; \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=pl[((y0+e)*RB+t)*2]; ai[e]=pl[((y0+e)*RB+t)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=0;e<8;e++){ buf[(t*8+e)*2]=ar[e]; buf[(t*8+e)*2+1]=ai[e]; } \
            } \
            KFN(buf, 1); \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=buf[(t*8+e)*2]; ai[e]=buf[(t*8+e)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=skip;e<8;e++){ pl[((y0+e)*RB+t)*2]=ar[e]; pl[((y0+e)*RB+t)*2+1]=ai[e]; } \
            } \
        } \
        for(long i=0;i<(long)LL*RB;i++){                         /* +c, map */ \
            v8 zr=pl[2*i]+cpl[2*i], zi=pl[2*i+1]+cpl[2*i+1]; \
            vmap(&zr,&zi); \
            pl[2*i]=zr; pl[2*i+1]=zi; \
        } \
    } \
}
GEN_PV(6,  6,  8, 1, k6, 0)
GEN_PV(8,  8,  8, 1, k8, 0)
GEN_PV(13, 13, 13, 2, k13, 0)
GEN_PV(17, 17, 17, 3, k17, 0)
GEN_PV(23, 23, 23, 3, k23, 0)
GEN_PV(36, 36, 36, 5, k36, 0)
GEN_PV(45, 45, 45, 6, k45, 0)
GEN_PV(64, 64, 64, 8, k64, 0)

'''
src = src[:old_start] + new + src[old_end:]
open('impl_tail.c','w').write(src)
print("ok")
