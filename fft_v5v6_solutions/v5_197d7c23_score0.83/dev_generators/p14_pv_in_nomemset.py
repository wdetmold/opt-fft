# B6 (log 02:19:40): trim conversion overhead - drop the per-volume memset in pv_in
# (keep it only for L=6 pad rows), zero z-pad lanes explicitly.
src=open('impl_tail.c').read()
old = '''static void pv_in(const double *restrict src, v8 *restrict V, long LL, long RB, long PB){
    memset(V, 0, (size_t)LL*PB*128);
    long full=LL/8, tail=LL%8;
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        const double *s = src + ((x*LL+y)*LL)*2;
        v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) deint(s+zc*16, &row[zc*2], &row[zc*2+1]);
        if(tail){
            double *rr=(double*)(row+full*2);
            for(long t=0;t<tail;t++){ rr[t]=s[(full*8+t)*2]; rr[8+t]=s[(full*8+t)*2+1]; }
        }
    }
}'''
new = '''static void pv_in(const double *restrict src, v8 *restrict V, long LL, long RB, long PB){
    long full=LL/8, tail=LL%8;
    if(LL<8) memset(V, 0, (size_t)LL*PB*128);  /* L=6: pad rows too */
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        const double *s = src + ((x*LL+y)*LL)*2;
        v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) deint(s+zc*16, &row[zc*2], &row[zc*2+1]);
        if(tail){
            double *rr=(double*)(row+full*2);
            for(long t=0;t<tail;t++){ rr[t]=s[(full*8+t)*2]; rr[8+t]=s[(full*8+t)*2+1]; }
            for(long t=tail;t<8;t++){ rr[t]=0.0; rr[8+t]=0.0; }   /* zero z-pad lanes */
        }
    }
}'''
assert old in src
src=src.replace(old,new)
open('impl_tail.c','w').write(src)
print("ok")
