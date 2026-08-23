# S09 (log 01:28:58): k64 with forced unrolling. NOTE: this truncates impl_head.c
# after k64 (dropping tr8); p09 restores tr8. Faithful to the original session.
src=open('impl_head.c').read()
# force-unroll loops in k64
old = src[src.index('// 64 = 8 x 8 Cooley-Tukey'):]
new = '''// 64 = 8 x 8 Cooley-Tukey
AI void k64(v8 *restrict p, const long st){
    v8 T[128]; // [k1][j1] pairs at (k1*8+j1)*2
    #pragma GCC unroll 8
    for(int j1=0;j1<8;j1++){
        v8 r[8], i[8];
        #pragma GCC unroll 8
        for(int t=0;t<8;t++){ r[t]=p[2*st*(j1+8*t)]; i[t]=p[2*st*(j1+8*t)+1]; }
        dft8(r,i);
        if(j1==0){
            #pragma GCC unroll 8
            for(int k1=0;k1<8;k1++){ T[(k1*8)*2]=r[k1]; T[(k1*8)*2+1]=i[k1]; }
        }else{
            T[(0*8+j1)*2]=r[0]; T[(0*8+j1)*2+1]=i[0];
            #pragma GCC unroll 8
            for(int k1=1;k1<8;k1++){
                v8 wr=splat(tw64[(j1*8+k1)*2]), wi=splat(tw64[(j1*8+k1)*2+1]);
                v8 tr,ti; CMUL(tr,ti, r[k1],i[k1], wr,wi);
                T[(k1*8+j1)*2]=tr; T[(k1*8+j1)*2+1]=ti;
            }
        }
    }
    #pragma GCC unroll 8
    for(int k1=0;k1<8;k1++){
        v8 r[8], i[8];
        #pragma GCC unroll 8
        for(int t=0;t<8;t++){ r[t]=T[(k1*8+t)*2]; i[t]=T[(k1*8+t)*2+1]; }
        dft8(r,i);
        #pragma GCC unroll 8
        for(int k2=0;k2<8;k2++){ p[2*st*(k1+8*k2)]=r[k2]; p[2*st*(k1+8*k2)+1]=i[k2]; }
    }
}
'''
src = src[:src.index('// 64 = 8 x 8 Cooley-Tukey')] + new
open('impl_head.c','w').write(src)
