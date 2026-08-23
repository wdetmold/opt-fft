# B9 (log 02:20:58): force loop unrolling in k36/k45 (PFA index-table folding) and k8
src=open('impl_head.c').read()
# k36: unroll loops
src=src.replace('''AI void k36(v8 *restrict p, const long st){
    v8 Tr[4][9], Ti[4][9];
    for(int n2=0;n2<9;n2++){''','''AI void k36(v8 *restrict p, const long st){
    v8 Tr[4][9], Ti[4][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){''')
src=src.replace('''    for(int k1=0;k1<4;k1++){
        dft9(Tr[k1], Ti[k1]);
        for(int k2=0;k2<9;k2++){
            p[2*st*K36T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K36T[k1][k2]+1] = Ti[k1][k2];
        }
    }''','''    #pragma GCC unroll 4
    for(int k1=0;k1<4;k1++){
        dft9(Tr[k1], Ti[k1]);
        #pragma GCC unroll 9
        for(int k2=0;k2<9;k2++){
            p[2*st*K36T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K36T[k1][k2]+1] = Ti[k1][k2];
        }
    }''')
# k45 same
src=src.replace('''AI void k45(v8 *restrict p, const long st){
    v8 Tr[5][9], Ti[5][9];
    for(int n2=0;n2<9;n2++){''','''AI void k45(v8 *restrict p, const long st){
    v8 Tr[5][9], Ti[5][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){''')
src=src.replace('''    for(int k1=0;k1<5;k1++){
        dft9(Tr[k1], Ti[k1]);
        for(int k2=0;k2<9;k2++){
            p[2*st*K45T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K45T[k1][k2]+1] = Ti[k1][k2];
        }
    }''','''    #pragma GCC unroll 5
    for(int k1=0;k1<5;k1++){
        dft9(Tr[k1], Ti[k1]);
        #pragma GCC unroll 9
        for(int k2=0;k2<9;k2++){
            p[2*st*K45T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K45T[k1][k2]+1] = Ti[k1][k2];
        }
    }''')
# k6 (PFA(2,3)) already fully unrolled manually. k8 loops: unroll
src=src.replace('''AI void k8(v8 *restrict p, const long st){
    v8 r[8], i[8];
    for(int t=0;t<8;t++){ r[t]=p[2*st*t]; i[t]=p[2*st*t+1]; }
    dft8(r,i);
    for(int t=0;t<8;t++){ p[2*st*t]=r[t]; p[2*st*t+1]=i[t]; }
}''','''AI void k8(v8 *restrict p, const long st){
    v8 r[8], i[8];
    #pragma GCC unroll 8
    for(int t=0;t<8;t++){ r[t]=p[2*st*t]; i[t]=p[2*st*t+1]; }
    dft8(r,i);
    #pragma GCC unroll 8
    for(int t=0;t<8;t++){ p[2*st*t]=r[t]; p[2*st*t+1]=i[t]; }
}''')
open('impl_head.c','w').write(src)
print("ok")
